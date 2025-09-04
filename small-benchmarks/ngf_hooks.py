# ngf_hooks.py — Stage-3: Geo + Detect (gain-only) + Soft Denoise at tap -9
from typing import Optional
import torch

def _resolve_tap(model, tap: int) -> int:
    n = len(model.transformer.h)
    return tap + n if tap < 0 else tap

@torch.no_grad()
def attach_ngf_hooks(
    model,
    tokenizer=None,
    device: Optional[torch.device] = None,
    *,
    tap: int = -9,
    g_det_max: float = 1.4,      # NEW: bound detect gain
    K_FADE: int = 16,            # NEW: widen fade over final tokens (MC scoring)
    # ---- Geo (v4b defaults) ----
    alpha0: float = 0.05,
    alpha_min: float = 0.006,
    trend_tau: float = 0.32,
    k_tr: int = 12,
    s_latch: float = 0.30,
    linger: int = 2,
    ema_center_beta: float = 0.05,
    gen_mode: str = "geo",
    # ---- Detect (v4b defaults, gain-only) ----
    use_detect: int = 1,
    detect_width: int = 20,
    null_K: int = 32,
    null_q: float = 0.92,
    k_det: int = 8,
    detect_sigma: float = 5.0,   # reserved
    # ---- Soft Denoise (v4b-ish defaults) ----
    use_denoise: int = 1,
    denoise_beta: float = 0.45,     # EMA strength for residual smoothing
    denoise_window: int = 3,        # tiny median window (time axis)
    denoise_k: int = 6,             # steps for phantom trend check
    denoise_tau: float = 0.40,      # sensitivity for phantom detection
    phantom_tr_tau: float = 0.65,   # EMA for norm trend baseline
    phantom_guard_gamma: float = 0.25,  # scale-down factor when phantom risk rises
    jitter_eps: float = 0.00,       # small noise on residual to avoid stickiness

    # older settings
    # use_denoise=1, denoise_beta= 0.60, denoise_window=5,
    # phantom_guard_gamma=0.45, jitter_eps=0.03

    # newer settings    
    # use_denoise=1, denoise_beta=0.45, denoise_window=3,
    # phantom_guard_gamma=0.25, jitter_eps=0.0
):
    layer_idx = _resolve_tap(model, tap)
    blocks = model.transformer.h
    if not (0 <= layer_idx < len(blocks)):
        raise ValueError(f"tap {tap} → layer {layer_idx} is out of range (0..{len(blocks)-1})")

    cfg = dict(
        tap=tap, alpha0=alpha0, alpha_min=alpha_min, trend_tau=trend_tau, k_tr=k_tr,
        s_latch=s_latch, linger=linger, ema_center_beta=ema_center_beta, gen_mode=gen_mode,
        use_detect=use_detect, detect_width=detect_width, null_K=null_K, null_q=null_q, k_det=k_det,
        use_denoise=use_denoise, denoise_beta=denoise_beta, denoise_window=denoise_window,
        denoise_k=denoise_k, denoise_tau=denoise_tau, phantom_tr_tau=phantom_tr_tau,
        phantom_guard_gamma=phantom_guard_gamma, jitter_eps=jitter_eps
    )
    print(f"[NGF] Hooking GPT-2 layer {layer_idx} (tap={tap}) [Stage-3: Geo+Detect+Denoise] cfg={cfg}")

    # Internal state across calls
    state = {
        "ema_center": None,      # (C,)
        "null_std": None,        # scalar baseline of motion (detect)
        "warmup": 0,             # frames to seed null
        "ema_r": None,           # (B, C) EMA of residual for denoise
        "ema_norm": None,        # scalar EMA of residual norm (phantom guard)
    }

    # Don’t modify final tokens used to score MC endings
    K_FADE = 8  # 6–12 is typical for HellaSwag

    def _compute_trend_gate(x, center):
        B, T, C = x.shape
        k = min(k_tr, T)
        dx = x[:, -k:, :] - center
        step_mag = dx.pow(2).sum(dim=-1).mean(dim=1)  # (B,)
        gap = (step_mag.mean() - step_mag.median()).clamp(min=0)
        g_tr = torch.sigmoid(gap / (trend_tau + 1e-6))
        return g_tr

    def _detect_gain(x, center):
        """
        Gain-only detect using motion magnitude over last `detect_width` tokens.
        g_det = 1 + tanh(k_det * max(0, (cur - null)/null))
        """
        B, T, C = x.shape
        w = min(max(detect_width, 1), T)
        dx = x[:, -w:, :] - center
        mot = dx.pow(2).sum(dim=-1).sqrt()     # (B, w)
        cur_std = mot.mean().detach()          # scalar

        if state["null_std"] is None:
            state["null_std"] = cur_std
            state["warmup"] = 1
        else:
            if state["warmup"] < null_K:
                state["null_std"] = (state["null_std"] * state["warmup"] + cur_std) / (state["warmup"] + 1)
                state["warmup"] += 1
            else:
                state["null_std"] = null_q * state["null_std"] + (1.0 - null_q) * cur_std

        base = state["null_std"].clamp(min=1e-6)
        delta = (cur_std - base).clamp(min=0.0) / base
        g_raw = 1.0 + torch.tanh(torch.tensor(k_det, dtype=cur_std.dtype, device=cur_std.device) * delta).item()
        g_det = float(min(g_raw, g_det_max))  # NEW: cap gain
        return g_det

    def _soft_denoise(center, y_pre):
        """
        Sign-safe residual smoothing + tiny median + phantom guard + jitter.
        Operates along time, returns y_post with same shape as y_pre.
        """
        B, T, C = y_pre.shape
        device_ = y_pre.device
        dtype_  = y_pre.dtype

        r = y_pre - center  # residual

        # init EMA state if needed
        if state["ema_r"] is None or state["ema_r"].shape[0] != B:
            state["ema_r"] = r[:, -1, :].detach()
        ema_r = state["ema_r"]

        # EMA for residual norm (phantom guard baseline)
        cur_norm = r.pow(2).sum(dim=-1).mean() ** 0.5  # scalar
        if state["ema_norm"] is None:
            state["ema_norm"] = cur_norm
        else:
            # smooth baseline
            beta_norm = float(phantom_tr_tau)
            state["ema_norm"] = beta_norm * state["ema_norm"] + (1.0 - beta_norm) * cur_norm
        base_norm = state["ema_norm"].clamp(min=1e-6)

        # residual std for jitter scale
        r_std = r.std(dim=(0, 1))                # (C,)

        # run denoise along T
        y_out_list = []
        # small buffer for median of last W EMA residuals (per-step)
        W = max(int(denoise_window), 1)
        med_buf = []

        for t in range(T):
            # EMA on residual
            ema_r = denoise_beta * ema_r + (1.0 - denoise_beta) * r[:, t, :]

            # median filter (over recent EMA residuals)
            med_buf.append(ema_r.unsqueeze(1))  # (B,1,C)
            if len(med_buf) > W:
                med_buf.pop(0)
            if W > 1 and len(med_buf) > 1:
                # stack over window and take median across time axis
                stack = torch.cat(med_buf, dim=1)     # (B,W,C)
                ema_r_med = stack.median(dim=1).values
            else:
                ema_r_med = ema_r

            # phantom guard: if residual norm spikes vs baseline, scale toward center
            # compute batch mean norm for current filtered residual
            rn = (ema_r_med.pow(2).sum(dim=-1).mean() ** 0.5)
            excess = (rn - base_norm).clamp(min=0.0) / base_norm
            # soft gate 0..1
            g_ph = torch.tanh(excess / max(denoise_tau, 1e-6))
            # shrink factor in [1-gamma, 1] depending on g_ph
            shrink = 1.0 - phantom_guard_gamma * g_ph.item()
            ema_r_guard = ema_r_med * shrink

            # tiny jitter (sign-safe; on residual)
            if jitter_eps > 0.0:
                noise = torch.randn_like(ema_r_guard) * (jitter_eps * r_std)
                ema_r_guard = ema_r_guard + noise

            y_step = center + ema_r_guard.unsqueeze(1)   # (B,1,C)
            y_out_list.append(y_step)

        y_post = torch.cat(y_out_list, dim=1)  # (B,T,C)
        state["ema_r"] = ema_r.detach()
        return y_post

    def ngf_forward_hook(module, inputs, output):
        # HF GPT-2 blocks can return Tuple[Tensor, ...]
        is_tuple = isinstance(output, (tuple, list))
        x = output[0] if is_tuple else output  # (B, T, C)
        B, T, C = x.shape

        # 1) EMA center at this tap
        seq_mean = x.mean(dim=1).mean(dim=0)
        if state["ema_center"] is None:
            state["ema_center"] = seq_mean.detach()
        else:
            state["ema_center"] = ema_center_beta * seq_mean.detach() + (1.0 - ema_center_beta) * state["ema_center"]
        center = state["ema_center"].view(1, 1, C)

        # 2) trend gate (telemetry)
        _ = _compute_trend_gate(x, center)

        # 3) base warp (constant small alpha for MC), detect gain (>=1)
        alpha = float(alpha_min)
        g_det = _detect_gain(x, center) if use_detect else 1.0

        # 4) last-K fade mask
        if K_FADE > 0:
            fade = torch.ones((1, T, 1), device=x.device, dtype=x.dtype)
            fade[:, -min(K_FADE, T):, :] = 0.0
        else:
            fade = 1.0

        # 5) Geo step with detect gain
        y_pre = x - (alpha * g_det) * fade * (x - center)

        # 6) Soft denoise (only where fade > 0)
        if use_denoise:
            y_dn = _soft_denoise(center, y_pre)
            # respect last-K fade: keep original y_pre for the scoring tokens
            y = fade * y_dn + (1.0 - fade) * y_pre
        else:
            y = y_pre

        # 7) return with original tuple shape
        if is_tuple:
            return (y,) + tuple(output[1:])
        else:
            return y

    handle = blocks[layer_idx].register_forward_hook(lambda m, inp, out: ngf_forward_hook(m, inp, out))
    if not hasattr(model, "_ngf_handles"):
        model._ngf_handles = []
    model._ngf_handles.append(handle)
    return {"status": "attached", "layer_idx": layer_idx, **cfg}
