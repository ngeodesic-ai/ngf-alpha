# ngf_hooks.py — Stage-2: Geo + Detect (gain-only) at tap -9
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
    # geo (v4b defaults)
    alpha0: float = 0.05,
    alpha_min: float = 0.006,
    trend_tau: float = 0.32,
    k_tr: int = 12,
    s_latch: float = 0.30,
    linger: int = 2,
    ema_center_beta: float = 0.05,
    gen_mode: str = "geo",
    # detect (v4b defaults)
    use_detect: int = 1,
    detect_width: int = 20,     # v4b: 20
    null_K: int = 32,           # number of frames to seed null
    null_q: float = 0.92,       # EMA factor for null
    k_det: int = 8,             # gain sharpness
    detect_sigma: float = 5.0,  # reserved / unused in minimal variant
    # denoise reserved for Stage-3
    use_denoise: int = 0,
):
    layer_idx = _resolve_tap(model, tap)
    blocks = model.transformer.h
    if not (0 <= layer_idx < len(blocks)):
        raise ValueError(f"tap {tap} → layer {layer_idx} is out of range (0..{len(blocks)-1})")

    cfg = dict(
        tap=tap, alpha0=alpha0, alpha_min=alpha_min, trend_tau=trend_tau, k_tr=k_tr,
        s_latch=s_latch, linger=linger, ema_center_beta=ema_center_beta, gen_mode=gen_mode,
        use_detect=use_detect, detect_width=detect_width, null_K=null_K, null_q=null_q, k_det=k_det
    )
    print(f"[NGF] Hooking GPT-2 layer {layer_idx} (tap={tap}) [Stage-2: Geo+Detect] cfg={cfg}")

    # Internal state
    state = {
        "ema_center": None,   # (C,)
        "burst": 0,           # unused for Stage-2 (kept for continuity)
        # detect state
        "null_std": None,     # scalar EMA baseline of motion/variance
        "warmup": 0,          # counts frames for null seed
    }

    # Fade warp on final tokens so MC option scoring stays discriminative
    K_FADE = 8  # 6–12 is OK for HellaSwag

    def _compute_trend_gate(x, center):
        # last-k trend gate used in Stage-1 (kept but not used to scale alpha now)
        B, T, C = x.shape
        k = min(k_tr, T)
        dx = x[:, -k:, :] - center
        step_mag = dx.pow(2).sum(dim=-1).mean(dim=1)  # (B,)
        gap = (step_mag.mean() - step_mag.median()).clamp(min=0)
        g_tr = torch.sigmoid(gap / (trend_tau + 1e-6))  # scalar (0,1)
        return g_tr

    def _detect_gain(x, center):
        """
        Matched-variance detect (gain-only):
        - Compute local motion magnitude over last `detect_width` tokens.
        - Compare to null_std (EMA baseline). If above, amplify warp.
        g_det = 1 + tanh(k_det * max(0, (cur - null)/null))
        """
        B, T, C = x.shape
        w = min(max(detect_width, 1), T)
        # motion magnitude relative to center over the local window
        dx = x[:, -w:, :] - center
        mot = dx.pow(2).sum(dim=-1).sqrt()     # (B, w)
        cur_std = mot.mean().detach()          # scalar

        # initialize / warmup null
        if state["null_std"] is None:
            state["null_std"] = cur_std
            state["warmup"] = 1
        else:
            if state["warmup"] < null_K:
                # seed null from the first K windows (simple average)
                state["null_std"] = (state["null_std"] * state["warmup"] + cur_std) / (state["warmup"] + 1)
                state["warmup"] += 1
            else:
                # EMA update thereafter
                state["null_std"] = null_q * state["null_std"] + (1.0 - null_q) * cur_std

        # gain-only amplification
        base = state["null_std"].clamp(min=1e-6)
        delta = (cur_std - base).clamp(min=0.0) / base
        g_det = 1.0 + torch.tanh(torch.tensor(k_det, dtype=cur_std.dtype, device=cur_std.device) * delta).item()
        return float(g_det)

    def ngf_forward_hook(module, inputs, output):
        # HF GPT-2 blocks can return Tuple[Tensor, ...]
        is_tuple = isinstance(output, (tuple, list))
        x = output[0] if is_tuple else output  # (B, T, C)
        B, T, C = x.shape

        # 1) EMA center
        seq_mean = x.mean(dim=1).mean(dim=0)  # (C,)
        if state["ema_center"] is None:
            state["ema_center"] = seq_mean.detach()
        else:
            state["ema_center"] = ema_center_beta * seq_mean.detach() + (1.0 - ema_center_beta) * state["ema_center"]
        center = state["ema_center"].view(1, 1, C)

        # 2) Trend gate (kept for telemetry; we’ll hold alpha constant to avoid washout)
        _ = _compute_trend_gate(x, center)

        # 3) Base warp strength (conservative for MC scoring)
        alpha = float(alpha_min)  # constant small pull; detect will scale it up when coherent

        # 4) Detect gain (gain-only)
        g_det = 1.0
        if use_detect:
            g_det = _detect_gain(x, center)

        # 5) Last-K fade so answer-ending tokens aren’t warped
        if K_FADE > 0:
            fade = torch.ones((1, T, 1), device=x.device, dtype=x.dtype)
            fade[:, -min(K_FADE, T):, :] = 0.0
        else:
            fade = 1.0

        # 6) Apply geo step with detect gain
        y = x - (alpha * g_det) * fade * (x - center)

        # 7) Re-wrap tuple if needed
        if is_tuple:
            return (y,) + tuple(output[1:])
        else:
            return y

    handle = blocks[layer_idx].register_forward_hook(lambda m, inp, out: ngf_forward_hook(m, inp, out))
    if not hasattr(model, "_ngf_handles"):
        model._ngf_handles = []
    model._ngf_handles.append(handle)
    return {"status": "attached", "layer_idx": layer_idx, **cfg}
