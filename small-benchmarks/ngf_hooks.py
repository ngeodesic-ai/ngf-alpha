# ngf_hooks.py — Stage-1: Geo (warp-only at tap -9)
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
    alpha0: float = 0.05,
    alpha_min: float = 0.006,
    trend_tau: float = 0.32,  # v4b profile
    k_tr: int = 12,
    # parsed but UNUSED in Stage-1 (reserved for next stages):
    use_detect: int = 0, detect_width: int = 24, detect_sigma: float = 5.0,
    null_K: int = 32, null_q: float = 0.92, k_det: int = 7,
    s_latch: float = 0.30, linger: int = 2,
    ema_center_beta: float = 0.05,
    gen_mode: str = "geo",
):
    layer_idx = _resolve_tap(model, tap)
    blocks = model.transformer.h
    if not (0 <= layer_idx < len(blocks)):
        raise ValueError(f"tap {tap} → layer {layer_idx} is out of range (0..{len(blocks)-1})")

    cfg = dict(
        tap=tap, alpha0=alpha0, alpha_min=alpha_min,
        trend_tau=trend_tau, k_tr=k_tr, s_latch=s_latch,
        linger=linger, ema_center_beta=ema_center_beta, gen_mode=gen_mode
    )
    print(f"[NGF] Hooking GPT-2 layer {layer_idx} (tap={tap}) [Stage-1: Geo] cfg={cfg}")

    state = {
        "ema_center": None,  # (C,)
        "burst": 0,          # short latch/linger counter
    }

    def ngf_forward_hook(module, inputs, output):
        # HF GPT-2 blocks can return Tuple[Tensor, ...]
        is_tuple = isinstance(output, (tuple, list))
        x = output[0] if is_tuple else output  # (B, T, C)
        B, T, C = x.shape

        # 1) EMA center (global, per call)
        seq_mean = x.mean(dim=1).mean(dim=0)  # (C,)
        if state["ema_center"] is None:
            state["ema_center"] = seq_mean.detach()
        else:
            state["ema_center"] = ema_center_beta * seq_mean.detach() + (1.0 - ema_center_beta) * state["ema_center"]
        center = state["ema_center"].view(1, 1, C)

        # 2) Soft trend gate from last-k token movement
        k = min(k_tr, T)
        dx = x[:, -k:, :] - center
        step_mag = dx.pow(2).sum(dim=-1).mean(dim=1)  # (B,)
        gap = (step_mag.mean() - step_mag.median()).clamp(min=0)
        g_tr = torch.sigmoid(gap / (trend_tau + 1e-6))  # scalar (0,1)

        # 3) Latch/linger (short bursts ≈2–3 tokens)
        if g_tr > 0.5:
            state["burst"] = min(3, state["burst"] + 1)
        else:
            state["burst"] = max(0, state["burst"] - 1)

        # 4) Alpha schedule (always-on inward step)
        alpha = float(alpha_min + (alpha0 - alpha_min) * g_tr.item())
        if state["burst"] > 0:
            alpha = max(alpha, alpha_min + 0.5 * (alpha0 - alpha_min))

        # 5) Geometric inward step (no detect/denoise in Stage-1)
        y = x - alpha * (x - center)

        # Re-wrap tuple if needed
        if is_tuple:
            return (y,) + tuple(output[1:])
        else:
            return y

    handle = blocks[layer_idx].register_forward_hook(
        lambda m, inp, out: ngf_forward_hook(m, inp, out)
    )
    if not hasattr(model, "_ngf_handles"):
        model._ngf_handles = []
    model._ngf_handles.append(handle)
    return {"status": "attached", "layer_idx": layer_idx, **cfg}
