# ngf_hooks.py
# Minimal adapter to attach an NGF-style hook at a GPT-2 layer (tap).
# You can replace the NO-OP with your warp/detect/denoise call.

from typing import Optional
import torch

def _resolve_tap(model, tap: int) -> int:
    # GPT-2: model.transformer.h = list of blocks
    n = len(model.transformer.h)
    return tap + n if tap < 0 else tap  # e.g., -9 on 12-layer → 3

def attach_ngf_hooks(
    model,
    tokenizer=None,
    device: Optional[torch.device] = None,
    *,
    tap: int = -9,
    alpha0: float = 0.05,
    alpha_min: float = 0.006,
    trend_tau: float = 0.35,
    k_tr: int = 12,
    use_detect: int = 1,
    detect_width: int = 24,
    detect_sigma: float = 5.0,
    null_K: int = 32,
    null_q: float = 0.92,
    k_det: int = 7,
    s_latch: float = 0.30,
    linger: int = 2,
    ema_center_beta: float = 0.05,
    gen_mode: str = "geo",
):
    """
    Attaches a forward hook at the requested tap that can modify hidden states.
    Replace the NO-OP section with your NGF warp/detect/denoise.
    """
    layer_idx = _resolve_tap(model, tap)
    blocks = model.transformer.h
    if not (0 <= layer_idx < len(blocks)):
        raise ValueError(f"tap {tap} → layer {layer_idx} is out of range (0..{len(blocks)-1})")

    cfg = dict(
        tap=tap, alpha0=alpha0, alpha_min=alpha_min, trend_tau=trend_tau, k_tr=k_tr,
        use_detect=use_detect, detect_width=detect_width, detect_sigma=detect_sigma,
        null_K=null_K, null_q=null_q, k_det=k_det, s_latch=s_latch, linger=linger,
        ema_center_beta=ema_center_beta, gen_mode=gen_mode,
    )

    print(f"[NGF] Hooking GPT-2 layer {layer_idx} (tap={tap}) with cfg={cfg}")

    def ngf_forward_hook(module, inputs, output):
        """
        inputs: tuple with (hidden_states) for GPT-2 blocks
        output: hidden_states tensor (B, T, C) — we can modify and return it.
        """
        # --------- NGF PLACEHOLDER (NO-OP) ----------
        # TODO: replace with your warp/detect/denoise using `cfg`.
        # For sanity, we leave the tensor unchanged:
        return output
        # Example mild debug (uncomment if needed):
        # if torch.rand(1).item() < 0.001:
        #     print("[NGF] hook fired:", tuple(output.shape))

    # Register the hook
    handle = blocks[layer_idx].register_forward_hook(
        lambda m, inp, out: ngf_forward_hook(m, inp, out)
    )
    # Keep a reference so it isn't GC’ed
    if not hasattr(model, "_ngf_handles"):
        model._ngf_handles = []
    model._ngf_handles.append(handle)

    return {"status": "attached", "layer_idx": layer_idx, **cfg}
