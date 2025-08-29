# =========================  NUDGE INTEGRATION  ================================
# Drop this anywhere above your evaluation loop in arc_benchmark_symbolic_nudge_fusion.py

from dataclasses import dataclass
from contextlib import contextmanager
import torch
import torch.nn as nn

# ---------------------- Config (edit or wire to argparse) ---------------------
NUDGE_ENABLE   = True         # <— set False to disable nudge quickly
NUDGE_LAYER    = -2           # which transformer block (0-based, negatives OK)
NUDGE_PART     = "mlp_in"     # one of: "attn_in","attn_out","mlp_in","mlp_out"
NUDGE_ALPHA    = 0.90         # strength of the injected direction (post-normalized)
NUDGE_MODE     = "in"         # "in" -> pre-hook (inputs), "out" -> forward-hook (outputs)
NUDGE_DIAG     = False        # print before/after tensor stats at the hook point

# --- Symbolic loop params (same spirit as v1) ---
PULL_STRENGTH  = 1.5
GAMMA          = 0.3
SYMB_STEPS     = 40
DT             = 0.05

# Optional: label set for 7 ops if you rely on lm_head letters
LETTER_TOKENS = list("ABCDEFG")  # map to your op classes if you use lm_head

# ---------------------------- Helper utilities --------------------------------
def _is_gpt2(model):
    return hasattr(model, "transformer") and hasattr(model.transformer, "h")

def _is_llama(model):
    return hasattr(model, "model") and hasattr(model.model, "layers")

def _get_block(model, layer_idx):
    if _is_gpt2(model):
        return model.transformer.h[layer_idx]
    elif _is_llama(model):
        return model.model.layers[layer_idx]
    raise ValueError("Unsupported model structure (expect GPT-2/LLaMA-style).")

def _get_target_module(block, part):
    """part in {"attn_in","attn_out","mlp_in","mlp_out"}."""
    attn = getattr(block, "attn", None) or getattr(block, "self_attn", None)
    mlp  = getattr(block, "mlp",  None) or getattr(block, "feed_forward", None)
    if part.startswith("attn"):
        if attn is None: raise ValueError("No attention submodule on this block.")
        return attn
    if part.startswith("mlp"):
        if mlp is None: raise ValueError("No MLP/feed-forward submodule on this block.")
        return mlp
    raise ValueError(f"Unknown part: {part}")

def _unit(v, eps=1e-12):
    return v / (v.norm().clamp_min(eps))

def _fmt(x):
    return f"shape={tuple(x.shape)} | mean={x.mean().item():.4f} | std={x.std().item():.4f} | norm={x.norm().item():.4f}"

@dataclass
class NudgeConfig:
    enable: bool      = NUDGE_ENABLE
    layer: int        = NUDGE_LAYER
    part: str         = NUDGE_PART
    alpha: float      = NUDGE_ALPHA
    mode: str         = NUDGE_MODE   # "in" or "out"
    diag: bool        = NUDGE_DIAG

# --------------------------- Install the nudge --------------------------------
@contextmanager
def install_symbolic_nudge(model, layer_idx, part, delta_vec, alpha=1.0, mode="in", diag=False):
    """
    Adds +alpha*unit(delta_vec) to the chosen tensor at a chosen layer/part.
    *_in  = forward_pre_hook (perturb inputs)
    *_out = forward_hook     (perturb outputs)
    """
    block = _get_block(model, layer_idx)
    target = _get_target_module(block, part)

    # Make the direction broadcastable to [B, T, H]
    if delta_vec.dim() == 1:
        delta = delta_vec.view(1, 1, -1).to(next(model.parameters()).device)
    else:
        delta = delta_vec.to(next(model.parameters()).device)

    delta = _unit(delta) * alpha
    handles = []

    def pre_hook(module, inputs):
        x = inputs[0]
        x_nudged = x + delta
        if diag:
            print(f"[NUDGE pre] {module.__class__.__name__} {part} +Δ  {_fmt(delta)}")
            print(f"  in  before: {_fmt(x)}")
            print(f"  in  after : {_fmt(x_nudged)}")
        return (x_nudged,) + tuple(inputs[1:])

    def fwd_hook(module, inputs, output):
        y = output
        y_nudged = y + delta
        if diag:
            print(f"[NUDGE fwd] {module.__class__.__name__} {part} +Δ  {_fmt(delta)}")
            print(f"  out before: {_fmt(y)}")
            print(f"  out after : {_fmt(y_nudged)}")
        return y_nudged

    if part.endswith("_in") or mode == "in":
        handles.append(target.register_forward_pre_hook(pre_hook, with_kwargs=False))
    if part.endswith("_out") or mode == "out":
        handles.append(target.register_forward_hook(fwd_hook, with_kwargs=False))

    try:
        if diag:
            print(f"[install_symbolic_nudge] layer={layer_idx} | part={part} | alpha={alpha} | delta={tuple(delta.shape)}")
        yield
    finally:
        for h in handles:
            h.remove()
        if diag:
            print("[install_symbolic_nudge] removed.")

# ----------------------- Build the symbolic direction -------------------------
def _get_hidden_size(model):
    if hasattr(model.config, "hidden_size"): return model.config.hidden_size
    if hasattr(model.config, "n_embd"): return model.config.n_embd
    raise ValueError("Could not infer hidden size from model.config.")

def _find_op_head(model):
    # Try common names
    if hasattr(model, "op_head") and isinstance(model.op_head, nn.Linear):
        return model.op_head
    # Some codebases place it under a submodule
    for name in ["classifier", "op_classifier", "arc_head"]:
        m = getattr(model, name, None)
        if isinstance(m, nn.Linear) and m.out_features in (7, 8):  # 7 ops + maybe 'other'
            return m
    return None

def _letter_token_ids(tokenizer):
    if tokenizer is None: return None
    ids = []
    for ch in LETTER_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(ch) if hasattr(tokenizer, "convert_tokens_to_ids") else tokenizer.encode(ch, add_special_tokens=False)
        if isinstance(tid, list): tid = tid[0]
        ids.append(tid)
    return torch.tensor(ids, dtype=torch.long)

def _direction_from_heads(model, tokenizer, target_idx=0, alt_idx=1):
    """
    Try to get a semantic direction from model heads:
    - Prefer op_head: use weight difference W[target]-W[alt] mapped back to hidden space.
    - Else fallback to lm_head letter rows for 'A'..'G'.
    - Else random.
    """
    H = _get_hidden_size(model)
    device = next(model.parameters()).device

    # 1) op_head
    op_head = _find_op_head(model)
    if op_head is not None and hasattr(op_head, "weight"):
        W = op_head.weight.data.to(device)   # [num_ops, H]
        if target_idx >= W.size(0) or alt_idx >= W.size(0):
            target_idx, alt_idx = 0, min(1, W.size(0)-1)
        return (W[target_idx] - W[alt_idx]).detach()

    # 2) lm_head with letter tokens
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "weight") and tokenizer is not None:
        letter_ids = _letter_token_ids(tokenizer)
        if letter_ids is not None:
            letter_ids = letter_ids.to(lm_head.weight.device)
            if target_idx >= len(letter_ids) or alt_idx >= len(letter_ids):
                target_idx, alt_idx = 0, min(1, len(letter_ids)-1)
            return (lm_head.weight[letter_ids[target_idx]] - lm_head.weight[letter_ids[alt_idx]]).detach()

    # 3) random fallback
    return torch.randn(H, device=device)

def build_symbolic_delta(model, tokenizer,
                         target_idx: int,
                         alt_idx: int,
                         pull_strength=PULL_STRENGTH,
                         gamma=GAMMA,
                         steps=SYMB_STEPS,
                         dt=DT):
    """
    Very lightweight v1-style refinement:
    Start with a head-derived direction; then do a tiny momentum-ish update.
    This does NOT backprop through the full model — it’s a cheap proxy.
    """
    d = _direction_from_heads(model, tokenizer, target_idx, alt_idx)  # [H]
    v = torch.zeros_like(d)
    d = _unit(d)

    for _ in range(steps):
        # Simple self-reinforcing update with decay:
        #   v <- gamma*v + pull_strength*d
        #   d <- unit(d + dt * v)
        v = gamma * v + pull_strength * d
        d = _unit(d + dt * v)

    return d.detach()  # [H]

# ----------------------- One-line context for your loop -----------------------
@contextmanager
def nudge_ctx(model, tokenizer, cfg: NudgeConfig, target_idx: int, alt_idx: int):
    """
    Computes the delta once per call, then installs the hook during the context.
    target_idx/alt_idx are indices in your op set (or letter set A..G).
    """
    if not cfg.enable:
        yield
        return

    delta_vec = build_symbolic_delta(
        model, tokenizer,
        target_idx=target_idx, alt_idx=alt_idx,
        pull_strength=PULL_STRENGTH, gamma=GAMMA,
        steps=SYMB_STEPS, dt=DT
    )
    with install_symbolic_nudge(
            model, layer_idx=cfg.layer, part=cfg.part,
            delta_vec=delta_vec, alpha=cfg.alpha,
            mode=cfg.mode, diag=cfg.diag):
        yield

# =======================  END NUDGE INTEGRATION  ==============================
