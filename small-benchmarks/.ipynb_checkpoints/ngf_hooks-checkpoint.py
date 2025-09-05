"""
# ==============================================================================
# Apache 2.0 License (ngeodesic.ai)
# ==============================================================================
# Copyright 2025 Ian C. Moore (Provisional Patents #63/864,726, #63/865,437, #63/871,647 and #63/872,334)
# Email: ngeodesic@gmail.com
# Part of Noetic Geodesic Framework (NGF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import math
import os
import atexit

import torch
import torch.nn as nn
import numpy as np


@torch.no_grad()
def attach_ngf_hooks(
    model: nn.Module,
    tokenizer=None,
    device: Optional[torch.device] = None,
    *,
    # Tap selection (negative = from top, like -9)
    tap: int = -9,

    # Stage-11 geo defaults (v4b-ish)
    alpha0: float = 0.05,
    alpha_min: float = 0.006,
    trend_tau: float = 0.35,
    k_tr: int = 12,

    # Detect (gain-only)
    use_detect: int = 1,
    detect_width: int = 24,
    detect_sigma: float = 5.0,
    null_K: int = 32,
    null_q: float = 0.92,
    k_det: int = 7,

    # Latch/linger + EMA for center
    s_latch: float = 0.30,
    linger: int = 2,
    ema_center_beta: float = 0.05,

    # Decode mode (kept for compatibility with your harness)
    gen_mode: str = "geo",

    # NEW: detect cap + tail fade are configurable (no override inside)
    g_det_max: float = 1.6,
    K_FADE: int = 4,

    # Noise / guards
    jitter_eps: float = 0.00,            # (fixed duplicate kwarg bug: appears only once)
    phantom_guard_gamma: float = 0.25,

    # Telemetry / capture for visualization
    save_hidden: int = 0,                # 1 => save pre/post at tap layer to disk
    hidden_dump_dir: str = "",
    hidden_cap: int = 50_000,            # max vectors stored per buffer (pre/post)
) -> str:
    """
    Returns a string describing attach status. On success, registers a forward hook at the tap layer.
    """

    # ---- Resolve tap layer index ----
    # GPT-2: blocks under model.transformer.h (length = n_layer)
    blocks = getattr(getattr(model, "transformer", model), "h", None)
    if blocks is None or not isinstance(blocks, (list, nn.ModuleList)):
        return "[NGF] attach failed: could not find model.transformer.h blocks"

    n_layers = len(blocks)
    tap_idx = tap if tap >= 0 else (n_layers + tap)
    if tap_idx < 0 or tap_idx >= n_layers:
        return f"[NGF] attach failed: tap={tap} resolved to invalid index {tap_idx} (n_layers={n_layers})"

    tap_mod = blocks[tap_idx]

    # We hook the block *output* hidden states. GPT-2 block forward returns (hidden_states, *extras)
    # So a standard forward_hook on the block will let us intercept the (B,T,C) tensor.
    # We'll be careful to preserve tuple shape if the block returns a tuple.

    # ---- Shared state for the hook ----
    state: Dict[str, Any] = {
        "ema_center": None,      # (C,)
        "ema_norm": None,        # scalar EMA for phantom guard
        "warmup": 0,
        "linger": 0,
        "latched": 0.0,          # for s_latch behavior
        "null_mu": None,         # detect null stats
        "null_sigma": None,
        "n_vecs_pre": 0,
        "n_vecs_post": 0,
        "pre_buf": [],
        "post_buf": [],
    }

    # ---- Helpers ----

    def _flatten_bt(x: torch.Tensor) -> torch.Tensor:
        # (B,T,C) -> (B*T, C)
        return x.reshape(-1, x.shape[-1])

    def _ema_update(prev: Optional[torch.Tensor], x: torch.Tensor, beta: float) -> torch.Tensor:
        if prev is None:
            return x.clone()
        return (1.0 - beta) * prev + beta * x

    def _compute_center(x: torch.Tensor) -> torch.Tensor:
        # center over batch*time on channel dim
        # x: (B,T,C)
        m = x.mean(dim=(0,1))
        return m

    def _trend_gate(x: torch.Tensor, center: torch.Tensor) -> float:
        # Soft "wobble" detector in [0,1].
        # We normalize average radial deviation by a scale tied to trend_tau.
        # Simple, smooth, and robust for benchmarking.
        r = x - center
        # average L2 over tokens, then soft map to [0,1]
        avg_norm = r.norm(dim=-1).mean().item()
        # Map via logistic so it expands near trend_tau
        # g_tr ≈ 0.5 when avg_norm ≈ trend_tau
        k = 8.0  # slope for logistic; not too sharp
        g = 1.0 / (1.0 + math.exp(-k * (avg_norm - trend_tau)))
        return float(max(0.0, min(1.0, g)))

    def _detect_gain(x: torch.Tensor, center: torch.Tensor) -> float:
        # Matched-filter-like soft gain; bounded by g_det_max.
        # Null stats are tracked on the *centered* norms.
        # Keep it conservative; return 1.0 when unused.
        r = x - center
        # mean radial energy over tokens
        z = r.norm(dim=-1).mean()  # scalar tensor
        # Update null stats with EMA (robust)
        if state["null_mu"] is None:
            state["null_mu"] = z.detach()
            state["null_sigma"] = z.detach() * 0.1 + z.detach().new_tensor(1e-6)
        else:
            state["null_mu"] =  (1.0 - null_q) * state["null_mu"] + null_q * z.detach()
            state["null_sigma"] = (1.0 - null_q) * state["null_sigma"] + null_q * (z.detach() - state["null_mu"]).abs().clamp_min(1e-6)

        # Convert to soft gain: >1 when above null mean by ~detect_sigma*sigma
        mu = state["null_mu"]
        sigma = state["null_sigma"].clamp_min(1e-6)
        raw = ((z - mu) / (detect_sigma * sigma)).clamp(min=-6.0, max=6.0)
        # squash to [0,1], then map to [1, g_det_max]
        s = torch.sigmoid(raw).item()
        g = 1.0 + s * (g_det_max - 1.0)
        return float(g)

    def _phantom_guard(scale: float) -> float:
        # Damp scale when EMA norm is high (very conservative).
        en = state["ema_norm"]
        if en is None:
            return scale
        # map en to [0,1] roughly; clamp then apply gamma
        damp = 1.0 / (1.0 + en.item())
        return scale * (1.0 - phantom_guard_gamma * (1.0 - damp))

    def _maybe_capture(tag: str, x: torch.Tensor):
        # x: (B,T,C). Capture flattened vectors with caps.
        if not save_hidden or not hidden_dump_dir:
            return
        vecs = _flatten_bt(x.detach()).cpu()
        if tag == "pre":
            rem = max(0, hidden_cap - state["n_vecs_pre"])
            if rem > 0:
                state["pre_buf"].append(vecs[:rem])
                state["n_vecs_pre"] += min(rem, vecs.shape[0])
        else:
            rem = max(0, hidden_cap - state["n_vecs_post"])
            if rem > 0:
                state["post_buf"].append(vecs[:rem])
                state["n_vecs_post"] += min(rem, vecs.shape[0])

    def _flush_hidden():
        if not save_hidden or not hidden_dump_dir:
            return
        os.makedirs(hidden_dump_dir, exist_ok=True)
        if state["pre_buf"]:
            pre = torch.cat(state["pre_buf"], dim=0).numpy()
            np.save(os.path.join(hidden_dump_dir, "tap9_pre.npy"), pre)
        if state["post_buf"]:
            post = torch.cat(state["post_buf"], dim=0).numpy()
            np.save(os.path.join(hidden_dump_dir, "tap9_post.npy"), post)

    atexit.register(_flush_hidden)

    # ---- Forward hook ----
    def ngf_forward_hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output):
        # Output may be (hidden_states, presents, attentions, ...)
        is_tuple = isinstance(output, (tuple, list))
        x = output[0] if is_tuple else output  # (B,T,C)
        if x is None or not isinstance(x, torch.Tensor):
            return output

        # Ensure float
        if x.dtype in (torch.float16, torch.bfloat16):
            x = x.to(torch.float32)

        # Set up EMA center
        center = state["ema_center"]
        cur_center = _compute_center(x)
        state["ema_center"] = _ema_update(center, cur_center, ema_center_beta)
        center = state["ema_center"]

        # Tail fade (protect last-K tokens for MC scoring tasks)
        if K_FADE > 0 and x.shape[1] >= 1:
            fade = torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
            if K_FADE > 0:
                fade[:, -K_FADE:] = 0.0
            fade = fade.unsqueeze(-1)  # (B,T,1)
        else:
            fade = 1.0

        # Trend gate in [0,1]
        g_tr = _trend_gate(x, center)

        # Base alpha with trend gating
        alpha = max(alpha_min, alpha0 * g_tr)

        # Optional detect gain
        g_det = 1.0
        if use_detect:
            g_det = _detect_gain(x, center)
            g_det = float(min(g_det, g_det_max))

        # Phantom guard (very conservative scaling)
        r = x - center
        norm = r.norm(dim=-1).mean()
        state["ema_norm"] = _ema_update(state["ema_norm"], norm, 0.05)
        alpha = _phantom_guard(alpha)

        # Optional tiny jitter to avoid stickiness (never required)
        if jitter_eps > 0.0:
            x = x + jitter_eps * torch.randn_like(x)

        # ---- Pre-capture
        _maybe_capture("pre", x)

        # ---- Geo step (never flips inward direction)
        # y = x - (alpha * g_det) * fade * (x - center)
        scale = (alpha * g_det)
        y = x - scale * (r * fade)

        # (Optional) extremely soft denoise (no sign flip; scale residuals slightly)
        # You can tune this to your liking; kept gentle by default.
        # Here we leave it as identity to keep the hook simple and safe.
        # y = center + (y - center)

        # ---- Post-capture
        _maybe_capture("post", y)

        # Return with original container type
        if is_tuple:
            # Rebuild tuple with y in slot 0 to preserve any extra outputs
            return (y,) + tuple(output[1:])
        else:
            return y

    # Register the forward hook
    hook_handle = tap_mod.register_forward_hook(ngf_forward_hook)

    # Pretty banner so your harness can surface it in logs/json
    status = (f"[NGF] NGF attached via ngf_hooks:attach_ngf_hooks at block index {tap_idx} "
              f"(tap={tap}) with cfg={{"
              f"alpha0={alpha0}, alpha_min={alpha_min}, trend_tau={trend_tau}, k_tr={k_tr}, "
              f"use_detect={use_detect}, detect_width={detect_width}, detect_sigma={detect_sigma}, "
              f"null_K={null_K}, null_q={null_q}, k_det={k_det}, "
              f"s_latch={s_latch}, linger={linger}, ema_center_beta={ema_center_beta}, "
              f"gen_mode='{gen_mode}', g_det_max={g_det_max}, K_FADE={K_FADE}, "
              f"phantom_guard_gamma={phantom_guard_gamma}, save_hidden={save_hidden}, "
              f"hidden_dump_dir='{hidden_dump_dir}', hidden_cap={hidden_cap}}}")
    print(status, flush=True)
    return status
