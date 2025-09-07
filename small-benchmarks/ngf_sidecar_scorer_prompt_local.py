# Write a concrete NGF sidecar scorer that reuses your Stage-11 warp from stage11_benchmark_latest.py
# It implements a prompt-local batch scorer suitable for HellaSwag (4 endings) and top-k token steering.
# It exposes:
#   - score_latents_prompt_local(H, cfg): batch scores + pick + meta
#   - score_latent(vec): single-vector shim (falls back to trivial gate; mainly for API parity)
# and a tiny CLI demo if run as a script.

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

# Reuse Stage-11 warp + params
from stage11_benchmark_latest import pca3_and_warp, WellParams  # noqa: E402

@dataclass
class NGFScore:
    depth: float
    margin: float
    conf: float
    gates_ok: bool

@dataclass
class NGFPromptMeta:
    phantom_index: float
    margin_prompt: float

def _squash_conf(x: np.ndarray) -> np.ndarray:
    x = (x - x.mean()) / (x.std() + 1e-9)
    return 1.0 / (1.0 + np.exp(-3.5 * x))

def score_latents_prompt_local(
    H: np.ndarray,
    well_cfg: Optional[Dict] = None,
    gamma: float = 0.5,
    zq_gate: float = 0.25,
    tau_phantom: float = 0.12,
    tau_margin: float = 0.06,
) -> Tuple[List[Dict], int, Dict]:
    """
    Prompt-local NGF scorer (no prefit). Use for HellaSwag (M=4) or top-k tokens.
      H: (M, D) candidate vectors for one prompt
      well_cfg: overrides for WellParams(...)
      gamma: margin weight in composite
      zq_gate: candidate passes if its depth is in the best (1 - zq_gate) quantile (default top-75% depth)
      tau_phantom, tau_margin: prompt-level gates

    Returns:
      scores: list of dict(depth, margin, gates_ok, conf) per candidate
      pick: index of chosen candidate
      meta: {'phantom_index', 'margin_prompt'}
    """
    if H.ndim != 2 or H.shape[0] < 1:
        raise ValueError("H must be (M,D) with M >= 1")
    # 1) Warp batch into single-well coords
    params = WellParams(**(well_cfg or {}))
    X3, metrics, _info = pca3_and_warp(H, energy=None, params=params)
    z = X3[:, 2]                               # more negative is deeper
    order = np.argsort(z)                      # smallest z is best
    best = order[0]
    runner = order[1] if len(order) > 1 else order[0]
    margin_prompt = float(z[runner] - z[best]) # positive if best is clearly deeper

    # 2) Per-candidate: flip & normalize depth
    depth = -z
    depth = (depth - depth.min()) / (depth.ptp() + 1e-9)

    # Margin relative to the best (0 for best, >0 otherwise)
    gaps = z - z[best]

    # 3) Prompt- and candidate-level gates
    gate_prompt = (metrics.get("phantom_index", 0.0) >= tau_phantom) and (margin_prompt >= tau_margin)
    z_thresh = float(np.quantile(z, zq_gate))  # allow candidates with small z (deep)
    gates_ok = [(gate_prompt and (z[i] <= z_thresh)) for i in range(len(z))]

    # 4) Composite & pick (prefer gate-passing; else best composite anyway)
    comp = depth + gamma * gaps
    if any(gates_ok):
        masked = np.where(gates_ok, comp, -1e9)
        pick = int(np.argmax(masked))
    else:
        pick = int(np.argmax(comp))

    conf = _squash_conf(depth)

    scores = [dict(depth=float(depth[i]),
                   margin=float(gaps[i]),
                   gates_ok=bool(gates_ok[i]),
                   conf=float(conf[i])) for i in range(len(z))]
    meta = dict(phantom_index=float(metrics.get("phantom_index", 0.0)),
                margin_prompt=float(margin_prompt))
    return scores, pick, meta

# Optional single-vector shim (mainly for API parity with earlier harnesses)
def score_latent(vec: np.ndarray) -> NGFScore:
    # Without a cohort to establish local geometry, we can only report a neutral score.
    # Use score_latents_prompt_local with a batch whenever possible.
    return NGFScore(depth=0.0, margin=0.0, conf=0.5, gates_ok=True)

# --------------- CLI demo (toy) ---------------
def _demo():
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=4, help="num candidates")
    parser.add_argument("--D", type=int, default=64, help="latent dim")
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    # Build a toy batch with one deeper candidate
    H = rng.normal(size=(args.M, args.D))
    H[0] += 0.8  # make candidate 0 a bit 'deeper'
    scores, pick, meta = score_latents_prompt_local(H)
    print("pick:", pick, "meta:", meta)
    for i, s in enumerate(scores):
        print(i, s)

if __name__ == "__main__":
    _demo()
