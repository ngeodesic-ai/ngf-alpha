# Rewrite with outer triple single-quoted string to avoid nested quote conflicts.
# code = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 11 — Well Parser Benchmark (v1)
------------------------------------
This script upgrades the Stage-10 hidden-target benchmark into an explicit
**warped manifold energy** parser as specified in Stage-11 (Well Parser).

Key ideas implemented (Steps 1–7):
1) Energy Landscape:
   U_p = - (w_perp * z_perp[p] + w_raw * z_raw[p] - w_cm * z_cm[p])
   Lower U is "deeper well" => more likely true primitive.
   We track the margin Δ = U_false - U_true during descent.

2) Improved Orthogonalization:
   Replace mean-subtraction with low-rank projection removal.
   We compute the top-r principal components across channels (time × K matrix)
   and project each channel onto the **span complement** to remove shared variance.

3) Stronger Calibration:
   Use a block-circular permutation null (preserves autocorrelation) to compute
   z-scores. Optionally average correlations under multiple tapers (Hann/Hamming/
   Blackman) to stabilize variance ("multitaper").

4) Sequential Residual Refinement:
   After selecting a primitive, subtract its aligned prototype contribution from
   all channels around the detected window, then recompute z-scores. True picks
   deepen the well; false picks flatten.

5) Lateral Inhibition:
   Penalize overlapping windows via a Gaussian kernel κ(t_i, t_j), discouraging
   distractor wells near already chosen peaks.

6) Controlled Descent:
   Select primitives with an annealed softmax over -U/T. Start hot (explore),
   anneal cold (exploit). Deterministic if --temp_min is very small.

7) Evaluation & Logging:
   We log margins (per step), hallucination/omission, precision/recall/F1/Jaccard,
   exact grid accuracy, and summary stats to CSV. Optional plots per sample.

Usage (examples):
  python3 stage11-benchmark-v1.py --samples 100 --plots
  python3 stage11-benchmark-v1.py --samples 150 --T 720 --proto_width 180 \
      --rank 2 --block 32 --tapers hann,hamming,blackman --temp0 1.5 --anneal 0.6 \
      --inhib_strength 0.8 --inhib_sigma 40

Author: ngeodesic — Stage 11 (v1)

---------------------------------------------
Step 1
---------------------------------------------
python3 stage11-benchmark-v0.py \
  --samples 50 --sigma 11 --proto_width 180 --rank 3 \
  --nperm 220 --block 48 --tapers hann,hamming,blackman \
  --temp0 1.0 --temp_min 0.10 --anneal 0.50 \
  --inhib_strength 1.1 --inhib_sigma 28 \
  --presence_floor 0.60 \
  --csv stage11_sweep_A.csv

python3 stage11-benchmark-v0.py \
  --samples 50 --sigma 13 --proto_width 190 --rank 4 \
  --nperm 280 --block 56 --tapers hann,hamming,blackman \
  --temp0 0.9 --temp_min 0.08 --anneal 0.45 \
  --inhib_strength 1.4 --inhib_sigma 22 \
  --presence_floor 0.75 \
  --csv stage11_sweep_B.csv

python3 stage11-benchmark-v0.py \
  --samples 50 --plot \
  --sigma 11 --proto_width 180 --rank 3 \
  --nperm 400 --block 64 --tapers hann,hamming,blackman \
  --temp0 1.1 --temp_min 0.12 --anneal 0.55 \
  --inhib_strength 1.0 --inhib_sigma 32 \
  --presence_floor 0.55 \
  --csv stage11_sweep_C.csv

---------------------------------------------
Step 2
---------------------------------------------
python3 stage11-benchmark-v0.py \
  --samples 50 --sigma 11 --proto_width 180 --rank 3 \
  --nperm 400 --block 64 --tapers hann,hamming,blackman \
  --temp0 1.1 --temp_min 0.12 --anneal 0.55 \
  --inhib_strength 1.1 --inhib_sigma 40 \
  --presence_floor 0.55 \
  --weights 0.8,0.4,0.6 \
  --csv stage11_step2_S2-1.csv

python3 stage11-benchmark-v0.py \
  --samples 50 --sigma 11 --proto_width 180 --rank 3 \
  --nperm 400 --block 64 --tapers hann,hamming,blackman \
  --temp0 1.1 --temp_min 0.12 --anneal 0.55 \
  --inhib_strength 1.6 --inhib_sigma 60 \
  --presence_floor 0.55 \
  --weights 0.9,0.4,0.5 \
  --csv stage11_step2_S2-2.csv

python3 stage11-benchmark-v0.py \
  --samples 50 --sigma 11 --proto_width 180 --rank 3 \
  --nperm 400 --block 64 --tapers hann,hamming,blackman \
  --temp0 1.1 --temp_min 0.12 --anneal 0.55 \
  --inhib_strength 1.8 --inhib_sigma 72 \
  --presence_floor 0.55 \
  --weights 0.6,0.45,0.7 \
  --csv stage11_step2_S2-3.csv

---------------------------------------------
Step 3
---------------------------------------------
python3 stage11-benchmark-v3.py \
  --samples 50 --sigma 11 --proto_width 180 --rank 3 \
  --nperm 400 --block 64 \
  --temp0 1.1 --temp_min 0.12 --anneal 0.55 \
  --inhib_strength 1.0 --inhib_sigma 32 --presence_floor 0.55 \
  --proto_modes half,skewL,skewR,triangle --proto_phases 0.0,0.2 \
  --csv stage11_step3_S3-1.csv 

python3 stage11-benchmark-v3.py \
  --samples 50 --sigma 11 --proto_width 180 --rank 3 \
  --nperm 400 --block 64 \
  --temp0 1.1 --temp_min 0.12 --anneal 0.55 \
  --inhib_strength 1.2 --inhib_sigma 40 --presence_floor 0.55 \
  --proto_modes half,skewL,skewR,triangle --proto_phases 0.0,0.2 \
  --raw_floor 0.30 --raw_floor_penalty 0.60 \
  --csv stage11_step3_S3-2.csv

python3 stage11-benchmark-v3.py \
  --samples 50 --sigma 11 --proto_width 180 --rank 3 \
  --nperm 400 --block 64 \
  --temp0 1.1 --temp_min 0.12 --anneal 0.55 \
  --inhib_strength 1.4 --inhib_sigma 48 --presence_floor 0.55 \
  --proto_modes half,skewL,skewR,triangle --proto_phases 0.0,0.2 \
  --raw_floor 0.35 --raw_floor_penalty 0.70 \
  --pick_penalty 0.20 \
  --csv stage11_step3_S3-3.csv

---------------------------------------------
Step 4
---------------------------------------------
python3 stage11-benchmark-v4.py \
  --samples 50 --sigma 11 --proto_width 180 --rank 3 \
  --nperm 400 --block 64 \
  --temp0 1.1 --temp_min 0.12 --anneal 0.55 \
  --inhib_strength 1.2 --inhib_sigma 48 --presence_floor 0.55 \
  --proto_modes half,skewL,skewR,triangle --proto_phases 0.0,0.2 \
  --proto_agg softmin --proto_tau 0.40 \
  --raw_floor_map "flip_v:0.55" \
  --weights_map "flip_h:1.0,0.4,0.3;flip_v:0.9,0.4,0.4;rotate:1.0,0.4,0.3" \
  --consensus_k 1 --consensus_eps 0.15 \
  --csv stage11_step4_S4-1.csv

python3 stage11-benchmark-v4.py \
  --samples 50 --sigma 11 --proto_width 180 --rank 3 \
  --nperm 400 --block 64 \
  --temp0 1.1 --temp_min 0.12 --anneal 0.55 \
  --inhib_strength 1.4 --inhib_sigma 60 --presence_floor 0.55 \
  --proto_modes half,skewL,skewR,triangle --proto_phases 0.0,0.2 \
  --proto_agg softmin --proto_tau 0.45 \
  --raw_floor_map "flip_v:0.60" \
  --residual_ceiling_map "flip_v:1.10" \
  --weights_map "flip_h:1.0,0.4,0.3;flip_v:0.85,0.4,0.45;rotate:1.0,0.4,0.3" \
  --consensus_k 2 --consensus_eps 0.12 \
  --csv stage11_step4_S4-2.csv

python3 stage11-benchmark-v4.py \
  --samples 50 --sigma 11 --proto_width 180 --rank 3 \
  --nperm 400 --block 64 \
  --temp0 1.1 --temp_min 0.12 --anneal 0.55 \
  --inhib_strength 1.6 --inhib_sigma 72 --presence_floor 0.55 \
  --proto_modes half,skewL,skewR,triangle --proto_phases 0.0,0.2 \
  --proto_agg median \
  --raw_floor_map "flip_v:0.62" \
  --weights_map "flip_h:1.0,0.4,0.3;flip_v:0.80,0.45,0.50;rotate:1.0,0.4,0.3" \
  --consensus_k 2 --consensus_eps 0.10 \
  --csv stage11_step4_S4-3.csv

---------------------------------------------
Step 5
---------------------------------------------
python3 stage11-benchmark-v5.py \
  --samples 50 --sigma 11 --proto_width 180 --rank 3 \
  --nperm 400 --block 64 \
  --temp0 1.1 --temp_min 0.12 --anneal 0.55 \
  --inhib_strength 1.2 --inhib_sigma 48 --presence_floor 0.55 \
  --proto_modes_map "flip_h:half,triangle,skewL,skewR;flip_v:half,halfN,halfW,hingeL,hingeR,deriv;rotate:half,triangle,skewR" \
  --proto_phases 0.0,0.2 \
  --proto_agg softmin --proto_tau 0.40 \
  --raw_floor_map "flip_v:0.60" --residual_ceiling_map "flip_v:1.10" \
  --consensus_k_map "flip_v:2" --consensus_eps_map "flip_v:0.12" \
  --ortho_flip_v \
  --csv stage11_step5_S5-1.csv

python3 stage11-benchmark-v5.py \
  --samples 50 --sigma 11 --proto_width 180 --rank 3 \
  --nperm 400 --block 64 \
  --temp0 1.1 --temp_min 0.12 --anneal 0.55 \
  --inhib_strength 1.4 --inhib_sigma 60 --presence_floor 0.55 \
  --proto_modes_map "flip_v:half,hingeL,hingeR,deriv;flip_h:half,triangle;rotate:half,triangle,skewR" \
  --proto_phases 0.0,0.2 \
  --proto_agg median \
  --raw_floor_map "flip_v:0.62" --residual_ceiling_map "flip_v:1.05" \
  --consensus_k_map "flip_v:2" --consensus_eps_map "flip_v:0.10" \
  --ortho_flip_v \
  --csv stage11_step5_S5-2.csv

python3 stage11-benchmark-v5.py \
  --samples 50 --sigma 11 --proto_width 180 --rank 3 \
  --nperm 400 --block 64 \
  --temp0 1.1 --temp_min 0.12 --anneal 0.55 \
  --inhib_strength 1.6 --inhib_sigma 72 --presence_floor 0.55 \
  --proto_modes_map "flip_v:half,halfN,halfW,hingeL,hingeR,deriv;flip_h:half,triangle;rotate:half,triangle" \
  --proto_phases 0.0,0.2 \
  --proto_agg softmin --proto_tau 0.45 \
  --raw_floor_map "flip_v:0.65" --residual_ceiling_map "flip_v:1.00" \
  --consensus_k_map "flip_v:3" --consensus_eps_map "flip_v:0.10" \
  --ortho_flip_v \
  --csv stage11_step5_S5-3.csv

---------------------------------------------
Step 6
---------------------------------------------
python3 stage11-benchmark-v5.py \
  --samples 100 --sigma 11 --proto_width 180 --rank 3 \
  --nperm 400 --block 64 \
  --temp0 1.1 --temp_min 0.12 --anneal 0.55 \
  --inhib_strength 1.6 --inhib_sigma 72 --presence_floor 0.55 \
  --proto_modes_map "flip_v:half,halfN,halfW,hingeL,hingeR,deriv;flip_h:half,triangle;rotate:half,triangle" \
  --proto_phases 0.0,0.2 \
  --proto_agg softmin --proto_tau 0.45 \
  --raw_floor_map "flip_v:0.65" --residual_ceiling_map "flip_v:1.00" \
  --consensus_k_map "flip_v:3" --consensus_eps_map "flip_v:0.10" \
  --ortho_flip_v \
  --csv stage11_step6_S6-1.csv

python3 stage11-benchmark-v5.py \
  --samples 100 --sigma 11 --proto_width 180 --rank 3 \
  --nperm 400 --block 64 \
  --temp0 1.1 --temp_min 0.12 --anneal 0.55 \
  --inhib_strength 1.8 --inhib_sigma 80 --presence_floor 0.60 \
  --proto_modes_map "flip_v:half,halfN,halfW,hingeL,hingeR,deriv;flip_h:half,triangle;rotate:half,triangle" \
  --proto_phases 0.0,0.2 \
  --proto_agg softmin --proto_tau 0.45 \
  --raw_floor_map "flip_v:0.68" --residual_ceiling_map "flip_v:0.95" \
  --consensus_k_map "flip_v:3" --consensus_eps_map "flip_v:0.10" \
  --ortho_flip_v \
  --csv stage11_step6_S6-2.csv

python3 stage11-benchmark-v5.py \
  --samples 100 --sigma 11 --proto_width 180 --rank 3 \
  --nperm 400 --block 64 \
  --temp0 1.1 --temp_min 0.12 --anneal 0.55 \
  --inhib_strength 1.2 --inhib_sigma 60 --presence_floor 0.50 \
  --proto_modes_map "flip_v:half,halfN,halfW,hingeL,hingeR,deriv;flip_h:half,triangle;rotate:half,triangle" \
  --proto_phases 0.0,0.2 \
  --proto_agg softmin --proto_tau 0.40 \
  --raw_floor_map "flip_v:0.60" --residual_ceiling_map "flip_v:1.10" \
  --consensus_k_map "flip_v:2" --consensus_eps_map "flip_v:0.12" \
  --ortho_flip_v \
  --csv stage11_step6_S6-3.csv

python3 stage11-benchmark-v5.py \
  --samples 100 --sigma 11 --proto_width 180 --rank 3 \
  --nperm 400 --block 64 \
  --temp0 1.1 --temp_min 0.12 --anneal 0.55 \
  --inhib_strength 1.6 --inhib_sigma 72 --presence_floor 0.55 \
  --proto_modes_map "flip_v:half,hingeL,hingeR,deriv;flip_h:half,triangle;rotate:half,triangle" \
  --proto_phases 0.0,0.2 \
  --proto_agg median \
  --raw_floor_map "flip_v:0.65" --residual_ceiling_map "flip_v:1.00" \
  --consensus_k_map "flip_v:3" --consensus_eps_map "flip_v:0.10" \
  --ortho_flip_v \
  --csv stage11_step6_S6-4.csv

python3 stage11-benchmark-v5.py \
  --samples 100 --sigma 11 --proto_width 180 --rank 3 \
  --nperm 400 --block 64 \
  --temp0 1.1 --temp_min 0.12 --anneal 0.55 \
  --inhib_strength 1.6 --inhib_sigma 72 --presence_floor 0.55 \
  --proto_modes_map "flip_v:half,halfN,halfW,hingeL,hingeR,deriv;flip_h:half,triangle;rotate:half,triangle" \
  --proto_phases 0.0,0.2 \
  --proto_agg softmin --proto_tau 0.45 \
  --raw_floor_map "flip_v:0.65" --residual_ceiling_map "flip_v:1.00" \
  --consensus_k_map "flip_v:3" --consensus_eps_map "flip_v:0.10" \
  --csv stage11_step6_S6-5.csv
"""




import argparse, os, csv, math
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- Utilities -----------------------------

def moving_average(x: np.ndarray, k: int = 9) -> np.ndarray:
    if k <= 1: 
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(xp, np.ones(k)/k, mode="valid")

def half_sine(width: int) -> np.ndarray:
    return np.sin(np.linspace(0, np.pi, width))

def half_sine_proto(width: int) -> np.ndarray:
    p = half_sine(width)
    return p / (np.linalg.norm(p) + 1e-8)

def gaussian_bump(T: int, center: int, width: int, amp: float = 1.0) -> np.ndarray:
    t = np.arange(T)
    sig2 = (width/2.355)**2  # FWHM->sigma
    return amp * np.exp(-(t-center)**2 / (2*sig2))

def add_noise(x: np.ndarray, sigma: float, rng) -> np.ndarray:
    return x + rng.normal(0, sigma, size=x.shape)

def common_mode(traces: Dict[str, np.ndarray]) -> np.ndarray:
    arr = np.stack([traces[p] for p in traces.keys()], 0)
    return arr.mean(0)

def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    sd = x.std() + 1e-12
    return (x - mu) / sd

def circ_shift(x: np.ndarray, k: int) -> np.ndarray:
    k = int(k) % len(x)
    if k == 0: return x
    return np.concatenate([x[-k:], x[:-k]])

def block_circ_shift(x: np.ndarray, rng, block: int) -> np.ndarray:
    """Rotate by a random multiple of `block` (preserve local autocorrelation)."""
    T = len(x)
    if block <= 1:
        s = rng.integers(1, T-1)
    else:
        nblocks = max(1, T // block)
        s = int(rng.integers(1, nblocks)) * block
    return circ_shift(x, s)

def normalized_corr_window(sig: np.ndarray, proto: np.ndarray, center_idx: int, width: int,
                           taper: str = None) -> float:
    """Cosine similarity on a centered window, optionally tapered (hann/hamming/blackman)."""
    T = len(sig)
    a, b = max(0, center_idx - width//2), min(T, center_idx + width//2)
    w = sig[a:b].astype(float)
    if len(w) < 3: 
        return 0.0
    pr = proto[:len(w)].astype(float)
    # Tapers (simple multitaper approximation)
    if taper is not None:
        t = np.linspace(0, np.pi, len(w))
        if taper.lower() == "hann":
            win = 0.5 * (1 - np.cos(2*np.pi*np.arange(len(w))/(len(w)-1)))
        elif taper.lower() == "hamming":
            win = 0.54 - 0.46 * np.cos(2*np.pi*np.arange(len(w))/(len(w)-1))
        elif taper.lower() == "blackman":
            n = np.arange(len(w))
            win = 0.42 - 0.5*np.cos(2*np.pi*n/(len(w)-1)) + 0.08*np.cos(4*np.pi*n/(len(w)-1))
        else:
            win = np.ones_like(w)
        w = w * win
        pr = pr * win
    # demean then cosine
    w = w - w.mean()
    pr = pr - pr.mean()
    denom = (np.linalg.norm(w) * np.linalg.norm(pr) + 1e-8)
    return float(np.dot(w, pr) / denom)

def multitaper_corr(sig: np.ndarray, proto: np.ndarray, center_idx: int, width: int,
                    tapers: List[str]) -> float:
    if not tapers:
        return normalized_corr_window(sig, proto, center_idx, width, taper=None)
    vals = [normalized_corr_window(sig, proto, center_idx, width, taper=t) for t in tapers]
    return float(np.mean(vals))

def perm_null_block_z(sig: np.ndarray, proto: np.ndarray, peak_idx: int, width: int, rng,
                      nperm: int = 200, block: int = 32, tapers: List[str] = None) -> Tuple[float, float]:
    """Return (z, obs_corr) under a block-circular permutation null with optional multitaper averaging."""
    obs = multitaper_corr(sig, proto, peak_idx, width, tapers or [])
    null = np.empty(nperm, dtype=float)
    for i in range(nperm):
        x = block_circ_shift(sig, rng, block)
        null[i] = multitaper_corr(x, proto, peak_idx, width, tapers or [])
    mu, sd = float(null.mean()), float(null.std() + 1e-8)
    z = (obs - mu) / sd
    return float(z), float(obs)

def span_complement_project(S: np.ndarray, rank: int) -> np.ndarray:
    """
    Remove top-`rank` shared components across channels from S (T x K).
    Returns residualized S_perp of same shape.
    """
    T, K = S.shape
    X = S - S.mean(axis=0, keepdims=True)        # center each channel (timewise)
    # SVD on time × channels to capture shared cross-channel structure
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    r = int(max(0, min(rank, len(s))))
    if r == 0:
        return X
    Ur = U[:, :r]                                # time-basis for shared variance
    # Project each column onto complement of Ur
    X_res = X - Ur @ (Ur.T @ X)
    return X_res

def perpendicular_energy(traces: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    mu = common_mode(traces)
    return {p: np.clip(traces[p] - mu, 0, None) for p in traces.keys()}

# ----------------------------- World (synthetic ARC-style) -----------------------------

PRIMS = ["flip_h","flip_v","rotate"]

@dataclass
class Sample:
    grid_in: np.ndarray
    tasks_true: List[str]
    order_true: List[str]
    grid_out_true: np.ndarray
    traces: Dict[str, np.ndarray]
    T: int

def random_grid(rng, H=8, W=8, ncolors=6): 
    return rng.integers(0, ncolors, size=(H, W))

def apply_primitive(grid: np.ndarray, prim: str) -> np.ndarray:
    if prim == "flip_h": return np.fliplr(grid)
    if prim == "flip_v": return np.flipud(grid)
    if prim == "rotate": return np.rot90(grid, k=-1)
    raise ValueError(prim)

def apply_sequence(grid: np.ndarray, seq: List[str]) -> np.ndarray:
    g = grid.copy()
    for p in seq:
        g = apply_primitive(g, p)
    return g

def gen_traces(tasks: List[str], T: int, rng, noise=0.02, cm_amp=0.02, overlap=0.5,
               amp_jitter=0.4, distractor_prob=0.4) -> Dict[str, np.ndarray]:
    # three centers to allow up to 3 tasks
    base = np.array([0.20, 0.50, 0.80]) * T
    centers = ((1.0 - overlap) * base + overlap * (T * 0.50)).astype(int)
    width = int(max(12, T * 0.10))
    t = np.arange(T)
    cm = cm_amp * (1.0 + 0.2 * np.sin(2*np.pi * t / max(30, T//6)))
    traces = {p: np.zeros(T, float) for p in PRIMS}
    # true bumps with jitter and amplitude noise
    for i, prim in enumerate(tasks):
        c = centers[i % len(centers)]
        amp = max(0.25, 1.0 + rng.normal(0, amp_jitter))
        c_jit = int(np.clip(c + rng.integers(-width//5, width//5 + 1), 0, T-1))
        traces[prim] += gaussian_bump(T, c_jit, width, amp=amp)
    # distractors
    for p in PRIMS:
        if p not in tasks and rng.random() < distractor_prob:
            c = int(rng.uniform(T*0.15, T*0.85))
            amp = max(0.25, 1.0 + rng.normal(0, amp_jitter))
            traces[p] += gaussian_bump(T, c, width, amp=0.9*amp)
    # add common-mode + noise
    for p in PRIMS:
        traces[p] = np.clip(traces[p] + cm, 0, None)
        traces[p] = add_noise(traces[p], noise, rng)
    return traces

def make_sample(rng, T=720, n_tasks=(1,3), grid_shape=(8,8), noise=0.02,
                allow_repeats=False, cm_amp=0.02, overlap=0.5, amp_jitter=0.4,
                distractor_prob=0.4) -> Sample:
    k = rng.integers(n_tasks[0], n_tasks[1]+1)
    tasks = list(rng.choice(PRIMS, size=k, replace=allow_repeats))
    if not allow_repeats:
        rng.shuffle(tasks)
    g0 = random_grid(rng, *grid_shape)
    g1 = apply_sequence(g0, tasks)
    traces = gen_traces(tasks, T=T, rng=rng, noise=noise, cm_amp=cm_amp,
                        overlap=overlap, amp_jitter=amp_jitter, distractor_prob=distractor_prob)
    return Sample(g0, tasks, tasks, g1, traces, T)

# ----------------------------- Metrics -----------------------------

def grid_similarity(gp: np.ndarray, gt: np.ndarray) -> float:
    return 0.0 if gp.shape != gt.shape else float((gp==gt).mean())

def set_metrics(true_list: List[str], pred_list: List[str]) -> Dict[str,float]:
    Tset, Pset = set(true_list), set(pred_list)
    tp, fp, fn = len(Tset & Pset), len(Pset - Tset), len(Tset - Pset)
    precision = tp / max(1, len(Pset))
    recall    = tp / max(1, len(Tset))
    f1        = 0.0 if precision+recall==0 else (2*precision*recall)/(precision+recall)
    jaccard   = tp / max(1, len(Tset | Pset))
    return dict(precision=precision, recall=recall, f1=f1, jaccard=jaccard,
                hallucination_rate=fp/max(1,len(Pset)), omission_rate=fn/max(1,len(Tset)))

# ----------------------------- Stage-11 Well Parser -----------------------------

def estimate_peak_indices(S_channel: Dict[str, np.ndarray], proto: np.ndarray) -> Dict[str, int]:
    """Return index of max correlation with prototype per channel (same-length corr)."""
    peak = {}
    for p, sig in S_channel.items():
        m = np.correlate(zscore(sig), proto, mode="same")
        peak[p] = int(np.argmax(m))
    return peak

def subtract_aligned_proto(S: np.ndarray, pick_idx: int, peak_idx: int, proto: np.ndarray, width: int) -> np.ndarray:
    """
    Subtract a scaled prototype around peak_idx for the picked channel from all channels.
    S is (T x K). We fit a per-channel scalar within the window and subtract that shaped proto.
    """
    T, K = S.shape
    a, b = max(0, peak_idx - width//2), min(T, peak_idx + width//2)
    L = b - a
    if L <= 2:
        return S
    pr = proto[:L]
    pr_c = pr - pr.mean()
    pr_den = (np.dot(pr_c, pr_c) + 1e-8)
    S2 = S.copy()
    for k in range(K):
        seg = S[a:b, k]
        seg_c = seg - seg.mean()
        alpha = float(np.dot(seg_c, pr_c) / pr_den)  # LS fit
        S2[a:b, k] = seg - alpha * pr  # subtract
    return S2

def well_parser(traces: Dict[str, np.ndarray],
                rng,
                sigma: int = 9,
                proto_width: int = 160,
                rank: int = 2,
                weights: Tuple[float,float,float] = (1.0, 0.4, 0.3),
                nperm: int = 200,
                block: int = 32,
                tapers: List[str] = None,
                temp0: float = 1.2,
                temp_min: float = 0.15,
                anneal: float = 0.7,
                max_picks: int = None,
                inhib_strength: float = 0.6,
                inhib_sigma: float = 40.0,
                presence_floor: float = 0.35) -> Tuple[List[str], List[str], Dict[str, float], Dict[str, int], List[float]]:
    """
    Return: (keep, order, energy_map, peak_idx, margins)
        keep: selected primitives (set membership)
        order: ordered primitives by peak time
        energy_map: final energies U_p for all primitives
        peak_idx: detected peaks for each primitive
        margins: list of Δ margins per selection step (U_false - U_true)
    """
    keys = list(traces.keys())
    K = len(keys)
    T = len(next(iter(traces.values())))

    # Smooth raw + residualize via common-mode
    Sraw = {p: moving_average(traces[p], k=sigma) for p in keys}
    Eres = perpendicular_energy(traces)
    Sres = {p: moving_average(Eres[p], k=sigma) for p in keys}
    Scm  = moving_average(common_mode(traces), k=sigma)

    # Low-rank shared removal on residuals
    M = np.stack([Sres[p] for p in keys], axis=1)  # (T x K)
    M_perp = span_complement_project(M, rank=rank) if rank > 0 else (M - M.mean(axis=0, keepdims=True))
    Sres_perp = {p: M_perp[:, i] for i, p in enumerate(keys)}

    proto = half_sine_proto(proto_width)
    peak_idx = estimate_peak_indices(Sres_perp, proto)

    # helper to compute Stage-11 energy for all channels
    w_perp, w_raw, w_cm = weights
    def compute_energy(Sres_local: Dict[str, np.ndarray],
                       Sraw_local: Dict[str, np.ndarray],
                       Scm_local: np.ndarray,
                       peaks: Dict[str, int],
                       lateral_peaks: List[int] = None) -> Dict[str, float]:
        E = {}
        for p in keys:
            idx = peaks[p]
            # z-scores via block-permutation null + multitaper average
            z_perp, _ = perm_null_block_z(zscore(Sres_local[p]), proto, idx, proto_width, rng,
                                          nperm=nperm, block=block, tapers=tapers or [])
            z_raw,  _ = perm_null_block_z(zscore(Sraw_local[p]),  proto, idx, proto_width, rng,
                                          nperm=nperm, block=block, tapers=tapers or [])
            z_cm,   _ = perm_null_block_z(zscore(Scm_local),      proto, idx, proto_width, rng,
                                          nperm=nperm, block=block, tapers=tapers or [])
            # Energy
            Up = - (w_perp * z_perp + w_raw * z_raw - w_cm * max(0.0, z_cm))
            # Lateral inhibition penalty
            if lateral_peaks:
                for t0 in lateral_peaks:
                    dt = float(idx - t0)
                    pen = inhib_strength * math.exp(-(dt*dt) / (2.0 * (inhib_sigma**2)))
                    Up += pen
            E[p] = float(Up)
        return E

    keep = []
    chosen_peaks = []  # list of idx of chosen peaks (for lateral inhibition)
    margins = []

    # Copy working matrices for refinement
    M_work = M_perp.copy()       # residual channels only for refinement
    Sraw_work = {p: Sraw[p].copy() for p in keys}
    Scm_work = Scm.copy()
    Sres_work = {p: M_work[:, i] for i, p in enumerate(keys)}
    peaks_work = dict(peak_idx)

    # Selection loop
    step = 0
    while True:
        energies = compute_energy(Sres_work, Sraw_work, Scm_work, peaks_work, lateral_peaks=chosen_peaks)
        # pick best via annealed softmax (lower U better)
        U_arr = np.array([energies[p] for p in keys])
        # Mask already selected
        mask = np.array([p not in keep for p in keys], dtype=bool)
        if not mask.any():
            break
        U_masked = U_arr.copy()
        U_masked[~mask] = np.inf  # exclude selected
        # stopping: if no finite energies remain
        if not np.isfinite(U_masked).any():
            break

        # compute probabilities over available candidates
        Tcur = max(temp_min, temp0 * (anneal ** step))
        # Convert to probabilities: P ∝ exp(-(U - Umin)/T)
        Uavail = U_masked[mask]
        Umin = float(np.min(Uavail))
        logits = - (Uavail - Umin) / max(1e-8, Tcur)
        probs = np.exp(logits - logits.max())
        probs = probs / (probs.sum() + 1e-12)
        # choose argmax (deterministic mode) or sample — we choose argmax for stability
        idx_avail = np.where(mask)[0]
        pick_pos = int(np.argmax(probs))
        pick_k = int(idx_avail[pick_pos])
        pick = keys[pick_k]
        keep.append(pick)

        # compute margin Δ = min_{false} U - U_true for logging
        false_U = [energies[p] for p in keys if p not in keep[:-1]]  # "false" relative to current
        if len(false_U) > 1:
            # true is the just-picked (lowest U among avail); margin = closest false minus true
            true_U = energies[pick]
            others = [energies[p] for p in keys if p != pick and p not in keep[:-1]]
            margin = float(min(others) - true_U) if others else float('nan')
            margins.append(margin)

        # Sequential residual refinement: subtract aligned prototype around peak
        t0 = peaks_work[pick]
        M_work = subtract_aligned_proto(M_work, pick_k, t0, proto, proto_width)
        # Rebuild working dicts after subtraction
        Sres_work = {p: M_work[:, i] for i, p in enumerate(keys)}
        # Optional: also lightly subtract from raw/common-mode to keep calibration consistent
        # (we apply a small fraction to avoid over-subtraction)
        frac = 0.25
        a, b = max(0, t0 - proto_width//2), min(T, t0 + proto_width//2)
        L = b - a
        if L > 2:
            pr = proto[:L]
            for i, p2 in enumerate(keys):
                seg = Sraw_work[p2][a:b]
                alpha = float(np.dot(seg - seg.mean(), (pr - pr.mean())) / (np.dot(pr - pr.mean(), pr - pr.mean()) + 1e-8))
                Sraw_work[p2][a:b] = seg - frac * alpha * pr
            Scm_work[a:b] = Scm_work[a:b] - frac * pr.mean()  # gentle adjustment

        # Update peak indices after refinement
        peaks_work = estimate_peak_indices(Sres_work, proto)
        chosen_peaks.append(t0)

        step += 1
        if max_picks is not None and len(keep) >= max_picks:
            break

        # Stopping criterion based on presence floor:
        # If the best (lowest) energy no longer improves relative to the median by presence_floor, stop.
        Uavail2 = np.array([energies[p] for p in keys if p not in keep])
        if Uavail2.size == 0:
            break
        # empirical test: if improvement small and temperature cold, break
        if Tcur <= (temp_min + 1e-6):
            u_best = float(np.min(Uavail2))
            u_med  = float(np.median(Uavail2))
            if (u_med - u_best) < presence_floor:
                break

    # Order by peak time (from final peaks_work recomputation)
    order = sorted(keep, key=lambda p: peaks_work[p])
    # Return also energy map from last iteration for introspection
    final_energies = {p: float(v) for p, v in energies.items()}
    return keep, order, final_energies, peaks_work, margins

# ----------------------------- Main Benchmark -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage 11 — Well Parser Benchmark (v1)")
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--sigma", type=int, default=9)
    ap.add_argument("--proto_width", type=int, default=160)
    ap.add_argument("--rank", type=int, default=2, help="top-r PCs to remove (span complement)")
    ap.add_argument("--nperm", type=int, default=150)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--tapers", type=str, default="hann,hamming,blackman")
    ap.add_argument("--temp0", type=float, default=1.2)
    ap.add_argument("--temp_min", type=float, default=0.15)
    ap.add_argument("--anneal", type=float, default=0.7)
    ap.add_argument("--max_picks", type=int, default=None)
    ap.add_argument("--inhib_strength", type=float, default=0.6)
    ap.add_argument("--inhib_sigma", type=float, default=40.0)
    ap.add_argument("--presence_floor", type=float, default=0.35)
    ap.add_argument("--weights", type=str, default="1.0,0.4,0.3")  # w_perp,w_raw,w_cm
    ap.add_argument("--tasks_range", type=str, default="1,3")
    ap.add_argument("--allow_repeats", action="store_true")
    ap.add_argument("--cm_amp", type=float, default=0.02)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--amp_jitter", type=float, default=0.4)
    ap.add_argument("--distractor_prob", type=float, default=0.4)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--csv", type=str, default="stage11_well_benchmark.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    min_k, max_k = map(int, args.tasks_range.split(","))
    w_perp, w_raw, w_cm = map(float, args.weights.split(","))
    tapers = [t.strip() for t in args.tapers.split(",") if t.strip()] if args.tapers else []

    rows = []
    agg = dict(
        acc_exact_geo=0, grid_geo=0.0, P_geo=0.0, R_geo=0.0, F1_geo=0.0, J_geo=0.0, H_geo=0.0, O_geo=0.0,
        # We'll keep "stock" fields for compatibility; they mirror well parser so names end with _geo
        margin_mean=0.0, margin_min=0.0
    )
    margins_all = []

    for i in range(1, args.samples+1):
        sample = make_sample(rng, T=args.T, n_tasks=(min_k, max_k), noise=args.noise,
                             allow_repeats=args.allow_repeats, cm_amp=args.cm_amp,
                             overlap=args.overlap, amp_jitter=args.amp_jitter,
                             distractor_prob=args.distractor_prob)

        keep, order, energies, peaks, margins = well_parser(
            sample.traces, rng, sigma=args.sigma, proto_width=args.proto_width,
            rank=args.rank, weights=(w_perp, w_raw, w_cm), nperm=args.nperm,
            block=args.block, tapers=tapers, temp0=args.temp0, temp_min=args.temp_min,
            anneal=args.anneal, max_picks=args.max_picks, inhib_strength=args.inhib_strength,
            inhib_sigma=args.inhib_sigma, presence_floor=args.presence_floor
        )

        # Evaluate
        grid_pred = apply_sequence(sample.grid_in, order)
        ok = int(np.array_equal(grid_pred, sample.grid_out_true))
        gs = grid_similarity(grid_pred, sample.grid_out_true)
        sm = set_metrics(sample.order_true, keep)

        agg["acc_exact_geo"] += ok
        agg["grid_geo"] += gs
        agg["P_geo"] += sm["precision"]; agg["R_geo"] += sm["recall"]; agg["F1_geo"] += sm["f1"]; agg["J_geo"] += sm["jaccard"]
        agg["H_geo"] += sm["hallucination_rate"]; agg["O_geo"] += sm["omission_rate"]

        m_mean = float(np.nanmean(margins)) if margins else float("nan")
        m_min  = float(np.nanmin(margins)) if margins else float("nan")
        margins_all.extend([m for m in margins if np.isfinite(m)])
        agg["margin_mean"] += (0.0 if np.isnan(m_mean) else m_mean)
        agg["margin_min"]  += (0.0 if np.isnan(m_min)  else m_min)

        rows.append(dict(
            sample=i, true="|".join(sample.order_true),
            keep="|".join(keep), order="|".join(order), ok=ok, grid=gs,
            precision=sm["precision"], recall=sm["recall"], f1=sm["f1"], jaccard=sm["jaccard"],
            hallucination=sm["hallucination_rate"], omission=sm["omission_rate"],
            energies=";".join([f"{k}:{energies[k]:.3f}" for k in PRIMS]),
            peaks=";".join([f"{k}:{peaks[k]}" for k in PRIMS]),
            margin_mean=m_mean, margin_min=m_min
        ))

        if args.plots:
            # Quick diagnostic plot per-sample
            T = sample.T
            Sres = perpendicular_energy(sample.traces)
            Sres_s = {p: moving_average(Sres[p], k=args.sigma) for p in PRIMS}
            Sraw = {p: moving_average(sample.traces[p], k=args.sigma) for p in PRIMS}

            fig, ax = plt.subplots(2,1, figsize=(11,6), sharex=True)
            # Residual (smoothed)
            for p in PRIMS:
                ax[0].plot(Sres_s[p], label=f"E⊥ {p}", linewidth=2)
                ax[0].axvline(peaks[p], color='k', linestyle=':', alpha=0.3)
            ax[0].legend(loc="upper right"); ax[0].set_ylabel("residual power (smoothed)")
            ax[0].set_title(f"[Stage11 Well] keep={keep} | order={' → '.join(order) if order else '—'} | ok={bool(ok)}")
            # Raw (smoothed)
            for p in PRIMS:
                ax[1].plot(Sraw[p], label=f"Eraw {p}", linewidth=2, alpha=0.9)
            ax[1].legend(loc="upper right"); ax[1].set_xlabel("step"); ax[1].set_ylabel("raw power (smoothed)")
            plt.tight_layout()
            out = f"stage11_sample_{i:02d}.png"; plt.savefig(out, dpi=120); plt.close()
            print(f"[Plot] {out}")

    n = float(args.samples)
    summary = dict(
        acc_exact_geo = agg["acc_exact_geo"]/n,
        grid_geo      = agg["grid_geo"]/n,
        precision     = agg["P_geo"]/n,
        recall        = agg["R_geo"]/n,
        f1            = agg["F1_geo"]/n,
        jaccard       = agg["J_geo"]/n,
        hallucination = agg["H_geo"]/n,
        omission      = agg["O_geo"]/n,
        margin_mean   = agg["margin_mean"]/n,
        margin_min    = agg["margin_min"]/n,
        n_samples     = int(n)
    )
    print("[Stage11 v1] Summary:", {k: (round(v,3) if isinstance(v,float) else v) for k,v in summary.items()})
    if margins_all:
        print(f"[Stage11 v1] Margins — mean={np.mean(margins_all):.3f}, min={np.min(margins_all):.3f}, n={len(margins_all)}")

    # CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"[WROTE] {args.csv}")

if __name__ == "__main__":
    main()

