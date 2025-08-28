
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 10 — v2 synthetic benchmark (parser + executor)
----------------------------------------------------
- Synthesizes ARC-12-like primitives: flip_h, flip_v, rotate
- Builds concept energy traces over time for each primitive
- Parses tasks + order from traces using:
    * Perpendicular (common-mode removed) energy
    * Smoothing + matched filter
    * Peak picking (top-K by matched-filter response)
- Executes parsed tasks via geodesic *primitive* executor (here: literal ops)
- Reports accuracy and saves diagnostic plots

Usage:
  python3 arc_stage10_v2_synth.py --samples 12 --seed 42 --T 720 --plot_dir plots_v2_synth

Author: Stage 10 v2 (synthetic) — consolidated by ChatGPT
"""

# python3 arc-benchmark-geodesic-latest.py --samples 10 --seed 42

# Flags you can tweak
# --samples (default 12)
# --seed (default 42)
# --T time steps (default 720)
# --noise trace noise (default 0.01)
# --sigma smoother window (default 9)
# --proto_width matched-filter width (default 140)
# --topk force a max number of concepts to keep (else auto)

# [01] -> Tasks: ['rotate'] | Order: rotate | ok=True
#      areas⊥=[1.092, 1.196, 51.425] corr_peak=[0.523, 0.11, 0.973] plot=plots_v2/sample01.png
# [02] -> Tasks: ['flip_h', 'flip_v'] | Order: flip_h → flip_v | ok=True
#      areas⊥=[51.358, 51.39, 0.755] corr_peak=[0.972, 0.972, 0.328] plot=plots_v2/sample02.png
# [03] -> Tasks: ['flip_h'] | Order: flip_h | ok=True
#      areas⊥=[51.567, 1.185, 1.285] corr_peak=[0.971, 0.093, 0.156] plot=plots_v2/sample03.png
# [04] -> Tasks: ['flip_h', 'flip_v', 'rotate'] | Order: flip_h → flip_v → rotate | ok=True
#      areas⊥=[51.029, 50.777, 50.976] corr_peak=[0.972, 0.972, 0.971] plot=plots_v2/sample04.png
# [05] -> Tasks: ['flip_h', 'rotate'] | Order: flip_h → rotate | ok=True
#      areas⊥=[51.214, 0.765, 51.548] corr_peak=[0.972, 0.289, 0.971] plot=plots_v2/sample05.png
# [06] -> Tasks: ['flip_h'] | Order: flip_h | ok=True
#      areas⊥=[51.815, 0.956, 1.15] corr_peak=[0.971, 0.275, 0.121] plot=plots_v2/sample06.png
# [07] -> Tasks: ['rotate'] | Order: rotate | ok=True
#      areas⊥=[1.18, 1.158, 51.721] corr_peak=[0.264, 0.202, 0.972] plot=plots_v2/sample07.png
# [08] -> Tasks: ['flip_h', 'flip_v', 'rotate'] | Order: flip_h → flip_v → rotate | ok=True
#      areas⊥=[50.844, 51.047, 50.994] corr_peak=[0.971, 0.971, 0.971] plot=plots_v2/sample08.png
# [09] -> Tasks: ['flip_h'] | Order: flip_h | ok=True
#      areas⊥=[51.822, 1.123, 1.264] corr_peak=[0.972, 0.41, 0.373] plot=plots_v2/sample09.png
# [10] -> Tasks: ['flip_h', 'flip_v', 'rotate'] | Order: rotate → flip_h → flip_v | ok=True
#      areas⊥=[51.127, 50.938, 50.934] corr_peak=[0.97, 0.971, 0.971] plot=plots_v2/sample10.png

# [Stage10 v2 — synthetic] Accuracy: 10/10 = 100.0%


import argparse
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

PRIMS = ["flip_h", "flip_v", "rotate"]

# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int):
    rng = np.random.default_rng(seed)
    return rng

def moving_average(x, k=9):
    # symmetric causal-ish smoother: reflect-pad then uniform filter
    if k <= 1: 
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    kernel = np.ones(k) / k
    y = np.convolve(xp, kernel, mode="valid")
    return y

def half_sine(width):
    t = np.linspace(0, np.pi, width)
    return np.sin(t)

def gaussian_bump(T, center, width, amp=1.0):
    t = np.arange(T)
    sig2 = (width/2.355)**2  # FWHM to sigma
    return amp * np.exp(-(t-center)**2 / (2*sig2))

def add_noise(x, sigma, rng):
    return x + rng.normal(0, sigma, size=x.shape)

# -----------------------------
# Synthetic generator
# -----------------------------

@dataclass
class Sample:
    grid_in: np.ndarray
    tasks_true: List[str]
    order_true: List[str]
    grid_out_true: np.ndarray
    traces: Dict[str, np.ndarray]
    T: int

def random_grid(rng, H=8, W=8, ncolors=5):
    return rng.integers(0, ncolors, size=(H, W))

def apply_primitive(grid, prim):
    if prim == "flip_h":
        return np.fliplr(grid)
    if prim == "flip_v":
        return np.flipud(grid)
    if prim == "rotate":
        # 90° clockwise
        return np.rot90(grid, k=-1)
    raise ValueError("unknown prim")

def apply_sequence(grid, seq):
    g = grid.copy()
    for p in seq:
        g = apply_primitive(g, p)
    return g

def gen_synthetic_traces(tasks: List[str], T: int, rng, noise=0.01) -> Dict[str, np.ndarray]:
    """
    For each primitive concept, produce an energy trace over T steps.
    We place broad gaussian bumps at time windows associated with tasks.
    """
    # positions for potential 3 tasks across T
    centers = [int(T*0.15), int(T*0.45), int(T*0.70)]
    width = int(T*0.10)
    base_drift = 0.002 * np.linspace(1.0, 1.1, T)

    traces = {p: np.zeros(T, dtype=float) for p in PRIMS}
    # Each task contributes a bump to its concept
    for idx, prim in enumerate(tasks):
        c = centers[idx % len(centers)]
        amp = 1.0
        bump = gaussian_bump(T, c, width, amp=amp)
        traces[prim] += bump

    # Add small correlated baseline to all traces + noise
    for p in PRIMS:
        traces[p] += base_drift
        traces[p] = add_noise(traces[p], noise, rng)
        # ensure non-neg
        traces[p] = np.clip(traces[p], 0, None)

    return traces

def make_sample(rng, T=720, n_tasks=(1,3), grid_shape=(8,8), noise=0.01) -> Sample:
    k = rng.integers(n_tasks[0], n_tasks[1]+1)
    tasks = list(rng.choice(PRIMS, size=k, replace=False))
    # random order:
    rng.shuffle(tasks)
    g0 = random_grid(rng, H=grid_shape[0], W=grid_shape[1])
    g1 = apply_sequence(g0, tasks)
    traces = gen_synthetic_traces(tasks, T=T, rng=rng, noise=noise)
    return Sample(grid_in=g0, tasks_true=tasks, order_true=tasks, grid_out_true=g1, traces=traces, T=T)

# -----------------------------
# Parser: perpendicular energy + matched filter
# -----------------------------

@dataclass
class ParseResult:
    tasks: List[str]
    order: List[str]
    peak_times: Dict[str, int]
    areas_perp: Dict[str, float]
    corr_peak: Dict[str, float]

def common_mode(traces: Dict[str, np.ndarray]) -> np.ndarray:
    X = np.stack([traces[p] for p in PRIMS], axis=0)
    mu = X.mean(axis=0)
    return mu

def perpendicular_energy(traces: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    mu = common_mode(traces)
    E = {}
    for p in PRIMS:
        E[p] = np.clip(traces[p] - mu, 0, None)  # residual positive part
    return E

def matched_filter_parse(traces: Dict[str, np.ndarray], sigma=9, proto_width=140, topk=None) -> ParseResult:
    # Residual energy
    E_perp = perpendicular_energy(traces)
    T = len(next(iter(traces.values())))

    # Smoothing
    E_s = {p: moving_average(E_perp[p], k=sigma) for p in PRIMS}

    # Prototype (half-sine) normalized
    proto = half_sine(proto_width)
    proto = proto / (np.linalg.norm(proto) + 1e-8)

    # Matched filtering via convolution (correlation) with same length
    MF = {}
    peak_idx = {}
    corr_peak = {}
    areas = {}

    for p in PRIMS:
        # correlate
        m = np.correlate(E_s[p], proto, mode="same")
        MF[p] = m
        # pick global peak
        idx = int(np.argmax(m))
        peak_idx[p] = idx
        # correlation proxy: cosine between window and proto
        L = proto_width
        a = max(0, idx - L//2)
        b = min(T, idx + L//2)
        w = E_s[p][a:b]
        w = (w - w.mean())  # demean before cosine
        pr = proto[:len(w)] - proto[:len(w)].mean()
        denom = (np.linalg.norm(w) * np.linalg.norm(pr) + 1e-8)
        corr_peak[p] = float(np.dot(w, pr) / denom)
        areas[p] = float(np.trapz(E_s[p]))

    # decide which concepts are present
    areas_arr = np.array([areas[p] for p in PRIMS])
    corrs_arr = np.array([corr_peak[p] for p in PRIMS])
    Amax = float(areas_arr.max() + 1e-12)
    Cmax = float(corrs_arr.max() + 1e-12)

    if topk is None:
        keep = []
        for p in PRIMS:
            a_ok = (areas[p] / Amax) >= 0.45
            c_ok = (corr_peak[p] / Cmax) >= 0.5
            if a_ok and c_ok:
                keep.append(p)
        # Fallback: if we filtered everything (e.g., weak signals), keep the best by (corr * area)
        if not keep:
            score = {p: corr_peak[p] * (areas[p] / (Amax+1e-12)) for p in PRIMS}
            keep = [max(score, key=score.get)]
    else:
        keep = sorted(PRIMS, key=lambda p: corr_peak[p], reverse=True)[:topk]

    # order by peak time increasing
    order = sorted(keep, key=lambda p: peak_idx[p])

    return ParseResult(tasks=keep, order=order, peak_times=peak_idx,
                       areas_perp=areas, corr_peak=corr_peak)

# -----------------------------
# Executor: apply predicted sequence to grid
# -----------------------------

def execute_plan(grid, order: List[str]):
    return apply_sequence(grid, order)

# -----------------------------
# Plotting
# -----------------------------

def plot_parse(sample_id: int, sample: Sample, E_perp_s: Dict[str, np.ndarray], parse: ParseResult, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    T = sample.T
    fig, ax = plt.subplots(figsize=(12,4))
    for p, color in zip(PRIMS, ["C0","C1","C2"]):
        ax.plot(E_perp_s[p], label=f"E⊥ {p}", color=color, linewidth=2)
    # overlay a prototype shifted to each picked peak (for the first in order)
    if len(parse.order) > 0:
        p0 = parse.order[0]
        proto_width = 140
        proto = half_sine(proto_width)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        shift = parse.peak_times[p0]
        xs = np.arange(proto_width) + shift - proto_width//2
        ys = proto * max(1e-8, E_perp_s[p0].max())
        # clip to plot bounds
        mask = (xs >= 0) & (xs < T)
        ax.plot(xs[mask], ys[mask], 'r--', label=f"proto⊥ {p0}", linewidth=2, alpha=0.8)

    ax.set_title(f"v2 parse — sample{sample_id:02d} — tasks: {parse.tasks} — order: {' → '.join(parse.order) if parse.order else '—'}")
    ax.set_xlabel("step"); ax.set_ylabel("residual aligned power (smoothed)")
    ax.legend(loc="upper right")
    fname = os.path.join(out_dir, f"sample{sample_id:02d}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=110)
    plt.close()
    return fname

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--plot_dir", type=str, default="plots_stage10_v2")
    ap.add_argument("--sigma", type=int, default=9, help="smoother window")
    ap.add_argument("--proto_width", type=int, default=140)
    ap.add_argument("--topk", type=int, default=None, help="max concepts to keep; default=auto")
    args = ap.parse_args()

    rng = set_seed(args.seed)

    correct = 0
    results = []

    for i in range(1, args.samples+1):
        # 1) build synthetic sample
        sample = make_sample(rng, T=args.T, noise=args.noise)

        # 2) compute residual (perpendicular) energy & smooth
        E_perp = perpendicular_energy(sample.traces)
        E_perp_s = {p: moving_average(E_perp[p], k=args.sigma) for p in PRIMS}

        # 3) parse tasks + order
        parse = matched_filter_parse(sample.traces, sigma=args.sigma, proto_width=args.proto_width, topk=args.topk)

        # 4) execute predicted order
        grid_pred = execute_plan(sample.grid_in, parse.order)

        # 5) evaluate
        ok = bool(np.array_equal(grid_pred, sample.grid_out_true))
        correct += int(ok)

        # 6) plot
        plot_path = plot_parse(i, sample, E_perp_s, parse, args.plot_dir)

        # 7) pretty print
        areas = [round(parse.areas_perp[p], 3) for p in PRIMS]
        corr = [round(parse.corr_peak[p], 3) for p in PRIMS]
        print(f"[{i:02d}] -> Tasks: {parse.tasks} | Order: {' → '.join(parse.order) if parse.order else '—'} | ok={ok}")
        print(f"     areas⊥={areas} corr_peak={corr} plot={plot_path}")

    acc = correct / args.samples
    print(f"\n[Stage10 v2 — synthetic] Accuracy: {correct}/{args.samples} = {acc:.1%}")

if __name__ == "__main__":
    main()
