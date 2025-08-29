
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC Stage-10 v2 - Geodesic vs. Stock Baseline (HARD mode)
----------------------------------------------------------
Purpose: your earlier run showed 100% for both methods because the synthetic traces were
too clean. This variant introduces realistic confounds (overlap, cross-talk, distractor bumps,
timing jitter, stronger drift, and lower SNR) and upgrades the geodesic parser to use an
EXCLUSIVE residual (QR span-complement) to suppress cross-talk — per Stage-10 v2 math.

Usage:
  python3 arc-benchmark-geodesic-hard.py --samples 24 --seed 42 --T 720 --noise 0.02 --plot_dir plots_hard --hard 1

Author: ngeodesic — 2025-08-25
"""
import argparse, os
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

PRIMS = ["flip_h", "flip_v", "rotate"]

def set_seed(seed: int):
    return np.random.default_rng(seed)

def moving_average(x, k=9):
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
    sig2 = (width/2.355)**2
    return amp * np.exp(-(t-center)**2 / (2*sig2))

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
    if prim == "flip_h": return np.fliplr(grid)
    if prim == "flip_v": return np.flipud(grid)
    if prim == "rotate": return np.rot90(grid, k=-1)
    raise ValueError("unknown prim" )

def apply_sequence(grid, seq):
    g = grid.copy()
    for p in seq:
        g = apply_primitive(g, p)
    return g

def _z(x):
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    sd = x.std() + 1e-12
    return (x - mu) / sd

def exclusive_residual_matrix(E):
    """E: (T,K) channels -> span-complement residual per channel via QR."""
    T, K = E.shape
    R = np.zeros_like(E)
    Z = np.stack([_z(E[:,k]) for k in range(K)], axis=1)
    for k in range(K):
        if K == 1:
            R[:,k] = Z[:,k]
            continue
        B = np.delete(Z, k, axis=1)  # others
        Q, _ = np.linalg.qr(B, mode='reduced')
        x = Z[:,k]
        x_expl = Q @ (Q.T @ x)
        r = x - x_expl
        R[:,k] = r
    return R

def gen_synthetic_traces_hard(tasks: List[str], T: int, rng, noise=0.02,
                              overlap_prob=0.6, distractor_per_chan=2,
                              cross_talk=0.15, drift_strength=0.04,
                              width_frac=0.10, jitter_frac=0.03):
    """Harder generator: overlaps, distractors, cross-talk, drift, jitter."""
    base = {p: np.zeros(T, float) for p in PRIMS}
    proto_width = int(T*width_frac)
    # nominal centers equally spaced then jitter
    centers = [int(T*0.2), int(T*0.45), int(T*0.7)]
    # choose windows for true tasks
    order = tasks.copy()
    # optionally force overlaps by pulling centers together
    if rng.random() < overlap_prob and len(order) >= 2:
        shift = int(0.08*T)
        centers[:len(order)] = [centers[0]] + [centers[0]+shift*i for i in range(1,len(order))]
    # add true bumps with jitter and varying amplitude/width
    for idx, prim in enumerate(order):
        c = centers[idx % len(centers)]
        c += int(rng.normal(0, jitter_frac*T))
        w = int(proto_width * rng.uniform(0.8, 1.4))
        amp = rng.uniform(0.8, 1.2)
        base[prim] += gaussian_bump(T, np.clip(c,0,T-1), max(5,w), amp=amp)

    # add small distractor bumps in non-true channels
    for p in PRIMS:
        for _ in range(distractor_per_chan):
            c = rng.integers(int(0.05*T), int(0.95*T))
            w = int(proto_width * rng.uniform(0.6, 1.2))
            amp = rng.uniform(0.15, 0.35)  # small but not tiny
            base[p] += gaussian_bump(T, c, max(5,w), amp=amp)

    # stack and inject cross-talk: X := X + A X with small off-diagonals
    X = np.stack([base[p] for p in PRIMS], axis=1)  # (T,K)
    A = np.eye(len(PRIMS)) + cross_talk*(np.ones((len(PRIMS),len(PRIMS))) - np.eye(len(PRIMS)))
    X = X @ A.T

    # drift and noise
    drift = drift_strength * np.linspace(0, 1.0, T)[:,None]
    X = np.maximum(X + drift + noise * rng.standard_normal(X.shape), 0.0)

    # back to dict
    traces = {p: X[:,i] for i,p in enumerate(PRIMS)}
    return traces

def make_sample(rng, T=720, n_tasks=(1,3), grid_shape=(8,8), hard=False, **hard_kwargs):
    k = rng.integers(n_tasks[0], n_tasks[1]+1)
    tasks = list(rng.choice(PRIMS, size=k, replace=False))
    rng.shuffle(tasks)
    g0 = random_grid(rng, H=grid_shape[0], W=grid_shape[1])
    g1 = apply_sequence(g0, tasks)
    if hard:
        traces = gen_synthetic_traces_hard(tasks, T=T, rng=rng, **hard_kwargs)
    else:
        # easy mode as before
        def gaussian_bump(T, center, width, amp=1.0):
            t = np.arange(T)
            sig2 = (width/2.355)**2
            return amp * np.exp(-(t-center)**2 / (2*sig2))
        centers = [int(T*0.15), int(T*0.45), int(T*0.70)]
        width = int(T*0.10)
        base_drift = 0.002 * np.linspace(1.0, 1.1, T)
        traces = {p: np.zeros(T, dtype=float) for p in PRIMS}
        for idx, prim in enumerate(tasks):
            c = centers[idx % len(centers)]
            bump = gaussian_bump(T, c, width, amp=1.0)
            traces[prim] += bump
        for p in PRIMS:
            traces[p] += base_drift
            traces[p] = traces[p] + 0.01 * rng.standard_normal(T)
            traces[p] = np.clip(traces[p], 0, None)
    return Sample(g0, tasks, tasks, g1, traces, T)

@dataclass
class ParseResult:
    tasks: List[str]
    order: List[str]
    peak_times: Dict[str, int]
    areas: Dict[str, float]
    corr_peak: Dict[str, float]

def parse_geodesic_exclusive(traces: Dict[str, np.ndarray], sigma=11, proto_width=120, presence_frac=0.35):
    # smooth
    Es = np.stack([moving_average(traces[p], k=sigma) for p in PRIMS], axis=1)  # (T,K)
    # exclusive residual across channels
    R = exclusive_residual_matrix(Es)
    # positive part
    R = np.maximum(R, 0.0)
    T = R.shape[0]
    # matched filter with half-sine
    proto = half_sine(proto_width)
    proto = proto / (np.linalg.norm(proto) + 1e-8)
    peak_idx = {}
    corr_peak = {}
    areas = {}
    for i,p in enumerate(PRIMS):
        m = np.correlate(_z(R[:,i]), _z(proto), mode='same')
        idx = int(np.argmax(m))
        peak_idx[p] = idx
        L = proto_width
        a,b = max(0, idx-L//2), min(T, idx+L//2)
        w = R[a:b, i]
        w = (w - w.mean())
        pr = proto[:len(w)] - proto[:len(w)].mean()
        corr_peak[p] = float(np.dot(w, pr) / (np.linalg.norm(w)*np.linalg.norm(pr) + 1e-8))
        areas[p] = float(np.trapz(R[:,i]))
    # keep by area fraction of best and correlation relative to max
    Amax = max(areas.values()) + 1e-12
    Cmax = max(corr_peak.values()) + 1e-12
    keep = [p for p in PRIMS if (areas[p]/Amax) >= presence_frac and (corr_peak[p]/Cmax) >= 0.45]
    if not keep:
        # fallback best score
        score = {p: corr_peak[p] * (areas[p]/Amax) for p in PRIMS}
        keep = [max(score, key=score.get)]
    order = sorted(keep, key=lambda p: peak_idx[p])
    return ParseResult(keep, order, peak_idx, areas, corr_peak)

def parse_stock_raw(traces: Dict[str, np.ndarray], sigma=11, proto_width=120, presence_frac=0.35):
    # just smooth the raw channels; no residualization
    Es = {p: moving_average(traces[p], k=sigma) for p in PRIMS}
    T = len(next(iter(Es.values())))
    proto = half_sine(proto_width)
    proto = proto / (np.linalg.norm(proto) + 1e-8)
    peak_idx = {}
    corr_peak = {}
    areas = {}
    for p in PRIMS:
        idx = int(np.argmax(Es[p]))
        peak_idx[p] = idx
        L = proto_width
        a,b = max(0, idx-L//2), min(T, idx+L//2)
        w = Es[p][a:b]
        w = (w - w.mean())
        pr = proto[:len(w)] - proto[:len(w)].mean()
        corr_peak[p] = float(np.dot(w, pr) / (np.linalg.norm(w)*np.linalg.norm(pr) + 1e-8))
        areas[p] = float(np.trapz(Es[p]))
    Amax = max(areas.values()) + 1e-12
    Cmax = max(corr_peak.values()) + 1e-12
    keep = [p for p in PRIMS if (areas[p]/Amax) >= presence_frac and (corr_peak[p]/Cmax) >= 0.45]
    if not keep:
        score = {p: corr_peak[p] * (areas[p]/Amax) for p in PRIMS}
        keep = [max(score, key=score.get)]
    order = sorted(keep, key=lambda p: peak_idx[p])
    return ParseResult(keep, order, peak_idx, areas, corr_peak)

def execute_plan(grid, order):  # exact same executor
    g = grid.copy()
    for p in order:
        if p == "flip_h": g = np.fliplr(g)
        elif p == "flip_v": g = np.flipud(g)
        elif p == "rotate": g = np.rot90(g, k=-1)
    return g

def seq_exact(pred, true_seq): return list(pred) == list(true_seq)
def set_exact(pred, true_seq): return set(pred) == set(true_seq)

def f1_for_sets(pred, true_set):
    pred_set, true_set = set(pred), set(true_set)
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1 = 2*prec*rec / (prec+rec + 1e-12)
    return f1, prec, rec

def plot_pair(i, sample, Rgeo, Es_raw, pg, ps, outdir):
    os.makedirs(outdir, exist_ok=True)
    T = sample.T
    fig, axes = plt.subplots(2,1, figsize=(12,7), sharex=True)
    # top: geodesic exclusive residuals
    for k,p in enumerate(PRIMS):
        axes[0].plot(np.maximum(Rgeo[:,k],0.0), label=f"E_ex {p}", linewidth=2)
    axes[0].set_title(f"[Geodesic excl] tasks={pg.tasks} order={' -> '.join(pg.order)}")
    axes[0].legend(loc='upper right'); axes[0].set_ylabel("excl residual")
    # bottom: stock raw
    for p in PRIMS:
        axes[1].plot(Es_raw[p], label=f"E_raw {p}", linewidth=2)
    axes[1].set_title(f"[Stock raw] tasks={ps.tasks} order={' -> '.join(ps.order)}")
    axes[1].legend(loc='upper right'); axes[1].set_xlabel("step"); axes[1].set_ylabel("raw power")
    plt.suptitle(f"Sample {i:02d} — true order: {' -> '.join(sample.order_true)}", y=0.98)
    plt.tight_layout(rect=[0,0,1,0.96])
    path = os.path.join(outdir, f"sample{i:02d}.png")
    plt.savefig(path, dpi=120); plt.close()
    return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--plot_dir", type=str, default="plots_hard")
    ap.add_argument("--sigma", type=int, default=11)
    ap.add_argument("--proto_width", type=int, default=120)
    ap.add_argument("--presence_frac", type=float, default=0.35)
    ap.add_argument("--hard", type=int, default=1, help="1=hard generator with overlaps/crosstalk/distractors")
    args = ap.parse_args()

    rng = set_seed(args.seed)

    # tallies
    grid_geod = grid_stock = 0
    seq_g = seq_s = 0
    setF1_g = []; setF1_s = []

    for i in range(1, args.samples+1):
        sample = make_sample(rng, T=args.T, noise=args.noise, hard=bool(args.hard),
                             overlap_prob=0.6, distractor_per_chan=2, cross_talk=0.15,
                             drift_strength=0.04, width_frac=0.10, jitter_frac=0.03)

        # signals
        Es_raw = {p: moving_average(sample.traces[p], k=args.sigma) for p in PRIMS}
        M_raw = np.stack([Es_raw[p] for p in PRIMS], axis=1)
        Rgeo = exclusive_residual_matrix(M_raw)

        # parse
        pg = parse_geodesic_exclusive(sample.traces, sigma=args.sigma, proto_width=args.proto_width, presence_frac=args.presence_frac)
        ps = parse_stock_raw(sample.traces, sigma=args.sigma, proto_width=args.proto_width, presence_frac=args.presence_frac)

        # execute & score
        g_geod = execute_plan(sample.grid_in, pg.order)
        g_stock = execute_plan(sample.grid_in, ps.order)

        ok_g = bool(np.array_equal(g_geod, sample.grid_out_true))
        ok_s = bool(np.array_equal(g_stock, sample.grid_out_true))
        grid_geod += int(ok_g); grid_stock += int(ok_s)
        seq_g += int(seq_exact(pg.order, sample.order_true)); seq_s += int(seq_exact(ps.order, sample.order_true))

        f1g, pgp, pgr = f1_for_sets(pg.tasks, sample.order_true)
        f1s, psp, psr = f1_for_sets(ps.tasks, sample.order_true)
        setF1_g.append(f1g); setF1_s.append(f1s)

        path = plot_pair(i, sample, Rgeo, Es_raw, pg, ps, args.plot_dir)
        print(f"[{i:02d}] TRUE: {sample.order_true}")
        print(f"     GEO: tasks={pg.tasks} F1={f1g:.2f} | order={' -> '.join(pg.order)} | grid_ok={ok_g}")
        print(f"     STK: tasks={ps.tasks} F1={f1s:.2f} | order={' -> '.join(ps.order)} | grid_ok={ok_s}")
        print(f"     plot={path}\n")

    n = args.samples
    print(f"[SUMMARY] Grid exact — Geodesic: {grid_geod}/{n} = {grid_geod/n:.1%} | Stock: {grid_stock}/{n} = {grid_stock/n:.1%}")
    print(f"[SUMMARY] Seq exact  — Geodesic: {seq_g}/{n} = {seq_g/n:.1%} | Stock: {seq_s}/{n} = {seq_s/n:.1%}")
    print(f"[SUMMARY] Task-set F1 — Geodesic: {np.mean(setF1_g):.2f} | Stock: {np.mean(setF1_s):.2f}")

if __name__ == "__main__":
    main()
