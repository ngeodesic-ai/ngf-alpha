# Regenerate (overwrite) Stage 11 v7 script at the same path.
# from textwrap import dedent

# code = dedent(r'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage11-well-benchmark-v7.py
----------------------------
Phase 6 (v7): Recall-first with precision guardrails.

Adds to v6:
- OS-CFAR / FDR-style adaptive gating (Pfa or BH q across primitives)
- Multi-scale matched filter (3 widths) with 2-of-3 time-consistency rule
- Confidence-weighted, overlap-adaptive inhibition
- SPRT+ cumulative refinement with trend check
- Recall-rescue pass (+1) with overlap constraint
- MDL prior on set size
- Fast/Full modes + richer logging

This file is a synthetic harness; substitute real traces where indicated to run on your data.

python3 stage11-well-benchmark-v7.py --mode full --samples 50 --nperm 500 \
  --scales 120,160,210 --pfa 0.10 --guard 24 --bg 72 \
  --lambda0 0.22 --inhib_sigma 1.6 --z0 1.0 --alpha 1.2 \
  --T0 2.8 --Tmin 0.7 --anneal_steps 6 --p_floor 0.10 \
  --sprt_tau 0.018 --drop_floor 0.004 --trend_need 3 \
  --rescue 1 --rescue_eps 0.06 --rescue_overlap 0.30 \
  --mdl_beta 0.015 \
  --pi --dump_surfaces_dir dumps/v7_surfaces --dump_manifold dumps/v7_manifold.npz

"""

import argparse, os, csv, json, math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

PRIMS = ["flip_h","flip_v","rotate"]

# ----------------------------- utils -----------------------------

def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)

def moving_average(x: np.ndarray, k: int=9) -> np.ndarray:
    if k <= 1:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(xp, np.ones(k)/k, mode="valid")

def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    sd = x.std() + 1e-12
    return (x - mu) / sd

def half_sine(width: int) -> np.ndarray:
    return np.sin(np.linspace(0, np.pi, width))

def half_sine_proto(width: int) -> np.ndarray:
    p = half_sine(width)
    return p / (np.linalg.norm(p) + 1e-12)

def gaussian(tdiff: float, sigma: float) -> float:
    return float(np.exp(-0.5 * (tdiff / (sigma + 1e-12))**2))

def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-x)))

def window(a: int, b: int, T: int) -> Tuple[int,int]:
    return max(0,a), min(T,b)

def overlap_frac(a1: int, b1: int, a2: int, b2: int) -> float:
    inter = max(0, min(b1, b2) - max(a1, a2))
    union = max(b1, b2) - min(a1, a2) + 1e-9
    return inter / union

# ------------------------ block permutation null ------------------------

def block_roll(x: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    if block_len <= 1 or block_len >= n:
        k = int(rng.integers(1, n-1))
        return np.roll(x, k)
    m = int(np.ceil(n / block_len)) * block_len
    xp = np.pad(x, (0, m - n), mode="wrap")
    xb = xp.reshape(m // block_len, block_len)
    k_blocks = int(rng.integers(1, xb.shape[0]))
    xb2 = np.roll(xb, k_blocks, axis=0)
    y = xb2.reshape(-1)[:n]
    return y

def corr_at(sig: np.ndarray, proto: np.ndarray, center_idx: int, width: int) -> float:
    T = len(sig)
    a, b = max(0, center_idx - width//2), min(T, center_idx + width//2)
    w = sig[a:b]
    if len(w) < 3:
        return 0.0
    w0 = w - w.mean()
    pr = proto[:len(w)] - proto[:len(w)].mean()
    denom = (np.linalg.norm(w0) * np.linalg.norm(pr) + 1e-12)
    return float(np.dot(w0, pr) / denom)

def perm_null_z(sig: np.ndarray, proto: np.ndarray, peak_idx: int, width: int, rng: np.random.Generator,
                nperm: int=500, block_frac: float=0.16) -> float:
    T = len(sig)
    obs = corr_at(sig, proto, peak_idx, width)
    block_len = max(1, int(block_frac * T))
    null = np.empty(nperm, dtype=float)
    for i in range(nperm):
        x = block_roll(sig, block_len, rng)
        null[i] = corr_at(x, proto, peak_idx, width)
    mu = float(null.mean())
    sd = float(null.std() + 1e-8)
    return (obs - mu) / sd

# ----------------------------- synthetic world -----------------------------

@dataclass
class Sample:
    grid_in: np.ndarray
    tasks_true: List[str]
    order_true: List[str]
    grid_out_true: np.ndarray
    traces: Dict[str, np.ndarray]
    centers_true: Dict[str, int]
    T: int

def random_grid(rng: np.random.Generator, H=8, W=8, ncolors=6) -> np.ndarray:
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

def gaussian_bump(T: int, center: int, width: int, amp: float=1.0) -> np.ndarray:
    t = np.arange(T)
    sig2 = (width / 2.355)**2  # FWHM -> sigma
    return amp * np.exp(-(t - center)**2 / (2*sig2))

def gen_traces(tasks: List[str], T: int, rng: np.random.Generator,
               noise: float=0.02, cm_amp: float=0.02, overlap: float=0.5,
               amp_jitter: float=0.4, distractor_prob: float=0.4) -> Tuple[Dict[str,np.ndarray], Dict[str,int]]:
    base = np.array([0.20, 0.50, 0.80]) * T
    centers = ((1.0 - overlap) * base + overlap * (T * 0.50)).astype(int)
    width = int(max(12, T * 0.10))
    t = np.arange(T)
    cm = cm_amp * (1.0 + 0.2 * np.sin(2*np.pi * t / max(30, T//6)))
    traces = {p: np.zeros(T, float) for p in PRIMS}
    centers_true = {}

    # true bumps
    for i, prim in enumerate(tasks):
        c = centers[i % len(centers)]
        amp = max(0.25, 1.0 + rng.normal(0, amp_jitter))
        c_jit = int(np.clip(c + rng.integers(-width//5, width//5 + 1), 0, T-1))
        traces[prim] += gaussian_bump(T, c_jit, width, amp=amp)
        centers_true[prim] = c_jit

    # distractors
    for p in PRIMS:
        if p not in tasks and rng.random() < distractor_prob:
            c = int(rng.uniform(T*0.15, T*0.85))
            amp = max(0.25, 1.0 + rng.normal(0, amp_jitter))
            traces[p] += gaussian_bump(T, c, width, amp=0.9*amp)

    # add small common-mode + noise
    for p in PRIMS:
        traces[p] = np.clip(traces[p] + cm, 0.0, None)
        traces[p] = traces[p] + rng.normal(0, noise, size=T)
        traces[p] = np.maximum(traces[p], 0.0)
    return traces, centers_true

def make_sample(rng: np.random.Generator, T: int=720, n_tasks=(1,3), noise=0.02,
                allow_repeats=False) -> Sample:
    k = rng.integers(n_tasks[0], n_tasks[1]+1)
    tasks = list(rng.choice(PRIMS, size=k, replace=allow_repeats))
    if not allow_repeats:
        rng.shuffle(tasks)
    g0 = random_grid(rng, H=8, W=8, ncolors=6)
    g1 = apply_sequence(g0, tasks)
    traces, centers_true = gen_traces(tasks, T=T, rng=rng, noise=noise)
    return Sample(g0, tasks, tasks, g1, traces, centers_true, T)

# ------------------------ Orthogonalization ------------------------

def low_rank_orthogonalize(traces: Dict[str,np.ndarray], rank_r: int = 1) -> Dict[str,np.ndarray]:
    keys = list(traces.keys())
    X = np.stack([traces[p] for p in keys], axis=0)  # (P,T)
    P, T = X.shape
    r = int(max(0, min(rank_r, P)))
    if r == 0:
        X_perp = X.copy()
    else:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        Ur = U[:, :r]  # (P,r)
        P_perp = np.eye(P) - Ur @ Ur.T  # (P,P)
        X_perp = P_perp @ X
    X_perp = np.clip(X_perp, 0.0, None)
    return {p: X_perp[i] for i,p in enumerate(keys)}

# ------------------------ Multi-scale matched filter ------------------------

def matched_z(sig: np.ndarray, center: int, width: int, rng: np.random.Generator, nperm: int, block_frac: float) -> float:
    proto = half_sine_proto(width)
    return perm_null_z(sig, proto, center, width, rng, nperm=nperm, block_frac=block_frac)

def multi_scale_peaks_and_z(sig: np.ndarray, widths: List[int], rng: np.random.Generator, nperm: int, block_frac: float, base_width: int) -> Tuple[Dict[int,int], Dict[int,float]]:
    base_proto = half_sine_proto(base_width)
    base_peak = int(np.argmax(np.correlate(sig, base_proto, mode="same")))
    peaks = {}
    zs = {}
    for w in widths:
        peaks[w] = base_peak
        zs[w] = matched_z(sig, base_peak, w, rng, nperm, block_frac)
    return peaks, zs

def twothree_consistent(peaks: Dict[int,int], tol: int) -> bool:
    ws = sorted(peaks.keys())
    ts = [peaks[w] for w in ws]
    ok = 0
    for i in range(len(ts)):
        for j in range(i+1,len(ts)):
            if abs(ts[i]-ts[j]) <= tol:
                ok += 1
    return ok >= 2  # at least two scales consistent

# ------------------------ OS-CFAR / FDR gating ------------------------

def ring_stats(series: np.ndarray, peak: int, guard: int, bg: int) -> np.ndarray:
    T = len(series)
    lo1, hi1 = max(0, peak - (guard + bg)), max(0, peak - guard)
    lo2, hi2 = min(T, peak + guard), min(T, peak + guard + bg)
    ring = []
    if hi1 > lo1: ring.append(series[lo1:hi1])
    if hi2 > lo2: ring.append(series[lo2:hi2])
    if not ring:
        return np.array(series)
    return np.concatenate(ring)

def os_cfar(series: np.ndarray, peak: int, guard: int, bg: int, pfa: float) -> Tuple[float,bool]:
    ring = ring_stats(series, peak, guard, bg)
    if len(ring) < 8:
        thr = float(series.mean() + 2.0*series.std())
    else:
        thr = float(np.quantile(ring, 1.0 - pfa))
    return thr, bool(series[peak] >= thr)

def bh_fdr(pvals: List[float], q: float) -> List[bool]:
    m = len(pvals)
    idx = np.argsort(pvals)
    passed = [False]*m
    thresh_idx = -1
    for k, i in enumerate(idx, start=1):
        if pvals[i] <= (k/m) * q:
            thresh_idx = k
    if thresh_idx >= 0:
        for k, i in enumerate(idx, start=1):
            passed[i] = (k <= thresh_idx)
    return passed

# ------------------------ Confidence-weighted inhibition ------------------------

def apply_conf_weighted_inhibition(scores: Dict[str,float], peaks: Dict[str,int], Z: Dict[str,float],
                                   lambda0: float, sigma: float, z0: float, alpha: float) -> Dict[str,float]:
    s = scores.copy()
    prims = list(scores.keys())
    weights = {p: 1.0 - sigmoid(alpha*(Z[p] - z0)) for p in prims}
    for p in prims:
        pen = 0.0
        for q in prims:
            if p == q: continue
            pen += lambda0 * gaussian(abs(peaks[p]-peaks[q]), sigma) * weights[p]
        s[p] = s[p] - pen
    return s

# ------------------------ SPRT+ cumulative refinement ------------------------

def cumulative_refinement_trend(Sres: Dict[str,np.ndarray], peaks: Dict[str,int], proto_width: int,
                                steps: int, drop_floor: float, tau_total: float, trend_need: int = 3) -> List[str]:
    proto = half_sine_proto(proto_width)
    keys = list(Sres.keys())
    T = len(next(iter(Sres.values())))
    selected = []
    cumulative = {p: 0.0 for p in keys}
    series = {p: [] for p in keys}
    Swork = {p: Sres[p].copy() for p in keys}

    for _ in range(max(1, steps)):
        base_energy = sum(float(np.trapz(Swork[p])) for p in keys)
        step_drop = {p: 0.0 for p in keys}
        for p in keys:
            a, b = window(peaks[p]-proto_width//2, peaks[p]+proto_width//2, T)
            shaved = {q: Swork[q].copy() for q in keys}
            for q in keys:
                if q == p: continue
                shaved[q][a:b] = 0.88 * shaved[q][a:b]
            new_energy = sum(float(np.trapz(shaved[q])) for q in keys)
            drop = (base_energy - new_energy) / max(1e-9, base_energy)
            drop = max(0.0, drop)
            step_drop[p] = drop
            series[p].append(drop)

        for p in keys:
            if step_drop[p] >= drop_floor:
                cumulative[p] += step_drop[p]

        best_p = max(keys, key=lambda k: step_drop[k])
        a, b = window(peaks[best_p]-proto_width//2, peaks[best_p]+proto_width//2, T)
        for q in keys:
            if q == best_p: continue
            Swork[q][a:b] = 0.92 * Swork[q][a:b]

    def rising_trend(xs: List[float]) -> bool:
        inc = sum(1 for i in range(1, len(xs)) if xs[i] >= xs[i-1]+1e-9)
        return inc >= trend_need

    for p in keys:
        if cumulative[p] >= tau_total and rising_trend(series[p]):
            selected.append(p)
    if not selected:
        selected = [max(keys, key=lambda k: cumulative[k])]
    return selected

# -------------------------- phantom index (PI) --------------------------

def find_local_minima(x: np.ndarray, delta: float=0.0) -> List[int]:
    mins = []
    for i in range(1, len(x)-1):
        if x[i] + delta < x[i-1] and x[i] + delta < x[i+1]:
            mins.append(i)
    return mins

def compute_phantom_index(Utime: Dict[str,np.ndarray], centers_true: Dict[str,int], window: int) -> float:
    total = 0
    phantom = 0
    T = len(next(iter(Utime.values())))
    tol = max(2, window // 2)
    for p in PRIMS:
        mins = find_local_minima(Utime[p])
        total += len(mins)
        true_center = centers_true.get(p, None)
        for m in mins:
            if true_center is None or abs(m - true_center) > tol:
                phantom += 1
    return float(phantom) / (max(1, total))

# ------------------------ surfaces & manifold dumps ------------------------

def dump_surfaces(sample_id: int, out_dir: str, Sraw: Dict[str,np.ndarray], Sres: Dict[str,np.ndarray],
                  Utime: Dict[str,np.ndarray]):
    if not out_dir: return
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"sample{sample_id:03d}.npz")
    np.savez_compressed(path, Sraw=Sraw, Sres=Sres, U=Utime, prims=np.array(PRIMS, dtype=object))

def synthetic_manifold_dump(path: str, Sraw: Dict[str,np.ndarray], Utime: Dict[str,np.ndarray]):
    if not path: return
    T = len(next(iter(Sraw.values())))
    d = 8
    X = np.zeros((T, d))
    B = np.zeros((3, d)); B[0,0]=1.0; B[1,1]=1.0; B[2,2]=1.0
    for i,p in enumerate(PRIMS):
        X[:, :d] += np.outer(Sraw[p], B[i])
    rng = np.random.default_rng(0)
    X[:, 3:] += 0.02 * rng.standard_normal((T, d-3))
    U = sum(Utime[p] for p in PRIMS)
    np.savez_compressed(path, Y=X, U=U, U_k=np.stack([Utime[p] for p in PRIMS], axis=1),
                        names=np.array(PRIMS, dtype=object))

# ------------------------------- metrics -------------------------------

def grid_similarity(gp: np.ndarray, gt: np.ndarray) -> float:
    if gp.shape != gt.shape: return 0.0
    return float((gp == gt).mean())

def set_metrics(true_list: List[str], pred_list: List[str]) -> Dict[str,float]:
    Tset, Pset = set(true_list), set(pred_list)
    tp = len(Tset & Pset); fp = len(Pset - Tset); fn = len(Tset - Pset)
    precision = tp / max(1, len(Pset))
    recall    = tp / max(1, len(Tset))
    f1 = 0.0 if precision+recall == 0 else (2*precision*recall)/(precision+recall)
    jacc = tp / max(1, len(Tset | Pset))
    return dict(precision=precision, recall=recall, f1=f1, jaccard=jacc,
                hallucination_rate=fp/max(1,len(Pset)), omission_rate=fn/max(1,len(Tset)))

# ------------------------------- main -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage 11 — Phase 6 (v7): adaptive gating + multi-scale + conf-weighted inhibition + SPRT+ trend + rescue + MDL")
    ap.add_argument("--mode", type=str, default="full", choices=["fast","full"])
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=48)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--sigma", type=int, default=9, help="smoother window")
    ap.add_argument("--proto_width", type=int, default=160, help="base prototype width")
    ap.add_argument("--scales", type=str, default="120,160,210", help="comma list of widths for multi-scale")
    # Orthogonalization
    ap.add_argument("--rank_r", type=int, default=1)
    # OS-CFAR / FDR
    ap.add_argument("--pfa", type=float, default=0.10)
    ap.add_argument("--guard", type=int, default=24)
    ap.add_argument("--bg", type=int, default=72)
    ap.add_argument("--use_fdr", action="store_true", help="use BH-FDR across primitives instead of OS-CFAR")
    ap.add_argument("--fdr_q", type=float, default=0.10)
    # Inhibition
    ap.add_argument("--lambda0", type=float, default=0.22)
    ap.add_argument("--inhib_sigma", type=float, default=1.6)
    ap.add_argument("--z0", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.2)
    # Anneal
    ap.add_argument("--T0", type=float, default=2.8)
    ap.add_argument("--Tmin", type=float, default=0.7)
    ap.add_argument("--anneal_steps", type=int, default=6)
    ap.add_argument("--p_floor", type=float, default=0.10)
    # SPRT+
    ap.add_argument("--sprt_tau", type=float, default=0.018)
    ap.add_argument("--drop_floor", type=float, default=0.004)
    ap.add_argument("--trend_need", type=int, default=3)
    # Rescue
    ap.add_argument("--rescue", type=int, default=1, help="0/1 to disable/enable rescue pass")
    ap.add_argument("--rescue_eps", type=float, default=0.06)
    ap.add_argument("--rescue_overlap", type=float, default=0.30)
    # MDL prior
    ap.add_argument("--mdl_beta", type=float, default=0.015)
    # Diagnostics
    ap.add_argument("--dump_surfaces_dir", type=str, default="")
    ap.add_argument("--dump_manifold", type=str, default="")
    ap.add_argument("--pi", action="store_true")
    # Null calibration
    ap.add_argument("--null-block-frac", type=float, default=0.16)
    ap.add_argument("--nperm", type=int, default=500)
    # Output
    ap.add_argument("--csv", type=str, default="stage11_v7.csv")
    args = ap.parse_args()

    # Mode presets
    if args.mode == "fast":
        args.nperm = min(args.nperm, 150)
        args.samples = min(args.samples, 20)
        args.anneal_steps = max(args.anneal_steps, 4)
        args.scales = "160"  # single scale
        args.rescue = 0

    rng = np.random.default_rng(args.seed)

    # Build widths
    try:
        widths = sorted(set(int(x.strip()) for x in args.scales.split(",") if x.strip()))
    except Exception:
        widths = [args.proto_width]
    if args.proto_width not in widths:
        widths.append(args.proto_width)
    widths = sorted(widths)

    rows = []
    agg = dict(acc=0, grid=0.0, P=0.0, R=0.0, F1=0.0, J=0.0, H=0.0, O=0.0,
               margin_mu=0.0, margin_min=0.0, PI=0.0)
    all_margins = []

    for i in range(1, args.samples+1):
        sample = make_sample(rng, T=args.T, noise=args.noise)

        # 1) Orthogonalization & smoothing
        Eperp = low_rank_orthogonalize(sample.traces, rank_r=args.rank_r)
        Sres = {p: moving_average(Eperp[p], k=args.sigma) for p in PRIMS}
        Sraw = {p: moving_average(sample.traces[p], k=args.sigma) for p in PRIMS}
        Scm  = moving_average(np.stack([sample.traces[p] for p in PRIMS],0).mean(0), k=args.sigma)

        # 2) Multi-scale peaks & z at base peak
        base_proto = half_sine_proto(args.proto_width)
        rng_i = np.random.default_rng(54321 + i)
        base_peak = {}
        peaks = {}
        z_res = {}
        scale_pass = {}
        tol_tc = int(0.1 * args.proto_width)

        for p in PRIMS:
            base_peak[p] = int(np.argmax(np.correlate(Sres[p], base_proto, mode="same")))
            p_peaks, p_zs = multi_scale_peaks_and_z(Sres[p], widths, rng_i, args.nperm, args.null_block_frac, args.proto_width)
            peaks[p] = p_peaks  # dict width->index (all equal to base_peak)
            z_res[p] = p_zs     # dict width->z
            scale_pass[p] = twothree_consistent(p_peaks, tol=tol_tc)

        # 3) OS-CFAR / FDR gating
        z_series = {p: _zscore(Sres[p]) for p in PRIMS}
        passed = []
        if args.use_fdr:
            from math import erf, sqrt
            def p_from_z(z):
                return 2.0 * (1.0 - 0.5*(1.0 + erf(abs(z)/sqrt(2.0))))
            pvals = [p_from_z(z_res[p][args.proto_width]) for p in PRIMS]
            keep_flags = bh_fdr(pvals, args.fdr_q)
            for idx, p in enumerate(PRIMS):
                if keep_flags[idx] and scale_pass[p]:
                    passed.append(p)
        else:
            for p in PRIMS:
                thr, ok = os_cfar(z_series[p], base_peak[p], args.guard, args.bg, args.pfa)
                if ok and scale_pass[p]:
                    passed.append(p)

        if not passed:
            passed = [max(PRIMS, key=lambda q: z_res[q][args.proto_width])]

        # 4) Scores (res/raw/cm) and confidence-weighted inhibition
        w_res, w_raw, w_cm = 1.0, 0.45, 0.25
        z_raw = {}
        z_cm = {}
        for p in PRIMS:
            z_raw[p] = perm_null_z(Sraw[p], base_proto, base_peak[p], args.proto_width, rng_i, nperm=max(120, args.nperm//3), block_frac=args.null_block_frac)
            z_cm[p]  = perm_null_z(Scm,      base_proto, base_peak[p], args.proto_width, rng_i, nperm=max(120, args.nperm//3), block_frac=args.null_block_frac)

        Zp_conf = {p: max(z_res[p].values()) for p in passed}
        score = {p: w_res*max(0.0, Zp_conf[p]) + w_raw*max(0.0, z_raw[p]) - w_cm*max(0.0, z_cm[p]) for p in passed}
        score_inhib = apply_conf_weighted_inhibition(
            score, {p: base_peak[p] for p in passed}, Zp_conf,
            lambda0=args.lambda0, sigma=args.inhib_sigma, z0=args.z0, alpha=args.alpha
        )

        # 5) Temperature soft presence (collect union across anneal)
        Up = {p: -score_inhib[p] for p in passed}
        keys = list(Up.keys())
        Ts = list(np.linspace(args.T0, args.Tmin, max(1, args.anneal_steps)))

        def soft_keep(Tcur: float) -> List[str]:
            Uv = np.array([Up[k] for k in keys], dtype=float)
            if len(Uv) == 0:
                return []
            m = np.max(-Uv / max(Tcur, 1e-6))
            ex = np.exp(-Uv / max(Tcur, 1e-6) - m)
            pi = ex / (np.sum(ex) + 1e-12)
            keep = [keys[i] for i,pv in enumerate(pi) if pv >= args.p_floor]
            if not keep:
                keep = [keys[int(np.argmax(pi))]]
            return keep

        cumul_set = set()
        for Tcur in Ts:
            cumul_set.update(soft_keep(Tcur))
        if not cumul_set:
            cumul_set = set(passed)

        # 6) SPRT+ cumulative refinement with trend
        Ssub = {p: Sres[p].copy() for p in cumul_set}
        keep = cumulative_refinement_trend(Ssub, {p: base_peak[p] for p in Ssub.keys()}, args.proto_width,
                                           steps=len(Ts), drop_floor=args.drop_floor, tau_total=args.sprt_tau, trend_need=args.trend_need)

        # 7) Recall rescue (+1) if residual energy suggests a missed basin
        if args.rescue and len(keep) < len(PRIMS):
            Tlen = len(next(iter(Sres.values())))
            w = args.proto_width
            kept_windows = [(window(base_peak[p]-w//2, base_peak[p]+w//2, Tlen)) for p in keep]
            residual_energy = {}
            for p in PRIMS:
                if p in keep: 
                    residual_energy[p] = 0.0
                    continue
                a, b = window(base_peak[p]-w//2, base_peak[p]+w//2, Tlen)
                if any(overlap_frac(a,b,ka,kb) > args.rescue_overlap for (ka,kb) in kept_windows):
                    residual_energy[p] = 0.0
                else:
                    residual_energy[p] = float(np.trapz(Sres[p][a:b])) / max(1e-9, float(np.trapz(Sres[p])))
            if residual_energy:
                cand = max(residual_energy, key=lambda k: residual_energy[k])
                if residual_energy[cand] >= args.rescue_eps:
                    keep.append(cand)

        # 8) MDL prior (light) on set size
        if len(keep) > 2 and args.mdl_beta > 0.0:
            proxy = {p: score_inhib.get(p, 0.0) - args.mdl_beta for p in keep}
            worst = min(keep, key=lambda p: proxy[p])
            if proxy[worst] < 0:
                keep.remove(worst)

        # 9) Ordering by matched filter time on raw
        proto = base_proto
        order_peaks = {}
        for p in keep:
            m = np.correlate(Sraw[p], proto, mode="same")
            order_peaks[p] = int(np.argmax(m))
        order = sorted(keep, key=lambda p: order_peaks[p])

        # --- Metrics & dumps ---
        gp = apply_sequence(sample.grid_in, order)
        ok = int(np.array_equal(gp, sample.grid_out_true))
        gs = grid_similarity(gp, sample.grid_out_true)
        sm = set_metrics(sample.order_true, keep)

        zcm_series = _zscore(Scm)
        Utime = {}
        for p in PRIMS:
            Utime[p] = -(_zscore(Sres[p]) + 0.4*_zscore(Sraw[p]) - 0.25*zcm_series)

        margins = []
        for p in PRIMS:
            t_true = sample.centers_true.get(p, None)
            if t_true is None: continue
            Tlen = len(Utime[p])
            w = args.proto_width
            a, b = window(t_true - w//2, t_true + w//2, Tlen)
            u_true = float(np.min(Utime[p][a:b]))
            others = [q for q in PRIMS if q != p]
            u_false = min(float(np.min(Utime[q])) for q in others)
            margins.append(u_false - u_true)
        margin_mu = float(np.mean(margins)) if margins else 0.0
        margin_min = float(min(margins)) if margins else 0.0
        all_margins.extend(margins)

        PI = compute_phantom_index(Utime, sample.centers_true, window=args.proto_width) if args.pi else 0.0

        dump_surfaces(i, args.dump_surfaces_dir, Sraw, Sres, Utime)
        if args.dump_manifold and i == 1:
            synthetic_manifold_dump(args.dump_manifold, Sraw, Utime)

        rows.append(dict(
            sample=i,
            true="|".join(sample.order_true),
            keep="|".join(keep),
            order="|".join(order),
            ok=ok, grid=gs, precision=sm["precision"], recall=sm["recall"],
            f1=sm["f1"], jaccard=sm["jaccard"], hallucination=sm["hallucination_rate"], omission=sm["omission_rate"],
            margin_mu=margin_mu, margin_min=margin_min, PI=PI,
        ))

        # aggregates
        agg["acc"] += ok; agg["grid"] += gs
        agg["P"] += sm["precision"]; agg["R"] += sm["recall"]; agg["F1"] += sm["f1"]; agg["J"] += sm["jaccard"]
        agg["H"] += sm["hallucination_rate"]; agg["O"] += sm["omission_rate"]
        agg["margin_mu"] += margin_mu; agg["margin_min"] += margin_min
        agg["PI"] += PI

        print(f"[{i:02d}] true={sample.order_true} | keep={keep} | order={order} | ok={bool(ok)} | P={sm['precision']:.3f} R={sm['recall']:.3f} H={sm['hallucination_rate']:.3f}")

    n = float(len(rows))
    summary = {k: (v/n if isinstance(v, (int,float)) else v) for k,v in agg.items()}
    if all_margins:
        summary["margin_mu"] = float(np.mean(all_margins))
        summary["margin_min"] = float(np.min(all_margins))

    # Write outputs
    if rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows: w.writerow(r)
    with open(os.path.splitext(args.csv)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[Stage11 v7] Summary:", {k: (round(v,3) if isinstance(v,float) else v) for k,v in summary.items()})
    print("[Stage11 v7] CSV:", args.csv)
    print("[Stage11 v7] Summary JSON:", os.path.splitext(args.csv)[0] + "_summary.json")

if __name__ == "__main__":
    main()
# ''')

# path = "/mnt/data/stage11-well-benchmark-v7.py"
# with open(path, "w") as f:
#     f.write(code)

# path
