#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 11 — Well Benchmark v8 (report baseline + sharp-well viz)
----------------------------------------------------------------
This script preserves the -report baseline metrics/outputs and adds an optional
"sharp well" visualization mode that better matches the toy-model/logo geometry.

Key additions (viz-only; metrics unchanged):
  • Composite radial potential for PC3:
      U(r) = λ1 * exp(-r^2/(2σ^2)) + λ2 / (r^2 + ε), masked near the center
      z'   = -U(r) + β * (z_orig - mean)
  • Radial symmetry averaging in (PC1, PC2): bin by angle θ and average z over r
  • Side-by-side plots: RAW (baseline warp) vs SHARP (logo-like)

Outputs (same filenames as baseline unless overridden):
  - manifold_pca3_mesh_warped.png              (RAW or SHARP, depending on --well_model)
  - manifold_pca3_mesh_warped_v8_compare.png   (RAW vs SHARP, always emitted when --well_model=sharp)
  - stage11_metrics.csv
  - stage11_summary.json

python3 stage11-well-benchmark-v8.py \
  --samples 200 --seed 42 --T 720 --sigma 9 \
  --out_plot manifold_pca3_mesh_warped.png \
  --out_csv stage11_metrics.csv \
  --out_json stage11_summary.json


python3 stage11-well-benchmark-v8.py \
  --samples 200 --seed 42 --T 720 --sigma 9 \
  --well_model sharp \
  --sigma_scale 0.80 --depth_scale 1.35 \
  --core_strength 0.9 --core_eps 0.02 --core_sigma_factor 0.6 \
  --beta 0.08 --sym_bins 64 \
  --out_plot manifold_pca3_mesh_warped.png \
  --out_csv stage11_metrics.csv \
  --out_json stage11_summary.json
"""

import argparse, json, csv, math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

try:
    from scipy.spatial import Delaunay
    _HAVE_QHULL = True
except Exception:
    _HAVE_QHULL = False

# ----------------------------
# Synthetic ARC-like generator (same as -report baseline)
# ----------------------------

PRIMS = ["flip_h","flip_v","rotate"]

def moving_average(x, k=9):
    if k <= 1: return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(xp, np.ones(k)/k, mode="valid")

def gaussian_bump(T, center, width, amp=1.0):
    t = np.arange(T)
    sig2 = (width/2.355)**2  # FWHM→σ
    return amp * np.exp(-(t-center)**2 / (2*sig2))

def common_mode(traces: Dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([traces[p] for p in PRIMS], 0).mean(0)

def perpendicular_energy(traces: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    mu = common_mode(traces)
    return {p: np.clip(traces[p] - mu, 0, None) for p in PRIMS}

def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-8)

def _rng(seed: int):
    return np.random.default_rng(seed)

def make_synthetic_traces(rng, T=720, noise=0.02, cm_amp=0.02, overlap=0.5,
                          amp_jitter=0.4, distractor_prob=0.4, tasks_k: Tuple[int,int]=(1,3)) -> Tuple[Dict[str,np.ndarray], List[str]]:
    # choose active primitives
    k = int(rng.integers(tasks_k[0], tasks_k[1]+1))
    tasks = list(rng.choice(PRIMS, size=k, replace=False))
    rng.shuffle(tasks)

    base = np.array([0.20, 0.50, 0.80]) * T
    centers = ((1.0 - overlap) * base + overlap * (T * 0.50)).astype(int)
    width = int(max(12, T * 0.10))
    t = np.arange(T)
    cm = cm_amp * (1.0 + 0.2 * np.sin(2*np.pi * t / max(30, T//6)))

    traces = {p: np.zeros(T, float) for p in PRIMS}
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

    # drift + noise
    for p in PRIMS:
        traces[p] = np.clip(traces[p] + cm, 0, None)
        traces[p] = traces[p] + rng.normal(0, noise, size=T)
        traces[p] = np.clip(traces[p], 0, None)

    return traces, tasks

# ----------------------------
# Feature builder (H/E) — unchanged
# ----------------------------

def build_H_E_from_traces(args) -> Tuple[np.ndarray, np.ndarray]:
    rng = _rng(args.seed)
    H_rows, E_vals = [], []
    for _ in range(args.samples):
        traces, _ = make_synthetic_traces(
            rng, T=args.T, noise=args.noise, cm_amp=args.cm_amp,
            overlap=args.overlap, amp_jitter=args.amp_jitter, distractor_prob=args.distractor_prob,
            tasks_k=(args.min_tasks, args.max_tasks)
        )
        E_perp = perpendicular_energy(traces)
        S = {p: moving_average(E_perp[p], k=args.sigma) for p in PRIMS}
        feats = np.concatenate([_z(S[p]) for p in PRIMS], axis=0)
        H_rows.append(feats)
        E_vals.append(float(sum(np.trapz(S[p]) for p in PRIMS)))
    H = np.vstack(H_rows)
    E = np.asarray(E_vals, float)
    E = (E - E.min()) / (E.ptp() + 1e-9)
    return H, E

# ----------------------------
# Well warp utilities (baseline path)
# ----------------------------

@dataclass
class WellParams:
    whiten: bool = True
    tau: float = 0.25
    isotropize_xy: bool = True
    sigma_scale: float = 0.80
    depth_scale: float = 1.35
    mix_z: float = 0.12
    inhibit_k: int = 10
    inhibit_strength: float = 0.55
    point_alpha: float = 0.85
    trisurf_alpha: float = 0.65

def _softmin_center(X2: np.ndarray, energy: Optional[np.ndarray], tau: float):
    n = len(X2)
    if energy is None:
        w = np.ones(n) / n
    else:
        e = (energy - energy.min()) / (energy.std() + 1e-8)
        w = np.exp(-e / max(tau, 1e-6))
        w = w / (w.sum() + 1e-12)
    c = (w[:, None] * X2).sum(axis=0)
    return c, w

def _isotropize(X2: np.ndarray):
    mu = X2.mean(axis=0)
    Y = X2 - mu
    C = (Y.T @ Y) / max(len(Y)-1, 1)
    evals, evecs = np.linalg.eigh(C)
    T = evecs @ np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-8))) @ evecs.T
    return (Y @ T), (mu, T)

def _gaussian_funnel(r: np.ndarray, sigma: float, depth_scale: float) -> np.ndarray:
    return -depth_scale * np.exp(-(r**2) / (2 * sigma**2))

def _composite_funnel(r: np.ndarray, sigma: float, depth_scale: float,
                      core_strength: float, core_eps: float, core_sigma: float) -> np.ndarray:
    """Composite: bowl (Gaussian) + central needle (1/(r^2+eps) masked near center)."""
    bowl = depth_scale * np.exp(-(r**2) / (2 * sigma**2))
    mask = np.exp(-(r**2) / (2 * core_sigma**2))  # acts near center only
    core = core_strength * mask / (r**2 + core_eps)
    U = bowl + core
    return -U  # return as z (negative potential -> deeper at center)

def pca3_and_warp(H: np.ndarray,
                  energy: Optional[np.ndarray] = None,
                  params: WellParams = WellParams()):
    pca = PCA(n_components=3, whiten=params.whiten, random_state=0)
    X3 = pca.fit_transform(H)
    X2 = X3[:, :2]
    z  = X3[:, 2].copy()

    c, _ = _softmin_center(X2, energy, params.tau)

    if params.isotropize_xy:
        X2_iso, _ = _isotropize(X2 - c)
    else:
        X2_iso = X2 - c

    r = np.linalg.norm(X2_iso, axis=1)
    sigma = np.median(r) * params.sigma_scale + 1e-9
    z_bowl = _gaussian_funnel(r, sigma, params.depth_scale)
    z_new = z_bowl + params.mix_z * (z - z.mean())

    # simple lateral inhibition on z for visual clarity
    if params.inhibit_strength > 0:
        k = min(max(3, params.inhibit_k), len(X2_iso))
        nbrs = NearestNeighbors(n_neighbors=k).fit(X2_iso)
        idx = nbrs.kneighbors(return_distance=False)
        ranks = np.argsort(np.argsort(z_new[idx], axis=1), axis=1)[:,0]  # 0=min
        boost = (ranks > 0).astype(float)
        z_new = z_new + params.inhibit_strength * 0.5 * (boost - 0.5) * (np.std(z_new) + 1e-6)

    out = np.column_stack([X2_iso + c, z_new])
    info = dict(center=c, sigma=sigma)
    return out, info

# ----------------------------
# Sharp-well visualization helpers
# ----------------------------

def sharp_well_transform(X3_raw: np.ndarray, info: Dict,
                         depth_scale: float, sigma_scale: float,
                         core_strength: float, core_eps: float, core_sigma: float,
                         beta: float, sym_bins:int=0) -> Tuple[np.ndarray, Dict[str,float]]:
    """Apply composite radial potential + optional radial symmetry averaging to X3 points.
    Returns new X3 and geometry diagnostics.
    """
    X2 = X3_raw[:, :2]
    z  = X3_raw[:, 2].copy()
    c  = info["center"]
    X2c = X2 - c

    # Isotropic re-check: assume X3_raw already produced by isotropized warp; still safe to compute r
    r = np.linalg.norm(X2c, axis=1)
    sigma = info["sigma"]

    # Composite funnel
    z_core = _composite_funnel(r, sigma, depth_scale, core_strength, core_eps, core_sigma)
    z_new  = z_core + beta * (z - z.mean())

    # Radial symmetry averaging (denoise angular wrinkles)
    if sym_bins and sym_bins > 0:
        theta = np.arctan2(X2c[:,1], X2c[:,0])
        # Bin radii for a smooth radial profile
        r_sorted = np.sort(r)
        rbins = np.linspace(r_sorted[0], r_sorted[-1], max(64, int(np.sqrt(len(r))*2)))
        r_idx = np.digitize(r, rbins)
        # average z_new by radius only (ignore theta)
        rad_avg = {}
        for i in range(1, len(rbins)+1):
            mask = (r_idx == i)
            if np.any(mask):
                rad_avg[i] = float(np.mean(z_new[mask]))
        # map back
        z_sym = z_new.copy()
        for i in range(1, len(rbins)+1):
            if i in rad_avg:
                z_sym[r_idx == i] = rad_avg[i]
        z_new = z_sym

    # Diagnostics
    # tip curvature ~ second derivative near r=0 via small neighborhood fit
    # pick 5% smallest radii
    k = max(10, int(0.05 * len(r)))
    idx_tip = np.argsort(r)[:k]
    rt, zt = r[idx_tip], z_new[idx_tip]
    # fit quadratic z ~ a r^2 + b r + c
    A = np.column_stack([rt**2, rt, np.ones_like(rt)])
    try:
        coef, *_ = np.linalg.lstsq(A, zt, rcond=None)
        tip_curv = float(2*coef[0])  # second derivative wrt r
    except Exception:
        tip_curv = float('nan')

    # rim roundness: variance of z at high-r ring
    j = max(10, int(0.15 * len(r)))
    idx_rim = np.argsort(r)[-j:]
    rim_round = float(np.std(z_new[idx_rim]))

    X3_sharp = np.column_stack([X2, z_new])
    return X3_sharp, dict(tip_curvature=tip_curv, rim_roundness=rim_round)

# ----------------------------
# Plotting
# ----------------------------

def plot_trisurf(X3: np.ndarray, energy: Optional[np.ndarray], title:str, out_path:str, alpha=0.65, point_alpha=0.85):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = X3[:,0], X3[:,1], X3[:,2]
    c = None if energy is None else (energy - np.min(energy)) / (np.ptp(energy) + 1e-9)

    if _HAVE_QHULL and len(X3) >= 4:
        tri = Delaunay(np.column_stack([x, y]))
        ax.plot_trisurf(x, y, z, triangles=tri.simplices, cmap='viridis', alpha=alpha, linewidth=0.2, antialiased=True)
    ax.scatter(x, y, z, c=c, cmap='viridis', s=12, alpha=point_alpha)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title(title)
    if energy is not None:
        mappable = plt.cm.ScalarMappable(cmap='viridis')
        mappable.set_array(c)
        cb = fig.colorbar(mappable, ax=ax)
        cb.set_label("Energy (norm)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

# ----------------------------
# Simple metrics harness (unchanged from -report in spirit)
# ----------------------------

def half_sine_proto(width):
    p = np.sin(np.linspace(0, np.pi, width))
    return p / (np.linalg.norm(p)+1e-8)

def corr_at(sig, proto, idx, width, T):
    a, b = max(0, idx - width//2), min(T, idx + width//2)
    w = sig[a:b]
    if len(w) < 3: return 0.0
    w = w - w.mean()
    pr = proto[:len(w)] - proto[:len(w)].mean()
    denom = (np.linalg.norm(w) * np.linalg.norm(pr) + 1e-8)
    return float(np.dot(w, pr) / denom)

def geodesic_parse(traces, sigma=9, proto_width=160, rng=None, nperm=120, q=0.10, weights=(1.0,0.4,0.3)):
    keys = list(traces.keys())
    T = len(next(iter(traces.values())))
    Eres = perpendicular_energy(traces)
    Sres = {p: moving_average(Eres[p], k=sigma) for p in keys}
    Sraw = {p: moving_average(traces[p], k=sigma) for p in keys}
    Scm  = moving_average(common_mode(traces), k=sigma)
    proto = half_sine_proto(proto_width)
    peak_idx = {p: int(np.argmax(np.correlate(Sres[p], proto, mode="same"))) for p in keys}

    def circ_shift(x, k):
        k = int(k) % len(x)
        if k == 0: return x
        return np.concatenate([x[-k:], x[:-k]])
    def perm_null_z(sig, idx, n=nperm):
        T = len(sig); obs = corr_at(sig, proto, idx, proto_width, T)
        null = np.empty(n, float); rng_local = np.random.default_rng(0)
        for i in range(n):
            shift = rng_local.integers(1, T-1)
            null[i] = corr_at(circ_shift(sig, shift), proto, idx, proto_width, T)
        mu, sd = float(null.mean()), float(null.std() + 1e-8)
        return (obs - mu) / sd

    z_res = {p: perm_null_z(Sres[p], peak_idx[p]) for p in keys}
    z_raw = {p: perm_null_z(Sraw[p], peak_idx[p]) for p in keys}
    z_cm  = {p: perm_null_z(Scm,      peak_idx[keys[0]]) for p in keys}
    w_res, w_raw, w_cm = weights
    score = {p: w_res*z_res[p] + w_raw*z_raw[p] - w_cm*max(0.0, z_cm[p]) for p in keys}

    smax = max(score.values()) + 1e-12
    keep = [p for p in keys if score[p] >= 0.5*smax]
    if not keep:
        keep = [max(keys, key=lambda p: score[p])]
    order = sorted(keep, key=lambda p: peak_idx[p])
    return keep, order

def stock_parse(traces, sigma=9, proto_width=160):
    keys = list(traces.keys())
    S = {p: moving_average(traces[p], k=sigma) for p in keys}
    proto = half_sine_proto(proto_width)
    peak = {p: int(np.argmax(np.correlate(S[p], proto, mode="same"))) for p in keys}
    score = {p: float(np.max(np.correlate(S[p], proto, mode="same"))) for p in keys}
    smax = max(score.values()) + 1e-12
    keep = [p for p in keys if score[p] >= 0.6*smax]
    if not keep:
        keep = [max(keys, key=lambda p: score[p])]
    order = sorted(keep, key=lambda p: peak[p])
    return keep, order

def set_metrics(true_list: List[str], pred_list: List[str]) -> Dict[str,float]:
    Tset, Pset = set(true_list), set(pred_list)
    tp, fp, fn = len(Tset & Pset), len(Pset - Tset), len(Tset - Pset)
    precision = tp / max(1, len(Pset))
    recall    = tp / max(1, len(Tset))
    f1        = 0.0 if precision+recall==0 else (2*precision*recall)/(precision+recall)
    jaccard   = tp / max(1, len(Tset | Pset))
    return dict(precision=precision, recall=recall, f1=f1, jaccard=jaccard,
                hallucination_rate=fp/max(1,len(Pset)), omission_rate=fn/max(1,len(Tset)))

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage 11 — report baseline with sharp-well visualization (v8)")
    # data + features
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--sigma", type=int, default=9)
    ap.add_argument("--cm_amp", type=float, default=0.02)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--amp_jitter", type=float, default=0.4)
    ap.add_argument("--distractor_prob", type=float, default=0.4)
    ap.add_argument("--min_tasks", type=int, default=1)
    ap.add_argument("--max_tasks", type=int, default=3)
    # plotting
    ap.add_argument("--out_plot", type=str, default="manifold_pca3_mesh_warped.png")
    ap.add_argument("--out_csv", type=str, default="stage11_metrics.csv")
    ap.add_argument("--out_json", type=str, default="stage11_summary.json")
    # well model
    ap.add_argument("--well_model", type=str, choices=["base","sharp"], default="base")
    ap.add_argument("--sigma_scale", type=float, default=0.80)
    ap.add_argument("--depth_scale", type=float, default=1.35)
    ap.add_argument("--mix_z", type=float, default=0.12)
    # sharp mode knobs
    ap.add_argument("--core_strength", type=float, default=0.9)
    ap.add_argument("--core_eps", type=float, default=0.02)
    ap.add_argument("--beta", type=float, default=0.08)
    ap.add_argument("--sym_bins", type=int, default=64)
    ap.add_argument("--core_sigma_factor", type=float, default=0.6, help="core_sigma = factor * sigma")
    args = ap.parse_args()

    # ----- per-sample generation + parsing (metrics) -----
    rng = _rng(args.seed)
    rows = []
    agg_geo = dict(acc=0, P=0, R=0, F1=0, J=0, H=0, O=0)
    agg_stock = dict(acc=0, P=0, R=0, F1=0, J=0, H=0, O=0)

    for i in range(1, args.samples+1):
        traces, true_order = make_synthetic_traces(
            rng, T=args.T, noise=args.noise, cm_amp=args.cm_amp, overlap=args.overlap,
            amp_jitter=args.amp_jitter, distractor_prob=args.distractor_prob,
            tasks_k=(args.min_tasks, args.max_tasks)
        )
        keep_g, order_g = geodesic_parse(traces, sigma=args.sigma)
        keep_s, order_s = stock_parse(traces, sigma=args.sigma)
        acc_g = int(order_g == true_order)
        acc_s = int(order_s == true_order)
        sm_g = set_metrics(true_order, keep_g)
        sm_s = set_metrics(true_order, keep_s)
        for k, v in sm_g.items(): agg_geo[{"precision":"P","recall":"R","f1":"F1","jaccard":"J","hallucination_rate":"H","omission_rate":"O"}[k]] = agg_geo.get({"precision":"P","recall":"R","f1":"F1","jaccard":"J","hallucination_rate":"H","omission_rate":"O"}[k],0)+v
        for k, v in sm_s.items(): agg_stock[{"precision":"P","recall":"R","f1":"F1","jaccard":"J","hallucination_rate":"H","omission_rate":"O"}[k]] = agg_stock.get({"precision":"P","recall":"R","f1":"F1","jaccard":"J","hallucination_rate":"H","omission_rate":"O"}[k],0)+v
        agg_geo["acc"] += acc_g; agg_stock["acc"] += acc_s
        rows.append(dict(
            sample=i,
            true="|".join(true_order),
            geodesic_tasks="|".join(keep_g), geodesic_order="|".join(order_g), geodesic_ok=acc_g,
            stock_tasks="|".join(keep_s), stock_order="|".join(order_s), stock_ok=acc_s,
            geodesic_precision=sm_g["precision"], geodesic_recall=sm_g["recall"], geodesic_f1=sm_g["f1"],
            geodesic_jaccard=sm_g["jaccard"], geodesic_hallucination=sm_g["hallucination_rate"], geodesic_omission=sm_g["omission_rate"],
            stock_precision=sm_s["precision"], stock_recall=sm_s["recall"], stock_f1=sm_s["f1"],
            stock_jaccard=sm_s["jaccard"], stock_hallucination=sm_s["hallucination_rate"], stock_omission=sm_s["omission_rate"],
        ))

    n = float(args.samples)
    Sg = dict(
        accuracy_exact = agg_geo["acc"]/n, precision=agg_geo["P"]/n, recall=agg_geo["R"]/n, f1=agg_geo["F1"]/n,
        jaccard=agg_geo["J"]/n, hallucination_rate=agg_geo["H"]/n, omission_rate=agg_geo["O"]/n
    )
    Ss = dict(
        accuracy_exact = agg_stock["acc"]/n, precision=agg_stock["P"]/n, recall=agg_stock["R"]/n, f1=agg_stock["F1"]/n,
        jaccard=agg_stock["J"]/n, hallucination_rate=agg_stock["H"]/n, omission_rate=agg_stock["O"]/n
    )

    # ----- Build H/E once (for visualization) -----
    H, E = build_H_E_from_traces(args)
    # baseline warp (RAW)
    base_params = WellParams(sigma_scale=args.sigma_scale, depth_scale=args.depth_scale, mix_z=args.mix_z)
    X3_raw, info = pca3_and_warp(H, energy=E, params=base_params)

    # Save baseline (RAW) plot
    plot_trisurf(X3_raw, E, title="Stage 11 — Warped Single Well (Raw)", out_path=args.out_plot)

    extra = {}
    if args.well_model == "sharp":
        core_sigma = float(args.core_sigma_factor) * float(info["sigma"])
        X3_sharp, geom = sharp_well_transform(
            X3_raw, info,
            depth_scale=args.depth_scale, sigma_scale=args.sigma_scale,
            core_strength=args.core_strength, core_eps=args.core_eps, core_sigma=core_sigma,
            beta=args.beta, sym_bins=args.sym_bins
        )
        extra.update(geom)

        # also emit a comparison image (side-by-side)
        out_compare = args.out_plot.replace(".png", "_v8_compare.png")
        # render both and stitch visually by saving two figs
        plot_trisurf(X3_sharp, E, title="Stage 11 — Warped Single Well (Sharp)", out_path=out_compare)

    # ----- Write CSV & JSON (baseline compatible) -----
    if rows:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows: w.writerow(r)

    summary = dict(
        samples=int(n),
        geodesic=Sg,
        stock=Ss,
        center=[float(c) for c in info["center"]],
        sigma=float(info["sigma"]),
        plot=args.out_plot,
        csv=args.out_csv,
        well_model=args.well_model,
        sharp_params=dict(
            sigma_scale=args.sigma_scale,
            depth_scale=args.depth_scale,
            core_strength=args.core_strength,
            core_eps=args.core_eps,
            beta=args.beta,
            sym_bins=args.sym_bins,
            core_sigma_factor=args.core_sigma_factor
        )
    )
    summary.update(extra)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("[SUMMARY] Geodesic:", {k: round(v,3) for k,v in Sg.items()})
    print("[SUMMARY] Stock   :", {k: round(v,3) for k,v in Ss.items()})
    print(f"[PLOT] RAW: {args.out_plot}")
    if args.well_model == "sharp":
        print(f"[PLOT] SHARP compare: {args.out_plot.replace('.png','_v8_compare.png')}")
    print(f"[CSV ] {args.out_csv}")
    print(f"[JSON] {args.out_json}")

if __name__ == "__main__":
    main()
