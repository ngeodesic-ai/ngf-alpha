#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 11 — Well Benchmark v9
----------------------------
Fork of the -report baseline that adds a *geometry-focused* visualization mode
which actually sharpens the funnel using three doctrine steps:

  1) Residual refinement loop (drainage) on the warped z field
  2) Lateral inhibition kernel (suppresses overlapping/phantom basins)
  3) Radial symmetry averaging for the final render (axisymmetric logo look)

⚠ Metrics/CSV/JSON remain the same as the -report baseline (parsing unchanged).
The sharpening acts only in the *visualization path* so we can A/B safely.

Outputs:
  - manifold_pca3_mesh_warped.png                 (RAW baseline)
  - manifold_pca3_mesh_warped_v9_sharp.png        (v9 sharpened viz)
  - stage11_metrics.csv
  - stage11_summary.json (augmented with viz diagnostics)

python3 stage11-well-benchmark-v9.py \
  --samples 200 --seed 42 --T 720 --sigma 9 \
  --iters 3 --drop_frac 0.28 \
  --inhibit_k 10 --inhibit_strength 0.55 \
  --sym_bins 64 \
  --out_plot manifold_pca3_mesh_warped.png \
  --out_csv stage11_metrics.csv \
  --out_json stage11_summary.json

python3 stage11-well-benchmark-v9.py \
  --samples 200 --seed 42 --T 720 --sigma 9 \
  --iters 3 --drop_frac 0.28 \
  --inhibit_k 10 --inhibit_strength 0.55 \
  --sym_bins 64 \
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
# Synthetic generator (baseline)
# ----------------------------

PRIMS = ["flip_h","flip_v","rotate"]

def moving_average(x, k=9):
    if k <= 1: return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(xp, np.ones(k)/k, mode="valid")

def gaussian_bump(T, center, width, amp=1.0):
    t = np.arange(T)
    sig2 = (width/2.355)**2
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

    for p in PRIMS:
        if p not in tasks and rng.random() < distractor_prob:
            c = int(rng.uniform(T*0.15, T*0.85))
            amp = max(0.25, 1.0 + rng.normal(0, amp_jitter))
            traces[p] += gaussian_bump(T, c, width, amp=0.9*amp)

    for p in PRIMS:
        traces[p] = np.clip(traces[p] + cm, 0, None)
        traces[p] = traces[p] + rng.normal(0, noise, size=T)
        traces[p] = np.clip(traces[p], 0, None)

    return traces, tasks

# ----------------------------
# Features (H/E) — baseline
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
# PCA warp (baseline)
# ----------------------------

from dataclasses import dataclass

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

def pca3_and_warp(H: np.ndarray, energy: Optional[np.ndarray], params: WellParams):
    pca = PCA(n_components=3, whiten=params.whiten, random_state=0)
    X3 = pca.fit_transform(H)
    X2 = X3[:, :2]
    z  = X3[:, 2].copy()

    c, _ = _softmin_center(X2, energy, params.tau)
    X2c = X2 - c
    if params.isotropize_xy:
        X2c, _ = _isotropize(X2c)

    # Gaussian bowl + small PC3 blend
    r = np.linalg.norm(X2c, axis=1)
    sigma = np.median(r) * params.sigma_scale + 1e-9
    z_bowl = -params.depth_scale * np.exp(-(r**2)/(2*sigma**2))
    z_new  = z_bowl + params.mix_z * (z - z.mean())

    # light inhibition (visual only)
    if params.inhibit_strength > 0:
        k = min(max(3, params.inhibit_k), len(X2c))
        nbrs = NearestNeighbors(n_neighbors=k).fit(X2c)
        idx = nbrs.kneighbors(return_distance=False)
        ranks = np.argsort(np.argsort(z_new[idx], axis=1), axis=1)[:,0]
        boost = (ranks > 0).astype(float)
        z_new = z_new + params.inhibit_strength * 0.5 * (boost - 0.5) * (np.std(z_new) + 1e-6)

    X3_out = np.column_stack([X2c + c, z_new])
    info = dict(center=c, sigma=sigma)
    return X3_out, info

# ----------------------------
# v9 Sharpening (viz path)
# ----------------------------

def _residual_refinement(X2c: np.ndarray, z_in: np.ndarray, *, iters:int, drop:float) -> np.ndarray:
    """Drain energy iteratively using neighborhood-aligned subtraction of the current minimum curve."""
    z = z_in.copy()
    k = min(10, max(4, int(np.sqrt(len(X2c))/3)))
    nbrs = NearestNeighbors(n_neighbors=k).fit(X2c)
    for t in range(iters):
        # find current global minimum (well center proxy)
        i0 = int(np.argmin(z))
        # construct a local template as the average of neighbors of i0
        idx = nbrs.kneighbors(X2c[[i0]], return_distance=False)[0]
        template = z[idx].mean()
        # subtract a fraction from all points proportional to proximity to center
        d = np.linalg.norm(X2c - X2c[i0], axis=1) + 1e-9
        w = np.exp(-(d**2)/(2*(np.median(d)*0.7 + 1e-9)**2))
        z = z - drop * w * (template - z.min())
        drop *= 0.9
    return z

def _lateral_inhibition_kernel(X2c: np.ndarray, z: np.ndarray, *, k:int, strength:float) -> np.ndarray:
    """Suppress local non-minima by pushing them up toward neighborhood mean."""
    k = min(max(3, k), len(X2c))
    nbrs = NearestNeighbors(n_neighbors=k).fit(X2c)
    idx = nbrs.kneighbors(return_distance=False)
    neigh_mean = z[idx].mean(axis=1)
    # pull each point toward neighborhood min; push non-minima up slightly
    ranks = np.argsort(np.argsort(z[idx], axis=1), axis=1)[:,0]
    nonmin = (ranks > 0).astype(float)
    z_out = z + strength * nonmin * (neigh_mean - z)
    return z_out

def _radial_symmetry_average(X2c: np.ndarray, z: np.ndarray, *, bins:int=64) -> np.ndarray:
    """Average z by radius only (axisymmetry), ignoring theta."""
    r = np.linalg.norm(X2c, axis=1)
    r_sorted = np.sort(r)
    rbins = np.linspace(r_sorted[0], r_sorted[-1], max(64, int(np.sqrt(len(r))*2)))
    r_idx = np.digitize(r, rbins)
    z_new = z.copy()
    for i in range(1, len(rbins)+1):
        mask = (r_idx == i)
        if np.any(mask):
            z_new[mask] = z[mask].mean()
    return z_new

def sharpen_funnel_v9(X3_raw: np.ndarray, info: Dict, *, iters:int=3, drop:float=0.28,
                      inhibit_k:int=10, inhibit_strength:float=0.55, sym_bins:int=64) -> Tuple[np.ndarray, Dict[str,float]]:
    X2 = X3_raw[:, :2]; z = X3_raw[:, 2].copy()
    c = info["center"]; X2c = X2 - c

    # 1) residual refinement
    z1 = _residual_refinement(X2c, z, iters=iters, drop=drop)

    # 2) lateral inhibition
    z2 = _lateral_inhibition_kernel(X2c, z1, k=inhibit_k, strength=inhibit_strength)

    # 3) radial symmetry averaging
    z3 = _radial_symmetry_average(X2c, z2, bins=sym_bins)

    # diagnostics
    r = np.linalg.norm(X2c, axis=1)
    k_tip = max(10, int(0.05 * len(r)))
    idx_tip = np.argsort(r)[:k_tip]
    A = np.column_stack([r[idx_tip]**2, r[idx_tip], np.ones_like(idx_tip)])
    try:
        coef, *_ = np.linalg.lstsq(A, z3[idx_tip], rcond=None)
        tip_curv = float(2*coef[0])
    except Exception:
        tip_curv = float('nan')
    rim_var = float(np.var(z3[np.argsort(r)[-max(10, int(0.15*len(r))):]]))

    X3_sharp = np.column_stack([X2, z3])
    return X3_sharp, dict(tip_curvature=tip_curv, rim_roundness=np.sqrt(rim_var))

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
# Metrics harness (baseline-like)
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

def geodesic_parse(traces, sigma=9, proto_width=160):
    keys = list(traces.keys())
    Eres = perpendicular_energy(traces)
    Sres = {p: moving_average(Eres[p], k=sigma) for p in keys}
    proto = half_sine_proto(proto_width)
    peak = {p: int(np.argmax(np.correlate(Sres[p], proto, mode="same"))) for p in keys}
    score = {p: float(np.max(np.correlate(Sres[p], proto, mode="same"))) for p in keys}
    smax = max(score.values()) + 1e-12
    keep = [p for p in keys if score[p] >= 0.5*smax]
    if not keep: keep = [max(keys, key=lambda p: score[p])]
    order = sorted(keep, key=lambda p: peak[p])
    return keep, order

def stock_parse(traces, sigma=9, proto_width=160):
    keys = list(traces.keys())
    S = {p: moving_average(traces[p], k=sigma) for p in keys}
    proto = half_sine_proto(proto_width)
    peak = {p: int(np.argmax(np.correlate(S[p], proto, mode="same"))) for p in keys}
    score = {p: float(np.max(np.correlate(S[p], proto, mode="same"))) for p in keys}
    smax = max(score.values()) + 1e-12
    keep = [p for p in keys if score[p] >= 0.6*smax]
    if not keep: keep = [max(keys, key=lambda p: score[p])]
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
    ap = argparse.ArgumentParser(description="Stage 11 — report baseline with v9 sharp-well viz (refine+inhibit+radial)")
    # data
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
    # outputs
    ap.add_argument("--out_plot", type=str, default="manifold_pca3_mesh_warped.png")
    ap.add_argument("--out_csv", type=str, default="stage11_metrics.csv")
    ap.add_argument("--out_json", type=str, default="stage11_summary.json")
    # viz params
    ap.add_argument("--sigma_scale", type=float, default=0.80)
    ap.add_argument("--depth_scale", type=float, default=1.35)
    ap.add_argument("--mix_z", type=float, default=0.12)
    ap.add_argument("--iters", type=int, default=3, help="residual refinement iterations")
    ap.add_argument("--drop_frac", type=float, default=0.28, help="drainage per iteration")
    ap.add_argument("--inhibit_k", type=int, default=10)
    ap.add_argument("--inhibit_strength", type=float, default=0.55)
    ap.add_argument("--sym_bins", type=int, default=64)
    args = ap.parse_args()

    # ----- metrics run (baseline) -----
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
        acc_g = int(order_g == true_order); acc_s = int(order_s == true_order)
        sm_g = set_metrics(true_order, keep_g); sm_s = set_metrics(true_order, keep_s)
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

    # ----- Visualization build -----
    H, E = build_H_E_from_traces(args)
    base_params = WellParams(sigma_scale=args.sigma_scale, depth_scale=args.depth_scale, mix_z=args.mix_z,
                             inhibit_k=args.inhibit_k, inhibit_strength=0.0)  # turn off here; we do a stronger one below
    X3_raw, info = pca3_and_warp(H, energy=E, params=base_params)
    plot_trisurf(X3_raw, E, "Stage 11 — Warped Single Well (Raw)", args.out_plot)

    # v9 sharpened viz
    X3_sharp, geom = sharpen_funnel_v9(
        X3_raw, info,
        iters=args.iters, drop=args.drop_frac,
        inhibit_k=args.inhibit_k, inhibit_strength=args.inhibit_strength,
        sym_bins=args.sym_bins
    )
    out_sharp = args.out_plot.replace(".png", "_v9_sharp.png")
    plot_trisurf(X3_sharp, E, "Stage 11 — Warped Single Well (v9: refined + inhibited + radial)", out_sharp)

    # ----- Outputs -----
    if rows:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows: w.writerow(r)

    summary = dict(
        samples=int(n),
        geodesic=Sg, stock=Ss,
        center=[float(c) for c in info["center"]],
        sigma=float(info["sigma"]),
        plot_raw=args.out_plot,
        plot_sharp=out_sharp,
        viz_params=dict(iters=args.iters, drop_frac=args.drop_frac, inhibit_k=args.inhibit_k,
                        inhibit_strength=args.inhibit_strength, sym_bins=args.sym_bins),
        geometry=geom
    )
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("[SUMMARY] Geodesic:", {k: round(v,3) for k,v in Sg.items()})
    print("[SUMMARY] Stock   :", {k: round(v,3) for k,v in Ss.items()})
    print(f"[PLOT] RAW:   {args.out_plot}")
    print(f"[PLOT] SHARP: {out_sharp}")
    print(f"[CSV ] {args.out_csv}")
    print(f"[JSON] {args.out_json}")

if __name__ == "__main__":
    main()
