#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 11 — Well Benchmark v8b (recall-friendly tuning)
------------------------------------------------------
Changes vs v8:
  • Softer residual refinement (drop_frac ↓)
  • Slower annealing (higher T0, gentler anneal)
  • Looser FDR gate (alpha ↑), more perms, smaller blocks
  • Adaptive gating: if we're under target picks, relax FDR on the fly
  • Candidate floor: always allow top-2 by prob to compete

Outputs:
  - manifold_pca3_mesh_warped_v8b.png
  - stage11_v8b_metrics.csv
  - stage11_v8b_summary.json

python3 stage11-well-benchmark-v8b.py \
  --samples 200 --seed 42 --T 720 --sigma 9 \
  --perm 180 --block 12 --alpha 0.12 \
  --max_iter 3 --drop_frac 0.22 \
  --T0 2.0 --Tmin 0.7 --anneal 0.92 \
  --out_plot manifold_pca3_mesh_warped_v8b.png \
  --out_csv stage11_v8b_metrics.csv \
  --out_json stage11_v8b_summary.json

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
# Synthetic ARC-like generator
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
# Feature builder (H/E) for manifold plot
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
# Single-well warp utilities
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

def _radial_funnel(X2_iso: np.ndarray, z: np.ndarray, sigma: float, depth_scale: float, mix_z: float):
    r = np.linalg.norm(X2_iso, axis=1) + 1e-9
    u = X2_iso / r[:, None]
    z_funnel = -np.exp(-(r**2) / (2 * sigma**2))  # [-1,0]
    z_new = depth_scale * z_funnel + mix_z * (z - z.mean())
    X2_new = (r[:, None] * u)
    return X2_new, z_new

def _phantom_metrics(X2: np.ndarray, z: np.ndarray) -> Dict[str, float]:
    nb = max(12, int(np.sqrt(len(X2)) / 2))
    xi = np.digitize(X2[:,0], np.linspace(X2[:,0].min(), X2[:,0].max(), nb))
    yi = np.digitize(X2[:,1], np.linspace(X2[:,1].min(), X2[:,1].max(), nb))
    grid_min = {}
    for i in range(len(X2)):
        key = (xi[i], yi[i])
        grid_min[key] = min(grid_min.get(key, np.inf), z[i])
    mins = sorted(grid_min.values())
    if len(mins) < 2:
        return {"phantom_index": 0.0, "margin": 0.0}
    z0, z1 = mins[0], mins[1]
    span = np.percentile(z, 95) - np.percentile(z, 5) + 1e-9
    phantom_index = (z1 - z0) / span
    margin = z1 - z0
    return {"phantom_index": float(phantom_index), "margin": float(margin)}

def _lateral_inhibition(z: np.ndarray, X2: np.ndarray, k:int, strength: float) -> np.ndarray:
    k = min(max(3, k), len(X2))
    nbrs = NearestNeighbors(n_neighbors=k).fit(X2)
    idx = nbrs.kneighbors(return_distance=False)
    ranks = np.argsort(np.argsort(z[idx], axis=1), axis=1)[:,0]  # 0=min
    boost = (ranks > 0).astype(float)
    z_adj = z + strength * 0.5 * (boost - 0.5) * (np.std(z) + 1e-6)
    return z_adj

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
    X2_new, z_new = _radial_funnel(X2_iso, z, sigma, params.depth_scale, params.mix_z)
    z_new = _lateral_inhibition(z_new, X2_new, k=params.inhibit_k, strength=params.inhibit_strength)

    metrics = _phantom_metrics(X2_new, z_new)
    out = np.column_stack([X2_new + c, z_new])
    return out, metrics, dict(center=c, sigma=sigma)

def plot_trisurf(X3: np.ndarray, energy: Optional[np.ndarray] = None, params: WellParams = WellParams(), title:str="Warped manifold (single well)"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = X3[:,0], X3[:,1], X3[:,2]
    c = None if energy is None else (energy - np.min(energy)) / (np.ptp(energy) + 1e-9)

    if _HAVE_QHULL and len(X3) >= 4:
        tri = Delaunay(np.column_stack([x, y]))
        ax.plot_trisurf(x, y, z, triangles=tri.simplices, cmap='viridis', alpha=params.trisurf_alpha, linewidth=0.2, antialiased=True)
    ax.scatter(x, y, z, c=c, cmap='viridis', s=12, alpha=params.point_alpha)

    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title(title)
    if energy is not None:
        mappable = plt.cm.ScalarMappable(cmap='viridis')
        mappable.set_array(c)
        cb = fig.colorbar(mappable, ax=ax)
        cb.set_label("Energy (norm)")
    plt.tight_layout()
    return fig, ax

# ----------------------------
# Calibrated nulls (block perms + optional multitaper)
# ----------------------------

def half_sine_proto(width):
    p = np.sin(np.linspace(0, np.pi, width))
    return p / (np.linalg.norm(p)+1e-8)

def circ_shift_blocks(x: np.ndarray, block: int, rng) -> np.ndarray:
    T = len(x)
    if block <= 1:
        k = int(rng.integers(1, T-1))
        return np.roll(x, k)
    step = int(block * int(rng.integers(1, max(2, T//block))))
    jitter = int(rng.integers(0, max(1, block//4)))
    k = (step + jitter) % T
    return np.roll(x, k)

def corr_at(sig, proto, idx, width, T):
    a, b = max(0, idx - width//2), min(T, idx + width//2)
    w = sig[a:b]
    if len(w) < 3: return 0.0
    w = w - w.mean()
    pr = proto[:len(w)] - proto[:len(w)].mean()
    denom = (np.linalg.norm(w) * np.linalg.norm(pr) + 1e-8)
    return float(np.dot(w, pr) / denom)

def z_from_null(obs, null):
    mu, sd = float(null.mean()), float(null.std() + 1e-8)
    return (obs - mu) / sd

def perm_null_z_block(sig, proto, peak_idx, width, rng, nperm=200, block=12):
    T = len(sig)
    obs = corr_at(sig, proto, peak_idx, width, T)
    null = np.empty(nperm, float)
    for i in range(nperm):
        x = circ_shift_blocks(sig, block, rng)
        null[i] = corr_at(x, proto, peak_idx, width, T)
    return z_from_null(obs, null)

def z_to_p(z):
    return 2.0 * (1.0 - 0.5 * (1 + math.erf(abs(z)/math.sqrt(2))))

def bh_fdr(pvals, q=0.10):
    m = len(pvals)
    if m == 0: return np.array([], dtype=bool)
    order = np.argsort(pvals)
    thresh = q * (np.arange(1, m+1) / m)
    passed = np.zeros(m, dtype=bool)
    pv_sorted = np.array(pvals)[order]
    max_i = np.where(pv_sorted <= thresh)[0]
    if len(max_i) > 0:
        cutoff = max_i.max()
        passed[order[:cutoff+1]] = True
    return passed

# ----------------------------
# v8b Geodesic parser (with adaptive gating)
# ----------------------------

def geodesic_parse_v8b(traces, *, sigma=9, proto_width=160, rng=None, nperm=180, q=0.12,
                       weights=(1.0,0.4,0.3), block=12,
                       max_iter=3, drop_frac=0.22, T0=2.0, Tmin=0.7, anneal=0.92,
                       expected_k: Optional[int]=None, candidate_floor:int=2):
    keys = list(traces.keys())
    T = len(next(iter(traces.values())))
    rng = np.random.default_rng() if rng is None else rng
    proto = half_sine_proto(proto_width)

    Eres0 = perpendicular_energy(traces)
    Sres = {p: moving_average(Eres0[p], k=sigma) for p in keys}
    Sraw = {p: moving_average(traces[p], k=sigma) for p in keys}
    Scm  = moving_average(common_mode(traces), k=sigma)

    keep, order = [], []
    temp = float(T0)

    def peak_idx_of(S):
        return {p: int(np.argmax(np.correlate(S[p], proto, mode="same"))) for p in keys}

    for it in range(max_iter):
        peak_idx = peak_idx_of(Sres)

        # calibrated z-scores
        z_res, z_raw, z_cm = {}, {}, {}
        for p in keys:
            idx = peak_idx[p]
            z_res[p] = perm_null_z_block(Sres[p], proto, idx, proto_width, rng, nperm=nperm, block=block)
            z_raw[p] = perm_null_z_block(Sraw[p], proto, idx, proto_width, rng, nperm=nperm, block=block)
            z_cm[p]  = perm_null_z_block(Scm,     proto, idx, proto_width, rng, nperm=nperm, block=block)

        w_res, w_raw, w_cm = weights
        score = {p: w_res*z_res[p] + w_raw*z_raw[p] - w_cm*max(0.0, z_cm[p]) for p in keys}
        U = {p: -(score[p]) for p in keys}

        # annealed probs
        logits = np.array([-U[p] / max(temp, 1e-3) for p in keys])
        logits = logits - logits.max()
        probs = np.exp(logits); probs = probs / (probs.sum() + 1e-12)

        # adaptive FDR: if we're under target picks, relax q_eff
        q_eff = q
        if expected_k is not None:
            deficit = max(0, expected_k - len(keep))
            q_eff = min(0.25, q * (1 + 0.5 * deficit))

        pvals = [z_to_p(max(0, z_res[p]) + 0.5*max(0, z_raw[p]) - 0.3*max(0, z_cm[p])) for p in keys]
        passed = bh_fdr(pvals, q=q_eff)
        cand = [k for k, ok in zip(keys, passed) if ok and score[k] > 0]

        # candidate floor: always allow top-k by prob into the pool
        top_idx = np.argsort(-probs)[:candidate_floor]
        for j in top_idx:
            k = keys[j]
            if k not in cand and score[k] > 0:
                cand.append(k)

        if not cand:
            break

        # pick best by current probability among candidates
        k_best = max(cand, key=lambda p: probs[keys.index(p)])
        keep.append(k_best); order.append(k_best)

        # residual refinement (softer) & decay drop_frac over iterations
        win_curve = Sres[k_best].copy()
        for p in keys:
            Sres[p] = np.clip(Sres[p] - drop_frac * win_curve, 0.0, None)
        drop_frac *= 0.9  # decay denoising strength

        # anneal
        temp = max(Tmin, temp * anneal)

    if not keep:
        peak_idx = peak_idx_of(Sres)
        z_res = {p: perm_null_z_block(Sres[p], proto, peak_idx[p], proto_width, rng, nperm=nperm, block=block) for p in keys}
        best = max(keys, key=lambda p: z_res[p])
        keep, order = [best], [best]
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
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage 11 v8b — recall-friendly tuning (adaptive gating)")
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
    # calibrated nulls
    ap.add_argument("--perm", type=int, default=180)
    ap.add_argument("--block", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=0.12)
    ap.add_argument("--proto_width", type=int, default=160)
    # refinement + anneal
    ap.add_argument("--max_iter", type=int, default=3)
    ap.add_argument("--drop_frac", type=float, default=0.22)
    ap.add_argument("--T0", type=float, default=2.0)
    ap.add_argument("--Tmin", type=float, default=0.7)
    ap.add_argument("--anneal", type=float, default=0.92)
    # outputs
    ap.add_argument("--out_plot", type=str, default="manifold_pca3_mesh_warped_v8b.png")
    ap.add_argument("--out_csv", type=str, default="stage11_v8b_metrics.csv")
    ap.add_argument("--out_json", type=str, default="stage11_v8b_summary.json")
    args = ap.parse_args()

    rng = _rng(args.seed)

    # ----- per-sample generation + parsing (for metrics) -----
    rows = []
    agg_geo = dict(acc=0, P=0, R=0, F1=0, J=0, H=0, O=0)
    agg_stock = dict(acc=0, P=0, R=0, F1=0, J=0, H=0, O=0)

    for i in range(1, args.samples+1):
        traces, true_order = make_synthetic_traces(
            rng, T=args.T, noise=args.noise, cm_amp=args.cm_amp, overlap=args.overlap,
            amp_jitter=args.amp_jitter, distractor_prob=args.distractor_prob,
            tasks_k=(args.min_tasks, args.max_tasks)
        )

        keep_g, order_g = geodesic_parse_v8b(
            traces, sigma=args.sigma, proto_width=args.proto_width, rng=rng,
            nperm=args.perm, q=args.alpha, block=args.block,
            max_iter=args.max_iter, drop_frac=args.drop_frac, T0=args.T0, Tmin=args.Tmin, anneal=args.anneal,
            expected_k=len(true_order), candidate_floor=2
        )
        keep_s, order_s = stock_parse(traces, sigma=args.sigma, proto_width=args.proto_width)

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

    # ----- build H/E once more (for manifold view) -----
    H, E = build_H_E_from_traces(args)
    X3_warp, m, info = pca3_and_warp(H, energy=E, params=WellParams())
    fig, ax = plot_trisurf(X3_warp, energy=E, title="Stage 11 — Warped Single Well (v8b)")
    fig.savefig(args.out_plot, dpi=220)

    # ----- write CSV & JSON -----
    if rows:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows: w.writerow(r)

    summary = dict(
        samples=int(n),
        geodesic=Sg,
        stock=Ss,
        phantom_index=float(m["phantom_index"]),
        margin=float(m["margin"]),
        center=[float(c) for c in info["center"]],
        sigma=float(info["sigma"]),
        plot=args.out_plot,
        csv=args.out_csv,
        params=dict(
            perm=args.perm, block=args.block, sigma=args.sigma,
            drop_frac=args.drop_frac, T0=args.T0, Tmin=args.Tmin, anneal=args.anneal,
            max_iter=args.max_iter, alpha=args.alpha
        )
    )
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("[SUMMARY] Geodesic:", {k: round(v,3) for k,v in Sg.items()})
    print("[SUMMARY] Stock   :", {k: round(v,3) for k,v in Ss.items()})
    print("[WELL] phantom_index:", round(m["phantom_index"], 4), "margin:", round(m["margin"], 4))
    print(f"[PLOT] {args.out_plot}")
    print(f"[CSV ] {args.out_csv}")
    print(f"[JSON] {args.out_json}")

if __name__ == "__main__":
    main()
