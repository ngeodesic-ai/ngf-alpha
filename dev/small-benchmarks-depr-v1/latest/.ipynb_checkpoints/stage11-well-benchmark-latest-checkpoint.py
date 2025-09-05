# Create an updated baseline script that keeps report metrics by default,
# but adds the fitted funnel overlay (and an optional prior toggle).
# script = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 11 — Well Benchmark + Report (funnel-updated)
---------------------------------------------------
Baseline 'report' pipeline with two additions:
  1) A *data-fit 360° funnel* is computed from the PCA(3) cloud and saved as an
     overlay figure for visualization (same shape we used in v9e).
  2) An optional *funnel prior* can be toggled on for parsing (--use_funnel_prior 1).
     By default it is OFF to preserve the original report metrics.

Outputs
  - manifold_pca3_mesh_warped.png        (raw trisurf, as before)
  - manifold_pca3_mesh_warped_fit.png    (funnel overlay using fitted profile)
  - stage11_metrics.csv
  - stage11_summary.json

Example:
python3 stage11-well-benchmark-latest-funnel.py \
  --samples 200 --seed 42 --T 720 --sigma 9 \
  --out_plot manifold_pca3_mesh_warped.png \
  --out_csv stage11_metrics.csv \
  --out_json stage11_summary.json \
  --use_funnel_prior 0   # keep the original report behavior

python3 stage11-well-benchmark-latest.py \
  --samples 200 --seed 42 --T 720 --sigma 9 \
  --out_plot manifold_pca3_mesh_warped.png \
  --out_csv stage11_metrics.csv \
  --out_json stage11_summary.json \
  --use_funnel_prior 1 --alpha 0.05 --beta_s 0.25 --q_s 2 \
  --tau_rel 0.60 --tau_abs_q 0.93 --null_K 40
"""

import argparse, json, csv
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

PRIMS = ["flip_h","flip_v","rotate"]

# ----------------------------
# Utils
# ----------------------------

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

def zscore_robust(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return (x - med) / (1.4826 * mad)

def _rng(seed: int):
    return np.random.default_rng(seed)

# ----------------------------
# Synthetic ARC-like generator
# ----------------------------

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
# Feature builder (H/E)
# ----------------------------

def build_H_E_from_traces(args) -> Tuple[np.ndarray, np.ndarray]:
    """Return feature matrix H and scalar energy E per sample (for PCA viz)."""
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
    E = (E - E.min()) / (E.ptp() + 1e-9)  # normalize for coloring
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
    ranks = np.argsort(np.argsort(z[idx], axis=1), axis=1)[:,0]
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
# Fitted funnel profile (for viz and optional priors)
# ----------------------------

def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    idx = np.argsort(values)
    v, w = values[idx], weights[idx]
    cum = np.cumsum(w)
    if cum[-1] <= 0:
        return float(np.median(v))
    t = q * cum[-1]
    j = int(np.searchsorted(cum, t, side="left"))
    j = min(max(j, 0), len(v)-1)
    return float(v[j])

def fit_radial_profile(X3: np.ndarray, center: np.ndarray, r_grid: np.ndarray,
                       h: float, q: float, r0_frac: float,
                       core_k: float, core_p: float) -> np.ndarray:
    x, y, z = X3[:,0], X3[:,1], X3[:,2]
    r = np.linalg.norm(np.c_[x-center[0], y-center[1]], axis=1)
    z_fit = np.zeros_like(r_grid)
    for i, rg in enumerate(r_grid):
        w = np.exp(-((r - rg)**2) / (2*h*h + 1e-12))
        if w.sum() < 1e-8:
            idx = np.argsort(np.abs(r - rg))[:8]
            z_fit[i] = float(np.median(z[idx]))
        else:
            z_fit[i] = weighted_quantile(z, w, q)
    # enforce monotone decreasing toward center
    last = z_fit[-1]
    for i in range(len(z_fit)-2, -1, -1):
        if z_fit[i] > last:
            z_fit[i] = last
        else:
            last = z_fit[i]
    # finite-core deepening
    r_max = float(r_grid[-1] + 1e-12)
    r0 = r0_frac * r_max
    core = core_k * (1.0 / (np.sqrt(r_grid**2 + r0**2) + 1e-12)**core_p)
    core -= core[-1]
    return z_fit - core

def analytic_core_template(r_grid: np.ndarray, D: float, p: float, r0_frac: float) -> np.ndarray:
    r_max = float(r_grid[-1] + 1e-12)
    r0 = r0_frac * r_max
    invp = 1.0 / (np.sqrt(r_grid**2 + r0**2) + 1e-12)**p
    invR = 1.0 / (np.sqrt(r_max**2 + r0**2) + 1e-12)**p
    return -D * (invp - invR)

def blend_profiles(z_data: np.ndarray, z_template: np.ndarray, alpha: float) -> np.ndarray:
    alpha = np.clip(alpha, 0.0, 1.0)
    return (1.0 - alpha) * z_data + alpha * z_template

def build_polar_surface(center, r_grid, z_prof, n_theta=160):
    theta = np.linspace(0, 2*np.pi, n_theta)
    R, TH = np.meshgrid(r_grid, theta)
    X = center[0] + R * np.cos(TH)
    Y = center[1] + R * np.sin(TH)
    Z = z_prof[None, :].repeat(n_theta, axis=0)
    return X, Y, Z

def priors_from_profile(r_grid: np.ndarray, z_prof: np.ndarray) -> Dict[str, np.ndarray]:
    """Return normalized depth φ(r) and slope g(r) on [0,1] radius grid."""
    phi_raw = (z_prof[-1] - z_prof)  # positive deeper toward center
    phi = phi_raw / (phi_raw.max() + 1e-12)
    dz = np.gradient(z_prof, r_grid + 1e-12)
    g_raw = np.maximum(0.0, -dz)  # positive where descending
    g = g_raw / (g_raw.max() + 1e-12)
    r_norm = r_grid / (r_grid[-1] + 1e-12)
    return dict(r=r_norm, phi=phi, g=g)

# ----------------------------
# Parsers (baseline + optional prior coupling)
# ----------------------------

def half_sine_proto(width): 
    P = np.sin(np.linspace(0, np.pi, width))
    return P / (np.linalg.norm(P) + 1e-8)

def radius_from_sample_energy(S: Dict[str,np.ndarray]) -> np.ndarray:
    T = len(next(iter(S.values())))
    M = np.stack([_z(S[p]) for p in PRIMS], axis=1)  # (T,3)
    M = M - M.mean(axis=0, keepdims=True)
    U = PCA(n_components=2, random_state=0).fit_transform(M)
    U = U - U.mean(axis=0, keepdims=True)
    r = np.linalg.norm(U, axis=1)
    R = r.max() + 1e-9
    return r / R  # [0,1]

def null_threshold(signal: np.ndarray, proto: np.ndarray, rng, K=40, q=0.95):
    """Null by circularly shifting the signal K times and taking max corr each time."""
    n = len(signal)
    vals = []
    for _ in range(K):
        s = int(rng.integers(0, n))
        xs = np.roll(signal, s)
        vals.append(np.max(np.correlate(xs, proto, mode="same")))
    return float(np.quantile(vals, q))

def corr_at(sig, proto, idx, width, T):
    a, b = max(0, idx - width//2), min(T, idx + width//2)
    w = sig[a:b]
    if len(w) < 3: return 0.0
    w = w - w.mean()
    pr = proto[:len(w)] - proto[:len(w)].mean()
    denom = (np.linalg.norm(w) * np.linalg.norm(pr) + 1e-8)
    return float(np.dot(w, pr) / denom)

def geodesic_parse_report(traces, sigma=9, proto_width=160):
    """Original report parser (kept to preserve baseline metrics)."""
    keys = list(traces.keys())
    T = len(next(iter(traces.values())))
    Eres = perpendicular_energy(traces)
    Sres = {p: moving_average(Eres[p], k=sigma) for p in keys}
    Sraw = {p: moving_average(traces[p], k=sigma) for p in keys}
    Scm  = moving_average(common_mode(traces), k=sigma)
    proto = half_sine_proto(proto_width)

    peak_idx = {p: int(np.argmax(np.correlate(Sres[p], proto, mode="same"))) for p in keys}

    # quick z via circular shifts
    def circ_shift(x, k):
        k = int(k) % len(x)
        if k == 0: return x
        return np.concatenate([x[-k:], x[:-k]])
    def perm_null_z(sig, idx, n=120):
        T = len(sig); obs = corr_at(sig, proto, idx, proto_width, T)
        null = np.empty(n, float); rng_local = np.random.default_rng(0)
        for i in range(n):
            shift = rng_local.integers(1, T-1)
            null[i] = corr_at(circ_shift(sig, shift), proto, idx, proto_width, T)
        mu, sd = float(null.mean()), float(null.std() + 1e-8)
        return (obs - mu) / sd

    z_res = {p: perm_null_z(Sres[p], peak_idx[p]) for p in keys}
    z_raw = {p: perm_null_z(Sraw[p], peak_idx[p]) for p in keys}
    z_cm  = {p: perm_null_z(Scm,      peak_idx[keys[0]]) for p in keys}  # same cm for all
    score = {p: 1.0*z_res[p] + 0.4*z_raw[p] - 0.3*max(0.0, z_cm[p]) for p in keys}

    smax = max(score.values()) + 1e-12
    keep = [p for p in keys if score[p] >= 0.5*smax]
    if not keep:
        keep = [max(keys, key=lambda p: score[p])]
    order = sorted(keep, key=lambda p: peak_idx[p])
    return keep, order

def geodesic_parse_with_prior(traces, priors, *, sigma=9, proto_width=160,
                              alpha=0.05, beta_s=0.25, q_s=2,
                              tau_rel=0.60, tau_abs_q=0.93, null_K=40, seed=0):
    """Report parser with slope/depth priors (optional)."""
    keys = list(traces.keys())
    Sres = {pname: moving_average(perpendicular_energy(traces)[pname], k=sigma) for pname in keys}
    proto = half_sine_proto(proto_width)

    # per-time radius in [0,1]
    r_t = radius_from_sample_energy(Sres)

    r_grid = priors["r"]; phi_prof = priors["phi"]; g_prof = priors["g"]
    phi_t = np.interp(r_t, r_grid, phi_prof)
    g_t   = np.interp(r_t, r_grid, g_prof)

    # slope-weighted energy with unit-mean normalization
    w_slope = 1.0 + beta_s * np.power(g_t, q_s)
    w_slope = w_slope / (np.mean(w_slope) + 1e-9)
    Snew = {pname: w_slope * Sres[pname] for pname in keys}

    # matched filtering
    corr = {p: np.correlate(Snew[p], proto, mode="same") for p in keys}
    peak = {p: int(np.argmax(corr[p])) for p in keys}
    score = {p: float(np.max(corr[p])) for p in keys}

    # robust depth prior at winner index: non-negative
    phi_r = zscore_robust(phi_t)
    phi_pos = np.maximum(0.0, phi_r)
    score_resc = {p: max(0.0, score[p] * (1.0 + alpha * phi_pos[peak[p]])) for p in keys}

    # thresholds
    smax = max(score_resc.values()) + 1e-12
    rng = _rng(int(seed) + 20259)
    tau_abs = {p: null_threshold(Snew[p], proto, rng, K=null_K, q=tau_abs_q) for p in keys}

    keep = [p for p in keys if (score_resc[p] >= tau_rel * smax) and (score_resc[p] >= tau_abs[p])]
    if not keep:
        keep = [max(keys, key=lambda k: score_resc[k])]
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

# ARC-like generator difficulty knobs (must match baseline)
def main():
    ap = argparse.ArgumentParser(description="Stage 11 — warped manifold + funnel fit + stats (CSV/JSON)")
    # data
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--sigma", type=int, default=9, help="smoother window for residual energy")
    ap.add_argument("--cm_amp", type=float, default=0.02)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--amp_jitter", type=float, default=0.4)
    ap.add_argument("--distractor_prob", type=float, default=0.4)
    ap.add_argument("--min_tasks", type=int, default=1)
    ap.add_argument("--max_tasks", type=int, default=3)
    ap.add_argument("--proto_width", type=int, default=160)
    # outputs
    ap.add_argument("--out_plot", type=str, default="manifold_pca3_mesh_warped.png")
    ap.add_argument("--out_csv", type=str, default="stage11_metrics.csv")
    ap.add_argument("--out_json", type=str, default="stage11_summary.json")
    # funnel warp (viz only)
    ap.add_argument("--sigma_scale", type=float, default=0.80)
    ap.add_argument("--depth_scale", type=float, default=1.35)
    ap.add_argument("--mix_z", type=float, default=0.12)
    # fitted funnel profile (viz + optional prior)
    ap.add_argument("--fit_quantile", type=float, default=0.65)
    ap.add_argument("--rbf_bw", type=float, default=0.30)
    ap.add_argument("--core_k", type=float, default=0.18)
    ap.add_argument("--core_p", type=float, default=1.7)
    ap.add_argument("--core_r0_frac", type=float, default=0.14)
    ap.add_argument("--blend_core", type=float, default=0.25)
    ap.add_argument("--template_D", type=float, default=1.2)
    ap.add_argument("--template_p", type=float, default=1.6)
    ap.add_argument("--n_theta", type=int, default=160)
    ap.add_argument("--n_r", type=int, default=220)
    # optional prior coupling (default OFF to preserve baseline metrics)
    ap.add_argument("--use_funnel_prior", type=int, default=0, choices=[0,1])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--beta_s", type=float, default=0.25)
    ap.add_argument("--q_s", type=int, default=2)
    ap.add_argument("--tau_rel", type=float, default=0.60)
    ap.add_argument("--tau_abs_q", type=float, default=0.93)
    ap.add_argument("--null_K", type=int, default=40)
    ap.add_argument("--use_baseline_arc", type=int, default=1, choices=[0,1],
               help="If 1, run metrics on the same ARC-like tasks/parsers as -latest-funnel.")
    args = ap.parse_args()

    rng = _rng(args.seed)

    # ----- per-sample generation + parsing (for metrics) -----
    rows = []
    agg_geo = dict(acc=0, P=0, R=0, F1=0, J=0, H=0, O=0)
    agg_stock = dict(acc=0, P=0, R=0, F1=0, J=0, H=0, O=0)

    # We will build priors from the global PCA cloud later; for now None.
    priors = None

    for i in range(1, args.samples+1):
        traces, true_order = make_synthetic_traces(
            rng, T=args.T, noise=args.noise, cm_amp=args.cm_amp, overlap=args.overlap,
            amp_jitter=args.amp_jitter, distractor_prob=args.distractor_prob,
            tasks_k=(args.min_tasks, args.max_tasks)
        )

        if args.use_funnel_prior and priors is not None:
            keep_g, order_g = geodesic_parse_with_prior(
                traces, priors, sigma=args.sigma, proto_width=args.proto_width,
                alpha=args.alpha, beta_s=args.beta_s, q_s=args.q_s,
                tau_rel=args.tau_rel, tau_abs_q=args.tau_abs_q, null_K=args.null_K, seed=args.seed + i
            )
        else:
            keep_g, order_g = geodesic_parse_report(traces, sigma=args.sigma, proto_width=args.proto_width)
        keep_s, order_s = stock_parse(traces, sigma=args.sigma, proto_width=args.proto_width)

        # exact accuracy = sequence equality
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
    base_params = WellParams(sigma_scale=args.sigma_scale, depth_scale=args.depth_scale, mix_z=args.mix_z)
    X3_warp, m, info = pca3_and_warp(H, energy=E, params=base_params)

    # Raw trisurf
    fig, ax = plot_trisurf(X3_warp, energy=E, title="Stage 11 — Warped Single Well (Report baseline)")
    fig.savefig(args.out_plot, dpi=220); plt.close(fig)

    # ----- Fit a radial funnel (viz) and *optionally* use as priors -----
    r_cloud = np.linalg.norm((X3_warp[:,:2] - info["center"]), axis=1)
    r_max = float(np.quantile(r_cloud, 0.98))
    r_grid = np.linspace(0.0, r_max, args.n_r)
    h = max(1e-6, args.rbf_bw * r_max)
    # data-fit + finite-core
    z_data = fit_radial_profile(
        X3_warp, info["center"], r_grid, h=h, q=args.fit_quantile,
        r0_frac=args.core_r0_frac, core_k=args.core_k, core_p=args.core_p
    )
    # blend with analytic template
    z_tmpl = analytic_core_template(r_grid, D=args.template_D, p=args.template_p, r0_frac=args.core_r0_frac)
    z_prof = blend_profiles(z_data, z_tmpl, args.blend_core)
    # build surface for viz
    Xs, Ys, Zs = build_polar_surface(info["center"], r_grid, z_prof, n_theta=args.n_theta)
    out_fit = args.out_plot.replace(".png", "_fit.png")
    def plot_surface(X, Y, Z, points: np.ndarray, energy: Optional[np.ndarray], title: str, out_path: str, alpha=0.9):
        fig = plt.figure(figsize=(10, 8)); ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=alpha, linewidth=0, antialiased=True)
        if points is not None:
            x, y, z = points[:,0], points[:,1], points[:,2]
            c = None if energy is None else (energy - np.min(energy)) / (np.ptp(energy) + 1e-9)
            ax.scatter(x, y, z, c=c, cmap='viridis', s=10, alpha=0.7)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        ax.set_title(title); fig.colorbar(surf, ax=ax, shrink=0.6, label="height")
        plt.tight_layout(); fig.savefig(out_path, dpi=220); plt.close(fig)
    plot_surface(Xs, Ys, Zs, X3_warp, E, f"Stage 11 — Data-fit Funnel (report, prior {'ON' if args.use_funnel_prior else 'OFF'})", out_fit)

    # prepare priors for optional coupling
    priors = priors_from_profile(r_grid, z_prof)

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
        plot_raw=args.out_plot,
        plot_fitted=out_fit,
        geometry=dict(fit_quantile=args.fit_quantile, rbf_bw=args.rbf_bw,
                      core_k=args.core_k, core_p=args.core_p, core_r0_frac=args.core_r0_frac,
                      blend_core=args.blend_core, template_D=args.template_D, template_p=args.template_p,
                      n_theta=args.n_theta, n_r=args.n_r),
        parser_prior=dict(use_funnel_prior=bool(args.use_funnel_prior),
                          alpha=args.alpha, beta_s=args.beta_s, q_s=args.q_s,
                          tau_rel=args.tau_rel, tau_abs_q=args.tau_abs_q, null_K=args.null_K),
        csv=args.out_csv
    )
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("[SUMMARY] Geodesic:", {k: round(v,3) for k,v in Sg.items()})
    print("[SUMMARY] Stock   :", {k: round(v,3) for k,v in Ss.items()})
    print("[WELL] phantom_index:", round(m["phantom_index"], 4), "margin:", round(m["margin"], 4))
    print(f"[PLOT] RAW:     {args.out_plot}")
    print(f"[PLOT] FITTED:  {out_fit}")
    print(f"[CSV ] {args.out_csv}")
    print(f"[JSON] {args.out_json}")

if __name__ == "__main__":
    main()

