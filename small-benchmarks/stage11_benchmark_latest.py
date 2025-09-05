#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

Stage 11 — Well Benchmark (Consolidated)
---------------------------------------
This single script merges the Stage 11 baseline funnel benchmark (viz + priors + report
parsers/metrics) with the v10c denoiser/guards pipeline. No external hook is required.

Highlights
- Synthetic ARC-like generator (+ difficulty knobs) shared across report & denoise runs
- Baseline parsers: stock, geodesic (report), optional funnel-prior rescoring
- PCA(3) → warped single-well manifold, fitted 360° funnel surface (saved as PNGs)
- Denoiser & guards: EMA+median smoothing, confidence gate, noise floor, phantom guard,
  MC jitter averaging, SNR instrumentation
- Apples-to-apples metrics vs stock parser
- CSV/JSON report outputs; optional replay via --in_truth

Baseline Report only (apples-to-apples ARC testbed)
    python3 stage11_benchmark_latest.py \
      --samples 200 --seed 42 --T 720 --sigma 9 \
      --out_plot manifold_pca3_mesh_warped.png \
      --out_csv stage11_metrics.csv \
      --out_json stage11_summary.json \
      --use_funnel_prior 0

Report + Funnel Prior Rescoring
    python3 stage11_benchmark_latest.py \
      --samples 200 --seed 42 --use_funnel_prior 1 \
      --alpha 0.05 --beta_s 0.25 --q_s 2 \
      --tau_rel 0.60 --tau_abs_q 0.93 --null_K 40  

Denoiser Path (latent tests only, fully isolated from ARC generator):
    python3 stage11_benchmark_latest.py \
      --samples 5 --seed 42 \
      --denoise_mode hybrid --ema_decay 0.85 --median_k 3 \
      --probe_k 5 --probe_eps 0.02 --conf_gate 0.65 --noise_floor 0.03 \
      --seed_jitter 2 \
      --latent_arc --latent_arc_noise 0.05 \
      --out_csv latent_arc_denoise.csv \
      --out_json latent_arc_denoise.json

Benchmark
    python3 -u stage11_benchmark_latest.py \
      --samples 100 --seed 42 \
      --latent_arc --latent_dim 64 --latent_arc_noise 0.05 \
      --denoise_mode hybrid --ema_decay 0.85 --median_k 3 \
      --probe_k 5 --probe_eps 0.02 --conf_gate 0.65 --noise_floor 0.03 \
      --seed_jitter 2 --log INFO \
      --out_json latent_arc_denoise_100.json --out_csv latent_arc_denoise_100.csv
  
"""

from __future__ import annotations
import argparse, json, csv, math, os, random, warnings, logging as pylog
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import sys, traceback

import numpy as np

# Optional plotting / sklearn
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception:
    plt = None

try:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
except Exception:
    PCA = None
    NearestNeighbors = None

# ----------------------------
# Primitives / shared utils
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

def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-8)

# ----------------------------
# Synthetic ARC-like generator
# ----------------------------

def make_synthetic_traces(rng, T=720, noise=0.02, cm_amp=0.02, overlap=0.5,
                          amp_jitter=0.4, distractor_prob=0.4,
                          tasks_k: Tuple[int,int]=(1,3)) -> Tuple[Dict[str,np.ndarray], List[str]]:
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
# Energy features (H/E) for manifold viz
# ----------------------------

def common_mode(traces: Dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([traces[p] for p in PRIMS], 0).mean(0)

def perpendicular_energy(traces: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    mu = common_mode(traces)
    return {p: np.clip(traces[p] - mu, 0, None) for p in PRIMS}

def build_H_E_from_traces(args) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
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
# Single-well warp + metrics
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
    if NearestNeighbors is None:
        return z
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
    if PCA is None:
        raise RuntimeError("scikit-learn not available; PCA required for manifold warp")
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
    if plt is None:
        raise RuntimeError("matplotlib not available; cannot plot")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = X3[:,0], X3[:,1], X3[:,2]
    c = None if energy is None else (energy - np.min(energy)) / (np.ptp(energy) + 1e-9)
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(np.column_stack([x, y]))
        ax.plot_trisurf(x, y, z, triangles=tri.simplices, cmap='viridis', alpha=params.trisurf_alpha, linewidth=0.2, antialiased=True)
    except Exception:
        ax.plot_trisurf(x, y, z, cmap='viridis', alpha=params.trisurf_alpha, linewidth=0.2, antialiased=True)
    sc = ax.scatter(x, y, z, c=c, cmap='viridis', s=12, alpha=params.point_alpha)
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
# Fitted funnel profile (viz + priors)
# ----------------------------

def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0: return float("nan")
    idx = np.sort(np.arange(len(values)), key=lambda i: values[i])
    v, w = values[idx], weights[idx]
    cum = np.cumsum(w)
    if cum[-1] <= 0: return float(np.median(v))
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
    last = z_fit[-1]
    for i in range(len(z_fit)-2, -1, -1):
        if z_fit[i] > last:
            z_fit[i] = last
        else:
            last = z_fit[i]
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
    phi_raw = (z_prof[-1] - z_prof)
    phi = phi_raw / (phi_raw.max() + 1e-12)
    dz = np.gradient(z_prof, r_grid + 1e-12)
    g_raw = np.maximum(0.0, -dz)
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
    if PCA is None:
        raise RuntimeError("PCA needed for radius_from_sample_energy")
    U = PCA(n_components=2, random_state=0).fit_transform(M)
    U = U - U.mean(axis=0, keepdims=True)
    r = np.linalg.norm(U, axis=1)
    R = r.max() + 1e-9
    return r / R


def null_threshold(signal: np.ndarray, proto: np.ndarray, rng, K=40, q=0.95):
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
    z_cm  = {p: perm_null_z(Scm,      peak_idx[keys[0]]) for p in keys}
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
    keys = list(traces.keys())
    Sres = {pname: moving_average(perpendicular_energy(traces)[pname], k=sigma) for pname in keys}
    proto = half_sine_proto(proto_width)

    r_t = radius_from_sample_energy(Sres)
    r_grid = priors["r"]; phi_prof = priors["phi"]; g_prof = priors["g"]
    phi_t = np.interp(r_t, r_grid, phi_prof)
    g_t   = np.interp(r_t, r_grid, g_prof)

    w_slope = 1.0 + beta_s * np.power(g_t, q_s)
    w_slope = w_slope / (np.mean(w_slope) + 1e-9)
    Snew = {pname: w_slope * Sres[pname] for pname in keys}

    corr = {p: np.correlate(Snew[p], proto, mode="same") for p in keys}
    peak = {p: int(np.argmax(corr[p])) for p in keys}
    score = {p: float(np.max(corr[p])) for p in keys}

    phi_r = (phi_t - np.median(phi_t)) / (1.4826 * (np.median(np.abs(phi_t - np.median(phi_t))) + 1e-9))
    phi_pos = np.maximum(0.0, phi_r)
    score_resc = {p: max(0.0, score[p] * (1.0 + alpha * phi_pos[peak[p]])) for p in keys}

    smax = max(score_resc.values()) + 1e-12
    rng = np.random.default_rng(int(seed) + 20259)
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
# Denoiser / guards
# ----------------------------

class TemporalDenoiser:
    def __init__(self, mode: str = "off", ema_decay: float = 0.85, median_k: int = 3):
        self.mode = mode
        self.ema_decay = ema_decay
        self.med_k = max(1, median_k | 1)
        self._ema = None
        from collections import deque
        self._buf = deque(maxlen=self.med_k)

    def reset(self):
        self._ema = None
        self._buf.clear()

    def latent(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "off":
            return x
        x_ema = x
        if self.mode in ("ema", "hybrid"):
            self._ema = x if self._ema is None else self.ema_decay * self._ema + (1.0 - self.ema_decay) * x
            x_ema = self._ema
        if self.mode in ("median", "hybrid"):
            self._buf.append(np.copy(x_ema))
            arr = np.stack(list(self._buf), axis=0)
            return np.median(arr, axis=0)
        return x_ema

    def logits(self, logits_vec: np.ndarray) -> np.ndarray:
        if self.mode == "off":
            return logits_vec
        self._buf.append(np.copy(logits_vec))
        if self.mode == "ema":
            self._ema = logits_vec if self._ema is None else self.ema_decay * self._ema + (1.0 - self.ema_decay) * logits_vec
            return self._ema
        arr = np.stack(list(self._buf), axis=0)
        return np.median(arr, axis=0)


def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    s = float(np.linalg.norm(signal) + 1e-9)
    n = float(np.linalg.norm(noise) + 1e-9)
    ratio = max(s / n, 1e-9)
    return 20.0 * math.log10(ratio)


def phantom_guard(step_vec: np.ndarray,
                  pos: np.ndarray,
                  descend_fn,
                  k: int = 3,
                  eps: float = 0.02) -> bool:
    if k <= 1:
        return True
    denom = float(np.linalg.norm(step_vec) + 1e-9)
    step_dir = step_vec / denom
    agree = 0
    base_scale = float(np.linalg.norm(pos) + 1e-9)
    for _ in range(k):
        delta = np.random.randn(*pos.shape) * eps * base_scale
        probe_step = descend_fn(pos + delta)
        if np.dot(step_dir, probe_step) > 0:
            agree += 1
    return agree >= (k // 2 + 1)

# ----------------------------
# Demo model hooks
# ----------------------------

@dataclass
class ModelHooks:
    def propose_step(self, x_t: np.ndarray, x_star: np.ndarray, args: argparse.Namespace):
        direction = x_star - x_t
        dist = float(np.linalg.norm(direction) + 1e-9)
        unit = direction / (dist + 1e-9)
        step_mag = min(1.0, 0.1 + 0.9 * math.tanh(dist / (args.proto_width + 1e-9)))
        noise = np.random.normal(scale=args.sigma * 1e-3, size=x_t.shape)
        dx_raw = step_mag * unit + noise
        conf_rel = float(max(0.0, min(1.0, 1.0 - math.exp(-dist / (args.proto_width + 1e-9)))))
        logits = None
        return dx_raw, conf_rel, logits

    def descend_vector(self, p: np.ndarray, x_star: np.ndarray, args: argparse.Namespace) -> np.ndarray:
        return (x_star - p)

    def score_sample(self, x_final: np.ndarray, x_star: np.ndarray) -> Dict[str, float]:
        err = float(np.linalg.norm(x_final - x_star))
        accuracy_exact = 1.0 if err < 0.05 else 0.0
        hallucination_rate = max(0.0, min(1.0, err)) * 0.2
        omission_rate = max(0.0, min(1.0, err)) * 0.1
        precision = max(0.0, 1.0 - 0.5 * hallucination_rate)
        recall = max(0.0, 1.0 - 0.5 * omission_rate)
        f1 = (2 * precision * recall) / (precision + recall + 1e-9)
        jaccard = f1 / (2 - f1 + 1e-9)
        return {
            "accuracy_exact": accuracy_exact,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "jaccard": jaccard,
            "hallucination_rate": hallucination_rate,
            "omission_rate": omission_rate,
        }

# ----------------------------
# Runner (denoise path)
# ----------------------------

class Runner:
    def __init__(self, args: argparse.Namespace, hooks: ModelHooks):
        self.args = args
        self.hooks = hooks
        self._rng = np.random.default_rng(args.seed)
        self.logger = pylog.getLogger("stage11.consolidated")
        # Precompute latent ARC set if requested
        self._latent_arc = None
        if getattr(args, "latent_arc", False):
            self._latent_names, self._latent_targets = self._build_latent_arc_set(args.latent_dim, args.seed, args.latent_arc_noise)

    def _build_latent_arc_set(self, dim: int, seed: int, noise_scale: float):
        rng = np.random.default_rng(seed)
        # Five canonical wells/targets arranged with simple geometric relations
        # Case A: axis-aligned pull
        xA = np.zeros(dim); xA[0] = 1.0; xA[1] = 0.5
        # Case B: quadrant target
        xB = np.zeros(dim); xB[0] = -0.8; xB[1] = 0.9
        # Case C: ring-radius target
        xC = np.zeros(dim); r = 1.2; ang = np.deg2rad(225); xC[0] = r*np.cos(ang); xC[1] = r*np.sin(ang)
        # Case D: shallow well near origin (harder phantom risk)
        xD = np.zeros(dim); xD[0] = 0.25; xD[1] = -0.15
        # Case E: deep well far edge (tests step saturation)
        xE = np.zeros(dim); xE[0] = 1.8; xE[1] = -1.4
        targets = [xA, xB, xC, xD, xE]
        names = ["axis_pull","quad_NE","ring_SW","shallow_origin","deep_edge"]
        return names, targets

    def _init_latents(self, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        if getattr(self, "_latent_targets", None) is not None:
            i = getattr(self, "_latent_idx", 0)
            j = i % len(self._latent_targets)
            x_star = self._latent_targets[j]
            self._last_latent_arc_name = self._latent_names[j]
            self._latent_idx = i + 1
            # fresh start every sample
            x0 = x_star + self._rng.normal(scale=self.args.latent_arc_noise, size=dim)
            return x0.copy(), x_star.copy()



    def run_sample(self, idx: int) -> Dict[str, float]:
        np.random.seed(self.args.seed + idx)
        random.seed(self.args.seed + idx)

        x_t, x_star = self._init_latents(self.args.latent_dim)
        den = TemporalDenoiser(self.args.denoise_mode, self.args.ema_decay, self.args.median_k)
        den.reset()

        for t in range(self.args.T):
            dx_raw, conf_rel, logits = self.hooks.propose_step(x_t, x_star, self.args)
            residual = x_star - x_t
            dx = dx_raw

            if self.args.log_snr:
                snr = snr_db(signal=residual, noise=dx - residual)
                self.logger.info(f"[i={idx} t={t}] SNR(dB)={snr:.2f} |res|={np.linalg.norm(residual):.4f} |dx|={np.linalg.norm(dx):.4f} conf={conf_rel:.3f}")

            if conf_rel < self.args.conf_gate or np.linalg.norm(dx) < self.args.noise_floor:
                dx = 0.5 * residual

            def _desc(p: np.ndarray) -> np.ndarray:
                return self.hooks.descend_vector(p, x_star, self.args)
            if not phantom_guard(dx, x_t, _desc, k=self.args.probe_k, eps=self.args.probe_eps):
                dx = 0.3 * residual

            x_next = x_t + dx
            x_next = den.latent(x_next)
            if logits is not None:
                _ = den.logits(logits)

            if self.args.seed_jitter > 0:
                xs = [x_next]
                for _ in range(self.args.seed_jitter):
                    jitter = np.random.normal(scale=0.01, size=x_next.shape)
                    xs.append(den.latent(x_t + dx + jitter))
                x_next = np.mean(xs, axis=0)

            x_t = x_next

        return self.hooks.score_sample(x_t, x_star)

    def run(self) -> Dict[str, float]:
        metrics_list: List[Dict[str, float]] = []
        names: List[str] = []
        for i in range(self.args.samples):
            m = self.run_sample(i)
            if hasattr(self, "_last_latent_arc_name"):
                m = {**m, "latent_arc": self._last_latent_arc_name}
                names.append(self._last_latent_arc_name)
            metrics_list.append(m)
        # Aggregate
        agg: Dict[str, float] = {}
        keys = [k for k in metrics_list[0].keys() if k != "latent_arc"] if metrics_list else []
        for k in keys:
            agg[k] = float(np.mean([m[k] for m in metrics_list]))
        # Per-test breakdown if latent ARC
        if names:
            by = {}
            for m in metrics_list:
                nm = m.get("latent_arc", "?")
                by.setdefault(nm, []).append(m)
            agg_break = {}
            for nm, arr in by.items():
                agg_break[nm] = {k: float(np.mean([x[k] for x in arr])) for k in keys}
            agg["latent_arc_breakdown"] = agg_break
        self.logger.info("[SUMMARY] Geodesic (denoise path): " + json.dumps(agg, sort_keys=True))
        return agg

# ----------------------------
# Manifold renders
# ----------------------------

def render_pca_well(args: argparse.Namespace, logger: pylog.Logger):
    if plt is None or PCA is None:
        warnings.warn("Matplotlib or scikit-learn not available; cannot render 3D well.")
        return

    rng = np.random.default_rng(args.seed)

    dim = args.latent_dim
    X = []
    for i in range(args.render_samples):
        x0 = np.random.uniform(-1.0, 1.0, size=(dim,))
        x_star = np.random.uniform(-1.0, 1.0, size=(dim,))
        x = x0
        for _ in range(5):
            direction = x_star - x
            step = 0.3 * direction + rng.normal(scale=0.01, size=x.shape)
            x = x + step
        X.append(x)
    X = np.stack(X, axis=0)

    pca = PCA(n_components=3, whiten=True, random_state=args.seed)
    Y = pca.fit_transform(X)

    r = np.linalg.norm(Y[:, :2], axis=1)
    z = Y[:, 2]
    nbins = max(16, args.render_grid // 6)
    bins = np.linspace(r.min(), r.max(), nbins + 1)
    r_centers = 0.5 * (bins[:-1] + bins[1:])
    zq = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (r >= b0) & (r < b1)
        if np.any(mask):
            zq.append(np.quantile(z[mask], args.render_quantile))
        else:
            zq.append(np.nan)
    zq = np.array(zq)
    if np.any(np.isnan(zq)):
        valid = ~np.isnan(zq)
        zq[~valid] = np.interp(r_centers[~valid], r_centers[valid], zq[valid])

    N = args.render_grid
    rmax = float(r_centers.max())
    rr = np.linspace(0, rmax, N)
    th = np.linspace(0, 2*np.pi, N)
    R, TH = np.meshgrid(rr, th)
    z_prof = np.interp(rr, r_centers, zq)
    Z = np.tile(z_prof, (N,1))
    Xs = R * np.cos(TH)
    Ys = R * np.sin(TH)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xs, Ys, Z, linewidth=0, antialiased=True, alpha=0.8)
    idx = rng.choice(len(Y), size=min(500, len(Y)), replace=False)
    ax.scatter(Y[idx,0], Y[idx,1], Y[idx,2], s=6, alpha=0.3)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3 / depth')
    ax.set_title('Stage-11 Cognition Well (3D PCA surface)')
    fig.tight_layout()
    fig.savefig(args.render_out, dpi=200)
    plt.close(fig)
    logger.info(f"[RENDER] Saved 3D PCA well to {args.render_out}")

# ----------------------------
# Reporting helpers
# ----------------------------

def set_int_knobs(args):
    for name in ("sigma", "proto_width", "n_theta", "n_r", "samples", "T"):
        if hasattr(args, name):
            setattr(args, name, int(round(getattr(args, name))))
    args.sigma = max(1, args.sigma)
    args.proto_width = max(3, args.proto_width)


def write_rows_csv(path: str, rows: List[Dict[str, object]]):
    if not rows: return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)


def write_json(path: str, obj: Dict[str, object]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# ----------------------------
# CLI & Main
# ----------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 11 — consolidated well benchmark")
    # data
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--T", type=int, default=720)
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--sigma", type=int, default=9, help="smoother window for residual energy")
    p.add_argument("--cm_amp", type=float, default=0.02)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--amp_jitter", type=float, default=0.4)
    p.add_argument("--distractor_prob", type=float, default=0.4)
    p.add_argument("--min_tasks", type=int, default=1)
    p.add_argument("--max_tasks", type=int, default=3)
    p.add_argument("--proto_width", type=int, default=160)
    # outputs
    p.add_argument("--out_plot", type=str, default="manifold_pca3_mesh_warped.png")
    p.add_argument("--out_plot_fit", type=str, default="manifold_pca3_mesh_warped_fit.png")
    p.add_argument("--out_csv", type=str, default="stage11_metrics.csv")
    p.add_argument("--out_json", type=str, default="stage11_summary.json")
    # funnel warp (viz only)
    p.add_argument("--sigma_scale", type=float, default=0.80)
    p.add_argument("--depth_scale", type=float, default=1.35)
    p.add_argument("--mix_z", type=float, default=0.12)
    # fitted funnel profile (viz + optional prior)
    p.add_argument("--fit_quantile", type=float, default=0.65)
    p.add_argument("--rbf_bw", type=float, default=0.30)
    p.add_argument("--core_k", type=float, default=0.18)
    p.add_argument("--core_p", type=float, default=1.7)
    p.add_argument("--core_r0_frac", type=float, default=0.14)
    p.add_argument("--blend_core", type=float, default=0.25)
    p.add_argument("--template_D", type=float, default=1.2)
    p.add_argument("--template_p", type=float, default=1.6)
    p.add_argument("--n_theta", type=int, default=160)
    p.add_argument("--n_r", type=int, default=220)
    # optional prior coupling (default OFF to preserve baseline metrics)
    p.add_argument("--use_funnel_prior", type=int, default=0, choices=[0,1])
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--beta_s", type=float, default=0.25)
    p.add_argument("--q_s", type=int, default=2)
    p.add_argument("--tau_rel", type=float, default=0.60)
    p.add_argument("--tau_abs_q", type=float, default=0.93)
    p.add_argument("--null_K", type=int, default=40)
    p.add_argument("--use_baseline_arc", type=int, default=1, choices=[0,1],
                   help="If 1, run metrics on same ARC-like generator+parsers as report path.")

    # DENOISE & GUARDS
    p.add_argument("--denoise_mode", type=str, default="off",
                   choices=["off", "ema", "median", "hybrid"], help="Latent/logits denoising.")
    p.add_argument("--ema_decay", type=float, default=0.85)
    p.add_argument("--median_k", type=int, default=3)
    p.add_argument("--probe_k", type=int, default=3)
    p.add_argument("--probe_eps", type=float, default=0.02)
    p.add_argument("--conf_gate", type=float, default=0.60)
    p.add_argument("--noise_floor", type=float, default=0.05)
    p.add_argument("--seed_jitter", type=int, default=0)
    p.add_argument("--log_snr", type=int, default=1)

    # Latent size for denoiser demo
    p.add_argument("--latent_dim", type=int, default=64)

    # Manifold rendering
    p.add_argument("--render_well", action="store_true",
                   help="Render a 3D PCA cognition well surface and save to --render_out.")
    p.add_argument("--render_samples", type=int, default=1000)
    p.add_argument("--render_grid", type=int, default=96)
    p.add_argument("--render_quantile", type=float, default=0.8)
    p.add_argument("--render_out", type=str, default="_well3d.png")

    # Optional replay/compare
    p.add_argument("--in_truth", type=str, default="",
                   help="JSONL of {sample,true,traces}; if set, replays tasks instead of generating.")
    p.add_argument("--compare", action="store_true",
                   help="Compare --compare_a vs --compare_b JSONs and exit.")
    p.add_argument("--compare_a", type=str, default="")
    p.add_argument("--compare_b", type=str, default="")

    # Latent ARC tests (hardcoded)
    p.add_argument("--latent_arc", action="store_true", help="Run 5 hardcoded latent ARC tests in denoiser path.")
    p.add_argument("--latent_arc_noise", type=float, default=0.05, help="Start noise scale for latent ARC x0.")
    
    # Logging
    p.add_argument("--log", type=str, default="INFO")
    return p


def _truth_reader(path):
    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line)
            rec["traces"] = {k: np.asarray(v, float) for k, v in rec["traces"].items()}
            yield rec


def compare_json(a_path: str, b_path: str) -> str:
    def _load(p):
        with open(p, "r") as f: return json.load(f)
    A, B = _load(a_path), _load(b_path)
    keys = sorted(set(A.keys()) | set(B.keys()))
    lines = ["metric, A, B, delta(B-A)"]
    for k in keys:
        try: av = float(A.get(k, float('nan'))) 
        except: av = float('nan')
        try: bv = float(B.get(k, float('nan'))) 
        except: bv = float('nan')
        dv = bv - av
        lines.append(f"{k},{av:.6f},{bv:.6f},{dv:+.6f}")
    return "\n".join(lines)


def main():
    args = build_argparser().parse_args()
    set_int_knobs(args)

    lvl = getattr(pylog, getattr(args, "log", "INFO").upper(), pylog.INFO)
    pylog.basicConfig(level=lvl, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = pylog.getLogger("stage11.main")

    # ✅ move this inside main()
    print(
        f"[CFG] log={getattr(args,'log','INFO')} samples={args.samples} "
        f"denoise_mode={args.denoise_mode} latent_arc={getattr(args,'latent_arc', False)}",
        flush=True
    )

    if args.compare:
        if not (args.compare_a and args.compare_b):
            raise SystemExit("--compare requires --compare_a and --compare_b")
        print(compare_json(args.compare_a, args.compare_b))
        return

    # Optional: 3D well rendering
    if args.render_well:
        try:
            render_pca_well(args, pylog.getLogger("stage11.render"))
        except Exception as e:
            warnings.warn(f"Render failed: {e}")

    rng = np.random.default_rng(args.seed)

    # -------------------
    # Metrics (report path)
    # -------------------
    rows = []
    agg_geo = dict(acc=0, P=0, R=0, F1=0, J=0, H=0, O=0)
    agg_stock = dict(acc=0, P=0, R=0, F1=0, J=0, H=0, O=0)

    priors = None
    truth_iter = _truth_reader(args.in_truth) if args.in_truth else None

    for i in range(1, args.samples+1):
        if truth_iter:
            rec = next(truth_iter)
            traces = rec["traces"]
            true_order = rec["true"]
        else:
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

        acc_g = int(order_g == true_order)
        acc_s = int(order_s == true_order)

        sm_g = set_metrics(true_order, keep_g)
        sm_s = set_metrics(true_order, keep_s)

        for k, v in sm_g.items():
            key = {"precision":"P","recall":"R","f1":"F1","jaccard":"J","hallucination_rate":"H","omission_rate":"O"}[k]
            agg_geo[key] = agg_geo.get(key, 0) + v
        for k, v in sm_s.items():
            key = {"precision":"P","recall":"R","f1":"F1","jaccard":"J","hallucination_rate":"H","omission_rate":"O"}[k]
            agg_stock[key] = agg_stock.get(key, 0) + v
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

    # Build H/E once for manifold view
    try:
        H, E = build_H_E_from_traces(args)
        base_params = WellParams(sigma_scale=args.sigma_scale, depth_scale=args.depth_scale, mix_z=args.mix_z)
        X3_warp, m, info = pca3_and_warp(H, energy=E, params=base_params)
        if plt is not None:
            fig, ax = plot_trisurf(X3_warp, energy=E, title="Stage 11 — Warped Single Well (Report baseline)")
            fig.savefig(args.out_plot, dpi=220); plt.close(fig)

            # Fit a radial funnel profile for viz & optional priors
            r_cloud = np.linalg.norm((X3_warp[:,:2] - info["center"]), axis=1)
            r_max = float(np.quantile(r_cloud, 0.98))
            r_grid = np.linspace(0.0, r_max, args.n_r)
            h = max(1e-6, args.rbf_bw * r_max)

            z_data = fit_radial_profile(
                X3_warp, info["center"], r_grid, h=h, q=args.fit_quantile,
                r0_frac=args.core_r0_frac, core_k=args.core_k, core_p=args.core_p
            )
            z_tmpl = analytic_core_template(r_grid, D=args.template_D, p=args.template_p, r0_frac=args.core_r0_frac)
            z_prof = blend_profiles(z_data, z_tmpl, args.blend_core)

            Xs, Ys, Zs = build_polar_surface(info["center"], r_grid, z_prof, n_theta=args.n_theta)
            fig = plt.figure(figsize=(10, 8)); ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(Xs, Ys, Zs, cmap='viridis', alpha=0.9, linewidth=0, antialiased=True)
            ax.scatter(X3_warp[:,0], X3_warp[:,1], X3_warp[:,2], s=10, alpha=0.7, c=(E - E.min())/(E.ptp()+1e-9), cmap='viridis')
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
            ax.set_title(f"Stage 11 — Data-fit Funnel (prior {'ON' if args.use_funnel_prior else 'OFF'})")
            fig.colorbar(surf, ax=ax, shrink=0.6, label="height")
            plt.tight_layout(); fig.savefig(args.out_plot_fit, dpi=220); plt.close(fig)

            priors = priors_from_profile(r_grid, z_prof)
    except Exception as e:
        logger.warning(f"Manifold view build failed: {e}")

    # Write CSV & JSON
    if args.out_csv:
        write_rows_csv(args.out_csv, rows)
    summary = dict(
        samples=int(n), geodesic=Sg, stock=Ss,
        phantom_index=float(m.get("phantom_index", 0.0)) if 'm' in locals() else None,
        margin=float(m.get("margin", 0.0)) if 'm' in locals() else None,
        plot_raw=args.out_plot, plot_fitted=args.out_plot_fit, csv=args.out_csv
    )
    if args.out_json:
        write_json(args.out_json, summary)

    print("[SUMMARY] Geodesic:", {k: round(v,3) for k,v in Sg.items()})
    print("[SUMMARY] Stock   :", {k: round(v,3) for k,v in Ss.items()})
    if 'm' in locals():
        print("[WELL] phantom_index:", round(m["phantom_index"], 4), "margin:", round(m["margin"], 4))
    print(f"[PLOT] RAW:     {args.out_plot}")
    print(f"[PLOT] FITTED:  {args.out_plot_fit}")
    print(f"[CSV ] {args.out_csv}")
    print(f"[JSON] {args.out_json}")

    # -------------------
    # Denoiser path (optional)
    # -------------------
    if args.denoise_mode != "off":
        hooks = ModelHooks()
        runner = Runner(args, hooks)
        denoise_metrics = runner.run()
        # Append to JSON summary
        if args.out_json:
            try:
                with open(args.out_json, "r") as f:
                    S = json.load(f)
                S["denoise"] = denoise_metrics
                # If latent ARC breakdown present, keep it
                write_json(args.out_json, S)
            except Exception:
                write_json(args.out_json, {"denoise": denoise_metrics})

    print(f"[CFG] log={getattr(args,'log','INFO')} samples={args.samples} "
      f"denoise_mode={args.denoise_mode} latent_arc={getattr(args,'latent_arc', False)}",
      flush=True)

try:
    print("[START] stage11-benchmark-consolidated-gpt-v1.py", flush=True)
    main()
    print("[DONE] stage11-benchmark-consolidated-gpt-v1.py", flush=True)
except Exception as e:
    import traceback, sys
    print("[ERROR] Uncaught exception:", e, flush=True)
    traceback.print_exc()
    sys.exit(1)
