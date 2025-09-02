#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 11 — Consolidated Well Benchmark (Rewrite)
------------------------------------------------
One self‑contained script that:
  • Generates ARC‑like traces and runs two report parsers (stock vs. geodesic)
  • Builds a PCA(3) manifold → single‑well warp → fits radial funnel + optional priors
  • Offers a denoiser/guards demo path on synthetic latent ARC targets
  • (New) Optional tap‑style warp configuration via --warp_config (pre‑warp + 3 passes)
  • Emits CSV + JSON summaries and optional 3D plots

This is a pragmatic rewrite focused on readability, fewer global side‑effects, and
clear extension points. It merges functionality that previously lived across multiple
files into one file while keeping helpers tight and well‑documented.

Example (report baseline only):
  python3 stage11_benchmark_rewrite.py \
    --samples 200 --seed 42 --out_json out/summary.json --out_csv out/metrics.csv

With prior coupling (rescoring):
  python3 stage11_benchmark_rewrite.py \
    --samples 200 --seed 42 --use_funnel_prior 1 --out_json out/summary.json

With denoiser path:
  python3 stage11_benchmark_rewrite.py \
    --samples 100 --latent_arc --denoise_mode hybrid --seed_jitter 2 --out_json out/summary.json

With tap‑style warp config (pre‑warp + 3 passes used for the manifold view stage):
  python3 stage11_benchmark_rewrite.py \
    --warp_config warp_config_tap9.json --out_plot out/well.png --out_json out/summary.json
"""

from __future__ import annotations
import argparse, json, csv, math, os, warnings, logging as pylog, random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np

# Optional plotting / sklearn
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
except Exception:
    PCA = None
    NearestNeighbors = None

PRIMS = ["flip_h","flip_v","rotate"]

# ----------------------------
# Small numeric helpers
# ----------------------------

def moving_average(x, k=9):
    if k <= 1: return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(xp, np.ones(k)/k, mode="valid")

def gaussian_bump(T, center, width, amp=1.0):
    t = np.arange(T); sig2 = (width/2.355)**2
    return amp * np.exp(-(t-center)**2 / (2*sig2))

def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float); return (x - x.mean()) / (x.std() + 1e-8)

# ----------------------------
# ARC‑like generator + energy
# ----------------------------

def make_synthetic_traces(rng, T=720, noise=0.02, cm_amp=0.02, overlap=0.5,
                          amp_jitter=0.4, distractor_prob=0.4,
                          tasks_k: Tuple[int,int]=(1,3)) -> Tuple[Dict[str,np.ndarray], List[str]]:
    k = int(rng.integers(tasks_k[0], tasks_k[1]+1))
    tasks = list(rng.choice(PRIMS, size=k, replace=False)); rng.shuffle(tasks)
    base = np.array([0.20, 0.50, 0.80]) * T
    centers = ((1.0 - overlap) * base + overlap * (T * 0.50)).astype(int)
    width = int(max(12, T * 0.10))
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
    t = np.arange(T); cm = cm_amp * (1.0 + 0.2*np.sin(2*np.pi*t/max(30, T//6)))
    for p in PRIMS:
        traces[p] = np.clip(traces[p] + cm, 0, None)
        traces[p] = traces[p] + rng.normal(0, noise, size=T)
        traces[p] = np.clip(traces[p], 0, None)
    return traces, tasks

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
# PCA(3) → single‑well warp (+ optional pre‑warp & passes)
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
    trisurf_alpha: float = 0.65
    point_alpha: float = 0.85

def _softmin_center(X2: np.ndarray, energy: Optional[np.ndarray], tau: float):
    n = len(X2)
    if energy is None:
        w = np.ones(n)/n
    else:
        e = (energy - energy.min()) / (energy.std() + 1e-8)
        w = np.exp(-e / max(tau, 1e-6)); w = w / (w.sum() + 1e-12)
    c = (w[:,None]*X2).sum(0)
    return c, w

def _isotropize(X2: np.ndarray):
    mu = X2.mean(0); Y = X2 - mu
    C = (Y.T @ Y) / max(len(Y)-1,1)
    evals, evecs = np.linalg.eigh(C)
    T = evecs @ np.diag(1.0/np.sqrt(np.maximum(evals,1e-8))) @ evecs.T
    return (Y @ T), (mu, T)

def _radial_funnel(X2_iso: np.ndarray, z: np.ndarray, sigma: float, depth_scale: float, mix_z: float):
    r = np.linalg.norm(X2_iso, axis=1) + 1e-9
    u = X2_iso / r[:,None]
    z_funnel = -np.exp(-(r**2)/(2*sigma*sigma))
    z_new = depth_scale*z_funnel + mix_z*(z - z.mean())
    return (r[:,None]*u), z_new

def _lateral_inhibition(z: np.ndarray, X2: np.ndarray, k:int, strength: float) -> np.ndarray:
    if NearestNeighbors is None: return z
    k = min(max(3,k), len(X2))
    nbrs = NearestNeighbors(n_neighbors=k).fit(X2)
    idx = nbrs.kneighbors(return_distance=False)
    ranks = np.argsort(np.argsort(z[idx], axis=1), axis=1)[:,0]
    boost = (ranks > 0).astype(float)
    return z + strength*0.5*(boost - 0.5)*(np.std(z)+1e-6)

def pca3_and_warp(H: np.ndarray, energy: Optional[np.ndarray], P: WellParams):
    if PCA is None: raise RuntimeError("scikit-learn not available; PCA required")
    pca = PCA(n_components=3, whiten=P.whiten, random_state=0)
    X3 = pca.fit_transform(H)
    X2, z = X3[:,:2], X3[:,2].copy()
    c, _ = _softmin_center(X2, energy, P.tau)
    X2_iso, _ = _isotropize(X2 - c) if P.isotropize_xy else (X2 - c, (None,None))
    r = np.linalg.norm(X2_iso, axis=1); sigma = np.median(r)*P.sigma_scale + 1e-9
    X2_new, z_new = _radial_funnel(X2_iso, z, sigma, P.depth_scale, P.mix_z)
    z_new = _lateral_inhibition(z_new, X2_new, k=P.inhibit_k, strength=P.inhibit_strength)
    out = np.column_stack([X2_new + c, z_new])
    return out, dict(center=c, sigma=sigma)

# --- histogram energy for phantom metrics / minima (2D) ---
from scipy.ndimage import gaussian_filter

def _hist_energy(X2: np.ndarray, nbins=120, sigma=5.0):
    x, y = X2[:,0], X2[:,1]
    H, xe, ye = np.histogram2d(x, y, bins=nbins)
    Hs = gaussian_filter(H, sigma=sigma)
    U  = -Hs
    return U, Hs, xe, ye

def _find_minima(U, Hs, density_floor=2.5, min_prom=0.38, merge_tol=1e-6):
    h, w = U.shape; mins = []
    for i in range(1,h-1):
        for j in range(1,w-1):
            if Hs[i,j] < density_floor: continue
            c = U[i,j]; neigh = U[i-1:i+2, j-1:j+2].copy(); neigh[1,1] = np.nan
            prom = np.nanmean(neigh) - c
            if prom >= min_prom and np.all(c < np.nan_to_num(neigh, nan=np.inf)):
                mins.append((c,i,j))
    if not mins: return []
    mins.sort(key=lambda t: t[0])
    uniq = [mins[0]]
    for c,i,j in mins[1:]:
        if abs(c - uniq[-1][0]) > merge_tol: uniq.append((c,i,j))
    return uniq

def _phantom_metrics(U):
    vals = np.sort(U.ravel())
    # proxy: gap between deepest and 2nd deepest bins normalized by span
    z0, z1 = vals[0], vals[1] if len(vals) > 1 else (vals[0], vals[0])
    span = vals[int(0.95*len(vals))-1] - vals[int(0.05*len(vals))]
    span = max(span, 1e-9)
    return float((z1 - z0)/span), float(z1 - z0)

def _radial_rewarp(X2: np.ndarray, center_xy, alpha: float, scale: float) -> np.ndarray:
    v = X2 - np.asarray(center_xy)[None,:]
    r = np.linalg.norm(v, axis=1) + 1e-12
    shrink = 1.0 - alpha * np.exp(-(r/scale)**2)
    return np.asarray(center_xy)[None,:] + (v.T * shrink).T

# ----------------------------
# Priors (from fitted funnel)
# ----------------------------

def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0: return float("nan")
    idx = np.sort(np.arange(len(values)), key=lambda i: values[i])
    v, w = values[idx], weights[idx]; cum = np.cumsum(w)
    if cum[-1] <= 0: return float(np.median(v))
    j = int(np.searchsorted(cum, q*cum[-1], side="left")); j = min(max(j,0), len(v)-1)
    return float(v[j])

def fit_radial_profile(X3: np.ndarray, center: np.ndarray, r_grid: np.ndarray,
                       h: float, q: float, r0_frac: float, core_k: float, core_p: float) -> np.ndarray:
    x,y,z = X3[:,0], X3[:,1], X3[:,2]
    r = np.linalg.norm(np.c_[x-center[0], y-center[1]], axis=1)
    z_fit = np.zeros_like(r_grid)
    for i, rg in enumerate(r_grid):
        w = np.exp(-((r-rg)**2)/(2*h*h + 1e-12))
        z_fit[i] = weighted_quantile(z, w if np.sum(w)>1e-8 else np.ones_like(z), q)
    # monotone enforce (non‑increasing)
    last = z_fit[-1]
    for i in range(len(z_fit)-2, -1, -1):
        if z_fit[i] > last: z_fit[i] = last
        else: last = z_fit[i]
    # add analytic core
    r_max = float(r_grid[-1] + 1e-12); r0 = r0_frac * r_max
    core = core_k * (1.0 / (np.sqrt(r_grid**2 + r0**2) + 1e-12)**core_p); core -= core[-1]
    return z_fit - core

def priors_from_profile(r_grid: np.ndarray, z_prof: np.ndarray) -> Dict[str, np.ndarray]:
    phi_raw = (z_prof[-1] - z_prof); phi = phi_raw / (phi_raw.max() + 1e-12)
    dz = np.gradient(z_prof, r_grid + 1e-12); g_raw = np.maximum(0.0, -dz)
    g = g_raw / (g_raw.max() + 1e-12)
    r_norm = r_grid / (r_grid[-1] + 1e-12)
    return dict(r=r_norm, phi=phi, g=g)

# ----------------------------
# Parsers (stock vs geodesic; optional prior coupling)
# ----------------------------

def half_sine_proto(width):
    P = np.sin(np.linspace(0, np.pi, width)); return P / (np.linalg.norm(P)+1e-8)

def geodesic_parse_report(traces, sigma=9, proto_width=160):
    keys = list(traces.keys()); T = len(next(iter(traces.values())))
    Eres = perpendicular_energy(traces)
    Sres = {p: moving_average(Eres[p], k=sigma) for p in keys}
    Sraw = {p: moving_average(traces[p], k=sigma) for p in keys}
    Scm  = moving_average(common_mode(traces), k=sigma)
    proto = half_sine_proto(proto_width)
    peak_idx = {p: int(np.argmax(np.correlate(Sres[p], proto, mode="same"))) for p in keys}
    def corr_at(sig, idx):
        a,b = max(0, idx - proto_width//2), min(T, idx + proto_width//2)
        w = sig[a:b]; pr = proto[:len(w)]
        w = w - w.mean(); pr = pr - pr.mean()
        return float(np.dot(w, pr) / (np.linalg.norm(w)*np.linalg.norm(pr) + 1e-8))
    rng = np.random.default_rng(0)
    def perm_null_z(sig, idx, n=120):
        obs = corr_at(sig, idx)
        null = np.empty(n, float)
        for i in range(n):
            shift = int(rng.integers(1, T-1))
            xs = np.concatenate([sig[-shift:], sig[:-shift]])
            null[i] = corr_at(xs, idx)
        mu, sd = float(null.mean()), float(null.std() + 1e-8)
        return (obs - mu) / sd
    z_res = {p: perm_null_z(Sres[p], peak_idx[p]) for p in keys}
    z_raw = {p: perm_null_z(Sraw[p], peak_idx[p]) for p in keys}
    z_cm  = {p: perm_null_z(Scm,      peak_idx[keys[0]]) for p in keys}
    score = {p: 1.0*z_res[p] + 0.4*z_raw[p] - 0.3*max(0.0, z_cm[p]) for p in keys}
    smax = max(score.values()) + 1e-12
    keep = [p for p in keys if score[p] >= 0.5*smax] or [max(keys, key=lambda p: score[p])]
    order = sorted(keep, key=lambda p: peak_idx[p])
    return keep, order

def radius_from_sample_energy(S: Dict[str,np.ndarray]) -> np.ndarray:
    T = len(next(iter(S.values())))
    M = np.stack([_z(S[p]) for p in PRIMS], axis=1); M = M - M.mean(0, keepdims=True)
    if PCA is None: raise RuntimeError("PCA required for radius_from_sample_energy")
    U = PCA(n_components=2, random_state=0).fit_transform(M); U = U - U.mean(0, keepdims=True)
    r = np.linalg.norm(U, axis=1); R = r.max() + 1e-9
    return r / R

def null_threshold(signal: np.ndarray, proto: np.ndarray, rng, K=40, q=0.93):
    n = len(signal); vals = []
    for _ in range(K):
        s = int(rng.integers(0, n)); xs = np.roll(signal, s)
        vals.append(np.max(np.correlate(xs, proto, mode="same")))
    return float(np.quantile(vals, q))

def geodesic_parse_with_prior(traces, priors, *, sigma=9, proto_width=160,
                              alpha=0.05, beta_s=0.25, q_s=2,
                              tau_rel=0.60, tau_abs_q=0.93, null_K=40, seed=0):
    keys = list(traces.keys())
    Sres = {p: moving_average(perpendicular_energy(traces)[p], k=sigma) for p in keys}
    proto = half_sine_proto(proto_width)
    r_t = radius_from_sample_energy(Sres)
    r_grid, phi_prof, g_prof = priors["r"], priors["phi"], priors["g"]
    phi_t = np.interp(r_t, r_grid, phi_prof); g_t = np.interp(r_t, r_grid, g_prof)
    w_slope = 1.0 + beta_s * np.power(g_t, q_s); w_slope = w_slope / (np.mean(w_slope) + 1e-9)
    Snew = {p: w_slope * Sres[p] for p in keys}
    corr = {p: np.correlate(Snew[p], proto, mode="same") for p in keys}
    peak = {p: int(np.argmax(corr[p])) for p in keys}
    score = {p: float(np.max(corr[p])) for p in keys}
    phi_r = (phi_t - np.median(phi_t)) / (1.4826*(np.median(np.abs(phi_t - np.median(phi_t))) + 1e-9))
    phi_pos = np.maximum(0.0, phi_r)
    score_resc = {p: max(0.0, score[p]*(1.0 + alpha*phi_pos[peak[p]])) for p in keys}
    smax = max(score_resc.values()) + 1e-12
    rng = np.random.default_rng(int(seed)+20259)
    tau_abs = {p: null_threshold(Snew[p], proto, rng, K=null_K, q=tau_abs_q) for p in keys}
    keep = [p for p in keys if (score_resc[p] >= tau_rel*smax) and (score_resc[p] >= tau_abs[p])] or [max(keys, key=lambda k: score_resc[k])]
    order = sorted(keep, key=lambda p: peak[p])
    return keep, order

def stock_parse(traces, sigma=9, proto_width=160):
    keys = list(traces.keys()); S = {p: moving_average(traces[p], k=sigma) for p in keys}
    proto = half_sine_proto(proto_width)
    peak = {p: int(np.argmax(np.correlate(S[p], proto, mode="same"))) for p in keys}
    score = {p: float(np.max(np.correlate(S[p], proto, mode="same"))) for p in keys}
    smax = max(score.values()) + 1e-12
    keep = [p for p in keys if score[p] >= 0.6*smax] or [max(keys, key=lambda p: score[p])]
    order = sorted(keep, key=lambda p: peak[p])
    return keep, order

def set_metrics(true_list: List[str], pred_list: List[str]) -> Dict[str,float]:
    Tset, Pset = set(true_list), set(pred_list)
    tp, fp, fn = len(Tset & Pset), len(Pset - Tset), len(Tset - Pset)
    precision = tp / max(1, len(Pset)); recall = tp / max(1, len(Tset))
    f1 = 0.0 if precision+recall==0 else (2*precision*recall)/(precision+recall)
    jaccard = tp / max(1, len(Tset | Pset))
    return dict(precision=precision, recall=recall, f1=f1, jaccard=jaccard,
                hallucination_rate=fp/max(1,len(Pset)), omission_rate=fn/max(1,len(Tset)))

# ----------------------------
# Denoiser/guards demo (latent ARC)
# ----------------------------

class TemporalDenoiser:
    def __init__(self, mode: str = "off", ema_decay: float = 0.85, median_k: int = 3):
        self.mode = mode; self.ema_decay = ema_decay; self.med_k = max(1, median_k|1)
        self._ema = None; from collections import deque; self._buf = deque(maxlen=self.med_k)
    def reset(self): self._ema=None; self._buf.clear()
    def latent(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "off": return x
        x_ema = x
        if self.mode in ("ema","hybrid"):
            self._ema = x if self._ema is None else self.ema_decay*self._ema + (1.0-self.ema_decay)*x
            x_ema = self._ema
        if self.mode in ("median","hybrid"):
            self._buf.append(x_ema.copy()); arr = np.stack(list(self._buf),0)
            return np.median(arr,0)
        return x_ema
    def logits(self, logits: np.ndarray) -> np.ndarray:
        if self.mode == "off": return logits
        self._buf.append(logits.copy())
        if self.mode == "ema":
            self._ema = logits if self._ema is None else self.ema_decay*self._ema + (1.0-self.ema_decay)*logits
            return self._ema
        arr = np.stack(list(self._buf),0); return np.median(arr,0)

def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    s = float(np.linalg.norm(signal)+1e-9); n = float(np.linalg.norm(noise)+1e-9)
    return 20.0*math.log10(max(s/n, 1e-9))

def phantom_guard(step_vec: np.ndarray, pos: np.ndarray, descend_fn, k:int=3, eps:float=0.02) -> bool:
    if k <= 1: return True
    denom = float(np.linalg.norm(step_vec) + 1e-9); step_dir = step_vec/denom
    agree = 0; base = float(np.linalg.norm(pos) + 1e-9)
    for _ in range(k):
        delta = np.random.randn(*pos.shape) * eps * base
        probe_step = descend_fn(pos + delta)
        if np.dot(step_dir, probe_step) > 0: agree += 1
    return agree >= (k//2 + 1)

@dataclass
class DemoHooks:
    def propose_step(self, x_t: np.ndarray, x_star: np.ndarray, args: argparse.Namespace):
        direction = x_star - x_t; dist = float(np.linalg.norm(direction) + 1e-9)
        unit = direction / (dist + 1e-9)
        step_mag = min(1.0, 0.1 + 0.9*math.tanh(dist/(args.proto_width + 1e-9)))
        noise = np.random.normal(scale=args.sigma*1e-3, size=x_t.shape)
        dx_raw = step_mag*unit + noise
        conf_rel = float(max(0.0, min(1.0, 1.0 - math.exp(-dist/(args.proto_width + 1e-9)))))
        logits = None
        return dx_raw, conf_rel, logits
    def descend_vector(self, p: np.ndarray, x_star: np.ndarray, args: argparse.Namespace) -> np.ndarray:
        return (x_star - p)
    def score_sample(self, x_final: np.ndarray, x_star: np.ndarray) -> Dict[str,float]:
        err = float(np.linalg.norm(x_final - x_star))
        accuracy_exact = 1.0 if err < 0.05 else 0.0
        hallucination_rate = max(0.0, min(1.0, err))*0.2
        omission_rate      = max(0.0, min(1.0, err))*0.1
        precision = max(0.0, 1.0 - 0.5*hallucination_rate)
        recall    = max(0.0, 1.0 - 0.5*omission_rate)
        f1 = (2*precision*recall)/(precision+recall+1e-9)
        jaccard = f1/(2 - f1 + 1e-9)
        return dict(accuracy_exact=accuracy_exact, precision=precision, recall=recall,
                    f1=f1, jaccard=jaccard, hallucination_rate=hallucination_rate, omission_rate=omission_rate)

class DenoiseRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args; self.rng = np.random.default_rng(args.seed)
        self.hooks = DemoHooks()
    def _targets(self, dim: int):
        rng = np.random.default_rng(self.args.seed)
        xA = np.zeros(dim); xA[0]=1.0; xA[1]=0.5
        xB = np.zeros(dim); xB[0]=-0.8; xB[1]=0.9
        r,ang = 1.2, np.deg2rad(225); xC = np.zeros(dim); xC[0]=r*np.cos(ang); xC[1]=r*np.sin(ang)
        xD = np.zeros(dim); xD[0]=0.25; xD[1]=-0.15
        xE = np.zeros(dim); xE[0]=1.8; xE[1]=-1.4
        return ["axis_pull","quad_NE","ring_SW","shallow_origin","deep_edge"], [xA,xB,xC,xD,xE]
    def run(self) -> Dict[str,float]:
        names, targets = self._targets(self.args.latent_dim)
        metrics = []
        for i in range(self.args.samples):
            nm = names[i % len(names)]; x_star = targets[i % len(targets)]
            x0 = x_star + self.rng.normal(scale=self.args.latent_arc_noise, size=self.args.latent_dim)
            den = TemporalDenoiser(self.args.denoise_mode, self.args.ema_decay, self.args.median_k)
            den.reset(); x_t = x0
            for t in range(self.args.T):
                dx_raw, conf_rel, logits = self.hooks.propose_step(x_t, x_star, self.args)
                residual = x_star - x_t; dx = dx_raw
                if self.args.log_snr:
                    _ = snr_db(signal=residual, noise=dx - residual)
                if conf_rel < self.args.conf_gate or np.linalg.norm(dx) < self.args.noise_floor:
                    dx = 0.5*residual
                if not phantom_guard(dx, x_t, lambda p: self.hooks.descend_vector(p, x_star, self.args), k=self.args.probe_k, eps=self.args.probe_eps):
                    dx = 0.3*residual
                x_next = den.latent(x_t + dx)
                if self.args.seed_jitter > 0:
                    xs=[x_next];
                    for _ in range(self.args.seed_jitter):
                        xs.append(den.latent(x_t + dx + np.random.normal(scale=0.01, size=x_t.shape)))
                    x_next = np.mean(xs,0)
                x_t = x_next
            m = self.hooks.score_sample(x_t, x_star); m["latent_arc"]=nm; metrics.append(m)
        keys = [k for k in metrics[0].keys() if k!="latent_arc"] if metrics else []
        agg = {k: float(np.mean([m[k] for m in metrics])) for k in keys}
        # breakdown
        by={}
        for m in metrics:
            by.setdefault(m["latent_arc"], []).append(m)
        agg["latent_arc_breakdown"] = {nm:{k: float(np.mean([x[k] for x in arr])) for k in keys} for nm,arr in by.items()}
        return agg

# ----------------------------
# Plotting
# ----------------------------

def plot_trisurf(X3: np.ndarray, energy: Optional[np.ndarray], P: WellParams, path: str, title: str):
    if plt is None: return
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure(figsize=(10,8)); ax = fig.add_subplot(111, projection='3d')
    x,y,z = X3[:,0], X3[:,1], X3[:,2]
    c = None if energy is None else (energy - np.min(energy)) / (np.ptp(energy) + 1e-9)
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(np.column_stack([x,y])); ax.plot_trisurf(x,y,z, triangles=tri.simplices, cmap='viridis', alpha=P.trisurf_alpha, linewidth=0.2, antialiased=True)
    except Exception:
        ax.plot_trisurf(x,y,z, cmap='viridis', alpha=P.trisurf_alpha, linewidth=0.2, antialiased=True)
    ax.scatter(x,y,z, c=c, cmap='viridis', s=12, alpha=P.point_alpha)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3'); ax.set_title(title)
    plt.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)

# ----------------------------
# IO helpers
# ----------------------------

def write_rows_csv(path: str, rows: List[Dict[str,object]]):
    if not rows: return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); [w.writerow(r) for r in rows]

def write_json(path: str, obj: Dict[str,object]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)

# ----------------------------
# CLI & main
# ----------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="Stage 11 — consolidated well benchmark (rewrite)")
    # data
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--T", type=int, default=720)
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--sigma", type=int, default=9)
    p.add_argument("--cm_amp", type=float, default=0.02)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--amp_jitter", type=float, default=0.4)
    p.add_argument("--distractor_prob", type=float, default=0.4)
    p.add_argument("--min_tasks", type=int, default=1)
    p.add_argument("--max_tasks", type=int, default=3)
    p.add_argument("--proto_width", type=int, default=160)
    # outputs
    p.add_argument("--out_plot", type=str, default="out/manifold_warped.png")
    p.add_argument("--out_plot_fit", type=str, default="out/manifold_warped_fit.png")
    p.add_argument("--out_csv", type=str, default="out/stage11_metrics.csv")
    p.add_argument("--out_json", type=str, default="out/stage11_summary.json")
    # warp viz params
    p.add_argument("--sigma_scale", type=float, default=0.80)
    p.add_argument("--depth_scale", type=float, default=1.35)
    p.add_argument("--mix_z", type=float, default=0.12)
    # funnel fit → priors
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
    # prior coupling
    p.add_argument("--use_funnel_prior", type=int, default=0, choices=[0,1])
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--beta_s", type=float, default=0.25)
    p.add_argument("--q_s", type=int, default=2)
    p.add_argument("--tau_rel", type=float, default=0.60)
    p.add_argument("--tau_abs_q", type=float, default=0.93)
    p.add_argument("--null_K", type=int, default=40)
    p.add_argument("--use_baseline_arc", type=int, default=1, choices=[0,1])
    # denoiser path
    p.add_argument("--denoise_mode", type=str, default="off", choices=["off","ema","median","hybrid"]) 
    p.add_argument("--ema_decay", type=float, default=0.85)
    p.add_argument("--median_k", type=int, default=3)
    p.add_argument("--probe_k", type=int, default=3)
    p.add_argument("--probe_eps", type=float, default=0.02)
    p.add_argument("--conf_gate", type=float, default=0.60)
    p.add_argument("--noise_floor", type=float, default=0.05)
    p.add_argument("--seed_jitter", type=int, default=0)
    p.add_argument("--log_snr", type=int, default=1)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--latent_arc", action="store_true")
    p.add_argument("--latent_arc_noise", type=float, default=0.05)
    # optional pre‑warp + passes for manifold stage (read from JSON)
    p.add_argument("--warp_config", type=str, default=None,
                   help="JSON with {prewarp:{enabled,mode,alpha,scale_pct,iters,decay}, passes:{count,alpha_schedule,scale_pct}}")
    # logging
    p.add_argument("--log", type=str, default="INFO")
    return p

# ----------------------------
# Main
# ----------------------------

def main():
    args = build_argparser().parse_args()
    lvl = getattr(pylog, args.log.upper(), pylog.INFO)
    pylog.basicConfig(level=lvl, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = pylog.getLogger("stage11")
    np.random.seed(args.seed); random.seed(args.seed)

    # ---- Report path (stock vs geodesic) ----
    rows = []
    agg_geo = dict(acc=0, P=0, R=0, F1=0, J=0, H=0, O=0)
    agg_stock = dict(acc=0, P=0, R=0, F1=0, J=0, H=0, O=0)

    rng = np.random.default_rng(args.seed)
    priors = None

    for i in range(1, args.samples+1):
        traces, true_order = make_synthetic_traces(
            rng, T=args.T, noise=args.noise, cm_amp=args.cm_amp,
            overlap=args.overlap, amp_jitter=args.amp_jitter, distractor_prob=args.distractor_prob,
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

        acc_g = int(order_g == true_order); acc_s = int(order_s == true_order)
        sm_g = set_metrics(true_order, keep_g); sm_s = set_metrics(true_order, keep_s)
        for k,v in sm_g.items():
            key = {"precision":"P","recall":"R","f1":"F1","jaccard":"J","hallucination_rate":"H","omission_rate":"O"}[k]
            agg_geo[key] = agg_geo.get(key,0) + v
        for k,v in sm_s.items():
            key = {"precision":"P","recall":"R","f1":"F1","jaccard":"J","hallucination_rate":"H","omission_rate":"O"}[k]
            agg_stock[key] = agg_stock.get(key,0) + v
        agg_geo["acc"] += acc_g; agg_stock["acc"] += acc_s

        rows.append(dict(
            sample=i,
            true="|".join(true_order),
            geodesic_tasks="|".join(keep_g), geodesic_order="|".join(order_g), geodesic_ok=acc_g,
            stock_tasks="|".join(keep_s),    stock_order="|".join(order_s),    stock_ok=acc_s,
            geodesic_precision=sm_g["precision"], geodesic_recall=sm_g["recall"], geodesic_f1=sm_g["f1"],
            geodesic_jaccard=sm_g["jaccard"], geodesic_hallucination=sm_g["hallucination_rate"], geodesic_omission=sm_g["omission_rate"],
            stock_precision=sm_s["precision"], stock_recall=sm_s["recall"], stock_f1=sm_s["f1"],
            stock_jaccard=sm_s["jaccard"], stock_hallucination=sm_s["hallucination_rate"], stock_omission=sm_s["omission_rate"],
        ))

    n = float(args.samples)
    Sg = dict(accuracy_exact=agg_geo["acc"]/n, precision=agg_geo["P"]/n, recall=agg_geo["R"]/n,
              f1=agg_geo["F1"]/n, jaccard=agg_geo["J"]/n, hallucination_rate=agg_geo["H"]/n, omission_rate=agg_geo["O"]/n)
    Ss = dict(accuracy_exact=agg_stock["acc"]/n, precision=agg_stock["P"]/n, recall=agg_stock["R"]/n,
              f1=agg_stock["F1"]/n, jaccard=agg_stock["J"]/n, hallucination_rate=agg_stock["H"]/n, omission_rate=agg_stock["O"]/n)

    # ---- Manifold view: PCA(3) → warp (with optional pre‑warp + passes) ----
    phantom_index = None; margin = None
    try:
        H, E = build_H_E_from_traces(args)
        P = WellParams(sigma_scale=args.sigma_scale, depth_scale=args.depth_scale, mix_z=args.mix_z)
        X3, info = None, None
        if args.warp_config and os.path.isfile(args.warp_config):
            cfg = json.load(open(args.warp_config))
            # Baseline PCA(3) warp
            X3_base, info = pca3_and_warp(H, E, P)
            X2 = X3_base[:,:2]
            # Pre‑warp (blind)
            pre = cfg.get("prewarp", {})
            if pre.get("enabled", False):
                mode = pre.get("mode","peak"); iters = max(1, int(pre.get("iters",2)))
                alpha0 = float(pre.get("alpha",0.06)); decay = float(pre.get("decay",0.75))
                pct = int(pre.get("scale_pct",60))
                for k in range(iters):
                    # center
                    U, Hs, xe, ye = _hist_energy(X2, nbins=120, sigma=5.0)
                    mins = _find_minima(U, Hs, density_floor=2.5, min_prom=0.38)
                    if mins:
                        mins.sort(key=lambda x: x[0]); i,j = mins[0][1], mins[0][2]
                        cx = 0.5*(xe[i]+xe[i+1]); cy = 0.5*(ye[j]+ye[j+1])
                    else:
                        cx, cy = np.median(X2[:,0]), np.median(X2[:,1])
                    v = X2 - np.array([cx,cy])[None,:]
                    r = np.linalg.norm(v, axis=1) + 1e-12
                    scale = max(np.percentile(r, pct), 1e-6)
                    alpha = alpha0 * (decay**k)
                    X2 = _radial_rewarp(X2, (cx,cy), alpha, scale)
                X3 = np.column_stack([X2, X3_base[:,2]])
            else:
                X3 = X3_base
            # Passes (anchored)
            passes = cfg.get("passes", {})
            Pn = int(passes.get("count",0))
            if Pn > 0:
                U, Hs, xe, ye = _hist_energy(X3[:,:2], nbins=120, sigma=5.0)
                mins = _find_minima(U, Hs, density_floor=2.5, min_prom=0.38)
                if mins:
                    mins.sort(key=lambda x: x[0]); ci,cj = mins[0][1], mins[0][2]
                    center = (0.5*(xe[ci]+xe[ci+1]), 0.5*(ye[cj]+ye[cj+1]))
                    X2 = X3[:,:2].copy()
                    al = list(passes.get("alpha_schedule", [0.10,0.06,0.03]))
                    sc = list(passes.get("scale_pct", [60,55,50]))
                    while len(al)<Pn: al.append(al[-1])
                    while len(sc)<Pn: sc.append(sc[-1])
                    for pidx in range(Pn):
                        v = X2 - np.array(center)[None,:]
                        r = np.linalg.norm(v, axis=1) + 1e-12
                        scale = max(np.percentile(r, int(sc[pidx])), 1e-6)
                        X2 = _radial_rewarp(X2, center, float(al[pidx]), scale)
                    X3 = np.column_stack([X2, X3[:,2]])
        else:
            X3, info = pca3_and_warp(H, E, P)
        # Phantom metrics (simple energy histogram proxy)
        U, _, _, _ = _hist_energy(X3[:,:2], nbins=120, sigma=5.0)
        pi, mg = _phantom_metrics(U); phantom_index, margin = float(pi), float(mg)
        # Render + fit radial profile
        if plt is not None:
            plot_trisurf(X3, E, P, args.out_plot, "Stage 11 — Warped Single Well")
            r_cloud = np.linalg.norm((X3[:,:2] - (info["center"] if info else np.array([0.0,0.0]))), axis=1) if info else np.linalg.norm(X3[:,:2],1)
            r_max = float(np.quantile(r_cloud, 0.98))
            r_grid = np.linspace(0.0, r_max, args.n_r)
            h = max(1e-6, args.rbf_bw * r_max)
            z_prof = fit_radial_profile(X3, (info["center"] if info else np.array([0.0,0.0])), r_grid,
                                        h=h, q=args.fit_quantile, r0_frac=args.core_r0_frac,
                                        core_k=args.core_k, core_p=args.core_p)
            # surface (for prior derivation only)
            theta = np.linspace(0, 2*np.pi, args.n_theta)
            R, TH = np.meshgrid(r_grid, theta)
            Xs = (info["center"][0] if info else 0.0) + R*np.cos(TH)
            Ys = (info["center"][1] if info else 0.0) + R*np.sin(TH)
            Zs = np.tile(z_prof, (args.n_theta,1))
            fig = plt.figure(figsize=(10,8)); ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(Xs, Ys, Zs, cmap='viridis', alpha=0.9, linewidth=0, antialiased=True)
            ax.scatter(X3[:,0], X3[:,1], X3[:,2], s=10, alpha=0.6, c=(E - E.min())/(E.ptp()+1e-9), cmap='viridis')
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
            ax.set_title("Stage 11 — Data‑fit Funnel")
            fig.colorbar(surf, ax=ax, shrink=0.6, label="height"); plt.tight_layout(); fig.savefig(args.out_plot_fit, dpi=220); plt.close(fig)
            priors = priors_from_profile(r_grid, z_prof)
    except Exception as e:
        logger.warning(f"Manifold stage failed: {e}")

    # ---- Write report outputs ----
    if args.out_csv: write_rows_csv(args.out_csv, rows)
    summary = dict(samples=int(n), geodesic=Sg, stock=Ss,
                   phantom_index=phantom_index, margin=margin,
                   plot_raw=args.out_plot, plot_fitted=args.out_plot_fit, csv=args.out_csv)
    if args.out_json: write_json(args.out_json, summary)

    # ---- Denoiser path (optional) ----
    if args.denoise_mode != "off":
        runner = DenoiseRunner(args)
        D = runner.run()
        if args.out_json:
            try:
                S = json.load(open(args.out_json)); S["denoise"] = D; write_json(args.out_json, S)
            except Exception:
                write_json(args.out_json, {"denoise": D})

    print("[SUMMARY] Geodesic:", {k: round(v,3) for k,v in Sg.items()})
    print("[SUMMARY] Stock   :", {k: round(v,3) for k,v in Ss.items()})
    if phantom_index is not None:
        print("[WELL] phantom_index:", round(phantom_index,4), "margin:", round(margin or 0.0,4))
    print(f"[PLOT] RAW:    {args.out_plot}")
    print(f"[PLOT] FITTED: {args.out_plot_fit}")
    print(f"[CSV ] {args.out_csv}")
    print(f"[JSON] {args.out_json}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback, sys
        print("[ERROR] Uncaught exception:", e)
        traceback.print_exc(); sys.exit(1)
