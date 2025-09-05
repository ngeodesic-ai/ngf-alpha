#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-11 — LLM Shadow Probe (Hijack → Localized Warp → Full Denoise-Ready)
v3 adds:
 • Depth-weighted + NMS-inhibited phantom metric (so hijack actually reduces PI)
 • Localized warp (fading with radius) around detected dominant basin
 • Smoothed inward-trend + control guards: EMA + Median + Confidence Gate + Phantom-Guard + (light) Jitter
 • SNR proxy logging for stabilization
Usage (example):
  python3 stage11_llm_shadow-hijack-v3.py \
    --model gpt2 --tap -3 \
    --calib ngf_calib_prompts_360.txt \
    --eval  ngf_eval_prompts_60.txt \
    --render \
    --use_depth_weighted_pi 1 --pi_beta 3.0 --nms_radius 5 \
    --ema_gamma 0.8 --med_k 5 --tau_conf 0.6 --eta 0.15 --jitter_sigma 0.03 --jitter_J 8 \
    --out_json llm_shadow_hijack_summary.json
"""
from __future__ import annotations
import argparse, json, math, random
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Callable, List

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter

# ------------------------------
# IO helpers
# ------------------------------

def read_lines(path: str):
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]

# ------------------------------
# Model hooks (CPU)
# ------------------------------

def load_model(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval(); model.to("cpu")
    return tok, model

@torch.no_grad()
def collect_hidden_states(model, tok, prompts, tap: int) -> np.ndarray:
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
    out = model(**enc, output_hidden_states=True)
    hs  = out.hidden_states[tap]            # (batch, seq, d)
    H   = hs.mean(1).cpu().numpy().astype(float)  # (batch, d)
    return H

# ------------------------------
# PCA rendering helpers
# ------------------------------

def render_Y3(Y3, title, out):
    fig = plt.figure(figsize=(7.5,6.2)); ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Y3[:,0], Y3[:,1], Y3[:,2], s=12, alpha=0.7)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3'); ax.set_title(title)
    plt.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)

# ------------------------------
# Phantom metrics (with depth weighting + non-max suppression option)
# ------------------------------

def _local_minima(U: np.ndarray) -> List[Tuple[float, Tuple[int,int]]]:
    """Return all strict 8-neighborhood local minima (value, (i,j))."""
    h, w = U.shape
    mins = []
    for i in range(1, h-1):
        for j in range(1, w-1):
            c = U[i, j]
            neigh = U[i-1:i+2, j-1:j+2].copy()
            neigh[1,1] = c + 1e9
            if (c < neigh).all():
                mins.append((float(c), (i,j)))
    mins.sort(key=lambda t: t[0])  # ascending by value (deeper first since U is "energy")
    return mins


def _nms_minima(minima: List[Tuple[float, Tuple[int,int]]], radius: int, max_keep: int = 32) -> List[Tuple[float, Tuple[int,int]]]:
    """Simple NMS over minima with integer radius in pixel space."""
    kept: List[Tuple[float, Tuple[int,int]]] = []
    def too_close(p, q):
        return (p[0]-q[0])**2 + (p[1]-q[1])**2 <= radius*radius
    for val, ij in minima:
        ok = True
        for _, k_ij in kept:
            if too_close(ij, k_ij):
                ok = False; break
        if ok:
            kept.append((val, ij))
            if len(kept) >= max_keep:
                break
    return kept


def phantom_metrics_Y3(Y3: np.ndarray, nbins=120, sigma=2.0, depth_weight_beta: float = 0.0, nms_radius: int = 0):
    """Compute phantom_index and normalized margin using a smoothed 2D histogram on (PC1,PC2).
       Options:
         - depth_weight_beta > 0 applies weights w = exp(-beta * z_norm) to favor deeper points
         - nms_radius > 0 performs NMS on minima to remove clustered phantoms
    """
    X2 = Y3[:, :2]; z = Y3[:, 2]
    x, y = X2[:,0], X2[:,1]
    weights = None
    if depth_weight_beta and depth_weight_beta > 0:
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-9)
        weights = np.exp(-depth_weight_beta * z_norm)
    H, xe, ye = np.histogram2d(x, y, bins=nbins, weights=weights)
    Hs = gaussian_filter(H, sigma=sigma)
    U = -Hs
    all_mins = _local_minima(U)
    mins = _nms_minima(all_mins, radius=nms_radius) if nms_radius and nms_radius > 0 else all_mins
    n = len(mins)
    span = float(U.max() - U.min() + 1e-9)
    if n <= 1:
        pi = 0.0; margin_norm = 0.0
        z0 = z1 = float(mins[0][0]) if n == 1 else float(U.min())
    else:
        z0 = float(mins[0][0]); z1 = float(mins[1][0])
        margin_norm = float((z1 - z0) / span)
        pi = float((n - 1) / n)
    dbg = {
        "mins": [m[0] for m in mins[:10]],
        "coords": [list(m[1]) for m in mins[:10]],
        "U_rng": span,
        "n_all": int(len(all_mins)),
        "n_after_nms": int(n)
    }
    return pi, margin_norm, dbg

# ------------------------------
# Hijack warp: localized funnel around detected basin
# ------------------------------
@dataclass
class WarpParams:
    sigma_scale: float = 0.80
    depth_scale: float = 1.35
    mix_z: float = 0.12
    local_radius: float = 1.40  # radius (in sigma units) where warp remains strong


def detect_basin_center(Y3: np.ndarray, nbins=120, sigma=2.0) -> Tuple[np.ndarray, float]:
    """Locate deepest basin center in X2 via smoothed 2D histogram (density)."""
    X2 = Y3[:, :2]
    x, y = X2[:,0], X2[:,1]
    H, xe, ye = np.histogram2d(x, y, bins=nbins)
    Hs = gaussian_filter(H, sigma=sigma)
    U = -Hs
    idx = np.unravel_index(np.argmin(U), U.shape)
    cx = 0.5 * (xe[idx[0]] + xe[idx[0]+1])
    cy = 0.5 * (ye[idx[1]] + ye[idx[1]+1])
    c = np.array([cx, cy], float)
    r = np.linalg.norm(X2 - c[None,:], axis=1)
    sigma_r = float(np.median(r) + 1e-9)
    return c, sigma_r


def build_warp_localized(Y3: np.ndarray, params: WarpParams):
    """Build localized warp function: z_out = depth_scale * funnel(r) + mix_z * z' (faded by radius)."""
    X2 = Y3[:, :2]; z = Y3[:, 2].copy()
    c, sigma_r = detect_basin_center(Y3)
    # isotropy via PCA whitening on local plane around center
    X2c = X2 - c
    C = (X2c.T @ X2c) / max(len(X2c)-1, 1)
    evals, evecs = np.linalg.eigh(C)
    T = evecs @ np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-8))) @ evecs.T

    r = np.linalg.norm(X2c @ T, axis=1) + 1e-9
    sigma = params.sigma_scale * np.median(r) + 1e-9

    def warp_fn(Y3_in: np.ndarray) -> np.ndarray:
        X2_in = Y3_in[:, :2]; z_in = Y3_in[:, 2].copy()
        Xw = (X2_in - c) @ T
        r_in = np.linalg.norm(Xw, axis=1) + 1e-9
        u_in = Xw / r_in[:, None]
        zf = -np.exp(-(r_in**2) / (2 * sigma**2))
        alpha = np.exp(- (r_in / (params.local_radius * sigma + 1e-9))**2)
        z_out = alpha * (params.depth_scale * zf + params.mix_z * (z_in - z_in.mean())) + (1 - alpha) * (z_in - z_in.mean())
        # back to original coords
        X2_out = r_in[:, None] * u_in @ np.linalg.inv(T) + c
        return np.column_stack([X2_out, z_out])
    info = dict(center=c.tolist(), sigma=float(sigma), sigma_r=float(sigma_r), local_radius=float(params.local_radius))
    return warp_fn, info

# ------------------------------
# Denoiser (EMA + Median + Confidence Gate + Phantom-Guard + Jitter) for token trajectories
# ------------------------------
@dataclass
class DenoiseParams:
    ema_gamma: float = 0.80
    med_k: int = 5
    tau_conf: float = 0.60
    eta: float = 0.15
    jitter_sigma: float = 0.03
    jitter_J: int = 8


def _median_filter_1d(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k) | 1)  # odd
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.array([np.median(xp[i:i+k]) for i in range(len(x))])


def denoise_token_path(X2_seq: np.ndarray, center: np.ndarray, dn: DenoiseParams):
    """Return denoised 2D path, r_trend, and SNR proxy stats."""
    T = X2_seq.shape[0]
    r_seq = np.linalg.norm(X2_seq - center[None,:], axis=1) + 1e-9
    # Normalize radii to [0,1] for stability
    Rn = (r_seq - r_seq.min()) / (r_seq.max() - r_seq.min() + 1e-9)

    # EMA on radius
    ema = np.zeros_like(Rn); acc = 0.0
    for i, v in enumerate(Rn):
        acc = dn.ema_gamma * acc + (1 - dn.ema_gamma) * v
        ema[i] = acc
    med = _median_filter_1d(ema, dn.med_k)

    # Confidence gate: fraction of last W diffs that are negative (inward); if low, pull toward center
    W = max(3, dn.med_k)
    Xhat = X2_seq.copy()
    diffs = np.diff(med, prepend=med[0])
    neg_frac = np.zeros(T)
    for t in range(T):
        a = max(0, t-W+1); b = t+1
        window = diffs[a:b]
        nf = float((window < 0).sum()) / max(1, len(window))
        neg_frac[t] = nf
        if nf < dn.tau_conf:
            Xhat[t] = Xhat[t] + dn.eta * (center - Xhat[t])

        # Phantom-guard: probe J jitters and ensure the gradient toward center aligns
        if dn.jitter_J > 0:
            good = 0
            for _ in range(dn.jitter_J):
                eps = np.random.normal(scale=dn.jitter_sigma, size=2)
                xp = Xhat[t] + eps
                gp = center - xp
                d_hat = (center - Xhat[t])
                if np.linalg.norm(gp) > 0 and np.linalg.norm(d_hat) > 0:
                    if (gp @ d_hat) / (np.linalg.norm(gp) * np.linalg.norm(d_hat)) > 0:
                        good += 1
            if good < (dn.jitter_J // 2):
                # fallback small descent
                Xhat[t] = Xhat[t] + 0.5 * dn.eta * (center - Xhat[t])

    # SNR proxy over steps (using Xhat)
    snrs = []
    for t in range(1, T):
        x = Xhat[t]; x_prev = Xhat[t-1]
        dx = x - x_prev
        delta = center - x_prev
        snr = np.linalg.norm(delta) / (np.linalg.norm(dx - delta) + 1e-9)
        snrs.append(20.0 * np.log10(snr + 1e-12))
    snr_med = float(np.median(snrs)) if snrs else 0.0
    snr_fin = float(snrs[-1]) if snrs else 0.0

    # r_trend from denoised radii
    r_hat = np.linalg.norm(Xhat - center[None,:], axis=1) + 1e-9
    rhat_n = (r_hat - r_hat.min()) / (r_hat.max() - r_hat.min() + 1e-9)
    r_diffs = np.diff(rhat_n)
    r_trend = float((r_diffs < 0).sum() / max(1, len(r_diffs)))

    return Xhat, r_trend, dict(snr_median=snr_med, snr_final=snr_fin, neg_frac_mean=float(neg_frac.mean()))


@torch.no_grad()
def token_path_stats(model, tok, prompt: str, tap: int, pca: PCA, warp2d: Optional[Callable], center2d: np.ndarray, dn: DenoiseParams):
    enc = tok(prompt, return_tensors="pt")
    out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states[tap][0].cpu().numpy()   # [T, d]
    Y  = pca.transform(hs)                         # [T, 3]
    if warp2d is not None:
        Y = warp2d(Y)
    X2 = Y[:, :2]
    Xhat, r_trend, snr = denoise_token_path(X2, center2d, dn)
    return r_trend, snr

# ------------------------------
# Gate thresholds
# ------------------------------
@dataclass
class GateParams:
    pi_max: float = 0.10
    margin_min: float = 0.04
    S_median_min: float = 0.55
    r_trend_min: float = 0.90

# ------------------------------
# Priors for simple S_median proxy (same as v2)
# ------------------------------

def S_median_proxy(Ye: np.ndarray) -> float:
    Rc = np.linalg.norm(Ye[:, :2], axis=1)
    Rn = (Rc - Rc.min()) / (Rc.max() - Rc.min() + 1e-9)
    r_grid = np.linspace(0, 1, 128)
    phi = 1.0 - np.power(r_grid, 1.3)
    g = np.gradient(phi, r_grid); g = np.abs(g)
    g = (g - g.min()) / (g.max() - g.min() + 1e-9)
    phi_e = np.interp(Rn, r_grid, phi)
    g_e = np.interp(Rn, r_grid, g)
    S = 0.05 * phi_e + 0.25 * (g_e ** 2)
    return float(np.median(S))

# ------------------------------
# Main
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage-11 LLM Shadow (hijack→localized warp→controls)")
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--tap",   type=int, default=-3, help="hidden layer index (negative from end)")
    ap.add_argument("--calib", type=str, required=True)
    ap.add_argument("--eval",  type=str, required=True)
    ap.add_argument("--out_json", type=str, default="llm_shadow_hijack_summary.json")
    ap.add_argument("--render", action="store_true")

    # Phantom metric knobs
    ap.add_argument("--use_depth_weighted_pi", type=int, default=1)
    ap.add_argument("--pi_beta", type=float, default=3.0)
    ap.add_argument("--nms_radius", type=int, default=5)

    # Warp knobs
    ap.add_argument("--sigma_scale", type=float, default=0.80)
    ap.add_argument("--depth_scale", type=float, default=1.35)
    ap.add_argument("--mix_z", type=float, default=0.12)
    ap.add_argument("--local_radius", type=float, default=1.40)

    # Denoiser knobs
    ap.add_argument("--ema_gamma", type=float, default=0.80)
    ap.add_argument("--med_k", type=int, default=5)
    ap.add_argument("--tau_conf", type=float, default=0.60)
    ap.add_argument("--eta", type=float, default=0.15)
    ap.add_argument("--jitter_sigma", type=float, default=0.03)
    ap.add_argument("--jitter_J", type=int, default=8)

    # Gates
    ap.add_argument("--pi_max", type=float, default=0.10)
    ap.add_argument("--margin_min", type=float, default=0.04)
    ap.add_argument("--S_median_min", type=float, default=0.55)
    ap.add_argument("--r_trend_min", type=float, default=0.90)

    args = ap.parse_args()

    gates = GateParams(args.pi_max, args.margin_min, args.S_median_min, args.r_trend_min)
    warp_params = WarpParams(args.sigma_scale, args.depth_scale, args.mix_z, args.local_radius)
    dn = DenoiseParams(args.ema_gamma, args.med_k, args.tau_conf, args.eta, args.jitter_sigma, args.jitter_J)

    tok, model = load_model(args.model)

    # 1) Calibration → PCA(3)
    calib_prompts = read_lines(args.calib); eval_prompts  = read_lines(args.eval)
    Hc = collect_hidden_states(model, tok, calib_prompts, args.tap)
    pca = PCA(n_components=3, whiten=True, random_state=0)
    Yc  = pca.fit_transform(Hc)

    # 2) Eval projection
    He = collect_hidden_states(model, tok, eval_prompts, args.tap)
    Ye = pca.transform(He)

    # 3) Pre-warp metrics
    pi0_d, m0_d, dbg0 = phantom_metrics_Y3(Ye, nbins=120, sigma=2.0, depth_weight_beta=0.0, nms_radius=args.nms_radius)
    if args.use_depth_weighted_pi:
        pi0_w, m0_w, _ = phantom_metrics_Y3(Ye, nbins=120, sigma=2.0, depth_weight_beta=args.pi_beta, nms_radius=args.nms_radius)
        pi0, m0 = pi0_w, m0_w
    else:
        pi0, m0 = pi0_d, m0_d
    S0 = S_median_proxy(Ye)

    # r_trend pre (smoothed, no denoise controls)
    rtr0_list = []
    for p in eval_prompts:
        rtr, _ = token_path_stats(model, tok, p, args.tap, pca, warp2d=None, center2d=np.array([0.0,0.0]), dn=dn)
        rtr0_list.append(rtr)
    rtr0 = float(np.mean(rtr0_list)) if rtr0_list else 0.0

    pre = dict(phantom_index=pi0, margin_norm=m0, S_median=S0, r_trend_tokens=rtr0,
               phantom_index_density=pi0_d, margin_norm_density=m0_d, debug=dbg0)

    # 4) Build hijack warp (localized) on eval cloud & recompute metrics
    warp_fn, winfo = build_warp_localized(Ye, warp_params)
    Ye_w = warp_fn(Ye)

    pi1_d, m1_d, dbg1 = phantom_metrics_Y3(Ye_w, nbins=120, sigma=2.0, depth_weight_beta=0.0, nms_radius=args.nms_radius)
    if args.use_depth_weighted_pi:
        pi1_w, m1_w, _ = phantom_metrics_Y3(Ye_w, nbins=120, sigma=2.0, depth_weight_beta=args.pi_beta, nms_radius=args.nms_radius)
        pi1, m1 = pi1_w, m1_w
    else:
        pi1, m1 = pi1_d, m1_d
    S1 = S_median_proxy(Ye_w)

    # r_trend post (with warp + full denoise controls)
    rtr1_list, snr_list = [], []
    c2 = np.array(winfo["center"], dtype=float)
    for p in eval_prompts:
        rtr, snr = token_path_stats(model, tok, p, args.tap, pca, warp2d=warp_fn, center2d=c2, dn=dn)
        rtr1_list.append(rtr); snr_list.append(snr["snr_median"])
    rtr1 = float(np.mean(rtr1_list)) if rtr1_list else 0.0
    snr_med = float(np.median(snr_list)) if snr_list else 0.0

    # 5) Improvements + Gates
    improve = dict(
        d_phantom_index = float(pi1 - pi0),
        d_margin_norm   = float(m1 - m0),
        d_S_median      = float(S1 - S0),
        d_r_trend_tokens= float(rtr1 - rtr0),
        snr_median_post = snr_med
    )
    go0 = (pi0 <= gates.pi_max) and (m0 >= gates.margin_min) and (S0 >= gates.S_median_min) and (rtr0 >= gates.r_trend_min)
    go1 = (pi1 <= gates.pi_max) and (m1 >= max(gates.margin_min, m0)) and (S1 >= max(gates.S_median_min, S0)) and (rtr1 >= max(gates.r_trend_min, rtr0))

    post = dict(phantom_index=pi1, margin_norm=m1, S_median=S1, r_trend_tokens=rtr1,
                phantom_index_density=pi1_d, margin_norm_density=m1_d, debug=dbg1)

    summary = dict(
        model=args.model, tap=args.tap,
        gates={ "pi_max":gates.pi_max, "margin_min":gates.margin_min, "S_median_min":gates.S_median_min, "r_trend_min":gates.r_trend_min },
        metric_mode=("depth_weighted" if args.use_depth_weighted_pi else "density_only"), pi_beta=(args.pi_beta if args.use_depth_weighted_pi else 0.0),
        nms_radius=args.nms_radius,
        warp_info=winfo,
        denoiser={ "ema_gamma": args.ema_gamma, "med_k": args.med_k, "tau_conf": args.tau_conf, "eta": args.eta, "jitter_sigma": args.jitter_sigma, "jitter_J": args.jitter_J },
        pre=pre, post=post, improve=improve, go_pre=bool(go0), go_post=bool(go1)
    )

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("[SUMMARY]", json.dumps(summary, indent=2))

    if args.render:
        render_Y3(Ye,   "LLM PCA(3) — pre-warp",  "llm_pca3_eval_pre.png")
        render_Y3(Ye_w, "LLM PCA(3) — post-warp (hijacked, localized)", "llm_pca3_eval_post.png")

if __name__ == "__main__":
    main()
