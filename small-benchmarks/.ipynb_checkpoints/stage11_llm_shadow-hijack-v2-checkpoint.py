#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-11 — LLM Shadow Probe (v4)
Detect → Hijack → Localized Warp → Stepwise Denoise (with Phantom-Guard) → Measure

What’s new vs v3
- Stepwise descent loop on the eval cloud in (PC1, PC2), with denoiser + gates applied EACH STEP
- Phantom-guard uses ∇U from a depth-weighted potential to avoid slipping into side wells
- Inline SNR/PI/Margin curves logged across steps so you can see convergence
- Final token r_trend computed with the same denoiser controls (as v3), centered at the hijacked basin

Usage (example):
  python3 stage11_llm_shadow-hijack-v4.py \
    --model gpt2 --tap -3 \
    --calib ngf_calib_prompts_360.txt \
    --eval  ngf_eval_prompts_60.txt \
    --render \
    --steps 24 --eta 0.20 \
    --use_depth_weighted_pi 1 --pi_beta 3.0 --nms_radius 5 \
    --ema_gamma 0.8 --med_k 5 --tau_conf 0.6 \
    --jitter_sigma 0.03 --jitter_J 8 --backoff 0.5 \
    --out_json llm_shadow_hijack_summary.json
"""
from __future__ import annotations
import argparse, json, math
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
# Phantom metrics (depth-weighted + NMS minima)
# ------------------------------

def _local_minima(U: np.ndarray) -> List[Tuple[float, Tuple[int,int]]]:
    h, w = U.shape
    mins = []
    for i in range(1, h-1):
        for j in range(1, w-1):
            c = U[i, j]
            neigh = U[i-1:i+2, j-1:j+2].copy()
            neigh[1,1] = c + 1e9
            if (c < neigh).all():
                mins.append((float(c), (i,j)))
    mins.sort(key=lambda t: t[0])
    return mins


def _nms_minima(minima: List[Tuple[float, Tuple[int,int]]], radius: int, max_keep: int = 64) -> List[Tuple[float, Tuple[int,int]]]:
    kept: List[Tuple[float, Tuple[int,int]]] = []
    def too_close(p, q):
        return (p[0]-q[0])**2 + (p[1]-q[1])**2 <= radius*radius
    for val, ij in minima:
        if all(not too_close(ij, k_ij) for _, k_ij in kept):
            kept.append((val, ij))
            if len(kept) >= max_keep:
                break
    return kept


def _potential_and_edges(X2: np.ndarray, z: np.ndarray, nbins=120, sigma=2.0, beta: float = 0.0):
    x, y = X2[:,0], X2[:,1]
    weights = None
    if beta and beta > 0:
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-9)
        weights = np.exp(-beta * z_norm)
    H, xe, ye = np.histogram2d(x, y, bins=nbins, weights=weights)
    Hs = gaussian_filter(H, sigma=sigma)
    U = -Hs
    return U, xe, ye


def _interp_grad_U(X2: np.ndarray, U: np.ndarray, xe: np.ndarray, ye: np.ndarray) -> np.ndarray:
    # average bin widths (edges -> nearly uniform grid)
    dx = float(np.mean(np.diff(xe)))
    dy = float(np.mean(np.diff(ye)))

    # gradient: gx = dU/dx, gy = dU/dy on the grid
    gx, gy = np.gradient(U, dx, dy, edge_order=2)

    # map points to nearest cell indices (keep away from borders for safety)
    xi = np.clip(np.searchsorted(xe, X2[:, 0]) - 1, 1, U.shape[0] - 2)
    yi = np.clip(np.searchsorted(ye, X2[:, 1]) - 1, 1, U.shape[1] - 2)

    # descend along -∇U and normalize
    g = np.column_stack([-gx[xi, yi], -gy[xi, yi]])
    n = np.linalg.norm(g, axis=1, keepdims=True) + 1e-9
    return g / n



def phantom_metrics_Y3(Y3: np.ndarray, nbins=120, sigma=2.0, beta: float = 0.0, nms_radius: int = 0):
    X2 = Y3[:, :2]; z = Y3[:, 2]
    U, xe, ye = _potential_and_edges(X2, z, nbins=nbins, sigma=sigma, beta=beta)
    mins_all = _local_minima(U)
    mins = _nms_minima(mins_all, radius=nms_radius) if nms_radius and nms_radius>0 else mins_all
    n = len(mins)
    span = float(U.max() - U.min() + 1e-9)
    if n <= 1:
        pi = 0.0; margin_norm = 0.0
    else:
        z0 = float(mins[0][0]); z1 = float(mins[1][0])
        margin_norm = float((z1 - z0) / span)
        pi = float((n - 1) / n)
    dbg = {"n_all": int(len(mins_all)), "n_after_nms": int(n), "U_rng": span}
    return pi, margin_norm, dbg, (U, xe, ye)

# ------------------------------
# Warp: localized funnel shaping of z (center stays fixed)
# ------------------------------
@dataclass
class WarpParams:
    sigma: float
    depth_scale: float = 1.35
    mix_z: float = 0.12
    local_radius: float = 1.40


def detect_center_density(Y3: np.ndarray, nbins=120, sigma=2.0) -> np.ndarray:
    X2 = Y3[:, :2]
    H, xe, ye = np.histogram2d(X2[:,0], X2[:,1], bins=nbins)
    Hs = gaussian_filter(H, sigma=sigma)
    U = -Hs
    idx = np.unravel_index(np.argmin(U), U.shape)
    cx = 0.5 * (xe[idx[0]] + xe[idx[0]+1])
    cy = 0.5 * (ye[idx[1]] + ye[idx[1]+1])
    return np.array([cx, cy], float)


def make_warp_params(Y3: np.ndarray, center: np.ndarray, sigma_scale=0.80, local_radius=1.40) -> WarpParams:
    r = np.linalg.norm(Y3[:, :2] - center[None,:], axis=1) + 1e-9
    sigma = float(np.median(r) * sigma_scale + 1e-9)
    return WarpParams(sigma=sigma, depth_scale=1.35, mix_z=0.12, local_radius=local_radius)


def apply_funnel_z(X2: np.ndarray, z_base: np.ndarray, center: np.ndarray, wp: WarpParams) -> np.ndarray:
    r = np.linalg.norm(X2 - center[None,:], axis=1) + 1e-9
    zf = -np.exp(-(r**2) / (2 * wp.sigma**2))
    alpha = np.exp(- (r / (wp.local_radius * wp.sigma + 1e-9))**2)
    z_centered = (z_base - z_base.mean())
    z_out = alpha * (wp.depth_scale * zf + wp.mix_z * z_centered) + (1 - alpha) * z_centered
    return z_out

# ------------------------------
# Denoiser controls for stepwise descent (EMA + median + confidence + jitter + phantom-guard)
# ------------------------------
@dataclass
class DenoiseParams:
    ema_gamma: float = 0.80
    med_k: int = 5
    tau_conf: float = 0.60
    jitter_sigma: float = 0.03
    jitter_J: int = 8
    backoff: float = 0.5


def stepwise_descent(Ye: np.ndarray, center: np.ndarray, wp: WarpParams,
                     steps: int = 24, eta: float = 0.20,
                     beta_pi: float = 3.0, nms_radius: int = 5,
                     dn: DenoiseParams = DenoiseParams()):
    """Iteratively pull X2 toward center with denoiser controls; recompute z via funnel each step."""
    X2 = Ye[:, :2].copy()
    z0 = Ye[:, 2].copy()
    N = len(X2)

    # histories
    R_hist = []  # mean radius per step
    PI_hist, M_hist, SNR_hist = [], [], []

    # initialize EMA of radius per-point
    r_prev = np.linalg.norm(X2 - center[None,:], axis=1) + 1e-9
    ema_r = r_prev.copy()
    med_buf_len = max(1, dn.med_k)
    med_buf = [ema_r.copy()]

    # build initial Y for metrics
    Y = np.column_stack([X2, apply_funnel_z(X2, z0, center, wp)])
    pi, m, _, pot = phantom_metrics_Y3(Y, beta=beta_pi, nms_radius=nms_radius)
    PI_hist.append(float(pi)); M_hist.append(float(m))
    R_hist.append(float(r_prev.mean()))
    SNR_hist.append(0.0)

    for t in range(steps):
        # Potential & gradient (depth-weighted)
        U, xe, ye = pot
        g_dir = _interp_grad_U(X2, U, xe, ye)  # unit vectors toward deeper energy

        # nominal step toward center
        v_center = (center[None,:] - X2)  # N x 2
        v = eta * v_center

        # phantom-guard: align with −∇U
        dot = np.sum(v * g_dir, axis=1)  # alignment
        misaligned = dot < 0.0
        if np.any(misaligned):
            v[misaligned] = v[misaligned] * dn.backoff + 0.3 * (g_dir[misaligned]) * (np.linalg.norm(v[misaligned], axis=1)[:,None] + 1e-9)

        # jitter averaging
        if dn.jitter_J > 0 and dn.jitter_sigma > 0:
            acc = v.copy()
            for _ in range(dn.jitter_J):
                acc += v + np.random.normal(scale=dn.jitter_sigma, size=v.shape)
            v = acc / float(dn.jitter_J + 1)

        # apply step
        X2_prop = X2 + v

        # confidence gate via EMA+median on radius
        r_new = np.linalg.norm(X2_prop - center[None,:], axis=1) + 1e-9
        ema_r = dn.ema_gamma * ema_r + (1 - dn.ema_gamma) * r_new
        med_buf.append(ema_r.copy())
        if len(med_buf) > med_buf_len: med_buf.pop(0)
        med_r = np.median(np.stack(med_buf, axis=0), axis=0)
        going_inward = (med_r - r_prev) < 0.0
        conf = np.mean(going_inward.astype(float))

        # adapt: if low confidence, back off the step globally
        if conf < dn.tau_conf:
            X2_prop = X2 + dn.backoff * v
            r_new = np.linalg.norm(X2_prop - center[None,:], axis=1) + 1e-9
            ema_r = dn.ema_gamma * ema_r + (1 - dn.ema_gamma) * r_new
            med_buf[-1] = ema_r.copy()

        # commit
        X2 = X2_prop
        r_prev = r_new

        # recompute z via funnel and metrics
        z = apply_funnel_z(X2, z0, center, wp)
        Y = np.column_stack([X2, z])
        pi, m, _, pot = phantom_metrics_Y3(Y, beta=beta_pi, nms_radius=nms_radius)
        PI_hist.append(float(pi)); M_hist.append(float(m))
        R_hist.append(float(r_prev.mean()))

        # SNR proxy: how much of v matched ideal centerward delta
        num = np.linalg.norm(v_center, axis=1) + 1e-9
        den = np.linalg.norm(v_center - v, axis=1) + 1e-9
        snr = np.median(20.0 * np.log10(num / den))
        SNR_hist.append(float(snr))

    return Y, dict(pi=PI_hist, margin=M_hist, R=R_hist, snr=SNR_hist)

# ------------------------------
# Token-path denoising (reused from v3 for final r_trend)
# ------------------------------
@dataclass
class TokenDenoiseParams:
    ema_gamma: float = 0.80
    med_k: int = 5
    tau_conf: float = 0.60
    eta: float = 0.15
    jitter_sigma: float = 0.03
    jitter_J: int = 8


def _median_filter_1d(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k) | 1)
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.array([np.median(xp[i:i+k]) for i in range(len(x))])


def denoise_token_path(X2_seq: np.ndarray, center: np.ndarray, dn: TokenDenoiseParams):
    T = X2_seq.shape[0]
    r_seq = np.linalg.norm(X2_seq - center[None,:], axis=1) + 1e-9
    Rn = (r_seq - r_seq.min()) / (r_seq.max() - r_seq.min() + 1e-9)

    ema = np.zeros_like(Rn); acc = 0.0
    for i, v in enumerate(Rn):
        acc = dn.ema_gamma * acc + (1 - dn.ema_gamma) * v
        ema[i] = acc
    med = _median_filter_1d(ema, dn.med_k)

    W = max(3, dn.med_k)
    Xhat = X2_seq.copy()
    diffs = np.diff(med, prepend=med[0])
    for t in range(T):
        a = max(0, t-W+1); b = t+1
        window = diffs[a:b]
        nf = float((window < 0).sum()) / max(1, len(window))
        if nf < dn.tau_conf:
            Xhat[t] = Xhat[t] + dn.eta * (center - Xhat[t])
        # light jitter average
        if dn.jitter_J > 0:
            outs = [Xhat[t] + np.random.normal(scale=dn.jitter_sigma, size=2) for _ in range(dn.jitter_J)]
            outs.append(Xhat[t])
            Xhat[t] = np.mean(outs, axis=0)

    r_hat = np.linalg.norm(Xhat - center[None,:], axis=1) + 1e-9
    rhat_n = (r_hat - r_hat.min()) / (r_hat.max() - r_hat.min() + 1e-9)
    r_diffs = np.diff(rhat_n)
    r_trend = float((r_diffs < 0).sum() / max(1, len(r_diffs)))
    return r_trend


@torch.no_grad()
def token_rtrend_for_prompts(model, tok, prompts, tap: int, pca: PCA, center2d: np.ndarray, dn: TokenDenoiseParams):
    vals = []
    for p in prompts:
        enc = tok(p, return_tensors="pt")
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[tap][0].cpu().numpy()   # [T, d]
        Y  = pca.transform(hs)                         # [T, 3]
        X2 = Y[:, :2]
        vals.append(denoise_token_path(X2, center2d, dn))
    return float(np.mean(vals)) if vals else 0.0

# ------------------------------
# Rendering
# ------------------------------

def render_Y3(Y3, title, out):
    fig = plt.figure(figsize=(7.5,6.2))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Y3[:,0], Y3[:,1], Y3[:,2], s=12, alpha=0.7)
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.set_zlabel('PC3')
    ax.set_title(title)
    plt.tight_layout(); fig.savefig(out, dpi=180); plt.close(fig)


def render_curves(hist: Dict[str, List[float]], out_prefix: str):
    plt.figure(figsize=(6,4))
    plt.plot(hist['pi']); plt.xlabel('step'); plt.ylabel('phantom_index'); plt.tight_layout()
    plt.savefig(f"{out_prefix}_pi.png", dpi=160); plt.close()
    plt.figure(figsize=(6,4))
    plt.plot(hist['margin']); plt.xlabel('step'); plt.ylabel('margin_norm'); plt.tight_layout()
    plt.savefig(f"{out_prefix}_margin.png", dpi=160); plt.close()
    plt.figure(figsize=(6,4))
    plt.plot(hist['snr']); plt.xlabel('step'); plt.ylabel('snr [dB]'); plt.tight_layout()
    plt.savefig(f"{out_prefix}_snr.png", dpi=160); plt.close()

# ------------------------------
# Gates
# ------------------------------
@dataclass
class GateParams:
    pi_max: float = 0.10
    margin_min: float = 0.04
    S_median_min: float = 0.55
    r_trend_min: float = 0.90


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
    ap = argparse.ArgumentParser(description="Stage-11 LLM Shadow (v4: stepwise denoise)")
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--tap",   type=int, default=-3)
    ap.add_argument("--calib", type=str, required=True)
    ap.add_argument("--eval",  type=str, required=True)
    ap.add_argument("--out_json", type=str, default="llm_shadow_hijack_summary.json")
    ap.add_argument("--render", action="store_true")

    # Steps
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--eta", type=float, default=0.20)

    # Phantom metric knobs
    ap.add_argument("--use_depth_weighted_pi", type=int, default=1)
    ap.add_argument("--pi_beta", type=float, default=3.0)
    ap.add_argument("--nms_radius", type=int, default=5)

    # Warp knobs
    ap.add_argument("--sigma_scale", type=float, default=0.80)
    ap.add_argument("--local_radius", type=float, default=1.40)

    # Denoiser knobs (stepwise)
    ap.add_argument("--ema_gamma", type=float, default=0.80)
    ap.add_argument("--med_k", type=int, default=5)
    ap.add_argument("--tau_conf", type=float, default=0.60)
    ap.add_argument("--jitter_sigma", type=float, default=0.03)
    ap.add_argument("--jitter_J", type=int, default=8)
    ap.add_argument("--backoff", type=float, default=0.5)

    # Token denoiser knobs (final r_trend)
    ap.add_argument("--tok_eta", type=float, default=0.15)

    # Gates
    ap.add_argument("--pi_max", type=float, default=0.10)
    ap.add_argument("--margin_min", type=float, default=0.04)
    ap.add_argument("--S_median_min", type=float, default=0.55)
    ap.add_argument("--r_trend_min", type=float, default=0.90)

    args = ap.parse_args()

    gates = GateParams(args.pi_max, args.margin_min, args.S_median_min, args.r_trend_min)

    tok, model = load_model(args.model)

    # 1) Calibration → PCA(3)
    calib_prompts = read_lines(args.calib); eval_prompts  = read_lines(args.eval)
    Hc = collect_hidden_states(model, tok, calib_prompts, args.tap)
    pca = PCA(n_components=3, whiten=True, random_state=0)
    Yc  = pca.fit_transform(Hc)

    # 2) Eval projection
    He = collect_hidden_states(model, tok, eval_prompts, args.tap)
    Ye = pca.transform(He)

    # 3) Pre metrics (density-only + depth-weighted) and S proxy
    beta = args.pi_beta if args.use_depth_weighted_pi else 0.0
    pi0, m0, dbg0, _ = phantom_metrics_Y3(Ye, beta=beta, nms_radius=args.nms_radius)
    S0 = S_median_proxy(Ye)

    # 4) Detect deepest basin (density) and build warp params
    center = detect_center_density(Ye)
    wp = make_warp_params(Ye, center, sigma_scale=args.sigma_scale, local_radius=args.local_radius)

    # 5) Stepwise descent with denoiser controls
    Y_post, hist = stepwise_descent(Ye, center, wp, steps=args.steps, eta=args.eta,
                                    beta_pi=beta, nms_radius=args.nms_radius,
                                    dn=DenoiseParams(args.ema_gamma, args.med_k, args.tau_conf,
                                                     args.jitter_sigma, args.jitter_J, args.backoff))

    # 6) Post metrics
    pi1, m1, dbg1, _ = phantom_metrics_Y3(Y_post, beta=beta, nms_radius=args.nms_radius)
    S1 = S_median_proxy(Y_post)

    # 7) Token r_trend using final center (same as v3)
    tok_dn = TokenDenoiseParams(args.ema_gamma, args.med_k, args.tau_conf, args.tok_eta, args.jitter_sigma, args.jitter_J)
    rtr1 = token_rtrend_for_prompts(model, tok, eval_prompts, args.tap, pca, center, tok_dn)

    # Gates
    go0 = (pi0 <= gates.pi_max) and (m0 >= gates.margin_min) and (S0 >= gates.S_median_min)
    go1 = (pi1 <= gates.pi_max) and (m1 >= max(gates.margin_min, m0)) and (S1 >= max(gates.S_median_min, S0)) and (rtr1 >= gates.r_trend_min)

    summary = dict(
        model=args.model, tap=args.tap,
        gates={"pi_max":gates.pi_max, "margin_min":gates.margin_min, "S_median_min":gates.S_median_min, "r_trend_min":gates.r_trend_min},
        center=center.tolist(),
        warp={"sigma": wp.sigma, "local_radius": wp.local_radius, "depth_scale": wp.depth_scale, "mix_z": wp.mix_z},
        phantom={"mode": ("depth_weighted" if beta>0 else "density_only"), "beta": beta, "nms_radius": args.nms_radius},
        pre={"phantom_index": pi0, "margin_norm": m0, "S_median": S0, "debug": dbg0},
        post={"phantom_index": pi1, "margin_norm": m1, "S_median": S1, "r_trend_tokens": rtr1, "debug": dbg1},
        improve={"d_phantom_index": float(pi1 - pi0), "d_margin_norm": float(m1 - m0), "d_S_median": float(S1 - S0)},
        hist=hist,
        go_pre=bool(go0), go_post=bool(go1)
    )

    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("[SUMMARY]", json.dumps(summary, indent=2))

    if args.render:
        render_Y3(Ye, "LLM PCA(3) — pre", "llm_pca3_eval_pre.png")
        render_Y3(Y_post, "LLM PCA(3) — post (stepwise hijack)", "llm_pca3_eval_post.png")
        render_curves(hist, "llm_shadow_step")
if __name__ == "__main__":
    main()
