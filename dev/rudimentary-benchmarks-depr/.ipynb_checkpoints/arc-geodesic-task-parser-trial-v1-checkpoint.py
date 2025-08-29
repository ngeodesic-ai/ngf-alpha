#!/usr/bin/env python3
"""
arc_task_parser_geodesic_v5p1_multitask.py

Stage-10 parser (fresh script) that implements the two key fixes:
  1) TWO-SIDED residual energy: use |-(R @ v)| instead of signed + ReLU
  2) PER-STEP CHANNEL NORMALIZATION on the all-fields run (after taking abs),
     then smooth; prototypes remain unnormalized (except smoothing) to preserve
     their informative shape for matched filtering.

This script is self-contained and will run in a small DEMO mode if you don't
plug it into your ARC-12 latent provider. To integrate with your pipeline,
replace `prepare_latents_and_centers()` with your real PCA/centers stage and
call `parse_sample(...)` on each sample.

Outputs per sample:
  Tasks: [list] | Order: A → B → ... | Concurrency: pairs | areas⊥=[...] corr=[...]
  Plus a plot of S(t) = smoothed, per-step-normalized |E_perp_all| with peak markers,
  and dashed prototype overlay for the top channel.

"""

import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

# -----------------------------
# Utility helpers
# -----------------------------

def smooth_ma(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    # 1D or 2D [T] or [T,K]
    if x.ndim == 1:
        pad = np.pad(x, (win-1, 0), mode='edge')
        k = np.ones(win) / win
        return np.convolve(pad, k, mode='valid')
    else:
        out = np.zeros_like(x)
        for j in range(x.shape[1]):
            out[:, j] = smooth_ma(x[:, j], win)
        return out

def trapz_area(y: np.ndarray) -> float:
    # y: [T]
    return float(np.trapz(y))

def normxcorr(a: np.ndarray, b: np.ndarray) -> float:
    # normalized correlation (cosine similarity of zero-mean vectors)
    a = a - a.mean(); b = b - b.mean()
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))

# Per-step channel normalization: remove common-mode and scale to simplex per time step
# E: [T,K] nonnegative

def per_step_channel_norm(E: np.ndarray) -> np.ndarray:
    Emin = E.min(axis=1, keepdims=True)
    E0 = np.clip(E - Emin, 0.0, None)
    S = E0.sum(axis=1, keepdims=True) + 1e-12
    return E0 / S

# -----------------------------
# Geodesic toy physics (replace with your real integrator)
# -----------------------------

class GeomCfg:
    def __init__(self, lam=0.35, gamma=0.04, dt=0.02, steps=700, mass_scale=4.0):
        self.lam = lam
        self.gamma = gamma
        self.dt = dt
        self.steps = steps
        self.mass_scale = mass_scale

# Potential: U_k(x) = 0.5 * ||x - c_k||^2; grad = x - c_k
# Residuals: r_k = g_k - mean_j g_j (channel-wise residual to common-mode)

def orth_residuals(grads: np.ndarray) -> np.ndarray:
    # grads: [K,d]
    g_bar = grads.mean(axis=0, keepdims=True)
    return grads - g_bar  # zero-sum residuals, good enough for parsing

# Core integrator returning raw grads, residual directions and energies

def run_geodesic(x0: np.ndarray,
                 centers: np.ndarray,
                 include_mask: np.ndarray,
                 cfg: GeomCfg) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
      E_raw:  [T,K] signed raw alignment (-(g_k @ v))
      E_perp: [T,K] two-sided residual energy |-(r_k @ v)|
      U_perp: [T,K,d] unit residual directions r_k/||r_k|| (for concurrency coherence)
    """
    K, d = centers.shape
    T = cfg.steps
    x = x0.copy()
    v = np.zeros_like(x)

    E_raw = np.zeros((T, K), dtype=np.float64)
    E_perp = np.zeros((T, K), dtype=np.float64)
    U_perp = np.zeros((T, K, d), dtype=np.float64)

    for t in range(T):
        # compute per-concept gradients (active only)
        diffs = x[None, :] - centers  # [K,d]
        grads = diffs  # grad of 0.5*||x-c||^2
        grads = grads * include_mask[:, None]

        # aggregate field (sum of active concept forces)
        g_sum = grads.sum(axis=0)
        # physics update: damped mass-spring towards centers
        a = -cfg.lam * g_sum - cfg.gamma * v
        v = v + cfg.dt * a / cfg.mass_scale
        x = x + cfg.dt * v

        # energies
        raw_signed = -(grads @ v)                   # [K]
        R = orth_residuals(grads)                   # [K,d]
        norms = np.linalg.norm(R, axis=1, keepdims=True) + 1e-12
        U = R / norms
        perp_signed = -(R @ v)
        perp_mag = np.abs(perp_signed)              # TWO-SIDED energy (v5.1)

        E_raw[t] = raw_signed
        E_perp[t] = perp_mag
        U_perp[t] = U

    return E_raw, E_perp, U_perp

# -----------------------------
# Parser (presence, order, concurrency)
# -----------------------------

def halfmax_window(y: np.ndarray) -> Tuple[int, int]:
    if y.size == 0:
        return (0, 0)
    peak = int(np.argmax(y))
    hm = 0.5 * y[peak]
    # left
    L = peak
    while L > 0 and y[L] > hm:
        L -= 1
    # right
    R = peak
    while R < len(y)-1 and y[R] > hm:
        R += 1
    return (L, R)

def parse_multitask(x0: np.ndarray,
                    centers: np.ndarray,
                    names: List[str],
                    outdir: str,
                    sample_id: str,
                    cfg: GeomCfg,
                    smooth_win: int = 7,
                    present_area_frac: float = 0.60,
                    present_mf_thresh: float = 0.20,
                    conc_overlap_frac: float = 0.30,
                    conc_coh_max: float = 0.40,
                    make_plot: bool = True) -> Dict:
    os.makedirs(outdir, exist_ok=True)
    K = len(names)

    # All-fields run
    include_all = np.ones(K, dtype=bool)
    E_raw_all, E_perp_all, U_perp_all = run_geodesic(x0, centers, include_all, cfg)

    # Build prototypes (exclusive single-field runs) and correlations
    protos = []
    corr = np.zeros(K, dtype=np.float64)
    proto_plot = None

    for k in range(K):
        mask = np.zeros(K, dtype=bool); mask[k] = True
        _, E_perp_only, _ = run_geodesic(x0, centers, mask, cfg)
        proto_k = smooth_ma(np.abs(E_perp_only[:, k]), smooth_win)
        protos.append(proto_k)
        S_k_for_corr = smooth_ma(np.abs(E_perp_all[:, k]), smooth_win)
        corr[k] = normxcorr(S_k_for_corr, proto_k)

    # Per-step normalize across channels (after abs) for multitask signal
    S = per_step_channel_norm(np.abs(E_perp_all))
    S = smooth_ma(S, smooth_win)   # [T, K]

    # Presence by area fraction + matched-filter corr
    areas = np.array([trapz_area(S[:, k]) for k in range(K)])
    k_max = int(np.argmax(areas))
    keep = []
    for k in range(K):
        if areas[k] >= present_area_frac * areas[k_max] and corr[k] >= present_mf_thresh:
            keep.append(k)
    if not keep:
        keep = [k_max]  # guarantee at least one task

    # Order by peak locations (on S)
    peaks = {k: int(np.argmax(S[:, k])) for k in keep}
    order_idx = sorted(keep, key=lambda kk: peaks[kk])
    order = [names[k] for k in order_idx]

    # Concurrency via half-max overlap + residual-direction incoherence
    pairs = []
    for i in range(len(order_idx)):
        ki = order_idx[i]
        yi = S[:, ki]
        Li, Ri = halfmax_window(yi)
        for j in range(i+1, len(order_idx)):
            kj = order_idx[j]
            yj = S[:, kj]
            Lj, Rj = halfmax_window(yj)
            L = max(Li, Lj); R = min(Ri, Rj)
            if R > L:
                overlap = (R - L) / (max(Ri, Rj) - min(Li, Lj) + 1e-9)
                # direction coherence over overlap (mean abs cosine)
                Ui = U_perp_all[L:R, ki, :]  # [M,d]
                Uj = U_perp_all[L:R, kj, :]
                num = np.abs(np.einsum('nd,nd->n', Ui, Uj))
                coh = float(num.mean()) if num.size else 1.0
                if overlap >= conc_overlap_frac and coh <= conc_coh_max:
                    pairs.append((names[ki], names[kj]))

    tasks = [names[k] for k in keep]

    # Plot
    figpath = None
    if make_plot:
        fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.0))
        for k in range(K):
            ax.plot(S[:, k], label=f"E⊥ {names[k]}")
        # overlay top prototype on its channel for visualization
        proto_plot = protos[k_max]
        ax.plot(proto_plot, '--', label=f"proto⊥ {names[k_max]}")
        # mark peaks of kept tasks
        for k in keep:
            ax.axvline(peaks[k], color='k', linestyle=':', alpha=0.4)
        title = f"v5.1 parse — {sample_id} — tasks: {tasks} — order: {' → '.join(order)}"
        ax.set_title(title)
        ax.set_xlabel('step'); ax.set_ylabel('residual aligned power (smoothed)')
        ax.legend()
        figpath = os.path.join(outdir, f"parse_v51_{sample_id}.png")
        plt.tight_layout(); plt.savefig(figpath); plt.close(fig)

    return {
        'tasks': tasks,
        'order': order,
        'concurrency': pairs,
        'areas': areas,
        'corr': corr,
        'plot': figpath,
    }

# -----------------------------
# DEMO data (optional). Replace with your PCA + centers.
# -----------------------------

def prepare_latents_and_centers(K: int = 3, d: int = 32, seed: int = 7):
    """Small synthetic setup to exercise the parser.
    Returns
      Z0: list of T0 latent starting points for a few demo samples
      centers: [K,d]
      names: list of concept names
    """
    rng = np.random.default_rng(seed)
    names = ['flip_h', 'flip_v', 'rotate'][:K]
    # random orthonormal-ish centers
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    centers = Q[:,:K].T * 2.0  # spread out
    # a few demo latents placed off-center
    Z0 = [rng.normal(scale=1.0, size=(d,)) + 0.5*centers[rng.integers(0, K)] for _ in range(6)]
    return Z0, centers, names

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=str, default='./plots_v5p1', help='where to save plots')
    ap.add_argument('--demo', action='store_true', help='run synthetic demo instead of external provider')
    ap.add_argument('--steps', type=int, default=700)
    args = ap.parse_args()

    cfg = GeomCfg(steps=args.steps)
    os.makedirs(args.outdir, exist_ok=True)

    # Replace this block with your real PCA/centers extraction
    Z0, centers, names = prepare_latents_and_centers()

    for i, z0 in enumerate(Z0, start=1):
        sid = f"sample{i:02d}"
        res = parse_multitask(z0, centers, names, args.outdir, sid, cfg,
                              smooth_win=7,
                              present_area_frac=0.60,
                              present_mf_thresh=0.20,
                              conc_overlap_frac=0.30,
                              conc_coh_max=0.40,
                              make_plot=True)
        tasks = res['tasks']
        order = res['order']
        conc = res['concurrency']
        areas = np.round(res['areas'], 6)
        corr = np.round(res['corr'], 3)
        print(f"[{i:02d}] -> Tasks: {tasks} | Order: {' → '.join(order) if order else '—'} | Concurrency: {conc}")
        print(f"     areas⊥={areas.tolist()} corr={corr.tolist()} plot={res['plot']}")

if __name__ == '__main__':
    main()
