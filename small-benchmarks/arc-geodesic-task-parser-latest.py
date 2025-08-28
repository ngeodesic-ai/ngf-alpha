# Write the integrated script to /mnt/data as arc-benchmark-geodesic-v5_3.py

# script = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arc-benchmark-geodesic-v5_3.py
--------------------------------
Geodesic task parser (Stage 10, v5.3)

Key changes vs v5.2:
  • Presence gating uses EXCLUSIVE residual energy (⊥ to the span of other prototypes).
  • Ordering via matched-filter lag; concurrency requires temporal proximity + correlation.
  • Clean CLI and plotting; optional synthetic mode for smoke tests.

This file is drop-in runnable in "synthetic" mode. In your real pipeline,
replace `compute_geodesic_traces()` to return your per-concept aligned power
traces (E_raw) and per-concept prototype traces (proto_all).

Expected shapes:
  - E_raw:    (T, K)   raw aligned power vs. time (one column per concept)
  - proto_all:(T, K)   prototype traces (same length & concept order)
  - names:    list[str] of length K, e.g. ['flip_h','flip_v','rotate']

Author: ngeodesic — v5.3
"""
import os, sys, math, argparse, textwrap
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

# ----------------------------- utils ---------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean()
    sd = x.std() + 1e-12
    return (x - mu) / sd

def smooth_pos(x: np.ndarray, s: int) -> np.ndarray:
    x = np.maximum(x, 0.0)
    if s and s > 1:
        return uniform_filter1d(x, size=s, mode='nearest')
    return x

# ----------------------- presence (exclusive ⊥) ----------------------

def exclusive_residual_traces(E_raw: np.ndarray,
                              protos: np.ndarray,
                              smooth: int = 11) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute exclusive residual for each concept: the part of E_raw[:,k] that cannot
    be explained by the subspace spanned by ALL OTHER prototypes.

    Returns:
        E_perp   : (T,K) residual traces (positive part, smoothed)
        areas_perp: (K,) integrated area under each residual trace
    """
    T, K = E_raw.shape
    E_perp = np.zeros_like(E_raw, dtype=np.float64)
    areas  = np.zeros(K, dtype=np.float64)

    # z-score timewise so projection is pattern-based, not scale-based
    Pz = np.stack([_z(protos[:, k]) for k in range(K)], axis=1)
    Ez = np.stack([_z(E_raw[:, k])  for k in range(K)], axis=1)

    for k in range(K):
        # basis of "others"
        if K == 1:
            # corner case
            r = smooth_pos(Ez[:, k], smooth)
            E_perp[:, k] = r
            areas[k] = np.trapz(r)
            continue

        B = np.delete(Pz, k, axis=1)            # (T, K-1)
        # Orthonormalize "others" with QR
        Q, _ = np.linalg.qr(B, mode='reduced')  # (T, K-1)
        x = Ez[:, k]
        x_expl = Q @ (Q.T @ x)
        r = x - x_expl
        r = smooth_pos(r, smooth)
        E_perp[:, k] = r
        areas[k] = np.trapz(r)

    return E_perp, areas

# -------------------- matched-filter ordering ------------------------

def matched_filter_lag_and_corr(x: np.ndarray, p: np.ndarray) -> Tuple[int, float]:
    """
    Normalized cross-correlation to find best alignment lag and similarity.
    Returns (lag, corr) where positive lag means p lags behind x (i.e., x leads).
    """
    xz, pz = _z(x), _z(p)
    c = np.correlate(xz, pz, mode='full')
    i = int(np.argmax(c))
    lag = i - (len(xz) - 1)
    corr = float(c[i] / (np.linalg.norm(xz) * np.linalg.norm(pz) + 1e-12))
    return lag, corr

def order_and_concurrency(E_raw: np.ndarray,
                          protos: np.ndarray,
                          sel_idx: List[int],
                          corr_thr: float = 0.35,
                          co_tau: int = 25) -> Tuple[List[int], List[Tuple[int,int]], Dict[int, Tuple[int,float]]]:
    """Return ordered indices, concurrent pairs, and {k:(lag,corr)} for selected concepts."""
    events = []
    for k in sel_idx:
        lag, corr = matched_filter_lag_and_corr(E_raw[:, k], protos[:, k])
        events.append((k, lag, corr))

    events.sort(key=lambda t: t[1])  # by lag
    order = [k for (k, _, _) in events]

    conc = []
    for i in range(len(events) - 1):
        k1, l1, c1 = events[i]
        k2, l2, c2 = events[i+1]
        if abs(l2 - l1) <= co_tau and c1 >= corr_thr and c2 >= corr_thr:
            conc.append((k1, k2))

    ev = {k: (lag, corr) for (k, lag, corr) in events}
    return order, conc, ev

# ---------------------- plotting helpers -----------------------------

def plot_sample(save_path: str,
                E_perp: np.ndarray,
                E_raw: np.ndarray,
                protos: np.ndarray,
                names: List[str],
                tasks_idx: List[int],
                order_idx: List[int],
                vertical_marks: List[int] = None,
                title_prefix: str = "v5.3 parse"):
    ensure_dir(os.path.dirname(save_path) or ".")
    T, K = E_perp.shape
    t = np.arange(T)

    plt.figure(figsize=(10, 4))
    # plot all E_perp to visualize separation
    for k in range(K):
        lbl = f"E⊥ {names[k]}"
        plt.plot(t, E_perp[:, k], label=lbl, linewidth=2 if k in tasks_idx else 1, alpha=0.9 if k in tasks_idx else 0.6)

    # overlay the first picked prototype for visual sanity
    if order_idx:
        k0 = order_idx[0]
        p = _z(protos[:, k0])
        # scale prototype to roughly match max of E_perp channel (for eyeballing)
        scale = (E_perp[:, k0].max() + 1e-12)
        p_vis = (p - p.min()) / (p.max() - p.min() + 1e-12) * scale
        plt.plot(t, p_vis, 'r--', linewidth=1.5, label=f"proto⊥ {names[k0]}")

    if vertical_marks:
        for vm in vertical_marks:
            plt.axvline(vm, color='k', linestyle=':', alpha=0.35)

    order_names = " → ".join([names[k] for k in order_idx]) if order_idx else "—"
    title = f"{title_prefix} — tasks: {[names[k] for k in tasks_idx]} — order: {order_names}"
    plt.title(title)
    plt.xlabel("step"); plt.ylabel("residual aligned power (smoothed)")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110)
    plt.close()

# -------------------- synthetic fallback (for smoke tests) ------------

def _make_sine_proto(T: int, phase: float, freq: float) -> np.ndarray:
    t = np.linspace(0, 2*np.pi, T)
    s = np.sin(freq * t + phase)
    # half-wave rectify to mimic positive "energy"
    s = np.maximum(s, 0.0)
    return s

def synthetic_traces(T: int = 720) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (E_raw, protos, names) with 3 concepts for quick sanity checks."""
    names = ['flip_h', 'flip_v', 'rotate']
    # Prototypes: different phases/frequencies
    P = np.stack([
        _make_sine_proto(T, phase=0.0,       freq=2.0),
        _make_sine_proto(T, phase=0.7*np.pi, freq=2.0),
        _make_sine_proto(T, phase=1.4*np.pi, freq=2.0),
    ], axis=1)

    # Raw signals = prototypes + small cross-talk + drift
    rng = np.random.default_rng(0)
    X = P.copy()
    X[:, 0] += 0.08*P[:, 1] + 0.03*P[:, 2]
    X[:, 1] += 0.08*P[:, 0] + 0.03*P[:, 2]
    X[:, 2] += 0.08*P[:, 0] + 0.03*P[:, 1]

    # slight upward drift + noise
    drift = np.linspace(0, 0.05, T)[:, None]
    X = np.maximum(X + drift + 0.01*rng.standard_normal(X.shape), 0.0)
    return X, P, names

# ---------------- Integrate with your geodesic pipeline ----------------

def compute_geodesic_traces(sample_idx: int,
                            args) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    REPLACE THIS with your real pipeline code that:
      - runs the geodesic for sample_idx
      - returns (E_raw, proto_all, names)
    If --synthetic is set, we return a synthetic triple.
    """
    if args.synthetic:
        return synthetic_traces(T=args.T)

    # ---- TEMPLATE ----
    # from your_module import run_sample_geodesic  # <-- provide this
    # E_raw, proto_all, names = run_sample_geodesic(sample_idx, args)
    # return E_raw, proto_all, names
    raise NotImplementedError("Hook up `compute_geodesic_traces` to your geodesic runner or use --synthetic.")

# ------------------------------- main ---------------------------------

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Stage-10 geodesic task parser (v5.3: exclusive residual + gated concurrency)"
    )
    ap.add_argument('--samples', type=int, default=6, help='number of samples to parse')
    ap.add_argument('--T', type=int, default=720, help='timesteps per trace (synthetic only)')
    ap.add_argument('--smooth', type=int, default=11, help='smoothing window for residual traces')
    ap.add_argument('--presence_frac', type=float, default=0.35, help='keep concepts with area >= frac * best')
    ap.add_argument('--corr_thr', type=float, default=0.35, help='min correlation for concurrency gating')
    ap.add_argument('--co_tau', type=int, default=25, help='max lag gap (steps) to consider concurrency')
    ap.add_argument('--outdir', type=str, default='plots_v5p3', help='output directory for plots')
    ap.add_argument('--synthetic', action='store_true', help='use synthetic traces (self-contained)')

    args = ap.parse_args()
    ensure_dir(args.outdir)

    for i in range(1, args.samples + 1):
        # 1) get traces
        E_raw, proto_all, names = compute_geodesic_traces(i, args)
        T, K = E_raw.shape

        # 2) presence via exclusive residual energy
        E_perp, areas_perp = exclusive_residual_traces(E_raw, proto_all, smooth=args.smooth)

        # 3) select concepts by area
        best = float(areas_perp.max()) if K > 0 else 0.0
        keep_idx = [k for k, a in enumerate(areas_perp) if a >= args.presence_frac * (best + 1e-12)]
        # fallback: if everything was pruned, keep the best one
        if not keep_idx and K > 0:
            keep_idx = [int(np.argmax(areas_perp))]

        # 4) order + concurrency with gating
        order_idx, conc_pairs, ev = order_and_concurrency(E_raw, proto_all, keep_idx,
                                                          corr_thr=args.corr_thr, co_tau=args.co_tau)

        corr_peak = [round(ev[k][1], 3) for k in keep_idx]
        areas_dbg = [round(float(a), 3) for a in areas_perp]

        # 5) plot
        save_path = os.path.join(args.outdir, f"sample{i:02d}.png")
        plot_sample(save_path, E_perp, E_raw, proto_all, names, keep_idx, order_idx,
                    vertical_marks=[0, T//3, 2*T//3], title_prefix=f"v5.3 parse — sample{i:02d}")

        # 6) log
        tasks_txt = [names[k] for k in keep_idx]
        order_txt = " → ".join([names[k] for k in order_idx]) if order_idx else "—"
        conc_txt = [(names[a], names[b]) for (a,b) in conc_pairs]
        print(f"[{i:02d}] -> Tasks: {tasks_txt} | Order: {order_txt} | Concurrency: {conc_txt}")
        print(f"     areas⊥={areas_dbg} corr_peak={corr_peak} plot={save_path}")

if __name__ == "__main__":
    main()
# '''
# path = "/mnt/data/arc-benchmark-geodesic-v5_3.py"
# with open(path, "w") as f:
#     f.write(script)
# path
