#!/usr/bin/env python3
"""
arc_task_parser_geodesic_v5p2_multitask.py

Adds --order_by flag (mf|peak). Default is 'mf' which orders tasks by
matched‑filter lag computed on the *raw (non-normalized)* residual energy.
Also updates the plot to draw vertical markers at the MF peaks when used.

Drop-in notes:
- Replace `demo_prepare_latents()` with your real geodesic run producing
  E_perp_all: [T, K] residual channel magnitudes per step (before any per-step
  normalization). K must equal len(CONCEPTS).
- Replace `build_prototypes(T)` with channel-specific prototypes for your fields
  if you have them; otherwise keep the defaults.

Outputs one plot per sample under ./plots_v5p2/ and logs the inferred tasks,
order, and naive concurrency hints.
"""
from __future__ import annotations
import argparse, os, math
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

# --------------------- helpers ---------------------
def smooth_ma(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: return x
    if x.ndim == 1:
        xpad = np.pad(x, (win, win), mode='edge')
        k = np.ones(2 * win + 1) / (2 * win + 1)
        return np.convolve(xpad, k, mode='same')[win:-win]
    else:
        # apply per column
        return np.stack([smooth_ma(x[:, i], win) for i in range(x.shape[1])], axis=1)

def norm_per_step(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # L2 normalize each time step vector across channels (keeps relative channel composition)
    denom = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / denom

def zscore(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = vec - vec.mean()
    s = vec.std() + eps
    return v / s

def ncc_full(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalized cross-correlation, full mode."""
    a0 = zscore(a)
    b0 = zscore(b)
    return np.correlate(a0, b0, mode='full')

# --------------------- prototypes ---------------------
CONCEPTS = ["flip_h", "flip_v", "rotate"]

def build_prototypes(T: int) -> Dict[int, np.ndarray]:
    """Simple wavy envelopes for demo. Replace with learned/exclusive traces if available."""
    t = np.linspace(0, 2.5 * math.pi, T)
    protos = {
        0: np.abs(np.sin(t * 0.9 + 0.15)),   # flip_h
        1: np.abs(np.sin(t * 0.9 + 0.45)),   # flip_v (phase shift)
        2: np.abs(np.sin(t * 0.9 + 0.85)) * 0.6,  # rotate, lower weight
    }
    # Keep all nonnegative
    return {k: (v - v.min()) for k, v in protos.items()}

# --------------------- parsing core ---------------------
@dataclass
class ParseResult:
    tasks: List[str]
    order: List[str]
    concurrency: List[Tuple[str, str]]
    areas_frac: np.ndarray  # [K]
    corr_peak: np.ndarray   # [K]
    mf_peaks: Dict[int, int]


def parse_multitask(
    E_perp_all: np.ndarray,               # [T, K] raw residual magnitudes per channel
    names: List[str],
    order_by: str = "mf",                # 'mf' or 'peak'
    smooth_win: int = 31,
    present_area_frac: float = 0.15,      # min fraction of total area to count as present
    present_mf_thresh: float = 0.05,      # min peak NCC on normalized traces
    overlap_slack: int = 25,
    out_plot: str | None = None,
) -> ParseResult:
    T, K = E_perp_all.shape
    assert K == len(names)

    # Presence channelization on per-step normalized energies
    S_norm = smooth_ma(norm_per_step(np.abs(E_perp_all)), smooth_win)  # [T,K]

    # Areas per channel
    area = S_norm.sum(axis=0)
    areas_frac = area / (area.sum() + 1e-8)

    # Prototypes (normalized for NCC on normalized signals)
    protos = build_prototypes(T)
    corr_peak = np.zeros(K)
    for k in range(K):
        c = ncc_full(S_norm[:, k], protos[k])
        corr_peak[k] = float(np.max(c) / (len(S_norm[:, k])))  # scale invariance across T

    # Presence: keep channels by area and mild NCC gate (helpful when protos differ)
    keep_idx = [k for k in range(K) if areas_frac[k] >= present_area_frac and corr_peak[k] >= present_mf_thresh]
    if not keep_idx:
        # Fall back: take top-1 by area
        keep_idx = [int(np.argmax(areas_frac))]

    # --- Ordering ---
    mf_peaks: Dict[int, int] = {}
    if order_by == "mf":
        # Use RAW magnitudes (smoothed) to preserve temporal structure
        S_raw = smooth_ma(np.abs(E_perp_all), smooth_win)
        for k in keep_idx:
            c = ncc_full(S_raw[:, k], protos[k])
            lag = int(np.argmax(c) - (len(protos[k]) - 1))
            # anchor to prototype internal peak
            p_peak = int(np.argmax(protos[k]))
            idx = max(0, min(T - 1, lag + p_peak))
            mf_peaks[k] = idx
        order_idx = sorted(keep_idx, key=lambda kk: mf_peaks[kk])
    elif order_by == "peak":
        peaks = {k: int(np.argmax(S_norm[:, k])) for k in keep_idx}
        mf_peaks = peaks
        order_idx = sorted(keep_idx, key=lambda kk: peaks[kk])
    else:
        raise ValueError("order_by must be 'mf' or 'peak'")

    tasks = [names[k] for k in keep_idx]
    order = [names[k] for k in order_idx]

    # --- Concurrency (naive): overlapping half-max windows in S_norm ---
    def halfmax_window(sig: np.ndarray, center: int) -> Tuple[int, int]:
        hm = 0.5 * sig.max()
        i = center
        L = i
        while L > 0 and sig[L] > hm:
            L -= 1
        R = i
        while R < len(sig) - 1 and sig[R] > hm:
            R += 1
        return L, R

    concurrency: List[Tuple[str, str]] = []
    for a, b in zip(order_idx, order_idx[1:]):
        La, Ra = halfmax_window(S_norm[:, a], mf_peaks[a])
        Lb, Rb = halfmax_window(S_norm[:, b], mf_peaks[b])
        overlap = (min(Ra, Rb) - max(La, Lb))
        if overlap > overlap_slack:
            concurrency.append((names[a], names[b]))

    # --- Plot ---
    if out_plot:
        os.makedirs(os.path.dirname(out_plot), exist_ok=True)
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        for k, color in zip(range(K), [None, None, None]):
            ax.plot(S_norm[:, k], label=f"E⊥ {names[k]}")
        # prototype over the selected channels only (for readability)
        for k in keep_idx:
            ax.plot(build_prototypes(T)[k], "r--", alpha=0.6, label=f"proto⊥ {names[k]}" if k == keep_idx[0] else None)
        for k in keep_idx:
            ax.axvline(mf_peaks[k], color='k', linestyle=':', alpha=0.35)
        ax.set_title(f"v5.2 parse — {os.path.basename(out_plot).split('.')[0]} — tasks: {tasks} — order: {' → '.join(order)}")
        ax.set_xlabel("step"); ax.set_ylabel("residual aligned power (smoothed)")
        ax.legend(); fig.tight_layout(); fig.savefig(out_plot); plt.close(fig)

    return ParseResult(tasks, order, concurrency, areas_frac, corr_peak, mf_peaks)

# --------------------- demo wiring ---------------------

def demo_prepare_latents(T: int = 720, K: int = 3, seed: int = 0) -> np.ndarray:
    """Toy residual magnitudes with channel preferences that drift.
    Replace with your real E_perp_all from the geodesic run.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1.0, T)
    base = np.stack([
        0.9 + 0.2 * t,         # flip_h growth
        0.8 + 0.25 * t,        # flip_v growth
        0.5 + 0.18 * t,        # rotate growth
    ], axis=1)
    # add tiny wiggles and noise
    wig = 0.02 * np.sin(2 * math.pi * (t * 3)[:, None] + np.array([0.2, 0.6, 1.0])[None, :])
    E = np.abs(base + wig + 0.005 * rng.standard_normal((T, K)))
    return E

# --------------------- CLI ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--order_by", type=str, default="mf", choices=["mf", "peak"])
    ap.add_argument("--smooth", type=int, default=31)
    ap.add_argument("--area_frac", type=float, default=0.15)
    ap.add_argument("--mf_thresh", type=float, default=0.05)
    ap.add_argument("--outdir", type=str, default="plots_v5p2")
    args = ap.parse_args()

    names = CONCEPTS
    for i in range(1, args.samples + 1):
        E = demo_prepare_latents(seed=41 + i)
        out_plot = os.path.join(args.outdir, f"sample{i:02d}.png")
        res = parse_multitask(
            E, names,
            order_by=args.order_by,
            smooth_win=args.smooth,
            present_area_frac=args.area_frac,
            present_mf_thresh=args.mf_thresh,
            out_plot=out_plot,
        )
        print(f"[{i:02d}] -> Tasks: {res.tasks} | Order: {' → '.join(res.order)} | Concurrency: {res.concurrency}")
        print(f"     areas⊥={np.round(res.areas_frac, 3).tolist()} corr_peak={np.round(res.corr_peak, 3).tolist()} plot={out_plot}")

if __name__ == "__main__":
    main()
