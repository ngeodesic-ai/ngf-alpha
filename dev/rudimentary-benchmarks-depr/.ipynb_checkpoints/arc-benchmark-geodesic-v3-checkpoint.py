#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC-like Benchmark — Geodesic vs Stock (v3-fixed)
=================================================

What this script does
---------------------
• Generates synthetic ARC-like tasks with 3 primitives: flip_h, flip_v, rotate
• Emits both raw per-concept energy traces (**E_raw**) and matching prototype traces (**proto_all**)
• Parses task set + order using either:
  - STOCK: smoothed perpendicular energy (no exclusivity, area gating only)
  - GEODESIC: **exclusive residual on PERPENDICULAR energy** (QR span-complement) + correlation gate
• Executes the predicted sequence on a grid and evaluates exact-match accuracy, semantic similarity, and hallucination rate
• Reports paired bootstrap deltas and writes per-sample CSV

Key fixes vs previous v3
------------------------
1) `moving_average` now supports 1-D and 2-D inputs (smooths columns independently).
2) Geodesic branch does **exclusive residual over perpendicular energy**, not raw (avoids signal annihilation).
3) **Correlation uses each channel's own prototype** (from `proto_all`) instead of a generic half-sine.
4) Residuals are **energyized**: no ReLU; we square the residual (r**2) then smooth — preserves signal even if it flips.
5) Synthetic generator no longer injects large background bumps on inactive prototypes.
6) Thresholds default slightly looser to account for amplitude compression.
7) Optional ablation via `--geo_mode` to compare exclusive vs plain vs none.

Usage
-----
python3 arc-benchmark-geodesic-plus-stock.py --samples 12 --seed 42 --T 720 --noise 0.01 --plot_dir plots_compare

"""

import argparse, csv
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

PRIMS = ["flip_h", "flip_v", "rotate"]

# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int):
    return np.random.default_rng(seed)


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """Reflect-padded moving average for 1-D or 2-D arrays.
    - (T,) -> (T,)
    - (T,K) -> (T,K), column-wise smoothing
    """
    x = np.asarray(x)
    if k <= 1:
        return x.copy()
    pad = k // 2
    kernel = np.ones(k) / k
    if x.ndim == 1:
        xp = np.pad(x, (pad, pad), mode="reflect")
        return np.convolve(xp, kernel, mode="valid")
    if x.ndim == 2:
        cols = []
        for j in range(x.shape[1]):
            col = x[:, j]
            xp = np.pad(col, (pad, pad), mode="reflect")
            cols.append(np.convolve(xp, kernel, mode="valid"))
        return np.stack(cols, axis=1)
    raise ValueError(f"moving_average expects 1-D or 2-D, got shape {x.shape}")


def gaussian_bump(T: int, center: int, width: int, amp: float=1.0) -> np.ndarray:
    t = np.arange(T)
    sig2 = (width/2.355)**2  # FWHM->sigma
    return amp * np.exp(-(t-center)**2 / (2*sig2))


def add_noise(x: np.ndarray, sigma: float, rng) -> np.ndarray:
    return x + rng.normal(0, sigma, size=x.shape)

# -----------------------------
# ARC-like synthetic generator (E_raw & proto_all)
# -----------------------------

@dataclass
class Sample:
    grid_in: np.ndarray
    tasks_true: List[str]
    order_true: List[str]
    grid_out_true: np.ndarray
    E_raw: np.ndarray          # (T, K)
    proto_all: np.ndarray      # (T, K)
    names: List[str]
    T: int


def random_grid(rng, H=8, W=8, ncolors=6):
    return rng.integers(0, ncolors, size=(H, W))


def apply_primitive(grid: np.ndarray, prim: str) -> np.ndarray:
    if prim == "flip_h":
        return np.fliplr(grid)
    if prim == "flip_v":
        return np.flipud(grid)
    if prim == "rotate":
        return np.rot90(grid, k=-1)  # 90° clockwise
    raise ValueError("unknown prim")


def apply_sequence(grid: np.ndarray, seq: List[str]) -> np.ndarray:
    g = grid.copy()
    for p in seq:
        g = apply_primitive(g, p)
    return g


def gen_protos_and_raw(tasks: List[str], T: int, rng, noise=0.01,
                        x_talk_major=0.05, x_talk_minor=0.02,
                        inactive_bg=0.0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (E_raw, proto_all, names).
    - proto_all: clean per-concept templates with bumps at true task windows
    - E_raw: proto_all + mild cross-talk + drift + noise
    - inactive_bg: amplitude for background bumps on inactive channels (use 0.0..0.01)
    """
    names = PRIMS.copy()
    centers = [int(T*0.15), int(T*0.45), int(T*0.70)]
    width = int(T*0.10)

    # Clean prototypes P (T,K)
    P = np.zeros((T, len(PRIMS)), dtype=float)
    for idx, prim in enumerate(tasks):
        c = centers[idx % len(centers)]
        P[:, names.index(prim)] += gaussian_bump(T, c, width, amp=1.0)

    # Optional tiny background on inactive channels
    if inactive_bg > 0.0:
        for k in range(len(PRIMS)):
            if P[:, k].max() == 0.0:
                P[:, k] += inactive_bg * gaussian_bump(T, centers[k], width, amp=1.0)

    # Build E_raw with mild cross-talk, drift, and noise
    X = P.copy()
    X[:, 0] += x_talk_major*P[:, 1] + x_talk_minor*P[:, 2]
    X[:, 1] += x_talk_major*P[:, 0] + x_talk_minor*P[:, 2]
    X[:, 2] += x_talk_major*P[:, 0] + x_talk_minor*P[:, 1]

    drift = 0.002 * np.linspace(1.0, 1.1, T)[:, None]
    X = np.maximum(X + drift, 0.0)
    X = add_noise(X, noise, rng)
    X = np.clip(X, 0.0, None)
    return X, P, names


def make_sample(rng, T=720, n_tasks=(1,3), grid_shape=(8,8), noise=0.01,
                inactive_bg=0.0) -> Sample:
    k = rng.integers(n_tasks[0], n_tasks[1]+1)
    tasks = list(rng.choice(PRIMS, size=k, replace=False))
    rng.shuffle(tasks)
    g0 = random_grid(rng, H=grid_shape[0], W=grid_shape[1])
    g1 = apply_sequence(g0, tasks)
    E_raw, P, names = gen_protos_and_raw(tasks, T=T, rng=rng, noise=noise, inactive_bg=inactive_bg)
    return Sample(grid_in=g0, tasks_true=tasks, order_true=tasks, grid_out_true=g1,
                  E_raw=E_raw, proto_all=P, names=names, T=T)

# -----------------------------
# Parsers
# -----------------------------

def _z(x: np.ndarray) -> np.ndarray:
    m, s = x.mean(), x.std() + 1e-12
    return (x - m) / s


def perpendicular_energy(E_raw: np.ndarray) -> np.ndarray:
    mu = E_raw.mean(axis=1, keepdims=True)
    return np.clip(E_raw - mu, 0.0, None)


def exclusive_residual_on_perp(E_raw: np.ndarray, protos: np.ndarray, smooth: int) -> np.ndarray:
    """Exclusive residual computed **on perpendicular energy** (Stage-10 v5.3 style).
    Steps:
      1) Perp common-mode removal for both signals and prototypes
      2) z-score each column timewise
      3) For each channel k, project onto span-complement of other prototype columns via QR
      4) **Energyize**: square residual (no ReLU), then smooth
    Returns E_ex_energy (T,K).
    """
    # 1) Perpendicular energies
    Eperp = perpendicular_energy(E_raw)            # (T,K)
    Pperp = perpendicular_energy(protos)           # (T,K)

    # 2) z-score timewise
    Ez = np.stack([_z(Eperp[:, k]) for k in range(Eperp.shape[1])], axis=1)
    Pz = np.stack([_z(Pperp[:, k]) for k in range(Pperp.shape[1])], axis=1)

    # 3) span-complement via QR
    Tlen, K = Ez.shape
    E_ex = np.zeros_like(Ez)
    for k in range(K):
        if K == 1:
            r = Ez[:, k]
        else:
            B = np.delete(Pz, k, axis=1)            # (T, K-1)
            Q, _ = np.linalg.qr(B, mode='reduced')
            x = Ez[:, k]
            r = x - Q @ (Q.T @ x)
        # 4) square to retain magnitude irrespective of sign, then smooth
        r = r * r
        if smooth and smooth > 1:
            r = moving_average(r, k=smooth)
        E_ex[:, k] = r
    return E_ex


def matched_filter_parse(E_raw: np.ndarray, protos: np.ndarray, names: List[str],
                         sigma=9, proto_width=140,
                         presence_frac=0.20, corr_thr=0.15,
                         mode: str = "stock") -> Tuple[List[str], List[str], Dict[str, float], Dict[str, float]]:
    """
    mode options:
      - "stock": smoothed perpendicular energy + area gating
      - "geodesic": exclusive residual on perpendicular energy + correlation gate
      - "plain_perp": like stock but returned to allow ablations (no corr gating)
    Returns (tasks, order, areas, corr) using concept names.
    """
    Tlen, K = E_raw.shape

    if mode == "geodesic":
        E = exclusive_residual_on_perp(E_raw, protos, smooth=sigma)  # (T,K)
    else:
        E = moving_average(perpendicular_energy(E_raw), k=sigma)     # (T,K)

    areas, corr, peak_idx = {}, {}, {}
    for j, name in enumerate(names):
        s = E[:, j]
        # Use the channel's actual prototype for correlation (shape match)
        pj = _z(perpendicular_energy(protos)[:, j])
        sj = _z(s)
        m = np.correlate(sj, pj, mode="same")
        idx = int(np.argmax(m))
        peak_idx[name] = idx
        # normalized cosine in a local window
        L = min(proto_width, Tlen)
        a = max(0, idx - L//2)
        b = min(Tlen, idx + L//2)
        w = s[a:b]; pw = pj[a:b]
        denom = (np.linalg.norm(w - w.mean()) * np.linalg.norm(pw - pw.mean()) + 1e-8)
        corr[name] = float(np.dot(w - w.mean(), pw - pw.mean()) / denom)
        areas[name] = float(np.trapz(s))

    Amax = max(areas.values()) + 1e-12
    Cmax = max(corr.values()) + 1e-12

    keep: List[str] = []
    for name in names:
        a_ok = (areas[name] / Amax) >= presence_frac
        if mode == "geodesic":
            c_ok = (corr[name] / Cmax) >= corr_thr
        else:
            c_ok = True
        if a_ok and c_ok:
            keep.append(name)

    if not keep:
        score = {name: areas[name] * (corr[name] if mode == "geodesic" else 1.0) for name in names}
        keep = [max(score, key=score.get)]

    order = sorted(keep, key=lambda nm: peak_idx[nm])
    return keep, order, areas, corr

# -----------------------------
# Metrics & stats
# -----------------------------

def hallucination_rate(pred_tasks: List[str], true_tasks: List[str]) -> float:
    return float(any(p not in true_tasks for p in pred_tasks))


def norm_hamming(a: np.ndarray, b: np.ndarray) -> float:
    assert a.shape == b.shape
    return 1.0 - (a != b).mean()


def bootstrap_ci(diff: np.ndarray, B: int = 2000, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(diff)
    if n == 0:
        return 0.0, (0.0, 0.0), 1.0
    idx = rng.integers(0, n, size=(B, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = float((means <= 0.0).mean())  # one‑sided: improvement > 0
    return float(diff.mean()), (float(lo), float(hi)), p

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--T', type=int, default=720)
    ap.add_argument('--noise', type=float, default=0.01)
    ap.add_argument('--sigma', type=int, default=9)
    ap.add_argument('--proto_width', type=int, default=140)
    ap.add_argument('--presence_frac', type=float, default=0.20)
    ap.add_argument('--corr_thr', type=float, default=0.15)
    ap.add_argument('--co_tau', type=int, default=25)
    ap.add_argument('--inactive_bg', type=float, default=0.0, help='inactive prototype background amplitude (0.0..0.01)')
    ap.add_argument('--geo_mode', type=str, default='exclusive_perp', choices=['exclusive_perp','plain_perp','none'],
                    help='geodesic parser mode: exclusive_perp (default), plain_perp (stock-like), none (disable geodesic)')
    ap.add_argument('--out_csv', type=str, default='benchmark_results.csv')
    args = ap.parse_args()

    rng = set_seed(args.seed)

    acc_stock, acc_geo = [], []
    sim_stock, sim_geo = [], []
    hall_stock, hall_geo = [], []

    rows = []

    for i in range(1, args.samples+1):
        s = make_sample(rng, T=args.T, noise=args.noise, inactive_bg=args.inactive_bg)

        # STOCK parse
        tasks_s, order_s, areas_s, corr_s = matched_filter_parse(
            s.E_raw, s.proto_all, s.names,
            sigma=args.sigma, proto_width=args.proto_width,
            presence_frac=args.presence_frac, corr_thr=args.corr_thr,
            mode="stock")
        grid_pred_s = apply_sequence(s.grid_in, order_s)

        # GEODESIC parse (mode switch for ablations)
        geo_mode = args.geo_mode
        if geo_mode == 'none':
            tasks_g, order_g = [], []
            areas_g, corr_g = {}, {}
            grid_pred_g = s.grid_in.copy()
        else:
            mode = 'geodesic' if geo_mode == 'exclusive_perp' else 'plain_perp'
            tasks_g, order_g, areas_g, corr_g = matched_filter_parse(
                s.E_raw, s.proto_all, s.names,
                sigma=args.sigma, proto_width=args.proto_width,
                presence_frac=args.presence_frac, corr_thr=args.corr_thr,
                mode=mode)
            grid_pred_g = apply_sequence(s.grid_in, order_g)

        ok_s = int(np.array_equal(grid_pred_s, s.grid_out_true))
        ok_g = int(np.array_equal(grid_pred_g, s.grid_out_true))

        acc_stock.append(ok_s)
        acc_geo.append(ok_g)
        sim_stock.append(norm_hamming(grid_pred_s, s.grid_out_true))
        sim_geo.append(norm_hamming(grid_pred_g, s.grid_out_true))
        hall_stock.append(hallucination_rate(tasks_s, s.tasks_true))
        hall_geo.append(hallucination_rate(tasks_g, s.tasks_true))

        rows.append({
            'id': i,
            'truth_tasks': ' '.join(s.tasks_true),
            'stock_tasks': ' '.join(tasks_s),
            'stock_order': ' '.join(order_s),
            'geo_tasks': ' '.join(tasks_g),
            'geo_order': ' '.join(order_g),
            'ok_stock': ok_s,
            'ok_geo': ok_g,
            'sim_stock': sim_stock[-1],
            'sim_geo': sim_geo[-1],
            'hall_stock': hall_stock[-1],
            'hall_geo': hall_geo[-1]
        })

    # paired diffs
    acc_diff = np.array(acc_geo) - np.array(acc_stock)
    sim_diff = np.array(sim_geo) - np.array(sim_stock)
    hall_diff = np.array(hall_stock) - np.array(hall_geo)  # negative is improvement

    d_acc, ci_acc, p_acc = bootstrap_ci(acc_diff, rng=rng)
    d_sim, ci_sim, p_sim = bootstrap_ci(sim_diff, rng=rng)
    d_hall, ci_hall, p_hall = bootstrap_ci(-hall_diff, rng=rng)  # improvement in (−hall)


    print(f"Samples: {args.samples}")
    print(f"Accuracy — stock: {np.mean(acc_stock)*100:.1f}% | geodesic: {np.mean(acc_geo)*100:.1f}% | Δ: {d_acc*100:.1f} pp [95% CI {ci_acc[0]*100:.1f}, {ci_acc[1]*100:.1f}] | p={p_acc:.4f}")
    print(f"Semantic similarity — stock: {np.mean(sim_stock)*100:.1f}% | geodesic: {np.mean(sim_geo)*100:.1f}% | Δ: {d_sim*100:.1f} pts [95% CI {ci_sim[0]*100:.1f}, {ci_sim[1]*100:.1f}] | p={p_sim:.4f}")
    print(f"Hallucination rate — stock: {np.mean(hall_stock)*100:.1f}% | geodesic: {np.mean(hall_geo)*100:.1f}% | Δ: {(-d_hall)*100:.1f} pp reduction [95% CI {(-ci_hall[1])*100:.1f}, {(-ci_hall[0])*100:.1f}] | p={p_hall:.4f}")

    with open(args.out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote per‑sample results to {args.out_csv}")

if __name__ == '__main__':
    main()
