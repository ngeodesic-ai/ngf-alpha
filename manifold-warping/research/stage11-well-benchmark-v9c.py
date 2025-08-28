#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 11 — Well Benchmark v9c
-----------------------------
This version bakes curvature into BOTH:
  (1) the visualization (analytic 360° polar funnel), and
  (2) the parsing energy path (radial potential + radius reweighting).

Defaults are baked-in per discussion, but exposed via CLI:
  kappa=0.15, p=1.6, beta=0.4, q=2

CSV/JSON schema is unchanged (geodesic vs stock), with an extra
diagnostic block `geometry_v9c` added to the JSON.

python3 stage11-well-benchmark-v9c.py \
  --samples 200 --seed 42 --T 720 --sigma 9 \
  --out_plot manifold_pca3_mesh_warped.png \
  --out_csv stage11_metrics.csv \
  --out_json stage11_summary.json

python3 stage11-well-benchmark-v9c.py \
  --samples 200 --seed 42 --T 720 --sigma 9 \
  --out_plot manifold_pca3_mesh_warped.png \
  --out_csv stage11_metrics.csv \
  --out_json stage11_summary.json \
  --kappa 0.00 \
  --beta 0.20 --q 2     

python3 stage11-well-benchmark-v9c.py \
  --samples 200 --seed 42 --sigma 9 \
  --out_plot manifold_pca3_mesh_warped.png \
  --out_csv stage11_metrics.csv \
  --out_json stage11_summary.json \
  --kappa 0.0 --beta 0.20 --q 2 \
  --funnel_D 0.4 --funnel_p 1.2

"""

import argparse, json, csv, math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay

# ----------------------------
# Synthetic generator (baseline)
# ----------------------------

PRIMS = ["flip_h","flip_v","rotate"]

def moving_average(x, k=9):
    if k <= 1: return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(xp, np.ones(k)/k, mode="valid")

def gaussian_bump(T, center, width, amp=1.0):
    t = np.arange(T)
    sig2 = (width/2.355)**2
    return amp * np.exp(-(t-center)**2 / (2*sig2))

def common_mode(traces: Dict[str,np.ndarray]) -> np.ndarray:
    return np.stack([traces[p] for p in PRIMS], 0).mean(0)

def perpendicular_energy(traces: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
    mu = common_mode(traces)
    return {p: np.clip(traces[p] - mu, 0, None) for p in PRIMS}

def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-8)

def _rng(seed: int):
    return np.random.default_rng(seed)

def make_synthetic_traces(rng, T=720, noise=0.02, cm_amp=0.02, overlap=0.5,
                          amp_jitter=0.4, distractor_prob=0.4, tasks_k: Tuple[int,int]=(1,3)):
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
# Features for viz (H/E over dataset)
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
# RAW PCA warp for viz
# ----------------------------

@dataclass
class WellParams:
    whiten: bool = True
    tau: float = 0.25
    isotropize_xy: bool = True
    sigma_scale: float = 0.80
    depth_scale: float = 1.35
    mix_z: float = 0.12

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

def pca3_and_warp(H: np.ndarray, energy: Optional[np.ndarray], params: WellParams):
    pca = PCA(n_components=3, whiten=params.whiten, random_state=0)
    X3 = pca.fit_transform(H)
    X2 = X3[:, :2]
    z  = X3[:, 2].copy()

    c_soft, _ = _softmin_center(X2, energy, params.tau)
    X2c = X2 - c_soft
    if params.isotropize_xy:
        X2c, _ = _isotropize(X2c)

    # Gaussian bowl + small PC3 blend (baseline look)
    r = np.linalg.norm(X2c, axis=1)
    sigma = np.median(r) * params.sigma_scale + 1e-9
    z_bowl = -params.depth_scale * np.exp(-(r**2)/(2*sigma**2))
    z_new  = z_bowl + params.mix_z * (z - z.mean())

    X3_out = np.column_stack([X2c + c_soft, z_new])
    info = dict(center=c_soft, sigma=sigma)
    return X3_out, info

# ----------------------------
# Analytic 360° funnel (viz)
# ----------------------------

def analytic_funnel(center, r_max, *, D=2.0, p=1.6, z_cap=0.0, eps=1e-3, n_theta=160, n_r=220):
    """z(r) = z_cap - D*(1/(r+eps)^p - 1/(r_max+eps)^p), revolved 360°."""
    r_grid = np.linspace(0.0, r_max, n_r)
    invp = (1.0 / (r_grid + eps))**p
    invp -= (1.0 / (r_max + eps))**p  # normalize so z(r_max) = z_cap
    z_prof = z_cap - D * invp

    # build surface
    theta = np.linspace(0, 2*np.pi, n_theta)
    R, TH = np.meshgrid(r_grid, theta)
    X = center[0] + R * np.cos(TH)
    Y = center[1] + R * np.sin(TH)
    Z = z_prof[None, :].repeat(n_theta, axis=0)
    return X, Y, Z, r_grid, z_prof

def plot_polar_surface(X, Y, Z, points: np.ndarray, energy: Optional[np.ndarray], title: str, out_path: str, alpha=0.85):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=alpha, linewidth=0, antialiased=True)
    if points is not None:
        x, y, z = points[:,0], points[:,1], points[:,2]
        c = None if energy is None else (energy - np.min(energy)) / (np.ptp(energy) + 1e-9)
        ax.scatter(x, y, z, c=c, cmap='viridis', s=10, alpha=0.7)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title(title)
    fig.colorbar(surf, ax=ax, shrink=0.6, label="height")
    plt.tight_layout(); fig.savefig(out_path, dpi=220); plt.close(fig)

# ----------------------------
# Parser with curvature coupling (metrics path)
# ----------------------------

def half_sine_proto(width):
    p = np.sin(np.linspace(0, np.pi, width))
    return p / (np.linalg.norm(p)+1e-8)

def radius_from_sample_energy(S: Dict[str,np.ndarray]) -> np.ndarray:
    """Compute per-time radius via PCA of the (flip_h, flip_v, rotate) energy triplet."""
    T = len(next(iter(S.values())))
    M = np.stack([_z(S[p]) for p in PRIMS], axis=1)  # (T,3)
    M = M - M.mean(axis=0, keepdims=True)
    U = PCA(n_components=2, random_state=0).fit_transform(M)
    U = U - U.mean(axis=0, keepdims=True)
    r = np.linalg.norm(U, axis=1)
    R = r.max() + 1e-9
    return r / R  # normalize to [0,~1]

def geodesic_parse_curved(traces, *, sigma=9, proto_width=160, kappa=0.15, p=1.6, beta=0.4, q=2):
    """Stage-11 geodesic parse with radial potential + radius reweighting baked in."""
    keys = list(traces.keys())
    Eres = perpendicular_energy(traces)
    Sres = {pname: moving_average(Eres[pname], k=sigma) for pname in keys}

    # per-time radius from sample PCA (T x 3 energies)
    r = radius_from_sample_energy(Sres)  # in [0,1]
    eps = 1e-6
    phi = 1.0 / np.power(r + eps, p)          # radial potential (big near center)
    w   = 1.0 + beta * np.power(1.0 - r, q)   # weight > 1 near center

    # apply coupling
    Snew = {}
    for pname in keys:
        Snew[pname] = w * Sres[pname] + kappa * phi

    # matched-filter selection
    proto = half_sine_proto(proto_width)
    peak = {pname: int(np.argmax(np.correlate(Snew[pname], proto, mode="same"))) for pname in keys}
    score = {pname: float(np.max(np.correlate(Snew[pname], proto, mode="same"))) for pname in keys}
    smax = max(score.values()) + 1e-12
    keep = [pname for pname in keys if score[pname] >= 0.5*smax]
    if not keep:
        keep = [max(keys, key=lambda k: score[k])]
    order = sorted(keep, key=lambda pname: peak[pname])
    return keep, order

def stock_parse(traces, sigma=9, proto_width=160):
    keys = list(traces.keys())
    S = {p: moving_average(traces[p], k=sigma) for p in keys}
    proto = half_sine_proto(proto_width)
    peak = {p: int(np.argmax(np.correlate(S[p], proto, mode="same"))) for p in keys}
    score = {p: float(np.max(np.correlate(S[p], proto, mode="same"))) for p in keys}
    smax = max(score.values()) + 1e-12
    keep = [p for p in keys if score[p] >= 0.6*smax]
    if not keep: keep = [max(keys, key=lambda p: score[p])]
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
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage 11 — baseline + v9c curvature coupling and analytic funnel viz")
    # data
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
    # outputs
    ap.add_argument("--out_plot", type=str, default="manifold_pca3_mesh_warped.png")
    ap.add_argument("--out_csv", type=str, default="stage11_metrics.csv")
    ap.add_argument("--out_json", type=str, default="stage11_summary.json")
    # viz params
    ap.add_argument("--sigma_scale", type=float, default=0.80)
    ap.add_argument("--depth_scale", type=float, default=1.35)
    ap.add_argument("--mix_z", type=float, default=0.12)
    ap.add_argument("--funnel_D", type=float, default=2.0)
    ap.add_argument("--funnel_p", type=float, default=1.6)
    ap.add_argument("--funnel_cap", type=float, default=0.0)
    ap.add_argument("--n_theta", type=int, default=160)
    ap.add_argument("--n_r", type=int, default=220)
    # coupling parameters (baked defaults)
    ap.add_argument("--kappa", type=float, default=0.15, help="radial potential strength")
    ap.add_argument("--p", type=float, default=1.6, help="radial potential power")
    ap.add_argument("--beta", type=float, default=0.4, help="radius reweight gain")
    ap.add_argument("--q", type=int, default=2, help="radius reweight power")
    args = ap.parse_args()

    # ----- metrics run (curvature-coupled geodesic) -----
    rng = _rng(args.seed)
    rows = []
    agg_geo = dict(acc=0, P=0, R=0, F1=0, J=0, H=0, O=0)
    agg_stock = dict(acc=0, P=0, R=0, F1=0, J=0, H=0, O=0)

    for i in range(1, args.samples+1):
        traces, true_order = make_synthetic_traces(
            rng, T=args.T, noise=args.noise, cm_amp=args.cm_amp, overlap=args.overlap,
            amp_jitter=args.amp_jitter, distractor_prob=args.distractor_prob,
            tasks_k=(args.min_tasks, args.max_tasks)
        )
        keep_g, order_g = geodesic_parse_curved(
            traces, sigma=args.sigma, proto_width=160,
            kappa=args.kappa, p=args.p, beta=args.beta, q=args.q
        )
        keep_s, order_s = stock_parse(traces, sigma=args.sigma)
        acc_g = int(order_g == true_order); acc_s = int(order_s == true_order)
        sm_g = set_metrics(true_order, keep_g); sm_s = set_metrics(true_order, keep_s)
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

    # ----- Visualization build: raw + analytic funnel -----
    H, E = build_H_E_from_traces(args)
    base_params = WellParams(sigma_scale=args.sigma_scale, depth_scale=args.depth_scale, mix_z=args.mix_z)
    X3_raw, info = pca3_and_warp(H, energy=E, params=base_params)
    # RAW trisurf for reference
    fig = plt.figure(figsize=(10, 8)); ax = fig.add_subplot(111, projection='3d')
    x, y, z = X3_raw[:,0], X3_raw[:,1], X3_raw[:,2]
    if len(X3_raw) >= 4:
        tri = Delaunay(np.column_stack([x, y]))
        ax.plot_trisurf(x, y, z, triangles=tri.simplices, cmap='viridis', alpha=0.65, linewidth=0.2, antialiased=True)
    c = (E - np.min(E)) / (np.ptp(E) + 1e-9)
    ax.scatter(x, y, z, c=c, cmap='viridis', s=12, alpha=0.85)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title("Stage 11 — Warped Single Well (Raw)")
    plt.tight_layout(); fig.savefig(args.out_plot, dpi=220); plt.close(fig)

    # Analytic funnel (360°)
    # radius scale from cloud
    r_cloud = np.linalg.norm((X3_raw[:,:2] - info["center"]), axis=1)
    r_max = float(np.quantile(r_cloud, 0.98))  # robust rim
    Xs, Ys, Zs, r_grid, z_prof = analytic_funnel(
        info["center"], r_max, D=2.0, p=args.funnel_p, z_cap=args.funnel_cap,
        n_theta=args.n_theta, n_r=args.n_r
    )
    out_polar = args.out_plot.replace(".png", "_v9c_funnel.png")
    plot_polar_surface(Xs, Ys, Zs, X3_raw, E, "Stage 11 — Analytic Funnel (v9c, 360°)", out_polar)

    # Diagnostics vs inverse-power profile
    phi_grid = 1.0 / np.power(r_grid + 1e-6, args.funnel_p)
    phi_grid = (phi_grid - phi_grid.min()) / (phi_grid.max() - phi_grid.min() + 1e-12)
    # Fit z_prof to a + b * phi
    A = np.c_[phi_grid, np.ones_like(phi_grid)]
    try:
        coef, *_ = np.linalg.lstsq(A, z_prof, rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((z_prof - pred)**2))
        ss_tot = float(np.sum((z_prof - np.mean(z_prof))**2))
        r2 = 1.0 - ss_res / max(1e-12, ss_tot)
    except Exception:
        r2 = float("nan")
    # Apex sharpness
    r5 = float(np.quantile(r_cloud, 0.05)); r25 = float(np.quantile(r_cloud, 0.25))
    apex = float(np.mean(X3_raw[np.linalg.norm((X3_raw[:,:2] - info["center"]), axis=1) <= r5, 2]))
    inner= float(np.mean(X3_raw[(r_cloud > r5) & (r_cloud <= r25), 2]))
    apex_sharp = float(inner - apex)

    # ----- Outputs -----
    if rows:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for rrow in rows: w.writerow(rrow)

    summary = dict(
        samples=int(n),
        geodesic=Sg, stock=Ss,
        center=[float(c) for c in info["center"]],
        sigma=float(info["sigma"]),
        plot_raw=args.out_plot,
        plot_funnel=out_polar,
        geometry_v9c=dict(
            r2_vs_invpower=float(r2),
            apex_sharpness=float(apex_sharp),
            params=dict(kappa=args.kappa, p=args.p, beta=args.beta, q=args.q,
                        funnel_p=args.funnel_p)
        )
    )
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("[SUMMARY] Geodesic:", {k: round(v,3) for k,v in Sg.items()})
    print("[SUMMARY] Stock   :", {k: round(v,3) for k,v in Ss.items()})
    print(f"[PLOT] RAW:    {args.out_plot}")
    print(f"[PLOT] FUNNEL: {out_polar}")
    print(f"[CSV ] {args.out_csv}")
    print(f"[JSON] {args.out_json}")

if __name__ == "__main__":
    main()
