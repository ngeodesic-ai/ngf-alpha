#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 11 — Well Benchmark v9a
-----------------------------
Target: eliminate flat-bottom basin and approach the "Curvature Engine" / logo look.

Visual-only warp additions (metrics unchanged):
  1) Conical radial pull on PC3:           z <- z - k_r * sqrt(r + eps)
  2) Energy-boosted throat near apex:      z <- z - k_e * (E^gamma) * exp(-r/rho)
  3) Radial monotonicity enforcement:      average z(r) in bins then cumulative-min toward center
  4) Optional Laplacian smoothing on z to hide trisurf facets

Emits both RAW and CONE views plus shape KPIs (R^2 vs sqrt(r), apex sharpness).

Outputs:
  - manifold_pca3_mesh_warped.png              (RAW baseline)
  - manifold_pca3_mesh_warped_v9a_cone.png     (cone+energy+monotone+smooth)
  - stage11_metrics.csv
  - stage11_summary.json (adds diagnostics under "geometry_v9a")

python3 stage11-well-benchmark-v9a.py \
  --samples 200 --seed 42 --T 720 --sigma 9 \
  --cone_k 0.35 --energy_k 0.15 --rho 0.6 --energy_gamma 1.7 \
  --monotone_bins 128 --smooth_iters 1 --smooth_lambda 0.12 \
  --out_plot manifold_pca3_mesh_warped.png \
  --out_csv stage11_metrics.csv \
  --out_json stage11_summary.json
  
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

def common_mode(traces: Dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([traces[p] for p in PRIMS], 0).mean(0)

def perpendicular_energy(traces: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    mu = common_mode(traces)
    return {p: np.clip(traces[p] - mu, 0, None) for p in PRIMS}

def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-8)

def _rng(seed: int):
    return np.random.default_rng(seed)

def make_synthetic_traces(rng, T=720, noise=0.02, cm_amp=0.02, overlap=0.5,
                          amp_jitter=0.4, distractor_prob=0.4, tasks_k: Tuple[int,int]=(1,3)) -> Tuple[Dict[str,np.ndarray], List[str]]:
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
# Features (H/E) — baseline
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
# PCA warp (baseline, RAW)
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

    # energy-aware center + isotropize
    c_soft, _ = _softmin_center(X2, energy, params.tau)
    X2c = X2 - c_soft
    if params.isotropize_xy:
        X2c, _ = _isotropize(X2c)

    # Gaussian bowl + small PC3 blend
    r = np.linalg.norm(X2c, axis=1)
    sigma = np.median(r) * params.sigma_scale + 1e-9
    z_bowl = -params.depth_scale * np.exp(-(r**2)/(2*sigma**2))
    z_new  = z_bowl + params.mix_z * (z - z.mean())

    X3_out = np.column_stack([X2c + c_soft, z_new])
    info = dict(center=c_soft, sigma=sigma)
    return X3_out, info

# ----------------------------
# v9a cone + energy + monotone + smoothing (viz only)
# ----------------------------

def energy_weighted_center(X2: np.ndarray, E: np.ndarray) -> np.ndarray:
    # Weight toward lower energy (deeper basin)
    w = (E - E.min()) / (E.ptp() + 1e-9) + 1e-6
    w = 1.0 / w
    w = w / (w.sum() + 1e-12)
    return (w[:, None] * X2).sum(axis=0)

def cone_energy_warp(X3_raw: np.ndarray, E: np.ndarray, *, k_r: float, k_e: float, rho: float, gamma: float, eps: float=1e-6):
    pc1, pc2, pc3 = X3_raw[:,0], X3_raw[:,1], X3_raw[:,2]

    # robust recenter
    c = energy_weighted_center(X3_raw[:,:2], E)
    x, y = pc1 - c[0], pc2 - c[1]
    r = np.sqrt(x*x + y*y)

    # normalized energy and power-law accent
    q10, q90 = np.quantile(E, 0.10), np.quantile(E, 0.90)
    En = np.clip((E - q10) / max(1e-8, (q90 - q10)), 0, 1)
    Eg = En**gamma

    # conical pull + energy accent near apex
    pc3_cone = pc3 - k_r * np.sqrt(r + eps)
    pc3_warp = pc3_cone - k_e * Eg * np.exp(-r / max(rho, eps))

    X3_cone = np.c_[pc1, pc2, pc3_warp]
    return X3_cone, r, c

def enforce_radial_monotone(X3_cone: np.ndarray, r: np.ndarray, bins: int = 128) -> np.ndarray:
    """Make z(r) monotonically decreasing toward center to avoid flat shelves."""
    pc1, pc2, z = X3_cone[:,0], X3_cone[:,1], X3_cone[:,2].copy()
    # bin radii
    rmin, rmax = float(r.min()), float(r.max())
    edges = np.linspace(rmin, rmax, bins+1)
    idx   = np.clip(np.digitize(r, edges)-1, 0, bins-1)

    # avg z per bin
    zbin = np.full(bins, np.nan)
    for b in range(bins):
        m = (idx == b)
        if np.any(m):
            zbin[b] = float(np.mean(z[m]))

    # cumulative minimum from outer rim (high r) to center (low r)
    # so that as we go inward, depth never gets shallower
    # Work on finite bins only
    finite = np.where(~np.isnan(zbin))[0]
    if len(finite) > 0:
        zb = zbin.copy()
        last = zb[finite[-1]]
        for k in range(len(finite)-2, -1, -1):
            i = finite[k]
            if np.isnan(last): 
                last = zb[i]
            else:
                if zb[i] > last:
                    zb[i] = last
                else:
                    last = zb[i]
        zbin = np.where(np.isnan(zbin), zb, zb)

    # map back to points
    z_new = z.copy()
    for b in range(bins):
        m = (idx == b)
        if np.any(m) and not np.isnan(zbin[b]):
            z_new[m] = zbin[b]

    return np.c_[pc1, pc2, z_new]

def laplacian_smooth_z(X3: np.ndarray, iters:int=1, lam:float=0.12) -> np.ndarray:
    """One-ring Laplacian smoothing on z using 2D Delaunay neighbors."""
    pts2 = X3[:, :2]
    z = X3[:, 2].copy()
    if len(pts2) < 4:
        return X3
    tri = Delaunay(pts2)
    # build adjacency
    N = len(pts2)
    adj = [set() for _ in range(N)]
    for a,b,c in tri.simplices:
        adj[a].update([b,c]); adj[b].update([a,c]); adj[c].update([a,b])
    for _ in range(max(0, iters)):
        z_new = z.copy()
        for i in range(N):
            if adj[i]:
                z_new[i] = z[i] + lam * (np.mean([z[j] for j in adj[i]]) - z[i])
        z = z_new
    return np.c_[pts2, z]

# ----------------------------
# Plotting
# ----------------------------

def plot_trisurf(X3: np.ndarray, energy: Optional[np.ndarray], title:str, out_path:str, alpha=0.65, point_alpha=0.85):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = X3[:,0], X3[:,1], X3[:,2]
    c = None if energy is None else (energy - np.min(energy)) / (np.ptp(energy) + 1e-9)

    tri = None
    if len(X3) >= 4:
        tri = Delaunay(np.column_stack([x, y]))
        ax.plot_trisurf(x, y, z, triangles=tri.simplices, cmap='viridis', alpha=alpha, linewidth=0.2, antialiased=True)
    ax.scatter(x, y, z, c=c, cmap='viridis', s=12, alpha=point_alpha)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title(title)
    if energy is not None:
        mappable = plt.cm.ScalarMappable(cmap='viridis')
        mappable.set_array(c)
        cb = fig.colorbar(mappable, ax=ax)
        cb.set_label("Energy (norm)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

# ----------------------------
# Minimal metrics harness (unchanged logic)
# ----------------------------

def half_sine_proto(width):
    p = np.sin(np.linspace(0, np.pi, width))
    return p / (np.linalg.norm(p)+1e-8)

def geodesic_parse(traces, sigma=9, proto_width=160):
    keys = list(traces.keys())
    Eres = perpendicular_energy(traces)
    Sres = {p: moving_average(Eres[p], k=sigma) for p in keys}
    proto = half_sine_proto(proto_width)
    peak = {p: int(np.argmax(np.correlate(Sres[p], proto, mode="same"))) for p in keys}
    score = {p: float(np.max(np.correlate(Sres[p], proto, mode="same"))) for p in keys}
    smax = max(score.values()) + 1e-12
    keep = [p for p in keys if score[p] >= 0.5*smax]
    if not keep: keep = [max(keys, key=lambda p: score[p])]
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
    ap = argparse.ArgumentParser(description="Stage 11 — report baseline with v9a cone/throat viz (non-flat basin)")
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
    # cone/energy/monotone/smooth params
    ap.add_argument("--cone_k", type=float, default=0.35, help="k_r: conical depth strength")
    ap.add_argument("--energy_k", type=float, default=0.15, help="k_e: energy accent near apex")
    ap.add_argument("--rho", type=float, default=0.6, help="radius falloff for energy accent (PC units)")
    ap.add_argument("--energy_gamma", type=float, default=1.7, help="power for energy remap")
    ap.add_argument("--monotone_bins", type=int, default=128, help="radial bins for monotone enforcement")
    ap.add_argument("--smooth_iters", type=int, default=1, help="Laplacian smoothing iterations")
    ap.add_argument("--smooth_lambda", type=float, default=0.12, help="Laplacian smoothing strength")
    # baseline warp params
    ap.add_argument("--sigma_scale", type=float, default=0.80)
    ap.add_argument("--depth_scale", type=float, default=1.35)
    ap.add_argument("--mix_z", type=float, default=0.12)
    args = ap.parse_args()

    # ----- metrics pass (baseline logic, unchanged) -----
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
        keep_g, order_g = geodesic_parse(traces, sigma=args.sigma)
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

    # ----- Visualization build: RAW then CONE -----
    H, E = build_H_E_from_traces(args)
    base_params = WellParams(sigma_scale=args.sigma_scale, depth_scale=args.depth_scale, mix_z=args.mix_z)
    X3_raw, info = pca3_and_warp(H, energy=E, params=base_params)
    plot_trisurf(X3_raw, E, "Stage 11 — Warped Single Well (Raw)", args.out_plot)

    # Cone + energy accent
    X3_cone, r, center_energy = cone_energy_warp(
        X3_raw, E,
        k_r=args.cone_k, k_e=args.energy_k, rho=args.rho, gamma=args.energy_gamma
    )

    # Enforce non-flat basin
    X3_mono = enforce_radial_monotone(X3_cone, r, bins=args.monotone_bins)

    # Laplacian smoothing on z
    X3_smooth = laplacian_smooth_z(X3_mono, iters=args.smooth_iters, lam=args.smooth_lambda)

    # KPI: R^2 of z vs sqrt(r), Apex sharpness (Δz inner ring vs apex)
    pc1, pc2, zf = X3_smooth[:,0], X3_smooth[:,1], X3_smooth[:,2]
    s = np.sqrt(r + 1e-6).reshape(-1,1)
    A = np.c_[s, np.ones_like(s)]
    try:
        coef, *_ = np.linalg.lstsq(A, zf.reshape(-1,1), rcond=None)
        pred = A @ coef
        ss_res = float(((zf.reshape(-1,1) - pred)**2).sum())
        ss_tot = float(((zf - zf.mean())**2).sum())
        r2 = 1.0 - ss_res / max(1e-12, ss_tot)
    except Exception:
        r2 = float("nan")
    r5, r25 = np.quantile(r, 0.05), np.quantile(r, 0.25)
    mask_apex  = (r <= r5)
    mask_inner = (r > r5) & (r <= r25)
    apex = float(zf[mask_apex].mean()) if np.any(mask_apex) else float("nan")
    inner= float(zf[mask_inner].mean()) if np.any(mask_inner) else float("nan")
    apex_sharp = float(inner - apex) if (not np.isnan(apex) and not np.isnan(inner)) else float("nan")

    out_cone = args.out_plot.replace(".png", "_v9a_cone.png")
    plot_trisurf(X3_smooth, E, "Stage 11 — Warped Single Well (v9a: cone+energy+monotone+smooth)", out_cone)

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
        plot_cone=out_cone,
        geometry_v9a=dict(
            r2_vs_sqrt_r=float(r2),
            apex_sharpness=float(apex_sharp),
            energy_center=[float(center_energy[0]), float(center_energy[1])],
            params=dict(cone_k=args.cone_k, energy_k=args.energy_k, rho=args.rho,
                        energy_gamma=args.energy_gamma, monotone_bins=args.monotone_bins,
                        smooth_iters=args.smooth_iters, smooth_lambda=args.smooth_lambda)
        )
    )
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("[SUMMARY] Geodesic:", {k: round(v,3) for k,v in Sg.items()})
    print("[SUMMARY] Stock   :", {k: round(v,3) for k,v in Ss.items()})
    print(f"[PLOT] RAW:   {args.out_plot}")
    print(f"[PLOT] CONE:  {out_cone}")
    print(f"[CSV ] {args.out_csv}")
    print(f"[JSON] {args.out_json}")

if __name__ == "__main__":
    main()
