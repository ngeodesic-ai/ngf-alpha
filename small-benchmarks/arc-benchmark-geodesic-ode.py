
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC Stage-10 v2 - ODE-integrated traces (Geodesic vs Stock) — thresholds + presence CSV
---------------------------------------------------------------------------------------
Upgrades:
- Uses fixed Stage-10v2 thresholds for presence: tau_area (default 10.0), tau_corr (default 0.7)
- Logs per-primitive presence metrics (area, max corr, label) to CSV
- Computes AUC-ROC and AUC-PR for presence if scikit-learn is available (no plots as requested)

Usage example:
  python3 arc-benchmark-geodesic-ode.py --samples 200 --seed 43 --T 720 \\
    --dim 19 --noise 0.02 --plot_dir plots_ode --tau_area 10 --tau_corr 0.7 \\
    --presence_csv presence_metrics.csv

Author: ngeodesic - 2025-08-25
"""
import argparse, os, csv
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

PRIMS = ["flip_h", "flip_v", "rotate"]

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def set_seed(seed: int):
    return np.random.default_rng(seed)

def moving_average(x, k=9):
    if k <= 1:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    kernel = np.ones(k) / k
    y = np.convolve(xp, kernel, mode="valid")
    return y

def _z(x):
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    sd = x.std() + 1e-12
    return (x - mu) / sd

def half_sine(width):
    t = np.linspace(0, np.pi, width)
    return np.sin(t)

def orthonormal_prototypes(rng, d: int, K: int):
    A = rng.standard_normal((d, K))
    Q, _ = np.linalg.qr(A)  # (d,K), columns orthonormal
    return Q

def gradient_U(y, t_idx, pk, ck, weights):
    d, K = pk.shape
    grad = np.zeros(d, dtype=float)
    for k in range(K):
        w = weights[k]
        if w <= 0: 
            continue
        p = pk[:, k]
        c = ck[:, k]
        diff = y - c
        proj = p * np.dot(p, diff)
        gk = diff - proj
        grad += w * gk
    return grad

def semi_implicit_euler(y0, v0, T, dt, m, lam, gamma, pk, ck, weight_schedule):
    d = y0.shape[0]
    Y = np.zeros((T, d), dtype=float)
    v = v0.copy()
    y = y0.copy()
    alpha = (1.0 - gamma * dt / (2.0 * m))
    for t in range(T):
        g_t = gradient_U(y, t, pk, ck, weight_schedule[t])
        v_half = alpha * v - (lam * dt / (2.0 * m)) * g_t
        y_next = y + dt * v_half
        g_next = gradient_U(y_next, t, pk, ck, weight_schedule[t])
        v_next = alpha * v_half - (lam * dt / (2.0 * m)) * g_next
        Y[t] = y_next
        y, v = y_next, v_next
    return Y

@dataclass
class Sample:
    grid_in: np.ndarray
    tasks_true: List[str]
    order_true: List[str]
    grid_out_true: np.ndarray
    traces: Dict[str, np.ndarray]
    T: int

def random_grid(rng, H=8, W=8, ncolors=5):
    return rng.integers(0, ncolors, size=(H, W))

def apply_primitive(grid, prim):
    if prim == "flip_h":
        return np.fliplr(grid)
    if prim == "flip_v":
        return np.flipud(grid)
    if prim == "rotate":
        return np.rot90(grid, k=-1)
    raise ValueError("unknown primitive")

def apply_sequence(grid, seq):
    g = grid.copy()
    for p in seq:
        g = apply_primitive(g, p)
    return g

def make_weight_schedule(T, order: List[str], overlap_prob, rng, jitter_frac=0.03):
    K = len(PRIMS)
    w = np.zeros((T, K), dtype=float)
    centers = [int(T * 0.20), int(T * 0.50), int(T * 0.78)]
    base_width = int(T * 0.14)
    if rng.random() < overlap_prob and len(order) >= 2:
        start = int(T * 0.25); gap = int(T * 0.10)
        centers = [start + i * gap for i in range(len(order))]
    for idx, prim in enumerate(order):
        k = PRIMS.index(prim)
        c = centers[idx % len(centers)]
        c += int(rng.normal(0, jitter_frac * T))
        width = int(base_width * rng.uniform(0.8, 1.3))
        a = max(0, c - width // 2); b = min(T, c + width // 2)
        ramp = int(0.15 * width)
        for t in range(a, b):
            if t < a + ramp:       w[t, k] = (t - a) / max(1, ramp)
            elif t > b - ramp:     w[t, k] = (b - t) / max(1, ramp)
            else:                  w[t, k] = 1.0
    row_sum = np.maximum(w.sum(axis=1, keepdims=True), 1e-8)
    w = np.minimum(w / row_sum, 1.0)
    return w

def make_sample_with_ode(rng, T=720, d=19, n_tasks=(1,3), noise=0.02,
                         m=4.0, lam=0.35, gamma=0.04, dt=0.02,
                         overlap_prob=0.6):
    k = rng.integers(n_tasks[0], n_tasks[1] + 1)
    tasks = list(rng.choice(PRIMS, size=k, replace=False))
    rng.shuffle(tasks)
    pk = orthonormal_prototypes(rng, d=d, K=len(PRIMS))
    ck = np.zeros((d, len(PRIMS)), dtype=float)
    Wt = make_weight_schedule(T, tasks, overlap_prob=overlap_prob, rng=rng)
    y0 = rng.standard_normal(d) * 0.1
    v0 = rng.standard_normal(d) * 0.05
    Y = semi_implicit_euler(y0, v0, T, dt, m, lam, gamma, pk, ck, Wt)
    # energies
    E_par = {}; E_perp = {}
    for i, p in enumerate(PRIMS):
        pvec = pk[:, i]
        s = (Y - ck[:, i]).dot(pvec)
        E_par[p] = s**2
        Ynorm2 = np.sum((Y - ck[:, i])**2, axis=1)
        E_perp[p] = np.maximum(Ynorm2 - E_par[p], 0.0)
    # measurement noise + drift
    t = np.arange(T); drift = 0.01 * (t / max(1, T-1))
    traces = {}
    for p in PRIMS:
        raw = E_perp[p] + noise * rng.standard_normal(T)
        raw = np.maximum(raw + drift, 0.0)
        traces[p] = raw
    g0 = rng.integers(0, 5, size=(8, 8))
    g1 = apply_sequence(g0, tasks)
    sample = Sample(grid_in=g0, tasks_true=tasks, order_true=tasks,
                    grid_out_true=g1, traces=traces, T=T)
    return sample, pk, ck, Wt, Y

@dataclass
class ParseResult:
    tasks: List[str]
    order: List[str]
    peak_times: Dict[str, int]
    areas: Dict[str, float]
    corr_peak: Dict[str, float]

def exclusive_residual_matrix(E):
    T, K = E.shape
    R = np.zeros_like(E)
    Z = np.stack([_z(E[:, k]) for k in range(K)], axis=1)
    for k in range(K):
        if K == 1:
            R[:, k] = Z[:, k]; continue
        B = np.delete(Z, k, axis=1)
        Q, _ = np.linalg.qr(B, mode='reduced')
        x = Z[:, k]
        x_expl = Q @ (Q.T @ x)
        r = x - x_expl
        R[:, k] = r
    return np.maximum(R, 0.0)

def geodesic_parse_exclusive(traces: Dict[str, np.ndarray], sigma=11, proto_width=120,
                             tau_area=10.0, tau_corr=0.7):
    E = np.stack([traces[p] for p in PRIMS], axis=1)
    Es = np.stack([moving_average(E[:, i], k=sigma) for i in range(E.shape[1])], axis=1)
    R = exclusive_residual_matrix(Es)
    Tlen = R.shape[0]
    proto = half_sine(proto_width); proto = proto / (np.linalg.norm(proto) + 1e-8)
    peak_idx, corr_peak, areas = {}, {}, {}
    for i, p in enumerate(PRIMS):
        m = np.correlate(_z(R[:, i]), _z(proto), mode="same")
        idx = int(np.argmax(m)); peak_idx[p] = idx
        L = proto_width; a, b = max(0, idx - L//2), min(Tlen, idx + L//2)
        w = R[a:b, i]; w_d = w - w.mean(); pr = proto[:len(w)] - proto[:len(w)].mean()
        corr_peak[p] = float(np.dot(w_d, pr) / (np.linalg.norm(w_d) * np.linalg.norm(pr) + 1e-8))
        areas[p] = float(np.trapz(np.maximum(R[:, i], 0.0)))
    # fixed thresholds (Stage-10v2)
    keep = [p for p in PRIMS if areas[p] > tau_area and corr_peak[p] > tau_corr]
    if not keep:
        # fallback: pick best by product
        score = {p: corr_peak[p] * areas[p] for p in PRIMS}
        keep = [max(score, key=score.get)]
    order = sorted(keep, key=lambda p: peak_idx[p])
    return ParseResult(keep, order, peak_idx, areas, corr_peak)

def stock_parse_raw(traces: Dict[str, np.ndarray], sigma=11, proto_width=120,
                    tau_area=10.0, tau_corr=0.7):
    Tlen = len(next(iter(traces.values())))
    proto = half_sine(proto_width); proto = proto / (np.linalg.norm(proto) + 1e-8)
    peak_idx, corr_peak, areas = {}, {}, {}
    Es = {p: moving_average(traces[p], k=sigma) for p in PRIMS}
    for p in PRIMS:
        idx = int(np.argmax(Es[p])); peak_idx[p] = idx
        L = proto_width; a, b = max(0, idx - L//2), min(Tlen, idx + L//2)
        w = Es[p][a:b]; w_d = w - w.mean(); pr = proto[:len(w)] - proto[:len(w)].mean()
        corr_peak[p] = float(np.dot(w_d, pr) / (np.linalg.norm(w_d) * np.linalg.norm(pr) + 1e-8))
        areas[p] = float(np.trapz(Es[p]))
    keep = [p for p in PRIMS if areas[p] > tau_area and corr_peak[p] > tau_corr]
    if not keep:
        score = {p: corr_peak[p] * areas[p] for p in PRIMS}
        keep = [max(score, key=score.get)]
    order = sorted(keep, key=lambda p: peak_idx[p])
    return ParseResult(keep, order, peak_idx, areas, corr_peak)

def plot_pair(sample_id, sample, Rgeo, Es_raw, pg, ps, outdir):
    ensure_dir(outdir)
    T = sample.T
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax = axes[0]
    for i, p in enumerate(PRIMS):
        ax.plot(np.maximum(Rgeo[:, i], 0.0), label=f"E_ex {p}", linewidth=2)
    ax.set_title(f"[Geodesic excl] tasks={pg.tasks} order={' -> '.join(pg.order)}")
    ax.legend(loc="upper right"); ax.set_ylabel("exclusive residual")
    ax = axes[1]
    for p in PRIMS:
        ax.plot(Es_raw[p], label=f"E_raw {p}", linewidth=2)
    ax.set_title(f"[Stock raw] tasks={ps.tasks} order={' -> '.join(ps.order)}")
    ax.legend(loc="upper right"); ax.set_xlabel("step"); ax.set_ylabel("raw energy")
    plt.suptitle(f"Sample {sample_id:02d} — true order: {' -> '.join(sample.order_true)}", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(outdir, f"sample{sample_id:02d}.png")
    plt.savefig(path, dpi=120); plt.close()
    return path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=48)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--dim", type=int, default=19)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--plot_dir", type=str, default="plots_ode")
    ap.add_argument("--sigma", type=int, default=11)
    ap.add_argument("--proto_width", type=int, default=120)
    ap.add_argument("--hard", type=int, default=1, help="use overlapping windows with jitter")
    ap.add_argument("--mass", type=float, default=4.0)
    ap.add_argument("--lam", type=float, default=0.35)
    ap.add_argument("--gamma", type=float, default=0.04)
    ap.add_argument("--dt", type=float, default=0.02)
    # New: thresholds + CSV path
    ap.add_argument("--tau_area", type=float, default=10.0)
    ap.add_argument("--tau_corr", type=float, default=0.7)
    ap.add_argument("--presence_csv", type=str, default="presence_metrics.csv")
    args = ap.parse_args()

    rng = set_seed(args.seed)
    grid_geod = grid_stock = 0
    seq_g = seq_s = 0
    setF1_g = []; setF1_s = []

    # presence logging buffers
    pres_rows = []
    def log_presence(method, sample_id, tasks_true, areas, corr_peak, keep):
        true_set = set(tasks_true)
        for p in PRIMS:
            pres_rows.append({
                "method": method,
                "sample": sample_id,
                "primitive": p,
                "present_gt": int(p in true_set),
                "area": float(areas[p]),
                "corr": float(corr_peak[p]),
                "pred": int(p in keep),
            })

    for i in range(1, args.samples + 1):
        sample, pk, ck, Wt, Y = make_sample_with_ode(
            rng, T=args.T, d=args.dim, noise=args.noise,
            m=args.mass, lam=args.lam, gamma=args.gamma, dt=args.dt,
            overlap_prob=0.6 if args.hard else 0.0
        )
        Es_raw = {p: moving_average(sample.traces[p], k=args.sigma) for p in PRIMS}
        M_raw = np.stack([Es_raw[p] for p in PRIMS], axis=1)
        Rgeo = exclusive_residual_matrix(M_raw)
        # parse with fixed thresholds
        pg = geodesic_parse_exclusive(sample.traces, sigma=args.sigma, proto_width=args.proto_width,
                                      tau_area=args.tau_area, tau_corr=args.tau_corr)
        ps = stock_parse_raw(sample.traces, sigma=args.sigma, proto_width=args.proto_width,
                             tau_area=args.tau_area, tau_corr=args.tau_corr)
        # log presence metrics
        log_presence("geodesic", i, sample.order_true, pg.areas, pg.corr_peak, set(pg.tasks))
        log_presence("stock",    i, sample.order_true, ps.areas, ps.corr_peak, set(ps.tasks))

        # execute
        g_geod = apply_sequence(sample.grid_in, pg.order)
        g_stock = apply_sequence(sample.grid_in, ps.order)
        ok_g = bool(np.array_equal(g_geod, sample.grid_out_true))
        ok_s = bool(np.array_equal(g_stock, sample.grid_out_true))
        grid_geod += int(ok_g); grid_stock += int(ok_s)
        seq_g += int(pg.order == sample.order_true); seq_s += int(ps.order == sample.order_true)
        set_true = set(sample.order_true)
        set_g = set(pg.tasks); set_s = set(ps.tasks)
        def f1(a, b):
            tp = len(a & b); fp = len(a - b); fn = len(b - a)
            prec = tp / (tp + fp + 1e-12); rec = tp / (tp + fn + 1e-12)
            return 2*prec*rec / (prec+rec + 1e-12)
        setF1_g.append(f1(set_g, set_true)); setF1_s.append(f1(set_s, set_true))
        path = plot_pair(i, sample, Rgeo, Es_raw, pg, ps, args.plot_dir)
        print(f"[{i:02d}] TRUE: {sample.order_true}")
        print(f"     GEO: tasks={pg.tasks} | order={' -> '.join(pg.order)} | grid_ok={ok_g}")
        print(f"     STK: tasks={ps.tasks} | order={' -> '.join(ps.order)} | grid_ok={ok_s}")
        print(f"     plot={path}\n")

    n = args.samples
    print(f"[SUMMARY] Grid exact — Geodesic: {grid_geod}/{n} = {grid_geod/n:.1%} | Stock: {grid_stock}/{n} = {grid_stock/n:.1%}")
    print(f"[SUMMARY] Seq exact  — Geodesic: {seq_g}/{n} = {seq_g/n:.1%} | Stock: {seq_s}/{n} = {seq_s/n:.1%}")
    import numpy as _np
    print(f"[SUMMARY] Task-set F1 — Geodesic: {_np.mean(setF1_g):.2f} | Stock: {_np.mean(setF1_s):.2f}")

    # Write presence CSV
    csv_path = args.presence_csv if os.path.isabs(args.presence_csv) else os.path.join('.', args.presence_csv)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["method","sample","primitive","present_gt","area","corr","pred"])
        writer.writeheader()
        for row in pres_rows:
            writer.writerow(row)
    print(f"[PRESENCE] Wrote metrics to {csv_path}")

    # Optional: compute AUCs without plotting
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        # Scores: we can use corr as the primary score; you can switch to area or product easily
        for method in ["geodesic", "stock"]:
            rows = [r for r in pres_rows if r["method"] == method]
            y_true = _np.array([r["present_gt"] for r in rows], dtype=float)
            score_corr = _np.array([r["corr"] for r in rows], dtype=float)
            score_area = _np.array([r["area"] for r in rows], dtype=float)
            # Normalize area for stability
            score_area_n = (score_area - score_area.min()) / (score_area.max() - score_area.min() + 1e-12)
            score_prod = score_corr * score_area_n
            auc_corr = roc_auc_score(y_true, score_corr)
            auc_area = roc_auc_score(y_true, score_area_n)
            auc_prod = roc_auc_score(y_true, score_prod)
            ap_corr = average_precision_score(y_true, score_corr)
            ap_area = average_precision_score(y_true, score_area_n)
            ap_prod = average_precision_score(y_true, score_prod)
            print(f"[AUC] {method} — ROC: corr={auc_corr:.3f}, area={auc_area:.3f}, prod={auc_prod:.3f}")
            print(f"[AUC] {method} — PR : corr={ap_corr:.3f}, area={ap_area:.3f}, prod={ap_prod:.3f}")
    except Exception as e:
        print(f"[AUC] Skipping AUC computation (sklearn not available or error: {e})")

if __name__ == "__main__":
    main()
