# Recreate the two scripts so you can download them.
from pathlib import Path

# root = Path("/mnt/data")

# arc_scaling_wall = root / "arc_scaling_wall.py"
# arc_hidden_target = root / "arc_hidden_target.py"

# arc_scaling_wall_code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC Scaling Wall Benchmark
--------------------------
Compares a geodesic task-parser vs a brute-force enumerator as we scale:
- number of primitives |ops|
- sequence length L

Geodesic parser infers tasks/order from residualized (perpendicular) energy traces.
Brute-force attempts to find a sequence that exactly transforms grid_in -> grid_out_true.

Outputs:
- per-sample CSV with metrics + runtimes
- summary printout
- optional scaling plots

Usage examples:
  python3 arc_scaling_wall.py --samples 100 --ops_set large --L_range 2,8 --plot_curves
  python3 arc_scaling_wall.py --samples 50 --ops_set medium --cm_amp 0.05 --overlap 0.5 \
      --amp_jitter 0.5 --distractor_prob 0.4 --allow_repeats --noise 0.02
"""
import argparse, os, csv, time
from dataclasses import dataclass
from typing import List, Dict, Callable, Tuple
import numpy as np
import matplotlib.pyplot as plt

# ------------------- Primitive library -------------------

def flip_h(G): return np.fliplr(G)
def flip_v(G): return np.flipud(G)
def rotate_cw(G): return np.rot90(G, k=-1)
def rotate_ccw(G): return np.rot90(G, k=1)
def rotate_180(G): return np.rot90(G, k=2)

def translate(G, dx=0, dy=0, fill=0):
    H, W = G.shape
    out = np.full_like(G, fill)
    xs = np.clip(np.arange(W) - dx, 0, W-1)
    ys = np.clip(np.arange(H) - dy, 0, H-1)
    out = G[np.ix_(ys, xs)]
    return out

def color_add_mod(G, k=1, ncolors=9):
    return (G + k) % ncolors

def color_swap(G, a=1, b=2):
    X = G.copy()
    X[G==a] = -999
    X[G==b] = a
    X[X==-999] = b
    return X

def color_mask_most(G):
    # return boolean mask of most frequent color
    vals, counts = np.unique(G, return_counts=True)
    c = vals[np.argmax(counts)]
    return (G==c).astype(int)

def invert_mod(G, ncolors=9): return (-G) % ncolors

def id_op(G): return G

def build_ops(set_name: str, ncolors: int = 9) -> Dict[str, Callable]:
    ops = {}
    # core
    ops["flip_h"] = flip_h
    ops["flip_v"] = flip_v
    ops["rot90"]  = rotate_cw
    if set_name in ("medium","large"):
        ops["rot270"] = rotate_ccw
        ops["rot180"] = rotate_180
        # translations
        for dx in (-2,-1,1,2):
            ops[f"tx{dx:+d}"] = (lambda G,dx=dx: translate(G, dx=dx, dy=0))
        for dy in (-2,-1,1,2):
            ops[f"ty{dy:+d}"] = (lambda G,dy=dy: translate(G, dx=0, dy=dy))
        # color ops
        for k in (1,2,3):
            ops[f"cadd{k}"] = (lambda G,k=k: color_add_mod(G, k=k, ncolors=ncolors))
        for a,b in [(1,2),(2,3),(3,4),(1,4)]:
            ops[f"cswap{a}{b}"] = (lambda G,a=a,b=b: color_swap(G,a=a,b=b))
        ops["invert"] = (lambda G: invert_mod(G, ncolors=ncolors))
    if set_name == "large":
        # extra translations
        for dx in (-3,3): ops[f"tx{dx:+d}"] = (lambda G,dx=dx: translate(G, dx=dx, dy=0))
        for dy in (-3,3): ops[f"ty{dy:+d}"] = (lambda G,dy=dy: translate(G, dx=0, dy=dy))
        # identity (distractor)
        ops["id"] = id_op
    return ops

# ------------------- Synthetic generator -------------------
@dataclass
class Sample:
    grid_in: np.ndarray
    tasks_true: List[str]
    order_true: List[str]
    grid_out_true: np.ndarray
    traces: Dict[str, np.ndarray]
    T: int

def random_grid(rng, H=12, W=12, ncolors=9): return rng.integers(0, ncolors, size=(H, W))

def apply_sequence(grid, seq, OPS):
    g = grid.copy()
    for p in seq:
        g = OPS[p](g)
    return g

def half_sine(width): return np.sin(np.linspace(0, np.pi, width))

def gaussian_bump(T, center, width, amp=1.0):
    t = np.arange(T)
    sig2 = (width/2.355)**2
    return amp * np.exp(-(t-center)**2 / (2*sig2))

def add_noise(x, sigma, rng): return x + rng.normal(0, sigma, size=x.shape)

def gen_traces(tasks: List[str], all_prims: List[str], T: int, rng, noise=0.01,
               cm_amp=0.002, overlap=0.0, amp_jitter=0.0, distractor_prob=0.0) -> Dict[str, np.ndarray]:
    base = np.array([0.20, 0.50, 0.80]) * T
    centers = ((1.0 - overlap) * base + overlap * (T * 0.50)).astype(int)
    width = int(max(12, T * 0.10))
    t = np.arange(T)
    cm = cm_amp * (1.0 + 0.2 * np.sin(2*np.pi * t / max(30, T//6)))

    traces = {p: np.zeros(T, float) for p in all_prims}

    for i, prim in enumerate(tasks):
        c = centers[i % len(centers)]
        amp = max(0.25, 1.0 + rng.normal(0, amp_jitter))
        c_jit = int(np.clip(c + rng.integers(-width//5, width//5 + 1), 0, T-1))
        traces[prim] += gaussian_bump(T, c_jit, width, amp=amp)

    # distractors on non-true primitives
    for p in all_prims:
        if p not in tasks and rng.random() < distractor_prob:
            c = int(rng.uniform(T*0.15, T*0.85))
            amp = max(0.25, 1.0 + rng.normal(0, amp_jitter))
            traces[p] += gaussian_bump(T, c, width, amp=0.9*amp)

    for p in all_prims:
        traces[p] = np.clip(traces[p] + cm, 0, None)
        traces[p] = add_noise(traces[p], noise, rng)
    return traces

def make_sample(rng, OPS, T=720, n_tasks=(2,5), grid_shape=(12,12), noise=0.01,
                allow_repeats=False, cm_amp=0.002, overlap=0.0, amp_jitter=0.0, distractor_prob=0.0) -> Sample:
    prims = list(OPS.keys())
    k = rng.integers(n_tasks[0], n_tasks[1]+1)
    tasks = list(rng.choice(prims, size=k, replace=allow_repeats))
    if not allow_repeats:
        rng.shuffle(tasks)
    g0 = random_grid(rng, *grid_shape)
    g1 = apply_sequence(g0, tasks, OPS)
    traces = gen_traces(tasks, prims, T=T, rng=rng, noise=noise,
                        cm_amp=cm_amp, overlap=overlap, amp_jitter=amp_jitter, distractor_prob=distractor_prob)
    return Sample(g0, tasks, tasks, g1, traces, T)

# ------------------- Geodesic parser -------------------
def moving_average(x, k=9):
    if k <= 1: return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(xp, np.ones(k)/k, mode="valid")

def common_mode(traces):
    arr = np.stack([traces[p] for p in traces.keys()], 0)
    return arr.mean(0)

def perpendicular_energy(traces):
    mu = common_mode(traces)
    return {p: np.clip(traces[p] - mu, 0, None) for p in traces.keys()}

def half_sine_proto(width): 
    P = half_sine(width)
    return P / (np.linalg.norm(P) + 1e-8)

def corr_at(sig, proto, idx, width, T):
    a, b = max(0, idx - width//2), min(T, idx + width//2)
    w = sig[a:b]
    if len(w) < 3: return 0.0
    w = w - w.mean()
    pr = proto[:len(w)] - proto[:len(w)].mean()
    denom = (np.linalg.norm(w) * np.linalg.norm(pr) + 1e-8)
    return float(np.dot(w, pr) / denom)

def circ_shift(x, k):
    k = int(k) % len(x)
    if k == 0: return x
    return np.concatenate([x[-k:], x[:-k]])

def perm_null_z(sig, proto, peak_idx, width, rng, nperm=200):
    T = len(sig)
    obs = corr_at(sig, proto, peak_idx, width, T)
    null = np.empty(nperm, dtype=float)
    for i in range(nperm):
        shift = rng.integers(1, T-1)
        x = circ_shift(sig, shift)
        null[i] = corr_at(x, proto, peak_idx, width, T)
    mu, sd = float(null.mean()), float(null.std() + 1e-8)
    z = (obs - mu) / sd
    # 2-sided normal p
    p = 2.0 * (1.0 - 0.5 * (1 + 0.0 * z) + 0.0)  # placeholder, z used elsewhere
    # (we only need z downstream)
    return z, 0.5

def bh_fdr(pvals, q=0.10):
    m = len(pvals)
    if m == 0: return np.array([], dtype=bool)
    order = np.argsort(pvals)
    thresh = q * (np.arange(1, m+1) / m)
    passed = np.zeros(m, dtype=bool)
    pv_sorted = np.array(pvals)[order]
    max_i = np.where(pv_sorted <= thresh)[0]
    if len(max_i) > 0:
        cutoff = max_i.max()
        passed[order[:cutoff+1]] = True
    return passed

def geodesic_detect(traces, sigma=9, proto_width=140, rng=None, nperm=200, q=0.10,
                    weights=(1.0, 0.4, 0.3)):
    """Hybrid residual/raw detection with common-mode penalty + FDR selection."""
    keys = list(traces.keys())
    T = len(next(iter(traces.values())))
    Eres = perpendicular_energy(traces)
    # smooth once
    Sres = {p: moving_average(Eres[p], k=sigma) for p in keys}
    Sraw = {p: moving_average(traces[p], k=sigma) for p in keys}
    Scm  = moving_average(common_mode(traces), k=sigma)

    proto = half_sine_proto(proto_width)
    peak_idx = {}
    for p in keys:
        m = np.correlate(Sres[p], proto, mode="same")
        peak_idx[p] = int(np.argmax(m))

    z_res, z_raw, z_cm = {}, {}, {}
    areas = {}
    for p in keys:
        idx = peak_idx[p]
        zr, _ = perm_null_z(Sres[p], proto, idx, proto_width, rng, nperm=nperm)
        z_res[p] = zr
        zr2, _ = perm_null_z(Sraw[p], proto, idx, proto_width, rng, nperm=nperm)
        z_raw[p] = zr2
        zc, _ = perm_null_z(Scm,      proto, idx, proto_width, rng, nperm=nperm)
        z_cm[p]  = zc
        areas[p] = float(np.trapz(Sres[p]))

    w_res, w_raw, w_cm = weights
    score = {p: w_res*z_res[p] + w_raw*z_raw[p] - w_cm*max(0.0, z_cm[p]) for p in keys}

    # simple selection: pick those >= 50% of max score
    smax = max(score.values()) + 1e-12
    keep = [p for p in keys if score[p] >= 0.5*smax]
    if not keep:
        keep = [max(keys, key=lambda p: score[p])]
    order = sorted(keep, key=lambda p: peak_idx[p])
    return keep, order

# ------------------- Brute force baseline -------------------
def brute_force_find(OPS, grid_in, grid_out, max_len=4, max_nodes=200000) -> Tuple[bool, List[str], int]:
    """Depth-first brute-force with a node cap; returns (found, seq, nodes_expanded)."""
    ops = list(OPS.keys())
    nodes = 0
    path = []
    seen = set()

    def dfs(g, depth):
        nonlocal nodes, path
        if nodes >= max_nodes: return False
        if np.array_equal(g, grid_out): return True
        if depth >= max_len: return False
        key = (g.tobytes(), depth)
        if key in seen: return False
        seen.add(key)
        for p in ops:
            nodes += 1
            g2 = OPS[p](g)
            path.append(p)
            if dfs(g2, depth+1): return True
            path.pop()
            if nodes >= max_nodes: break
        return False

    found = dfs(grid_in, 0)
    return found, path.copy(), nodes

# ------------------- Metrics -------------------
def grid_similarity(gp: np.ndarray, gt: np.ndarray) -> float:
    return 0.0 if gp.shape != gt.shape else float((gp==gt).mean())

def set_metrics(true_list: List[str], pred_list: List[str]) -> Dict[str,float]:
    Tset, Pset = set(true_list), set(pred_list)
    tp, fp, fn = len(Tset & Pset), len(Pset - Tset), len(Tset - Pset)
    precision = tp / max(1, len(Pset))
    recall    = tp / max(1, len(Tset))
    f1        = 0.0 if precision+recall==0 else (2*precision*recall)/(precision+recall)
    jaccard   = tp / max(1, len(Tset | Pset))
    return dict(precision=precision, recall=recall, f1=f1, jaccard=jaccard,
                hallucination_rate=fp/max(1,len(Pset)), omission_rate=fn/max(1,len(Tset)))

# ------------------- Main -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--plot_curves", action="store_true")
    ap.add_argument("--ops_set", type=str, default="large", choices=["small","medium","large"])
    ap.add_argument("--tasks_range", type=str, default="2,6")
    ap.add_argument("--L_range", type=str, default="2,6", help="sequence length for brute-force cap")
    ap.add_argument("--allow_repeats", action="store_true")
    # hard-mode knobs
    ap.add_argument("--cm_amp", type=float, default=0.02)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--amp_jitter", type=float, default=0.4)
    ap.add_argument("--distractor_prob", type=float, default=0.4)
    # geodesic detector
    ap.add_argument("--sigma", type=int, default=9)
    ap.add_argument("--proto_width", type=int, default=160)
    ap.add_argument("--perm", type=int, default=150)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--weights", type=str, default="1.0,0.4,0.3")
    # brute force
    ap.add_argument("--bf_max_len", type=int, default=4)
    ap.add_argument("--bf_max_nodes", type=int, default=200000)
    ap.add_argument("--csv", type=str, default="scaling_wall.csv")
    args = ap.parse_args()

    w_res, w_raw, w_cm = map(float, args.weights.split(","))
    min_k, max_k = map(int, args.tasks_range.split(","))
    minL, maxL = map(int, args.L_range.split(","))

    rng = np.random.default_rng(args.seed)
    OPS = build_ops(args.ops_set)
    prims = list(OPS.keys())

    rows = []
    agg = dict(geo_acc=0, geo_grid=0.0, geo_P=0.0, geo_R=0.0, geo_F1=0.0, geo_J=0.0,
               geo_H=0.0, geo_O=0.0, geo_time=0.0,
               bf_found=0, bf_time=0.0, bf_nodes=0)

    for i in range(1, args.samples+1):
        sample = make_sample(rng, OPS, T=args.T, n_tasks=(min_k, max_k), noise=args.noise,
                             allow_repeats=args.allow_repeats, cm_amp=args.cm_amp,
                             overlap=args.overlap, amp_jitter=args.amp_jitter, distractor_prob=args.distractor_prob)

        # Geodesic detection (no target used)
        t0 = time.perf_counter()
        keep, order = geodesic_detect(sample.traces, sigma=args.sigma, proto_width=args.proto_width,
                                      rng=rng, nperm=args.perm, q=args.alpha, weights=(w_res, w_raw, w_cm))
        g = apply_sequence(sample.grid_in, order, OPS)
        geo_ok = int(np.array_equal(g, sample.grid_out_true))
        geo_grid = grid_similarity(g, sample.grid_out_true)
        sm = set_metrics(sample.order_true, keep)
        t1 = time.perf_counter()
        geo_dt = t1 - t0

        # Brute-force (with target), length bounded by bf_max_len
        bf_len = min(args.bf_max_len, maxL)
        t2 = time.perf_counter()
        found, seq, nodes = brute_force_find(OPS, sample.grid_in, sample.grid_out_true,
                                             max_len=bf_len, max_nodes=args.bf_max_nodes)
        t3 = time.perf_counter()
        bf_dt = t3 - t2

        agg["geo_acc"] += geo_ok
        agg["geo_grid"] += geo_grid
        agg["geo_P"] += sm["precision"]; agg["geo_R"] += sm["recall"]; agg["geo_F1"] += sm["f1"]; agg["geo_J"] += sm["jaccard"]
        agg["geo_H"] += sm["hallucination_rate"]; agg["geo_O"] += sm["omission_rate"]
        agg["geo_time"] += geo_dt
        agg["bf_found"] += int(found); agg["bf_time"] += bf_dt; agg["bf_nodes"] += nodes

        rows.append(dict(
            sample=i,
            true="|".join(sample.order_true),
            geodesic_tasks="|".join(keep),
            geodesic_order="|".join(order),
            geodesic_ok=geo_ok,
            geodesic_grid=geo_grid,
            geodesic_precision=sm["precision"],
            geodesic_recall=sm["recall"],
            geodesic_f1=sm["f1"],
            geodesic_jaccard=sm["jaccard"],
            geodesic_hallucination=sm["hallucination_rate"],
            geodesic_omission=sm["omission_rate"],
            geodesic_time=geo_dt,
            bf_found=int(found),
            bf_seq="|".join(seq),
            bf_nodes=nodes,
            bf_time=bf_dt,
            n_ops=len(prims),
            L_true=len(sample.order_true),
        ))

    n = float(args.samples)
    summary = dict(
        n_ops=len(prims),
        geo_acc=agg["geo_acc"]/n, geo_grid=agg["geo_grid"]/n,
        geo_P=agg["geo_P"]/n, geo_R=agg["geo_R"]/n, geo_F1=agg["geo_F1"]/n, geo_J=agg["geo_J"]/n,
        geo_H=agg["geo_H"]/n, geo_O=agg["geo_O"]/n,
        geo_time=agg["geo_time"]/n,
        bf_found=agg["bf_found"]/n, bf_time=agg["bf_time"]/n, bf_nodes=agg["bf_nodes"]/n,
    )
    print("[SCALING WALL] Summary:", {k: (round(v,3) if isinstance(v,float) else v) for k,v in summary.items()})

    # CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"[WROTE] {args.csv}")

    # Optional rough plot of runtimes distributions
    if args.plot_curves:
        import matplotlib.pyplot as plt
        geo_times = [r["geodesic_time"] for r in rows]
        bf_times  = [r["bf_time"] for r in rows]
        plt.figure(figsize=(8,4))
        plt.boxplot([geo_times, bf_times], labels=["Geodesic", "Brute-Force"])
        plt.ylabel("runtime (s)"); plt.title(f"Runtime dist — ops={len(prims)}")
        plt.tight_layout()
        plt.savefig("scaling_wall_runtimes.png", dpi=120)
        print("[PLOT] scaling_wall_runtimes.png")

if __name__ == "__main__":
    main()

# '''

# arc_scaling_wall.write_text(arc_scaling_wall_code)
# arc_hidden_target.write_text(arc_hidden_target_code)

# print(str(arc_scaling_wall))
# print(str(arc_hidden_target))
