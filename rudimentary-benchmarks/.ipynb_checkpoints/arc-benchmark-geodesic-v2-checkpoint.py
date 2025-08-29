#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 10 - v2 synthetic benchmark (parser + executor) + METRICS
----------------------------------------------------------------
Adds:
- Task-set precision/recall/F1 and hallucination rate (FP / predicted)
- Order similarity (Kendall-tau on common items) and edit-distance similarity
- Grid semantic similarity (cell-wise accuracy) and exact-execution accuracy
- Per-sample CSV and summary aggregates

Usage:
  python3 arc-benchmark-geodesic-v2.py --samples 25 --seed 42 --T 720 --plot_dir plots_v2 
"""
import argparse
import os
import csv
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt

PRIMS = ["flip_h", "flip_v", "rotate"]

# -----------------------------
# Utilities
# -----------------------------
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

def half_sine(width):
    t = np.linspace(0, np.pi, width)
    return np.sin(t)

def gaussian_bump(T, center, width, amp=1.0):
    t = np.arange(T)
    sig2 = (width/2.355)**2  # FWHM -> sigma
    return amp * np.exp(-(t-center)**2 / (2*sig2))

def add_noise(x, sigma, rng):
    return x + rng.normal(0, sigma, size=x.shape)

# -----------------------------
# Synthetic generator
# -----------------------------
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
        # 90° clockwise
        return np.rot90(grid, k=-1)
    raise ValueError("unknown prim")

def apply_sequence(grid, seq):
    g = grid.copy()
    for p in seq:
        g = apply_primitive(g, p)
    return g

def gen_synthetic_traces(tasks: List[str], T: int, rng, noise=0.01) -> Dict[str, np.ndarray]:
    """
    For each primitive concept, produce an energy trace over T steps.
    Place Gaussian bumps at windows associated with tasks; add a small shared drift.
    """
    centers = [int(T*0.15), int(T*0.45), int(T*0.70)]
    width = int(T*0.10)
    base_drift = 0.002 * np.linspace(1.0, 1.1, T)

    traces = {p: np.zeros(T, dtype=float) for p in PRIMS}
    for idx, prim in enumerate(tasks):
        c = centers[idx % len(centers)]
        bump = gaussian_bump(T, c, width, amp=1.0)
        traces[prim] += bump

    for p in PRIMS:
        traces[p] += base_drift
        traces[p] = add_noise(traces[p], noise, rng)
        traces[p] = np.clip(traces[p], 0, None)

    return traces

def make_sample(rng, T=720, n_tasks=(1,3), grid_shape=(8,8), noise=0.01) -> Sample:
    k = rng.integers(n_tasks[0], n_tasks[1]+1)
    tasks = list(rng.choice(PRIMS, size=k, replace=False))
    rng.shuffle(tasks)
    g0 = random_grid(rng, H=grid_shape[0], W=grid_shape[1])
    g1 = apply_sequence(g0, tasks)
    traces = gen_synthetic_traces(tasks, T=T, rng=rng, noise=noise)
    return Sample(grid_in=g0, tasks_true=tasks, order_true=tasks, grid_out_true=g1, traces=traces, T=T)

# -----------------------------
# Parser: perpendicular energy + matched filter
# -----------------------------
@dataclass
class ParseResult:
    tasks: List[str]
    order: List[str]
    peak_times: Dict[str, int]
    areas_perp: Dict[str, float]
    corr_peak: Dict[str, float]

def common_mode(traces: Dict[str, np.ndarray]) -> np.ndarray:
    X = np.stack([traces[p] for p in PRIMS], axis=0)
    mu = X.mean(axis=0)
    return mu

def perpendicular_energy(traces: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    mu = common_mode(traces)
    E = {}
    for p in PRIMS:
        E[p] = np.clip(traces[p] - mu, 0, None)  # residual positive part
    return E

def matched_filter_parse(traces: Dict[str, np.ndarray], sigma=9, proto_width=140, topk=None) -> ParseResult:
    E_perp = perpendicular_energy(traces)
    T = len(next(iter(traces.values())))
    E_s = {p: moving_average(E_perp[p], k=sigma) for p in PRIMS}

    proto = half_sine(proto_width)
    proto = proto / (np.linalg.norm(proto) + 1e-8)

    peak_idx = {}
    corr_peak = {}
    areas = {}

    for p in PRIMS:
        m = np.correlate(E_s[p], proto, mode="same")
        idx = int(np.argmax(m))
        peak_idx[p] = idx

        L = proto_width
        a = max(0, idx - L//2)
        b = min(T, idx + L//2)
        w = E_s[p][a:b]
        w = (w - w.mean())
        pr = proto[:len(w)] - proto[:len(w)].mean()
        denom = (np.linalg.norm(w) * np.linalg.norm(pr) + 1e-8)
        corr_peak[p] = float(np.dot(w, pr) / denom)

        areas[p] = float(np.trapz(E_s[p]))

    areas_arr = np.array([areas[p] for p in PRIMS])
    corrs_arr = np.array([corr_peak[p] for p in PRIMS])
    Amax = float(areas_arr.max() + 1e-12)
    Cmax = float(corrs_arr.max() + 1e-12)

    if topk is None:
        keep = []
        for p in PRIMS:
            a_ok = (areas[p] / Amax) >= 0.45
            c_ok = (corr_peak[p] / Cmax) >= 0.5
            if a_ok and c_ok:
                keep.append(p)
        if not keep:
            score = {p: corr_peak[p] * (areas[p] / (Amax+1e-12)) for p in PRIMS}
            keep = [max(score, key=score.get)]
    else:
        keep = sorted(PRIMS, key=lambda p: corr_peak[p], reverse=True)[:topk]

    order = sorted(keep, key=lambda p: peak_idx[p])

    return ParseResult(tasks=keep, order=order, peak_times=peak_idx,
                       areas_perp=areas, corr_peak=corr_peak)

# -----------------------------
# Executor & plotting
# -----------------------------
def execute_plan(grid, order: List[str]):
    return apply_sequence(grid, order)

def plot_parse(sample_id: int, sample: Sample, E_perp_s: Dict[str, np.ndarray],
               parse: ParseResult, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    T = sample.T
    fig, ax = plt.subplots(figsize=(12,4))
    for p in PRIMS:
        ax.plot(E_perp_s[p], label=f"E⊥ {p}", linewidth=2)
    if len(parse.order) > 0:
        p0 = parse.order[0]
        proto_width = 140
        proto = half_sine(proto_width)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        shift = parse.peak_times[p0]
        xs = np.arange(proto_width) + shift - proto_width//2
        ys = proto * max(1e-8, E_perp_s[p0].max())
        mask = (xs >= 0) & (xs < T)
        ax.plot(xs[mask], ys[mask], linestyle="--", linewidth=2)
    ax.set_title(f"v2 parse - sample{sample_id:02d} - tasks: {parse.tasks} - order: {' -> '.join(parse.order) if parse.order else '—'}")
    ax.set_xlabel("step"); ax.set_ylabel("residual aligned power (smoothed)")
    ax.legend(loc="upper right")
    fname = os.path.join(out_dir, f"sample{sample_id:02d}.png")
    plt.tight_layout(); plt.savefig(fname, dpi=110); plt.close()
    return fname

# -----------------------------
# Metrics
# -----------------------------
def kendall_similarity(order_true: List[str], order_pred: List[str]) -> float:
    """
    Kendall-tau similarity on the intersection of items in both orders.
    Returns 1.0 if identical order on common items, 0.0 if fully reversed (for m>=2), and 1.0 for m<=1.
    """
    common = [x for x in order_true if x in order_pred]
    if len(common) <= 1:
        return 1.0
    rank_true = {x: i for i, x in enumerate(order_true) if x in common}
    rank_pred = {x: i for i, x in enumerate(order_pred) if x in common}
    discord = 0
    total = 0
    for i in range(len(common)):
        for j in range(i+1, len(common)):
            a, b = common[i], common[j]
            total += 1
            sign_true = np.sign(rank_true[a] - rank_true[b])
            sign_pred = np.sign(rank_pred[a] - rank_pred[b])
            if sign_true != sign_pred:
                discord += 1
    if total == 0:
        return 1.0
    return 1.0 - (discord / total)

def edit_similarity(a: List[str], b: List[str]) -> float:
    """
    Normalized Levenshtein similarity: 1 - dist / max(len(a), len(b)).
    """
    na, nb = len(a), len(b)
    if max(na, nb) == 0:
        return 1.0
    dp = np.zeros((na+1, nb+1), dtype=int)
    for i in range(na+1): dp[i,0] = i
    for j in range(nb+1): dp[0,j] = j
    for i in range(1,na+1):
        for j in range(1,nb+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i,j] = min(dp[i-1,j]+1, dp[i,j-1]+1, dp[i-1,j-1]+cost)
    dist = dp[na, nb]
    return 1.0 - dist / max(na, nb)

def grid_similarity(g_pred: np.ndarray, g_true: np.ndarray) -> float:
    """
    Fraction of matching cells (semantic similarity on the output grid).
    """
    if g_pred.shape != g_true.shape:
        return 0.0
    return float((g_pred == g_true).mean())

def set_metrics(true_list: List[str], pred_list: List[str]) -> Dict[str, float]:
    Tset = set(true_list)
    Pset = set(pred_list)
    tp = len(Tset & Pset)
    fp = len(Pset - Tset)
    fn = len(Tset - Pset)
    precision = tp / max(1, len(Pset))
    recall = tp / max(1, len(Tset))
    f1 = 0.0 if precision+recall == 0 else 2*precision*recall/(precision+recall)
    jaccard = tp / max(1, len(Tset | Pset))
    hallucination_rate = fp / max(1, len(Pset))  # FP among predicted
    omission_rate = fn / max(1, len(Tset))      # FN among true
    return dict(precision=precision, recall=recall, f1=f1, jaccard=jaccard,
                hallucination_rate=hallucination_rate, omission_rate=omission_rate)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--plot_dir", type=str, default="plots_stage10_v2")
    ap.add_argument("--sigma", type=int, default=9, help="smoother window")
    ap.add_argument("--proto_width", type=int, default=140)
    ap.add_argument("--topk", type=int, default=None, help="max concepts to keep; default=auto")
    ap.add_argument("--metrics_csv", type=str, default="metrics_stage10v2.csv")
    args = ap.parse_args()

    rng = set_seed(args.seed)

    # Aggregates
    agg = dict(
        exact_ok=0,
        grid_sim=0.0,
        precision=0.0, recall=0.0, f1=0.0, jaccard=0.0,
        hallucination=0.0, omission=0.0,
        kendall=0.0, edit_sim=0.0
    )

    rows = []
    os.makedirs(args.plot_dir, exist_ok=True)

    for i in range(1, args.samples+1):
        # Build synthetic sample
        sample = make_sample(rng, T=args.T, noise=args.noise)
        # Residual energy & smoothing
        from_traces = perpendicular_energy(sample.traces)
        E_perp_s = {p: moving_average(from_traces[p], k=args.sigma) for p in PRIMS}
        # Parse
        parse = matched_filter_parse(sample.traces, sigma=args.sigma, proto_width=args.proto_width, topk=args.topk)
        # Execute
        grid_pred = execute_plan(sample.grid_in, parse.order)
        ok = bool(np.array_equal(grid_pred, sample.grid_out_true))

        # Metrics
        sm = set_metrics(sample.order_true, parse.tasks)
        kend = kendall_similarity(sample.order_true, parse.order)
        edit_sim = edit_similarity(sample.order_true, parse.order)
        gsim = grid_similarity(grid_pred, sample.grid_out_true)

        # Aggregate
        agg["exact_ok"] += int(ok)
        agg["grid_sim"] += gsim
        agg["precision"] += sm["precision"]; agg["recall"] += sm["recall"]
        agg["f1"] += sm["f1"]; agg["jaccard"] += sm["jaccard"]
        agg["hallucination"] += sm["hallucination_rate"]; agg["omission"] += sm["omission_rate"]
        agg["kendall"] += kend; agg["edit_sim"] += edit_sim

        # Plot
        plot_path = plot_parse(i, sample, E_perp_s, parse, args.plot_dir)

        # Log line
        areas = [round(parse.areas_perp[p], 3) for p in PRIMS]
        corr = [round(parse.corr_peak[p], 3) for p in PRIMS]
        print(f"[{i:02d}] -> Tasks: {parse.tasks} | Order: {' → '.join(parse.order) if parse.order else '—'} | ok={ok}")
        print(f"     areas⊥={areas} corr_peak={corr} plot={plot_path}")
        print(f"     set: P={sm['precision']:.3f} R={sm['recall']:.3f} F1={sm['f1']:.3f} Jacc={sm['jaccard']:.3f}  Halluc={sm['hallucination_rate']:.3f} Omit={sm['omission_rate']:.3f}")
        print(f"     order: Kendall={kend:.3f}  EditSim={edit_sim:.3f}  grid-sim={gsim:.3f}")

        # CSV row
        rows.append(dict(
            sample=i,
            tasks_true="|".join(sample.order_true),
            tasks_pred="|".join(parse.tasks),
            order_pred="|".join(parse.order),
            ok_exact=int(ok),
            grid_similarity=gsim,
            precision=sm["precision"], recall=sm["recall"], f1=sm["f1"], jaccard=sm["jaccard"],
            hallucination_rate=sm["hallucination_rate"], omission_rate=sm["omission_rate"],
            kendall=kend, edit_similarity=edit_sim,
            areas_flip_h=parse.areas_perp["flip_h"], areas_flip_v=parse.areas_perp["flip_v"], areas_rotate=parse.areas_perp["rotate"],
            corr_flip_h=parse.corr_peak["flip_h"], corr_flip_v=parse.corr_peak["flip_v"], corr_rotate=parse.corr_peak["rotate"],
            plot=plot_path
        ))

    # Summary
    n = float(args.samples)
    summary = dict(
        accuracy_exact = agg["exact_ok"]/n,
        grid_similarity = agg["grid_sim"]/n,
        precision = agg["precision"]/n,
        recall = agg["recall"]/n,
        f1 = agg["f1"]/n,
        jaccard = agg["jaccard"]/n,
        hallucination_rate = agg["hallucination"]/n,
        omission_rate = agg["omission"]/n,
        kendall = agg["kendall"]/n,
        edit_similarity = agg["edit_sim"]/n
    )
    print("\n[Stage10 v2 - synthetic] SUMMARY")
    for k,v in summary.items():
        print(f"  {k:20s}: {v:.3f}" if isinstance(v, float) else f"  {k:20s}: {v}")

    # Write CSV
    csv_path = args.metrics_csv
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
            # append summary as a final row with sample=TOTAL
            summ_row = {k: "" for k in fieldnames}
            summ_row["sample"] = "TOTAL"
            
            # Map summary metrics into columns that actually exist
            summ_row["ok_exact"] = summary["accuracy_exact"]           # use the existing column
            summ_row["grid_similarity"] = summary["grid_similarity"]
            summ_row["precision"] = summary["precision"]
            summ_row["recall"] = summary["recall"]
            summ_row["f1"] = summary["f1"]
            summ_row["jaccard"] = summary["jaccard"]
            summ_row["hallucination_rate"] = summary["hallucination_rate"]
            summ_row["omission_rate"] = summary["omission_rate"]
            summ_row["kendall"] = summary["kendall"]
            summ_row["edit_similarity"] = summary["edit_similarity"]
            
            w.writerow(summ_row)
        print(f"\n[WROTE] per-sample metrics -> {csv_path}")
    else:
        print("[WARN] No rows to write; zero samples?")

if __name__ == "__main__":
    main()
