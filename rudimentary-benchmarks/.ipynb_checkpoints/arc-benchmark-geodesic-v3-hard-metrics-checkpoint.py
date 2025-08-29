#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARC Stage-10 v3 — Geodesic vs Stock + HARD-MODE METRICS (synthetic)
-------------------------------------------------------------------
- Geodesic parser: perpendicular residual (common-mode removed) + smoothing + matched filter
- Stock baseline: smoothed raw traces (no residualization)
- Hard-mode controls to separate the methods:
    * --cm_amp            : common-mode bias amplitude (hurts stock > geodesic)
    * --overlap           : 0..1, larger → peaks crowd at mid timeline
    * --amp_jitter        : per-peak amplitude jitter
    * --distractor_prob   : spurious bumps on unused primitives (tests hallucinations)
    * --allow_repeats     : allow repeated primitives in the true sequence
    * --tasks_range a,b   : min,max #tasks to sample per instance

Metrics (both methods):
  * Task-set P/R/F1/Jaccard + hallucination (FP/|pred|) + omission (FN/|true|)
  * Order: Kendall (on common items) + normalized edit similarity
  * Grid semantic similarity (cell-wise) + exact-match accuracy
  * Per-sample CSV with paths to plots; two TOTAL summary rows (one per method)

Usage:
  python3 arc-benchmark-geodesic-v3-hard-metrics.py --samples 50
  python3 arc-benchmark-geodesic-v3-hard-metrics.py --samples 200 \
    --cm_amp 0.02 --overlap 0.65 --amp_jitter 0.5 --distractor_prob 0.5 \
    --tasks_range 1,4 --allow_repeats
"""
import argparse, os, csv
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt

PRIMS = ["flip_h", "flip_v", "rotate"]

# =========================
# Utilities
# =========================
def set_seed(seed: int): return np.random.default_rng(seed)

def moving_average(x, k=9):
    if k <= 1: return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(xp, np.ones(k)/k, mode="valid")

def half_sine(width): return np.sin(np.linspace(0, np.pi, width))

def gaussian_bump(T, center, width, amp=1.0):
    t = np.arange(T)
    sig2 = (width/2.355)**2  # FWHM -> sigma
    return amp * np.exp(-(t-center)**2 / (2*sig2))

def add_noise(x, sigma, rng): return x + rng.normal(0, sigma, size=x.shape)

# =========================
# Synthetic world
# =========================
@dataclass
class Sample:
    grid_in: np.ndarray
    tasks_true: List[str]
    order_true: List[str]
    grid_out_true: np.ndarray
    traces: Dict[str, np.ndarray]
    T: int

def random_grid(rng, H=8, W=8, ncolors=5): return rng.integers(0, ncolors, size=(H, W))

def apply_primitive(grid, prim):
    if prim == "flip_h": return np.fliplr(grid)
    if prim == "flip_v": return np.flipud(grid)
    if prim == "rotate": return np.rot90(grid, k=-1)  # 90° CW
    raise ValueError(f"unknown prim {prim}")

def apply_sequence(grid, seq):
    g = grid.copy()
    for p in seq: g = apply_primitive(g, p)
    return g

def gen_synthetic_traces(tasks: List[str], T: int, rng, noise=0.01,
                         cm_amp=0.002, overlap=0.0, amp_jitter=0.0, distractor_prob=0.0) -> Dict[str, np.ndarray]:
    """
    Hard-mode traces:
      - Common-mode bias (cm_amp)
      - Overlapping centers (overlap)
      - Peak amplitude jitter per task (amp_jitter)
      - Spurious bumps on non-true primitives (distractor_prob)
    """
    # Base centers then collapse toward middle as overlap↑
    base = np.array([0.20, 0.50, 0.80]) * T
    centers = ((1.0 - overlap) * base + overlap * (T * 0.50)).astype(int)
    width = int(max(12, T * 0.10))

    # Common-mode bias (hurts raw/stock more than residualized/geodesic)
    t = np.arange(T)
    cm = cm_amp * (1.0 + 0.2 * np.sin(2*np.pi * t / max(30, T//6)))  # slow wobble

    traces = {p: np.zeros(T, float) for p in PRIMS}

    # True task bumps with amplitude jitter
    for i, prim in enumerate(tasks):
        c = centers[i % len(centers)]
        amp = max(0.25, 1.0 + rng.normal(0, amp_jitter))
        c_jit = c + rng.integers(-width//5, width//5 + 1)
        traces[prim] += gaussian_bump(T, int(np.clip(c_jit, 0, T-1)), width, amp=amp)

    # Distractor bumps on primitives not in tasks
    for p in PRIMS:
        if p not in tasks and rng.random() < distractor_prob:
            c = int(rng.uniform(T*0.15, T*0.85))
            amp = max(0.25, 1.0 + rng.normal(0, amp_jitter))
            traces[p] += gaussian_bump(T, c, width, amp=0.9*amp)

    # Add common-mode + noise and clamp
    for p in PRIMS:
        traces[p] = np.clip(traces[p] + cm, 0, None)
        traces[p] = add_noise(traces[p], noise, rng)
    return traces

def make_sample(rng, T=720, n_tasks=(1,3), grid_shape=(8,8), noise=0.01, allow_repeats=False,
                cm_amp=0.002, overlap=0.0, amp_jitter=0.0, distractor_prob=0.0) -> Sample:
    k = rng.integers(n_tasks[0], n_tasks[1]+1)
    tasks = list(rng.choice(PRIMS, size=k, replace=allow_repeats))
    if not allow_repeats:
        rng.shuffle(tasks)
    g0 = random_grid(rng, H=grid_shape[0], W=grid_shape[1])
    g1 = apply_sequence(g0, tasks)
    traces = gen_synthetic_traces(tasks, T=T, rng=rng, noise=noise,
                                  cm_amp=cm_amp, overlap=overlap,
                                  amp_jitter=amp_jitter, distractor_prob=distractor_prob)
    return Sample(g0, tasks, tasks, g1, traces, T)

# =========================
# Parsers
# =========================
@dataclass
class ParseResult:
    tasks: List[str]
    order: List[str]
    peak_times: Dict[str, int]
    areas: Dict[str, float]
    corr_peak: Dict[str, float]

def common_mode(traces): return np.stack([traces[p] for p in PRIMS], 0).mean(0)

def perpendicular_energy(traces):  # residualize
    mu = common_mode(traces)
    return {p: np.clip(traces[p] - mu, 0, None) for p in PRIMS}

def _parse_from_signal(sig_dict, sigma=9, proto_width=140, topk=None) -> ParseResult:
    T = len(next(iter(sig_dict.values())))
    S = {p: moving_average(sig_dict[p], k=sigma) for p in PRIMS}
    proto = half_sine(proto_width); proto /= (np.linalg.norm(proto) + 1e-8)
    peak, corr, area = {}, {}, {}
    for p in PRIMS:
        m = np.correlate(S[p], proto, mode="same")
        idx = int(np.argmax(m)); peak[p] = idx
        L = proto_width; a, b = max(0, idx-L//2), min(T, idx+L//2)
        w = S[p][a:b]; w = (w - w.mean())
        pr = proto[:len(w)] - proto[:len(w)].mean()
        corr[p] = float(np.dot(w, pr) / (np.linalg.norm(w)*np.linalg.norm(pr) + 1e-8))
        area[p] = float(np.trapz(S[p]))
    Amax = max(area.values()) + 1e-12; Cmax = max(corr.values()) + 1e-12
    if topk is None:
        keep = [p for p in PRIMS if (area[p]/Amax) >= 0.45 and (corr[p]/Cmax) >= 0.5]
        if not keep:
            score = {p: corr[p]*(area[p]/Amax) for p in PRIMS}
            keep = [max(score, key=score.get)]
    else:
        keep = sorted(PRIMS, key=lambda p: corr[p], reverse=True)[:topk]
    order = sorted(keep, key=lambda p: peak[p])
    return ParseResult(keep, order, peak, area, corr)

def geodesic_parse(traces, sigma=9, proto_width=140, topk=None):
    return _parse_from_signal(perpendicular_energy(traces), sigma, proto_width, topk)

def stock_parse(traces, sigma=9, proto_width=140, topk=None):
    return _parse_from_signal(traces, sigma, proto_width, topk)

# =========================
# Metrics
# =========================
def kendall_similarity(true_order: List[str], pred_order: List[str]) -> float:
    common = [x for x in true_order if x in pred_order]
    if len(common) <= 1: return 1.0
    rT = {x:i for i,x in enumerate(true_order) if x in common}
    rP = {x:i for i,x in enumerate(pred_order) if x in common}
    disc = total = 0
    for i in range(len(common)):
        for j in range(i+1, len(common)):
            a,b = common[i], common[j]; total += 1
            if np.sign(rT[a]-rT[b]) != np.sign(rP[a]-rP[b]): disc += 1
    return 1.0 - (disc/total if total else 0.0)

def edit_similarity(a: List[str], b: List[str]) -> float:
    na, nb = len(a), len(b)
    if max(na,nb) == 0: return 1.0
    dp = np.zeros((na+1, nb+1), int)
    for i in range(na+1): dp[i,0]=i
    for j in range(nb+1): dp[0,j]=j
    for i in range(1,na+1):
        for j in range(1,nb+1):
            dp[i,j] = min(dp[i-1,j]+1, dp[i,j-1]+1, dp[i-1,j-1]+(a[i-1]!=b[j-1]))
    return 1.0 - dp[na,nb]/max(na,nb)

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

# =========================
# Plotting
# =========================
def plot_compare(i: int, sample: Sample, geod_sig: Dict[str,np.ndarray],
                 stock_sig: Dict[str,np.ndarray], geod_parse: ParseResult,
                 stock_parse: ParseResult, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    T = sample.T
    fig, axes = plt.subplots(2,1, figsize=(12,7), sharex=True)
    # geodesic
    ax = axes[0]
    for p in PRIMS: ax.plot(geod_sig[p], label=f"E⊥ {p}", linewidth=2)
    if geod_parse.order:
        p0 = geod_parse.order[0]; W=140; proto = half_sine(W); proto/= (np.linalg.norm(proto)+1e-8)
        s = geod_parse.peak_times[p0]; xs = np.arange(W)+s-W//2; ys = proto*max(1e-8, geod_sig[p0].max())
        m = (xs>=0)&(xs<T); ax.plot(xs[m], ys[m], 'r--', linewidth=2, alpha=0.8)
    ax.set_title(f"[Geodesic] tasks={geod_parse.tasks} order={' -> '.join(geod_parse.order) if geod_parse.order else '—'}")
    ax.legend(loc="upper right"); ax.set_ylabel("residual power")
    # stock
    ax = axes[1]
    for p in PRIMS: ax.plot(stock_sig[p], label=f"Eraw {p}", linewidth=2)
    if stock_parse.order:
        p0 = stock_parse.order[0]; W=140; proto = half_sine(W); proto/= (np.linalg.norm(proto)+1e-8)
        s = stock_parse.peak_times[p0]; xs = np.arange(W)+s-W//2; ys = proto*max(1e-8, stock_sig[p0].max())
        m = (xs>=0)&(xs<T); ax.plot(xs[m], ys[m], 'r--', linewidth=2, alpha=0.8)
    ax.set_title(f"[Stock] tasks={stock_parse.tasks} order={' -> '.join(stock_parse.order) if stock_parse.order else '—'}")
    ax.legend(loc="upper right"); ax.set_xlabel("step"); ax.set_ylabel("raw power")
    plt.suptitle(f"Sample {i:02d} — true: {' -> '.join(sample.order_true)}", y=0.98)
    plt.tight_layout(rect=[0,0,1,0.96])
    fname = os.path.join(out_dir, f"sample{i:02d}.png")
    plt.savefig(fname, dpi=120); plt.close()
    return fname

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--plot_dir", type=str, default="plots_compare_v3_hard")
    ap.add_argument("--sigma", type=int, default=9)
    ap.add_argument("--proto_width", type=int, default=140)
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--metrics_csv", type=str, default="compare_metrics_hard.csv")
    # hard-mode knobs
    ap.add_argument("--cm_amp", type=float, default=0.002, help="common-mode bias amplitude")
    ap.add_argument("--overlap", type=float, default=0.0, help="0=no overlap, 1=tight overlap")
    ap.add_argument("--amp_jitter", type=float, default=0.0, help="std of per-peak amplitude jitter")
    ap.add_argument("--distractor_prob", type=float, default=0.0, help="prob of spurious bump on non-true primitive")
    ap.add_argument("--allow_repeats", action="store_true", help="allow repeated primitives in the true sequence")
    ap.add_argument("--tasks_range", type=str, default="1,3", help="min,max number of tasks")
    args = ap.parse_args()

    min_k, max_k = map(int, args.tasks_range.split(","))
    task_range = (min_k, max_k)
    rng = set_seed(args.seed)

    # aggregates per method
    agg = {
        "geod": dict(exact=0, grid=0.0, P=0.0, R=0.0, F1=0.0, J=0.0, H=0.0, O=0.0, K=0.0, E=0.0),
        "stock": dict(exact=0, grid=0.0, P=0.0, R=0.0, F1=0.0, J=0.0, H=0.0, O=0.0, K=0.0, E=0.0),
    }
    rows = []
    os.makedirs(args.plot_dir, exist_ok=True)

    for i in range(1, args.samples+1):
        sample = make_sample(
            rng, T=args.T, noise=args.noise, n_tasks=task_range,
            allow_repeats=args.allow_repeats, cm_amp=args.cm_amp,
            overlap=args.overlap, amp_jitter=args.amp_jitter,
            distractor_prob=args.distractor_prob
        )
        # signals
        geod_sig = {p: moving_average(perpendicular_energy(sample.traces)[p], k=args.sigma) for p in PRIMS}
        stock_sig = {p: moving_average(sample.traces[p], k=args.sigma) for p in PRIMS}
        # parses
        geod = _parse_from_signal(geod_sig, sigma=1, proto_width=args.proto_width, topk=args.topk)  # already smoothed
        stok = _parse_from_signal(stock_sig, sigma=1, proto_width=args.proto_width, topk=args.topk)

        # execute
        grid_geod = apply_sequence(sample.grid_in, geod.order)
        grid_stok = apply_sequence(sample.grid_in, stok.order)
        ok_geod = bool(np.array_equal(grid_geod, sample.grid_out_true))
        ok_stok = bool(np.array_equal(grid_stok, sample.grid_out_true))

        # metrics per method
        def compute_block(A, parse, gp):
            sm = set_metrics(sample.order_true, parse.tasks)
            kend = kendall_similarity(sample.order_true, parse.order)
            edit = edit_similarity(sample.order_true, parse.order)
            gsim = grid_similarity(gp, sample.grid_out_true)
            ok = bool(np.array_equal(gp, sample.grid_out_true))
            A["exact"] += int(ok); A["grid"] += gsim
            A["P"] += sm["precision"]; A["R"] += sm["recall"]; A["F1"] += sm["f1"]; A["J"] += sm["jaccard"]
            A["H"] += sm["hallucination_rate"]; A["O"] += sm["omission_rate"]; A["K"] += kend; A["E"] += edit
            return sm, kend, edit, gsim

        sm_g, K_g, E_g, GS_g = compute_block(agg["geod"], geod, grid_geod)
        sm_s, K_s, E_s, GS_s = compute_block(agg["stock"], stok, grid_stok)

        plot = plot_compare(i, sample, geod_sig, stock_sig, geod, stok, args.plot_dir)

        # logging
        print(f"[{i:02d}] TRUE: {sample.order_true}")
        print(f"     GEODESIC: tasks={geod.tasks} | order={' -> '.join(geod.order) if geod.order else '—'} | grid_sim={GS_g:.3f} | P={sm_g['precision']:.3f} R={sm_g['recall']:.3f} F1={sm_g['f1']:.3f} H={sm_g['hallucination_rate']:.3f} O={sm_g['omission_rate']:.3f} K={K_g:.3f} E={E_g:.3f}")
        print(f"     STOCK   : tasks={stok.tasks} | order={' -> '.join(stok.order) if stok.order else '—'} | grid_sim={GS_s:.3f} | P={sm_s['precision']:.3f} R={sm_s['recall']:.3f} F1={sm_s['f1']:.3f} H={sm_s['hallucination_rate']:.3f} O={sm_s['omission_rate']:.3f} K={K_s:.3f} E={E_s:.3f}")
        print(f"     plot={plot}\n")

        # CSV rows (two rows per sample: geod & stock)
        for method, parse, sm, K, E, GS in [
            ("geodesic", geod, sm_g, K_g, E_g, GS_g),
            ("stock",    stok, sm_s, K_s, E_s, GS_s),
        ]:
            rows.append(dict(
                sample=i, method=method,
                tasks_true="|".join(sample.order_true),
                tasks_pred="|".join(parse.tasks),
                order_pred="|".join(parse.order),
                ok_exact=int(GS==1.0),
                grid_similarity=GS,
                precision=sm["precision"], recall=sm["recall"], f1=sm["f1"], jaccard=sm["jaccard"],
                hallucination_rate=sm["hallucination_rate"], omission_rate=sm["omission_rate"],
                kendall=K, edit_similarity=E,
                areas_flip_h=parse.areas["flip_h"], areas_flip_v=parse.areas["flip_v"], areas_rotate=parse.areas["rotate"],
                corr_flip_h=parse.corr_peak["flip_h"], corr_flip_v=parse.corr_peak["flip_v"], corr_rotate=parse.corr_peak["rotate"],
                plot=plot
            ))

    # summaries
    n = float(args.samples)
    def summarize(A):
        return dict(
            accuracy_exact = A["exact"]/n,
            grid_similarity = A["grid"]/n,
            precision=A["P"]/n, recall=A["R"]/n, f1=A["F1"]/n, jaccard=A["J"]/n,
            hallucination_rate=A["H"]/n, omission_rate=A["O"]/n,
            kendall=A["K"]/n, edit_similarity=A["E"]/n
        )
    Sg, Ss = summarize(agg["geod"]), summarize(agg["stock"])
    print("[SUMMARY] Geodesic:", {k: round(v,3) for k,v in Sg.items()})
    print("[SUMMARY] Stock   :", {k: round(v,3) for k,v in Ss.items()})

    # CSV
    if rows:
        fieldnames = list(rows[0].keys())
        # add summary columns to header for TOTAL rows
        for c in ["accuracy_exact","grid_similarity","precision","recall","f1","jaccard",
                  "hallucination_rate","omission_rate","kendall","edit_similarity"]:
            if c not in fieldnames: fieldnames.append(c)
        with open(args.metrics_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
            for r in rows: w.writerow(r)
            # TOTAL rows (one per method)
            for name, S in [("geodesic", Sg), ("stock", Ss)]:
                row = {k:"" for k in fieldnames}
                row["sample"] = "TOTAL"; row["method"] = name
                for k,v in S.items(): row[k] = v
                w.writerow(row)
        print(f"[WROTE] {args.metrics_csv}")
    else:
        print("[WARN] No rows to write; zero samples?")

if __name__ == "__main__":
    main()
