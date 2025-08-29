
"""
ARC Hidden-Target Planning (Geodesic vs Stock)
----------------------------------------------
Plans a sequence from input grid and *traces only*. Target grid is used
only for evaluation; the parser must infer tasks + order without seeing it.

Outputs: per-sample CSV, summary metrics, optional plots per sample.

python3 arc_hidden_target.py --samples 150 --cm_amp 0.20 --overlap 0.75 \
  --amp_jitter 0.60 --distractor_prob 0.60 --noise 0.03 --proto_width 180 \
  --perm 200 --alpha 0.10 --weights 1.0,0.4,0.3 --plots

"""
import argparse, os, csv
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt

PRIMS_SMALL = ["flip_h","flip_v","rotate"]

# ------------- Utilities -------------
def moving_average(x, k=9):
    if k <= 1: return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(xp, np.ones(k)/k, mode="valid")

def half_sine(width): return np.sin(np.linspace(0, np.pi, width))
def half_sine_proto(width):
    p = half_sine(width)
    return p / (np.linalg.norm(p)+1e-8)

def gaussian_bump(T, center, width, amp=1.0):
    t = np.arange(T)
    sig2 = (width/2.355)**2
    return amp * np.exp(-(t-center)**2 / (2*sig2))

def common_mode(traces): return np.stack([traces[p] for p in traces],0).mean(0)
def perpendicular_energy(traces): 
    mu = common_mode(traces)
    return {p: np.clip(traces[p]-mu, 0, None) for p in traces}

def add_noise(x, sigma, rng): return x + rng.normal(0, sigma, size=x.shape)

def circ_shift(x, k):
    k = int(k) % len(x)
    if k == 0: return x
    return np.concatenate([x[-k:], x[:-k]])

def corr_at(sig, proto, idx, width, T):
    a, b = max(0, idx - width//2), min(T, idx + width//2)
    w = sig[a:b]
    if len(w) < 3: return 0.0
    w = w - w.mean()
    pr = proto[:len(w)] - proto[:len(w)].mean()
    denom = (np.linalg.norm(w) * np.linalg.norm(pr) + 1e-8)
    return float(np.dot(w, pr) / denom)

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
    p = 2.0 * (1.0 - 0.5 * (1 + np.math.erf(abs(z)/np.sqrt(2))))
    return z, p

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

# ------------- World -------------
@dataclass
class Sample:
    grid_in: np.ndarray
    tasks_true: List[str]
    order_true: List[str]
    grid_out_true: np.ndarray
    traces: Dict[str, np.ndarray]
    T: int

def random_grid(rng, H=8, W=8, ncolors=6): return rng.integers(0, ncolors, size=(H, W))

def apply_primitive(grid, prim):
    if prim == "flip_h": return np.fliplr(grid)
    if prim == "flip_v": return np.flipud(grid)
    if prim == "rotate": return np.rot90(grid, k=-1)
    raise ValueError(prim)

def apply_sequence(grid, seq):
    g = grid.copy()
    for p in seq: g = apply_primitive(g, p)
    return g

def gen_traces(tasks: List[str], T: int, rng, noise=0.01,
               cm_amp=0.02, overlap=0.5, amp_jitter=0.4, distractor_prob=0.4) -> Dict[str, np.ndarray]:
    base = np.array([0.20, 0.50, 0.80]) * T
    centers = ((1.0 - overlap) * base + overlap * (T * 0.50)).astype(int)
    width = int(max(12, T * 0.10))
    t = np.arange(T)
    cm = cm_amp * (1.0 + 0.2 * np.sin(2*np.pi * t / max(30, T//6)))
    traces = {p: np.zeros(T, float) for p in PRIMS_SMALL}

    for i, prim in enumerate(tasks):
        c = centers[i % len(centers)]
        amp = max(0.25, 1.0 + rng.normal(0, amp_jitter))
        c_jit = int(np.clip(c + rng.integers(-width//5, width//5 + 1), 0, T-1))
        traces[prim] += gaussian_bump(T, c_jit, width, amp=amp)

    for p in PRIMS_SMALL:
        if p not in tasks and rng.random() < distractor_prob:
            c = int(rng.uniform(T*0.15, T*0.85))
            amp = max(0.25, 1.0 + rng.normal(0, amp_jitter))
            traces[p] += gaussian_bump(T, c, width, amp=0.9*amp)

    for p in PRIMS_SMALL:
        traces[p] = np.clip(traces[p] + cm, 0, None)
        traces[p] = add_noise(traces[p], noise, rng)
    return traces

def make_sample(rng, T=720, n_tasks=(1,3), grid_shape=(8,8), noise=0.01, allow_repeats=False,
                cm_amp=0.02, overlap=0.5, amp_jitter=0.4, distractor_prob=0.4) -> Sample:
    k = rng.integers(n_tasks[0], n_tasks[1]+1)
    tasks = list(rng.choice(PRIMS_SMALL, size=k, replace=allow_repeats))
    if not allow_repeats:
        rng.shuffle(tasks)
    g0 = random_grid(rng, *grid_shape)
    g1 = apply_sequence(g0, tasks)
    traces = gen_traces(tasks, T=T, rng=rng, noise=noise,
                        cm_amp=cm_amp, overlap=overlap,
                        amp_jitter=amp_jitter, distractor_prob=distractor_prob)
    return Sample(g0, tasks, tasks, g1, traces, T)

# ------------- Parsers -------------
def geodesic_parse(traces, sigma=9, proto_width=140, rng=None, nperm=150, q=0.10, weights=(1.0,0.4,0.3)):
    keys = list(traces.keys()); T = len(next(iter(traces.values())))
    Eres = perpendicular_energy(traces)
    Sres = {p: moving_average(Eres[p], k=sigma) for p in keys}
    Sraw = {p: moving_average(traces[p], k=sigma) for p in keys}
    Scm  = moving_average(common_mode(traces), k=sigma)
    proto = half_sine_proto(proto_width)

    # detect peaks from residual channel
    peak_idx = {p: int(np.argmax(np.correlate(Sres[p], proto, mode="same"))) for p in keys}

    def z(sig, idx): return perm_null_z(sig, proto, idx, proto_width, rng, nperm=nperm)[0]
    z_res = {p: z(Sres[p], peak_idx[p]) for p in keys}
    z_raw = {p: z(Sraw[p], peak_idx[p]) for p in keys}
    z_cm  = {p: z(Scm,      peak_idx[p]) for p in keys}
    w_res, w_raw, w_cm = weights
    score = {p: w_res*z_res[p] + w_raw*z_raw[p] - w_cm*max(0.0, z_cm[p]) for p in keys}

    # convert to p via normal tail
    def z_to_p(v): return 2.0 * (1.0 - 0.5 * (1 + np.math.erf(abs(v)/np.sqrt(2))))
    p_comb = [z_to_p(max(0,z_res[p]) + 0.5*max(0,z_raw[p]) - 0.3*max(0,z_cm[p])) for p in keys]
    passed = bh_fdr(p_comb, q=q)
    keep = [keys[i] for i, ok in enumerate(passed) if ok and score[keys[i]] > 0]
    if not keep: keep = [max(keys, key=lambda p: score[p])]
    order = sorted(keep, key=lambda p: peak_idx[p])
    return keep, order

def stock_parse(traces, sigma=9, proto_width=140):
    keys = list(traces.keys()); T = len(next(iter(traces.values())))
    S = {p: moving_average(traces[p], k=sigma) for p in keys}
    proto = half_sine_proto(proto_width)
    peak = {p: int(np.argmax(np.correlate(S[p], proto, mode="same"))) for p in keys}
    # simple score = peak correlation height
    score = {p: float(np.max(np.correlate(S[p], proto, mode="same"))) for p in keys}
    # pick those >= 60% of max
    smax = max(score.values()) + 1e-12
    keep = [p for p in keys if score[p] >= 0.6*smax]
    if not keep: keep = [max(keys, key=lambda p: score[p])]
    order = sorted(keep, key=lambda p: peak[p])
    return keep, order

# ------------- Metrics -------------
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

# ------------- Main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--sigma", type=int, default=9)
    ap.add_argument("--proto_width", type=int, default=160)
    ap.add_argument("--perm", type=int, default=150)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--weights", type=str, default="1.0,0.4,0.3")
    ap.add_argument("--tasks_range", type=str, default="1,3")
    ap.add_argument("--allow_repeats", action="store_true")
    ap.add_argument("--cm_amp", type=float, default=0.02)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--amp_jitter", type=float, default=0.4)
    ap.add_argument("--distractor_prob", type=float, default=0.4)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--csv", type=str, default="hidden_target.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    min_k, max_k = map(int, args.tasks_range.split(","))
    w_res, w_raw, w_cm = map(float, args.weights.split(","))

    rows = []
    agg = dict(geo_acc=0, geo_grid=0.0, geo_P=0.0, geo_R=0.0, geo_F1=0.0, geo_J=0.0, geo_H=0.0, geo_O=0.0,
               stock_acc=0, stock_grid=0.0, stock_P=0.0, stock_R=0.0, stock_F1=0.0, stock_J=0.0, stock_H=0.0, stock_O=0.0)

    for i in range(1, args.samples+1):
        sample = make_sample(rng, T=args.T, n_tasks=(min_k,max_k), noise=args.noise, allow_repeats=args.allow_repeats,
                             cm_amp=args.cm_amp, overlap=args.overlap, amp_jitter=args.amp_jitter, distractor_prob=args.distractor_prob)

        keep_g, order_g = geodesic_parse(sample.traces, sigma=args.sigma, proto_width=args.proto_width,
                                         rng=rng, nperm=args.perm, q=args.alpha, weights=(w_res,w_raw,w_cm))
        keep_s, order_s = stock_parse(sample.traces, sigma=args.sigma, proto_width=args.proto_width)

        grid_g = apply_sequence(sample.grid_in, order_g)
        grid_s = apply_sequence(sample.grid_in, order_s)

        ok_g = int(np.array_equal(grid_g, sample.grid_out_true))
        ok_s = int(np.array_equal(grid_s, sample.grid_out_true))
        gs_g = grid_similarity(grid_g, sample.grid_out_true)
        gs_s = grid_similarity(grid_s, sample.grid_out_true)
        sm_g = set_metrics(sample.order_true, keep_g)
        sm_s = set_metrics(sample.order_true, keep_s)

        agg["geo_acc"] += ok_g; agg["geo_grid"] += gs_g
        agg["geo_P"] += sm_g["precision"]; agg["geo_R"] += sm_g["recall"]; agg["geo_F1"] += sm_g["f1"]; agg["geo_J"] += sm_g["jaccard"]
        agg["geo_H"] += sm_g["hallucination_rate"]; agg["geo_O"] += sm_g["omission_rate"]
        agg["stock_acc"] += ok_s; agg["stock_grid"] += gs_s
        agg["stock_P"] += sm_s["precision"]; agg["stock_R"] += sm_s["recall"]; agg["stock_F1"] += sm_s["f1"]; agg["stock_J"] += sm_s["jaccard"]
        agg["stock_H"] += sm_s["hallucination_rate"]; agg["stock_O"] += sm_s["omission_rate"]

        rows.append(dict(
            sample=i, true="|".join(sample.order_true),
            geodesic_tasks="|".join(keep_g), geodesic_order="|".join(order_g),
            geodesic_ok=ok_g, geodesic_grid=gs_g, geodesic_precision=sm_g["precision"], geodesic_recall=sm_g["recall"],
            geodesic_f1=sm_g["f1"], geodesic_jaccard=sm_g["jaccard"], geodesic_hallucination=sm_g["hallucination_rate"], geodesic_omission=sm_g["omission_rate"],
            stock_tasks="|".join(keep_s), stock_order="|".join(order_s),
            stock_ok=ok_s, stock_grid=gs_s, stock_precision=sm_s["precision"], stock_recall=sm_s["recall"],
            stock_f1=sm_s["f1"], stock_jaccard=sm_s["jaccard"], stock_hallucination=sm_s["hallucination_rate"], stock_omission=sm_s["omission_rate"],
        ))

        if args.plots:
            # quick plot per-sample
            T = sample.T; proto = half_sine_proto(args.proto_width)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
            # geodesic residual
            Eres = perpendicular_energy(sample.traces)
            for p in PRIMS_SMALL:
                ax[0].plot(moving_average(Eres[p], k=args.sigma), label=f"E⊥ {p}", linewidth=2)
            ax[0].legend(loc="upper right"); ax[0].set_title(f"[Geodesic] {keep_g} | {' -> '.join(order_g)}")
            # stock raw
            for p in PRIMS_SMALL:
                ax[1].plot(moving_average(sample.traces[p], k=args.sigma), label=f"Eraw {p}", linewidth=2)
            ax[1].legend(loc="upper right"); ax[1].set_title(f"[Stock] {keep_s} | {' -> '.join(order_s)}")
            ax[1].set_xlabel("step"); ax[0].set_ylabel("power"); ax[1].set_ylabel("power")
            plt.tight_layout()
            out = f"hidden_target_{i:02d}.png"; plt.savefig(out, dpi=120); plt.close()
            print(f"[Plot] {out}")

    n = float(args.samples)
    Sg = dict(
        accuracy_exact = agg["geo_acc"]/n, grid_similarity=agg["geo_grid"]/n,
        precision=agg["geo_P"]/n, recall=agg["geo_R"]/n, f1=agg["geo_F1"]/n, jaccard=agg["geo_J"]/n,
        hallucination_rate=agg["geo_H"]/n, omission_rate=agg["geo_O"]/n
    )
    Ss = dict(
        accuracy_exact = agg["stock_acc"]/n, grid_similarity=agg["stock_grid"]/n,
        precision=agg["stock_P"]/n, recall=agg["stock_R"]/n, f1=agg["stock_F1"]/n, jaccard=agg["stock_J"]/n,
        hallucination_rate=agg["stock_H"]/n, omission_rate=agg["stock_O"]/n
    )
    print("[HIDDEN-TARGET] Geodesic:", {k: round(v,3) for k,v in Sg.items()})
    print("[HIDDEN-TARGET] Stock   :", {k: round(v,3) for k,v in Ss.items()})

    if rows:
        fieldnames = list(rows[0].keys())
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
            for r in rows: w.writerow(r)
        print(f"[WROTE] {args.csv}")

if __name__ == "__main__":
    main()