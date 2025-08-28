
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage11-well-benchmark-v1.py
----------------------------
Phase 0 starter for Stage 11:
- Freezes S5-3 baseline profile
- Adds diagnostics: surface dumps, manifold dump, Phantom Index
- Provides knobs for null calibration, inhibition, and gates (inactive by default)

Self-contained synthetic mode for smoke tests.
Replace `compute_traces_for_sample()` with your pipeline hooks when ready.
"""

import argparse, os, csv, json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

PRIMS = ["flip_h","flip_v","rotate"]

# ----------------------------- utils -----------------------------

def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)

def moving_average(x: np.ndarray, k: int=9) -> np.ndarray:
    if k <= 1: 
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(xp, np.ones(k)/k, mode="valid")

def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    sd = x.std() + 1e-12
    return (x - mu) / sd

def half_sine(width: int) -> np.ndarray:
    return np.sin(np.linspace(0, np.pi, width))

def half_sine_proto(width: int) -> np.ndarray:
    p = half_sine(width)
    return p / (np.linalg.norm(p) + 1e-12)

def common_mode(traces: Dict[str, np.ndarray]) -> np.ndarray:
    X = np.stack([traces[p] for p in PRIMS], axis=0)
    return X.mean(axis=0)

def perpendicular_energy(traces: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    mu = common_mode(traces)
    return {p: np.clip(traces[p] - mu, 0.0, None) for p in PRIMS}

def corr_at(sig: np.ndarray, proto: np.ndarray, center_idx: int, width: int) -> float:
    T = len(sig)
    a, b = max(0, center_idx - width//2), min(T, center_idx + width//2)
    w = sig[a:b]
    if len(w) < 3:
        return 0.0
    # demean for cosine-like correlation
    w0 = w - w.mean()
    pr = proto[:len(w)] - proto[:len(w)].mean()
    denom = (np.linalg.norm(w0) * np.linalg.norm(pr) + 1e-12)
    return float(np.dot(w0, pr) / denom)

def find_local_minima(x: np.ndarray, delta: float=0.0) -> List[int]:
    """Return indices of strict local minima (x[i] < neighbors - delta)."""
    mins = []
    for i in range(1, len(x)-1):
        if x[i] + delta < x[i-1] and x[i] + delta < x[i+1]:
            mins.append(i)
    return mins

def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / (sigma + 1e-12))**2)

# ------------------------ block permutation null ------------------------

def block_roll(x: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Block-circular shift: split x into blocks of length block_len and roll by k blocks."""
    n = len(x)
    if block_len <= 1 or block_len >= n:
        # fall back to simple circular shift
        k = int(rng.integers(1, n-1))
        return np.roll(x, k)
    # pad to multiple of block_len
    m = int(np.ceil(n / block_len)) * block_len
    xp = np.pad(x, (0, m - n), mode="wrap")
    # reshape to (m_blocks, block_len)
    xb = xp.reshape(m // block_len, block_len)
    # roll by random number of whole blocks
    k_blocks = int(rng.integers(1, xb.shape[0]))
    xb2 = np.roll(xb, k_blocks, axis=0)
    y = xb2.reshape(-1)[:n]
    return y

def perm_null_z(sig: np.ndarray, proto: np.ndarray, peak_idx: int, width: int, rng: np.random.Generator,
                nperm: int=600, block_frac: float=0.20) -> float:
    """Compute a z-score for correlation at peak_idx vs block-permutation null."""
    T = len(sig)
    obs = corr_at(sig, proto, peak_idx, width)
    block_len = max(1, int(block_frac * T))
    null = np.empty(nperm, dtype=float)
    for i in range(nperm):
        x = block_roll(sig, block_len, rng)
        null[i] = corr_at(x, proto, peak_idx, width)
    mu = float(null.mean())
    sd = float(null.std() + 1e-8)
    return (obs - mu) / sd

# ----------------------------- synthetic world -----------------------------

@dataclass
class Sample:
    grid_in: np.ndarray
    tasks_true: List[str]
    order_true: List[str]
    grid_out_true: np.ndarray
    traces: Dict[str, np.ndarray]
    centers_true: Dict[str, int]
    T: int

def random_grid(rng: np.random.Generator, H=8, W=8, ncolors=6) -> np.ndarray:
    return rng.integers(0, ncolors, size=(H, W))

def apply_primitive(grid: np.ndarray, prim: str) -> np.ndarray:
    if prim == "flip_h": return np.fliplr(grid)
    if prim == "flip_v": return np.flipud(grid)
    if prim == "rotate": return np.rot90(grid, k=-1)
    raise ValueError(prim)

def apply_sequence(grid: np.ndarray, seq: List[str]) -> np.ndarray:
    g = grid.copy()
    for p in seq:
        g = apply_primitive(g, p)
    return g

def gaussian_bump(T: int, center: int, width: int, amp: float=1.0) -> np.ndarray:
    t = np.arange(T)
    sig2 = (width / 2.355)**2  # FWHM -> sigma
    return amp * np.exp(-(t - center)**2 / (2*sig2))

def gen_traces(tasks: List[str], T: int, rng: np.random.Generator,
               noise: float=0.02, cm_amp: float=0.02, overlap: float=0.5,
               amp_jitter: float=0.4, distractor_prob: float=0.4) -> Tuple[Dict[str,np.ndarray], Dict[str,int]]:
    base = np.array([0.20, 0.50, 0.80]) * T
    centers = ((1.0 - overlap) * base + overlap * (T * 0.50)).astype(int)
    width = int(max(12, T * 0.10))
    t = np.arange(T)
    cm = cm_amp * (1.0 + 0.2 * np.sin(2*np.pi * t / max(30, T//6)))
    traces = {p: np.zeros(T, float) for p in PRIMS}
    centers_true = {}

    # true bumps
    for i, prim in enumerate(tasks):
        c = centers[i % len(centers)]
        amp = max(0.25, 1.0 + rng.normal(0, amp_jitter))
        c_jit = int(np.clip(c + rng.integers(-width//5, width//5 + 1), 0, T-1))
        traces[prim] += gaussian_bump(T, c_jit, width, amp=amp)
        centers_true[prim] = c_jit

    # distractors
    for p in PRIMS:
        if p not in tasks and rng.random() < distractor_prob:
            c = int(rng.uniform(T*0.15, T*0.85))
            amp = max(0.25, 1.0 + rng.normal(0, amp_jitter))
            traces[p] += gaussian_bump(T, c, width, amp=0.9*amp)

    # add small common-mode + noise
    for p in PRIMS:
        traces[p] = np.clip(traces[p] + cm, 0.0, None)
        traces[p] = traces[p] + rng.normal(0, noise, size=T)
        traces[p] = np.maximum(traces[p], 0.0)
    return traces, centers_true

def make_sample(rng: np.random.Generator, T: int=720, n_tasks=(1,3), noise=0.02,
                allow_repeats=False) -> Sample:
    k = rng.integers(n_tasks[0], n_tasks[1]+1)
    tasks = list(rng.choice(PRIMS, size=k, replace=allow_repeats))
    if not allow_repeats:
        rng.shuffle(tasks)
    g0 = random_grid(rng, H=8, W=8, ncolors=6)
    g1 = apply_sequence(g0, tasks)
    traces, centers_true = gen_traces(tasks, T=T, rng=rng, noise=noise)
    return Sample(g0, tasks, tasks, g1, traces, centers_true, T)

# ------------------------ stage-11 scoring (v1) ------------------------

def stage11_scores(traces: Dict[str,np.ndarray], sigma: int, proto_width: int,
                   rng: np.random.Generator, nperm: int, block_frac: float
                   ) -> Tuple[Dict[str,float], Dict[str,int], Dict[str,np.ndarray], Dict[str,np.ndarray], np.ndarray]:
    """
    Compute per-primitive scores + peaks using calibrated nulls.
    Returns:
      score[p], peak_idx[p], Sraw[p], Sres[p], Scm
    """
    Eres = perpendicular_energy(traces)
    Sraw = {p: moving_average(traces[p], k=sigma) for p in PRIMS}
    Sres = {p: moving_average(Eres[p],     k=sigma) for p in PRIMS}
    Scm  = moving_average(common_mode(traces), k=sigma)
    proto = half_sine_proto(proto_width)

    peak_idx = {p: int(np.argmax(np.correlate(Sres[p], proto, mode="same"))) for p in PRIMS}

    # z-scores via block permutation null (calibrated)
    z_res = {p: perm_null_z(Sres[p], proto, peak_idx[p], proto_width, rng, nperm=nperm, block_frac=block_frac) for p in PRIMS}
    z_raw = {p: perm_null_z(Sraw[p], proto, peak_idx[p], proto_width, rng, nperm=nperm, block_frac=block_frac) for p in PRIMS}
    # z_cm is the same sequence for all, but evaluated at each primitive's peak index
    z_cm  = {p: perm_null_z(Scm,     proto, peak_idx[p], proto_width, rng, nperm=nperm, block_frac=block_frac) for p in PRIMS}

    # S5-3 weights
    w_res, w_raw, w_cm = 1.0, 0.4, 0.3
    score = {p: w_res*max(0.0, z_res[p]) + w_raw*max(0.0, z_raw[p]) - w_cm*max(0.0, z_cm[p]) for p in PRIMS}
    return score, peak_idx, Sraw, Sres, Scm

def presence_consensus(score: Dict[str,float], Sres: Dict[str,np.ndarray], proto_width: int) -> List[str]:
    """Consensus presence: top-by-score, with residual area guard (>=45% of best)."""
    areas = {p: float(np.trapz(Sres[p])) for p in PRIMS}
    smax = max(score.values()) + 1e-12
    amax = max(areas.values()) + 1e-12
    keep = [p for p in PRIMS if score[p] >= 0.50*smax and areas[p] >= 0.45*amax]
    if not keep:
        keep = [max(PRIMS, key=lambda q: score[q])]
    return keep

def order_by_matched_filter(Sraw: Dict[str,np.ndarray], keep: List[str], proto_width: int) -> List[str]:
    proto = half_sine_proto(proto_width)
    peaks = {}
    for p in keep:
        m = np.correlate(Sraw[p], proto, mode="same")
        peaks[p] = int(np.argmax(m))
    return sorted(keep, key=lambda p: peaks[p])

# -------------------------- phantom index (PI) --------------------------

def compute_phantom_index(Utime: Dict[str,np.ndarray], centers_true: Dict[str,int], window: int) -> float:
    """
    PI = (# phantom minima across primitives) / (total minima + 1e-9)
    A minimum is 'true' if it's within +/- window/2 of the true center for that primitive (if present).
    """
    total = 0
    phantom = 0
    T = len(next(iter(Utime.values())))
    tol = max(2, window // 2)
    for p in PRIMS:
        mins = find_local_minima(Utime[p])
        total += len(mins)
        true_center = centers_true.get(p, None)
        for m in mins:
            if true_center is None or abs(m - true_center) > tol:
                phantom += 1
    return float(phantom) / (max(1, total))

# ------------------------ surfaces & manifold dumps ------------------------

def dump_surfaces(sample_id: int, out_dir: str, Sraw: Dict[str,np.ndarray], Sres: Dict[str,np.ndarray],
                  Utime: Dict[str,np.ndarray]):
    if not out_dir: return
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"sample{sample_id:03d}.npz")
    np.savez_compressed(path, Sraw=Sraw, Sres=Sres, U=Utime, prims=np.array(PRIMS, dtype=object))

def synthetic_manifold_dump(path: str, Sraw: Dict[str,np.ndarray], Utime: Dict[str,np.ndarray]):
    if not path: return
    # Build a toy latent chart Y(t) in d=8 by embedding smoothed traces + small noise
    T = len(next(iter(Sraw.values())))
    d = 8
    t = np.arange(T)
    # Map each primitive to a basis direction (3 of them), fill rest with combinations
    B = np.zeros((3, d))
    B[0,0] = 1.0; B[1,1] = 1.0; B[2,2] = 1.0
    X = np.zeros((T, d))
    for i,p in enumerate(PRIMS):
        X[:, :d] += np.outer(Sraw[p], B[i])
    # add tiny residual dims
    rng = np.random.default_rng(0)
    X[:, 3:] += 0.02 * rng.standard_normal((T, d-3))
    # Total energy U(t) as sum of Utime across prims
    U = sum(Utime[p] for p in PRIMS)
    np.savez_compressed(path, Y=X, U=U, U_k=np.stack([Utime[p] for p in PRIMS], axis=1),
                        names=np.array(PRIMS, dtype=object))

# ------------------------------- metrics -------------------------------

def grid_similarity(gp: np.ndarray, gt: np.ndarray) -> float:
    if gp.shape != gt.shape: return 0.0
    return float((gp == gt).mean())

def set_metrics(true_list: List[str], pred_list: List[str]) -> Dict[str,float]:
    Tset, Pset = set(true_list), set(pred_list)
    tp = len(Tset & Pset); fp = len(Pset - Tset); fn = len(Tset - Pset)
    precision = tp / max(1, len(Pset))
    recall    = tp / max(1, len(Tset))
    f1 = 0.0 if precision+recall == 0 else (2*precision*recall)/(precision+recall)
    jacc = tp / max(1, len(Tset | Pset))
    return dict(precision=precision, recall=recall, f1=f1, jaccard=jacc,
                hallucination_rate=fp/max(1,len(Pset)), omission_rate=fn/max(1,len(Tset)))

# ------------------------------- main -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage 11 — Phase 0 baseline + diagnostics (v1)")
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--sigma", type=int, default=9, help="smoother window")
    ap.add_argument("--proto_width", type=int, default=160)
    ap.add_argument("--profile", type=str, default="S5-3", choices=["S5-3","custom"])
    # Diagnostics
    ap.add_argument("--dump_surfaces_dir", type=str, default="", help="directory to save per-sample surfaces npz")
    ap.add_argument("--dump_manifold", type=str, default="", help="npz path to save synthetic manifold dump (single run)")
    ap.add_argument("--pi", action="store_true", help="compute Phantom Index")
    # Null + inhibition + gates (off by default for baseline run)
    ap.add_argument("--null-block-frac", type=float, default=0.20)
    ap.add_argument("--nperm", type=int, default=600)
    ap.add_argument("--inhib_sigma", type=float, default=1.6, help="time-offset inhibition kernel sigma (not used in v1 scoring)")
    ap.add_argument("--strict-gates", action="store_true", help="enable per-primitive acceptance gates")
    ap.add_argument("--raw_floor_v", type=float, default=0.15)
    ap.add_argument("--res_ceiling_v", type=float, default=0.65)
    # Output
    ap.add_argument("--csv", type=str, default="stage11_v1.csv")

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # S5-3 baseline guardrail (profile just pins defaults in v1; knobs still exposed for ablations)
    if args.profile == "S5-3":
        # Defaults already match the S5-3 spirit: calibrated nulls + consensus + soft ordering.
        pass

    rows = []
    agg = dict(acc=0, grid=0.0, P=0.0, R=0.0, F1=0.0, J=0.0, H=0.0, O=0.0,
               margin_mu=0.0, margin_min=0.0, PI=0.0)
    all_margins = []

    for i in range(1, args.samples+1):
        sample = make_sample(rng, T=args.T, noise=args.noise)

        # scores + peaks
        score, peak_idx, Sraw, Sres, Scm = stage11_scores(sample.traces, sigma=args.sigma,
                                                          proto_width=args.proto_width,
                                                          rng=rng, nperm=args.nperm,
                                                          block_frac=args.null-block-frac if hasattr(args,'null-block-frac') else args.__dict__.get('null-block-frac',0.20))
        # presence consensus
        keep = presence_consensus(score, Sres, args.proto_width)
        order = order_by_matched_filter(Sraw, keep, args.proto_width)

        # Strict gates for flip_v (optional)
        if args.strict_gates and "flip_v" in keep:
            # crude gate using raw/residual levels near the picked peak
            pv = peak_idx["flip_v"]
            Tlen = len(Sraw["flip_v"])
            a, b = max(0, pv - args.proto_width//2), min(Tlen, pv + args.proto_width//2)
            raw_win = float(np.mean(Sraw["flip_v"][a:b]))
            res_win = float(np.mean(Sres["flip_v"][a:b]))
            if raw_win < args.raw_floor_v or res_win > args.res_ceiling_v:
                keep = [p for p in keep if p != "flip_v"]
                order = [p for p in order if p != "flip_v"]

        # execute predicted sequence on grid (synthetic executor)
        gp = apply_sequence(sample.grid_in, order)
        ok = int(np.array_equal(gp, sample.grid_out_true))
        gs = grid_similarity(gp, sample.grid_out_true)
        sm = set_metrics(sample.order_true, keep)

        # Energy time series U_p(t) for surfaces (v1: proxy via standardized signals)
        # U_p(t) ~ - ( z(Sres_p) + 0.4 z(Sraw_p) - 0.3 z(Scm) )
        zcm = _zscore(Scm)
        Utime = {}
        for p in PRIMS:
            Utime[p] = -(_zscore(Sres[p]) + 0.4*_zscore(Sraw[p]) - 0.3*zcm)

        # Margins: difference between (best false min) and (true min) depths
        # We approximate by taking min U over window near true center vs min over other primitives
        margins = []
        for p in PRIMS:
            t_true = sample.centers_true.get(p, None)
            if t_true is None:
                continue
            Tlen = len(Utime[p])
            w = args.proto_width
            a, b = max(0, t_true - w//2), min(Tlen, t_true + w//2)
            u_true = float(np.min(Utime[p][a:b]))
            # pick the deepest min in *other* primitives
            others = [q for q in PRIMS if q != p]
            u_false = min(float(np.min(Utime[q])) for q in others)
            margins.append(u_false - u_true)
        margin_mu = float(np.mean(margins)) if margins else 0.0
        margin_min = float(min(margins)) if margins else 0.0
        all_margins.extend(margins)

        # Phantom Index (optional)
        PI = compute_phantom_index(Utime, sample.centers_true, window=args.proto_width) if args.pi else 0.0

        # dump surfaces
        dump_surfaces(i, args.dump_surfaces_dir, Sraw, Sres, Utime)

        # manifold dump (only once if requested)
        if args.dump_manifold and i == 1:
            synthetic_manifold_dump(args.dump_manifold, Sraw, Utime)

        rows.append(dict(
            sample=i,
            true="|".join(sample.order_true),
            keep="|".join(keep),
            order="|".join(order),
            ok=ok, grid=gs, precision=sm["precision"], recall=sm["recall"],
            f1=sm["f1"], jaccard=sm["jaccard"], hallucination=sm["hallucination_rate"], omission=sm["omission_rate"],
            margin_mu=margin_mu, margin_min=margin_min, PI=PI
        ))

        agg["acc"] += ok; agg["grid"] += gs
        agg["P"] += sm["precision"]; agg["R"] += sm["recall"]; agg["F1"] += sm["f1"]; agg["J"] += sm["jaccard"]
        agg["H"] += sm["hallucination_rate"]; agg["O"] += sm["omission_rate"]
        agg["margin_mu"] += margin_mu; agg["margin_min"] += margin_min
        agg["PI"] += PI

        # pretty print
        print(f"[{i:02d}] tasks_true={sample.order_true} | keep={keep} | order={order} | ok={bool(ok)} | grid={gs:.3f} | P={sm['precision']:.3f} R={sm['recall']:.3f} H={sm['hallucination_rate']:.3f}")

    n = float(len(rows))
    summary = {k: (v/n if isinstance(v, (int,float)) else v) for k,v in agg.items()}
    # summarize margins properly
    if all_margins:
        summary["margin_mu"] = float(np.mean(all_margins))
        summary["margin_min"] = float(np.min(all_margins))

    print("\n[Stage11 v1] Summary:", {k: (round(v,3) if isinstance(v,float) else v) for k,v in summary.items()})
    # CSV
    if rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"[WROTE] {args.csv}")
    # JSON summary
    with open(os.path.splitext(args.csv)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        print(f"[WROTE] {os.path.splitext(args.csv)[0]}_summary.json")

if __name__ == "__main__":
    main()
