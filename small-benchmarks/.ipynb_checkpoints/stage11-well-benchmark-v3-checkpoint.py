
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage11-well-benchmark-v3.py
----------------------------
Phase 2: Orthogonalized residuals (Step 2) + Strict sequential residual refinement (Step 4).

Goals:
- Flip margins positive by removing shared low-rank structure across primitives.
- Keep recall high while cutting phantoms via "accept only if residual energy drops" rule.
- Retain diagnostics: PI, surfaces, manifold dump.

This remains a synthetic harness; connect your real trace generator where indicated.
"""

import argparse, os, csv, json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

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

def gaussian_bump(T: int, center: int, width: int, amp: float=1.0) -> np.ndarray:
    t = np.arange(T)
    sig2 = (width / 2.355)**2  # FWHM -> sigma
    return amp * np.exp(-(t - center)**2 / (2*sig2))

def find_local_minima(x: np.ndarray, delta: float=0.0) -> List[int]:
    mins = []
    for i in range(1, len(x)-1):
        if x[i] + delta < x[i-1] and x[i] + delta < x[i+1]:
            mins.append(i)
    return mins

# ------------------------ block permutation null ------------------------

def block_roll(x: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Block-circular shift: split x into blocks of length block_len and roll by k blocks."""
    n = len(x)
    if block_len <= 1 or block_len >= n:
        k = int(rng.integers(1, n-1))
        return np.roll(x, k)
    m = int(np.ceil(n / block_len)) * block_len
    xp = np.pad(x, (0, m - n), mode="wrap")
    xb = xp.reshape(m // block_len, block_len)
    k_blocks = int(rng.integers(1, xb.shape[0]))
    xb2 = np.roll(xb, k_blocks, axis=0)
    y = xb2.reshape(-1)[:n]
    return y

def corr_at(sig: np.ndarray, proto: np.ndarray, center_idx: int, width: int) -> float:
    T = len(sig)
    a, b = max(0, center_idx - width//2), min(T, center_idx + width//2)
    w = sig[a:b]
    if len(w) < 3:
        return 0.0
    w0 = w - w.mean()
    pr = proto[:len(w)] - proto[:len(w)].mean()
    denom = (np.linalg.norm(w0) * np.linalg.norm(pr) + 1e-12)
    return float(np.dot(w0, pr) / denom)

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

# ------------------------ Step 2: Orthogonalization ------------------------

def low_rank_orthogonalize(traces: Dict[str,np.ndarray], rank_r: int = 1) -> Dict[str,np.ndarray]:
    """
    Remove shared low-rank structure across primitives using SVD on the (P x T) matrix.
    X_perp = (I - U_r U_r^T) @ X, where X stacks primitive traces row-wise.
    """
    keys = list(traces.keys())
    X = np.stack([traces[p] for p in keys], axis=0)  # (P,T)
    P, T = X.shape
    r = int(max(0, min(rank_r, P)))  # r ∈ [0,P]
    if r == 0:
        X_perp = X.copy()
    else:
        # SVD on row-space (P x T)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        Ur = U[:, :r]  # (P,r)
        P_perp = np.eye(P) - Ur @ Ur.T  # (P,P)
        X_perp = P_perp @ X
    # clip to positive energy (like perpendicular_energy), avoid negatives
    X_perp = np.clip(X_perp, 0.0, None)
    return {p: X_perp[i] for i,p in enumerate(keys)}

# ------------------------ Step 4: Strict refinement ------------------------

def strict_sequential_refinement(Sres: Dict[str,np.ndarray], proto_width: int,
                                 drop_frac: float = 0.03, max_add: int = 3) -> List[str]:
    """
    Greedy selection: add primitive p only if assigning its window reduces *global* residual area
    by at least drop_frac (e.g., 3%). Stops when no candidate passes or max_add reached.
    """
    proto = half_sine_proto(proto_width)
    keys = list(Sres.keys())
    T = len(next(iter(Sres.values())))
    remaining = set(keys)
    selected: List[str] = []

    # Precompute peak indices by matched filter on residual channel
    peaks = {}
    for p in keys:
        m = np.correlate(Sres[p], proto, mode="same")
        peaks[p] = int(np.argmax(m))

    # Greedy loop
    for _ in range(max_add):
        base_energy = sum(float(np.trapz(Sres[p])) for p in keys)
        best_p, best_drop = None, 0.0

        for p in list(remaining):
            a, b = max(0, peaks[p] - proto_width//2), min(T, peaks[p] + proto_width//2)
            shaved = {q: Sres[q].copy() for q in keys}
            for q in keys:
                if q == p:
                    continue
                shaved[q][a:b] = 0.80 * shaved[q][a:b]  # stronger shaving for strictness
            new_energy = sum(float(np.trapz(shaved[q])) for q in keys)
            drop = (base_energy - new_energy) / max(1e-9, base_energy)
            if drop > best_drop:
                best_drop, best_p = drop, p

        if best_p is not None and best_drop >= drop_frac:
            selected.append(best_p)
            remaining.remove(best_p)
            # commit the shave
            a, b = max(0, peaks[best_p] - proto_width//2), min(T, peaks[best_p] + proto_width//2)
            for q in keys:
                if q == best_p: 
                    continue
                Sres[q][a:b] = 0.80 * Sres[q][a:b]
        else:
            break

    if not selected:
        # fallback: keep the strongest residual-peak primitive
        top = max(keys, key=lambda p: float(np.max(np.correlate(Sres[p], proto, mode="same"))))
        selected = [top]
    return selected

def order_by_matched_filter(Sraw: Dict[str,np.ndarray], keep: List[str], proto_width: int) -> List[str]:
    proto = half_sine_proto(proto_width)
    peaks = {}
    for p in keep:
        m = np.correlate(Sraw[p], proto, mode="same")
        peaks[p] = int(np.argmax(m))
    return sorted(keep, key=lambda p: peaks[p])

# -------------------------- phantom index (PI) --------------------------

def compute_phantom_index(Utime: Dict[str,np.ndarray], centers_true: Dict[str,int], window: int) -> float:
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
    # Toy latent chart Y(t) in d=8 by embedding smoothed traces + small noise
    T = len(next(iter(Sraw.values())))
    d = 8
    t = np.arange(T)
    B = np.zeros((3, d))
    B[0,0] = 1.0; B[1,1] = 1.0; B[2,2] = 1.0
    X = np.zeros((T, d))
    for i,p in enumerate(PRIMS):
        X[:, :d] += np.outer(Sraw[p], B[i])
    rng = np.random.default_rng(0)
    X[:, 3:] += 0.02 * rng.standard_normal((T, d-3))
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
    ap = argparse.ArgumentParser(description="Stage 11 — Phase 2 (v3): orthogonalization + strict refinement")
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--sigma", type=int, default=9, help="smoother window")
    ap.add_argument("--proto_width", type=int, default=160)
    # Step 2: orthogonalization
    ap.add_argument("--rank_r", type=int, default=1, help="low-rank to remove across primitives (0..P)")
    # Diagnostics
    ap.add_argument("--dump_surfaces_dir", type=str, default="")
    ap.add_argument("--dump_manifold", type=str, default="")
    ap.add_argument("--pi", action="store_true")
    # Null calibration (kept strong to avoid z-inflation)
    ap.add_argument("--null-block-frac", type=float, default=0.20)
    ap.add_argument("--nperm", type=int, default=600)
    # Step 4: strict refinement threshold
    ap.add_argument("--res_drop_frac", type=float, default=0.03, help="min global residual drop to accept a primitive (e.g., 0.03=3%)")
    ap.add_argument("--max_add", type=int, default=3, help="max primitives to add (upper bound)")
    # Output
    ap.add_argument("--csv", type=str, default="stage11_v3.csv")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = []
    agg = dict(acc=0, grid=0.0, P=0.0, R=0.0, F1=0.0, J=0.0, H=0.0, O=0.0,
               margin_mu=0.0, margin_min=0.0, PI=0.0)
    all_margins = []

    for i in range(1, args.samples+1):
        sample = make_sample(rng, T=args.T, noise=args.noise)

        # Step 2: orthogonalized residuals
        Eperp = low_rank_orthogonalize(sample.traces, rank_r=args.rank_r)
        Sres = {p: moving_average(Eperp[p], k=args.sigma) for p in PRIMS}
        Sraw = {p: moving_average(sample.traces[p], k=args.sigma) for p in PRIMS}
        # common-mode (for energy penalty)
        Scm  = moving_average(np.stack([sample.traces[p] for p in PRIMS],0).mean(0), k=args.sigma)

        # Peaks (using residual channel)
        proto = half_sine_proto(args.proto_width)
        peak_idx = {p: int(np.argmax(np.correlate(Sres[p], proto, mode="same"))) for p in PRIMS}

        # z-scores with calibrated nulls
        z_res = {p: perm_null_z(Sres[p], proto, peak_idx[p], args.proto_width, rng, nperm=args.nperm, block_frac=args.null_block_frac) for p in PRIMS}
        z_raw = {p: perm_null_z(Sraw[p], proto, peak_idx[p], args.proto_width, rng, nperm=args.nperm, block_frac=args.null_block_frac) for p in PRIMS}
        z_cm  = {p: perm_null_z(Scm,     proto, peak_idx[p], args.proto_width, rng, nperm=args.nperm, block_frac=args.null_block_frac) for p in PRIMS}

        # Combined score (S5-3 style; raw term modest, cm is penalty)
        w_res, w_raw, w_cm = 1.0, 0.45, 0.30
        score = {p: w_res*max(0.0, z_res[p]) + w_raw*max(0.0, z_raw[p]) - w_cm*max(0.0, z_cm[p]) for p in PRIMS}

        # Candidate pool = those >= 45% of max score (not too permissive)
        smax = max(score.values()) + 1e-12
        cands = [p for p in PRIMS if score[p] >= 0.45*smax]
        if not cands:
            cands = [max(PRIMS, key=lambda q: score[q])]

        # Step 4: strict refinement
        keep = strict_sequential_refinement({p: Sres[p].copy() for p in PRIMS if p in cands},
                                            args.proto_width, drop_frac=args.res_drop_frac, max_add=args.max_add)
        order = order_by_matched_filter(Sraw, keep, args.proto_width)

        # execute predicted sequence on grid (synthetic executor)
        gp = apply_sequence(sample.grid_in, order)
        ok = int(np.array_equal(gp, sample.grid_out_true))
        gs = grid_similarity(gp, sample.grid_out_true)
        sm = set_metrics(sample.order_true, keep)

        # Energy time series U_p(t) (proxy)
        zcm = _zscore(Scm)
        Utime = {}
        for p in PRIMS:
            Utime[p] = -(_zscore(Sres[p]) + 0.4*_zscore(Sraw[p]) - 0.3*zcm)

        # Margins
        margins = []
        for p in PRIMS:
            t_true = sample.centers_true.get(p, None)
            if t_true is None:
                continue
            Tlen = len(Utime[p])
            w = args.proto_width
            a, b = max(0, t_true - w//2), min(Tlen, t_true + w//2)
            u_true = float(np.min(Utime[p][a:b]))
            others = [q for q in PRIMS if q != p]
            u_false = min(float(np.min(Utime[q])) for q in others)
            margins.append(u_false - u_true)
        margin_mu = float(np.mean(margins)) if margins else 0.0
        margin_min = float(min(margins)) if margins else 0.0
        all_margins.extend(margins)

        PI = compute_phantom_index(Utime, sample.centers_true, window=args.proto_width) if args.pi else 0.0

        dump_surfaces(i, args.dump_surfaces_dir, Sraw, Sres, Utime)
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

        # aggregates
        agg["acc"] += ok; agg["grid"] += gs
        agg["P"] += sm["precision"]; agg["R"] += sm["recall"]; agg["F1"] += sm["f1"]; agg["J"] += sm["jaccard"]
        agg["H"] += sm["hallucination_rate"]; agg["O"] += sm["omission_rate"]
        agg["margin_mu"] += margin_mu; agg["margin_min"] += margin_min
        agg["PI"] += PI

        print(f"[{i:02d}] true={sample.order_true} | keep={keep} | order={order} | ok={bool(ok)} | P={sm['precision']:.3f} R={sm['recall']:.3f} H={sm['hallucination_rate']:.3f}")

    n = float(len(rows))
    summary = {k: (v/n if isinstance(v, (int,float)) else v) for k,v in agg.items()}
    if all_margins:
        summary["margin_mu"] = float(np.mean(all_margins))
        summary["margin_min"] = float(np.min(all_margins))

    print("\n[Stage11 v3] Summary:", {k: (round(v,3) if isinstance(v,float) else v) for k,v in summary.items()})
    if rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"[WROTE] {args.csv}")
    with open(os.path.splitext(args.csv)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        print(f"[WROTE] {os.path.splitext(args.csv)[0]}_summary.json")

if __name__ == "__main__":
    main()
