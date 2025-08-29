
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage11-well-benchmark-v4.py
----------------------------
Phase 3: Lateral inhibition (Step 5) + Controlled descent / temperature (Step 6)
with an explicit signal-processing framing.

What’s new vs v3:
- Lateral inhibition: time-offset-aware Gaussian penalty between primitives
  (suppresses overlapping distractors without hard pruning).
- Temperature descent: convert hard selection into soft probabilities π ∝ exp(−Up/T),
  then anneal temperature to avoid premature rejections.
- Candidate floor exposed (--cand_floor).
- Per-primitive gates kept (esp. flip_v) but gentler by default.
- Keeps diagnostics: PI, surfaces, manifold dump.

This is still a synthetic harness. Replace the trace generator with your pipeline hooks.

python3 stage11-well-benchmark-v4.py \
  --samples 50 \
  --rank_r 1 \
  --cand_floor 0.40 \
  --inhib_sigma 1.6 --inhib_lambda 0.35 \
  --T0 1.6 --Tmin 0.7 --anneal_steps 2 --p_floor 0.18 \
  --pi \
  --dump_surfaces_dir dumps/v4_surfaces \
  --dump_manifold dumps/v4_manifold.npz

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

def gaussian_penalty(ti: int, tj: int, sigma: float) -> float:
    return float(np.exp(-0.5 * ((ti - tj) / (sigma + 1e-12))**2))

def find_local_minima(x: np.ndarray, delta: float=0.0) -> List[int]:
    mins = []
    for i in range(1, len(x)-1):
        if x[i] + delta < x[i-1] and x[i] + delta < x[i+1]:
            mins.append(i)
    return mins

# ------------------------ block permutation null ------------------------

def block_roll(x: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
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
                nperm: int=500, block_frac: float=0.16) -> float:
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

# ------------------------ Step 2: Orthogonalization (kept from v3) ------------------------

def low_rank_orthogonalize(traces: Dict[str,np.ndarray], rank_r: int = 1) -> Dict[str,np.ndarray]:
    keys = list(traces.keys())
    X = np.stack([traces[p] for p in keys], axis=0)  # (P,T)
    P, T = X.shape
    r = int(max(0, min(rank_r, P)))
    if r == 0:
        X_perp = X.copy()
    else:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        Ur = U[:, :r]  # (P,r)
        P_perp = np.eye(P) - Ur @ Ur.T  # (P,P)
        X_perp = P_perp @ X
    X_perp = np.clip(X_perp, 0.0, None)
    return {p: X_perp[i] for i,p in enumerate(keys)}

# ------------------------ Step 5: Lateral inhibition ------------------------

def apply_lateral_inhibition(scores: Dict[str,float], peaks: Dict[str,int], sigma: float, lambd: float) -> Dict[str,float]:
    """Penalize a primitive by Gaussian-overlap with others: s_p' = s_p - λ * sum_q≠p κ(t_p, t_q)."""
    prims = list(scores.keys())
    s = scores.copy()
    for i,p in enumerate(prims):
        pen = 0.0
        for j,q in enumerate(prims):
            if p == q: 
                continue
            pen += gaussian_penalty(peaks[p], peaks[q], sigma)
        s[p] = s[p] - lambd * pen
    return s

# ------------------------ Step 4: Strict refinement (gentle) ------------------------

def strict_sequential_refinement(Sres: Dict[str,np.ndarray], proto_width: int,
                                 drop_frac: float = 0.02, max_add: int = 3) -> List[str]:
    proto = half_sine_proto(proto_width)
    keys = list(Sres.keys())
    T = len(next(iter(Sres.values())))
    remaining = set(keys)
    selected: List[str] = []

    peaks = {}
    for p in keys:
        m = np.correlate(Sres[p], proto, mode="same")
        peaks[p] = int(np.argmax(m))

    for _ in range(max_add):
        base_energy = sum(float(np.trapz(Sres[p])) for p in keys)
        best_p, best_drop = None, 0.0

        for p in list(remaining):
            a, b = max(0, peaks[p] - proto_width//2), min(T, peaks[p] + proto_width//2)
            shaved = {q: Sres[q].copy() for q in keys}
            for q in keys:
                if q == p:
                    continue
                shaved[q][a:b] = 0.85 * shaved[q][a:b]
            new_energy = sum(float(np.trapz(shaved[q])) for q in keys)
            drop = (base_energy - new_energy) / max(1e-9, base_energy)
            if drop > best_drop:
                best_drop, best_p = drop, p

        if best_p is not None and best_drop >= drop_frac:
            selected.append(best_p)
            remaining.remove(best_p)
            a, b = max(0, peaks[best_p] - proto_width//2), min(T, peaks[best_p] + proto_width//2)
            for q in keys:
                if q == best_p: 
                    continue
                Sres[q][a:b] = 0.85 * Sres[q][a:b]
        else:
            break

    if not selected:
        top = max(keys, key=lambda p: float(np.max(np.correlate(Sres[p], proto, mode="same"))))
        selected = [top]
    return selected

# ------------------------ Temperature descent ------------------------

def soft_presence_from_energy(Up: Dict[str,float], T: float, floor: float=0.18) -> List[str]:
    """Compute π ∝ exp(−Up/T), keep those with π ≥ floor (ensure at least top-1)."""
    keys = list(Up.keys())
    Uv = np.array([Up[k] for k in keys], dtype=float)
    # Numerically stable softmax over −U/T
    m = np.max(-Uv / max(T, 1e-6))
    ex = np.exp(-Uv / max(T, 1e-6) - m)
    pi = ex / (np.sum(ex) + 1e-12)
    keep = [keys[i] for i,p in enumerate(pi) if p >= floor]
    if not keep:
        keep = [keys[int(np.argmax(pi))]]
    return keep

def anneal_schedule(T0: float, Tmin: float, steps: int) -> List[float]:
    if steps <= 1:
        return [Tmin]
    return list(np.linspace(T0, Tmin, steps))

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
    T = len(next(iter(Sraw.values())))
    d = 8
    B = np.zeros((3, d)); B[0,0]=1.0; B[1,1]=1.0; B[2,2]=1.0
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
    ap = argparse.ArgumentParser(description="Stage 11 — Phase 3 (v4): inhibition + temperature descent")
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=45)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--sigma", type=int, default=9, help="smoother window")
    ap.add_argument("--proto_width", type=int, default=160)
    # Orthogonalization
    ap.add_argument("--rank_r", type=int, default=1)
    # Candidate floor
    ap.add_argument("--cand_floor", type=float, default=0.40, help="relative score floor vs max (0..1)")
    # Inhibition
    ap.add_argument("--inhib_sigma", type=float, default=1.6, help="time-offset kernel sigma (in samples)")
    ap.add_argument("--inhib_lambda", type=float, default=0.35, help="penalty strength")
    # Temperature descent
    ap.add_argument("--T0", type=float, default=1.6)
    ap.add_argument("--Tmin", type=float, default=0.7)
    ap.add_argument("--anneal_steps", type=int, default=2)
    ap.add_argument("--p_floor", type=float, default=0.18, help="min probability to keep under soft presence")
    # Diagnostics
    ap.add_argument("--dump_surfaces_dir", type=str, default="")
    ap.add_argument("--dump_manifold", type=str, default="")
    ap.add_argument("--pi", action="store_true")
    # Null calibration
    ap.add_argument("--null-block-frac", type=float, default=0.16)
    ap.add_argument("--nperm", type=int, default=500)
    # Per-primitive gates (gentle; mainly flip_v)
    ap.add_argument("--strict_gates", action="store_true")
    ap.add_argument("--raw_floor_v", type=float, default=0.10)
    ap.add_argument("--res_ceiling_v", type=float, default=0.70)
    # Output
    ap.add_argument("--csv", type=str, default="stage11_v4.csv")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = []
    agg = dict(acc=0, grid=0.0, P=0.0, R=0.0, F1=0.0, J=0.0, H=0.0, O=0.0,
               margin_mu=0.0, margin_min=0.0, PI=0.0)
    all_margins = []

    for i in range(1, args.samples+1):
        # --- generate sample ---
        # (replace with real traces from your pipeline)
        sample = make_sample(rng, T=args.T, noise=args.noise)

        # --- SP pipeline ---
        # 1) orthogonalize residuals (subspace projection / adaptive filtering)
        Eperp = low_rank_orthogonalize(sample.traces, rank_r=args.rank_r)
        Sres = {p: moving_average(Eperp[p], k=args.sigma) for p in PRIMS}
        Sraw = {p: moving_average(sample.traces[p], k=args.sigma) for p in PRIMS}
        Scm  = moving_average(np.stack([sample.traces[p] for p in PRIMS],0).mean(0), k=args.sigma)

        # 2) matched-filter peaks
        proto = half_sine_proto(args.proto_width)
        peak_idx = {p: int(np.argmax(np.correlate(Sres[p], proto, mode="same"))) for p in PRIMS}

        # 3) calibrated z-scores
        z_res = {p: perm_null_z(Sres[p], proto, peak_idx[p], args.proto_width, rng, nperm=args.nperm, block_frac=args.null_block_frac) for p in PRIMS}
        z_raw = {p: perm_null_z(Sraw[p], proto, peak_idx[p], args.proto_width, rng, nperm=args.nperm, block_frac=args.null_block_frac) for p in PRIMS}
        z_cm  = {p: perm_null_z(Scm,     proto, peak_idx[p], args.proto_width, rng, nperm=args.nperm, block_frac=args.null_block_frac) for p in PRIMS}

        # 4) energy / score before inhibition
        w_res, w_raw, w_cm = 1.0, 0.45, 0.25  # slightly lighter CM penalty to aid recall
        score = {p: w_res*max(0.0, z_res[p]) + w_raw*max(0.0, z_raw[p]) - w_cm*max(0.0, z_cm[p]) for p in PRIMS}

        # 5) lateral inhibition (time-offset aware)
        score_inhib = apply_lateral_inhibition(score, peak_idx, sigma=args.inhib_sigma, lambd=args.inhib_lambda)

        # 6) candidate pool by relative floor
        smax = max(score_inhib.values()) + 1e-12
        cands = [p for p in PRIMS if score_inhib[p] >= args.cand_floor * smax]
        if not cands:
            cands = [max(PRIMS, key=lambda q: score_inhib[q])]

        # 7) temperature descent presence (soft presence)
        # build Up (lower is better) from z-scores w/ inhibition-adjusted score
        # Use a monotone map: Up = -score_inhib[p]
        Up = {p: -score_inhib[p] for p in cands}
        keep_soft = set()
        for Tcur in anneal_schedule(args.T0, args.Tmin, args.anneal_steps):
            ks = soft_presence_from_energy(Up, Tcur, floor=args.p_floor)
            keep_soft.update(ks)
        keep_soft = list(sorted(keep_soft))

        # 8) strict residual refinement (gentle) on the soft set
        keep = strict_sequential_refinement({p: Sres[p].copy() for p in keep_soft}, args.proto_width, drop_frac=0.02, max_add=3)

        # 9) per-primitive gentle gate (optional): flip_v sanity
        if args.strict_gates and "flip_v" in keep:
            pv = peak_idx["flip_v"]
            Tlen = len(Sraw["flip_v"])
            a, b = max(0, pv - args.proto_width//2), min(Tlen, pv + args.proto_width//2)
            raw_win = float(np.mean(Sraw["flip_v"][a:b]))
            res_win = float(np.mean(Sres["flip_v"][a:b]))
            if raw_win < args.raw_floor_v or res_win > args.res_ceiling_v:
                keep = [p for p in keep if p != "flip_v"]

        # 10) ordering by matched filter on Sraw
        peaks = {}
        for p in keep:
            m = np.correlate(Sraw[p], proto, mode="same")
            peaks[p] = int(np.argmax(m))
        order = sorted(keep, key=lambda p: peaks[p])

        # --- scoring / metrics ---
        gp = apply_sequence(sample.grid_in, order)
        ok = int(np.array_equal(gp, sample.grid_out_true))
        gs = grid_similarity(gp, sample.grid_out_true)
        sm = set_metrics(sample.order_true, keep)

        zcm = _zscore(Scm)
        Utime = {}
        for p in PRIMS:
            Utime[p] = -(_zscore(Sres[p]) + 0.4*_zscore(Sraw[p]) - 0.25*zcm)

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

    print("\n[Stage11 v4] Summary:", {k: (round(v,3) if isinstance(v,float) else v) for k,v in summary.items()})
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
