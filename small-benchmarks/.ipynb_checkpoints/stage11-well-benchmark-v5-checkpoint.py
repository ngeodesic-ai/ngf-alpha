
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage11-well-benchmark-v5.py
----------------------------
Phase 4: Recall recovery with safer defaults + small operating-point sweep.

What’s new vs v4:
- Recall-oriented defaults (wider candidate set, softer inhibition, hotter/longer anneal,
  gentler residual drop threshold).
- Optional quick sweep over a tiny grid of {cand_floor, inhib_lambda, res_drop_frac}
  to pick an operating point that maximizes F1 subject to Recall≥0.90 and Halluc≤0.26.

Still a synthetic harness; replace the trace generator with your pipeline hooks for real data.

python3 stage11-well-benchmark-v5.py --samples 50 --sweep quick --pi \
  --dump_surfaces_dir dumps/v5_surfaces --dump_manifold dumps/v5_manifold.npz

"""

import argparse, os, csv, json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import itertools
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

# ------------------------ Step 2: Orthogonalization ------------------------

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
                                 drop_frac: float = 0.01, max_add: int = 3) -> List[str]:
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
                shaved[q][a:b] = 0.88 * shaved[q][a:b]
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
                Sres[q][a:b] = 0.88 * Sres[q][a:b]
        else:
            break

    if not selected:
        top = max(keys, key=lambda p: float(np.max(np.correlate(Sres[p], proto, mode="same"))))
        selected = [top]
    return selected

# ------------------------ Temperature descent ------------------------

def soft_presence_from_energy(Up: Dict[str,float], T: float, floor: float=0.15) -> List[str]:
    keys = list(Up.keys())
    Uv = np.array([Up[k] for k in keys], dtype=float)
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
    X = np.zeros((T, d))
    B = np.zeros((3, d)); B[0,0]=1.0; B[1,1]=1.0; B[2,2]=1.0
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

# ------------------------------- runner -------------------------------

def evaluate_on_dataset(dataset, params, dump_surfaces_dir="", dump_manifold_path="", pi=False, proto_width=160, sigma=9,
                        null_block_frac=0.16, nperm=500, inhib_sigma=1.6):
    # unpack params
    rank_r = params["rank_r"]; cand_floor = params["cand_floor"]
    inhib_lambda = params["inhib_lambda"]; T0 = params["T0"]; Tmin = params["Tmin"]
    anneal_steps = params["anneal_steps"]; p_floor = params["p_floor"]
    res_drop_frac = params["res_drop_frac"]

    rows = []
    agg = dict(acc=0, grid=0.0, P=0.0, R=0.0, F1=0.0, J=0.0, H=0.0, O=0.0,
               margin_mu=0.0, margin_min=0.0, PI=0.0)
    all_margins = []

    proto = half_sine_proto(proto_width)

    for i, sample in enumerate(dataset, start=1):
        # --- SP pipeline ---
        Eperp = low_rank_orthogonalize(sample.traces, rank_r=rank_r)
        Sres = {p: moving_average(Eperp[p], k=sigma) for p in PRIMS}
        Sraw = {p: moving_average(sample.traces[p], k=sigma) for p in PRIMS}
        Scm  = moving_average(np.stack([sample.traces[p] for p in PRIMS],0).mean(0), k=sigma)

        # peaks
        peak_idx = {p: int(np.argmax(np.correlate(Sres[p], proto, mode="same"))) for p in PRIMS}

        # z-scores
        rng = np.random.default_rng(12345 + i)  # deterministic per-sample
        z_res = {p: perm_null_z(Sres[p], proto, peak_idx[p], proto_width, rng, nperm=nperm, block_frac=null_block_frac) for p in PRIMS}
        z_raw = {p: perm_null_z(Sraw[p], proto, peak_idx[p], proto_width, rng, nperm=nperm, block_frac=null_block_frac) for p in PRIMS}
        z_cm  = {p: perm_null_z(Scm,     proto, peak_idx[p], proto_width, rng, nperm=nperm, block_frac=null_block_frac) for p in PRIMS}

        # scores
        w_res, w_raw, w_cm = 1.0, 0.45, 0.25
        score = {p: w_res*max(0.0, z_res[p]) + w_raw*max(0.0, z_raw[p]) - w_cm*max(0.0, z_cm[p]) for p in PRIMS}

        # inhibition
        score_inhib = apply_lateral_inhibition(score, peak_idx, sigma=inhib_sigma, lambd=inhib_lambda)

        # candidate pool
        smax = max(score_inhib.values()) + 1e-12
        cands = [p for p in PRIMS if score_inhib[p] >= cand_floor * smax]
        if not cands:
            cands = [max(PRIMS, key=lambda q: score_inhib[q])]

        # temperature descent (soft presence)
        Up = {p: -score_inhib[p] for p in cands}
        keep_soft = set()
        for Tcur in anneal_schedule(T0, Tmin, anneal_steps):
            ks = soft_presence_from_energy(Up, Tcur, floor=p_floor)
            keep_soft.update(ks)
        keep_soft = list(sorted(keep_soft))

        # refinement
        keep = strict_sequential_refinement({p: Sres[p].copy() for p in keep_soft}, proto_width, drop_frac=res_drop_frac, max_add=3)

        # ordering
        peaks = {}
        for p in keep:
            m = np.correlate(Sraw[p], proto, mode="same")
            peaks[p] = int(np.argmax(m))
        order = sorted(keep, key=lambda p: peaks[p])

        # metrics
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
            w = proto_width
            a, b = max(0, t_true - w//2), min(Tlen, t_true + w//2)
            u_true = float(np.min(Utime[p][a:b]))
            others = [q for q in PRIMS if q != p]
            u_false = min(float(np.min(Utime[q])) for q in others)
            margins.append(u_false - u_true)
        margin_mu = float(np.mean(margins)) if margins else 0.0
        margin_min = float(min(margins)) if margins else 0.0
        all_margins.extend(margins)

        PI = compute_phantom_index(Utime, sample.centers_true, window=proto_width) if pi else 0.0

        if dump_surfaces_dir:
            dump_surfaces(i, dump_surfaces_dir, Sraw, Sres, Utime)
        if dump_manifold_path and i == 1:
            synthetic_manifold_dump(dump_manifold_path, Sraw, Utime)

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

    n = float(len(rows))
    summary = {k: (v/n if isinstance(v, (int,float)) else v) for k,v in agg.items()}
    if all_margins:
        summary["margin_mu"] = float(np.mean(all_margins))
        summary["margin_min"] = float(np.min(all_margins))
    return rows, summary

# ------------------------------- main -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage 11 — Phase 4 (v5): recall recovery + sweep")
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=46)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--sigma", type=int, default=9, help="smoother window")
    ap.add_argument("--proto_width", type=int, default=160)
    # Defaults (recall-oriented)
    ap.add_argument("--rank_r", type=int, default=1)
    ap.add_argument("--cand_floor", type=float, default=0.30)
    ap.add_argument("--inhib_sigma", type=float, default=1.6)
    ap.add_argument("--inhib_lambda", type=float, default=0.25)
    ap.add_argument("--T0", type=float, default=2.2)
    ap.add_argument("--Tmin", type=float, default=0.7)
    ap.add_argument("--anneal_steps", type=int, default=4)
    ap.add_argument("--p_floor", type=float, default=0.15)
    ap.add_argument("--res_drop_frac", type=float, default=0.01)
    # Diagnostics
    ap.add_argument("--dump_surfaces_dir", type=str, default="")
    ap.add_argument("--dump_manifold", type=str, default="")
    ap.add_argument("--pi", action="store_true")
    # Null calibration
    ap.add_argument("--null-block-frac", type=float, default=0.16)
    ap.add_argument("--nperm", type=int, default=500)
    # Sweep
    ap.add_argument("--sweep", type=str, default="", choices=["","quick"], help="run a tiny grid sweep to pick operating point")
    ap.add_argument("--csv", type=str, default="stage11_v5.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Pre-generate a fixed dataset so sweeps compare on identical samples
    dataset = [make_sample(rng, T=args.T, noise=args.noise) for _ in range(args.samples)]

    def run_with(params, tag=""):
        rows, summary = evaluate_on_dataset(
            dataset, params,
            dump_surfaces_dir=args.dump_surfaces_dir if not tag else "",
            dump_manifold_path=args.dump_manifold if (not tag) else "",
            pi=args.pi, proto_width=args.proto_width, sigma=args.sigma,
            null_block_frac=args.null_block_frac, nperm=args.nperm, inhib_sigma=args.inhib_sigma
        )
        return rows, summary

    # Base params from CLI
    base_params = dict(
        rank_r=args.rank_r, cand_floor=args.cand_floor,
        inhib_lambda=args.inhib_lambda, T0=args.T0, Tmin=args.Tmin,
        anneal_steps=args.anneal_steps, p_floor=args.p_floor,
        res_drop_frac=args.res_drop_frac
    )

    sweep_results = []
    best = None

    if args.sweep == "quick":
        cand_floor_vals = [max(0.20, args.cand_floor-0.10), args.cand_floor, min(0.60, args.cand_floor+0.10)]
        inhib_lambda_vals = [max(0.10, args.inhib_lambda-0.10), args.inhib_lambda, min(0.60, args.inhib_lambda+0.10)]
        res_drop_vals = [max(0.005, args.res_drop_frac-0.005), args.res_drop_frac, min(0.04, args.res_drop_frac+0.01)]
        for cf, il, rdf in itertools.product(cand_floor_vals, inhib_lambda_vals, res_drop_vals):
            params = dict(base_params); params.update(dict(cand_floor=cf, inhib_lambda=il, res_drop_frac=rdf))
            _, summ = run_with(params, tag="sweep")
            # Constraint-first selection: Recall ≥ 0.90 and Halluc ≤ 0.26 preferred; otherwise relax recall to ≥0.85 then ≥0.80
            score_tuple = (
                int(summ["R"] >= 0.90 and summ["H"] <= 0.26),
                int(summ["R"] >= 0.85 and summ["H"] <= 0.26),
                int(summ["R"] >= 0.80 and summ["H"] <= 0.26),
                round(summ["F1"], 4)
            )
            sweep_results.append(dict(cand_floor=cf, inhib_lambda=il, res_drop_frac=rdf, **{k:float(round(v,4)) for k,v in summ.items()}))
            if (best is None) or (score_tuple > best["score_tuple"]):
                best = dict(params=params, summary=summ, score_tuple=score_tuple)

    # Final run: base or best
    final_params = best["params"] if best else base_params
    rows, summary = run_with(final_params)

    # Write outputs
    if rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows: w.writerow(r)
    with open(os.path.splitext(args.csv)[0] + "_summary.json", "w") as f:
        json.dump(dict(summary=summary, chosen_params=final_params, sweep_results=sweep_results), f, indent=2)

    # Print console summary
    print("[Stage11 v5] Params:", final_params)
    print("[Stage11 v5] Summary:", {k: (round(v,3) if isinstance(v,float) else v) for k,v in summary.items()})
    if best:
        print("[Stage11 v5] Best-by-constraints F1:", round(best["summary"]["F1"],3))

if __name__ == "__main__":
    main()
