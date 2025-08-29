
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage11-well-benchmark-v6.py
----------------------------
Phase 5 (v6): Recall recovery with adaptive, SP-inspired controls.

Adds:
- CFAR candidate gating (per-primitive, per-sample noise-aware)
- Multi-scale matched filter (two prototype widths; take max z)
- Overlap-adaptive inhibition (lambda_{pq} = lambda0 * exp(-Δt^2 / (2σ^2)))
- Cumulative refinement (SPRT-style evidence across anneal steps)
- Fast/Full mode presets

Retains diagnostics: PI, surfaces, manifold dump, margins.
This is a synthetic harness; wire your real traces where noted.
"""

import argparse, os, csv, json
from dataclasses import dataclass
from typing import Dict, List, Tuple
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

def gaussian(tdiff: float, sigma: float) -> float:
    return float(np.exp(-0.5 * (tdiff / (sigma + 1e-12))**2))

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

# ------------------------ Multi-scale matched filter + CFAR ------------------------

def matched_z(sig: np.ndarray, center: int, width: int, rng: np.random.Generator, nperm: int, block_frac: float) -> float:
    proto = half_sine_proto(width)
    return perm_null_z(sig, proto, center, width, rng, nperm=nperm, block_frac=block_frac)

def multi_scale_z(sig: np.ndarray, center: int, widths: List[int], rng: np.random.Generator, nperm: int, block_frac: float) -> float:
    zs = [matched_z(sig, center, w, rng, nperm, block_frac) for w in widths]
    return float(max(zs))

def cfar_threshold(series: np.ndarray, peak: int, guard: int, bg: int) -> Tuple[float, float]:
    """Compute mean/std from a ring excluding guard cells around the peak."""
    T = len(series)
    lo1, hi1 = max(0, peak - (guard + bg)), max(0, peak - guard)
    lo2, hi2 = min(T, peak + guard), min(T, peak + guard + bg)
    ring = []
    if hi1 > lo1: ring.append(series[lo1:hi1])
    if hi2 > lo2: ring.append(series[lo2:hi2])
    if not ring:
        return float(series.mean()), float(series.std() + 1e-9)
    ring = np.concatenate(ring)
    return float(ring.mean()), float(ring.std() + 1e-9)

# ------------------------ Overlap-adaptive inhibition ------------------------

def apply_adaptive_inhibition(scores: Dict[str,float], peaks: Dict[str,int], lambda0: float, sigma: float) -> Dict[str,float]:
    s = scores.copy()
    prims = list(scores.keys())
    for p in prims:
        pen = 0.0
        for q in prims:
            if p == q: continue
            pen += lambda0 * gaussian(abs(peaks[p]-peaks[q]), sigma)
        s[p] = s[p] - pen
    return s

# ------------------------ Cumulative refinement (SPRT-style) ------------------------

def cumulative_refinement(Sres: Dict[str,np.ndarray], peaks: Dict[str,int], proto_width: int,
                          anneal_steps: int, drop_floor: float, tau_total: float) -> List[str]:
    """
    Accumulate residual-drop evidence across virtual anneal steps.
    Accept p if cumulative drop ≥ tau_total; each step requires at least drop_floor to count.
    """
    proto = half_sine_proto(proto_width)
    keys = list(Sres.keys())
    T = len(next(iter(Sres.values())))
    selected = []

    # Precompute base energies per step (simulate anneal steps as repeated shaving)
    cumulative = {p: 0.0 for p in keys}
    Swork = {p: Sres[p].copy() for p in keys}

    for _ in range(max(1, anneal_steps)):
        base_energy = sum(float(np.trapz(Swork[p])) for p in keys)

        # Evaluate each candidate's effect this step
        step_drop = {p: 0.0 for p in keys}
        for p in keys:
            a, b = max(0, peaks[p] - proto_width//2), min(T, peaks[p] + proto_width//2)
            shaved = {q: Swork[q].copy() for q in keys}
            for q in keys:
                if q == p: continue
                shaved[q][a:b] = 0.88 * shaved[q][a:b]
            new_energy = sum(float(np.trapz(shaved[q])) for q in keys)
            drop = (base_energy - new_energy) / max(1e-9, base_energy)
            step_drop[p] = max(0.0, drop)

        # Update cumulative only for those above the minimal floor
        for p in keys:
            if step_drop[p] >= drop_floor:
                cumulative[p] += step_drop[p]

        # Commit a gentle shave to simulate anneal progression around the currently strongest p
        best_p = max(keys, key=lambda k: step_drop[k])
        a, b = max(0, peaks[best_p] - proto_width//2), min(T, peaks[best_p] + proto_width//2)
        for q in keys:
            if q == best_p: continue
            Swork[q][a:b] = 0.92 * Swork[q][a:b]

    for p in keys:
        if cumulative[p] >= tau_total:
            selected.append(p)

    if not selected:
        # Fallback: keep the strongest
        selected = [max(keys, key=lambda k: cumulative[k])]
    return selected

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

# ------------------------------- main -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage 11 — Phase 5 (v6): CFAR + multi-scale + adaptive inhibition + cumulative refinement")
    ap.add_argument("--mode", type=str, default="full", choices=["fast","full"])
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--sigma", type=int, default=9, help="smoother window")
    ap.add_argument("--proto_width", type=int, default=160, help="base prototype width")
    ap.add_argument("--proto_widths", type=str, default="130,210", help="comma list of extra widths; include base implicitly")
    # Orthogonalization
    ap.add_argument("--rank_r", type=int, default=1)
    # CFAR
    ap.add_argument("--cfar_k", type=float, default=2.2)
    ap.add_argument("--cfar_guard", type=int, default=20)
    ap.add_argument("--cfar_bg", type=int, default=60)
    # Inhibition
    ap.add_argument("--lambda0", type=float, default=0.25)
    ap.add_argument("--inhib_sigma", type=float, default=1.6)
    # Anneal
    ap.add_argument("--T0", type=float, default=2.6)
    ap.add_argument("--Tmin", type=float, default=0.7)
    ap.add_argument("--anneal_steps", type=int, default=5)
    ap.add_argument("--p_floor", type=float, default=0.12)
    # Cumulative refinement
    ap.add_argument("--sprt_tau", type=float, default=0.02, help="cumulative residual drop to accept")
    ap.add_argument("--drop_floor", type=float, default=0.006, help="minimum per-step drop to accumulate")
    # Diagnostics
    ap.add_argument("--dump_surfaces_dir", type=str, default="")
    ap.add_argument("--dump_manifold", type=str, default="")
    ap.add_argument("--pi", action="store_true")
    # Null calibration
    ap.add_argument("--null-block-frac", type=float, default=0.16)
    ap.add_argument("--nperm", type=int, default=500)
    # Output
    ap.add_argument("--csv", type=str, default="stage11_v6.csv")
    args = ap.parse_args()

    # Mode presets
    if args.mode == "fast":
        args.nperm = min(args.nperm, 150)
        args.samples = min(args.samples, 20)
        args.anneal_steps = max(args.anneal_steps, 3)
        args.proto_widths = "160"  # single scale

    rng = np.random.default_rng(args.seed)

    # Prepare extra widths list (include base width implicitly)
    widths = [args.proto_width]
    try:
        extra = [int(x.strip()) for x in args.proto_widths.split(",") if x.strip()]
        for w in extra:
            if w not in widths:
                widths.append(w)
    except Exception:
        pass
    widths = sorted(set(widths))

    rows = []
    agg = dict(acc=0, grid=0.0, P=0.0, R=0.0, F1=0.0, J=0.0, H=0.0, O=0.0,
               margin_mu=0.0, margin_min=0.0, PI=0.0)
    all_margins = []

    for i in range(1, args.samples+1):
        sample = make_sample(rng, T=args.T, noise=args.noise)

        # 1) Orthogonalization
        Eperp = low_rank_orthogonalize(sample.traces, rank_r=args.rank_r)
        Sres = {p: moving_average(Eperp[p], k=args.sigma) for p in PRIMS}
        Sraw = {p: moving_average(sample.traces[p], k=args.sigma) for p in PRIMS}
        Scm  = moving_average(np.stack([sample.traces[p] for p in PRIMS],0).mean(0), k=args.sigma)

        # 2) Matched filter peaks from residual channel
        base_proto = half_sine_proto(args.proto_width)
        peak_idx = {p: int(np.argmax(np.correlate(Sres[p], base_proto, mode="same"))) for p in PRIMS}

        # 3) Multi-scale z-score at peaks (per primitive)
        rng_i = np.random.default_rng(123450 + i)
        z_res = {}
        for p in PRIMS:
            z_res[p] = multi_scale_z(Sres[p], peak_idx[p], widths, rng_i, args.nperm, args.null_block_frac)

        # We'll also compute z for raw & CM at base width for energy shaping (optional)
        z_raw = {}
        z_cm = {}
        for p in PRIMS:
            z_raw[p] = perm_null_z(Sraw[p], base_proto, peak_idx[p], args.proto_width, rng_i, nperm=max(120, args.nperm//3), block_frac=args.null_block_frac)
            z_cm[p]  = perm_null_z(Scm,      base_proto, peak_idx[p], args.proto_width, rng_i, nperm=max(120, args.nperm//3), block_frac=args.null_block_frac)

        # 4) CFAR gating
        # Build a rough "series" by sliding correlation (approximate with Sres itself z-scored)
        # For gating, we use local ring stats around peak on z_res surrogate.
        z_series = {}
        for p in PRIMS:
            z_series[p] = _zscore(Sres[p])  # surrogate series for local stats
        passed = []
        for p in PRIMS:
            mu, sd = cfar_threshold(z_series[p], peak_idx[p], guard=args.cfar_guard, bg=args.cfar_bg)
            thr = mu + args.cfar_k * sd
            if z_series[p][peak_idx[p]] >= thr:
                passed.append(p)
        if not passed:
            # Ensure at least one candidate remains: pick max z_res
            passed = [max(PRIMS, key=lambda q: z_res[q])]

        # 5) Base score + adaptive inhibition
        w_res, w_raw, w_cm = 1.0, 0.45, 0.25
        score = {p: w_res*max(0.0, z_res[p]) + w_raw*max(0.0, z_raw[p]) - w_cm*max(0.0, z_cm[p]) for p in passed}
        score_inhib = apply_adaptive_inhibition(score, {p: peak_idx[p] for p in passed}, lambda0=args.lambda0, sigma=args.inhib_sigma)

        # 6) Temperature soft presence
        # Convert to energies Up = -score_inhib
        Up = {p: -score_inhib[p] for p in passed}
        keys = list(Up.keys())

        def soft_keep(Tcur: float) -> List[str]:
            Uv = np.array([Up[k] for k in keys], dtype=float)
            if len(Uv) == 0:
                return []
            m = np.max(-Uv / max(Tcur, 1e-6))
            ex = np.exp(-Uv / max(Tcur, 1e-6) - m)
            pi = ex / (np.sum(ex) + 1e-12)
            keep = [keys[i] for i,pv in enumerate(pi) if pv >= args.p_floor]
            if not keep:
                keep = [keys[int(np.argmax(pi))]]
            return keep

        # Anneal schedule
        Ts = list(np.linspace(args.T0, args.Tmin, max(1, args.anneal_steps)))

        # 7) Cumulative refinement across anneal steps
        # Build Sres subset for current candidates
        cumul_set = set()
        for Tcur in Ts:
            cumul_set.update(soft_keep(Tcur))
        Ssub = {p: Sres[p].copy() for p in cumul_set} if cumul_set else {passed[0]: Sres[passed[0]].copy()}
        keep = cumulative_refinement(Ssub, {p: peak_idx[p] for p in Ssub.keys()}, args.proto_width,
                                     anneal_steps=len(Ts), drop_floor=args.drop_floor, tau_total=args.sprt_tau)

        # 8) Ordering by matched filter on Sraw
        proto = base_proto
        peaks = {}
        for p in keep:
            m = np.correlate(Sraw[p], proto, mode="same")
            peaks[p] = int(np.argmax(m))
        order = sorted(keep, key=lambda p: peaks[p])

        # --- Metrics & dumps ---
        gp = apply_sequence(sample.grid_in, order)
        ok = int(np.array_equal(gp, sample.grid_out_true))
        gs = grid_similarity(gp, sample.grid_out_true)
        sm = set_metrics(sample.order_true, keep)

        zcm_series = _zscore(Scm)
        Utime = {}
        for p in PRIMS:
            Utime[p] = -(_zscore(Sres[p]) + 0.4*_zscore(Sraw[p]) - 0.25*zcm_series)

        margins = []
        for p in PRIMS:
            t_true = sample.centers_true.get(p, None)
            if t_true is None: continue
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

    # Write outputs
    if rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows: w.writerow(r)
    with open(os.path.splitext(args.csv)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[Stage11 v6] Summary:", {k: (round(v,3) if isinstance(v,float) else v) for k,v in summary.items()})
    print("[Stage11 v6] CSV:", args.csv)
    print("[Stage11 v6] Summary JSON:", os.path.splitext(args.csv)[0] + "_summary.json")

if __name__ == "__main__":
    main()
