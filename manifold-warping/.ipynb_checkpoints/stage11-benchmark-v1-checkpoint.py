#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 11 — Well Parser Benchmark (v1)  [fixed]
----------------------------------------------
- Adds tunable refinement (--refine_frac)
- Adds sparsity clamps (--lambda_sparsity, --z_perp_gate)
- Adds MDL-style cumulative objective in selection loop
- Adds compat print mode (--compat_hidden_target) and a simple Stock baseline
- Fixes argparse types and earlier duplicate flag issues

python3 stage11-benchmark-v1.py --samples 150 --compat_hidden_target \
  --sigma 11 --proto_width 150 --rank 4 \
  --nperm 320 --block 64 --tapers hann,hamming,blackman \
  --temp0 0.08 --temp_min 0.08 --anneal 0.40 \
  --inhib_strength 1.5 --inhib_sigma 18 \
  --presence_floor 0.95 \
  --refine_frac 0.7 \
  --lambda_sparsity 1.00 \
  --z_perp_gate 1.6 \
  --weights 1.0,0.35,0.8

"""
import argparse, csv, math
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- Utilities -----------------------------

def moving_average(x: np.ndarray, k: int = 9) -> np.ndarray:
    if k <= 1:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(xp, np.ones(k)/k, mode="valid")

def half_sine(width: int) -> np.ndarray:
    return np.sin(np.linspace(0, np.pi, width))

def half_sine_proto(width: int) -> np.ndarray:
    p = half_sine(width)
    return p / (np.linalg.norm(p) + 1e-8)

def gaussian_bump(T: int, center: int, width: int, amp: float = 1.0) -> np.ndarray:
    t = np.arange(T)
    sig2 = (width/2.355)**2  # FWHM->sigma
    return amp * np.exp(-(t-center)**2 / (2*sig2))

def add_noise(x: np.ndarray, sigma: float, rng) -> np.ndarray:
    return x + rng.normal(0, sigma, size=x.shape)

def common_mode(traces: Dict[str, np.ndarray]) -> np.ndarray:
    arr = np.stack([traces[p] for p in traces.keys()], 0)
    return arr.mean(0)

def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    sd = x.std() + 1e-12
    return (x - mu) / sd

def circ_shift(x: np.ndarray, k: int) -> np.ndarray:
    k = int(k) % len(x)
    if k == 0: return x
    return np.concatenate([x[-k:], x[:-k]])

def block_circ_shift(x: np.ndarray, rng, block: int) -> np.ndarray:
    """Rotate by a random multiple of `block` (preserve local autocorrelation)."""
    T = len(x)
    if block <= 1:
        s = rng.integers(1, T-1)
    else:
        nblocks = max(1, T // block)
        s = int(rng.integers(1, nblocks)) * block
    return circ_shift(x, s)

def normalized_corr_window(sig: np.ndarray, proto: np.ndarray, center_idx: int, width: int,
                           taper: str = None) -> float:
    """Cosine similarity on a centered window, optionally tapered (hann/hamming/blackman)."""
    T = len(sig)
    a, b = max(0, center_idx - width//2), min(T, center_idx + width//2)
    w = sig[a:b].astype(float)
    if len(w) < 3:
        return 0.0
    pr = proto[:len(w)].astype(float)
    # Tapers
    if taper is not None:
        if taper.lower() == "hann":
            win = 0.5 * (1 - np.cos(2*np.pi*np.arange(len(w))/(len(w)-1)))
        elif taper.lower() == "hamming":
            win = 0.54 - 0.46 * np.cos(2*np.pi*np.arange(len(w))/(len(w)-1))
        elif taper.lower() == "blackman":
            n = np.arange(len(w))
            win = 0.42 - 0.5*np.cos(2*np.pi*n/(len(w)-1)) + 0.08*np.cos(4*np.pi*n/(len(w)-1))
        else:
            win = np.ones_like(w)
        w = w * win; pr = pr * win
    # demean then cosine
    w = w - w.mean(); pr = pr - pr.mean()
    denom = (np.linalg.norm(w) * np.linalg.norm(pr) + 1e-8)
    return float(np.dot(w, pr) / denom)

def multitaper_corr(sig: np.ndarray, proto: np.ndarray, center_idx: int, width: int,
                    tapers: List[str]) -> float:
    if not tapers:
        return normalized_corr_window(sig, proto, center_idx, width, taper=None)
    vals = [normalized_corr_window(sig, proto, center_idx, width, taper=t) for t in tapers]
    return float(np.mean(vals))

def perm_null_block_z(sig: np.ndarray, proto: np.ndarray, peak_idx: int, width: int, rng,
                      nperm: int = 200, block: int = 32, tapers: List[str] = None) -> Tuple[float, float]:
    """Return (z, obs_corr) under a block-circular permutation null with optional multitaper averaging."""
    obs = multitaper_corr(sig, proto, peak_idx, width, tapers or [])
    null = np.empty(nperm, dtype=float)
    for _ in range(nperm):
        x = block_circ_shift(sig, rng, block)
        null[_] = multitaper_corr(x, proto, peak_idx, width, tapers or [])
    mu, sd = float(null.mean()), float(null.std() + 1e-8)
    z = (obs - mu) / sd
    return float(z), float(obs)

def span_complement_project(S: np.ndarray, rank: int) -> np.ndarray:
    """
    Remove top-`rank` shared components across channels from S (T x K).
    Returns residualized S_perp of same shape.
    """
    X = S - S.mean(axis=0, keepdims=True)        # center each channel (timewise)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    r = int(max(0, min(rank, len(s))))
    if r == 0: return X
    Ur = U[:, :r]                                # time-basis for shared variance
    X_res = X - Ur @ (Ur.T @ X)                  # project to complement
    return X_res

def perpendicular_energy(traces: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    mu = common_mode(traces)
    return {p: np.clip(traces[p] - mu, 0, None) for p in traces.keys()}

# ----------------------------- World (synthetic ARC-style) -----------------------------

PRIMS = ["flip_h","flip_v","rotate"]

@dataclass
class Sample:
    grid_in: np.ndarray
    tasks_true: List[str]
    order_true: List[str]
    grid_out_true: np.ndarray
    traces: Dict[str, np.ndarray]
    T: int

def random_grid(rng, H=8, W=8, ncolors=6):
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

def gen_traces(tasks: List[str], T: int, rng, noise=0.02, cm_amp=0.02, overlap=0.5,
               amp_jitter=0.4, distractor_prob=0.4) -> Dict[str, np.ndarray]:
    # three centers to allow up to 3 tasks
    base = np.array([0.20, 0.50, 0.80]) * T
    centers = ((1.0 - overlap) * base + overlap * (T * 0.50)).astype(int)
    width = int(max(12, T * 0.10))
    t = np.arange(T)
    cm = cm_amp * (1.0 + 0.2 * np.sin(2*np.pi * t / max(30, T//6)))
    traces = {p: np.zeros(T, float) for p in PRIMS}
    # true bumps with jitter and amplitude noise
    for i, prim in enumerate(tasks):
        c = centers[i % len(centers)]
        amp = max(0.25, 1.0 + rng.normal(0, amp_jitter))
        c_jit = int(np.clip(c + rng.integers(-width//5, width//5 + 1), 0, T-1))
        traces[prim] += gaussian_bump(T, c_jit, width, amp=amp)
    # distractors
    for p in PRIMS:
        if p not in tasks and rng.random() < distractor_prob:
            c = int(rng.uniform(T*0.15, T*0.85))
            amp = max(0.25, 1.0 + rng.normal(0, amp_jitter))
            traces[p] += gaussian_bump(T, c, width, amp=0.9*amp)
    # add common-mode + noise
    for p in PRIMS:
        traces[p] = np.clip(traces[p] + cm, 0, None)
        traces[p] = add_noise(traces[p], noise, rng)
    return traces

def make_sample(rng, T=720, n_tasks=(1,3), grid_shape=(8,8), noise=0.02,
                allow_repeats=False, cm_amp=0.02, overlap=0.5, amp_jitter=0.4,
                distractor_prob=0.4) -> Sample:
    k = rng.integers(n_tasks[0], n_tasks[1]+1)
    tasks = list(rng.choice(PRIMS, size=k, replace=allow_repeats))
    if not allow_repeats:
        rng.shuffle(tasks)
    g0 = random_grid(rng, *grid_shape)
    g1 = apply_sequence(g0, tasks)
    traces = gen_traces(tasks, T=T, rng=rng, noise=noise, cm_amp=cm_amp,
                        overlap=overlap, amp_jitter=amp_jitter, distractor_prob=distractor_prob)
    return Sample(g0, tasks, tasks, g1, traces, T)

# ----------------------------- Metrics -----------------------------

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

# ----------------------------- Stage-11 Well Parser -----------------------------

def estimate_peak_indices(S_channel: Dict[str, np.ndarray], proto: np.ndarray) -> Dict[str, int]:
    """Return index of max correlation with prototype per channel (same-length corr)."""
    peak = {}
    for p, sig in S_channel.items():
        m = np.correlate(zscore(sig), proto, mode="same")
        peak[p] = int(np.argmax(m))
    return peak

def subtract_aligned_proto(S: np.ndarray, pick_idx: int, peak_idx: int, proto: np.ndarray, width: int) -> np.ndarray:
    """
    Subtract a scaled prototype around peak_idx for the picked channel from all channels.
    S is (T x K). We fit a per-channel scalar within the window and subtract that shaped proto.
    """
    T, K = S.shape
    a, b = max(0, peak_idx - width//2), min(T, peak_idx + width//2)
    L = b - a
    if L <= 2:
        return S
    pr = proto[:L]
    pr_c = pr - pr.mean()
    pr_den = (np.dot(pr_c, pr_c) + 1e-8)
    S2 = S.copy()
    for k in range(K):
        seg = S[a:b, k]
        seg_c = seg - seg.mean()
        alpha = float(np.dot(seg_c, pr_c) / pr_den)  # LS fit
        S2[a:b, k] = seg - alpha * pr  # subtract
    return S2

def well_parser(traces: Dict[str, np.ndarray],
                rng,
                sigma: int = 9,
                proto_width: int = 160,
                rank: int = 2,
                weights: Tuple[float,float,float] = (1.0, 0.4, 0.3),
                nperm: int = 200,
                block: int = 32,
                tapers: List[str] = None,
                temp0: float = 1.2,
                temp_min: float = 0.15,
                anneal: float = 0.7,
                max_picks: int = None,
                inhib_strength: float = 0.6,
                inhib_sigma: float = 40.0,
                presence_floor: float = 0.35,
                refine_frac: float = 0.5,
                # gating / sparsity
                lambda_sparsity: float = 0.55,
                z_perp_gate: float = 1.0
                ) -> Tuple[List[str], List[str], Dict[str, float], Dict[str, int], List[float]]:
    """
    Return: (keep, order, energy_map, peak_idx, margins)
    """
    keys = list(traces.keys())
    T = len(next(iter(traces.values())))

    # Smooth raw + residualize via common-mode
    Sraw = {p: moving_average(traces[p], k=sigma) for p in keys}
    Eres = perpendicular_energy(traces)
    Sres = {p: moving_average(Eres[p], k=sigma) for p in keys}
    Scm  = moving_average(common_mode(traces), k=sigma)

    # Low-rank shared removal on residuals
    M = np.stack([Sres[p] for p in keys], axis=1)  # (T x K)
    M_perp = span_complement_project(M, rank=rank) if rank > 0 else (M - M.mean(axis=0, keepdims=True))
    Sres_perp = {p: M_perp[:, i] for i, p in enumerate(keys)}

    proto = half_sine_proto(proto_width)
    peak_idx = estimate_peak_indices(Sres_perp, proto)

    # helper to compute Stage-11 energy for all channels
    w_perp, w_raw, w_cm = weights
    def compute_energy(Sres_local: Dict[str, np.ndarray],
                       Sraw_local: Dict[str, np.ndarray],
                       Scm_local: np.ndarray,
                       peaks: Dict[str, int],
                       lateral_peaks: List[int] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        E = {}
        Zperp = {}
        for p in keys:
            idx = peaks[p]
            # z-scores via block-permutation null + multitaper average
            z_perp, _ = perm_null_block_z(zscore(Sres_local[p]), proto, idx, proto_width, rng,
                                          nperm=nperm, block=block, tapers=tapers or [])
            z_raw,  _ = perm_null_block_z(zscore(Sraw_local[p]),  proto, idx, proto_width, rng,
                                          nperm=nperm, block=block, tapers=tapers or [])
            z_cm,   _ = perm_null_block_z(zscore(Scm_local),      proto, idx, proto_width, rng,
                                          nperm=nperm, block=block, tapers=tapers or [])
            # Energy
            Up = - (w_perp * z_perp + w_raw * z_raw - w_cm * max(0.0, z_cm))
            # Lateral inhibition penalty
            if lateral_peaks:
                for t0 in lateral_peaks:
                    dt = float(idx - t0)
                    pen = inhib_strength * math.exp(-(dt*dt) / (2.0 * (inhib_sigma**2)))
                    Up += pen
            E[p] = float(Up)
            Zperp[p] = float(z_perp)
        return E, Zperp

    keep = []
    chosen_peaks = []  # list of idx of chosen peaks (for lateral inhibition)
    margins = []

    # Working copies for refinement
    M_work = M_perp.copy()       # residual channels only for refinement
    Sraw_work = {p: Sraw[p].copy() for p in keys}
    Scm_work = Scm.copy()
    Sres_work = {p: M_work[:, i] for i, p in enumerate(keys)}
    peaks_work = dict(peak_idx)

    # Selection loop
    step = 0
    J_acc = 0.0   # MDL-style cumulative objective: sum(depth) - λ*k
    while True:
        energies, zmap = compute_energy(Sres_work, Sraw_work, Scm_work, peaks_work, lateral_peaks=chosen_peaks)
        # Mask already selected
        mask = np.array([p not in keep for p in keys], dtype=bool)
        if not mask.any():
            break
        U_arr = np.array([energies[p] for p in keys])
        U_masked = U_arr.copy()
        U_masked[~mask] = np.inf  # exclude selected
        if not np.isfinite(U_masked).any():
            break

        # annealed selection (lower U better)
        Tcur = max(temp_min, temp0 * (anneal ** step))
        Uavail = U_masked[mask]
        Umin = float(np.min(Uavail))
        logits = - (Uavail - Umin) / max(1e-8, Tcur)
        probs = np.exp(logits - logits.max()); probs = probs / (probs.sum() + 1e-12)
        idx_avail = np.where(mask)[0]
        pick_k = int(idx_avail[int(np.argmax(probs))])
        pick = keys[pick_k]

        # Always-on gating + MDL cumulative objective
        depth = -float(energies[pick])        # well depth (higher is better)
        if zmap[pick] < z_perp_gate:          # require sufficient residual evidence
            break
        if depth < lambda_sparsity:           # per-pick minimum depth
            break
        J_next = J_acc + depth - lambda_sparsity
        if J_next <= J_acc + 1e-9:            # no improvement => stop
            break

        keep.append(pick)
        J_acc = J_next

        # margin Δ = closest false minus true
        others = [energies[p] for p in keys if p != pick and p not in keep[:-1]]
        if others:
            margins.append(float(min(others) - energies[pick]))

        # Sequential residual refinement
        t0 = peaks_work[pick]
        M_work = subtract_aligned_proto(M_work, pick_k, t0, proto, proto_width)
        # Rebuild working dicts after subtraction
        Sres_work = {p: M_work[:, i] for i, p in enumerate(keys)}
        # Also subtract from raw/common-mode with configurable strength
        a, b = max(0, t0 - proto_width//2), min(Sraw_work[keys[0]].shape[0], t0 + proto_width//2)
        L = b - a
        if L > 2:
            pr = proto[:L]
            for i, p2 in enumerate(keys):
                seg = Sraw_work[p2][a:b]
                alpha = float(np.dot(seg - seg.mean(), (pr - pr.mean())) / (np.dot(pr - pr.mean(), pr - pr.mean()) + 1e-8))
                Sraw_work[p2][a:b] = seg - refine_frac * alpha * pr
            Scm_work[a:b] = Scm_work[a:b] - refine_frac * pr.mean()

        # Update peaks after refinement
        peaks_work = estimate_peak_indices(Sres_work, proto)
        chosen_peaks.append(t0)

        step += 1
        if max_picks is not None and len(keep) >= max_picks:
            break

        # presence-floor stopping (only when cold)
        Uavail2 = np.array([energies[p] for p in keys if p not in keep])
        if Uavail2.size == 0: break
        if Tcur <= (temp_min + 1e-6):
            u_best = float(np.min(Uavail2)); u_med = float(np.median(Uavail2))
            if (u_med - u_best) < presence_floor: break

    # Order by peak time (from final peaks_work)
    order = sorted(keep, key=lambda p: peaks_work[p])
    final_energies = {p: float(v) for p, v in energies.items()}
    return keep, order, final_energies, peaks_work, margins

# ----------------------------- Simple Stock Baseline -----------------------------

def stock_parse(traces: Dict[str, np.ndarray], sigma: int = 9, proto_width: int = 160):
    """Raw channel matched-filter + threshold (≈ Stage-10 stock)."""
    keys = list(traces.keys())
    S = {p: moving_average(traces[p], k=sigma) for p in keys}
    proto = half_sine_proto(proto_width)
    peak = {p: int(np.argmax(np.correlate(S[p], proto, mode="same"))) for p in keys}
    score = {p: float(np.max(np.correlate(S[p], proto, mode="same"))) for p in keys}
    smax = max(score.values()) + 1e-12
    keep = [p for p in keys if score[p] >= 0.6*smax]
    if not keep: keep = [max(keys, key=lambda p: score[p])]
    order = sorted(keep, key=lambda p: peak[p])
    return keep, order

# ----------------------------- Main Benchmark -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage 11 — Well Parser Benchmark (v1)")
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--sigma", type=int, default=9)
    ap.add_argument("--proto_width", type=int, default=160)
    ap.add_argument("--rank", type=int, default=2, help="top-r PCs to remove (span complement)")
    ap.add_argument("--nperm", type=int, default=150)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--tapers", type=str, default="hann,hamming,blackman")
    ap.add_argument("--temp0", type=float, default=1.2)
    ap.add_argument("--temp_min", type=float, default=0.15)
    ap.add_argument("--anneal", type=float, default=0.7)
    ap.add_argument("--max_picks", type=int, default=None)
    ap.add_argument("--inhib_strength", type=float, default=0.6)
    ap.add_argument("--inhib_sigma", type=float, default=40.0)
    ap.add_argument("--presence_floor", type=float, default=0.35)
    ap.add_argument("--refine_frac", type=float, default=0.5, help="strength of raw/CM subtraction after each pick")
    ap.add_argument("--lambda_sparsity", type=float, default=0.55, help="model-order penalty λ per pick")
    ap.add_argument("--z_perp_gate", type=float, default=1.0, help="minimum residual z⊥ required per pick")
    ap.add_argument("--weights", type=str, default="1.0,0.4,0.3")  # w_perp,w_raw,w_cm
    ap.add_argument("--tasks_range", type=str, default="1,3")
    ap.add_argument("--allow_repeats", action="store_true")
    ap.add_argument("--cm_amp", type=float, default=0.02)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--amp_jitter", type=float, default=0.4)
    ap.add_argument("--distractor_prob", type=float, default=0.4)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--csv", type=str, default="stage11_well_benchmark.csv")
    ap.add_argument("--compat_hidden_target", action="store_true",
                    help="print results in [HIDDEN-TARGET] Geodesic/Stock style")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    min_k, max_k = map(int, args.tasks_range.split(","))
    w_perp, w_raw, w_cm = map(float, args.weights.split(","))
    tapers = [t.strip() for t in args.tapers.split(",") if t.strip()] if args.tapers else []

    rows = []

    # Aggregates for Stage-11 (well)
    agg_w = dict(acc=0, grid=0.0, P=0.0, R=0.0, F1=0.0, J=0.0, H=0.0, O=0.0,
                 margin_mean=0.0, margin_min=0.0)
    margins_all = []

    # Aggregates for Stock baseline (for compat print)
    agg_s = dict(acc=0, grid=0.0, P=0.0, R=0.0, F1=0.0, J=0.0, H=0.0, O=0.0)

    for i in range(1, args.samples+1):
        sample = make_sample(rng, T=args.T, n_tasks=(min_k, max_k), noise=args.noise,
                             allow_repeats=args.allow_repeats, cm_amp=args.cm_amp,
                             overlap=args.overlap, amp_jitter=args.amp_jitter,
                             distractor_prob=args.distractor_prob)

        keep, order, energies, peaks, margins = well_parser(
            sample.traces, rng, sigma=args.sigma, proto_width=args.proto_width,
            rank=args.rank, weights=(w_perp, w_raw, w_cm), nperm=args.nperm,
            block=args.block, tapers=tapers, temp0=args.temp0, temp_min=args.temp_min,
            anneal=args.anneal, max_picks=args.max_picks, inhib_strength=args.inhib_strength,
            inhib_sigma=args.inhib_sigma, presence_floor=args.presence_floor, refine_frac=args.refine_frac,
            lambda_sparsity=args.lambda_sparsity, z_perp_gate=args.z_perp_gate
        )

        # Stock baseline
        keep_s, order_s = stock_parse(sample.traces, sigma=args.sigma, proto_width=args.proto_width)

        # Evaluate (well)
        grid_pred = apply_sequence(sample.grid_in, order)
        ok = int(np.array_equal(grid_pred, sample.grid_out_true))
        gs = grid_similarity(grid_pred, sample.grid_out_true)
        sm = set_metrics(sample.order_true, keep)

        agg_w["acc"] += ok; agg_w["grid"] += gs
        agg_w["P"] += sm["precision"]; agg_w["R"] += sm["recall"]; agg_w["F1"] += sm["f1"]; agg_w["J"] += sm["jaccard"]
        agg_w["H"] += sm["hallucination_rate"]; agg_w["O"] += sm["omission_rate"]

        m_mean = float(np.nanmean(margins)) if margins else float("nan")
        m_min  = float(np.nanmin(margins)) if margins else float("nan")
        margins_all.extend([m for m in margins if np.isfinite(m)])
        agg_w["margin_mean"] += (0.0 if np.isnan(m_mean) else m_mean)
        agg_w["margin_min"]  += (0.0 if np.isnan(m_min)  else m_min)

        # Evaluate (stock)
        grid_stock = apply_sequence(sample.grid_in, order_s)
        ok_s = int(np.array_equal(grid_stock, sample.grid_out_true))
        gs_s = grid_similarity(grid_stock, sample.grid_out_true)
        sm_s = set_metrics(sample.order_true, keep_s)
        agg_s["acc"] += ok_s; agg_s["grid"] += gs_s
        agg_s["P"] += sm_s["precision"]; agg_s["R"] += sm_s["recall"]; agg_s["F1"] += sm_s["f1"]; agg_s["J"] += sm_s["jaccard"]
        agg_s["H"] += sm_s["hallucination_rate"]; agg_s["O"] += sm_s["omission_rate"]

        rows.append(dict(
            sample=i, true="|".join(sample.order_true),
            keep="|".join(keep), order="|".join(order), ok=ok, grid=gs,
            precision=sm["precision"], recall=sm["recall"], f1=sm["f1"], jaccard=sm["jaccard"],
            hallucination=sm["hallucination_rate"], omission=sm["omission_rate"],
            energies=";".join([f"{k}:{energies[k]:.3f}" for k in PRIMS]),
            peaks=";".join([f"{k}:{peaks[k]}" for k in PRIMS]),
            margin_mean=m_mean, margin_min=m_min,
            stock_tasks="|".join(keep_s), stock_order="|".join(order_s),
            stock_ok=ok_s, stock_grid=gs_s, stock_precision=sm_s["precision"],
            stock_recall=sm_s["recall"], stock_f1=sm_s["f1"], stock_jaccard=sm_s["jaccard"],
            stock_hallucination=sm_s["hallucination_rate"], stock_omission=sm_s["omission_rate"]
        ))

    n = float(args.samples)
    summary = dict(
        acc_exact_geo = agg_w["acc"]/n,
        grid_geo      = agg_w["grid"]/n,
        precision     = agg_w["P"]/n,
        recall        = agg_w["R"]/n,
        f1            = agg_w["F1"]/n,
        jaccard       = agg_w["J"]/n,
        hallucination = agg_w["H"]/n,
        omission      = agg_w["O"]/n,
        margin_mean   = agg_w["margin_mean"]/n,
        margin_min    = agg_w["margin_min"]/n,
        n_samples     = int(n)
    )
    print("[Stage11 v1] Summary:", {k: (round(v,3) if isinstance(v,float) else v) for k,v in summary.items()})
    if margins_all:
        print(f"[Stage11 v1] Margins — mean={np.mean(margins_all):.3f}, min={np.min(margins_all):.3f}, n={len(margins_all)}")

    # Compat printout in Stage-10 style
    if args.compat_hidden_target:
        G = dict(
            accuracy_exact = summary["acc_exact_geo"],
            grid_similarity= summary["grid_geo"],
            precision      = summary["precision"],
            recall         = summary["recall"],
            f1             = summary["f1"],
            jaccard        = summary["jaccard"],
            hallucination_rate = summary["hallucination"],
            omission_rate      = summary["omission"]
        )
        S = dict(
            accuracy_exact = (agg_s["acc"]/n),
            grid_similarity= (agg_s["grid"]/n),
            precision      = (agg_s["P"]/n),
            recall         = (agg_s["R"]/n),
            f1             = (agg_s["F1"]/n),
            jaccard        = (agg_s["J"]/n),
            hallucination_rate = (agg_s["H"]/n),
            omission_rate      = (agg_s["O"]/n)
        )
        print("[HIDDEN-TARGET] Geodesic:", {k: round(v,3) for k,v in G.items()})
        print("[HIDDEN-TARGET] Stock   :", {k: round(v,3) for k,v in S.items()})

    # CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"[WROTE] {args.csv}")

if __name__ == "__main__":
    main()
