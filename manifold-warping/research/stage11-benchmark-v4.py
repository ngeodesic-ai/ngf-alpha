
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, csv, math
from typing import List, Dict, Tuple
import numpy as np

PRIMS = ["flip_h","flip_v","rotate"]

# ----------------------------- Utilities -----------------------------

def moving_average(x: np.ndarray, k: int = 9) -> np.ndarray:
    if k <= 1: 
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(xp, np.ones(k)/k, mode="valid")

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
    T = len(x); 
    if block <= 1:
        s = rng.integers(1, T-1)
    else:
        nblocks = max(1, T // block)
        s = int(rng.integers(1, nblocks)) * block
    return circ_shift(x, s)

def half_sine(width: int) -> np.ndarray:
    return np.sin(np.linspace(0, np.pi, width))

def triangle(width: int) -> np.ndarray:
    w = np.linspace(0, 1, width)
    tri = np.where(w <= 0.5, w/0.5, (1-w)/0.5)
    return tri

def skew_half_sine(width: int, skew: float = 0.5) -> np.ndarray:
    x = np.linspace(0, 1, width)
    beta = 3*(1-skew) + 1e-6
    return np.sin(np.pi * x**beta)

def normalize(p: np.ndarray) -> np.ndarray:
    return p / (np.linalg.norm(p) + 1e-8)

def shift_proto(p: np.ndarray, frac: float) -> np.ndarray:
    k = int(round(len(p)*frac))
    return circ_shift(p, k)

def generate_prototypes(width: int, modes: List[str], phases: List[float]) -> List[np.ndarray]:
    base = []
    for m in modes:
        if m == "half": base.append(half_sine(width))
        elif m == "triangle": base.append(triangle(width))
        elif m == "skewL": base.append(skew_half_sine(width, skew=0.35))
        elif m == "skewR": base.append(skew_half_sine(width, skew=0.65))
        else: base.append(half_sine(width))
    protos = []
    for b in base:
        b = normalize(b)
        for ph in phases:
            protos.append(normalize(shift_proto(b, ph)))
    return protos

def normalized_corr_window(sig: np.ndarray, proto: np.ndarray, center_idx: int, width: int) -> float:
    T = len(sig)
    a, b = max(0, center_idx - width//2), min(T, center_idx + width//2)
    w = sig[a:b].astype(float)
    if len(w) < 3: 
        return 0.0
    pr = proto[:len(w)].astype(float)
    w = w - w.mean()
    pr = pr - pr.mean()
    denom = (np.linalg.norm(w) * np.linalg.norm(pr) + 1e-8)
    return float(np.dot(w, pr) / denom)

def perm_null_block_z(sig: np.ndarray, proto: np.ndarray, peak_idx: int, width: int, rng,
                      nperm: int = 200, block: int = 32) -> Tuple[float, float]:
    obs = normalized_corr_window(sig, proto, peak_idx, width)
    null = np.empty(nperm, dtype=float)
    for i in range(nperm):
        x = block_circ_shift(sig, rng, block)
        null[i] = normalized_corr_window(x, proto, peak_idx, width)
    mu, sd = float(null.mean()), float(null.std() + 1e-8)
    z = (obs - mu) / sd
    return float(z), float(obs)

def span_complement_project(S: np.ndarray, rank: int) -> np.ndarray:
    T, K = S.shape
    X = S - S.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    r = int(max(0, min(rank, len(s))))
    if r == 0: return X
    Ur = U[:, :r]
    X_res = X - Ur @ (Ur.T @ X)
    return X_res

def common_mode(traces: Dict[str, np.ndarray]) -> np.ndarray:
    arr = np.stack([traces[p] for p in traces.keys()], 0)
    return arr.mean(0)

def estimate_peak_indices(S_channel: Dict[str, np.ndarray], proto: np.ndarray) -> Dict[str, int]:
    peak = {}
    for p, sig in S_channel.items():
        m = np.correlate(zscore(sig), proto, mode="same")
        peak[p] = int(np.argmax(m))
    return peak

def subtract_aligned_proto(M: np.ndarray, pick_k: int, peak_idx: int, proto: np.ndarray, width: int) -> np.ndarray:
    T, K = M.shape
    a, b = max(0, peak_idx - width//2), min(T, peak_idx + width//2)
    L = b - a
    if L <= 2: return M
    pr = proto[:L]
    pr_c = pr - pr.mean()
    pr_den = (np.dot(pr_c, pr_c) + 1e-8)
    S2 = M.copy()
    for k in range(K):
        seg = M[a:b, k]
        seg_c = seg - seg.mean()
        alpha = float(np.dot(seg_c, pr_c) / pr_den)
        S2[a:b, k] = seg - alpha * pr
    return S2

# ----------------------------- Synthetic world -----------------------------

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

def gaussian_bump(T: int, center: int, width: int, amp: float = 1.0) -> np.ndarray:
    t = np.arange(T)
    sig2 = (width/2.355)**2
    return amp * np.exp(-(t-center)**2 / (2*sig2))

def add_noise(x: np.ndarray, sigma: float, rng) -> np.ndarray:
    return x + rng.normal(0, sigma, size=x.shape)

def gen_traces(tasks: List[str], T: int, rng, noise=0.02, cm_amp=0.02, overlap=0.5,
               amp_jitter=0.4, distractor_prob=0.4) -> Dict[str, np.ndarray]:
    base = np.array([0.20, 0.50, 0.80]) * T
    centers = ((1.0 - overlap) * base + overlap * (T * 0.50)).astype(int)
    width = int(max(12, T * 0.10))
    t = np.arange(T)
    cm = cm_amp * (1.0 + 0.2 * np.sin(2*np.pi * t / max(30, T//6)))
    traces = {p: np.zeros(T, float) for p in PRIMS}
    for i, prim in enumerate(tasks):
        c = centers[i % len(centers)]
        amp = max(0.25, 1.0 + rng.normal(0, amp_jitter))
        c_jit = int(np.clip(c + rng.integers(-width//5, width//5 + 1), 0, T-1))
        traces[prim] += gaussian_bump(T, c_jit, width, amp=amp)
    for p in PRIMS:
        if p not in tasks and rng.random() < distractor_prob:
            c = int(rng.uniform(T*0.15, T*0.85))
            amp = max(0.25, 1.0 + rng.normal(0, amp_jitter))
            traces[p] += gaussian_bump(T, c, width, amp=0.9*amp)
    for p in PRIMS:
        traces[p] = np.clip(traces[p] + cm, 0, None)
        traces[p] = add_noise(traces[p], noise, rng)
    return traces

def make_sample(rng, T=720, n_tasks=(1,3), grid_shape=(8,8), noise=0.02,
                allow_repeats=False, cm_amp=0.02, overlap=0.5, amp_jitter=0.4,
                distractor_prob=0.4):
    k = rng.integers(n_tasks[0], n_tasks[1]+1)
    tasks = list(rng.choice(PRIMS, size=k, replace=allow_repeats))
    if not allow_repeats:
        rng.shuffle(tasks)
    g0 = random_grid(rng, *grid_shape)
    g1 = apply_sequence(g0, tasks)
    traces = gen_traces(tasks, T=T, rng=rng, noise=noise, cm_amp=cm_amp,
                        overlap=overlap, amp_jitter=amp_jitter, distractor_prob=distractor_prob)
    return g0, tasks, g1, traces, T

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

# ----------------------------- Step-4 Well Parser -----------------------------

def parse_map_triplets(s: str, default=(1.0,0.4,0.3)) -> Dict[str, Tuple[float,float,float]]:
    m = {p: default for p in PRIMS}
    if not s: return m
    for part in s.split(";"):
        if not part.strip(): continue
        name, trip = part.split(":")
        vals = tuple(float(x) for x in trip.split(","))
        m[name.strip()] = vals
    return m

def parse_map_scalars(s: str, default=None) -> Dict[str, float]:
    m = {p: default for p in PRIMS}
    if not s: return m
    for part in s.split(";"):
        if not part.strip(): continue
        name, val = part.split(":")
        m[name.strip()] = float(val)
    return m

def softmin(x: np.ndarray, tau: float = 0.5) -> float:
    # returns soft minimum (smaller = better) to avoid cherry-picking single proto
    # tau > 0; as tau->0, becomes hard min; large tau -> averages
    x = np.asarray(x, dtype=float)
    m = np.min(x)
    logits = - (x - m) / max(1e-8, tau)
    w = np.exp(logits - logits.max())
    w = w / (w.sum() + 1e-12)
    return float(np.sum(w * x))

def well_parser_step4(traces: Dict[str, np.ndarray],
                rng,
                sigma: int = 9,
                proto_width: int = 160,
                rank: int = 2,
                weights_map: Dict[str, Tuple[float,float,float]] = None,
                nperm: int = 200,
                block: int = 32,
                temp0: float = 1.2,
                temp_min: float = 0.15,
                anneal: float = 0.7,
                max_picks: int = None,
                inhib_strength: float = 0.6,
                inhib_sigma: float = 40.0,
                presence_floor: float = 0.35,
                proto_modes: List[str] = None,
                proto_phases: List[float] = None,
                proto_agg: str = "softmin",
                proto_tau: float = 0.4,
                raw_floor_map: Dict[str, float] = None,
                residual_ceiling_map: Dict[str, float] = None,
                consensus_k: int = 0,
                consensus_eps: float = 0.0,
                pick_penalty: float = 0.0
                ):
    keys = list(traces.keys())
    K = len(keys)
    T = len(next(iter(traces.values())))

    Sraw = {p: moving_average(traces[p], k=sigma) for p in keys}
    mu = common_mode(traces)
    Eres = {p: np.clip(traces[p] - mu, 0, None) for p in keys}
    Sres = {p: moving_average(Eres[p], k=sigma) for p in keys}
    M = np.stack([Sres[p] for p in keys], axis=1)
    M_perp = span_complement_project(M, rank=rank) if rank > 0 else (M - M.mean(axis=0, keepdims=True))
    Sres_perp = {p: M_perp[:, i] for i, p in enumerate(keys)}
    Scm  = moving_average(mu, k=sigma)

    # prototypes
    modes = proto_modes or ["half","skewL","skewR","triangle"]
    phases = proto_phases or [0.0, 0.2]
    PROTOS = generate_prototypes(proto_width, modes, phases)

    # peak idx (rough) using first proto
    proto0 = PROTOS[0]
    peak_idx = estimate_peak_indices(Sres_perp, proto0)

    if raw_floor_map is None:
        raw_floor_map = {p: None for p in keys}
    if residual_ceiling_map is None:
        residual_ceiling_map = {p: None for p in keys}

    chosen_peaks = []
    keep = []
    margins = []

    M_work = M_perp.copy()
    Sraw_work = {p: Sraw[p].copy() for p in keys}
    Scm_work = Scm.copy()
    Sres_work = {p: M_work[:, i] for i, p in enumerate(keys)}
    peaks_work = dict(peak_idx)

    def aggregate_proto(energies_list: List[float]) -> float:
        if proto_agg == "min":
            return float(np.min(energies_list))
        elif proto_agg == "median":
            return float(np.median(energies_list))
        else:  # softmin
            return softmin(np.array(energies_list), tau=proto_tau)

    def compute_energy(Sres_local, Sraw_local, Scm_local, peaks, lateral_peaks=None, picks_count=0):
        E = {}
        for p in keys:
            idx = peaks[p]
            parts = []  # per-proto energies
            support = 0
            for proto in PROTOS:
                z_perp, _ = perm_null_block_z(zscore(Sres_local[p]), proto, idx, proto_width, rng, nperm=nperm, block=block)
                z_raw,  _ = perm_null_block_z(zscore(Sraw_local[p]),  proto, idx, proto_width, rng, nperm=nperm, block=block)
                z_cm,   _ = perm_null_block_z(zscore(Scm_local),      proto, idx, proto_width, rng, nperm=nperm, block=block)
                w_perp, w_raw, w_cm = weights_map[p]
                Up = - (w_perp * z_perp + w_raw * z_raw - w_cm * max(0.0, z_cm))
                # Per-primitive constraints
                rf = raw_floor_map.get(p, None)
                if rf is not None and z_raw < rf:
                    Up += (rf - z_raw) * 1.0  # 1.0 penalty per shortfall unit
                rc = residual_ceiling_map.get(p, None)
                if rc is not None and z_perp > rc:
                    Up += (z_perp - rc) * 1.0  # penalize residual-only dominance
                parts.append(Up)
                if (rf is None or z_raw >= rf) and (rc is None or z_perp <= rc):
                    # proto is supportive if above raw floor and below residual ceiling
                    if Up <= (np.min(parts) + consensus_eps):
                        support += 1
            # aggregate
            Up_agg = aggregate_proto(parts)
            # consensus requirement
            if consensus_k > 0 and support < consensus_k:
                Up_agg += (consensus_k - support) * 0.5
            # pick penalty
            if pick_penalty > 0.0:
                Up_agg += pick_penalty * picks_count
            # lateral inhibition
            if lateral_peaks:
                for t0 in lateral_peaks:
                    dt = float(idx - t0)
                    pen = inhib_strength * math.exp(-(dt*dt) / (2.0 * (inhib_sigma**2)))
                    Up_agg += pen
            E[p] = float(Up_agg)
        return E

    step = 0
    while True:
        energies = compute_energy(Sres_work, Sraw_work, Scm_work, peaks_work, lateral_peaks=chosen_peaks, picks_count=len(keep))
        U_arr = np.array([energies[p] for p in keys])
        mask = np.array([p not in keep for p in keys], dtype=bool)
        if not mask.any(): break
        U_masked = U_arr.copy(); U_masked[~mask] = np.inf
        if not np.isfinite(U_masked).any(): break

        Tcur = max(temp_min, temp0 * (anneal ** step))
        Uavail = U_masked[mask]
        Umin = float(np.min(Uavail))
        logits = - (Uavail - Umin) / max(1e-8, Tcur)
        probs = np.exp(logits - logits.max()); probs = probs / (probs.sum() + 1e-12)
        idx_avail = np.where(mask)[0]
        pick_pos = int(np.argmax(probs))
        pick_k = int(idx_avail[pick_pos])
        pick = PRIMS[pick_k]
        keep.append(pick)

        # margin
        false_U = [energies[p] for p in PRIMS if p not in keep[:-1]]
        if len(false_U) > 1:
            true_U = energies[pick]
            others = [energies[p] for p in PRIMS if p != pick and p not in keep[:-1]]
            margin = float(min(others) - true_U) if others else float('nan')
            margins.append(margin)

        # subtract aligned proto using the first proto
        t0 = peaks_work[pick]
        proto0 = generate_prototypes(proto_width, ["half"], [0.0])[0]
        M_work = subtract_aligned_proto(M_work, pick_k, t0, proto0, proto_width)
        Sres_work = {p: M_work[:, i] for i, p in enumerate(PRIMS)}

        # light subtraction from raw/common-mode
        frac = 0.25
        a, b = max(0, t0 - proto_width//2), min(T, t0 + proto_width//2)
        L = b - a
        if L > 2:
            pr = proto0[:L]
            for i, p2 in enumerate(PRIMS):
                seg = Sraw_work[p2][a:b]
                alpha = float(np.dot(seg - seg.mean(), (pr - pr.mean())) / (np.dot(pr - pr.mean(), pr - pr.mean()) + 1e-8))
                Sraw_work[p2][a:b] = seg - frac * alpha * pr
            Scm_work[a:b] = Scm_work[a:b] - frac * pr.mean()

        peaks_work = estimate_peak_indices(Sres_work, proto0)
        chosen_peaks.append(t0)
        step += 1
        if max_picks is not None and len(keep) >= max_picks: break

        Uavail2 = np.array([energies[p] for p in PRIMS if p not in keep])
        if Uavail2.size == 0: break
        if Tcur <= (temp_min + 1e-6):
            u_best = float(np.min(Uavail2))
            u_med  = float(np.median(Uavail2))
            if (u_med - u_best) < presence_floor: break

    order = sorted(keep, key=lambda p: peaks_work[p])
    final_energies = {p: float(v) for p, v in energies.items()}
    return keep, order, final_energies, peaks_work, margins

# ----------------------------- Main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Stage 11 — Step 4: Per-Primitive Thresholds & Robust Prototype Aggregation")
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--T", type=int, default=720)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--sigma", type=int, default=9)
    ap.add_argument("--proto_width", type=int, default=160)
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--nperm", type=int, default=150)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--temp0", type=float, default=1.2)
    ap.add_argument("--temp_min", type=float, default=0.15)
    ap.add_argument("--anneal", type=float, default=0.7)
    ap.add_argument("--max_picks", type=int, default=None)
    ap.add_argument("--inhib_strength", type=float, default=0.6)
    ap.add_argument("--inhib_sigma", type=float, default=40.0)
    ap.add_argument("--presence_floor", type=float, default=0.35)
    ap.add_argument("--weights_map", type=str, default="flip_h:1.0,0.4,0.3;flip_v:0.9,0.4,0.4;rotate:1.0,0.4,0.3")
    ap.add_argument("--tasks_range", type=str, default="1,3")
    ap.add_argument("--allow_repeats", action="store_true")
    ap.add_argument("--cm_amp", type=float, default=0.02)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--amp_jitter", type=float, default=0.4)
    ap.add_argument("--distractor_prob", type=float, default=0.4)
    ap.add_argument("--proto_modes", type=str, default="half,skewL,skewR,triangle")
    ap.add_argument("--proto_phases", type=str, default="0.0,0.2")
    ap.add_argument("--proto_agg", type=str, default="softmin", choices=["softmin","median","min"])
    ap.add_argument("--proto_tau", type=float, default=0.4, help="temperature for softmin aggregation")
    ap.add_argument("--raw_floor_map", type=str, default="flip_v:0.55")
    ap.add_argument("--residual_ceiling_map", type=str, default="")
    ap.add_argument("--consensus_k", type=int, default=1, help="require at least k supportive prototypes")
    ap.add_argument("--consensus_eps", type=float, default=0.15, help="support if within eps of proto aggregate")
    ap.add_argument("--pick_penalty", type=float, default=0.15)
    ap.add_argument("--csv", type=str, default="stage11_step4.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    min_k, max_k = map(int, args.tasks_range.split(","))
    modes = [m.strip() for m in args.proto_modes.split(",") if m.strip()]
    phases = [float(x) for x in args.proto_phases.split(",") if x.strip()]
    weights_map = parse_map_triplets(args.weights_map, default=(1.0,0.4,0.3))
    raw_floor_map = parse_map_scalars(args.raw_floor_map, default=None)
    residual_ceiling_map = parse_map_scalars(args.residual_ceiling_map, default=None)

    rows = []
    agg = dict(acc_exact=0, grid=0.0, P=0.0, R=0.0, F1=0.0, J=0.0, H=0.0, O=0.0, margin_mean=0.0, margin_min=0.0)
    margins_all = []

    for i in range(1, args.samples+1):
        g0, tasks_true, g1, traces, TT = make_sample(rng, T=args.T, n_tasks=(min_k, max_k), noise=args.noise,
                             allow_repeats=args.allow_repeats, cm_amp=args.cm_amp,
                             overlap=args.overlap, amp_jitter=args.amp_jitter,
                             distractor_prob=args.distractor_prob)

        keep, order, energies, peaks, margins = well_parser_step4(
            traces, rng, sigma=args.sigma, proto_width=args.proto_width,
            rank=args.rank, weights_map=weights_map, nperm=args.nperm,
            block=args.block, temp0=args.temp0, temp_min=args.temp_min,
            anneal=args.anneal, max_picks=args.max_picks, inhib_strength=args.inhib_strength,
            inhib_sigma=args.inhib_sigma, presence_floor=args.presence_floor,
            proto_modes=modes, proto_phases=phases, proto_agg=args.proto_agg, proto_tau=args.proto_tau,
            raw_floor_map=raw_floor_map, residual_ceiling_map=residual_ceiling_map,
            consensus_k=args.consensus_k, consensus_eps=args.consensus_eps, pick_penalty=args.pick_penalty
        )

        gp = apply_sequence(g0, order)
        ok = int(np.array_equal(gp, g1))
        gs = grid_similarity(gp, g1)
        sm = set_metrics(tasks_true, keep)

        agg["acc_exact"] += ok
        agg["grid"] += gs
        agg["P"] += sm["precision"]; agg["R"] += sm["recall"]; agg["F1"] += sm["f1"]; agg["J"] += sm["jaccard"]
        agg["H"] += sm["hallucination_rate"]; agg["O"] += sm["omission_rate"]

        m_mean = float(np.nanmean(margins)) if margins else float("nan")
        m_min  = float(np.nanmin(margins)) if margins else float("nan")
        if margins:
            margins_all.extend([m for m in margins if np.isfinite(m)])
        agg["margin_mean"] += (0.0 if np.isnan(m_mean) else m_mean)
        agg["margin_min"]  += (0.0 if np.isnan(m_min)  else m_min)

        rows.append(dict(
            sample=i, true="|".join(tasks_true),
            keep="|".join(keep), order="|".join(order), ok=ok, grid=gs,
            precision=sm["precision"], recall=sm["recall"], f1=sm["f1"], jaccard=sm["jaccard"],
            hallucination=sm["hallucination_rate"], omission=sm["omission_rate"],
            energies=";".join([f"{k}:{energies[k]:.3f}" for k in PRIMS]),
            peaks=";".join([f"{k}:{peaks[k]}" for k in PRIMS]),
            margin_mean=m_mean, margin_min=m_min
        ))

    n = float(args.samples)
    summary = dict(
        acc_exact = agg["acc_exact"]/n,
        grid      = agg["grid"]/n,
        precision = agg["P"]/n,
        recall    = agg["R"]/n,
        f1        = agg["F1"]/n,
        jaccard   = agg["J"]/n,
        hallucination = agg["H"]/n,
        omission      = agg["O"]/n,
        margin_mean   = agg["margin_mean"]/n,
        margin_min    = agg["margin_min"]/n,
        n_samples     = int(n)
    )
    print("[Step4] Summary:", {k: (round(v,3) if isinstance(v,float) else v) for k,v in summary.items()})
    if margins_all:
        print(f"[Step4] Margins — mean={np.mean(margins_all):.3f}, min={np.min(margins_all):.3f}, n={len(margins_all)}")

    if rows:
        fieldnames = list(rows[0].keys())
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"[WROTE] {args.csv}")

if __name__ == "__main__":
    main()
