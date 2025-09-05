
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage11-llm-hook.py
-------------------
Minimal Stage-11 doctrine over LLM hidden states:
Warp (PCA whiten) → Detect (matched filter with null calibration + exclusive residual)
→ Denoise (EMA + median), with optional lateral inhibition.

This builds directly on "stage10-llm-hook.py" but adds the critical robustness pieces.
"""

import argparse, random, json, os, math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np

# Soft deps
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:
    raise SystemExit("Please install transformers & torch: pip install transformers torch --upgrade\n" + str(e))

try:
    from sklearn.decomposition import PCA
except Exception as e:
    raise SystemExit("Please install scikit-learn: pip install scikit-learn --upgrade\n" + str(e))

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

PRIMS = ["reverse", "uppercase", "sort"]

# -----------------------------
# String primitives
# -----------------------------

def apply_primitive(s: str, prim: str) -> str:
    if prim == "reverse": return s[::-1]
    if prim == "uppercase": return s.upper()
    if prim == "sort": return "".join(sorted(list(s)))
    raise ValueError(f"Unknown primitive: {prim}")

def apply_sequence(s: str, seq: List[str]) -> str:
    out = s
    for p in seq: out = apply_primitive(out, p)
    return out

# -----------------------------
# HuggingFace helpers
# -----------------------------

def load_model(model_name: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model.eval().to(device)
    return tok, model

@torch.no_grad()
def get_hidden_states(tok, model, text: str, layer: int, device: str) -> np.ndarray:
    inp = tok(text, return_tensors="pt")
    inp = {k: v.to(device) for k, v in inp.items()}
    out = model(**inp, output_hidden_states=True)
    hs = out.hidden_states[layer]  # [1, T, H]
    return hs.squeeze(0).detach().cpu().numpy()

# -----------------------------
# Stage-11 utilities
# -----------------------------

@dataclass
class Params:
    pca_dim: int = 19
    sigma_ma: int = 9
    ema_gamma: float = 0.25
    med_k: int = 7
    proto_width: int = 140
    area_rel: float = 0.45
    corr_rel: float = 0.50
    null_shifts: int = 64
    z_thresh: float = 3.0
    lateral_width: int = 80
    lateral_gain: float = 0.35

@dataclass
class PrimitiveStats:
    anchor: np.ndarray
    proto: np.ndarray

def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1: return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(xp, np.ones(k)/k, mode="valid")

def ema_filter(x: np.ndarray, gamma: float) -> np.ndarray:
    y = np.zeros_like(x)
    acc = x[0]
    for t in range(len(x)):
        acc = gamma*acc + (1-gamma)*x[t]
        y[t] = acc
    return y

def med_filter(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1: return x.copy()
    pad = k//2
    xp = np.pad(x, (pad, pad), mode="edge")
    y = np.empty_like(x)
    for i in range(len(x)):
        y[i] = np.median(xp[i:i+k])
    return y

def half_sine(width: int) -> np.ndarray:
    t = np.linspace(0, np.pi, width)
    v = np.sin(t)
    return v / (np.linalg.norm(v) + 1e-8)

# Exclusive residual: project each channel's z-scored trace against span of others; keep positive part
def exclusive_residual(traces: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    keys = list(traces.keys())
    Z = {}
    # z-score each
    for k in keys:
        x = traces[k]
        mu, sd = x.mean(), x.std() + 1e-8
        Z[k] = (x - mu) / sd
    # build span matrices
    E = {}
    for i, ki in enumerate(keys):
        zi = Z[ki]
        others = [Z[kj] for j, kj in enumerate(keys) if j != i]
        if not others:
            E[ki] = np.maximum(zi, 0)
            continue
        A = np.stack(others, axis=1)  # [T, K-1]
        # least-squares projection of zi onto span(others)
        # solve min ||A w - zi|| ; projection = A w_hat
        # Use normal equations with ridge for stability
        lam = 1e-3
        ATA = A.T @ A + lam*np.eye(A.shape[1])
        ATz = A.T @ zi
        w = np.linalg.solve(ATA, ATz)
        proj = A @ w
        resid = zi - proj
        E[ki] = np.maximum(resid, 0.0)  # positive part
    return E

def matched_filter(x: np.ndarray, proto: np.ndarray) -> Tuple[np.ndarray, int, float]:
    m = np.correlate(x, proto, mode="same")
    idx = int(np.argmax(m))
    # cosine similarity proxy in a local window around idx
    L = len(proto)
    a = max(0, idx - L//2); b = min(len(x), idx + L//2)
    w = x[a:b]; w = (w - w.mean())
    pr = proto[:len(w)] - proto[:len(w)].mean()
    denom = (np.linalg.norm(w) * np.linalg.norm(pr) + 1e-8)
    corr = float(np.dot(w, pr) / denom)
    return m, idx, corr

def circular_shift(x: np.ndarray, s: int) -> np.ndarray:
    s = s % len(x)
    if s == 0: return x.copy()
    return np.concatenate([x[-s:], x[:-s]])

def null_calibration(x: np.ndarray, proto: np.ndarray, n_shifts: int) -> Tuple[float, float]:
    # return mean and std of null correlation maxima
    T = len(x)
    peaks = []
    for i in range(n_shifts):
        s = (i * (T // n_shifts + 1)) % T
        xs = circular_shift(x, s)
        _, _, c = matched_filter(xs, proto)
        peaks.append(c)
    peaks = np.array(peaks, dtype=float)
    return float(peaks.mean()), float(peaks.std() + 1e-8)

def lateral_inhibition(E: Dict[str, np.ndarray], winner: str, peak_idx: Dict[str, int], width: int, gain: float):
    T = len(next(iter(E.values())))
    # build a gaussian penalty centered at winner's peak
    L = width
    t0 = peak_idx[winner]
    t = np.arange(T)
    sig = L/2.355
    pen = gain * np.exp(-0.5*((t - t0)/sig)**2)
    for k in E.keys():
        if k == winner: continue
        E[k] = np.clip(E[k] - pen, 0.0, None)
    return E

# -----------------------------
# Demos and dataset
# -----------------------------

def demos_for_prims() -> Dict[str, List[str]]:
    return {
        "reverse": [
            "Task: reverse\nInput: abcdef\nOutput: fedcba",
            "Task: reverse\nInput: tulip\nOutput: pilut",
            "Task: reverse\nInput: panda\nOutput: adnap",
        ],
        "uppercase": [
            "Task: uppercase\nInput: helloWorld\nOutput: HELLOWORLD",
            "Task: uppercase\nInput: ngfAlpha\nOutput: NGFALPHA",
            "Task: uppercase\nInput: mixEd\nOutput: MIXED",
        ],
        "sort": [
            "Task: sort\nInput: zebra\nOutput: aberz",
            "Task: sort\nInput: casino\nOutput: acinos",
            "Task: sort\nInput: letter\nOutput: eelrtt",
        ],
    }

def make_dataset(rng, n: int) -> List[Dict[str, Any]]:
    alpha = list("abcdefghijklmnopqrstuvwxyz")
    ds = []
    for i in range(n):
        L = rng.integers(5, 9)
        s = "".join(rng.choice(alpha, size=L, replace=True))
        k = int(rng.integers(1, 4))
        order = list(rng.choice(PRIMS, size=k, replace=False))
        rng.shuffle(order)
        target = apply_sequence(s, order)
        ds.append(dict(input=s, order_true=order, output_true=target))
    return ds

def prompt_for(inp: str) -> str:
    return f"Task: (unknown)\nInput: {inp}\nOutput:"

# -----------------------------
# Stats from demos
# -----------------------------

@dataclass
class Stats:
    anchor: Dict[str, np.ndarray]
    proto: Dict[str, np.ndarray]

def build_pca(tok, model, texts: List[str], layer: int, device: str, dim: int):
    H = []
    for t in texts:
        hs = get_hidden_states(tok, model, t, layer, device)
        H.append(hs)
    H = np.vstack(H)
    pca = PCA(n_components=dim, whiten=True, random_state=0).fit(H)
    return pca

def learn_stats(tok, model, demos: Dict[str, List[str]], pca, layer: int, device: str) -> Stats:
    anchor, proto = {}, {}
    for prim, texts in demos.items():
        Y = []
        for t in texts:
            h = get_hidden_states(tok, model, t, layer, device)
            y = pca.transform(h)
            T = y.shape[0]
            seg = y[max(0, T*2//3):, :] if T >= 6 else y
            Y.append(seg)
        Y = np.vstack(Y)
        ck = Y.mean(axis=0)
        U = Y - ck
        _, _, vh = np.linalg.svd(U, full_matrices=False)
        pk = vh[0] / (np.linalg.norm(vh[0]) + 1e-8)
        anchor[prim], proto[prim] = ck, pk
    return Stats(anchor=anchor, proto=proto)

# -----------------------------
# Main detection routine
# -----------------------------

def run(args):
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and (not args.cpu) else "cpu"
    tok, model = load_model(args.model, device)
    P = Params(pca_dim=args.pca_dim, sigma_ma=args.sigma, ema_gamma=args.ema_gamma, med_k=args.med_k,
               proto_width=args.proto_width, null_shifts=args.null_shifts, z_thresh=args.z_thresh,
               lateral_width=args.lateral_width, lateral_gain=args.lateral_gain)

    # Build PCA on demos + a few neutral prompts
    demos = demos_for_prims()
    corpus = [t for texts in demos.values() for t in texts] + [
        "Explain the rules of chess in one sentence.",
        "Summarize: Warping latent space into a single well can stabilize reasoning.",
        "Translate to French: The cat sleeps on the mat.",
        "List three colors: red, green, blue."
    ]
    pca = build_pca(tok, model, corpus, args.layer, device, P.pca_dim)
    stats = learn_stats(tok, model, demos, pca, args.layer, device)

    # Prepare dataset
    dataset = make_dataset(rng, n=args.samples)

    results = []
    correct = 0

    proto = half_sine(P.proto_width)

    for i, rec in enumerate(dataset, 1):
        # Token-time latent trajectory in whitened space
        Y = pca.transform(get_hidden_states(tok, model, prompt_for(rec["input"]), args.layer, device))  # [T, d]
        T = Y.shape[0]

        # Raw per-channel energies (Stage-10 style)
        E_par, E_perp = {}, {}
        for prim in PRIMS:
            ck, pk = stats.anchor[prim], stats.proto[prim]
            diff = Y - ck[None, :]
            s_par = diff @ pk
            E_par[prim] = s_par**2
            E_tot = np.sum(diff**2, axis=1)
            E_perp[prim] = np.clip(E_tot - E_par[prim], 0.0, None)

        # DENOISE: EMA + Median + short MA (Stage-11 control smoothing)
        for prim in PRIMS:
            e = E_perp[prim]
            e = ema_filter(e, P.ema_gamma)
            e = med_filter(e, P.med_k)
            e = moving_average(e, P.sigma_ma)
            E_perp[prim] = e

        # EXCLUSIVE residual to remove cross-talk
        E_ex = exclusive_residual(E_perp)

        # Matched filter + NULL calibration per channel
        corr, area, peak_idx = {}, {}, {}
        keep = []
        for prim in PRIMS:
            m, idx, c = matched_filter(E_ex[prim], proto)
            peak_idx[prim] = idx
            corr[prim] = c
            area[prim] = float(np.trapz(E_ex[prim]))
        # Relative gates
        Amax = max(area.values()) + 1e-12
        Cmax = max(corr.values()) + 1e-12

        # Absolute gate via permutation/circular-shift nulls
        abs_ok = {}
        for prim in PRIMS:
            mu, sd = null_calibration(E_ex[prim], proto, P.null_shifts)
            z = (corr[prim] - mu) / (sd + 1e-8)
            abs_ok[prim] = (z >= P.z_thresh)
            corr[prim] = float(corr[prim])
        # Combine gates
        for prim in PRIMS:
            rel_ok = (area[prim]/Amax >= P.area_rel) and (corr[prim]/Cmax >= P.corr_rel)
            if rel_ok and abs_ok[prim]:
                keep.append(prim)

        # Fallback if all filtered out: keep best z-score
        if not keep:
            zscores = {}
            for prim in PRIMS:
                mu, sd = null_calibration(E_ex[prim], proto, P.null_shifts)
                zscores[prim] = (corr[prim] - mu) / (sd + 1e-8)
            keep = [max(zscores, key=zscores.get)]

        # Order by peak time
        order = sorted(keep, key=lambda k: peak_idx[k])

        # Optional lateral inhibition around winner to refine set
        if len(order) >= 1:
            winner = order[0]
            E_ex = lateral_inhibition(E_ex, winner, peak_idx, P.lateral_width, P.lateral_gain)

        pred_out = apply_sequence(rec["input"], order)
        ok = int(pred_out == rec["output_true"])
        correct += ok

        results.append(dict(
            sample=i,
            input=rec["input"],
            order_true="|".join(rec["order_true"]),
            order_pred="|".join(order),
            ok=bool(ok),
            areas={k: float(area[k]) for k in PRIMS},
            corr=corr,
        ))

        if args.verbose:
            print(f"[{i:02d}] true={rec['order_true']} | pred={order} | ok={bool(ok)}")

    acc = correct / max(1, len(dataset))
    summary = dict(model=args.model, layer=args.layer, samples=len(dataset), accuracy_exact=acc)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(dict(rows=results, summary=summary), f, indent=2)

    print(f"[SUMMARY] Stage11-LLM — acc_exact={acc:.3f} over {len(dataset)} samples")
    print(f"[JSON] wrote {args.out_json}")

def build_argparser():
    ap = argparse.ArgumentParser(description="Stage-11 → LLM integration (minimal doctrine)")
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--pca_dim", type=int, default=19)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sigma", type=int, default=9)
    ap.add_argument("--ema_gamma", type=float, default=0.25)
    ap.add_argument("--med_k", type=int, default=7)
    ap.add_argument("--proto_width", type=int, default=140)
    ap.add_argument("--null_shifts", type=int, default=64)
    ap.add_argument("--z_thresh", type=float, default=3.0)
    ap.add_argument("--lateral_width", type=int, default=80)
    ap.add_argument("--lateral_gain", type=float, default=0.35)
    ap.add_argument("--out_json", type=str, default="stage11_llm_results.json")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    return ap

if __name__ == "__main__":
    run(build_argparser().parse_args())
