
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage10-llm-hook.py
-------------------
Proof-of-concept Stage 10 → LLM integration.
- Taps mid-layer hidden states from a small HuggingFace decoder-only model (default: gpt2).
- Learns per-primitive anchors/prototypes from short demonstrations.
- Runs the Stage-10 parser (perpendicular energy + smoothing + matched filter) over token-time.
- Executes the predicted primitive order on the input string and evaluates metrics.

Usage (example):
  python3 stage10-llm-hook.py --model gpt2 --layer 8 --samples 12 --seed 42

Notes:
- Requires: transformers, torch, scikit-learn, numpy, matplotlib (optional for plots).
- Internet may be required on first run to download the HF model.
- This is a minimal, readable baseline. It is engineered for clarity over speed.
"""

import argparse, random, math, json, os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
import numpy as np

# Soft dependencies (install if missing): pip install transformers torch scikit-learn matplotlib
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
# String primitive executors
# -----------------------------

def apply_primitive(s: str, prim: str) -> str:
    if prim == "reverse":
        return s[::-1]
    if prim == "uppercase":
        return s.upper()
    if prim == "sort":
        return "".join(sorted(list(s)))
    raise ValueError(f"Unknown primitive: {prim}")

def apply_sequence(s: str, seq: List[str]) -> str:
    out = s
    for p in seq:
        out = apply_primitive(out, p)
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
    # hidden_states is a tuple: (embeddings, layer1, ..., layerN)
    hs = out.hidden_states[layer]  # [1, T, H]
    return hs.squeeze(0).detach().cpu().numpy()  # [T, H]

# -----------------------------
# Stage-10 math utilities
# -----------------------------

def moving_average(x: np.ndarray, k: int = 9) -> np.ndarray:
    if k <= 1: return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(xp, np.ones(k)/k, mode="valid")

def half_sine(width: int) -> np.ndarray:
    t = np.linspace(0, np.pi, width)
    v = np.sin(t)
    return v / (np.linalg.norm(v) + 1e-8)

@dataclass
class Stage10Params:
    pca_dim: int = 19
    sigma: int = 9
    proto_width: int = 140
    area_rel: float = 0.45
    corr_rel: float = 0.50

@dataclass
class PrimitiveStats:
    anchor: np.ndarray  # ck \in R^d
    proto: np.ndarray   # pk \in R^d, unit

# -----------------------------
# Learning anchors/prototypes from demos
# -----------------------------

def build_prim_stats(tok, model, demos: Dict[str, List[str]], layer: int, device: str, params: Stage10Params,
                     pca_whitener) -> Dict[str, PrimitiveStats]:
    stats = {}
    for prim, texts in demos.items():
        H = []
        for t in texts:
            hs = get_hidden_states(tok, model, t, layer, device)    # [T, H_orig]
            y = pca_whitener.transform(hs)                          # [T, d]
            # focus on last third of tokens (often contains 'Output:' or transformed string)
            T = y.shape[0]
            seg = y[max(0, T*2//3):, :] if T >= 6 else y
            H.append(seg)
        Y = np.vstack(H)  # [N, d]
        ck = Y.mean(axis=0)
        # Prototype: first principal component of (Y - ck)
        U = Y - ck
        u, s, vh = np.linalg.svd(U, full_matrices=False)
        pk = vh[0]  # top right-singular vector
        # normalize
        pk = pk / (np.linalg.norm(pk) + 1e-8)
        stats[prim] = PrimitiveStats(anchor=ck, proto=pk)
    return stats

# -----------------------------
# Parser over token-time
# -----------------------------

def perpendicular_energy_traces(Y: np.ndarray, prim_stats: Dict[str, PrimitiveStats]) -> Dict[str, np.ndarray]:
    # Y: [T, d] whitened token-time trajectory
    T = Y.shape[0]
    # common mode
    # (here we don't have per-prim traces yet, so we compute E⊥ per channel directly)
    E = {}
    for prim, st in prim_stats.items():
        ck, pk = st.anchor, st.proto
        diff = Y - ck[None, :]
        s_par = diff @ pk  # [T]
        E_par = s_par**2
        E_tot = np.sum(diff**2, axis=1)
        E_perp = np.clip(E_tot - E_par, 0.0, None)  # [T]
        E[prim] = E_perp
    # remove common mode across channels (mean)
    keys = list(prim_stats.keys())
    M = np.stack([E[k] for k in keys], axis=0).mean(axis=0)
    for k in keys:
        E[k] = np.clip(E[k] - M, 0.0, None)
    return E  # dict prim -> [T]

def matched_filter_parse(E_perp: Dict[str, np.ndarray], params: Stage10Params) -> Tuple[List[str], List[str], Dict[str,int], Dict[str,float], Dict[str,float]]:
    keys = list(E_perp.keys())
    T = len(next(iter(E_perp.values())))
    # smooth
    S = {k: moving_average(E_perp[k], k=params.sigma) for k in keys}
    proto = half_sine(params.proto_width)
    # correlation (same-length, then peak)
    peak_idx, corr_peak, area = {}, {}, {}
    for k in keys:
        m = np.correlate(S[k], proto, mode="same")
        idx = int(np.argmax(m))
        peak_idx[k] = idx
        L = params.proto_width
        a, b = max(0, idx - L//2), min(T, idx + L//2)
        w = S[k][a:b]
        w = (w - w.mean())
        pr = proto[:len(w)] - proto[:len(w)].mean()
        denom = (np.linalg.norm(w) * np.linalg.norm(pr) + 1e-8)
        corr_peak[k] = float(np.dot(w, pr) / denom)
        area[k] = float(np.trapz(S[k]))
    # dual relative gates vs best
    Amax = max(area.values()) + 1e-12
    Cmax = max(corr_peak.values()) + 1e-12
    keep = [k for k in keys if (area[k]/Amax >= params.area_rel) and (corr_peak[k]/Cmax >= params.corr_rel)]
    if not keep:
        score = {k: area[k] * corr_peak[k] for k in keys}
        keep = [max(score, key=score.get)]
    order = sorted(keep, key=lambda k: peak_idx[k])
    return keep, order, peak_idx, area, corr_peak

# -----------------------------
# Data synthesis (toy tasks on strings)
# -----------------------------

def make_dataset(rng, n: int = 12) -> List[Dict[str, Any]]:
    # draw a random base string from lowercase ascii
    alpha = list("abcdefghijklmnopqrstuvwxyz")
    ds = []
    for i in range(n):
        L = rng.integers(5, 9)
        s = "".join(rng.choice(alpha, size=L, replace=True))
        # pick 1-3 primitives and order
        k = int(rng.integers(1, 4))
        order = list(rng.choice(PRIMS, size=k, replace=False))
        rng.shuffle(order)
        target = apply_sequence(s, order)
        ds.append(dict(input=s, order_true=order, output_true=target))
    return ds

def demo_texts_for_prims() -> Dict[str, List[str]]:
    # short demonstrations that imprint each primitive in hidden states
    # (few-shot style prompts; customize freely)
    demos = {
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
    return demos

def prompt_for_sample(inp: str) -> str:
    # A generic instruction prompt that does NOT reveal the true order.
    return f"Task: (unknown)\nInput: {inp}\nOutput:"

# -----------------------------
# Main experiment
# -----------------------------

def run(args):
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and (not args.cpu) else "cpu"
    tok, model = load_model(args.model, device)
    params = Stage10Params(pca_dim=args.pca_dim, sigma=args.sigma, proto_width=args.proto_width)

    # 1) Build PCA whitener from a small corpus of hidden states (demos + a few dummy prompts)
    demos = demo_texts_for_prims()
    H_all = []
    for prim, texts in demos.items():
        for t in texts:
            hs = get_hidden_states(tok, model, t, args.layer, device)  # [T, H]
            H_all.append(hs)
    # add some neutral prompts to cover broader space
    neutral = [
        "Explain the rules of chess in one sentence.",
        "Summarize: Warping latent space into a single well can stabilize reasoning.",
        "Translate to French: The cat sleeps on the mat.",
        "List three colors: red, green, blue."
    ]
    for t in neutral:
        H_all.append(get_hidden_states(tok, model, t, args.layer, device))
    H_stack = np.vstack(H_all)  # [N, H]
    pca = PCA(n_components=params.pca_dim, whiten=True, random_state=0).fit(H_stack)

    # 2) Learn per-primitive anchors/prototypes in whitened space
    prim_stats = build_prim_stats(tok, model, demos, args.layer, device, params, pca)

    # 3) Build dataset
    dataset = make_dataset(rng, n=args.samples)

    # 4) Evaluate
    correct = 0
    rows = []
    for i, rec in enumerate(dataset, 1):
        # (a) get token-time trajectory for the test prompt
        prompt = prompt_for_sample(rec["input"])
        Y = pca.transform(get_hidden_states(tok, model, prompt, args.layer, device))  # [T, d]

        # (b) compute E_perp per channel and parse
        E = perpendicular_energy_traces(Y, prim_stats)
        keep, order, peaks, areas, corr = matched_filter_parse(E, params)

        # (c) execute predicted order
        pred_out = apply_sequence(rec["input"], order)

        ok = int(pred_out == rec["output_true"])
        correct += ok

        rows.append(dict(
            sample=i,
            input=rec["input"],
            order_true="|".join(rec["order_true"]),
            order_pred="|".join(order),
            tasks_pred="|".join(keep),
            ok=bool(ok),
            areas={k: float(areas[k]) for k in PRIMS},
            corr={k: float(corr[k]) for k in PRIMS},
        ))

        if args.plot_dir and plt is not None:
            os.makedirs(args.plot_dir, exist_ok=True)
            T = len(next(iter(E.values())))
            fig, ax = plt.subplots(figsize=(11,4))
            for j, k in enumerate(PRIMS):
                ax.plot(moving_average(E[k], k=params.sigma), label=f"E⊥ {k}", linewidth=2)
            ax.set_title(f"sample{i:02d} — true: {rec['order_true']} — pred: {order}")
            ax.set_xlabel("token step"); ax.set_ylabel("residual aligned power (smoothed)")
            ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(os.path.join(args.plot_dir, f"sample{i:02d}.png"), dpi=130)
            plt.close(fig)

        if args.verbose:
            print(f"[{i:02d}] true={rec['order_true']} | pred={order} | ok={bool(ok)}")

    acc = correct / max(1, len(dataset))
    summary = dict(
        model=args.model,
        layer=args.layer,
        samples=len(dataset),
        accuracy_exact=acc,
    )

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(dict(rows=rows, summary=summary), f, indent=2)

    print(f"[SUMMARY] Stage10-LLM — acc_exact={acc:.3f} over {len(dataset)} samples")
    print(f"[JSON] wrote {args.out_json}")
    if args.plot_dir and plt is not None:
        print(f"[PLOTS] wrote dir: {args.plot_dir}")

def build_argparser():
    ap = argparse.ArgumentParser(description="Stage 10 → LLM integration hook")
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--layer", type=int, default=8, help="0..n_layers (embedding=0, first block=1)")
    ap.add_argument("--pca_dim", type=int, default=19)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sigma", type=int, default=9)
    ap.add_argument("--proto_width", type=int, default=140)
    ap.add_argument("--plot_dir", type=str, default="plots_stage10_llm")
    ap.add_argument("--out_json", type=str, default="stage10_llm_results.json")
    ap.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is available")
    ap.add_argument("--verbose", action="store_true")
    return ap

if __name__ == "__main__":
    run(build_argparser().parse_args())
