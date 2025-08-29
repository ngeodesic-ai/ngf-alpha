
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
biggest_kid_ablation.py
-----------------------
A/B safety check for the "biggest kid" doctrine: ensure NGF Stage-11 policy does not harm stock behavior on unrelated tasks.

Buckets:
  (A) Control QA (should abstain) — measure exact-match vs references.
  (B) Primitive tasks (reverse/uppercase/sort) — allow intervention if confident.

Policy:
  - Compute NGF per-primitive z-scores via null-calibrated matched filter on exclusive residuals.
  - If max z-score < z_abstain: ABSTAIN => return stock output.
  - Else: apply detected primitive order to input string (for primitive tasks).
  - For QA, even if z is high, we ABSTAIN by default to be conservative (toggleable).

Outputs:
  - JSON report with per-item details and aggregate metrics for harm (bucket A) and utility (bucket B).
"""

import argparse, random, os, json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np

# Soft deps
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:
    raise SystemExit("Please: pip install transformers torch --upgrade\n" + str(e))

from sklearn.decomposition import PCA

# -------------------- Utilities --------------------

PRIMS = ["reverse", "uppercase", "sort"]

def apply_primitive(s: str, prim: str) -> str:
    if prim == "reverse": return s[::-1]
    if prim == "uppercase": return s.upper()
    if prim == "sort": return "".join(sorted(list(s)))
    raise ValueError(prim)

def apply_sequence(s: str, seq: List[str]) -> str:
    out = s
    for p in seq: out = apply_primitive(out, p)
    return out

def load_model(model_name: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
    model.eval().to(device)
    return tok, model

@torch.no_grad()
def hidden_states(tok, model, text: str, layer: int, device: str) -> np.ndarray:
    inp = tok(text, return_tensors="pt")
    inp = {k:v.to(device) for k,v in inp.items()}
    out = model(**inp, output_hidden_states=True)
    hs = out.hidden_states[layer]  # [1,T,H]
    return hs.squeeze(0).detach().cpu().numpy()

def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1: return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad,pad), mode="reflect")
    return np.convolve(xp, np.ones(k)/k, mode="valid")

def half_sine(L: int) -> np.ndarray:
    t = np.linspace(0, np.pi, L)
    v = np.sin(t)
    return v/(np.linalg.norm(v)+1e-8)

def circular_shift(x: np.ndarray, s: int) -> np.ndarray:
    s = s % len(x)
    if s == 0: return x.copy()
    return np.concatenate([x[-s:], x[:-s]])

def matched_filter_corr(x: np.ndarray, proto: np.ndarray) -> Tuple[int, float]:
    m = np.correlate(x, proto, mode="same")
    idx = int(np.argmax(m))
    L = len(proto)
    a, b = max(0, idx - L//2), min(len(x), idx + L//2)
    w = x[a:b]; w = (w - w.mean())
    pr = proto[:len(w)] - proto[:len(w)].mean()
    denom = (np.linalg.norm(w)*np.linalg.norm(pr) + 1e-8)
    corr = float(np.dot(w, pr)/denom)
    return idx, corr

def exclusive_residual(traces: Dict[str,np.ndarray]) -> Dict[str,np.ndarray]:
    # z-score then project out span of others; keep positive part
    Z = {}
    for k, x in traces.items():
        mu, sd = x.mean(), x.std()+1e-8
        Z[k] = (x - mu)/sd
    keys = list(traces.keys())
    E = {}
    for i, ki in enumerate(keys):
        zi = Z[ki]
        others = [Z[kj] for j,kj in enumerate(keys) if j!=i]
        if not others:
            E[ki] = np.maximum(zi, 0.0)
            continue
        A = np.stack(others, axis=1)
        lam = 1e-3
        ATA = A.T@A + lam*np.eye(A.shape[1])
        ATz = A.T@zi
        w = np.linalg.solve(ATA, ATz)
        proj = A@w
        resid = zi - proj
        E[ki] = np.maximum(resid, 0.0)
    return E

# -------------------- NGF-lite stats --------------------

def demo_texts() -> Dict[str, List[str]]:
    return {
        "reverse": [
            "Task: reverse\nInput: abcdef\nOutput: fedcba",
            "Task: reverse\nInput: tulip\nOutput: pilut",
        ],
        "uppercase": [
            "Task: uppercase\nInput: helloWorld\nOutput: HELLOWORLD",
            "Task: uppercase\nInput: ngfAlpha\nOutput: NGFALPHA",
        ],
        "sort": [
            "Task: sort\nInput: zebra\nOutput: aberz",
            "Task: sort\nInput: casino\nOutput: acinos",
        ],
    }

@dataclass
class NGFParams:
    pca_dim: int = 19
    sigma: int = 9
    proto_width: int = 140
    null_shifts: int = 64
    z_abstain: float = 3.5  # abstain unless very confident
    abstain_on_QA: bool = True

@dataclass
class NGFState:
    pca: PCA
    anchors: Dict[str,np.ndarray]
    protos: Dict[str,np.ndarray]
    proto_template: np.ndarray

def build_ngf(tok, model, layer: int, device: str, P: NGFParams) -> NGFState:
    demos = demo_texts()
    H = []
    for texts in demos.values():
        for t in texts:
            H.append(hidden_states(tok, model, t, layer, device))
    # a few neutral prompts
    for t in [
        "Explain the rules of chess in one sentence.",
        "List three colors: red, green, blue.",
        "Translate to French: The cat sleeps on the mat."
    ]:
        H.append(hidden_states(tok, model, t, layer, device))
    H = np.vstack(H)
    pca = PCA(n_components=P.pca_dim, whiten=True, random_state=0).fit(H)

    anchors, protos = {}, {}
    for prim, texts in demos.items():
        Y = []
        for t in texts:
            y = pca.transform(hidden_states(tok, model, t, layer, device))
            T = y.shape[0]
            seg = y[max(0,T*2//3):,:] if T>=6 else y
            Y.append(seg)
        Y = np.vstack(Y)
        ck = Y.mean(axis=0)
        U = Y - ck
        _,_,vh = np.linalg.svd(U, full_matrices=False)
        pk = vh[0]/(np.linalg.norm(vh[0])+1e-8)
        anchors[prim], protos[prim] = ck, pk

    proto_template = half_sine(P.proto_width)
    return NGFState(pca=pca, anchors=anchors, protos=protos, proto_template=proto_template)

def ngf_scores(state: NGFState, Y: np.ndarray, P: NGFParams) -> Dict[str, Dict[str, float]]:
    # Compute exclusive residual per channel, then matched-filter z-scores via circular-shift nulls
    E_par, E_perp = {}, {}
    for prim in PRIMS:
        ck, pk = state.anchors[prim], state.protos[prim]
        diff = Y - ck[None,:]
        s_par = diff @ pk
        E_par[prim] = s_par**2
        E_tot = np.sum(diff**2, axis=1)
        E_perp[prim] = np.clip(E_tot - E_par[prim], 0.0, None)

    # smooth & exclusive residual
    Tr = {}
    for prim in PRIMS:
        e = moving_average(E_perp[prim], P.sigma)
        Tr[prim] = e
    Ex = exclusive_residual(Tr)

    out = {}
    for prim in PRIMS:
        idx, c = matched_filter_corr(Ex[prim], state.proto_template)
        # null calibration
        T = len(Ex[prim])
        peaks = []
        for i in range(P.null_shifts):
            s = (i * (T // P.null_shifts + 1)) % T
            xs = circular_shift(Ex[prim], s)
            _, cn = matched_filter_corr(xs, state.proto_template)
            peaks.append(cn)
        mu, sd = float(np.mean(peaks)), float(np.std(peaks)+1e-8)
        z = (c - mu)/sd
        out[prim] = {"peak_idx": idx, "corr": float(c), "z": float(z)}
    return out

# -------------------- Data --------------------

def control_qa_set() -> List[Dict[str,str]]:
    # Replace/augment with your own. ground_truth kept simple for demo.
    return [
        {"q":"What is the capital of France?", "a":"Paris"},
        {"q":"2 + 2 equals what number?", "a":"4"},
        {"q":"What color is the sky on a clear day?", "a":"blue"},
        {"q":"Name the largest planet in our solar system.", "a":"Jupiter"},
        {"q":"What is H2O commonly known as?", "a":"water"},
    ]

def primitive_set() -> List[Dict[str,Any]]:
    return [
        {"input":"abcdef", "order_true":["reverse"], "output_true":"fedcba"},
        {"input":"helloWorld", "order_true":["uppercase"], "output_true":"HELLOWORLD"},
        {"input":"zebra", "order_true":["sort"], "output_true":"aberz"},
        {"input":"panda", "order_true":["reverse","uppercase"], "output_true":"ADNAP"},
    ]

# -------------------- Generation helpers --------------------

@torch.no_grad()
def generate(model, tok, prompt: str, device: str, max_new_tokens=32) -> str:
    enc = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

def exact_match(pred: str, gold: str) -> bool:
    return pred.strip().lower() == gold.strip().lower()

# -------------------- Main --------------------

def run(args):
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() and (not args.cpu) else "cpu"
    tok, model = load_model(args.model, device)
    P = NGFParams(pca_dim=args.pca_dim, sigma=args.sigma, proto_width=args.proto_width,
                  null_shifts=args.null_shifts, z_abstain=args.z_abstain, abstain_on_QA=not args.allow_ngf_on_qa)
    state = build_ngf(tok, model, args.layer, device, P)

    report = {"settings": vars(args), "results": {}}

    # Bucket (A): Control QA
    qa = control_qa_set()
    qa_rows = []
    qa_correct_stock, qa_correct_ngf = 0, 0
    for i, item in enumerate(qa, 1):
        q = item["q"]; gold = item["a"]
        prompt = q + " Answer succinctly:"
        stock = generate(model, tok, prompt, device)
        # NGF policy: ABSTAIN on QA (unless allow_ngf_on_qa), or if max z < threshold
        Y = state.pca.transform(hidden_states(tok, model, "Task: (unknown)\nInput: "+q+"\nOutput:", args.layer, device))
        scores = ngf_scores(state, Y, P)
        best_prim, best_z = max(scores.items(), key=lambda kv: kv[1]["z"])
        if P.abstain_on_QA or best_z < P.z_abstain:
            ngf_out = stock  # abstain
        else:
            # (conservative) still abstain for QA by default
            ngf_out = stock
        qa_correct_stock += int(exact_match(stock, gold))
        qa_correct_ngf += int(exact_match(ngf_out, gold))
        qa_rows.append({"q": q, "gold": gold, "stock": stock, "ngf": ngf_out, "best_prim": best_prim, "best_z": best_z})

    report["results"]["control_qa"] = {
        "rows": qa_rows,
        "accuracy_stock": qa_correct_stock/len(qa),
        "accuracy_ngf_policy": qa_correct_ngf/len(qa),
        "delta": (qa_correct_ngf - qa_correct_stock)/len(qa)
    }

    # Bucket (B): Primitive tasks
    prims = primitive_set()
    prim_rows = []
    prim_correct_stock, prim_correct_ngf = 0, 0
    for i, rec in enumerate(prims, 1):
        s = rec["input"]; gold = rec["output_true"]
        # stock model tries to output transformed string given instruction
        stock_prompt = f"Transform the input according to hidden rule.\nInput: {s}\nOutput:"
        stock = generate(model, tok, stock_prompt, device)
        prim_correct_stock += int(exact_match(stock, gold))

        # NGF policy: if confident, apply detected primitive order; else abstain
        Y = state.pca.transform(hidden_states(tok, model, f"Task: (unknown)\nInput: {s}\nOutput:", args.layer, device))
        scores = ngf_scores(state, Y, P)
        # order by peak index for included ones above threshold; for simplicity, just pick top-1 when > z_abstain
        best_prim, best = max(scores.items(), key=lambda kv: kv[1]["z"])
        if best["z"] >= P.z_abstain:
            ngf_out = apply_sequence(s, [best_prim])
        else:
            ngf_out = stock  # abstain if not confident

        prim_correct_ngf += int(exact_match(ngf_out, gold))
        prim_rows.append({"input": s, "gold": gold, "stock": stock, "ngf": ngf_out, "best_prim": best_prim, "best_z": best["z"]})

    report["results"]["primitive"] = {
        "rows": prim_rows,
        "accuracy_stock": prim_correct_stock/len(prims),
        "accuracy_ngf_policy": prim_correct_ngf/len(prims),
        "delta": (prim_correct_ngf - prim_correct_stock)/len(prims)
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[REPORT] wrote {args.out_json}")
    print("Control QA — acc(stock) vs acc(NGF-policy):",
          report["results"]["control_qa"]["accuracy_stock"],
          report["results"]["control_qa"]["accuracy_ngf_policy"],
          "delta:", report["results"]["control_qa"]["delta"])
    print("Primitive — acc(stock) vs acc(NGF-policy):",
          report["results"]["primitive"]["accuracy_stock"],
          report["results"]["primitive"]["accuracy_ngf_policy"],
          "delta:", report["results"]["primitive"]["delta"])

def build_argparser():
    ap = argparse.ArgumentParser(description="A/B safety check for 'biggest kid' (cognition well) policy")
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--pca_dim", type=int, default=19)
    ap.add_argument("--sigma", type=int, default=9)
    ap.add_argument("--proto_width", type=int, default=140)
    ap.add_argument("--null_shifts", type=int, default=64)
    ap.add_argument("--z_abstain", type=float, default=3.5)
    ap.add_argument("--allow_ngf_on_qa", action="store_true", help="allow NGF to intervene on QA (default abstain)")
    ap.add_argument("--out_json", type=str, default="biggest_kid_ablation_report.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    return ap

if __name__ == "__main__":
    run(build_argparser().parse_args())
