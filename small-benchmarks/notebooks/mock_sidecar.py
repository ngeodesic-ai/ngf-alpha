# mock_sidecar.py — augmented to run on real HellaSwag
# Demo NGF-style sidecar (mock scorer) + HellaSwag loader/runner.
# Requires: pip install datasets

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import argparse, time, sys
import numpy as np
from datasets import load_dataset
from typing import Tuple
import numpy as np

# --- ADD: imports for latent mode ---
from typing import Tuple
import numpy as np

# Optional heavy deps are imported lazily inside functions:
#   transformers, sklearn, datasets (already used above)

# ========= Latent WDD (external) =========
def _latent_pool_lastk(hs: "torch.Tensor", k_last: int) -> "np.ndarray":
    # hs: [B, T, D] → mean of last k tokens per item, returns [B, D] (float32 np)
    import torch
    k = min(k_last, hs.shape[1])
    pooled = hs[:, -k:, :].mean(1)
    return pooled.detach().cpu().numpy().astype(np.float32)

def _calibrate_pca_basis(model, tok, split: str, tap: int, k_last: int, calib_n: int = 256) -> "sklearn.decomposition.PCA":
    """
    Fit a PCA(3, whiten=True) basis ONCE using a small slice of HellaSwag prompts.
    We use just the prefix (ctx+ctx_a) for speed/stability (like your layer scan). 
    """
    from datasets import load_dataset
    from sklearn.decomposition import PCA
    import torch

    ds = load_dataset("hellaswag", split=split)
    n = min(calib_n, len(ds))
    prefixes = []
    for i in range(n):
        ex = ds[i]
        ctx = (ex.get("ctx","") or "").strip()
        ctx_a = (ex.get("ctx_a","") or "").strip()
        if ctx_a and not ctx.endswith(ctx_a):
            if ctx and ctx in ctx_a:
                prefix = ctx_a
            else:
                prefix = (ctx + " " + ctx_a).strip()
        else:
            prefix = ctx
        # keep the same extra cue you used in your runner
        prefixes.append((prefix + " The most plausible next event is:").strip())

    with torch.no_grad():
        batch = tok(prefixes, return_tensors="pt", padding=True, truncation=True).to(model.device)
        out = model(**batch, output_hidden_states=True)
        hs = out.hidden_states[tap]  # [B, T, D]
        H = _latent_pool_lastk(hs, k_last=k_last)  # [B, D]

    pca = PCA(n_components=3, whiten=True, random_state=0)
    pca.fit(H)
    return pca  # will be reused for all items

def _project_choices(model, tok, pca, prompt: str, endings: list[str], tap: int, k_last: int) -> "np.ndarray":
    """Return a 4×D' = 4×3 array of PCA coords for the four choices (using pooled latents at tap)."""
    import torch
    with torch.no_grad():
        texts = [(prompt + " " + e).strip() for e in endings]
        batch = tok(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        out = model(**batch, output_hidden_states=True)
        hs = out.hidden_states[tap]  # [4, T, D]
        H = _latent_pool_lastk(hs, k_last=k_last)  # [4, D]
    Y3 = pca.transform(H)  # [4, 3] whitened
    return Y3

def _warp_detect(Y3: "np.ndarray", detect_z: float = 0.3) -> Tuple[int, bool, "np.ndarray"]:
    """
    Warp: use PC1-PC2 as plane; depth = -||Z2|| normalized.
    Detect: robust gate vs. null using MAD; abstain if no candidate clears gate.
    Returns (best_idx, abstain, depth_scores).
    """
    Z2 = Y3[:, :2]
    r = np.linalg.norm(Z2, axis=1) + 1e-9
    depth_raw = -r
    depth = (depth_raw - depth_raw.min()) / (depth_raw.ptp() + 1e-9)

    # Robust gate (MAD-based)
    med = np.median(depth)
    mad = np.median(np.abs(depth - med)) + 1e-9
    sigma = 1.4826 * mad
    gate = med + detect_z * sigma  # higher = deeper
    ok = depth > gate
    idx_ok = np.where(ok)[0]

    if idx_ok.size == 0:
        return int(depth.argmax()), True, depth

    # Add a small margin bonus for the best among 'ok' to stabilize ties
    order = np.argsort(-depth[idx_ok])
    best_local = idx_ok[order[0]]
    return int(best_local), False, depth

def rerank_with_latent_wdd(model_name: str = "gpt2",
                           split: str = "validation",
                           n: int = 100,
                           tap: int = -9,
                           k_last: int = 6,
                           calib_n: int = 256,
                           detect_z: float = 0.3,
                           denoise: str = "off",
                           debug: int = 0,
                           device: str = "auto"):
    """
    External W→D(+D) sidecar using real latents.
    denoise: "off" | "ema"
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    # device selection
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Calibrate PCA basis once (fast)
    pca = _calibrate_pca_basis(model, tok, split=split, tap=tap, k_last=k_last, calib_n=calib_n)

    # Load dataset slice
    ds = load_dataset("hellaswag", split=split)
    if n and n < len(ds):
        ds = ds.select(range(n))

    correct = 0; abstains = 0
    # Simple EMA denoiser state (per-item), only if requested
    ema_beta = 0.6

    for i, ex in enumerate(ds):
        # Build the same prefix you use in the existing runner
        prefix = safe_prefix(ex.get("ctx",""), ex.get("ctx_a",""))
        prompt = (prefix + " The most plausible next event is:").strip()
        endings = list(ex["endings"]); gold = int(ex["label"])

        # One projection pass
        Y3 = _project_choices(model, tok, pca, prompt, endings, tap=tap, k_last=k_last)

        # Optional denoise (EMA over a tiny synthetic jitter on the plane)
        if denoise.lower() == "ema":
            # jitter PC1-2 slightly to simulate sampling noise and smooth depth
            jit = 2
            acc_depth = None
            for j in range(jit):
                Z2 = Y3[:, :2].copy()
                Z2 += np.random.normal(0, 0.02, size=Z2.shape)  # small plane jitter
                r = np.linalg.norm(Z2, axis=1) + 1e-9
                d = (-(r) - (-(r)).min()) / ((-(r)).ptp() + 1e-9)
                acc_depth = d if acc_depth is None else (ema_beta*acc_depth + (1-ema_beta)*d)
            # replace the depth axis by smoothed version
            Y3 = np.column_stack([Y3[:,0], Y3[:,1], acc_depth])

        # Warp+Detect in the plane (depth uses −||PC12||)
        pred, abstain, depth = _warp_detect(Y3, detect_z=detect_z)
        correct += int(pred == gold); abstains += int(abstain)

        if debug and i < debug:
            print(f"[{i}] gold={gold} pred={pred} abstain={abstain} depth={np.round(depth,3)}\n") 
            best = endings[pred][:64].replace('\\n',' ')
            print(f"best: {best}")

        # progress pulse
        n_done = i + 1
        if n_done % 10 == 0 or n_done == len(ds):
            print(f"{n_done} acc: {correct}/{n_done}={correct/n_done:.4f} (abstains={abstains})")

    total = len(ds) or 1
    print("\n=== HellaSwag (sidecar/latent WDD) ===")
    print(f"model={model_name} tap={tap} k_last={k_last} calib_n={calib_n} "
          f"split={split} n={total} acc_top1={correct/total:.4f} abstain_rate={abstains/total:.4f}")

# ========= Wire into CLI =========
def run_hellaswag_latent_wdd_cli(args):
    rerank_with_latent_wdd(model_name=args.model,
                           split=args.split,
                           n=args.n,
                           tap=args.tap,
                           k_last=args.k_last,
                           calib_n=args.calib_n,
                           detect_z=args.detect_z,
                           denoise=args.denoise,
                           debug=args.debug,
                           device=args.device)


# ---------------------------
# Mock encoder (unchanged)
# ---------------------------
def simple_text_vec(text: str) -> np.ndarray:
    import re
    lowers = sum(c.islower() for c in text)
    uppers = sum(c.isupper() for c in text)
    digits = sum(c.isdigit() for c in text)
    spaces = text.count(" ")
    dots = text.count(".")
    commas = text.count(",")
    other_punct = sum(c in "!?:;-%" for c in text)
    length = len(text)
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    has_num = 1 if nums else 0
    num_len_avg = np.mean([len(n) for n in nums]) if nums else 0.0
    has_one_decimal = 1 if any(re.match(r"^-?\d+\.\d$", n) for n in nums) else 0
    has_two_decimals = 1 if any(re.match(r"^-?\d+\.\d\d$", n) for n in nums) else 0
    has_three_decimals = 1 if any(re.match(r"^-?\d+\.\d\d\d$", n) for n in nums) else 0
    v = np.array([lowers, uppers, digits, spaces, dots, commas, other_punct, length,
                  has_num, num_len_avg, has_one_decimal, has_two_decimals, has_three_decimals],
                 dtype=float)
    v = (v - v.mean()) / (v.std() + 1e-6)
    return v

# ---------------------------
# Mock Warp→Detect→Denoise scorer (unchanged)
# ---------------------------
@dataclass
class Score:
    depth: float
    margin: float
    conf: float
    gates_ok: bool

class MockScorer:
    def __init__(self, context: str):
        self.context = context
        self._history = []

    def score_batch(self, prompt: str, candidates: List[str]) -> List[Score]:
        base = simple_text_vec(prompt)
        mats = np.stack([simple_text_vec(prompt + " " + c) for c in candidates], axis=0)
        X = mats - mats.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        Z = X @ Vt.T[:, :2]
        radii = np.linalg.norm(Z, axis=1)
        depth_raw = -radii
        import re
        asks_one_dec = "one decimal" in prompt.lower()
        fmt_bonus = np.array([1.0 if asks_one_dec and re.search(r"\b-?\d+\.\d\b", c) else 0.0 for c in candidates])
        numeric_mask = np.array([1.0 if re.search(r"\d", c) else 0.0 for c in candidates])
        depth = (depth_raw - depth_raw.min()) / (depth_raw.ptp() + 1e-6)
        depth = 0.75*depth + 0.35*fmt_bonus + 0.10*numeric_mask
        null = np.mean(depth) - 0.25*np.std(depth)
        gates = depth > null
        order = np.argsort(-depth)
        best = order[0]
        second = order[1] if len(order) > 1 else best
        margin = np.zeros_like(depth)
        margin[best] = max(0.0, depth[best] - depth[second])
        conf = 1 / (1 + np.exp(-5*(depth - depth.mean())))
        return [Score(depth=float(depth[i]), margin=float(margin[i]), conf=float(conf[i]), gates_ok=bool(gates[i])) for i in range(len(candidates))]

# ---------------------------
# NGF Sidecar harness (unchanged API)
# ---------------------------
class NGFSidecar:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = {}

    def rerank(self, prompt: str, candidates: List[str]) -> Dict[str, Any]:
        scorer = MockScorer(context=prompt)
        scores = scorer.score_batch(prompt, candidates)
        gamma = 0.5
        idx_ok = [i for i,s in enumerate(scores) if s.gates_ok]
        if not idx_ok:
            best_idx = int(np.argmax([s.depth for s in scores]))
            return {"best": candidates[best_idx], "best_idx": best_idx, "abstain": True,
                    "scores": [s.__dict__ for s in scores]}
        comp = np.array([scores[i].depth + gamma*scores[i].margin for i in idx_ok])
        best_local = idx_ok[int(np.argmax(comp))]
        return {"best": candidates[best_local], "best_idx": best_local, "abstain": False,
                "scores": [s.__dict__ for s in scores]}

# ---------------------------
# Helpers (mirrors ngf_benchmark.py behavior)
# ---------------------------
def safe_prefix(ctx: str, ctx_a: str) -> str:
    ctx = (ctx or "").strip()
    ctx_a = (ctx_a or "").strip()
    if not ctx_a:
        return ctx
    if ctx.endswith(ctx_a):
        return ctx
    if ctx and ctx in ctx_a:
        return ctx_a
    return (ctx + " " + ctx_a).strip()

# ---------------------------
# HellaSwag runner
# ---------------------------
def run_hellaswag(n: int, split: str, debug: int) -> None:

    ds = load_dataset("hellaswag", split=split)
    if n and n < len(ds):
        ds = ds.select(range(n))

    sidecar = NGFSidecar()
    correct = 0
    abstains = 0
    t0 = time.time()

    for i, ex in enumerate(ds):
        # Build the same prefix/endings the benchmark uses
        prefix = safe_prefix(ex.get("ctx", ""), ex.get("ctx_a", ""))
        # Optional: add a small cue like the toy demo
        prompt = (prefix + " The most plausible next event is:").strip()
        endings = list(ex["endings"])
        label = int(ex["label"])

        out = sidecar.rerank(prompt, endings)
        pred = out["best_idx"]
        correct += int(pred == label)
        abstains += int(out.get("abstain", False))

        if debug and i < debug:
            print("[DEBUG]", {
                "i": i,
                "gold": label,
                "pred": pred,
                "best_text": endings[pred][:60].replace("\n"," "),
                "gold_text": endings[label][:60].replace("\n"," "),
                "abstain": out.get("abstain", False)
            })

        n_done = i + 1
        if n_done % 10 == 0 or n_done == len(ds):
            print(f"{n_done} acc: {correct}/{n_done}={correct/n_done:.4f} (abstains={abstains})")

    total = len(ds)
    acc = correct / total if total else 0.0
    print("\n=== HellaSwag (sidecar/mock) ===")
    print(f"split={split} n={total} acc_top1={acc:.4f} abstain_rate={abstains/total:.4f} elapsed_sec={time.time()-t0:.1f}")

# ---------------------------
# Demos (keep originals)
# ---------------------------
def run_demos():
    prompt1 = "Round 7.432 to one decimal."
    cands1  = ["7.4", "7.43", "7.5", "7.40", "8"]
    sidecar = NGFSidecar()
    out1 = sidecar.rerank(prompt1, cands1)

    prompt2 = "A man is riding a skateboard down a ramp. He loses his balance. The most plausible next event is:"
    cands2  = [
        "He falls off and lands on the ground.",
        "A cat jumps onto the table.",
        "The man starts typing on a keyboard.",
        "The skateboard floats into space."
    ]
    out2 = sidecar.rerank(prompt2, cands2)

    print("=== Demo 1: Rounding ===")
    print("Prompt:", prompt1)
    print("Best:", out1["best"], "Abstain:", out1["abstain"])
    for s, c in zip(out1["scores"], cands1):
        print(f"{c:8s} -> depth={s['depth']:.3f} margin={s['margin']:.3f} conf={s['conf']:.2f} gate={s['gates_ok']}")

    print("\n=== Demo 2: HellaSwag Toy ===")
    print("Prompt:", prompt2)
    print("Best:", out2["best"], "Abstain:", out2["abstain"])
    for s, c in zip(out2["scores"], cands2):
        print(f"{c:45s} -> depth={s['depth']:.3f} margin={s['margin']:.3f} conf={s['conf']:.2f} gate={s['gates_ok']}")

def run_hellaswag_demo(n = 3, split="validation"):
    ds = load_dataset("hellaswag", split=split)
    ds = ds.select(range(n))
    sidecar = NGFSidecar()
    correct = 0
    abstains = 0

    for i, ex in enumerate(ds):
        # Build the same prefix/endings the benchmark uses
        prefix = safe_prefix(ex.get("ctx", ""), ex.get("ctx_a", ""))
        prompt = (prefix + " The most plausible next event is:").strip()
        endings = list(ex["endings"])
        label = int(ex["label"])
        out = sidecar.rerank(prompt, endings)
        pred = out["best_idx"]
        
        print(f"------ Prompt {i} ------ ")
        print(f"Prompt: {prompt}\n")
        for i, ending in enumerate(endings):
            print(f"({i}): {ending}")

        print(f"\nCorrect: {label}, Pred {pred}\n")

        correct += int(pred == label)
        abstains += int(out.get("abstain", False))

    total = len(ds)
    acc = correct / total if total else 0.0
    print("\n=== HellaSwag (sidecar/mock) ===")
    print(f"split={split} n={total} acc_top1={acc:.4f} abstain_rate={abstains/total:.4f}")
    
    pass


# ---------------------------
# CLI
# ---------------------------
# --- Unified CLI + Notebook runner ---
def run_entry(mode: str = "hellaswag", split: str = "validation", n: int = 100, debug: int = 0):
    """Entry point usable from both CLI and notebooks."""
    if mode == "hellaswag":
        run_hellaswag(n=n, split=split, debug=debug)
    else:
        run_demos()

import argparse
def get_args(notebook=True):
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--mode", choices=["demos", "hellaswag", "hellaswag_latent_wdd"], default="hellaswag",
                    help="Toy demos, toy HellaSwag (mock), or real-latent WDD sidecar.")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--tap", type=int, default=-9)
    ap.add_argument("--k_last", type=int, default=6)
    ap.add_argument("--calib_n", type=int, default=256)
    ap.add_argument("--detect_z", type=float, default=0.3, help="Gate strength (MAD-z units).")
    ap.add_argument("--denoise", choices=["off","ema"], default="off")
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    
    if notebook:
        return ap.parse_args([
            "--split", "validation",
            "--n", "50",
            "--debug", "0",
            "--mode", "hellaswag",
            "--model", "gpt2",
            "--tap", "-9",
            "--k_last", "6", 
            "--calib_n","256",
            "--detect_z", "0.3",
            "--denoise", "off",
            "--device", "cpu"
        ])
    else:
        return ap.parse_args()

if __name__ == "__main__":
    args = get_args(False)
    run_hellaswag_latent_wdd_cli(args)
