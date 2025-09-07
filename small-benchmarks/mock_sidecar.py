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

def _torch_pca_from_sklearn(pca, device, dtype):
    """
    Convert sklearn PCA (whiten=True) to torch tensors for fast on-GPU transform.
    Returns dict with mean (mu), components (W), scale (S) so that:
      Y = (X - mu) @ W.T / sqrt(var)   (since whiten=True)
    """
    import torch, numpy as np
    W = torch.tensor(pca.components_, device=device, dtype=dtype)          # [k, D]
    mu = torch.tensor(pca.mean_, device=device, dtype=dtype)               # [D]
    var = torch.tensor(pca.explained_variance_, device=device, dtype=dtype) # [k]
    S = torch.sqrt(var + 1e-9)
    gmu = getattr(pca, "_ngf_depth_mu", 0.0)
    gsig = getattr(pca, "_ngf_depth_sig", 1e-9)
    return {"W": W, "mu": mu, "S": S,
            "gmu": torch.tensor(gmu, device=device, dtype=dtype),
            "gsig": torch.tensor(gsig, device=device, dtype=dtype)}

def _detect_kd_torch(Yk_row, gmu, gsig, detect_z: float):
    """
    Yk_row: [4, k] torch; returns (best_idx, abstain, depth[4] as torch).
    depth_raw = -||Y||; global z-norm with (gmu, gsig); local [0,1] scaling.
    """
    import torch
    r = torch.linalg.vector_norm(Yk_row, dim=1) + 1e-9
    depth_raw = -r
    depth_g = (depth_raw - gmu) / gsig
    # local [0,1]
    d_min = depth_g.min()
    d_ptp = torch.clamp(depth_g.max() - d_min, min=1e-9)
    depth = (depth_g - d_min) / d_ptp
    # robust MAD gate in torch
    med = depth.median()
    mad = (depth - med).abs().median() + 1e-9
    sigma = 1.4826 * mad
    gate = med + detect_z * sigma
    ok = depth > gate
    idx_ok = torch.nonzero(ok, as_tuple=False).flatten()
    if idx_ok.numel() == 0:
        return int(depth.argmax().item()), True, depth
    # choose best among ok
    best = int(depth[idx_ok].argmax().item())
    return int(idx_ok[best].item()), False, depth


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

    Y3 = pca.transform(H)                   # [B,3]
    Z2 = Y3[:, :2]
    r  = np.linalg.norm(Z2, axis=1) + 1e-9
    depth_raw = -r
    g_mu  = depth_raw.mean()
    g_sig = depth_raw.std() + 1e-9
    pca._ngf_depth_mu = float(g_mu)
    pca._ngf_depth_sig = float(g_sig)
    
    return pca  # will be reused for all items

def _project_choices_batched(model, tok, pcaT, prompts: list[str], endings_list: list[list[str]],
                             tap: int, k_last: int, max_length: int):
    """
    Project a batch of B prompts, each with 4 endings, in one forward.
    Returns Yk: torch.Tensor [B, 4, k]
    """
    import torch
    texts = []
    for prompt, ends in zip(prompts, endings_list):
        texts.extend([(prompt + " " + e).strip() for e in ends])  # B*4 strings
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=max_length).to(model.device)
    with torch.inference_mode():
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[tap]  # [B*4, T, D]
        k = min(k_last, hs.shape[1])
        H = hs[:, -k:, :].mean(1).to(dtype=pcaT["W"].dtype)  # [B*4, D]
        # PCA (whiten=True): Y = ((H - mu) @ W.T) / S
        Xc = H - pcaT["mu"]
        Y = (Xc @ pcaT["W"].t()) / pcaT["S"]  # [B*4, k]
    B = len(prompts)
    return Y.view(B, 4, -1)  # [B,4,k]

def _warp_detect(Y3: "np.ndarray", detect_z: float = 0.3) -> Tuple[int, bool, "np.ndarray"]:
    """
    Warp: use PC1-PC2 as plane; depth = -||Z2|| normalized.
    Detect: robust gate vs. null using MAD; abstain if no candidate clears gate.
    Returns (best_idx, abstain, depth_scores).
    """
    Z2 = Y3[:, :2]
    r = np.linalg.norm(Z2, axis=1) + 1e-9
    depth_raw = -r
    depth_raw = -np.linalg.norm(Y3[:, :2], axis=1) + 1e-9

    # global normalize (from calibration)
    gmu = getattr(pca, "_ngf_depth_mu", 0.0)
    gs  = getattr(pca, "_ngf_depth_sig", 1.0)
    depth_g = (depth_raw - gmu) / gs
    
    # local rescale to [0,1] to keep interpretability
    depth = (depth_g - depth_g.min()) / (depth_g.ptp() + 1e-9)

    pc3 = Y3[:, 2]
    pc3 = (pc3 - pc3.mean()) / (pc3.std() + 1e-9)
    depth = 0.85 * depth + 0.15 * (-np.abs(pc3))  # deeper if |PC3| small (converged)

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

def rerank_with_latent_wdd(model_name="gpt2", split="validation", n=100,
                           tap=-9, k_last=6, calib_n=256, detect_z=0.3,
                           denoise="off", debug=0, device="auto",
                           batch_size=16, max_length=160, amp_dtype="auto"):
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    # device + dtype
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if amp_dtype == "auto":
        amp_dtype = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else ("float16" if device=="cuda" else "float32")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    amp = dtype_map[amp_dtype]

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=amp if device=="cuda" else None).to(device).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Calibrate PCA and convert to torch params on device
    print(f"_calibrate_pca_basis ... ")
    pca = _calibrate_pca_basis(model, tok, split=split, tap=tap, k_last=k_last, calib_n=calib_n)

    print(f"_torch_pca_from_sklearn ... ")
    pcaT = _torch_pca_from_sklearn(pca, device=device, dtype=amp)

    ds = load_dataset("hellaswag", split=split)
    if n and n < len(ds):
        ds = ds.select(range(n))

    # Prebuild prompt+endings batches
    prompts, endings_list, labels = [], [], []
    for ex in ds:
        prefix = safe_prefix(ex.get("ctx",""), ex.get("ctx_a",""))
        prompt = (prefix + " The most plausible next event is:").strip()
        prompts.append(prompt)
        endings_list.append(list(ex["endings"]))
        labels.append(int(ex["label"]))
        #print(f"prompt: {prompt}")

    correct = 0; abstains = 0
    # Warmup (helps cuBLAS autotune)
    with torch.inference_mode():
        _ = model(**tok(["warmup"], return_tensors="pt").to(device), output_hidden_states=True)

    for i in range(0, len(prompts), batch_size):
        Ps = prompts[i:i+batch_size]
        Es = endings_list[i:i+batch_size]
        Y = _project_choices_batched(model, tok, pcaT, Ps, Es, tap=tap, k_last=k_last, max_length=max_length)  # [B,4,k]
        # Optional denoise: skip jitter loops for speed (keep 'off' as default)
        B = Y.shape[0]
        for b in range(B):
            pred, abstain, depth = _detect_kd_torch(Y[b], pcaT["gmu"], pcaT["gsig"], detect_z)
            gold = labels[i + b]
            correct += int(pred == gold); abstains += int(abstain)
            if debug and (i+b) < debug:
                dnp = depth.detach().cpu().numpy()
                print(f"[{i+b}] gold={gold} pred={pred} abstain={abstain} depth={np.round(dnp,3)}")
        done = min(i+batch_size, len(prompts))
        if done % 50 == 0 or done == len(prompts):
            print(f"{done} acc: {correct}/{done}={correct/done:.4f} (abstains={abstains})")

    total = len(prompts) or 1
    print("\n=== HellaSwag (sidecar/latent WDD, batched) ===")
    print(f"model={model_name} dev={device} dtype={amp_dtype} tap={tap} k_last={k_last} "
          f"calib_n={calib_n} batch={batch_size} max_len={max_length} "
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
    ap.add_argument("--batch_size", type=int, default=16, help="Number of prompts per batch (each has 4 endings).")
    ap.add_argument("--max_length", type=int, default=160, help="Token cap for speed.")
    ap.add_argument("--amp_dtype", choices=["auto","float16","bfloat16","float32"], default="auto",
                    help="Autocast/inference dtype for model/sidecar math.")
    
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
    ap.add_argument("--batch_size", type=int, default=16, help="Number of prompts per batch (each has 4 endings).")
    ap.add_argument("--max_length", type=int, default=160, help="Token cap for speed.")
    ap.add_argument("--amp_dtype", choices=["auto","float16","bfloat16","float32"], default="auto",
                    help="Autocast/inference dtype for model/sidecar math.")

    args = ap.parse_args()

    if args.mode == "hellaswag_latent_wdd":
        run_hellaswag_latent_wdd_cli(args)
    elif args.mode == "hellaswag":
        run_hellaswag(n=args.n, split=args.split, debug=args.debug)
    else:
        run_demos()
