#!/usr/bin/env python3
# mock_sidecar.py — Latent WDD sidecar with Random Projection (no PCA calibration)

import os, argparse, time
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from datasets import load_dataset

"""
python3 mock_sidecar_v2.py --mode hellaswag_latent_wdd --split validation --n 250 \
  --proj pca --pca_k 8 --calib_split train --calib_n 512 \ 
  --pca_cache .cache/pca_gpt2_tap-9_k8.pkl --svd_solver randomized \
  --tap -9 --k_last 8 --batch_size 32 --max_length 160 \
  --detect_z 0.7 --device cpu
"""

# Avoid HF tokenizers fork warning
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------
# Helpers
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
# Mock (toy) scorer for demos only
# ---------------------------
@dataclass
class Score:
    depth: float
    margin: float
    conf: float
    gates_ok: bool

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
    has_one_decimal = 1 if any(re.match(r"\b-?\d+\.\d\b", n) for n in nums) else 0
    has_two_decimals = 1 if any(re.match(r"\b-?\d+\.\d\d\b", n) for n in nums) else 0
    has_three_decimals = 1 if any(re.match(r"\b-?\d+\.\d\d\d\b", n) for n in nums) else 0
    v = np.array([lowers, uppers, digits, spaces, dots, commas, other_punct, length,
                  has_num, num_len_avg, has_one_decimal, has_two_decimals, has_three_decimals],
                 dtype=float)
    v = (v - v.mean()) / (v.std() + 1e-6)
    return v

class MockScorer:
    def score_batch(self, prompt: str, candidates: List[str]) -> List[Score]:
        mats = np.stack([simple_text_vec(prompt + " " + c) for c in candidates], axis=0)
        X = mats - mats.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        Z = X @ Vt.T[:, :2]
        r = np.linalg.norm(Z, axis=1)
        depth = ( -r - (-r).min() ) / ( (-r).ptp() + 1e-6 )
        order = np.argsort(-depth)
        margin = np.zeros_like(depth)
        if len(depth) > 1:
            margin[order[0]] = max(0.0, depth[order[0]] - depth[order[1]])
        conf = 1 / (1 + np.exp(-5*(depth - depth.mean())))
        gate = depth > (depth.mean() - 0.25*depth.std())
        return [Score(float(depth[i]), float(margin[i]), float(conf[i]), bool(gate[i]))
                for i in range(len(candidates))]

# ---------------------------
# Demo runner (unchanged)
# ---------------------------
def run_demos():
    sidecar = MockScorer()
    prompt1 = "Round 7.432 to one decimal."
    cands1  = ["7.4", "7.43", "7.5", "7.40", "8"]
    s1 = sidecar.score_batch(prompt1, cands1)
    best1 = int(np.argmax([s.depth for s in s1]))
    print("=== Demo 1 ===")
    print("Best:", cands1[best1])

    prompt2 = "A man is riding a skateboard down a ramp. He loses his balance. The most plausible next event is:"
    cands2  = [
        "He falls off and lands on the ground.",
        "A cat jumps onto the table.",
        "The man starts typing on a keyboard.",
        "The skateboard floats into space."
    ]
    s2 = sidecar.score_batch(prompt2, cands2)
    best2 = int(np.argmax([s.depth for s in s2]))
    print("=== Demo 2 ===")
    print("Best:", cands2[best2])

def run_hellaswag_mock(n: int, split: str, debug: int):
    ds = load_dataset("hellaswag", split=split)
    if n and n < len(ds):
        ds = ds.select(range(n))
    scorer = MockScorer()
    correct, abstains = 0, 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        prefix = safe_prefix(ex.get("ctx",""), ex.get("ctx_a",""))
        prompt = (prefix + " The most plausible next event is:").strip()
        endings = list(ex["endings"]); gold = int(ex["label"])
        scores = scorer.score_batch(prompt, endings)
        pred = int(np.argmax([s.depth for s in scores]))
        correct += int(pred == gold)
        abstains += int(not scores[pred].gates_ok)
        if debug and i < debug:
            best_line = endings[pred][:64].replace("\n"," ")
            print(f"[{i}] gold={gold} pred={pred} abstain={not scores[pred].gates_ok} best: {best_line}")
        if (i+1) % 10 == 0 or (i+1) == len(ds):
            print(f"{i+1} acc: {correct}/{i+1}={correct/(i+1):.4f} (abstains={abstains})")
    print("\n=== HellaSwag (mock) ===")
    print(f"split={split} n={len(ds)} acc_top1={correct/len(ds):.4f} abstain_rate={abstains/len(ds):.4f} elapsed={time.time()-t0:.1f}s")

# --- PCA calibrator with caching ---
def _calibrate_pca_basis(model, tok, tap: int, k_last: int,
                         calib_split: str, calib_n: int,
                         n_components: int, max_length: int,
                         pca_cache: str = "", svd_solver: str = "randomized"):
    """
    Fit PCA(k, whiten=True) on pooled tap latents from calib_split.
    Caches to pca_cache if provided.
    """
    import os, pickle, numpy as np, torch
    from datasets import load_dataset
    from sklearn.decomposition import PCA

    if pca_cache and os.path.exists(pca_cache):
        with open(pca_cache, "rb") as f:
            pca = pickle.load(f)
        return pca

    ds = load_dataset("hellaswag", split=calib_split)
    m = min(calib_n, len(ds))
    # build short prefixes (consistent with runner)
    prefixes = []
    for i in range(m):
        ex = ds[i]
        ctx  = (ex.get("ctx","") or "").strip()
        ctxa = (ex.get("ctx_a","") or "").strip()
        if ctxa and not ctx.endswith(ctxa):
            pref = ctxa if (ctx and ctx in ctxa) else (ctx + (" " if ctx and ctxa else "") + ctxa)
        else:
            pref = ctx
        prefixes.append((pref + " The most plausible next event is:").strip())

    # batched latent harvest
    H_all = []
    bs = 64
    with torch.inference_mode():
        for i0 in range(0, m, bs):
            chunk = prefixes[i0:i0+bs]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_length).to(model.device)
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states[tap]             # [B,T,D]
            k = min(k_last, hs.shape[1])
            H = hs[:, -k:, :].mean(1).cpu().numpy() # [B,D]
            H_all.append(H.astype(np.float32))
    H = np.concatenate(H_all, axis=0)               # [M,D]

    pca = PCA(n_components=n_components, whiten=True, random_state=0,
              svd_solver=svd_solver).fit(H)

    # global depth stats (kD radial) for Detect’s z-norm
    Yk = pca.transform(H)                            # [M,k]
    r  = np.linalg.norm(Yk, axis=1) + 1e-9
    depth_raw = -r
    pca._ngf_depth_mu  = float(depth_raw.mean())
    pca._ngf_depth_sig = float(depth_raw.std() + 1e-9)

    if pca_cache:
        os.makedirs(os.path.dirname(pca_cache), exist_ok=True)
        with open(pca_cache, "wb") as f:
            pickle.dump(pca, f)
    return pca

# --- load sklearn PCA into torch tensors for fast transforms ---
def _torch_pca_from_sklearn(pca, device, dtype):
    import torch, numpy as np
    W = torch.tensor(pca.components_, device=device, dtype=dtype)            # [k,D]
    mu = torch.tensor(pca.mean_, device=device, dtype=dtype)                 # [D]
    var = torch.tensor(pca.explained_variance_, device=device, dtype=dtype)  # [k]
    S = torch.sqrt(var + 1e-9)
    gmu  = torch.tensor(getattr(pca, "_ngf_depth_mu", 0.0), device=device, dtype=dtype)
    gsig = torch.tensor(getattr(pca, "_ngf_depth_sig", 1e-9), device=device, dtype=dtype)
    return {"W": W, "mu": mu, "S": S, "gmu": gmu, "gsig": gsig}

# --- batched projection using PCA on device ---
def _project_choices_batched_pca(model, tok, pcaT, prompts, endings_list, tap, k_last, max_length):
    import torch
    texts = []
    for p, ends in zip(prompts, endings_list):
        texts.extend([(p + " " + e).strip() for e in ends])  # B*4
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=max_length).to(model.device)
    with torch.inference_mode():
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[tap]                    # [B*4,T,D]
        k = min(k_last, hs.shape[1])
        H = hs[:, -k:, :].mean(1).to(dtype=pcaT["W"].dtype)
        Xc = H - pcaT["mu"]
        Y = (Xc @ pcaT["W"].t()) / pcaT["S"]          # [B*4,k]
    B = len(prompts)
    return Y.view(B, 4, -1)


# ---------------------------
# Latent WDD (Random Projection path)
# ---------------------------
def _init_random_projector(hidden_dim: int, k: int, device, dtype, seed: int = 123, rp_cache: str = ""):
    """
    Make an orthonormal random projector R: [D, k] using QR. Cache if requested.
    """
    import torch, numpy as np
    if rp_cache and os.path.exists(rp_cache):
        data = np.load(rp_cache)
        R = torch.tensor(data["R"], device=device, dtype=dtype)  # [D,k]
        return {"R": R}
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(hidden_dim, k, generator=g, dtype=torch.float32)  # CPU for QR stability
    Q, _ = torch.linalg.qr(A, mode="reduced")  # [D,k]
    R = Q.to(device=device, dtype=dtype)
    if rp_cache:
        os.makedirs(os.path.dirname(rp_cache), exist_ok=True)
        np.savez(rp_cache, R=R.detach().cpu().numpy())
    return {"R": R}

def _project_choices_batched_rp(model, tok, rpT, prompts, endings_list, tap, k_last, max_length):
    """
    One forward for B×4 texts; returns torch.Tensor [B,4,k].
    """
    import torch
    texts = []
    for p, ends in zip(prompts, endings_list):
        texts.extend([(p + " " + e).strip() for e in ends])  # B*4
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=max_length).to(model.device)
    with torch.inference_mode():
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[tap]                    # [B*4, T, D]
        k = min(k_last, hs.shape[1])
        H = hs[:, -k:, :].mean(1)                     # [B*4, D]
        Y = H @ rpT["R"]                               # [B*4, k]
    B = len(prompts)
    return Y.view(B, 4, -1)

def _detect_kd_torch(Yk_row, detect_z: float, gmu=None, gsig=None):
    """
    Yk_row: [4,k] torch.
    If gmu/gsig provided (PCA), apply global z-norm; else (RP) skip it.
    Depth = -||Y||, locally rescaled to [0,1], MAD gate.
    """
    import torch
    r = torch.linalg.vector_norm(Yk_row, dim=1) + 1e-9
    depth_raw = -r
    if (gmu is not None) and (gsig is not None):
        depth_raw = (depth_raw - gmu) / gsig
    # local [0,1]
    dmin = depth_raw.min()
    rng  = torch.clamp(depth_raw.max() - dmin, min=1e-9)
    depth = (depth_raw - dmin) / rng
    # robust MAD gate
    med = depth.median()
    mad = (depth - med).abs().median() + 1e-9
    sigma = 1.4826 * mad
    gate = med + detect_z * sigma
    ok = depth > gate
    idx_ok = torch.nonzero(ok, as_tuple=False).flatten()
    if idx_ok.numel() == 0:
        return int(depth.argmax().item()), True, depth
    best = int(depth[idx_ok].argmax().item())
    return int(idx_ok[best].item()), False, depth


def rerank_with_latent_wdd_rp(args, model_name="gpt2", split="validation", n=100,
                              tap=-9, k_last=8, detect_z=0.6, rp_k=8, rp_seed=123,
                              rp_cache="", debug=0, device="auto",
                              batch_size=16, max_length=160, amp_dtype="auto"):
    """
    External sidecar: Random Projection (no PCA calibration) + kD well + MAD gate.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    use_pca = True

    # Pick device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Pick dtype
    if amp_dtype == "auto":
        if device == "cuda":
            amp_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        elif device == "mps":
            amp_dtype = "float16"
        else:
            amp_dtype = "float32"
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    amp = dtype_map[amp_dtype]

    tok = AutoTokenizer.from_pretrained(model_name)
    model_kwargs = {}
    if device in ("cuda","mps") and amp in (torch.float16, torch.bfloat16):
        model_kwargs["torch_dtype"] = amp
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Warmup (helps CUDA/MPS autotune) and get hidden dim
    with torch.inference_mode():
        wout = model(**tok(["_warmup_"], return_tensors="pt").to(device), output_hidden_states=True)
        hidden_dim = wout.hidden_states[tap].shape[-1]

    if use_pca:
        print("_calibrate_pca_basis ... ")
        pca = _calibrate_pca_basis(model, tok,
                                   tap=tap, k_last=k_last,
                                   calib_split=args.calib_split,
                                   calib_n=args.calib_n,
                                   n_components=args.pca_k,
                                   max_length=max_length,
                                   pca_cache=args.pca_cache,
                                   svd_solver=args.svd_solver)
        print("_torch_pca_from_sklearn ... ")
        P = _torch_pca_from_sklearn(pca, device=device, dtype=amp)
    else:
        # RP path you already have:
        with torch.inference_mode():
            wout = model(**tok(["_warmup_"], return_tensors="pt").to(device), output_hidden_states=True)
            hidden_dim = wout.hidden_states[tap].shape[-1]
        RP = _init_random_projector(hidden_dim, k=args.rp_k, device=device, dtype=amp,
                                    seed=args.rp_seed, rp_cache=args.rp_cache)

    # Dataset slice
    ds = load_dataset("hellaswag", split=split)
    if n and n < len(ds): ds = ds.select(range(n))

    prompts, endings_list, labels = [], [], []
    for ex in ds:
        prefix = safe_prefix(ex.get("ctx",""), ex.get("ctx_a",""))
        prompts.append((prefix + " The most plausible next event is:").strip())
        endings_list.append(list(ex["endings"]))
        labels.append(int(ex["label"]))


    abstains = 0
    correct = 0
    # ... batching loop:
    for i in range(0, len(prompts), batch_size):
        Ps = prompts[i:i+batch_size]
        Es = endings_list[i:i+batch_size]
        if use_pca:
            Y = _project_choices_batched_pca(model, tok, P, Ps, Es, tap=tap, k_last=k_last, max_length=max_length)
        else:
            Y = _projec
        B = Y.shape[0]
        for b in range(B):
            if use_pca:
                pred, abstain, depth = _detect_kd_torch(Y[b], detect_z, P["gmu"], P["gsig"])
            else:
                pred, abstain, depth = _detect_kd_torch(Y[b], detect_z)
            gold = labels[i+b]
            correct += int(pred == gold)
            abstains += int(abstain)
            if debug and (i+b) < debug:
                dnp = depth.detach().cpu().numpy()
                best_line = Es[b][pred][:64].replace("\n"," ")
                print(f"[{i+b}] gold={gold} pred={pred} abstain={abstain} depth={np.round(dnp,3)} best: {best_line}")

        done = min(i+batch_size, len(prompts))
        if done % 50 == 0 or done == len(prompts):
            print(f"{done} acc: {correct}/{done}={correct/done:.4f} (abstains={abstains})")

    total = len(prompts) or 1
    print("\n=== HellaSwag (sidecar/latent WDD • RP) ===")
    print(f"model={model_name} dev={device} dtype={amp_dtype} tap={tap} k_last={k_last} "
          f"rp_k={rp_k} seed={rp_seed} batch={batch_size} max_len={max_length} "
          f"split={split} n={total} acc_top1={correct/total:.4f} abstain_rate={abstains/total:.4f}")

# ---------------------------
# HellaSwag entrypoints
# ---------------------------
def run_hellaswag_latent_wdd_cli(args):
    return rerank_with_latent_wdd_rp(args,
        model_name=args.model, split=args.split, n=args.n,
        tap=args.tap, k_last=args.k_last, detect_z=args.detect_z,
        rp_k=args.rp_k, rp_seed=args.rp_seed, rp_cache=args.rp_cache,
        debug=args.debug, device=args.device,
        batch_size=args.batch_size, max_length=args.max_length, amp_dtype=args.amp_dtype
    )

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["demos","hellaswag","hellaswag_latent_wdd"], default="hellaswag",
                    help="Toy demos, toy HellaSwag, or real-latent WDD sidecar (RP).")
    ap.add_argument("--split", default="validation", choices=["train","validation","test"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--debug", type=int, default=0)

    # Latent sidecar knobs (RP path)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", choices=["auto","cpu","cuda","mps"], default="auto")
    ap.add_argument("--amp_dtype", choices=["auto","float16","bfloat16","float32"], default="auto")
    ap.add_argument("--tap", type=int, default=-9)
    ap.add_argument("--k_last", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=160)
    ap.add_argument("--detect_z", type=float, default=0.6)

    ap.add_argument("--rp_k", type=int, default=8, help="Random-projection output dim k.")
    ap.add_argument("--rp_seed", type=int, default=123, help="Seed for RP matrix.")
    ap.add_argument("--rp_cache", default="", help="Optional path to save/load RP matrix (npz).")
    ap.add_argument("--proj", choices=["pca","rp"], default="pca",
                    help="Projection method: pca (calibrated, whitened) or rp (random, no calibration).")
    ap.add_argument("--pca_k", type=int, default=8, help="PCA output dim k.")
    ap.add_argument("--calib_n", type=int, default=1024, help="Calibration sample size for PCA.")
    ap.add_argument("--calib_split", default="train", choices=["train","validation","test"],
                    help="Split used to fit PCA once.")
    ap.add_argument("--pca_cache", default="", help="Optional path to save/load PCA (pickle).")
    ap.add_argument("--svd_solver", choices=["auto","full","arpack","randomized"], default="randomized",
                    help="sklearn PCA solver; 'randomized' is fast for small k.")

    args = ap.parse_args()

    if args.mode == "hellaswag_latent_wdd":
        run_hellaswag_latent_wdd_cli(args)
    elif args.mode == "hellaswag":
        run_hellaswag_mock(n=args.n, split=args.split, debug=args.debug)
    else:
        run_demos()

if __name__ == "__main__":
    main()
