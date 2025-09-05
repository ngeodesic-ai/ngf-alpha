#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage11_llm_prewarp_benchmark.py
- Learn a PCA subspace at a target tap from calibration prompts.
- Build a 2D density/energy map from eval prompts in that subspace.
- Choose the dominant minimum as warp center.
- Register a runtime hook at that tap that contracts points toward the center
  in the PCA plane (nonlinear radial warp), then inverse-transform back to model space.
- Generate text for eval prompts and save JSONL of prompt, output.
Notes:
- Works with GPT-2-like causal LMs from HuggingFace.
- 'tap' is negative indexing into hidden_states (e.g., -1 is last layer). The model hook is mapped accordingly.


python3 stage11_llm_prewarp_benchmark_rewrite.py \
  --model gpt2 \
  --tap -9 \
  --calib calib_prompts_v2_900.txt \
  --eval calib_eval_style_200.txt \
  --safe_mode \
  --refine_passes 1 \
  --batch_size 6 \
  --nbins 96 \
  --max_new_tokens 48 \
  --temperature 0.7 --top_p 0.95 \
  --out_jsonl logs/prewarp_generations.jsonl

"""
import argparse, os, json, math
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter


torch.set_grad_enabled(False)
torch.set_num_threads(1)                 # tame threading on macOS/Accelerate
try:
    torch.backends.mkldnn.enabled = False
except Exception:
    pass

def collect_hidden_states(model, tok, prompts, tap: int, pool="lastk", k_last=6):
    with torch.no_grad():
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        hs  = out.hidden_states[tap]  # (B,T,D)
        if pool == "lastk":
            k = min(k_last, hs.shape[1])
            H = hs[:, -k:, :].mean(1)
        else:
            H = hs.mean(1)
        return H.detach().cpu().numpy().astype(float)

def pca3(H):
    p = PCA(n_components=3, whiten=True, random_state=0)
    return p, p.fit_transform(H)

def energy_from_eval(Y2, nbins=120, sigma=3.5):
    x, y = Y2[:,0], Y2[:,1]
    H, xe, ye = np.histogram2d(x, y, bins=nbins)
    Hs = gaussian_filter(H, sigma=sigma)
    U  = -Hs
    return U, Hs, xe, ye

def register_layer_warp_pre_hook(model, tap: int, pca, center_xy, r0: float, alpha: float):
    """
    Safer for CPU/mac: pre-hook warps the *input* hidden states of GPT-2 block.
    Inputs shape: (hidden_states, layer_past, attention_mask, ...)
    We modify only the first (hidden_states) tensor and pass through the rest.
    """
    n_layers = len(model.transformer.h)
    layer_idx = max(0, min(n_layers - 1, n_layers + tap))  # e.g., -9 -> index 3 for GPT-2 small

    # Precompute PCA bits as torch tensors (float32)
    W_np   = pca.components_[:2, :].astype("float32")   # (2, D)
    mean   = pca.mean_.astype("float32")                # (D,)
    b_np   = (-W_np @ mean).astype("float32")           # (2,)
    c_np   = np.asarray(center_xy, dtype="float32")     # (2,)

    W_t  = torch.from_numpy(W_np)                       # (2,D)
    WT_t = W_t.transpose(0, 1).contiguous()             # (D,2)
    b_t  = torch.from_numpy(b_np)                       # (2,)
    c_t  = torch.from_numpy(c_np)                       # (2,)
    r0_f = float(max(1e-6, r0))
    a_f  = float(alpha)

    def pre_hook(module, inputs):
        """
        inputs is a tuple: (hidden_states, layer_past, attention_mask, ...).
        We must return a *tuple* of same length/types.
        """
        if not isinstance(inputs, tuple) or len(inputs) == 0:
            return inputs  # nothing to do

        H = inputs[0]  # [B,T,D] hidden states IN to this block
        if not torch.is_tensor(H):
            return inputs

        B, T, D = H.shape
        dev, dt = H.device, H.dtype

        W  = W_t.to(device=dev, dtype=dt)
        WT = WT_t.to(device=dev, dtype=dt)
        b  = b_t.to(device=dev, dtype=dt)
        c  = c_t.to(device=dev, dtype=dt)

        X  = H.reshape(-1, D)           # (N,D)
        Y2 = X @ WT + b                 # (N,2)
        V  = Y2 - c
        R  = torch.linalg.norm(V, dim=-1, keepdim=True) + 1e-8
        S  = 1.0 - a_f * torch.exp(- (R / r0_f) ** 2)
        Y2p = c + S * V
        dY2 = Y2p - Y2
        Xp  = X + dY2 @ W               # (N,D)
        Hp  = Xp.reshape(B, T, D)

        # Return a tuple of inputs with hidden_states replaced
        new_inputs = (Hp,) + inputs[1:]
        return new_inputs

    handle = model.transformer.h[layer_idx].register_forward_pre_hook(pre_hook, with_kwargs=False)
    return handle, layer_idx



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--tap", type=int, default=-9)
    ap.add_argument("--calib", type=str, required=True, help="one prompt per line")
    ap.add_argument("--eval", type=str, required=True, help="one prompt per line for generation")
    ap.add_argument("--pool_mode", type=str, default="lastk", choices=["lastk","mean"])
    ap.add_argument("--k_last", type=int, default=8)
    ap.add_argument("--nbins", type=int, default=120)
    ap.add_argument("--sigma_px", type=float, default=5.0)
    ap.add_argument("--density_floor", type=float, default=4.0)
    ap.add_argument("--min_prom", type=float, default=0.55)
    ap.add_argument("--alpha", type=float, default=0.6, help="warp strength (0..1.5)")
    ap.add_argument("--r0_frac", type=float, default=0.35, help="r0 as fraction of PCA radius (0..1)")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--out_jsonl", type=str, default="prewarp_generations.jsonl")
    ap.add_argument("--safe_mode", action="store_true",
                help="CPU/mac-safe path: pre-hook warp, no cache, 1 thread, eager attention")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok   = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    if args.safe_mode:
        torch.set_grad_enabled(False)
        torch.set_num_threads(1)
        model.config.use_cache = False
        try:
            model.config.attn_implementation = "eager"
        except Exception:
            pass
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    with open(args.calib) as f: calib_prompts = [ln.strip() for ln in f if ln.strip()]
    with open(args.eval)  as f: eval_prompts  = [ln.strip() for ln in f if ln.strip()]

    Hc = collect_hidden_states(model, tok, calib_prompts, args.tap, args.pool_mode, args.k_last)
    pca, Yc = pca3(Hc)
    r_max = float(np.linalg.norm(Yc[:, :2], axis=1).max() + 1e-8)

    He = collect_hidden_states(model, tok, eval_prompts, args.tap, args.pool_mode, args.k_last)
    Ye = pca.transform(He)
    # Build energy map
    x, y = Ye[:,0], Ye[:,1]
    H, xe, ye = np.histogram2d(x, y, bins=args.nbins)
    Hs = gaussian_filter(H, sigma=args.sigma_px)
    U  = -Hs

    # Find dominant minimum (with gating), else fallback to global minimum
    center_idx = None
    h, w = U.shape
    best_val = np.inf
    for i in range(1, h-1):
        for j in range(1, w-1):
            if Hs[i, j] < args.density_floor:
                continue
            c = U[i,j]
            neigh = U[i-1:i+2, j-1:j+2].copy()
            neigh[1,1] = np.nan
            prom = np.nanmean(neigh) - c
            if prom >= args.min_prom and np.all(c < np.nan_to_num(neigh, nan=np.inf)):
                if c < best_val:
                    best_val = c
                    center_idx = (i,j)
    if center_idx is None:
        center_idx = np.unravel_index(np.argmin(U), U.shape)

    xc = (xe[:-1] + xe[1:]) / 2.0
    yc = (ye[:-1] + ye[1:]) / 2.0
    center_xy = np.array([xc[center_idx[1]], yc[center_idx[0]]], dtype=np.float32)

    r0 = max(1e-6, args.r0_frac * r_max)
    handle, layer_idx = register_layer_warp_pre_hook(model, args.tap, pca, center_xy, r0, args.alpha)
    print(f"[HOOK] Pre-warp active at layer index {layer_idx} for tap {args.tap}; center={center_xy.tolist()}, r0={r0:.4f}, alpha={args.alpha:.3f}")

    out_path = Path(args.out_jsonl)
    with out_path.open("w") as f:
        for p in eval_prompts:
            # A) Stock
            enc = tok(p, return_tensors="pt").to(device)
            with torch.no_grad():
                y_stock = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens,
                    do_sample=True, temperature=args.temperature, top_p=args.top_p,
                    pad_token_id=tok.eos_token_id,
                    use_cache=False
                )
            txt_stock = tok.decode(y_stock[0], skip_special_tokens=True)
            # B) Geodesic (pre-warp)
            enc2 = tok(p, return_tensors="pt").to(device)
            with torch.no_grad():
                y_geo = model.generate(
                    **enc2, max_new_tokens=args.max_new_tokens,
                    do_sample=True, temperature=args.temperature, top_p=args.top_p,
                    pad_token_id=tok.eos_token_id,
                    use_cache=False
                )
            txt_geo = tok.decode(y_geo[0], skip_special_tokens=True)
            f.write(json.dumps({"prompt": p, "stock": txt_stock, "geodesic_prewarp": txt_geo}) + "\n")
    handle.remove()
    print(f"[WRITE] {str(out_path)}")

if __name__ == "__main__":
    main()
