#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage11_warp_run.py
-------------------
Step 2 of a two-pass pipeline.

• Loads model+tokenizer.
• Loads a previously saved warp_config.npz from stage11_warp_fit.py.
• Registers a forward pre-hook that applies a radial contraction in PCA(2).
• Generates completions for eval prompts and writes a JSONL (stock vs prewarp).

Usage:
  python3 stage11_warp_run.py \
    --model gpt2 \
    --warp_npz warp_config_tap9.npz \
    --eval eval_prompts.txt \
    --out_jsonl logs/prewarp_generations.jsonl \
    --dtype auto \
    --batch_size 6
"""

import argparse, os, sys, json, math
from pathlib import Path
from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----- Utils -----
def set_safe_mode():
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

def resolve_dtype(device, arg_dtype: str):
    if arg_dtype == "float32":
        return torch.float32
    if arg_dtype == "float16":
        return torch.float16
    if arg_dtype == "bfloat16":
        return torch.bfloat16
    # auto
    if device == "cuda" and torch.cuda.is_available():
        major = torch.cuda.get_device_capability()[0]
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32

class PCAProj:
    def __init__(self, mean: np.ndarray, components: np.ndarray, scales: np.ndarray):
        self.mean = mean.astype(np.float32, copy=True)
        self.components = components.astype(np.float32, copy=True)
        self.scales = np.maximum(scales.astype(np.float32, copy=True), 1e-12)
    def inverse_delta(self, dY: torch.Tensor) -> torch.Tensor:
        device = dY.device
        m = dY.shape[-1]
        scales = torch.from_numpy(self.scales[:m]).to(device=device, dtype=dY.dtype)  # (m,)
        compsT = torch.from_numpy(self.components[:m]).to(device=device, dtype=dY.dtype)  # (m, D)
        return (dY * scales) @ compsT  # (N, D)

class WarpHook:
    def __init__(self, model, layer_idx: int, pca: PCAProj, center_xy: np.ndarray, r0: float, alpha: float):
        self.model = model
        self.layer_idx = int(layer_idx)
        self.pca = pca
        self.center = torch.from_numpy(center_xy.astype(np.float32))
        self.r0 = float(max(1e-6, r0))
        self.alpha = float(alpha)
        self.handle = None
    def __enter__(self):
        def pre_hook(module, inputs):
            if not isinstance(inputs, tuple) or len(inputs) == 0: return inputs
            H = inputs[0]
            if not torch.is_tensor(H): return inputs
            B, T, D = H.shape
            dev, dt = H.device, H.dtype
            center = self.center.to(dev, dt)  # (2,)
            # project to PCA(2) whitened
            X = H.reshape(-1, D)
            compsT = torch.from_numpy(self.pca.components[:2, :]).to(dev, dt)  # (2, D)
            mean = torch.from_numpy(self.pca.mean).to(dev, dt)  # (D,)
            scales = torch.from_numpy(self.pca.scales[:2]).to(dev, dt)  # (2,)
            Y2 = ((X - mean) @ compsT.T) / scales  # (N,2)
            # radial contraction
            V = Y2 - center
            R = torch.linalg.norm(V, dim=-1, keepdim=True).clamp_min(1e-8)
            S = 1.0 - self.alpha * torch.exp(- (R / self.r0) ** 2)
            Y2p = center + S * V
            dY2 = Y2p - Y2
            dX = self.pca.inverse_delta(dY2)  # (N, D)
            Xp = X + dX
            Hp = Xp.reshape(B, T, D)
            return (Hp,) + inputs[1:]
        self.handle = self.model.transformer.h[self.layer_idx].register_forward_pre_hook(pre_hook, with_kwargs=False)
        return self
    def __exit__(self, exc_type, exc, tb):
        try:
            if self.handle is not None: self.handle.remove()
        finally:
            self.handle = None

def _tokenize(tok, texts: List[str], device):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True)
    return {k: v.to(device) for k, v in enc.items()}

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--warp_npz", type=str, required=True)
    ap.add_argument("--eval", type=str, required=True)
    ap.add_argument("--out_jsonl", type=str, default="prewarp_generations.jsonl")
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto","float32","float16","bfloat16"])
    ap.add_argument("--safe_mode", action="store_true")
    args = ap.parse_args()

    if args.safe_mode:
        set_safe_mode()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dtype = resolve_dtype(device, args.dtype)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model.to(run_dtype)
    except Exception:
        run_dtype = torch.float32

    # Load warp config
    data = np.load(args.warp_npz)
    mean = data["mean"]; components = data["components"]; scales = data["scales"]
    center_xy = data["center_xy"]; r0 = float(data["r0"][0]); alpha = float(data["alpha"][0])
    layer_idx = int(data["layer_idx"][0])
    pca = PCAProj(mean, components, scales)

    # Load eval prompts
    with open(args.eval, "r") as f:
        eval_prompts = [ln.strip() for ln in f if ln.strip()]

    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    amp_cast = (device == "cuda" and run_dtype != torch.float32)

    with outp.open("w", encoding="utf-8") as fw:
        for ptxt in eval_prompts:
            # stock
            enc = tok(ptxt, return_tensors="pt").to(device)
            with torch.inference_mode():
                if amp_cast:
                    with torch.autocast(device_type="cuda", dtype=run_dtype):
                        y0 = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                            do_sample=True, temperature=args.temperature, top_p=args.top_p,
                                            pad_token_id=tok.eos_token_id, use_cache=False)
                else:
                    y0 = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                        do_sample=True, temperature=args.temperature, top_p=args.top_p,
                                        pad_token_id=tok.eos_token_id, use_cache=False)
            stock = tok.decode(y0[0], skip_special_tokens=True)

            # pre-warp
            with WarpHook(model, layer_idx, pca, center_xy, r0, alpha):
                enc2 = tok(ptxt, return_tensors="pt").to(device)
                with torch.inference_mode():
                    if amp_cast:
                        with torch.autocast(device_type="cuda", dtype=run_dtype):
                            y1 = model.generate(**enc2, max_new_tokens=args.max_new_tokens,
                                                do_sample=True, temperature=args.temperature, top_p=args.top_p,
                                                pad_token_id=tok.eos_token_id, use_cache=False)
                    else:
                        y1 = model.generate(**enc2, max_new_tokens=args.max_new_tokens,
                                            do_sample=True, temperature=args.temperature, top_p=args.top_p,
                                            pad_token_id=tok.eos_token_id, use_cache=False)
            geo = tok.decode(y1[0], skip_special_tokens=True)

            rec = {"prompt": ptxt, "stock": stock, "geodesic_prewarp": geo}
            fw.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[WRITE] {str(outp)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr); sys.exit(1)
