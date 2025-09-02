#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage11_warp_fit.py
-------------------
Step 1 of a two-pass pipeline.

• Loads model+tokenizer.
• Collects hidden-states at the chosen tap from calibration + eval prompts.
• Fits PCA (k=3, whitened), builds a smoothed density in PCA(2),
  selects a dominant center, sets r0 by spread and chosen fraction.
• Writes an NPZ warp package (mean, components, scales, center_xy, r0, alpha, layer_idx, meta)
  plus a small JSON manifest for human inspection.

Usage:
  python3 stage11_warp_fit.py \
    --model gpt2 \
    --tap -9 \
    --calib calib_prompts.txt \
    --eval eval_prompts.txt \
    --alpha 0.60 \
    --r0_frac 0.35 \
    --out_npz warp_config_tap9.npz \
    --manifest warp_manifest.json
"""

import argparse, os, json, math, sys
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------- Utils ----------
def set_safe_mode():
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

def map_tap(n_layers: int, tap: int) -> int:
    return max(0, min(n_layers-1, tap if tap >= 0 else n_layers + tap))

def _tokenize(tok, texts: List[str], device):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True)
    return {k: v.to(device) for k, v in enc.items()}

class PCAProj:
    def __init__(self, mean: np.ndarray, components: np.ndarray, scales: np.ndarray):
        self.mean = mean.astype(np.float32, copy=True)
        self.components = components.astype(np.float32, copy=True)
        self.scales = np.maximum(scales.astype(np.float32, copy=True), 1e-12)
    @staticmethod
    def fit(H: np.ndarray, k: int = 3, whiten: bool = True):
        H = H.astype(np.float32, copy=False)
        mean = H.mean(axis=0, dtype=np.float64).astype(np.float32)
        X = H - mean
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        comps = Vt[:k, :]
        denom = max(1, (H.shape[0]-1))
        eig_sqrt = S[:k] / math.sqrt(denom)
        scales = eig_sqrt if whiten else np.ones_like(eig_sqrt, dtype=np.float32)
        proj = PCAProj(mean, comps, scales)
        Y = proj.transform(H)
        return proj, Y
    def transform(self, H: np.ndarray) -> np.ndarray:
        X = (H - self.mean).astype(np.float32, copy=False)
        Y = X @ self.components.T
        Y /= self.scales
        return Y

def gaussian_kernel2d(sigma: float, truncate: float = 3.0):
    sigma = max(1e-3, float(sigma))
    radius = int(truncate * sigma + 0.5)
    size = 2*radius + 1
    x = torch.arange(-radius, radius+1, dtype=torch.float32)
    xx, yy = torch.meshgrid(x, x, indexing="ij")
    g = torch.exp(-(xx*xx + yy*yy) / (2.0 * sigma * sigma))
    g /= g.sum().clamp_min(1e-12)
    return g

def smooth2d(hist2d: np.ndarray, sigma_px: float) -> np.ndarray:
    h = torch.from_numpy(hist2d.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    k = gaussian_kernel2d(sigma_px).unsqueeze(0).unsqueeze(0)
    pad = k.shape[-1] // 2
    h_pad = F.pad(h, (pad, pad, pad, pad), mode="replicate")
    out = F.conv2d(h_pad, k)
    return out.squeeze(0).squeeze(0).numpy()

def collect_hidden(model, tok, prompts: List[str], tap: int, pool="lastk", k_last=8, device="cpu", batch_size=8):
    assert pool in ("lastk", "mean")
    outs = []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i+batch_size]
            if not chunk: continue
            enc = _tokenize(tok, chunk, device)
            out = model(**enc, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states[tap]  # (B,T,D)
            if pool == "lastk":
                k = min(k_last, hs.shape[1])
                pooled = hs[:, -k:, :].mean(dim=1)
            else:
                pooled = hs.mean(dim=1)
            outs.append(pooled.detach().cpu().to(torch.float32))
            del enc, out, hs, pooled
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    H = torch.cat(outs, dim=0).numpy()
    return H

def energy_map(Y2: np.ndarray, nbins=96, sigma_px=5.0):
    x, y = Y2[:,0], Y2[:,1]
    H, xe, ye = np.histogram2d(x, y, bins=nbins)
    Hs = smooth2d(H, sigma_px)
    U = -Hs
    xc = 0.5*(xe[:-1] + xe[1:])
    yc = 0.5*(ye[:-1] + ye[1:])
    return U, Hs, xc, yc

def pick_center(U, Hs, xc, yc, density_floor=4.0, min_prom=0.55):
    h, w = U.shape
    best, best_val = None, np.inf
    for i in range(1, h-1):
        for j in range(1, w-1):
            if Hs[i,j] < density_floor: continue
            c = U[i,j]
            neigh = U[i-1:i+2, j-1:j+2].copy()
            neigh[1,1] = np.nan
            prom = np.nanmean(neigh) - c
            if prom >= min_prom and np.all(c < np.nan_to_num(neigh, nan=np.inf)):
                if c < best_val:
                    best_val, best = c, (i,j)
    if best is None:
        best = np.unravel_index(np.argmin(U), U.shape)
    i, j = best
    center = np.array([xc[j], yc[i]], dtype=np.float32)
    return center

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--tap", type=int, default=-9)
    ap.add_argument("--calib", type=str, required=True)
    ap.add_argument("--eval", type=str, required=True)
    ap.add_argument("--pool_mode", type=str, default="lastk", choices=["lastk","mean"])
    ap.add_argument("--k_last", type=int, default=8)
    ap.add_argument("--nbins", type=int, default=96)
    ap.add_argument("--sigma_px", type=float, default=5.0)
    ap.add_argument("--density_floor", type=float, default=4.0)
    ap.add_argument("--min_prom", type=float, default=0.55)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--r0_frac", type=float, default=0.35)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--out_npz", type=str, default="warp_config.npz")
    ap.add_argument("--manifest", type=str, default="warp_manifest.json")
    ap.add_argument("--safe_mode", action="store_true")
    args = ap.parse_args()

    if args.safe_mode:
        set_safe_mode()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load prompts
    with open(args.calib, "r") as f:
        calib_prompts = [ln.strip() for ln in f if ln.strip()]
    with open(args.eval, "r") as f:
        eval_prompts = [ln.strip() for ln in f if ln.strip()]
    if not calib_prompts or not eval_prompts:
        print("[ERR] Empty calib/eval.", file=sys.stderr); sys.exit(2)

    # Hidden-states
    tap_idx = args.tap
    Hc = collect_hidden(model, tok, calib_prompts, tap_idx, args.pool_mode, args.k_last, device, args.batch_size)
    pca, Yc = PCAProj.fit(Hc, k=3, whiten=True)
    He = collect_hidden(model, tok, eval_prompts, tap_idx, args.pool_mode, args.k_last, device, args.batch_size)
    Ye = pca.transform(He)

    # Energy & center
    U, Hs, xc, yc = energy_map(Ye[:, :2], nbins=args.nbins, sigma_px=args.sigma_px)
    center_xy = pick_center(U, Hs, xc, yc, args.density_floor, args.min_prom)

    # Scale r0 by spread
    r_max = float(np.linalg.norm(Yc[:, :2], axis=1).max() + 1e-8)
    r0 = max(1e-6, args.r0_frac * r_max)

    # Save NPZ package
    n_layers = len(model.transformer.h)
    layer_idx = map_tap(n_layers, args.tap)
    np.savez(
        args.out_npz,
        mean=pca.mean,
        components=pca.components,
        scales=pca.scales,
        center_xy=center_xy,
        r0=np.array([r0], dtype=np.float32),
        alpha=np.array([args.alpha], dtype=np.float32),
        layer_idx=np.array([layer_idx], dtype=np.int32),
        meta=np.array([0], dtype=np.int32),
    )

    # Manifest JSON
    manifest = {
        "model": args.model,
        "tap": args.tap,
        "layer_idx": int(layer_idx),
        "alpha": float(args.alpha),
        "r0": float(r0),
        "pool_mode": args.pool_mode,
        "k_last": int(args.k_last),
        "nbins": int(args.nbins),
        "sigma_px": float(args.sigma_px),
        "density_floor": float(args.density_floor),
        "min_prom": float(args.min_prom),
        "calib_count": len(calib_prompts),
        "eval_count": len(eval_prompts),
        "notes": "Use stage11_warp_run.py with this NPZ to register the warp hook."
    }
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[WRITE] NPZ: {args.out_npz}")
    print(f"[WRITE] Manifest: {args.manifest}")
    print(f"[INFO] center_xy={center_xy.tolist()} r0={r0:.4f} alpha={args.alpha:.3f} layer_idx={layer_idx}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {type(e).__name__}: {e}", file=sys.stderr); sys.exit(1)
