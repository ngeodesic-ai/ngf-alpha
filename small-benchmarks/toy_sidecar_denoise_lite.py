
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
toy_sidecar_denoise_lite.py
A drop-in simple inspector for HellaSwag that:
- farms pooled latents for 4 endings,
- projects with PCA(k) (whiten=True),
- scores depth using k-D Mahalanobis to the per-item centroid,
- applies lite denoiser controls: winsorization + jitter averaging (null-like) + EMA,
- computes a null-style z from jittered margins,
- gates with --detect_z,
- saves 2D PC scatter PNGs + JSON + CSV, matching toy_sidecar output format.
"""
import argparse, os, json, csv, math
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None

# ---------------- utils ----------------

def set_device(name: str):
    if name == "auto":
        if torch.cuda.is_available(): return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
        return "cpu"
    return name

def pool_lastk(h: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, h.shape[1])
    return h[:, -k:, :].mean(1)

@dataclass
class PCABasis:
    mean: np.ndarray
    components: np.ndarray  # [k, D]
    whiten: bool
    var_: np.ndarray        # explained variances for k comps

def fit_pca_all_endings(model, tok, ds, tap: int, k_last: int, max_len: int, calib_n: int, device: str, k: int = 3) -> PCABasis:
    n = min(calib_n, len(ds))
    Xs = []
    with torch.no_grad():
        for i in range(n):
            ctx = ds[i]["ctx"]; ends = ds[i]["endings"]
            for j in range(4):
                text = (ctx.strip() + " " + ends[j].strip())
                enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
                out = model(**enc, output_hidden_states=True)
                hs  = out.hidden_states[tap]  # [1,T,D]
                pooled = pool_lastk(hs, k_last)  # [1,D]
                Xs.append(pooled[0].cpu().numpy())
    X = np.stack(Xs, 0).astype(np.float32)
    if PCA is None:
        raise RuntimeError("sklearn not available for PCA")
    pca = PCA(n_components=k, whiten=True, random_state=0).fit(X)
    basis = PCABasis(mean=pca.mean_.astype(np.float32),
                     components=pca.components_.astype(np.float32),
                     whiten=bool(pca.whiten),
                     var_=pca.explained_variance_.astype(np.float32))
    return basis

def pca_transform(basis: PCABasis, X: np.ndarray) -> np.ndarray:
    Xc = X - basis.mean[None, :]
    Y  = Xc @ basis.components.T
    # whiten=True already scales by sqrt(var), sklearn does this internally in transform;
    # since we manually apply components, emulate whiten by dividing by std if requested
    # (components_ are on raw; sklearn applies whitening inside transform; so recompute with PCA object ideally.
    # For simplicity here, we approximate by dividing by sqrt(var_).)
    if basis.whiten:
        Y = Y / np.sqrt(basis.var_[None, :] + 1e-8)
    return Y.astype(np.float32)

def harvest_item_latents(model, tok, tap: int, k_last: int, max_len: int, ctx: str, endings: List[str], device: str) -> np.ndarray:
    vecs = []
    with torch.no_grad():
        for j in range(4):
            text = (ctx.strip() + " " + endings[j].strip())
            enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            out = model(**enc, output_hidden_states=True)
            hs  = out.hidden_states[tap]  # [1,T,D]
            pooled = pool_lastk(hs, k_last)  # [1,D]
            vecs.append(pooled[0].cpu().numpy())
    H = np.stack(vecs, 0).astype(np.float32)  # [4,D]
    return H

def winsorize(x: np.ndarray, hi_q: float) -> np.ndarray:
    hi = np.quantile(x, hi_q)
    return np.minimum(x, hi)

def ema(prev: Optional[float], x: float, beta: float) -> float:
    return x if prev is None else (beta*prev + (1-beta)*x)

def null_jitter_margins(Yk: np.ndarray, J: int = 24, eps: float = 0.02, rng=None) -> np.ndarray:
    """Return distribution of margins (best-second) under small Gaussian jitter in k-D PCA space."""
    rng = rng or np.random.default_rng(20259)
    k = Yk.shape[1]
    margins = []
    for _ in range(int(J)):
        jitter = rng.normal(0.0, eps, size=Yk.shape).astype(np.float32)
        Yj = Yk + jitter
        c  = Yj.mean(0, keepdims=True)
        d  = np.linalg.norm(Yj - c, axis=1)  # since whitened, Euclidean ~ Mahalanobis
        order = np.argsort(d)  # smaller is better
        m = d[order[1]] - d[order[0]]
        margins.append(float(m))
    return np.array(margins, dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="hellaswag_latent_wdd")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=250)

    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tap", type=int, default=-9)
    ap.add_argument("--k_last", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=160)

    ap.add_argument("--pca_k", type=int, default=3)
    ap.add_argument("--calib_split", default="train")
    ap.add_argument("--calib_n", type=int, default=512)

    ap.add_argument("--inspect_n", type=int, default=5)
    ap.add_argument("--detect_z", type=float, default=0.7)

    ap.add_argument("--winsor_q", type=float, default=0.985)
    ap.add_argument("--jitter_J", type=int, default=32)
    ap.add_argument("--jitter_eps", type=float, default=0.02)
    ap.add_argument("--ema_beta", type=float, default=0.0, help="0 disables EMA on per-ending depths")

    ap.add_argument("--dump_dir", default="latent_inspect_denoise")
    args = ap.parse_args()

    device = set_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    ds = load_dataset("hellaswag", split=args.calib_split)
    ds_eval = load_dataset("hellaswag", split=args.split)
    n_eval = min(args.n, len(ds_eval))

    # --- PCA basis (calibration) ---
    basis = fit_pca_all_endings(model, tok, ds, args.tap, args.k_last, args.max_length, args.calib_n, device, k=args.pca_k)

    os.makedirs(args.dump_dir, exist_ok=True)
    out_json = {"items": []}
    rows = []

    # plotting
    import matplotlib.pyplot as plt

    # Precompute global winsor clip from calib radii distribution (k-D to per-item centroid not available here;
    # approximate with radii to global origin for calib samples in k-D).
    # We compute Y on calib X used in PCA fit.
    # (For simplicity, re-harvest a small subset.)
    # NOTE: Because PCA fitted with whiten=True, Euclidean in Y ~ Mahalanobis in X.
    rng = np.random.default_rng(20259)

    for i in range(min(args.inspect_n, n_eval)):
        ctx   = ds_eval[i]["ctx"]
        ends  = ds_eval[i]["endings"]
        label = int(ds_eval[i]["label"])

        H = harvest_item_latents(model, tok, args.tap, args.k_last, args.max_length, ctx, ends, device)  # [4,D]
        Y = pca_transform(basis, H)  # [4,k]
        Y2 = Y[:, :2].copy()

        # k-D centroid distance (smaller = better)
        c = Y.mean(0, keepdims=True)
        d_raw = np.linalg.norm(Y - c, axis=1)

        # lite denoiser controls: (1) winsorize, (2) EMA on per-ending depth across tiny jitters, (3) null jitter margins -> z
        # Winsor clip reference: use per-item d_raw for simplicity
        d_w = winsorize(d_raw, args.winsor_q)

        # Jitter averaging (stabilize): compute mean distance over J jitters per ending
        depths = []
        if args.ema_beta > 0.0:
            prev = [None]*4
        for j in range(4):
            dj = []
            for _ in range(int(args.jitter_J)):
                jitter = rng.normal(0.0, args.jitter_eps, size=Y.shape[1]).astype(np.float32)
                yy = Y[j] + jitter
                cc = c + rng.normal(0.0, args.jitter_eps, size=c.shape).astype(np.float32)
                dj.append(np.linalg.norm(yy - cc[0]))
            djm = float(np.mean(dj))
            if args.ema_beta > 0.0:
                prev[j] = ema(prev[j], djm, args.ema_beta)
                depths.append(prev[j])
            else:
                depths.append(djm)
        depth01 = np.array(depths, dtype=np.float32)

        # Null-style z: compare observed margin to jittered margins distribution
        margins_null = null_jitter_margins(Y, J=max(8, args.jitter_J), eps=args.jitter_eps, rng=rng)
        margin_obs = float(np.partition(depth01, 1)[1] - np.min(depth01))  # second-best minus best (since smaller is better)
        # robust z using MAD around null margins
        m = float(np.median(margins_null)); mad = float(np.median(np.abs(margins_null - m)) + 1e-9)
        z = (margin_obs - m) / (1.4826*mad + 1e-9)

        pick = int(np.argmin(depth01))
        abstain = bool(z < args.detect_z)

        # --- save plot (PC1/PC2) ---
        fig = plt.figure()
        xs, ys = Y2[:, 0], Y2[:, 1]
        plt.scatter(xs, ys)
        for j in range(4):
            txt = f"{j}"
            if j == pick: txt += "★"
            plt.annotate(txt, (xs[j], ys[j]))
        plt.title(f"item {i}: pick={pick} gold={label} z={z:.2f} {'ABSTAIN' if abstain else ''}")
        png_path = os.path.join(args.dump_dir, f"item_{i:03d}.png")
        fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)

        # pack outputs (similar fields to toy_sidecar)
        item = dict(
            i=i, gold=label, pick=pick, abstain=int(abstain), z=float(z),
            pc10=float(Y2[0,0]), pc11=float(Y2[0,1]),
            pc20=float(Y2[1,0]), pc21=float(Y2[1,1]),
            pc30=float(Y2[2,0]), pc31=float(Y2[2,1]),
            pc40=float(Y2[3,0]), pc41=float(Y2[3,1]),
            d0=float(depth01[0]), d1=float(depth01[1]), d2=float(depth01[2]), d3=float(depth01[3]),
            dr0=float(d_raw[0]), dr1=float(d_raw[1]), dr2=float(d_raw[2]), dr3=float(d_raw[3]),
        )
        out_json["items"].append(item)
        rows.append(item)

    json_path = os.path.join(args.dump_dir, "inspect_items.json")
    csv_path  = os.path.join(args.dump_dir, "inspect_items.csv")
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        for r in rows: w.writerow(r)

    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"PNGs:  {args.dump_dir}/item_###.png")

if __name__ == "__main__":
    main()
