# Write a super-simple inspector that farms latents for the first N items,
# projects to PCA-2, applies the toy "semantic well" depth (−||Y||),
# computes MAD z, and dumps everything to a folder with PNG plots + JSON + CSV.

import json, os, csv, math, numpy as np

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json, csv, math, numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None

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
    var_: Optional[np.ndarray] = None

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xc = X - self.mean
        Y = Xc @ self.components.T
        if self.whiten and self.var_ is not None:
            Y = Y / np.sqrt(np.maximum(self.var_, 1e-12))
        return Y

def fit_pca_all_endings(model, tok, ds, tap: int, k_last: int, max_len: int, calib_n: int, device: str, k: int = 3) -> PCABasis:
    n = min(calib_n, len(ds))
    ctxs  = [ds[i]["ctx"] for i in range(n)]
    ends  = [ds[i]["endings"] for i in range(n)]

    Xs = []
    with torch.no_grad():
        for i in range(n):
            for j in range(4):
                text = (ctxs[i].strip() + " " + ends[i][j].strip())
                enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
                out = model(**enc, output_hidden_states=True)
                hs = out.hidden_states[tap]  # [1, T, D]
                pooled = pool_lastk(hs, k_last)  # [1, D]
                Xs.append(pooled[0].cpu().numpy())
    X = np.stack(Xs, axis=0)  # [n*4, D]

    if PCA is None:
        raise RuntimeError("scikit-learn is required (pip install scikit-learn).")

    p = PCA(n_components=k, whiten=True, random_state=0).fit(X)
    return PCABasis(
        mean=p.mean_.astype(np.float32),
        components=p.components_.astype(np.float32),
        whiten=True,
        var_=p.explained_variance_.astype(np.float32),
    )

def harvest_item_latents(model, tok, ctx: str, endings: List[str], tap: int, k_last: int, max_len: int, device: str) -> np.ndarray:
    with torch.no_grad():
        vecs = []
        for j in range(4):
            text = (ctx.strip() + " " + endings[j].strip())
            enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states[tap]
            pooled = pool_lastk(hs, k_last)
            vecs.append(pooled[0].cpu().numpy())
    H = np.stack(vecs, axis=0)  # [4, D]
    return H

def mad_z(depth01: np.ndarray) -> float:
    m = float(np.median(depth01))
    mad = float(np.median(np.abs(depth01 - m))) + 1e-9
    z = (float(depth01.max()) - m) / (1.4826 * mad)
    return z

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
    ap.add_argument("--calib_n", type=int, default=2000)

    ap.add_argument("--inspect_n", type=int, default=5)
    ap.add_argument("--detect_z", type=float, default=0.7)

    ap.add_argument("--dump_dir", default="latent_inspect")
    args = ap.parse_args()

    device = set_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    ds_calib = load_dataset("hellaswag", split=args.calib_split)
    ds_eval  = load_dataset("hellaswag", split=args.split)
    n_eval = min(args.n, len(ds_eval))
    os.makedirs(args.dump_dir, exist_ok=True)

    print(f"Fitting PCA({args.pca_k}) on {args.calib_n} items × 4 endings ...")
    pca = fit_pca_all_endings(model, tok, ds_calib, args.tap, args.k_last, args.max_length, args.calib_n, device, k=args.pca_k)
    print("PCA fit complete. Inspecting items...")

    rows = []
    out_json = {"items": []}

    # plotting inline not required; we will save PNGs per item
    import matplotlib.pyplot as plt

    for i in range(min(args.inspect_n, n_eval)):
        ctx   = ds_eval[i]["ctx"]
        ends  = ds_eval[i]["endings"]
        gold  = int(ds_eval[i]["label"])

        H = harvest_item_latents(model, tok, ctx, ends, args.tap, args.k_last, args.max_length, device)  # [4,D]
        Y = pca.transform(H)  # [4, k]; use first 2 comps for the toy funnel
        Y2 = Y[:, :2]
        vec = -Y2  # inward pull
        depth_raw = -np.linalg.norm(vec, axis=1)  # more negative = deeper
        depth_pos = -depth_raw
        lo, hi = float(np.min(depth_pos)), float(np.max(depth_pos))
        depth01 = (depth_pos - lo) / max(1e-9, (hi - lo))
        z = mad_z(depth01)
        pick = int(np.argmax(depth01))
        abstain = bool(z < args.detect_z)

        # save JSON entry
        item = {
            "index": i,
            "gold": gold,
            "ctx": ctx,
            "endings": ends,
            "Y2": Y2.tolist(),  # PC1, PC2 per option
            "depth_raw": depth_raw.tolist(),
            "depth01": depth01.tolist(),
            "z": z,
            "pick": pick,
            "abstain": abstain
        }
        out_json["items"].append(item)

        # CSV row (flatten a bit)
        r = {
            "idx": i, "gold": gold, "pick": pick, "abstain": int(abstain), "z": z,
            "d0": depth01[0], "d1": depth01[1], "d2": depth01[2], "d3": depth01[3],
            "dr0": depth_raw[0], "dr1": depth_raw[1], "dr2": depth_raw[2], "dr3": depth_raw[3],
            "pc10": Y2[0,0], "pc20": Y2[0,1],
            "pc11": Y2[1,0], "pc21": Y2[1,1],
            "pc12": Y2[2,0], "pc22": Y2[2,1],
            "pc13": Y2[3,0], "pc23": Y2[3,1],
        }
        rows.append(r)

        # Make a simple plot: PC1 vs PC2 with labels 0..3 and mark picked option
        fig = plt.figure()
        xs, ys = Y2[:,0], Y2[:,1]
        plt.scatter(xs, ys)
        for j in range(4):
            txt = f"{j}"
            if j == pick: txt += "★"
            plt.annotate(txt, (xs[j], ys[j]))
        plt.axhline(0); plt.axvline(0)
        plt.title(f"Item {i} PC1–PC2; gold={gold} pick={pick} z={z:.3f} abstain={abstain}")
        png_path = os.path.join(args.dump_dir, f"item_{i:03d}.png")
        fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)

    # write JSON + CSV
    json_path = os.path.join(args.dump_dir, "inspect_items.json")
    with open(json_path, "w") as f:
        json.dump(out_json, f, indent=2)

    csv_path = os.path.join(args.dump_dir, "inspect_items.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows: w.writerow(r)

    print(f"Wrote: {json_path}")
    print(f"Wrote: {csv_path}")
    print(f"PNGs: {args.dump_dir}/item_###.png")

if __name__ == "__main__":
    main()



