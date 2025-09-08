#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
toy_sidecar_denoise_lite_v2.py
A denoiser-wired variant of the toy sidecar inspector.

- Farms pooled latents for 4 endings (HellaSwag),
- PCA(k) with whitening (cached to disk),
- Scores by k-D distance to per-item centroid (smaller = better),
- Stage-11-inspired SoftDenoiser to stabilize per-ending depths:
    * EMA smoothing
    * confidence-gated margin shrink (sigmoid on null margin)
    * phantom guard for isolated winners under weak evidence
- Gates with a null-style z (observed margin vs jittered null margins)
- Saves PNGs + JSON + CSV, plus a run summary (coverage, acc_on_accepted, z stats)
"""

import argparse, os, json, csv, hashlib, json as _json
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

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

def _pca_cache_key(model_name: str, tap: int, k_last: int, max_len: int,
                   pca_k: int, calib_split: str, calib_n: int) -> str:
    payload = _json.dumps(
        dict(model=model_name, tap=tap, k_last=k_last, max_len=max_len,
             pca_k=pca_k, calib_split=calib_split, calib_n=calib_n),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()

def save_pca_basis(path: Path, basis: PCABasis) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(path),
        mean=basis.mean,
        components=basis.components,
        whiten=np.array([basis.whiten], dtype=np.int8),
        var_=basis.var_,
    )

def load_pca_basis(path: Path) -> Optional[PCABasis]:
    if not path.exists():
        return None
    data = np.load(str(path))
    return PCABasis(
        mean=data["mean"].astype(np.float32),
        components=data["components"].astype(np.float32),
        whiten=bool(int(data["whiten"][0])),
        var_=data["var_"].astype(np.float32),
    )

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
    # emulate sklearn whitening during transform
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

def winsorize(x: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
    lo = np.quantile(x, lo_q); hi = np.quantile(x, hi_q)
    return np.clip(x, lo, hi)

# ---------------- Stage-11–style SoftDenoiser (adapted) ----------------

class SoftDenoiserLite:
    """
    EMA + confidence gating + phantom guard for 4 scalar depths.
    """
    def __init__(self, beta=0.6, k=8.0, tau=0.35,
                 phantom_guard_gamma=0.35, evidence_floor=0.15):
        self.beta=float(beta)
        self.k=float(k)
        self.tau=float(tau)
        self.phantom_guard_gamma=float(phantom_guard_gamma)
        self.evidence_floor=float(evidence_floor)
        self._ema=None

    @staticmethod
    def _sigmoid(x): return 1.0/(1.0+np.exp(-x))

    def reset(self):
        self._ema=None

    def step(self, depths: np.ndarray, z_conf: float) -> np.ndarray:
        """
        depths: array of 4 positive distances (smaller = better)
        z_conf: null-style z for the item
        """
        x = np.asarray(depths, dtype=np.float32)
        if self._ema is None:
            self._ema = x.copy()
        else:
            self._ema = self.beta*self._ema + (1.0-self.beta)*x
        d = self._ema.copy()

        order = np.argsort(d)  # best first
        margin = float(d[order[1]] - d[order[0]])
        score  = float(self._sigmoid(self.k*(z_conf - self.tau)))  # 0..1
        shrink = (1.0 - score)
        d[order[0]] += 0.5 * shrink * margin   # soften winner under low evidence

        med = float(np.median(d))
        mad = float(np.median(np.abs(d - med)) + 1e-9)
        z_winner = (med - d[order[0]]) / (1.4826*mad)
        if (z_conf < self.evidence_floor) and (z_winner > 2.0):
            d[order[0]] = med - self.phantom_guard_gamma*(med - d[order[0]])

        return d

# ---------------- null margin (jitter) ----------------

def null_jitter_margins(Yk: np.ndarray, J: int = 24, eps: float = 0.02, rng=None) -> np.ndarray:
    """Distribution of (second - best) margin under small Gaussian jitter in k-D PCA space."""
    rng = rng or np.random.default_rng(20259)
    margins = []
    for _ in range(int(J)):
        jitter = rng.normal(0.0, eps, size=Yk.shape).astype(np.float32)
        Yj = Yk + jitter
        c  = Yj.mean(0, keepdims=True)
        d  = np.linalg.norm(Yj - c, axis=1)   # whitened → Euclidean ~ Mahalanobis
        order = np.argsort(d)
        margins.append(float(d[order[1]] - d[order[0]]))
    return np.array(margins, dtype=np.float32)

# ---------------- main ----------------

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

    # PCA cache
    ap.add_argument("--pca_cache_dir", default=".pca_cache")
    ap.add_argument("--force_recompute_pca", type=int, default=0)

    ap.add_argument("--inspect_n", type=int, default=5)
    ap.add_argument("--detect_z", type=float, default=0.7)

    # denoiser-lite knobs
    ap.add_argument("--winsor_lo", type=float, default=0.00)
    ap.add_argument("--winsor_hi", type=float, default=0.985)
    ap.add_argument("--jitter_J", type=int, default=32)
    ap.add_argument("--jitter_eps", type=float, default=0.02)
    ap.add_argument("--ema_beta", type=float, default=0.6)
    ap.add_argument("--denoise_k", type=float, default=8.0)
    ap.add_argument("--denoise_tau", type=float, default=0.35)
    ap.add_argument("--phantom_guard_gamma", type=float, default=0.35)
    ap.add_argument("--evidence_floor", type=float, default=0.15)

    ap.add_argument("--dump_dir", default="latent_inspect_denoise_v2")
    args = ap.parse_args()

    device = set_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    ds = load_dataset("hellaswag", split=args.calib_split)
    ds_eval = load_dataset("hellaswag", split=args.split)
    n_eval = min(args.n, len(ds_eval))

    # --- PCA basis with cache ---
    cache_dir = Path(args.pca_cache_dir)
    cache_key = _pca_cache_key(args.model, args.tap, args.k_last, args.max_length,
                               args.pca_k, args.calib_split, args.calib_n)
    cache_file = cache_dir / f"pca_{cache_key}.npz"

    basis = None
    if not args.force_recompute_pca:
        basis = load_pca_basis(cache_file)

    if basis is None:
        print(f"[PCA] cache miss → fitting (k={args.pca_k}, calib_n={args.calib_n}) …")
        basis = fit_pca_all_endings(model, tok, ds,
                                    args.tap, args.k_last, args.max_length,
                                    args.calib_n, device, k=args.pca_k)
        save_pca_basis(cache_file, basis)
        print(f"[PCA] saved → {cache_file}")
    else:
        print(f"[PCA] cache hit → {cache_file}")

    os.makedirs(args.dump_dir, exist_ok=True)
    out_json = {"items": []}
    rows = []

    # scoring harness counters
    n_items = 0
    n_abstain = 0
    n_accept = 0
    n_correct_on_accept = 0
    z_values = []

    import matplotlib.pyplot as plt
    rng = np.random.default_rng(20259)

    denoiser = SoftDenoiserLite(beta=args.ema_beta, k=args.denoise_k, tau=args.denoise_tau,
                                phantom_guard_gamma=args.phantom_guard_gamma,
                                evidence_floor=args.evidence_floor)

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

        # winsorize per-item distances (mainly for logging; we score on jitter-average below)
        d_w = winsorize(d_raw, args.winsor_lo, args.winsor_hi)

        # build null margins distribution via jitter
        margins_null = null_jitter_margins(Y, J=max(8, args.jitter_J), eps=args.jitter_eps, rng=rng)
        order_w = np.argsort(d_w)
        margin_obs = float(d_w[order_w[1]] - d_w[order_w[0]])
        # robust z using MAD
        m = float(np.median(margins_null)); mad = float(np.median(np.abs(margins_null - m)) + 1e-9)
        z = (margin_obs - m) / (1.4826*mad + 1e-9)

        # jitter averaging per-ending (stabilize) + denoiser
        depths = []
        for j in range(4):
            dj = []
            for _ in range(int(args.jitter_J)):
                jitter = rng.normal(0.0, args.jitter_eps, size=Y.shape[1]).astype(np.float32)
                yy = Y[j] + jitter
                cc = c + rng.normal(0.0, args.jitter_eps, size=c.shape).astype(np.float32)
                dj.append(np.linalg.norm(yy - cc[0]))
            depths.append(float(np.mean(dj)))
        depth01 = np.array(depths, dtype=np.float32)

        denoiser.reset()
        depth_stab = denoiser.step(depth01, z_conf=float(z))

        pick = int(np.argmin(depth_stab))
        abstain = bool(z < args.detect_z)

        # --- plot ---
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

        item = dict(
            i=i, gold=label, pick=pick, abstain=int(abstain), z=float(z),
            pc10=float(Y2[0,0]), pc11=float(Y2[0,1]),
            pc20=float(Y2[1,0]), pc21=float(Y2[1,1]),
            pc30=float(Y2[2,0]), pc31=float(Y2[2,1]),
            pc40=float(Y2[3,0]), pc41=float(Y2[3,1]),
            d0=float(depth_stab[0]), d1=float(depth_stab[1]), d2=float(depth_stab[2]), d3=float(depth_stab[3]),
            dr0=float(d_raw[0]), dr1=float(d_raw[1]), dr2=float(d_raw[2]), dr3=float(d_raw[3]),
        )
        out_json["items"].append(item)
        rows.append(item)

        # scoring harness updates
        n_items += 1
        z_values.append(float(z))
        if abstain:
            n_abstain += 1
        else:
            n_accept += 1
            if pick == label:
                n_correct_on_accept += 1

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

    # --- scoring harness summary ---
    import numpy as _np, json as _json
    cov = 1 - (n_abstain / max(1, n_items))
    acc_on_accept = (n_correct_on_accept / max(1, n_accept))
    summary = {
        "items": int(n_items),
        "accepted": int(n_accept),
        "abstained": int(n_abstain),
        "coverage": float(cov),
        "acc_on_accepted": float(acc_on_accept),
        "z_mean": float(_np.mean(_np.array(z_values))) if z_values else 0.0,
        "z_median": float(_np.median(_np.array(z_values))) if z_values else 0.0,
        "detect_z": float(args.detect_z),
        "pca_k": int(args.pca_k),
        "tap": int(args.tap),
        "k_last": int(args.k_last),
    }
    sum_path = os.path.join(args.dump_dir, "summary.json")
    with open(sum_path, "w") as f:
        _json.dump(summary, f, indent=2)
    print("[summary]", _json.dumps(summary, indent=2))
    print(f"Saved summary → {sum_path}")

if __name__ == "__main__":
    main()
