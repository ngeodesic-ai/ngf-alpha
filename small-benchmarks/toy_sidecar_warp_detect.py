#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toy sidecar: Warp + Detect (no denoise)
--------------------------------------
- Farms pooled latents for 4 endings (HellaSwag)
- Global PCA(k>=3) with whitening (cached)
- Builds a *global* Stage‑11–style single‑well warp from calibration data
  (2D isotropization + radial funnel depth profile)
- Scores each ending by 3D depth to the warped well center (centroid distance in warped coords)
- Detect gate via null-style z using jittered null margins
- Saves PNGs + JSON + CSV + summary for the first N items (inspect_n)

New in this version:
- `--ending_only 1` pools **only the ending tokens** (not ctx) when forming each latent.
- `--taps -9,-10` allows **multi-tap** harvesting. Per-tap PCA+warp are fit and scoring
  is **ensembled** across taps.
- `--ensemble {mean_depth,vote,mean_dist}` controls how taps are combined (default: mean_depth).
  * mean_depth → average warped 3D distances to centroid across taps (default existing behavior generalized)
  * mean_dist  → identical to mean_depth (alias kept for clarity)
  * vote       → pick-per-tap and majority vote; detect uses aggregated z (conservative)

CLI example (first 5 items):
    python3 toy_sidecar_warp_detect.py \
      --model gpt2 --tap -9 --k_last 16 --pca_k 3 \
      --calib_split train --calib_n 512 \
      --split validation --n 250 --inspect_n 5 \
      --detect_z 0.7 --dump_dir latent_inspect_warp_detect
"""

import argparse, os, json, csv, hashlib, json as _json
from dataclasses import dataclass
from typing import List, Optional, Tuple
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
                   pca_k: int, calib_split: str, calib_n: int, ending_only: int) -> str:
    payload = _json.dumps(
        dict(model=model_name, tap=tap, k_last=k_last, max_len=max_len,
             pca_k=pca_k, calib_split=calib_split, calib_n=calib_n,
             ending_only=int(ending_only)),
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


def fit_pca_all_endings(model, tok, ds, tap: int, k_last: int, max_len: int, calib_n: int, device: str, k: int = 3, ending_only: int = 0) -> Tuple[PCABasis, np.ndarray]:
    """Fit PCA(k) on pooled latents from calibration split; also returns the matrix X used."""
    n = min(calib_n, len(ds))
    Xs = []
    with torch.no_grad():
        for i in range(n):
            ctx = ds[i]["ctx"]; ends = ds[i]["endings"]
            # pre-encode ctx if using ending_only to find the junction index
            enc_ctx = tok(ctx, return_tensors="pt", truncation=True, max_length=max_len).to(device) if ending_only else None
            for j in range(4):
                text = (ctx.strip() + " " + ends[j].strip())
                enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
                out = model(**enc, output_hidden_states=True)
                hs  = out.hidden_states[tap]  # [1, T, D]
                if ending_only and enc_ctx is not None:
                    n_ctx = enc_ctx["input_ids"].shape[1]
                    end_slice = hs[:, n_ctx:, :]
                    if end_slice.shape[1] == 0:
                        pooled = pool_lastk(hs, k_last)
                    else:
                        k_use = min(k_last, end_slice.shape[1])
                        pooled = end_slice[:, -k_use:, :].mean(1)
                else:
                    pooled = pool_lastk(hs, k_last)
                Xs.append(pooled[0].cpu().numpy())
    X = np.stack(Xs, 0).astype(np.float32)
    if PCA is None:
        raise RuntimeError("sklearn not available for PCA")
    pca = PCA(n_components=k, whiten=True, random_state=0).fit(X)
    basis = PCABasis(mean=pca.mean_.astype(np.float32),
                     components=pca.components_.astype(np.float32),
                     whiten=bool(pca.whiten),
                     var_=pca.explained_variance_.astype(np.float32))
    return basis, X


def pca_transform(basis: PCABasis, X: np.ndarray) -> np.ndarray:
    Xc = X - basis.mean[None, :]
    Y  = Xc @ basis.components.T
    if basis.whiten:
        Y = Y / np.sqrt(basis.var_[None, :] + 1e-8)
    return Y.astype(np.float32)


def harvest_item_latents(model, tok, tap: int, k_last: int, max_len: int, ctx: str, endings: List[str], device: str, ending_only: int = 0) -> np.ndarray:
    vecs = []
    with torch.no_grad():
        enc_ctx = tok(ctx, return_tensors="pt", truncation=True, max_length=max_len).to(device) if ending_only else None
        for j in range(4):
            text = (ctx.strip() + " " + endings[j].strip())
            enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            out = model(**enc, output_hidden_states=True)
            hs  = out.hidden_states[tap]  # [1, T, D]
            if ending_only and enc_ctx is not None:
                n_ctx = enc_ctx["input_ids"].shape[1]
                end_slice = hs[:, n_ctx:, :]
                if end_slice.shape[1] == 0:
                    pooled = pool_lastk(hs, k_last)
                else:
                    k_use = min(k_last, end_slice.shape[1])
                    pooled = end_slice[:, -k_use:, :].mean(1)
            else:
                pooled = pool_lastk(hs, k_last)
            vecs.append(pooled[0].cpu().numpy())
    H = np.stack(vecs, 0).astype(np.float32)
    return H


def winsorize(x: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
    lo = np.quantile(x, lo_q); hi = np.quantile(x, hi_q)
    return np.clip(x, lo, hi)

# ---------------- Stage‑11 style warp (global), then detect ----------------

@dataclass
class WarpParams:
    center2: np.ndarray          # (2,) center in PCA2
    iso_mu: np.ndarray           # (2,) mean used for isotropization
    iso_T: np.ndarray            # (2,2) whitening transform
    sigma: float                 # radial scale for funnel core
    depth_scale: float           # scales funnel depth (z)
    mix_z: float                 # mixes original PC3 as residual depth


def isotropize_fit(X2: np.ndarray):
    mu = X2.mean(axis=0)
    Y = X2 - mu
    C = (Y.T @ Y) / max(len(Y)-1, 1)
    evals, evecs = np.linalg.eigh(C)
    T = evecs @ np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-8))) @ evecs.T
    return mu, T


def build_global_warp(basis: PCABasis, calib_X: np.ndarray, sigma_scale=0.80, depth_scale=1.35, mix_z=0.12) -> WarpParams:
    """Use calibration latents to set global center, isotropization and funnel scales."""
    Y = pca_transform(basis, calib_X)  # [N,k]
    if Y.shape[1] < 3:
        raise ValueError("pca_k must be >= 3 for warp")
    X2 = Y[:, :2]
    Z3 = Y[:, 2]
    center2 = X2.mean(axis=0)
    iso_mu, iso_T = isotropize_fit(X2 - center2)
    X2_iso = (X2 - center2 - iso_mu) @ iso_T
    r = np.linalg.norm(X2_iso, axis=1)
    sigma = float(np.median(r) * sigma_scale + 1e-9)
    return WarpParams(center2=center2.astype(np.float32), iso_mu=iso_mu.astype(np.float32), iso_T=iso_T.astype(np.float32), sigma=sigma, depth_scale=float(depth_scale), mix_z=float(mix_z))


def funnel_depth(X2_iso: np.ndarray, z3: np.ndarray, wp: WarpParams) -> np.ndarray:
    """Stage‑11 radial funnel depth profile + PC3 residual (negative is deeper)."""
    r = np.linalg.norm(X2_iso, axis=1) + 1e-9
    z_funnel = -np.exp(-(r**2) / (2 * wp.sigma**2))  # [-1,0]
    z_new = wp.depth_scale * z_funnel + wp.mix_z * (z3 - np.mean(z3))
    return z_new.astype(np.float32)


def depth_vector(Yrow: np.ndarray, wp: WarpParams, gamma: float = 1.0) -> np.ndarray:
    """Map one PCA row → 3D position (x',y',z') in warped well coords for distance scoring."""
    x2 = Yrow[:2]
    z3 = Yrow[2] if len(Yrow) >= 3 else 0.0
    X2_iso = (x2 - wp.center2 - wp.iso_mu) @ wp.iso_T
    z = funnel_depth(X2_iso[None, :], np.array([z3], np.float32), wp)[0]
    # Distance in warped coords to well center at (0,0, z_at_r=0)
    # We scale z by gamma so user can tune z influence on depth.
    return np.array([X2_iso[0], X2_iso[1], gamma * (-z)], dtype=np.float32)


# ---------------- null margin (jitter) for Detect ----------------

def null_jitter_margins_depth(Y: np.ndarray, wp: WarpParams, J: int = 24, eps: float = 0.02, gamma: float = 1.0, rng=None) -> np.ndarray:
    """Distribution of (second - best) margins under small jitter, using *warped depth* distances."""
    rng = rng or np.random.default_rng(20259)
    margins = []
    for _ in range(int(J)):
        jitter = rng.normal(0.0, eps, size=Y.shape).astype(np.float32)
        Yj = Y + jitter
        P = np.stack([depth_vector(Yj[i], wp, gamma) for i in range(Yj.shape[0])], axis=0)
        c = P.mean(0, keepdims=True)
        d = np.linalg.norm(P - c, axis=1)
        order = np.argsort(d)
        margins.append(float(d[order[1]] - d[order[0]]))
    return np.array(margins, dtype=np.float32)


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()

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

    # Warp knobs (Stage‑11 inspired)
    ap.add_argument("--sigma_scale", type=float, default=0.80)
    ap.add_argument("--depth_scale", type=float, default=1.35)
    ap.add_argument("--mix_z", type=float, default=0.12)
    ap.add_argument("--z_gamma", type=float, default=1.0, help="Influence of funnel depth on distance")

    # Jitter null for Detect
    ap.add_argument("--jitter_J", type=int, default=32)
    ap.add_argument("--jitter_eps", type=float, default=0.02)

    # New: harvesting/ensembling controls
    ap.add_argument("--ending_only", type=int, default=0, help="If 1, pool only ending tokens (not ctx)")
    ap.add_argument("--taps", type=str, default=None, help="Comma-separated list of taps (e.g. -9,-10)")
    ap.add_argument("--ensemble", type=str, default="mean_depth", choices=["mean_depth", "mean_dist", "vote"],
                    help="How to combine per-tap scores")
    ap.add_argument("--probe_dump", type=str, default="",
                    help="If set, saves raw pooled latents + labels to NPZ for probing.")
    ap.add_argument("--probe_items", type=int, default=1000,
                    help="How many eval items to dump for the probe.")

    ap.add_argument("--dump_dir", default="latent_inspect_warp_detect")
    args = ap.parse_args()

    if args.pca_k < 3:
        raise SystemExit("--pca_k must be >=3 for warp-detect")

    device = set_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()

    ds_calib = load_dataset("hellaswag", split=args.calib_split)
    ds_eval  = load_dataset("hellaswag", split=args.split)
    n_eval = min(args.n, len(ds_eval))

    probe_X, probe_y_gold, probe_item_id, probe_end_id = [], [], [], []

    # --- taps handling ---
    taps = [args.tap] if not args.taps else [int(t.strip()) for t in args.taps.split(",") if t.strip()]

    # --- PCA + warp per tap (cached) ---
    bases = {}
    warps = {}
    for t in taps:
        cache_dir = Path(args.pca_cache_dir)
        cache_key = _pca_cache_key(args.model, t, args.k_last, args.max_length,
                                   args.pca_k, args.calib_split, args.calib_n, args.ending_only)
        cache_file = cache_dir / f"pca_{cache_key}.npz"

        basis = None
        if not args.force_recompute_pca:
            basis = load_pca_basis(cache_file)

        calib_X = None
        if basis is None:
            print(f"[PCA] tap {t}: cache miss → fitting (k={args.pca_k}, calib_n={args.calib_n}, ending_only={args.ending_only}) …")
            basis, calib_X = fit_pca_all_endings(model, tok, ds_calib,
                                                 t, args.k_last, args.max_length,
                                                 args.calib_n, device, k=args.pca_k, ending_only=args.ending_only)
            save_pca_basis(cache_file, basis)
            print(f"[PCA] tap {t}: saved → {cache_file}")
        else:
            print(f"[PCA] tap {t}: cache hit → {cache_file}")
            # Re-harvest small subset for warp fit if we didn't just compute X
            m = min(args.calib_n, 256)
            Xs = []
            with torch.no_grad():
                for i in range(m):
                    ctx = ds_calib[i]["ctx"]; ends = ds_calib[i]["endings"]
                    enc_ctx = tok(ctx, return_tensors="pt", truncation=True, max_length=args.max_length).to(device) if args.ending_only else None
                    for j in range(4):
                        text = (ctx.strip() + " " + ends[j].strip())
                        enc = tok(text, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
                        out = model(**enc, output_hidden_states=True)
                        hs  = out.hidden_states[t]
                        if args.ending_only and enc_ctx is not None:
                            n_ctx = enc_ctx["input_ids"].shape[1]
                            end_slice = hs[:, n_ctx:, :]
                            if end_slice.shape[1] == 0:
                                pooled = pool_lastk(hs, args.k_last)
                            else:
                                k_use = min(args.k_last, end_slice.shape[1])
                                pooled = end_slice[:, -k_use:, :].mean(1)
                        else:
                            pooled = pool_lastk(hs, args.k_last)
                        Xs.append(pooled[0].cpu().numpy())
            calib_X = np.stack(Xs, 0).astype(np.float32)

        bases[t] = basis
        warps[t] = build_global_warp(basis, calib_X, sigma_scale=args.sigma_scale,
                                     depth_scale=args.depth_scale, mix_z=args.mix_z)

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

    for i in range(min(args.inspect_n, n_eval)):
        ctx   = ds_eval[i]["ctx"]
        ends  = ds_eval[i]["endings"]
        label = int(ds_eval[i]["label"])

        # Per-tap scoring artifacts
        per_tap_d = []   # list of [4] arrays of distances
        per_tap_pick = []
        per_tap_Y = {}
        per_tap_wp = {}
        per_tap_P = {}
        margins_all = []

        for t in taps:
            H = harvest_item_latents(model, tok, t, args.k_last, args.max_length, ctx, ends, device, ending_only=args.ending_only)  # [4,D]
            Y = pca_transform(bases[t], H)  # [4,k]
            per_tap_Y[t] = Y
            wp = warps[t]
            per_tap_wp[t] = wp

            # 3D warped positions per ending
            P = np.stack([depth_vector(Y[j], wp, gamma=args.z_gamma) for j in range(4)], axis=0)  # [4,3]
            per_tap_P[t] = P

            # Distance to per-item centroid in warped coordinates (smaller = better)
            c = P.mean(0, keepdims=True)
            d_raw = np.linalg.norm(P - c, axis=1)
            per_tap_d.append(d_raw)
            order_w = np.argsort(d_raw)
            per_tap_pick.append(int(order_w[0]))

            # Null margins for this tap
            margins_null_t = null_jitter_margins_depth(Y, wp, J=max(8, args.jitter_J), eps=args.jitter_eps, gamma=args.z_gamma, rng=rng)
            margins_all.append(margins_null_t)

        per_tap_d = np.stack(per_tap_d, axis=0)  # [T,4]

        # Ensemble combine
        if args.ensemble in ("mean_depth", "mean_dist"):
            d_raw_avg = per_tap_d.mean(axis=0)
            order = np.argsort(d_raw_avg)
            pick = int(order[0])
            margin_obs = float(d_raw_avg[order[1]] - d_raw_avg[order[0]])
        elif args.ensemble == "vote":
            # majority vote; tie-break by mean distance
            votes = np.bincount(np.array(per_tap_pick), minlength=4)
            top = np.flatnonzero(votes == votes.max())
            if len(top) == 1:
                pick = int(top[0])
            else:
                d_mean = per_tap_d.mean(axis=0)
                pick = int(np.argsort(d_mean)[0])
            # compute margin on mean distances for detect
            d_raw_avg = per_tap_d.mean(axis=0)
            order = np.argsort(d_raw_avg)
            margin_obs = float(d_raw_avg[order[1]] - d_raw_avg[order[0]])
        else:
            raise ValueError(f"Unknown ensemble mode: {args.ensemble}")


        # ---- PROBE DUMP (raw pooled latents) ----
        if args.probe_dump:
            m = min(args.probe_items, len(ds_eval))
            with torch.no_grad():
                for i in range(m):
                    ctx   = ds_eval[i]["ctx"]
                    ends  = ds_eval[i]["endings"]
                    gold  = int(ds_eval[i]["label"])
                    H = harvest_item_latents(model, tok, taps[0], args.k_last, args.max_length,
                                             ctx, ends, device, ending_only=args.ending_only)  # [4, D]
                    for j in range(4):
                        probe_X.append(H[j])
                        probe_y_gold.append(1 if j == gold else 0)
                        probe_item_id.append(i)
                        probe_end_id.append(j)
            np.savez_compressed(
                args.probe_dump,
                X=np.asarray(probe_X, dtype=np.float32),          # [N, D]
                y=np.asarray(probe_y_gold, dtype=np.int8),        # [N]
                item=np.asarray(probe_item_id, dtype=np.int32),   # [N]
                end=np.asarray(probe_end_id, dtype=np.int8),      # [N]
                meta=np.array([dict(
                    model=args.model, tap=taps[0], k_last=args.k_last,
                    ending_only=args.ending_only, pca_k=args.pca_k,
                )], dtype=object)
            )
            print(f"[probe] saved → {args.probe_dump}")


        # Aggregate null margins across taps (conservative):
        # concatenate all tap-wise jitter margins to form a pooled null
        margins_null = np.concatenate(margins_all, axis=0)
        m = float(np.median(margins_null)); mad = float(np.median(np.abs(margins_null - m)) + 1e-9)
        z = (margin_obs - m) / (1.4826*mad + 1e-9)

        abstain = bool(z < args.detect_z)

        # --- plot (PCA2 for view) using the first tap ---
        Yviz = per_tap_Y[taps[0]]
        Y2 = Yviz[:, :2]
        xs, ys = Y2[:, 0], Y2[:, 1]
        fig = plt.figure()
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
            d0=float(d_raw_avg[0]), d1=float(d_raw_avg[1]), d2=float(d_raw_avg[2]), d3=float(d_raw_avg[3]),
            pc10=float(Y2[0,0]), pc11=float(Y2[0,1]),
            pc20=float(Y2[1,0]), pc21=float(Y2[1,1]),
            pc30=float(Y2[2,0]), pc31=float(Y2[2,1]),
            pc40=float(Y2[3,0]), pc41=float(Y2[3,1]),
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
        "sigma_scale": float(args.sigma_scale),
        "depth_scale": float(args.depth_scale),
        "mix_z": float(args.mix_z),
        "z_gamma": float(args.z_gamma),
        "ending_only": int(args.ending_only),
        "taps": taps,
        "ensemble": args.ensemble,
    }
    sum_path = os.path.join(args.dump_dir, "summary.json")
    with open(sum_path, "w") as f:
        _json.dump(summary, f, indent=2)
    print("[summary]", _json.dumps(summary, indent=2))
    print(f"Saved summary → {sum_path}")


if __name__ == "__main__":
    main()
