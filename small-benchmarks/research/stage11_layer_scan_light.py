#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage‑11 Layer Scan (Light) — batch‑safe for modest CPUs

Key differences vs. stage11_layer_scan.py
- Streams prompts in minibatches (configurable --batch_size)
- Caps tokenized length (--max_len) to keep tensors small
- Uses IncrementalPCA for calibration fit to avoid huge matrices
- Avoids keeping all hidden states in memory at once
- Uses inference_mode() and torch.set_grad_enabled(False)
- Optional thread pinning to reduce CPU thrash

Example:
  python3 stage11_layer_scan_light.py \
    --model gpt2 \
    --calib calib_prompts_v2_900.txt \
    --eval  calib_eval_style_200.txt \
    --batch_size 6 --max_len 192 \
    --pool_mode lastk --k_last 6 \
    --sigma_px 3.5 --density_floor 2.0 --min_prom 0.35 \
    --nbins 100 \
    --out_csv layer_scan_gpt2_light.csv
"""

import argparse, json, os, math, csv
import numpy as np

def _lazy_imports():
    global torch, AutoTokenizer, AutoModelForCausalLM, IncrementalPCA, PCA, gaussian_filter
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sklearn.decomposition import IncrementalPCA, PCA
    from scipy.ndimage import gaussian_filter

def set_threads(num: int):
    try:
        import torch
        torch.set_num_threads(max(1, num))
        torch.set_num_interop_threads(max(1, num))
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", str(num))
    os.environ.setdefault("MKL_NUM_THREADS", str(num))

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def tokenize_batch(tok, prompts, max_len):
    enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    return enc

def collect_pooled(model, tok, prompts, tap, pool_mode, k_last, batch_size, max_len):
    """Stream-pooled hidden states for a list of prompts; returns (N, D)."""
    import torch
    reps = []
    with torch.inference_mode():
        for batch in chunks(prompts, batch_size):
            enc = tokenize_batch(tok, batch, max_len)
            out = model(**enc, output_hidden_states=True)
            hs  = out.hidden_states[tap]      # (B, T, D)
            if pool_mode == "lastk":
                k = min(k_last, hs.shape[1])
                H = hs[:, -k:, :].mean(1)
            else:
                H = hs.mean(1)
            reps.append(H.cpu().numpy().astype(np.float32))
            del enc, out, hs, H
    return np.concatenate(reps, axis=0) if reps else np.zeros((0, model.config.n_embd), dtype=np.float32)

def build_funnel_priors_from_Y3(Y3):
    r = np.linalg.norm(Y3[:, :2], axis=1)
    r = (r - r.min()) / (r.ptp() + 1e-8)
    r_grid = np.linspace(0, 1, 128)
    p = 1.3
    phi = 1.0 - np.power(r_grid, p)
    g = np.abs(np.gradient(phi, r_grid))
    g = (g - g.min()) / (g.ptp() + 1e-8)
    return r_grid, phi, g

def phantom_metrics_from_Y3(Y3, nbins=100, sigma=3.0, density_floor=2.0, min_prom=0.35, merge_tol=1e-6):
    # 2D histogram density -> energy U = -gaussian(H)
    X2 = Y3[:, :2]
    H, xe, ye = np.histogram2d(X2[:,0], X2[:,1], bins=nbins)
    Hs = gaussian_filter(H, sigma=sigma)
    U = -Hs
    h, w = U.shape
    if h < 3 or w < 3 or Hs.max() <= 0:
        return 0.0, 0.0, 0.0
    # Find local minima w/ gating
    mins = []
    for i in range(1, h-1):
        for j in range(1, w-1):
            c = U[i, j]
            neigh = U[i-1:i+2, j-1:j+2].copy()
            neigh[1,1] = c + 1e9
            if (c < neigh).all():
                if Hs[i, j] < density_floor:  # density floor
                    continue
                prom = (np.nanmean(neigh) - c)  # deeper if positive
                if prom < min_prom:
                    continue
                mins.append(c)
    if not mins:
        return 0.0, 0.0, 0.0
    mins = np.array(sorted(mins))
    uniq = [mins[0]]
    for v in mins[1:]:
        if abs(v - uniq[-1]) > merge_tol:
            uniq.append(v)
    uniq = np.array(uniq)
    if len(uniq) == 1:
        return 0.0, 0.0, 0.0
    pi = float((len(uniq) - 1) / len(uniq))
    margin_raw = float(uniq[1] - uniq[0])
    rng = float(U.max() - U.min() + 1e-8)
    margin_norm = float(margin_raw / rng)
    return pi, margin_raw, margin_norm

def token_inward_trend_stream(model, tok, prompts, tap, pca, r_max, batch_size, max_len):
    import torch
    vals = []
    with torch.inference_mode():
        for batch in chunks(prompts, batch_size):
            enc = tokenize_batch(tok, batch, max_len)
            out = model(**enc, output_hidden_states=True)
            hs  = out.hidden_states[tap]  # (B, T, D)
            B = hs.shape[0]
            for b in range(B):
                Y = pca.transform(hs[b].cpu().numpy())
                R = np.linalg.norm(Y[:, :2], axis=1)
                Rn = R / (r_max + 1e-8)
                diffs = np.diff(Rn)
                vals.append(float((diffs < 0).sum() / max(1, len(diffs))))
            del enc, out, hs
    return float(np.mean(vals)) if vals else float("nan")

def scan_layers(args):
    _lazy_imports()
    set_threads(args.threads)

    tok   = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval().to("cpu")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    with open(args.calib) as f: calib_prompts = [ln.strip() for ln in f if ln.strip()]
    with open(args.eval)  as f: eval_prompts  = [ln.strip() for ln in f if ln.strip()]

    # detect number of hidden layers
    _enc = tok(["hello"], return_tensors="pt")
    _out = model(**_enc, output_hidden_states=True)
    L = len(_out.hidden_states)  # includes embeddings for GPT-2
    taps = list(range(-1, -L-1, -1))
    del _enc, _out

    rows = []
    best = None

    for tap in taps:
        # --- Incremental PCA on calibration ---
        from sklearn.decomposition import IncrementalPCA
        ipca = IncrementalPCA(n_components=3, whiten=True, batch_size=args.ipca_batch)

        # Pass 1: partial fit
        for batch in chunks(calib_prompts, args.batch_size):
            H = collect_pooled(model, tok, batch, tap, args.pool_mode, args.k_last, args.batch_size, args.max_len)
            if H.size:
                ipca.partial_fit(H)

        # Pass 2: transform all calib
        Yc_parts = []
        for batch in chunks(calib_prompts, args.batch_size):
            H = collect_pooled(model, tok, batch, tap, args.pool_mode, args.k_last, args.batch_size, args.max_len)
            if H.size:
                Yc_parts.append(ipca.transform(H))
        Yc = np.concatenate(Yc_parts, axis=0) if Yc_parts else np.zeros((0,3), dtype=np.float32)

        r_cal = np.linalg.norm(Yc[:, :2], axis=1) if Yc.size else np.array([1.0], dtype=np.float32)
        r_max = float(r_cal.max() + 1e-8)

        r_grid, phi_cal, g_cal = build_funnel_priors_from_Y3(Yc if Yc.size else np.zeros((1,3), dtype=np.float32))

        # Eval reps -> transform
        Ye_parts = []
        for batch in chunks(eval_prompts, args.batch_size):
            H = collect_pooled(model, tok, batch, tap, args.pool_mode, args.k_last, args.batch_size, args.max_len)
            if H.size:
                Ye_parts.append(ipca.transform(H))
        Ye = np.concatenate(Ye_parts, axis=0) if Ye_parts else np.zeros((0,3), dtype=np.float32)

        if Ye.size:
            R  = np.linalg.norm(Ye[:, :2], axis=1)
            Rn = R / (r_max + 1e-8)
            phi_e = np.interp(Rn, r_grid, phi_cal)
            g_e   = np.interp(Rn, r_grid, g_cal)
            S     = 0.05 * phi_e + 0.25 * (g_e ** 2)
            S_median = float(np.median(S))
            pi, _, margin_norm = phantom_metrics_from_Y3(Ye, nbins=args.nbins, sigma=args.sigma_px,
                                                        density_floor=args.density_floor, min_prom=args.min_prom)
            r_trend_tokens = token_inward_trend_stream(model, tok, eval_prompts, tap, ipca, r_max,
                                                       args.batch_size, args.max_len)
        else:
            S_median = 0.0
            pi = 1.0
            margin_norm = 0.0
            r_trend_tokens = 0.5

        row = dict(
            tap=tap, phantom_index=pi, margin_norm=margin_norm,
            r_trend_tokens=r_trend_tokens, S_median=S_median
        )
        rows.append(row)
        print(f"[SCAN/LIGHT] tap={tap:3d} | PI={pi:.3f} | margin_norm={margin_norm:.3f} | r_trend={r_trend_tokens:.3f} | S_med={S_median:.3f}")

        score = (pi, -margin_norm, -r_trend_tokens)
        if (best is None) or (score < best["score"]):
            best = dict(score=score, tap=tap)

    # write CSV
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows: w.writerow(r)

    summary = {"model": args.model, "best_tap": best["tap"], "csv": args.out_csv}
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print("[RESULT/LIGHT]", json.dumps(summary, indent=2))

def build_argparser():
    import argparse
    ap = argparse.ArgumentParser(description="Stage‑11 Layer Scan (Light)")
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--calib", type=str, required=True)
    ap.add_argument("--eval",  type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--ipca_batch", type=int, default=256)
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--pool_mode", type=str, default="lastk", choices=["mean","lastk"])
    ap.add_argument("--k_last", type=int, default=6)
    ap.add_argument("--nbins", type=int, default=100)
    ap.add_argument("--sigma_px", type=float, default=3.0)
    ap.add_argument("--density_floor", type=float, default=2.0)
    ap.add_argument("--min_prom", type=float, default=0.35)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out_csv", type=str, default="layer_scan_light.csv")
    ap.add_argument("--out_json", type=str, default="layer_scan_light_summary.json")
    return ap

def main():
    args = build_argparser().parse_args()
    scan_layers(args)

if __name__ == "__main__":
    main()
