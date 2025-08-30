#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
for t in {-24..-1}; do
  echo "[SCAN] tap=$t"
  python3 stage11_llm_layer_scan.py \
    --model gpt2 \
    --tap_range "$t" \
    --calib calib_prompts_v2_900.txt \
    --eval calib_eval_style_200.txt \
    --pool_mode lastk --k_last 6 \
    --sigma_px 4.0 --density_floor 3.0 --min_prom 0.45 \
    --out_csv logs/layer_scan_metrics_t${t}.csv \
    --out_png logs/layer_scan_plot_t${t}.png \
    --out_json logs/layer_scan_summary_t${t}.json
done
"""

import argparse, csv, json, math, os, numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, minimum_filter

# ---------------------------
# Utilities
# ---------------------------

def parse_tap_range(spec: str) -> list[int]:
    # e.g., "-24:-1" (inclusive), "-12:-3", or single "-6"
    spec = spec.strip()
    if ":" in spec:
        a, b = spec.split(":")
        a, b = int(a), int(b)
        step = 1 if a <= b else -1
        return list(range(a, b + step, step))
    return [int(spec)]

def collect_hidden_states(model, tok, prompts, tap: int, pool_mode="lastk", k_last=6):
    with torch.no_grad():
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
        out = model(**enc, output_hidden_states=True)
        hs  = out.hidden_states[tap]  # (B, T, D)
        if pool_mode == "lastk":
            k = min(k_last, hs.shape[1])
            H = hs[:, -k:, :].mean(1)
        else:
            H = hs.mean(1)
        return H.cpu().numpy().astype(float)

def phantom_metrics_from_Y3(
    Y3, nbins=120, sigma=3.5, density_floor=2.0, min_prom=0.35, merge_tol=1e-6
):
    """
    Return (phantom_index, margin_raw, margin_norm).
    Uses histogram density -> smoothed -> energy U=-H.
    Applies density floor and prominence gating; merges near-equal minima.
    """
    X2 = Y3[:, :2]
    x, y = X2[:, 0], X2[:, 1]
    H, xe, ye = np.histogram2d(x, y, bins=nbins)
    Hs = gaussian_filter(H, sigma=sigma)
    U = -Hs
    h, w = U.shape
    if h < 3 or w < 3:
        return 0.0, 0.0, 0.0

    mins = []
    for i in range(1, h-1):
        for j in range(1, w-1):
            if Hs[i, j] < density_floor:
                continue
            c = U[i, j]
            neigh = U[i-1:i+2, j-1:j+2].copy()
            neigh[1,1] = np.nan
            prom = np.nanmean(neigh) - c  # positive if center is deeper
            if prom >= min_prom:
                # strict local minimum check vs neighbors
                if np.all(c < np.nan_to_num(neigh, nan=np.inf)):
                    mins.append(c)

    if not mins:
        return 0.0, 0.0, 0.0

    mins = np.sort(np.array(mins))
    uniq = [mins[0]]
    for v in mins[1:]:
        if abs(v - uniq[-1]) > merge_tol:
            uniq.append(v)
    uniq = np.array(uniq)
    n = len(uniq)
    if n == 1:
        return 0.0, 0.0, 0.0

    pi = float((n - 1) / n)
    margin_raw = float(uniq[1] - uniq[0])
    rng = float(U.max() - U.min() + 1e-8)
    margin_norm = float(margin_raw / rng)
    return pi, margin_raw, margin_norm

def token_inward_trend(model, tok, prompt, tap, pca, r_max):
    with torch.no_grad():
        enc = tok(prompt, return_tensors="pt")
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[tap][0].cpu().numpy()   # [T, D]
        Y  = pca.transform(hs)                         # [T, 3]
        R  = np.linalg.norm(Y[:, :2], axis=1)
        Rn = R / (r_max + 1e-8)                        # global (calibration) normalization
        diffs = np.diff(Rn)
        return float((diffs < 0).sum() / max(1, len(diffs)))

# ---------------------------
# Main: layer scan
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--tap_range", type=str, required=True, help="e.g. -24:-1 or -12:-3")
    ap.add_argument("--calib", type=str, required=True)
    ap.add_argument("--eval", type=str, required=True)
    ap.add_argument("--pool_mode", type=str, default="lastk", choices=["lastk","mean"])
    ap.add_argument("--k_last", type=int, default=6)
    ap.add_argument("--sigma_px", type=float, default=3.5)
    ap.add_argument("--density_floor", type=float, default=2.0)
    ap.add_argument("--min_prom", type=float, default=0.35)
    ap.add_argument("--eval_trend_limit", type=int, default=64, help="prompts used for token trend")
    ap.add_argument("--out_csv", type=str, default="layer_scan_metrics.csv")
    ap.add_argument("--out_png", type=str, default="layer_scan_plot.png")
    ap.add_argument("--out_json", type=str, default="layer_scan_summary.json")
    args = ap.parse_args()

    tok   = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval(); model.to("cpu")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    with open(args.calib) as f: calib_prompts = [ln.strip() for ln in f if ln.strip()]
    with open(args.eval)  as f: eval_prompts  = [ln.strip() for ln in f if ln.strip()]
    eval_prompts_trend = eval_prompts[: min(args.eval_trend_limit, len(eval_prompts))]

    taps = parse_tap_range(args.tap_range)
    rows = []

    for t in taps:
        # ---- Calibration PCA(3) at this tap
        Hc = collect_hidden_states(model, tok, calib_prompts, t,
                                   pool_mode=args.pool_mode, k_last=args.k_last)
        pca = PCA(n_components=3, whiten=True, random_state=0)
        Yc  = pca.fit_transform(Hc)
        r_max = float(np.linalg.norm(Yc[:, :2], axis=1).max() + 1e-8)

        # Funnel priors (optional S score; not strictly needed for scan summary)
        # r_grid, phi_cal, g_cal = build_funnel_priors_from_Y3(Yc)  # if you have this in your base

        # ---- Eval at this tap
        He = collect_hidden_states(model, tok, eval_prompts, t,
                                   pool_mode=args.pool_mode, k_last=args.k_last)
        Ye = pca.transform(He)

        # Phantom metrics
        pi, margin_raw, margin_norm = phantom_metrics_from_Y3(
            Ye, nbins=120, sigma=args.sigma_px,
            density_floor=args.density_floor, min_prom=args.min_prom
        )

        # Token-wise inward trend (average over a subset for speed)
        trends = []
        for p in eval_prompts_trend:
            trends.append(token_inward_trend(model, tok, p, t, pca, r_max))
        r_trend_tokens = float(np.mean(trends)) if trends else 0.0

        rows.append({
            "tap": t,
            "phantom_index": float(pi),
            "margin_norm": float(margin_norm),
            "margin_raw": float(margin_raw),
            "r_trend_tokens": float(r_trend_tokens),
            "n_eval": len(eval_prompts),
            "n_calib": len(calib_prompts),
        })
        print(f"[SCAN] tap={t:>3} | PI={pi:.3f} | margin_norm={margin_norm:.4f} | "
              f"trend={r_trend_tokens:.3f}")

    # ---- Save CSV
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- Pick best tap (lowest PI, then highest margin_norm, then highest trend)
    def tap_score(r):
        return (round(r["phantom_index"], 6),
                -round(r["margin_norm"], 6),
                -round(r["r_trend_tokens"], 6))
    best = sorted(rows, key=tap_score)[0]

    # ---- Plot
    taps_sorted = [r["tap"] for r in rows]
    pi_list     = [r["phantom_index"]   for r in rows]
    mn_list     = [r["margin_norm"]     for r in rows]
    tr_list     = [r["r_trend_tokens"]  for r in rows]

    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(taps_sorted, pi_list, marker="o", label="phantom_index (↓ better)")
    ax.plot(taps_sorted, mn_list, marker="s", label="margin_norm (↑ better)")
    ax.plot(taps_sorted, tr_list, marker="^", label="r_trend_tokens (↑ better)")
    ax.axvline(best["tap"], linestyle="--", alpha=0.5, label=f"best tap={best['tap']}")
    ax.set_xlabel("Layer tap (negative = from top)")
    ax.set_title("Layer Scan — Well Competition Profile")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=160)
    plt.close(fig)

    # ---- Summary JSON
    with open(args.out_json, "w") as f:
        json.dump({"best_tap": best, "rows": rows}, f, indent=2)

    print(f"\n[RESULT] Best tap ≈ {best['tap']} | PI={best['phantom_index']:.3f} | "
          f"margin_norm={best['margin_norm']:.4f} | trend={best['r_trend_tokens']:.3f}")
    print(f"[WRITE] {args.out_csv}, {args.out_png}, {args.out_json}")

if __name__ == "__main__":
    main()
