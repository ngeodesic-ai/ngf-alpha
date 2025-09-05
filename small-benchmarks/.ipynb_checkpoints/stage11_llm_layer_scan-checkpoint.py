#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, json, math, os, numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, minimum_filter


"""
for t in {-12..-6}; do
  for k in 8 12; do
    python3 stage11_llm_layer_scan.py \
      --model gpt2 --tap_range "$t" \
      --calib calib/calib_prompts_v2_900.txt --eval calib/calib_eval_style_200.txt \
      --pool_mode lastk --k_last $k \
      --sigma_px 5.0 --density_floor 4.0 --min_prom 0.55 \
      --with_detect --with_denoise \
      --out_csv logs/wdd_t${t}_k${k}.csv \
      --out_png logs/wdd_t${t}_k${k}.png \
      --out_json logs/wdd_t${t}_k${k}.json
  done
done

python3 stage11_llm_layer_scan.py \
      --model gpt2 --tap_range -9 \
      --calib calib/calib_prompts_v2_900.txt --eval calib/calib_eval_style_200.txt \
      --pool_mode lastk --k_last 12 \
      --sigma_px 5.0 --density_floor 4.0 --min_prom 0.55 \
      --with_detect --with_denoise \
      --out_csv logs/wdd_t9_k12.csv \
      --out_png logs/wdd_t9_k12.png \
      --out_json logs/wdd_t9_k12.json

"""

# ---------------------------
# Helpers
# ---------------------------

def parse_tap_range(spec: str):
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
        hs  = out.hidden_states[tap]  # (B,T,D)
        if pool_mode == "lastk":
            k = min(k_last, hs.shape[1])
            H = hs[:, -k:, :].mean(1)
        else:
            H = hs.mean(1)
        return H.cpu().numpy().astype(float)

def pca3_and_center(Hc):
    pca = PCA(n_components=3, whiten=True, random_state=0)
    Yc  = pca.fit_transform(Hc)
    return pca, Yc

def hist_energy(Y, nbins=120, sigma=3.5):
    X2 = Y[:, :2]
    x, y = X2[:, 0], X2[:, 1]
    H, xe, ye = np.histogram2d(x, y, bins=nbins)
    Hs = gaussian_filter(H, sigma=sigma)
    U  = -Hs  # energy ∝ -density
    return U, Hs

def find_minima(U, Hs, density_floor=2.0, min_prom=0.35, merge_tol=1e-6):
    h, w = U.shape
    mins = []
    for i in range(1, h-1):
        for j in range(1, w-1):
            if Hs[i, j] < density_floor: 
                continue
            c = U[i, j]
            neigh = U[i-1:i+2, j-1:j+2].copy()
            neigh[1,1] = np.nan
            prom = np.nanmean(neigh) - c
            if prom >= min_prom and np.all(c < np.nan_to_num(neigh, nan=np.inf)):
                mins.append((c, i, j))
    if not mins:
        return []
    # sort and merge near-equals (plateaus)
    mins.sort(key=lambda t: t[0])
    uniq = [mins[0]]
    for c,i,j in mins[1:]:
        if abs(c - uniq[-1][0]) > merge_tol:
            uniq.append((c,i,j))
    return uniq

def phantom_metrics(U, minima):
    if len(minima) <= 1:
        return 0.0, 0.0, 0.0
    vals = np.array([m[0] for m in minima])  # energies (more negative = deeper)
    vals.sort()
    n = len(vals)
    pi = float((n-1)/n)
    margin_raw = float(vals[1] - vals[0])             # gap (#2 minus #1) in energy space
    rng = float(U.max() - U.min() + 1e-8)
    margin_norm = float(margin_raw / rng)
    return pi, margin_raw, margin_norm

def token_inward_trend(model, tok, prompt, tap, pca, r_max, smooth_k=3):
    with torch.no_grad():
        enc = tok(prompt, return_tensors="pt")
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[tap][0].cpu().numpy()   # [T,D]
        Y  = pca.transform(hs)                         # [T,3]
        R  = np.linalg.norm(Y[:, :2], axis=1)
        Rn = R / (r_max + 1e-8)
        if smooth_k >= 3:
            pad = smooth_k//2
            xp = np.pad(Rn, (pad,pad), mode="edge")
            kernel = np.ones(smooth_k) / smooth_k
            Rn = np.convolve(xp, kernel, mode="valid")
        diffs = np.diff(Rn)
        return float((diffs < 0).sum() / max(1, len(diffs)))

# ---------------------------
# Denoise controls (lite)
# ---------------------------

def lateral_inhibition(U, center_rc, radius_px=6, strength=0.75):
    """Suppress nearby competitors by deepening the chosen minimum and lightly raising others in a band."""
    Ui, Uj = center_rc
    V = U.copy()
    h, w = U.shape
    yy, xx = np.mgrid[0:h, 0:w]
    d2 = (yy - Ui)**2 + (xx - Uj)**2
    # deepen core (make energy lower near center)
    V -= strength * np.exp(-d2 / (2*(max(1, radius_px/2)**2)))
    # mild bump in an annulus to reduce nearby minima
    ann = (d2 > (radius_px**2)) & (d2 < (3*radius_px**2))
    V[ann] += 0.15 * strength
    return V

# ---------------------------
# Main: Layer Scan with stages
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--tap_range", type=str, required=True)   # e.g. "-24:-1"
    ap.add_argument("--calib", type=str, required=True)
    ap.add_argument("--eval", type=str, required=True)
    ap.add_argument("--pool_mode", type=str, default="lastk", choices=["lastk","mean"])
    ap.add_argument("--k_last", type=int, default=6)
    ap.add_argument("--sigma_px", type=float, default=4.0)
    ap.add_argument("--density_floor", type=float, default=3.0)
    ap.add_argument("--min_prom", type=float, default=0.45)
    ap.add_argument("--trend_limit", type=int, default=64)
    ap.add_argument("--with_detect", action="store_true")
    ap.add_argument("--with_denoise", action="store_true")
    ap.add_argument("--out_csv", type=str, default="layer_scan_plus.csv")
    ap.add_argument("--out_png", type=str, default="layer_scan_plus.png")
    ap.add_argument("--out_json", type=str, default="layer_scan_plus.json")
    args = ap.parse_args()

    tok   = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval(); model.to("cpu")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    with open(args.calib) as f: calib_prompts = [ln.strip() for ln in f if ln.strip()]
    with open(args.eval)  as f: eval_prompts  = [ln.strip() for ln in f if ln.strip()]
    eval_prompts_trend = eval_prompts[: min(args.trend_limit, len(eval_prompts))]

    taps = parse_tap_range(args.tap_range)
    rows = []

    for t in taps:
        # ---- Calibration PCA at tap
        Hc = collect_hidden_states(model, tok, calib_prompts, t, args.pool_mode, args.k_last)
        pca, Yc = pca3_and_center(Hc)
        r_max = float(np.linalg.norm(Yc[:, :2], axis=1).max() + 1e-8)

        # ---- Eval embeddings at same PCA
        He = collect_hidden_states(model, tok, eval_prompts, t, args.pool_mode, args.k_last)
        Ye = pca.transform(He)

        # --- Stage A: WARP (density → energy)
        U_warp, Hs = hist_energy(Ye, nbins=120, sigma=args.sigma_px)
        minima_w = find_minima(U_warp, Hs, args.density_floor, args.min_prom)
        pi_w, mraw_w, mnorm_w = phantom_metrics(U_warp, minima_w)

        # trend (warp-only geometry)
        trend_vals = [token_inward_trend(model, tok, p, t, pca, r_max, smooth_k=3) for p in eval_prompts_trend]
        trend_w = float(np.mean(trend_vals)) if trend_vals else 0.0

        # Defaults for next stages
        pi_d, mnorm_d, trend_d = pi_w, mnorm_w, trend_w
        pi_z, mnorm_z, trend_z = pi_w, mnorm_w, trend_w

        # --- Stage B: DETECT-lite (optional)
        c_rc = None
        if args.with_detect and minima_w:
            # choose deepest gated minimum; require tiny positive margin
            minima_w.sort(key=lambda x: x[0])            # most negative first
            c_rc = (minima_w[0][1], minima_w[0][2])      # (i,j)
            if len(minima_w) > 1:
                margin_raw = minima_w[1][0] - minima_w[0][0]
                rng = U_warp.max() - U_warp.min() + 1e-8
                if (margin_raw / rng) < 0.01:            # fail gate -> no detect stage
                    c_rc = None

            # re-score phantom metrics restricted by a shallow “keep region” around center (optional)
            pi_d, _, mnorm_d = pi_w, mraw_w, mnorm_w  # (we keep same metrics; anchor only)

            # trend doesn’t change here; anchor just tells denoiser what to protect
            trend_d = trend_w

        # --- Stage C: DENOISE-lite (optional)
        if args.with_denoise and c_rc is not None:
            # lateral inhibition around the chosen center; recompute phantom metrics
            U_den = lateral_inhibition(U_warp, c_rc, radius_px=6, strength=0.75)
            # smooth a touch to stabilize metric
            U_den = gaussian_filter(U_den, sigma=1.0)
            # rebuild minima on denoised energy
            minima_z = find_minima(U_den, Hs, args.density_floor, args.min_prom)
            pi_z, _, mnorm_z = phantom_metrics(U_den, minima_z)
            trend_z = trend_w  # (token trend is computed from sequences; denoise is energy-map level)

        rows.append({
            "tap": int(t),
            # warp
            "warp_phantom_index": float(pi_w),
            "warp_margin_norm": float(mnorm_w),
            "warp_trend": float(trend_w),
            # detect (anchor only)
            "detect_phantom_index": float(pi_d),
            "detect_margin_norm": float(mnorm_d),
            "detect_trend": float(trend_d),
            # denoise
            "denoise_phantom_index": float(pi_z),
            "denoise_margin_norm": float(mnorm_z),
            "denoise_trend": float(trend_z),
        })

        print(f"[SCAN] tap={t:>3} | WARP: PI={pi_w:.3f} mN={mnorm_w:.4f} trend={trend_w:.3f}"
              f"{' | DENOISE: PI='+format(pi_z,'.3f')+' mN='+format(mnorm_z,'.4f') if args.with_denoise and c_rc is not None else ''}")

    # save CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # plot
    taps_sorted = [r["tap"] for r in rows]
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(taps_sorted, [r["warp_phantom_index"]   for r in rows], "o-", label="PI (warp)")
    if args.with_detect:
        ax.plot(taps_sorted, [r["detect_phantom_index"] for r in rows], "s--", label="PI (detect)")
    if args.with_denoise:
        ax.plot(taps_sorted, [r["denoise_phantom_index"] for r in rows], "^-", label="PI (denoise)")
    ax.set_xlabel("Layer tap (negative = higher layer)"); ax.set_ylabel("phantom_index (↓ better)")
    ax.grid(True, alpha=0.3); ax.legend(); ax.set_title("Layer Scan — Warp vs Detect vs Denoise")
    plt.tight_layout(); plt.savefig(args.out_png, dpi=160); plt.close(fig)

    with open(args.out_json, "w") as f:
        json.dump({"rows": rows}, f, indent=2)

    print(f"[WRITE] {args.out_csv}, {args.out_png}, {args.out_json}")

if __name__ == "__main__":
    main()
