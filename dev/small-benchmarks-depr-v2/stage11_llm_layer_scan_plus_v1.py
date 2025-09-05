#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, json, math, os, numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

"""
Stage-11 Layer Scan (+ passes) — now with optional **pre‑warp**

New: `--prewarp` runs a blind radial warp (no anchor required) to encourage a single, deeper
basin before DETECT/DENOISE. This helps layers where minima are too shallow to anchor.

Pipeline:
  PCA (calib) → Eval → (optional) Pre‑warp → WARP (energy) → DETECT → DENOISE → PASS(1..3)

Pre‑warp details:
  • Center: robust median (XY) or densest-bin peak (if `--prewarp_mode peak`).
  • Scale: percentile of radii vs center (default 60th; `--prewarp_scale_pct`).
  • Strength: small α (default 0.06) to avoid flattening; `--prewarp_alpha`.
  • Iterations: repeat pre‑warp a few times (default 2) with α annealed by `--prewarp_decay`.
  • After pre‑warp we recompute energy + minima; detect/denoise/passes then proceed normally.

CSV/JSON include pass details, as before.
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


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


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
    return U, Hs, xe, ye


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
    mins.sort(key=lambda t: t[0])
    uniq = [mins[0]]
    for c,i,j in mins[1:]:
        if abs(c - uniq[-1][0]) > merge_tol:
            uniq.append((c,i,j))
    return uniq


def phantom_metrics(U, minima):
    if len(minima) <= 1:
        return 0.0, 0.0, 0.0
    vals = np.array([m[0] for m in minima])
    vals.sort()
    n = len(vals)
    pi = float((n-1)/n)
    margin_raw = float(vals[1] - vals[0])
    rng = float(U.max() - U.min() + 1e-8)
    margin_norm = float(margin_raw / rng)
    return pi, margin_raw, margin_norm


def radial_rewarp_Y(Ye, center_xy, alpha=0.10, scale=None):
    Yw = Ye.copy()
    xy  = Yw[:, :2]
    v   = xy - np.array(center_xy)[None, :]
    r   = np.linalg.norm(v, axis=1) + 1e-12
    if scale is None:
        scale = np.percentile(r, 60)
        scale = max(scale, 1e-6)
    shrink = 1.0 - alpha * np.exp(-(r/scale)**2)
    Yw[:, :2] = np.array(center_xy)[None, :] + (v.T * shrink).T
    return Yw


def token_inward_trend(model, tok, prompt, tap, pca, r_max, smooth_k=3,
                       warp_center_xy=None, warp_alpha=0.0, warp_scale=None):
    with torch.no_grad():
        enc = tok(prompt, return_tensors="pt")
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[tap][0].cpu().numpy()
        Y  = pca.transform(hs)
        if warp_center_xy is not None and warp_alpha > 0.0:
            Y = radial_rewarp_Y(Y, warp_center_xy, alpha=warp_alpha, scale=warp_scale)
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
# Main: Layer Scan (+ optional pre‑warp) and up to 3 passes
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--tap_range", type=str, required=True)
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
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--alpha_schedule", type=str, default="0.10,0.06,0.03")
    ap.add_argument("--scale_pct", type=str, default="60,55,50")
    # Pre‑warp knobs
    ap.add_argument("--prewarp", action="store_true", help="apply blind radial pre‑warp before detect")
    ap.add_argument("--prewarp_mode", type=str, default="median", choices=["median","peak"], help="center choice")
    ap.add_argument("--prewarp_alpha", type=float, default=0.06)
    ap.add_argument("--prewarp_scale_pct", type=int, default=60)
    ap.add_argument("--prewarp_iters", type=int, default=2)
    ap.add_argument("--prewarp_decay", type=float, default=0.75, help="multiply alpha by this each iter")

    args = ap.parse_args()

    # schedules
    args.passes = max(1, min(3, int(args.passes)))
    alpha_sched = parse_float_list(args.alpha_schedule)
    scale_pcts  = parse_int_list(args.scale_pct)
    while len(alpha_sched) < args.passes: alpha_sched.append(alpha_sched[-1])
    while len(scale_pcts)  < args.passes: scale_pcts.append(scale_pcts[-1])

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
        # PCA on calibration
        Hc = collect_hidden_states(model, tok, calib_prompts, t, args.pool_mode, args.k_last)
        pca, Yc = pca3_and_center(Hc)
        r_max = float(np.linalg.norm(Yc[:, :2], axis=1).max() + 1e-8)

        # Eval embeddings in PCA
        He = collect_hidden_states(model, tok, eval_prompts, t, args.pool_mode, args.k_last)
        Ye = pca.transform(He)

        # Optional blind pre‑warp (no anchor needed)
        pre_center = None
        pre_iters_run = 0
        pre_alphas = []
        if args.prewarp:
            for k in range(max(1, int(args.prewarp_iters))):
                # center: robust median or densest peak
                if args.prewarp_mode == "median":
                    cx, cy = np.median(Ye[:,0]), np.median(Ye[:,1])
                else:
                    Utmp, Hstmp, xet, yet = hist_energy(Ye, nbins=120, sigma=args.sigma_px)
                    mins_tmp = find_minima(Utmp, Hstmp, args.density_floor*0.8, args.min_prom*0.8)
                    if mins_tmp:
                        mins_tmp.sort(key=lambda x: x[0])
                        ii, jj = mins_tmp[0][1], mins_tmp[0][2]
                        cx = 0.5 * (xet[ii] + xet[ii+1]); cy = 0.5 * (yet[jj] + yet[jj+1])
                    else:
                        cx, cy = np.median(Ye[:,0]), np.median(Ye[:,1])
                center_xy = (float(cx), float(cy))
                # scale: percentile radius from current center
                v   = Ye[:, :2] - np.array(center_xy)[None, :]
                r   = np.linalg.norm(v, axis=1) + 1e-12
                scale = max(np.percentile(r, int(args.prewarp_scale_pct)), 1e-6)
                # annealed alpha
                alpha = float(args.prewarp_alpha * (args.prewarp_decay ** k))
                Ye = radial_rewarp_Y(Ye, center_xy, alpha=alpha, scale=scale)
                pre_center = center_xy; pre_iters_run += 1; pre_alphas.append(alpha)
            print(f"[PREWARP] iters={pre_iters_run} center=({pre_center[0]:.3f},{pre_center[1]:.3f}) alphas={[round(a,3) for a in pre_alphas]}")

        # WARP (energy)
        U_warp, Hs, xe, ye = hist_energy(Ye, nbins=120, sigma=args.sigma_px)
        minima_w = find_minima(U_warp, Hs, args.density_floor, args.min_prom)
        pi_w, mraw_w, mnorm_w = phantom_metrics(U_warp, minima_w)

        # token trend (pre‑warp aware)
        trend_vals = [token_inward_trend(model, tok, p, t, pca, r_max, 3) for p in eval_prompts_trend]
        trend_w = float(np.mean(trend_vals)) if trend_vals else 0.0

        # Defaults
        pi_d, mnorm_d, trend_d = pi_w, mnorm_w, trend_w
        pi_z, mnorm_z, trend_z = pi_w, mnorm_w, trend_w

        # DETECT-lite
        c_rc = None
        if args.with_detect and minima_w:
            minima_w.sort(key=lambda x: x[0])
            c_rc = (minima_w[0][1], minima_w[0][2])
            if len(minima_w) > 1:
                margin_raw = minima_w[1][0] - minima_w[0][0]
                rng = U_warp.max() - U_warp.min() + 1e-8
                if (margin_raw / rng) < 0.01:
                    c_rc = None
            pi_d, _, mnorm_d = pi_w, mraw_w, mnorm_w
            trend_d = trend_w

        # DENOISE-lite
        if args.with_denoise and c_rc is not None:
            U_den = U_warp.copy()
            Ui, Uj = c_rc
            h, w = U_den.shape
            yy, xx = np.mgrid[0:h, 0:w]
            d2 = (yy - Ui)**2 + (xx - Uj)**2
            U_den -= 0.75 * np.exp(-d2 / (2*(max(1, 6/2)**2)))
            ann = (d2 > (6**2)) & (d2 < (3*6**2))
            U_den[ann] += 0.15 * 0.75
            U_den = gaussian_filter(U_den, sigma=1.0)
            minima_z = find_minima(U_den, Hs, args.density_floor, args.min_prom)
            pi_z, _, mnorm_z = phantom_metrics(U_den, minima_z)
            trend_z = trend_w

        # Passes (anchored re‑warp)
        pass_metrics = []
        Ye_p = Ye.copy()
        pass_details = []
        center_xy = None
        if c_rc is not None and args.passes >= 1:
            ci, cj = c_rc
            xc = 0.5 * (xe[ci] + xe[ci+1])
            yc = 0.5 * (ye[cj] + ye[cj+1])
            center_xy = (float(xc), float(yc))
            for p_idx in range(args.passes):
                alpha = float(alpha_sched[p_idx])
                pct   = int(scale_pcts[p_idx])
                v   = Ye_p[:, :2] - np.array(center_xy)[None, :]
                r   = np.linalg.norm(v, axis=1) + 1e-12
                scale = max(np.percentile(r, pct), 1e-6)
                Ye_p = radial_rewarp_Y(Ye_p, center_xy, alpha=alpha, scale=scale)
                U_p, Hs_p, _, _ = hist_energy(Ye_p, nbins=120, sigma=args.sigma_px)
                mins_p = find_minima(U_p, Hs_p, args.density_floor, args.min_prom)
                pi_p, _, mnorm_p = phantom_metrics(U_p, mins_p)
                nmin = int(len(mins_p))
                trend_vals_p = [
                    token_inward_trend(
                        model, tok, pr, t, pca, r_max, 3,
                        warp_center_xy=center_xy, warp_alpha=alpha, warp_scale=scale
                    ) for pr in eval_prompts_trend
                ]
                trend_p = float(np.mean(trend_vals_p)) if trend_vals_p else 0.0
                pass_metrics.append((pi_p, mnorm_p, trend_p, alpha, scale, nmin))
                pass_details.append({
                    "pass": p_idx+1,
                    "alpha": float(alpha),
                    "scale": float(scale),
                    "center_x": float(center_xy[0]),
                    "center_y": float(center_xy[1]),
                    "phantom_index": float(pi_p),
                    "margin_norm": float(mnorm_p),
                    "trend": float(trend_p),
                    "minima_count": int(nmin)
                })

        if pass_metrics:
            pi_fin, mnorm_fin, trend_fin, _, _, _ = pass_metrics[-1]
        else:
            pi_fin, mnorm_fin, trend_fin = pi_z, mnorm_z, trend_z

        row = {
            "tap": int(t),
            "warp_phantom_index": float(pi_w),
            "warp_margin_norm": float(mnorm_w),
            "warp_trend": float(trend_w),
            "detect_phantom_index": float(pi_d),
            "detect_margin_norm": float(mnorm_d),
            "detect_trend": float(trend_d),
            "denoise_phantom_index": float(pi_z),
            "denoise_margin_norm": float(mnorm_z),
            "denoise_trend": float(trend_z),
            "final_phantom_index": float(pi_fin),
            "final_margin_norm": float(mnorm_fin),
            "final_trend": float(trend_fin),
        }
        for idx, (pi_p, mnorm_p, trend_p, alpha_p, scale_p, nmin_p) in enumerate(pass_metrics, start=1):
            row[f"pass{idx}_phantom_index"] = float(pi_p)
            row[f"pass{idx}_margin_norm"]   = float(mnorm_p)
            row[f"pass{idx}_trend"]         = float(trend_p)
            row[f"pass{idx}_alpha"]         = float(alpha_p)
            row[f"pass{idx}_scale"]         = float(scale_p)
            row[f"pass{idx}_minima_count"]  = int(nmin_p)
            if center_xy is not None:
                row[f"pass{idx}_center_x"] = float(center_xy[0])
                row[f"pass{idx}_center_y"] = float(center_xy[1])
        if args.prewarp:
            row["prewarp_iters"] = int(pre_iters_run)
            if pre_center is not None:
                row["prewarp_center_x"] = float(pre_center[0])
                row["prewarp_center_y"] = float(pre_center[1])

        rows.append(row)

        log = f"[SCAN] tap={t:>3} | WARP: PI={pi_w:.3f} mN={mnorm_w:.4f} trend={trend_w:.3f}"
        if args.prewarp and pre_center is not None:
            log = "[PREWARP]→ " + log
        if args.with_denoise and (c_rc is not None):
            log += f" | DENOISE: PI={pi_z:.3f} mN={mnorm_z:.4f}"
        if pass_metrics:
            pi_p, mn_p, tr_p, a_p, sc_p, nmin_p = pass_metrics[-1]
            log += f" | PASS{len(pass_metrics)}: PI={pi_p:.3f} mN={mn_p:.4f} trend={tr_p:.3f} α={a_p:.3f} R≈{sc_p:.3f} nmin={nmin_p}"
        print(log)

    # save CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        fieldnames = sorted({k for r in rows for k in r.keys()}, key=lambda x: ("pass" in x, x))
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

    # plot (PI only)
    taps_sorted = [r["tap"] for r in rows]
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(taps_sorted, [r["warp_phantom_index"]   for r in rows], "o-", label="PI (warp)")
    if any("detect_phantom_index" in r for r in rows):
        ax.plot(taps_sorted, [r["detect_phantom_index"] for r in rows], "s--", label="PI (detect)")
    if any("denoise_phantom_index" in r for r in rows):
        ax.plot(taps_sorted, [r["denoise_phantom_index"] for r in rows], "^-", label="PI (denoise)")
    ax.plot(taps_sorted, [r["final_phantom_index"]  for r in rows], "x-", label="PI (passN)")
    ax.set_xlabel("Layer tap (negative = higher layer)"); ax.set_ylabel("phantom_index (↓ better)")
    ax.grid(True, alpha=0.3); ax.legend(); ax.set_title("Layer Scan — Pre‑warp (opt) + Warp/Detect/Denoise/PassN")
    plt.tight_layout(); plt.savefig(args.out_png, dpi=160); plt.close(fig)

    with open(args.out_json, "w") as f:
        json.dump({"rows": rows}, f, indent=2)

    print(f"[WRITE] {args.out_csv}, {args.out_png}, {args.out_json}")


if __name__ == "__main__":
    main()
