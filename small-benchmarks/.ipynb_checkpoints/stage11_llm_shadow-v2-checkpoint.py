#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-11 Live LLM Shadow Probe (CPU-only)

Usage:
  python3 stage11_llm_shadow.py \
    --model gpt2 --tap -3 \
    --calib prompts_calib.txt \
    --eval  prompts_eval.txt \
    --render_well \
    --out_json llm_shadow_summary.json
"""

import argparse, json, numpy as np
from sklearn.decomposition import PCA
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, minimum_filter
from scipy.ndimage import gaussian_filter

def safe_radius_bounds(Yc, Ye, lo=5.0, hi=95.0, eps=1e-6):
    """
    Choose non-degenerate [r_lo, r_hi] for the funnel:
      1) use calibration percentiles [lo, hi]
      2) if degenerate, mix calib+eval
      3) if still degenerate, fallback to unit span
    """
    Rc = np.linalg.norm(Yc[:, :2], axis=1).astype(float)
    Re = np.linalg.norm(Ye[:, :2], axis=1).astype(float)

    r_lo, r_hi = np.percentile(Rc, lo), np.percentile(Rc, hi)
    if (r_hi - r_lo) < eps:
        Rmix = np.concatenate([Rc, Re]) if Re.size else Rc
        r_lo, r_hi = np.percentile(Rmix, lo), np.percentile(Rmix, hi)
        if (r_hi - r_lo) < eps:
            # final fallback: unit span around the mean
            mu = float(np.mean(Rmix)) if Rmix.size else 0.0
            r_lo, r_hi = mu - 0.5, mu + 0.5
    return float(r_lo), float(r_hi)

def safe_interp01(x, y, t, eps=1e-8):
    """
    Interpolate y(x) at normalized t in [0,1] robustly.
    x must be increasing; if degenerate, return y.mean.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or (x[-1] - x[0]) < eps:
        return np.full_like(t, float(np.nanmean(y) if np.isfinite(np.nanmean(y)) else 0.0))
    x01 = (x - x[0]) / (x[-1] - x[0] + eps)
    return np.interp(t, x01, y)


def funnel_priors_from_calib_density(Yc, nbins_xy=120, sigma_xy=1.8, nbins_r=128, eps=1e-8):
    """
    Build φ(r) and g(r) from calibration density (PC1/PC2) robustly.
    - Uses 2D density to get an energy per point, then radial-bins that energy.
    - Fully guards against rmax==rmin (flat radius), empty bins, NaNs, etc.
    Returns: (rmin, rmax, r_grid, phi, g)
    """
    X2 = Yc[:, :2]
    x, y = X2[:, 0], X2[:, 1]

    # 2D smoothed density
    H, xe, ye = np.histogram2d(x, y, bins=nbins_xy)
    Hs = gaussian_filter(H.astype(float), sigma=sigma_xy)
    U  = -Hs  # energy

    # Map each calib point to its grid energy (nearest-cell lookup)
    xi = np.clip(np.digitize(x, xe) - 1, 0, Hs.shape[0]-1)
    yi = np.clip(np.digitize(y, ye) - 1, 0, Hs.shape[1]-1)
    e_pt = U[xi, yi].astype(float)

    # Radial coordinates
    r = np.linalg.norm(X2, axis=1)
    rmin, rmax = float(np.nanmin(r)), float(np.nanmax(r))
    if not np.isfinite(rmin) or not np.isfinite(rmax):
        rmin, rmax = 0.0, 1.0
    if (rmax - rmin) < 1e-8:
        rmax = rmin + 1e-8  # avoid zero span

    # Radial binning: mean energy per radial bin (weights / counts)
    sum_e, edges = np.histogram(r, bins=nbins_r, range=(rmin, rmax), weights=e_pt)
    cnt,  _      = np.histogram(r, bins=nbins_r, range=(rmin, rmax))
    cnt = np.maximum(cnt, 1)
    prof = sum_e / cnt
    prof = np.nan_to_num(prof, nan=np.nanmean(prof) if np.isfinite(np.nanmean(prof)) else 0.0)

    # r-grid at bin centers
    r_grid = 0.5 * (edges[:-1] + edges[1:])

    # Normalize to [0,1], flip so center-deeper => larger φ
    pmin, pmax = float(np.min(prof)), float(np.max(prof))
    scale = (pmax - pmin) if (pmax - pmin) > eps else eps
    prof_n = (prof - pmin) / scale
    phi = 1.0 - prof_n

    # Slope g = dφ/dr (uniform spacing, safe)
    dr = float(r_grid[1] - r_grid[0]) if len(r_grid) > 1 else 1.0
    g = np.gradient(phi, dr)
    g = np.abs(g)
    gmin, gmax = float(np.min(g)), float(np.max(g))
    g = (g - gmin) / (gmax - gmin + eps)

    return (rmin, rmax, r_grid, phi.astype(float), g.astype(float))

def find_density_peaks(Hs, tau=0.30, nms_radius=3):
    H = Hs.copy()
    thresh = float(H.max()) * float(tau)
    mask = H >= thresh
    peaks = []
    W = H * mask
    h, w = W.shape
    while True:
        i, j = np.unravel_index(np.argmax(W), W.shape)
        v = W[i, j]
        if v <= 0:
            break
        # ignore borders
        if i in (0, h-1) or j in (0, w-1):
            W[i, j] = 0.0
            continue
        peaks.append((i, j, float(v)))
        i0, i1 = max(0, i-nms_radius), min(h, i+nms_radius+1)
        j0, j1 = max(0, j-nms_radius), min(w, j+nms_radius+1)
        W[i0:i1, j0:j1] = 0.0
        if (W >= thresh).sum() == 0:
            break
    return peaks

def phantom_metrics_from_Y3_density_peaks(Y3, nbins=80, sigma=2.0,
                                          tau=0.30, nms_radius=4,
                                          height_tol=1e-6, prom_frac=0.02, eps=1e-9):
    """
    Returns:
      phantom_index: (K-1)/K  where K = # of unique, prominent peaks
      margin_norm  : (top - second) / (max-min) on peak heights; 0 if K<2
    """
    X2 = Y3[:, :2]
    x, y = X2[:, 0], X2[:, 1]
    H, xe, ye = np.histogram2d(x, y, bins=nbins)
    Hs = gaussian_filter(H, sigma=sigma)

    # raw peaks
    peaks = find_density_peaks(Hs, tau=tau, nms_radius=nms_radius)
    if not peaks:
        return 0.0, 0.0

    # compute simple local background to get prominence
    # (mean of 3x3 neighborhood excluding center)
    def local_mean(i, j):
        i0, i1 = max(0, i-1), min(Hs.shape[0], i+2)
        j0, j1 = max(0, j-1), min(Hs.shape[1], j+2)
        patch = Hs[i0:i1, j0:j1].copy()
        patch[1 if i0>0 else 0, 1 if j0>0 else 0] = np.nan  # skip center approx
        return float(np.nanmean(patch))

    # filter by prominence
    heights = []
    for (i, j, v) in peaks:
        bg = local_mean(i, j)
        if v - bg >= prom_frac * (Hs.max() - Hs.min() + eps):
            heights.append(v)

    if not heights:
        return 0.0, 0.0

    # merge near-duplicate heights (ties)
    heights = np.array(sorted(heights))
    uniq = [heights[0]]
    for h in heights[1:]:
        if abs(h - uniq[-1]) > height_tol:
            uniq.append(h)
    uniq = np.array(uniq)
    K = len(uniq)

    if K == 1:
        return 0.0, 0.0

    top, second = uniq[-1], uniq[-2]
    rng = float(Hs.max() - Hs.min() + eps)
    margin_norm = float((top - second) / rng)
    phantom_index = float((K - 1) / K)
    return phantom_index, margin_norm

# ---- Funnel priors from calibration PCA(3) cloud ----
def build_funnel_priors_from_Y3(Y3):
    r = np.linalg.norm(Y3[:, :2], axis=1)
    r_n = (r - r.min()) / (r.max() - r.min() + 1e-8)
    r_grid = np.linspace(0, 1, 128)
    # monotone funnel depth profile and its slope proxy
    p = 1.3
    phi = 1.0 - np.power(r_grid, p)             # deeper near center
    g   = np.gradient(phi, r_grid)
    g   = np.abs(g)
    g   = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return r_grid, phi, g

# ---- Batch hidden state collection (mean over seq) ----
def collect_hidden_states(model, tok, prompts, tap: int, k_last: int = 8):
    with torch.no_grad():
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
        out = model(**enc, output_hidden_states=True)
        hs  = out.hidden_states[tap]                  # (B, T, d)
        k   = min(k_last, hs.shape[1])
        # H   = hs[:, -k:, :].mean(1).cpu().numpy().astype(float)
        H = hs[:, -min(16, hs.shape[1]):, :].mean(1).cpu().numpy().astype(float)  # set k_last=16
    return H


# ---- Phantom metrics from eval PCA3 (no second PCA fit) ----
# --- robust density-peak metric over PC1/PC2 (no extra deps) ---
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

def phantom_metrics_from_Y3(Y3, nbins=120, sigma=2.0, peak_min_dist=3, eps=1e-9):
    """
    Returns:
      pi           : phantom index = (num_peaks-1)/num_peaks
      margin_norm  : (top_peak - 2nd_peak) / dynamic_range   [0..1]
    Notes:
      - Work on the *smoothed density* Hs
      - Peaks = local maxima found via max filter, borders excluded
      - If <2 peaks, return pi=0, margin_norm=0 (no decision)
    """
    X2 = Y3[:, :2]
    x, y = X2[:, 0], X2[:, 1]
    H, xe, ye = np.histogram2d(x, y, bins=nbins)
    Hs = gaussian_filter(H, sigma=sigma)

    # local maxima mask
    Hmax = maximum_filter(Hs, size=(peak_min_dist, peak_min_dist), mode="nearest")
    peaks_mask = (Hs == Hmax)

    # exclude borders (avoid false plateaus at edges)
    peaks_mask[[0,-1], :] = False
    peaks_mask[:, [0,-1]] = False

    peaks = Hs[peaks_mask].ravel()
    if peaks.size < 2:
        return 0.0, 0.0

    peaks.sort()                 # ascending
    top, second = peaks[-1], peaks[-2]
    rng = float(Hs.max() - Hs.min() + eps)
    margin_norm = float((top - second) / rng)
    pi = float((peaks.size - 1) / peaks.size)
    return pi, margin_norm



def render_Y3(Y3, out="llm_pca3.png"):
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Y3[:,0], Y3[:,1], Y3[:,2], s=15, alpha=0.6)
    ax.set_title("LLM PCA(3) shadow manifold (eval)")
    plt.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

def token_inward_trend(model, tok, prompt, tap, pca):
    with torch.no_grad():
        enc = tok(prompt, return_tensors="pt")
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[tap][0].cpu().numpy()   # [T, d]
        Y  = pca.transform(hs)                         # [T, 3]
        R  = np.linalg.norm(Y[:, :2], axis=1)
        Rn = (R - R.min()) / (R.max() - R.min() + 1e-8)
        diffs = np.diff(Rn)
        return float((diffs < 0).sum() / max(1, len(diffs)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--tap",   type=int, default=-3, help="hidden layer index (negative from end)")
    ap.add_argument("--calib", type=str, required=True)
    ap.add_argument("--eval",  type=str, required=True)
    ap.add_argument("--render_well", action="store_true")
    ap.add_argument("--out_json", type=str, default="llm_shadow.json")
    args = ap.parse_args()

    torch.set_grad_enabled(False)

    tok   = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval(); model.to("cpu")

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    with open(args.calib) as f: calib_prompts = [ln.strip() for ln in f if ln.strip()]
    with open(args.eval)  as f: eval_prompts  = [ln.strip() for ln in f if ln.strip()]

    # 1) Collect calibration hidden states at tap; fit PCA(3)
    Hc = collect_hidden_states(model, tok, calib_prompts, args.tap, k_last=16)     # [Nc, d]
    pca = PCA(n_components=3, whiten=True, random_state=0)
    Yc  = pca.fit_transform(Hc)                                         # [Nc, 3]

    phantom_index, margin_norm = phantom_metrics_from_Y3_density_peaks(
        Yc, nbins=100, sigma=2.0, tau=0.30, nms_radius=4, prom_frac=0.02
    )
    print(f"[DBG2] pi={phantom_index:.3f} margin_norm={margin_norm:.3f}")
            
    # Build funnel priors on calibration cloud
    r_grid, phi_cal, g_cal = build_funnel_priors_from_Y3(Yc)

    # 2) Eval hidden states -> project with same PCA
    He = collect_hidden_states(model, tok,  eval_prompts,  args.tap, k_last=16)      # [Ne, d]
    Ye = pca.transform(He)                                              # [Ne, 3]

    # 4) Stage-11 well score S = 0.05*phi + 0.25*g^2 (use eval radii against calib priors)
    rmin, rmax, r_grid, phi_cal, g_cal = funnel_priors_from_calib_density(Yc)
        
    # inward trend over eval order
    R = np.linalg.norm(Ye[:, :2], axis=1).astype(float)
    if R.size < 2:
        r_trend_tokens = 0.0
    else:
        Rn = (R - R.min()) / (R.max() - R.min() + 1e-8)
        diffs = np.diff(Rn)
        r_trend_tokens = float((diffs < 0).sum() / max(1, diffs.size))

    r_lo, r_hi = safe_radius_bounds(Yc, Ye, lo=5.0, hi=95.0)  # percentiles
    # Normalize eval radii to grid domain safely
    R_eval = np.linalg.norm(Ye[:, :2], axis=1).astype(float)
    Rn_eval = (R_eval - r_lo) / (r_hi - r_lo + 1e-8)
    Rn_eval = np.clip(Rn_eval, 0.0, 1.0)
    
    # Build a normalized x-grid for interpolation (0..1)
    t_grid = (r_grid - rmin) / (rmax - rmin + 1e-8)
    phi_e = safe_interp01(r_grid, phi_cal, Rn_eval)
    g_e   = safe_interp01(r_grid, g_cal,   Rn_eval)
    
    S = 0.05 * phi_e + 0.25 * (g_e ** 2)
    S_median = float(np.median(S))

    # 5) Simple inward-trend proxy over eval order (fraction of negative steps)
    diffs = np.diff(Rn)
    r_trend_proxy = float((diffs < 0).sum() / max(1, len(diffs)))

    trend_vals = [token_inward_trend(model, tok, p, args.tap, pca) for p in eval_prompts]
    r_trend_tokens = float(np.mean(trend_vals))
    
    # phantom_index, margin_norm = phantom_metrics_from_Y3(Ye, nbins=120, sigma=2.0, peak_min_dist=5)
    out = {
        "phantom_index": float(phantom_index),
        "margin_norm":   float(margin_norm),
        "S_median":      float(S_median),
        "r_trend_tokens": float(r_trend_tokens),
    }

    import matplotlib.pyplot as plt
    H, xe, ye = np.histogram2d(Yc[:,0], Yc[:,1], bins=120)
    from scipy.ndimage import gaussian_filter
    Hs = gaussian_filter(H, sigma=1.2)
    plt.imshow(Hs.T, origin="lower", aspect="auto")
    plt.title("Calib density (PC1/PC2)"); plt.colorbar(); plt.tight_layout()
    plt.savefig("calib_density_debug.png", dpi=140); plt.close()

    print(f"[DBG] rmin={rmin:.3g} rmax={rmax:.3g} dr≈{(rmax-rmin)/max(1,len(r_grid)):.3g} "
      f"phi∈[{phi_cal.min():.3f},{phi_cal.max():.3f}] g∈[{g_cal.min():.3f},{g_cal.max():.3f}]")

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("[SUMMARY]", json.dumps(out, indent=2))

    if args.render_well:
        render_Y3(Ye, out="llm_pca3_eval.png")

if __name__ == "__main__":
    main()
