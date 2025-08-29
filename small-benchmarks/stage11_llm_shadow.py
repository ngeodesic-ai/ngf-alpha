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
def collect_hidden_states(model, tok, prompts, tap: int):
    with torch.no_grad():
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
        out = model(**enc, output_hidden_states=True)
        hs  = out.hidden_states[tap]            # (batch, seq, d)
        H   = hs.mean(1).cpu().numpy().astype(float)  # (batch, d)
    return H

# ---- Phantom metrics from eval PCA3 (no second PCA fit) ----
def phantom_metrics_from_Y3(Y3, nbins=80, sigma=1.2):
    """
    Estimate phantom index and margin from a smoothed 2D density over PC1,PC2.
    - PI: fraction of minima that are not the deepest (count-1)/count
    - margin: difference between the two deepest minima in 'energy' units (U = -density)
    - margin_norm: margin normalized by dynamic range of U (0..1-ish)
    """
    X2 = Y3[:, :2]
    x, y = X2[:, 0], X2[:, 1]
    H, xe, ye = np.histogram2d(x, y, bins=nbins)
    Hs = gaussian_filter(H, sigma=sigma)

    U = -Hs  # higher density == deeper well -> lower energy
    # Find local minima (3x3 neighborhood)
    neigh = minimum_filter(U, size=3, mode="nearest")
    mins_mask = (U == neigh)
    mins = U[mins_mask].ravel()
    mins = np.sort(mins)

    if mins.size < 2:
        return 0.0, 0.0, 0.0

    # Phantom index: everything except the deepest basin
    pi = float((mins.size - 1) / mins.size)

    # Margin between deepest and 2nd deepest minima
    margin = float(mins[1] - mins[0])

    # Normalize by dynamic range for a stable threshold
    rng = float(U.max() - U.min() + 1e-8)
    margin_norm = float(margin / rng)

    return pi, margin, margin_norm

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

    tok   = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval(); model.to("cpu")

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    with open(args.calib) as f: calib_prompts = [ln.strip() for ln in f if ln.strip()]
    with open(args.eval)  as f: eval_prompts  = [ln.strip() for ln in f if ln.strip()]

    # 1) Collect calibration hidden states at tap; fit PCA(3)
    Hc = collect_hidden_states(model, tok, calib_prompts, args.tap)     # [Nc, d]
    pca = PCA(n_components=3, whiten=True, random_state=0)
    Yc  = pca.fit_transform(Hc)                                         # [Nc, 3]

    # Build funnel priors on calibration cloud
    r_grid, phi_cal, g_cal = build_funnel_priors_from_Y3(Yc)

    # 2) Eval hidden states -> project with same PCA
    He = collect_hidden_states(model, tok, eval_prompts, args.tap)      # [Ne, d]
    Ye = pca.transform(He)                                              # [Ne, 3]

    # 3) Phantom metrics from eval PCA(3)
    phantom_index, margin_raw, margin_norm = phantom_metrics_from_Y3(Ye, nbins=80, sigma=1.2)

    # 4) Stage-11 well score S = 0.05*phi + 0.25*g^2 (use eval radii against calib priors)
    R  = np.linalg.norm(Ye[:, :2], axis=1)
    Rn = (R - R.min()) / (R.max() - R.min() + 1e-8)
    phi_e = np.interp(Rn, r_grid, phi_cal)
    g_e   = np.interp(Rn, r_grid, g_cal)
    S     = 0.05 * phi_e + 0.25 * (g_e ** 2)
    S_median = float(np.median(S))

    # 5) Simple inward-trend proxy over eval order (fraction of negative steps)
    diffs = np.diff(Rn)
    r_trend_proxy = float((diffs < 0).sum() / max(1, len(diffs)))

    trend_vals = [token_inward_trend(model, tok, p, args.tap, pca) for p in eval_prompts]
    r_trend_tokens = float(np.mean(trend_vals))
    
    out = {
        "phantom_index": phantom_index,
        "margin_norm":   margin_norm,
        "S_median":      S_median,
        "r_trend_tokens": r_trend_tokens
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("[SUMMARY]", json.dumps(out, indent=2))

    if args.render_well:
        render_Y3(Ye, out="llm_pca3_eval.png")

if __name__ == "__main__":
    main()
