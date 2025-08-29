#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-11 - LLM Shadow Probe v4

Adds:
 - --phantom_backend {hist2d, radial_kde}
 - 2-D NMS (--nms_radius) for hist2d backend
 - Tunable smoothing (--sigma) for hist2d
 - Optional radial warp (--radial_warp, --warp_gamma)

Retains Step-2 Go/No-Go gate and summary JSON.
"""
import argparse, json, numpy as np
from typing import Tuple, List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def collect_hidden_states(model, tok, prompts, tap: int, mode="lastk", k_last=16):
    with torch.no_grad():
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True)
        out = model(**enc, output_hidden_states=True)
        hs  = out.hidden_states[tap].cpu().numpy()  # (B, T, D)
        if mode == "mean":
            H = hs.mean(1)
        else:
            k = np.minimum(k_last, hs.shape[1])
            H = hs[:, -k:, :].mean(1)
        return H.astype(float)

def build_funnel_priors_from_Y3(Y3):
    r = np.linalg.norm(Y3[:, :2], axis=1)
    r_n = (r - r.min()) / (r.max() - r.min() + 1e-8)
    r_grid = np.linspace(0, 1, 128)
    p = 1.3
    phi = 1.0 - np.power(r_grid, p)
    g   = np.gradient(phi, r_grid)
    g   = np.abs(g)
    g   = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return r_grid, phi, g

def radial_warp(Y3, gamma=0.85):
    X = Y3.copy()
    r = np.linalg.norm(X[:, :2], axis=1) + 1e-12
    s = (r ** gamma) / r
    X[:,0] *= s
    X[:,1] *= s
    return X

def _find_minima(U, eps=1e-9, min_prom=0.0):
    h, w = U.shape
    mins = []
    for i in range(1, h-1):
        for j in range(1, w-1):
            c = U[i, j]
            neigh = U[i-1:i+2, j-1:j+2].copy()
            neigh[1,1] = c + 1e9
            if (c + eps < neigh).all():
                if min_prom > 0.0 and c - neigh.mean() > -min_prom:
                    continue
                mins.append((i, j, c))
    return mins

def _nms_2d(mins, radius=3):
    mins = sorted(mins, key=lambda t: t[2])  # deepest first
    kept = []
    taken = np.zeros(len(mins), dtype=bool)
    for a, (ia, ja, ua) in enumerate(mins):
        if taken[a]: 
            continue
        kept.append((ia, ja, ua))
        for b in range(a+1, len(mins)):
            if taken[b]:
                continue
            ib, jb, ub = mins[b]
            if (ia-ib)**2 + (ja-jb)**2 <= radius*radius:
                taken[b] = True
    return kept

def phantom_hist2d(Y3, nbins=160, sigma=3.5, min_prom=0.02, nms_radius=4):
    X2 = Y3[:, :2]
    x, y = X2[:, 0], X2[:, 1]
    H, xe, ye = np.histogram2d(x, y, bins=nbins)
    cov = np.cov(np.vstack([x, y]))
    w, V = np.linalg.eigh(cov)          # columns of V are eigenvectors
    # project grid to eigenbasis; emulate anisotropic blur by two passes
    Hs = gaussian_filter(H, sigma=(sigma*1.6, sigma*0.8))  # simple anisotropic approx
    U  = -Hs
    mins = _find_minima(U, min_prom=min_prom)
    mins = _nms_2d(mins, radius=nms_radius)
    if not mins:
        return 0.0, 0.0, 0.0, 0
    vals = np.array(sorted([m[2] for m in mins]))
    K = len(vals)
    if K == 1:
        return 0.0, 0.0, 0.0, 1
    pi = float((K - 1) / K)
    margin_raw = float(vals[1] - vals[0])
    rng = float(U.max() - U.min() + 1e-8)
    margin_norm = float(margin_raw / rng)
    return pi, margin_raw, margin_norm, K

def phantom_radial_kde(Y3, bandwidth=0.05):
    R = np.linalg.norm(Y3[:, :2], axis=1)
    Rn = (R - R.min()) / (R.max() - R.min() + 1e-8)
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(Rn[:,None])
    grid = np.linspace(0,1,200)[:,None]
    logd = kde.score_samples(grid)
    U = -logd
    mins = []
    for i in range(1, len(U)-1):
        if U[i] < U[i-1] and U[i] < U[i+1]:
            mins.append(U[i])
    if not mins:
        return 0.0, 0.0, 0.0, 0
    mins = np.sort(np.array(mins))
    K = len(mins)
    if K == 1:
        return 0.0, 0.0, 0.0, 1
    pi = float((K - 1) / K)
    margin_raw = float(mins[1]-mins[0])
    margin_norm = float(margin_raw/(U.max()-U.min()+1e-8))
    return pi, margin_raw, margin_norm, K

def detector_stability_hist(Ye) -> bool:
    def stable(vals, tol=0.05):
        arr = np.asarray(vals, float)
        if arr.size <= 1: 
            return True
        return (arr.max() - arr.min()) <= tol * (abs(arr.mean()) + 1e-9)
    pis, mags, Ks = [], [], []
    for nb in (120, 160, 180):
        for sg in (2.5, 3.0, 3.5):
            piX, _, mX, KX = phantom_hist2d(Ye, nbins=nb, sigma=sg, min_prom=0.02, nms_radius=4)
            pis.append(piX); mags.append(mX); Ks.append(KX)
    ok = (stable(pis) and stable(mags) and len(set(Ks)) == 1)
    return not ok

def token_inward_trend(model, tok, prompt, tap, pca) -> float:
    with torch.no_grad():
        enc = tok(prompt, return_tensors="pt", truncation=True)
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[tap][0].cpu().numpy()   # [T, D]
        Y  = pca.transform(hs)                         # [T, 3]
        R  = np.linalg.norm(Y[:, :2], axis=1)
        Rn = (R - R.min()) / (R.max() - R.min() + 1e-8)
        diffs = np.diff(Rn)
        return float((diffs < 0).sum() / max(1, len(diffs)))

def should_proceed_step2(m: Dict) -> Tuple[str, List[str]]:
    reasons = []
    def block(cond, msg):
        if cond: reasons.append("BLOCK: " + msg)
    block(m.get('phantom_index', 1.0) > 0.10, f"PI={m.get('phantom_index', float('nan')):.3f} > 0.10")
    block(m.get('margin_norm', 0.0)   < 0.04, f"margin_norm={m.get('margin_norm', float('nan')):.3f} < 0.04")
    if m.get('K', 1) > 1 and not m.get('margin_improved', False):
        block(True, f"K={m.get('K')} and margin did not improve after \u03c4\u2191/NMS\u2191")
    block(m.get('S_median', 0.0)      < 0.55, f"S_median={m.get('S_median', float('nan')):.2f} < 0.55")
    block(m.get('r_trend_tokens', 0.0)< 0.90, f"r_trend_tokens={m.get('r_trend_tokens', float('nan')):.3f} < 0.90")
    block(m.get('calibration_prompts', 0) < 300, f"calibration_prompts={m.get('calibration_prompts', 0)} < 300")
    block(m.get('k_last') not in {8,16,24}, f"k_last={m.get('k_last')} not in {{8,16,24}}")
    block(m.get('detector_unstable', False), "detector unstable (PI/margin/K oscillate)")
    block(m.get('dual_lobes', False), "density plot shows \u22652 lobes")
    block(m.get('r_span_collapsed', False), "rmin\u2248rmax (collapsed radial span)")
    block(m.get('margin_tied', False), "peak heights tied \u2192 margin_norm\u22480")
    all_pass = (
        m.get('phantom_index', 1.0) <= 0.10 and
        m.get('margin_norm', 0.0)   >= 0.04 and
        m.get('S_median', 0.0)      >= 0.55 and
        m.get('r_trend_tokens', 0.0)>= 0.90 and
        m.get('K', 99)              == 1 and
        m.get('calibration_prompts', 0) >= 300 and
        m.get('k_last') in {8,16,24} and
        bool(m.get('dominant_basin', False))
    )
    decision = "GO" if (not reasons and all_pass) else "NO-GO"
    return decision, reasons

def render_Y3(Y3, out="llm_pca3_eval.png"):
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(Y3[:,0], Y3[:,1], Y3[:,2], s=15, alpha=0.6)
    ax.set_title("LLM PCA(3) shadow manifold (eval)")
    plt.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--tap",   type=int, default=-3)
    ap.add_argument("--calib", type=str, required=True)
    ap.add_argument("--eval",  type=str, required=True)
    ap.add_argument("--pool_mode", type=str, default="lastk", choices=["mean","lastk"])
    ap.add_argument("--k_last", type=int, default=16)
    ap.add_argument("--render_well", action="store_true")
    ap.add_argument("--out_json", type=str, default="llm_shadow.json")
    # New flags
    ap.add_argument("--phantom_backend", type=str, default="hist2d", choices=["hist2d","radial_kde"])
    ap.add_argument("--nms_radius", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=3.5)
    ap.add_argument("--radial_warp", type=int, default=0, help="1 to enable radial warp before phantom detection")
    ap.add_argument("--warp_gamma", type=float, default=0.85)
    ap.add_argument("--kde_bw", type=float, default=0.05, help="bandwidth for radial_kde backend")
    args = ap.parse_args()

    tok   = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval(); model.to("cpu")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    with open(args.calib) as f: calib_prompts = [ln.strip() for ln in f if ln.strip()]
    with open(args.eval)  as f: eval_prompts  = [ln.strip() for ln in f if ln.strip()]

    calib_count = len(calib_prompts)
    eval_count  = len(eval_prompts)

    if calib_count < 300:
        print(f"[GATE] NO-GO: calibration_prompts={calib_count} < 300")
    if eval_count < 50:
        print(f"[GATE] NO-GO: eval_prompts={eval_count} < 50")

    Hc  = collect_hidden_states(model, tok, calib_prompts, args.tap, mode=args.pool_mode, k_last=args.k_last)
    pca = PCA(n_components=3, whiten=True, random_state=0)
    Yc  = pca.fit_transform(Hc)
    r_grid, phi_cal, g_cal = build_funnel_priors_from_Y3(Yc)

    He = collect_hidden_states(model, tok, eval_prompts, args.tap, mode=args.pool_mode, k_last=args.k_last)
    Ye = pca.transform(He)

    from sklearn.cluster import DBSCAN
    Y2 = Ye[:, :2]
    db = DBSCAN(eps=0.12, min_samples=3).fit(Y2)  # eps ~ 10–15% of unit box
    labels = db.labels_
    # if -1 (noise) dominates or single cluster already, skip; else keep largest non-noise
    if (labels >= 0).any():
        counts = [(lab, (labels==lab).sum()) for lab in set(labels) if lab >= 0]
        if counts:
            main_lab = max(counts, key=lambda t: t[1])[0]
            Ye = Ye[labels==main_lab]
    
    Ye_w = radial_warp(Ye, gamma=args.warp_gamma) if args.radial_warp == 1 else Ye

    R  = np.linalg.norm(Ye_w[:, :2], axis=1)
    Rn = (R - R.min()) / (R.max() - R.min() + 1e-8)
    phi_e = np.interp(Rn, r_grid, phi_cal)
    g_e   = np.interp(Rn, r_grid, g_cal)
    S     = 0.05 * phi_e + 0.25 * (g_e ** 2)
    S_median = float(np.median(S))

    diffs = np.diff(Rn)
    r_trend_proxy  = float((diffs < 0).sum() / max(1, len(diffs)))
    trend_vals = [token_inward_trend(model, tok, p, args.tap, pca) for p in eval_prompts]
    r_trend_tokens = float(np.mean(trend_vals)) if trend_vals else 0.0

    if args.phantom_backend == "hist2d":
        pi2, mraw2, mnorm2, K2 = phantom_hist2d(Ye_w, nbins=160, sigma=args.sigma, min_prom=0.02, nms_radius=args.nms_radius)
        detector_unstable = detector_stability_hist(Ye_w)
    else:
        pi2, mraw2, mnorm2, K2 = phantom_radial_kde(Ye_w, bandwidth=args.kde_bw)
        detector_unstable = False

    margin_improved = True
    phantom_index  = float(pi2)
    margin_norm    = float(mnorm2)

    R_abs  = np.linalg.norm(Ye_w[:, :2], axis=1)
    rmin, rmax = float(R_abs.min()), float(R_abs.max())
    dual_lobes = (K2 > 1)
    r_span_collapsed = (abs(rmax - rmin) < 1e-6)
    margin_tied = (abs(mraw2) < 1e-9) or (mnorm2 < 1e-6)
    dominant_basin = not dual_lobes

    gate_input = dict(
        phantom_index=phantom_index,
        margin_norm=margin_norm,
        S_median=S_median,
        r_trend_tokens=r_trend_tokens,
        K=K2,
        calibration_prompts=calib_count,
        k_last=args.k_last,
        dominant_basin=dominant_basin,
        margin_improved=margin_improved,
        detector_unstable=detector_unstable,
        dual_lobes=dual_lobes,
        r_span_collapsed=r_span_collapsed,
        margin_tied=margin_tied
    )
    decision, reasons = should_proceed_step2(gate_input)

    out = dict(
        model=args.model,
        tap=args.tap,
        pool_mode=args.pool_mode,
        k_last=args.k_last,
        phantom_backend=args.phantom_backend,
        radial_warp=bool(args.radial_warp),
        warp_gamma=args.warp_gamma,
        nms_radius=args.nms_radius,
        sigma=args.sigma,
        kde_bw=args.kde_bw,
        calibration_prompts=calib_count,
        eval_prompts=eval_count,
        phantom_index=phantom_index,
        margin_norm=margin_norm,
        S_median=S_median,
        r_trend_tokens=r_trend_tokens,
        K=K2,
        margin_improved=margin_improved,
        detector_unstable=detector_unstable,
        dual_lobes=dual_lobes,
        r_span=[rmin, rmax],
        margin_tied=margin_tied,
        dominant_basin=dominant_basin,
        decision=decision,
        no_go_reasons=reasons
    )

    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("[SUMMARY]", json.dumps(out, indent=2))

    if args.render_well:
        render_Y3(Ye_w, out="llm_pca3_eval.png")

if __name__ == "__main__":
    main()
