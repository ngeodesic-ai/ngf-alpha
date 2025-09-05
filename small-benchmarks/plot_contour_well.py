#!/usr/bin/env python3
# plot_tap9_contour_well.py  —  pre/post “semantic well” contours (PCA-2)
# - If --pre is provided, renders PRE and POST side-by-side
# - Shared PCA axes (fit_on: pre|post|combined), shared extent, shared colormap scale
# - mmap loads, subsampling, quantile clipping, Gaussian smoothing
# - Saves PNG + PDF

import argparse, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter

"""
OUT=results/maxwarpC_tap9_noOutlier

python3 plot_contour_well.py \
  --pre "$OUT/tap-9_pre.npy" \
  --post "$OUT/tap-9_post.npy" \
  --out_png "$OUT/tap9_well_compare.png" \
  --out_pdf "$OUT/tap9_well_compare.pdf" \
  --fit_on post \
  --sample 80000 \
  --bins 220 \
  --sigma 2.0 \
  --clip_q 0.01 \
  --levels 14

OUT=results/gpt2_n1000  

python3 plot_contour_well.py \
  --pre "$OUT/tap9_pre.npy" \
  --post "$OUT/tap9_post.npy" \
  --out_png "$OUT/tap9_well_compare.png" \
  --out_pdf "$OUT/tap9_well_compare.pdf" \
  --fit_on post \
  --sample 80000 \
  --bins 220 \
  --sigma 2.0 \
  --clip_q 0.01 \
  --levels 14
  
"""

# ---------- PCA helpers ----------
def pca_fit_transform(X, k=2):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:k].T
    Z = Xc @ W
    return Z, W

def pca_transform(X, W):
    Xc = X - X.mean(axis=0, keepdims=True)
    return Xc @ W

# ---------- utils ----------
def subsample(X, n, seed=0):
    if X is None: return None
    N = X.shape[0]
    if N <= n: return np.asarray(X)
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=n, replace=False); idx.sort()
    return np.asarray(X[idx])

def apply_clip_pair(Z_pre, Z_post, clip_q=0.01):
    """Compute joint quantile bounds on (pre ∪ post) and clip both to them."""
    Zs = [z for z in [Z_pre, Z_post] if z is not None]
    if not Zs: return Z_pre, Z_post, None
    Zcat = np.vstack(Zs)
    x_lo, x_hi = np.quantile(Zcat[:,0], [clip_q, 1-clip_q])
    y_lo, y_hi = np.quantile(Zcat[:,1], [clip_q, 1-clip_q])
    def mask(Z):
        if Z is None: return None
        m = (Z[:,0]>=x_lo)&(Z[:,0]<=x_hi)&(Z[:,1]>=y_lo)&(Z[:,1]<=y_hi)
        return Z[m]
    return mask(Z_pre), mask(Z_post), (x_lo, x_hi, y_lo, y_hi)

def hist2d(Z, bins=220, extent=None):
    x = Z[:,0]; y = Z[:,1]
    if extent is None:
        x_min, x_max = np.quantile(x, [0.01, 0.99])
        y_min, y_max = np.quantile(y, [0.01, 0.99])
    else:
        x_min, x_max, y_min, y_max = extent
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=[[x_min,x_max],[y_min,y_max]])
    return H.T, (x_min, x_max, y_min, y_max)

def normalize(H):
    s = H.sum()
    return H / max(1e-12, s)

def to_well_field(P, sigma):
    if sigma and sigma > 0:
        P = gaussian_filter(P, sigma=sigma, mode="nearest")
    # “well” = negative of centered density; scale to [-1, 1]
    F = -(P - P.max())
    F /= max(1e-12, np.abs(F).max())
    return F

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=False, help="tap9_pre.npy (N,C)")
    ap.add_argument("--post", required=True,  help="tap9_post.npy (N,C)")
    ap.add_argument("--out_png", default="tap9_contour_well.png")
    ap.add_argument("--out_pdf", default="tap9_contour_well.pdf")
    ap.add_argument("--sample", type=int, default=80000)
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--fit_on", choices=["post","pre","combined"], default="post")
    ap.add_argument("--bins",   type=int, default=220)
    ap.add_argument("--sigma",  type=float, default=2.0, help="Gaussian smoothing (pixels)")
    ap.add_argument("--clip_q", type=float, default=0.01, help="Joint quantile clip (both panels)")
    ap.add_argument("--levels", type=int, default=14)
    args = ap.parse_args()

    # mmap loads
    post = np.load(args.post, mmap_mode="r")
    pre  = np.load(args.pre,  mmap_mode="r") if args.pre else None

    # Subsample
    rng = np.random.default_rng(args.seed)
    def sub(X):
        if X is None: return None
        N = X.shape[0]
        if N <= args.sample: return np.asarray(X)
        idx = rng.choice(N, size=args.sample, replace=False); idx.sort()
        return np.asarray(X[idx])
    post_s, pre_s = sub(post), sub(pre)

    # PCA → shared frame
    if args.fit_on == "combined" and pre_s is not None:
        Zc, W = pca_fit_transform(np.vstack([pre_s, post_s]), k=2)
        Z_pre  = Zc[:pre_s.shape[0]] if pre_s is not None else None
        Z_post = Zc[-post_s.shape[0]:]
    elif args.fit_on == "pre" and pre_s is not None:
        Z_pre, W  = pca_fit_transform(pre_s, k=2)
        Z_post    = pca_transform(post_s, W)
    else:  # fit on post
        Z_post, W = pca_fit_transform(post_s, k=2)
        Z_pre     = pca_transform(pre_s, W) if pre_s is not None else None

    # Joint outlier clipping → both panels get same bounds
    Z_pre, Z_post, _ = apply_clip_pair(Z_pre, Z_post, clip_q=args.clip_q)

    # Common extent (quantile padded over both)
    if Z_pre is not None:
        x_all = np.concatenate([Z_pre[:,0], Z_post[:,0]])
        y_all = np.concatenate([Z_pre[:,1], Z_post[:,1]])
    else:
        x_all, y_all = Z_post[:,0], Z_post[:,1]
    x_min, x_max = np.quantile(x_all, [0.01, 0.99])
    y_min, y_max = np.quantile(y_all, [0.01, 0.99])
    extent = (x_min, x_max, y_min, y_max)

    # Densities → well fields
    H_post, _ = hist2d(Z_post, bins=args.bins, extent=extent)
    P_post    = normalize(H_post)
    F_post    = to_well_field(P_post, sigma=args.sigma)

    F_pre = None
    if Z_pre is not None:
        H_pre, _ = hist2d(Z_pre, bins=args.bins, extent=extent)
        P_pre    = normalize(H_pre)
        F_pre    = to_well_field(P_pre, sigma=args.sigma)

    # Shared color scale for fair comparison
    vmin = -1.0; vmax = 1.0

    # Render
    def render(ax, F, title):
        x_min, x_max, y_min, y_max = extent
        im = ax.contourf(
            np.linspace(x_min, x_max, F.shape[1]),
            np.linspace(y_min, y_max, F.shape[0]),
            F, levels=args.levels, cmap="RdBu_r", vmin=vmin, vmax=vmax, alpha=0.9
        )
        ax.contour(
            np.linspace(x_min, x_max, F.shape[1]),
            np.linspace(y_min, y_max, F.shape[0]),
            F, levels=args.levels, colors="k", linewidths=0.3, alpha=0.35
        )
        ax.set_title(title)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        return im

    if F_pre is not None:
        fig = plt.figure(figsize=(12,5.4))
        ax1 = fig.add_subplot(1,2,1); ax2 = fig.add_subplot(1,2,2)
        im1 = render(ax1, F_pre,  "Tap −9 Pre-warp: Semantic Field (PCA-2)")
        im2 = render(ax2, F_post, "Tap −9 Post-warp: Semantic Well (PCA-2)")
        cbar = fig.colorbar(im2, ax=[ax1, ax2], fraction=0.046, pad=0.04)
        cbar.set_label("well depth (relative)")
    else:
        fig = plt.figure(figsize=(7.5,6.2))
        ax  = fig.add_subplot(1,1,1)
        im  = render(ax, F_post, "Tap −9 Post-warp: Semantic Well (PCA-2)")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("well depth (relative)")

    fig.tight_layout()
    fig.savefig(args.out_png, dpi=240)
    with PdfPages(args.out_pdf) as pp:
        pp.savefig(fig, dpi=320)
    plt.close(fig)

    print(f"[WRITE] {args.out_png}")
    print(f"[WRITE] {args.out_pdf}")

if __name__ == "__main__":
    main()
