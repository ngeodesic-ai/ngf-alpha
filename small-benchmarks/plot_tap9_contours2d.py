# Create a "well-style" single-panel contour map that matches the patent-like color scheme
# (blue well in the center, red outside), with light contours.
# It computes PCA->2D on post-warp (or pre/combined), builds a smoothed density,
# then plots a diverging map of negative density so the well is blue.
#

"""
python3 plot_tap9_contours2d.py \   
  --post "$OUT/tap9_post.npy" \
  --out_png "$OUT/tap9_semantic_well_map.png" \
  --out_pdf "$OUT/tap9_semantic_well_map.pdf" \
  --fit_on post \
  --sample 80000 \
  --bins 220 \
  --sigma 2.0 \
  --clip_q 0.01 \
  --levels 14
"""


import argparse, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter

def pca_fit_transform(X, k=2):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:k].T
    Z = Xc @ W
    return Z, W

def pca_transform(X, W):
    Xc = X - X.mean(axis=0, keepdims=True)
    return Xc @ W

def subsample(X, n, seed=0):
    if X is None: return None
    N = X.shape[0]
    if N <= n: return np.asarray(X)
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=n, replace=False)
    idx.sort()
    return np.asarray(X[idx])

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

def apply_clip(Z, clip_q=None, bounds=None):
    if Z is None: return None
    if clip_q is not None:
        x_lo, x_hi = np.quantile(Z[:,0], [clip_q, 1-clip_q])
        y_lo, y_hi = np.quantile(Z[:,1], [clip_q, 1-clip_q])
        if bounds is None: bounds = [x_lo, x_hi, y_lo, y_hi]
        else:
            bounds[0] = max(bounds[0], x_lo) if bounds[0] is not None else x_lo
            bounds[1] = min(bounds[1], x_hi) if bounds[1] is not None else x_hi
            bounds[2] = max(bounds[2], y_lo) if bounds[2] is not None else y_lo
            bounds[3] = min(bounds[3], y_hi) if bounds[3] is not None else y_hi
    if bounds is not None:
        x_min,x_max,y_min,y_max = bounds
        m = np.ones(len(Z), dtype=bool)
        if x_min is not None: m &= (Z[:,0] >= x_min)
        if x_max is not None: m &= (Z[:,0] <= x_max)
        if y_min is not None: m &= (Z[:,1] >= y_min)
        if y_max is not None: m &= (Z[:,1] <= y_max)
        Z = Z[m]
    return Z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=False, help="tap9_pre.npy (N,C)")
    ap.add_argument("--post", required=True, help="tap9_post.npy (N,C)")
    ap.add_argument("--out_png", default="tap9_contour_well.png")
    ap.add_argument("--out_pdf", default="tap9_contour_well.pdf")
    ap.add_argument("--sample", type=int, default=80000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit_on", choices=["post","pre","combined"], default="post")
    ap.add_argument("--bins", type=int, default=220)
    ap.add_argument("--sigma", type=float, default=2.0, help="Gaussian smoothing (pixels)")
    ap.add_argument("--clip_q", type=float, default=0.01, help="Quantile clip per axis")
    ap.add_argument("--levels", type=int, default=14)
    args = ap.parse_args()

    post = np.load(args.post, mmap_mode="r")
    pre = np.load(args.pre, mmap_mode="r") if args.pre else None

    # Subsample
    rng = np.random.default_rng(args.seed)
    def sub(X): 
        if X is None: return None
        N = X.shape[0]
        if N <= args.sample: return np.asarray(X)
        idx = rng.choice(N, size=args.sample, replace=False); idx.sort()
        return np.asarray(X[idx])
    post_s = sub(post)
    pre_s  = sub(pre)

    # PCA
    if args.fit_on == "combined" and pre_s is not None:
        Zc, W = pca_fit_transform(np.vstack([pre_s, post_s]), k=2)
        Z_post = Zc[-post_s.shape[0]:]
    elif args.fit_on == "pre" and pre_s is not None:
        Z_pre, W = pca_fit_transform(pre_s, k=2)
        Z_post = pca_transform(post_s, W)
    else:
        Z_post, W = pca_fit_transform(post_s, k=2)

    # Clip outliers for stability
    Z_post = apply_clip(Z_post, clip_q=args.clip_q)

    # Build smoothed density
    H, extent = hist2d(Z_post, bins=args.bins)
    P = normalize(H)
    if args.sigma and args.sigma > 0:
        P = gaussian_filter(P, sigma=args.sigma, mode="nearest")

    # Convert to "well" potential so center is BLUE, outskirts RED
    # We use negative, scaled to [-1, 1] for a clean diverging map.
    F = -(P - P.max())
    F = F / max(1e-12, np.abs(F).max())

    # Render
    x_min, x_max, y_min, y_max = extent
    fig = plt.figure(figsize=(7.5,6.2))
    ax = fig.add_subplot(1,1,1)

    # Use RdBu_r so lows (blue) are wells, highs (red) are ridges
    im = ax.contourf(
        np.linspace(x_min, x_max, F.shape[1]),
        np.linspace(y_min, y_max, F.shape[0]),
        F, levels=args.levels, cmap="RdBu_r", alpha=0.9
    )
    # light contour lines
    ax.contour(
        np.linspace(x_min, x_max, F.shape[1]),
        np.linspace(y_min, y_max, F.shape[0]),
        F, levels=args.levels, colors="k", linewidths=0.3, alpha=0.35
    )
    ax.set_title("Tap −9 Post-warp: Semantic Well (PCA-2)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

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

