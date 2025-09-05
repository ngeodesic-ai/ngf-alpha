# plot_tap9_semantic_well.py
# Visualize the *semantic well* at tap −9 as a smooth funnel surface (no scatter).
# - PCA → 3D (fit on post by default)
# - Robust radial funnel fit ẑ(r)
# - Rotational surface render
# - Phantom index in title + sidecar TXT
#
# Usage:
#   python3 plot_tap9_semantic_well.py \
#     --post results/gpt2_n1000/tap9_post.npy \
#     --out_png results/gpt2_n1000/tap9_semantic_well.png \
#     --out_pdf results/gpt2_n1000/tap9_semantic_well.pdf \
#     --fit_on post --deg 2 --sample 30000
#
# Side-by-side pre vs post (surfaces only):
#   python3 plot_tap9_semantic_well.py \
#     --pre  results/gpt2_n1000/tap9_pre.npy \
#     --post results/gpt2_n1000/tap9_post.npy \
#     --render_pre_surface 1 \
#     --out_png results/gpt2_n1000/tap9_semantic_well_compare.png \
#     --out_pdf results/gpt2_n1000/tap9_semantic_well_compare.pdf

"""
OUT=results/gpt2_n1000
python3 plot_tap9_funnel3d.py \
  --post "$OUT/tap9_post.npy" \
  --out_png "$OUT/tap9_semantic_well.png" \
  --out_pdf "$OUT/tap9_semantic_well.pdf" \
  --fit_on post --deg 2 --sample 30000
"""

import argparse, os, numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------- PCA helpers ----------
def pca_fit_transform(X, k=3):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:k].T  # [C, k]
    Z = Xc @ W
    return Z, W

def pca_transform(X, W):
    Xc = X - X.mean(axis=0, keepdims=True)
    return Xc @ W

# ---------- Funnel profile ----------
def robust_polyfit_r_z(R, Z, deg=2, iters=3, trim=0.1):
    """Fit z = poly(r) robustly by iterative trimming (ascending powers)."""
    r = R.copy().reshape(-1)
    z = Z.copy().reshape(-1)
    mask = np.isfinite(r) & np.isfinite(z)
    r, z = r[mask], z[mask]
    for _ in range(iters):
        V = np.vander(r, deg + 1, increasing=True)
        coefs, *_ = np.linalg.lstsq(V, z, rcond=None)
        z_hat = V @ coefs
        resid = np.abs(z - z_hat)
        thr = np.quantile(resid, 0.9)
        keep = resid <= thr
        r, z = r[keep], z[keep]
    return coefs

def eval_poly_anyshape(coefs, r):
    r = np.asarray(r)
    shape = r.shape
    r_flat = r.reshape(-1)
    V = np.vander(r_flat, len(coefs), increasing=True)
    z_flat = V @ np.asarray(coefs)
    return z_flat.reshape(shape)

def phantom_index_from_hist(R, nbins=64, smooth=3):
    """Simple multiwell proxy: mass outside primary radial mode."""
    r = R.reshape(-1)
    r = r[np.isfinite(r)]
    if r.size == 0: return 0.0
    hist, edges = np.histogram(r, bins=nbins)
    if smooth > 0:
        from numpy.lib.stride_tricks import sliding_window_view
        w = 2*smooth+1
        pad = np.pad(hist, (smooth, smooth), mode='edge')
        win = sliding_window_view(pad, w)
        hist = win.mean(axis=-1)
    i = int(np.argmax(hist))
    keep = np.zeros_like(hist, dtype=bool)
    keep[max(0,i-1):min(len(hist),i+2)] = True
    mass_total = hist.sum()
    mass_out = hist[~keep].sum()
    return float(mass_out / max(1e-9, mass_total))

def make_surface(coefs, r_max, n_r=96, n_th=144):
    r = np.linspace(0, r_max, n_r)
    th = np.linspace(0, 2*np.pi, n_th, endpoint=True)
    R, TH = np.meshgrid(r, th, indexing='ij')
    Z = eval_poly_anyshape(coefs, R)
    X = R * np.cos(TH)
    Y = R * np.sin(TH)
    return X, Y, Z

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=False, help="tap9_pre.npy (N,C)")
    ap.add_argument("--post", required=True, help="tap9_post.npy (N,C)")
    ap.add_argument("--out_png", default="tap9_semantic_well.png")
    ap.add_argument("--out_pdf", default="tap9_semantic_well.pdf")
    ap.add_argument("--sample", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deg", type=int, default=2, help="Polynomial degree for z(r)")
    ap.add_argument("--fit_on", choices=["post","pre","combined"], default="post",
                    help="Fit PCA axes on which set (default: post)")
    ap.add_argument("--render_pre_surface", type=int, default=0,
                    help="Also render a pre-warp surface (side-by-side)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Load
    post = np.load(args.post)
    pre = np.load(args.pre) if args.pre else None

    # Subsample
    def subsample(X):
        if X is None: return None
        if X.shape[0] <= args.sample: return X
        idx = rng.choice(X.shape[0], size=args.sample, replace=False)
        return X[idx]

    post_s = subsample(post)
    pre_s  = subsample(pre) if pre is not None else None

    # PCA
    if args.fit_on == "combined" and pre_s is not None:
        Zc, W = pca_fit_transform(np.vstack([pre_s, post_s]), k=3)
        Z_pre  = Zc[:pre_s.shape[0]] if pre_s is not None else None
        Z_post = Zc[-post_s.shape[0]:]
    elif args.fit_on == "pre" and pre_s is not None:
        Z_pre, W = pca_fit_transform(pre_s, k=3)
        Z_post = pca_transform(post_s, W)
    else:  # default: fit on post
        Z_post, W = pca_fit_transform(post_s, k=3)
        Z_pre = pca_transform(pre_s, W) if pre_s is not None else None

    # Fit funnel(s)
    r_post = np.sqrt(Z_post[:,0]**2 + Z_post[:,1]**2)
    coefs_post = robust_polyfit_r_z(r_post, Z_post[:,2], deg=args.deg)
    phi_post = phantom_index_from_hist(r_post)

    if Z_pre is not None and args.render_pre_surface:
        r_pre = np.sqrt(Z_pre[:,0]**2 + Z_pre[:,1]**2)
        coefs_pre = robust_polyfit_r_z(r_pre, Z_pre[:,2], deg=args.deg)
        phi_pre = phantom_index_from_hist(r_pre)
    else:
        coefs_pre = None
        phi_pre = None

    # Render (surfaces only)
    if coefs_pre is not None and args.render_pre_surface:
        fig = plt.figure(figsize=(12,5))
        ax1 = fig.add_subplot(1,2,1, projection='3d')
        ax2 = fig.add_subplot(1,2,2, projection='3d')

        r_max_pre  = np.quantile(np.sqrt(Z_pre[:,0]**2 + Z_pre[:,1]**2), 0.99)
        r_max_post = np.quantile(r_post, 0.99)

        X1,Y1,Z1 = make_surface(coefs_pre,  r_max_pre)
        X2,Y2,Z2 = make_surface(coefs_post, r_max_post)

        ax1.plot_surface(X1, Y1, Z1, linewidth=0, antialiased=False, alpha=0.9)
        ax2.plot_surface(X2, Y2, Z2, linewidth=0, antialiased=False, alpha=0.9)

        ax1.set_title(f"Pre-warp Funnel (surface)\nphantom≈{phi_pre:.3f}")
        ax2.set_title(f"Post-warp Semantic Well (surface)\nphantom≈{phi_post:.3f}")
        for ax in (ax1, ax2):
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        fig.tight_layout()
    else:
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(1,1,1, projection='3d')

        r_max_post = np.quantile(r_post, 0.99)
        X2,Y2,Z2 = make_surface(coefs_post, r_max_post)
        ax.plot_surface(X2, Y2, Z2, linewidth=0, antialiased=False, alpha=0.95)
        ax.set_title(f"Post-warp Semantic Well (surface)\nphantom≈{phi_post:.3f}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        fig.tight_layout()

    fig.savefig(args.out_png, dpi=240)
    pp = PdfPages(args.out_pdf); pp.savefig(fig, dpi=300); pp.close()

    # Sidecar with params
    meta_path = os.path.splitext(args.out_png)[0] + ".txt"
    with open(meta_path, "w") as f:
        f.write(f"coefs_post: {coefs_post.tolist()}\n")
        f.write(f"phantom_post: {phi_post:.6f}\n")
        if coefs_pre is not None:
            f.write(f"coefs_pre: {coefs_pre.tolist()}\n")
            f.write(f"phantom_pre: {phi_pre:.6f}\n")

    print(f"[WRITE] {args.out_png}")
    print(f"[WRITE] {args.out_pdf}")
    print(f"[WRITE] {meta_path}")

if __name__ == "__main__":
    main()
