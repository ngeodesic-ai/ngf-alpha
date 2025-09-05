
import argparse, os, numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

"""
python3 plot_tap9_pca3.py \
  --pre  "$OUT/tap9_pre.npy" \
  --post "$OUT/tap9_post.npy" \
  --out_png "$OUT/tap9_pca3_compare.png" 
"""

def pca_fit_transform(X, k=3):
    # X: [N, C], mean-center then take top-k PCs via SVD
    X = X - X.mean(axis=0, keepdims=True)
    # economy SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    W = Vt[:k].T  # [C, k]
    Z = X @ W     # [N, k]
    return Z, W

def pca_transform(X, W):
    X = X - X.mean(axis=0, keepdims=True)  # center with its own mean for visualization
    return X @ W

def jitter_points(Z, eps=1e-6):
    # tiny jitter to avoid exact overlaps
    if eps <= 0: return Z
    return Z + eps * np.random.randn(*Z.shape)

def plot_3d(ax, Z, title="", s=2, alpha=0.7):
    ax.scatter(Z[:,0], Z[:,1], Z[:,2], s=s, alpha=alpha, depthshade=False)
    ax.set_title(title)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True, help="Path to pre-warp hidden states .npy [N,C]")
    ap.add_argument("--post", required=True, help="Path to post-warp hidden states .npy [N,C]")
    ap.add_argument("--out_png", default="tap9_pca3_compare.png")
    ap.add_argument("--out_pdf", default="tap9_pca3_compare.pdf")
    ap.add_argument("--sample", type=int, default=20000, help="Max points to plot for each set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit_on", choices=["pre","combined"], default="pre",
                    help="Fit PCA on 'pre' only (default) or pre+post combined")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pre = np.load(args.pre)   # [N,C]
    post = np.load(args.post) # [N,C]

    # Subsample to manageable size
    def subsample(X, n):
        if X.shape[0] <= n: return X
        idx = rng.choice(X.shape[0], size=n, replace=False)
        return X[idx]
    pre_s  = subsample(pre, args.sample)
    post_s = subsample(post, args.sample)

    # Fit PCA
    if args.fit_on == "combined":
        combo = np.vstack([pre_s, post_s])
        Z_combo, W = pca_fit_transform(combo, k=3)
        # split back
        Z_pre  = Z_combo[:pre_s.shape[0]]
        Z_post = Z_combo[pre_s.shape[0]:]
    else:
        Z_pre, W = pca_fit_transform(pre_s, k=3)
        Z_post = pca_transform(post_s, W)

    # Tiny jitter (optional) to reduce overplotting artifacts
    Z_pre  = jitter_points(Z_pre, 1e-8)
    Z_post = jitter_points(Z_post, 1e-8)

    # Plot side-by-side PNG
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    plot_3d(ax1, Z_pre,  title="Tap −9 Pre‑warp (PCA‑3)")
    plot_3d(ax2, Z_post, title="Tap −9 Post‑warp (PCA‑3)")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=200)

    # Also save a PDF (vector)
    pp = PdfPages(args.out_pdf)
    pp.savefig(fig, dpi=300)
    pp.close()

    print(f"[WRITE] {args.out_png}")
    print(f"[WRITE] {args.out_pdf}")

if __name__ == "__main__":
    main()
