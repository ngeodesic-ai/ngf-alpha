
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 11 - 3D Manifold Visualization Pipeline
Loads a manifold dump (npz) from Stage11 v6/v7 runs and produces 3D visuals:
  - PCA(3D) scatter colored by energy U(t)
  - Optional "surface" triangulation for a mesh look
  - Per-primitive overlays from U_k(t, primitive)

Inputs (npz with keys):
  Y: (T x d) latent path over time
  U: (T,) total energy over time
  U_k: (T x K) per-primitive energy over time
  names: (K,) primitive names

Outputs (PNG):
  <outdir>/manifold_pca3_scatter.png
  <outdir>/manifold_pca3_mesh.png
  <outdir>/manifold_pca3_overlay_<prim>.png

Usage:
  python3 stage11_manifold_viz.py --npz dumps/v7_manifold.npz --out viz/ --surface --normalize
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import os

def pca3(X):
    X = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Z = X @ Vt[:3].T  # (T,3)
    return Z, S[:3]

def norm01(x):
    m = np.min(x)
    M = np.max(x)
    if M - m < 1e-12: return np.zeros_like(x)
    return (x - m) / (M - m)

def build_triangulation(Z3):
    # simple Delaunay in the (x,y) plane for a mesh look
    x, y = Z3[:,0], Z3[:,1]
    tri = Triangulation(x, y)
    return tri

def plot_scatter3(Z3, color, title, path):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Z3[:,0], Z3[:,1], Z3[:,2], c=color, s=10, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    plt.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def plot_mesh3(Z3, zvals, title, path):
    # Triangulate on XY; use PC3 as height; color by zvals
    tri = build_triangulation(Z3)
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(tri, Z3[:,2], cmap='viridis', alpha=0.9, linewidth=0.2, antialiased=True)
    c = ax.scatter(Z3[:,0], Z3[:,1], Z3[:,2], c=zvals, s=6, alpha=0.8)
    fig.colorbar(c, ax=ax, shrink=0.6, label="Energy (norm)")
    ax.set_title(title)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    plt.tight_layout()
    fig.savefig(path, dpi=320)
    plt.close(fig)

def plot_overlay(Z3, U_k, names, outdir):
    # One panel per primitive: scatter colored by that primitive's energy
    for j, name in enumerate(names):
        cj = norm01(U_k[:, j])
        title = f"PCA3 Overlay - {name}"
        path = os.path.join(outdir, f"manifold_pca3_overlay_{name}.png")
        plot_scatter3(Z3, cj, title, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to manifold npz (from --dump_manifold)")
    ap.add_argument("--out", default="viz", help="Output directory")
    ap.add_argument("--surface", action="store_true", help="Also render a mesh/tri-surface view")
    ap.add_argument("--normalize", action="store_true", help="Normalize Y features before PCA")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = np.load(args.npz, allow_pickle=True)
    Y = data["Y"]      # (T,d)
    U = data["U"]      # (T,)
    U_k = data["U_k"]  # (T,K)
    names = [str(s) for s in data["names"]]

    if args.normalize:
        Y = (Y - Y.mean(0, keepdims=True)) / (Y.std(0, keepdims=True) + 1e-9)

    Z3, S3 = pca3(Y)
    U_norm = norm01(U)

    # 1) Scatter colored by total energy
    plot_scatter3(Z3, U_norm, "PCA(3) - colored by total energy U(t)", os.path.join(args.out, "manifold_pca3_scatter.png"))

    # 2) Optional mesh surface
    if args.surface:
        plot_mesh3(Z3, U_norm, "PCA(3) - mesh view (trisurf) with energy coloring", os.path.join(args.out, "manifold_pca3_mesh.png"))

    # 3) Per-primitive overlays
    plot_overlay(Z3, U_k, names, args.out)

    # Save a tiny summary json
    summary = {
        "npz": args.npz,
        "out": args.out,
        "explained_variance_hint": [float(s) for s in S3],
        "primitives": names,
        "files": ["manifold_pca3_scatter.png"] + \
                 (["manifold_pca3_mesh.png"] if args.surface else []) + \
                 [f"manifold_pca3_overlay_{n}.png" for n in names]
    }
    with open(os.path.join(args.out, "viz_summary.json"), "w") as f:
        import json; json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
