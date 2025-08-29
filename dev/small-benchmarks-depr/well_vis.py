# well_vis.py
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

try:
    from scipy.spatial import Delaunay
    _HAVE_QHULL = True
except Exception:
    _HAVE_QHULL = False

@dataclass
class WellParams:
    # PCA / scaling
    whiten: bool = True
    # center estimation
    tau: float = 0.25         # softmin temperature for energy weighting
    # plane isotropization
    isotropize_xy: bool = True
    # radial funnel shaping
    sigma_scale: float = 0.80 # funnel width ~ 0.8 * median radius
    depth_scale: float = 1.25 # funnel depth multiplier
    mix_z: float = 0.15       # retain a bit of original z to preserve structure
    # lateral inhibition
    inhibit_k: int = 12       # neighbor count for local min scan
    inhibit_strength: float = 0.6  # how hard to suppress phantoms [0..1]
    # plotting
    point_alpha: float = 0.85
    trisurf_alpha: float = 0.65

def _softmin_center(X2: np.ndarray, energy: Optional[np.ndarray], tau: float) -> Tuple[np.ndarray, np.ndarray]:
    n = len(X2)
    if energy is None:
        w = np.ones(n) / n
    else:
        e = (energy - energy.min()) / (energy.std() + 1e-8)
        w = np.exp(-e / max(tau, 1e-6))
        w = w / (w.sum() + 1e-12)
    c = (w[:, None] * X2).sum(axis=0)
    return c, w

def _isotropize(X2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # linear map that whitens PC1–PC2 plane to unit covariance
    mu = X2.mean(axis=0)
    Y = X2 - mu
    C = (Y.T @ Y) / max(len(Y)-1, 1)
    evals, evecs = np.linalg.eigh(C)
    T = evecs @ np.diag(1.0 / np.sqrt(np.maximum(evals, 1e-8))) @ evecs.T
    return (Y @ T), (mu, T)

def _radial_funnel(X2_iso: np.ndarray, z: np.ndarray, c: np.ndarray, sigma: float, depth_scale: float, mix_z: float):
    R = X2_iso - c[None, :]
    r = np.linalg.norm(R, axis=1) + 1e-9
    u = R / r[:, None]
    # axisymmetric funnel along z_funnel
    z_funnel = -np.exp(-(r**2) / (2 * sigma**2))           # in [-1, 0]
    z_new = depth_scale * z_funnel + mix_z * (z - z.mean())
    # keep points at same angle but snap XY slightly toward circularity
    X2_new = (r[:, None] * u)
    return X2_new, z_new

def _phantom_metrics(X2: np.ndarray, z: np.ndarray) -> Dict[str, float]:
    # Estimate global min and second-lowest basin via coarse binning
    # (robust, no SciPy required)
    nb = max(12, int(np.sqrt(len(X2)) / 2))
    xi = np.digitize(X2[:,0], np.linspace(X2[:,0].min(), X2[:,0].max(), nb))
    yi = np.digitize(X2[:,1], np.linspace(X2[:,1].min(), X2[:,1].max(), nb))
    grid_min = {}
    for i in range(len(X2)):
        grid_min[(xi[i], yi[i])] = min(grid_min.get((xi[i], yi[i]), np.inf), z[i])
    mins = sorted(grid_min.values())
    if len(mins) < 2: 
        return {"phantom_index": 0.0, "margin": 0.0}
    z0, z1 = mins[0], mins[1]
    span = np.percentile(z, 95) - np.percentile(z, 5) + 1e-9
    phantom_index = (z1 - z0) / span   # higher is better (true deeper than 2nd)
    margin = z1 - z0
    return {"phantom_index": float(phantom_index), "margin": float(margin)}

def _lateral_inhibition(z: np.ndarray, X2: np.ndarray, k:int, strength: float) -> np.ndarray:
    # Push up (less negative) the non-global minima
    # Simple heuristic: compute local z-rank in kNN neighborhood and damp
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(k, len(X2))).fit(X2)
    idx = nbrs.kneighbors(return_distance=False)
    ranks = np.argsort(np.argsort(z[idx], axis=1), axis=1)[:,0]  # rank of each point within its neighborhood
    # points with rank 0 are local minima; >0 are higher
    boost = (ranks > 0).astype(float)
    z_adj = z + strength * 0.5 * (boost - 0.5) * (np.std(z) + 1e-6)
    return z_adj

def pca3_and_warp(H: np.ndarray,
                  energy: Optional[np.ndarray] = None,
                  params: WellParams = WellParams()):
    # 1) PCA(3) with whitening
    pca = PCA(n_components=3, whiten=params.whiten, random_state=0)
    X3 = pca.fit_transform(H)
    X2 = X3[:, :2]
    z  = X3[:, 2].copy()

    # 2) center by softmin of energy (favor true basin)
    c, w = _softmin_center(X2, energy, params.tau)

    # 3) isotropize plane (optional)
    if params.isotropize_xy:
        X2_iso, (mu, T) = _isotropize(X2 - c)
    else:
        X2_iso = X2 - c

    # 4) radial funnel shaping
    r = np.linalg.norm(X2_iso, axis=1)
    sigma = np.median(r) * params.sigma_scale + 1e-9
    X2_new, z_new = _radial_funnel(X2_iso, z, np.array([0.0, 0.0]), sigma, params.depth_scale, params.mix_z)

    # 5) lateral inhibition to suppress phantoms
    z_new = _lateral_inhibition(z_new, X2_new, k=params.inhibit_k, strength=params.inhibit_strength)

    # diagnostics
    metrics = _phantom_metrics(X2_new, z_new)
    out = np.column_stack([X2_new + c, z_new])  # restore center
    return out, metrics, dict(center=c, sigma=sigma)

def plot_trisurf(X3: np.ndarray, energy: Optional[np.ndarray] = None, params: WellParams = WellParams(), title:str="PCA(3) → warped single well"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = X3[:,0], X3[:,1], X3[:,2]
    c = None if energy is None else (energy - np.min(energy)) / (np.ptp(energy) + 1e-9)

    # trisurf if available, else scatter fallback
    if _HAVE_QHULL and len(X3) >= 4:
        tri = Delaunay(np.column_stack([x, y]))
        ax.plot_trisurf(x, y, z, triangles=tri.simplices, cmap='viridis', alpha=params.trisurf_alpha, linewidth=0.2, antialiased=True)
    ax.scatter(x, y, z, c=c, cmap='viridis', s=12, alpha=params.point_alpha)

    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title(title)
    mappable = plt.cm.ScalarMappable(cmap='viridis')
    if energy is not None:
        mappable.set_array(c)
        cb = fig.colorbar(mappable, ax=ax)
        cb.set_label("Energy (norm)")
    plt.tight_layout()
    return fig, ax
