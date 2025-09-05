# step9_hybrid_geodesic.py
# CPU-only, NumPy + transformers. Precision Step 9 (geodesics), optional tiny damping.

from sklearn.decomposition import PCA
import numpy as np
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.decomposition import PCA
from numpy.linalg import inv


# ---------- Config ----------
REDUCED_VAR = 0.99
LAMBDA = 0.35          # give a bit more curvature
DT = 0.02
STEPS  = 600    # was 800 (early stop will usually end sooner)
GAMMA = 0.04           # tiny damping
SEED = 42
MODE = "geodesic"
EPS = 1e-6             # used in potential/grad

# --- PCA ---
REDUCED_VAR = 0.99
MIN_DIMS = 8
MAX_DIMS = 16



# 1) Drop this list where you define training data
train_prompts = [
    # ---- rotate (10) ----
    "Identify the pattern: Input grid [[1,2],[3,4]] -> Output [[3,1],[4,2]] (90° cw). Apply the same idea.",
    "Perform a 90° clockwise rotation: [[5,6],[7,8]] → [[7,5],[8,6]]. Generalize this transformation.",
    "Rotate right by 90 degrees: [[2,1],[4,3]] → [[4,2],[3,1]]. Use this rotation behavior.",
    "Turn the grid 90° to the right: [[0,1],[2,3]] → [[2,0],[3,1]]. Maintain this mapping rule.",
    "Apply a quarter‑turn clockwise: [[9,8],[7,6]] → [[7,9],[6,8]]. Keep consistent with 90° cw.",
    "Use a 90° cw rotation on a 3×3: [[1,2,3],[4,5,6],[7,8,9]] → [[7,4,1],[8,5,2],[9,6,3]].",
    "Clockwise quarter‑turn: [[3,0],[0,1]] → [[0,3],[1,0]]. Follow the same rotation rule.",
    # "Rotate 90° cw: [[1,0],[0,2]] → [[0,1],[2,0]]. Preserve clockwise orientation.",
    "Rotate 90° cw: [[1,2,0],[0,3,4],[0,0,5]] → [[0,0,1],[0,3,2],[5,4,0]].",
    "Quarter‑turn right: [[2,0],[5,7]] → [[5,2],[7,0]]. Use standard 90° cw mapping.",
    "3×3 right rotation: [[0,2,0],[1,0,1],[0,2,0]] → [[0,1,0],[2,0,2],[0,1,0]].",

    # ---- flip_h (10) ----
    # "Flip horizontally: [[1,2],[3,4]] → [[2,1],[4,3]]. Mirror columns.",
    "Flip horizontally: [[1,0,2],[3,0,4],[5,0,6]] → [[2,0,1],[4,0,3],[6,0,5]].",
    "Reflect left‑right: [[5,6],[7,8]] → [[6,5],[8,7]]. Keep rows, swap columns.",
    "Horizontal mirror: [[0,1],[2,3]] → [[1,0],[3,2]]. Apply left↔right reflection.",
    "Left‑right flip on 3×3: [[1,2,3],[4,5,6],[7,8,9]] → [[3,2,1],[6,5,4],[9,8,7]].",
    "Mirror columns: [[2,1],[4,3]] → [[1,2],[3,4]]. Keep the same flip rule.",
    "Flip horizontally: [[0,2],[5,7]] → [[2,0],[7,5]]. Swap each row’s ends.",
    "Reflect across vertical axis: [[3,0],[0,1]] → [[0,3],[1,0]]. Preserve row order.",
    "Left↔right reflect: [[1,0],[0,2]] → [[0,1],[2,0]]. Symmetric across the center line.",
    "Horizontal flip (3×3): [[0,2,0],[1,0,1],[0,2,0]] → [[0,2,0],[1,0,1],[0,2,0]].",
    "Mirror columns: [[9,8],[7,6]] → [[8,9],[6,7]].",

    # ---- flip_v (10) ----
    # "Flip vertically: [[1,2],[3,4]] → [[3,4],[1,2]]. Mirror rows.",
    "Flip vertically: [[1,2,3],[0,0,0],[4,5,6]] → [[4,5,6],[0,0,0],[1,2,3]].",
    "Reflect top‑bottom: [[5,6],[7,8]] → [[7,8],[5,6]]. Keep columns, swap rows.",
    "Vertical mirror: [[0,1],[2,3]] → [[2,3],[0,1]]. Apply top↔bottom reflection.",
    "Top‑bottom flip on 3×3: [[1,2,3],[4,5,6],[7,8,9]] → [[7,8,9],[4,5,6],[1,2,3]].",
    "Mirror rows: [[2,1],[4,3]] → [[4,3],[2,1]]. Standard vertical flip.",
    "Flip vertically: [[0,2],[5,7]] → [[5,7],[0,2]]. Swap row order.",
    "Reflect across horizontal axis: [[3,0],[0,1]] → [[0,1],[3,0]]. Keep columns fixed.",
    "Top↔bottom reflect: [[1,0],[0,2]] → [[0,2],[1,0]]. Maintain column structure.",
    "Vertical flip (3×3): [[0,2,0],[1,0,1],[0,2,0]] → [[0,2,0],[1,0,1],[0,2,0]].",
    "Mirror rows: [[9,8],[7,6]] → [[7,6],[9,8]].",
]

labels = (
    ["rotate"]*10 +
    ["flip_h"]*10 +
    ["flip_v"]*10
)


np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------- Latent extraction ----------
_device = torch.device("cpu")

_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
_model = GPT2LMHeadModel.from_pretrained("gpt2").to(_device)
_model.eval()

def get_latent(prompt: str) -> np.ndarray:
    """Mean-pool last hidden state as latent."""
    with torch.no_grad():
        toks = _tokenizer(prompt, return_tensors="pt").to(_device)
        out = _model(**toks, output_hidden_states=True)
        hs = out.hidden_states[-1][0].cpu().numpy()  # (seq, hidden)
    return hs.mean(axis=0)  # (hidden,)






def pca_fit_transform(vectors, target_var=0.95, MIN_DIMS=2, MAX_DIMS=8, verbose=True):
    """
    vectors: (n_samples, n_features)
    Chooses n_components safely based on available samples/features and target variance.
    """
    vectors = np.asarray(vectors)
    n_samples, n_features = vectors.shape
    n_cap = min(n_samples, n_features)          # hard cap from sklearn
    if n_cap <= 0:
        raise ValueError("Empty training set for PCA.")
    
    # First fit with full dimensionality up to the cap to measure variance captured.
    probe_n = min(n_cap, MAX_DIMS)              # don't waste time probing huge dims
    pca_probe = PCA(n_components=probe_n, whiten=False, svd_solver="full").fit(vectors)
    cumvar = np.cumsum(pca_probe.explained_variance_ratio_)
    # Smallest k meeting target variance (or probe_n if target not reached)
    k = int(np.searchsorted(cumvar, target_var) + 1)
    
    # Final n: respect MIN/MAX and the cap
    n = max(MIN_DIMS, min(MAX_DIMS, k))
    n = min(n, n_cap)
    
    # If we only have 1–2 samples/features, we may be forced to 1D.
    if n < MIN_DIMS and n_cap >= 1:
        # fall back gracefully; keep whiten=False when n is tiny
        n = n_cap
    
    whiten = (n >= 2)   # avoid whitening in degenerate 1D cases
    pca = PCA(n_components=n, whiten=whiten, svd_solver="full").fit(vectors)
    Z = pca.transform(vectors)
    
    if verbose:
        print(f"[PCA] samples={n_samples} feat={n_features} cap={n_cap} "
              f"-> n={pca.n_components_} (cum var={cumvar[min(n-1, len(cumvar)-1)]:.4f}, "
              f"whiten={whiten})")
        if pca.n_components_ < 2:
            print("NOTE: PCA ended up 1D due to limited samples/features. "
                  "Add more training prompts or lower MIN_DIMS if downstream expects ≥2D.")
    return pca, Z



# ---------- Conformally flat metric from multi-mass potential ----------
def potential(x: np.ndarray, centers: np.ndarray, masses: np.ndarray) -> float:
    # V(x) = - sum_i M_i / (||x - c_i|| + eps)
    diffs = x[None, :] - centers
    dists = np.linalg.norm(diffs, axis=1) + EPS
    return -np.sum(masses / dists)

def grad_potential(x: np.ndarray, centers: np.ndarray, masses: np.ndarray) -> np.ndarray:
    # ∇V = - sum_i M_i * (-(x - c_i)) / (||x - c_i||^3 + eps)
    diffs = x[None, :] - centers  # (k,d)
    dists = np.linalg.norm(diffs, axis=1) + EPS  # (k,)
    # d/dx (1/r) = - (x-c)/r^3
    terms = masses[:, None] * diffs / (dists**3)[:, None]  # (k,d)
    return -np.sum(terms, axis=0)

def lnphi_and_grad(x: np.ndarray, centers: np.ndarray, masses: np.ndarray):
    V = potential(x, centers, masses)
    lnphi = LAMBDA * V
    # ∇lnφ = λ ∇V
    g_lnphi = LAMBDA * grad_potential(x, centers, masses)
    return lnphi, g_lnphi

def christoffel_conformal(x: np.ndarray, v: np.ndarray, centers: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """
    Γ^i_{jk} v^j v^k for g_ij = φ^2 δ_ij with φ = exp(λV).
    Γ^i_{jk} = δ^i_j ∂_k lnφ + δ^i_k ∂_j lnφ - δ_jk ∂^i lnφ
    Contract with v^j v^k: 2 v_i (v·∇lnφ) - ||v||^2 ∂^i lnφ
    Returned as vector with upper index i.
    """
    _, grad_lnphi = lnphi_and_grad(x, centers, masses)
    v_dot = float(np.dot(v, grad_lnphi))
    v_norm2 = float(np.dot(v, v))
    return 2.0 * v * v_dot - v_norm2 * grad_lnphi

def rk4_step(x: np.ndarray, v: np.ndarray, dt: float, centers: np.ndarray, masses: np.ndarray):
    """
    Geodesic ODE in coordinates (x, v):
      dx/dτ = v
      dv/dτ = - Γ(x)[v,v] - γ v
    """
    def a(x_, v_):
        return -christoffel_conformal(x_, v_, centers, masses) - GAMMA * v_

    k1x = v
    k1v = a(x, v)

    k2x = v + 0.5 * dt * k1v
    k2v = a(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v)

    k3x = v + 0.5 * dt * k2v
    k3v = a(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v)

    k4x = v + dt * k3v
    k4v = a(x + dt * k3x, v + dt * k3v)

    x_new = x + (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
    v_new = v + (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)
    return x_new, v_new

def integrate_geodesic(x0: np.ndarray, v0: np.ndarray, centers: np.ndarray, masses: np.ndarray,
                       steps: int = STEPS, dt: float = DT):
    x, v = x0.copy(), v0.copy()
    traj = [x.copy()]
    for _ in range(steps):
        x, v = rk4_step(x, v, dt, centers, masses)
        traj.append(x.copy())
    return np.array(traj)

def class_stats(Z, labels, uniq):
    stats = {}
    for u in uniq:
        idxs = [i for i,l in enumerate(labels) if l == u]
        G = Z[idxs]
        mu = G.mean(axis=0)
        # regularized covariance (diagonal if tiny sample)
        C = np.cov(G.T) if G.shape[0] > G.shape[1] else np.diag(np.var(G, axis=0) + 1e-6)
        # ridge for stability
        C = C + 1e-4 * np.eye(C.shape[0])
        stats[u] = {"mu": mu, "invC": inv(C)}
    return stats

def shared_cov_stats(Z, labels, order, ridge=1e-3):
    # pool all class-centered residuals
    mus = {}
    resids = []
    for u in order:
        idx = [i for i,l in enumerate(labels) if l == u]
        G = Z[idx]
        mu = G.mean(axis=0)
        mus[u] = mu
        resids.append(G - mu)
    R = np.concatenate(resids, axis=0)
    Sigma = (R.T @ R) / max(1, R.shape[0]-1)
    # shrinkage toward diagonal
    diag = np.diag(np.diag(Sigma))
    alpha = 0.2  # shrink factor (tune 0.1–0.3)
    Sigma = (1-alpha)*Sigma + alpha*diag
    Sigma += ridge*np.eye(Sigma.shape[0])
    invS = inv(Sigma)
    return {u: {"mu": mus[u], "invS": invS} for u in order}

def mahalanobis_distances(z, stats, order):
    ds = []
    for u in order:
        mu, invC = stats[u]["mu"], stats[u]["invC"]
        d = z - mu
        ds.append(float(np.sqrt(d @ invC @ d)))
    return np.array(ds)


def mahalanobis_shared(z, stats, order):
    ds = []
    invS = next(iter(stats.values()))["invS"]
    for u in order:
        mu = stats[u]["mu"]
        d = z - mu
        ds.append(float(np.sqrt(d @ invS @ d)))
    return np.array(ds)

# Use this instead of Euclidean in classify_endpoint:
def classify_endpoint_maha(z_final_red, stats, order):
    d = mahalanobis_distances(z_final_red, stats, order)
    j = int(np.argmin(d))
    return j, float(d[j]), d

# --- Compute anchor centers and semantic masses ---
def compute_anchors(Z, labels):
    uniq = sorted(set(labels))  # actual order used to build centers
    centers, masses = [], []
    for u in uniq:
        idxs = [i for i,l in enumerate(labels) if l == u]
        group = Z[idxs]
        c = group.mean(axis=0)
        centers.append(c)
        spread = float(np.mean(np.linalg.norm(group - c, axis=1)) + 1e-6)
        masses.append(1.0 / spread)  # tighter cluster => heavier mass
    centers = np.stack(centers, axis=0)
    masses  = np.array(masses, dtype=float)
    masses /= (masses.sum() + 1e-9)
    return centers, masses, uniq




# ---------- Example end-to-end ----------
def solve_with_geodesics(prompt: str,
                         pca: PCA,
                         anchor_centers_red: np.ndarray,
                         masses: np.ndarray):
    z_full = get_latent(prompt)                 # (hidden,)
    z_red  = pca.transform(z_full[None, :])[0]  # (d,)

    # Start slightly displaced with small initial velocity toward negative grad V
    lnphi, g_lnphi = lnphi_and_grad(z_red, anchor_centers_red, masses)
    v0 = -0.1 * g_lnphi / (np.linalg.norm(g_lnphi) + 1e-9)

    traj = integrate_geodesic(z_red, v0, anchor_centers_red, masses, steps=STEPS, dt=DT)
    z_final_red = traj[-1]
    # Map back to full space (optional): x_full ≈ z_red * P^T + mean
    z_full_final = pca.inverse_transform(z_final_red[None, :])[0]
    return z_full_final, traj


# ----------------------------
# ARC warp-interference solver
# ----------------------------

# 1) Minimal ARC transforms we’ll support right now
def rot90_cw(grid):
    g = np.array(grid)
    return np.rot90(g, k=3).tolist()

def rot180(grid):
    g = np.array(grid)
    return np.rot90(g, k=2).tolist()

def rot270_cw(grid):
    g = np.array(grid)
    return np.rot90(g, k=1).tolist()

def flip_h(grid):
    g = np.array(grid)
    return np.flip(g, axis=1).tolist()

def flip_v(grid):
    g = np.array(grid)
    return np.flip(g, axis=0).tolist()

# Map anchor index -> transform; adjust to your labels/anchor order
TRANSFORM_BY_LABEL = {
    "rotate": rot90_cw,       # label 0
    "flip_h": flip_h,         # label 1
    "flip_v": flip_v,         # label 2
    # add more labels/anchors as you train them:
    # "rot180": rot180,
    # "rot270": rot270_cw,
}

# If you created anchors via sorted(set(labels)), keep the same order here:
ANCHOR_LABELS_IN_ORDER = ["rotate", "flip_h", "flip_v"]  # <- keep in sync with your training labels

def grid_to_prompt(grid):
    # simple text promptization so GPT-2 latent “sees” the exact instance
    return f"Identify the pattern: Input grid {grid} -> Output ? (choose rotate 90° cw, flip_h, or flip_v)."

def classify_endpoint(z_final_red, centers):
    d = np.linalg.norm(centers - z_final_red, axis=1)
    j = int(np.argmin(d))
    return j, float(d[j]), d

def arc_latent_for_grid(grid, pca):
    prompt = grid_to_prompt(grid)
    z_full = get_latent(prompt)
    z_red = pca.transform(z_full.reshape(1, -1))[0]
    return z_red

def run_warp_interference(z_red, centers, masses, steps, dt, mode="geodesic"):
    # build initial velocity along −∇lnφ (fallback to small random)
    _, g_lnphi = lnphi_and_grad(z_red, centers, masses)
    g_norm = float(np.linalg.norm(g_lnphi))
    if g_norm < 1e-8:
        rng = np.random.default_rng(0)
        v0 = rng.normal(size=z_red.shape).astype(float)
        v0 /= (np.linalg.norm(v0) + 1e-9)
        v0 *= 0.05
    else:
        v0 = -g_lnphi / (g_norm + 1e-9) * 0.1

    if mode == "geodesic":
        traj = integrate_geodesic(x0=z_red, v0=v0, centers=centers, masses=masses, steps=steps, dt=dt)
        z_final = traj[-1]
    else:
        # Stage-9 linearized nudge (if you kept integrate_nudge from earlier)
        traj = integrate_nudge(z_red, centers, masses, steps=350, dt=0.05, k=2.0, gamma=0.2)
        z_final = traj[-1]

    return z_final

def solve_arc_task(input_grid, *, verbose=True):
    """
    input_grid: e.g. [[1,2],[3,4]] or a 3x3 integer grid.
    Returns: predicted_output_grid (same shape as input)
    """
    # 1) embed this specific instance
    z_red = arc_latent_for_grid(input_grid, pca)

    # 2) warp-interference geodesic to pick the anchor
    z_final = run_warp_interference(z_red, anchor_centers_red, masses, steps=STEPS, dt=DT, mode=MODE)

    # 3) nearest anchor = chosen transform
    # j, dmin, all_d = classify_endpoint(z_final, anchor_centers_red)
    # chosen_label = ANCHOR_LABELS_IN_ORDER[j]

    j, dmin, all_d = classify_maha(z_final)
    chosen_label = ANCHOR_LABELS_IN_ORDER[j]
    
    transform = TRANSFORM_BY_LABEL.get(chosen_label, None)

    if verbose:
        print(f"[ARC] Endpoint nearest anchor: {j} ({chosen_label}) at distance {dmin:.4f}")
        print(f"[ARC] Distances to anchors:", np.array_str(all_d, precision=4, suppress_small=True))

    if transform is None:
        raise ValueError(f"No transform mapped for anchor label '{chosen_label}'")

    # 4) apply transform to the grid
    output_grid = transform(input_grid)
    return output_grid

def classify_maha(z_red):
    d = mahalanobis_shared(z_red, cls_stats, ANCHOR_LABELS_IN_ORDER)
    j = int(np.argmin(d))
    return j, float(d[j]), d

def anchor_confidence(distances, tau=1.6):  # was 0.8
    d = np.asarray(distances, float)
    s = np.exp(-(d - d.min()) / max(1e-9, tau))
    p = s / (s.sum() + 1e-9)
    j = int(np.argmin(d))
    return j, float(p[j]), p


def integrate_geodesic_with_early_stop(x0, v0, centers, masses, steps=STEPS, dt=DT,
                                       stable_window=12):
    x, v = x0.copy(), v0.copy()
    traj = [x.copy()]
    last = []
    for t in range(steps):
        x, v = rk4_step(x, v, dt, centers, masses)
        traj.append(x.copy())
        # check stability of the predicted anchor in reduced space
        d = np.linalg.norm(centers - x, axis=1)
        j = int(np.argmin(d))
        last.append(j)
        if len(last) > stable_window:
            last.pop(0)
            if len(set(last)) == 1:  # same pick for 'stable_window' steps
                break
    return np.array(traj)

# Build 12 toy cases covering rotate / flip_h / flip_v evenly
def make_cases():
    # small 2x2 and 3x3 to vary structure
    cases = [
        # rotate 90° cw
        {"grid": [[1,2],[3,4]], "label":"rotate"},
        {"grid": [[5,6],[7,8]], "label":"rotate"},
        {"grid": [[1,0,2],[0,1,0],[2,0,1]], "label":"rotate"},
        {"grid": [[9,8,7],[6,5,4],[3,2,1]], "label":"rotate"},
        # flip_h
        {"grid": [[1,2],[3,4]], "label":"flip_h"},
        {"grid": [[5,6],[7,8]], "label":"flip_h"},
        {"grid": [[1,0,2],[0,1,0],[2,0,1]], "label":"flip_h"},
        {"grid": [[9,8,7],[6,5,4],[3,2,1]], "label":"flip_h"},
        # flip_v
        {"grid": [[1,2],[3,4]], "label":"flip_v"},
        {"grid": [[5,6],[7,8]], "label":"flip_v"},
        {"grid": [[1,0,2],[0,1,0],[2,0,1]], "label":"flip_v"},
        {"grid": [[9,8,7],[6,5,4],[3,2,1]], "label":"flip_v"},
    ]
    # attach ground-truth transformed grids using your transform map
    out = []
    for c in cases:
        tfunc = TRANSFORM_BY_LABEL[c["label"]]
        out.append({"grid": c["grid"], "label": c["label"], "target": tfunc(c["grid"])})
    return out

def grid_to_prompt(grid, choices):
    return f"Identify the pattern: Input grid {grid} -> Output ? (choose {', '.join(choices)})."

# then arc_latent_for_grid:
def arc_latent_for_grid(grid, pca):
    prompt = grid_to_prompt(grid, ANCHOR_LABELS_IN_ORDER)
    z_full = get_latent(prompt)
    return pca.transform(z_full.reshape(1, -1))[0]

def run_arc_benchmark_12(verbose=True, tau=2.0):
    cases = make_cases()
    correct = 0
    confs = []
    per_case = []

    for idx, case in enumerate(cases, 1):
        z_red = arc_latent_for_grid(case["grid"], pca)
        z_final = run_warp_interference(z_red, anchor_centers_red, masses, steps=STEPS, dt=DT, mode=MODE)
        # j, dmin, dists = classify_endpoint(z_final, anchor_centers_red)
        # j_hat, conf, probs = anchor_confidence(dists, tau=tau)
        # pred_label = ANCHOR_LABELS_IN_ORDER[j_hat]
        
        j, dmin, dists = classify_maha(z_final)
        j_hat, conf, probs = anchor_confidence(dists, tau=0.8)
        pred_label = ANCHOR_LABELS_IN_ORDER[j_hat]
        pred_grid = TRANSFORM_BY_LABEL[pred_label](case["grid"])

        ok = (pred_label == case["label"]) and (pred_grid == case["target"])
        correct += int(ok)
        confs.append(conf)

        if verbose:
            print(f"[{idx:02d}] true={case['label']:7s} pred={pred_label:7s} "
                  f"ok={ok} dist*={dmin:.3f} conf={conf:.3f} probs={np.array_str(probs, precision=3)}")

        per_case.append({
            "idx": idx,
            "true": case["label"],
            "pred": pred_label,
            "ok": ok,
            "conf": conf,
            "dmin": dmin,
        })

    acc = correct / len(cases)
    mean_conf = float(np.mean(confs)) if confs else 0.0
    print(f"\n[ARC-12] Accuracy: {acc*100:.1f}% | Mean confidence: {mean_conf:.3f} | Mode={MODE}")
    return {"accuracy": acc, "mean_confidence": mean_conf, "details": per_case}


# ----- run it -----
if __name__ == "__main__":
    # 1) Build training latents and PCA first
    train_latents = np.stack([get_latent(p) for p in train_prompts], axis=0)
    pca, Z = pca_fit_transform(train_latents, target_var=0.99, MIN_DIMS=8, MAX_DIMS=16)

    # 2) Anchors + masses (and record the actual order they were built in)
    anchor_centers_red, masses, ANCHOR_LABELS_IN_ORDER = compute_anchors(Z, labels)
    masses = masses / (masses.sum() + 1e-9)
    masses *= 4.0               # stronger curvature
    print("[Anchors] order:", ANCHOR_LABELS_IN_ORDER)

    # 3) Optional: Mahalanobis classifier stats
    #cls_stats = class_stats(Z, labels, ANCHOR_LABELS_IN_ORDER)
    cls_stats = shared_cov_stats(Z, labels, ANCHOR_LABELS_IN_ORDER, ridge=1e-3)

    # 4) Run mini-benchmark (uses global pca/anchors/masses/MODE)
    _ = run_arc_benchmark_12(verbose=True, tau=1.0)

    # 5) Optional sanity: one demo grid end-to-end
    demo = [[1,2],[3,4]]
    z0 = arc_latent_for_grid(demo, pca)
    zf = run_warp_interference(z0, anchor_centers_red, masses, steps=STEPS, dt=DT, mode=MODE)
    j, dmin, all_d = classify_endpoint(zf, anchor_centers_red)
    pred_label = ANCHOR_LABELS_IN_ORDER[j]
    print("[Sanity] anchor order:", ANCHOR_LABELS_IN_ORDER)
    print("[Sanity] distances:", np.array_str(all_d, precision=3))
    print("[Sanity] chosen:", pred_label)




