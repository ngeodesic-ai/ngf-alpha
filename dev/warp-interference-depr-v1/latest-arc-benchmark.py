
# latest-arc-benchmark.py
# Stage-10 (upgraded): hybrid geodesic/nudge, pooled-cov Mahalanobis, early stopping, CPU-friendly.
# Author: ngeodesic.ai (Alpha)
# License: Apache-2.0

import argparse, sys, math, time, random
from dataclasses import dataclass
import numpy as np
from numpy.linalg import inv
from sklearn.decomposition import PCA

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel


# -------------------------
# Config (defaults; overridable by CLI)
# -------------------------
DEFAULTS = dict(
    target_var=0.99,
    pca_min_dims=8,
    pca_max_dims=16,
    lam=0.35,              # curvature strength (λ) in φ = exp(λ V)
    dt=0.02,               # ODE step
    steps=600,             # ODE max steps (early stop usually stops earlier)
    gamma=0.04,            # damping in dv/dτ = ... - γ v
    seed=42,
    mode="geodesic",       # "geodesic" or "nudge"
    tau=1.6,               # softness for confidence from distances
    mass_scale=4.0,        # multiply masses to deepen wells
    early_stop_window=12,  # consecutive identical picks to stop ODE
)

# -------------------------
# Utility: seeding & device
# -------------------------
def seed_all(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def pick_device():
    # Force CPU for reproducibility & portability; flip to cuda if you want
    return torch.device("cpu")

# -------------------------
# Latent extraction (GPT-2)
# -------------------------
class LatentExtractor:
    def __init__(self, device):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self.model.eval()

    @torch.no_grad()
    def get(self, prompt:str) -> np.ndarray:
        toks = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model(**toks, output_hidden_states=True)
        hs = out.hidden_states[-1][0].cpu().numpy()  # (seq, hidden)
        return hs.mean(axis=0)  # (hidden,)

# -------------------------
# PCA fit/transform (robust)
# -------------------------
def pca_fit_transform(vectors, target_var=0.99, MIN_DIMS=8, MAX_DIMS=16, verbose=True):
    X = np.asarray(vectors)
    n_samples, n_features = X.shape
    n_cap = min(n_samples, n_features)
    if n_cap <= 0:
        raise ValueError("Empty training set for PCA.")
    probe_n = min(n_cap, MAX_DIMS)
    probe = PCA(n_components=probe_n, whiten=False, svd_solver="full").fit(X)
    cum = np.cumsum(probe.explained_variance_ratio_)
    k = int(np.searchsorted(cum, target_var) + 1)
    n = max(MIN_DIMS, min(MAX_DIMS, k))
    n = min(n, n_cap)
    whiten = (n >= 2)
    pca = PCA(n_components=n, whiten=whiten, svd_solver="full").fit(X)
    Z = pca.transform(X)
    if verbose:
        print(f"[PCA] samples={n_samples} feat={n_features} cap={n_cap} -> n={pca.n_components_} "
              f"(cum var={cum[min(n-1, len(cum)-1)]:.4f}, whiten={whiten})")
    return pca, Z

# -------------------------
# Training prompts (de-ambiguous)
# -------------------------
def training_prompts():
    # Non-symmetric examples so each transform actually changes the grid.
    rotate = [
        "Identify the pattern: Input grid [[1,2],[3,4]] -> Output [[3,1],[4,2]] (90° cw). Apply the same idea.",
        "Perform a 90° clockwise rotation: [[5,6],[7,8]] → [[7,5],[8,6]]. Generalize this transformation.",
        "Rotate right by 90 degrees: [[2,1],[4,3]] → [[4,2],[3,1]]. Use this rotation behavior.",
        "Turn the grid 90° to the right: [[0,1],[2,3]] → [[2,0],[3,1]]. Maintain this mapping rule.",
        "Apply a quarter‑turn clockwise: [[9,8],[7,6]] → [[7,9],[6,8]]. Keep consistent with 90° cw.",
        "Use a 90° cw rotation on a 3×3: [[1,2,3],[4,5,6],[7,8,9]] → [[7,4,1],[8,5,2],[9,6,3]].",
        "Clockwise quarter‑turn: [[3,0],[0,1]] → [[0,3],[1,0]]. Follow the same rotation rule.",
        "Rotate 90° cw: [[1,2,0],[0,3,4],[0,0,5]] → [[0,0,1],[0,3,2],[5,4,0]].",
        "Quarter‑turn right: [[2,0],[5,7]] → [[5,2],[7,0]]. Use standard 90° cw mapping.",
        "3×3 right rotation: [[2,0,1],[0,1,0],[1,0,2]] → [[1,0,2],[0,1,0],[2,0,1]].",
    ]
    flip_h = [
        "Flip horizontally: [[1,2],[3,4]] → [[2,1],[4,3]]. Mirror columns.",
        "Reflect left‑right: [[5,6],[7,8]] → [[6,5],[8,7]]. Keep rows, swap columns.",
        "Horizontal mirror: [[0,1],[2,3]] → [[1,0],[3,2]]. Apply left↔right reflection.",
        "Left‑right flip on 3×3: [[1,2,4],[4,5,6],[7,8,9]] → [[4,2,1],[6,5,4],[9,8,7]].",
        "Mirror columns: [[2,1],[4,3]] → [[1,2],[3,4]]. Keep the same flip rule.",
        "Flip horizontally: [[0,2],[5,7]] → [[2,0],[7,5]]. Swap each row’s ends.",
        "Reflect across vertical axis: [[3,0],[0,1]] → [[0,3],[1,0]]. Preserve row order.",
        "Left↔right reflect: [[1,0],[0,2]] → [[0,1],[2,0]]. Symmetric across the center line.",
        "Horizontal flip (3×3): [[1,0,2],[0,1,0],[2,0,1]] → [[2,0,1],[0,1,0],[1,0,2]].",
        "Mirror columns: [[9,8],[7,6]] → [[8,9],[6,7]].",
    ]
    flip_v = [
        "Flip vertically: [[1,2],[3,4]] → [[3,4],[1,2]]. Mirror rows.",
        "Reflect top‑bottom: [[5,6],[7,8]] → [[7,8],[5,6]]. Keep columns, swap rows.",
        "Vertical mirror: [[0,1],[2,3]] → [[2,3],[0,1]]. Apply top↔bottom reflection.",
        "Top‑bottom flip on 3×3: [[1,2,4],[4,5,6],[7,8,9]] → [[7,8,9],[4,5,6],[4,2,1]].",
        "Mirror rows: [[2,1],[4,3]] → [[4,3],[2,1]]. Standard vertical flip.",
        "Flip vertically: [[0,2],[5,7]] → [[5,7],[0,2]]. Swap row order.",
        "Reflect across horizontal axis: [[3,0],[0,1]] → [[0,1],[3,0]]. Keep columns fixed.",
        "Top↔bottom reflect: [[1,0],[0,2]] → [[0,2],[1,0]]. Maintain column structure.",
        "Vertical flip (3×3): [[1,0,2],[0,1,0],[2,0,1]] → [[2,0,1],[0,1,0],[1,0,2]].",
        "Mirror rows: [[9,8],[7,6]] → [[7,6],[9,8]].",
    ]
    prompts = rotate + flip_h + flip_v
    labels = (["rotate"]*len(rotate) + ["flip_h"]*len(flip_h) + ["flip_v"]*len(flip_v))
    return prompts, labels

# -------------------------
# Metric / geometry
# -------------------------
EPS = 1e-6

def potential(x: np.ndarray, centers: np.ndarray, masses: np.ndarray) -> float:
    diffs = x[None,:] - centers
    dists = np.linalg.norm(diffs, axis=1) + EPS
    return -np.sum(masses / dists)

def grad_potential(x: np.ndarray, centers: np.ndarray, masses: np.ndarray) -> np.ndarray:
    diffs = x[None,:] - centers      # (k,d)
    dists = np.linalg.norm(diffs, axis=1) + EPS
    terms = masses[:,None] * diffs / (dists**3)[:,None]
    return -np.sum(terms, axis=0)

def lnphi_and_grad(x: np.ndarray, centers: np.ndarray, masses: np.ndarray, lam: float):
    V = potential(x, centers, masses)
    gV = grad_potential(x, centers, masses)
    return lam*V, lam*gV

def christoffel_conformal(x: np.ndarray, v: np.ndarray, centers: np.ndarray, masses: np.ndarray, lam: float):
    _, grad_lnphi = lnphi_and_grad(x, centers, masses, lam)
    v_dot = float(np.dot(v, grad_lnphi))
    v_norm2 = float(np.dot(v, v))
    return 2.0 * v * v_dot - v_norm2 * grad_lnphi

def rk4_step(x: np.ndarray, v: np.ndarray, dt: float, centers: np.ndarray, masses: np.ndarray, lam: float, gamma: float):
    def acc(x_, v_):
        return -christoffel_conformal(x_, v_, centers, masses, lam) - gamma * v_
    k1x = v
    k1v = acc(x, v)
    k2x = v + 0.5*dt*k1v
    k2v = acc(x + 0.5*dt*k1x, v + 0.5*dt*k1v)
    k3x = v + 0.5*dt*k2v
    k3v = acc(x + 0.5*dt*k2x, v + 0.5*dt*k2v)
    k4x = v + dt*k3v
    k4v = acc(x + dt*k3x, v + dt*k3v)
    x_new = x + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
    v_new = v + (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
    return x_new, v_new

def integrate_geodesic_with_early_stop(x0: np.ndarray, v0: np.ndarray, centers: np.ndarray, masses: np.ndarray,
                                       lam: float, gamma: float, dt: float, steps: int, window:int=12):
    x, v = x0.copy(), v0.copy()
    picks = []
    for t in range(steps):
        x, v = rk4_step(x, v, dt, centers, masses, lam, gamma)
        d = np.linalg.norm(centers - x, axis=1)
        picks.append(int(np.argmin(d)))
        if len(picks) > window:
            picks.pop(0)
            if len(set(picks)) == 1:
                break
    return x

# Simple symbolic nudge (linearized geodesic) fallback
def integrate_nudge(x0: np.ndarray, centers: np.ndarray, masses: np.ndarray, steps=350, dt=0.05, k=2.0, gamma=0.2, tau=0.8):
    x = x0.copy()
    v = np.zeros_like(x)
    for _ in range(steps):
        d = np.linalg.norm(centers - x, axis=1)
        score = np.exp(-(d - d.min()) / max(1e-9, tau))
        w = (score * (masses + 1e-9))
        w = w / (w.sum() + 1e-9)
        target = (w[:,None] * centers).sum(axis=0)
        a = k*(target - x) - gamma*v
        v = v + dt*a
        x = x + dt*v
    return x

# -------------------------
# Anchors, masses, classifiers
# -------------------------
def compute_anchors(Z, labels):
    uniq = sorted(set(labels))  # anchor labels in order
    centers, masses = [], []
    for u in uniq:
        idx = [i for i,l in enumerate(labels) if l == u]
        G = Z[idx]
        c = G.mean(axis=0)
        centers.append(c)
        spread = float(np.mean(np.linalg.norm(G - c, axis=1)) + 1e-6)
        masses.append(1.0 / spread)   # tighter cluster => heavier mass
    centers = np.stack(centers, axis=0)
    masses = np.array(masses, dtype=float)
    masses = masses / (masses.sum() + 1e-9)
    return centers, masses, uniq

def shared_cov_stats(Z, labels, order, ridge=1e-3, shrink=0.2):
    mus, resids = {}, []
    for u in order:
        idx = [i for i,l in enumerate(labels) if l == u]
        G = Z[idx]
        mu = G.mean(axis=0)
        mus[u] = mu
        resids.append(G - mu)
    R = np.concatenate(resids, axis=0)
    Sigma = (R.T @ R) / max(1, R.shape[0]-1)
    diag = np.diag(np.diag(Sigma))
    Sigma = (1-shrink)*Sigma + shrink*diag
    Sigma += ridge*np.eye(Sigma.shape[0])
    invS = inv(Sigma)
    return {u: {"mu": mus[u], "invS": invS} for u in order}

def mahalanobis_shared(z, stats, order):
    invS = next(iter(stats.values()))["invS"]
    ds = []
    for u in order:
        mu = stats[u]["mu"]
        d = z - mu
        ds.append(float(np.sqrt(d @ invS @ d)))
    return np.array(ds)

def anchor_confidence(distances, tau=1.6):
    d = np.asarray(distances, float)
    s = np.exp(-(d - d.min()) / max(1e-9, tau))
    p = s / (s.sum() + 1e-9)
    j = int(np.argmin(d))
    return j, float(p[j]), p

# -------------------------
# ARC transforms and prompts
# -------------------------
def rot90_cw(grid):
    g = np.array(grid); return np.rot90(g, k=3).tolist()
def flip_h(grid):
    g = np.array(grid); return np.flip(g, axis=1).tolist()
def flip_v(grid):
    g = np.array(grid); return np.flip(g, axis=0).tolist()

TRANSFORM_BY_LABEL = {"rotate": rot90_cw, "flip_h": flip_h, "flip_v": flip_v}

def grid_to_prompt(grid, choices):
    return f"Identify the pattern: Input grid {grid} -> Output ? (choose {', '.join(choices)})."

# -------------------------
# Benchmark (ARC-12)
# -------------------------
def make_cases():
    cases = [
        # rotate
        {"grid": [[1,2],[3,4]], "label": "rotate"},
        {"grid": [[5,6],[7,8]], "label": "rotate"},
        {"grid": [[1,0,2],[0,1,0],[2,0,1]], "label": "rotate"},
        {"grid": [[2,0,1],[0,1,0],[1,0,2]], "label": "rotate"},
        # flip_h
        {"grid": [[1,2],[3,4]], "label": "flip_h"},
        {"grid": [[5,6],[7,8]], "label": "flip_h"},
        {"grid": [[1,0,2],[0,1,0],[2,0,1]], "label": "flip_h"},
        {"grid": [[1,2,4],[4,5,6],[7,8,9]], "label": "flip_h"},
        # flip_v
        {"grid": [[1,2],[3,4]], "label": "flip_v"},
        {"grid": [[5,6],[7,8]], "label": "flip_v"},
        {"grid": [[1,0,2],[0,1,0],[2,0,1]], "label": "flip_v"},
        {"grid": [[1,2,4],[4,5,6],[7,8,9]], "label": "flip_v"},
    ]
    out = []
    for c in cases:
        t = TRANSFORM_BY_LABEL[c["label"]]
        out.append({**c, "target": t(c["grid"])})
    return out

# -------------------------
# End-to-end helpers
# -------------------------
@dataclass
class ARCContext:
    pca: PCA
    centers: np.ndarray
    masses: np.ndarray
    labels_in_order: list
    cls_stats: dict
    extractor: LatentExtractor
    args: argparse.Namespace

def arc_latent_for_grid(grid, ctx: ARCContext):
    prompt = grid_to_prompt(grid, ctx.labels_in_order)
    z_full = ctx.extractor.get(prompt)
    z_red = ctx.pca.transform(z_full.reshape(1, -1))[0]
    return z_red

def run_warp_interference(z_red, ctx: ARCContext):
    # initial velocity along -∇lnφ
    _, g_lnphi = lnphi_and_grad(z_red, ctx.centers, ctx.masses, ctx.args.lam)
    g_norm = float(np.linalg.norm(g_lnphi))
    if g_norm < 1e-8:
        rng = np.random.default_rng(ctx.args.seed)
        v0 = rng.normal(size=z_red.shape).astype(float)
        v0 /= (np.linalg.norm(v0) + 1e-9)
        v0 *= 0.05
    else:
        v0 = -g_lnphi / (g_norm + 1e-9) * 0.1

    if ctx.args.mode == "geodesic":
        zf = integrate_geodesic_with_early_stop(
            x0=z_red, v0=v0, centers=ctx.centers, masses=ctx.masses,
            lam=ctx.args.lam, gamma=ctx.args.gamma, dt=ctx.args.dt,
            steps=ctx.args.steps, window=ctx.args.early_stop_window
        )
    else:
        zf = integrate_nudge(
            x0=z_red, centers=ctx.centers, masses=ctx.masses,
            steps=350, dt=0.05, k=2.0, gamma=0.2, tau=0.8
        )
    return zf

def classify_maha(z_red, ctx: ARCContext):
    d = mahalanobis_shared(z_red, ctx.cls_stats, ctx.labels_in_order)
    j = int(np.argmin(d))
    return j, float(d[j]), d

def run_arc_benchmark_12(ctx: ARCContext):
    cases = make_cases()
    correct = 0
    confs = []
    details = []
    for idx, case in enumerate(cases, 1):
        z0 = arc_latent_for_grid(case["grid"], ctx)
        zf = run_warp_interference(z0, ctx)
        j, dmin, dists = classify_maha(zf, ctx)
        j_hat, conf, probs = anchor_confidence(dists, tau=ctx.args.tau)
        pred_label = ctx.labels_in_order[j_hat]
        pred_grid = TRANSFORM_BY_LABEL[pred_label](case["grid"])
        ok = (pred_label == case["label"]) and (pred_grid == case["target"])
        correct += int(ok)
        confs.append(conf)
        print(f"[{idx:02d}] true={case['label']:7s} pred={pred_label:7s} ok={ok} "
              f"dist*={dmin:.3f} conf={conf:.3f} probs={np.array_str(probs, precision=3)}")
        details.append(dict(idx=idx, true=case["label"], pred=pred_label, ok=ok, conf=conf, dmin=dmin))
    acc = correct / len(cases)
    mean_conf = float(np.mean(confs)) if confs else 0.0
    print(f"\n[ARC-12] Accuracy: {acc*100:.1f}% | Mean confidence: {mean_conf:.3f} | Mode={ctx.args.mode}")
    # Sanity readout of anchor space geometry
    demo = [[1,2],[3,4]]
    z_demo = arc_latent_for_grid(demo, ctx)
    # distances from demo latent to centers (no traversal)
    base_d = np.linalg.norm(ctx.centers - z_demo, axis=1)
    print("[Sanity] anchor order:", ctx.labels_in_order)
    print("[Sanity] distances:", np.array_str(base_d, precision=3))
    print("[Sanity] chosen:", ctx.labels_in_order[int(np.argmin(base_d))])
    return dict(accuracy=acc, mean_confidence=mean_conf, details=details)

# -------------------------
# CLI / main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Stage-10 Upgraded ARC Benchmark (Hybrid Geodesic/Nudge)")
    p.add_argument("--mode", type=str, default=DEFAULTS["mode"], choices=["geodesic","nudge"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--target-var", type=float, default=DEFAULTS["target_var"])
    p.add_argument("--pca-min-dims", type=int, default=DEFAULTS["pca_min_dims"])
    p.add_argument("--pca-max-dims", type=int, default=DEFAULTS["pca_max_dims"])
    p.add_argument("--lam", type=float, default=DEFAULTS["lam"])
    p.add_argument("--dt", type=float, default=DEFAULTS["dt"])
    p.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    p.add_argument("--gamma", type=float, default=DEFAULTS["gamma"])
    p.add_argument("--tau", type=float, default=DEFAULTS["tau"])
    p.add_argument("--mass-scale", type=float, default=DEFAULTS["mass_scale"])
    p.add_argument("--early-stop-window", type=int, default=DEFAULTS["early_stop_window"])
    return p.parse_args()

def main():
    args = parse_args()
    seed_all(args.seed)
    device = pick_device()
    print(f"[Init] mode={args.mode} seed={args.seed} device={device.type}")
    print(f"[Geom] λ={args.lam} γ={args.gamma} dt={args.dt} steps={args.steps} mass_scale={args.mass_scale}")

    # Build training latents
    prompts, labels = training_prompts()
    extractor = LatentExtractor(device)
    train_latents = np.stack([extractor.get(p) for p in prompts], axis=0)

    # PCA & reduced space
    pca, Z = pca_fit_transform(train_latents, target_var=args.target_var,
                               MIN_DIMS=args.pca_min_dims, MAX_DIMS=args.pca_max_dims, verbose=True)

    # Anchors/masses and order
    centers, masses, labels_in_order = compute_anchors(Z, labels)
    masses = masses / (masses.sum() + 1e-9)
    masses *= args.mass_scale
    print("[Anchors] order:", labels_in_order)

    # Pooled-covariance classifier
    cls_stats = shared_cov_stats(Z, labels, labels_in_order, ridge=1e-3, shrink=0.2)

    ctx = ARCContext(
        pca=pca, centers=centers, masses=masses,
        labels_in_order=labels_in_order, cls_stats=cls_stats,
        extractor=extractor, args=args
    )

    # Run benchmark
    run_arc_benchmark_12(ctx)

if __name__ == "__main__":
    main()
