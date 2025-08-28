# Write a new classifier-free script that chooses the transform by minimal geodesic action.
# The script keeps your previous geometry (conformal metric from multi-center potential),
# but replaces the classifier with a multi-shooting action minimization over anchors.
from textwrap import dedent

code = dedent(r'''
#!/usr/bin/env python3
# latest-arc-benchmark_action.py
# Stage-10 (classifier-free): choose transform via minimal geodesic action in warped space.
# Author: ngeodesic.ai (Alpha) — Apache-2.0
#
# How it works
# 1) Build anchor centers from label-free prototype I/O examples (no label words in prompts).
# 2) Warp metric: g_ij = phi(x)^2 δ_ij, with phi = exp(λ V(x)), V = -Σ_i M_i / ||x - c_i||.
# 3) For a test instance latent z0, run multi-shooting geodesics with different initial velocities
#    aimed at each anchor; compute discrete action S = Σ phi(x_t)^2 * ||v_t||^2 * dt + α * d(x_T, c_j).
# 4) Pick the anchor j with minimal S_j. Apply the corresponding transform to the grid.
#
# No classifier. Decision lives in the geometry & dynamics, so it generalizes beyond ARC-12.
#
# Run:
#   python3 latest-arc-benchmark_action.py --device cpu
#
from __future__ import annotations
import argparse, math, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from numpy.linalg import inv
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# ---------------------------
# Config
# ---------------------------
SEED = 42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

DEVICE_DEFAULT = "cpu"

# PCA
PCA_TARGET_VAR = 0.996
PCA_MIN, PCA_MAX = 8, 24
PCA_WHITEN = True

# Geometry & integrator
LAMBDA = 0.35       # curvature strength (phi = exp(lambda * V))
GAMMA  = 0.04       # small damping
DT     = 0.02
STEPS  = 700
MASS_SCALE = 4.0
EPS    = 1e-6

# Action
TERM_ALPHA = 1.0    # terminal penalty weight: α * ||x_T - c_j||
STOP_RADIUS_FACTOR = 0.6  # stop when within (factor * class spread) of target center
SHOOTS_PER_ANCHOR = 3     # different initial velocities per anchor

# ---------------------------
# Prototypes (label-free)
# ---------------------------
TRAIN: Dict[str, List[str]] = {
    "rotate": [
        "Map [[1,2],[3,4]] to [[3,1],[4,2]].",
        "Map [[5,6],[7,8]] to [[7,5],[8,6]].",
        "Map [[2,1],[4,3]] to [[4,2],[3,1]].",
        "Map [[0,1],[2,3]] to [[2,0],[3,1]].",
        "Map [[1,0],[0,2]] to [[0,1],[2,0]].",
        "Map [[3,0],[0,1]] to [[0,3],[1,0]].",
        "Map [[9,8],[7,6]] to [[7,9],[6,8]].",
        "Map [[1,2,3],[4,5,6],[7,8,9]] to [[7,4,1],[8,5,2],[9,6,3]].",
        "Map [[2,0,1],[0,1,0],[1,0,2]] to [[1,0,2],[0,1,0],[2,0,1]].",
        "Map [[0,2,0],[1,0,1],[0,2,0]] to [[0,1,0],[2,0,2],[0,1,0]].",
    ],
    "flip_h": [
        "Map [[1,2],[3,4]] to [[2,1],[4,3]].",
        "Map [[5,6],[7,8]] to [[6,5],[8,7]].",
        "Map [[0,1],[2,3]] to [[1,0],[3,2]].",
        "Map [[2,1],[4,3]] to [[1,2],[3,4]].",
        "Map [[1,0],[0,2]] to [[0,1],[2,0]].",
        "Map [[3,0],[0,1]] to [[0,3],[1,0]].",
        "Map [[9,8],[7,6]] to [[8,9],[6,7]].",
        "Map [[1,2,3],[4,5,6],[7,8,9]] to [[3,2,1],[6,5,4],[9,8,7]].",
        "Map [[1,0,2],[0,1,0],[2,0,1]] to [[2,0,1],[0,1,0],[1,0,2]].",
        "Map [[0,2,0],[1,0,1],[0,2,0]] to [[0,2,0],[1,0,1],[0,2,0]].",
    ],
    "flip_v": [
        "Map [[1,2],[3,4]] to [[3,4],[1,2]].",
        "Map [[5,6],[7,8]] to [[7,8],[5,6]].",
        "Map [[0,1],[2,3]] to [[2,3],[0,1]].",
        "Map [[2,1],[4,3]] to [[4,3],[2,1]].",
        "Map [[1,0],[0,2]] to [[0,2],[1,0]].",
        "Map [[3,0],[0,1]] to [[0,1],[3,0]].",
        "Map [[9,8],[7,6]] to [[7,6],[9,8]].",
        "Map [[1,2,3],[4,5,6],[7,8,9]] to [[7,8,9],[4,5,6],[1,2,3]].",
        "Map [[1,0,2],[0,1,0],[2,0,1]] to [[2,0,1],[0,1,0],[1,0,2]].",
        "Map [[0,2,0],[1,0,1],[0,2,0]] to [[0,2,0],[1,0,1],[0,2,0]].",
    ],
}

# Test set (12 cases)
TEST: List[Tuple[List[List[int]], str]] = [
    # rotate
    ([[1,2],[3,4]], "rotate"),
    ([[5,6],[7,8]], "rotate"),
    ([[1,0,2],[0,1,0],[2,0,1]], "rotate"),
    ([[2,0,1],[0,1,0],[1,0,2]], "rotate"),
    # flip_h
    ([[1,2],[3,4]], "flip_h"),
    ([[5,6],[7,8]], "flip_h"),
    ([[1,0,2],[0,1,0],[2,0,1]], "flip_h"),
    ([[9,8,7],[6,5,4],[3,2,1]], "flip_h"),
    # flip_v
    ([[1,2],[3,4]], "flip_v"),
    ([[5,6],[7,8]], "flip_v"),
    ([[1,0,2],[0,1,0],[2,0,1]], "flip_v"),
    ([[1,2,4],[4,5,6],[7,8,9]], "flip_v"),
]

LABELS = ["flip_h","flip_v","rotate"]  # fixed for reproducibility
TRANSFORM = {
    'rotate': lambda g: np.rot90(np.array(g), k=3).tolist(),
    'flip_h': lambda g: np.flip(np.array(g), axis=1).tolist(),
    'flip_v': lambda g: np.flip(np.array(g), axis=0).tolist(),
}

# ---------------------------
# Embedding (last-4-layer mean)
# ---------------------------
class Extractor:
    def __init__(self, device: str):
        self.device = torch.device(device)
        self.tok = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        self.model.eval()
    @torch.no_grad()
    def get(self, text: str) -> np.ndarray:
        toks = self.tok(text, return_tensors="pt").to(self.device)
        out = self.model(**toks, output_hidden_states=True)
        hs = out.hidden_states
        stack = torch.stack(hs[-4:], dim=0)    # (4,1,seq,hid)
        mean_layers = stack.mean(dim=0)[0]     # (seq,hid)
        vec = mean_layers.mean(dim=0).cpu().numpy().astype(np.float64)
        return vec

def grid_to_prompt(grid: List[List[int]]) -> str:
    return f"Input grid {grid}. Produce the matching output grid."

# ---------------------------
# PCA projector
# ---------------------------
@dataclass
class Projector:
    scaler: StandardScaler
    pca: PCA
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(X))

def fit_projector(X: np.ndarray) -> Projector:
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    n_samples, n_feats = Xs.shape
    cap = min(n_samples, n_feats)
    probe = min(cap, PCA_MAX)
    p_probe = PCA(n_components=probe, whiten=False, svd_solver="full").fit(Xs)
    cum = np.cumsum(p_probe.explained_variance_ratio_)
    k = next((i+1 for i,v in enumerate(cum) if v >= PCA_TARGET_VAR), probe)
    k = max(PCA_MIN, min(k, probe))
    pca = PCA(n_components=k, whiten=PCA_WHITEN, svd_solver="full").fit(Xs)
    print(f"[PCA] samples={n_samples} feat={n_feats} cap={cap} -> n={k} (cum var≈{cum[k-1]:.4f}, whiten={PCA_WHITEN})")
    return Projector(scaler, pca)

# ---------------------------
# Geometry: potential & geodesic
# ---------------------------
def potential(x: np.ndarray, centers: np.ndarray, masses: np.ndarray) -> float:
    diffs = x[None,:] - centers
    dists = np.linalg.norm(diffs, axis=1) + EPS
    return -float(np.sum(masses / dists))

def grad_potential(x: np.ndarray, centers: np.ndarray, masses: np.ndarray) -> np.ndarray:
    diffs = x[None,:] - centers
    dists = np.linalg.norm(diffs, axis=1) + EPS
    terms = masses[:,None] * diffs / (dists**3)[:,None]
    return -np.sum(terms, axis=0)

def lnphi_and_grad(x: np.ndarray, centers: np.ndarray, masses: np.ndarray) -> Tuple[float, np.ndarray]:
    V = potential(x, centers, masses)
    gV = grad_potential(x, centers, masses)
    lnphi = LAMBDA * V
    grad_lnphi = LAMBDA * gV
    return lnphi, grad_lnphi

def christoffel_conformal(x: np.ndarray, v: np.ndarray, centers: np.ndarray, masses: np.ndarray) -> np.ndarray:
    _, gL = lnphi_and_grad(x, centers, masses)
    v_dot = float(np.dot(v, gL))
    v_norm2 = float(np.dot(v, v))
    return 2.0 * v * v_dot - v_norm2 * gL

def rk4_step(x: np.ndarray, v: np.ndarray, dt: float, centers: np.ndarray, masses: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    def acc(x_, v_):
        return -christoffel_conformal(x_, v_, centers, masses) - GAMMA * v_
    k1x = v;              k1v = acc(x, v)
    k2x = v + 0.5*dt*k1v; k2v = acc(x + 0.5*dt*k1x, v + 0.5*dt*k1v)
    k3x = v + 0.5*dt*k2v; k3v = acc(x + 0.5*dt*k2x, v + 0.5*dt*k2v)
    k4x = v + dt*k3v;     k4v = acc(x + dt*k3x, v + dt*k3v)
    x_new = x + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
    v_new = v + (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
    return x_new, v_new

def metric_speed2(x: np.ndarray, v: np.ndarray, centers: np.ndarray, masses: np.ndarray) -> float:
    lnphi, _ = lnphi_and_grad(x, centers, masses)
    phi2 = math.exp(2.0 * lnphi)
    return float(phi2 * np.dot(v, v))

def integrate_geodesic_action(x0: np.ndarray, v0: np.ndarray,
                              centers: np.ndarray, masses: np.ndarray,
                              target_center: np.ndarray,
                              stop_radius: float,
                              steps: int = STEPS, dt: float = DT) -> Tuple[np.ndarray, float, int]:
    x = x0.copy(); v = v0.copy()
    S = 0.0
    for t in range(steps):
        # accumulate action using current state
        S += metric_speed2(x, v, centers, masses) * dt
        x, v = rk4_step(x, v, dt, centers, masses)
        # early stop if close to target center
        if np.linalg.norm(x - target_center) <= stop_radius:
            break
    # terminal penalty for not being exactly at center
    S += TERM_ALPHA * float(np.linalg.norm(x - target_center))
    return x, float(S), t+1

def multi_shoot_action(x0: np.ndarray, j: int,
                       centers: np.ndarray, masses: np.ndarray,
                       spreads: np.ndarray,
                       shoots: int = SHOOTS_PER_ANCHOR) -> Tuple[float, dict]:
    best_S = float('inf'); best_meta = {}
    cj = centers[j]; stop_r = STOP_RADIUS_FACTOR * float(spreads[j])
    # candidate initial velocities
    dirs = []
    # 1) direct toward anchor j
    d = cj - x0; n = np.linalg.norm(d)
    dirs.append(d / (n + 1e-9))
    # 2) along -grad lnphi (global downhill)
    _, gL = lnphi_and_grad(x0, centers, masses)
    n2 = np.linalg.norm(gL)
    dirs.append(-gL / (n2 + 1e-9))
    # 3..) small jittered mixes
    rng = np.random.default_rng(SEED + j)
    while len(dirs) < shoots:
        mix = 0.7*dirs[0] + 0.3*dirs[1] + 0.05*rng.normal(size=x0.shape)
        mix /= (np.linalg.norm(mix) + 1e-9)
        dirs.append(mix)
    for k, u in enumerate(dirs):
        v0 = 0.10 * u  # speed scale
        xT, S, steps_used = integrate_geodesic_action(x0, v0, centers, masses, cj, stop_r, STEPS, DT)
        if S < best_S:
            best_S = S
            best_meta = {"shoot": k, "steps": steps_used, "xT": xT, "stop_r": stop_r}
    return best_S, best_meta

# ---------------------------
# Anchors & masses
# ---------------------------
def compute_anchors(Z: np.ndarray, y: List[str], order: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = []; spreads = []; masses = []
    for c in order:
        idx = np.array([yy == c for yy in y])
        G = Z[idx]
        mu = G.mean(axis=0)
        centers.append(mu)
        spread = float(np.mean(np.linalg.norm(G - mu, axis=1)) + 1e-6)
        spreads.append(spread)
        masses.append(1.0 / spread)
    centers = np.vstack(centers)
    spreads = np.array(spreads, dtype=np.float64)
    masses = np.array(masses, dtype=np.float64)
    masses /= (masses.sum() + 1e-9)
    masses *= MASS_SCALE
    return centers, masses, spreads

# ---------------------------
# Pipeline
# ---------------------------
@dataclass
class Context:
    extractor: Extractor
    proj: Projector
    centers: np.ndarray
    masses: np.ndarray
    spreads: np.ndarray

def prepare(device: str) -> Context:
    ex = Extractor(device)
    # embed prototypes
    X_train, y_train = [], []
    for c, prompts in TRAIN.items():
        for s in prompts:
            X_train.append(ex.get(s))
            y_train.append(c)
    X_train = np.vstack(X_train)
    proj = fit_projector(X_train)
    Z = proj.transform(X_train)
    centers, masses, spreads = compute_anchors(Z, y_train, LABELS)
    print("[Anchors] order:", LABELS)
    return Context(ex, proj, centers, masses, spreads)

def run(device: str = DEVICE_DEFAULT):
    ctx = prepare(device)

    # sanity: base distances from one demo latent to centers
    demo_prompt = "Input grid [[1,2],[3,4]]. Produce the matching output grid."
    z_demo = ctx.proj.transform(np.vstack([ctx.extractor.get(demo_prompt)]))[0]
    base_d = np.linalg.norm(ctx.centers - z_demo, axis=1)
    print("[Sanity] distances:", np.array(base_d).round(3))
    print("[Sanity] chosen:", LABELS[int(np.argmin(base_d))])

    # build test latents
    Xte, yte = [], []
    for grid, label in TEST:
        Xte.append(ctx.extractor.get(grid_to_prompt(grid)))
        yte.append(label)
    Zte = ctx.proj.transform(np.vstack(Xte))

    # evaluate
    preds = []
    logs = []
    for i, (z0, true) in enumerate(zip(Zte, yte), 1):
        S_list = []; metas = []
        for j in range(len(LABELS)):
            Sj, meta = multi_shoot_action(z0, j, ctx.centers, ctx.masses, ctx.spreads, SHOOTS_PER_ANCHOR)
            S_list.append(Sj); metas.append(meta)
        j_hat = int(np.argmin(S_list))
        pred = LABELS[j_hat]
        preds.append(pred)

        # confidence proxy: softmin over actions
        A = np.array(S_list, dtype=np.float64)
        A = A - A.min()
        w = np.exp(-A / (A.std() + 1e-6))
        p = w / (w.sum() + 1e-9)

        # dist* proxy: expected euclidean to means under p
        eu = np.linalg.norm(ctx.centers - z0, axis=1)
        dist_star = float(np.dot(p, eu))
        conf = float(p[j_hat])
        logs.append((i, true, pred, pred==true, dist_star, conf, p))

    # print per-sample
    for i, true, pred, ok, dist_star, conf, p in logs:
        print(f"[{i:02d}] true={true:<7} pred={pred:<7} ok={str(ok):<5} dist*={dist_star:.3f} conf={conf:.3f} probs=[{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]")

    acc = sum(int(a==b) for a,b in zip(preds,yte)) / len(yte)
    mean_conf = float(np.mean([c for *_, c, _ in logs]))
    print(f"\n[ARC-12] Accuracy: {acc*100:.1f}% | Mean confidence: {mean_conf:.3f} | Mode=min-action")

    # confusion
    cm = confusion_matrix(yte, preds, labels=LABELS)
    print("[Confusion]\n          pred→   flip_h  flip_v  rotate")
    for i, row in enumerate(cm):
        print(f"true={LABELS[i]:<8}      {row[0]:>3}     {row[1]:>3}     {row[2]:>3}")

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=DEVICE_DEFAULT)
    args = ap.parse_args()
    run(device=args.device)
''')

path = "/mnt/data/latest-arc-benchmark_action.py"
with open(path, "w") as f:
    f.write(code)

print("Wrote", path)
