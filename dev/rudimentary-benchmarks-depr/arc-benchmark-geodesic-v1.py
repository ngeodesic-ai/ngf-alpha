#!/usr/bin/env python3
# arc-benchmark-geodesic-v1_perp.py
# Oracle execution check using *orthogonalized residual energy* (E_perp).
# This is a drop-in replacement for the scoring part of arc-benchmark-geodesic-v1.py
# to avoid common-mode raw-energy ties.

# python3 arc-benchmark-geodesic-v1.py --device cpu

# Output
# [PCA] samples=30 feat=768 cap=30 -> n=19 (cum var≈0.9968, whiten=True)
# [01] true=rotate   pred=rotate   ok=True  areas_raw=[0.921 0.891 0.857]  areas_perp=[0.    0.    0.024]
# [02] true=rotate   pred=rotate   ok=True  areas_raw=[0.729 0.718 0.678]  areas_perp=[0.    0.    0.015]
# [03] true=rotate   pred=rotate   ok=True  areas_raw=[1.195 1.093 1.023]  areas_perp=[0.    0.    0.032]
# [04] true=rotate   pred=rotate   ok=True  areas_raw=[1.357 1.269 1.242]  areas_perp=[0.    0.    0.051]
# [05] true=flip_h   pred=flip_h   ok=True  areas_raw=[0.941 0.891 0.84 ]  areas_perp=[0.026 0.    0.   ]
# [06] true=flip_h   pred=flip_h   ok=True  areas_raw=[0.741 0.717 0.668]  areas_perp=[0.016 0.    0.   ]
# [07] true=flip_h   pred=flip_h   ok=True  areas_raw=[1.223 1.091 1.   ]  areas_perp=[0.039 0.    0.   ]
# [08] true=flip_h   pred=flip_h   ok=True  areas_raw=[1.072 0.971 0.916]  areas_perp=[0.032 0.    0.   ]
# [09] true=flip_v   pred=flip_v   ok=True  areas_raw=[0.915 0.916 0.834]  areas_perp=[0.    0.034 0.   ]
# [10] true=flip_v   pred=flip_v   ok=True  areas_raw=[0.725 0.733 0.664]  areas_perp=[0.    0.022 0.   ]
# [11] true=flip_v   pred=flip_v   ok=True  areas_raw=[1.181 1.13  0.99 ]  areas_perp=[0.    0.052 0.   ]
# [12] true=flip_v   pred=flip_v   ok=True  areas_raw=[1.139 1.133 0.965]  areas_perp=[0.    0.051 0.   ]

# [Oracle-Exec ⟂] Accuracy with known primitive (perp energy): 12/12 = 100.0%

from __future__ import annotations
import argparse, math, os, random, time
from typing import List, Dict, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import GPT2Tokenizer, GPT2LMHeadModel

SEED = 42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

DEVICE_DEFAULT = "cpu"

# PCA
PCA_TARGET_VAR = 0.996
PCA_MIN, PCA_MAX = 8, 24
PCA_WHITEN = True

# Geometry
LAMBDA = 0.35
GAMMA  = 0.00
DT     = 0.02
STEPS  = 700
EPS    = 1e-9

SPEED_EUCLID = 0.12   # keep ||v|| constant

LABELS = ["flip_h","flip_v","rotate"]

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

TEST = [
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

# ---------- Embedding ----------
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
        stack = torch.stack(hs[-4:], dim=0)         # last 4 layers
        mean_layers = stack.mean(dim=0)[0]          # (seq, hid)
        vec = mean_layers.mean(dim=0).cpu().numpy().astype(np.float64)
        return vec

def grid_to_prompt(grid):
    return f"Input grid {grid}. Produce the matching output grid."

# ---------- PCA ----------
class Projector:
    def __init__(self, scaler: StandardScaler, pca: PCA):
        self.scaler = scaler; self.pca = pca
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

def compute_anchors(Z: np.ndarray, y: List[str], labels: List[str]):
    centers = []
    for c in labels:
        idx = np.array([yy == c for yy in y])
        centers.append(Z[idx].mean(axis=0))
    return np.vstack(centers)

# ---------- Geometry ----------
def V_center(x: np.ndarray, c: np.ndarray, mass: float = 1.0) -> float:
    r = np.linalg.norm(x - c) + EPS
    return -mass / r

def gradV_center(x: np.ndarray, c: np.ndarray, mass: float = 1.0) -> np.ndarray:
    # grad of -1/r is +(x-c)/r^3
    d = x - c
    r = np.linalg.norm(d) + EPS
    return mass * d / (r**3)

def lnphi_and_grad(x: np.ndarray, centers: np.ndarray, include_mask: np.ndarray):
    lnphi = 0.0
    g = np.zeros_like(x)
    for k, inc in enumerate(include_mask):
        if not inc: 
            continue
        lnphi += LAMBDA * V_center(x, centers[k])
        g     += gradV_center(x, centers[k])
    return lnphi, LAMBDA * g

def rk4_step_const_euclid(x: np.ndarray, v: np.ndarray, dt: float, centers: np.ndarray, include_mask: np.ndarray):
    def acc(x_, v_):
        _, gL = lnphi_and_grad(x_, centers, include_mask)
        v_dot = float(np.dot(v_, gL))
        v_norm2 = float(np.dot(v_, v_))
        ch = 2.0 * v_ * v_dot - v_norm2 * gL
        return -ch - GAMMA * v_
    k1x = v;              k1v = acc(x, v)
    k2x = v + 0.5*dt*k1v; k2v = acc(x + 0.5*dt*k1x, v + 0.5*dt*k1v)
    k3x = v + 0.5*dt*k2v; k3v = acc(x + 0.5*dt*k2x, v + 0.5*dt*k2v)
    k4x = v + dt*k3v;     k4v = acc(x + dt*k3x, v + dt*k3v)
    x_new = x + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
    v_new = v + (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
    n = np.linalg.norm(v_new) + 1e-12
    v_new = (SPEED_EUCLID / n) * v_new
    return x_new, v_new

# ---------- Discriminative residuals ----------
def orth_residuals(grads: np.ndarray) -> np.ndarray:
    """
    For each concept gradient g_k, remove its projection onto the span of {g_j: j!=k}.
    grads: (K, d) -> residuals (K, d)
    """
    K, d = grads.shape
    R = np.zeros_like(grads)
    for k in range(K):
        idx = [j for j in range(K) if j != k]
        B = grads[idx]            # (K-1, d)
        gk = grads[k]
        if B.shape[0] == 0:
            R[k] = gk
            continue
        BBt = B @ B.T + 1e-9*np.eye(B.shape[0])
        coef = np.linalg.solve(BBt, B @ gk)     # (K-1,)
        proj = B.T @ coef                       # (d,)
        R[k] = gk - proj
    return R

# ---------- Geodesic run ----------
def run_geodesic(x0: np.ndarray, centers: np.ndarray, include_mask: np.ndarray):
    K = centers.shape[0]
    # init v toward nearest included
    dists = np.linalg.norm(centers - x0, axis=1)
    nearest = int(np.argmin(np.where(include_mask, dists, np.inf)))
    dir0 = centers[nearest] - x0; n0 = np.linalg.norm(dir0) + 1e-9
    v = SPEED_EUCLID * (dir0 / n0)

    x = x0.copy()
    E_raw  = np.zeros((int(STEPS), int(K)), dtype=np.float64)
    E_perp = np.zeros((int(STEPS), int(K)), dtype=np.float64)

    for t in range(int(STEPS)):
        lnphi, _ = lnphi_and_grad(x, centers, include_mask)

        grads = np.array([gradV_center(x, centers[k]) for k in range(K)])  # (K,d)
        raw = -(grads @ v)                          # (K,)
        raw = np.clip(raw, 0.0, None)
        E_raw[t, :] = raw

        R = orth_residuals(grads)
        perp = -(R @ v)                             # (K,)
        perp = np.clip(perp, 0.0, None)
        E_perp[t, :] = perp

        x, v = rk4_step_const_euclid(x, v, DT, centers, include_mask)

    return E_raw, E_perp

# ---------- Pipeline ----------
def prepare(device: str):
    ex = Extractor(device)
    Xtr, ytr = [], []
    for c, prompts in TRAIN.items():
        for s in prompts:
            Xtr.append(ex.get(s)); ytr.append(c)
    Xtr = np.vstack(Xtr)
    proj = fit_projector(Xtr)
    Ztr = proj.transform(Xtr)
    centers = compute_anchors(Ztr, ytr, LABELS)
    return ex, proj, centers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=DEVICE_DEFAULT)
    args = ap.parse_args()

    ex, proj, centers = prepare(args.device)

    # Build test set
    Xte, yte = [], []
    for grid, label in TEST:
        Xte.append(ex.get(grid_to_prompt(grid))); yte.append(label)
    Zte = proj.transform(np.vstack(Xte))

    correct = 0
    for i, (z0, true) in enumerate(zip(Zte, yte), 1):
        mask = np.zeros(len(LABELS), dtype=bool)
        k_true = LABELS.index(true)
        mask[k_true] = True

        E_raw, E_perp = run_geodesic(z0, centers, mask)
        areas_raw  = E_raw.sum(axis=0)
        areas_perp = E_perp.sum(axis=0)

        pred = LABELS[int(np.argmax(areas_perp))]
        ok = (pred == true)
        correct += int(ok)
        print(f"[{i:02d}] true={true:<7}  pred={pred:<7}  ok={ok}  areas_raw={np.round(areas_raw,3)}  areas_perp={np.round(areas_perp,3)}")

    acc = correct / len(yte)
    print(f"\n[Oracle-Exec ⟂] Accuracy with known primitive (perp energy): {correct}/{len(yte)} = {acc*100:.1f}%")

if __name__ == "__main__":
    main()
