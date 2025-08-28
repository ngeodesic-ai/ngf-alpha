# Create a "v4 matched-filter" parser that solves the toy case by picking exactly one task
# using cross-correlation with exclusive-run prototypes.
# from textwrap import dedent

# code = dedent(r"""
#!/usr/bin/env python3
# arc_task_parser_geodesic_v4_matched.py
# Parse-only: one-of-K via matched filters from exclusive geodesics.
# - Runs a single include-all geodesic to get RAW energy traces E_k(t)
# - For each concept k, runs an exclusive geodesic (only k field) to get prototype P_k(t)
# - Scores by normalized cross-correlation corr_k = corr(E_k, P_k)
# - Picks argmax corr_k as the single task; reports order=that task, no concurrency
# This is designed to *solve the toy ARC-12 case* decisively.

from __future__ import annotations
import argparse, math, os, time, random
from typing import List, Tuple, Dict

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

# Geometry
LAMBDA = 0.35
GAMMA  = 0.00
DT     = 0.02
STEPS  = 700
EPS    = 1e-9

SPEED_EUCLID = 0.12   # keep ||v|| constant

# Smoothing for correlation
SMOOTH_WIN = 7

PLOT = True

LABELS = ["flip_h","flip_v","rotate"]

# ---------------------------
# Tiny synthetic corpus
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

# ---------------------------
# Embedding
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
        stack = torch.stack(hs[-4:], dim=0)
        mean_layers = stack.mean(dim=0)[0]
        vec = mean_layers.mean(dim=0).cpu().numpy().astype(np.float64)
        return vec

def grid_to_prompt(grid):
    return f"Input grid {grid}. Produce the matching output grid."

# ---------------------------
# PCA
# ---------------------------
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

# ---------------------------
# Fields & geometry
# ---------------------------
def compute_anchors(Z: np.ndarray, y: List[str], labels: List[str]):
    centers = []
    for c in labels:
        idx = np.array([yy == c for yy in y])
        G = Z[idx]
        centers.append(G.mean(axis=0))
    return np.vstack(centers)

def V_center(x: np.ndarray, c: np.ndarray, mass: float = 1.0) -> float:
    r = np.linalg.norm(x - c) + EPS
    return -mass / r

def gradV_center(x: np.ndarray, c: np.ndarray, mass: float = 1.0) -> np.ndarray:
    # Correct sign for V=-1/r: grad V = +(x-c)/r^3
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

def metric_speed2(x: np.ndarray, v: np.ndarray, lnphi: float) -> float:
    phi2 = math.exp(2.0 * lnphi)
    return float(phi2 * np.dot(v, v))

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
    # renormalize to constant Euclidean speed
    n = np.linalg.norm(v_new) + 1e-12
    v_new = (SPEED_EUCLID / n) * v_new
    return x_new, v_new

# ---------------------------
# Action & aligned energy
# ---------------------------
def run_geodesic(x0: np.ndarray, centers: np.ndarray, include_mask: np.ndarray):
    K = centers.shape[0]
    # init toward nearest included
    dists = np.linalg.norm(centers - x0, axis=1)
    nearest = int(np.argmin(np.where(include_mask, dists, np.inf)))
    dir0 = centers[nearest] - x0; n0 = np.linalg.norm(dir0) + 1e-9
    v = SPEED_EUCLID * (dir0 / n0)

    x = x0.copy()
    S = 0.0
    P_raw    = np.zeros((STEPS, K), dtype=np.float64)
    RHO = np.zeros((STEPS, K, K), dtype=np.float64)

    for t in range(STEPS):
        lnphi, _ = lnphi_and_grad(x, centers, include_mask)
        vs2 = metric_speed2(x, v, lnphi)
        S += vs2 * DT

        grads = np.array([gradV_center(x, centers[k]) for k in range(K)])
        raw = -(grads @ v)                # positive when moving toward centers
        raw = np.clip(raw, 0.0, None)
        P_raw[t]    = raw

        norms = np.linalg.norm(grads, axis=1) + 1e-9
        dirs  = grads / norms[:, None]
        RHO[t] = dirs @ dirs.T

        x, v = rk4_step_const_euclid(x, v, DT, centers, include_mask)

    return S, P_raw, RHO

def smooth_ma(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: return x
    k = np.ones(win) / win
    return np.apply_along_axis(lambda s: np.convolve(s, k, mode="same"), axis=0, arr=x)

def normxcorr(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    a = a - a.mean()
    b = b - b.mean()
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))

# ---------------------------
# Ablation-free matched parse
# ---------------------------
def matched_parse(x0: np.ndarray, centers: np.ndarray, labels: List[str]):
    K = len(labels)
    include_all = np.ones(K, dtype=bool)
    S_all, Pr_all, RHO_all = run_geodesic(x0, centers, include_all)
    Ps = smooth_ma(Pr_all, SMOOTH_WIN)

    # Exclusive prototypes per channel
    corrs = []
    protos = []
    for k in range(K):
        mask = np.zeros(K, dtype=bool); mask[k] = True
        S_only, Pr_only, _ = run_geodesic(x0, centers, mask)
        proto = smooth_ma(Pr_only[:,k], SMOOTH_WIN)
        protos.append(proto)
        corrs.append(normxcorr(Ps[:,k], proto))
    corrs = np.array(corrs, dtype=np.float64)

    k_hat = int(np.argmax(corrs))
    present = [labels[k_hat]]
    order = [labels[k_hat]]
    conc = []

    return {
        "S_all": float(S_all),
        "corrs": corrs.tolist(),
        "present": present,
        "order": order,
        "concurrency": conc,
        "Ps": Ps,
        "protos": protos,
    }

# ---------------------------
# Pipeline
# ---------------------------
def prepare(device: str):
    ex = Extractor(device)
    # build anchor set
    Xtr, ytr = [], []
    for c, prompts in TRAIN.items():
        for s in prompts:
            Xtr.append(ex.get(s)); ytr.append(c)
    Xtr = np.vstack(Xtr)
    proj = fit_projector(Xtr)
    Ztr = proj.transform(Xtr)
    centers = compute_anchors(Ztr, ytr, LABELS)
    return ex, proj, centers

def run(device: str = DEVICE_DEFAULT, outdir: str = "."):
    os.makedirs(outdir, exist_ok=True)
    ex, proj, centers = prepare(device)

    demo = "Input grid [[1,2],[3,4]]. Produce the matching output grid."
    z_demo = proj.transform(np.vstack([ex.get(demo)]))[0]
    dists = np.linalg.norm(centers - z_demo, axis=1)
    print("[Anchors] order:", LABELS)
    print("[Sanity] distances:", np.array(dists).round(3))
    print("[Sanity] chosen:", LABELS[int(np.argmin(dists))])

    Xte, yte = [], []
    for grid, label in TEST:
        Xte.append(ex.get(grid_to_prompt(grid))); yte.append(label)
    Zte = proj.transform(np.vstack(Xte))

    for i, (z0, true) in enumerate(zip(Zte, yte), 1):
        tag = f"sample{i:02d}"
        res = matched_parse(z0, centers, LABELS)
        present = res["present"]; order = res["order"]; conc = res["concurrency"]
        corrs = np.array(res["corrs"]).round(3).tolist()
        print(f"[{i:02d}] true={true:<7} -> Task: {present[0]} | Order: {order[0]} | Concurrency: none | corrs={corrs}")

        if PLOT:
            try:
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(7.6, 4.0))
                for k, name in enumerate(LABELS):
                    plt.plot(res["Ps"][:,k], label=f"E_raw {name}")
                # overlay the chosen prototype for visual compare
                k_hat = LABELS.index(present[0])
                plt.plot(res["protos"][k_hat], linestyle="--", linewidth=2.0, label=f"proto {present[0]} (exclusive)")
                plt.xlabel("step"); plt.ylabel("aligned power (RAW, smoothed)")
                plt.title(f"Matched parse — {tag} — picked: {present[0]}")
                plt.legend()
                ts = time.strftime("%Y%m%d-%H%M%S")
                fname = os.path.join(outdir, f"parse_energy_v4_matched_{tag}_{ts}.png")
                plt.savefig(fname, bbox_inches="tight"); plt.close(fig)
                print(f"[Plot] saved: {fname}")
            except Exception as e:
                print("[Plot] skipped:", e)

# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=DEVICE_DEFAULT)
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()
    run(device=args.device, outdir=args.outdir)

if __name__ == "__main__":
    main()
# """)

# path = "/mnt/data/arc_task_parser_geodesic_v4_matched.py"
# with open(path, "w") as f:
#     f.write(code)

# print("Wrote", path)
