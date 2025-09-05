# Create a new script that implements the mixture-of-fields controller (classifier-free)
# and saves per-sample plots of the concept weights w(t) in the current directory.

# from textwrap import dedent

# code = dedent(r"""
#!/usr/bin/env python3
# latest-arc-benchmark_mixture.py
# Classifier-free Step-9: joint geodesic + simplex weights controller.
# Chooses and executes concepts (flip_h, flip_v, rotate) by evolving x(t) and w(t).
# Saves per-sample plots of w(t) in the working directory (or --outdir).

from __future__ import annotations
import argparse, math, random, os, time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# ---------------
# Config
# ---------------
SEED = 42
np.random.seed(SEED); random.seed(SEED); torch.manual_seed(SEED)

DEVICE_DEFAULT = "cpu"

# PCA
PCA_TARGET_VAR = 0.996
PCA_MIN, PCA_MAX = 8, 24
PCA_WHITEN = True

# Geometry / integrator
LAMBDA = 0.35       # conformal strength
GAMMA  = 0.04       # damping
DT     = 0.02
STEPS  = 700
EPS    = 1e-6

# Controller (weights on simplex)
LAMBDA_SWITCH = 0.10  # TV/switching penalty (prox on deltas)
HYST_DWELL_WIN = 10   # consecutive steps above threshold to trigger lock
HYST_DWELL_STEPS = 30
HYST_THRESHOLD = 0.7
MU_PROGRESS = 0.35   # a bit more progress pull
ETA_W = 0.12         # slightly smaller step

# Decision smoothing
FINAL_AVG_TAIL = 25   # average w over last N steps for the decision

# Plotting
PLOT = True

# ---------------
# Prototypes (label-free)
# ---------------
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

LABELS = ["flip_h","flip_v","rotate"]

TRANSFORM = {
    'rotate': lambda g: np.rot90(np.array(g), k=3).tolist(),
    'flip_h': lambda g: np.flip(np.array(g), axis=1).tolist(),
    'flip_v': lambda g: np.flip(np.array(g), axis=0).tolist(),
}

# ---------------
# Embedding (last-4-layer mean)
# ---------------
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

# ---------------
# PCA projector
# ---------------
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

# ---------------
# Field registry (V_k, grad V_k, H_k, grad H_k)
# ---------------
@dataclass
class Field:
    name: str
    center: np.ndarray   # (d,)
    spread: float

def V_center(x: np.ndarray, c: np.ndarray, mass: float = 1.0) -> float:
    r = np.linalg.norm(x - c) + EPS
    return -mass / r

def gradV_center(x: np.ndarray, c: np.ndarray, mass: float = 1.0) -> np.ndarray:
    d = x - c
    r = np.linalg.norm(d) + EPS
    return -mass * d / (r**3)

def H_quadratic(x: np.ndarray, c: np.ndarray) -> float:
    # simple progress: higher when closer; gradient is -(x - c)
    return -0.5 * float(np.dot(x - c, x - c))

def gradH_quadratic(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    return -(x - c)

# ---------------
# Simplex ops (sum=1, w>=0)
# ---------------
def project_simplex(w: np.ndarray) -> np.ndarray:
    # project onto { w >= 0, sum w = 1 }
    w = np.asarray(w, dtype=np.float64)
    if w.ndim != 1:
        raise ValueError("w must be 1-D")
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(w)+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    w_proj = np.maximum(w - theta, 0.0)
    return w_proj

def soft_threshold(delta: np.ndarray, lam: float) -> np.ndarray:
    # elementwise soft-threshold
    return np.sign(delta) * np.maximum(np.abs(delta) - lam, 0.0)

# ---------------
# Geodesic pieces
# ---------------
def christoffel_conformal(x: np.ndarray, v: np.ndarray, grad_lnphi: np.ndarray) -> np.ndarray:
    v_dot = float(np.dot(v, grad_lnphi))
    v_norm2 = float(np.dot(v, v))
    return 2.0 * v * v_dot - v_norm2 * grad_lnphi

def rk4_step(x: np.ndarray, v: np.ndarray, dt: float, grad_lnphi_fn) -> Tuple[np.ndarray,np.ndarray]:
    def acc(x_, v_):
        gL = grad_lnphi_fn(x_)
        return -christoffel_conformal(x_, v_, gL) - GAMMA * v_
    k1x = v;              k1v = acc(x, v)
    k2x = v + 0.5*dt*k1v; k2v = acc(x + 0.5*dt*k1x, v + 0.5*dt*k1v)
    k3x = v + 0.5*dt*k2v; k3v = acc(x + 0.5*dt*k2x, v + 0.5*dt*k2v)
    k4x = v + dt*k3v;     k4v = acc(x + dt*k3x, v + dt*k3v)
    x_new = x + (dt/6.0)*(k1x + 2*k2x + 2*k3x + k4x)
    v_new = v + (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
    return x_new, v_new

# ---------------
# Mixture controller integration
# ---------------
def integrate_mixture(x0: np.ndarray, fields: List[Field], steps: int, dt: float,
                      outdir: str, tag: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    K = len(fields)
    # init velocity small toward the weighted center
    centers = np.vstack([f.center for f in fields])
    dists = np.linalg.norm(centers - x0, axis=1) + 1e-6
    w = dists.max() - dists
    if w.sum() == 0:
        w = np.ones(K) / K
    else:
        w = w / w.sum()
    v = 0.10 * ( (centers[w.argmax()] - x0) / (np.linalg.norm(centers[w.argmax()] - x0) + 1e-9) )

    # logs
    W_log = np.zeros((steps, K), dtype=np.float64)
    x = x0.copy()

    # hysteresis state
    dwell = 0
    last_winner = None
    streak = 0

    # helper closures
    def lnphi_and_grad(x_: np.ndarray, w_: np.ndarray) -> Tuple[float, np.ndarray]:
        Vks = np.array([V_center(x_, f.center) for f in fields])
        lnphi = LAMBDA * float(np.dot(w_, Vks))
        # grad ln phi = LAMBDA * sum_k w_k * grad V_k
        g = np.zeros_like(x_)
        for k,f in enumerate(fields):
            g += w_[k] * gradV_center(x_, f.center)
        gL = LAMBDA * g
        return lnphi, gL

    def grad_lnphi_fn(x_: np.ndarray) -> np.ndarray:
        _, gL = lnphi_and_grad(x_, w)
        return gL

    for t in range(steps):
        # --- geometry step ---
        x, v = rk4_step(x, v, dt, grad_lnphi_fn)
    
        # --- weight update ---
        # gradients per concept
        gradsV = np.array([gradV_center(x, f.center) for f in fields])     # (K,d)
        Vks    = np.array([V_center(x, f.center) for f in fields])         # (K,)
        
        # 1) directional descent speed toward each center
        desc = - (gradsV @ v)                  # (K,) larger if moving downhill wrt field k
        desc = np.clip(desc, 0.0, None)        # only reward true descent
        desc = desc - desc.mean()              # remove global bias
        
        # 2) progress gated by alignment to k
        to_k  = np.array([f.center - x for f in fields])                   # (K,d)
        cos   = np.einsum('kd,d->k', to_k / (np.linalg.norm(to_k,axis=1,keepdims=True)+1e-9),
                          v / (np.linalg.norm(v)+1e-9))
        gate  = np.clip(cos - 0.2, 0.0, None)                               # hard gate
        gradH = -to_k                                                        # ∇H_k = -(x - c_k)
        dH    = np.einsum('kd,d->k', gradH, v)                               # directional progress
        prog  = gate * dH
        prog  = prog - prog.mean()                                           # center
        
        # 3) concurrency penalty from field conflicts
        # (penalize co-activation when fields point in similar/opposite directions)
        norms = np.linalg.norm(gradsV, axis=1) + 1e-9
        dirs  = gradsV / norms[:, None]
        rho   = dirs @ dirs.T                                                # (K,K), cosine of field gradients
        conflict = np.sum(w * np.abs(rho), axis=0)                           # (K,)
        
        # combine into gradient for weights
        g_w = -(1.0 * desc + MU_PROGRESS * prog) + 0.5 * conflict
        # (minus because we want larger descent/progress to INCREASE weight)
        
        # hysteresis / dwell as before, but with lower threshold
        winner = int(np.argmax(w))
        if w[winner] > 0.7:  # was 0.90
            streak += 1
        else:
            streak = 0
        if dwell > 0:
            penal = np.ones_like(g_w); penal[winner] = 0.0
            g_w += 5.0 * penal
            dwell -= 1
        elif streak >= HYST_DWELL_WIN:
            dwell = 30  # was 20
            streak = 0
        
        # 4) one step + soft-threshold + projection to sum≤1 simplex (with idle)
        w_prev = w.copy()
        w_tilde = w_prev - ETA_W * g_w
        delta = w_tilde - w_prev
        delta = soft_threshold(delta, LAMBDA_SWITCH * ETA_W)
        w = np.maximum(w_prev + delta, 0.0)
        s = w.sum()
        if s > 1.0:                         # sum≤1 budget
            w = w / s
        # (optional) implicit idle = 1 - w.sum()


        # hysteresis (lock winner)
        winner = int(np.argmax(w))
        if w[winner] > HYST_THRESHOLD:
            streak += 1
        else:
            streak = 0
        if dwell > 0:
            # during dwell, heavily penalize non-winner
            penal = np.ones_like(g_w)
            penal[winner] = 0.0
            g_w += 5.0 * penal
            dwell -= 1
        elif streak >= HYST_DWELL_WIN:
            last_winner = winner
            dwell = HYST_DWELL_STEPS
            streak = 0

        # proximal TV (on delta) + step + simplex projection
        w_prev = w.copy()
        w_tilde = w_prev - ETA_W * g_w
        delta = w_tilde - w_prev
        delta = soft_threshold(delta, LAMBDA_SWITCH * ETA_W)
        w = project_simplex(w_prev + delta)

        W_log[t] = w

    # save plot
    if PLOT:
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            # one chart, default style/colors
            fig = plt.figure()
            for k in range(len(fields)):
                plt.plot(W_log[:, k], label=fields[k].name)
            plt.xlabel("step")
            plt.ylabel("weight")
            plt.title(f"Concept weights over time — {tag}")
            plt.legend()
            ts = time.strftime("%Y%m%d-%H%M%S")
            fname = os.path.join(outdir, f"mixture_weights_{tag}_{ts}.png")
            plt.savefig(fname, bbox_inches="tight")
            plt.close(fig)
            print(f"[Plot] saved: {fname}")
        except Exception as e:
            print("[Plot] skipped:", e)

    return x, v, W_log

# ---------------
# Anchors
# ---------------
def compute_fields(Z: np.ndarray, y: List[str], labels: List[str]) -> List[Field]:
    fields: List[Field] = []
    for c in labels:
        idx = np.array([yy == c for yy in y])
        G = Z[idx]
        mu = G.mean(axis=0)
        spread = float(np.mean(np.linalg.norm(G - mu, axis=1)) + 1e-6)
        fields.append(Field(c, mu, spread))
    return fields

# ---------------
# Pipeline / eval
# ---------------
@dataclass
class ProjectorWrap:
    scaler: StandardScaler
    pca: PCA
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(X))

def fit_projector_wrap(X: np.ndarray) -> ProjectorWrap:
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
    return ProjectorWrap(scaler, pca)

def prepare(device: str) -> Tuple[Extractor, ProjectorWrap, List[Field]]:
    ex = Extractor(device)
    Xtr, ytr = [], []
    for c, prompts in TRAIN.items():
        for s in prompts:
            Xtr.append(ex.get(s))
            ytr.append(c)
    Xtr = np.vstack(Xtr)
    proj = fit_projector_wrap(Xtr)
    Ztr = proj.transform(Xtr)
    fields = compute_fields(Ztr, ytr, LABELS)
    print("[Anchors] order:", [f.name for f in fields])
    return ex, proj, fields

def run(device: str = DEVICE_DEFAULT, outdir: str = "."):
    os.makedirs(outdir, exist_ok=True)
    ex, proj, fields = prepare(device)

    # sanity
    demo = "Input grid [[1,2],[3,4]]. Produce the matching output grid."
    z_demo = proj.transform(np.vstack([ex.get(demo)]))[0]
    dists = [np.linalg.norm(z_demo - f.center) for f in fields]
    print("[Sanity] distances:", np.array(dists).round(3))
    print("[Sanity] chosen:", fields[int(np.argmin(dists))].name)

    # test set
    Xte, yte = [], []
    for grid, label in TEST:
        Xte.append(ex.get(f"Input grid {grid}. Produce the matching output grid."))
        yte.append(label)
    Zte = proj.transform(np.vstack(Xte))

    preds = []
    logs = []
    for i, (z0, true) in enumerate(zip(Zte, yte), 1):
        tag = f"sample{i:02d}"
        xT, vT, W_log = integrate_mixture(z0, fields, STEPS, DT, outdir, tag)
        w_tail = W_log[-FINAL_AVG_TAIL:].mean(axis=0)
        j_hat = int(np.argmax(w_tail))
        pred = fields[j_hat].name
        preds.append(pred)

        # probs via softmax over average negative distances to centers (rough proxy)
        eu = np.array([np.linalg.norm(z0 - f.center) for f in fields], dtype=np.float64)
        eu = eu - eu.min()
        q = np.exp(-(eu / (eu.std() + 1e-9)))
        p = q / (q.sum() + 1e-9)
        conf = float(p[j_hat])
        dist_star = float(np.dot(p, eu))

        logs.append((i, true, pred, pred==true, dist_star, conf, p))

    for i, true, pred, ok, dist_star, conf, p in logs:
        print(f"[{i:02d}] true={true:<7} pred={pred:<7} ok={str(ok):<5} dist*={dist_star:.3f} conf={conf:.3f} probs=[{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]")

    acc = sum(int(a==b) for a,b in zip(preds,yte)) / len(yte)
    mean_conf = float(np.mean([c for *_, c, _ in logs]))
    print(f"\n[ARC-12] Accuracy: {acc*100:.1f}% | Mean confidence: {mean_conf:.3f} | Mode=mixture-controller")

    cm = confusion_matrix(yte, preds, labels=LABELS)
    print("[Confusion]\n          pred→   flip_h  flip_v  rotate")
    for i, row in enumerate(cm):
        print(f"true={LABELS[i]:<8}      {row[0]:>3}     {row[1]:>3}     {row[2]:>3}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=DEVICE_DEFAULT)
    ap.add_argument("--outdir", type=str, default=".")
    args = ap.parse_args()
    run(device=args.device, outdir=args.outdir)

if __name__ == "__main__":
    main()
# """)

# path = "/mnt/data/latest-arc-benchmark_mixture.py"
# with open(path, "w") as f:
#     f.write(code)

# print("Wrote", path)
