#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 10 — ARC-12 patched benchmark (flip_h vs flip_v fix, reduced bias)

Key changes
- Latents = mean of last 4 GPT-2 hidden layers (less lexical drift)
- Training prompts use I/O examples WITHOUT the words "rotate"/"flip" to reduce label-bias
- PCA with standardization + whitening; auto n-components to hit target variance
- FlipAwareClassifier: pooled-cov head + rotate-null flip-only Fisher head, blended
- Softer temperature (main=2.4, flip=1.6) to avoid p≈{0,1} overconfidence
- Reports per-sample logs and confusion matrix

Run:
  python3 latest-arc-benchmark_patched.py --device cpu
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from numpy.linalg import inv
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from transformers import GPT2Tokenizer, GPT2LMHeadModel

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE_DEFAULT = "cpu"

PCA_TARGET_VAR = 0.996
PCA_MIN, PCA_MAX = 8, 24
PCA_WHITEN = True

# Flip classifier temperatures (softer than before)
TEMP_MAIN = 3.0
TEMP_FLIP = 2.4
VOTE_GATE = 0.70  # if you later add a vote, override when top prob < gate

# ---------------------------
# Training/test data
# ---------------------------
# Training prompts avoid label words; show example input->output pairs only.
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

# Test uses the classic 12 cases (grids embedded in the prompt, no label words)
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

# ---------------------------
# Embedding (last 4 layers mean)
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
        # last 4 layer mean, then token mean
        stack = torch.stack(hs[-4:], dim=0)  # (4, batch=1, seq, hid)
        mean_layers = stack.mean(dim=0)[0]   # (seq, hid)
        vec = mean_layers.mean(dim=0).cpu().numpy().astype(np.float64)
        return vec

# ---------------------------
# PCA + scaling
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
# Flip-aware classifier (rotate-null + flip-only Fisher)
# ---------------------------
@dataclass
class FitState:
    label_names: List[str]
    mu: Dict[str, np.ndarray]
    pooled_inv: np.ndarray
    proj_null_rotate: np.ndarray
    W_flip: np.ndarray  # (d,1)

class FlipAwareClassifier:
    def __init__(self, temp_main: float = TEMP_MAIN, temp_flip: float = TEMP_FLIP, ridge: float = 1e-3):
        self.temp_main = temp_main
        self.temp_flip = temp_flip
        self.ridge = ridge
        self.state: Optional[FitState] = None

    def _safe_inv(self, M: np.ndarray) -> np.ndarray:
        M = M.copy(); k = M.shape[0]; M.flat[::k+1] += self.ridge
        try:
            return inv(M)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(M)

    def fit(self, Z: np.ndarray, y: List[str], label_order: List[str]):
        labs = list(label_order)
        mu = {c: Z[np.array([yy==c for yy in y])].mean(axis=0, keepdims=True) for c in labs}
        # pooled cov LDA
        resids = []
        for c in labs:
            Zi = Z[np.array([yy==c for yy in y])]
            resids.append(Zi - mu[c])
        R = np.vstack(resids)
        cov = (R.T @ R) / max(1, R.shape[0]-1)
        pooled_inv = self._safe_inv(cov)
        # rotate-null projector for flip head
        flips_mean = 0.5*(mu['flip_h'] + mu['flip_v'])
        a = (mu['rotate'] - flips_mean).reshape(-1)
        a = a / (np.linalg.norm(a) + 1e-9)
        P = np.eye(Z.shape[1]) - np.outer(a, a)
        # 1D Fisher between flips (in original space, projected at score time)
        Cf = 0.5*( (R[np.array([yy in ('flip_h','flip_v') for yy in y])].T @ R[np.array([yy in ('flip_h','flip_v') for yy in y])]) / max(1, sum(yy in ('flip_h','flip_v') for yy in y)-1) )
        Cf_inv = self._safe_inv(Cf)
        w = Cf_inv @ ( (mu['flip_h'] - mu['flip_v']).reshape(-1,1) )
        w = w / (np.linalg.norm(w) + 1e-9)
        self.state = FitState(lab
el_names=labs, mu=mu, pooled_inv=pooled_inv, proj_null_rotate=P, W_flip=w)
        return self

    def _softmax(self, x: np.ndarray, tau: float) -> np.ndarray:
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x / max(tau, 1e-6))
        return e / np.clip(e.sum(axis=1, keepdims=True), 1e-9, None)

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        st = self.state; assert st is not None
        # main head: pooled-cov Mahalanobis to each mean
        logits = []
        for c in st.label_names:
            d = Z - st.mu[c]
            d2 = np.einsum('nd,dk,nk->n', d, st.pooled_inv, d)
            logits.append((-d2)[:,None])
        main_logits = np.concatenate(logits, axis=1)
        main_probs  = self._softmax(main_logits, tau=self.temp_main)
        # flip head: rotate-null + 1D Fisher
        Zp = Z @ st.proj_null_rotate.T
        t  = (Zp @ st.W_flip).reshape(-1)
        m_h = float((st.mu['flip_h'] @ st.proj_null_rotate.T @ st.W_flip).reshape(()))
        m_v = float((st.mu['flip_v'] @ st.proj_null_rotate.T @ st.W_flip).reshape(()))
        s_h = -(t - m_h)**2; s_v = -(t - m_v)**2
        flips = np.stack([s_h, s_v], axis=1)
        flips = self._softmax(flips, tau=self.temp_flip)
        # blend flips multiplicatively
        idx_h = st.label_names.index('flip_h')
        idx_v = st.label_names.index('flip_v')
        out = main_probs.copy()
        out[:, [idx_h, idx_v]] = np.sqrt(np.clip(out[:, [idx_h, idx_v]], 1e-9, None) * flips)
        out /= np.clip(out.sum(axis=1, keepdims=True), 1e-9, None)
        return out

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return self.predict_proba(Z).argmax(axis=1)

# ---------------------------
# Helpers
# ---------------------------
TRANSFORM = {
    'rotate': lambda g: np.rot90(np.array(g), k=3).tolist(),
    'flip_h': lambda g: np.flip(np.array(g), axis=1).tolist(),
    'flip_v': lambda g: np.flip(np.array(g), axis=0).tolist(),
}

LABELS = ['flip_h','flip_v','rotate']

def grid_to_prompt(grid: List[List[int]]) -> str:
    return f"Input grid {grid}. Produce the matching output grid."

# ---------------------------
# Main flow
# ---------------------------
@dataclass
class Context:
    proj: Projector
    clf: FlipAwareClassifier
    extractor: Extractor


def prepare(device: str) -> Context:
    ex = Extractor(device)
    # embed training
    X, y = [], []
    for c, arr in TRAIN.items():
        for s in arr:
            X.append(ex.get(s))
            y.append(c)
    X = np.vstack(X)
    proj = fit_projector(X)
    Z = proj.transform(X)
    clf = FlipAwareClassifier(temp_main=TEMP_MAIN, temp_flip=TEMP_FLIP).fit(Z, y, LABELS)
    return Context(proj=proj, clf=clf, extractor=ex)


def run(device: str = DEVICE_DEFAULT):
    ctx = prepare(device)
    print(f"[Anchors] order: {LABELS}")

    # sanity distances (euclidean to means in reduced space)
    means = {c: ctx.clf.state.mu[c] for c in LABELS}  # type: ignore
    d_means = []
    for c in LABELS:
        d = np.linalg.norm(ctx.proj.transform(np.vstack([ctx.extractor.get(p) for p in TRAIN[c]])) - means[c], axis=1).mean()
        d_means.append(d)
    print("[Sanity] distances:", np.array(d_means).round(3))
    print("[Sanity] chosen:", LABELS[int(np.argmin(d_means))])

    # build test set latents
    Xte, yte = [], []
    for grid, label in TEST:
        Xte.append(ctx.extractor.get(grid_to_prompt(grid)))
        yte.append(label)
    Zte = ctx.proj.transform(np.vstack(Xte))

    probs = ctx.clf.predict_proba(Zte)
    preds_idx = probs.argmax(axis=1)
    preds = [LABELS[i] for i in preds_idx]

    # print per-sample
    correct = 0; rows = []
    for i, (grid, true) in enumerate(TEST, 1):
        p = probs[i-1]
        pred = preds[i-1]
        ok = pred == true; correct += int(ok)
        conf = float(p.max())
        # dist*: expected euclidean distance to class means under p
        means_stack = np.vstack([means[c] for c in LABELS])  # (3,d)
        z = Zte[i-1]
        eu = np.linalg.norm(means_stack - z, axis=1)
        dist_star = float(np.dot(p, eu))
        print(f"[{i:02d}] true={true:7s}  pred={pred:7s}  ok={ok}  dist*={dist_star:.3f}  conf={conf:.3f}  probs=[{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]")

    acc = correct / len(TEST)
    print(f"
[ARC-12] Accuracy: {acc*100:.1f}% | Mean confidence: {float(probs.max(axis=1).mean()):.3f} | Mode=flip-aware")

    # confusion matrix
    cm = confusion_matrix(yte, preds, labels=LABELS)
    print("[Confusion]
          pred→   flip_h  flip_v  rotate")
    for i, row in enumerate(cm):
        print(f"true={LABELS[i]:<8}      {row[0]:>3}     {row[1]:>3}     {row[2]:>3}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=DEVICE_DEFAULT)
    args = ap.parse_args()
    run(device=args.device)
