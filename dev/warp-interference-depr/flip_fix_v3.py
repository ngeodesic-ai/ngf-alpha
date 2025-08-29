# flip_fix_v3.py — Apache-2.0 (ngeodesic.ai)
# Fixes flip_h vs flip_v collapse by rotate-null projection + flip-only LDA.

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict

_EPS = 1e-9

def _softmax(x, tau=1.0, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x / max(tau, 1e-6))
    return ex / np.clip(ex.sum(axis=axis, keepdims=True), _EPS, None)

def _mahalanobis(x, mu, Si_inv):
    d = x - mu
    return np.einsum('...i,ij,...j->...', d, Si_inv, d)

def _safe_inv(mat, ridge=1e-4):
    m = mat.copy()
    k = m.shape[0]
    m.flat[::k+1] += ridge
    try:
        return np.linalg.inv(m)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(m)

@dataclass
class FitState:
    label_names: List[str]
    mu: Dict[str, np.ndarray]           # class means (1,d)
    Si: Dict[str, np.ndarray]           # class covs (d,d)
    Si_inv: Dict[str, np.ndarray]       # inverses (d,d)
    pooled_cov: np.ndarray
    pooled_inv: np.ndarray
    rotate_axis: np.ndarray             # unit vector of (rotate - mean flips)
    W_flip: np.ndarray                  # (d,1) LDA direction for flips
    proj_null_rotate: np.ndarray        # projector: I - a a^T
    temp_main: float
    temp_flip: float

class FlipAwareClassifier:
    """
    Minimal external API:
        clf = FlipAwareClassifier()
        clf.fit(train_Z, train_y, label_names=['flip_h','flip_v','rotate'])
        probs = clf.predict_proba(Z_test)
        pred  = clf.predict(Z_test)
    Optionally merge a trajectory vote: clf.predict_proba(Z, vote=vote_idx)
    """
    def __init__(self, temp_main: float = 1.0, temp_flip: float = 0.8, ridge: float = 1e-3):
        self.temp_main = float(temp_main)
        self.temp_flip = float(temp_flip)
        self.ridge = float(ridge)
        self.state: Optional[FitState] = None

    def fit(self, Z: np.ndarray, y: np.ndarray, label_names: List[str]):
        labs = list(label_names)
        req = {'flip_h','flip_v','rotate'}
        if set(labs) != req:
            raise ValueError(f"label_names must be a permutation of {req}, got {labs}")
        if Z.ndim != 2: raise ValueError("Z must be (n,d)")
        n, d = Z.shape

        y = np.asarray(y)
        if np.issubdtype(y.dtype, np.integer):
            y_names = np.array([labs[i] for i in y])
        else:
            y_names = y.astype(object)

        # class stats (means, covs, inverses)
        mu, Si, Si_inv = {}, {}, {}
        for name in labs:
            Zi = Z[y_names == name]
            if Zi.size == 0: raise ValueError(f"No samples for class {name}")
            mu[name] = Zi.mean(axis=0, keepdims=True)
            C = np.cov(Zi.T) if Zi.shape[0] > 1 else np.eye(d)
            C = C + self.ridge * np.eye(d)
            Si[name] = C
            Si_inv[name] = _safe_inv(C, ridge=self.ridge)

        # pooled cov (for main head)
        pooled_cov = np.zeros((d,d)); dof = 0
        for name in labs:
            Zi = Z[y_names == name]
            ni = max(Zi.shape[0]-1, 1)
            pooled_cov += ni * Si[name]
            dof += ni
        pooled_cov /= max(dof, 1)
        pooled_inv = _safe_inv(pooled_cov, ridge=self.ridge)

        # rotate axis and projector to null it when deciding flips
        flips_mean = 0.5 * (mu['flip_h'] + mu['flip_v'])
        a = (mu['rotate'] - flips_mean).reshape(-1)
        a_unit = a / (np.linalg.norm(a) + _EPS)
        P_null = np.eye(d) - np.outer(a_unit, a_unit)

        # Fisher direction for flips only in rotate-null subspace
        m_h = (mu['flip_h'] @ P_null.T).reshape(-1,1)
        m_v = (mu['flip_v'] @ P_null.T).reshape(-1,1)
        Cf = 0.5 * (Si['flip_h'] + Si['flip_v'])
        Cf_inv = _safe_inv(Cf, ridge=self.ridge)
        w = Cf_inv @ (m_h - m_v)
        w = w / (np.linalg.norm(w) + _EPS)

        self.state = FitState(
            label_names=labs, mu=mu, Si=Si, Si_inv=Si_inv,
            pooled_cov=pooled_cov, pooled_inv=pooled_inv,
            rotate_axis=a_unit, W_flip=w, proj_null_rotate=P_null,
            temp_main=float(self.temp_main), temp_flip=float(self.temp_flip)
        )
        return self

    def _score_main(self, Z: np.ndarray) -> np.ndarray:
        st = self.state
        if st is None: raise RuntimeError("Call fit first")
        logits = []
        for name in st.label_names:
            d2 = _mahalanobis(Z, st.mu[name], st.pooled_inv)
            logits.append((-d2).reshape(-1,1))
        return np.concatenate(logits, axis=1)  # (n,3)

    def _score_flips(self, Z: np.ndarray) -> np.ndarray:
        st = self.state
        if st is None: raise RuntimeError("Call fit first")
        Zp = Z @ st.proj_null_rotate.T
        t  = (Zp @ st.W_flip).reshape(-1)
        m_h = float((st.mu['flip_h'] @ st.proj_null_rotate.T @ st.W_flip).reshape(()))
        m_v = float((st.mu['flip_v'] @ st.proj_null_rotate.T @ st.W_flip).reshape(()))
        s_h = -(t - m_h)**2
        s_v = -(t - m_v)**2
        return np.stack([s_h, s_v], axis=1)  # (n,2)

    def predict_proba(self, Z: np.ndarray, vote: Optional[np.ndarray] = None, vote_gate: float = 0.65) -> np.ndarray:
        st = self.state
        if st is None: raise RuntimeError("Call fit first")
        # head 1: pooled-cov distances over all 3
        main_logits = self._score_main(Z)
        main_probs  = _softmax(main_logits, tau=st.temp_main, axis=1)
        # head 2: flip-only scores in rotate-null space
        flips_logits = self._score_flips(Z)
        flips_probs  = _softmax(flips_logits, tau=st.temp_flip, axis=1)
        # blend flips multiplicatively, renormalize
        idx_h = st.label_names.index('flip_h')
        idx_v = st.label_names.index('flip_v')
        blended = main_probs.copy()
        flip_mass = blended[:, [idx_h, idx_v]]
        blended[:, [idx_h, idx_v]] = np.sqrt(np.clip(flip_mass, _EPS, None) * flips_probs)
        blended /= np.clip(blended.sum(1, keepdims=True), _EPS, None)
        # optional vote override under low confidence
        if vote is not None:
            vote = np.asarray(vote)
            vote_idx = np.array([st.label_names.index(v) if isinstance(v, str) else int(v) for v in vote])
            top = blended.max(1)
            override = top < float(vote_gate)
            for i in np.where(override)[0]:
                b = np.zeros(3, dtype=float); b[vote_idx[i]] = 1.0
                blended[i] = b
        return blended

    def predict(self, Z: np.ndarray, vote: Optional[np.ndarray] = None, vote_gate: float = 0.65) -> np.ndarray:
        return self.predict_proba(Z, vote=vote, vote_gate=vote_gate).argmax(1)

    def merge_with_vote(self, probs: np.ndarray, vote: np.ndarray, vote_gate: float = 0.65) -> np.ndarray:
        st = self.state
        if st is None: raise RuntimeError("Call fit first")
        vote = np.asarray(vote)
        vote_idx = np.array([st.label_names.index(v) if isinstance(v, str) else int(v) for v in vote])
        top = probs.max(1)
        override = top < float(vote_gate)
        out = probs.copy()
        for i in np.where(override)[0]:
            b = np.zeros(3, dtype=float); b[vote_idx[i]] = 1.0
            out[i] = b
        return out
