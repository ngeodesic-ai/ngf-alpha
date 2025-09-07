#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NGF Reno v2 Hook — Full-Dimensional Warp (with optional PCA telemetry)


export NGF_RENO_CFG="tap=-9 \
alpha0=0.10 alpha_min=0.012 trend_tau=0.50 k_tr=12 \
use_detect=1 detect_width=32 detect_sigma=5 null_K=64 null_q=0.92 k_det=6 \
s_latch=0.40 linger=3 ema_center_beta=0.05 eps=0.25 \
use_denoise=1 denoise_beta=0.6 denoise_window=3 denoise_k=8.0 denoise_tau=0.35 \
phantom_tr_tau=0.60 phantom_guard_gamma=0.35 jitter_eps=0.03 \
center_mode=full pca_telemetry=1"

Usage (example):
  python3 ngf_benchmark.py \
    --mode ngf --ngf_import ngf_hooks_v3:attach_ngf_hooks \
    --model gpt2 --split validation --n 1000 --max_length 768 --device auto \
    --tap -9 \
    --alpha0 0.10 --alpha_min 0.012 --trend_tau 0.50 --k_tr 12 \
    --use_detect 1 --detect_width 32 --detect_sigma 5 --k_det 6 \
    --null_K 64 --null_q 0.92 \
    --s_latch 0.40 --linger 3 --ema_center_beta 0.05 \
    --center_mode full --pca_telemetry 1 \
    --gen_mode geo --max_new_tokens 96 \
    --out_json results/ngf_fullR768_tap9.json
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
from torch import nn

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None


# ------------------------------ Small utils ------------------------------

def moving_average(x: np.ndarray, k: int = 9) -> np.ndarray:
    if k <= 1:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(xp, np.ones(k)/k, mode="valid")


def half_sine_proto(width: int) -> np.ndarray:
    P = np.sin(np.linspace(0, np.pi, int(max(2, width))))
    P = P / (np.linalg.norm(P) + 1e-8)
    return P


def xcorr_same(sig: np.ndarray, proto: np.ndarray) -> np.ndarray:
    T = len(sig)
    L = min(len(proto), T)
    pr = proto[:L] - np.mean(proto[:L])
    prn = pr / (np.linalg.norm(pr) + 1e-8)
    out = np.zeros(T, dtype=float)
    for i in range(T):
        a = max(0, i - L//2); b = min(T, a + L)
        a = max(0, b - L)
        w = sig[a:b]
        wn = w - np.mean(w)
        denom = (np.linalg.norm(wn) * np.linalg.norm(prn) + 1e-8)
        out[i] = float(np.dot(wn, prn[:len(wn)]) / denom)
    return out


def null_threshold(sig: np.ndarray, proto: np.ndarray, K: int = 24, q: float = 0.90, rng=None) -> float:
    rng = rng or np.random.default_rng(20259)
    T = len(sig); L = min(len(proto), T)
    vals = []
    for _ in range(int(K)):
        s = int(rng.integers(1, max(2, T-1)))
        xs = np.roll(sig, s)
        vals.append(float(np.max(xcorr_same(xs[-L:], proto[:L]))))
    return float(np.quantile(vals, q))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


# ------------------------------ Telemetry-only PCA-2 ------------------------------

@dataclass
class PCA2Projector:
    pca: Optional[PCA]
    center: Optional[np.ndarray]
    warm: List[np.ndarray]
    max_warm: int
    ema_center_beta: float
    ema_c: Optional[np.ndarray]

    @classmethod
    def make(cls, max_warm: int = 256, center_xy: Optional[Tuple[float, float]] = None, ema_center_beta: float = 0.0):
        c = np.array(center_xy, dtype=float) if center_xy is not None else None
        return cls(pca=None, center=c, warm=[], max_warm=max_warm, ema_center_beta=float(ema_center_beta), ema_c=None)

    def update_fit(self, vec: np.ndarray):
        if len(self.warm) < self.max_warm:
            self.warm.append(vec.astype(np.float32))
            if PCA is not None and self.pca is None and len(self.warm) >= max(32, self.max_warm//4):
                X = np.stack(self.warm, 0)
                self.pca = PCA(n_components=2, whiten=True, random_state=0).fit(X)

    def project(self, vec: np.ndarray) -> Tuple[np.ndarray, float]:
        x = vec.astype(np.float32)
        if self.pca is None and PCA is not None and len(self.warm) >= 8:
            X = np.stack(self.warm, 0)
            self.pca = PCA(n_components=2, whiten=True, random_state=0).fit(X)
        if self.pca is not None:
            y2 = self.pca.transform(x[None, :])[0]
        else:
            y2 = x[:2].copy()
        if self.ema_center_beta > 0.0:
            if self.ema_c is None:
                self.ema_c = y2.copy()
            else:
                self.ema_c = (1.0 - self.ema_center_beta) * self.ema_c + self.ema_center_beta * y2
            c = self.ema_c
        else:
            if self.center is not None:
                c = self.center
            elif self.warm:
                c = np.mean(np.stack(self.warm, 0)[:, :2], axis=0)
            else:
                c = np.zeros(2, dtype=float)
        r = float(np.linalg.norm(y2 - c) + 1e-9)
        return y2, r


# ------------------------------ Full-D EMA center (control path) ------------------------------

@dataclass
class FullCenter:
    """Full-dimensional EMA center & radius in R^D (no PCA)."""
    ema_center_beta: float = 0.05
    max_warm: int = 256
    _ema: Optional[np.ndarray] = None
    _warm: Optional[List[np.ndarray]] = None

    def __post_init__(self):
        if self._warm is None:
            self._warm = []

    def update_fit(self, vec: np.ndarray):
        if len(self._warm) < self.max_warm:
            self._warm.append(vec.astype(np.float32))

    def radius(self, vec: np.ndarray) -> float:
        x = vec.astype(np.float32)
        if self.ema_center_beta > 0.0:
            if self._ema is None:
                self._ema = x.copy()
            else:
                b = float(self.ema_center_beta)
                self._ema = (1.0 - b) * self._ema + b * x
            c = self._ema
        else:
            if self._warm:
                c = np.mean(np.stack(self._warm, 0), axis=0)
            else:
                c = np.zeros_like(x)
        return float(np.linalg.norm(x - c) + 1e-9)


# ------------------------------ Soft Denoiser ------------------------------

class SoftDenoiser:
    """
    Smooth & scale the warp residual (never flip direction).
    - EMA smoothing on residual vector
    - Soft confidence gate (sigmoid of evidence)
    - Phantom guard: attenuate isolated spikes with low evidence
    - Micro-jitter averaging (stabilizes brittle spikes)
    """
    def __init__(self, beta=0.6, window=3, k=8.0, tau=0.35,
                 phantom_tr_tau=0.60, phantom_guard_gamma=0.35,
                 jitter_eps=0.03):
        from collections import deque as _dq
        self.beta=float(beta); self.window=int(window)
        self.k=float(k); self.tau=float(tau)
        self.phantom_tr_tau=float(phantom_tr_tau)
        self.phantom_guard_gamma=float(phantom_guard_gamma)
        self.jitter_eps=float(jitter_eps)
        self._ema=None
        self._buf=_dq(maxlen=self.window)

    @staticmethod
    def _sigmoid(x): return 1.0/(1.0+np.exp(-x))
    def reset(self):
        self._ema=None; self._buf.clear()

    def step(self, resid_vec: np.ndarray, tr: float, g_det: float, s: float, prev_s: float):
        if resid_vec is None:
            return None, dict(dn_gain=0.0, dn_guard=0, dn_ema_norm=0.0, dn_med_norm=0.0)
        r=resid_vec
        rn=float(np.linalg.norm(r)+1e-12)
        self._buf.append(rn)
        med=float(np.median(self._buf))
        # EMA smoothing
        if self._ema is None: self._ema=r.copy()
        else: self._ema=self.beta*self._ema+(1.0-self.beta)*r
        ema=float(np.linalg.norm(self._ema)+1e-12)
        # Phantom guard: low evidence + isolated → attenuate
        guard=0
        if (g_det<0.25) and (s<0.15) and (abs(tr)>self.phantom_tr_tau) and (prev_s<0.15):
            self._ema*=self.phantom_guard_gamma; guard=1
        # Soft confidence gate (scale only)
        score=0.6*float(g_det)+0.4*float(s)
        gain=float(self._sigmoid(self.k*(score-self.tau)))
        out=self._ema*gain
        # Micro-jitter averaging
        if self.jitter_eps>0:
            j=self.jitter_eps
            out=0.5*(out*(1.0+j)+out*(1.0-j))
        return out, dict(dn_gain=gain, dn_guard=int(guard), dn_ema_norm=ema, dn_med_norm=med)


# ------------------------------ Terraform Hook (Reno v2) ------------------------------

class TerraformHook:
    def __init__(self,
                 layer_module: nn.Module,
                 center_ctl: Optional[FullCenter],    # full-D center/radius for control (None → legacy PCA control)
                 proj_tele: Optional[PCA2Projector],  # PCA-2 telemetry only
                 alpha0: float = 0.06, alpha_min: float = 0.01,
                 trend_tau: float = 0.32, k_tr: float = 8.0,
                 use_detect: int = 0, detect_width: int = 40, detect_sigma: int = 7,
                 null_K: int = 24, null_q: float = 0.90, k_det: float = 8.0,
                 linger: int = 2, s_latch: float = 0.6,
                 eps: float = 0.0,             # relative step clip; 0 disables
                 print_every: int = 32,
                 log_prefix: str = "[HOOK]",
                 denoiser: Optional[SoftDenoiser]=None):
        self.layer_module = layer_module
        self.center_ctl = center_ctl
        self.proj_tele = proj_tele
        self.denoiser = denoiser
        self.alpha0 = float(alpha0)
        self.alpha_min = float(alpha_min)
        self.trend_tau = float(trend_tau)
        self.k_tr = float(k_tr)
        self.use_detect = int(use_detect)
        self.detect_sigma = int(detect_sigma)
        self.proto = half_sine_proto(int(detect_width)) if use_detect else None
        self.null_K = int(null_K)
        self.null_q = float(null_q)
        self.k_det = float(k_det)
        from collections import deque
        self.trend_hist = deque(maxlen=max(192, int(detect_width))) if use_detect else None
        self.rng = np.random.default_rng(20259)
        self.prev_r = None
        self.linger = int(linger)
        self.s_latch = float(s_latch)
        self.linger_left = 0
        self.eps = float(eps)
        self.print_every = int(print_every)
        self.log_prefix = log_prefix
        # stats
        self.steps_seen = 0
        self.steps_applied = 0
        self.alpha_last = 0.0
        self.trend_last = 0.0
        self.radius_last = 0.0
        self.step_norm_last = 0.0
        self.g_tr_last = 0.0
        self.g_det_last = 1.0 if not use_detect else 0.0
        # sequences (per prompt)
        self.last_print_T = -1
        self.alpha_seq: List[float] = []
        self.s_seq: List[float] = []
        self.trend_seq: List[float] = []
        self.g_tr_seq: List[float] = []
        self.g_det_seq: List[Optional[float]] = []
        self.detect_score_seq: List[Optional[float]] = []
        self.tau_abs_seq: List[Optional[float]] = []
        self.radius_seq: List[float] = []
        # denoiser telemetry
        self.dn_gain_seq: List[float] = []
        self.dn_guard_seq: List[int] = []
        self.dn_ema_norm_seq: List[float] = []
        self.dn_med_norm_seq: List[float] = []
        self._prev_s = 0.0
        self.enabled = True
        self._hook_handle = None

    def reset_for_prompt(self):
        self.prev_r = None
        self.linger_left = 0
        self.steps_seen = 0
        self.steps_applied = 0
        self.alpha_last = 0.0
        self.trend_last = 0.0
        self.radius_last = 0.0
        self.step_norm_last = 0.0
        self.g_tr_last = 0.0
        self.g_det_last = 1.0 if not self.use_detect else 0.0
        self.last_print_T = -1
        # seqs
        self.alpha_seq = []; self.s_seq = []; self.trend_seq = []
        self.g_tr_seq = []; self.g_det_seq = []
        self.detect_score_seq = []; self.tau_abs_seq = []
        self.radius_seq = []
        self.dn_gain_seq = []; self.dn_guard_seq = []
        self.dn_ema_norm_seq = []; self.dn_med_norm_seq = []
        if self.trend_hist is not None: self.trend_hist.clear()
        if self.denoiser is not None: self.denoiser.reset()

    def _detect_soft(self) -> Tuple[float, Optional[float], Optional[float]]:
        if not self.use_detect:
            return 1.0, None, None
        if self.trend_hist is None or len(self.trend_hist) < 8:
            return 0.0, None, None
        sig = np.asarray(self.trend_hist, dtype=float)
        S = moving_average(sig, k=self.detect_sigma)
        L = min(len(self.proto), len(S))
        proto = self.proto[:L]
        seg = S[-L:]
        corr = xcorr_same(seg, proto)
        score = float(np.max(corr))
        tau_abs = null_threshold(seg, proto, K=self.null_K, q=self.null_q, rng=self.rng)
        g_det = float(_sigmoid(self.k_det * (score - tau_abs)))
        return g_det, score, tau_abs

    def attach(self):
        if self._hook_handle is not None:
            return

        def _unwrap(out):
            if isinstance(out, (tuple, list)): return out[0]
            return out
        def _rewrap(new_hs, out_orig):
            if isinstance(out_orig, tuple): return (new_hs,) + tuple(out_orig[1:])
            if isinstance(out_orig, list):  return [new_hs] + list(out_orig[1:])
            return new_hs

        def _forward_hook(module, inputs, output):
            if not self.enabled:
                return output
            hs = _unwrap(output)
            if not torch.is_tensor(hs):
                return output
            self.steps_seen += 1
            with torch.no_grad():
                h_last = hs[:, -1, :]                        # (B,D)
                h_np = h_last[0].detach().cpu().float().numpy()

                # ---- CONTROL: full-D radius / trend (or legacy PCA control) ----
                if self.center_ctl is not None:
                    self.center_ctl.update_fit(h_np)
                    r = self.center_ctl.radius(h_np)        # full-R^D control
                else:
                    # legacy: use PCA-2 for control if provided
                    if self.proj_tele is not None:
                        self.proj_tele.update_fit(h_np)
                        _, r = self.proj_tele.project(h_np)
                    else:
                        r = float(np.linalg.norm(h_np))

                if self.prev_r is None:
                    trend = 0.0
                else:
                    trend = max(0.0, float((self.prev_r - r) / max(self.prev_r, 1e-6)))
                self.prev_r = r
                self.trend_last = float(trend)
                self.radius_last = float(r)
                self.trend_seq.append(self.trend_last)
                self.radius_seq.append(self.radius_last)
                if self.trend_hist is not None:
                    self.trend_hist.append(self.trend_last)

                # Telemetry-only PCA projection (never used for control)
                if self.proj_tele is not None:
                    try:
                        self.proj_tele.update_fit(h_np)
                        _y2, _ = self.proj_tele.project(h_np)
                    except Exception:
                        pass

                # ---- Gates & latch ----
                g_det, score, tau_abs = self._detect_soft()
                g_tr = float(_sigmoid(self.k_tr * (self.trend_last - self.trend_tau)))
                if (g_tr * g_det) >= 0.5:
                    self.linger_left = max(self.linger_left, self.linger)
                g_latch = self.s_latch if self.linger_left > 0 else 0.0
                if self.linger_left > 0:
                    self.linger_left -= 1
                s_pre = g_tr * g_det
                s = max(s_pre, g_latch)

                # ---- Warp step (full-D) ----
                alpha_t = float(self.alpha_min + (self.alpha0 - self.alpha_min) * s)
                dx = -alpha_t * h_last
                # relative step clip if requested
                if self.eps > 0.0:
                    # Per-sample relative step control (B,1) tensors
                    hnorm = torch.norm(h_last, dim=-1, keepdim=True) + 1e-9  # (B,1)
                    dnorm = torch.norm(dx,     dim=-1, keepdim=True)         # (B,1)
                    ratio = (dnorm / hnorm).clamp_min(1e-9)                  # (B,1)
                    scale = torch.clamp(self.eps / ratio, max=1.0)           # (B,1)
                    dx = dx * scale                                          # broadcast
                    # Represent alpha_t as the mean relative step across batch
                    alpha_t = float(ratio.mean().item())

                # ---- Denoiser (vector-level smoothing & scaling) ----
                dn_gain=0.0; dn_guard=0; dn_ema=0.0; dn_med=0.0
                if self.denoiser is not None:
                    dx_np = dx[0].detach().cpu().float().numpy()
                    r_dn, meta = self.denoiser.step(
                        dx_np,
                        self.trend_last,
                        g_det if self.use_detect else 1.0,
                        s, getattr(self, "_prev_s", 0.0)
                    )
                    if r_dn is not None:
                        r_t = torch.from_numpy(r_dn).to(hs.device).view_as(dx[0])
                        dx = dx.clone(); dx[0] = r_t
                    dn_gain=float(meta["dn_gain"]); dn_guard=int(meta["dn_guard"])
                    dn_ema=float(meta["dn_ema_norm"]); dn_med=float(meta["dn_med_norm"])
                    self._prev_s = float(s)

                hs_new = hs.clone()
                hs_new[:, -1, :] = hs_new[:, -1, :] + dx

                # telemetry
                self.step_norm_last = float(torch.norm(dx).item())
                self.alpha_last = alpha_t
                self.steps_applied += int(alpha_t > 0.0)
                self.alpha_seq.append(self.alpha_last)
                self.s_seq.append(float(s))
                self.g_tr_seq.append(float(g_tr))
                self.g_det_seq.append(float(g_det) if self.use_detect else None)
                self.detect_score_seq.append(float(score) if score is not None else None)
                self.tau_abs_seq.append(float(tau_abs) if tau_abs is not None else None)
                self.dn_gain_seq.append(dn_gain)
                self.dn_guard_seq.append(dn_guard)
                self.dn_ema_norm_seq.append(dn_ema)
                self.dn_med_norm_seq.append(dn_med)

                T = hs.shape[1]
                if self.print_every > 0 and (self.steps_seen % self.print_every) == 0 and (self.last_print_T != T):
                    self.last_print_T = T
                    gdet_print = (g_det if self.use_detect else 1.0)
                    print(f"{self.log_prefix} fired: shape={tuple(hs.shape)} tr={self.trend_last:.3f} g_tr={g_tr:.3f} g_det={gdet_print:.3f} s_pre={s_pre:.3f} s={s:.3f} alpha={self.alpha_last:.4f}")
                return _rewrap(hs_new, output)

        self._hook_handle = self.layer_module.register_forward_hook(_forward_hook)

    def detach(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


# ------------------------------ Public API ------------------------------

def _choose_layer(model: nn.Module, tap: int) -> nn.Module:
    """Tap index is negative from top (e.g., -1 = last)."""
    h = model.transformer.h
    n = len(h)
    i = tap if tap >= 0 else n + tap
    i = max(0, min(n-1, i))
    return h[i]


def attach_ngf_hooks(model: nn.Module, cfg: Dict[str, Any] = None, tokenizer=None, **kwargs):
    """
    Attach NGF warp/detect/denoise hook to a HF transformer block.

    Args:
      model: AutoModelForCausalLM (e.g., gpt2)
      cfg: dict of parameters (all optional, good defaults provided):
        tap, alpha0, alpha_min, trend_tau, k_tr,
        use_detect, detect_width, detect_sigma, null_K, null_q, k_det,
        s_latch, linger, ema_center_beta, eps,
        center_mode ('full' | 'pca2'), pca_telemetry (0|1),
        use_denoise (0|1), denoise_* params, print_every
    Returns:
      (model, hook, cfg_used)
    """

    if cfg is None:
        cfg = {}
    # if kwargs has extra keys, fold them into cfg just in case
    for k, v in (kwargs or {}).items():
        cfg.setdefault(k, v)
    
    # defaults
    tap = int(cfg.get("tap", -9))
    alpha0 = float(cfg.get("alpha0", 0.10))
    alpha_min = float(cfg.get("alpha_min", 0.012))
    trend_tau = float(cfg.get("trend_tau", 0.50))
    k_tr = float(cfg.get("k_tr", 12.0))
    use_detect = int(cfg.get("use_detect", 1))
    detect_width = int(cfg.get("detect_width", 32))
    detect_sigma = int(cfg.get("detect_sigma", 5))
    null_K = int(cfg.get("null_K", 64))
    null_q = float(cfg.get("null_q", 0.92))
    k_det = float(cfg.get("k_det", 6.0))
    s_latch = float(cfg.get("s_latch", 0.40))
    linger = int(cfg.get("linger", 3))
    ema_center_beta = float(cfg.get("ema_center_beta", 0.05))
    eps = float(cfg.get("eps", 0.25))
    print_every = int(cfg.get("print_every", 32))
    center_mode = str(cfg.get("center_mode", "full")).lower()
    pca_telemetry = int(cfg.get("pca_telemetry", 1))
    # denoiser
    use_denoise = int(cfg.get("use_denoise", 1))
    denoise_beta = float(cfg.get("denoise_beta", 0.6))
    denoise_window = int(cfg.get("denoise_window", 3))
    denoise_k = float(cfg.get("denoise_k", 8.0))
    denoise_tau = float(cfg.get("denoise_tau", 0.35))
    phantom_tr_tau = float(cfg.get("phantom_tr_tau", 0.60))
    phantom_guard_gamma = float(cfg.get("phantom_guard_gamma", 0.35))
    jitter_eps = float(cfg.get("jitter_eps", 0.03))

    # build helpers
    if center_mode == "full":
        center_ctl = FullCenter(ema_center_beta=ema_center_beta, max_warm=256)
        proj_tele = PCA2Projector.make(max_warm=256, center_xy=None, ema_center_beta=0.0) if pca_telemetry else None
    else:
        center_ctl = None
        proj_tele = PCA2Projector.make(max_warm=256, center_xy=None, ema_center_beta=ema_center_beta) if pca_telemetry or True else None

    denoiser = SoftDenoiser(denoise_beta, denoise_window, denoise_k, denoise_tau,
                            phantom_tr_tau, phantom_guard_gamma, jitter_eps) if use_denoise else None

    # choose tap and attach
    layer = _choose_layer(model, tap)
    hook = TerraformHook(layer_module=layer,
                         center_ctl=center_ctl,
                         proj_tele=proj_tele,
                         alpha0=alpha0, alpha_min=alpha_min,
                         trend_tau=trend_tau, k_tr=k_tr,
                         use_detect=use_detect, detect_width=detect_width, detect_sigma=detect_sigma,
                         null_K=null_K, null_q=null_q, k_det=k_det,
                         linger=linger, s_latch=s_latch,
                         eps=eps, print_every=print_every,
                         log_prefix=f"[NGF hook@tap {tap}]",
                         denoiser=denoiser)
    hook.attach()

    # Ensure model returns hidden states so the hook sees full tensors as expected.
    # (If your caller already sets this, it's harmless.)
    try:
        model.config.output_hidden_states = True
    except Exception:
        pass

    cfg_used = dict(cfg)
    cfg_used.update({
        "tap": tap, "alpha0": alpha0, "alpha_min": alpha_min,
        "trend_tau": trend_tau, "k_tr": k_tr,
        "use_detect": use_detect, "detect_width": detect_width, "detect_sigma": detect_sigma,
        "null_K": null_K, "null_q": null_q, "k_det": k_det,
        "s_latch": s_latch, "linger": linger,
        "ema_center_beta": ema_center_beta, "eps": eps,
        "center_mode": center_mode, "pca_telemetry": pca_telemetry,
        "use_denoise": use_denoise,
        "denoise_beta": denoise_beta, "denoise_window": denoise_window,
        "denoise_k": denoise_k, "denoise_tau": denoise_tau,
        "phantom_tr_tau": phantom_tr_tau, "phantom_guard_gamma": phantom_guard_gamma,
        "jitter_eps": jitter_eps,
    })

    print(f"[NGF] Hooking {model.__class__.__name__} layer tap {tap} "
          f"[Full-D center={center_mode=='full'} | Detect={use_detect} | Denoise={use_denoise}] cfg={cfg_used}")

    return model, hook, cfg_used
