
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Stage-11 A/B/C Eval — with integrated Denoiser
# - Always-on warp (alpha_min > 0), soft trend gate, optional Detect (gain-only).
# - Integrated SoftDenoiser that smooths/scales residuals (never flips direction).
# - gen_mode: {stock|geo} to decide whether decoding happens under warp.
# - Telemetry includes denoiser counters and norms.


"""

# Quick run, GPU-friendly
python3 stage11_ab_eval_base_denoise.py \
  --model gpt2 --layer -9 \
  --prompts wobble_prompts_v1.txt --max_new_tokens 96 \
  --alpha0 0.05 --alpha_min 0.006 \
  --trend_tau 0.35 --k_tr 12 \
  --use_detect 1 --detect_width 24 --detect_sigma 5 \
  --null_K 32 --null_q 0.92 --k_det 7 \
  --s_latch 0.30 --linger 2 --ema_center_beta 0.05 \
  --gen_mode geo --device cuda --print_every 128 \
  --use_denoise 1 \
  --denoise_beta 0.6 --denoise_window 3 \
  --denoise_k 8.0 --denoise_tau 0.35 \
  --phantom_tr_tau 0.60 --phantom_guard_gamma 0.35 \
  --jitter_eps 0.03 \
  --out_json ab_results_geo_v4b_denoise_wobble_v1.json

#2 Disable Detect, prove burst capacity (denoiser ON)
python3 stage11_ab_eval_base_denoise.py \
  --model gpt2 --layer -9 \
  --prompts wobble_prompts_v1.txt --max_new_tokens 96 \
  --alpha0 0.05 --alpha_min 0.006 \
  --trend_tau 0.35 --k_tr 12 \
  --use_detect 0 \
  --linger 3 --s_latch 0.30 --ema_center_beta 0.05 \
  --eps 0.25 --burst_thr 0.30 \
  --gen_mode geo --device cuda \
  --use_denoise 1 --denoise_beta 0.6 --denoise_window 3 \
  --denoise_k 8.0 --denoise_tau 0.35 --phantom_tr_tau 0.60 --phantom_guard_gamma 0.35 \
  --jitter_eps 0.03 \
  --out_json ab_geo_v4b_denoise_noDetect.json

#3 Soften Detect, extend linger
python3 stage11_ab_eval_base_denoise.py \
  --model gpt2 --layer -9 \
  --prompts wobble_prompts_v1.txt --max_new_tokens 96 \
  --alpha0 0.05 --alpha_min 0.006 \
  --trend_tau 0.35 --k_tr 12 \
  --use_detect 1 --detect_width 24 --detect_sigma 5 \
  --null_K 24 --null_q 0.88 --k_det 9 \
  --linger 4 --s_latch 0.25 --ema_center_beta 0.05 \
  --eps 0.25 --burst_thr 0.30 \
  --gen_mode geo --device cuda \
  --use_denoise 1 --denoise_beta 0.6 --denoise_window 3 \
  --denoise_k 8.0 --denoise_tau 0.35 --phantom_tr_tau 0.60 --phantom_guard_gamma 0.35 \
  --jitter_eps 0.03 \
  --out_json ab_geo_v4b_denoise_softDetect.json
  
"""

import argparse, json, os, random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import deque

import numpy as np

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# ------------------------------ Utilities ------------------------------

def moving_average(x: np.ndarray, k: int = 9) -> np.ndarray:
    if k <= 1: return x.copy()
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


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

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
        if resid_vec is None:  # defensive
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
        # Micro-jitter averaging (single-pass; cheap)
        if self.jitter_eps>0:
            j=self.jitter_eps
            out=0.5*(out*(1.0+j)+out*(1.0-j))
        return out, dict(dn_gain=gain, dn_guard=int(guard), dn_ema_norm=ema, dn_med_norm=med)

# ------------------------------ PCA-2 projector ------------------------------

@dataclass
class PCA2Projector:
    pca: Optional[PCA]
    center: Optional[np.ndarray]
    warm: List[np.ndarray]
    max_warm: int
    ema_center_beta: float
    ema_c: Optional[np.ndarray]

    @classmethod
    def make(cls, max_warm: int = 256, center_xy: Optional[Tuple[float,float]] = None, ema_center_beta: float = 0.0):
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
                c = np.mean(np.stack(self.warm,0)[:,:2], axis=0)
            else:
                c = np.zeros(2, dtype=float)
        r = float(np.linalg.norm(y2 - c) + 1e-9)
        return y2, r

# ------------------------------ Terraform Hook (with Denoiser) ------------------------------

class TerraformHook:
    def __init__(self, layer_module: nn.Module, projector: PCA2Projector,
                 alpha0: float = 0.06, alpha_min: float = 0.01,
                 trend_tau: float = 0.32, k_tr: float = 8.0,
                 use_detect: int = 0, detect_width: int = 40, detect_sigma: int = 7,
                 null_K: int = 24, null_q: float = 0.90, k_det: float = 8.0,
                 linger: int = 2, s_latch: float = 0.6,
                 eps: float = 0.0,  # relative step clip; 0 disables
                 print_every: int = 32,
                 log_prefix: str = "[HOOK]",
                 denoiser: Optional[SoftDenoiser]=None):
        self.layer_module = layer_module
        self.projector = projector
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
        self.trend_hist = deque(maxlen=max(192, int(detect_width))) if use_detect else None
        self.rng = np.random.default_rng(20259)
        self.prev_r = None
        self.linger = int(linger)
        self.s_latch = float(s_latch)
        self.linger_left = 0
        self.eps = float(eps)
        self.print_every = int(print_every)
        # stats
        self.steps_seen = 0
        self.steps_applied = 0
        self.alpha_last = 0.0
        self.trend_last = 0.0
        self.radius_last = 0.0
        self.step_norm_last = 0.0
        self.g_tr_last = 0.0
        self.g_det_last = 1.0 if not use_detect else 0.0
        # sequences (fresh per prompt)
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
        self.log_prefix = log_prefix
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
        # new lists (no aliasing)
        self.alpha_seq = []
        self.s_seq = []
        self.trend_seq = []
        self.g_tr_seq = []
        self.g_det_seq = []
        self.detect_score_seq = []
        self.tau_abs_seq = []
        self.radius_seq = []
        self.dn_gain_seq = []
        self.dn_guard_seq = []
        self.dn_ema_norm_seq = []
        self.dn_med_norm_seq = []
        self._prev_s = 0.0
        self.last_print_T = -1
        if self.trend_hist is not None:
            self.trend_hist.clear()
        if self.denoiser is not None:
            self.denoiser.reset()

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
        g_det = float(sigmoid(self.k_det * (score - tau_abs)))
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
                h_last = hs[:, -1, :]
                h_np = h_last[0].detach().cpu().float().numpy()
                self.projector.update_fit(h_np)
                _, r = self.projector.project(h_np)
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

                g_det, score, tau_abs = self._detect_soft()
                g_tr = float(sigmoid(self.k_tr * (self.trend_last - self.trend_tau)))
                self.g_tr_last = g_tr
                self.g_det_last = g_det
                if (g_tr * g_det) >= 0.5:
                    self.linger_left = max(self.linger_left, self.linger)
                g_latch = self.s_latch if self.linger_left > 0 else 0.0
                if self.linger_left > 0:
                    self.linger_left -= 1
                s_pre = g_tr * g_det
                s = max(s_pre, g_latch)

                alpha_t = float(self.alpha_min + (self.alpha0 - self.alpha_min) * s)
                dx = -alpha_t * h_last
                # relative step clip if requested
                if self.eps > 0.0:
                    hnorm = torch.norm(h_last, dim=-1, keepdim=True) + 1e-9
                    dnorm = torch.norm(dx, dim=-1, keepdim=True)
                    ratio = (dnorm / hnorm).clamp_min(1e-9)
                    scale = torch.clamp(self.eps / ratio, max=1.0)
                    dx = dx * scale
                    # revise alpha to reflect the effective step size
                    alpha_t = float((dx.norm().item() / (hnorm.squeeze().item() + 1e-9)))

                # ---- DENOISE residual (vector-level smoothing & scaling) ----
                dn_gain=0.0; dn_guard=0; dn_ema=0.0; dn_med=0.0
                if self.denoiser is not None:
                    dx_np = dx[0].detach().cpu().float().numpy()
                    r_dn, meta = self.denoiser.step(dx_np, self.trend_last,
                                                    self.g_det_last if self.use_detect else 1.0,
                                                    s, self._prev_s)
                    if r_dn is not None:
                        r_t = torch.from_numpy(r_dn).to(hs.device).view_as(dx[0])
                        dx = dx.clone(); dx[0] = r_t
                    dn_gain=float(meta["dn_gain"]); dn_guard=int(meta["dn_guard"])
                    dn_ema=float(meta["dn_ema_norm"]); dn_med=float(meta["dn_med_norm"])
                    self._prev_s = float(s)

                hs_new = hs.clone()
                hs_new[:, -1, :] = hs_new[:, -1, :] + dx

                self.step_norm_last = float(torch.norm(dx).item())
                self.alpha_last = alpha_t
                self.steps_applied += int(alpha_t > 0.0)
                self.alpha_seq.append(self.alpha_last)
                self.s_seq.append(float(s))
                self.g_tr_seq.append(self.g_tr_last)
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

# ------------------------------ Scoring ------------------------------

@torch.no_grad()
def score_stepwise_dlp(model_stock, model_geo, tok, prompt: str, max_new_tokens: int,
                       hook: TerraformHook, device: str, gen_mode: str, burst_thr: float) -> Dict:
    model_stock.eval(); model_geo.eval()
    enc = tok(prompt, return_tensors="pt").to(device)

    # Greedy continuation under chosen generator
    out_ids = enc.input_ids
    gen_model = model_stock if (gen_mode == "stock") else model_geo
    for _ in range(max_new_tokens):
        logits = gen_model(input_ids=out_ids).logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        out_ids = torch.cat([out_ids, next_id], dim=1)
    gen_ids = out_ids

    def _token_logprob(model, ids_ctx, next_id):
        logits = model(input_ids=ids_ctx).logits[:, -1, :]
        logprobs = torch.log_softmax(logits, dim=-1)
        return float(logprobs[0, int(next_id.item())].item())

    hook.reset_for_prompt()
    dlp_seq = []
    for t in range(enc.input_ids.shape[1], gen_ids.shape[1]):
        ctx = gen_ids[:, :t]
        nxt = gen_ids[:, t:t+1]
        lp_stock = _token_logprob(model_stock, ctx, nxt)
        lp_geo   = _token_logprob(model_geo,   ctx, nxt)  # hook runs here
        dlp_seq.append(lp_geo - lp_stock)

    txt = tok.decode(gen_ids[0], skip_special_tokens=True)
    dlp = np.array(dlp_seq, dtype=float) if dlp_seq else np.zeros(0)

    def _safe_mean(x):
        x = np.asarray([v for v in x if v is not None], dtype=float)
        return float(x.mean()) if x.size else 0.0

    # Split by burst membership (s ≥ burst_thr)
    s_arr = np.array(hook.s_seq, dtype=float)
    if s_arr.size:
        inside = s_arr >= float(burst_thr)
    else:
        inside = np.zeros_like(dlp, dtype=bool)
    dlp_in  = _safe_mean([dlp[i] for i in range(min(len(dlp), len(inside))) if inside[i]])
    dlp_out = _safe_mean([dlp[i] for i in range(min(len(dlp), len(inside))) if not inside[i]])

    # Convergence metrics
    r = np.array(hook.radius_seq, dtype=float)
    if r.size >= 2:
        shrink = (r[:-1] - r[1:]) / np.maximum(r[:-1], 1e-9)
        mean_shrink = float(np.mean(shrink))
        start_r, end_r = float(r[0]), float(r[-1])
    else:
        mean_shrink = 0.0; start_r = end_r = float(r[0]) if r.size else 0.0

    # Burst metrics from inside mask
    def _runs(mask: np.ndarray) -> List[int]:
        lens = []
        cnt = 0
        for v in mask.tolist():
            if v:
                cnt += 1
            else:
                if cnt>0: lens.append(cnt)
                cnt = 0
        if cnt>0: lens.append(cnt)
        return lens
    runs = _runs(inside) if s_arr.size else []
    mean_burst = float(np.mean(runs)) if runs else 0.0
    n_bursts = int(len(runs))
    inside_tokens = int(np.sum(inside)) if s_arr.size else 0
    adjacency_ratio = float((inside_tokens - n_bursts) / max(1, inside_tokens)) if inside_tokens>0 else 0.0

    steps_seen = hook.steps_seen
    steps_applied = int(np.sum(np.array(hook.alpha_seq) > 0.0))

    return dict(
        text=txt,
        dlp=float(dlp.mean()) if dlp.size else 0.0,
        dlp_seq=dlp_seq,
        dlp_in=dlp_in,
        dlp_out=dlp_out,
        mean_shrink=mean_shrink,
        start_r=start_r,
        end_r=end_r,
        mean_burst_len=mean_burst,
        n_bursts=n_bursts,
        adjacency_ratio=adjacency_ratio,
        steps_seen=steps_seen,
        steps_applied=steps_applied,
        applied_rate=float(steps_applied / max(1, steps_seen)),
        alpha_seq=hook.alpha_seq,
        s_seq=hook.s_seq,
        trend_seq=hook.trend_seq,
        g_tr_seq=hook.g_tr_seq,
        g_det_seq=hook.g_det_seq,
        detect_score_seq=hook.detect_score_seq,
        tau_abs_seq=hook.tau_abs_seq,
        radius_seq=hook.radius_seq,
        dn_gain_seq=hook.dn_gain_seq,
        dn_guard_seq=hook.dn_guard_seq,
        dn_ema_norm_seq=hook.dn_ema_norm_seq,
        dn_med_norm_seq=hook.dn_med_norm_seq,
    )

# ------------------------------ Runner ------------------------------

def load_prompts(path: str) -> List[str]:
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


def choose_layer(model, idx: int) -> nn.Module:
    h = model.transformer.h
    n = len(h)
    i = idx if idx >= 0 else n + idx
    i = max(0, min(n-1, i))
    return h[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--layer", type=int, default=-9)
    ap.add_argument("--prompts", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Always-on + soft gating
    ap.add_argument("--alpha0", type=float, default=0.07)
    ap.add_argument("--alpha_min", type=float, default=0.012)
    ap.add_argument("--trend_tau", type=float, default=0.30)
    ap.add_argument("--k_tr", type=float, default=10.0)
    ap.add_argument("--ema_center_beta", type=float, default=0.05, help="EMA center adapt rate in PCA-2 (0 disables)")
    ap.add_argument("--eps", type=float, default=0.25, help="Relative step clip; 0 disables")
    # Detect (optional, soft)
    ap.add_argument("--use_detect", type=int, default=0)
    ap.add_argument("--detect_width", type=int, default=40)
    ap.add_argument("--detect_sigma", type=int, default=7)
    ap.add_argument("--null_K", type=int, default=24)
    ap.add_argument("--null_q", type=float, default=0.88)
    ap.add_argument("--k_det", type=float, default=9.0)
    # Bursts
    ap.add_argument("--linger", type=int, default=3)
    ap.add_argument("--s_latch", type=float, default=0.7)
    ap.add_argument("--burst_thr", type=float, default=0.5)
    # Denoiser
    ap.add_argument("--use_denoise", type=int, default=0)
    ap.add_argument("--denoise_beta", type=float, default=0.6)
    ap.add_argument("--denoise_window", type=int, default=3)
    ap.add_argument("--denoise_k", type=float, default=8.0)
    ap.add_argument("--denoise_tau", type=float, default=0.35)
    ap.add_argument("--phantom_tr_tau", type=float, default=0.60)
    ap.add_argument("--phantom_guard_gamma", type=float, default=0.35)
    ap.add_argument("--jitter_eps", type=float, default=0.03)
    # Logging / decode mode
    ap.add_argument("--gen_mode", type=str, default="stock", choices=["stock","geo"], help="Decode tokens with stock (baseline) or geo (warp-on) model")
    ap.add_argument("--print_every", type=int, default=32)
    # Output
    ap.add_argument("--out_json", type=str, default="ab_results.json")
    args = ap.parse_args()

    set_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = args.device

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    stock = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    geo   = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    layer = choose_layer(geo, args.layer)
    projector = PCA2Projector.make(max_warm=256, center_xy=None, ema_center_beta=args.ema_center_beta)
    denoiser = None
    if int(args.use_denoise) == 1:
        denoiser = SoftDenoiser(beta=args.denoise_beta, window=args.denoise_window,
                                 k=args.denoise_k, tau=args.denoise_tau,
                                 phantom_tr_tau=args.phantom_tr_tau,
                                 phantom_guard_gamma=args.phantom_guard_gamma,
                                 jitter_eps=args.jitter_eps)
    hook = TerraformHook(layer, projector,
                         alpha0=args.alpha0, alpha_min=args.alpha_min,
                         trend_tau=args.trend_tau, k_tr=args.k_tr,
                         use_detect=args.use_detect, detect_width=args.detect_width, detect_sigma=args.detect_sigma,
                         null_K=args.null_K, null_q=args.null_q, k_det=args.k_det,
                         linger=args.linger, s_latch=args.s_latch,
                         eps=args.eps, print_every=args.print_every,
                         denoiser=denoiser)
    hook.attach()

    prompts = load_prompts(args.prompts)
    rows = []
    for i, prompt in enumerate(prompts, 1):
        try:
            rec = dict(idx=i, prompt=prompt)
            out = score_stepwise_dlp(stock, geo, tok, prompt, args.max_new_tokens, hook, device, args.gen_mode, args.burst_thr)
            rec.update(out)
        except Exception as e:
            rec = dict(idx=i, prompt=prompt, error=str(e))
        rows.append(rec)

    def _safe_mean(xs):
        xs = [x for x in xs if isinstance(x, (int,float))]
        return float(np.mean(xs)) if xs else 0.0

    dlp_mean = _safe_mean([r.get("dlp") for r in rows])
    mean_shrink_mean = _safe_mean([r.get("mean_shrink") for r in rows])
    steps_seen_total = int(sum(r.get("steps_seen", 0) for r in rows))
    steps_applied_total = int(sum(r.get("steps_applied", 0) for r in rows))
    applied_rate = float(steps_applied_total / max(1, steps_seen_total))

    out = dict(
        config=dict(model=args.model, layer=args.layer,
                    alpha0=args.alpha0, alpha_min=args.alpha_min,
                    trend_tau=args.trend_tau, k_tr=args.k_tr,
                    use_detect=args.use_detect, detect_width=args.detect_width, detect_sigma=args.detect_sigma,
                    null_K=args.null_K, null_q=args.null_q, k_det=args.k_det,
                    linger=args.linger, s_latch=args.s_latch,
                    ema_center_beta=args.ema_center_beta,
                    eps=args.eps, burst_thr=args.burst_thr,
                    gen_mode=args.gen_mode, print_every=args.print_every,
                    max_new_tokens=args.max_new_tokens,
                    prompts_file=args.prompts,
                    use_denoise=int(args.use_denoise),
                    denoise=dict(beta=args.denoise_beta, window=args.denoise_window,
                                 k=args.denoise_k, tau=args.denoise_tau,
                                 phantom_tr_tau=args.phantom_tr_tau,
                                 phantom_guard_gamma=args.phantom_guard_gamma,
                                 jitter_eps=args.jitter_eps) if int(args.use_denoise)==1 else None),
        aggregate=dict(n=len(rows), dlp_mean=dlp_mean, mean_shrink_mean=mean_shrink_mean,
                       steps_seen_total=steps_seen_total, steps_applied_total=steps_applied_total,
                       applied_rate=applied_rate),
        rows=rows,
    )

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[JSON] {args.out_json}")

if __name__ == "__main__":
    main()
