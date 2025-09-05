#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-11 A/B Eval — Always-On Warp (α_min) + Soft Gating + Optional Detect
---------------------------------------------------------------------------
Purpose: honor the "warped manifold helps" thesis by applying a tiny, always-on
warp (α_min) every token, while smoothly increasing it when evidence supports it.
No hard on/off gate. Optional burst "linger" to avoid single-token flickers.

Key differences from hard-gated variants:
- α_t = α_min + (α0 - α_min) * s_t, where s_t ∈ [0,1] is a *soft* product of
  trend and detect sigmoids (both optional). If detect is off, s_t uses trend only.
- Optional burst dilation (linger L>0): once s_t clears 0.5, keep a minimum s≈s_latch
  for the next L tokens to create short bursts.
- Trend is inward radial velocity in a 2D PCA plane (EMA center optional).

CLI example:
  python3 stage11_ab_eval_v3.py \
    --model gpt2 --layer -9 --prompts wobble_prompts_v1.txt \
    --max_new_tokens 96 --alpha0 0.06 --alpha_min 0.01 \
    --trend_tau 0.32 --k_tr 8.0 \
    --use_detect 1 --detect_width 40 --detect_sigma 7 --null_K 24 --null_q 0.90 --k_det 8.0 \
    --linger 2 --s_latch 0.6 \
    --seed 42 --out_json ab_results.json

python3 stage11_ab_eval_v3.py \
  --model gpt2 --layer -9 --prompts wobble_prompts_v1.txt --max_new_tokens 96 \
  --alpha0 0.07 --alpha_min 0.012 \
  --trend_tau 0.30 --k_tr 10 \
  --use_detect 1 --detect_width 32 --detect_sigma 7 --null_K 24 --null_q 0.88 --k_det 9 \
  --linger 3 --s_latch 0.7 \
  --ema_center_beta 0.05 \
  --out_json ab_results.json

    
"""
from __future__ import annotations
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

# ------------------------------ Helpers ------------------------------

def moving_average(x: np.ndarray, k: int = 9) -> np.ndarray:
    if k <= 1: return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(xp, np.ones(k)/k, mode="valid")


def half_sine_proto(width: int) -> np.ndarray:
    P = np.sin(np.linspace(0, np.pi, int(width)))
    P = P / (np.linalg.norm(P) + 1e-8)
    return P


def xcorr_same(sig: np.ndarray, proto: np.ndarray) -> np.ndarray:
    T = len(sig); L = min(len(proto), T)
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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

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
            c = self.center if self.center is not None else (np.mean(np.stack(self.warm,0)[:,:2], axis=0) if self.warm else np.zeros(2, dtype=float))
        r = float(np.linalg.norm(y2 - c) + 1e-9)
        return y2, r

# ------------------------------ Terraform Hook (always-on + soft gating) ------------------------------

class TerraformHook:
    def __init__(self, layer_module: nn.Module, projector: PCA2Projector,
                 alpha0: float = 0.06, alpha_min: float = 0.01,
                 trend_tau: float = 0.32, k_tr: float = 8.0,
                 use_detect: int = 1, detect_width: int = 40, detect_sigma: int = 7,
                 null_K: int = 24, null_q: float = 0.90, k_det: float = 8.0,
                 linger: int = 2, s_latch: float = 0.6,
                 log_prefix: str = "[HOOK]"):
        self.layer_module = layer_module
        self.projector = projector
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
        # stats
        self.steps_seen = 0
        self.steps_applied = 0
        self.alpha_last = 0.0
        self.trend_last = 0.0
        self.radius_last = 0.0
        self.step_norm_last = 0.0
        # sequences
        self.alpha_seq: List[float] = []
        self.trend_seq: List[float] = []
        self.detect_score_seq: List[Optional[float]] = []
        self.tau_abs_seq: List[Optional[float]] = []
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
        self.alpha_seq = []
        self.trend_seq = []
        self.detect_score_seq = []
        self.tau_abs_seq = []
        if self.trend_hist is not None:
            self.trend_hist.clear()

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
        # Soft gate in [0,1]
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
                if self.trend_hist is not None:
                    self.trend_hist.append(self.trend_last)

                g_det, score, tau_abs = self._detect_soft()
                # Soft trend gate
                g_tr = float(sigmoid(self.k_tr * (self.trend_last - self.trend_tau)))
                # Linger/dilate bursts
                if (g_tr * g_det) >= 0.5:
                    self.linger_left = max(self.linger_left, self.linger)
                g_latch = self.s_latch if self.linger_left > 0 else 0.0
                if self.linger_left > 0:
                    self.linger_left -= 1
                # Final soft gate s in [0,1]
                s = max(g_tr * g_det, g_latch)
                alpha_t = float(self.alpha_min + (self.alpha0 - self.alpha_min) * s)

                dx = -alpha_t * h_last  # always-on + scaled bonus
                hs_new = hs.clone()
                hs_new[:, -1, :] = hs_new[:, -1, :] + dx
                self.step_norm_last = float(torch.norm(dx).item())
                self.alpha_last = alpha_t
                self.steps_applied += 1  # counts all steps (since α_min>0 always applies)
                self.alpha_seq.append(self.alpha_last)
                self.detect_score_seq.append(float(score) if score is not None else None)
                self.tau_abs_seq.append(float(tau_abs) if tau_abs is not None else None)

                if (self.steps_seen % 32) == 0:
                    print(f"{self.log_prefix} fired: shape={tuple(hs.shape)} tr={self.trend_last:.3f} s={s:.3f} alpha={self.alpha_last:.4f}")
                return _rewrap(hs_new, output)

        self._hook_handle = self.layer_module.register_forward_hook(_forward_hook)

    def detach(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

# ------------------------------ Scoring (stock decode, per-step dlp) ------------------------------

@torch.no_grad()
def score_stepwise_dlp(model_stock, model_geo, tok, prompt: str, max_new_tokens: int,
                       hook: TerraformHook, device: str) -> Dict:
    model_stock.eval(); model_geo.eval()
    enc = tok(prompt, return_tensors="pt").to(device)

    # STOCK greedy continuation
    out_ids = enc.input_ids
    for _ in range(max_new_tokens):
        logits = model_stock(input_ids=out_ids).logits[:, -1, :]
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

    # Inside vs outside by s>=0.5 proxy: reconstruct s from alpha
    a = np.array(hook.alpha_seq, dtype=float)
    if a.size:
        s_proxy = np.clip((a - hook.alpha_min) / max(1e-9, (hook.alpha0 - hook.alpha_min)), 0.0, 1.0)
    else:
        s_proxy = np.zeros(0)
    dlp_in  = _safe_mean([dlp[i] for i in range(len(dlp)) if i < len(s_proxy) and s_proxy[i] >= 0.5])
    dlp_out = _safe_mean([dlp[i] for i in range(len(dlp)) if i < len(s_proxy) and s_proxy[i] < 0.5])

    return dict(
        dlp=float(dlp.mean()) if dlp.size else 0.0,
        dlp_seq=dlp_seq,
        dlp_in=dlp_in,
        dlp_out=dlp_out,
        text=txt,
        steps_seen=hook.steps_seen,
        steps_applied=hook.steps_applied,  # now equals steps_seen since α_min>0
        applied_rate=1.0,
        trend_last=hook.trend_last,
        alpha_last=hook.alpha_last,
        radius_last=hook.radius_last,
        step_norm_last=hook.step_norm_last,
        alpha_seq=hook.alpha_seq,
        trend_seq=hook.trend_seq,
        detect_score_seq=hook.detect_score_seq,
        tau_abs_seq=hook.tau_abs_seq,
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
    ap.add_argument("--alpha0", type=float, default=0.06)
    ap.add_argument("--alpha_min", type=float, default=0.01)
    ap.add_argument("--trend_tau", type=float, default=0.32)
    ap.add_argument("--k_tr", type=float, default=8.0)
    ap.add_argument("--ema_center_beta", type=float, default=0.0, help="EMA center adapt rate in PCA plane (0 disables)")
    # Detect (optional)
    ap.add_argument("--use_detect", type=int, default=1)
    ap.add_argument("--detect_width", type=int, default=40)
    ap.add_argument("--detect_sigma", type=int, default=7)
    ap.add_argument("--null_K", type=int, default=24)
    ap.add_argument("--null_q", type=float, default=0.90)
    ap.add_argument("--k_det", type=float, default=8.0)
    # Bursts
    ap.add_argument("--linger", type=int, default=2)
    ap.add_argument("--s_latch", type=float, default=0.6)
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
    hook = TerraformHook(layer, projector,
                         alpha0=args.alpha0, alpha_min=args.alpha_min,
                         trend_tau=args.trend_tau, k_tr=args.k_tr,
                         use_detect=args.use_detect, detect_width=args.detect_width, detect_sigma=args.detect_sigma,
                         null_K=args.null_K, null_q=args.null_q, k_det=args.k_det,
                         linger=args.linger, s_latch=args.s_latch)
    hook.attach()

    prompts = load_prompts(args.prompts)
    rows = []
    for i, prompt in enumerate(prompts, 1):
        try:
            rec = dict(idx=i, prompt=prompt)
            out = score_stepwise_dlp(stock, geo, tok, prompt, args.max_new_tokens, hook, device)
            rec.update(out)
        except Exception as e:
            rec = dict(idx=i, prompt=prompt, error=str(e))
        rows.append(rec)

    def _safe_mean(xs):
        xs = [x for x in xs if isinstance(x, (int,float))]
        return float(np.mean(xs)) if xs else 0.0

    dlp_mean = _safe_mean([r.get("dlp") for r in rows])
    steps_seen_total = int(sum(r.get("steps_seen", 0) for r in rows))
    # with α_min>0, steps_applied_total equals steps_seen_total by design
    steps_applied_total = steps_seen_total

    out = dict(
        config=dict(model=args.model, layer=args.layer,
                    alpha0=args.alpha0, alpha_min=args.alpha_min,
                    trend_tau=args.trend_tau, k_tr=args.k_tr,
                    use_detect=args.use_detect, detect_width=args.detect_width, detect_sigma=args.detect_sigma,
                    null_K=args.null_K, null_q=args.null_q, k_det=args.k_det,
                    linger=args.linger, s_latch=args.s_latch,
                    ema_center_beta=args.ema_center_beta),
        aggregate=dict(n=len(rows), dlp_mean=dlp_mean,
                       steps_seen_total=steps_seen_total,
                       steps_applied_total=steps_applied_total,
                       applied_rate=1.0),
        rows=rows,
    )

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[JSON] {args.out_json}")

if __name__ == "__main__":
    main()
