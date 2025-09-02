#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-11 A/B Eval — Detect-Gated Warp with Burst Telemetry
----------------------------------------------------------
- Hooks a transformer layer (e.g., GPT-2 layer -9) with a warp that only fires
  when a DETECT gate is positive *and* the inward-trend gate passes.
- Re-scores the stock-decoded sequence step-by-step under stock vs geo
  to get per-token Δlogprob (geo − stock).
- Logs per-token sequences: alpha_seq, trend_seq, detect_score_seq, tau_abs_seq.
- Summarizes burst structure (contiguous alpha>0 runs), and computes dlp_in/out.

Run example:
    python3 stage11_ab_eval_v2.py \
      --model gpt2 --layer -9 \
      --prompts prompts.txt --max_new_tokens 64 \
      --alpha0 0.06 --trend_tau 0.35 \
      --detect_width 60 --detect_sigma 9 --null_K 40 --null_q 0.93 --tau_rel 0.60 \
      --seed 42 --out_json ab_results.json

    python3 stage11_ab_eval_v2.py \
      --model gpt2 --layer -9 \
      --prompts wobble_prompts_v1.txt --max_new_tokens 64 \
      --alpha0 0.06 --trend_tau 0.32 \
      --detect_width 32 --detect_sigma 7 --null_K 40 --null_q 0.93 --tau_rel 0.60 \
      --out_json ab_results.json

python3 stage11_ab_eval_v2.py \
      --model gpt2 --layer -9 \
      --prompts wobble_prompts_v1.txt --max_new_tokens 64 \
      --alpha0 0.06 --trend_tau 0.32 \
      --detect_width 16 --detect_sigma 5 --null_K 24 --null_q 0.93 --tau_rel 0.45 \
      --out_json ab_results.json

Notes:
- Trend is inward radial *velocity* in a PCA-2 plane: trend_t = max(0, (r_{t-1}-r_t)/max(r_{t-1},eps)).
  This makes positive spikes when moving toward the center.
- DETECT uses matched filter (half-sine) on the smoothed trend history with dual gates:
  relative (tau_rel) + absolute null threshold (quantile from circular shifts).
- Warp modifies ONLY the last token's hidden state at the hooked layer: dx = -alpha * h_last.
- We unwrap/rewrap outputs in the hook to be robust to tuple-returning layers.
"""
from __future__ import annotations
import argparse, json, os, random, sys
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

# ------------------------------ Helpers: smoothing, matched filter, null ------------------------------

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


def null_threshold(sig: np.ndarray, proto: np.ndarray, K: int = 40, q: float = 0.93, rng=None) -> float:
    rng = rng or np.random.default_rng(20259)
    T = len(sig); L = min(len(proto), T)
    vals = []
    for _ in range(int(K)):
        s = int(rng.integers(1, max(2, T-1)))
        xs = np.roll(sig, s)
        vals.append(float(np.max(xcorr_same(xs[-L:], proto[:L]))))
    return float(np.quantile(vals, q))

# ------------------------------ PCA-2 projector for radius/trend ------------------------------

@dataclass
class PCA2Projector:
    pca: Optional[PCA]
    center: Optional[np.ndarray]
    warm: List[np.ndarray]
    max_warm: int

    @classmethod
    def make(cls, max_warm: int = 256, center_xy: Optional[Tuple[float,float]] = None):
        c = np.array(center_xy, dtype=float) if center_xy is not None else None
        return cls(pca=None, center=c, warm=[], max_warm=max_warm)

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
        c = self.center if self.center is not None else (np.mean(np.stack(self.warm,0)[:,:2], axis=0) if self.warm else np.zeros(2, dtype=float))
        r = float(np.linalg.norm(y2 - c) + 1e-9)
        return y2, r

# ------------------------------ Terraform Hook (warp + detect + logging) ------------------------------

class TerraformHook:
    def __init__(self, layer_module: nn.Module, projector: PCA2Projector,
                 alpha0: float = 0.06, trend_tau: float = 0.35,
                 detect_width: int = 60, detect_sigma: int = 9,
                 null_K: int = 40, null_q: float = 0.93, tau_rel: float = 0.60,
                 log_prefix: str = "[HOOK]"):
        self.layer_module = layer_module
        self.projector = projector
        self.alpha0 = float(alpha0)
        self.trend_tau = float(trend_tau)
        # detect state
        self.detect_sigma = int(detect_sigma)
        self.proto = half_sine_proto(int(detect_width))
        self.null_K = int(null_K)
        self.null_q = float(null_q)
        self.tau_rel = float(tau_rel)
        self.trend_hist = deque(maxlen=max(192, int(detect_width)))
        self.rng = np.random.default_rng(20259)
        # stats
        self.prev_r = None
        self.steps_seen = 0
        self.steps_applied = 0
        self.alpha_last = 0.0
        self.trend_last = 0.0
        self.radius_last = 0.0
        self.step_norm_last = 0.0
        # per-token sequences
        self.alpha_seq: List[float] = []
        self.trend_seq: List[float] = []
        self.detect_score_seq: List[Optional[float]] = []
        self.tau_abs_seq: List[Optional[float]] = []
        # control
        self.enabled = True
        self.log_prefix = log_prefix
        self._hook_handle = None

    def reset_for_prompt(self):
        self.trend_hist.clear()
        self.prev_r = None
        self.steps_seen = 0
        self.steps_applied = 0
        self.alpha_last = 0.0
        self.trend_last = 0.0
        self.radius_last = 0.0
        self.step_norm_last = 0.0
        self.alpha_seq.clear()
        self.trend_seq.clear()
        self.detect_score_seq.clear()
        self.tau_abs_seq.clear()

    def _detect_ok(self) -> Tuple[bool, Optional[float], Optional[float]]:
        if len(self.trend_hist) < 8:
            return False, None, None
        sig = np.asarray(self.trend_hist, dtype=float)
        S = moving_average(sig, k=self.detect_sigma)
        L = min(len(self.proto), len(S))
        proto = self.proto[:L]
        seg = S[-L:]
        corr = xcorr_same(seg, proto)
        score = float(np.max(corr))
        tau_abs = null_threshold(seg, proto, K=self.null_K, q=self.null_q, rng=self.rng)
        # relative gate (single-channel => trivial but keep margin)
        rel_ok = (score >= self.tau_rel * max(score, 1e-12))
        abs_ok = (score >= tau_abs)
        return (rel_ok and abs_ok), score, tau_abs

    def attach(self):
        if self._hook_handle is not None:
            return

        def _unwrap(out):
            if isinstance(out, (tuple, list)):
                return out[0]
            return out

        def _rewrap(new_hs, out_orig):
            if isinstance(out_orig, tuple):
                return (new_hs,) + tuple(out_orig[1:])
            if isinstance(out_orig, list):
                return [new_hs] + list(out_orig[1:])
            return new_hs

        def _forward_hook(module, inputs, output):
            if not self.enabled:
                return output
            hs = _unwrap(output)
            if not torch.is_tensor(hs):
                return output
            self.steps_seen += 1
            with torch.no_grad():
                h_last = hs[:, -1, :]  # (B,H)
                h_np = h_last[0].detach().cpu().float().numpy()
                self.projector.update_fit(h_np)
                _, r = self.projector.project(h_np)
                if self.prev_r is None:
                    trend = 0.0
                else:
                    trend = max(0.0, float((self.prev_r - r) / max(self.prev_r, 1e-6)))
                self.prev_r = r

                self.trend_hist.append(trend)
                detect_ok, score, tau_abs = self._detect_ok()

                gate_ok = (trend >= self.trend_tau) and detect_ok
                if gate_ok:
                    dx = -self.alpha0 * h_last
                    hs_new = hs.clone()
                    hs_new[:, -1, :] = hs_new[:, -1, :] + dx
                    self.step_norm_last = float(torch.norm(dx).item())
                    self.alpha_last = float(self.alpha0)
                    self.steps_applied += 1
                    out_ret = _rewrap(hs_new, output)
                else:
                    self.alpha_last = 0.0
                    self.step_norm_last = 0.0
                    out_ret = output

                # telemetry
                self.trend_last = float(trend)
                self.radius_last = float(r)
                self.alpha_seq.append(float(self.alpha_last))
                self.trend_seq.append(float(trend))
                self.detect_score_seq.append(float(score) if score is not None else None)
                self.tau_abs_seq.append(float(tau_abs) if tau_abs is not None else None)

                # occasional print
                if (self.steps_seen % 32) == 0:
                    print(f"{self.log_prefix} fired: shape={tuple(hs.shape)} tr={self.trend_last:.3f} alpha={self.alpha_last:.4f}")
                return out_ret

        self._hook_handle = self.layer_module.register_forward_hook(_forward_hook)

    def detach(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

# ------------------------------ Scoring: per-token Δlogprob via stock decode ------------------------------

@torch.no_grad()
def score_stepwise_dlp(model_stock, model_geo, tokenizer, prompt: str, max_new_tokens: int,
                       hook: TerraformHook, device: str) -> Dict:
    model_stock.eval(); model_geo.eval()
    enc = tokenizer(prompt, return_tensors="pt").to(device)

    # Greedy STOCK decode to get the reference continuation
    out_ids = enc.input_ids
    for _ in range(max_new_tokens):
        logits = model_stock(input_ids=out_ids).logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        out_ids = torch.cat([out_ids, next_id], dim=1)
    gen_ids = out_ids

    # Per-token logprob under both models, evaluating the chosen next_id
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
        lp_geo   = _token_logprob(model_geo,   ctx, nxt)  # hook fires inside this call
        dlp_seq.append(lp_geo - lp_stock)

    # Summaries
    alpha = np.array(hook.alpha_seq, dtype=float) if hook.alpha_seq else np.zeros(0)
    trend = np.array(hook.trend_seq, dtype=float) if hook.trend_seq else np.zeros(0)
    dlp = np.array(dlp_seq, dtype=float) if dlp_seq else np.zeros(0)

    def _safe_mean(x):
        x = np.asarray([v for v in x if v is not None], dtype=float)
        return float(x.mean()) if x.size else 0.0

    # Burst stats
    bursts = []
    i = 0
    while i < len(alpha):
        if alpha[i] > 0:
            j = i
            while j < len(alpha) and alpha[j] > 0: j += 1
            bursts.append((i, j))  # [i, j)
            i = j
        else:
            i += 1
    burst_lens = [b[1]-b[0] for b in bursts]
    mean_burst_len = float(np.mean(burst_lens)) if burst_lens else 0.0
    n_bursts = int(len(bursts))
    # adjacency ratio: fraction of alpha>0 tokens that have an alpha>0 neighbor
    adj_hits = 0; total_pos = int((alpha > 0).sum())
    for k in range(len(alpha)):
        if alpha[k] > 0:
            if (k>0 and alpha[k-1]>0) or (k+1<len(alpha) and alpha[k+1]>0):
                adj_hits += 1
    adjacency_ratio = float(adj_hits / total_pos) if total_pos else 0.0

    dlp_in  = _safe_mean([dlp[t] for t in range(len(dlp)) if t < len(alpha) and alpha[t] > 0])
    dlp_out = _safe_mean([dlp[t] for t in range(len(dlp)) if t < len(alpha) and alpha[t] == 0])

    # Decode text (for sanity; both paths use stock continuation so strings match)
    txt = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    return dict(
        dlp=float(dlp.mean()) if dlp.size else 0.0,
        dlp_seq=dlp_seq,
        dlp_in=dlp_in,
        dlp_out=dlp_out,
        text=txt,
        steps_seen=hook.steps_seen,
        steps_applied=hook.steps_applied,
        applied_rate=(hook.steps_applied / max(1, hook.steps_seen)),
        trend_last=hook.trend_last,
        alpha_last=hook.alpha_last,
        radius_last=hook.radius_last,
        step_norm_last=hook.step_norm_last,
        alpha_seq=hook.alpha_seq,
        trend_seq=hook.trend_seq,
        detect_score_seq=hook.detect_score_seq,
        tau_abs_seq=hook.tau_abs_seq,
        n_bursts=n_bursts,
        mean_burst_len=mean_burst_len,
        adjacency_ratio=adjacency_ratio,
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
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Warp + trend
    ap.add_argument("--alpha0", type=float, default=0.06)
    ap.add_argument("--trend_tau", type=float, default=0.35)
    ap.add_argument("--center", type=float, nargs=2, default=None, help="Optional fixed PCA2 center (x y)")
    # Detect
    ap.add_argument("--detect_width", type=int, default=60)
    ap.add_argument("--detect_sigma", type=int, default=9)
    ap.add_argument("--null_K", type=int, default=40)
    ap.add_argument("--null_q", type=float, default=0.93)
    ap.add_argument("--tau_rel", type=float, default=0.60)
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
    projector = PCA2Projector.make(max_warm=256, center_xy=tuple(args.center) if args.center else None)
    hook = TerraformHook(layer, projector,
                         alpha0=args.alpha0, trend_tau=args.trend_tau,
                         detect_width=args.detect_width, detect_sigma=args.detect_sigma,
                         null_K=args.null_K, null_q=args.null_q, tau_rel=args.tau_rel)
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
    steps_applied_total = int(sum(r.get("steps_applied", 0) for r in rows))

    out = dict(
        config=dict(model=args.model, layer=args.layer, alpha0=args.alpha0, trend_tau=args.trend_tau,
                    detect_width=args.detect_width, detect_sigma=args.detect_sigma,
                    null_K=args.null_K, null_q=args.null_q, tau_rel=args.tau_rel,
                    center=(args.center if args.center else None)),
        aggregate=dict(n=len(rows), dlp_mean=dlp_mean,
                       steps_seen_total=steps_seen_total,
                       steps_applied_total=steps_applied_total,
                       applied_rate=(steps_applied_total / max(1, steps_seen_total))),
        rows=rows,
    )

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[JSON] {args.out_json}")

if __name__ == "__main__":
    main()
