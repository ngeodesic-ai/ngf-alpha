# Create a unified Stage-11 runner that covers steps 1–6:
# - Steps 1–4: always-on warp with PCA-2 plane + EMA center (already in previous script)
# - Step 5: soft Detect (gain-only) via matched-filter + null model → g_det
# - Step 6: Soft Denoiser that smooths/scales residuals (never flips sign)
# The script keeps stock vs geo decode via --gen_mode and emits ARC-compatible JSONL.
# Defaults mirror the v4b baseline where reasonable.


# -*- coding: utf-8 -*-
"""
text_arc_unified_steps1_6.py
Stage-11 unified text runner (Steps 1–6): Warp + soft trend gate + Detect (gain) + Soft Denoiser.

Key ideas (short):
- Always-on curvature: alpha_min applied every token on tap layer (default −9).
- Evidence → soft gain s (0..1): product of soft trend gate and (optional) Detect; with latch & linger.
- Denoiser: smooth/scale residuals; never flip inward direction.
- Decode modes: stock (no warp while decoding) vs geo (decode under warp).
- Output: ARC-compatible JSONL with (id, prompt, generation).

Example (v4b-ish defaults):

python3 text_arc_unified.py \
  --config calib/profile_v4b_tap9_text.json \
  --calib calib/math_calib_prompts_420.txt \
  --prompts  calib/ngf_eval_prompts_60.txt \
  --telemetry_jsonl geo_steps1_6.v4b.tap9.telemetry.jsonl \
  --metrics_json    benchmark_results/metrics_geo.v4b.tap9.json \
  --metrics_csv     benchmark_results/metrics_geo.v4b.tap9.csv \
  --perf_profile gpu_T4 --dtype auto --compile 0 \
  --out generations_geo_steps.v4b.tap9.jsonl

python3 text_arc_unified.py \
  --config calib/profile_v4b_tap9_text.json \
  --use_denoise 0 \
  --prompts  calib/ngf_eval_prompts_60.txt \
  --metrics_json    benchmark_results/metrics_geo.v4b.tap9.no_denoise.json \
  --out generations_geo_steps.v4b.tap9.no_denoise.jsonl

python3 text_arc_unified.py \
  --gen_mode stock \
  --prompts  calib/ngf_eval_prompts_60.txt \
  --metrics_json benchmark_results/metrics_stock.v4b.tap9.json \
  --out generations_stock.v4b.tap9.jsonl

# T4-style path: float16, no compile
python3 text_arc_unified.py \
  --config calib/profile_v4b_tap9_text.json \
  --prompts  calib/ngf_eval_prompts_60.txt \
  --perf_profile gpu_T4 --dtype float16 --compile 0 \
  --metrics_json benchmark_results/metrics_geo.v4b.tap9.t4_fp16.json \
  --out generations_geo_steps.v4b.tap9.t4_fp16.jsonl


python3 text_arc_unified.py \
  --config calib/profile_v4b_tap9_text.json \
  --calib calib/math_calib_prompts_420.txt \
  --prompts  calib/ngf_eval_prompts_60.txt \
  --telemetry_jsonl geo_steps1_6.v4b.tap9.telemetry.jsonl \
  --out generations_geo_steps.v4b.tap9.jsonl \
  --save_config run_effective_config.json

python3 text_arc_unified.py \
  --config calib/profile_v4b_tap9_text.json \
  --calib calib/math_calib_prompts_420.txt \
  --prompts  calib/ngf_eval_prompts_60.txt \
  --telemetry_jsonl geo_steps1_6.v4b.tap9.telemetry.jsonl \
  --out generations_geo_steps.v4b.tap9.jsonl \
  --save_config run_effective_config.json

python3 text_arc_unified.py \
  --model gpt2 --tap -9 \
  --calib calib/math_calib_prompts_420.txt \
  --prompts  calib/ngf_eval_prompts_60.txt \
  --gen_mode geo \
  --alpha0 0.05 --alpha_min 0.006 \
  --trend_tau 0.35 --k_tr 12 \
  --use_detect 1 --detect_width 24 --null_K 24 --null_q 0.88 --k_det 9 \
  --s_latch 0.30 --linger 2 --ema_center_beta 0.05 \
  --use_denoise 1 --denoise_beta 0.6 --denoise_window 3 \
  --denoise_k 8.0 --denoise_tau 0.35 --phantom_tr_tau 0.60 --phantom_guard_gamma 0.35 \
  --jitter_eps 0.03 \
  --max_new_tokens 96 \
  --telemetry_jsonl geo_steps.telemetry.jsonl \
  --out generations_geo_steps.jsonl

python3 text_arc_unified.py \
  --model gpt2 --tap -9 \
  --calib calib/math_calib_prompts_420.txt \
  --prompts  calib/ngf_eval_prompts_60.txt \
  --gen_mode geo \
  --alpha0 0.05 --alpha_min 0.006 \
  --trend_tau 0.32 --k_tr 12 \
  --use_detect 1 --detect_width 20 --null_K 32 --null_q 0.92 --k_det 8 \
  --s_latch 0.30 --linger 4 --ema_center_beta 0.05 \
  --use_denoise 1 --denoise_beta 0.6 --denoise_window 5 \
  --denoise_k 6.0 --denoise_tau 0.40 --phantom_tr_tau 0.65 --phantom_guard_gamma 0.45 \
  --jitter_eps 0.03 \
  --max_new_tokens 96 \
  --telemetry_jsonl geo_steps1_6.tweakA.telemetry.jsonl \
  --out generations_geo_steps.tweakA.jsonl


Stock baseline (no warp during decode):
  python3 text_arc_unified.py \
    --model gpt2 --tap -9 \
    --prompts  calib/ngf_eval_prompts_60.txt \
    --gen_mode stock \
    --max_new_tokens 64 \
    --out generations_stock.jsonl
"""

import argparse, json, os, sys, math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None  # optional

# -------------------------- Telemetry --------------------------
# -------------------------- Perf & Metrics helpers --------------------------
def resolve_dtype(device:str, dtype_flag:str):
    if dtype_flag == "float32":
        return torch.float32
    if dtype_flag == "float16":
        return torch.float16
    if dtype_flag == "bfloat16":
        return torch.bfloat16
    # auto
    if device == "cuda":
        # Prefer bfloat16 on Ampere+; else float16
        cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0,0)
        if cap[0] >= 8:  # Ampere+
            return torch.bfloat16
        return torch.float16
    return torch.float32

def apply_perf_profile(args):
    # Global switches
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if args.perf_profile == "cpu_fast":
        # nothing special beyond defaults
        return
    if args.perf_profile in ("gpu_T4","gpu_L4","gpu_A100"):
        # enable SDPA if available (PyTorch 2+)
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception:
            pass

def write_metrics(args, metrics:dict):
    if args.metrics_json:
        try:
            import os, json
            os.makedirs(os.path.dirname(args.metrics_json) or ".", exist_ok=True)
            with open(args.metrics_json, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"[WRITE] Metrics JSON → {args.metrics_json}")
        except Exception as e:
            print(f"[WARN] Failed to write metrics JSON: {e}")
    if args.metrics_csv:
        try:
            import os, csv
            os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)
            keys = sorted(metrics.keys())
            with open(args.metrics_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.writer(f); w.writerow(keys); w.writerow([metrics[k] for k in keys])
            print(f"[WRITE] Metrics CSV → {args.metrics_csv}")
        except Exception as e:
            print(f"[WARN] Failed to write metrics CSV: {e}")

class Telemetry:
    def __init__(self, path=None, log_every=64):
        self.path = path
        self.log_every = int(log_every)
        self._buf = []
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
    def log(self, rec):
        if self.path is None: return
        self._buf.append(rec)
        if len(self._buf) >= self.log_every:
            with open(self.path, "a", encoding="utf-8") as f:
                for r in self._buf:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            self._buf = []
    def flush(self):
        if self.path is None or not self._buf: return
        with open(self.path, "a", encoding="utf-8") as f:
            for r in self._buf:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self._buf = []


# -------------------------- Config profiles --------------------------
def build_parser():
    ap = argparse.ArgumentParser(description="Stage-11 (Steps 1–6) unified runner: warp + detect + denoise")
    # model / io
    ap.add_argument("--model", type=str, default="gpt2")
    #ap.add_argument("--model", type=str, default="gpt2-medium")
    ap.add_argument("--tap", type=int, default=-9)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","mps"])
    ap.add_argument("--seed", type=int, default=20259)
    ap.add_argument("--calib", type=str, default=None, help="Calibration prompts (one per line)")
    ap.add_argument("--prompts", type=str, required=True, help="Eval prompts (one per line)")
    ap.add_argument("--out", type=str, required=True, help="Output JSONL path")
    ap.add_argument("--print_every", type=int, default=32)
    ap.add_argument("--telemetry_jsonl", type=str, default=None,
                     help="If set, write per-token telemetry JSONL (id, t, alpha, s, g_tr, g_det, radius, step_norm)")
    ap.add_argument("--log_every", type=int, default=64)
    # metrics export
    ap.add_argument("--metrics_json", type=str, default=None, help="Write aggregate run metrics to JSON")
    ap.add_argument("--metrics_csv", type=str, default=None, help="Write aggregate run metrics to CSV")
    # perf profiles
    ap.add_argument("--perf_profile", type=str, default=None, choices=[None, "cpu_fast", "gpu_T4", "gpu_L4", "gpu_A100"], help="Preset perf profile")
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto","float32","float16","bfloat16"], help="Model dtype override")
    ap.add_argument("--compile", type=int, default=0, help="Use torch.compile if available (1/0)")

    # decode
    ap.add_argument("--gen_mode", type=str, default="stock", choices=["stock","geo"])
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--do_sample", action="store_true")

    # geometry & gating
    ap.add_argument("--alpha0", type=float, default=0.05, help="Max warp step (when s≈1)")
    ap.add_argument("--alpha_min", type=float, default=0.006, help="Always-on inward step")
    ap.add_argument("--ema_center_beta", type=float, default=0.05)
    ap.add_argument("--eps", type=float, default=None, help="Relative step clip (safety), e.g., 0.25")

    # trend gate (soft)
    ap.add_argument("--trend_tau", type=float, default=0.35)
    ap.add_argument("--k_tr", type=float, default=12.0)
    ap.add_argument("--linger", type=int, default=2, help="short decay after a burst")
    ap.add_argument("--s_latch", type=float, default=0.30, help="floor for s inside a burst")

    # Detect (gain only)
    ap.add_argument("--use_detect", type=int, default=1)
    ap.add_argument("--detect_width", type=int, default=24)
    ap.add_argument("--null_K", type=int, default=24)
    ap.add_argument("--null_q", type=float, default=0.88)
    ap.add_argument("--k_det", type=float, default=9.0)

    # Denoiser
    ap.add_argument("--use_denoise", type=int, default=1)
    ap.add_argument("--denoise_beta", type=float, default=0.6)
    ap.add_argument("--denoise_window", type=int, default=3)
    ap.add_argument("--denoise_k", type=float, default=8.0)
    ap.add_argument("--denoise_tau", type=float, default=0.35)
    ap.add_argument("--phantom_tr_tau", type=float, default=0.60)
    ap.add_argument("--phantom_guard_gamma", type=float, default=0.35)
    ap.add_argument("--jitter_eps", type=float, default=0.03)

    # Profiles
    ap.add_argument("--config", type=str, default=None, help="Load defaults from JSON profile (CLI overrides)")
    ap.add_argument("--save_config", type=str, default=None, help="Save effective args to JSON at end")
    return ap

def parse_args():
    # Pre-parse to fetch --config if present
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    known, _ = pre.parse_known_args()
    ap = build_parser()

    # If config provided, load and set as defaults BEFORE full parse
    if known.config:
        try:
            with open(known.config, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for action in ap._actions:
                if action.dest in cfg:
                    action.default = cfg[action.dest]
        except Exception as e:
            print(f"[WARN] Failed to load config {known.config}: {e}")

    args = ap.parse_args()
    return args

# -------------------------- Utils --------------------------
def choose_device(pref: str) -> torch.device:
    if pref == "cpu": return torch.device("cpu")
    if pref == "cuda" and torch.cuda.is_available(): return torch.device("cuda")
    if pref == "mps" and torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if len(ln) > 0]

def sigmoid(x: float) -> float:
    return 1.0/(1.0+math.exp(-x))

# -------------------------- PCA-2 plane & EMA center --------------------------
@dataclass
class GeoState:
    center: torch.Tensor             # [H]
    U: Optional[torch.Tensor] = None # [H,2]
    have_basis: bool = False

def fit_pca2(hidden_stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = hidden_stack.mean(axis=0)
    X = hidden_stack - mean
    if PCA is not None:
        pca = PCA(n_components=2, svd_solver="auto", random_state=20259)
        pca.fit(X)
        U = pca.components_.T  # [H,2]
    else:
        Xt = torch.from_numpy(X).float()
        Ufull, S, Vt = torch.linalg.svd(Xt, full_matrices=False)
        U = Vt[:2, :].T.detach().cpu().numpy()
    return mean, U

def collect_hidden_for_pca(model, tokenizer, device, prompts: List[str], max_tokens: int=64) -> np.ndarray:
    model.eval()
    hstack = []
    with torch.no_grad():
        for p in prompts:
            toks = tokenizer(p, return_tensors="pt").to(device)
            out = model(**toks, output_hidden_states=True)
            H = out.hidden_states[-1][0]  # [T,H] final layer approx
            K = min(max_tokens, H.shape[0])
            hstack.append(H[-K:, :].detach().cpu().numpy())
    if not hstack:
        return np.zeros((0, model.config.n_embd), dtype=np.float32)
    return np.concatenate(hstack, axis=0)

# -------------------------- Detect: matched filter + null --------------------------
def half_sine_proto(width: int) -> np.ndarray:
    L = max(2, int(width))
    P = np.sin(np.linspace(0, np.pi, L))
    P = P / (np.linalg.norm(P) + 1e-8)
    return P

def xcorr_at(sig: np.ndarray, proto: np.ndarray, t: int) -> float:
    T = len(sig); L = min(len(proto), T)
    a = max(0, t - L//2); b = min(T, a + L); a = max(0, b - L)
    w = sig[a:b]
    pr = proto[:len(w)]
    w = w - w.mean()
    pr = pr - pr.mean()
    denom = (np.linalg.norm(w) * np.linalg.norm(pr) + 1e-8)
    return float(np.dot(w, pr) / denom)

def null_threshold(sig: np.ndarray, proto: np.ndarray, K: int=24, q: float=0.88, rng=None) -> float:
    rng = rng or np.random.default_rng(20259)
    T = len(sig); L = min(len(proto), T)
    vals = []
    for _ in range(int(K)):
        s = int(rng.integers(1, max(2, T)))
        xs = np.roll(sig, s)
        vals.append(float(max(0.0, xcorr_at(xs[-L:], proto[:L], len(xs[-L:])//2))))
    return float(np.quantile(vals, q))

# -------------------------- Soft Denoiser --------------------------
class SoftDenoiser:
    def __init__(self, beta=0.6, window=3, k=8.0, tau=0.35,
                 phantom_tr_tau=0.60, phantom_guard_gamma=0.35, jitter_eps=0.03):
        from collections import deque
        self.beta=float(beta); self.window=int(window)
        self.k=float(k); self.tau=float(tau)
        self.phantom_tr_tau=float(phantom_tr_tau)
        self.phantom_guard_gamma=float(phantom_guard_gamma)
        self.jitter_eps=float(jitter_eps)
        self._ema=None
        self._buf=deque(maxlen=self.window)

    def reset(self):
        self._ema=None; self._buf.clear()

    def step(self, resid_vec: torch.Tensor, tr: float, g_det: float, s: float, prev_s: float):
        if resid_vec is None:
            return resid_vec
        r = resid_vec.detach().cpu().numpy()
        if self._ema is None:
            self._ema = r.copy()
        else:
            self._ema = (1.0 - self.beta)*self._ema + self.beta*r
        self._buf.append(self._ema.copy())
        med = np.median(np.stack(list(self._buf), axis=0), axis=0)
        conf = 1.0/(1.0 + np.exp(-self.k*((tr) - self.tau)))
        if tr < self.phantom_tr_tau and (s > prev_s):
            conf *= (1.0 - self.phantom_guard_gamma)
        med = med * (1.0 - self.jitter_eps) + r * self.jitter_eps
        out = torch.from_numpy(med * conf).to(resid_vec.device).type_as(resid_vec)
        if torch.dot(out.flatten(), resid_vec.flatten()) < 0:
            out = 0.0 * resid_vec
        return out

# -------------------------- Inward residual & Hook --------------------------
def make_inward(delta_vec: torch.Tensor, alpha: float, eps: Optional[float], x_last: torch.Tensor) -> torch.Tensor:
    step = alpha * delta_vec
    if eps is not None and eps > 0:
        xmax = torch.norm(x_last, dim=-1, keepdim=True).clamp_min(1e-8)
        sn = torch.norm(step, dim=-1, keepdim=True).clamp_min(1e-8)
        scale = torch.minimum(torch.ones_like(sn), (eps * xmax) / sn)
        step = step * scale
    return step

class TapWarpHook(nn.Module):
    def __init__(self, geo: GeoState, args):
        super().__init__()
        self.geo = geo
        self.args = args
        self.prev_s = 0.0
        self.denoiser = SoftDenoiser(beta=args.denoise_beta,
                                     window=args.denoise_window,
                                     k=args.denoise_k,
                                     tau=args.denoise_tau,
                                     phantom_tr_tau=args.phantom_tr_tau,
                                     phantom_guard_gamma=args.phantom_guard_gamma,
                                     jitter_eps=args.jitter_eps) if args.use_denoise else None
        self.trend_hist = []  # radius shrink history for detect
        self.proto = half_sine_proto(args.detect_width)
        # Telemetry
        self.telemetry = None
        self.current_id = None
        self.tok_idx = 0
        # Aggregates for metrics
        self.tot_tokens = 0
        self.sum_s = 0.0
        self.sum_gdet = 0.0
        self.sum_alpha = 0.0
        self.bursts = 0
        self.sum_burst_len = 0
        self._in_burst = False
        self._cur_burst = 0
    def set_telemetry(self, telemetry): self.telemetry = telemetry
    def set_current_id(self, pid:int): self.current_id = int(pid); self.tok_idx = 0

    def forward(self, module, input, output):
        if not isinstance(output, torch.Tensor) or output.dim()!=3:
            return output
        B,T,H = output.shape
        if T<1: return output
        x = output
        x_last = x[:, -1, :]
        c = self.geo.center.to(x.device)

        v = x_last - c
        if self.geo.have_basis and self.geo.U is not None:
            U = self.geo.U.to(x.device)
            proj = (v @ U) @ U.T
            inward = -proj
        else:
            inward = -v
        norm = torch.norm(inward, dim=-1, keepdim=True).clamp_min(1e-8)
        unit_inward = inward / norm

        tr = torch.ones(B, device=x.device)
        tr_val = float(tr.mean().item())
        g_tr = 1.0/(1.0+math.exp(-self.args.k_tr*(tr_val - self.args.trend_tau)))

        if self.args.use_detect:
            self.trend_hist.append(tr_val)
            if len(self.trend_hist) > max(4, self.args.detect_width*3):
                self.trend_hist = self.trend_hist[-max(4, self.args.detect_width*3):]
            sig = np.array(self.trend_hist, dtype=float)
            g_raw = xcorr_at(sig, self.proto, len(sig)-1)
            thr = null_threshold(sig, self.proto, K=self.args.null_K, q=self.args.null_q) if len(sig)>=4 else 0.0
            g_det = 1.0/(1.0+math.exp(-self.args.k_det*(g_raw - thr)))
        else:
            g_det = 1.0

        s_pre = g_tr * g_det
        s = max(self.args.s_latch, s_pre) if s_pre > self.prev_s else max(0.0, self.prev_s - 1e-3*self.args.linger)
        self.prev_s = s
        alpha = self.args.alpha_min + (self.args.alpha0 - self.args.alpha_min) * s

        resid = make_inward(unit_inward, alpha=alpha, eps=self.args.eps, x_last=x_last)
        if self.denoiser is not None and self.args.use_denoise:
            resid = self.denoiser.step(resid, tr=tr_val, g_det=g_det, s=s, prev_s=s)
        x[:, -1, :] = x_last + resid

        # Telemetry
        try:
            if self.telemetry is not None and self.current_id is not None:
                radius = float(torch.norm((x_last[0]-self.geo.center.to(x.device))).item())
                step_norm = float(torch.norm(resid[0]).item())
                rec = {
                    "id": int(self.current_id),
                    "t": int(self.tok_idx),
                    "alpha": float(alpha),
                    "s": float(s),
                    "g_tr": float(g_tr),
                    "g_det": float(g_det),
                    "radius": radius,
                    "step_norm": step_norm
                }
                self.telemetry.log(rec)
                self.tok_idx += 1
        except Exception:
            pass
        return x

# -------------------------- Generation --------------------------
@torch.no_grad()
def generate_one(model, tokenizer, device, prompt: str, max_new_tokens: int, temperature: float, top_p: float, do_sample: bool):
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    gen_ids = model.generate(
        **toks,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=(temperature if do_sample else None),
        top_p=(top_p if do_sample else None),
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_tok_cnt = int(gen_ids[0].shape[0] - toks['input_ids'].shape[1])
    text = tokenizer.decode(gen_ids[0][toks['input_ids'].shape[1]:], skip_special_tokens=True)
    return text, new_tok_cnt

# -------------------------- Main --------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device); print(f"[INFO] Using device: {device}")

    # perf setup
    apply_perf_profile(args)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    torch_dtype_opt = resolve_dtype(device.type, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch_dtype_opt)
    model.to(device)
    try:
        if int(args.compile) == 1 and device.type in ('cuda','cpu') and hasattr(torch, 'compile'):
            model = torch.compile(model, mode='reduce-overhead')
            print('[INFO] torch.compile enabled')
    except Exception as e:
        print(f'[WARN] torch.compile unavailable: {e}')

    n_layers = len(model.transformer.h)
    tap = args.tap if args.tap>=0 else (n_layers + args.tap)
    if not (0 <= tap < n_layers):
        raise ValueError(f"tap {args.tap} resolves out of range for n_layers={n_layers}")
    print(f"[INFO] Tap resolved to block index {tap} (tap={args.tap})")

    H = model.config.n_embd
    center = torch.zeros(H)
    U2 = None; have_basis = False

    if args.calib and os.path.exists(args.calib):
        cprompts = read_lines(args.calib)
        if len(cprompts)>0:
            print(f"[INFO] Calibrating PCA-2 over {len(cprompts)} prompts …")
            hs = collect_hidden_for_pca(model, tok, device, cprompts)
            if hs.shape[0] >= 8:
                mean, U = fit_pca2(hs)
                center = torch.from_numpy(mean).float()
                U2 = torch.from_numpy(U).float()
                have_basis = True
                print(f"[INFO] PCA-2 OK: stack={hs.shape}")
            else:
                print("[WARN] Not enough tokens for PCA; fallback to EMA center only.")
        else:
            print("[WARN] Calibration file empty; skipping PCA.")
    else:
        if args.gen_mode == "geo":
            print("[WARN] No calibration provided; geo mode will run with EMA-only center.")
    geo = GeoState(center=center, U=U2, have_basis=have_basis)

    # Telemetry
    telemetry = Telemetry(args.telemetry_jsonl, args.log_every)

    hook_handle = None
    if args.gen_mode == "geo":
        warp_hook = TapWarpHook(geo=geo, args=args)
        warp_hook.set_telemetry(telemetry)
        hook_handle = model.transformer.h[tap].register_forward_hook(warp_hook)
        print(f"[HOOK] Warp+Detect+Denoise active at block {tap}")

    eval_prompts = read_lines(args.prompts)
    if len(eval_prompts)==0:
        print("[ERROR] No prompts found.", file=sys.stderr); sys.exit(2)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n=0
    total_new_tokens=0
    with open(args.out, "w", encoding="utf-8") as fout:
        for i,p in enumerate(eval_prompts,1):
            if args.gen_mode == "geo":
                toks = tok(p, return_tensors="pt").to(device)
                out = model(**toks, output_hidden_states=True)
                Hlast = out.hidden_states[-1][0, -1, :].detach().cpu()
                beta = float(args.ema_center_beta)
                geo.center = (1.0 - beta)*geo.center + beta*Hlast
                warp_hook.set_current_id(i)
            text, newtok = generate_one(model, tok, device, p, args.max_new_tokens, args.temperature, args.top_p, args.do_sample)
            telemetry.flush()
            total_new_tokens += int(newtok)
            fout.write(json.dumps({"id": i, "prompt": p, "generation": text.strip()}, ensure_ascii=False) + "\n")
            n+=1
            if n % max(1, args.print_every) == 0:
                print(f"[GEN] {n} prompts …")

    if hook_handle is not None:
        hook_handle.remove()
    print(f"[WRITE] Generations → {args.out}  (count={n})")
    # Aggregate metrics from hook/runner
    metrics = {
        "prompts": int(n),
        "total_new_tokens": int(total_new_tokens),
        "avg_new_tokens_per_prompt": (float(total_new_tokens)/max(1,int(n))),
    }
    try:
        if args.gen_mode == "geo":
            # Access hook aggregates
            metrics.update({
                "hook_tot_tokens": int(warp_hook.tot_tokens),
                "hook_mean_s": (warp_hook.sum_s/max(1,warp_hook.tot_tokens)),
                "hook_mean_g_det": (warp_hook.sum_gdet/max(1,warp_hook.tot_tokens)),
                "hook_mean_alpha": (warp_hook.sum_alpha/max(1,warp_hook.tot_tokens)),
                "hook_mean_burst_len": (warp_hook.sum_burst_len/max(1,warp_hook.bursts) if warp_hook.bursts>0 else 0.0),
                "hook_n_bursts": int(warp_hook.bursts)
            })
    except Exception as e:
        print(f"[WARN] Metrics aggregation failed: {e}")
    write_metrics(args, metrics)
    if getattr(args, "save_config", None):
        try:
            with open(args.save_config, "w", encoding="utf-8") as f:
                json.dump(vars(args), f, indent=2, ensure_ascii=False)
            print(f"[WRITE] Effective config → {args.save_config}")
        except Exception as e:
            print(f"[WARN] Failed to save config: {e}")
    print("[DONE] Stage-11 Steps 1–6 complete.")

if __name__ == "__main__":
    main()
