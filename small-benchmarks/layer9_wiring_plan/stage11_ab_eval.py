#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage‑11 A/B harness — persist+debug build
-----------------------------------------
Adds four fixes:
1) **PCA small‑K guard**: avoids q=2 error when <2 samples in the window.
2) **Correct PCs**: uses V[:, :2] from pca_lowrank (feature PCs), not U.
3) **Stepwise scorer**: measures Δlogprob token‑by‑token so hook effects register.
4) **Center control**: optional --freeze_center to forbid runtime overwrites; optional --scan to inject center into calib if missing.

Also: cache disabled in config and generate(); explicit GPT‑2 block bind with fallback; per‑prompt stats; aggregate totals across prompts.

python3 stage11_ab_eval.py \
  --model gpt2 --layer -9 \
  --calib resources/calib_t-9_k12.json \
  --prompts prompts.txt \
  --alpha 0.06 --eps 0.25 --trend_tau 0.50 \
  --freeze_center \
  --out_json ab_results.json

"""
from __future__ import annotations
import argparse, json, os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# -------------------------
# Utils
# -------------------------

def read_prompts(path_or_inline: str) -> List[str]:
    if os.path.isfile(path_or_inline):
        with open(path_or_inline, 'r') as f:
            lines = [ln.strip("\n") for ln in f]
        return [x for x in lines if x.strip()]
    if "\n" in path_or_inline:
        return [s.strip() for s in path_or_inline.split("\n") if s.strip()]
    if "," in path_or_inline:
        return [s.strip() for s in path_or_inline.split(",") if s.strip()]
    return [path_or_inline.strip()]

@dataclass
class Calib:
    center: Optional[Tuple[float,float]] = None
    k_last: int = 12
    alpha: float = 0.05
    eps: float = 0.25
    trend_tau: float = 0.60

    @staticmethod
    def load(path: Optional[str]) -> 'Calib':
        if not path:
            return Calib()
        with open(path, 'r') as f:
            obj = json.load(f)
        c = obj.get('center', None)
        if c is not None and len(c) >= 2:
            center = (float(c[0]), float(c[1]))
        else:
            center = None
        return Calib(
            center=center,
            k_last=int(obj.get('k_last', 12)),
            alpha=float(obj.get('alpha', obj.get('α', 0.05))),
            eps=float(obj.get('eps', 0.25)),
            trend_tau=float(obj.get('tau', obj.get('trend_tau', 0.60))),
        )

# -------------------------
# Scan importer
# -------------------------

def _extract_center_from_scan(path: str) -> Optional[Tuple[float,float]]:
    try:
        with open(path, 'r') as f:
            obj = json.load(f)
    except Exception:
        return None
    for k in ['center','pca_center','c_star','c*']:
        v = obj.get(k)
        if isinstance(v, (list,tuple)) and len(v) >= 2:
            return (float(v[0]), float(v[1]))
    for sect in ['info','warp','detect','denoise','metrics','profile','calib','terraform']:
        d = obj.get(sect, {})
        if isinstance(d, dict):
            for k in ['center','pca_center','c_star','c*']:
                v = d.get(k)
                if isinstance(v, (list,tuple)) and len(v) >= 2:
                    return (float(v[0]), float(v[1]))
    rows = obj.get('rows')
    if isinstance(rows, list) and rows:
        r0 = rows[0]
        if isinstance(r0, dict):
            for k in ['center','pca_center','c_star','c*']:
                v = r0.get(k)
                if isinstance(v, (list,tuple)) and len(v) >= 2:
                    return (float(v[0]), float(v[1]))
    return None

# -------------------------
# Hook
# -------------------------

class TerraformHook(nn.Module):
    def __init__(self, layer_module: nn.Module, calib: Calib, alpha_max: float, device: torch.device, verbose: bool=False):
        super().__init__()
        self.layer = layer_module
        self.calib = calib
        self.alpha_max = float(alpha_max)
        self.device = device
        self.buf: List[torch.Tensor] = []  # (hidden_dim,) per token
        self.center2d: Optional[torch.Tensor] = None  # (2,)
        self.handle = None
        self.verbose = verbose
        # live stats
        self.steps_seen = 0
        self.steps_applied = 0
        self.last_trend = 0.0
        self.last_alpha = 0.0
        self.last_radius = 0.0
        self.last_step_norm = 0.0
        self._printed = 0

    def reset_stats(self):
        self.steps_seen = 0
        self.steps_applied = 0
        self.last_trend = 0.0
        self.last_alpha = 0.0
        self.last_radius = 0.0
        self.last_step_norm = 0.0

    def _pca2(self, X: torch.Tensor):
        mu = X.mean(dim=0)
        Xc = X - mu
        # PCs from feature space (V), not U
        U, S, V = torch.pca_lowrank(Xc, q=2, center=False)
        PCs = V[:, :2]               # (D,2)
        Z = Xc @ PCs                 # (K,2)
        return mu, PCs, Z

    def _trend(self, R: torch.Tensor) -> float:
        if R.numel() < 3:
            return 0.0
        r = R.detach().float()
        t = torch.linspace(1.0, 0.0, steps=r.numel(), device=r.device)
        num = torch.dot(r - r.mean(), t - t.mean())
        den = torch.linalg.vector_norm(r - r.mean()) * torch.linalg.vector_norm(t - t.mean()) + 1e-8
        return float((num / den).clamp(-1, 1).item())

    def enable(self):
        if self.handle is not None:
            return
        self.reset_stats()

        def _hook(module, inputs, output):
            try:
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden.dim() != 3:
                    return output
                B, T, D = hidden.shape
                last = hidden[:, -1, :]
                # append last-token state(s)
                for b in range(B):
                    self.buf.append(last[b].detach().to(self.device))
                # keep last-k
                K = max(4, int(self.calib.k_last))
                if len(self.buf) > K:
                    self.buf = self.buf[-K:]
                # small-K guard: need >=2 to fit 2D PCA
                if len(self.buf) < 2:
                    self.steps_seen += 1
                    self.last_trend = 0.0
                    self.last_alpha = 0.0
                    return output
                X = torch.stack(self.buf, dim=0)  # (k,D)
                mu, PCs, Z = self._pca2(X)
                # ensure shapes are aligned
                if PCs.shape[0] != D or PCs.shape[1] != 2:
                    return output
                z_t = (last[-1] - mu) @ PCs       # (2,)
                # center selection
                if self.center2d is None:
                    if self.calib.center is not None:
                        self.center2d = torch.tensor(self.calib.center, device=self.device, dtype=z_t.dtype)
                    else:
                        self.center2d = Z.mean(dim=0)
                d = (self.center2d - z_t)
                r = torch.linalg.vector_norm(d) + 1e-9
                u = d / r
                # trend over window radii
                Rwin = torch.linalg.vector_norm(Z - self.center2d, dim=1)
                tr = self._trend(Rwin)
                # gate + step
                if tr < self.calib.trend_tau:
                    alpha = 0.0
                else:
                    base = min(self.alpha_max, max(0.0, self.calib.alpha))
                    alpha = float(base * (0.5 + 0.5*tr))
                step2d = alpha * torch.clamp(r, max=self.calib.eps) * u  # (2,)
                stepD = step2d @ PCs.T                                   # (D,)
                # stats
                self.steps_seen += 1
                self.last_trend = float(tr)
                self.last_alpha = float(alpha)
                self.last_radius = float(r)
                self.last_step_norm = float(torch.linalg.vector_norm(stepD).item()) if alpha > 0 else 0.0
                if self.verbose and self._printed < 5:
                    print(f"[HOOK] fired: shape={tuple(hidden.shape)} tr={tr:.3f} alpha={alpha:.4f}")
                    self._printed += 1
                # apply
                if alpha > 0.0:
                    last = last.clone(); last[-1] = last[-1] + stepD
                    self.steps_applied += 1
                    hidden = hidden.clone(); hidden[:, -1, :] = last
                return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
            except Exception as e:
                if self.verbose:
                    print("[HOOK] error:", repr(e))
                return output
        self.handle = self.layer.register_forward_hook(_hook)

    def disable(self):
        if self.handle is not None:
            try:
                self.handle.remove()
            except Exception:
                pass
            self.handle = None

# -------------------------
# Model & eval
# -------------------------

def get_block(module: nn.Module, layer_index: int) -> nn.Module:
    for attr in ["transformer", "model"]:
        if hasattr(module, attr):
            root = getattr(module, attr)
            break
    else:
        root = module
    for name in ["h", "layers", "decoder.layers"]:
        obj = root
        for part in name.split('.'):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                obj = None; break
        if obj is not None:
            blocks = obj
            break
    else:
        raise RuntimeError("Could not locate transformer blocks on this model")
    n = len(blocks)
    idx = layer_index if layer_index >= 0 else (n + layer_index)
    if idx < 0 or idx >= n:
        raise IndexError(f"layer index out of range: {layer_index} in [0,{n-1}]")
    return blocks[idx]

@torch.no_grad()
def greedy_generate(model, tokenizer, prompt, max_new_tokens=64, use_cache=False):
    device = next(model.parameters()).device
    ids = tokenizer(prompt, return_tensors='pt').to(device)
    out_ids = model.generate(**ids, do_sample=False, max_new_tokens=max_new_tokens, use_cache=use_cache)
    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    comp_ids = out_ids[0][ids['input_ids'].shape[1]:]
    return text, out_ids[0], comp_ids

@torch.no_grad()
def mean_logprob_on_tokens_stepwise(model, full_ids: torch.Tensor, target_tail: torch.Tensor) -> float:
    device = next(model.parameters()).device
    L = full_ids.shape[0]
    T = target_tail.shape[0]
    start = L - T
    if start < 1:
        return float('nan')
    total = 0.0
    for i in range(start, L):
        ctx = full_ids[:i].unsqueeze(0).to(device)
        out = model(ctx)  # hook runs here on last position
        logits = out.logits[:, -1, :]
        lp = logits.log_softmax(dim=-1)[0, target_tail[i - start].to(device)]
        total += float(lp.item())
    return total / T

# -------------------------
# Main A/B
# -------------------------

def run_ab(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model_stock = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model_geo   = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model_stock.eval(); model_geo.eval()

    # Disable cache so forwards traverse the full stack and our hook runs
    model_geo.config.use_cache = False
    if hasattr(model_geo, "generation_config"):
        model_geo.generation_config.use_cache = False

    # Load/adjust calib
    calib = Calib.load(args.calib)
    if args.force_fire:
        calib.trend_tau = -1.0

    # Inject center from scan if missing
    if calib.center is None and args.scan:
        ctry = _extract_center_from_scan(args.scan)
        if ctry is not None:
            calib.center = (ctry[0], ctry[1])
            # persist back to file for reproducibility
            try:
                with open(args.calib, 'r') as f:
                    _cal = json.load(f)
            except Exception:
                _cal = {}
            _cal['center'] = [ctry[0], ctry[1]]
            with open(args.calib, 'w') as f:
                json.dump(_cal, f, indent=2)
            print(f"[CALIB] Injected center from scan {os.path.abspath(args.scan)} -> {tuple(calib.center)}")

    # Block bind (explicit GPT‑2 first, fallback otherwise)
    try:
        block = model_geo.transformer.h[(model_geo.config.n_layer + args.layer) if args.layer < 0 else args.layer]
    except Exception:
        block = get_block(model_geo, args.layer)

    hook = TerraformHook(block, calib, alpha_max=args.alpha, device=device, verbose=True)
    hook.enable()

    prompts = read_prompts(args.prompts)
    rows = []
    dlp_sum = 0.0
    total_seen = 0
    total_applied = 0

    for i, p in enumerate(prompts, 1):
        hook.reset_stats()
        stock_text, stock_full_ids, comp_ids = greedy_generate(model_stock, tok, p, args.max_new_tokens, use_cache=False)
        lp_stock = mean_logprob_on_tokens_stepwise(model_stock, stock_full_ids, comp_ids)
        lp_geo   = mean_logprob_on_tokens_stepwise(model_geo,   stock_full_ids, comp_ids)
        dlp = lp_geo - lp_stock
        dlp_sum += dlp
        geo_text, _, _ = greedy_generate(model_geo, tok, p, args.max_new_tokens, use_cache=False)
        rows.append(dict(
            idx=i,
            prompt=p,
            dlp=dlp,
            stock_text=stock_text,
            geo_text=geo_text,
            trend_last=hook.last_trend,
            alpha_last=hook.last_alpha,
            radius_last=hook.last_radius,
            step_norm_last=hook.last_step_norm,
            steps_seen=hook.steps_seen,
            steps_applied=hook.steps_applied,
            applied_rate=(hook.steps_applied / hook.steps_seen if hook.steps_seen else 0.0),
        ))
        total_seen += hook.steps_seen
        total_applied += hook.steps_applied

    # Optionally persist learned center only if not frozen and if calib.center was None
    if (not args.freeze_center) and hook.center2d is not None and calib.center is None:
        calib.center = tuple(float(x) for x in hook.center2d.detach().cpu().tolist())

    agg = dict(
        n=len(prompts),
        dlp_mean=(dlp_sum/len(prompts) if prompts else 0.0),
        steps_seen_total=int(total_seen),
        steps_applied_total=int(total_applied),
        applied_rate=(float(total_applied) / float(total_seen) if total_seen else 0.0),
        trend_tau=float(calib.trend_tau),
    )

    result = dict(
        config=dict(model=args.model, layer=args.layer, alpha=args.alpha, eps=args.eps,
                    k_last=calib.k_last, trend_tau=calib.trend_tau, center=calib.center),
        aggregate=agg,
        rows=rows,
    )

    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump(result, f, indent=2)
    print("[A/B] n=%d | mean Δlogprob@chosen=%+0.4f | applied_rate=%.3f | steps_total=%d" % (
        agg['n'], agg['dlp_mean'], agg['applied_rate'], agg['steps_seen_total']))
    if args.out_json:
        print(f"[JSON] {args.out_json}")

# -------------------------
# CLI
# -------------------------

def build_argparser():
    ap = argparse.ArgumentParser(description="Stage‑11 A/B — persist+debug build")
    ap.add_argument('--model', type=str, default='gpt2')
    ap.add_argument('--layer', type=int, default=-9)
    ap.add_argument('--calib', type=str, required=True, help='path to calibration JSON (may have center:null)')
    ap.add_argument('--scan', type=str, default='', help='optional layer-scan JSON to import center if calib.center is null')
    ap.add_argument('--prompts', type=str, required=True)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--eps', type=float, default=0.25)
    ap.add_argument('--k_last', type=int, default=12)
    ap.add_argument('--trend_tau', type=float, default=0.60)
    ap.add_argument('--max_new_tokens', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--out_json', type=str, default='ab_results.json')
    ap.add_argument('--force_fire', action='store_true')
    ap.add_argument('--freeze_center', action='store_true', help='do not persist any learned center; honor calib center only')
    return ap

if __name__ == '__main__':
    args = build_argparser().parse_args()
    # Hard fail if calib path is wrong (prevents accidental null stub regen elsewhere)
    if not os.path.isfile(args.calib):
        raise FileNotFoundError(f"--calib not found: {args.calib}")
    # Optionally synchronize tau in the calib file to make runs reproducible from JSON alone
    try:
        with open(args.calib, 'r') as f:
            _cal = json.load(f)
        _cal['tau'] = float(args.trend_tau)
        with open(args.calib, 'w') as f:
            json.dump(_cal, f, indent=2)
    except Exception:
        pass
    run_ab(args)
