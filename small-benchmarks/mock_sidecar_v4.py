
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latent-WDD sidecar (v4): hybrid fallback, margin logging, softmax picker,
lite denoiser, sim-style guards (conf_gate, noise_floor, phantom_guard),
and improved PCA calibration over all four endings.

Usage (example):
python3 mock_sidecar_v4.py --mode hellaswag_latent_wdd --split validation --n 250 \
  --tap -9 --k_last 16 --batch_size 32 --max_length 160 \
  --pca_k 32 --calib_split train --calib_n 2000 \
  --pca_cache .cache/pca_gpt2_tap-9_k32_n2000.pkl --svd_solver randomized \
  --detect_z 0.9 --softmax_tau 0.7 --hybrid_fallback 1 --log_margin 1 \
  --use_denoise 1 --dn_ema 0.85 --dn_med_k 3 --dn_guard_tau 0.15 --dn_guard_gamma 0.35 \
  --conf_gate 0.65 --noise_floor 0.03 --probe_k 3 --probe_eps 0.02 \
  --model gpt2 --device auto --amp_dtype auto --debug 20

python3 mock_sidecar_v4.py --mode hellaswag_latent_wdd --split validation --n 250 \
  --tap -9 --k_last 16 --batch_size 32 --max_length 160 \
  --pca_k 32 --calib_split train --calib_n 2000 \
  --pca_cache .cache/pca_gpt2_tap-9_k32_n2000.pkl --svd_solver randomized \
  --detect_z 0.9 --softmax_tau 0.7 --hybrid_fallback 1 --log_margin 1 \
  --use_denoise 1 --dn_ema 0.85 --dn_med_k 3 --dn_guard_tau 0.15 --dn_guard_gamma 0.35 \
  --conf_gate 0.65 --noise_floor 0.03 --probe_k 3 --probe_eps 0.02 \
  --model gpt2 --device auto --amp_dtype auto --debug 20
  
"""

import argparse, os, math, json, numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None

# ------------------------- Utils -------------------------

def set_device(name: str):
    if name == "auto":
        if torch.cuda.is_available(): return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
        return "cpu"
    return name

def pool_lastk(h: torch.Tensor, k: int) -> torch.Tensor:
    # h: [B, T, D] → pooled [B, D]
    k = min(k, h.shape[1])
    return h[:, -k:, :].mean(1)

def softmax_temperature(x: np.ndarray, tau: float) -> np.ndarray:
    if tau <= 0:  # argmax mode
        z = np.zeros_like(x); z[np.argmax(x)] = 1.0; return z
    x = x / max(1e-9, tau)
    x = x - x.max()
    p = np.exp(x); p = p / max(1e-12, p.sum())
    return p

# ------------------------- Lite Denoiser -------------------------

class LiteDenoiser:
    """EMA + tiny median window + guard. Vector in, vector out. Never flips direction."""
    def __init__(self, ema=0.85, med_k=3, guard_tau=0.15, guard_gamma=0.35, jitter_eps=0.0):
        from collections import deque
        self.ema = float(ema)
        self.med_k = int(med_k)
        self.guard_tau = float(guard_tau)
        self.guard_gamma = float(guard_gamma)
        self.jitter_eps = float(jitter_eps)
        self._ema_vec = None
        self._buf = deque(maxlen=self.med_k)
        self._prev_s = 0.0

    def reset(self):
        self._ema_vec = None
        self._buf.clear()
        self._prev_s = 0.0

    def step(self, vec: np.ndarray, s: float, detect_score: float) -> np.ndarray:
        # EMA on vector; keep direction
        v = vec.astype(np.float32)
        if self._ema_vec is None: self._ema_vec = v.copy()
        else: self._ema_vec = self.ema * self._ema_vec + (1.0 - self.ema) * v

        # tiny median smoothing on norm (keep as side info; not used directly)
        rn = float(np.linalg.norm(self._ema_vec) + 1e-12)
        self._buf.append(rn)

        out = self._ema_vec.copy()
        # guard: low strength + low detect → attenuate
        if (s < self.guard_tau) and (detect_score < 0.25):
            out *= self.guard_gamma

        # micro jitter average (cheap)
        if self.jitter_eps > 0.0:
            j = self.jitter_eps
            out = 0.5 * (out * (1.0 + j) + out * (1.0 - j))
        self._prev_s = s
        return out

# ------------------------- Guards (sim-style, simplified) -------------------------

def conf_gate_pass(zscore: float, raw_margin: float, tau: float = 0.65, mmin: float = 0.05) -> bool:
    """
    Gate on detect strength (zscore from MAD) and best-vs-second raw margin.
    """
    return (zscore >= tau) and (raw_margin >= mmin)

def noise_floor_pass(raw_margin: float, floor: float = 0.03) -> bool:
    """Block tiny 'steps' that look like numerical noise."""
    return raw_margin >= floor

def phantom_guard_pass(best_vec: np.ndarray, descend_dir: np.ndarray,
                       k: int = 3, eps: float = 0.02) -> bool:
    """
    Cheap agreement test: jitter the candidate vector and check directional agreement
    with an estimated descent direction. 2/3 majority to pass.
    """
    if k <= 1: return True
    b = best_vec / (np.linalg.norm(best_vec) + 1e-9)
    agree = 0
    base = float(np.linalg.norm(best_vec) + 1e-9)
    for _ in range(k):
        j = np.random.normal(scale=eps * base, size=best_vec.shape)
        v = (best_vec + j) / (np.linalg.norm(best_vec + j) + 1e-9)
        if float(np.dot(v, descend_dir)) > 0: agree += 1
    return agree >= (k // 2 + 1)

# ------------------------- PCA cache -------------------------

@dataclass
class PCABasis:
    mean: np.ndarray
    components: np.ndarray  # [k, D]
    whiten: bool
    var_: Optional[np.ndarray] = None  # for whitening

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xc = X - self.mean
        Y = Xc @ self.components.T
        if self.whiten and self.var_ is not None:
            Y = Y / np.sqrt(np.maximum(self.var_, 1e-12))
        return Y

def fit_pca_cache(X: np.ndarray, k: int, cache_path: str, svd_solver: str = "auto") -> PCABasis:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    if os.path.exists(cache_path):
        obj = np.load(cache_path, allow_pickle=True)
        # Case A: dict saved by v3/v4 (np.save of a dict)
        if isinstance(obj, np.lib.npyio.NpzFile):
            keys = list(obj.keys())
            if {"mean","components","whiten"}.issubset(keys):
                W = {k: obj[k] for k in keys}
                return PCABasis(W["mean"], W["components"], bool(W.get("whiten", True)), W.get("var_"))
            obj = obj["arr_0"]
        if isinstance(obj, dict):
            W = obj
            return PCABasis(W["mean"], W["components"], bool(W.get("whiten", True)), W.get("var_"))
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            obj = obj.item()
            if isinstance(obj, dict):
                W = obj
                return PCABasis(W["mean"], W["components"], bool(W.get("whiten", True)), W.get("var_"))
        # Case B: legacy sklearn PCA object
        try:
            mean = np.asarray(obj.mean_, dtype=np.float32)
            comps = np.asarray(obj.components_, dtype=np.float32)
            var_  = np.asarray(getattr(obj, "explained_variance_", None), dtype=np.float32) if hasattr(obj, "explained_variance_") else None
            return PCABasis(mean, comps, True, var_)
        except Exception:
            pass
        raise RuntimeError(f"Unrecognized PCA cache format at {cache_path}. Delete it or pass a new --pca_cache path.")
    # ---- No cache: fit and save dict format
    if PCA is None:
        raise RuntimeError("scikit-learn is required for PCA caching.")
    p = PCA(n_components=k, whiten=True, random_state=0, svd_solver=svd_solver).fit(X)
    W = dict(
        mean=p.mean_.astype(np.float32),
        components=p.components_.astype(np.float32),
        whiten=True,
        var_=p.explained_variance_.astype(np.float32),
    )
    np.save(cache_path, W, allow_pickle=True)
    return PCABasis(W["mean"], W["components"], True, W["var_"])

# ------------------------- HellaSwag helpers -------------------------

def load_hellaswag(split: str):
    ds = load_dataset("hellaswag", split=split)
    # fields: 'ctx', 'endings' (4), 'label'
    return ds

def make_prompt(ctx: str, ending: str) -> str:
    # simple concat; adjust if you use templates elsewhere
    return ctx.strip() + " " + ending.strip()

@torch.no_grad()
def option_loglik(model, tok, ctx: str, ending: str, device: str, max_length: int) -> float:
    # Stock fallback scorer: sum logprobs of ending conditioned on ctx
    text = make_prompt(ctx, ending)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    ids = enc.input_ids
    logits = model(**enc).logits[:, :-1, :]
    tgt = ids[:, 1:]
    logp = torch.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
    return float(logp)

# ------------------------- Latent harvest → depth -------------------------

def harvest_latents(model, tok, ctx_batch: List[str], endings_batch: List[List[str]],
                    tap: int, k_last: int, max_length: int, device: str):
    """
    Return pooled hidden states per option:
      H: [B, 4, D]
    """
    model.eval()
    with torch.no_grad():
        all_H = []
        for i in range(len(ctx_batch)):
            h4 = []
            for j in range(4):
                text = make_prompt(ctx_batch[i], endings_batch[i][j])
                enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
                out = model(**enc, output_hidden_states=True)
                hs = out.hidden_states[tap]  # [1, T, D]
                pooled = pool_lastk(hs, k_last)  # [1, D]
                h4.append(pooled[0].cpu())
            all_H.append(torch.stack(h4, dim=0))  # [4, D]
        H = torch.stack(all_H, dim=0).numpy().astype(np.float32)  # [B, 4, D]
        return H

def depths_from_latents(H: np.ndarray, pca: PCABasis, denoiser: Optional[LiteDenoiser], s_dbg: float, det_dbg: float):
    """
    Map latents → kD → depth scores per option.
    Returns:
      depth_raw: negative norm in PCA-k (before any rescale), shape [4]
      depth_01 : per-item local [0,1] rescale, shape [4]
      Y        : PCA-projected vectors [4,k] (for guards)
    """
    Y = pca.transform(H)  # [4, k]
    # residual-like vector = -Y (pull inward)
    vec = -Y  # [4, k]
    if denoiser is not None:
        vec = np.stack([denoiser.step(vec[i], s_dbg, det_dbg) for i in range(vec.shape[0])], axis=0)
    depth_raw = -np.linalg.norm(vec, axis=1)  # more negative = deeper (relative energy)
    depth_pos = -depth_raw  # positive magnitude of “well”
    lo, hi = float(np.min(depth_pos)), float(np.max(depth_pos))
    depth_01 = (depth_pos - lo) / max(1e-9, (hi - lo))
    return depth_raw, depth_01, Y

def mad_gate(depth_01: np.ndarray, z_thr: float) -> Tuple[bool, float]:
    """Return (pass, z_score). Higher depth_01 is better."""
    x = depth_01
    m = np.median(x); mad = np.median(np.abs(x - m)) + 1e-9
    z = (x.max() - m) / (1.4826 * mad)
    return (z >= z_thr), float(z)

# ------------------------- Main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="hellaswag_latent_wdd")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--tap", type=int, default=-9)
    ap.add_argument("--k_last", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=160)

    # PCA calibration
    ap.add_argument("--pca_k", type=int, default=8)
    ap.add_argument("--calib_split", default="train")
    ap.add_argument("--calib_n", type=int, default=512)
    ap.add_argument("--pca_cache", default=".cache/pca_gpt2_tap-9_k8.pkl")
    ap.add_argument("--svd_solver", default="auto")

    # Gate and selection
    ap.add_argument("--detect_z", type=float, default=0.7, help="MAD z-threshold")
    ap.add_argument("--softmax_tau", type=float, default=0.0, help="τ=0 → argmax on raw depth; τ>0 → softmax over raw depth")
    ap.add_argument("--hybrid_fallback", type=int, default=1, help="If abstain → use stock loglik scorer")
    ap.add_argument("--log_margin", type=int, default=1, help="Print (best-second) margins for raw and [0,1] depths")

    # Denoiser (optional)
    ap.add_argument("--use_denoise", type=int, default=0)
    ap.add_argument("--dn_ema", type=float, default=0.85)
    ap.add_argument("--dn_med_k", type=int, default=3)
    ap.add_argument("--dn_guard_tau", type=float, default=0.15)
    ap.add_argument("--dn_guard_gamma", type=float, default=0.35)
    ap.add_argument("--dn_jitter_eps", type=float, default=0.0)

    # Guards
    ap.add_argument("--conf_gate", type=float, default=0.65)
    ap.add_argument("--noise_floor", type=float, default=0.03)
    ap.add_argument("--probe_k", type=int, default=3)
    ap.add_argument("--probe_eps", type=float, default=0.02)

    # Model/runtime
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda","mps"])
    ap.add_argument("--amp_dtype", default="auto")
    ap.add_argument("--debug", type=int, default=20)
    args = ap.parse_args()

    device = set_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    # Load HellaSwag
    assert args.mode.startswith("hellaswag"), "Only HellaSwag mode implemented in v4."
    ds_calib = load_hellaswag(args.calib_split)
    ds_eval  = load_hellaswag(args.split)

    # Pick slice
    n = min(args.n, len(ds_eval))
    ctxs  = [ds_eval[i]["ctx"] for i in range(n)]
    ends  = [ds_eval[i]["endings"] for i in range(n)]
    golds = [int(ds_eval[i]["label"]) for i in range(n)]

    # ---- Calibrate PCA on pooled latents from calib set (ALL 4 endings) ----
    calib_n = min(args.calib_n, len(ds_calib))
    calib_ctx = [ds_calib[i]["ctx"] for i in range(calib_n)]
    calib_end = [ds_calib[i]["endings"] for i in range(calib_n)]
    Hc_all = harvest_latents(model, tok, calib_ctx, calib_end, args.tap, args.k_last, args.max_length, device)
    X = Hc_all.reshape(-1, Hc_all.shape[-1])  # [calib_n*4, D]
    print("_calibrate_pca_basis ... ")
    pca = fit_pca_cache(X, args.pca_k, args.pca_cache, svd_solver=args.svd_solver)
    print("_torch_pca_from_sklearn ... ")

    # Denoiser
    deno = LiteDenoiser(args.dn_ema, args.dn_med_k, args.dn_guard_tau, args.dn_guard_gamma, args.dn_jitter_eps) if args.use_denoise else None

    sidecar_correct = 0
    hybrid_correct = 0
    abstains = 0

    for i in range(n):
        # Harvest latents for the 4 options
        Hi = harvest_latents(model, tok, [ctxs[i]], [ends[i]], args.tap, args.k_last, args.max_length, device)[0]  # [4, D]

        # Depths + PCA vecs
        depth_raw, depth_01, Y4 = depths_from_latents(Hi, pca, deno, s_dbg=0.0, det_dbg=1.0)

        # MAD gate
        passed, zscore = mad_gate(depth_01, args.detect_z)

        # RAW preference & margin
        raw_pref = -depth_raw  # larger = better (well magnitude)
        order_raw = np.argsort(-raw_pref)     # desc
        raw_margin = float(raw_pref[order_raw[0]] - raw_pref[order_raw[1]])

        # Guards (stateless; BEFORE selection)
        if not conf_gate_pass(zscore, raw_margin, args.conf_gate) or not noise_floor_pass(raw_margin, args.noise_floor):
            passed = False

        # Phantom guard: estimate a 'descent' direction using mean vector
        best_idx = int(order_raw[0])
        vec_best = -Y4[best_idx]    # inward pull vector
        mean_vec = -Y4.mean(0)
        descend_dir = mean_vec / (np.linalg.norm(mean_vec) + 1e-9)
        if not phantom_guard_pass(vec_best, descend_dir, k=args.probe_k, eps=args.probe_eps):
            passed = False

        # Sidecar selection (softmax over RAW)
        probs = softmax_temperature(raw_pref, args.softmax_tau)
        sidecar_pick = int(np.argmax(probs))

        # Hybrid fallback
        abstain = False
        hybrid_pick = sidecar_pick
        if not passed:
            if args.hybrid_fallback:
                lps = [option_loglik(model, tok, ctxs[i], ends[i][j], device, args.max_length) for j in range(4)]
                hybrid_pick = int(np.argmax(lps))
                abstain = True
            else:
                abstain = True  # pure abstain (no pick change)

        gold = golds[i]
        sidecar_correct += int(sidecar_pick == gold)
        hybrid_correct  += int(hybrid_pick  == gold)
        abstains += int(abstain)

        # pretty-print (debug)
        if args.debug and (i < args.debug):
            order_01  = np.argsort(-depth_01)
            best_01,  second_01  = depth_01[order_01[0]], depth_01[order_01[1]]
            if args.log_margin:
                print(f"[{i}] gold={gold} pred(hybrid)={hybrid_pick} pred(sidecar)={sidecar_pick} "
                      f"abstain={abstain} z={zscore:.3f} raw_margin={raw_margin:+.3f} "
                      f"norm_margin={best_01-second_01:+.3f} depth_01={np.round(depth_01,3)}")
            best_txt = ends[i][hybrid_pick][:84].replace("\\n"," ")
            print(f"    best(hybrid): {best_txt}")

    acc_sidecar = sidecar_correct / max(1, n)
    acc_hybrid  = hybrid_correct  / max(1, n)

    print(f"{n} acc_sidecar: {sidecar_correct}/{n}={acc_sidecar:.4f} | acc_hybrid: {hybrid_correct}/{n}={acc_hybrid:.4f} (abstains={abstains})")

    print("\n=== HellaSwag (sidecar/latent WDD • PCA) ===")
    print(f"model={args.model} dev={set_device(args.device)} tap={args.tap} k_last={args.k_last} pca_k={args.pca_k} "
          f"tau={args.softmax_tau} z_thr={args.detect_z} hybrid={args.hybrid_fallback} dn={args.use_denoise} "
          f"conf_gate={args.conf_gate} noise_floor={args.noise_floor} probe_k={args.probe_k} probe_eps={args.probe_eps} "
          f"split={args.split} n={n} acc_top1_sidecar={acc_sidecar:.4f} acc_top1_hybrid={acc_hybrid:.4f} abstain_rate={abstains/max(1,n):.4f}")

if __name__ == "__main__":
    main()
