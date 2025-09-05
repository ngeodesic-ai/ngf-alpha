# Write a new Stage‑11 "Reno" hook with Detect, Denoiser, and OutlierGuard.
# File: /mnt/data/ngf_hooks_v2.py


import torch
from torch import nn
import numpy as np
import atexit, os, re
from typing import Optional, Tuple

EPS = 1e-8


"""
# stock model run
python3 ngf_benchmark.py \
  --mode stock --model gpt2 --split validation --n 1000 --max_length 768 --device auto \
  --out_json stock_gpt2_n1000.json

# warp-detect-denoise: outlier supression
export NGF_RENO_CFG="use_denoise=1 denoise_mode=ema denoise_beta=0.22 denoise_ph_lambda=0.35 \
phantom_k=8 phantom_lambda=0.28 squeeze_orth_lambda=0.20 \
k_det=9 g_det_max=1.26 det_robust=mad winsor_q=0.985 \
alpha_min=0.034 alpha0=0.14 alpha_r_gamma=0.45 alpha_r_p=1.6 \
anneal_tokens=40 anneal_scale=1.85 outlier_q=1.0 outlier_alpha_scale=1.0 tap=-9"

python3 ngf_benchmark.py --mode ngf --ngf_import ngf_hooks_v2:attach_ngf_hooks \
  --model gpt2 --tap -9 --n 1000 \
  --alpha0 0.06 --alpha_min 0.012 --trend_tau 0.30 --k_tr 12 \
  --use_detect 1 --detect_width 22 --detect_sigma 4.5 --k_det 8 \
  --s_latch 0.35 --linger 3 --ema_center_beta=0.04 \
  --gen_mode geo --save_hidden 1 --hidden_dump_dir results/maxwarpC_tap9_noOutlier \
  --out_json results/maxwarpC_tap9_noOutlier/metrics.json

F1: 0.356 up from 0.322 baseline

# # warp-detect-denoise: no outlier supression
export NGF_RENO_CFG="use_denoise=1 denoise_mode=ema denoise_beta=0.22 denoise_ph_lambda=0.35 \
phantom_k=8 phantom_lambda=0.28 squeeze_orth_lambda=0.20 \
k_det=9 g_det_max=1.26 det_robust=mad winsor_q=0.985 \
alpha_min=0.034 alpha0=0.14 alpha_r_gamma=0.45 alpha_r_p=1.6 \
anneal_tokens=40 anneal_scale=1.85 outlier_q=1.0 outlier_alpha_scale=1.0 tap=-9"

python3 ngf_benchmark.py --mode ngf --ngf_import ngf_hooks_v2:attach_ngf_hooks \
  --model gpt2 --tap -9 --n 1000 \
  --alpha0 0.06 --alpha_min 0.012 --trend_tau 0.30 --k_tr 12 \
  --use_detect 1 --detect_width 22 --detect_sigma 4.5 --k_det 8 \
  --s_latch 0.35 --linger 3 --ema_center_beta=0.04 \
  --gen_mode geo --save_hidden 1 --hidden_dump_dir results/maxwarpC_tap9_noOutlier \
  --out_json results/maxwarpC_tap9_noOutlier/metrics.json

F1: 0.3570

export NGF_RENO_CFG="use_denoise=1 denoise_mode=ema denoise_beta=0.26 denoise_ph_lambda=0.42 \
phantom_k=12 phantom_lambda=0.36 squeeze_orth_lambda=0.26 \
k_det=9 g_det_max=1.36 det_robust=mad winsor_q=0.985 \
alpha_min=0.038 alpha0=0.18 alpha_r_gamma=0.55 alpha_r_p=1.80 \
anneal_tokens=56 anneal_scale=1.95 outlier_q=1.0 outlier_alpha_scale=1.0 tap=-9"

python3 ngf_benchmark.py --mode ngf --ngf_import ngf_hooks_v2:attach_ngf_hooks \
  --model gpt2 --tap -9 --n 1000 \
  --alpha0 0.06 --alpha_min 0.012 --trend_tau 0.30 --k_tr 12 \
  --use_detect 1 --detect_width 22 --detect_sigma 4.5 --k_det 8 \
  --s_latch 0.35 --linger 3 --ema_center_beta=0.04 \
  --gen_mode geo --save_hidden 1 \
  --hidden_dump_dir results/maxwarpD_<D1|D2|D3> \
  --out_json results/maxwarpD_<D1|D2|D3>/metrics.json
  
F1: 3560


for S in 0 1 2 3 4; do
  OUT="results/maxwarpC_tap9_s${S}"
  export PYTHONHASHSEED=$S; export CUDA_VISIBLE_DEVICES=0
  export NGF_RENO_CFG="use_denoise=1 denoise_mode=ema denoise_beta=0.22 denoise_ph_lambda=0.35 \
phantom_k=8 phantom_lambda=0.28 squeeze_orth_lambda=0.20 k_det=9 g_det_max=1.26 \
winsor_q=0.985 alpha_min=0.034 alpha0=0.14 alpha_r_gamma=0.45 alpha_r_p=1.6 \
anneal_tokens=40 anneal_scale=1.85 tap=-9"
  python3 ngf_benchmark.py --mode ngf --ngf_import ngf_hooks_v2:attach_ngf_hooks \
    --model gpt2 --tap -9 --alpha0 0.06 --alpha_min 0.012 --trend_tau 0.30 --k_tr 12 \
    --use_detect 1 --detect_width 22 --detect_sigma 4.5 --k_det 8 \
    --s_latch 0.35 --linger 3 --ema_center_beta 0.04 \
    --gen_mode geo --save_hidden 1 \
    --hidden_dump_dir "$OUT" --out_json "$OUT/metrics.json"
done

"""

# ------------------------ Utilities ------------------------

def _merge_env_overrides(cfg):
    raw = os.environ.get("NGF_RENO_CFG", "").strip()
    if not raw:
        return cfg
    parts = re.split(r"[,\s]+", raw)
    for p in parts:
        if not p or "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip(); v = v.strip()
        if v.lower() in ("true","false"):
            val = 1 if v.lower()=="true" else 0
        else:
            try:
                if "." in v or "e" in v.lower():
                    val = float(v)
                else:
                    val = int(v)
            except Exception:
                try:
                    val = float(v)
                except Exception:
                    val = v
        cfg[k] = val
    return cfg

def _get_tap_index(model, tap: int) -> int:
    try:
        n_layers = len(model.transformer.h)
    except Exception:
        n_layers = len([m for m in model.modules() if isinstance(m, nn.Module)])
    if tap < 0:
        idx = n_layers + tap
    else:
        idx = tap
    if idx < 0 or idx >= n_layers:
        raise ValueError(f"tap index out of bounds: {tap} -> {idx} with n_layers={n_layers}")
    return idx

def _winsorize(x: torch.Tensor, q: float) -> torch.Tensor:
    if q is None: return x
    qv = torch.quantile(x.detach(), q)
    return torch.minimum(x, qv)

def _robust_stats(norms_w: torch.Tensor, method: str = "mad", q_cap: float = 0.99) -> Tuple[torch.Tensor, torch.Tensor]:
    # norms_w shape [...,1] or [...]; return (mu, sigma) scalars
    x = norms_w.view(-1)
    if method == "mad":
        med = x.median()
        mad = (x - med).abs().median().clamp_min(1e-4)
        mu = med
        sigma = 1.4826 * mad
    elif method == "trim":
        cap = torch.quantile(x.detach(), q_cap)
        core = x[x <= cap]
        mu = core.mean()
        sigma = core.std().clamp_min(1e-4)
    else:  # naive
        mu = x.mean()
        sigma = x.std().clamp_min(1e-4)
    return mu, sigma

def _pca_plane(X: torch.Tensor, k: int = 2) -> torch.Tensor:
    # X: [N,D], mean-centered
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    return Vh[:k, :].T.contiguous()  # [D,k]

def _apply_plane_ops(delta: torch.Tensor, U_par: Optional[torch.Tensor], U_ph: Optional[torch.Tensor],
                     squeeze_lambda: float, ph_lambda: float):
    # delta: [B,T,D]
    if U_par is not None and squeeze_lambda > 0:
        P = U_par @ U_par.T  # [D,D]
        delta = delta - squeeze_lambda * (delta @ P - delta)  # suppress orth-to-plane
    if U_ph is not None and ph_lambda > 0:
        Pph = U_ph @ U_ph.T
        delta = delta - ph_lambda * (delta @ Pph)
    return delta

def _ensure_numpy(x: torch.Tensor):
    return x.detach().cpu().float().numpy()

# ------------------------ State ------------------------

class _RenoV2State:
    def __init__(self, cfg, d_model: int):
        self.cfg = cfg
        self.center = None
        self.r0 = None
        self.step_tokens = 0
        # plane + phantom dirs
        self.U_par = None
        self.U_ph  = None
        # buffer for calibration
        self._buf = []
        self._buf_max = int(cfg.get('calib_max_tokens', 12000))
        self._calibrated = False
        self._d = d_model
        # detect/null stats
        self.null_mean = None
        self.null_std  = None
        self.latch_val = 0.0
        self.linger_left = 0
        # denoiser feature-state (shape-agnostic)
        self.ema_vec = None        # [1,1,D]
        self._median_vecs = []     # list of [1,1,D]
        # capture
        self.cap_pre = []
        self.cap_post = []

# ------------------------ Hook ------------------------

def attach_ngf_hooks(model, **cfg):
    """
    Stage‑11 Reno v2 hook:
      - Always‑on warp with EMA center and radius scaling
      - Detect as soft gain with robust stats (MAD/trimmed) + latch
      - Phantom suppression + orthogonal squeeze
      - Soft Denoiser (EMA or median on pooled feature) + phantom damping
      - OutlierGuard for detect & alpha scaling
      - Hidden capture dumps as (N,C)
    """
    device = next(model.parameters()).device
    cfg = _merge_env_overrides(cfg)

    # ---- Core defaults ----
    cfg.setdefault('tap', -9)
    cfg.setdefault('alpha0', 0.05)
    cfg.setdefault('alpha_min', 0.006)
    cfg.setdefault('ema_center_beta', 0.05)

    # Detect
    cfg.setdefault('use_detect', 1)
    cfg.setdefault('k_det', 7.0)
    cfg.setdefault('winsor_q', 0.99)
    cfg.setdefault('g_det_max', 1.35)
    cfg.setdefault('s_latch', 0.30)
    cfg.setdefault('linger', 2)
    cfg.setdefault('det_robust', 'mad')   # 'mad' | 'trim' | 'mean'
    cfg.setdefault('det_q_cap', 0.99)     # only for 'trim'

    # Reno extras
    cfg.setdefault('alpha_r_gamma', 0.20)
    cfg.setdefault('alpha_r_p', 1.2)
    cfg.setdefault('squeeze_orth_lambda', 0.10)
    cfg.setdefault('phantom_k', 3)
    cfg.setdefault('phantom_lambda', 0.12)
    cfg.setdefault('anneal_tokens', 12)
    cfg.setdefault('anneal_scale', 1.35)
    cfg.setdefault('calib_max_tokens', 12000)

    # Denoiser
    cfg.setdefault('use_denoise', 0)          # 0/1
    cfg.setdefault('denoise_mode', 'ema')     # 'ema' | 'median'
    cfg.setdefault('denoise_beta', 0.15)
    cfg.setdefault('denoise_window', 5)
    cfg.setdefault('jitter_std', 0.02)
    cfg.setdefault('denoise_ph_lambda', 0.35)

    # Outlier guard
    cfg.setdefault('outlier_q', 0.99)             # percentile for marking outliers
    cfg.setdefault('outlier_alpha_scale', 0.25)   # scale alpha on outliers
    # (winsor_q above already tightens stats)

    # Capture
    save_hidden = int(cfg.get('save_hidden', 0)) == 1
    dump_dir = cfg.get('hidden_dump_dir', None)

    # Tap layer
    tap = int(cfg['tap'])
    layer_idx = _get_tap_index(model, tap)
    block = model.transformer.h[layer_idx]

    # Hidden size
    d_model = getattr(getattr(model, "config", None), "n_embd", None)
    if d_model is None and hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        d_model = model.transformer.wte.weight.shape[1]
    if d_model is None:
        d_model = 768

    state = _RenoV2State(cfg, d_model)

    def _hook(module, args, output):
        if isinstance(output, tuple):
            h = output[0]; rest = output[1:]
        else:
            h = output; rest = None

        B,T,D = h.shape

        # --- EMA center ---
        batch_mean = h.mean(dim=(0,1), keepdim=True)
        if state.center is None:
            state.center = batch_mean.detach()
        else:
            beta_c = float(cfg['ema_center_beta'])
            state.center = (1.0 - beta_c) * state.center + beta_c * batch_mean.detach()

        delta = h - state.center  # [B,T,D]

        # --- r0 EMA (robust median) ---
        if state.r0 is None:
            r = torch.linalg.norm(delta.detach(), dim=-1, keepdim=True)
            state.r0 = r.median().view(1,1,1).to(h.device).clamp_min(1e-3)
        else:
            r = torch.linalg.norm(delta.detach(), dim=-1, keepdim=True)
            r_med = r.median().view(1,1,1).clamp_min(1e-3)
            state.r0 = 0.95 * state.r0 + 0.05 * r_med

        # --- Calibration buffer for plane & phantom dirs ---
        if not state._calibrated and len(state._buf) < state._buf_max:
            take = min(256, B*T)
            idx = torch.randperm(B*T, device=h.device)[:take]
            X = delta.detach().reshape(B*T, D)[idx]
            state._buf.append(X.cpu())
            if sum(x.shape[0] for x in state._buf) >= state._buf_max:
                Xall = torch.cat(state._buf, dim=0)  # [N,D]
                Xall = Xall - Xall.mean(dim=0, keepdim=True)
                try:
                    U_par = _pca_plane(Xall.to(h.device), k=2)
                    U, S, Vh = torch.linalg.svd(Xall.to(h.device), full_matrices=False)
                    V = Vh.T
                    k_ph = int(cfg['phantom_k'])
                    U_ph = V[:, 2:2+k_ph].contiguous() if (k_ph > 0 and V.shape[1] > 2) else None
                    state.U_par = U_par.detach()
                    state.U_ph  = U_ph.detach() if U_ph is not None else None
                except Exception:
                    state.U_par = None; state.U_ph = None
                state._calibrated = True
                state._buf = []

        # --- Detect: robust gain ---
        if int(cfg.get('use_detect', 1)) == 1:
            norms = torch.linalg.norm(delta, dim=-1, keepdim=True)  # [B,T,1]
            norms_w = _winsorize(norms, float(cfg['winsor_q']))
            mu, sigma = _robust_stats(norms_w, method=str(cfg['det_robust']), q_cap=float(cfg['det_q_cap']))
            z = (norms - mu) / (sigma + EPS)
            s_det = torch.sigmoid(float(cfg['k_det']) * z)  # [B,T,1]

            # latch/linger
            if float(cfg['s_latch']) > 0:
                s_mean = s_det.mean().item()
                if s_mean > state.latch_val:
                    state.latch_val = s_mean; state.linger_left = int(cfg['linger'])
                elif state.linger_left > 0:
                    state.linger_left -= 1
                else:
                    state.latch_val *= 0.95
                s_det = torch.clamp(s_det + state.latch_val, 0.0, float(cfg['g_det_max']))
        else:
            s_det = torch.ones(B, T, 1, device=h.device)

        # --- Alpha eff ---
        alpha0 = float(cfg['alpha0']); alpha_min = float(cfg['alpha_min'])
        alpha_eff = alpha_min + alpha0 * s_det  # [B,T,1]

        # radius scaling
        r = torch.linalg.norm(delta, dim=-1, keepdim=True)  # [B,T,1]
        r0 = state.r0 if state.r0 is not None else r.median().view(1,1,1)
        gamma = float(cfg['alpha_r_gamma']); p = float(cfg['alpha_r_p'])
        alpha_eff = alpha_eff * (1.0 + gamma * torch.clamp((r / (r0 + EPS))**p, 0, 3))

        # anneal
        if state.step_tokens < int(cfg['anneal_tokens']):
            alpha_eff = alpha_eff * float(cfg['anneal_scale'])
        state.step_tokens += T

        # outlier alpha scaling
        q_out = torch.quantile(r.detach().view(-1), float(cfg['outlier_q']))
        is_out = (r > q_out).float()  # [B,T,1]
        alpha_eff = alpha_eff * (1.0 - is_out * (1.0 - float(cfg['outlier_alpha_scale'])))

        alpha_eff = torch.clamp(alpha_eff, max=0.25)

        # --- Phantom suppression & squeeze ---
        delta_mod = _apply_plane_ops(delta, state.U_par, state.U_ph,
                                     float(cfg['squeeze_orth_lambda']),
                                     float(cfg['phantom_lambda']))

        # --- Warp ---
        h_new = state.center + (1.0 - alpha_eff) * delta_mod

        # --- Denoiser (optional) ---
        if int(cfg['use_denoise']) == 1:
            # residual after warp
            rres = (h_new - state.center)  # [B,T,D]

            # extra phantom damping
            if state.U_ph is not None and float(cfg['denoise_ph_lambda']) > 0:
                Pph = state.U_ph @ state.U_ph.T
                rres = rres - float(cfg['denoise_ph_lambda']) * (rres @ Pph)

            # pooled feature vector [1,1,D], shape-agnostic
            m_cur = rres.detach().mean(dim=(0,1), keepdim=True)  # [1,1,D]
            mode = str(cfg.get('denoise_mode', 'ema'))
            beta = float(cfg['denoise_beta'])

            if mode == 'ema':
                if state.ema_vec is None:
                    state.ema_vec = m_cur
                else:
                    state.ema_vec = (1.0 - beta) * state.ema_vec + beta * m_cur
                rres = rres - beta * (rres - state.ema_vec)

            elif mode == 'median':
                k = max(3, int(cfg['denoise_window']) | 1)  # odd
                jit = float(cfg['jitter_std'])
                state._median_vecs.append(m_cur + jit * torch.randn_like(m_cur))
                if len(state._median_vecs) > k:
                    state._median_vecs.pop(0)
                stack = torch.stack(state._median_vecs, dim=0)  # [k,1,1,D]
                m_med, _ = torch.median(stack, dim=0)          # [1,1,D]
                rres = rres - beta * (rres - m_med)

            h_new = state.center + rres

        # --- Capture ---
        if save_hidden:
            take = min(64, B*T)
            idx = torch.randperm(B*T, device=h.device)[:take]
            pre_slice = delta.detach().reshape(B*T, D)[idx].unsqueeze(0)
            post_delta = (h_new - state.center).detach().reshape(B*T, D)[idx].unsqueeze(0)
            state.cap_pre.append(_ensure_numpy(pre_slice))
            state.cap_post.append(_ensure_numpy(post_delta))

        return (h_new,) + rest if rest is not None else h_new

    handle = block.register_forward_hook(_hook)

    def _save_hidden():
        if not save_hidden or dump_dir is None:
            return
        try:
            os.makedirs(dump_dir, exist_ok=True)
            pre  = np.concatenate(state.cap_pre,  axis=1) if len(state.cap_pre)  else np.zeros((1,0,state._d))
            post = np.concatenate(state.cap_post, axis=1) if len(state.cap_post) else np.zeros((1,0,state._d))
            pre  = pre.reshape(-1, pre.shape[-1])
            post = post.reshape(-1, post.shape[-1])
            np.save(os.path.join(dump_dir, f"tap{tap}_pre.npy"),  pre)
            np.save(os.path.join(dump_dir, f"tap{tap}_post.npy"), post)
        except Exception as e:
            print(f"[NGF] Warning: failed to save hidden dumps: {e}")

    atexit.register(_save_hidden)

    print(f"[NGF] Hooking GPT-2 layer {layer_idx} (tap={tap}) [Reno‑v2] cfg={cfg}")
    return {"handle": handle, "layer_idx": layer_idx, "cfg": cfg}

