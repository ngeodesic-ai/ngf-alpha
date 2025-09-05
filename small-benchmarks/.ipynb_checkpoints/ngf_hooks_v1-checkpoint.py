
import torch
from torch import nn
import numpy as np
import atexit
import os
import re
from typing import Optional, Tuple

EPS = 1e-8

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

class _RenoV1State:
    def __init__(self, cfg, d_model: int):
        self.cfg = cfg
        self.center = None
        self.r0 = None
        self.step_tokens = 0
        self.U_par = None
        self.U_ph  = None
        self._buf = []
        self._buf_max = int(cfg.get('calib_max_tokens', 12000))
        self._calibrated = False
        self._d = d_model
        self.null_mean = None
        self.null_std  = None
        self.latch_val = 0.0
        self.linger_left = 0
        self.cap_pre = []
        self.cap_post = []

def _pca_plane(X: torch.Tensor, k: int = 2) -> torch.Tensor:
    U,S,Vh = torch.linalg.svd(X, full_matrices=False)
    return Vh[:k, :].T.contiguous()

def _robust_center_radius(delta: torch.Tensor):
    B,T,D = delta.shape
    center = delta.mean(dim=(0,1), keepdim=True)
    r = torch.linalg.norm(delta, dim=-1, keepdim=True)
    r0 = r.median().clamp_min(1e-3).view(1,1,1)
    return center, r0

def _winsorize(x: torch.Tensor, q: float) -> torch.Tensor:
    if q is None: return x
    qv = torch.quantile(x.detach(), q)
    return torch.minimum(x, qv)

def _apply_plane_ops(delta: torch.Tensor, U_par: Optional[torch.Tensor], U_ph: Optional[torch.Tensor], squeeze_lambda: float, ph_lambda: float):
    B,T,D = delta.shape
    if U_par is not None and squeeze_lambda > 0:
        P = U_par @ U_par.T
        delta = delta - squeeze_lambda * (delta @ P - delta)
    if U_ph is not None and ph_lambda > 0:
        Pph = U_ph @ U_ph.T
        delta = delta - ph_lambda * (delta @ Pph)
    return delta

def _ensure_numpy(x: torch.Tensor):
    return x.detach().cpu().float().numpy()

def attach_ngf_hooks(model, **cfg):
    device = next(model.parameters()).device
    cfg = _merge_env_overrides(cfg)

    cfg.setdefault('tap', -9)
    cfg.setdefault('alpha0', 0.05)
    cfg.setdefault('alpha_min', 0.006)
    cfg.setdefault('trend_tau', 0.35)
    cfg.setdefault('k_tr', 12)

    cfg.setdefault('use_detect', 1)
    cfg.setdefault('detect_width', 24)
    cfg.setdefault('detect_sigma', 5.0)
    cfg.setdefault('null_K', 32)
    cfg.setdefault('null_q', 0.92)
    cfg.setdefault('k_det', 7)
    cfg.setdefault('g_det_max', 1.35)
    cfg.setdefault('winsor_q', 0.99)

    cfg.setdefault('s_latch', 0.30)
    cfg.setdefault('linger', 2)
    cfg.setdefault('ema_center_beta', 0.05)

    cfg.setdefault('alpha_r_gamma', 0.20)
    cfg.setdefault('alpha_r_p', 1.2)
    cfg.setdefault('squeeze_orth_lambda', 0.10)
    cfg.setdefault('phantom_k', 3)
    cfg.setdefault('phantom_lambda', 0.12)
    cfg.setdefault('anneal_tokens', 12)
    cfg.setdefault('anneal_scale', 1.35)
    cfg.setdefault('calib_max_tokens', 12000)

    save_hidden = int(cfg.get('save_hidden', 0)) == 1
    dump_dir = cfg.get('hidden_dump_dir', None)
    tap = int(cfg['tap'])

    layer_idx = _get_tap_index(model, tap)
    block = model.transformer.h[layer_idx]

    d_model = getattr(getattr(model, "config", None), "n_embd", None)
    if d_model is None and hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        d_model = model.transformer.wte.weight.shape[1]
    if d_model is None:
        d_model = 768

    state = _RenoV1State(cfg, d_model)

    def _hook(module, args, output):
        if isinstance(output, tuple):
            h = output[0]; rest = output[1:]
        else:
            h = output; rest = None

        B,T,D = h.shape

        batch_mean = h.mean(dim=(0,1), keepdim=True)
        if state.center is None:
            state.center = batch_mean.detach()
        else:
            beta = float(cfg['ema_center_beta'])
            state.center = (1.0 - beta)*state.center + beta*batch_mean.detach()

        delta = h - state.center
        if state.r0 is None:
            _, r0_now = _robust_center_radius(delta.detach())
            state.r0 = r0_now.to(h.device)
        else:
            r = torch.linalg.norm(delta.detach(), dim=-1, keepdim=True)
            r_med = r.median().view(1,1,1).clamp_min(1e-3)
            state.r0 = 0.95*state.r0 + 0.05*r_med

        if not state._calibrated and len(state._buf) < state._buf_max:
            take = min(256, B*T)
            idx = torch.randperm(B*T, device=h.device)[:take]
            X = delta.detach().reshape(B*T, D)[idx]
            state._buf.append(X.cpu())
            if sum(x.shape[0] for x in state._buf) >= state._buf_max:
                Xall = torch.cat(state._buf, dim=0)
                Xall = Xall - Xall.mean(dim=0, keepdim=True)
                try:
                    U_par = _pca_plane(Xall.to(h.device), k=2)
                    U, S, Vh = torch.linalg.svd(Xall.to(h.device), full_matrices=False)
                    V = Vh.T
                    k_ph = int(cfg['phantom_k'])
                    if k_ph > 0 and V.shape[1] > 2:
                        U_ph = V[:, 2:2+k_ph].contiguous()
                    else:
                        U_ph = None
                    state.U_par = U_par.detach()
                    state.U_ph  = U_ph.detach() if U_ph is not None else None
                except Exception:
                    state.U_par = None
                    state.U_ph = None
                state._calibrated = True
                state._buf = []

        if int(cfg.get('use_detect', 1)) == 1:
            norms = torch.linalg.norm(delta, dim=-1)
            norms_w = _winsorize(norms, float(cfg['winsor_q']))
            mean = norms_w.mean(); std = norms_w.std().clamp_min(1e-4)
            if state.null_mean is None:
                state.null_mean = mean.detach(); state.null_std = std.detach()
            else:
                state.null_mean = 0.98*state.null_mean + 0.02*mean.detach()
                state.null_std  = 0.98*state.null_std  + 0.02*std.detach()
            z = (norms - state.null_mean) / (state.null_std + EPS)
            k_det = float(cfg['k_det'])
            s_det = torch.sigmoid(k_det * z).unsqueeze(-1)
            latch = float(cfg['s_latch'])
            if latch > 0:
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

        alpha0 = float(cfg['alpha0']); alpha_min = float(cfg['alpha_min'])
        alpha_eff = alpha_min + alpha0 * s_det

        r = torch.linalg.norm(delta, dim=-1, keepdim=True)
        r0 = state.r0 if state.r0 is not None else r.median().view(1,1,1)
        gamma = float(cfg['alpha_r_gamma']); p = float(cfg['alpha_r_p'])
        alpha_eff = alpha_eff * (1.0 + gamma * torch.clamp((r / (r0 + EPS))**p, 0, 3))

        if state.step_tokens < int(cfg['anneal_tokens']):
            alpha_eff = alpha_eff * float(cfg['anneal_scale'])
        state.step_tokens += T

        alpha_eff = torch.clamp(alpha_eff, max=0.25)

        delta_mod = _apply_plane_ops(delta, state.U_par, state.U_ph,
                                     float(cfg['squeeze_orth_lambda']),
                                     float(cfg['phantom_lambda']))

        h_new = state.center + (1.0 - alpha_eff) * delta_mod


        # ----------------- SoftDenoiser (optional) -----------------
        if int(cfg['use_denoise']) == 1:
            # residual after warp
            r = (h_new - state.center)
        
            # (a) phantom-direction damping on residual
            if state.U_ph is not None and float(cfg['denoise_ph_lambda']) > 0:
                Pph = state.U_ph @ state.U_ph.T
                r = r - float(cfg['denoise_ph_lambda']) * (r @ Pph)
        
            mode = cfg['denoise_mode']
            if mode == 'ema':
                beta = float(cfg['denoise_beta'])
                if state.ema_residual is None:
                    state.ema_residual = r.detach()
                else:
                    state.ema_residual = (1.0 - beta) * state.ema_residual + beta * r.detach()
                r = r - beta * (r - state.ema_residual)  # pull toward EMA smoothed residual
        
            elif mode == 'median':
                # Keep a small window of residuals across time; add tiny jitter to avoid ties
                k = max(3, int(cfg['denoise_window']) | 1)  # make odd
                jitter = float(cfg['jitter_std'])
                # Store a shallow copy (detach to freeze for window)
                state._median_buf.append((r.detach() + jitter * torch.randn_like(r)))
                if len(state._median_buf) > k:
                    state._median_buf.pop(0)
                # Compute elementwise median across window
                stack = torch.stack(state._median_buf, dim=0)  # [k,B,T,D]
                r_med, _ = torch.median(stack, dim=0)          # [B,T,D]
                # Blend current residual toward median
                beta = float(cfg['denoise_beta'])
                r = r - beta * (r - r_med)
        
            h_new = state.center + r
        # ----------------------------------------------------------

        

        if save_hidden:
            take = min(64, B*T)
            idx = torch.randperm(B*T, device=h.device)[:take]
            pre_slice = delta.detach().reshape(B*T, D)[idx].unsqueeze(0)
            post_delta = (h_new - state.center).detach().reshape(B*T, D)[idx].unsqueeze(0)
            state.cap_pre.append(_ensure_numpy(pre_slice))
            state.cap_post.append(_ensure_numpy(post_delta))

        if rest is None:
            return h_new
        else:
            return (h_new,) + rest

    handle = block.register_forward_hook(_hook, with_kwargs=False)

    def _save_hidden():
        if not save_hidden or dump_dir is None:
            return
        try:
            os.makedirs(dump_dir, exist_ok=True)
            pre = np.concatenate(state.cap_pre, axis=1) if len(state.cap_pre) else np.zeros((1,0,state._d))
            post = np.concatenate(state.cap_post, axis=1) if len(state.cap_post) else np.zeros((1,0,state._d))
            pre  = pre.reshape(-1, pre.shape[-1])    # <-- ensure (N,C)
            post = post.reshape(-1, post.shape[-1])  # <-- ensure (N,C)
            np.save(os.path.join(dump_dir, f"tap{tap}_pre.npy"), pre)
            np.save(os.path.join(dump_dir, f"tap{tap}_post.npy"), post)
        except Exception as e:
            print(f"[NGF] Warning: failed to save hidden dumps: {e}")

    atexit.register(_save_hidden)

    print(f"[NGF] Hooking GPT-2 layer {layer_idx} (tap={tap}) [Reno-v1] cfg={cfg}")
    print("[NGF] NGF attached via ngf_hooks_v1:attach_ngf_hooks with cfg=" + str(cfg))
    return {"handle": handle, "layer_idx": layer_idx, "cfg": cfg}
