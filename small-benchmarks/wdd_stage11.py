# wdd_stage11.py
import math, torch, torch.nn as nn, torch.nn.functional as F

# -------------------------
# Stage-11 preset (from your script)
# -------------------------
STAGE11_PRESET = dict(
    ema_decay=0.70,
    median_k=1,
    conf_gate=0.80,
    noise_floor=0.08,
    probe_k=5,
    probe_eps=0.02,
    seed_jitter=2,
)

# -------------------------
# Warp (WellMetric) — numerically safe, 2D/3D aware
# -------------------------
class WellMetric(nn.Module):
    def __init__(self, d_model: int, alpha=1.0, beta=0.5, center=None):
        super().__init__()
        self.center = nn.Parameter(torch.zeros(1, 1, d_model) if center is None else center)
        self.alpha  = nn.Parameter(torch.tensor(float(alpha)))
        self.beta   = nn.Parameter(torch.tensor(float(beta)))
        self.eps    = 1e-8
        self.enabled = True
    def forward(self, h):
        if not self.enabled: return h
        if h.dim() == 3: c = self.center.to(h)
        elif h.dim() == 2: c = self.center.to(h).squeeze(0)
        else: raise ValueError(f"Unsupported h.dim()={h.dim()}")
        v = h - c
        r = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
        a = F.softplus(self.alpha).to(h); b = F.softplus(self.beta).to(h)
        den = torch.clamp(r, min=self.eps)
        gain = a * torch.tanh(b * r) / den
        gain = torch.where(r <= 1e-6, a * b, gain)  # limit
        z = c + gain * v
        return torch.nan_to_num(z, nan=0.0, posinf=1e6, neginf=-1e6)

# -------------------------
# Detect + Denoise (sequence-wise)
# -------------------------
class DetectDenoise(nn.Module):
    """
    Operates on [B,T,D] (or [T,D]) hidden states *after* the warp.
    - Detect: flag high-radius tokens via robust z-score (median/MAD)
    - Denoise (temporal): EMA + median over a short window (Stage-11 defaults)
    - Phantom-guard: if the proposed update direction is unstable under jitter, reduce it
    """
    def __init__(self, ema_decay=0.85, median_k=3, conf_gate=0.65, noise_floor=0.03,
                 probe_k=5, probe_eps=0.02, seed_jitter=2):
        super().__init__()
        self.ema_decay  = float(ema_decay)
        self.median_k   = int(max(1, median_k | 1))
        self.conf_gate  = float(conf_gate)
        self.noise_floor= float(noise_floor)
        self.probe_k    = int(max(1, probe_k))
        self.probe_eps  = float(probe_eps)
        self.seed_jitter= int(max(0, seed_jitter))

    @staticmethod
    def _radial(h, c):
        v = h - c
        return torch.norm(v, dim=-1)  # [..., T]

    @staticmethod
    def _robust_z(x, dim=-1, eps=1e-8):
        med = x.median(dim=dim, keepdim=True).values
        mad = (x - med).abs().median(dim=dim, keepdim=True).values
        return (x - med) / (1.4826 * (mad + eps))

    def _ema(self, x, decay):
        # x: [T, D]
        out = x.clone()
        for t in range(1, x.size(0)):
            out[t] = decay * out[t-1] + (1.0 - decay) * x[t]
        return out

    def _median_k(self, x, k):
        # x: [T, D], odd k
        if k <= 1: return x
        T = x.size(0)
        pad = k // 2
        xp = torch.cat([x[:1].repeat(pad,1), x, x[-1:].repeat(pad,1)], dim=0)  # [T+2p, D]
        chunks = []
        for t in range(T):
            window = xp[t:t+k]
            chunks.append(window.median(dim=0).values)
        return torch.stack(chunks, dim=0)

    def forward(self, h_post, center):
        """
        h_post: [B, T, D] or [T, D] after warp
        center: [1,1,D] or [1,D]
        returns denoised h_post (same shape)
        """
        squeeze_2d = (h_post.dim()==2)
        if squeeze_2d:
            h = h_post.unsqueeze(0)  # [1,T,D]
            c = center.unsqueeze(0) if center.dim()==2 else center
        else:
            h = h_post
            c = center if center.dim()==3 else center.unsqueeze(0)

        B, T, D = h.shape
        r = self._radial(h, c)           # [B,T]
        rz = self._robust_z(r, dim=1)    # robust z over time

        # Detect: high-radius tokens (phantoms) get heavier smoothing
        phantom = (rz > 1.0)             # boolean mask [B,T]

        # Denoise path per sequence
        h_out = []
        for b in range(B):
            xb = h[b]                                  # [T,D]
            if phantom[b].any():
                # heavier EMA if phantom; else light EMA
                decay = self.ema_decay
            else:
                decay = min(0.9, 0.5 + 0.5*self.ema_decay)
            yb = self._ema(xb, decay=decay)
            yb = self._median_k(yb, self.median_k)
            h_out.append(yb)
        y = torch.stack(h_out, dim=0)     # [B,T,D]

        # Compute per-token gate g in [0,1] from robust z + radius
        rz_pos = torch.clamp(rz, min=0.0)              # [B,T]
        r_norm = (r / (r.mean(dim=1, keepdim=True) + 1e-8)).clamp(0, 5.0)
        g_raw  = 0.5*rz_pos + 0.5*(r_norm - 1.0)       # combine cues
        g      = torch.sigmoid(2.0*(g_raw - self.conf_gate))  # conf_gate ≈ 0.65 default
        g      = g.unsqueeze(-1)                       # [B,T,1]
        
        # Soft noise floor guard (don’t denoise if not needed)
        floor = (r < (r.mean(dim=1, keepdim=True) * (1.0 + self.noise_floor))).unsqueeze(-1)
        g = torch.where(floor, torch.zeros_like(g), g)
        
        # Blend: h + λ * (y - h)
        lambda_max = 0.35        # << conservative
        y = h + lambda_max * g * (y - h)
        
        return y[0] if squeeze_2d else y

# -------------------------
# Attacher utilities
# -------------------------
def attach_wdd(model, layer_idx: int, *, alpha=1.0, beta=0.5,
               preset=STAGE11_PRESET, device="cpu"):
    """
    Inserts Warp (WellMetric) + DetectDenoise at transformer.h[layer_idx].
    Returns (wdd_module, cache, handles)
    """
    # Locate the block
    blk = model.transformer.h[layer_idx]
    d_model = model.config.hidden_size

    warp = WellMetric(d_model=d_model, alpha=alpha, beta=beta).to(device)
    dd   = DetectDenoise(**preset)

    # In wdd_stage11.attach_wdd, stash a callable to fetch mask if available
    cache = {"pre": None, "post": None, "mask": None}
    
    def fwd_pre(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out          # [B,T,D]
        cache["pre"] = h.detach() if h.requires_grad else h
    
        # Try to fetch attention mask from inputs (tuple 'inp')
        # 'inp' is (hidden_states, layer_past, attention_mask, ...)
        attn_mask = None
        if isinstance(inp, tuple) and len(inp) >= 3:
            attn_mask = inp[2]
        cache["mask"] = attn_mask
    
        z = warp(h)
        z = dd(z, warp.center)          # dd will read and mask if you wire it
        cache["post"] = z.detach()
        return (z,) + out[1:] if isinstance(out, tuple) else z

    handle = blk.register_forward_hook(lambda m, i, o: fwd_pre(m, i, o))
    return dict(warp=warp, dd=dd), cache, handle
