# Create a small helper that generates a dimension-aware NGF config.
# It prints either a shell `export NGF_RENO_CFG="..."` line or a JSON for programmatic use.
import math, json, textwrap
from pathlib import Path


"""
export NGF_RENO_CFG="tap=-9 \
alpha0=0.10 alpha_min=0.012 trend_tau=0.50 k_tr=12 \
use_detect=1 detect_width=32 detect_sigma=5 null_K=64 null_q=0.92 k_det=6 \
s_latch=0.40 linger=3 ema_center_beta=0.05 eps=0.25 \
use_denoise=1 denoise_beta=0.6 denoise_window=3 denoise_k=8.0 denoise_tau=0.35 \
phantom_tr_tau=0.60 phantom_guard_gamma=0.35 jitter_eps=0.03 \
center_mode=full pca_telemetry=1"
"""

#!/usr/bin/env python3
# reno_dimscale.py
# Dimension-aware parameter recommender for NGF/Stage-11 warp+detect+denoise.
#
# Usage examples:
#   python3 reno_dimscale.py --dim 768
#   python3 reno_dimscale.py --dim 256 --pca_rank 32 --apply_lowrank 1 --blend 0.5 --print json
#
import argparse, math, json, sys

def clamp(x, lo, hi): 
    return max(lo, min(hi, x))

def scale_params(dim: int, pca_rank: int=None, apply_lowrank: int=0, blend: float=0.0):
    # ----- Scale laws -----
    # Warp strength ~ 1/sqrt(d)
    c0, cmin = 0.30, 0.04
    alpha0   = round(c0 / math.sqrt(dim), 6)
    alpha_min= round(cmin / math.sqrt(dim), 6)

    # Trust region (relative step cap)
    eps = round(min(0.25, 1.5 / math.sqrt(dim)), 6)

    # EMA center responsiveness (increase with d, cap around 0.14)
    log2d = math.log2(max(dim, 9)/9.0)
    ema_center_beta = round(clamp(0.05 + 0.02*log2d, 0.05, 0.14), 6)

    # Trend gate
    trend_tau = round(clamp(0.50 - 0.03*log2d, 0.38, 0.50), 6)
    k_tr      = int(round(clamp(12 + 1.0*log2d, 12, 15)))

    # Detector smoothing / thresholding
    width = int(round(clamp(24 + 8*log2d, 32, 72)))
    sigma = int(max(3, round(width/6)))  # ~width/6-7
    null_K = int(round(clamp(32 * math.sqrt(dim/9.0), 32, 192)))
    null_q = 0.90
    k_det  = 5

    # Latch / linger
    s_latch = 0.50
    linger  = int(clamp(4 + log2d, 4, 6))

    # PCA control rank heuristic if requested
    if pca_rank is None:
        pca_rank = int(clamp(math.ceil(dim/12), 16, 64))

    cfg = {
        # warp
        "alpha0": alpha0, "alpha_min": alpha_min, "eps": eps,
        "trend_tau": trend_tau, "k_tr": k_tr,
        # detect
        "use_detect": 1, "detect_width": width, "detect_sigma": sigma,
        "null_K": null_K, "null_q": null_q, "k_det": k_det,
        # stickiness / center
        "s_latch": s_latch, "linger": linger, "ema_center_beta": ema_center_beta,
        # PCA control / application
        "pca_k": pca_rank, "whiten": 1,
        "apply_lowrank": int(apply_lowrank), "apply_blend_lambda": float(blend),
        # misc
        "tap": -9
    }
    return cfg

def to_export_line(cfg: dict) -> str:
    # Keep ordering a bit human-friendly
    keys = ["tap",
            "alpha0","alpha_min","eps",
            "trend_tau","k_tr",
            "use_detect","detect_width","detect_sigma","null_K","null_q","k_det",
            "s_latch","linger","ema_center_beta",
            "pca_k","whiten","apply_lowrank","apply_blend_lambda"]
    parts = []
    for k in keys:
        if k in cfg:
            v = cfg[k]
            parts.append(f"{k}={v}")
    return 'export NGF_RENO_CFG="' + " ".join(parts) + '"'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, required=True, help="latent / hidden dimension (e.g., 768)")
    ap.add_argument("--pca_rank", type=int, default=None, help="PCA control rank; default heuristic scales with dim")
    ap.add_argument("--apply_lowrank", type=int, default=0, help="Project Δh to PCA subspace (0/1)")
    ap.add_argument("--blend", type=float, default=0.0, help="Blend factor if apply_lowrank=1 (0..1)")
    ap.add_argument("--print", dest="fmt", choices=["export","json"], default="export")
    args = ap.parse_args()

    cfg = scale_params(args.dim, args.pca_rank, args.apply_lowrank, args.blend)
    if args.fmt == "json":
        print(json.dumps(cfg, indent=2))
    else:
        print(to_export_line(cfg))

if __name__ == "__main__":
    main()

