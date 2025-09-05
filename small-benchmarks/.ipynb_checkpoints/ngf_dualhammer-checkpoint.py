# ngf_dualhammer.py
from ngf_hooks_v2 import attach_ngf_hooks as attach_v2

def attach_dualhammer(model, tokenizer=None, **kwargs):
    """
    Primary: tap -9 (heavy) — uses env NGF_RENO_CFG as-is.
    Secondary: tap -6 (light) — tiny floor/ceiling; only engages on strong evidence.
    Returns both handles so state is kept separately.
    """
    # primary (your MaxWarp-C)
    h1 = attach_v2(model, tokenizer, **kwargs)

    # secondary (light assist)
    cfg2 = dict(kwargs)
    cfg2.update(dict(
        tap=-6,
        alpha_min=0.006,   # tiny always-on pull
        alpha0=0.045,      # small cap on bursts
        k_tr=12,
        trend_tau=0.32,    # a hair steadier so it won't flicker
        s_latch=0.25, linger=2,
        # detector: narrower + slightly higher cap, robust stats on
        use_detect=1, detect_width=20, detect_sigma=4.0, k_det=8,
        # keep denoise, but lighter
        ema_center_beta=0.05
    ))
    h2 = attach_v2(model, tokenizer, **cfg2)
    return [h1, h2]
