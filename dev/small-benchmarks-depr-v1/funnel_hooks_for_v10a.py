# funnel_hooks_for_v10a.py
# Adapts stage11-well-benchmark-latest-funnel.py to v10a's hook API.
import argparse, numpy as np
from typing import Dict, Tuple, Optional
# Import your funnel generator + parsers + metrics
from stage11_well_benchmark_latest_funnel import (
    make_synthetic_traces, geodesic_parse_report, geodesic_parse_with_prior,
    stock_parse, set_metrics, _rng
)

# Per-sample state so score_sample() can see the exact traces/answer used at init.
_LAST = {"traces": None, "true_order": None, "priors": None, "seed": None}

def _geodesic_eval(args):
    # choose parser w/ or w/o prior, mirroring the funnel script
    if getattr(args, "use_funnel_prior", 0) and _LAST["priors"] is not None:
        keep_g, order_g = geodesic_parse_with_prior(
            _LAST["traces"], _LAST["priors"], sigma=args.sigma, proto_width=args.proto_width,
            alpha=args.alpha, beta_s=args.beta_s, q_s=args.q_s,
            tau_rel=args.tau_rel, tau_abs_q=args.tau_abs_q, null_K=args.null_K, seed=_LAST["seed"]+1
        )
    else:
        keep_g, order_g = geodesic_parse_report(_LAST["traces"], sigma=args.sigma, proto_width=args.proto_width)
    keep_s, order_s = stock_parse(_LAST["traces"], sigma=args.sigma, proto_width=args.proto_width)
    return keep_g, order_g, keep_s, order_s

# v10a hook: optional latent initializer
def init_xy(dim: int = 64):
    # generate one funnel task using the *same knobs* as the funnel script
    seed = getattr(init_xy, "_seed", 42)
    rng = _rng(seed)
    traces, true_order = make_synthetic_traces(
        rng,
        T=getattr(init_xy, "T", 720),
        noise=getattr(init_xy, "noise", 0.02),
        cm_amp=getattr(init_xy, "cm_amp", 0.02),
        overlap=getattr(init_xy, "overlap", 0.5),
        amp_jitter=getattr(init_xy, "amp_jitter", 0.4),
        distractor_prob=getattr(init_xy, "distractor_prob", 0.4),
        tasks_k=(getattr(init_xy, "min_tasks", 1), getattr(init_xy, "max_tasks", 3)),
    )
    # stash for score_sample
    _LAST.update(dict(traces=traces, true_order=true_order, priors=None, seed=seed))
    # v10a expects (x0, x_star) latents; they won't affect scoring here
    x_star = np.zeros(dim, dtype=float)
    x0 = x_star + np.random.normal(0, 0.3, size=dim)
    init_xy._seed = seed + 1
    return x0, x_star

# v10a hook: propose the next step in latent space (not used for scoring; keep simple)
def propose_step(x_t: np.ndarray, x_star: np.ndarray, args: argparse.Namespace):
    direction = x_star - x_t
    dist = float(np.linalg.norm(direction) + 1e-9)
    unit = direction / (dist + 1e-9)
    # mimic funnel "smoothness" via proto_width/sigma scaling
    step_mag = min(1.0, 0.1 + 0.9 * np.tanh(dist / (getattr(args, "proto_width", 160) + 1e-9)))
    noise = np.random.normal(scale=getattr(args, "sigma", 9) * 1e-3, size=x_t.shape)
    dx = step_mag * unit + noise
    conf_rel = float(max(0.0, min(1.0, 1.0 - np.exp(-dist / (getattr(args, "proto_width", 160) + 1e-9)))))
    return dx, conf_rel, None

# v10a hook: local descent vector (unused by this shim; keep simple)
def descend_vector(p: np.ndarray, x_star: np.ndarray, args: argparse.Namespace):
    return x_star - p

# v10a hook: compute per-sample metrics — this is where we ensure identical difficulty
def score_sample(x_final: np.ndarray, x_star: np.ndarray) -> Dict[str, float]:
    keep_g, order_g, keep_s, order_s = _geodesic_eval(score_sample._args)
    true_order = _LAST["true_order"]
    sm_g = set_metrics(true_order, keep_g)
    sm_s = set_metrics(true_order, keep_s)
    # exact sequence accuracy
    acc_g = int(order_g == true_order)
    acc_s = int(order_s == true_order)
    # emit the same keys v10a aggregates
    out = {
        "accuracy_exact": float(acc_g),
        "precision": float(sm_g["precision"]),
        "recall": float(sm_g["recall"]),
        "f1": float(sm_g["f1"]),
        "jaccard": float(sm_g["jaccard"]),
        "hallucination_rate": float(sm_g["hallucination_rate"]),
        "omission_rate": float(sm_g["omission_rate"]),
        # (optional) mirror stock via prefixed keys if you want both printed from v10a
        "stock_accuracy_exact": float(acc_s),
        "stock_precision": float(sm_s["precision"]),
        "stock_recall": float(sm_s["recall"]),
        "stock_f1": float(sm_s["f1"]),
        "stock_jaccard": float(sm_s["jaccard"]),
        "stock_hallucination_rate": float(sm_s["hallucination_rate"]),
        "stock_omission_rate": float(sm_s["omission_rate"]),
    }
    return out

# v10a will call this if present to pass args so score_sample can access parser knobs
def configure_args(args: argparse.Namespace):
    score_sample._args = args
    # also share generator knobs with init_xy so they mirror the funnel defaults/flags
    for k in ("T","noise","cm_amp","overlap","amp_jitter","distractor_prob","min_tasks","max_tasks"):
        setattr(init_xy, k, getattr(args, k, getattr(init_xy, k, None)))
    init_xy._seed = getattr(args, "seed", 42)
