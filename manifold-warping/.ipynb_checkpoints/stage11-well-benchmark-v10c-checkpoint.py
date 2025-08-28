#!/usr/bin/env python3
"""
Stage 11 — Well Benchmark v10

Denoise-focused variant derived from the stage11 funnel baseline.
This script adds:
  • Temporal denoising (EMA / Median / Hybrid) for latent states and logits
  • Confidence-gated updates & noise floor fallback
  • Phantom-well guard via local field probes
  • Optional MC smoothing via seed jitter
  • SNR/residual instrumentation
  • Optional auto-wiring to your baseline funnel script
  • JSON comparator utility

Plug points (minimal surgery):
  1) Provide `propose_step()` for your model to compute dx_raw and conf_rel.
  2) Provide `descend_vector()` that returns a descent direction from pos -> x_star.
  3) (Optional) Provide `decode_logits()` per step if you want logit denoising.

All knobs are exposed as CLI flags to preserve baseline behavior when disabled.

Author: ngeodesic — stage 11 line.

python3 stage11-well-benchmark-v10c.py \
  --baseline_path ./stage11-well-benchmark-latest-funnel.py \
  --out_plot ./manifold_pca3_mesh_warped.png \
  --out_plot_fit ./manifold_pca3_mesh_warped_fit.png \
  --render_well --render_samples 1500 --render_grid 120 --render_quantile 0.8 \
  --render_out _well3d.png \
  --samples 200 --seed 42 --use_funnel_prior 1 --T 720 --sigma 9 \
  --proto_width 160 --cm_amp 0.02 --overlap 0.5 --amp_jitter 0.4 \
  --distractor_prob 0.4 --calib_samples 300 --alpha 0.03 --beta_s 0.15 \
  --q_s 2 --tau_rel 0.62 --tau_abs_q 0.92 --null_K 0 \
  --denoise_mode hybrid --ema_decay 0.85 --median_k 3 \
  --probe_k 5 --probe_eps 0.02 --conf_gate 0.65 --noise_floor 0.03 \
  --seed_jitter 2 \
  --out_csv _denoise_hybrid.csv --out_json _denoise_hybrid.json

python3 stage11-well-benchmark-v10c.py \
  --baseline_path ./stage11-well-benchmark-latest-funnel.py \
  --out_plot ./manifold_pca3_mesh_warped.png \
  --out_plot_fit ./manifold_pca3_mesh_warped_fit.png \
  --render_well --render_samples 1500 --render_grid 120 --render_quantile 0.8 \
  --render_out _well3d.png \
  --samples 200 --seed 42 --use_funnel_prior 1 --T 720 --sigma 9 \
  --proto_width 160 --cm_amp 0.02 --overlap 0.5 --amp_jitter 0.4 \
  --distractor_prob 0.4 --calib_samples 300 --alpha 0.03 --beta_s 0.15 \
  --q_s 2 --tau_rel 0.62 --tau_abs_q 0.92 --null_K 0 \
  --denoise_mode hybrid --ema_decay 0.85 --median_k 3 \
  --probe_k 5 --probe_eps 0.02 --conf_gate 0.65 --noise_floor 0.03 \
  --seed_jitter 2 \
  --out_csv _denoise_hybrid.csv --out_json _denoise_hybrid.json

"""
from __future__ import annotations
import argparse
import json
import logging as pylog
import math
import numpy as np
import os
import random
from collections import deque
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# For optional 3D rendering
try:
    import matplotlib
    matplotlib.use("Agg")  # headless safe
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception:
    plt = None

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None

# -------------------------------
# CLI
# -------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage11 Well Benchmark v10 (denoise + phantom guard)")

    # Baseline / data / evaluation knobs
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_funnel_prior", type=int, default=1)
    p.add_argument("--T", type=int, default=720, help="Steps per sample")
    p.add_argument("--sigma", type=int, default=9)
    p.add_argument("--proto_width", type=float, default=160.0)
    p.add_argument("--cm_amp", type=float, default=0.02)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--amp_jitter", type=float, default=0.4)
    p.add_argument("--distractor_prob", type=float, default=0.4)
    p.add_argument("--calib_samples", type=int, default=300)
    p.add_argument("--alpha", type=float, default=0.03)
    p.add_argument("--beta_s", type=float, default=0.15)
    p.add_argument("--q_s", type=float, default=2.0)
    p.add_argument("--tau_rel", type=float, default=0.60)
    p.add_argument("--tau_abs_q", type=float, default=0.92)
    p.add_argument("--null_K", type=int, default=0)

    # Outputs / logging
    p.add_argument("--out_csv", type=str, default="_stage11_v10.csv")
    p.add_argument("--out_json", type=str, default="_stage11_v10.json")
    p.add_argument("--log", type=str, default="INFO")

    # DENOISE & GUARDS
    p.add_argument("--denoise_mode", type=str, default="off",
                   choices=["off", "ema", "median", "hybrid"],
                   help="Denoising on latent/logits path.")
    p.add_argument("--ema_decay", type=float, default=0.85,
                   help="EMA decay for latent smoothing.")
    p.add_argument("--median_k", type=int, default=3,
                   help="Window (odd) for median filtering.")
    p.add_argument("--probe_k", type=int, default=3,
                   help="Phantom-well guard: number of local probes.")
    p.add_argument("--probe_eps", type=float, default=0.02,
                   help="Perturbation scale for probes (relative).")
    p.add_argument("--conf_gate", type=float, default=0.60,
                   help="Relative confidence gate (0-1).")
    p.add_argument("--noise_floor", type=float, default=0.05,
                   help="Minimum |dx|; below this, fallback to prior.")
    p.add_argument("--seed_jitter", type=int, default=0,
                   help=">0 enables MC smoothing over N jitters.")
    p.add_argument("--log_snr", type=int, default=1,
                   help="Log SNR / residual diagnostics each step.")

    # Latent size (for demo mode only)
    p.add_argument("--latent_dim", type=int, default=64,
                   help="Latent dimension for the demo path.")

    # Integration & Compare
    p.add_argument("--baseline_path", type=str, default="",
                   help="Path to stage11-well-benchmark-latest-funnel.py to auto-wire hooks.")
    p.add_argument("--compare", action="store_true",
                   help="If set, compare --compare_a vs --compare_b JSONs and exit.")
    p.add_argument("--compare_a", type=str, default="",
                   help="Path to baseline JSON (A).")
    p.add_argument("--compare_b", type=str, default="",
                   help="Path to experiment JSON (B).")

    # Rendering (3D PCA Well)
    p.add_argument("--render_well", action="store_true",
                   help="Render a 3D PCA cognition well surface and save to --render_out.")
    p.add_argument("--render_samples", type=int, default=1000,
                   help="Number of latent points to sample for PCA well fit.")
    p.add_argument("--render_grid", type=int, default=96,
                   help="Resolution of the polar grid for the surface (N x N).")
    p.add_argument("--render_quantile", type=float, default=0.8,
                   help="Radial height quantile used to fit bowl depth (0-1).")
    p.add_argument("--render_out", type=str, default="_well3d.png",
                   help="Output image path for the 3D well rendering.")

    # --- Baseline PCA/warp + plot outputs ---
    p.add_argument("--out_plot", type=str, default="manifold_pca3_mesh_warped.png",
                   help="Raw PCA(3) warped trisurf path (baseline-style).")
    p.add_argument("--out_plot_fit", type=str, default="manifold_pca3_mesh_warped_fit.png",
                   help="Funnel overlay trisurf path (baseline-style).")

    # (reuse baseline fit knobs so we can call its helpers)
    p.add_argument("--fit_quantile", type=float, default=0.65)
    p.add_argument("--rbf_bw", type=float, default=0.30)
    p.add_argument("--core_k", type=float, default=0.18)
    p.add_argument("--core_p", type=float, default=1.7)
    p.add_argument("--core_r0_frac", type=float, default=0.14)
    p.add_argument("--blend_core", type=float, default=0.25)
    p.add_argument("--template_D", type=float, default=1.2)
    p.add_argument("--template_p", type=float, default=1.6)
    # --- Fitted-funnel mesh resolution (for overlay) ---
    p.add_argument("--n_theta", type=int, default=160,
                   help="Angular samples for funnel surface.")
    p.add_argument("--n_r", type=int, default=220,
                   help="Radial samples for funnel surface.")

    # --- Data synth knobs baseline expects for H/E builder ---
    p.add_argument("--noise", type=float, default=0.02,
                   help="Noise level for synthetic traces used for PCA viz.")
    p.add_argument("--min_tasks", type=int, default=1,
                   help="Min tasks per sample for PCA viz.")
    p.add_argument("--max_tasks", type=int, default=3,
                   help="Max tasks per sample for PCA viz.")

    # --- Funnel warp params for PCA->well warp (WellParams) ---
    p.add_argument("--sigma_scale", type=float, default=0.80,
                   help="Radial width scale for funnel warp (matches baseline).")
    p.add_argument("--depth_scale", type=float, default=1.35,
                   help="Depth multiplier for funnel warp (matches baseline).")
    p.add_argument("--mix_z", type=float, default=0.12,
                   help="How much to mix original PC3 height into warped z (matches baseline).")

    p.add_argument("--use_baseline_arc", type=int, default=1, choices=[0,1],
                   help="If 1, run metrics on the same ARC-like tasks/parsers as -latest-funnel.")

    # in build args (near outputs)
    p.add_argument("--in_truth", type=str, default="",
                   help="JSONL of {sample,true,traces} from the baseline; if set, v10c replays tasks instead of generating.")



    return p



# -------------------------------
# Utils: Denoiser / SNR / Guard
# -------------------------------
class TemporalDenoiser:
    def __init__(self, mode: str = "off", ema_decay: float = 0.85, median_k: int = 3):
        self.mode = mode
        self.ema_decay = ema_decay
        self.med_k = max(1, median_k | 1)  # ensure odd
        self._ema = None
        self._buf = deque(maxlen=self.med_k)

    def reset(self):
        self._ema = None
        self._buf.clear()

    def latent(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "off":
            return x
        x_ema = x
        if self.mode in ("ema", "hybrid"):
            self._ema = x if self._ema is None else self.ema_decay * self._ema + (1.0 - self.ema_decay) * x
            x_ema = self._ema
        if self.mode in ("median", "hybrid"):
            self._buf.append(np.copy(x_ema))
            arr = np.stack(list(self._buf), axis=0)
            return np.median(arr, axis=0)
        return x_ema

    def logits(self, logits_vec: np.ndarray) -> np.ndarray:
        if self.mode == "off":
            return logits_vec
        self._buf.append(np.copy(logits_vec))
        if self.mode == "ema":
            self._ema = logits_vec if self._ema is None else self.ema_decay * self._ema + (1.0 - self.ema_decay) * logits_vec
            return self._ema
        # median or hybrid for logits
        arr = np.stack(list(self._buf), axis=0)
        return np.median(arr, axis=0)


def snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    s = float(np.linalg.norm(signal) + 1e-9)
    n = float(np.linalg.norm(noise) + 1e-9)
    ratio = max(s / n, 1e-9)
    return 20.0 * math.log10(ratio)


def phantom_guard(step_vec: np.ndarray,
                  pos: np.ndarray,
                  descend_fn: Callable[[np.ndarray], np.ndarray],
                  k: int = 3,
                  eps: float = 0.02) -> bool:
    """Probe local field around pos; True if majority agrees with step direction."""
    if k <= 1:
        return True
    denom = float(np.linalg.norm(step_vec) + 1e-9)
    step_dir = step_vec / denom
    agree = 0
    base_scale = float(np.linalg.norm(pos) + 1e-9)
    for _ in range(k):
        delta = np.random.randn(*pos.shape) * eps * base_scale
        probe_step = descend_fn(pos + delta)
        if np.dot(step_dir, probe_step) > 0:
            agree += 1
    return agree >= (k // 2 + 1)


# -------------------------------
# Demo model hooks (replace with your real ones)
# -------------------------------
@dataclass
class ModelHooks:
    # Provide your own implementations to integrate the real pipeline.
    def propose_step(self, x_t: np.ndarray, x_star: np.ndarray, args: argparse.Namespace
                     ) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
        """Return (dx_raw, conf_rel, logits_optional).
        Placeholder: noisy pull toward x_star with confidence ~ cosine.
        """
        direction = x_star - x_t
        dist = float(np.linalg.norm(direction) + 1e-9)
        unit = direction / (dist + 1e-9)
        # Simulate a raw step with noise controlled by sigma
        step_mag = min(1.0, 0.1 + 0.9 * math.tanh(dist / (args.proto_width + 1e-9)))
        noise = np.random.normal(scale=args.sigma * 1e-3, size=x_t.shape)
        dx_raw = step_mag * unit + noise
        # Simulate confidence
        conf_rel = float(max(0.0, min(1.0, 1.0 - math.exp(-dist / (args.proto_width + 1e-9)))))
        logits = None  # supply if you decode per step
        return dx_raw, conf_rel, logits

    def descend_vector(self, p: np.ndarray, x_star: np.ndarray, args: argparse.Namespace) -> np.ndarray:
        """Return a descent vector at position p toward the proto-well x_star.
        Replace with your funnel/prior gradient.
        """
        return (x_star - p)

    def score_sample(self, x_final: np.ndarray, x_star: np.ndarray) -> Dict[str, float]:
        """Compute per-sample metrics. Replace with your real evaluation.
        Here we emit toy metrics consistent with our reporting format.
        """
        err = float(np.linalg.norm(x_final - x_star))
        # Map error to pseudo-metrics
        accuracy_exact = 1.0 if err < 0.05 else 0.0
        hallucination_rate = max(0.0, min(1.0, err)) * 0.2
        omission_rate = max(0.0, min(1.0, err)) * 0.1
        precision = max(0.0, 1.0 - 0.5 * hallucination_rate)
        recall = max(0.0, 1.0 - 0.5 * omission_rate)
        f1 = (2 * precision * recall) / (precision + recall + 1e-9)
        jaccard = f1 / (2 - f1 + 1e-9)
        return {
            "accuracy_exact": accuracy_exact,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "jaccard": jaccard,
            "hallucination_rate": hallucination_rate,
            "omission_rate": omission_rate,
        }


# -------------------------------
# Optional: Baseline adapter (auto-wire to your funnel script)
# -------------------------------
import importlib.util as _ilspec

class BaselineAdapter(ModelHooks):
    def __init__(self, path: str, logger: logging.Logger):
        self.logger = logger
        self.mod = self._load_module(path)
        # Heuristic discovery of hooks
        self._propose = self._find_callable([
            "propose_step", "step_proposal", "propose", "compute_step"
        ])
        self._descend = self._find_callable([
            "descend_vector", "prior_descent", "funnel_descent", "descent"
        ])
        self._score = self._find_callable([
            "score_sample", "score", "evaluate_sample", "compute_metrics"
        ])
        self._init_xy = self._find_callable([
            "init_latents", "init_state", "make_latents", "sample_latents"
        ])
        if not self._propose:
            self.logger.warning("Baseline adapter: no step proposal found; using placeholder.")
        if not self._descend:
            self.logger.warning("Baseline adapter: no descent found; using placeholder.")
        if not self._score:
            self.logger.warning("Baseline adapter: no scorer found; using placeholder.")
        if not self._init_xy:
            self.logger.info("Baseline adapter: no init_latents; using Runner default.")

    def _load_module(self, path: str):
        spec = _ilspec.spec_from_file_location("stage11_baseline", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import baseline from {path}")
        mod = _ilspec.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.logger.info(f"Loaded baseline module from {path}")
        return mod

    def _find_callable(self, names):
        for n in names:
            fn = getattr(self.mod, n, None)
            if callable(fn):
                return fn
        # Also search for a class with method name
        for attr in dir(self.mod):
            obj = getattr(self.mod, attr)
            if hasattr(obj, "__dict__") and hasattr(obj, "__call__") is False:
                for n in names:
                    cand = getattr(obj, n, None)
                    if callable(cand):
                        return cand
        return None

    # --- Hooks (fallback to parent if missing) ---
    def propose_step(self, x_t: np.ndarray, x_star: np.ndarray, args: argparse.Namespace):
        if self._propose:
            out = self._propose(x_t, x_star, args) if getattr(self._propose, "__code__", None) and self._propose.__code__.co_argcount >= 3 else self._propose(x_t, x_star)
            # Normalize returns to (dx_raw, conf_rel, logits|None)
            if isinstance(out, tuple) and len(out) == 3:
                return out
            if isinstance(out, tuple) and len(out) == 2:
                dx, conf = out
                return dx, float(conf), None
            # assume only dx returned
            return out, 1.0, None
        return super().propose_step(x_t, x_star, args)

    def descend_vector(self, p: np.ndarray, x_star: np.ndarray, args: argparse.Namespace) -> np.ndarray:
        if self._descend:
            try:
                return self._descend(p, x_star, args)
            except TypeError:
                try:
                    return self._descend(p, x_star)
                except TypeError:
                    return self._descend(p)
        return super().descend_vector(p, x_star, args)

    def score_sample(self, x_final: np.ndarray, x_star: np.ndarray) -> Dict[str, float]:
        if self._score:
            try:
                return dict(self._score(x_final, x_star))
            except TypeError:
                return super().score_sample(x_final, x_star)
        return super().score_sample(x_final, x_star)

    def maybe_init_latents(self, dim: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._init_xy:
            try:
                return self._init_xy(dim)
            except TypeError:
                try:
                    return self._init_xy()
                except Exception:
                    return None
        return None

# Patch Runner to use baseline init if available

# -------------------------------
# Runner
# -------------------------------
class Runner:
    def __init__(self, args: argparse.Namespace, hooks: ModelHooks):
        self.args = args
        self.hooks = hooks
        self.logger = pylog.getLogger("stage11.v10")

    def _init_latents(self, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (x0, x_star). Replace with actual dataset/targets as needed."""
        if isinstance(self.hooks, BaselineAdapter):
            pair = self.hooks.maybe_init_latents(dim)
            if pair is not None:
                return pair
        x_star = np.random.uniform(-1.0, 1.0, size=(dim,))
        x0 = x_star + np.random.normal(scale=0.5, size=(dim,))
        return x0, x_star

    def run_sample(self, idx: int) -> Dict[str, float]:
        np.random.seed(self.args.seed + idx)
        random.seed(self.args.seed + idx)

        x_t, x_star = self._init_latents(self.args.latent_dim)
        den = TemporalDenoiser(self.args.denoise_mode, self.args.ema_decay, self.args.median_k)
        den.reset()

        for t in range(self.args.T):
            dx_raw, conf_rel, logits = self.hooks.propose_step(x_t, x_star, self.args)
            residual = x_star - x_t
            dx = dx_raw

            # Instrumentation
            if self.args.log_snr:
                snr = snr_db(signal=residual, noise=dx - residual)
                self.logger.info(f"[i={idx} t={t}] SNR(dB)={snr:.2f} |res|={np.linalg.norm(residual):.4f} |dx|={np.linalg.norm(dx):.4f} conf={conf_rel:.3f}")

            # Gate & noise floor
            if conf_rel < self.args.conf_gate or np.linalg.norm(dx) < self.args.noise_floor:
                dx = 0.5 * residual  # fallback toward proto-well

            # Phantom guard
            def _desc(p: np.ndarray) -> np.ndarray:
                return self.hooks.descend_vector(p, x_star, self.args)
            if not phantom_guard(dx, x_t, _desc, k=self.args.probe_k, eps=self.args.probe_eps):
                dx = 0.3 * residual

            # Commit tentative next
            x_next = x_t + dx

            # Latent denoise
            x_next = den.latent(x_next)

            # Optional logits denoise (if provided)
            if logits is not None:
                _ = den.logits(logits)

            # Optional MC smoothing
            if self.args.seed_jitter > 0:
                xs = [x_next]
                for _ in range(self.args.seed_jitter):
                    jitter = np.random.normal(scale=0.01, size=x_next.shape)
                    xs.append(den.latent(x_t + dx + jitter))
                x_next = np.mean(xs, axis=0)

            x_t = x_next

        return self.hooks.score_sample(x_t, x_star)

    def run(self) -> Dict[str, float]:
        metrics_list: List[Dict[str, float]] = []
        for i in range(self.args.samples):
            m = self.run_sample(i)
            metrics_list.append(m)

        # Aggregate
        agg: Dict[str, float] = {}
        keys = metrics_list[0].keys() if metrics_list else []
        for k in keys:
            agg[k] = float(np.mean([m[k] for m in metrics_list]))

        # Compatibility with prior summaries
        self.logger.info("[SUMMARY] Geodesic (v10 denoise): " + json.dumps(agg, sort_keys=True))
        return agg


# -------------------------------
# Comparator utility
# -------------------------------

def compare_json(a_path: str, b_path: str) -> str:
    def _load(p):
        with open(p, "r") as f:
            return json.load(f)
    A = _load(a_path)
    B = _load(b_path)
    keys = sorted(set(A.keys()) | set(B.keys()))
    lines = ["metric, A, B, delta(B-A)"]
    for k in keys:
        try:
            av = float(A.get(k, float('nan')))
        except Exception:
            av = float('nan')
        try:
            bv = float(B.get(k, float('nan')))
        except Exception:
            bv = float('nan')
        dv = bv - av
        lines.append(f"{k},{av:.6f},{bv:.6f},{dv:+.6f}")
    return "\n".join(lines)


# -------------------------------
# Rendering: 3D PCA Cognition Well
# -------------------------------

def render_pca_well(args: argparse.Namespace, hooks: ModelHooks, logger: logging.Logger):
    if plt is None or PCA is None:
        warnings.warn("Matplotlib or scikit-learn not available; cannot render 3D well.")
        return

    rng = np.random.default_rng(args.seed)

    # 1) Collect a latent cloud around wells using baseline init + small descent steps
    dim = args.latent_dim
    X = []
    for i in range(args.render_samples):
        x0, x_star = Runner._init_latents(self=None, dim=dim) if not isinstance(hooks, BaselineAdapter) else (hooks.maybe_init_latents(dim) or Runner._init_latents(self=None, dim=dim))
        # take a couple of guided steps toward the well to populate the basin
        x = x0
        for _ in range(5):
            dx, conf, _ = hooks.propose_step(x, x_star, args)
            x = x + dx
        X.append(x)
    X = np.stack(X, axis=0)

    # 2) PCA to 3D
    pca = PCA(n_components=3, whiten=True, random_state=args.seed)
    Y = pca.fit_transform(X)

    # 3) Fit a radial bowl profile by quantile over radius bins
    r = np.linalg.norm(Y[:, :2], axis=1)
    z = Y[:, 2]
    nbins = max(16, args.render_grid // 6)
    bins = np.linspace(r.min(), r.max(), nbins + 1)
    r_centers = 0.5 * (bins[:-1] + bins[1:])
    zq = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (r >= b0) & (r < b1)
        if np.any(mask):
            zq.append(np.quantile(z[mask], args.render_quantile))
        else:
            zq.append(np.nan)
    zq = np.array(zq)
    # simple inpainting of nans by nearest valid
    if np.any(np.isnan(zq)):
        valid = ~np.isnan(zq)
        zq[~valid] = np.interp(r_centers[~valid], r_centers[valid], zq[valid])

    # 4) Create a smooth polar surface from the fitted profile
    N = args.render_grid
    rmax = float(r_centers.max())
    rr = np.linspace(0, rmax, N)
    th = np.linspace(0, 2*np.pi, N)
    R, TH = np.meshgrid(rr, th)
    # interpolate radial profile onto rr
    z_prof = np.interp(rr, r_centers, zq)
    Z = np.tile(z_prof, (N,1))  # TH dimension x R dimension
    Xs = R * np.cos(TH)
    Ys = R * np.sin(TH)

    # 5) Plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xs, Ys, Z, linewidth=0, antialiased=True, alpha=0.8)
    # scatter a small subset of projected points for visual sanity
    idx = rng.choice(len(Y), size=min(500, len(Y)), replace=False)
    ax.scatter(Y[idx,0], Y[idx,1], Y[idx,2], s=6, alpha=0.3)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3 / depth')
    ax.set_title('Stage-11 Cognition Well (3D PCA surface)')
    fig.tight_layout()
    fig.savefig(args.render_out, dpi=200)
    plt.close(fig)
    logger.info(f"[RENDER] Saved 3D PCA well to {args.render_out}")

def _import_baseline_module(baseline_path: str):
    import importlib.util, os
    if not baseline_path or not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")
    spec = importlib.util.spec_from_file_location("s11funnel", baseline_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _sanitize_for_baseline(args):
    # enforce ints for any window/length knobs the baseline expects
    for name in ("sigma", "proto_width", "n_theta", "n_r", "samples", "T"):
        if hasattr(args, name):
            setattr(args, name, int(round(getattr(args, name))))
    # guardrails
    args.sigma = max(1, args.sigma)
    args.proto_width = max(3, args.proto_width)

def _ensure_baseline_defaults_and_types(args):
    # Fill any missing attributes baseline needs (defensive for older checkpoints)
    for k, v in dict(
        noise=0.02, min_tasks=1, max_tasks=3,
        sigma_scale=0.80, depth_scale=1.35, mix_z=0.12,
        n_theta=160, n_r=220, proto_width=160
    ).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    # Coerce widths/counts to int so np.pad etc. don't choke
    for name in ("samples", "T", "sigma", "proto_width", "n_theta", "n_r"):
        if hasattr(args, name):
            setattr(args, name, int(getattr(args, name)))

def _truth_reader(path):
    import json
    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line)
            # convert lists back to np arrays
            rec["traces"] = {k: np.asarray(v, float) for k, v in rec["traces"].items()}
            yield rec

def save_manifold_plots_with_baseline(args, logger):
    import importlib.util, numpy as np, matplotlib.pyplot as plt
    if not getattr(args, "baseline_path", None):
        logger.warning("No --baseline_path provided; skipping manifold plots.")
        return

    _ensure_baseline_defaults_and_types(args)

    # Import baseline module
    spec = importlib.util.spec_from_file_location("s11funnel", args.baseline_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    # Build feature cloud + energy using baseline synth (needs noise/cm_amp/overlap/...)
    H, E = mod.build_H_E_from_traces(args)  # uses args.noise, cm_amp, overlap, amp_jitter, distractor_prob, min/max_tasks, sigma
    # Warp into single well using baseline's WellParams + PCA(3)
    params = mod.WellParams(sigma_scale=args.sigma_scale, depth_scale=args.depth_scale, mix_z=args.mix_z)
    X3_warp, metrics, info = mod.pca3_and_warp(H, energy=E, params=params)

    # Raw trisurf -> args.out_plot
    fig, ax = mod.plot_trisurf(X3_warp, energy=E, title="Stage 11 — Warped Single Well (v10)")
    fig.savefig(args.out_plot, dpi=220); plt.close(fig)
    logger.info(f"[PLOT] Saved {args.out_plot}")

    # Fitted funnel overlay -> args.out_plot_fit
    r_cloud = np.linalg.norm((X3_warp[:, :2] - info["center"]), axis=1)
    r_max = float(np.quantile(r_cloud, 0.98))
    r_grid = np.linspace(0.0, r_max, args.n_r)

    z_data = mod.fit_radial_profile(
        X3_warp, info["center"], r_grid,
        h=max(1e-6, 0.30 * r_max), q=0.65,
        r0_frac=0.14, core_k=0.18, core_p=1.7
    )
    z_tmpl = mod.analytic_core_template(r_grid, D=1.2, p=1.6, r0_frac=0.14)
    z_prof = mod.blend_profiles(z_data, z_tmpl, 0.25)

    Xs, Ys, Zs = mod.build_polar_surface(info["center"], r_grid, z_prof, n_theta=args.n_theta)
    fig = plt.figure(figsize=(10, 8)); ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Xs, Ys, Zs, cmap="viridis", alpha=0.9, linewidth=0, antialiased=True)
    ax.scatter(X3_warp[:,0], X3_warp[:,1], X3_warp[:,2], s=10, alpha=0.7, c=(E - E.min())/(E.ptp()+1e-9), cmap="viridis")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
    ax.set_title("Stage 11 — Data-fit Funnel (v10 overlay)")
    fig.colorbar(surf, ax=ax, shrink=0.6, label="height")
    plt.tight_layout(); fig.savefig(args.out_plot_fit, dpi=220); plt.close(fig)
    logger.info(f"[PLOT] Saved {args.out_plot_fit}")



# -------------------------------
# I/O helpers
# -------------------------------

def save_csv(path: str, metrics: Dict[str, float]):
    import csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, f"{v:.6f}"])


def save_json(path: str, metrics: Dict[str, float]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


# -------------------------------
# Main
# -------------------------------

def main():
    args = build_argparser().parse_args()
    _sanitize_for_baseline(args)
    rng = np.random.default_rng(args.seed)  

    # near start of main(), after args parsed:
    truth_iter = _truth_reader(args.in_truth) if args.in_truth else None

    lvl = getattr(pylog, getattr(args, "log", "INFO").upper(), pylog.INFO)
    pylog.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger = pylog.getLogger("stage11.v10c")

    # Quick compare mode
    if args.compare:
        if not (args.compare_a and args.compare_b):
            raise SystemExit("--compare requires --compare_a and --compare_b")
        print(compare_json(args.compare_a, args.compare_b))
        return

    # Optional: 3D well rendering (can be run standalone)
    if args.render_well:
        # Use hooks as below to leverage baseline init/proposal if available
        if args.baseline_path:
            hooks = BaselineAdapter(args.baseline_path, pylog.getLogger("stage11.adapter"))
        else:
            hooks = ModelHooks()
        try:
            render_pca_well(args, hooks, logging.getLogger("stage11.render"))
        except Exception as e:
            warnings.warn(f"Render failed: {e}")
        # Continue to full run after rendering (so it can do both), or return if you prefer
        # return

    # Seed global RNGs for reproducibility of the synthetic demo
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Choose hooks: baseline adapter if provided, else defaults
    if args.baseline_path:
        hooks = BaselineAdapter(args.baseline_path, pylog.getLogger("stage11.adapter"))
    else:
        hooks = ModelHooks()

    runner = Runner(args, hooks)
    metrics = runner.run()

    if args.out_csv:
        save_csv(args.out_csv, metrics)
    if args.out_json:
        save_json(args.out_json, metrics)

    print("[DONE] Wrote:")
    if args.out_csv:
        print(f"  - {args.out_csv}")
    if args.out_json:
        print(f"  - {args.out_json}")

    # If requested, run metrics using the baseline ARC-like tasks/parsers (apples-to-apples)
    if args.use_baseline_arc:
        BL = _import_baseline_module(args.baseline_path)

        sigma = int(round(args.sigma))
        proto_width = int(round(args.proto_width))
        if sigma < 1: sigma = 1
        if proto_width < 3: proto_width = 3  # minimal useful prototype length
    
        rows = []
        agg_geo = dict(acc=0, P=0, R=0, F1=0, J=0, H=0, O=0)
        agg_stock = dict(acc=0, P=0, R=0, F1=0, J=0, H=0, O=0)
    
        for i in range(1, args.samples + 1):
            # --- same generator the baseline uses ---
            if truth_iter:
                rec = next(truth_iter)              # raises StopIteration if mismatch
                traces = rec["traces"]
                true_order = rec["true"]
            else:
                traces, true_order = BL.make_synthetic_traces(
                    rng, T=args.T, noise=args.noise, cm_amp=args.cm_amp, overlap=args.overlap,
                    amp_jitter=args.amp_jitter, distractor_prob=args.distractor_prob,
                    tasks_k=(args.min_tasks, args.max_tasks)
                )
    
            # --- same parsers as the baseline's default report path ---
            keep_g, order_g = BL.geodesic_parse_report(traces, sigma=args.sigma, proto_width=args.proto_width)  # 
            keep_s, order_s = BL.stock_parse(traces, sigma=args.sigma, proto_width=args.proto_width)            # 
    
            # exact sequence accuracy (not just set-membership)
            acc_g = int(order_g == true_order)
            acc_s = int(order_s == true_order)
    
            sm_g = BL.set_metrics(true_order, keep_g)
            sm_s = BL.set_metrics(true_order, keep_s)  # 
    
            # accumulate like the baseline
            for k, v in sm_g.items():
                key = {"precision":"P","recall":"R","f1":"F1","jaccard":"J","hallucination_rate":"H","omission_rate":"O"}[k]
                agg_geo[key] = agg_geo.get(key, 0) + v
            for k, v in sm_s.items():
                key = {"precision":"P","recall":"R","f1":"F1","jaccard":"J","hallucination_rate":"H","omission_rate":"O"}[k]
                agg_stock[key] = agg_stock.get(key, 0) + v
            agg_geo["acc"] += acc_g
            agg_stock["acc"] += acc_s
    
            rows.append(dict(
                sample=i,
                true="|".join(true_order),
                geodesic_tasks="|".join(keep_g), geodesic_order="|".join(order_g), geodesic_ok=acc_g,
                stock_tasks="|".join(keep_s), stock_order="|".join(order_s), stock_ok=acc_s,
                geodesic_precision=sm_g["precision"], geodesic_recall=sm_g["recall"], geodesic_f1=sm_g["f1"],
                geodesic_jaccard=sm_g["jaccard"], geodesic_hallucination=sm_g["hallucination_rate"], geodesic_omission=sm_g["omission_rate"],
                stock_precision=sm_s["precision"], stock_recall=sm_s["recall"], stock_f1=sm_s["f1"],
                stock_jaccard=sm_s["jaccard"], stock_hallucination=sm_s["hallucination_rate"], stock_omission=sm_s["omission_rate"],
            ))

    
        # finalize summary exactly like baseline (means over samples)
        n = float(args.samples)
        Sg = dict(
            accuracy_exact = agg_geo["acc"]/n, precision=agg_geo["P"]/n, recall=agg_geo["R"]/n, f1=agg_geo["F1"]/n,
            jaccard=agg_geo["J"]/n, hallucination_rate=agg_geo["H"]/n, omission_rate=agg_geo["O"]/n
        )
        Ss = dict(
            accuracy_exact = agg_stock["acc"]/n, precision=agg_stock["P"]/n, recall=agg_stock["R"]/n, f1=agg_stock["F1"]/n,
            jaccard=agg_stock["J"]/n, hallucination_rate=agg_stock["H"]/n, omission_rate=agg_stock["O"]/n
        )
        print("[SUMMARY] Geodesic:", {k: round(v,3) for k,v in Sg.items()})
        print("[SUMMARY] Stock   :", {k: round(v,3) for k,v in Ss.items()})
    
        # (Optional) write your CSV/JSON here if you keep that in v10c



    try:
        import logging
        save_manifold_plots_with_baseline(args, logging.getLogger("stage11.v10a.plots"))
    except Exception as e:
        import logging
        logging.getLogger("stage11.v10a.plots").warning(f"Plot generation failed: {e}")


if __name__ == "__main__":
    main()
