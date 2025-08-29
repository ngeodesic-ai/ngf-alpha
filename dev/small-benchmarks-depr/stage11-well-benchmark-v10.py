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

Plug points (minimal surgery):
  1) Provide `propose_step()` for your model to compute dx_raw and conf_rel.
  2) Provide `descend_vector()` that returns a descent direction from pos -> x_star.
  3) (Optional) Provide `decode_logits()` per step if you want logit denoising.

All knobs are exposed as CLI flags to preserve baseline behavior when disabled.

Author: ngeodesic — stage 11 line.

python3 stage11-well-benchmark-v10.py \
  --baseline_path ./stage11-well-benchmark-latest-funnel.py \
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
import logging
import math
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# -------------------------------
# CLI
# -------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage11 Well Benchmark v10 (denoise + phantom guard)")

    # Baseline / data / evaluation knobs (kept from prior runs)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_funnel_prior", type=int, default=1)
    p.add_argument("--T", type=int, default=720, help="Steps per sample")
    p.add_argument("--sigma", type=float, default=9.0)
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

    # Outputs
    p.add_argument("--out_csv", type=str, default="_stage11_v10.csv")
    p.add_argument("--out_json", type=str, default="_stage11_v10.json")
    p.add_argument("--log", type=str, default="INFO")

    # ---------------- DENOISE & GUARDS ----------------
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

    # Latent size (only used if you run the demo loop w/o your model)
    p.add_argument("--latent_dim", type=int, default=64,
                   help="Latent dimension for the demo path.")

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
# Runner
# -------------------------------
class Runner:
    def __init__(self, args: argparse.Namespace, hooks: ModelHooks):
        self.args = args
        self.hooks = hooks
        self.logger = logging.getLogger("stage11.v10")

    def _init_latents(self, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (x0, x_star). Replace with actual dataset/targets as needed."""
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
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Seed global RNGs for reproducibility of the synthetic demo
    np.random.seed(args.seed)
    random.seed(args.seed)

    hooks = ModelHooks()  # Replace with wired hooks to your actual pipeline
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


if __name__ == "__main__":
    main()
