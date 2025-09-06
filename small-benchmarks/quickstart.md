# Small Benchmarks (Stage-11)

This folder contains Stage-11 benchmark scripts and results for the **Noetic Geodesic Framework (NGF)**.  
These “small benchmarks” provide reproducible experiments on compact datasets (HellaSwag, ARC-style prompts) to validate the **Warp → Detect → Denoise** doctrine.

## Contents
- **[stage11_benchmark_latest.py](stage11_benchmark_latest.py)** — (Part A) Benchmark runner; latent-ARC simulations.
- **[ngf_benchmark.py](ngf_benchmark.py)** — (Part B) Benchmark runner; supports both stock GPT-2 and NGF-augmented runs.
- **[ngf_hooks_v2.py](ngf_hooks_v2.py)** — (Part B) Stage-11 Reno v2 hook (warp, detect, denoise, outlier guard).
- **[stage11_ab_eval.py](stage11_ab_eval.py)** — (Part B) Stage-11 A/B test evaluation framework.
- **[stage11_llm_layer_scan.py](stage11_llm_layer_scan.py)** — (Part B) Layer scan tool for phantom index, margin, and trend metrics.
- **[plot_contour_well.py](plot_contour_well.py)** — (Part B) PCA-2 visualization of “semantic wells” (pre vs post warp).
- **Results files** (`*.json`, `*.pdf`, `*.png`) — Example benchmark outputs and comparison reports.

## Usage

### 1. Stock Baseline
```bash
python3 ngf_benchmark.py \
  --mode stock --model gpt2 --split validation --n 1000 \
  --max_length 768 --device auto \
  --out_json results/stock/metrics.json
```

### 2. NGF Warp + Detect + Denoise
```bash

export NGF_RENO_CFG="use_denoise=1 denoise_mode=ema denoise_beta=0.26 \
denoise_ph_lambda=0.42 phantom_k=12 phantom_lambda=0.36 squeeze_orth_lambda=0.26 \
k_det=9 g_det_max=1.36 det_robust=mad winsor_q=0.985 \
alpha_min=0.038 alpha0=0.18 alpha_r_gamma=0.55 alpha_r_p=1.80 \
anneal_tokens=56 anneal_scale=1.95 outlier_q=1.0 outlier_alpha_scale=1.0 tap=-9"

python3 ngf_benchmark.py --mode ngf --ngf_import ngf_hooks_v2:attach_ngf_hooks \
  --model gpt2 --tap -9 --n 1000 \
  --alpha0 0.06 --alpha_min 0.012 --trend_tau 0.30 --k_tr 12 \
  --use_detect 1 --detect_width 22 --detect_sigma 4.5 --k_det 8 \
  --s_latch 0.35 --linger 3 --ema_center_beta=0.04 \
  --gen_mode geo --save_hidden 1 \
  --hidden_dump_dir results/maxwarp \
  --out_json results/maxwarp/metrics.json

```

### 3. PCA Visualization
```bash

OUT=results/stock

python3 plot_contour_well.py \
  --pre "$OUT/tap-9_pre.npy" \
  --post "$OUT/tap-9_post.npy" \
  --out_png "$OUT/tap9_well_compare.png" \
  --out_pdf "$OUT/tap9_well_compare.pdf" \
  --fit_on post \
  --sample 80000 \
  --bins 220 \
  --sigma 2.0 \
  --clip_q 0.01 \
  --levels 14

OUT=results/maxwarp

python3 plot_contour_well.py \
  --pre "$OUT/tap-9_pre.npy" \
  --post "$OUT/tap-9_post.npy" \
  --out_png "$OUT/tap9_well_compare.png" \
  --out_pdf "$OUT/tap9_well_compare.pdf" \
  --fit_on post \
  --sample 80000 \
  --bins 220 \
  --sigma 2.0 \
  --clip_q 0.01 \
  --levels 14

```

## Example Results

| Metric            | Stock GPT-2 | MaxWarpC (no outlier) | Δ |
|-------------------|-------------|------------------------|---|
| Top-1 Accuracy    | 0.324       | 0.355                 | +0.031 |
| Macro F1          | 0.323       | 0.355                 | +0.032 |
| ECE (Calibration) | 0.112       | 0.080                 | −0.032 |

_Source: [Stock_vs_MaxWarp Report.pdf](./benchmark_report.pdf)_

**Interpretation:** Aggressive warp reshaping yields a +3 F1 point improvement, with better calibration and stability.

## References
- Stage-11 math: [stage11_math.pdf](./stage11_math.pdf)
- Technical paper (work in progress): [article_latest.pdf](../docs/article_latest.pdf)
