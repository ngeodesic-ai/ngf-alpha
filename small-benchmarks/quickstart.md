# Small Benchmarks Quickstart (Stage-11)

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

## Basic Usage
Use this section for running stage 11 benchmarks

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

## Advanced Usage
Use this section for scoping new models (eg, GPT-large)

### 4. A/B test evaluation
Runs Stage-11 A/B evaluation tests on GPT-2 with NGF hooks. Supports multiple modes (Stock, Warp, Warp+Detect, Warp+Detect+Denoise) and logs convergence metrics, burst dynamics, and denoiser telemetry for prompt sets. Useful for side-by-side comparisons of baseline vs NGF-augmented reasoning.

```bash

# STOCK
python3 stage11_ab_eval.py \
  --model gpt2 --layer -9 \
  --prompts calib/wobble_prompts_v1.txt --max_new_tokens 64 \
  --gen_mode stock --device cuda \
  --out_json results/ab_stock_patterned.json

# GEO (Warp only)
python3 stage11_ab_eval.py \
  --model gpt2 --layer -9 \
  --prompts calib/wobble_prompts_v1.txt --max_new_tokens 64 \
  --alpha0 0.05 --alpha_min 0.006 \
  --trend_tau 0.35 --k_tr 12 \
  --s_latch 0.30 --linger 2 --ema_center_beta 0.05 \
  --gen_mode geo --device cuda \
  --out_json results/ab_geo_patterned.json

# GEO+Detect (Warp + Detect, no denoise)
python3 stage11_ab_eval.py \
  --model gpt2 --layer -9 \
  --prompts calib/wobble_prompts_v1.txt --max_new_tokens 64 \
  --alpha0 0.05 --alpha_min 0.006 \
  --trend_tau 0.35 --k_tr 12 \
  --use_detect 1 --detect_width 24 --detect_sigma 5 \
  --null_K 32 --null_q 0.92 --k_det 7 \
  --s_latch 0.30 --linger 2 --ema_center_beta 0.05 \
  --gen_mode geo --device cuda \
  --out_json results/ab_geo_detect_patterned.json
  
# GEO+Detect+Denoise 
python3 stage11_ab_eval.py \
  --model gpt2 --layer -9 \
  --prompts calib/wobble_prompts_v1.txt --max_new_tokens 64 \
  --alpha0 0.05 --alpha_min 0.006 \
  --trend_tau 0.35 --k_tr 12 \
  --use_detect 1 --detect_width 24 --detect_sigma 5 \
  --null_K 24 --null_q 0.88 --k_det 9 \
  --linger 4 --s_latch 0.25 --ema_center_beta 0.05 \
  --gen_mode geo --device cuda \
  --use_denoise 1 \
  --denoise_beta 0.6 --denoise_window 3 \
  --denoise_k 8.0 --denoise_tau 0.35 \
  --phantom_tr_tau 0.60 --phantom_guard_gamma 0.35 \
  --jitter_eps 0.03 \
  --out_json results/ab_geo_detect_denoise_patterned.json

```

### 5. LLM Layer Scan
Scans GPT-2 layers to measure warp, detect, and denoise effects. Computes phantom index, margin, and inward-trend metrics across hidden states, saving results as CSV/JSON and plotting layer-wise stability curves. Helps identify optimal tap layers for NGF integration.

```bash
for t in {-12..-6}; do
  for k in 8 12; do
    python3 stage11_llm_layer_scan.py \
      --model gpt2 --tap_range "$t" \
      --calib calib/calib_prompts_v2_900.txt --eval calib/calib_eval_style_200.txt \
      --pool_mode lastk --k_last $k \
      --sigma_px 5.0 --density_floor 4.0 --min_prom 0.55 \
      --with_detect --with_denoise \
      --out_csv logs/wdd_t${t}_k${k}.csv \
      --out_png logs/wdd_t${t}_k${k}.png \
      --out_json logs/wdd_t${t}_k${k}.json
  done
done

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
