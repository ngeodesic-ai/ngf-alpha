# Small Benchmarks (Stage-11)

This folder contains Stage-11 benchmark scripts and results for the **Noetic Geodesic Framework (NGF)**.  
These “small benchmarks” provide reproducible experiments on compact datasets (HellaSwag, ARC-style prompts) to validate the **Warp → Detect → Denoise** doctrine.

## Contents

- **`ngf_benchmark.py`** — HellaSwag benchmark runner. Supports both stock GPT-2 and NGF-augmented runs.
- **`ngf_hooks_v2.py`** — Stage-11 Reno v2 hook (warp, detect, denoise, outlier guard).
- **`stage11_ab_eval_base_denoise.py`** — A/B/C evaluation framework with integrated denoiser.
- **`stage11_llm_layer_scan_plus.py`** — Layer scan tool for phantom index, margin, and trend metrics.
- **`plot_tap9_contour_well.py`** — PCA-2 visualization of “semantic wells” (pre vs post warp).
- **Results files** (`*.json`, `*.pdf`, `*.png`) — Example benchmark outputs and comparison reports.

## Usage

### 1. Stock Baseline
```bash
python3 ngf_benchmark.py   --mode stock --model gpt2 --split validation --n 1000 --max_length 768 --device auto   --out_json results/stock_gpt2_n1000.json
```

### 2. NGF Warp + Detect + Denoise
```bash
python3 ngf_benchmark.py   --mode ngf --ngf_import ngf_hooks_v2:attach_ngf_hooks   --model gpt2 --tap -9 --n 1000   --alpha0 0.06 --alpha_min 0.012 --trend_tau 0.30 --k_tr 12   --use_detect 1 --detect_width 22 --detect_sigma 4.5 --k_det 8   --s_latch 0.35 --linger 3 --ema_center_beta 0.04   --gen_mode geo --save_hidden 1   --hidden_dump_dir results/maxwarpC_tap9_noOutlier   --out_json results/maxwarpC_tap9_noOutlier/metrics.json
```

### 3. Visualization
```bash
python3 plot_tap9_contour_well.py   --pre results/maxwarpC_tap9_noOutlier/tap-9_pre.npy   --post results/maxwarpC_tap9_noOutlier/tap-9_post.npy   --out_png results/maxwarpC_tap9_noOutlier/tap9_well_compare.png   --out_pdf results/maxwarpC_tap9_noOutlier/tap9_well_compare.pdf   --fit_on post --sample 80000 --bins 220 --sigma 2.0 --levels 14
```

## Example Results

| Metric            | Stock GPT-2 | MaxWarpC (no outlier) | Δ |
|-------------------|-------------|------------------------|---|
| Top-1 Accuracy    | 0.324       | 0.355                 | +0.031 |
| Macro F1          | 0.323       | 0.355                 | +0.032 |
| ECE (Calibration) | 0.112       | 0.080                 | −0.032 |

_Source: [Stock_vs_MaxWarpC_Report.pdf](./Stock_vs_MaxWarpC_Report.pdf)_

**Interpretation:** Aggressive warp reshaping yields a +3 F1 point improvement, with better calibration and stability.

## References
- Stage-11 technical paper (work in progress): [article_latest.pdf](../docs/article_latest.pdf)
- Provisional patent appendices: [Appendix A (Math)](../docs/patent_appendix_a.pdf), [Appendix B (Benchmarks)](../docs/patent_appendix_b.pdf)
