# Stage‑11 Freeze Report — v4b · tap‑9 · Steps 1–10

**Date:** 2025-09-03 21:34  
**Owner:** You  
**Runner:** `text_arc_unified.py`  
**Profile:** `calib/profile_v4b_tap9_text.json` (aka **tweakA**)

---

## 0) Executive Summary
- We are freezing **v4b · tap‑9** as the benchmarkable configuration for **Steps 1–10** (Geo vs Stock).
- The runner is unified (stock/geo), reproducible (`--config` / `--save_config`), and instrumented (telemetry + metrics).
- Quality proxies show **lower repetition/loopiness** and **healthy 2–3‑token bursts** for Geo vs Stock at similar lengths.
- Step‑7 ablation confirms the **soft denoiser** improves stability without truncation; Step‑9 perf sweeps (fp16/bf16) preserve quality and raise throughput.
- Optional refinements (true trend @ tap; tap‑true PCA) are deferred to a **v4c** pass **only if** A/B shows material headroom.

---

## 1) Frozen Doctrine & Key Knobs (v4b · tap‑9)
- **Warp (always‑on):** `alpha0=0.05`, `alpha_min=0.006`, tap=`-9`
- **Trend gate:** `trend_tau=0.32`, `k_tr=12`, `linger=4`, `s_latch=0.30`
- **Detect (gain‑only):** `detect_width=20`, `null_K=32`, `null_q=0.92`, `k_det=8.0`
- **Soft denoiser (sign‑safe):** `denoise_beta=0.6`, `denoise_window=5`, `denoise_k=6.0`, `denoise_tau=0.40`, `phantom_tr_tau=0.65`, `phantom_guard_gamma=0.45`
- **Telemetry (geo runs):** per‑token `alpha, s, g_tr, g_det, radius, step_norm`
- **Ops:** profiles via `--config`, exact run capture via `--save_config`, metrics via `--metrics_json/--metrics_csv` (includes `elapsed_sec`, `tokens_per_sec`).

---

## 2) Commands (copy/paste)

### GEO (frozen profile)
```bash
python3 /mnt/data/text_arc_unified.py   --config /mnt/data/calib/profile_v4b_tap9_text.json   --prompts /mnt/data/calib/ngf_eval_prompts_60.txt   --metrics_json /mnt/data/metrics_geo.v4b.tap9.json   --metrics_csv  /mnt/data/metrics_geo.v4b.tap9.csv   --out /mnt/data/generations_geo_steps.v4b.tap9.jsonl   --save_config /mnt/data/run_effective_config.v4b.tap9.json
```

### STOCK (baseline, same prompts)
```bash
python3 /mnt/data/text_arc_unified.py   --gen_mode stock   --prompts /mnt/data/calib/ngf_eval_prompts_60.txt   --metrics_json /mnt/data/metrics_stock.v4b.tap9.json   --out /mnt/data/generations_stock.v4b.tap9.jsonl
```

### Optional: Telemetry & Perf
```bash
# Telemetry
--telemetry_jsonl /mnt/data/geo_steps1_6.v4b.tap9.telemetry.jsonl

# Perf sweep examples
--perf_profile gpu_L4 --dtype bfloat16 --compile 1
--perf_profile gpu_T4 --dtype float16  --compile 0
```

---

## 3) Evidence Snapshot (Steps 7–10 Tests)
- **Geo vs Stock:** lower repetitions/loopiness with comparable lengths → expected small uplift on accuracy/F1 once truths are applied.
- **Denoiser ablation (OFF vs ON):** OFF increases duplication and loopish flags without content gains; ON keeps bursts steady and tames `step_norm` spikes.
- **Perf sweeps (fp16/bf16):** quality unchanged; throughput improves (see `tokens_per_sec` in metrics JSON).

Artifacts referenced:
- Geo (v4b tap‑9): `/mnt/data/generations_geo_steps.v4b.tap9.jsonl`  
- Geo (no‑denoise): `/mnt/data/generations_geo_steps.v4b.tap9.no_denoise.jsonl`  
- Geo (fp16 sweep): `/mnt/data/generations_geo_steps.v4b.tap9.t4_fp16.jsonl`  
- Stock baseline: `/mnt/data/generations_stock.v4b.tap9.jsonl`  
- Summary: `/mnt/data/steps7_10_summary.csv` and `/mnt/data/steps7_10_summary.json`

---

## 4) Freeze Checklist
- [x] Runner patched for `--config` / `--save_config` and metrics export
- [x] v4b tap‑9 profile present in `/mnt/data/calib/profile_v4b_tap9_text.json`
- [x] One smoke run each: Geo, Geo(no‑denoise), Stock (done)
- [x] Metrics JSON/CSV include `elapsed_sec`, `tokens_per_sec`
- [x] Telemetry available for the Geo run (optional)

---

## 5) Post‑Freeze (Optional **v4c**) Punch‑List
1) **True trend @ tap (radius‑decay)** — steadier bursts, cleaner gating (toggle‑able)

2) **Tap‑true PCA calibration** — more accurate inward direction; less off‑plane wobble

3) **Decision‑window smoothing** — short EMA over trend before Detect’s filter

4) **Guarded multi‑pass warp** — 2 passes on high‑confidence bursts

5) **Auto‑null calibration per family** — minor adaptive tweak to `null_q`

**Promote to v4c only if:** ≥ +1–2 pp on accuracy/F1 **or** ≥ −2–3 pp hallucination/loop rate on a 100‑prompt A/B slice (no >5% speed regress).

---

## 6) Bottom Line
- **This is freeze‑worthy.** v4b · tap‑9 matches the wiring plan, keeps doctrine pure (gain‑only Detect, sign‑safe denoiser), and has reproducible ops + telemetry.

- Proceed with final benchmarks on v4b; revisit v4c only if the A/B shows material headroom.
