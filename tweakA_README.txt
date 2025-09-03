Param patch tweakA for steps 5–6 (detect gain + soft denoise)\n\nGoals:\n - Lengthen bursts from ~1 token to ~2–3 tokens (reduce chatter).\n - Tame detect micro-peaks and smooth denoiser knee.\n - Keep doctrine constants intact (alpha_min always-on).\n\nChanges:\n{
  "linger": "2 \u2192 4",
  "trend_tau": "0.35 \u2192 0.32",
  "denoise_window": "3 \u2192 5",
  "denoise_k": "8.0 \u2192 6.0",
  "denoise_tau": "0.35 \u2192 0.40",
  "phantom_guard_gamma": "0.35 \u2192 0.45",
  "phantom_tr_tau": "0.60 \u2192 0.65",
  "detect_width": "24 \u2192 20",
  "null_K": "24/32 \u2192 32",
  "null_q": "0.88/0.92 \u2192 0.92",
  "k_det": "9 \u2192 8"
}\n\nUsage:\nRun with telemetry:
  python3 /mnt/data/text_arc_unified_steps1_6.py \
    --model gpt2 --tap -9 \
    --calib calib_prompts.txt \
    --prompts eval_prompts.txt \
    --gen_mode geo \
    --alpha0 0.05 --alpha_min 0.006 \
    --trend_tau 0.32 --k_tr 12.0 \
    --use_detect 1 --detect_width 20 --null_K 32 --null_q 0.92 --k_det 8.0 \
    --s_latch 0.3 --linger 4 --ema_center_beta 0.05 \
    --use_denoise 1 --denoise_beta 0.6 --denoise_window 5 \
    --denoise_k 6.0 --denoise_tau 0.4 --phantom_tr_tau 0.65 --phantom_guard_gamma 0.45 \
    --jitter_eps 0.03 \
    --max_new_tokens 96 \
    --telemetry_jsonl geo_steps1_6.tweakA.telemetry.jsonl \
    --out generations_geo_steps.tweakA.jsonl\n