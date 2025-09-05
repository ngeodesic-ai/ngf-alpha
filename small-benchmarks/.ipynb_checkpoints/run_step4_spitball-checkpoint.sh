#!/usr/bin/env bash
set -euo pipefail

MATRIX="step4_spitball_matrix.csv"
OUTDIR="logs"
mkdir -p "$OUTDIR"

# You can 'pip install pandas' if needed; otherwise, we parse with awk.
# We'll use awk to iterate and build commands.
tail -n +2 "$MATRIX" | while IFS=, read -r exp_id runner tap calib eval nbins sigma steps eta rescore notes; do
  echo "[RUN] $exp_id :: $runner"
  stamp=$(date +%Y%m%d-%H%M%S)
  json="$OUTDIR/${exp_id}_${stamp}.json"
  png="$OUTDIR/${exp_id}_${stamp}.png"

  if [[ "$runner" == "stage11_llm_shadow_base.py" ]]; then
    # Optional smoothing args are source edits in base; if your base supports CLI, append here.
    python3 "$runner" \
      --model gpt2 --tap "$tap" \
      --calib $(echo "$calib" | tr -d '"') \
      --eval  $(echo "$eval"  | tr -d '"') \
      --render_well \
      --out_json "$json" || true
  else
    # Hijack runner
    args=(
      --model gpt2 --tap "$tap"
      --calib $(echo "$calib" | tr -d '"')
      --eval  $(echo "$eval"  | tr -d '"')
      --steps "${steps:-96}" --eta "${eta:-0.15}"
      --ema_gamma 0.96 --med_k 17
      --jitter_sigma 0.015 --jitter_J 6
      --use_depth_weighted_pi 1 --pi_beta 6.5
      --nms_radius 7 --sigma_scale 0.72
      --local_radius 1.35
      --tau_conf 0.72 --backoff 0.78
      --tok_eta 0.20
      --out_json "$json" --render
    )
    if [[ "$rescore" == "1" ]]; then
      args+=( --enable_rescore 1 --rescore_eps 0.25 --rescore_lambda 1.0 --rescore_min_steps 1 )
    fi
    python3 stage11_llm_shadow-hijack-v4.py "${args[@]}" || true
  fi
done

echo "All Step-4 experiments queued. Outputs in $OUTDIR/"

