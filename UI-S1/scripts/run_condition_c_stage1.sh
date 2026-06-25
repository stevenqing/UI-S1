#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/aiscuser/UI-S1/UI-S1}"
cd "$PROJECT_ROOT"

OUT="${OUT:-$PROJECT_ROOT/outputs/text_error_horizon_probe/condition_c/stage1_main}"
STAGE0_RESULTS="${STAGE0_RESULTS:-$PROJECT_ROOT/outputs/text_error_horizon_probe/condition_c/stage0_zero_point/pair_rows.jsonl}"
PAIRS="$OUT/pairs.jsonl"
JSONL_FILE="${JSONL_FILE:-$PROJECT_ROOT/datasets/GUI-Odyssey/gui_odyssey_random_split_test.jsonl}"
ROLLOUT_RESULTS="${ROLLOUT_RESULTS:-$PROJECT_ROOT/outputs/qwen35_9b_baseline_template_gui_odyssey_full_random_split_test_8gpu_512threads/merged_corrected_direct1k/trajectory_results.jsonl}"
PILOT_EPISODES="${PILOT_EPISODES:-$PROJECT_ROOT/outputs/text_error_horizon_probe/qwen35_pilot300_action_only/pilot_episodes.jsonl}"
FORCE_BUILD="${FORCE_BUILD:-0}"
FORCE_RUN="${FORCE_RUN:-0}"

if [[ ! -f "$STAGE0_RESULTS" ]]; then
  echo "Missing STAGE0_RESULTS=$STAGE0_RESULTS" >&2
  exit 1
fi

mkdir -p "$OUT"
if [[ "$FORCE_BUILD" == "1" || ! -s "$PAIRS" ]]; then
  .venv/bin/python scripts/build_condition_c_pairs.py \
    --jsonl-file "$JSONL_FILE" \
    --rollout-results "$ROLLOUT_RESULTS" \
    --pilot-episode-ids "$PILOT_EPISODES" \
    --output "$PAIRS" \
    --stage stage1 \
    --inject-mode wrong \
    --distances 1-30
else
  echo "Reusing pairs: $PAIRS"
fi

if [[ "$FORCE_RUN" == "1" || ! -s "$OUT/pair_rows.jsonl" ]]; then
  PAIRS_FILE="$PAIRS" OUTPUT_ROOT="$OUT" bash scripts/run_condition_c_sharded.sh
else
  echo "Reusing pair rows: $OUT/pair_rows.jsonl"
fi

.venv/bin/python scripts/summarize_condition_c_probe.py \
  --results "$OUT/pair_rows.jsonl" \
  --output-dir "$OUT" \
  --stage stage1 \
  --stage0-results "$STAGE0_RESULTS"

echo "Summary: $OUT/summary.md"