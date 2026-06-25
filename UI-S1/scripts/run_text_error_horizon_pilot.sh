#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/aiscuser/UI-S1/UI-S1}"
cd "$PROJECT_ROOT"

JSONL_FILE="${JSONL_FILE:-$PROJECT_ROOT/datasets/GUI-Odyssey/gui_odyssey_random_split_test.jsonl}"
ROLLOUT_RESULTS="${ROLLOUT_RESULTS:-$PROJECT_ROOT/outputs/qwen35_9b_baseline_template_gui_odyssey_full_random_split_test_8gpu_512threads/merged_corrected_direct1k/trajectory_results.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/text_error_horizon_probe/qwen35_pilot300_gt_history}"
PILOT_FILE="${PILOT_FILE:-$OUTPUT_ROOT/pilot_episodes.jsonl}"
MAX_EPISODES="${MAX_EPISODES:-300}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
PORT_BASE="${PORT_BASE:-8110}"
SHARD_COUNT="${SHARD_COUNT:-}"
MAX_WORKERS="${MAX_WORKERS:-8}"
FORCE_BUILD="${FORCE_BUILD:-0}"
FORCE_RUN="${FORCE_RUN:-0}"

mkdir -p "$OUTPUT_ROOT"

if [[ "$FORCE_BUILD" == "1" || ! -s "$PILOT_FILE" ]]; then
  .venv/bin/python scripts/build_text_error_horizon_pilot.py \
    --jsonl-file "$JSONL_FILE" \
    --rollout-results "$ROLLOUT_RESULTS" \
    --output "$PILOT_FILE" \
    --max-episodes "$MAX_EPISODES"
else
  echo "Reusing pilot file: $PILOT_FILE"
fi

if [[ "$FORCE_RUN" == "1" || ! -s "$OUTPUT_ROOT/probe_rows.jsonl" ]]; then
  EPISODE_IDS="$PILOT_FILE" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  JSONL_FILE="$JSONL_FILE" \
  GPU_LIST="$GPU_LIST" \
  PORT_BASE="$PORT_BASE" \
  SHARD_COUNT="$SHARD_COUNT" \
  MAX_WORKERS="$MAX_WORKERS" \
    bash scripts/run_text_error_horizon_probe_sharded.sh
else
  echo "Reusing probe rows: $OUTPUT_ROOT/probe_rows.jsonl"
fi

.venv/bin/python scripts/summarize_text_error_horizon_probe.py \
  --probe-rows "$OUTPUT_ROOT/probe_rows.jsonl" \
  --output-dir "$OUTPUT_ROOT"

echo "Report: $OUTPUT_ROOT/text_error_horizon_report.md"