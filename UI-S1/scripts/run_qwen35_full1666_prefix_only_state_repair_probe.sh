#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/aiscuser/UI-S1/UI-S1}"
cd "$PROJECT_ROOT"

TRAJECTORY_RESULTS="${TRAJECTORY_RESULTS:-$PROJECT_ROOT/outputs/qwen35_9b_baseline_template_gui_odyssey_full_random_split_test_8gpu_512threads/merged_corrected_direct1k/trajectory_results.jsonl}"
GUI_ODYSSEY_TEST_JSONL="${GUI_ODYSSEY_TEST_JSONL:-$PROJECT_ROOT/datasets/GUI-Odyssey/gui_odyssey_random_split_test.jsonl}"
SEGMENTS_JSONL="${SEGMENTS_JSONL:-$PROJECT_ROOT/datasets/segmentation_test/gui_odyssey_segments.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/gui_odyssey_on_policy_state_repair_probe/qwen35_prefix_only_no_future_full1666}"
PROBE_FILE="${PROBE_FILE:-$OUTPUT_ROOT/probes_first_error_all_prefix_only_no_future.jsonl}"
MANIFEST_FILE="$PROBE_FILE.manifest.json"

MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/checkpoints/Qwen3.5-9B}"
MODEL_NAME="${MODEL_NAME:-qwen3.5-9b}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
PORT_BASE="${PORT_BASE:-8080}"
SHARD_COUNT="${SHARD_COUNT:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
MAX_WORKERS="${MAX_WORKERS:-16}"
MAX_TOKENS="${MAX_TOKENS:-256}"
FORCE_BUILD="${FORCE_BUILD:-0}"
FORCE_RUN="${FORCE_RUN:-0}"
FORCE_SUMMARY="${FORCE_SUMMARY:-1}"

mkdir -p "$OUTPUT_ROOT"

echo "Full GUI-Odyssey first-error prefix-only/no-future state repair probe"
echo "Project:      $PROJECT_ROOT"
echo "Trajectory:   $TRAJECTORY_RESULTS"
echo "Dataset:      $GUI_ODYSSEY_TEST_JSONL"
echo "Segments:     $SEGMENTS_JSONL"
echo "Output:       $OUTPUT_ROOT"
echo "Probe file:   $PROBE_FILE"
echo "Model:        $MODEL_PATH"
echo "GPUs:         $GPU_LIST"
echo "Ports:        $PORT_BASE+"

for required in "$TRAJECTORY_RESULTS" "$GUI_ODYSSEY_TEST_JSONL" "$SEGMENTS_JSONL"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Missing MODEL_PATH=$MODEL_PATH" >&2
  exit 1
fi

if [[ "$FORCE_BUILD" == "1" || ! -s "$PROBE_FILE" ]]; then
  echo "[1/3] Building full first-error probes"
  .venv/bin/python scripts/build_on_policy_state_repair_probes.py \
    --trajectory-results "$TRAJECTORY_RESULTS" \
    --jsonl-file "$GUI_ODYSSEY_TEST_JSONL" \
    --segments "$SEGMENTS_JSONL" \
    --output "$PROBE_FILE" \
    --selection-mode first_error_all \
    --state-mode prefix_only_no_future \
    --min-num-steps 0 \
    --max-probes 0
else
  echo "[1/3] Reusing existing probe file: $PROBE_FILE"
fi

echo "Probe lines: $(wc -l < "$PROBE_FILE")"
if [[ -f "$MANIFEST_FILE" ]]; then
  echo "Manifest: $MANIFEST_FILE"
fi

if [[ "$FORCE_RUN" == "1" || ! -s "$OUTPUT_ROOT/probe_results.jsonl" ]]; then
  echo "[2/3] Running sharded probe evaluation"
  PROBE_FILE="$PROBE_FILE" \
  OUTPUT_ROOT="$OUTPUT_ROOT" \
  MODEL_PATH="$MODEL_PATH" \
  MODEL_NAME="$MODEL_NAME" \
  GPU_LIST="$GPU_LIST" \
  PORT_BASE="$PORT_BASE" \
  SHARD_COUNT="$SHARD_COUNT" \
  MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
  MAX_WORKERS="$MAX_WORKERS" \
  MAX_TOKENS="$MAX_TOKENS" \
    bash scripts/run_on_policy_state_repair_probe_sharded.sh
else
  echo "[2/3] Reusing existing probe results: $OUTPUT_ROOT/probe_results.jsonl"
fi

if [[ "$FORCE_SUMMARY" == "1" || ! -s "$OUTPUT_ROOT/population_report.md" ]]; then
  echo "[3/3] Writing population summary"
  .venv/bin/python scripts/summarize_on_policy_state_repair_population.py \
    --probe-results "$OUTPUT_ROOT/probe_results.jsonl" \
    --probe-manifest "$MANIFEST_FILE" \
    --output-json "$OUTPUT_ROOT/population_summary.json" \
    --output-md "$OUTPUT_ROOT/population_report.md" \
    --total-test-episodes 1666
else
  echo "[3/3] Reusing existing population report: $OUTPUT_ROOT/population_report.md"
fi

echo "Done. Read: $OUTPUT_ROOT/population_report.md"