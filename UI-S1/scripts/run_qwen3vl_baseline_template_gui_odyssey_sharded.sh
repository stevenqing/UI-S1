#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/aiscuser/UI-S1/UI-S1}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/checkpoints/Qwen3-VL-8B-Instruct}"
MODEL_NAME="${MODEL_NAME:-qwen3-vl-8b-instruct}"
DATASET_DIR="${DATASET_DIR:-$PROJECT_ROOT/datasets/GUI-Odyssey}"
SPLIT="${SPLIT:-random_split}"
SUBSET="${SUBSET:-test}"
JSONL_FILE="${JSONL_FILE:-$DATASET_DIR/gui_odyssey_${SPLIT}_${SUBSET}.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/qwen3vl_baseline_template_gui_odyssey_full_${SPLIT}_${SUBSET}_sharded}"
MERGED_DIR="${MERGED_DIR:-$OUTPUT_ROOT/merged}"
GPU_LIST="${GPU_LIST:-0}"
PORT_BASE="${PORT_BASE:-8020}"
SHARD_COUNT="${SHARD_COUNT:-}"
THREADS_PER_SHARD="${THREADS_PER_SHARD:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
RESUME="${RESUME:-0}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---enforce-eager --generation-config vllm}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

if [[ ! -f "$JSONL_FILE" ]]; then
  .venv/bin/python gui_odyssey_eval/convert_to_eval_format.py \
    --data_dir "$DATASET_DIR" \
    --split "$SPLIT" \
    --subset "$SUBSET" \
    --output_dir "$DATASET_DIR"
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then
  echo "GPU_LIST is empty" >&2
  exit 1
fi
if [[ -z "$SHARD_COUNT" ]]; then
  SHARD_COUNT="${#GPU_ARRAY[@]}"
fi

TOTAL_EPISODES=$(wc -l < "$JSONL_FILE" | tr -d ' ')
mkdir -p "$OUTPUT_ROOT"

echo "Total episodes: $TOTAL_EPISODES"
echo "Shard count: $SHARD_COUNT"
echo "GPU list: $GPU_LIST"
echo "Model path: $MODEL_PATH"
echo "Model name: $MODEL_NAME"
echo "Output root: $OUTPUT_ROOT"

pids=()
shard_dirs=()

for shard_idx in $(seq 0 $((SHARD_COUNT - 1))); do
  start_episode=$((TOTAL_EPISODES * shard_idx / SHARD_COUNT))
  end_episode=$((TOTAL_EPISODES * (shard_idx + 1) / SHARD_COUNT))
  gpu="${GPU_ARRAY[$((shard_idx % ${#GPU_ARRAY[@]}))]}"
  port=$((PORT_BASE + shard_idx))
  shard_dir="$OUTPUT_ROOT/shard_${shard_idx}_episodes_${start_episode}_${end_episode}"
  mkdir -p "$shard_dir"
  shard_dirs+=("$shard_dir")

  echo "Launching shard $shard_idx episodes [$start_episode, $end_episode) on GPU $gpu port $port"
  (
    START_EPISODE="$start_episode" \
    END_EPISODE="$end_episode" \
    OUTPUT_DIR="$shard_dir" \
    PORT="$port" \
    GPUS="$gpu" \
    MODEL_PATH="$MODEL_PATH" \
    MODEL_NAME="$MODEL_NAME" \
    THREADS="$THREADS_PER_SHARD" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
    RESUME="$RESUME" \
    VLLM_EXTRA_ARGS="$VLLM_EXTRA_ARGS" \
    bash scripts/run_qwen3vl_baseline_template_gui_odyssey_full.sh
  ) > "$shard_dir/run.log" 2>&1 &
  pids+=("$!")
done

failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "Shard $i failed. Last log lines:" >&2
    tail -120 "${shard_dirs[$i]}/run.log" >&2 || true
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "At least one shard failed; not merging results." >&2
  exit 1
fi

mkdir -p "$MERGED_DIR"
: > "$MERGED_DIR/trajectory_results.jsonl"
for shard_dir in "${shard_dirs[@]}"; do
  cat "$shard_dir/trajectory_results.jsonl" >> "$MERGED_DIR/trajectory_results.jsonl"
done

.venv/bin/python scripts/summarize_gui_odyssey_trajectory_results.py \
  --trajectory-results "$MERGED_DIR/trajectory_results.jsonl" \
  --output-dir "$MERGED_DIR"

echo "Merged outputs:"
echo "  $MERGED_DIR/trajectory_results.jsonl"
echo "  $MERGED_DIR/summary_enriched.json"
echo "  $MERGED_DIR/trajectory_metrics.jsonl"
echo "  $MERGED_DIR/error_samples.jsonl"