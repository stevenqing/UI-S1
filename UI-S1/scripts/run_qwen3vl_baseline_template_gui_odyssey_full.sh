#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/aiscuser/UI-S1/UI-S1}"
VLLM_ENV_DIR="${VLLM_ENV_DIR:-$PROJECT_ROOT/.venv-qwen3-vllm-stable}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/checkpoints/Qwen3-VL-8B-Instruct}"
MODEL_NAME="${MODEL_NAME:-qwen3-vl-8b-instruct}"
DATASET_DIR="${DATASET_DIR:-$PROJECT_ROOT/datasets/GUI-Odyssey}"
SPLIT="${SPLIT:-random_split}"
SUBSET="${SUBSET:-test}"
JSONL_FILE="${JSONL_FILE:-$DATASET_DIR/gui_odyssey_${SPLIT}_${SUBSET}.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/qwen3vl_baseline_template_gui_odyssey_full_${SPLIT}_${SUBSET}}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-0}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
THREADS="${THREADS:-32}"
MAX_EPISODES="${MAX_EPISODES:-}"
START_EPISODE="${START_EPISODE:-0}"
END_EPISODE="${END_EPISODE:-}"
RESUME="${RESUME:-0}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export QWENVL_ENDPOINT="http://localhost:$PORT/v1"
mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$JSONL_FILE" ]]; then
  .venv/bin/python gui_odyssey_eval/convert_to_eval_format.py \
    --data_dir "$DATASET_DIR" \
    --split "$SPLIT" \
    --subset "$SUBSET" \
    --output_dir "$DATASET_DIR"
fi

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

CUDA_VISIBLE_DEVICES="$GPUS" "$VLLM_ENV_DIR/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --port "$PORT" \
  --tensor-parallel-size "$TP_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --limit-mm-per-prompt '{"image": 2}' \
  $VLLM_EXTRA_ARGS \
  > "$OUTPUT_DIR/vllm_server.log" 2>&1 &
VLLM_PID=$!

echo "Started vLLM pid=$VLLM_PID on GPUs=$GPUS port=$PORT env=$VLLM_ENV_DIR"

for _ in $(seq 1 240); do
  if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "vLLM ready"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited early; tailing log" >&2
    tail -240 "$OUTPUT_DIR/vllm_server.log" >&2 || true
    exit 1
  fi
  sleep 5
done

if ! curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then
  echo "vLLM did not become ready; tailing log" >&2
  tail -240 "$OUTPUT_DIR/vllm_server.log" >&2 || true
  exit 1
fi

MAX_EP_ARGS=()
if [[ -n "$MAX_EPISODES" ]]; then
  MAX_EP_ARGS=(--max_episodes "$MAX_EPISODES")
fi
RANGE_ARGS=(--start_episode "$START_EPISODE")
if [[ -n "$END_EPISODE" ]]; then
  RANGE_ARGS+=(--end_episode "$END_EPISODE")
fi
RESUME_ARGS=()
if [[ "$RESUME" == "1" || "$RESUME" == "true" || "$RESUME" == "TRUE" ]]; then
  RESUME_ARGS=(--resume)
fi

.venv/bin/python gui_odyssey_eval/eval_ar_trajectory.py \
  --jsonl_file "$JSONL_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --model_name "$MODEL_NAME" \
  --split_name "$SPLIT" \
  --n_history_image_limit 2 \
  --max_workers "$THREADS" \
  --no_stop \
  "${RANGE_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "${MAX_EP_ARGS[@]}"

.venv/bin/python scripts/summarize_gui_odyssey_trajectory_results.py \
  --trajectory-results "$OUTPUT_DIR/trajectory_results.jsonl" \
  --output-dir "$OUTPUT_DIR"

echo "Outputs:"
echo "  $OUTPUT_DIR/trajectory_results.jsonl"
echo "  $OUTPUT_DIR/summary.json"
echo "  $OUTPUT_DIR/summary_enriched.json"
echo "  $OUTPUT_DIR/trajectory_metrics.jsonl"
echo "  $OUTPUT_DIR/error_samples.jsonl"