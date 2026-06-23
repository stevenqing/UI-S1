#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/aiscuser/UI-S1/UI-S1}"
VLLM_ENV_DIR="${VLLM_ENV_DIR:-$PROJECT_ROOT/.venv-qwen3-vllm-stable}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/checkpoints/Qwen3-VL-8B-Instruct}"
MODEL_NAME="${MODEL_NAME:-qwen3-vl-8b-instruct}"
JSONL_FILE="${JSONL_FILE:-$PROJECT_ROOT/datasets/GUI-Odyssey/gui_odyssey_random_split_test.jsonl}"
SAMPLE_KEYS="${SAMPLE_KEYS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/qwen3vl_baseline_template_full_test}"
PORT="${PORT:-8000}"
GPUS="${GPUS:-0}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
THREADS="${THREADS:-32}"
LIMIT="${LIMIT:-0}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$JSONL_FILE" ]]; then
  .venv/bin/python gui_odyssey_eval/convert_to_eval_format.py \
    --data_dir datasets/GUI-Odyssey \
    --split random_split \
    --subset test \
    --output_dir datasets/GUI-Odyssey
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

echo "Started vLLM pid=$VLLM_PID on GPUs=$GPUS port=$PORT"

for _ in $(seq 1 180); do
  if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "vLLM ready"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM exited early; tailing log" >&2
    tail -200 "$OUTPUT_DIR/vllm_server.log" >&2 || true
    exit 1
  fi
  sleep 5
done

if ! curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then
  echo "vLLM did not become ready; tailing log" >&2
  tail -200 "$OUTPUT_DIR/vllm_server.log" >&2 || true
  exit 1
fi

if [[ -n "$SAMPLE_KEYS" ]]; then
  SAMPLE_ARGS=(--sample-keys "$SAMPLE_KEYS")
else
  SAMPLE_ARGS=(--all-steps)
fi

LIMIT_ARGS=()
if [[ "$LIMIT" != "0" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

.venv/bin/python scripts/eval_gui_odyssey_baseline_template_rows.py \
  --jsonl-file "$JSONL_FILE" \
  "${SAMPLE_ARGS[@]}" \
  --output-dir "$OUTPUT_DIR" \
  --api-url "http://localhost:$PORT/v1" \
  --model-name "$MODEL_NAME" \
  --threads "$THREADS" \
  "${LIMIT_ARGS[@]}"