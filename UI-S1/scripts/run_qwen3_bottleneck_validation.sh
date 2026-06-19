#!/usr/bin/env bash
set -euo pipefail

# Run Qwen3-VL-8B and Qwen3.5-9B bottleneck behavior validation.
# Requires two OpenAI-compatible vLLM servers:
#   Qwen3-VL-8B-Instruct at http://localhost:8000/v1
#   Qwen3.5-9B at http://localhost:8001/v1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VL_ENV="${VL_ENV:-${QWEN3_ENV:-$PROJECT_ROOT/.venv-qwen3-vllm}}"
TEXT_ENV="${TEXT_ENV:-$PROJECT_ROOT/.venv-qwen35-vllm}"
VL_MODEL_PATH="${VL_MODEL_PATH:-$PROJECT_ROOT/checkpoints/Qwen3-VL-8B-Instruct}"
TEXT_MODEL_PATH="${TEXT_MODEL_PATH:-$PROJECT_ROOT/checkpoints/Qwen3.5-9B}"
VL_SERVED_NAME="${VL_SERVED_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
TEXT_SERVED_NAME="${TEXT_SERVED_NAME:-Qwen/Qwen3.5-9B}"
VL_PORT="${VL_PORT:-8000}"
TEXT_PORT="${TEXT_PORT:-8001}"
OUTPUT_DIR="${OUTPUT_DIR:-datasets/model_bottleneck_validation}"
MAX_CASES="${MAX_CASES:-40}"
INPUTS="${INPUTS:-datasets/segmentation_train/gui_odyssey_segments.jsonl}"
MODELS="${MODELS:-vl qwen35}"
MAX_TOKENS="${MAX_TOKENS:-256}"
REQUEST_WORKERS="${REQUEST_WORKERS:-1}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
THINKING_MODES="${THINKING_MODES:-non_thinking thinking}"
START_SEQUENTIAL="${START_SEQUENTIAL:-0}"
RESUME_PARTIAL="${RESUME_PARTIAL:-0}"
CASE_SHARD_INDEX="${CASE_SHARD_INDEX:-0}"
CASE_SHARD_COUNT="${CASE_SHARD_COUNT:-1}"
read -r -a VL_EXTRA_ARGS_ARRAY <<< "${VL_EXTRA_ARGS:-}"
read -r -a TEXT_EXTRA_ARGS_ARRAY <<< "${TEXT_EXTRA_ARGS:-}"

RUN_VL=0
RUN_TEXT=0
for MODEL_KIND in $MODELS; do
  case "$MODEL_KIND" in
    vl) RUN_VL=1 ;;
    text|qwen35) RUN_TEXT=1 ;;
    *) echo "Unknown model kind in MODELS: $MODEL_KIND" >&2; exit 2 ;;
  esac
done

if [[ "$RUN_VL" -eq 1 && ! -d "$VL_ENV" ]]; then
  echo "Missing Qwen3-VL uv environment: $VL_ENV" >&2
  echo "Create it with uv or set VL_ENV." >&2
  exit 2
fi
if [[ "$RUN_TEXT" -eq 1 && ! -d "$TEXT_ENV" ]]; then
  echo "Missing Qwen3.5 uv environment: $TEXT_ENV" >&2
  echo "Create it with uv or set TEXT_ENV." >&2
  exit 2
fi

VL_PYTHON="$VL_ENV/bin/python"
VL_VLLM="$VL_ENV/bin/vllm"
TEXT_PYTHON="$TEXT_ENV/bin/python"
TEXT_VLLM="$TEXT_ENV/bin/vllm"
if [[ "$RUN_VL" -eq 1 && (! -x "$VL_PYTHON" || ! -x "$VL_VLLM") ]]; then
  echo "Missing python/vllm binaries under $VL_ENV" >&2
  exit 2
fi
if [[ "$RUN_TEXT" -eq 1 && (! -x "$TEXT_PYTHON" || ! -x "$TEXT_VLLM") ]]; then
  echo "Missing python/vllm binaries under $TEXT_ENV" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

cleanup() {
  if [[ -n "${VL_PID:-}" ]]; then kill "$VL_PID" 2>/dev/null || true; fi
  if [[ -n "${TEXT_PID:-}" ]]; then kill "$TEXT_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT

WAIT_URLS=()
EVAL_PYTHON="${VL_PYTHON:-$TEXT_PYTHON}"
if [[ "$RUN_VL" -eq 1 ]]; then
  EVAL_PYTHON="$VL_PYTHON"
  echo "Starting Qwen3-VL server on port $VL_PORT: $VL_MODEL_PATH"
  VLLM_USE_V1="${VL_VLLM_USE_V1:-1}" CUDA_VISIBLE_DEVICES="${VL_CUDA_VISIBLE_DEVICES:-1}" "$VL_VLLM" serve "$VL_MODEL_PATH" \
    --served-model-name "$VL_SERVED_NAME" \
    --host 127.0.0.1 --port "$VL_PORT" \
    --trust-remote-code \
    --tensor-parallel-size "${VL_TP:-1}" \
    --max-model-len "${VL_MAX_MODEL_LEN:-8192}" \
    --gpu-memory-utilization "${VL_GPU_MEMORY_UTILIZATION:-0.65}" \
    --enforce-eager \
    "${VL_EXTRA_ARGS_ARRAY[@]}" \
    > "$OUTPUT_DIR/qwen3_vl_server.log" 2>&1 &
  VL_PID=$!
  WAIT_URLS+=("http://127.0.0.1:$VL_PORT/v1/models")
  if [[ "$START_SEQUENTIAL" == "1" && "$RUN_TEXT" -eq 1 ]]; then
    echo "Waiting for Qwen3-VL before starting Qwen3.5..."
    WAIT_URLS_TEXT="http://127.0.0.1:$VL_PORT/v1/models" "$VL_PYTHON" - <<'PY'
import time, requests, sys
import os
urls = os.environ['WAIT_URLS_TEXT'].split()
deadline = time.time() + 900
ready = {url: False for url in urls}
while time.time() < deadline:
    for url in urls:
        if ready[url]:
            continue
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                print('ready', url)
                ready[url] = True
        except Exception:
            pass
    if all(ready.values()):
        sys.exit(0)
    time.sleep(10)
print('Timeout waiting for servers', ready)
sys.exit(1)
PY
  fi
fi

if [[ "$RUN_TEXT" -eq 1 ]]; then
  EVAL_PYTHON="$TEXT_PYTHON"
  echo "Starting Qwen3.5 text server on port $TEXT_PORT: $TEXT_MODEL_PATH"
  CUDA_VISIBLE_DEVICES="${TEXT_CUDA_VISIBLE_DEVICES:-2}" "$TEXT_VLLM" serve "$TEXT_MODEL_PATH" \
    --served-model-name "$TEXT_SERVED_NAME" \
    --host 127.0.0.1 --port "$TEXT_PORT" \
    --trust-remote-code \
    --tensor-parallel-size "${TEXT_TP:-1}" \
    --max-model-len "${TEXT_MAX_MODEL_LEN:-4096}" \
    --gpu-memory-utilization "${TEXT_GPU_MEMORY_UTILIZATION:-0.55}" \
    --enforce-eager \
    "${TEXT_EXTRA_ARGS_ARRAY[@]}" \
    > "$OUTPUT_DIR/qwen35_text_server.log" 2>&1 &
  TEXT_PID=$!
  WAIT_URLS+=("http://127.0.0.1:$TEXT_PORT/v1/models")
fi

echo "Waiting for servers..."
WAIT_URLS_TEXT="${WAIT_URLS[*]}" "$EVAL_PYTHON" - <<'PY'
import time, requests, sys
import os
urls = os.environ['WAIT_URLS_TEXT'].split()
deadline = time.time() + 900
ready = {url: False for url in urls}
while time.time() < deadline:
    for url in urls:
        if ready[url]:
            continue
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                print('ready', url)
                ready[url] = True
        except Exception:
            pass
    if all(ready.values()):
        sys.exit(0)
    time.sleep(10)
print('Timeout waiting for servers', ready)
sys.exit(1)
PY

"$EVAL_PYTHON" scripts/eval_model_bottleneck_behavior.py \
  --inputs $INPUTS \
  --output-dir "$OUTPUT_DIR" \
  --vl-api-url http://127.0.0.1:$VL_PORT/v1 \
  --vl-model "$VL_SERVED_NAME" \
  --text-api-url http://127.0.0.1:$TEXT_PORT/v1 \
  --text-model "$TEXT_SERVED_NAME" \
  --max-cases "$MAX_CASES" \
  --case-shard-index "$CASE_SHARD_INDEX" \
  --case-shard-count "$CASE_SHARD_COUNT" \
  --max-tokens "$MAX_TOKENS" \
  --request-workers "$REQUEST_WORKERS" \
  --timeout "$REQUEST_TIMEOUT" \
  --thinking-modes $THINKING_MODES \
  --models $MODELS \
  $(if [[ "$RESUME_PARTIAL" == "1" ]]; then printf '%s' '--resume-partial'; fi)
