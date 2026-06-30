#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-.venv-qwen3-vllm/bin/python}
MODEL_PATH=${MODEL_PATH:-train_GUI_360/llamafactory/output/gui360_gt_history_full_sft/checkpoint-39}
MODEL_NAME=${MODEL_NAME:-checkpoint-39-original-template}
TEST_DATA=${TEST_DATA:-outputs/gui360_history_ab/original_eval/gui360_test_1000_balanced_reconstructed.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/gui360_history_ab/original_template_ckpt39_$(date +%Y%m%d_%H%M%S)}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8135}
TP=${TP:-4}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-131072}
LIMIT_MM_IMAGES=${LIMIT_MM_IMAGES:-2}
KV_CACHE_MEMORY_BYTES=${KV_CACHE_MEMORY_BYTES:-}
SKIP_MM_PROFILING=${SKIP_MM_PROFILING:-0}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}
THREADS=${THREADS:-128}
MAX_TOKENS=${MAX_TOKENS:-1024}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-600}
START=${START:-0}
END=${END:-}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

SERVER_PID=""

log() {
  printf '[original-template-eval] %s\n' "$*"
}

stop_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    log "stopping vLLM pid=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}

wait_for_server() {
  local base_url="http://$HOST:$PORT"
  local deadline=$((SECONDS + 900))
  while (( SECONDS < deadline )); do
    if "$PYTHON" - <<PY >/dev/null 2>&1
import urllib.request
urllib.request.urlopen('$base_url/health', timeout=5).read()
PY
    then
      log "vLLM ready"
      return 0
    fi
    if grep -Eq 'Traceback \(most recent call last\)|CUDA out of memory|Address already in use|RuntimeError' "$OUTPUT_DIR/server.log" 2>/dev/null; then
      tail -160 "$OUTPUT_DIR/server.log" || true
      return 3
    fi
    sleep 5
  done
  tail -160 "$OUTPUT_DIR/server.log" || true
  return 2
}

main() {
  mkdir -p "$OUTPUT_DIR"
  log "model: $MODEL_PATH"
  log "test_data: $TEST_DATA"
  log "output_dir: $OUTPUT_DIR"
  log "shard: [$START:${END:-end}]"
  server_args=(
    -m vllm.entrypoints.openai.api_server
    --model "$MODEL_PATH"
    --served-model-name "$MODEL_NAME"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size "$TP"
    --max-model-len "$MAX_MODEL_LEN"
    --trust-remote-code
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --limit-mm-per-prompt "{\"image\": $LIMIT_MM_IMAGES}"
    --disable-log-requests
    --disable-log-stats
  )
  if [[ -n "${KV_CACHE_MEMORY_BYTES:-}" ]]; then
    server_args+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
  fi
  if [[ "$SKIP_MM_PROFILING" == "1" ]]; then
    server_args+=(--skip-mm-profiling)
  fi
  if [[ "$ENFORCE_EAGER" == "1" ]]; then
    server_args+=(--enforce-eager)
  fi
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" VLLM_USE_V1=1 "$PYTHON" "${server_args[@]}" >"$OUTPUT_DIR/server.log" 2>&1 &
  SERVER_PID=$!
  trap stop_server EXIT
  wait_for_server

  eval_args=(
    --test_data "$TEST_DATA"
    --api_url "http://$HOST:$PORT/v1"
    --model_name "$MODEL_NAME"
    --output_dir "$OUTPUT_DIR"
    --threads "$THREADS"
    --gt_history
    --max_tokens "$MAX_TOKENS"
    --request_timeout "$REQUEST_TIMEOUT"
    --start "$START"
  )
  if [[ -n "${END:-}" ]]; then
    eval_args+=(--end "$END")
  fi
  "$PYTHON" v13_gui_360/eval_gui360_template.py "${eval_args[@]}" 2>&1 | tee "$OUTPUT_DIR/eval.log"
  log "done"
}

main "$@"