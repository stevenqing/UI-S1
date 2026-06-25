#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/aiscuser/UI-S1/UI-S1}"
JSONL_FILE="${JSONL_FILE:-$PROJECT_ROOT/datasets/GUI-Odyssey/gui_odyssey_random_split_test.jsonl}"
PAIRS_FILE="${PAIRS_FILE:?PAIRS_FILE is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/checkpoints/Qwen3.5-9B}"
MODEL_NAME="${MODEL_NAME:-qwen3.5-9b}"
GPU_LIST="${GPU_LIST:-0}"
PORT_BASE="${PORT_BASE:-8190}"
SHARD_COUNT="${SHARD_COUNT:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-8G}"
MAX_WORKERS="${MAX_WORKERS:-4}"
N_SAMPLES="${N_SAMPLES:-3}"
SERVER_START_RETRIES="${SERVER_START_RETRIES:-2}"
RESUME="${RESUME:-0}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---enforce-eager --generation-config vllm}"

cd "$PROJECT_ROOT"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then
  echo "GPU_LIST is empty" >&2
  exit 1
fi
if [[ -z "$SHARD_COUNT" ]]; then
  SHARD_COUNT="${#GPU_ARRAY[@]}"
fi

mkdir -p "$OUTPUT_ROOT/pair_shards"
.venv/bin/python - <<PY
import hashlib
import json
from pathlib import Path
pairs = Path("$PAIRS_FILE")
out = Path("$OUTPUT_ROOT/pair_shards")
shards = int("$SHARD_COUNT")
handles = [(out / f"shard_{idx}.jsonl").open("w", encoding="utf-8") for idx in range(shards)]
try:
    for line in pairs.open("r", encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        key = f"{row['episode_id']}:{row['target_step']}"
        shard = int(hashlib.md5(key.encode()).hexdigest(), 16) % shards
        handles[shard].write(line)
finally:
    for handle in handles:
        handle.close()
for idx in range(shards):
    path = out / f"shard_{idx}.jsonl"
    print(path, sum(1 for _ in path.open("r", encoding="utf-8")))
PY

server_pids=()
runner_pids=()
shard_dirs=()
cleanup() {
  for pid in "${server_pids[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  for pid in "${server_pids[@]:-}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

for shard_idx in $(seq 0 $((SHARD_COUNT - 1))); do
  gpu="${GPU_ARRAY[$((shard_idx % ${#GPU_ARRAY[@]}))]}"
  port=$((PORT_BASE + shard_idx))
  shard_dir="$OUTPUT_ROOT/shard_$shard_idx"
  mkdir -p "$shard_dir"
  shard_dirs+=("$shard_dir")
  ready=0
  for attempt in $(seq 1 "$SERVER_START_RETRIES"); do
    log_file="$shard_dir/vllm_server_attempt_${attempt}.log"
    rm -f "$shard_dir/vllm_server.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PROJECT_ROOT/.venv-qwen3-vllm-stable/bin/python" -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_PATH" \
      --served-model-name "$MODEL_NAME" \
      --port "$port" \
      --tensor-parallel-size 1 \
      --max-model-len "$MAX_MODEL_LEN" \
      --trust-remote-code \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES" \
      --limit-mm-per-prompt '{"image": 2}' \
      $VLLM_EXTRA_ARGS \
      > "$log_file" 2>&1 &
    server_pid="$!"
    ln -sf "$(basename "$log_file")" "$shard_dir/vllm_server.log"
    echo "started server shard=$shard_idx attempt=$attempt gpu=$gpu port=$port pid=$server_pid"
    for _ in $(seq 1 240); do
      if curl -fsS "http://localhost:$port/health" >/dev/null 2>&1; then
        echo "ready shard=$shard_idx port=$port attempt=$attempt"
        server_pids+=("$server_pid")
        ready=1
        break
      fi
      if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "server shard=$shard_idx attempt=$attempt exited" >&2
        tail -160 "$log_file" >&2 || true
        break
      fi
      sleep 5
    done
    if [[ "$ready" == "1" ]]; then
      break
    fi
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  done
  if [[ "$ready" != "1" ]]; then
    echo "server shard=$shard_idx failed after $SERVER_START_RETRIES attempts" >&2
    exit 1
  fi
done

for shard_idx in $(seq 0 $((SHARD_COUNT - 1))); do
  port=$((PORT_BASE + shard_idx))
  shard_dir="${shard_dirs[$shard_idx]}"
  RESUME_ARGS=()
  if [[ "$RESUME" == "1" || "$RESUME" == "true" || "$RESUME" == "TRUE" ]]; then
    RESUME_ARGS=(--resume)
  fi
  .venv/bin/python scripts/run_condition_c_probe.py \
    --jsonl-file "$JSONL_FILE" \
    --pairs "$OUTPUT_ROOT/pair_shards/shard_$shard_idx.jsonl" \
    --output "$shard_dir/pair_rows.jsonl" \
    --model-name "$MODEL_NAME" \
    --endpoint "http://localhost:$port/v1" \
    --n-samples "$N_SAMPLES" \
    --max-workers "$MAX_WORKERS" \
    "${RESUME_ARGS[@]}" \
    > "$shard_dir/run.log" 2>&1 &
  runner_pids+=("$!")
  echo "started runner shard=$shard_idx pid=${runner_pids[-1]}"
done

failed=0
for shard_idx in $(seq 0 $((SHARD_COUNT - 1))); do
  if ! wait "${runner_pids[$shard_idx]}"; then
    echo "runner shard=$shard_idx failed" >&2
    tail -160 "${shard_dirs[$shard_idx]}/run.log" >&2 || true
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

: > "$OUTPUT_ROOT/pair_rows.jsonl"
for shard_idx in $(seq 0 $((SHARD_COUNT - 1))); do
  cat "$OUTPUT_ROOT/shard_$shard_idx/pair_rows.jsonl" >> "$OUTPUT_ROOT/pair_rows.jsonl"
done

cat > "$OUTPUT_ROOT/manifest.json" <<JSON
{
  "jsonl_file": "$JSONL_FILE",
  "pairs_file": "$PAIRS_FILE",
  "model_path": "$MODEL_PATH",
  "model_name": "$MODEL_NAME",
  "n_samples": "$N_SAMPLES",
  "output": "$OUTPUT_ROOT/pair_rows.jsonl"
}
JSON

echo "Merged: $OUTPUT_ROOT/pair_rows.jsonl"