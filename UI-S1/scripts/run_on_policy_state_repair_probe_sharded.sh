#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/aiscuser/UI-S1/UI-S1}"
PROBE_FILE="${PROBE_FILE:?PROBE_FILE is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/checkpoints/Qwen3.5-9B}"
MODEL_NAME="${MODEL_NAME:-qwen3.5-9b}"
GPU_LIST="${GPU_LIST:-0}"
PORT_BASE="${PORT_BASE:-8070}"
SHARD_COUNT="${SHARD_COUNT:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"
MAX_WORKERS="${MAX_WORKERS:-16}"
MAX_TOKENS="${MAX_TOKENS:-256}"
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

mkdir -p "$OUTPUT_ROOT/probe_shards"
.venv/bin/python - <<PY
from pathlib import Path
probe_file = Path("$PROBE_FILE")
out = Path("$OUTPUT_ROOT/probe_shards")
shards = int("$SHARD_COUNT")
handles = [(out / f"shard_{idx}.jsonl").open("w", encoding="utf-8") for idx in range(shards)]
try:
    for idx, line in enumerate(probe_file.open("r", encoding="utf-8")):
        handles[idx % shards].write(line)
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
  CUDA_VISIBLE_DEVICES="$gpu" "$PROJECT_ROOT/.venv-qwen3-vllm-stable/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$MODEL_NAME" \
    --port "$port" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --limit-mm-per-prompt '{"image": 1}' \
    $VLLM_EXTRA_ARGS \
    > "$shard_dir/vllm_server.log" 2>&1 &
  server_pids+=("$!")
  echo "started server shard=$shard_idx gpu=$gpu port=$port pid=${server_pids[-1]}"
done

for shard_idx in $(seq 0 $((SHARD_COUNT - 1))); do
  port=$((PORT_BASE + shard_idx))
  for _ in $(seq 1 240); do
    if curl -fsS "http://localhost:$port/health" >/dev/null 2>&1; then
      echo "ready shard=$shard_idx port=$port"
      break
    fi
    if ! kill -0 "${server_pids[$shard_idx]}" 2>/dev/null; then
      echo "server shard=$shard_idx exited" >&2
      tail -160 "${shard_dirs[$shard_idx]}/vllm_server.log" >&2 || true
      exit 1
    fi
    sleep 5
  done
done

for shard_idx in $(seq 0 $((SHARD_COUNT - 1))); do
  port=$((PORT_BASE + shard_idx))
  shard_dir="${shard_dirs[$shard_idx]}"
  .venv/bin/python scripts/run_on_policy_state_repair_probe.py \
    --probes "$OUTPUT_ROOT/probe_shards/shard_$shard_idx.jsonl" \
    --output-dir "$shard_dir" \
    --model-name "$MODEL_NAME" \
    --endpoint "http://localhost:$port/v1" \
    --max-tokens "$MAX_TOKENS" \
    --max-workers "$MAX_WORKERS" \
    > "$shard_dir/run.log" 2>&1 &
  runner_pids+=("$!")
  echo "started runner shard=$shard_idx pid=${runner_pids[-1]}"
done

failed=0
for shard_idx in $(seq 0 $((SHARD_COUNT - 1))); do
  if ! wait "${runner_pids[$shard_idx]}"; then
    echo "runner shard=$shard_idx failed" >&2
    tail -120 "${shard_dirs[$shard_idx]}/run.log" >&2 || true
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

: > "$OUTPUT_ROOT/probe_results.jsonl"
for shard_idx in $(seq 0 $((SHARD_COUNT - 1))); do
  cat "$OUTPUT_ROOT/shard_$shard_idx/probe_results.jsonl" >> "$OUTPUT_ROOT/probe_results.jsonl"
done

.venv/bin/python - <<PY
import json
from pathlib import Path
from scripts.run_on_policy_state_repair_probe import summarize
root = Path("$OUTPUT_ROOT")
rows = [json.loads(line) for line in (root / "probe_results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
summary = summarize(rows)
(root / "probe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY