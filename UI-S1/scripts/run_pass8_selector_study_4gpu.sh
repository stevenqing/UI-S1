#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

variant="${1:-}"
split="${2:-smoke}"
case "$variant" in
  current|strong) ;;
  *) echo "usage: $0 {current|strong} {smoke|dev|locked_test}" >&2; exit 2 ;;
esac
case "$split" in
  smoke|dev|locked_test) ;;
  *) echo "invalid split: $split" >&2; exit 2 ;;
esac

protected_pid="${PROTECTED_PID:-1911}"
if ! kill -0 "$protected_pid" 2>/dev/null; then
  echo "protected PID $protected_pid is not alive; refusing to launch" >&2
  exit 3
fi

root="outputs/pass8_selector_study"
frozen="$root/frozen_v1"
runtime="$root/runtime"
mkdir -p "$runtime/logs" "$runtime/selectors/$variant"

if [[ "$variant" == "current" ]]; then
  physical_gpus="4"
  model_path="checkpoints/Qwen3.5-9B"
  served_model="qwen35-9b-selector"
  tp=1
  port=8734
  threads="${THREADS:-8}"
else
  physical_gpus="4,5,6,7"
  model_path="checkpoints/Qwen3.5-35B-A3B"
  served_model="qwen35-35b-a3b-selector"
  tp=4
  port=8735
  threads="${THREADS:-16}"
fi

IFS=',' read -ra requested_gpus <<< "$physical_gpus"
for gpu in "${requested_gpus[@]}"; do
  if [[ "$gpu" != "4" && "$gpu" != "5" && "$gpu" != "6" && "$gpu" != "7" ]]; then
    echo "forbidden physical GPU requested: $gpu" >&2
    exit 4
  fi
done

server_log="$runtime/logs/server_${variant}_${split}.log"
selector_log="$runtime/logs/selector_${variant}_${split}.log"
output="$runtime/selectors/$variant/$split.jsonl"

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="$physical_gpus" .venv-qwen35-vllm/bin/vllm serve "$model_path" \
  --served-model-name "$served_model" \
  --host 127.0.0.1 \
  --port "$port" \
  --trust-remote-code \
  --tensor-parallel-size "$tp" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.65 \
  --kv-cache-memory-bytes 8G \
  --enforce-eager \
  --limit-mm-per-prompt '{"image":1}' \
  --reasoning-parser qwen3 \
  >"$server_log" 2>&1 &
server_pid=$!

.venv-qwen35-vllm/bin/python - "$port" "$server_pid" <<'PY'
import json
import os
import sys
import time
import urllib.request

port, pid = int(sys.argv[1]), int(sys.argv[2])
deadline = time.time() + 1800
while time.time() < deadline:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3) as response:
            json.load(response)
        raise SystemExit(0)
    except Exception:
        try:
            os.kill(pid, 0)
        except OSError:
            raise SystemExit("vLLM server exited before becoming ready")
        time.sleep(5)
raise SystemExit("timed out waiting for vLLM server")
PY

.venv-qwen35-vllm/bin/python scripts/run_pass8_selector.py \
  --manifest "$frozen/manifest.json" \
  --blind "$frozen/blind/$split.jsonl" \
  --output "$output" \
  --selector-name "$variant" \
  --model "$served_model" \
  --api-urls "http://127.0.0.1:$port/v1" \
  --threads "$threads" \
  --max-tokens 512 \
  2>&1 | tee "$selector_log"

echo "completed selector=$variant split=$split output=$output"