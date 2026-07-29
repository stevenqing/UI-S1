#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ( "$1" != "smoke" && "$1" != "full" ) ]]; then
  echo "usage: $0 {smoke|full}" >&2
  exit 2
fi

run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$run_dir/../../.." && pwd)"
python="$workspace/.venv-ac-vllm/bin/python"
model="$run_dir/models/UI-TARS-72B-SFT"
metadata="$workspace/runs/mind2web-showui/2026-07-28/data/Mind2Web/metadata/hf_test_task.json"
images="$workspace/runs/mind2web/2026-07-27/data/ming2web_images"
output="$run_dir/artifacts/72b/$1"
limit_args=()
if [[ "$1" == "smoke" ]]; then
  limit_args=(--limit 1)
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 "$python" "$run_dir/infer.py" \
  --model-dir "$model" \
  --model-name ByteDance-Seed/UI-TARS-72B-SFT \
  --metadata "$metadata" \
  --image-root "$images" \
  --output-dir "$output" \
  --tensor-parallel-size 4 \
  --kv-cache-memory-bytes 1073741824 \
  --batch-size 4 \
  --enforce-eager \
  --resume \
  "${limit_args[@]}"