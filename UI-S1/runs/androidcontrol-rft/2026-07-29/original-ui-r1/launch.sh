#!/usr/bin/env bash
set -euo pipefail

run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$run_dir/../../../.." && pwd)"
python="$workspace/.venv-ac-vllm/bin/python"
model="$run_dir/../models/UI-R1-3B-v1"
source_json="$run_dir/../ui-r1-repo/dataset/ac_test.json"
prepared_images="$workspace/runs/androidcontrol-pro/2026-07-27/data/prepared/images"
kv_cache_memory_bytes=2147483648

if [[ "$#" -ne 1 || ( "$1" != "smoke" && "$1" != "full" ) ]]; then
  echo "usage: $0 {smoke|full}" >&2
  exit 2
fi

if [[ "$1" == "smoke" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$python" "$run_dir/infer.py" \
    --model-dir "$model" \
    --source-json "$source_json" \
    --prepared-images "$prepared_images" \
    --recovered-images "$run_dir/images" \
    --output "$run_dir/artifacts/smoke.jsonl" \
    --shard-index 0 \
    --batch-size 16 \
    --limit 1 \
    --kv-cache-memory-bytes "$kv_cache_memory_bytes" \
    --resume
  exit
fi

shard_root="$run_dir/artifacts/full-shards"
mkdir -p "$shard_root"
pids=()
for shard_index in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard_index" "$python" "$run_dir/infer.py" \
    --model-dir "$model" \
    --source-json "$source_json" \
    --prepared-images "$prepared_images" \
    --recovered-images "$run_dir/images" \
    --output "$shard_root/shard-$shard_index.jsonl" \
    --shard-index "$shard_index" \
    --batch-size 16 \
    --kv-cache-memory-bytes "$kv_cache_memory_bytes" \
    --resume \
    >"$shard_root/shard-$shard_index.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi
"$python" "$run_dir/merge_score_audit.py" \
  --source-json "$source_json" \
  --prepared-images "$prepared_images" \
  --recovered-images "$run_dir/images" \
  --shard-root "$shard_root" \
  --model-dir "$model" \
  --image-manifest "$run_dir/image_manifest.json" \
  --gcs-manifest "$run_dir/../data/official-gcs/official_gcs_manifest.json" \
  --output-dir "$run_dir/artifacts/full" \
  --kv-cache-memory-bytes "$kv_cache_memory_bytes"