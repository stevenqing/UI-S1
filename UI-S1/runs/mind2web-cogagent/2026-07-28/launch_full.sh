#!/usr/bin/env bash
set -euo pipefail

run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$run_dir/../../.." && pwd)"
model_dir="${MODEL_DIR:-$HOME/.cache/huggingface/hub/models--zai-org--cogagent-chat-hf/snapshots/26eec27a44348fbe0c9fad89348cf6a505f5a5ae}"
tokenizer_dir="${TOKENIZER_DIR:-$HOME/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d}"
metadata="$workspace/runs/mind2web-showui/2026-07-28/data/Mind2Web/metadata/hf_test_task.json"
image_root="$workspace/runs/mind2web/2026-07-27/data/ming2web_images"
artifact_root="$run_dir/artifacts/full-shards"

mkdir -p "$artifact_root"
pids=()
for shard_index in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard_index" "$run_dir/run_python.sh" "$run_dir/infer.py" \
    --model-dir "$model_dir" \
    --tokenizer-dir "$tokenizer_dir" \
    --metadata "$metadata" \
    --image-root "$image_root" \
    --output-dir "$artifact_root/shard-$shard_index" \
    --num-shards 4 \
    --shard-index "$shard_index" \
    --resume \
    --max-new-tokens 256 \
    >"$artifact_root/shard-$shard_index.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"