#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 || ( "$1" != "2b" && "$1" != "7b" ) || ( "$2" != "low" && "$2" != "high" ) ]]; then
  echo "usage: $0 {2b|7b} {low|high}" >&2
  exit 2
fi
run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$run_dir/../../.." && pwd)"
if [[ "$1" == "2b" ]]; then
  model_name="ByteDance-Seed/UI-TARS-2B-SFT"
  model_dir="$workspace/runs/mind2web-tongui/2026-07-28/models/UI-TARS-2B-SFT"
else
  model_name="ByteDance-Seed/UI-TARS-7B-SFT"
  model_dir="$workspace/runs/mind2web-tongui/2026-07-28/models/UI-TARS-7B-SFT"
fi
python="$workspace/.venv-ac-vllm/bin/python"
data="$workspace/runs/androidcontrol-pro/2026-07-27/data/prepared/ac_high.jsonl"
image_root="$workspace/runs/androidcontrol-pro/2026-07-27"
artifact_root="$run_dir/artifacts/$1-$2/full-shards"
mkdir -p "$artifact_root"
pids=()
for shard_index in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard_index" "$python" "$run_dir/infer.py" \
    --data "$data" --image-root "$image_root" \
    --model-dir "$model_dir" --model-name "$model_name" --setting "$2" \
    --output-dir "$artifact_root/shard-$shard_index" \
    --num-shards 4 --shard-index "$shard_index" --resume \
    >"$artifact_root/shard-$shard_index.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"