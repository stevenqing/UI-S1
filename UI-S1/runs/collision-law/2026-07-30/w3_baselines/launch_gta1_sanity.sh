#!/usr/bin/env bash
set -euo pipefail
run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$run_dir/../../../.." && pwd)"
python="$workspace/.venv-ac-vllm/bin/python"
cell="$run_dir/../w3_artifacts/gta1_screenspot_pro"
mkdir -p "$cell/shards"
pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard" "$python" "$run_dir/gta1_sanity.py" infer \
    --output "$cell/shards/shard-$shard.jsonl" --num-shards 4 --shard-index "$shard" --resume \
    >"$cell/shards/shard-$shard.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || exit "$status"
if [[ ! -f "$cell/predictions.jsonl" ]]; then
  "$python" "$run_dir/gta1_sanity.py" merge --shard-root "$cell/shards" --output "$cell/predictions.jsonl"
fi
"$python" "$run_dir/gta1_sanity.py" score --predictions "$cell/predictions.jsonl" --output "$cell/score.json"