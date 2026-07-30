#!/usr/bin/env bash
set -euo pipefail

run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$run_dir/../../.." && pwd)"
python="$workspace/.venv-ac-vllm/bin/python"
upstream="$workspace/runs/androidcontrol-rft/2026-07-29"

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 {ui-agile-3b|ui-agile-7b|ui-r1-e-3b|gui-r1-3b|gui-r1-7b} {low|high}" >&2
  exit 2
fi
model="$1"; setting="$2"
cell="$run_dir/w4_artifacts/$model/$setting"; shards="$cell/shards"
mkdir -p "$shards"
pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard" "$python" "$run_dir/w4_curated.py" infer \
    --model "$model" --setting "$setting" --output "$shards/shard-$shard.jsonl" \
    --num-shards 4 --shard-index "$shard" --batch-size 16 \
    --kv-cache-memory-bytes 2147483648 --resume \
    >"$shards/shard-$shard.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || exit "$status"
if [[ ! -f "$cell/predictions.jsonl" ]]; then
  "$python" "$upstream/merge_shards.py" --shard-root "$shards" \
    --output "$cell/predictions.jsonl" --num-shards 4 --expected-rows 8377
fi
"$python" "$run_dir/w4_curated.py" score --predictions "$cell/predictions.jsonl" \
  --setting "$setting" --output "$cell/score.json" --require-complete