#!/usr/bin/env bash
set -euo pipefail

run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$run_dir/../../../.." && pwd)"
python="$workspace/runs/mind2web-tongui/2026-07-28/.venv/bin/python"
overlay="$run_dir/../w3_assets/mvp-overlay"
cell="$run_dir/../w3_artifacts/gta1_self_consistency_n5_screenspot_pro"
[[ -x "$python" && -d "$overlay/transformers" ]] || { echo "missing MVP runtime" >&2; exit 2; }
export PYTHONPATH="$run_dir/..:$overlay${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$cell/shards"
pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard" "$python" "$run_dir/gta1_sanity.py" infer \
    --output "$cell/shards/shard-$shard.jsonl" --num-shards 4 --shard-index "$shard" \
    --samples 5 --temperature 0.7 --seed 20260730 --resume \
    >"$cell/shards/shard-$shard.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || exit "$status"
if [[ ! -f "$cell/predictions.jsonl" ]]; then
  "$python" "$run_dir/gta1_sanity.py" merge \
    --shard-root "$cell/shards" --output "$cell/predictions.jsonl"
fi
"$python" "$run_dir/gta1_sanity.py" score \
  --predictions "$cell/predictions.jsonl" --output "$cell/score.json"