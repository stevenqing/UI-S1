#!/usr/bin/env bash
set -euo pipefail

run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
stage_dir="$(cd "$run_dir/.." && pwd)"
workspace="$(cd "$stage_dir/../../.." && pwd)"
python="$workspace/runs/mind2web-tongui/2026-07-28/.venv/bin/python"
analysis_python="$workspace/.venv-ac-vllm/bin/python"
overlay="$workspace/runs/collision-law/2026-07-30/w3_assets/mvp-overlay"
mvp="$workspace/runs/collision-law/2026-07-30/w3_assets/MVP"
shards="$run_dir/shards/top18"
raw="$run_dir/raw"
logs="$run_dir/logs"
mkdir -p "$shards" "$raw" "$logs"

[[ -x "$python" && -x "$analysis_python" && -d "$overlay/transformers" && -d "$mvp/.git" ]] || {
  echo "missing pinned H1 runtime" >&2
  exit 2
}

export PYTHONPATH="$run_dir:$stage_dir:$mvp:$overlay${PYTHONPATH:+:$PYTHONPATH}"
pids=()
for shard in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$shard" "$python" "$run_dir/generate_candidates.py" \
    --output "$shards/shard-$shard.jsonl" --num-shards 8 --shard-index "$shard" \
    --max-subimages 18 --resume \
    >"$logs/generate-shard-$shard.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || { echo "H1 candidate shard failed" >&2; exit "$status"; }

PYTHONPATH="$run_dir:$stage_dir:$workspace/runs/collision-law/2026-07-30" \
  "$analysis_python" "$run_dir/merge_candidates.py" \
  --shard-root "$shards" --output-dir "$raw" --num-shards 8

PYTHONPATH="$run_dir:$stage_dir:$workspace/runs/collision-law/2026-07-30" \
  "$analysis_python" "$run_dir/h1_eval.py" \
  --candidate-dir "$raw" --output "$stage_dir/h1_headtohead.json"
