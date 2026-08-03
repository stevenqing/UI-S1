#!/usr/bin/env bash
set -euo pipefail

run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$run_dir/../../../.." && pwd)"
python="$workspace/runs/mind2web-tongui/2026-07-28/.venv/bin/python"
analysis_python="$workspace/.venv-ac-vllm/bin/python"
artifact_root="$run_dir/../w2_artifacts/mind2web/tongui-7b"
upstream="$workspace/runs/mind2web-tongui/2026-07-28"

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 {v1|v2|v3|v4}" >&2
  exit 2
fi
view="$1"
case "$view" in v1|v2|v3|v4) ;; *) exit 2 ;; esac

cell="$artifact_root/$view"
shards="$cell/shards"
mkdir -p "$shards"
pids=()
for shard in 0 1 2 3; do
  PYTHONPATH="$workspace:$run_dir/..:$run_dir" CUDA_VISIBLE_DEVICES="$shard" \
    "$python" "$run_dir/infer_mind2web.py" \
      --view "$view" --output-dir "$shards/shard-$shard" \
      --num-shards 4 --shard-index "$shard" --resume \
      >"$shards/shard-$shard.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || exit "$status"
if [[ ! -f "$cell/predictions.jsonl" ]]; then
  "$python" "$upstream/merge_predictions.py" --shard-root "$shards" --output-dir "$cell"
fi
PYTHONPATH="$run_dir/.." "$analysis_python" "$run_dir/score_mind2web.py" \
  --predictions "$cell/predictions.jsonl" --output "$cell/score.json" \
  --scored-rows "$cell/scored_rows.jsonl" --require-complete