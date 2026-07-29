#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 || ( "$1" != "smoke" && "$1" != "full" && "$1" != "audit" ) ]]; then
  echo "usage: $0 {smoke|full|audit}" >&2
  exit 2
fi

run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$run_dir/../../.." && pwd)"
source_dir="$workspace/runs/mind2web/2026-07-27/repos/Mind2Web/src"
common_args=(
  --source-dir "$source_dir"
  --data-dir "$run_dir/data/Mind2Web"
  --score-file "$run_dir/data/source/scores_all_data.pkl"
  --model-dir "$run_dir/model/flan-t5-xl"
  --tokenizer-dir "$run_dir/model/flan-t5-xl-tokenizer"
)

if [[ "$1" == "audit" ]]; then
  "$run_dir/run_python.sh" "$run_dir/audit.py" \
    --source-dir "$source_dir" \
    --data-dir "$run_dir/data/Mind2Web" \
    --score-file "$run_dir/data/source/scores_all_data.pkl" \
    --predictions "$run_dir/artifacts/full/test_task_predictions_top50.json" \
    --results "$run_dir/artifacts/full/test_task_results_top50.json" \
    --provenance "$run_dir/artifacts/full/provenance.json" \
    --manifest "$run_dir/artifact_manifest.json" \
    --output "$run_dir/artifacts/full/audit.json"
  exit
fi

output="$run_dir/artifacts/$1"
limit_args=()
if [[ "$1" == "smoke" ]]; then
  limit_args=(--limit 3)
fi
if [[ "$1" == "full" ]]; then
  shard_root="$run_dir/artifacts/full-shards"
  mkdir -p "$shard_root"
  pids=()
  for shard_index in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$shard_index" "$run_dir/run_python.sh" "$run_dir/run_mindact.py" \
      "${common_args[@]}" \
      --output-dir "$shard_root/shard-$shard_index" \
      --num-shards 4 \
      --shard-index "$shard_index" \
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
  "$run_dir/run_python.sh" "$run_dir/merge_shards.py" \
    --source-dir "$source_dir" \
    --data-dir "$run_dir/data/Mind2Web" \
    --score-file "$run_dir/data/source/scores_all_data.pkl" \
    --shard-root "$shard_root" \
    --output-dir "$run_dir/artifacts/full"
  exit
fi
CUDA_VISIBLE_DEVICES=0 "$run_dir/run_python.sh" "$run_dir/run_mindact.py" \
  "${common_args[@]}" \
  --output-dir "$output" \
  "${limit_args[@]}"