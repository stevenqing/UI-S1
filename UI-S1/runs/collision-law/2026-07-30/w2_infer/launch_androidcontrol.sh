#!/usr/bin/env bash
set -euo pipefail

run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$run_dir/../../../.." && pwd)"
python="$workspace/.venv-ac-vllm/bin/python"
artifact_root="$run_dir/../w2_artifacts/androidcontrol"
upstream="$workspace/runs/androidcontrol-rft/2026-07-29"
gpu_memory_utilization="${W2_GPU_MEMORY_UTILIZATION:-0.65}"

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 {gui-r1-7b|ui-agile-7b} {low|high} {v1|v2|v3|v4}" >&2
  exit 2
fi

model="$1"
setting="$2"
view="$3"
case "$model" in gui-r1-7b|ui-agile-7b) ;; *) exit 2 ;; esac
case "$setting" in low|high) ;; *) exit 2 ;; esac
case "$view" in v1|v2|v3|v4) ;; *) exit 2 ;; esac

cell="$artifact_root/$model/$setting/$view"
shards="$cell/shards"
if [[ "$model/$setting/$view" == "gui-r1-7b/high/v4" ]]; then
  inherited="$workspace/runs/complementarity/2026-07-30/e5_artifacts/androidcontrol/original_768/predictions.jsonl"
  [[ -f "$inherited" ]] || { echo "missing preregistered inherited v4 cell" >&2; exit 3; }
  mkdir -p "$cell"
  PYTHONPATH="$run_dir/.." "$python" "$run_dir/score_androidcontrol.py" \
    --predictions "$inherited" --output "$cell/score.json" --require-complete \
    --model "$model" --view-id "$view"
  exit 0
fi
mkdir -p "$shards"
pids=()
for shard in 0 1 2 3; do
  PYTHONPATH="$run_dir/..:$run_dir" CUDA_VISIBLE_DEVICES="$shard" \
    "$python" "$run_dir/infer_androidcontrol.py" \
      --model "$model" --setting "$setting" --view "$view" \
      --output "$shards/shard-$shard.jsonl" \
      --num-shards 4 --shard-index "$shard" --batch-size 16 \
      --kv-cache-memory-bytes 2147483648 \
      --gpu-memory-utilization "$gpu_memory_utilization" --resume \
      >"$shards/shard-$shard.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
[[ "$status" -eq 0 ]] || exit "$status"

if [[ ! -f "$cell/predictions.jsonl" ]]; then
  "$python" "$upstream/merge_shards.py" \
    --shard-root "$shards" --output "$cell/predictions.jsonl" \
    --num-shards 4 --expected-rows 7708
fi
PYTHONPATH="$run_dir/.." "$python" "$run_dir/score_androidcontrol.py" \
  --predictions "$cell/predictions.jsonl" --output "$cell/score.json" --require-complete