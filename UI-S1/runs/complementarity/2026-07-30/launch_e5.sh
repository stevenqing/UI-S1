#!/usr/bin/env bash
set -euo pipefail

run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$run_dir/../../.." && pwd)"
ac_run="$workspace/runs/androidcontrol-rft/2026-07-29"
m2w_run="$workspace/runs/mind2web-tongui/2026-07-28"
ac_python="$workspace/.venv-ac-vllm/bin/python"
m2w_python="$m2w_run/.venv/bin/python"
artifact_root="$run_dir/e5_artifacts"

for required in "$ac_run/infer.py" "$m2w_run/infer.py" \
  "$ac_run/models/GUI-R1/GUI-R1-7B" "$m2w_run/models/TongUI-7B"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required E5 input: $required" >&2
    exit 1
  fi
done

if [[ "$#" -ne 1 || ( "$1" != "androidcontrol" && "$1" != "mind2web" && "$1" != "all" ) ]]; then
  echo "usage: $0 {androidcontrol|mind2web|all}" >&2
  exit 2
fi

run_ac_cell() {
  local prompt_variant="$1"
  local max_tokens="$2"
  local cell="$artifact_root/androidcontrol/${prompt_variant}_${max_tokens}"
  local shards="$cell/shards"
  mkdir -p "$shards"
  local pids=()
  for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$shard" "$ac_python" "$ac_run/infer.py" \
      --model-dir "$ac_run/models/GUI-R1/GUI-R1-7B" \
      --model-name "ritzzai/GUI-R1:GUI-R1-7B" \
      --data-path "$ac_run/data/UI-AGILE-Data/android_control/androidcontrol_high_test.parquet" \
      --data-setting high \
      --prompt-template gui_r1 \
      --prompt-variant "$prompt_variant" \
      --max-visual-tokens "$max_tokens" \
      --output "$shards/shard-$shard.jsonl" \
      --num-shards 4 --shard-index "$shard" --batch-size 16 \
      --kv-cache-memory-bytes 2147483648 --resume \
      >"$shards/shard-$shard.log" 2>&1 &
    pids+=("$!")
  done
  local status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  [[ "$status" -eq 0 ]] || return "$status"
  "$ac_python" "$ac_run/merge_shards.py" --shard-root "$shards" --output "$cell/predictions.jsonl"
  "$ac_python" "$ac_run/score.py" --predictions "$cell/predictions.jsonl" --output "$cell/score.json" --require-complete
}

run_m2w_cell() {
  local prompt_variant="$1"
  local max_tokens="$2"
  local cell="$artifact_root/mind2web/${prompt_variant}_${max_tokens}"
  local shards="$cell/shards"
  mkdir -p "$shards"
  local pids=()
  for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$shard" "$m2w_python" "$m2w_run/infer.py" \
      --model-dir "$m2w_run/models/TongUI-7B" \
      --model-name "TongUI-7B" \
      --model-revision "a3e0cf46c3164bbd885dea2694f2ad7a31f1661d" \
      --data-root "$m2w_run/data/evaluation_data" \
      --repo-dir "$m2w_run/repos/TongUI-agent" \
      --output-dir "$shards/shard-$shard" \
      --num-shards 4 --shard-index "$shard" --resume \
      --prompt-variant "$prompt_variant" \
      --max-visual-tokens "$max_tokens" \
      >"$shards/shard-$shard.log" 2>&1 &
    pids+=("$!")
  done
  local status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  [[ "$status" -eq 0 ]] || return "$status"
  if [[ ! -f "$cell/predictions.jsonl" ]]; then
    "$m2w_python" "$m2w_run/merge_predictions.py" --shard-root "$shards" --output-dir "$cell"
  fi
  "$m2w_python" "$m2w_run/score.py" --predictions "$cell/predictions.jsonl" --output "$cell/score.json"
}

run_benchmark() {
  local benchmark="$1"
  local original_tokens
  if [[ "$benchmark" == "androidcontrol" ]]; then
    original_tokens=12800
  else
    original_tokens=1344
  fi
  for prompt in original please_carry_out your_objective; do
    for tokens in "$original_tokens" 768; do
      if [[ "$prompt" == "original" && "$tokens" == "$original_tokens" ]]; then
        continue
      fi
      if [[ "$benchmark" == "androidcontrol" ]]; then
        run_ac_cell "$prompt" "$tokens"
      else
        run_m2w_cell "$prompt" "$tokens"
      fi
    done
  done
}

if [[ "$1" == "all" ]]; then
  run_benchmark androidcontrol
  run_benchmark mind2web
else
  run_benchmark "$1"
fi

"$ac_python" "$run_dir/e5_noise.py" \
  --artifact-root "$artifact_root" \
  --output "$run_dir/e5_noise.json" \
  --table "$run_dir/transition_table.md"