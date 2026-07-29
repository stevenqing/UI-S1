#!/usr/bin/env bash
set -euo pipefail

run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$run_dir/../../.." && pwd)"
python="$workspace/.venv-ac-vllm/bin/python"
kv_cache_memory_bytes=2147483648

model_config() {
  case "$1" in
    ui-agile-3b)
      model_dir="$run_dir/models/UI-AGILE-3B"
      model_name="KDEGroup/UI-AGILE-3B"
      prompt_template="android_control_detailed"
      ;;
    ui-agile-7b)
      model_dir="$run_dir/models/UI-AGILE-7B"
      model_name="KDEGroup/UI-AGILE"
      prompt_template="android_control_detailed"
      ;;
    ui-r1-e-3b)
      model_dir="$run_dir/models/UI-R1-E-3B"
      model_name="LZXzju/Qwen2.5-VL-3B-UI-R1-E"
      prompt_template="ui_r1"
      ;;
    gui-r1-3b)
      model_dir="$run_dir/models/GUI-R1/GUI-R1-3B"
      model_name="ritzzai/GUI-R1:GUI-R1-3B"
      prompt_template="gui_r1"
      ;;
    gui-r1-7b)
      model_dir="$run_dir/models/GUI-R1/GUI-R1-7B"
      model_name="ritzzai/GUI-R1:GUI-R1-7B"
      prompt_template="gui_r1"
      ;;
    *)
      echo "unknown model key: $1" >&2
      exit 2
      ;;
  esac
}

run_setting() {
  local mode="$1"
  local model_key="$2"
  local setting="$3"
  model_config "$model_key"
  local data_path="$run_dir/data/UI-AGILE-Data/android_control/androidcontrol_${setting}_test.parquet"
  local artifact_root="$run_dir/artifacts/$model_key/$setting"
  local shard_root="$artifact_root/shards"
  mkdir -p "$shard_root"
  if [[ "$mode" == "smoke" ]]; then
    CUDA_VISIBLE_DEVICES=0 "$python" "$run_dir/infer.py" \
      --model-dir "$model_dir" \
      --model-name "$model_name" \
      --data-path "$data_path" \
      --data-setting "$setting" \
      --prompt-template "$prompt_template" \
      --output "$artifact_root/smoke.jsonl" \
      --num-shards 4 \
      --shard-index 0 \
      --batch-size 16 \
      --limit 1 \
      --kv-cache-memory-bytes "$kv_cache_memory_bytes" \
      --resume
    return
  fi

  local pids=()
  for shard_index in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$shard_index" "$python" "$run_dir/infer.py" \
      --model-dir "$model_dir" \
      --model-name "$model_name" \
      --data-path "$data_path" \
      --data-setting "$setting" \
      --prompt-template "$prompt_template" \
      --output "$shard_root/shard-$shard_index.jsonl" \
      --num-shards 4 \
      --shard-index "$shard_index" \
      --batch-size 16 \
      --kv-cache-memory-bytes "$kv_cache_memory_bytes" \
      --resume \
      >"$shard_root/shard-$shard_index.log" 2>&1 &
    pids+=("$!")
  done
  local status=0
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  if [[ "$status" -ne 0 ]]; then
    return "$status"
  fi
  "$python" "$run_dir/merge_shards.py" \
    --shard-root "$shard_root" \
    --output "$artifact_root/predictions.jsonl"
  "$python" "$run_dir/score.py" \
    --predictions "$artifact_root/predictions.jsonl" \
    --output "$artifact_root/score.json" \
    --require-complete
  "$python" "$run_dir/audit.py" \
    --data-path "$data_path" \
    --data-setting "$setting" \
    --predictions "$artifact_root/predictions.jsonl" \
    --score "$artifact_root/score.json" \
    --manifest "$run_dir/artifact_manifest.json" \
    --model-name "$model_name" \
    --prompt-template "$prompt_template" \
    --kv-cache-memory-bytes "$kv_cache_memory_bytes" \
    --output "$artifact_root/audit.json"
}

if [[ "$#" -eq 1 && "$1" == "all" ]]; then
  "$python" "$run_dir/build_manifest.py"
  for model_key in ui-agile-3b ui-agile-7b ui-r1-e-3b gui-r1-3b gui-r1-7b; do
    for setting in low high; do
      run_setting full "$model_key" "$setting"
    done
  done
elif [[ "$#" -eq 3 && ( "$1" == "smoke" || "$1" == "full" ) && ( "$3" == "low" || "$3" == "high" ) ]]; then
  "$python" "$run_dir/build_manifest.py"
  run_setting "$1" "$2" "$3"
else
  echo "usage: $0 {smoke|full} {ui-agile-3b|ui-agile-7b|ui-r1-e-3b|gui-r1-3b|gui-r1-7b} {low|high}" >&2
  echo "       $0 all" >&2
  exit 2
fi