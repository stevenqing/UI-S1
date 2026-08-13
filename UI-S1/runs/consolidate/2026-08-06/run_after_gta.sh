#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$root"

run_dir="runs/consolidate/2026-08-06"
regions="$run_dir/raw/q1_regions.jsonl"
crops="$run_dir/raw/q2b_crops.jsonl"
logs="$run_dir/logs"
mkdir -p "$logs"

guard_external() {
  if [[ ! -d /proc/2274 ]]; then
    echo "protected PID 2274 is not alive; refusing to continue" >&2
    return 1
  fi
}

validate_rows() {
  local directory="$1"
  local expected="${2:-1581}"
  local total=0
  local file
  for file in "$directory"/shard-*.jsonl; do
    [[ -f "$file" ]] || continue
    total=$((total + $(wc -l < "$file")))
  done
  if [[ "$total" -ne "$expected" ]]; then
    echo "row-count mismatch: $directory has $total, expected $expected" >&2
    return 1
  fi
  echo "validated $directory rows=$total"
}

run_q1_qwen3() {
  local output_dir="$run_dir/raw/q1-qwen3"
  mkdir -p "$output_dir"
  local pids=()
  local shard
  for shard in 0 1 2 3 4 5 6 7; do
    env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$shard" \
      .venv-qwen3-vllm/bin/python "$run_dir/q1_infer_crops.py" \
      --regions "$regions" \
      --model-dir runs/ccm-h2h/2026-07-31/h3/models/Qwen3-VL-8B-Instruct \
      --model-type qwen3 --model-id Qwen3-VL-8B-Instruct \
      --model-revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
      --output "$output_dir/shard-$shard.jsonl" --num-shards 8 --shard-index "$shard" --resume \
      >"$logs/q1-qwen3-shard-$shard.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  local pid
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" -eq 0 ]]
  validate_rows "$output_dir"
}

run_q1_uitars() {
  local output_dir="$run_dir/raw/q1-uitars"
  mkdir -p "$output_dir"
  local pids=()
  local shard
  for shard in 0 1 2 3 4 5 6 7; do
    env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$shard" \
      runs/mind2web-tongui/2026-07-28/.venv/bin/python "$run_dir/q1_infer_crops.py" \
      --regions "$regions" \
      --model-dir runs/mind2web-tongui/2026-07-28/models/UI-TARS-7B-SFT \
      --model-type uitars --model-id UI-TARS-7B-SFT \
      --model-revision 3434901a9dd04dd3625617d839a5724fe5e2db20 \
      --output "$output_dir/shard-$shard.jsonl" --num-shards 8 --shard-index "$shard" --resume \
      >"$logs/q1-uitars-shard-$shard.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  local pid
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" -eq 0 ]]
  validate_rows "$output_dir"
}

run_q2b_smoke() {
  rm -f "$run_dir/raw/q2b-smoke-gta1.jsonl" "$run_dir/raw/q2b-smoke-qwen3.jsonl" "$run_dir/raw/q2b-smoke-uitars.jsonl"
  CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="$root/runs/collision-law/2026-07-30/w3_assets/MVP:$root/runs/collision-law/2026-07-30/w3_assets/mvp-overlay" \
    runs/mind2web-tongui/2026-07-28/.venv/bin/python "$run_dir/q2b_infer_verification.py" \
    --crops "$crops" --model-dir runs/collision-law/2026-07-30/w3_assets/GTA1-7B \
    --model-type gta1 --model-id GTA1-7B --model-revision 701bedc80b447863bd60e3318ae44f6cbbfafd78 \
    --output "$run_dir/raw/q2b-smoke-gta1.jsonl" --num-shards 1 --shard-index 0 --limit 1 \
    >"$logs/q2b-smoke-gta1.log" 2>&1
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 \
    .venv-qwen3-vllm/bin/python "$run_dir/q2b_infer_verification.py" \
    --crops "$crops" --model-dir runs/ccm-h2h/2026-07-31/h3/models/Qwen3-VL-8B-Instruct \
    --model-type qwen3 --model-id Qwen3-VL-8B-Instruct --model-revision 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
    --output "$run_dir/raw/q2b-smoke-qwen3.jsonl" --num-shards 1 --shard-index 0 --limit 1 \
    >"$logs/q2b-smoke-qwen3.log" 2>&1
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES=0 \
    runs/mind2web-tongui/2026-07-28/.venv/bin/python "$run_dir/q2b_infer_verification.py" \
    --crops "$crops" --model-dir runs/mind2web-tongui/2026-07-28/models/UI-TARS-7B-SFT \
    --model-type uitars --model-id UI-TARS-7B-SFT --model-revision 3434901a9dd04dd3625617d839a5724fe5e2db20 \
    --output "$run_dir/raw/q2b-smoke-uitars.jsonl" --num-shards 1 --shard-index 0 --limit 1 \
    >"$logs/q2b-smoke-uitars.log" 2>&1
  [[ $(wc -l < "$run_dir/raw/q2b-smoke-gta1.jsonl") -eq 1 ]]
  [[ $(wc -l < "$run_dir/raw/q2b-smoke-qwen3.jsonl") -eq 1 ]]
  [[ $(wc -l < "$run_dir/raw/q2b-smoke-uitars.jsonl") -eq 1 ]]
}

run_q2b_model() {
  local model_type="$1"
  local model_id="$2"
  local revision="$3"
  local model_dir="$4"
  local runtime="$5"
  local output_name="$6"
  local output_dir="$run_dir/raw/$output_name"
  mkdir -p "$output_dir"
  local pids=()
  local shard
  for shard in 0 1 2 3 4 5 6 7; do
    if [[ "$model_type" == "gta1" ]]; then
      CUDA_VISIBLE_DEVICES="$shard" \
        PYTHONPATH="$root/runs/collision-law/2026-07-30/w3_assets/MVP:$root/runs/collision-law/2026-07-30/w3_assets/mvp-overlay" \
        "$runtime" "$run_dir/q2b_infer_verification.py" \
        --crops "$crops" --model-dir "$model_dir" --model-type "$model_type" --model-id "$model_id" \
        --model-revision "$revision" --output "$output_dir/shard-$shard.jsonl" --num-shards 8 --shard-index "$shard" --resume \
        >"$logs/$output_name-shard-$shard.log" 2>&1 &
    else
      env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$shard" \
        "$runtime" "$run_dir/q2b_infer_verification.py" \
        --crops "$crops" --model-dir "$model_dir" --model-type "$model_type" --model-id "$model_id" \
        --model-revision "$revision" --output "$output_dir/shard-$shard.jsonl" --num-shards 8 --shard-index "$shard" --resume \
        >"$logs/$output_name-shard-$shard.log" 2>&1 &
    fi
    pids+=("$!")
  done
  local failed=0
  local pid
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  [[ "$failed" -eq 0 ]]
  validate_rows "$output_dir"
}

guard_external
validate_rows "$run_dir/raw/q1-gta1"
run_q1_qwen3
guard_external
run_q1_uitars
guard_external
.venv-scaleup/bin/python "$run_dir/q1_sequential.py" >"$logs/q1-evaluate.log" 2>&1

run_q2b_smoke
guard_external
run_q2b_model gta1 GTA1-7B 701bedc80b447863bd60e3318ae44f6cbbfafd78 \
  runs/collision-law/2026-07-30/w3_assets/GTA1-7B \
  runs/mind2web-tongui/2026-07-28/.venv/bin/python q2b-gta1
guard_external
run_q2b_model qwen3 Qwen3-VL-8B-Instruct 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  runs/ccm-h2h/2026-07-31/h3/models/Qwen3-VL-8B-Instruct \
  .venv-qwen3-vllm/bin/python q2b-qwen3
guard_external
run_q2b_model uitars UI-TARS-7B-SFT 3434901a9dd04dd3625617d839a5724fe5e2db20 \
  runs/mind2web-tongui/2026-07-28/models/UI-TARS-7B-SFT \
  runs/mind2web-tongui/2026-07-28/.venv/bin/python q2b-uitars
guard_external

.venv-scaleup/bin/python "$run_dir/q2b_verification.py" >"$logs/q2b-evaluate.log" 2>&1
.venv-scaleup/bin/python "$run_dir/finalize.py" >"$logs/finalize.log" 2>&1
guard_external
echo "Q pipeline PASS"