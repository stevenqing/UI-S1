#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv-qwen3-vllm/bin/python}"
DATA_ROOT="${DATA_ROOT:-outputs/multiagent_trajectory_revision/full_v1/lora_screen/data}"
OUT="${OUT:-outputs/multiagent_trajectory_revision/full_v1/lora_screen}"
TEST_EPISODES="${TEST_EPISODES:-outputs/multiagent_trajectory_revision/full_v1/test_eval_episodes.jsonl}"
BASELINE="${BASELINE:-outputs/validation_2k/eval_sft_greedy.jsonl}"
MODEL_PATH="${MODEL_PATH:-checkpoints/gui360-fullparam-sft-step250}"
PROTECTED_PID="${PROTECTED_PID:-1911}"
MAX_STEPS="${MAX_STEPS:-100}"
EVAL_SHARD_INDEX="${EVAL_SHARD_INDEX:-0}"
EVAL_SHARD_COUNT="${EVAL_SHARD_COUNT:-8}"

ARMS=(
  a1_gt_target_gt_history
  a2_random_target_gt_history
  a4_revision_target_revision_history
  a5_revision_target_gt_history
  a6_gt_target_revision_history
  a7_revision_clean_prefix
  a9_revision_internvl3_only
  a10_revision_qwen3_vl_only
)

mkdir -p "$OUT/logs"
rm -rf "$OUT/models" "$OUT/eval" "$OUT/report"
mkdir -p "$OUT/models" "$OUT/eval" "$OUT/report"

if ps -p "$PROTECTED_PID" >/dev/null 2>&1; then
  ps -p "$PROTECTED_PID" -o pid=,stat=,etime= > "$OUT/protected_pid_before.txt"
fi

export PYTHONPATH="$ROOT/train_GUI_360/compat${PYTHONPATH:+:$PYTHONPATH}"

train_pids=()
for index in "${!ARMS[@]}"; do
  arm="${ARMS[$index]}"
  data="$DATA_ROOT/$arm.jsonl"
  [[ -s "$data" ]] || { echo "missing screen data: $data" >&2; exit 2; }
  MASTER_ADDR=127.0.0.1 MASTER_PORT="$((29700 + index))" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
  CUDA_VISIBLE_DEVICES="$index" "$PYTHON_BIN" scripts/train_hetero_inject_sft.py \
    --model_path "$MODEL_PATH" --train_data "$data" --output_dir "$OUT/models/$arm" \
    --max_steps "$MAX_STEPS" --num_epochs 1 --gradient_accumulation_steps 8 \
    --logging_steps 10 --save_steps 0 --lora_r 64 --lora_alpha 128 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --trainable_mode lora_only --image_max_pixels 602112 \
    --label_smoothing 0.0 --kl_weight 0.0 --entropy_bonus 0.0 \
    --lora_lr 2e-5 --seed 42 > "$OUT/logs/train_${arm}.log" 2>&1 &
  train_pids+=("$!")
done

status=0
for pid in "${train_pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if (( status != 0 )); then
  echo "one or more LoRA screen training arms failed" >&2
  exit "$status"
fi

"$PYTHON_BIN" - "$OUT" "$MAX_STEPS" "${ARMS[@]}" <<'PY'
import sys, torch
from pathlib import Path
root=Path(sys.argv[1])
expected=int(sys.argv[2])
for arm in sys.argv[3:]:
    path=root/'models'/arm/'final'/'training_state.pt'
    state=torch.load(path,map_location='cpu',weights_only=True)
  if int(state.get('global_step',-1)) != expected:
        raise SystemExit(f'{arm}: {state}')
print(f'[train] all arms reached global_step={expected}')
PY

eval_pids=()
for index in "${!ARMS[@]}"; do
  arm="${ARMS[$index]}"
  CUDA_VISIBLE_DEVICES="$index" "$PYTHON_BIN" scripts/evaluate_multiagent_revision_pilot.py \
    --arm "$arm" --episode-data "$TEST_EPISODES" \
    --output "$OUT/eval/$arm.jsonl" --batch-size 8 --seed 42 \
    --model-path "$MODEL_PATH" --adapter-dir "$OUT/models/$arm/final/cooperative" \
    --shard-index "$EVAL_SHARD_INDEX" --shard-count "$EVAL_SHARD_COUNT" \
    > "$OUT/logs/eval_${arm}.log" 2>&1 &
  eval_pids+=("$!")
done

status=0
for pid in "${eval_pids[@]}"; do
  if ! wait "$pid"; then status=1; fi
done
if (( status != 0 )); then
  echo "one or more LoRA screen evaluation arms failed" >&2
  exit "$status"
fi

"$PYTHON_BIN" scripts/report_revision_lora_screen.py \
  --baseline "$BASELINE" --test-episodes "$TEST_EPISODES" \
  --post-glob "$OUT/eval/*.jsonl" --data-manifest "$DATA_ROOT/manifest.json" \
  --training-root "$OUT/models" --output-dir "$OUT/report" \
  --shard-index "$EVAL_SHARD_INDEX" --shard-count "$EVAL_SHARD_COUNT" \
  --bootstrap-draws 10000 --seed 42 | tee "$OUT/logs/report.log"

if ps -p "$PROTECTED_PID" >/dev/null 2>&1; then
  ps -p "$PROTECTED_PID" -o pid=,stat=,etime= > "$OUT/protected_pid_after.txt"
fi

echo "[complete] equal-budget revision LoRA screen"
