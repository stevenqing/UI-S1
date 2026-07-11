#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv-qwen3-vllm/bin/python}"
FULL_ROOT="${FULL_ROOT:-outputs/multiagent_trajectory_revision/full_v1}"
OVERNIGHT="$FULL_ROOT/overnight"
CAUSAL="$FULL_ROOT/causal_arms"
A4_ROOT="$FULL_ROOT/causal_eval/a4_starting_student"
A5_ROOT="$FULL_ROOT/causal_eval/a5_gt_history_grid"
HISTORY_OUT="$FULL_ROOT/causal_analysis/history_intervention"
LORA_ROOT="$FULL_ROOT/lora_screen"
PROTECTED_PID="${PROTECTED_PID:-1911}"
GIT_COMMIT="$(git rev-parse HEAD)"

mkdir -p "$OVERNIGHT/logs"
exec > >(tee -a "$OVERNIGHT/logs/overnight.log") 2>&1

finalize() {
  "$PYTHON_BIN" scripts/summarize_revision_overnight.py \
    --root "$FULL_ROOT" --output-dir "$OVERNIGHT" --git-commit "$GIT_COMMIT" || true
  if ps -p "$PROTECTED_PID" >/dev/null 2>&1; then
    ps -p "$PROTECTED_PID" -o pid=,stat=,etime= > "$OVERNIGHT/protected_pid_final.txt" || true
  fi
}
trap finalize EXIT

require_file() {
  [[ -s "$1" ]] || { echo "required file missing: $1" >&2; exit 2; }
}

run_causal_eval_8way() {
  local arm="$1" input="$2" out="$3" expected="$4"
  if [[ -s "$out/merged.jsonl" && -s "$out/merged.summary.json" ]]; then
    echo "[resume] causal eval $arm"
    return
  fi
  rm -rf "$out"
  mkdir -p "$out/shards" "$out/logs"
  local pids=() status=0
  for shard in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="$shard" "$PYTHON_BIN" scripts/evaluate_revision_causal_arm.py \
      --arm "$arm" --input "$input" --output "$out/shards/shard_${shard}.jsonl" \
      --model-path checkpoints/gui360-fullparam-sft-step250 --batch-size 8 --seed 42 \
      --shard-index "$shard" --shard-count 8 > "$out/logs/shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do if ! wait "$pid"; then status=1; fi; done
  if (( status != 0 )); then echo "$arm causal shard failure" >&2; exit "$status"; fi
  "$PYTHON_BIN" scripts/merge_revision_causal_eval.py \
    --input "$input" --shards "$out/shards/shard_*.jsonl" \
    --output "$out/merged.jsonl" --expected-rows "$expected" > "$out/logs/merge.log"
}

run_full_lora_eval() {
  local arm="$1"
  local out="$LORA_ROOT/full_eval/$arm"
  if [[ -s "$out/report/summary.json" ]]; then
    echo "[resume] full LoRA eval $arm"
    return
  fi
  rm -rf "$out"
  mkdir -p "$out/shards" "$out/logs" "$out/report"
  local adapter="$LORA_ROOT/models/$arm/final/cooperative"
  require_file "$LORA_ROOT/models/$arm/final/training_state.pt"
  [[ -d "$adapter" ]] || { echo "missing adapter $adapter" >&2; exit 2; }
  local pids=() status=0
  for shard in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="$shard" "$PYTHON_BIN" scripts/evaluate_multiagent_revision_pilot.py \
      --arm "$arm" --episode-data "$FULL_ROOT/test_eval_episodes.jsonl" \
      --output "$out/shards/shard_${shard}.jsonl" --batch-size 8 --seed 42 \
      --model-path checkpoints/gui360-fullparam-sft-step250 --adapter-dir "$adapter" \
      --shard-index "$shard" --shard-count 8 > "$out/logs/shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do if ! wait "$pid"; then status=1; fi; done
  if (( status != 0 )); then echo "$arm full LoRA shard failure" >&2; exit "$status"; fi
  "$PYTHON_BIN" scripts/merge_multiagent_revision_eval.py \
    --episode-data "$FULL_ROOT/test_eval_episodes.jsonl" \
    --shards "$out/shards/shard_*.jsonl" --output "$out/merged.jsonl" \
    --expected-episodes 1000 --expected-steps 7498 > "$out/logs/merge.log"
  "$PYTHON_BIN" scripts/report_revision_lora_screen.py \
    --baseline outputs/validation_2k/eval_sft_greedy.jsonl \
    --test-episodes "$FULL_ROOT/test_eval_episodes.jsonl" --post-glob "$out/merged.jsonl" \
    --data-manifest "$LORA_ROOT/data/manifest.json" --training-root "$LORA_ROOT/models" \
    --output-dir "$out/report" --shard-index 0 --shard-count 1 \
    --bootstrap-draws 10000 --seed 42 > "$out/logs/report.log"
}

export PYTHONPATH="$ROOT/train_GUI_360/compat${PYTHONPATH:+:$PYTHONPATH}"
if ps -p "$PROTECTED_PID" >/dev/null 2>&1; then
  ps -p "$PROTECTED_PID" -o pid=,stat=,etime= > "$OVERNIGHT/protected_pid_before.txt"
fi

require_file "$A4_ROOT/merged.jsonl"
require_file "$A4_ROOT/analysis/student_relative_revision_summary.json"
require_file "$CAUSAL/eval_grid_a4_revision_history.jsonl"
require_file "$CAUSAL/eval_grid_a5_gt_history.jsonl"

echo "[stage] A5 paired history intervention"
run_causal_eval_8way \
  a5_revision_target_gt_history "$CAUSAL/eval_grid_a5_gt_history.jsonl" "$A5_ROOT" 2048
rm -rf "$HISTORY_OUT"
"$PYTHON_BIN" scripts/analyze_student_relative_revision.py \
  --revision-history-input "$CAUSAL/eval_grid_a4_revision_history.jsonl" \
  --revision-history-eval "$A4_ROOT/merged.jsonl" --gt-history-eval "$A5_ROOT/merged.jsonl" \
  --population-input "$CAUSAL/a4_revision_target_revision_history.jsonl" \
  --output-dir "$HISTORY_OUT" --bootstrap-draws 10000 --seed 42 \
  > "$OVERNIGHT/logs/history_analysis.log"

echo "[stage] one-step LoRA training smoke"
SMOKE="$FULL_ROOT/lora_screen_smoke"
if [[ ! -f "$SMOKE/PASS" ]]; then
  rm -rf "$SMOKE"
  mkdir -p "$SMOKE"
  MASTER_ADDR=127.0.0.1 MASTER_PORT=29790 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/train_hetero_inject_sft.py \
    --model_path checkpoints/gui360-fullparam-sft-step250 \
    --train_data "$LORA_ROOT/data/a5_revision_target_gt_history.jsonl" \
    --output_dir "$SMOKE/model" --max_rows 8 --max_steps 1 --num_epochs 1 \
    --gradient_accumulation_steps 8 --logging_steps 1 --save_steps 0 \
    --lora_r 64 --lora_alpha 128 --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --trainable_mode lora_only --image_max_pixels 602112 --label_smoothing 0.0 \
    --kl_weight 0.0 --entropy_bonus 0.0 --lora_lr 2e-5 --seed 42 \
    > "$SMOKE/train.log" 2>&1
  "$PYTHON_BIN" - "$SMOKE/model/final/training_state.pt" "$SMOKE/model/final/cooperative" <<'PY'
import sys,torch
from pathlib import Path
state=torch.load(sys.argv[1],map_location='cpu',weights_only=True)
if int(state.get('global_step',-1)) != 1: raise SystemExit(state)
if not Path(sys.argv[2]).is_dir(): raise SystemExit('missing adapter')
PY
  CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" scripts/evaluate_multiagent_revision_pilot.py \
    --arm lora_smoke --episode-data "$FULL_ROOT/test_eval_episodes.jsonl" \
    --output "$SMOKE/eval.jsonl" --batch-size 8 --seed 42 \
    --model-path checkpoints/gui360-fullparam-sft-step250 \
    --adapter-dir "$SMOKE/model/final/cooperative" --shard-index 0 --shard-count 125 \
    > "$SMOKE/eval.log" 2>&1
  touch "$SMOKE/PASS"
fi

echo "[stage] equal-budget eight-arm LoRA screen"
if [[ ! -s "$LORA_ROOT/report/summary.json" ]]; then
  bash scripts/run_revision_lora_screen.sh
else
  echo "[resume] LoRA screen"
fi

mapfile -t candidates < <("$PYTHON_BIN" - "$LORA_ROOT/report/summary.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
for arm in x.get('full_eval_candidates',[]): print(arm)
PY
)
printf '[gate] screen candidates: %s\n' "${candidates[*]:-none}"

for arm in "${candidates[@]}"; do
  echo "[stage] full 1,000-episode LoRA confirmation: $arm"
  run_full_lora_eval "$arm"
done

TOP_CANDIDATE="$($PYTHON_BIN - "$LORA_ROOT/full_eval" <<'PY'
import glob,json,sys
rows=[]
for path in glob.glob(sys.argv[1]+'/*/report/summary.json'):
    x=json.load(open(path))
    for row in x.get('arms',[]):
        if row.get('gate')=='HELPS' and row.get('deployable_selector') and row.get('arm')!='a4_revision_target_revision_history':
            rows.append(row)
rows.sort(key=lambda r:(r['tsr_delta'],r['step_accuracy_delta']),reverse=True)
print(rows[0]['arm'] if rows else '')
PY
)"

if [[ -n "$TOP_CANDIDATE" ]]; then
  echo "[stage] top full-grid candidate enters 6-GPU fullparam: $TOP_CANDIDATE"
  if [[ ! -f "$FULL_ROOT/fullparam_candidates/$TOP_CANDIDATE/COMPLETE" ]]; then
    bash scripts/run_revision_fullparam_candidate.sh "$TOP_CANDIDATE"
  else
    echo "[resume] fullparam $TOP_CANDIDATE"
  fi
else
  echo "[gate] no deployable full-grid LoRA candidate; stop before fullparam"
fi

touch "$OVERNIGHT/COMPLETE"
echo "[complete] revision research overnight pipeline"
