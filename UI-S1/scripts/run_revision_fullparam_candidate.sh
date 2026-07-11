#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ARM="${1:?usage: run_revision_fullparam_candidate.sh ARM [DATA] [DATA_SUMMARY]}"
DATA="${2:-outputs/multiagent_trajectory_revision/full_v1/causal_arms/${ARM}.jsonl}"
DATA_SUMMARY="${3:-outputs/multiagent_trajectory_revision/full_v1/causal_arms/${ARM}.summary.json}"
PYTHON_BIN="${PYTHON_BIN:-.venv-qwen3-vllm/bin/python}"
ROOT_OUT="${ROOT_OUT:-outputs/multiagent_trajectory_revision/full_v1/fullparam_candidates}"
OUT="$ROOT_OUT/$ARM"
MODEL_PATH="${MODEL_PATH:-checkpoints/gui360-fullparam-sft-step250}"
BASELINE="${BASELINE:-outputs/validation_2k/eval_sft_greedy.jsonl}"
TEST_EPISODES="${TEST_EPISODES:-outputs/multiagent_trajectory_revision/full_v1/test_eval_episodes.jsonl}"
SCREEN_MANIFEST="${SCREEN_MANIFEST:-outputs/multiagent_trajectory_revision/full_v1/lora_screen/data/manifest.json}"
PROTECTED_PID="${PROTECTED_PID:-1911}"
MODEL_OUT="$OUT/model"
COMPAT_OUT="$OUT/training_compat"
EVAL_OUT="$OUT/training_eval"
LOGS="$OUT/logs"
SHARDS="$EVAL_OUT/shards"

rm -rf "$OUT"
mkdir -p "$LOGS" "$SHARDS"
if ps -p "$PROTECTED_PID" >/dev/null 2>&1; then
  ps -p "$PROTECTED_PID" -o pid=,stat=,etime= > "$OUT/protected_pid_before.txt"
fi

"$PYTHON_BIN" scripts/prepare_multiagent_fullparam_llamafactory.py \
  --input "$DATA" --input-summary "$DATA_SUMMARY" \
  --output-dir "$OUT/llamafactory_data" --config-out "$OUT/fullparam_6gpu.yaml" \
  --model-path "$MODEL_PATH" --model-output "$MODEL_OUT" \
  --pad-to-multiple 48 --gradient-accumulation-steps 8 --learning-rate 6e-6 \
  > "$LOGS/prepare.log"

export FORCE_TORCHRUN=1
export PATH="$ROOT/.venv-qwen3-vllm/bin:$PATH"
export PYTHONPATH="$ROOT/train_GUI_360/compat${PYTHONPATH:+:$PYTHONPATH}"
export NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 NPROC_PER_NODE=6
export NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SMOKE_CONFIG="$OUT/fullparam_6gpu_smoke.yaml"
SMOKE_OUT="$OUT/smoke_model"
"$PYTHON_BIN" - "$OUT/fullparam_6gpu.yaml" "$SMOKE_CONFIG" "$SMOKE_OUT" <<'PY'
import sys,yaml
source,target,output=sys.argv[1:]
cfg=yaml.safe_load(open(source)); cfg['output_dir']=output; cfg['max_steps']=1; cfg['max_samples']=48
cfg['logging_steps']=1; cfg['run_name']=cfg.get('run_name','revision_fullparam')+'_smoke'
open(target,'w').write(yaml.safe_dump(cfg,sort_keys=False))
PY
export MASTER_PORT=29820
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 .venv-qwen3-vllm/bin/llamafactory-cli train "$SMOKE_CONFIG" \
  > "$LOGS/smoke.log" 2>&1
"$PYTHON_BIN" - "$SMOKE_OUT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); state=json.load(open(root/'trainer_state.json'))
if int(state.get('global_step',-1)) != 1: raise SystemExit(state)
print('[smoke] PASS')
PY
rm -rf "$SMOKE_OUT"

export MASTER_PORT=29821
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 .venv-qwen3-vllm/bin/llamafactory-cli train "$OUT/fullparam_6gpu.yaml" \
  > "$LOGS/train.log" 2>&1
"$PYTHON_BIN" scripts/export_llamafactory_training_compat.py \
  --model-dir "$MODEL_OUT" --output-dir "$COMPAT_OUT" > "$LOGS/export.log"
"$PYTHON_BIN" - "$MODEL_OUT" "$OUT/llamafactory_data/preparation_manifest.json" <<'PY'
import json,sys
from pathlib import Path
model=Path(sys.argv[1]); prep=json.load(open(sys.argv[2])); state=json.load(open(model/'trainer_state.json'))
if int(state.get('global_step',-1)) != int(prep['optimizer_steps']): raise SystemExit(state)
index=json.load(open(model/'model.safetensors.index.json')); shards={model/name for name in index['weight_map'].values()}
if not shards or not all(path.is_file() and path.stat().st_size>0 for path in shards): raise SystemExit('incomplete model shards')
print('[train] checkpoint complete')
PY

pids=()
for shard in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES="$shard" "$PYTHON_BIN" scripts/evaluate_multiagent_revision_pilot.py \
    --arm "$ARM" --episode-data "$TEST_EPISODES" \
    --output "$SHARDS/shard_${shard}.jsonl" --batch-size 8 --seed 42 \
    --model-path "$MODEL_OUT" --shard-index "$shard" --shard-count 8 \
    > "$LOGS/eval_shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then status=1; fi; done
if (( status != 0 )); then echo "fullparam evaluation shard failure" >&2; exit "$status"; fi

"$PYTHON_BIN" scripts/merge_multiagent_revision_eval.py \
  --episode-data "$TEST_EPISODES" --shards "$SHARDS/shard_*.jsonl" \
  --output "$EVAL_OUT/merged.jsonl" --expected-episodes 1000 --expected-steps 7498 \
  > "$LOGS/merge.log"
"$PYTHON_BIN" scripts/report_revision_lora_screen.py \
  --baseline "$BASELINE" --test-episodes "$TEST_EPISODES" \
  --post-glob "$EVAL_OUT/merged.jsonl" --data-manifest "$SCREEN_MANIFEST" \
  --output-dir "$EVAL_OUT/report" --shard-index 0 --shard-count 1 \
  --bootstrap-draws 10000 --seed 42 > "$LOGS/report.log"

if ps -p "$PROTECTED_PID" >/dev/null 2>&1; then
  ps -p "$PROTECTED_PID" -o pid=,stat=,etime= > "$OUT/protected_pid_after.txt"
fi

touch "$OUT/COMPLETE"
echo "[complete] full-parameter candidate $ARM"
