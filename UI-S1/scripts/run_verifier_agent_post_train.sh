#!/usr/bin/env bash
# Post-train runbook for Verifier Agent SFT.
# Locates the trained checkpoint, generates verifier decisions, evaluates them,
# writes coordinator commands, and writes compact summaries.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}
BASE_MODEL=${BASE_MODEL:-$PROJECT_DIR/checkpoints/Qwen3.5-9B}
RUN_DIR=${RUN_DIR:-$PROJECT_DIR/outputs/verifier_agent_sft_qwen35_4gpu_fp32_len2048}
DATA_DIR=${DATA_DIR:-$PROJECT_DIR/datasets/verifier_agent_gui_odyssey_sft_balanced}
EVAL_OUT=${EVAL_OUT:-$RUN_DIR/post_train_eval}
COORDINATOR_OUT=${COORDINATOR_OUT:-$RUN_DIR/coordinator_eval}
CHECKPOINT=${CHECKPOINT:-}
DTYPE=${DTYPE:-bf16}
DEVICE=${DEVICE:-cuda}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-96}
BATCH_SIZE=${BATCH_SIZE:-8}
LIMIT=${LIMIT:-0}
SKIP_EXISTING=${SKIP_EXISTING:-0}
WAIT_FOR_CHECKPOINT=${WAIT_FOR_CHECKPOINT:-0}
WAIT_SECONDS=${WAIT_SECONDS:-21600}
POLL_SECONDS=${POLL_SECONDS:-300}

latest_checkpoint() {
  find "$RUN_DIR/checkpoints" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sort -V | tail -1
}

wait_for_checkpoint() {
  local waited=0
  local ckpt=""
  while true; do
    ckpt=$(latest_checkpoint || true)
    if [[ -n "$ckpt" ]]; then
      echo "$ckpt"
      return 0
    fi
    if [[ "$WAIT_FOR_CHECKPOINT" != "1" ]]; then
      return 1
    fi
    if (( waited >= WAIT_SECONDS )); then
      echo "Timed out waiting for checkpoint under $RUN_DIR/checkpoints" >&2
      return 1
    fi
    echo "No checkpoint yet; waiting ${POLL_SECONDS}s... (${waited}/${WAIT_SECONDS})" >&2
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
  done
}

mkdir -p "$EVAL_OUT"

if [[ -z "$CHECKPOINT" ]]; then
  CHECKPOINT=$(wait_for_checkpoint || true)
fi
if [[ -z "$CHECKPOINT" ]]; then
  echo "No checkpoint found. Set CHECKPOINT=/path/to/global_step_* or run with WAIT_FOR_CHECKPOINT=1." >&2
  exit 1
fi
if [[ -f "$CHECKPOINT/.dcp_checkpoint" ]]; then
  echo "Checkpoint is DCP sharded: $CHECKPOINT" >&2
  echo "Use the final HF checkpoint, or convert DCP before inference." >&2
  exit 1
fi

cat > "$EVAL_OUT/run_info.json" <<JSON
{
  "run_dir": "$RUN_DIR",
  "checkpoint": "$CHECKPOINT",
  "base_model": "$BASE_MODEL",
  "data_dir": "$DATA_DIR",
  "coordinator_out": "$COORDINATOR_OUT",
  "dtype": "$DTYPE",
  "device": "$DEVICE",
  "max_new_tokens": $MAX_NEW_TOKENS,
  "batch_size": $BATCH_SIZE,
  "skip_existing": $SKIP_EXISTING,
  "limit": $LIMIT
}
JSON

run_split() {
  local split="$1"
  local data_file="$DATA_DIR/${split}.jsonl"
  local pred_file="$EVAL_OUT/${split}_predictions.jsonl"
  local out_dir="$EVAL_OUT/${split}_eval"
  local coord_dir="$COORDINATOR_OUT/${split}"
  local command_file="$coord_dir/verifier_safety_gate_commands.jsonl"
  local command_summary="$coord_dir/command_summary.json"
  local metrics_file="$out_dir/verifier_eval_metrics.json"
  if [[ ! -f "$data_file" ]]; then
    echo "skip missing $data_file" >&2
    return 0
  fi
  if [[ "$SKIP_EXISTING" == "1" && -f "$pred_file" ]]; then
    echo "Skipping $split generation; predictions already exist."
  else
    echo "Generating $split predictions..."
    "$PYTHON_BIN" scripts/generate_verifier_agent_predictions.py \
      --base-model "$BASE_MODEL" \
      --checkpoint "$CHECKPOINT" \
      --data "$data_file" \
      --output "$pred_file" \
      --dtype "$DTYPE" \
      --device "$DEVICE" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --batch-size "$BATCH_SIZE" \
      ${LIMIT:+--limit "$LIMIT"}
  fi
  if [[ "$SKIP_EXISTING" == "1" && -f "$metrics_file" ]]; then
    echo "Skipping $split verifier metrics; metrics already exist."
  else
    echo "Evaluating $split predictions..."
    "$PYTHON_BIN" scripts/evaluate_verifier_agent.py \
      --data "$data_file" \
      --predictions "$pred_file" \
      --output-dir "$out_dir" \
      --mode predictions \
      ${LIMIT:+--limit "$LIMIT"}
  fi

  if [[ "$SKIP_EXISTING" == "1" && -f "$command_file" && -f "$command_summary" ]]; then
    echo "Skipping $split coordinator commands; commands already exist."
  else
    echo "Writing $split coordinator commands..."
    "$PYTHON_BIN" scripts/apply_verifier_agent_coordinator.py \
      --data "$data_file" \
      --predictions "$pred_file" \
      --output "$command_file" \
      --summary "$command_summary" \
      ${LIMIT:+--limit "$LIMIT"}
  fi

  echo "Evaluating $split coordinator replay..."
  "$PYTHON_BIN" scripts/evaluate_verifier_agent_coordinator.py \
    --data "$data_file" \
    --predictions "$pred_file" \
    --output-dir "$coord_dir" \
    ${LIMIT:+--limit "$LIMIT"}
}

run_split dev
run_split test
run_split dev_balanced
run_split test_balanced

POST_TRAIN_EVAL_OUT="$EVAL_OUT" POST_TRAIN_CHECKPOINT="$CHECKPOINT" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ['POST_TRAIN_EVAL_OUT'])
checkpoint_name = Path(os.environ['POST_TRAIN_CHECKPOINT']).name
lines = ['# Verifier Agent Post-Train Evaluation', '']
lines.append(f'Checkpoint: `{checkpoint_name}`')
lines.append('')
lines.append('| split | accuracy | macro F1 | commit P/R | full P/R | replan P/R | invalid pred |')
lines.append('|---|---:|---:|---:|---:|---:|---:|')
for split in ['dev', 'test', 'dev_balanced', 'test_balanced']:
    path = root / f'{split}_eval' / 'verifier_eval_metrics.json'
    if not path.exists():
        continue
    data = json.loads(path.read_text())['metrics']
    per = data['per_class']
    pred_counts = data.get('pred_counts', {})
    invalid = pred_counts.get('invalid', 0)
    lines.append(
        f"| {split} | {data['accuracy']:.4f} | {data['macro_f1']:.4f} | "
        f"{per['commit_segment']['precision']:.4f}/{per['commit_segment']['recall']:.4f} | "
        f"{per['use_full_history']['precision']:.4f}/{per['use_full_history']['recall']:.4f} | "
        f"{per['replan']['precision']:.4f}/{per['replan']['recall']:.4f} | {invalid} |"
    )
(root / 'post_train_summary.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
PY

POST_TRAIN_EVAL_OUT="$EVAL_OUT" POST_TRAIN_COORDINATOR_OUT="$COORDINATOR_OUT" POST_TRAIN_CHECKPOINT="$CHECKPOINT" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

eval_root = Path(os.environ['POST_TRAIN_EVAL_OUT'])
coord_root = Path(os.environ['POST_TRAIN_COORDINATOR_OUT'])
checkpoint_name = Path(os.environ['POST_TRAIN_CHECKPOINT']).name
lines = ['# Verifier Agent Runtime Summary', '']
lines.append(f'Checkpoint: `{checkpoint_name}`')
lines.append('')
lines.append('## Verifier Route Metrics')
lines.append('')
lines.append('| split | accuracy | macro F1 | commit P/R | full P/R | replan P/R | invalid pred |')
lines.append('|---|---:|---:|---:|---:|---:|---:|')
for split in ['dev', 'test', 'dev_balanced', 'test_balanced']:
  path = eval_root / f'{split}_eval' / 'verifier_eval_metrics.json'
  if not path.exists():
    continue
  data = json.loads(path.read_text())['metrics']
  per = data['per_class']
  invalid = data.get('pred_counts', {}).get('invalid', 0)
  lines.append(
    f"| {split} | {data['accuracy']:.4f} | {data['macro_f1']:.4f} | "
    f"{per['commit_segment']['precision']:.4f}/{per['commit_segment']['recall']:.4f} | "
    f"{per['use_full_history']['precision']:.4f}/{per['use_full_history']['recall']:.4f} | "
    f"{per['replan']['precision']:.4f}/{per['replan']['recall']:.4f} | {invalid} |"
  )
lines.append('')
lines.append('## Coordinator Command Metrics')
lines.append('')
lines.append('| split | execute rate | action acc all | executed acc | unsafe exec | replan rate | replan abstain recall | missed executable |')
lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
for split in ['dev', 'test', 'dev_balanced', 'test_balanced']:
  path = coord_root / split / 'command_summary.json'
  if not path.exists():
    continue
  data = json.loads(path.read_text())
  lines.append(
    f"| {split} | {data['execute_rate']:.4f} | {data['action_accuracy_all']:.4f} | "
    f"{data['executed_action_accuracy']:.4f} | {data['unsafe_execution_rate']:.4f} | "
    f"{data['replan_rate']:.4f} | {data['replan_abstain_recall']:.4f} | "
    f"{data['missed_executable_count']} |"
  )
lines.append('')
lines.append('Interpretation: coordinator commands are the agent-facing artifact. `execute` rows carry a concrete GUI action; `replan` rows carry a replan_request for a resolver agent. Replan is intentionally an abstention/escalation signal, not a forced fallback route.')
(eval_root / 'runtime_summary.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
PY

echo "Post-train evaluation written to $EVAL_OUT"
echo "Coordinator commands written to $COORDINATOR_OUT"
