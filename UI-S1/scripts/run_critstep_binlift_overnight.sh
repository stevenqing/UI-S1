#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/home/aiscuser/UI-S1/UI-S1}
cd "$PROJECT_DIR"

PYTHON=${PYTHON:-$PROJECT_DIR/.venv-qwen3-vllm/bin/python}
OUT=${OUT:-outputs/critstep_binlift}
LOG_DIR=${LOG_DIR:-$OUT/logs}
mkdir -p "$LOG_DIR"

TRAIN_CANDIDATES=${TRAIN_CANDIDATES:-}
TEST_CANDIDATES=${TEST_CANDIDATES:-outputs/verifier_e2e/slice200/candidates/per_step.jsonl}
TRAIN_VERIFIER_ROOT=${TRAIN_VERIFIER_ROOT:-}
TEST_VERIFIER_ROOT=${TEST_VERIFIER_ROOT:-}
TRAIN_TASKS=${TRAIN_TASKS:-}
TEST_TASKS=${TEST_TASKS:-outputs/critstep_eval/per_task.jsonl}
N_CANDIDATES=${N_CANDIDATES:-50}

echo "[binlift] start $(date -u)" | tee "$LOG_DIR/overnight.log"
"$PYTHON" scripts/critstep_binlift.py \
  --train-candidates "$TRAIN_CANDIDATES" \
  --test-candidates "$TEST_CANDIDATES" \
  --train-tasks "$TRAIN_TASKS" \
  --test-tasks "$TEST_TASKS" \
  --train-verifier-root "$TRAIN_VERIFIER_ROOT" \
  --test-verifier-root "$TEST_VERIFIER_ROOT" \
  --n-candidates "$N_CANDIDATES" \
  --output-dir "$OUT" \
  2>&1 | tee -a "$LOG_DIR/overnight.log"
echo "[binlift] end $(date -u)" | tee -a "$LOG_DIR/overnight.log"