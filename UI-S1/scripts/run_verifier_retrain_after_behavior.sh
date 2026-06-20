#!/usr/bin/env bash
# Resume the Verifier Agent retraining pipeline after behavior validation finishes.
#
# This script is intended to be started while the Qwen3-VL behavior validation
# job is still running. Set WAIT_PID to that job's top-level PID and this script
# will block until it exits, then run the downstream data/model steps.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}
WAIT_PID=${WAIT_PID:-}

BEHAVIOR_DIR=${BEHAVIOR_DIR:-$PROJECT_DIR/datasets/model_bottleneck_validation_qwen3vl_restore_20260620}
BEHAVIOR_RESULTS=${BEHAVIOR_RESULTS:-$BEHAVIOR_DIR/model_behavior_results.jsonl}
SEGMENT_EPISODES=${SEGMENT_EPISODES:-$PROJECT_DIR/datasets/segmentation_train/gui_odyssey_segments.jsonl}

CMU_DIR=${CMU_DIR:-$PROJECT_DIR/datasets/counterfactual_memory_utility_restore_20260620}
PROPOSAL_DIR=${PROPOSAL_DIR:-$PROJECT_DIR/datasets/counterfactual_memory_utility_specificity_progress_restore_20260620}
VERIFIER_ALL_DIR=${VERIFIER_ALL_DIR:-$PROJECT_DIR/datasets/verifier_agent_gui_odyssey_all_restore_20260620}
VERIFIER_HARD_DIR=${VERIFIER_HARD_DIR:-$PROJECT_DIR/datasets/verifier_agent_gui_odyssey_hard_restore_20260620}
SFT_DIR=${SFT_DIR:-$PROJECT_DIR/datasets/verifier_agent_gui_odyssey_sft_balanced}

RUN_DIR=${RUN_DIR:-$PROJECT_DIR/outputs/verifier_agent_sft_qwen35_retrain_fp32_len2048}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-verifier_agent_qwen35_retrain_fp32_len2048}
MASTER_PORT=${MASTER_PORT:-29615}
N_GPUS=${N_GPUS:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
MODEL_PATH=${MODEL_PATH:-$PROJECT_DIR/checkpoints/Qwen3.5-9B}
BASE_MODEL=${BASE_MODEL:-$MODEL_PATH}

TRAIN_PER_CLASS=${TRAIN_PER_CLASS:-1024}
BALANCED_EVAL_PER_CLASS=${BALANCED_EVAL_PER_CLASS:-128}
MAX_LENGTH=${MAX_LENGTH:-2048}
MODEL_DTYPE=${MODEL_DTYPE:-fp32}
LOGGER=${LOGGER:-console}

POST_TRAIN_EVAL_OUT=${POST_TRAIN_EVAL_OUT:-$RUN_DIR/post_train_eval}
POST_TRAIN_COORDINATOR_OUT=${POST_TRAIN_COORDINATOR_OUT:-$RUN_DIR/coordinator_eval}
POST_TRAIN_BATCH_SIZE=${POST_TRAIN_BATCH_SIZE:-8}
POST_TRAIN_MAX_NEW_TOKENS=${POST_TRAIN_MAX_NEW_TOKENS:-96}

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "Missing required directory: $path" >&2
    exit 1
  fi
}

wait_for_pid_if_needed() {
  if [[ -z "$WAIT_PID" ]]; then
    return 0
  fi
  if kill -0 "$WAIT_PID" 2>/dev/null; then
    log "Waiting for behavior validation PID $WAIT_PID to exit..."
    # GNU tail --pid blocks until the process exits without manual polling.
    tail --pid="$WAIT_PID" -f /dev/null
  else
    log "WAIT_PID=$WAIT_PID is not running; continuing."
  fi
}

main() {
  cd "$PROJECT_DIR"
  require_file "$PYTHON_BIN"
  require_file "$SEGMENT_EPISODES"
  require_dir "$MODEL_PATH"

  wait_for_pid_if_needed

  require_file "$BEHAVIOR_RESULTS"
  log "Behavior validation complete: $BEHAVIOR_RESULTS"

  log "Building counterfactual memory utility data -> $CMU_DIR"
  "$PYTHON_BIN" scripts/build_counterfactual_memory_utility_data.py \
    --results "$BEHAVIOR_RESULTS" \
    --episodes "$SEGMENT_EPISODES" \
    --output-dir "$CMU_DIR"

  log "Training specificity+progress proposal scorer -> $PROPOSAL_DIR"
  "$PYTHON_BIN" scripts/train_counterfactual_memory_utility.py \
    --data-dir "$CMU_DIR" \
    --output-dir "$PROPOSAL_DIR" \
    --specificity-features \
    --progress-features

  require_file "$PROPOSAL_DIR/memory_utility_model.joblib"

  log "Building all verifier-agent packets -> $VERIFIER_ALL_DIR"
  "$PYTHON_BIN" scripts/build_verifier_agent_data.py \
    --data-dir "$CMU_DIR" \
    --proposal-model "$PROPOSAL_DIR/memory_utility_model.joblib" \
    --output-dir "$VERIFIER_ALL_DIR" \
    --proposal-threshold 0.0

  log "Building hard-only verifier-agent packets -> $VERIFIER_HARD_DIR"
  "$PYTHON_BIN" scripts/build_verifier_agent_data.py \
    --data-dir "$CMU_DIR" \
    --proposal-model "$PROPOSAL_DIR/memory_utility_model.joblib" \
    --output-dir "$VERIFIER_HARD_DIR" \
    --proposal-threshold 0.0 \
    --hard-only

  log "Preparing balanced Verifier Agent SFT data -> $SFT_DIR"
  "$PYTHON_BIN" scripts/prepare_verifier_agent_sft_data.py \
    --input-dir "$VERIFIER_HARD_DIR" \
    --output-dir "$SFT_DIR" \
    --train-per-class "$TRAIN_PER_CLASS" \
    --balanced-eval-per-class "$BALANCED_EVAL_PER_CLASS"

  require_file "$SFT_DIR/train_balanced.parquet"
  require_file "$SFT_DIR/dev.parquet"

  log "Launching Verifier Agent SFT -> $RUN_DIR"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  PYTHON_BIN="$PYTHON_BIN" \
  MODEL_PATH="$MODEL_PATH" \
  TRAIN_PARQUET="$SFT_DIR/train_balanced.parquet" \
  VAL_PARQUET="$SFT_DIR/dev.parquet" \
  OUTPUT_DIR="$RUN_DIR" \
  EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  N_GPUS="$N_GPUS" \
  MASTER_PORT="$MASTER_PORT" \
  MODEL_DTYPE="$MODEL_DTYPE" \
  MAX_LENGTH="$MAX_LENGTH" \
  LOGGER="$LOGGER" \
  bash scripts/run_verifier_agent_sft.sh

  log "Running post-train Verifier Agent evaluation -> $POST_TRAIN_EVAL_OUT"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" \
  PYTHON_BIN="$PYTHON_BIN" \
  RUN_DIR="$RUN_DIR" \
  BASE_MODEL="$BASE_MODEL" \
  DATA_DIR="$SFT_DIR" \
  EVAL_OUT="$POST_TRAIN_EVAL_OUT" \
  COORDINATOR_OUT="$POST_TRAIN_COORDINATOR_OUT" \
  BATCH_SIZE="$POST_TRAIN_BATCH_SIZE" \
  MAX_NEW_TOKENS="$POST_TRAIN_MAX_NEW_TOKENS" \
  SKIP_EXISTING=1 \
  bash scripts/run_verifier_agent_post_train.sh

  log "Verifier retraining pipeline complete."
}

main "$@"