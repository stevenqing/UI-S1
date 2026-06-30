#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-.venv-qwen3-vllm/bin/python}
BUILD_G=${BUILD_G:-1}
BUILD_O=${BUILD_O:-0}
RUN_RECIPE_DIFF=${RUN_RECIPE_DIFF:-1}
RUN_TRAIN=${RUN_TRAIN:-0}
DRY_RUN=${DRY_RUN:-1}
WRITE_VERDICT=${WRITE_VERDICT:-0}
BUILD_V3_PAIRS=${BUILD_V3_PAIRS:-1}
CHECK_TRAIN_DEPS=${CHECK_TRAIN_DEPS:-1}
LLAMAFACTORY_CMD=${LLAMAFACTORY_CMD:-$(dirname "$PYTHON")/llamafactory-cli}

BALANCED_DATA_DIR=${BALANCED_DATA_DIR:-datasets/gui360-balanced/data}
DATA_DIR=${DATA_DIR:-train_GUI_360/llamafactory/data}
IMAGE_DIR=${IMAGE_DIR:-train_GUI_360/llamafactory/data/gui360_history_arm_images}
BASE_MODEL=${BASE_MODEL:-checkpoints/Qwen2.5-VL-7B-Instruct}
DS_CONFIG=${DS_CONFIG:-train_GUI_360/llamafactory/ds_z3_config.json}
IMAGE_MAX_PIXELS=${IMAGE_MAX_PIXELS:-200704}
CUTOFF_LEN=${CUTOFF_LEN:-16384}
SAVE_STRATEGY=${SAVE_STRATEGY:-epoch}
EVAL_STRATEGY=${EVAL_STRATEGY:-epoch}
MAX_TRAIN_EPISODES=${MAX_TRAIN_EPISODES:--1}
MAX_VAL_EPISODES=${MAX_VAL_EPISODES:-32}
PATCH_BUDGET=${PATCH_BUDGET:-3}

S_YAML=${S_YAML:-train_GUI_360/llamafactory/qwen25vl_gui360_balanced_full_sft_repro.yaml}
G_YAML=${G_YAML:-train_GUI_360/llamafactory/qwen25vl_gui360_gt_history_full_sft.yaml}
O_YAML=${O_YAML:-train_GUI_360/llamafactory/qwen25vl_gui360_own_history_full_sft.yaml}
TRAIN_ARMS=${TRAIN_ARMS:-S,G,O}
RESULTS_JSON=${RESULTS_JSON:-gui360_long_horizon/reports/capstone_results.json}
VERDICT_JSON=${VERDICT_JSON:-gui360_long_horizon/reports/verdict.json}
V3_OUT_DIR=${V3_OUT_DIR:-outputs/gui360_history_ab/v3_candidates_$(date +%Y%m%d_%H%M%S)}
V3_DATASET=${V3_DATASET:-train_GUI_360/llamafactory/data/gui360_gt_history_val.json}
V3_PAIRS=${V3_PAIRS:-$V3_OUT_DIR/G_v3_pairs.json}
V3_SHUFFLE_PAIRS=${V3_SHUFFLE_PAIRS:-$V3_OUT_DIR/G_v3_pairs_shuffle.json}
V3_CANDIDATE_SUMMARY=${V3_CANDIDATE_SUMMARY:-$V3_OUT_DIR/G_v3_candidates_summary.json}
V3_MIN_PAIRS=${V3_MIN_PAIRS:-8}

O_HARNESS_BASE_URL=${O_HARNESS_BASE_URL:-}
O_HARNESS_MODEL=${O_HARNESS_MODEL:-$BASE_MODEL}
O_HARNESS_MAX_TOKENS=${O_HARNESS_MAX_TOKENS:-256}
O_HARNESS_TEMPERATURE=${O_HARNESS_TEMPERATURE:-0.0}

log() {
  printf '[history-ab] %s\n' "$*"
}

run_or_echo() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

require_python() {
  if [[ ! -x "$PYTHON" ]]; then
    echo "Python executable not found or not executable: $PYTHON" >&2
    exit 2
  fi
  export PATH="$(dirname "$PYTHON"):$PATH"
}

build_arm() {
  local arm=$1
  shift
  "$PYTHON" train_GUI_360/build_history_arms.py \
    --arm "$arm" \
    --balanced-data-dir "$BALANCED_DATA_DIR" \
    --data-dir "$DATA_DIR" \
    --image-dir "$IMAGE_DIR" \
    --base-model "$BASE_MODEL" \
    --ds-config "$DS_CONFIG" \
    --image-max-pixels "$IMAGE_MAX_PIXELS" \
    --cutoff-len "$CUTOFF_LEN" \
    --save-strategy "$SAVE_STRATEGY" \
    --eval-strategy "$EVAL_STRATEGY" \
    --max-train-episodes "$MAX_TRAIN_EPISODES" \
    --max-val-episodes "$MAX_VAL_EPISODES" \
    --patch-budget "$PATCH_BUDGET" \
    --prepare-data \
    --write-config \
    "$@"
}

yaml_for_arm() {
  case "$1" in
    S|single_step) echo "$S_YAML" ;;
    G|gt_history) echo "$G_YAML" ;;
    O|own_history) echo "$O_YAML" ;;
    *) echo "unknown arm in TRAIN_ARMS: $1" >&2; exit 2 ;;
  esac
}

train_arm() {
  local arm=$1
  local yaml
  yaml=$(yaml_for_arm "$arm")
  if [[ ! -f "$yaml" ]]; then
    echo "missing YAML for $arm: $yaml" >&2
    exit 2
  fi
  read -r -a train_cmd <<< "$LLAMAFACTORY_CMD"
  if [[ "$DRY_RUN" != "1" ]] && ! command -v "${train_cmd[0]}" >/dev/null 2>&1; then
    echo "${train_cmd[0]} is not in PATH; cannot train $arm ($yaml). Set LLAMAFACTORY_CMD to a valid launcher." >&2
    exit 2
  fi
  if [[ "$DRY_RUN" != "1" && "$CHECK_TRAIN_DEPS" == "1" ]]; then
    "$PYTHON" - <<'PY'
import importlib.util, sys
missing = [name for name in ("llamafactory", "deepspeed") if importlib.util.find_spec(name) is None]
if missing:
    print("missing training Python packages: " + ", ".join(missing), file=sys.stderr)
    sys.exit(2)
PY
  fi
  run_or_echo "${train_cmd[@]}" train "$yaml"
}

build_v3_pairs() {
  if [[ ! -f "$V3_DATASET" ]]; then
    echo "missing V3 dataset: $V3_DATASET" >&2
    exit 2
  fi
  "$PYTHON" -m gui360_long_horizon.data.longdep_candidates \
    --dataset "$V3_DATASET" \
    --out-pairs "$V3_PAIRS" \
    --out-summary "$V3_CANDIDATE_SUMMARY" \
    --out-shuffle "$V3_SHUFFLE_PAIRS" \
    --min-pairs "$V3_MIN_PAIRS"
}

main() {
  require_python

  if [[ "$BUILD_G" == "1" ]]; then
    log "build order 1: gt_history data + dataset_info + YAML"
    build_arm gt_history
  fi

  if [[ "$BUILD_O" == "1" ]]; then
    log "build order 2: own_history data + dataset_info + YAML"
    if [[ -z "$O_HARNESS_BASE_URL" ]]; then
      echo "BUILD_O=1 requires O_HARNESS_BASE_URL; refusing to run _Unwired O arm" >&2
      exit 2
    fi
    build_arm own_history \
      --harness-base-url "$O_HARNESS_BASE_URL" \
      --harness-model "$O_HARNESS_MODEL" \
      --harness-max-tokens "$O_HARNESS_MAX_TOKENS" \
      --harness-temperature "$O_HARNESS_TEMPERATURE"
  fi

  if [[ "$RUN_RECIPE_DIFF" == "1" ]]; then
    log "build order 3 guard: recipe diff"
    candidates=()
    [[ -f "$G_YAML" ]] && candidates+=(--candidate "$G_YAML")
    [[ -f "$O_YAML" ]] && candidates+=(--candidate "$O_YAML")
    if [[ ${#candidates[@]} -eq 0 ]]; then
      echo "no G/O candidate YAML found for recipe diff" >&2
      exit 2
    fi
    "$PYTHON" -m gui360_long_horizon.analysis.guards recipe-diff --reference "$S_YAML" "${candidates[@]}"
  fi

  if [[ "$BUILD_V3_PAIRS" == "1" ]]; then
    log "build order 7 prep: conservative V3 pair candidates"
    build_v3_pairs
  fi

  if [[ "$RUN_TRAIN" == "1" ]]; then
    log "build order 3: matched S/G/O training"
    IFS=',' read -r -a arms <<< "$TRAIN_ARMS"
    for arm in "${arms[@]}"; do
      train_arm "$arm"
    done
  else
    log "training disabled (set RUN_TRAIN=1, DRY_RUN=0 to launch LLaMA-Factory)"
  fi

  if [[ "$WRITE_VERDICT" == "1" ]]; then
    log "build order 8: write verdict"
    "$PYTHON" -m gui360_long_horizon.analysis.capstone_report --results "$RESULTS_JSON" --out "$VERDICT_JSON"
  fi
}

main "$@"