#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-.venv-qwen3-vllm/bin/python}
DRY_RUN=${DRY_RUN:-1}
RUN_G=${RUN_G:-1}
RUN_O=${RUN_O:-0}
PROBES=${PROBES:-v1,v4,v2}
LIMIT=${LIMIT:--1}
MAX_TOKENS=${MAX_TOKENS:-256}
TEMPERATURE=${TEMPERATURE:-0.0}
COORD_TOL=${COORD_TOL:-20.0}
OUT_DIR=${OUT_DIR:-outputs/gui360_history_ab/probes_$(date +%Y%m%d_%H%M%S)}
AGGREGATE=${AGGREGATE:-1}
WRITE_VERDICT=${WRITE_VERDICT:-0}
RESULTS_JSON=${RESULTS_JSON:-$OUT_DIR/capstone_results.json}
VERDICT_JSON=${VERDICT_JSON:-$OUT_DIR/verdict.json}
V3_SUMMARY=${V3_SUMMARY:-}

G_DATASET=${G_DATASET:-train_GUI_360/llamafactory/data/gui360_gt_history_val.json}
O_DATASET=${O_DATASET:-train_GUI_360/llamafactory/data/gui360_own_history_val.json}
G_BASE_URL=${G_BASE_URL:-}
O_BASE_URL=${O_BASE_URL:-}
G_MODEL=${G_MODEL:-train_GUI_360/llamafactory/output/gui360_gt_history_full_sft}
O_MODEL=${O_MODEL:-train_GUI_360/llamafactory/output/gui360_own_history_full_sft}
G_V3_PAIRS=${G_V3_PAIRS:-}
G_V3_SHUFFLE_PAIRS=${G_V3_SHUFFLE_PAIRS:-}
O_V3_PAIRS=${O_V3_PAIRS:-}
O_V3_SHUFFLE_PAIRS=${O_V3_SHUFFLE_PAIRS:-}

log() {
  printf '[history-ab-probes] %s\n' "$*"
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
}

run_probe() {
  local arm=$1
  local history_format=$2
  local dataset=$3
  local base_url=$4
  local model=$5
  local probe=$6
  local pair_file=${7:-}
  local shuffle_pair_file=${8:-}

  if [[ ! -f "$dataset" ]]; then
    echo "missing dataset for $arm: $dataset" >&2
    exit 2
  fi
  if [[ -z "$base_url" ]]; then
    echo "missing ${arm}_BASE_URL; cannot run $probe for $arm" >&2
    exit 2
  fi
  extra=()
  if [[ "$probe" == "v3" ]]; then
    if [[ -z "$pair_file" || ! -f "$pair_file" ]]; then
      echo "missing ${arm}_V3_PAIRS for V3: $pair_file" >&2
      exit 2
    fi
    extra+=(--pairs "$pair_file")
    [[ -n "$shuffle_pair_file" ]] && extra+=(--shuffle-pairs "$shuffle_pair_file")
  fi
  local stem="${arm}_${probe}"
  run_or_echo "$PYTHON" -m gui360_long_horizon.experiments.capstone_runtime \
    --dataset "$dataset" \
    --probe "$probe" \
    --arm "$arm" \
    --history-format "$history_format" \
    --base-url "$base_url" \
    --model "$model" \
    --out-rows "$OUT_DIR/rows/${stem}.jsonl" \
    --out-summary "$OUT_DIR/summaries/${stem}.json" \
    --limit "$LIMIT" \
    --max-tokens "$MAX_TOKENS" \
    --temperature "$TEMPERATURE" \
    --coord-tol "$COORD_TOL" \
    "${extra[@]}"
}

main() {
  require_python
  mkdir -p "$OUT_DIR/rows" "$OUT_DIR/summaries"
  log "out_dir: $OUT_DIR"
  IFS=',' read -r -a probes <<< "$PROBES"
  for probe in "${probes[@]}"; do
    case "$probe" in
      v1|v2|v3|v4) ;;
      *) echo "unsupported probe in PROBES: $probe (supported: v1,v2,v3,v4)" >&2; exit 2 ;;
    esac
    if [[ "$RUN_G" == "1" ]]; then
      run_probe G gt_history "$G_DATASET" "$G_BASE_URL" "$G_MODEL" "$probe" "$G_V3_PAIRS" "$G_V3_SHUFFLE_PAIRS"
    fi
    if [[ "$RUN_O" == "1" ]]; then
      run_probe O own_history "$O_DATASET" "$O_BASE_URL" "$O_MODEL" "$probe" "$O_V3_PAIRS" "$O_V3_SHUFFLE_PAIRS"
    fi
  done
  log "summaries: $OUT_DIR/summaries"
  if [[ "$AGGREGATE" == "1" ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
      log "aggregation skipped in dry-run"
    else
      args=(--summary-dir "$OUT_DIR/summaries" --results-out "$RESULTS_JSON" --out "$VERDICT_JSON")
      [[ -n "$V3_SUMMARY" ]] && args+=(--v3-summary "$V3_SUMMARY")
      if [[ "$WRITE_VERDICT" == "1" ]]; then
        "$PYTHON" -m gui360_long_horizon.analysis.capstone_report "${args[@]}"
      else
        "$PYTHON" -m gui360_long_horizon.analysis.capstone_report "${args[@]}" --out "$OUT_DIR/_verdict_preview.json"
        rm -f "$OUT_DIR/_verdict_preview.json"
        log "results: $RESULTS_JSON"
      fi
    fi
  fi
}

main "$@"