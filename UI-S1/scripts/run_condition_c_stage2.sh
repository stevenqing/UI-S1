#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/aiscuser/UI-S1/UI-S1}"
cd "$PROJECT_ROOT"

OUT="${OUT:-$PROJECT_ROOT/outputs/text_error_horizon_probe/condition_c/stage2_dose}"
JSONL_FILE="${JSONL_FILE:-$PROJECT_ROOT/datasets/GUI-Odyssey/gui_odyssey_random_split_test.jsonl}"
ROLLOUT_RESULTS="${ROLLOUT_RESULTS:-$PROJECT_ROOT/outputs/qwen35_9b_baseline_template_gui_odyssey_full_random_split_test_8gpu_512threads/merged_corrected_direct1k/trajectory_results.jsonl}"
PILOT_EPISODES="${PILOT_EPISODES:-$PROJECT_ROOT/outputs/text_error_horizon_probe/qwen35_pilot300_action_only/pilot_episodes.jsonl}"
ZERO_PAIRS="$OUT/zeropoint_multi/pairs.jsonl"
MAIN_PAIRS="$OUT/main/pairs.jsonl"
FORCE_BUILD="${FORCE_BUILD:-0}"
FORCE_RUN="${FORCE_RUN:-0}"

mkdir -p "$OUT/zeropoint_multi" "$OUT/main"
if [[ "$FORCE_BUILD" == "1" || ! -s "$ZERO_PAIRS" ]]; then
  .venv/bin/python scripts/build_condition_c_stage2_pairs.py \
    --jsonl-file "$JSONL_FILE" \
    --rollout-results "$ROLLOUT_RESULTS" \
    --pilot-episode-ids "$PILOT_EPISODES" \
    --output "$ZERO_PAIRS" \
    --stage zeropoint
else
  echo "Reusing zero pairs: $ZERO_PAIRS"
fi
if [[ "$FORCE_BUILD" == "1" || ! -s "$MAIN_PAIRS" ]]; then
  .venv/bin/python scripts/build_condition_c_stage2_pairs.py \
    --jsonl-file "$JSONL_FILE" \
    --rollout-results "$ROLLOUT_RESULTS" \
    --pilot-episode-ids "$PILOT_EPISODES" \
    --output "$MAIN_PAIRS" \
    --stage main
else
  echo "Reusing main pairs: $MAIN_PAIRS"
fi

if [[ "$FORCE_RUN" == "1" || ! -s "$OUT/zeropoint_multi/pair_rows.jsonl" ]]; then
  PAIRS_FILE="$ZERO_PAIRS" OUTPUT_ROOT="$OUT/zeropoint_multi" bash scripts/run_condition_c_sharded.sh
else
  echo "Reusing zero rows: $OUT/zeropoint_multi/pair_rows.jsonl"
fi

.venv/bin/python - <<PY
import json
from pathlib import Path
from scripts.summarize_condition_c_stage2 import iter_jsonl, mean_ci
rows = list(iter_jsonl(Path("$OUT/zeropoint_multi/pair_rows.jsonl")))
values = [float(row.get("gap_value", 0.0)) for row in rows if not row.get("error")]
summary = mean_ci(values)
print(json.dumps({"zeropoint_rows": len(rows), "value_gap": summary, "gate_pass": abs(summary["mean"]) <= 0.01}, ensure_ascii=False, indent=2))
if abs(summary["mean"]) > 0.01:
    raise SystemExit("Stage2 zero-point multi gate failed; stop before main dose run.")
PY

if [[ "$FORCE_RUN" == "1" || ! -s "$OUT/main/pair_rows.jsonl" ]]; then
  PAIRS_FILE="$MAIN_PAIRS" OUTPUT_ROOT="$OUT/main" bash scripts/run_condition_c_sharded.sh
else
  echo "Reusing main rows: $OUT/main/pair_rows.jsonl"
fi

.venv/bin/python scripts/summarize_condition_c_stage2.py \
  --zero-results "$OUT/zeropoint_multi/pair_rows.jsonl" \
  --main-results "$OUT/main/pair_rows.jsonl" \
  --output-dir "$OUT"

echo "Summary: $OUT/summary.md"