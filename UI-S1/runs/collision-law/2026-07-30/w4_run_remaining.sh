#!/usr/bin/env bash
set -euo pipefail

run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$run_dir/../../.." && pwd)"
python="$workspace/.venv-ac-vllm/bin/python"
freeze="$run_dir/AMENDMENT_007_CCM_CONFIRMATION.md"

[[ -f "$freeze" ]] || {
  echo "W4 blocked: freeze A5 in $freeze before confirmation inference" >&2
  exit 2
}

score_passes() {
  local score="$1"
  [[ -f "$score" ]] && "$python" -c \
    'import json, sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status") == "PASS" else 1)' \
    "$score"
}

models=(ui-agile-3b ui-agile-7b ui-r1-e-3b gui-r1-3b gui-r1-7b)
for model in "${models[@]}"; do
  for setting in low high; do
    score="$run_dir/w4_artifacts/$model/$setting/score.json"
    if score_passes "$score"; then
      echo "W4 skip PASS: $model/$setting"
      continue
    fi
    echo "W4 run: $model/$setting"
    "$run_dir/w4_launch.sh" "$model" "$setting"
    score_passes "$score"
  done
done

"$python" "$run_dir/w4_analyze.py" \
  --curated "$run_dir/w4_curated.json" \
  --threshold "$run_dir/w4_threshold.json"
score_passes "$run_dir/w4_curated.json"
score_passes "$run_dir/w4_threshold.json"