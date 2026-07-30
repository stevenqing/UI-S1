#!/usr/bin/env bash
set -euo pipefail

run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
stage_dir="$(cd "$run_dir/.." && pwd)"
workspace="$(cd "$run_dir/../../../.." && pwd)"
analysis_python="$workspace/.venv-ac-vllm/bin/python"

score_passes() {
  local score="$1"
  [[ -f "$score" ]] && "$analysis_python" -c \
    'import json, sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status") == "PASS" else 1)' \
    "$score"
}

refresh_summaries() {
  PYTHONPATH="$stage_dir" "$analysis_python" "$stage_dir/w2_analyze.py" \
    --flips "$stage_dir/w2_flips.json" \
    --noise "$stage_dir/w2_noise.json" \
    --allocation "$stage_dir/w2_allocation.json"
}

android_cells=(
  "gui-r1-7b high v2" "gui-r1-7b high v3" "gui-r1-7b high v4"
  "gui-r1-7b low v2" "gui-r1-7b low v3" "gui-r1-7b low v4"
  "ui-agile-7b high v2" "ui-agile-7b high v3" "ui-agile-7b high v4"
  "ui-agile-7b low v2" "ui-agile-7b low v3" "ui-agile-7b low v4"
)

for specification in "${android_cells[@]}"; do
  read -r model setting view <<<"$specification"
  score="$stage_dir/w2_artifacts/androidcontrol/$model/$setting/$view/score.json"
  if score_passes "$score"; then
    echo "W2 skip PASS: androidcontrol/$model/$setting/$view"
    continue
  fi
  echo "W2 run: androidcontrol/$model/$setting/$view"
  "$run_dir/launch_androidcontrol.sh" "$model" "$setting" "$view"
  score_passes "$score"
  refresh_summaries
done

for view in v2 v3 v4; do
  score="$stage_dir/w2_artifacts/mind2web/tongui-7b/$view/score.json"
  if score_passes "$score"; then
    echo "W2 skip PASS: mind2web/tongui-7b/$view"
    continue
  fi
  echo "W2 run: mind2web/tongui-7b/$view"
  "$run_dir/launch_mind2web.sh" "$view"
  score_passes "$score"
  refresh_summaries
done

refresh_summaries