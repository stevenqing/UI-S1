#!/usr/bin/env bash
set -euo pipefail

run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$run_dir/../../.." && pwd)"
python="$workspace/.venv-ac-vllm/bin/python"

score_passes() {
  local score="$1"
  [[ -f "$score" ]] && "$python" -c \
    'import json, sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status") == "PASS" else 1)' \
    "$score"
}

bare_score="$run_dir/w3_artifacts/gta1_screenspot_pro/score.json"
if score_passes "$bare_score"; then
  echo "W3 skip PASS: bare GTA1"
else
  echo "W3 run: bare GTA1"
  "$run_dir/w3_baselines/launch_gta1_sanity.sh"
  score_passes "$bare_score"
fi

self_consistency_score="$run_dir/w3_artifacts/gta1_self_consistency_n5_screenspot_pro/score.json"
if score_passes "$self_consistency_score"; then
  echo "W3 skip PASS: GTA1 self-consistency N=5"
else
  echo "W3 run: GTA1 self-consistency N=5"
  "$run_dir/w3_baselines/launch_gta1_self_consistency.sh"
  score_passes "$self_consistency_score"
fi

"$run_dir/w4_run_remaining.sh"