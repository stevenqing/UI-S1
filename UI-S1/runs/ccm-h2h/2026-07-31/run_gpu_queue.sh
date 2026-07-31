#!/usr/bin/env bash
set -euo pipefail

run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
protected_pid="${PROTECTED_PID:-1814}"

if kill -0 "$protected_pid" 2>/dev/null; then
  echo "Waiting for protected PID $protected_pid to release GPUs 0-7"
  tail --pid="$protected_pid" -f /dev/null
fi

echo "Protected PID released; starting H1 on GPUs 0-7"
"$run_dir/h1/run_h1.sh"

python_bin="$(cd "$run_dir/../../.." && pwd)/.venv-ac-vllm/bin/python"
"$python_bin" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("status") == "PASS"' \
  "$run_dir/h1_headtohead.json"
echo "H1 PASS"

h2_gate=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["summary"]["h3_gate"])' "$run_dir/h2_collision_floor.json")
if [[ "$h2_gate" == "OPEN" ]]; then
  echo "H2 gate OPEN; H3 implementation/model preflight is required before inference"
else
  echo "H2 gate closed; H3 will not run"
fi
