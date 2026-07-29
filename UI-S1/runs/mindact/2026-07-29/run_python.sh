#!/usr/bin/env bash
set -euo pipefail

run_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "$run_dir/../../.." && pwd)"
overlay="$run_dir/.venv/lib/python3.12/site-packages"

exec env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="$overlay" \
  "$workspace/.venv-ac-vllm/bin/python" "$@"