#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 USER@HOST:/absolute/path/to/UI-S1" >&2
  exit 2
fi

source_root="${1%/}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

paths=(
  runs/allocation-law/2026-08-01/raw/shared_regions_n12.jsonl
  runs/allocation-law/2026-08-01/shards/
  runs/ccm-h2h/2026-07-31/h1/shards/top18/
  runs/ccm-h2h/2026-07-31/h3/shards/qwen3_views/
  runs/ccm-h2h/2026-07-31/h3/shards/uitars_views/
  runs/ccm-h2h/2026-07-31/h3/raw/shared_regions_n4.jsonl
  runs/scaleup/2026-08-02/raw/g2-regions.jsonl
  runs/scaleup/2026-08-02/raw/g2-score-gta1.jsonl
  runs/scaleup/2026-08-02/raw/g2-score-venus.jsonl
  runs/scaleup/2026-08-02/raw/g2-score-qwen35.jsonl
  runs/complementarity/2026-07-30/rows.parquet
)

for relative in "${paths[@]}"; do
  destination="$project_root/$relative"
  if [[ "$relative" == */ ]]; then
    mkdir -p "$destination"
  else
    mkdir -p "$(dirname "$destination")"
  fi
  rsync -aH --partial --info=progress2 "$source_root/$relative" "$destination"
done

"$project_root/.venv-scaleup/bin/python" \
  "$project_root/runs/final/2026-08-04/asset_preflight.py"
