#!/usr/bin/env bash
# Run Qwen3-VL behavior validation as four independent vLLM servers/shards.
# This is faster than one tensor-parallel server for the many small requests in
# the intervention protocol.

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$PROJECT_ROOT"

BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-datasets/model_bottleneck_validation_qwen3vl_restore_20260620_sharded}
MERGED_DIR=${MERGED_DIR:-datasets/model_bottleneck_validation_qwen3vl_restore_20260620_sharded/merged}
INPUTS=${INPUTS:-datasets/segmentation_train/gui_odyssey_segments.jsonl}
VL_ENV=${VL_ENV:-$PROJECT_ROOT/.venv-qwen3-vllm}
GPUS_CSV=${GPUS_CSV:-4,5,6,7}
BASE_PORT=${BASE_PORT:-8000}
REQUEST_WORKERS=${REQUEST_WORKERS:-32}
VL_GPU_MEMORY_UTILIZATION=${VL_GPU_MEMORY_UTILIZATION:-0.55}
VL_MAX_MODEL_LEN=${VL_MAX_MODEL_LEN:-8192}
MAX_TOKENS=${MAX_TOKENS:-512}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-600}
RESUME_PARTIAL=${RESUME_PARTIAL:-1}
LONG_STEP_THRESHOLD=${LONG_STEP_THRESHOLD:-10}
LONG_PREV_SEGMENTS_THRESHOLD=${LONG_PREV_SEGMENTS_THRESHOLD:-2}

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
SHARD_COUNT=${#GPUS[@]}

if [[ "$SHARD_COUNT" -lt 1 ]]; then
  echo "No GPUs provided in GPUS_CSV=$GPUS_CSV" >&2
  exit 1
fi
if [[ ! -x "$VL_ENV/bin/python" || ! -x "$VL_ENV/bin/vllm" ]]; then
  echo "Missing Qwen3-VL vLLM env under $VL_ENV" >&2
  exit 1
fi
if [[ ! -f "$INPUTS" ]]; then
  echo "Missing INPUTS=$INPUTS" >&2
  exit 1
fi

mkdir -p "$BASE_OUTPUT_DIR" "$MERGED_DIR"

echo "Qwen3-VL sharded validation"
echo "Output base: $BASE_OUTPUT_DIR"
echo "Merged dir:  $MERGED_DIR"
echo "Inputs:      $INPUTS"
echo "GPUs:        $GPUS_CSV"
echo "Shards:      $SHARD_COUNT"
echo "Workers/shard: $REQUEST_WORKERS"

pids=()
for shard in "${!GPUS[@]}"; do
  gpu="${GPUS[$shard]}"
  port=$((BASE_PORT + shard))
  shard_dir="$BASE_OUTPUT_DIR/shard_${shard}"
  mkdir -p "$shard_dir"
  log="$BASE_OUTPUT_DIR/shard_${shard}.log"
  echo "Launching shard $shard/$SHARD_COUNT gpu=$gpu port=$port output=$shard_dir"
  (
    cd "$PROJECT_ROOT"
    VL_ENV="$VL_ENV" \
    VL_CUDA_VISIBLE_DEVICES="$gpu" \
    VL_PORT="$port" \
    VL_TP=1 \
    VL_GPU_MEMORY_UTILIZATION="$VL_GPU_MEMORY_UTILIZATION" \
    VL_MAX_MODEL_LEN="$VL_MAX_MODEL_LEN" \
    VL_EXTRA_ARGS='' \
    REQUEST_WORKERS="$REQUEST_WORKERS" \
    REQUEST_TIMEOUT="$REQUEST_TIMEOUT" \
    MAX_TOKENS="$MAX_TOKENS" \
    RESUME_PARTIAL="$RESUME_PARTIAL" \
    CASE_SHARD_INDEX="$shard" \
    CASE_SHARD_COUNT="$SHARD_COUNT" \
    OUTPUT_DIR="$shard_dir" \
    INPUTS="$INPUTS" \
    bash scripts/run_qwen3_vl_overnight_all_samples.sh
  ) > "$log" 2>&1 &
  pids+=("$!")
done

failed=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  if wait "$pid"; then
    echo "Shard $i finished successfully."
  else
    echo "Shard $i failed; see $BASE_OUTPUT_DIR/shard_${i}.log" >&2
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi

echo "Merging shard outputs into $MERGED_DIR"
BASE_OUTPUT_DIR="$BASE_OUTPUT_DIR" MERGED_DIR="$MERGED_DIR" python3 - <<'PY'
import json
import os
from pathlib import Path

base = Path(os.environ['BASE_OUTPUT_DIR'])
merged = Path(os.environ['MERGED_DIR'])
merged.mkdir(parents=True, exist_ok=True)
rows = []
summaries = {}
progress = {}
for shard_dir in sorted(path for path in base.glob('shard_*') if path.is_dir()):
    result_path = shard_dir / 'model_behavior_results.jsonl'
    if not result_path.exists():
        raise SystemExit(f'missing {result_path}')
    shard_rows = []
    with result_path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                shard_rows.append(row)
                rows.append(row)
    summary_path = shard_dir / 'summary.json'
    if summary_path.exists():
        summaries[shard_dir.name] = json.loads(summary_path.read_text())
    progress_path = shard_dir / 'qwen3_vl_8b_progress.json'
    if progress_path.exists():
        progress[shard_dir.name] = json.loads(progress_path.read_text())
    print(f'{shard_dir.name}: {len(shard_rows)} rows')

rows.sort(key=lambda row: (
    str(row.get('model_key', '')),
    str(row.get('thinking_mode', '')),
    str(row.get('case_kind', '')),
    int(row.get('case_id', -1)),
    str(row.get('condition', '')),
))
with (merged / 'model_behavior_results.jsonl').open('w', encoding='utf-8') as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
summary = {
    'rows': len(rows),
    'shards': summaries,
    'progress': progress,
}
(merged / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
report_lines = ['# Qwen3-VL Sharded Behavior Validation Merge', '', f'- rows: {len(rows)}', '']
for name, item in sorted(progress.items()):
    report_lines.append(f"- {name}: {item.get('completed_requests')} / {item.get('total_requests')} requests; errors={item.get('error_requests')}")
(merged / 'model_behavior_report.md').write_text('\n'.join(report_lines) + '\n', encoding='utf-8')
print(f'merged rows={len(rows)} output={merged / "model_behavior_results.jsonl"}')
PY

python3 scripts/analyze_model_bottleneck_hard_cases.py \
  --results "$MERGED_DIR/model_behavior_results.jsonl" \
  --model-key qwen3_vl_8b \
  --episodes "$INPUTS" \
  --long-step-threshold "$LONG_STEP_THRESHOLD" \
  --long-prev-segments-threshold "$LONG_PREV_SEGMENTS_THRESHOLD" \
  --output-dir "$MERGED_DIR/hard_case_analysis"

echo "Done: $MERGED_DIR"