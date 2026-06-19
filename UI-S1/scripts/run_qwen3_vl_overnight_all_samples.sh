#!/usr/bin/env bash
set -euo pipefail

# Overnight all-sample Qwen3-VL bottleneck validation.
# Runs all current-protocol samples: every real segment boundary plus one random
# control per episode, with thinking and non-thinking modes.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-datasets/model_bottleneck_validation_qwen3vl_all_samples_overnight_${STAMP}}"
INPUTS="${INPUTS:-datasets/segmentation_train/gui_odyssey_segments.jsonl}"

VL_ENV="${VL_ENV:-$PROJECT_ROOT/.venv-qwen3-vllm}"
VL_CUDA_VISIBLE_DEVICES="${VL_CUDA_VISIBLE_DEVICES:-4,5}"
VL_PORT="${VL_PORT:-8000}"
VL_TP="${VL_TP:-2}"
VL_GPU_MEMORY_UTILIZATION="${VL_GPU_MEMORY_UTILIZATION:-0.70}"
VL_MAX_MODEL_LEN="${VL_MAX_MODEL_LEN:-8192}"
REQUEST_WORKERS="${REQUEST_WORKERS:-16}"
MAX_TOKENS="${MAX_TOKENS:-512}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
RESUME_PARTIAL="${RESUME_PARTIAL:-0}"
CASE_SHARD_INDEX="${CASE_SHARD_INDEX:-0}"
CASE_SHARD_COUNT="${CASE_SHARD_COUNT:-1}"
LONG_STEP_THRESHOLD="${LONG_STEP_THRESHOLD:-10}"
LONG_PREV_SEGMENTS_THRESHOLD="${LONG_PREV_SEGMENTS_THRESHOLD:-2}"

mkdir -p "$OUTPUT_DIR"

echo "Output: $OUTPUT_DIR"
python - <<'PY'
import json
from pathlib import Path
path = Path('datasets/segmentation_train/gui_odyssey_segments.jsonl')
episodes = steps = real = 0
with path.open() as handle:
    for line in handle:
        if not line.strip():
            continue
        episode = json.loads(line)
        episodes += 1
        steps += len(episode.get('steps', []))
        real += sum(1 for segment in episode.get('segments', []) if segment.get('start_step', 0) > 0)
print(f'episodes={episodes} steps={steps} real_boundaries={real} random_controls={episodes}')
print(f'cases={real + episodes} requests={(real + episodes) * 2 * 4}')
PY

VL_ENV="$VL_ENV" \
TEXT_ENV="$PROJECT_ROOT/.venv-qwen35-vllm" \
VL_CUDA_VISIBLE_DEVICES="$VL_CUDA_VISIBLE_DEVICES" \
VL_PORT="$VL_PORT" \
VL_TP="$VL_TP" \
VL_GPU_MEMORY_UTILIZATION="$VL_GPU_MEMORY_UTILIZATION" \
VL_MAX_MODEL_LEN="$VL_MAX_MODEL_LEN" \
REQUEST_WORKERS="$REQUEST_WORKERS" \
REQUEST_TIMEOUT="$REQUEST_TIMEOUT" \
MAX_CASES=999999999 \
MAX_TOKENS="$MAX_TOKENS" \
THINKING_MODES="non_thinking thinking" \
MODELS="vl" \
INPUTS="$INPUTS" \
OUTPUT_DIR="$OUTPUT_DIR" \
RESUME_PARTIAL="$RESUME_PARTIAL" \
CASE_SHARD_INDEX="$CASE_SHARD_INDEX" \
CASE_SHARD_COUNT="$CASE_SHARD_COUNT" \
bash scripts/run_qwen3_bottleneck_validation.sh

python scripts/analyze_model_bottleneck_hard_cases.py \
  --results "$OUTPUT_DIR/model_behavior_results.jsonl" \
  --model-key qwen3_vl_8b \
  --episodes $INPUTS \
  --long-step-threshold "$LONG_STEP_THRESHOLD" \
  --long-prev-segments-threshold "$LONG_PREV_SEGMENTS_THRESHOLD" \
  --output-dir "$OUTPUT_DIR/hard_case_analysis"

echo "Done: $OUTPUT_DIR"