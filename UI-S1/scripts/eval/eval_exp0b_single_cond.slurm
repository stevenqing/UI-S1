#!/bin/bash
#SBATCH --job-name=exp0b_%x
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/exp0b_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/exp0b_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# Single-condition Exp 0b job — launch one per condition for parallelism.
# Required env: CONDITION, OUTPUT_DIR
# Optional env: MODEL_PATH, VLLM_PORT, EVAL_WORKERS, MAX_SAMPLES

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="$PROJECT_DIR/evaluation"
LOG_DIR="$PROJECT_DIR/train_GUI_360/logs"

MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/train_GUI_360/llamafactory/output/gui360_full_sft_v2/checkpoint-796}"
MODEL_NAME="$MODEL_PATH"
PARQUET="${PARQUET:-$PROJECT_DIR/train_GUI_360/data/gui360_test_sft_eval_format_with_bbox.parquet}"

CONDITION="${CONDITION:?ERROR: must set CONDITION}"
OUTPUT_DIR="${OUTPUT_DIR:?ERROR: must set OUTPUT_DIR}"

VLLM_PORT="${VLLM_PORT:-8000}"
API_URL="http://localhost:${VLLM_PORT}/v1"
EVAL_WORKERS="${EVAL_WORKERS:-128}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

echo "============================================================"
echo "Exp 0b: $CONDITION"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Model: $MODEL_PATH"
echo "Condition: $CONDITION"
echo "Workers: $EVAL_WORKERS"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

export TMPDIR=/tmp/${SLURM_JOB_ID}
mkdir -p $TMPDIR 2>/dev/null || true
if ! touch "$TMPDIR/.test_write" 2>/dev/null; then
    echo "WARNING: /tmp full, using scratch"
    export TMPDIR=/scratch/a5l/shuqing.a5l/tmp/${SLURM_JOB_ID}
    mkdir -p $TMPDIR
fi
rm -f "$TMPDIR/.test_write" 2>/dev/null
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor
export TORCH_HOME=$TMPDIR/torch
export XDG_CACHE_HOME=$TMPDIR/cache
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TORCH_HOME $XDG_CACHE_HOME

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

# Start vLLM
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee "$LOG_DIR/vllm_exp0b_${CONDITION}_${SLURM_JOB_ID}.log" &
VLLM_PID=$!

MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM ready!"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Run single condition
cd "$EVAL_DIR"

echo ""
echo "=== Running: $CONDITION ($(date)) ==="

python eval_exp0b_f4_replication.py \
    --model_name "$MODEL_NAME" \
    --condition "$CONDITION" \
    --parquet_file "$PARQUET" \
    --api_url "$API_URL" \
    --output_dir "$OUTPUT_DIR" \
    --max_workers $EVAL_WORKERS \
    --max_samples $MAX_SAMPLES \
    --save_debug \
    --resume

echo "=== $CONDITION done ($(date)) ==="

# Print summary
for f in "$OUTPUT_DIR"/*${CONDITION}*_summary.json; do
    [ -f "$f" ] && echo "--- $(basename $f) ---" && python3 -m json.tool "$f" && echo
done

kill $VLLM_PID 2>/dev/null || true
echo "Done at $(date)"
