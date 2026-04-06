#!/bin/bash
#SBATCH --job-name=mv_ac_abcd
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/multiview_ac_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/multiview_ac_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Multi-View AC Evaluation — All Variants A/B/C/D
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="$PROJECT_DIR/evaluation"
LOG_DIR="$PROJECT_DIR/scripts/eval/logs"
AC_JSONL="$PROJECT_DIR/evaluation/dataset/android_control_evaluation_fixed.jsonl"

MODEL_PATH="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
MODEL_NAME="$MODEL_PATH"
VLLM_PORT=8000
API_URL="http://localhost:${VLLM_PORT}/v1"

MULTIVIEW_WORKERS=256
OUTPUT_DIR="$PROJECT_DIR/evaluation/results/multiview_ac_variants_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "Multi-View AC Evaluation — Variants A/B/C/D"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Model: $MODEL_PATH"
echo "Dataset: $AC_JSONL"
echo "Output: $OUTPUT_DIR"
echo "Workers: $MULTIVIEW_WORKERS"
echo "=========================================="

# Environment setup
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

# Fix TMPDIR/cache
export TMPDIR=/tmp/${SLURM_JOB_ID}
mkdir -p $TMPDIR
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor
export TORCH_HOME=$TMPDIR/torch
export XDG_CACHE_HOME=$TMPDIR/cache
export HOME_OVERRIDE=$TMPDIR
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TORCH_HOME $XDG_CACHE_HOME

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

# Validate
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"; exit 1
fi
if [ ! -f "$AC_JSONL" ]; then
    echo "ERROR: Dataset not found at $AC_JSONL"; exit 1
fi

# Start vLLM Server
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee "$LOG_DIR/vllm_multiview_ac_${SLURM_JOB_ID}.log" &
VLLM_PID=$!

# Wait for vLLM
MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    echo "Waiting for vLLM... ($WAIT_COUNT/$MAX_WAIT seconds)"
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

cd "$EVAL_DIR"

# Run all 4 variants sequentially
for VARIANT in A B C D; do
    echo ""
    echo "=== Variant $VARIANT (max_workers=$MULTIVIEW_WORKERS) ==="
    echo "Start: $(date)"
    python eval_multiview.py \
        --jsonl_file "$AC_JSONL" \
        --output_dir "$OUTPUT_DIR" \
        --model_name "$MODEL_NAME" \
        --mode multiview \
        --variant $VARIANT \
        --api_url "$API_URL" \
        --max_workers $MULTIVIEW_WORKERS \
        --n_history_image_limit 2 \
        --save_debug
    echo "Variant $VARIANT completed at $(date)."
done

# === Cleanup ===
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true

echo ""
echo "=========================================="
echo "All AC Variants Complete"
echo "End: $(date)"
echo "Results: $OUTPUT_DIR"
echo "=========================================="
