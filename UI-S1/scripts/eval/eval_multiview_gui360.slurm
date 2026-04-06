#!/bin/bash
#SBATCH --job-name=mv_g360_abcd
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/multiview_gui360_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/multiview_gui360_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Multi-View GUI-360 Evaluation — A/B/C/D in PARALLEL
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="$PROJECT_DIR/evaluation"
LOG_DIR="$PROJECT_DIR/scripts/eval/logs"
PARQUET_FILE="$PROJECT_DIR/train_GUI_360/data/gui360_test_sft_eval_format_with_bbox.parquet"

MODEL_PATH="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
MODEL_NAME="$MODEL_PATH"
VLLM_PORT=8000
API_URL="http://localhost:${VLLM_PORT}/v1"

# Each variant gets 64 workers (256 total across 4 parallel variants)
PER_VARIANT_WORKERS=64

# Reuse existing output dir (Variant A has partial results)
OUTPUT_DIR="$PROJECT_DIR/evaluation/results/multiview_gui360_variants_20260326_073551"

echo "=========================================="
echo "Multi-View GUI-360 — A/B/C/D PARALLEL"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Workers per variant: $PER_VARIANT_WORKERS (x4 = $((PER_VARIANT_WORKERS * 4)) total)"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Environment setup
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

export TMPDIR=/tmp/${SLURM_JOB_ID}
mkdir -p $TMPDIR
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor
export TORCH_HOME=$TMPDIR/torch
export XDG_CACHE_HOME=$TMPDIR/cache
export HOME_OVERRIDE=$TMPDIR
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TORCH_HOME $XDG_CACHE_HOME

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"; exit 1
fi
if [ ! -f "$PARQUET_FILE" ]; then
    echo "ERROR: Dataset not found at $PARQUET_FILE"; exit 1
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
    2>&1 | tee "$LOG_DIR/vllm_multiview_gui360_${SLURM_JOB_ID}.log" &
VLLM_PID=$!

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

# Run all 4 variants IN PARALLEL (each as a background process)
for VARIANT in A B C D; do
    echo "=== Starting Variant $VARIANT (workers=$PER_VARIANT_WORKERS) at $(date) ==="
    python eval_multiview_gui360.py \
        --parquet_file "$PARQUET_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --model_name "$MODEL_NAME" \
        --mode multiview \
        --variant $VARIANT \
        --api_url "$API_URL" \
        --max_workers $PER_VARIANT_WORKERS \
        --save_debug \
        --resume \
        > "$LOG_DIR/gui360_variant_${VARIANT}_${SLURM_JOB_ID}.log" 2>&1 &
    echo "Variant $VARIANT PID: $!"
done

echo ""
echo "All 4 variants launched in parallel. Waiting for completion..."
wait
echo ""

# === Cleanup ===
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true

echo "=========================================="
echo "All GUI-360 Variants Complete"
echo "End: $(date)"
echo "Results: $OUTPUT_DIR"
echo "=========================================="
