#!/bin/bash
#SBATCH --job-name=eval_qwen25vl_base
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_qwen25vl_base_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_qwen25vl_base_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Evaluate Qwen2.5-VL-7B Base Model on GUI-360
# ============================================
# This script:
# 1. Deploys Qwen2.5-VL-7B base model using vLLM
# 2. Runs GUI-360 evaluation (grounding, action_prediction, screen_parsing)
# ============================================

set -e

# ============================================
# Configuration
# ============================================
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
EVAL_DIR="$TRAIN_DIR/GUI-360-eval"
LOG_DIR="$TRAIN_DIR/logs"
RESULTS_DIR="$EVAL_DIR/results/qwen25vl_base"

# Model configuration (use local checkpoint to avoid HuggingFace download + config issues)
MODEL_NAME="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/checkpoints/Qwen2.5-VL-7B-Instruct"
MODEL_TYPE="qwen2.5_vl_7b"

# Dataset path
DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"

# vLLM server configuration
VLLM_PORT=19806
VLLM_HOST="localhost"

# Evaluation configuration
EVAL_THREADS=5
MAX_SAMPLES=""  # Empty for all samples, set number to limit

# ============================================
# Environment Setup
# ============================================
echo "=========================================="
echo "Qwen2.5-VL-7B Base Model Evaluation"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_ROOT"
echo "=========================================="

# Activate conda environment (use /scratch path for compute node compatibility)
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

# Create directories
mkdir -p $LOG_DIR
mkdir -p $RESULTS_DIR

# ============================================
# Check Dataset
# ============================================
if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: Dataset not found at $DATASET_ROOT"
    exit 1
fi

echo "Dataset verified: $DATASET_ROOT"
ls -la $DATASET_ROOT/

# ============================================
# Start vLLM Server
# ============================================
echo "=========================================="
echo "Starting vLLM Server"
echo "=========================================="

# Kill any existing vLLM processes
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

# Start vLLM server in background
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $LOG_DIR/vllm_server_${SLURM_JOB_ID}.log &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Wait for server to be ready
echo "Waiting for vLLM server to be ready..."
MAX_WAIT=600  # 10 minutes
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://${VLLM_HOST}:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "vLLM server is ready!"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    echo "Waiting... ($WAIT_COUNT/$MAX_WAIT seconds)"
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server failed to start within $MAX_WAIT seconds"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# ============================================
# Run Evaluations
# ============================================
cd $EVAL_DIR

API_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Function to run evaluation
run_eval() {
    local EVAL_TYPE=$1
    echo "=========================================="
    echo "Running $EVAL_TYPE evaluation"
    echo "=========================================="

    local OUTPUT_DIR="$RESULTS_DIR/${EVAL_TYPE}_${TIMESTAMP}"
    mkdir -p $OUTPUT_DIR

    local CMD="python evaluation.py \
        --root_dir $DATASET_ROOT \
        --type $EVAL_TYPE \
        --model_type $MODEL_TYPE \
        --model_name $MODEL_NAME \
        --api_url $API_URL \
        --threads $EVAL_THREADS \
        --output_dir $OUTPUT_DIR"

    if [ -n "$MAX_SAMPLES" ]; then
        CMD="$CMD --max_samples $MAX_SAMPLES"
    fi

    echo "Command: $CMD"
    eval $CMD

    echo "$EVAL_TYPE evaluation completed. Results: $OUTPUT_DIR"
}

# Run all three evaluation types
run_eval "grounding"
run_eval "action_prediction"
run_eval "action_prediction_a11y"

# ============================================
# Cleanup
# ============================================
echo "=========================================="
echo "Stopping vLLM Server"
echo "=========================================="
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true

# ============================================
# Summary
# ============================================
echo "=========================================="
echo "Evaluation Completed"
echo "=========================================="
echo "End time: $(date)"
echo "Results directory: $RESULTS_DIR"
echo ""
echo "Results:"
ls -la $RESULTS_DIR/

echo ""
echo "To view results:"
echo "  cat $RESULTS_DIR/*/evaluation_summary_*.json"
echo "=========================================="
