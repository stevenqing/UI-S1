#!/bin/bash
#SBATCH --job-name=eval_sft_v2_ar
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_sft_v2_ar_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_sft_v2_ar_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Level 2 (Semi-Online AR) Evaluation
# Model: Qwen2.5-VL-7B SFT v2 on GUI-360
# ============================================
# Uses model's own predicted actions as history
# for subsequent steps (autoregressive rollout).
# stop_on_error=True by default.
# ============================================

set -e

# ============================================
# Configuration
# ============================================
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
EVAL_DIR="$TRAIN_DIR/GUI-360-eval"
LOG_DIR="$TRAIN_DIR/logs"
RESULTS_DIR="$EVAL_DIR/results/qwen25vl_sft_v2"

MODEL_NAME="$TRAIN_DIR/llamafactory/output/gui360_full_sft_v2"
MODEL_TYPE="qwen2.5_vl_7b"

DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"

VLLM_PORT=19808
VLLM_HOST="localhost"

EVAL_THREADS=5

# ============================================
# Environment Setup
# ============================================
echo "=========================================="
echo "Qwen2.5-VL-7B SFT v2 - AR Evaluation (Level 2)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Model: $MODEL_NAME"
echo "Eval type: action_prediction_ar (stop_on_error=True)"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

mkdir -p $LOG_DIR
mkdir -p $RESULTS_DIR

# ============================================
# Check Dataset & Model
# ============================================
if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: Dataset not found at $DATASET_ROOT"
    exit 1
fi

if [ ! -f "$MODEL_NAME/model.safetensors.index.json" ]; then
    echo "ERROR: Model checkpoint incomplete at $MODEL_NAME"
    exit 1
fi

echo "Dataset verified: $DATASET_ROOT"
echo "Model verified: $MODEL_NAME"

# ============================================
# Start vLLM Server
# ============================================
echo "=========================================="
echo "Starting vLLM Server (port $VLLM_PORT)"
echo "=========================================="

pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $LOG_DIR/vllm_sft_v2_ar_${SLURM_JOB_ID}.log &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

echo "Waiting for vLLM server to be ready..."
MAX_WAIT=600
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
# Run AR Evaluation
# ============================================
cd $EVAL_DIR

API_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$RESULTS_DIR/action_prediction_ar_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "Running action_prediction_ar evaluation"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

python evaluation.py \
    --root_dir $DATASET_ROOT \
    --type action_prediction_ar \
    --model_type $MODEL_TYPE \
    --model_name $MODEL_NAME \
    --api_url $API_URL \
    --threads $EVAL_THREADS \
    --output_dir $OUTPUT_DIR

echo "AR evaluation completed. Results: $OUTPUT_DIR"

# ============================================
# Run Trajectory Analysis
# ============================================
echo "=========================================="
echo "Running trajectory analysis"
echo "=========================================="

EVAL_JSON=$(find $OUTPUT_DIR -name "*evaluation_results_*.json" | head -1)
if [ -n "$EVAL_JSON" ]; then
    ANALYSIS_DIR="$EVAL_DIR/results/trajectory_analysis/sft_v2_ar"
    mkdir -p $ANALYSIS_DIR

    python scripts/analyze_trajectory_metrics.py \
        --results_file "$EVAL_JSON" \
        --output_dir "$ANALYSIS_DIR" \
        --enable_subgoals \
        --format both

    echo "Trajectory analysis saved: $ANALYSIS_DIR"
else
    echo "WARNING: No evaluation results JSON found for trajectory analysis"
fi

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
echo "AR Evaluation Completed"
echo "=========================================="
echo "End time: $(date)"
echo "Results: $OUTPUT_DIR"
echo ""
echo "Summaries:"
cat $OUTPUT_DIR/evaluation_summary_*.json 2>/dev/null || echo "No summaries found"
echo "=========================================="
