#!/bin/bash
#SBATCH --job-name=eval_moe_v2
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_moe_v2_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_moe_v2_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Evaluate MoE v2 (Multi-Expert with Routing)
# ============================================
# MoE v2: 6 experts with entropy balance loss
# Uses custom MoE server (not vLLM) for inference
# ============================================

set -e

# ============================================
# Configuration
# ============================================
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
EVAL_DIR="$TRAIN_DIR/GUI-360-eval"
LOG_DIR="$TRAIN_DIR/logs"
RESULTS_DIR="$TRAIN_DIR/moe_sft/output/eval_moe_v2"

# MoE model paths
BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
MOE_CHECKPOINT="$TRAIN_DIR/moe_sft/output/moe_sft_v2/final"
SERVE_SCRIPT="$TRAIN_DIR/moe_sft/serve_moe.py"

MODEL_TYPE="qwen2.5_vl_7b"
MODEL_NAME="moe_v2"

# Dataset path
DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"

# Server configuration
SERVER_PORT=19809
SERVER_HOST="localhost"

# Evaluation configuration
EVAL_THREADS=1  # Sequential for MoE model (GPU bound)

# ============================================
# Environment Setup
# ============================================
echo "=========================================="
echo "MoE v2 Evaluation (Multi-Expert Routing)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "MoE Checkpoint: $MOE_CHECKPOINT"
echo "=========================================="

# Activate conda environment
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

# CUDA setup
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Source environment configuration
if [ -f "$PROJECT_DIR/train/env_config.sh" ]; then
    source "$PROJECT_DIR/train/env_config.sh"
fi

# Create directories
mkdir -p $LOG_DIR
mkdir -p $RESULTS_DIR

# Verify MoE checkpoint exists
if [ ! -f "$MOE_CHECKPOINT/moe_config.json" ]; then
    echo "ERROR: MoE checkpoint not found at $MOE_CHECKPOINT"
    echo "Expected file: $MOE_CHECKPOINT/moe_config.json"
    exit 1
fi
echo "MoE checkpoint verified: $MOE_CHECKPOINT"

# ============================================
# Step 1: Start MoE Server
# ============================================
echo "=========================================="
echo "Starting MoE Server (port $SERVER_PORT)"
echo "=========================================="

cd "$PROJECT_DIR"

python $SERVE_SCRIPT \
    --base_model $BASE_MODEL \
    --moe_checkpoint $MOE_CHECKPOINT \
    --port $SERVER_PORT \
    2>&1 | tee $LOG_DIR/moe_v2_server_${SLURM_JOB_ID}.log &

SERVER_PID=$!
echo "MoE server started with PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for MoE server to be ready..."
MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://${SERVER_HOST}:${SERVER_PORT}/health > /dev/null 2>&1; then
        echo "MoE server is ready!"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    echo "Waiting... ($WAIT_COUNT/$MAX_WAIT seconds)"
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: MoE server failed to start within $MAX_WAIT seconds"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# ============================================
# Step 2: Run Evaluations
# ============================================
cd $EVAL_DIR

API_URL="http://${SERVER_HOST}:${SERVER_PORT}/v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

run_eval() {
    local EVAL_TYPE=$1
    echo "=========================================="
    echo "Running $EVAL_TYPE evaluation"
    echo "=========================================="

    local OUTPUT_DIR="$RESULTS_DIR/${EVAL_TYPE}_${TIMESTAMP}"
    mkdir -p $OUTPUT_DIR

    python evaluation.py \
        --root_dir $DATASET_ROOT \
        --type $EVAL_TYPE \
        --model_type $MODEL_TYPE \
        --model_name $MODEL_NAME \
        --api_url $API_URL \
        --threads $EVAL_THREADS \
        --output_dir $OUTPUT_DIR

    echo "$EVAL_TYPE completed. Results: $OUTPUT_DIR"
}

run_eval "grounding"
run_eval "action_prediction"
run_eval "action_prediction_a11y"

# ============================================
# Cleanup
# ============================================
echo "=========================================="
echo "Stopping MoE Server"
echo "=========================================="
kill $SERVER_PID 2>/dev/null || true

# ============================================
# Summary
# ============================================
echo "=========================================="
echo "MoE v2 Evaluation Completed"
echo "=========================================="
echo "End time: $(date)"
echo "Results: $RESULTS_DIR"
echo ""
echo "Results:"
ls -la $RESULTS_DIR/ 2>/dev/null
echo ""
echo "Summaries:"
cat $RESULTS_DIR/*/evaluation_summary_*.json 2>/dev/null || echo "No summaries found"
echo "=========================================="
