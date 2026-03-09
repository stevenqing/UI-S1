#!/bin/bash
#SBATCH --job-name=eval_moe_v3
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_moe_v3_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_moe_v3_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Evaluate single MoE v3 checkpoint
# Usage: sbatch --export=CKPT_NAME=checkpoint-200,SERVER_PORT=19810 eval_moe_v3_single.slurm
# ============================================

set -e

# Fix TMPDIR for nodes without /local/user
export TMPDIR=/tmp/${SLURM_JOB_ID}
mkdir -p $TMPDIR

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
EVAL_DIR="$TRAIN_DIR/GUI-360-eval"
LOG_DIR="$TRAIN_DIR/logs"

BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
MOE_OUTPUT="$TRAIN_DIR/moe_sft/output/moe_sft_v3"
SERVE_SCRIPT="$TRAIN_DIR/moe_sft/serve_moe.py"
DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"

CKPT_PATH="$MOE_OUTPUT/$CKPT_NAME"
RESULTS_DIR="$TRAIN_DIR/moe_sft/output/eval_moe_v3/$CKPT_NAME"
MODEL_NAME="moe_v3_${CKPT_NAME}"
MODEL_TYPE="qwen2.5_vl_7b"
SERVER_HOST="localhost"
EVAL_THREADS=1

echo "=========================================="
echo "MoE v3 Eval: $CKPT_NAME"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Checkpoint: $CKPT_PATH"
echo "Server port: $SERVER_PORT"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

if [ -f "$PROJECT_DIR/train/env_config.sh" ]; then
    source "$PROJECT_DIR/train/env_config.sh"
fi

mkdir -p $LOG_DIR $RESULTS_DIR

# Verify checkpoint
if [ ! -f "$CKPT_PATH/moe_config.json" ]; then
    echo "ERROR: Checkpoint not found at $CKPT_PATH"
    exit 1
fi

# Start MoE server
cd "$PROJECT_DIR"
python $SERVE_SCRIPT \
    --base_model $BASE_MODEL \
    --moe_checkpoint $CKPT_PATH \
    --port $SERVER_PORT \
    2>&1 | tee $LOG_DIR/moe_v3_server_${CKPT_NAME}_${SLURM_JOB_ID}.log &

SERVER_PID=$!

# Wait for server
MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://${SERVER_HOST}:${SERVER_PORT}/health > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: Server failed to start"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# Run evaluations
cd $EVAL_DIR
API_URL="http://${SERVER_HOST}:${SERVER_PORT}/v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

for EVAL_TYPE in grounding action_prediction; do
    echo "Running $EVAL_TYPE..."
    OUTPUT_DIR="$RESULTS_DIR/${EVAL_TYPE}_${TIMESTAMP}"
    mkdir -p $OUTPUT_DIR

    python evaluation.py \
        --root_dir $DATASET_ROOT \
        --type $EVAL_TYPE \
        --model_type $MODEL_TYPE \
        --model_name $MODEL_NAME \
        --api_url $API_URL \
        --threads $EVAL_THREADS \
        --output_dir $OUTPUT_DIR || echo "WARNING: $EVAL_TYPE failed"

    echo "$EVAL_TYPE done"
done

# Cleanup
kill $SERVER_PID 2>/dev/null || true

echo "=========================================="
echo "Evaluation Complete: $CKPT_NAME"
echo "=========================================="
echo "End time: $(date)"
echo "Results:"
cat $RESULTS_DIR/*/evaluation_summary_*.json 2>/dev/null || echo "No summaries"
echo "=========================================="
