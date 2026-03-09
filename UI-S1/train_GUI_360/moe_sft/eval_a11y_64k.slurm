#!/bin/bash
#SBATCH --job-name=eval_a11y_64k
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_a11y_64k_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_a11y_64k_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Re-evaluate A11y with 64K context length
# to fix Excel context overflow (85% empty outputs)
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
EVAL_DIR="$TRAIN_DIR/GUI-360-eval"
LOG_DIR="$TRAIN_DIR/logs"

MODEL_TYPE="qwen2.5_vl_7b"
DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"

VLLM_PORT=19807
VLLM_HOST="localhost"
EVAL_THREADS=5
MAX_WAIT=600

echo "=========================================="
echo "A11y Re-eval with 64K context"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# ==========================================
# 1) SFT v2 A11y evaluation
# ==========================================
SFT_V2_MODEL="$TRAIN_DIR/llamafactory/output/gui360_full_sft_v2"
SFT_V2_RESULTS="$EVAL_DIR/results/qwen25vl_sft_v2"

pkill -f "vllm.entrypoints.openai.api_server.*${VLLM_PORT}" || true
sleep 2

echo "Starting vLLM for SFT v2 (64K context)..."
python -m vllm.entrypoints.openai.api_server \
    --model $SFT_V2_MODEL \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 65536 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $LOG_DIR/vllm_sft_v2_a11y_64k_${SLURM_JOB_ID}.log &

VLLM_PID=$!

WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://${VLLM_HOST}:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "SFT v2 vLLM server is ready!"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    echo "Waiting... ($WAIT_COUNT/$MAX_WAIT seconds)"
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

cd $EVAL_DIR
API_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

SFT_V2_OUTPUT="$SFT_V2_RESULTS/action_prediction_a11y_64k_${TIMESTAMP}"
mkdir -p $SFT_V2_OUTPUT

echo "=========================================="
echo "Running SFT v2 A11y evaluation (64K)"
echo "=========================================="

python evaluation.py \
    --root_dir $DATASET_ROOT \
    --type action_prediction_a11y \
    --model_type $MODEL_TYPE \
    --model_name $SFT_V2_MODEL \
    --api_url $API_URL \
    --threads $EVAL_THREADS \
    --output_dir $SFT_V2_OUTPUT

echo "SFT v2 A11y completed. Results: $SFT_V2_OUTPUT"

# ==========================================
# 2) Base model A11y evaluation
# ==========================================
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server.*${VLLM_PORT}" || true
sleep 5

BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
BASE_RESULTS="$EVAL_DIR/results/qwen25vl_base"

echo "Starting vLLM for Base model (64K context)..."
python -m vllm.entrypoints.openai.api_server \
    --model $BASE_MODEL \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 65536 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $LOG_DIR/vllm_base_a11y_64k_${SLURM_JOB_ID}.log &

VLLM_PID=$!

WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://${VLLM_HOST}:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "Base model vLLM server is ready!"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    echo "Waiting... ($WAIT_COUNT/$MAX_WAIT seconds)"
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: Base model vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

BASE_OUTPUT="$BASE_RESULTS/action_prediction_a11y_64k_${TIMESTAMP}"
mkdir -p $BASE_OUTPUT

echo "=========================================="
echo "Running Base A11y evaluation (64K)"
echo "=========================================="

python evaluation.py \
    --root_dir $DATASET_ROOT \
    --type action_prediction_a11y \
    --model_type $MODEL_TYPE \
    --model_name $BASE_MODEL \
    --api_url $API_URL \
    --threads $EVAL_THREADS \
    --output_dir $BASE_OUTPUT

echo "Base A11y completed. Results: $BASE_OUTPUT"

# Cleanup
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server.*${VLLM_PORT}" || true

echo "=========================================="
echo "All done!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "SFT v2 A11y (64K) results:"
cat $SFT_V2_OUTPUT/evaluation_summary_*.json 2>/dev/null || echo "No summary found"
echo ""
echo "Base A11y (64K) results:"
cat $BASE_OUTPUT/evaluation_summary_*.json 2>/dev/null || echo "No summary found"
