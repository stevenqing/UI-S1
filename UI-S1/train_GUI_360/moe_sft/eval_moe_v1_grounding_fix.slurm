#!/bin/bash
#SBATCH --job-name=eval_moe_v1_gfix
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_moe_v1_gfix_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_moe_v1_gfix_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Re-evaluate MoE v1 Grounding with parser fix
# (added x1/y1 attribute parsing in step 10)
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
EVAL_DIR="$TRAIN_DIR/GUI-360-eval"
LOG_DIR="$TRAIN_DIR/logs"
RESULTS_DIR="$TRAIN_DIR/moe_sft/output/eval_moe_v1"

MERGED_MODEL="$TRAIN_DIR/moe_sft/output/moe_v1_merged"
MODEL_TYPE="qwen2.5_vl_7b"
DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"

VLLM_PORT=19807
VLLM_HOST="localhost"
EVAL_THREADS=5

echo "=========================================="
echo "MoE v1 Grounding Re-eval (parser fix)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Start vLLM Server
pkill -f "vllm.entrypoints.openai.api_server.*${VLLM_PORT}" || true
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model $MERGED_MODEL \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $LOG_DIR/vllm_moe_v1_gfix_${SLURM_JOB_ID}.log &

VLLM_PID=$!

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
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Run grounding only
cd $EVAL_DIR

API_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$RESULTS_DIR/grounding_fixed_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "Running grounding evaluation"
echo "=========================================="

python evaluation.py \
    --root_dir $DATASET_ROOT \
    --type grounding \
    --model_type $MODEL_TYPE \
    --model_name $MERGED_MODEL \
    --api_url $API_URL \
    --threads $EVAL_THREADS \
    --output_dir $OUTPUT_DIR

echo "Grounding completed. Results: $OUTPUT_DIR"

# Also re-run base model grounding with same parser fix for fair comparison
echo "=========================================="
echo "Re-running base model grounding for comparison"
echo "=========================================="

# Swap to base model
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server.*${VLLM_PORT}" || true
sleep 5

BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
BASE_RESULTS_DIR="$EVAL_DIR/results/qwen25vl_base"

python -m vllm.entrypoints.openai.api_server \
    --model $BASE_MODEL \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $LOG_DIR/vllm_base_gfix_${SLURM_JOB_ID}.log &

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

BASE_OUTPUT_DIR="$BASE_RESULTS_DIR/grounding_fixed_${TIMESTAMP}"
mkdir -p $BASE_OUTPUT_DIR

python evaluation.py \
    --root_dir $DATASET_ROOT \
    --type grounding \
    --model_type $MODEL_TYPE \
    --model_name $BASE_MODEL \
    --api_url $API_URL \
    --threads $EVAL_THREADS \
    --output_dir $BASE_OUTPUT_DIR

echo "Base grounding completed. Results: $BASE_OUTPUT_DIR"

# Cleanup
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server.*${VLLM_PORT}" || true

echo "=========================================="
echo "All done!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "MoE v1 grounding results:"
cat $OUTPUT_DIR/evaluation_summary_*.json 2>/dev/null || echo "No summary found"
echo ""
echo "Base grounding results:"
cat $BASE_OUTPUT_DIR/evaluation_summary_*.json 2>/dev/null || echo "No summary found"
