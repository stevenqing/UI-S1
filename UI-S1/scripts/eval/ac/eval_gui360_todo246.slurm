#!/bin/bash
#SBATCH --job-name=gui360_todo
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/gui360_todo246_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/gui360_todo246_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="$PROJECT_DIR/train_GUI_360/GUI-360-eval"
SCRIPT_DIR="$PROJECT_DIR/scripts/eval/ac"
LOG_DIR="$SCRIPT_DIR/logs"

MODEL_NAME="$PROJECT_DIR/train_GUI_360/llamafactory/output/gui360_full_sft_v3_grounding"
DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"

VLLM_PORT=19815
VLLM_HOST="localhost"

echo "=========================================="
echo "GUI-360 TODO 2+4+6 Evaluation"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Model: $MODEL_NAME"
echo "=========================================="

# Use qwen3-eval conda env
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

mkdir -p $LOG_DIR

# Start vLLM server
echo "Starting vLLM server on port $VLLM_PORT..."
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $LOG_DIR/vllm_gui360_todo246_${SLURM_JOB_ID}.log &

VLLM_PID=$!

# Wait for server
MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://${VLLM_HOST}:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "vLLM server is ready! (waited ${WAIT_COUNT}s)"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

cd $EVAL_DIR

# ─── TODO 6: Error Classification (fastest, uses existing multisample data) ───
echo ""
echo "=========================================="
echo "Running TODO 6: Error Classification"
echo "=========================================="
MULTISAMPLE="$PROJECT_DIR/outputs/eval_gui360_multisample/multisample_results.jsonl"
if [ -f "$MULTISAMPLE" ]; then
    python $SCRIPT_DIR/eval_gui360_todo6_error_classify.py \
        --model_name $MODEL_NAME \
        --api_url http://${VLLM_HOST}:${VLLM_PORT}/v1 \
        --multisample_results $MULTISAMPLE \
        --output_dir $PROJECT_DIR/outputs/eval_gui360_todo6 \
        --max_workers 16
    echo "TODO 6 done."
else
    echo "SKIP: multisample_results.jsonl not found"
fi

# ─── TODO 2: Sub-Task Decomposition ───
echo ""
echo "=========================================="
echo "Running TODO 2: Sub-Task Decomposition"
echo "=========================================="
python $SCRIPT_DIR/eval_gui360_todo2_subtask.py \
    --model_name $MODEL_NAME \
    --api_url http://${VLLM_HOST}:${VLLM_PORT}/v1 \
    --root_dir $DATASET_ROOT \
    --output_dir $PROJECT_DIR/outputs/eval_gui360_todo2 \
    --max_workers 32
echo "TODO 2 done."

# ─── TODO 4: Summary Augmented ───
echo ""
echo "=========================================="
echo "Running TODO 4: Summary Augmented"
echo "=========================================="
python $SCRIPT_DIR/eval_gui360_todo4_summary.py \
    --model_name $MODEL_NAME \
    --api_url http://${VLLM_HOST}:${VLLM_PORT}/v1 \
    --root_dir $DATASET_ROOT \
    --output_dir $PROJECT_DIR/outputs/eval_gui360_todo4 \
    --max_workers 16
echo "TODO 4 done."

# Cleanup
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true

echo ""
echo "=========================================="
echo "All GUI-360 TODO 2+4+6 completed at $(date)"
echo "=========================================="
