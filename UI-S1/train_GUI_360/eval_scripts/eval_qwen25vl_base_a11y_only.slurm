#!/bin/bash
#SBATCH --job-name=eval_qwen25vl_a11y
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_qwen25vl_base_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_qwen25vl_base_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Re-run action_prediction_a11y only (bugfix: smart_resize factor)
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
EVAL_DIR="$TRAIN_DIR/GUI-360-eval"
LOG_DIR="$TRAIN_DIR/logs"
RESULTS_DIR="$EVAL_DIR/results/qwen25vl_base"

MODEL_NAME="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/checkpoints/Qwen2.5-VL-7B-Instruct"
MODEL_TYPE="qwen2.5_vl_7b"
DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"

VLLM_PORT=19806
VLLM_HOST="localhost"
EVAL_THREADS=5

echo "=========================================="
echo "Qwen2.5-VL-7B A11y Evaluation (re-run)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

mkdir -p $LOG_DIR
mkdir -p $RESULTS_DIR

# Start vLLM Server
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $LOG_DIR/vllm_server_${SLURM_JOB_ID}.log &

VLLM_PID=$!

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

# Run only action_prediction_a11y
cd $EVAL_DIR
API_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$RESULTS_DIR/action_prediction_a11y_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "Running action_prediction_a11y evaluation"
echo "=========================================="

python evaluation.py \
    --root_dir $DATASET_ROOT \
    --type action_prediction_a11y \
    --model_type $MODEL_TYPE \
    --model_name $MODEL_NAME \
    --api_url $API_URL \
    --threads $EVAL_THREADS \
    --output_dir $OUTPUT_DIR

echo "action_prediction_a11y completed. Results: $OUTPUT_DIR"

# Cleanup
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true

echo "=========================================="
echo "Done. End time: $(date)"
echo "=========================================="
