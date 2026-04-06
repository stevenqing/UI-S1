#!/bin/bash
#SBATCH --job-name=sft3_s3_g360
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/sft_v3_step3_gui360_summary_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/sft_v3_step3_gui360_summary_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# SFT v3: GUI-360 Summary Context Evaluation
# Tests action_prediction_ar_summary mode
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="$PROJECT_DIR/train_GUI_360/GUI-360-eval"
LOG_DIR="$PROJECT_DIR/scripts/exp2/logs"
RESULTS_DIR="$PROJECT_DIR/scripts/exp2/results/sft_v3/gui360"

MODEL_NAME="$PROJECT_DIR/train_GUI_360/llamafactory/output/gui360_full_sft_v3_grounding"
MODEL_TYPE="qwen2.5_vl_7b"

DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"

VLLM_PORT=19806
VLLM_HOST="localhost"
EVAL_THREADS=5

echo "=========================================="
echo "SFT v3: GUI-360 Summary Context Evaluation"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Model: $MODEL_NAME"
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
    2>&1 | tee $LOG_DIR/vllm_sft3_s3_gui360_${SLURM_JOB_ID}.log &

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

API_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd $EVAL_DIR

# Run Summary mode evaluation
OUTPUT_DIR="$RESULTS_DIR/summary_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

echo "Running action_prediction_ar_summary evaluation (SFT v3)"
python evaluation.py \
    --root_dir $DATASET_ROOT \
    --type action_prediction_ar_summary \
    --model_type $MODEL_TYPE \
    --model_name $MODEL_NAME \
    --api_url $API_URL \
    --threads $EVAL_THREADS \
    --output_dir $OUTPUT_DIR

echo "Summary evaluation (SFT v3) completed."
cp $OUTPUT_DIR/*summary*.json $RESULTS_DIR/ 2>/dev/null || true
ln -sf $OUTPUT_DIR $RESULTS_DIR/latest_summary

# Cleanup
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true

echo "End: $(date)"
echo "Results: $OUTPUT_DIR"
cat $OUTPUT_DIR/*summary*.json 2>/dev/null || echo "No summaries found"
