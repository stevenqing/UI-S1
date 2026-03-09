#!/bin/bash
#SBATCH --job-name=eval_lora_v3c
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_lora_v3_cont_%x_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_lora_v3_cont_%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Evaluate a single LoRA v3 Cont checkpoint
# Usage: sbatch --export=CKPT_NAME=checkpoint-400 eval_lora_v3_cont_single.slurm
# ============================================

set -e

if [ -z "$CKPT_NAME" ]; then
    echo "ERROR: CKPT_NAME not set. Use --export=CKPT_NAME=checkpoint-XXX"
    exit 1
fi

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
EVAL_DIR="$TRAIN_DIR/GUI-360-eval"
LOG_DIR="$TRAIN_DIR/logs"
RESULTS_DIR="$EVAL_DIR/results/qwen25vl_lora_v3_cont/$CKPT_NAME"

BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
LORA_ADAPTER="$TRAIN_DIR/llamafactory/output/gui360_lora_sft_v3_cont/$CKPT_NAME"
MERGED_MODEL="$TRAIN_DIR/llamafactory/output/gui360_lora_sft_v3_cont_merged_${CKPT_NAME}"

MODEL_TYPE="qwen2.5_vl_7b"
DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"

# Use different ports to avoid conflicts when running in parallel on same node
VLLM_PORT=$((19800 + RANDOM % 100))
VLLM_HOST="localhost"
EVAL_THREADS=5

echo "=========================================="
echo "LoRA v3 Cont - Evaluate $CKPT_NAME"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Adapter: $LORA_ADAPTER"
echo "vLLM Port: $VLLM_PORT"
echo "=========================================="

module load cuda/12.6
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

# Fix temp directory - /local/user may not exist on all nodes
export TMPDIR=/tmp/${SLURM_JOB_ID}
mkdir -p $TMPDIR

mkdir -p $LOG_DIR $RESULTS_DIR

# Verify checkpoint exists
if [ ! -f "$LORA_ADAPTER/adapter_config.json" ]; then
    echo "ERROR: Adapter not found at $LORA_ADAPTER"
    exit 1
fi

# ============================================
# Step 1: Merge LoRA adapter into base model
# ============================================
if [ -f "$MERGED_MODEL/model.safetensors.index.json" ]; then
    echo "Merged model already exists, skipping merge."
else
    echo "Merging LoRA adapter into base model..."
    llamafactory-cli export \
        --model_name_or_path $BASE_MODEL \
        --adapter_name_or_path $LORA_ADAPTER \
        --template qwen2_vl \
        --finetuning_type lora \
        --export_dir $MERGED_MODEL \
        --export_size 5 \
        --export_legacy_format False
    echo "Merge completed at $(date)"
fi

# ============================================
# Step 2: Start vLLM Server
# ============================================
echo "Starting vLLM server on port $VLLM_PORT..."

python -m vllm.entrypoints.openai.api_server \
    --model $MERGED_MODEL \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $LOG_DIR/vllm_lora_v3_cont_${CKPT_NAME}_${SLURM_JOB_ID}.log &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://${VLLM_HOST}:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "vLLM server ready!"
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

# ============================================
# Step 3: Run Evaluations
# ============================================
cd $EVAL_DIR
API_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

for EVAL_TYPE in grounding action_prediction action_prediction_a11y; do
    echo "=========================================="
    echo "Running $EVAL_TYPE for $CKPT_NAME"
    echo "=========================================="

    OUTPUT_DIR="$RESULTS_DIR/${EVAL_TYPE}_${TIMESTAMP}"
    mkdir -p $OUTPUT_DIR

    python evaluation.py \
        --root_dir $DATASET_ROOT \
        --type $EVAL_TYPE \
        --model_type $MODEL_TYPE \
        --model_name $MERGED_MODEL \
        --api_url $API_URL \
        --threads $EVAL_THREADS \
        --output_dir $OUTPUT_DIR || echo "WARNING: $EVAL_TYPE failed for $CKPT_NAME"

    echo "$EVAL_TYPE completed."
done

# ============================================
# Cleanup
# ============================================
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true

# Clean up merged model to save disk space
echo "Cleaning up merged model..."
rm -rf $MERGED_MODEL

echo "=========================================="
echo "Evaluation Complete: $CKPT_NAME"
echo "=========================================="
echo "End time: $(date)"
echo "Results: $RESULTS_DIR"
echo ""
echo "Summaries:"
cat $RESULTS_DIR/*/evaluation_summary_*.json 2>/dev/null || echo "No summaries found"
echo "=========================================="
