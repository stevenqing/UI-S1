#!/bin/bash
#SBATCH --job-name=eval_lora_v3
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_lora_v3_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/eval_lora_v3_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Evaluate LoRA SFT v3 on GUI-360
# ============================================
# Step 1: Merge LoRA adapter into base model
# Step 2: Start vLLM server with merged model
# Step 3: Run grounding + action_prediction + action_prediction_a11y
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
EVAL_DIR="$TRAIN_DIR/GUI-360-eval"
LOG_DIR="$TRAIN_DIR/logs"
RESULTS_DIR="$EVAL_DIR/results/qwen25vl_lora_v3"

BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
LORA_ADAPTER="$TRAIN_DIR/llamafactory/output/gui360_lora_sft_v3"
MERGED_MODEL="$TRAIN_DIR/llamafactory/output/gui360_lora_sft_v3_merged"
MODEL_TYPE="qwen2.5_vl_7b"

DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"

VLLM_PORT=19808
VLLM_HOST="localhost"
EVAL_THREADS=5

echo "=========================================="
echo "LoRA SFT v3 Evaluation"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Base Model: $BASE_MODEL"
echo "LoRA Adapter: $LORA_ADAPTER"
echo "=========================================="

# Load CUDA and activate conda environment
module load cuda/12.6
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

mkdir -p $LOG_DIR
mkdir -p $RESULTS_DIR

# ============================================
# Step 1: Merge LoRA into base model
# ============================================
if [ -f "$MERGED_MODEL/model.safetensors.index.json" ]; then
    echo "Merged model already exists, skipping merge step."
else
    echo "=========================================="
    echo "Merging LoRA adapter into base model..."
    echo "=========================================="

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
echo "=========================================="
echo "Starting vLLM Server (port $VLLM_PORT)"
echo "=========================================="

pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model $MERGED_MODEL \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $LOG_DIR/vllm_lora_v3_${SLURM_JOB_ID}.log &

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
# Step 3: Run Evaluations
# ============================================
cd $EVAL_DIR

API_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
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
        --model_name $MERGED_MODEL \
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
echo "Stopping vLLM Server"
echo "=========================================="
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true

echo "=========================================="
echo "Evaluation Completed"
echo "=========================================="
echo "End time: $(date)"
echo "Results directory: $RESULTS_DIR"
echo ""
echo "Results:"
ls -la $RESULTS_DIR/
echo ""
echo "Summaries:"
cat $RESULTS_DIR/*/evaluation_summary_*.json 2>/dev/null || echo "No summaries found"
echo "=========================================="
