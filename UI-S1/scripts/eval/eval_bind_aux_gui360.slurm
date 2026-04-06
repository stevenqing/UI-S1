#!/bin/bash
#SBATCH --job-name=eval_bind_aux
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_bind_aux_%j.out
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_bind_aux_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================================
# Step 2α Eval Track 2: GUI-360 Action Accuracy
# Merge LoRA → vLLM → grounding + action_prediction + a11y
# ============================================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
EVAL_DIR="$TRAIN_DIR/GUI-360-eval"

BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
LORA_ADAPTER="$TRAIN_DIR/llamafactory/output/bind_aux_r32/final"
MERGED_MODEL="$TRAIN_DIR/llamafactory/output/bind_aux_r32/final_merged"
MODEL_TYPE="qwen2.5_vl_7b"

DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$EVAL_DIR/results/bind_aux_r32_${TIMESTAMP}"

VLLM_PORT=19812
VLLM_HOST="localhost"
EVAL_THREADS=5

echo "============================================================"
echo "Step 2α Eval: GUI-360 Action Accuracy (bind_aux_r32)"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "LoRA: $LORA_ADAPTER"
echo "Output: $RESULTS_DIR"
echo "============================================================"

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export PYTHONUNBUFFERED=1

if [ -f "$PROJECT_DIR/train/env_config.sh" ]; then
    source "$PROJECT_DIR/train/env_config.sh"
fi

mkdir -p "$RESULTS_DIR" "$PROJECT_DIR/scripts/eval/logs"

# Step 1: Merge LoRA
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

# Step 2: Start vLLM
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model $MERGED_MODEL \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $PROJECT_DIR/scripts/eval/logs/vllm_bind_aux_${SLURM_JOB_ID}.log &

VLLM_PID=$!
echo "Waiting for vLLM server..."
MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://${VLLM_HOST}:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "vLLM server is ready!"
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

# Step 3: Eval
cd $EVAL_DIR
API_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

run_eval() {
    local EVAL_TYPE=$1
    local OUTPUT_DIR="$RESULTS_DIR/${EVAL_TYPE}_${TIMESTAMP}"
    mkdir -p $OUTPUT_DIR
    echo "Running $EVAL_TYPE evaluation..."
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

kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true

echo ""
echo "============================================================"
echo "All evaluations completed at $(date)"
echo "Results: $RESULTS_DIR"
echo "============================================================"
cat $RESULTS_DIR/*/evaluation_summary_*.json 2>/dev/null || echo "No summaries found"
