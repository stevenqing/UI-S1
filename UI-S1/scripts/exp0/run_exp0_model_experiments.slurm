#!/bin/bash
#SBATCH --job-name=exp0-model
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/outputs/exp0_model_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/outputs/exp0_model_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# =============================================================================
# Experiment 0.1 + 0.2 + 0.5: Model-dependent experiments
#
# Starts vLLM server with SFT model, then runs:
#   0.1: Uncertainty Analysis (K=10 sampling)
#   0.2: Zero-Shot Verification
#   0.5: Retry with Error Context
#
# Usage:
#   sbatch scripts/exp0/run_exp0_model_experiments.slurm
# =============================================================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
LOG_DIR="$PROJECT_DIR/outputs"
cd "${PROJECT_DIR}"

# --- Model config ---
# Option A: Use a merged LoRA checkpoint (recommended)
BASE_MODEL="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
LORA_ADAPTER="$TRAIN_DIR/llamafactory/output/gui360_lora_sft_v4/checkpoint-354"
MERGED_MODEL="$TRAIN_DIR/llamafactory/output/gui360_lora_sft_v4/checkpoint-354_merged"
MODEL_NAME="gui360_lora_v4_ckpt354"

# Option B: Use a full-SFT checkpoint directly (uncomment below)
# MERGED_MODEL="$PROJECT_DIR/checkpoints/gui360_full_sft_v3"
# MODEL_NAME="gui360_full_sft_v3"

VLLM_PORT=19825

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

mkdir -p $LOG_DIR outputs/exp0_1 outputs/exp0_2_v2 outputs/exp0_5_v2

echo "=========================================="
echo "Experiment 0.x: Model Experiments"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

# --- Step 0: Merge LoRA if needed ---
if [ -d "$LORA_ADAPTER" ]; then
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
fi

# --- Step 1: Start vLLM ---
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model $MERGED_MODEL \
    --served-model-name $MODEL_NAME \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $LOG_DIR/vllm_exp0_${SLURM_JOB_ID}.log &

VLLM_PID=$!

echo "Waiting for vLLM server (PID: $VLLM_PID)..."
MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://localhost:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "vLLM server ready after ${WAIT_COUNT}s"
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

ENDPOINT="http://localhost:${VLLM_PORT}/v1"

# --- Step 2: Build pair index cache (for 0.2 & 0.5) ---
echo ""
echo "--- Building pair index cache ---"
python3 -c "
from scripts.exp0.data_utils import _build_request_index
index = _build_request_index(rebuild=False)
paired = sum(1 for v in index.values() if v.get('success') and v.get('fail'))
print(f'Index ready: {paired} paired requests')
"

# --- Step 3: Experiment 0.1 — SKIPPED (already completed) ---
echo ""
echo "Skipping Experiment 0.1 (already completed: outputs/exp0_1/summary.json)"

# --- Step 4: Experiment 0.2 v2 — Zero-Shot Verification (fixed: scrambled negatives) ---
echo ""
echo "=========================================="
echo "Experiment 0.2 v2: Zero-Shot Verification (fixed)"
echo "=========================================="

python3 scripts/exp0/exp0_2_zeroshot_verification.py \
    --model_name "${MODEL_NAME}" \
    --num_examples 200 \
    --endpoint "${ENDPOINT}" \
    --output_dir outputs/exp0_2_v2 \
    || echo "WARNING: Experiment 0.2 failed, continuing..."

# --- Step 5: Experiment 0.5 v2 — Retry with Error Context (fixed: tool_call format) ---
echo ""
echo "=========================================="
echo "Experiment 0.5 v2: Retry with Error Context (fixed)"
echo "=========================================="

python3 scripts/exp0/exp0_5_retry_with_context.py \
    --model_name "${MODEL_NAME}" \
    --num_examples 100 \
    --endpoint "${ENDPOINT}" \
    --output_dir outputs/exp0_5_v2 \
    || echo "WARNING: Experiment 0.5 failed, continuing..."

# --- Cleanup ---
echo ""
echo "=========================================="
echo "All model experiments complete."
echo "End: $(date)"
echo "=========================================="

kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true

echo "Results:"
for d in outputs/exp0_1 outputs/exp0_2_v2 outputs/exp0_5_v2; do
    echo "  $d/"
    ls -la $d/ 2>/dev/null || echo "    (empty)"
done
