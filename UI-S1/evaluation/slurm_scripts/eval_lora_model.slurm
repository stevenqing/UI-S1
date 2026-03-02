#!/bin/bash
#SBATCH --job-name=eval_lora
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/results/eval_lora_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/results/eval_lora_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ========================================
# LORA Model Evaluation Script
# ========================================
# Evaluates LoRA checkpoints from gui_traj_grpo
#
# Usage:
#   sbatch eval_lora_model.slurm [CHECKPOINT_NAME] [STEP]
#
# Example:
#   sbatch eval_lora_model.slurm qwenvl_uis1_LORA_r64_a128_gamma_0.5 30
#
# ========================================

# Activate environment
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

# Set CUDA library path
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

# Paths
BASE_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="${BASE_DIR}/evaluation"
DATASET_PATH="${EVAL_DIR}/dataset/android_control_evaluation_std.jsonl"
OUTPUT_DIR="${EVAL_DIR}/results"

# Default checkpoint configuration
CHECKPOINT_NAME="${1:-qwenvl_uis1_LORA_r64_a128_gamma_0.5}"
STEP="${2:-30}"

CHECKPOINT_DIR="${BASE_DIR}/checkpoints/gui_traj_grpo/${CHECKPOINT_NAME}/global_step_${STEP}"
LORA_ADAPTER_PATH="${CHECKPOINT_DIR}/actor/lora_adapter"
BASE_MODEL="${BASE_DIR}/checkpoints/Qwen2.5-VL-7B-Instruct"
MODEL_NAME="LORA_${CHECKPOINT_NAME}_step${STEP}"

echo "=========================================="
echo "Starting LORA Model Evaluation"
echo "Checkpoint: ${CHECKPOINT_NAME}"
echo "Step: ${STEP}"
echo "LORA Adapter: ${LORA_ADAPTER_PATH}"
echo "Base Model: ${BASE_MODEL}"
echo "Start time: $(date)"
echo "=========================================="

# Check if LORA adapter exists
if [ ! -f "${LORA_ADAPTER_PATH}/adapter_model.safetensors" ]; then
    echo "ERROR: LORA adapter not found at ${LORA_ADAPTER_PATH}"
    echo "Available checkpoints:"
    ls -la "${BASE_DIR}/checkpoints/gui_traj_grpo/${CHECKPOINT_NAME}/"
    exit 1
fi

# Start vLLM server with LoRA support
echo ""
echo "Starting vLLM server with LoRA adapter..."
echo "=========================================="

cd $EVAL_DIR

vllm serve $BASE_MODEL \
    --served-model-name $MODEL_NAME \
    --enable-lora \
    --lora-modules "${MODEL_NAME}=${LORA_ADAPTER_PATH}" \
    --tensor-parallel-size 4 \
    --limit-mm-per-prompt image=2 \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 16384 \
    --max-lora-rank 64 \
    &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Wait for vLLM server to be ready
echo "Waiting for vLLM server to start..."
MAX_RETRIES=60
RETRY_COUNT=0
while ! curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 10
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Waiting for server... ($RETRY_COUNT/$MAX_RETRIES)"
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: vLLM server failed to start within timeout"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
done

echo "vLLM server is ready!"

# Run evaluation
echo ""
echo "Running evaluation..."
echo "=========================================="

python $EVAL_DIR/eval_qwenvl.py \
    --model_name $MODEL_NAME \
    --jsonl_file $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --n_history_image_limit 2 \
    --max_workers 4

EVAL_EXIT_CODE=$?

# Cleanup - kill vLLM server
echo ""
echo "Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "=========================================="
echo "Evaluation completed at $(date)"
echo "Exit code: $EVAL_EXIT_CODE"
echo "Results saved to: $OUTPUT_DIR/${MODEL_NAME}.jsonl"
echo "=========================================="

# Analyze results
if [ -f "$OUTPUT_DIR/${MODEL_NAME}.jsonl" ]; then
    echo ""
    echo "Results Summary:"
    echo "=========================================="
    python -c "
import json
results = []
with open('$OUTPUT_DIR/${MODEL_NAME}.jsonl', 'r') as f:
    for line in f:
        results.append(json.loads(line))
success = sum(1 for r in results if r.get('task_success', False))
total = len(results)
print(f'Total tasks: {total}')
print(f'Successful tasks: {success}')
print(f'Success rate: {success/total*100:.2f}%' if total > 0 else 'N/A')
"
fi

exit $EVAL_EXIT_CODE
