#!/bin/bash
#SBATCH --job-name=eval_os_genesis_hist
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/results/eval_os_genesis_with_history_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/results/eval_os_genesis_with_history_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# Activate environment
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

# Set CUDA library path
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

# Set paths
MODEL_PATH="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/checkpoints/OS-Genesis-7B-AC"
MODEL_NAME="OS-Genesis-7B"
EVAL_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation"
DATASET_PATH="${EVAL_DIR}/dataset/android_control_evaluation_std.jsonl"
OUTPUT_DIR="${EVAL_DIR}/results"

# Change to evaluation directory
cd $EVAL_DIR

echo "=========================================="
echo "Starting OS-Genesis-7B Evaluation WITH HISTORY"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "Start time: $(date)"
echo "=========================================="

# Create output directory if needed
mkdir -p $OUTPUT_DIR

# Fix dataset image paths (replace /datasets/ with actual path)
FIXED_DATASET="${EVAL_DIR}/dataset/android_control_evaluation_fixed.jsonl"
if [ ! -f "$FIXED_DATASET" ]; then
    echo "Fixing dataset image paths..."
    sed 's|/datasets/|/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/|g' \
        "$DATASET_PATH" > "$FIXED_DATASET"
    echo "Fixed dataset created at: $FIXED_DATASET"
fi
DATASET_PATH="$FIXED_DATASET"

# Use legacy vLLM engine for Qwen2-VL models (not Qwen2.5-VL)
export VLLM_USE_V1=0

# Start vLLM server in background
echo "Starting vLLM server..."
vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --tensor-parallel-size 4 \
    --limit-mm-per-prompt image=2 \
    --port 8000 \
    --trust-remote-code \
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

# Run evaluation with history enabled
echo "Starting evaluation WITH HISTORY..."
python $EVAL_DIR/eval_os-genesis-7b-with-history.py \
    --model_name $MODEL_NAME \
    --jsonl_file $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --max_workers 4

EVAL_EXIT_CODE=$?

# Cleanup - kill vLLM server
echo "Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "=========================================="
echo "Evaluation completed at $(date)"
echo "Exit code: $EVAL_EXIT_CODE"
echo "Results saved to: $OUTPUT_DIR/OS_Genesis_7b_with_history.jsonl"
echo "Summary saved to: $OUTPUT_DIR/OS_Genesis_7b_with_history_summary.json"
echo "=========================================="

exit $EVAL_EXIT_CODE
