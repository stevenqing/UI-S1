#!/bin/bash
#SBATCH --job-name=eval_qwen25vl
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/results/eval_qwen25vl_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/results/eval_qwen25vl_%j.err
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
MODEL_PATH="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/checkpoints/Qwen2.5-VL-7B-Instruct"
MODEL_NAME="Qwen2.5-VL-7B"
EVAL_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation"
DATASET_PATH="${EVAL_DIR}/dataset/android_control_evaluation_std.jsonl"
OUTPUT_DIR="${EVAL_DIR}/results"

# Change to evaluation directory
cd $EVAL_DIR

echo "=========================================="
echo "Starting Qwen2.5-VL-7B Evaluation"
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

# Run evaluation
echo "Starting evaluation..."
python $EVAL_DIR/eval_qwenvl.py \
    --model_name $MODEL_NAME \
    --jsonl_file $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --n_history_image_limit 2 \
    --max_workers 4

EVAL_EXIT_CODE=$?

# Cleanup - kill vLLM server
echo "Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "=========================================="
echo "Evaluation completed at $(date)"
echo "Exit code: $EVAL_EXIT_CODE"
echo "Results saved to: $OUTPUT_DIR/${MODEL_NAME}.jsonl"
echo "=========================================="

exit $EVAL_EXIT_CODE
