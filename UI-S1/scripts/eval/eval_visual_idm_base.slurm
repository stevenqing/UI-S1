#!/bin/bash
#SBATCH --job-name=vidm_base
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/vidm_base_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/vidm_base_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ============================================
# Visual IDM Evaluation: Base Model (Qwen2.5-VL-7B-Instruct)
# Tests VLM's zero-shot visual state transition understanding
# Runs 3 modes: standard, hindsight, pure_idm
# ============================================

set -e

source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="${PROJECT_DIR}/evaluation"
SCRIPT_DIR="${PROJECT_DIR}/scripts/eval"

MODEL_DIR="${PROJECT_DIR}/checkpoints/Qwen2.5-VL-7B-Instruct"
MODEL_NAME="qwen25vl_7b_base"
DATASET_PATH="${EVAL_DIR}/dataset/android_control_evaluation_fixed.jsonl"
OUTPUT_DIR="${EVAL_DIR}/results/visual_idm"
MAX_SAMPLES=500

mkdir -p $OUTPUT_DIR
mkdir -p ${SCRIPT_DIR}/logs

echo "=========================================="
echo "Visual IDM Evaluation: Base Model"
echo "=========================================="
echo "Model: $MODEL_DIR"
echo "Dataset: $DATASET_PATH"
echo "Max samples: $MAX_SAMPLES"
echo "Start time: $(date)"
echo "=========================================="

# Fix dataset paths if needed
if [ ! -f "$DATASET_PATH" ]; then
    echo "Fixing dataset image paths..."
    sed 's|/datasets/|/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/|g' \
        "${EVAL_DIR}/dataset/android_control_evaluation_std.jsonl" > "$DATASET_PATH"
fi

# ============================================
# Start vLLM server
# ============================================
echo "Starting vLLM server..."
vllm serve "$MODEL_DIR" \
    --served-model-name $MODEL_NAME \
    --tensor-parallel-size 4 \
    --limit-mm-per-prompt image=3 \
    --port 8000 \
    --trust-remote-code \
    &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for server
MAX_RETRIES=60
RETRY_COUNT=0
while ! curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 10
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Waiting... ($RETRY_COUNT/$MAX_RETRIES)"
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: vLLM server failed to start"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
done
echo "vLLM server ready!"

cd $PROJECT_DIR

# ============================================
# Run 3 modes sequentially
# ============================================

for MODE in standard hindsight pure_idm; do
    echo ""
    echo "=========================================="
    echo "Running mode: $MODE"
    echo "Time: $(date)"
    echo "=========================================="

    python ${SCRIPT_DIR}/eval_visual_idm.py \
        --model_name $MODEL_NAME \
        --jsonl_file $DATASET_PATH \
        --output_dir $OUTPUT_DIR \
        --mode $MODE \
        --max_samples $MAX_SAMPLES \
        --max_workers 128

    echo "Mode $MODE completed at $(date)"
done

# ============================================
# Cleanup
# ============================================
echo "Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "=========================================="
echo "All Visual IDM evaluations completed at $(date)"
echo "Results: $OUTPUT_DIR/"
echo "=========================================="
