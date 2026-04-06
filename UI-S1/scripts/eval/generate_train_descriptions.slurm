#!/bin/bash
#SBATCH --job-name=gen_desc
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/gen_desc_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/gen_desc_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ============================================================
# Offline pi_V description generation for training data
#
# 1000 episodes, ~6536 steps, ~5536 unique s_{t+1} screenshots
# Uses base model to generate consistent descriptions
# Output: ui_s1_train_with_desc.jsonl (adds desc_t1 to each step)
# ============================================================

set -e

source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
SCRIPT_DIR="${PROJECT_DIR}/scripts/eval"

MODEL_DIR="${PROJECT_DIR}/checkpoints/Qwen2.5-VL-7B-Instruct"
MODEL_NAME="qwen25vl_7b_base"
INPUT_JSONL="${PROJECT_DIR}/datasets/ui_s1_dataset/ui_s1_train.jsonl"
OUTPUT_JSONL="${PROJECT_DIR}/datasets/ui_s1_dataset/ui_s1_train_with_desc.jsonl"

echo "=========================================="
echo "Generate pi_V descriptions for training data"
echo "=========================================="
echo "Model: $MODEL_DIR"
echo "Input: $INPUT_JSONL"
echo "Output: $OUTPUT_JSONL"
echo "Start time: $(date)"
echo "=========================================="

# Start vLLM server
vllm serve "$MODEL_DIR" \
    --served-model-name $MODEL_NAME \
    --tensor-parallel-size 4 \
    --limit-mm-per-prompt image=2 \
    --port 8000 \
    --trust-remote-code \
    &

VLLM_PID=$!

MAX_RETRIES=60
RETRY_COUNT=0
while ! curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 10
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: vLLM server failed to start"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
done
echo "vLLM server ready!"

cd $PROJECT_DIR

python ${SCRIPT_DIR}/generate_descriptions.py \
    --input_jsonl $INPUT_JSONL \
    --output_jsonl $OUTPUT_JSONL \
    --model_name $MODEL_NAME \
    --max_workers 128

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo ""
echo "=========================================="
echo "Description generation completed at $(date)"
echo "=========================================="

# Verify output
echo "Output file:"
ls -la $OUTPUT_JSONL
echo "Lines:"
wc -l $OUTPUT_JSONL
