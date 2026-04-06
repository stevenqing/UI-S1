#!/bin/bash
#SBATCH --job-name=vidm_v8s50
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/vidm_v8s50_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/vidm_v8s50_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ============================================
# Visual IDM Evaluation: V8 Step 50 (peak checkpoint)
# ============================================

set -e

source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="${PROJECT_DIR}/evaluation"
SCRIPT_DIR="${PROJECT_DIR}/scripts/eval"

MODEL_DIR="${PROJECT_DIR}/checkpoints/sp_gigpo_ac/sp_gigpo_spwa_k8_v8/merged_step_50"
MODEL_NAME="v8_step50"
DATASET_PATH="${EVAL_DIR}/dataset/android_control_evaluation_fixed.jsonl"
OUTPUT_DIR="${EVAL_DIR}/results/visual_idm"
MAX_SAMPLES=500

mkdir -p $OUTPUT_DIR

echo "=========================================="
echo "Visual IDM Evaluation: V8 Step 50"
echo "=========================================="
echo "Model: $MODEL_DIR"
echo "Start time: $(date)"
echo "=========================================="

# Fix dataset paths if needed
if [ ! -f "$DATASET_PATH" ]; then
    sed 's|/datasets/|/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/|g' \
        "${EVAL_DIR}/dataset/android_control_evaluation_std.jsonl" > "$DATASET_PATH"
fi

# Start vLLM
vllm serve "$MODEL_DIR" \
    --served-model-name $MODEL_NAME \
    --tensor-parallel-size 4 \
    --limit-mm-per-prompt image=3 \
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

for MODE in standard hindsight pure_idm; do
    echo ""
    echo "Running mode: $MODE ($(date))"
    python ${SCRIPT_DIR}/eval_visual_idm.py \
        --model_name $MODEL_NAME \
        --jsonl_file $DATASET_PATH \
        --output_dir $OUTPUT_DIR \
        --mode $MODE \
        --max_samples $MAX_SAMPLES \
        --max_workers 128
done

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "V8 Step 50 Visual IDM evaluation completed at $(date)"
