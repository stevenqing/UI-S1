#!/bin/bash
#SBATCH --job-name=vidm_v6s90
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/vidm_v6s90_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/vidm_v6s90_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ============================================
# Visual IDM Evaluation: V6 Step 90 (peak checkpoint)
# Merge FSDP checkpoint first, then run 3-mode eval
# ============================================

set -e

source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="${PROJECT_DIR}/evaluation"
SCRIPT_DIR="${PROJECT_DIR}/scripts/eval"

CKPT_STEP=90
CKPT_DIR="${PROJECT_DIR}/checkpoints/sp_gigpo_ac/sp_gigpo_spwa_k8_v6/global_step_${CKPT_STEP}/actor"
MERGED_DIR="${PROJECT_DIR}/checkpoints/sp_gigpo_ac/sp_gigpo_spwa_k8_v6/merged_step_${CKPT_STEP}"

MODEL_NAME="v6_step90"
DATASET_PATH="${EVAL_DIR}/dataset/android_control_evaluation_fixed.jsonl"
OUTPUT_DIR="${EVAL_DIR}/results/visual_idm"
MAX_SAMPLES=500

mkdir -p $OUTPUT_DIR

cd $PROJECT_DIR

echo "=========================================="
echo "Visual IDM Evaluation: V6 Step 90"
echo "=========================================="
echo "Start time: $(date)"
echo "=========================================="

# ============================================
# Step 1: Merge FSDP checkpoint
# ============================================
if [ -d "$MERGED_DIR" ] && [ -f "$MERGED_DIR/config.json" ]; then
    echo "Merged checkpoint already exists, skipping merge."
else
    echo "Merging FSDP checkpoint..."
    python ${PROJECT_DIR}/scripts/model_merger.py merge \
        --backend fsdp \
        --local_dir "$CKPT_DIR" \
        --target_dir "$MERGED_DIR"

    if [ $? -ne 0 ]; then
        echo "ERROR: Checkpoint merge failed"
        exit 1
    fi
    echo "Merge completed: $MERGED_DIR"
fi

# ============================================
# Step 2: Start vLLM server
# ============================================
echo "Starting vLLM server..."
vllm serve "$MERGED_DIR" \
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

# ============================================
# Step 3: Run 3 modes
# ============================================
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

echo "V6 Step 90 Visual IDM evaluation completed at $(date)"
