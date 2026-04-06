#!/bin/bash
#SBATCH --job-name=eval_v8_s60
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_v8_s60_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_v8_s60_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ============================================
# SP + GiGPO + SPWA + Hindsight V8 Step 50 Eval
# ============================================

set -e

source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="${PROJECT_DIR}/evaluation"

CKPT_STEP=60
CKPT_DIR="${PROJECT_DIR}/checkpoints/sp_gigpo_ac/sp_gigpo_spwa_k8_v8/global_step_${CKPT_STEP}/actor"
MERGED_DIR="${PROJECT_DIR}/checkpoints/sp_gigpo_ac/sp_gigpo_spwa_k8_v8/merged_step_${CKPT_STEP}"

MODEL_NAME="sp_gigpo_v8_step${CKPT_STEP}"
DATASET_PATH="${EVAL_DIR}/dataset/android_control_evaluation_std.jsonl"
OUTPUT_DIR="${EVAL_DIR}/results"

cd $PROJECT_DIR

echo "=========================================="
echo "SP+GiGPO+Hindsight V8 Evaluation"
echo "=========================================="
echo "Checkpoint: step ${CKPT_STEP}"
echo "CKPT_DIR: $CKPT_DIR"
echo "MERGED_DIR: $MERGED_DIR"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p $OUTPUT_DIR

# ============================================
# Step 1: Merge FSDP checkpoint to HuggingFace format
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
        echo "ERROR: Checkpoint merge failed!"
        exit 1
    fi
    echo "Merge completed: $MERGED_DIR"
fi

# ============================================
# Step 2: Fix dataset image paths
# ============================================
FIXED_DATASET="${EVAL_DIR}/dataset/android_control_evaluation_fixed.jsonl"
if [ ! -f "$FIXED_DATASET" ]; then
    echo "Fixing dataset image paths..."
    sed 's|/datasets/|/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/|g' \
        "$DATASET_PATH" > "$FIXED_DATASET"
fi
DATASET_PATH="$FIXED_DATASET"

# ============================================
# Step 3: Start vLLM server
# ============================================
echo "Starting vLLM server with merged model..."
vllm serve "$MERGED_DIR" \
    --served-model-name $MODEL_NAME \
    --tensor-parallel-size 4 \
    --limit-mm-per-prompt image=2 \
    --port 8000 \
    --trust-remote-code \
    &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID"

# Wait for vLLM server to be ready
echo "Waiting for vLLM server..."
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

# ============================================
# Step 4: Run evaluation
# ============================================
echo "Starting evaluation..."
cd $EVAL_DIR
python eval_qwenvl.py \
    --model_name $MODEL_NAME \
    --jsonl_file $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --n_history_image_limit 2 \
    --max_workers 16

EVAL_EXIT_CODE=$?

# Cleanup
echo "Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "=========================================="
echo "Evaluation completed at $(date)"
echo "Exit code: $EVAL_EXIT_CODE"
echo "Results: $OUTPUT_DIR/${MODEL_NAME}.jsonl"
echo "=========================================="

exit $EVAL_EXIT_CODE
