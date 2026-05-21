#!/bin/bash
#SBATCH --job-name=v12_eval_vllm
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v12_general/scripts/logs/eval_sft_ac_vllm_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v12_general/scripts/logs/eval_sft_ac_vllm_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V12 SFT Eval — AC trajectory via vLLM
#
# 1) Merge r=0.5 + save correction adapters
# 2) Launch vLLM server with lossless V12 hooks
# 3) Run AR trajectory eval (128 parallel workers)
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
COOP_CKPT="$PROJECT_ROOT/checkpoints/v12_sft/cooperative_final"
MERGED_MODEL="/tmp/v12_merged_lossless_${SLURM_JOB_ID}"
MODEL_NAME="v12_sft_ep4"
DATASET_PATH="$PROJECT_ROOT/datasets/android_control_evaluation_std.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/outputs/v12_sft_ep4_ac_vllm"

echo "=========================================="
echo "V12 Eval — SFT epoch 4, AC (vLLM lossless)"
echo "=========================================="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Base:      $BASE_MODEL"
echo "COOP_CKPT: $COOP_CKPT"
echo "Merged:    $MERGED_MODEL"
echo "Output:    $OUTPUT_DIR"
echo "Start:     $(date)"
echo "=========================================="

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_ROOT/v12_general/scripts/logs"

# Fix image paths: JSONL has /datasets/... but images are at $PROJECT_ROOT/datasets/...
if [ ! -e /datasets ]; then
    ln -sf "$PROJECT_ROOT/datasets" /datasets 2>/dev/null || {
        echo "Cannot create /datasets symlink (no permission). Creating local fix..."
        # Fallback: rewrite JSONL paths
        FIXED_JSONL="/tmp/ac_eval_fixed_${SLURM_JOB_ID}.jsonl"
        sed "s|/datasets/|${PROJECT_ROOT}/datasets/|g" "$DATASET_PATH" > "$FIXED_JSONL"
        DATASET_PATH="$FIXED_JSONL"
        echo "Fixed JSONL: $FIXED_JSONL"
    }
fi

# Step 1: Merge + save corrections
echo "Step 1: Merging r=0.5 + saving correction adapters..."
python v12_general/merge_to_hf.py \
    --base_model "$BASE_MODEL" \
    --coop_checkpoint "$COOP_CKPT" \
    --output_dir "$MERGED_MODEL"

MERGE_EXIT=$?
if [ $MERGE_EXIT -ne 0 ]; then
    echo "ERROR: Merge failed with exit code $MERGE_EXIT"
    exit $MERGE_EXIT
fi
echo "Merge complete."

# Step 2: Start vLLM server with V12 correction hooks
echo "Step 2: Starting vLLM server with V12 lossless corrections..."
python v12_general/serve_v12_vllm.py \
    --model "$MERGED_MODEL" \
    --served-model-name "$MODEL_NAME" \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image": 2}' \
    &
VLLM_PID=$!

MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM server ready after ${WAIT_COUNT}s."
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
    if [ $((WAIT_COUNT % 60)) -eq 0 ]; then
        echo "Waiting for vLLM server... ($WAIT_COUNT/$MAX_WAIT seconds)"
    fi
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null
    rm -rf "$MERGED_MODEL"
    exit 1
fi

# Step 3: Run trajectory eval
echo "Step 3: Running AR trajectory eval..."
cd "$PROJECT_ROOT/evaluation"

python eval_qwenvl.py \
    --model_name "$MODEL_NAME" \
    --jsonl_file "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --n_history_image_limit 2 \
    --max_workers 128

EVAL_EXIT=$?

# Cleanup
kill $VLLM_PID 2>/dev/null || true
pkill -f "serve_v12_vllm" || true
wait $VLLM_PID 2>/dev/null
rm -rf "$MERGED_MODEL"
rm -f "/tmp/ac_eval_fixed_${SLURM_JOB_ID}.jsonl" 2>/dev/null

echo "=========================================="
echo "Done: $(date)  exit=$EVAL_EXIT"
echo "Results: $OUTPUT_DIR"
echo "=========================================="
exit $EVAL_EXIT
