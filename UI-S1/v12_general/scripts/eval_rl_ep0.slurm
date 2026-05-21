#!/bin/bash
#SBATCH --job-name=v12_rl_eval
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v12_general/scripts/logs/eval_rl_ep0_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v12_general/scripts/logs/eval_rl_ep0_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V12 RL Eval — Epoch 0 checkpoint
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
DATASET_PATH="$PROJECT_ROOT/datasets/android_control_evaluation_std.jsonl"
COOP_DIR="$PROJECT_ROOT/checkpoints/v12_rl/epoch-0/cooperative"
MERGED_MODEL="/tmp/v12_rl_ep0_${SLURM_JOB_ID}"
MODEL_NAME="v12_rl_ep0"
OUTPUT_DIR="$PROJECT_ROOT/outputs/v12_rl_ep0_ac"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Fix image paths
FIXED_JSONL="/tmp/ac_eval_fixed_${SLURM_JOB_ID}.jsonl"
if [ ! -e /datasets ]; then
    ln -sf "$PROJECT_ROOT/datasets" /datasets 2>/dev/null || {
        sed "s|/datasets/|${PROJECT_ROOT}/datasets/|g" "$DATASET_PATH" > "$FIXED_JSONL"
        DATASET_PATH="$FIXED_JSONL"
        echo "Fixed JSONL paths -> $FIXED_JSONL"
    }
fi

echo "=========================================="
echo "V12 RL Eval — Epoch 0"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"

# Step 1: Merge cooperative LoRA into base model
echo "Merging cooperative LoRA..."
python v12_general/merge_to_hf.py \
    --base_model "$BASE_MODEL" \
    --coop_checkpoint "$COOP_DIR" \
    --output_dir "$MERGED_MODEL"

# Step 2: Start vLLM server
echo "Starting vLLM server..."
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

# Wait for server
WAIT_COUNT=0
while [ $WAIT_COUNT -lt 600 ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM ready after ${WAIT_COUNT}s"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
done

if [ $WAIT_COUNT -ge 600 ]; then
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null || true
    rm -rf "$MERGED_MODEL"
    exit 1
fi

# Step 3: Eval — stop-on-error
echo "Running eval (stop-on-error)..."
python v12_general/eval_ac_vllm.py \
    --model_name "$MODEL_NAME" \
    --jsonl_file "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --n_history_image_limit 2 \
    --max_workers 128

# Step 4: Cleanup
echo "Shutting down vLLM..."
kill $VLLM_PID 2>/dev/null || true
pkill -f "serve_v12_vllm" 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
rm -rf "$MERGED_MODEL"
rm -f "$FIXED_JSONL" 2>/dev/null

echo "=========================================="
echo "Done: $(date)"
echo "Results: $OUTPUT_DIR"
echo "=========================================="
