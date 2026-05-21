#!/bin/bash
#SBATCH --job-name=ev_v15_coop
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/eval_cooperative_sft_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/eval_cooperative_sft_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V15: Eval Cooperative LoRA SFT (all epoch checkpoints)
#
# Uses serve_cooperative_direct.py (same as V13/V14 eval).
# Base model: Qwen2.5-VL-7B-Instruct (matches training).
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
CKPT_ROOT="$PROJECT_ROOT/checkpoints/v15_balanced_cooperative_sft"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p "$PROJECT_ROOT/v15_gui_360/scripts/logs"

echo "=========================================="
echo "V15 Eval: Cooperative LoRA SFT (balanced data)"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

eval_coop() {
    local CKPT_TAG="$1"
    local CKPT_PATH="$CKPT_ROOT/${CKPT_TAG}/cooperative"
    local MODEL_NAME="v15_coop_sft_${CKPT_TAG}"
    local OUTPUT_DIR="$PROJECT_ROOT/v15_gui_360/outputs/eval_coop_sft_${CKPT_TAG}"

    if ls "$OUTPUT_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
        echo "[${CKPT_TAG}] Already evaluated, skipping."
        return 0
    fi

    if [ ! -f "$CKPT_PATH/lora_weights.pt" ]; then
        echo "[${CKPT_TAG}] Checkpoint not found at $CKPT_PATH, skipping."
        return 0
    fi

    echo ""
    echo "=========================================="
    echo "Evaluating: cooperative SFT ${CKPT_TAG}"
    echo "Time: $(date)"
    echo "=========================================="

    mkdir -p "$OUTPUT_DIR"

    python "$PROJECT_ROOT/v13_gui_360/serve_cooperative_direct.py" \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$CKPT_PATH" \
        --version v13 \
        --served-model-name "$MODEL_NAME" \
        --port 8000 \
        &
    local SERVER_PID=$!

    local WAIT_COUNT=0
    while [ $WAIT_COUNT -lt 600 ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "Server ready after ${WAIT_COUNT}s"
            break
        fi
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
    done

    if [ $WAIT_COUNT -ge 600 ]; then
        echo "ERROR: Server failed to start for ${CKPT_TAG}"
        kill $SERVER_PID 2>/dev/null || true
        return 1
    fi

    python "$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --threads 4

    echo "Shutting down server..."
    kill $SERVER_PID 2>/dev/null || true
    pkill -f "serve_cooperative_direct" 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 10
}

# Eval all epoch checkpoints
for ep in epoch-0 epoch-1 epoch-2 epoch-3; do
    eval_coop "$ep"
done

echo "=========================================="
echo "All done: $(date)"
echo "=========================================="
