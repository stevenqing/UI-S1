#!/bin/bash
#SBATCH --job-name=v13_resume_eval
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/eval_rl_resumed_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/eval_rl_resumed_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V13 Resumed Eval — epoch-0,1 (logical epoch 3,4)
# from checkpoints/v13_gui360_rl_resumed/
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
TEST_DATA="$PROJECT_ROOT/v13_gui_360/data/gui360_test_968.jsonl"
CKPT_ROOT="$PROJECT_ROOT/checkpoints/v13_gui360_rl_resumed"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=========================================="
echo "V13 Resumed Eval — epoch 3,4 (from resumed training)"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "=========================================="

eval_epoch() {
    local CKPT_TAG="$1"
    local LOGICAL_TAG="$2"
    local COOP_DIR="$CKPT_ROOT/${CKPT_TAG}/cooperative"
    local MODEL_NAME="v13_gui360_rl_${LOGICAL_TAG}"
    local OUTPUT_DIR="$PROJECT_ROOT/v13_gui_360/outputs/${LOGICAL_TAG}"

    if ls "$OUTPUT_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
        echo "[${LOGICAL_TAG}] Already evaluated, skipping."
        return 0
    fi

    if [ ! -f "$COOP_DIR/lora_weights.pt" ]; then
        echo "[${LOGICAL_TAG}] Checkpoint not found at $COOP_DIR, skipping."
        return 0
    fi

    echo ""
    echo "=========================================="
    echo "Evaluating: ${LOGICAL_TAG} (ckpt: ${CKPT_TAG})"
    echo "Time: $(date)"
    echo "=========================================="

    mkdir -p "$OUTPUT_DIR"

    python "$PROJECT_ROOT/v13_gui_360/serve_cooperative_direct.py" \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$COOP_DIR" \
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
        echo "ERROR: Server failed to start for ${LOGICAL_TAG}"
        kill $SERVER_PID 2>/dev/null || true
        return 1
    fi

    python "$PROJECT_ROOT/v13_gui_360/eval_cooperative.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --threads 4

    echo "Shutting down server for ${LOGICAL_TAG}..."
    kill $SERVER_PID 2>/dev/null || true
    pkill -f "serve_cooperative_direct" 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 5
}

# epoch-0 in resumed dir = logical epoch-3 (from epoch-2 checkpoint)
# epoch-1 in resumed dir = logical epoch-4
eval_epoch "epoch-0" "epoch-3-resumed"
eval_epoch "epoch-1" "epoch-4-resumed"

echo ""
echo "=========================================="
echo "All epochs done: $(date)"
echo "=========================================="
