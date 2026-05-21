#!/bin/bash
#SBATCH --job-name=ev_v14_g360
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v14_gui_360/scripts/logs/eval_gui360_tmpl_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v14_gui_360/scripts/logs/eval_gui360_tmpl_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Eval v14_warmstart_rl epoch-0 with GUI-360 template
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/gui_action"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
CKPT_ROOT="$PROJECT_ROOT/checkpoints/v14_warmstart_rl"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v14_gui_360/scripts/logs

echo "=========================================="
echo "Eval: V14 Warmstart RL — GUI-360 Template"
echo "Job ID: ${SLURM_JOB_ID} | Start: $(date)"
echo "=========================================="

eval_v14() {
    local CKPT_TAG="$1"
    local CKPT_PATH="$CKPT_ROOT/${CKPT_TAG}/cooperative"
    local MODEL_NAME="v14_ws_${CKPT_TAG}_g360"
    local OUTPUT_DIR="$PROJECT_ROOT/v14_gui_360/outputs/gui360tmpl_v14_ws_${CKPT_TAG}"

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
    echo "Evaluating: v14_ws ${CKPT_TAG} (gui360 template)"
    echo "Time: $(date)"
    echo "=========================================="

    mkdir -p "$OUTPUT_DIR"

    python "$PROJECT_ROOT/v13_gui_360/serve_cooperative_direct.py" \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$CKPT_PATH" \
        --version v14 \
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

# Eval epoch-0 (best V14 checkpoint based on val reward)
eval_v14 "epoch-0"

echo "=========================================="
echo "All done: $(date)"
echo "=========================================="
