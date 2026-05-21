#!/bin/bash
#SBATCH --job-name=ev_ws_g360c
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_warmstart/scripts/logs/eval_gui360_tmpl_coop_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_warmstart/scripts/logs/eval_gui360_tmpl_coop_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Eval warmstart RL with GUI-360 template
# Cooperative: V12 SP ep2, V13 SP ep1
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/gui_action"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=========================================="
echo "Eval: Warmstart RL Cooperative — GUI-360 Template"
echo "Job ID: ${SLURM_JOB_ID} | Start: $(date)"
echo "=========================================="

eval_cooperative() {
    local COOP_DIR="$1"
    local VERSION="$2"     # v12 or v13
    local TAG="$3"         # e.g. v12_sp, v13_sp
    local EPOCH="$4"
    local MODEL_NAME="ws_${TAG}_ep${EPOCH}_g360"
    local OUTPUT_DIR="$PROJECT_ROOT/v13_warmstart/outputs/gui360tmpl_${TAG}_epoch-${EPOCH}"

    if ls "$OUTPUT_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
        echo "[${TAG} epoch-${EPOCH}] Already evaluated, skipping."
        return 0
    fi

    local CKPT_PATH="$COOP_DIR/epoch-${EPOCH}/cooperative"
    if [ ! -f "$CKPT_PATH/lora_weights.pt" ]; then
        echo "[${TAG} epoch-${EPOCH}] Checkpoint not found at $CKPT_PATH, skipping."
        return 0
    fi

    echo ""
    echo "=========================================="
    echo "Evaluating: ${TAG} epoch-${EPOCH} (gui360 template)"
    echo "Time: $(date)"
    echo "=========================================="

    mkdir -p "$OUTPUT_DIR"

    # Start cooperative server (4 GPU replicas)
    python "$PROJECT_ROOT/v13_gui_360/serve_cooperative_direct.py" \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$CKPT_PATH" \
        --version "$VERSION" \
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
        echo "ERROR: Server failed to start for ${TAG} epoch-${EPOCH}"
        kill $SERVER_PID 2>/dev/null || true
        return 1
    fi

    # Eval with GUI-360 template (NOT eval_cooperative.py)
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

# V12 SP warmstart — best epoch (epoch-2)
eval_cooperative "$PROJECT_ROOT/checkpoints/v12_warmstart_rl" "v12" "v12_sp" 2

# V13 SP warmstart — best epoch (epoch-1)
eval_cooperative "$PROJECT_ROOT/checkpoints/v13_warmstart_rl" "v13" "v13_sp" 1

echo "=========================================="
echo "All done: $(date)"
echo "=========================================="
