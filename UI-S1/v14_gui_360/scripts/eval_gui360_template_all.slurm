#!/bin/bash
#SBATCH --job-name=ev_g360_all
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v14_gui_360/scripts/logs/eval_gui360_tmpl_all_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v14_gui_360/scripts/logs/eval_gui360_tmpl_all_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Fair eval: ALL RL checkpoints with GUI-360 template
# Baseline: gui_action SFT = 14.4% TSR
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
echo "GUI-360 Template Eval — Fair comparison vs SFT 14.4%"
echo "Job ID: ${SLURM_JOB_ID} | Start: $(date)"
echo "=========================================="

eval_cooperative() {
    local CKPT_PATH="$1"     # path to cooperative/ dir
    local VERSION="$2"       # v12, v13, or v14
    local TAG="$3"           # output tag
    local MODEL_NAME="g360_${TAG}"
    local OUTPUT_DIR="$PROJECT_ROOT/v14_gui_360/outputs/gui360tmpl_${TAG}"

    if ls "$OUTPUT_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
        echo "[${TAG}] Already evaluated, skipping."
        return 0
    fi

    if [ ! -f "$CKPT_PATH/lora_weights.pt" ]; then
        echo "[${TAG}] Checkpoint not found at $CKPT_PATH, skipping."
        return 0
    fi

    echo ""
    echo "=========================================="
    echo "Evaluating: ${TAG} (gui360 template, version=${VERSION})"
    echo "Time: $(date)"
    echo "=========================================="

    mkdir -p "$OUTPUT_DIR"

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
        echo "ERROR: Server failed to start for ${TAG}"
        kill $SERVER_PID 2>/dev/null || true
        return 1
    fi

    # Use gui360 template eval (NOT eval_cooperative.py)
    python "$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --threads 4

    echo "Shutting down server for ${TAG}..."
    kill $SERVER_PID 2>/dev/null || true
    pkill -f "serve_cooperative_direct" 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 10
}

# ── V13 warmstart (old run, shared B) ──
eval_cooperative \
    "$PROJECT_ROOT/checkpoints/v13_warmstart_rl/epoch-0/cooperative" \
    "v13" "v13_ws_ep0"

eval_cooperative \
    "$PROJECT_ROOT/checkpoints/v13_warmstart_rl/epoch-1/cooperative" \
    "v13" "v13_ws_ep1"

# ── V13 warmstart v3 (current run) ──
eval_cooperative \
    "$PROJECT_ROOT/checkpoints/v13_warmstart_rl_v3/epoch-0/cooperative" \
    "v13" "v13_ws_v3_ep0"

# ── V14 warmstart (separate B) ──
eval_cooperative \
    "$PROJECT_ROOT/checkpoints/v14_warmstart_rl/epoch-0/cooperative" \
    "v14" "v14_ws_ep0"

eval_cooperative \
    "$PROJECT_ROOT/checkpoints/v14_warmstart_rl/epoch-1/cooperative" \
    "v14" "v14_ws_ep1"

echo ""
echo "=========================================="
echo "All done: $(date)"
echo "=========================================="
