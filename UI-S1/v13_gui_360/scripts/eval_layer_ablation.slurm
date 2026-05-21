#!/bin/bash
#SBATCH --job-name=v13_layer_ablation
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/eval_layer_ablation_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/eval_layer_ablation_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V13 Layer Ablation — evaluate with individual layers active
# Uses epoch-3 checkpoint (best: 18.7% TSR)
# ============================================

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
TEST_DATA="$PROJECT_ROOT/v13_gui_360/data/gui360_test_968.jsonl"
COOP_DIR="$PROJECT_ROOT/checkpoints/v13_gui360_rl/epoch-3/cooperative"
OUTPUT_BASE="$PROJECT_ROOT/v13_gui_360/outputs/layer_ablation"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=========================================="
echo "V13 Layer Ablation Eval"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "=========================================="

eval_config() {
    local CONFIG_NAME="$1"
    local ACTIVE_LAYERS="$2"  # comma-separated layer indices, or "all"
    local OUTPUT_DIR="$OUTPUT_BASE/${CONFIG_NAME}"
    local MODEL_NAME="v13_ablation_${CONFIG_NAME}"

    # Skip if already evaluated
    if ls "$OUTPUT_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
        echo "[${CONFIG_NAME}] Already evaluated, skipping."
        return 0
    fi

    echo ""
    echo "=========================================="
    echo "Config: ${CONFIG_NAME} (layers=${ACTIVE_LAYERS})"
    echo "Time: $(date)"
    echo "=========================================="

    mkdir -p "$OUTPUT_DIR"

    # Build server args
    local LAYER_ARG=""
    if [ "$ACTIVE_LAYERS" != "all" ]; then
        LAYER_ARG="--active_layers $ACTIVE_LAYERS"
    fi

    # Start server
    python "$PROJECT_ROOT/v13_gui_360/serve_cooperative_direct.py" \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$COOP_DIR" \
        --version v13 \
        --served-model-name "$MODEL_NAME" \
        --port 8000 \
        $LAYER_ARG \
        &
    local SERVER_PID=$!

    # Wait for server
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
        echo "ERROR: Server failed to start for ${CONFIG_NAME}"
        kill $SERVER_PID 2>/dev/null || true
        return 1
    fi

    # Run eval
    python "$PROJECT_ROOT/v13_gui_360/eval_cooperative.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --threads 4

    # Shutdown
    echo "Shutting down server for ${CONFIG_NAME}..."
    kill $SERVER_PID 2>/dev/null || true
    pkill -f "serve_cooperative_direct" 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 5
}

# ── Individual key layers (from weight analysis: strongest comm) ──
eval_config "only_L27" "27"
eval_config "only_L10" "10"
eval_config "only_L18" "18"

# ── Layer groups ──
eval_config "early_L0-9"   "0,1,2,3,4,5,6,7,8,9"
eval_config "mid_L10-19"   "10,11,12,13,14,15,16,17,18,19"
eval_config "late_L20-27"  "20,21,22,23,24,25,26,27"

# ── All-but-one key layer (leave-one-out) ──
eval_config "no_L10" "0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27"
eval_config "no_L18" "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,27"
eval_config "no_L27" "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26"

# ── Top-3 comm layers only ──
eval_config "top3_L10_L18_L27" "10,18,27"

echo ""
echo "=========================================="
echo "All configs done: $(date)"
echo "=========================================="
