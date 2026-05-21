#!/bin/bash
#SBATCH --job-name=v13_exp_bal
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/eval_expert_balanced_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/eval_expert_balanced_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V13 Expert Ablation — Balanced checkpoint epoch-2
# Each expert outputs alone, with and without communication
# Tests: expert_1_only, expert_2_only × {comm, no_comm}
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
COOP_DIR="$PROJECT_ROOT/checkpoints/v13_gui360_rl_balanced/epoch-2/cooperative"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=========================================="
echo "V13 Expert Ablation — Balanced epoch-2"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "=========================================="

# Check checkpoint exists
if [ ! -f "$COOP_DIR/lora_weights.pt" ]; then
    echo "ERROR: Checkpoint not found at $COOP_DIR"
    exit 1
fi

eval_expert() {
    local EXPERT_MODE="$1"
    local COMM_FLAG="$2"  # "" or "--no_comm"
    local COMM_TAG="$3"   # "comm" or "nocomm"
    local MODEL_NAME="v13_bal_ep2_${COMM_TAG}_${EXPERT_MODE}"
    local OUTPUT_DIR="$PROJECT_ROOT/v13_gui_360/outputs/balanced_epoch-2_${COMM_TAG}_${EXPERT_MODE}"

    if ls "$OUTPUT_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
        echo "[${COMM_TAG}/${EXPERT_MODE}] Already evaluated, skipping."
        return 0
    fi

    echo ""
    echo "=========================================="
    echo "Evaluating: ${EXPERT_MODE} (${COMM_TAG})"
    echo "Time: $(date)"
    echo "=========================================="

    mkdir -p "$OUTPUT_DIR"

    python "$PROJECT_ROOT/v13_gui_360/serve_cooperative_direct.py" \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$COOP_DIR" \
        --version v13 \
        --served-model-name "$MODEL_NAME" \
        --expert_mode "$EXPERT_MODE" \
        $COMM_FLAG \
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
        echo "ERROR: Server failed to start for ${EXPERT_MODE} (${COMM_TAG})"
        kill $SERVER_PID 2>/dev/null || true
        return 1
    fi

    python "$PROJECT_ROOT/v13_gui_360/eval_cooperative.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --threads 4

    echo "Shutting down server for ${EXPERT_MODE} (${COMM_TAG})..."
    kill $SERVER_PID 2>/dev/null || true
    pkill -f "serve_cooperative_direct" 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 5
}

# With communication — each expert alone but comm rounds still execute
eval_expert "expert_1_only" "" "comm"
eval_expert "expert_2_only" "" "comm"

# Without communication — comm rounds skipped entirely
eval_expert "expert_1_only" "--no_comm" "nocomm"
eval_expert "expert_2_only" "--no_comm" "nocomm"

echo ""
echo "=========================================="
echo "Expert ablation (balanced epoch-2) done: $(date)"
echo "=========================================="
