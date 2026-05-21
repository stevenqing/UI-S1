#!/bin/bash
#SBATCH --job-name=v13ws_expert
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_warmstart/scripts/logs/eval_expert_ablation_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_warmstart/scripts/logs/eval_expert_ablation_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V13 Warmstart — Expert Ablation
# Tests: expert_1_only, expert_2_only × {comm, no_comm}
# For both SP and GRPO best epochs
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
echo "V13 Warmstart — Expert Ablation"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "=========================================="

eval_expert() {
    local COOP_DIR="$1"
    local EXPERT_MODE="$2"
    local COMM_FLAG="$3"   # "" or "--no_comm"
    local COMM_TAG="$4"    # "comm" or "nocomm"
    local RUN_TAG="$5"     # e.g. "grpo_epoch-1"
    local MODEL_NAME="v13ws_${RUN_TAG}_${COMM_TAG}_${EXPERT_MODE}"
    local OUTPUT_DIR="$PROJECT_ROOT/v13_warmstart/outputs/${RUN_TAG}_${COMM_TAG}_${EXPERT_MODE}"

    if ls "$OUTPUT_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
        echo "[${RUN_TAG}/${COMM_TAG}/${EXPERT_MODE}] Already evaluated, skipping."
        return 0
    fi

    if [ ! -f "$COOP_DIR/lora_weights.pt" ]; then
        echo "[${RUN_TAG}] Checkpoint not found at $COOP_DIR, skipping."
        return 0
    fi

    echo ""
    echo "=========================================="
    echo "Evaluating: ${RUN_TAG} ${EXPERT_MODE} (${COMM_TAG})"
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
        echo "ERROR: Server failed to start"
        kill $SERVER_PID 2>/dev/null || true
        return 1
    fi

    python "$PROJECT_ROOT/v13_gui_360/eval_cooperative.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --threads 4

    echo "Shutting down server..."
    kill $SERVER_PID 2>/dev/null || true
    pkill -f "serve_cooperative_direct" 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 5
}

# --- V13 Warmstart GRPO epoch-1 ---
GRPO_COOP="$PROJECT_ROOT/checkpoints/v13_warmstart_grpo/epoch-1/cooperative"

eval_expert "$GRPO_COOP" "expert_1_only" ""         "comm"   "grpo_epoch-1"
eval_expert "$GRPO_COOP" "expert_2_only" ""         "comm"   "grpo_epoch-1"
eval_expert "$GRPO_COOP" "expert_1_only" "--no_comm" "nocomm" "grpo_epoch-1"
eval_expert "$GRPO_COOP" "expert_2_only" "--no_comm" "nocomm" "grpo_epoch-1"

# --- V13 Warmstart SP epoch-1 ---
SP_COOP="$PROJECT_ROOT/checkpoints/v13_warmstart_rl/epoch-1/cooperative"

eval_expert "$SP_COOP" "expert_1_only" ""         "comm"   "sp_epoch-1"
eval_expert "$SP_COOP" "expert_2_only" ""         "comm"   "sp_epoch-1"
eval_expert "$SP_COOP" "expert_1_only" "--no_comm" "nocomm" "sp_epoch-1"
eval_expert "$SP_COOP" "expert_2_only" "--no_comm" "nocomm" "sp_epoch-1"

echo ""
echo "=========================================="
echo "All expert ablations done: $(date)"
echo "=========================================="
