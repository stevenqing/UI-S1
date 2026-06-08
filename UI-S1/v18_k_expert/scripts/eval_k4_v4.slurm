#!/bin/bash
#SBATCH --job-name=ev_v18_k4
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v18_k_expert/scripts/logs/eval_k4_v4_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v18_k_expert/scripts/logs/eval_k4_v4_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V18: Eval K=4 Expert Cooperative LoRA (v4)
#
# Uses serve_k_expert_direct.py (4 GPU replicas)
# Eval: eval_gui360_template.py on 1000 episodes
#
# Checkpoints: step-3, step-9 (best val FSR=35%)
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
CKPT_ROOT="$PROJECT_ROOT/checkpoints/v18_rl_k4_v4"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p "$PROJECT_ROOT/v18_k_expert/scripts/logs"

echo "=========================================="
echo "V18 Eval: K=4 Expert Cooperative LoRA (v4)"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

eval_k_expert() {
    local CKPT_TAG="$1"
    local CKPT_PATH="$CKPT_ROOT/${CKPT_TAG}/cooperative"
    local MODEL_NAME="v18_k4_v4_${CKPT_TAG}"
    local OUTPUT_DIR="$PROJECT_ROOT/v18_k_expert/outputs/eval_k4_v4_${CKPT_TAG}"

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
    echo "Evaluating: K=4 v4 ${CKPT_TAG}"
    echo "Time: $(date)"
    echo "=========================================="

    mkdir -p "$OUTPUT_DIR"

    python "$PROJECT_ROOT/v18_k_expert/serve_k_expert_direct.py" \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$CKPT_PATH" \
        --served-model-name "$MODEL_NAME" \
        --port 8000 \
        --max_batch_size 8 \
        --batch_timeout 0.3 \
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
        --threads 32

    echo "Shutting down server..."
    kill $SERVER_PID 2>/dev/null || true
    pkill -f "serve_k_expert_direct" 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 10
}

# Eval best checkpoints (val FSR=35% at step-3 and step-9)
for step in epoch-0_step-9 epoch-0_step-3 epoch-0_step-30; do
    eval_k_expert "$step"
done

echo "=========================================="
echo "All done: $(date)"
echo "=========================================="
