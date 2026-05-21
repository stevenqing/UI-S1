#!/bin/bash
#SBATCH --job-name=ev_ws_g360
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_warmstart/scripts/logs/eval_gui360_tmpl_std_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_warmstart/scripts/logs/eval_gui360_tmpl_std_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Eval warmstart RL with GUI-360 template
# Standard LoRA: SP ep2, GRPO ep2
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
echo "Eval: Warmstart RL — GUI-360 Template"
echo "Job ID: ${SLURM_JOB_ID} | Start: $(date)"
echo "=========================================="

eval_std_lora() {
    local CKPT_ROOT="$1"
    local EPOCH="$2"
    local TAG="$3"
    local LORA_DIR="$CKPT_ROOT/epoch-${EPOCH}/lora"
    local MERGED_DIR="$CKPT_ROOT/epoch-${EPOCH}/merged"
    local OUTPUT_DIR="$PROJECT_ROOT/v13_warmstart/outputs/gui360tmpl_${TAG}_epoch-${EPOCH}"
    local MODEL_NAME="ws_${TAG}_ep${EPOCH}"

    if ls "$OUTPUT_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
        echo "[${TAG} epoch-${EPOCH}] Already evaluated, skipping."
        return 0
    fi

    if [ ! -f "$LORA_DIR/lora_weights.pt" ]; then
        echo "[${TAG} epoch-${EPOCH}] LoRA not found, skipping."
        return 0
    fi

    # Merge if needed
    if [ ! -f "$MERGED_DIR/config.json" ]; then
        echo "[${TAG} epoch-${EPOCH}] Merging LoRA..."
        python "$PROJECT_ROOT/v12_gui_360/merge_standard_lora.py" \
            --base_model "$BASE_MODEL" \
            --lora_checkpoint "$LORA_DIR" \
            --output_dir "$MERGED_DIR"
    fi

    echo ""
    echo "=========================================="
    echo "Evaluating: ${TAG} epoch-${EPOCH} (gui360 template)"
    echo "Time: $(date)"
    echo "=========================================="

    mkdir -p "$OUTPUT_DIR"

    python -m vllm.entrypoints.openai.api_server \
        --model "$MERGED_DIR" \
        --served-model-name "$MODEL_NAME" \
        --port 8000 \
        --tensor-parallel-size 4 \
        --max-model-len 32768 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --limit-mm-per-prompt '{"image": 2}' \
        &
    local VLLM_PID=$!

    local WAIT_COUNT=0
    while [ $WAIT_COUNT -lt 600 ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "vLLM ready after ${WAIT_COUNT}s"
            break
        fi
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
    done

    if [ $WAIT_COUNT -ge 600 ]; then
        echo "ERROR: vLLM failed for ${TAG} epoch-${EPOCH}"
        kill $VLLM_PID 2>/dev/null || true
        return 1
    fi

    python "$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --threads 128

    echo "Shutting down vLLM..."
    kill $VLLM_PID 2>/dev/null || true
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    sleep 10
}

# Std LoRA SP warmstart
eval_std_lora "$PROJECT_ROOT/checkpoints/standard_lora_gui360_rl_warmstart" 2 "std_sp"

# Std LoRA GRPO warmstart
eval_std_lora "$PROJECT_ROOT/checkpoints/standard_lora_grpo_gui360_rl_warmstart" 2 "std_grpo"

echo "=========================================="
echo "All done: $(date)"
echo "=========================================="
