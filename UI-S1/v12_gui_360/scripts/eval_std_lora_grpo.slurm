#!/bin/bash
#SBATCH --job-name=eval_stdlora_grpo
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v12_gui_360/scripts/logs/eval_std_lora_grpo_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v12_gui_360/scripts/logs/eval_std_lora_grpo_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# Evaluate: Standard LoRA + GRPO, epochs 0-3

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
TEST_DATA="$PROJECT_ROOT/v13_gui_360/data/gui360_test_968.jsonl"
CKPT_ROOT="$PROJECT_ROOT/checkpoints/standard_lora_grpo_gui360_rl"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=========================================="
echo "Eval: Standard LoRA + GRPO"
echo "Job ID: ${SLURM_JOB_ID} | Start: $(date)"
echo "=========================================="

eval_epoch() {
    local EPOCH="$1"
    local LORA_DIR="$CKPT_ROOT/epoch-${EPOCH}/lora"
    local MERGED_DIR="$CKPT_ROOT/epoch-${EPOCH}/merged"
    local OUTPUT_DIR="$PROJECT_ROOT/v12_gui_360/outputs/std_lora_grpo/epoch-${EPOCH}"
    local MODEL_NAME="std_lora_grpo_epoch${EPOCH}"

    if ls "$OUTPUT_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
        echo "[epoch-${EPOCH}] Already evaluated, skipping."
        return 0
    fi

    if [ ! -f "$MERGED_DIR/config.json" ]; then
        echo "[epoch-${EPOCH}] Merging LoRA..."
        python "$PROJECT_ROOT/v12_gui_360/merge_standard_lora.py" \
            --base_model "$BASE_MODEL" \
            --lora_checkpoint "$LORA_DIR" \
            --output_dir "$MERGED_DIR"
    fi

    echo ""
    echo "=========================================="
    echo "Evaluating: ${MODEL_NAME}"
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
        echo "ERROR: vLLM failed for ${MODEL_NAME}"
        kill $VLLM_PID 2>/dev/null || true
        return 1
    fi

    python "$PROJECT_ROOT/v13_gui_360/eval_cooperative.py" \
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

for EPOCH in 0 1 2 3; do
    eval_epoch "$EPOCH"
done

echo "=========================================="
echo "All epochs done: $(date)"
echo "=========================================="
