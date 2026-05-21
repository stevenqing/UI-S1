#!/bin/bash
#SBATCH --job-name=extra_baselines
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/eval_extra_baselines_gui360_test_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/eval_extra_baselines_gui360_test_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Evaluate extra baselines on GUI-360 test
# 1. UI-S1-GRPO-Trained (Qwen2.5-VL)
# 2. UI-TARS-7B-DPO (Qwen2-VL / UI-TARS)
# ============================================

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
GUI360_EVAL="$PROJECT_ROOT/train_GUI_360/GUI-360-eval"
GUI360_DATA="$PROJECT_ROOT/datasets/GUI-360/test"

cd "$GUI360_EVAL"
export PYTHONPATH="$PROJECT_ROOT:$GUI360_EVAL:$PYTHONPATH"

mkdir -p "${PROJECT_ROOT}/v13_gui_360/scripts/logs"

echo "=========================================="
echo "Extra Baselines GUI-360 Test Eval"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "=========================================="

eval_model() {
    local MODEL_PATH="$1"
    local MODEL_NAME="$2"
    local OUTPUT_DIR="$3"
    local MODEL_TYPE="${4:-qwen2.5_vl_7b}"

    echo ""
    echo "=========================================="
    echo "Evaluating: $MODEL_NAME"
    echo "Model:   $MODEL_PATH"
    echo "Type:    $MODEL_TYPE"
    echo "Output:  $OUTPUT_DIR"
    echo "Time:    $(date)"
    echo "=========================================="

    mkdir -p "$OUTPUT_DIR"

    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
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
        echo "ERROR: vLLM server failed to start for $MODEL_NAME"
        kill $VLLM_PID 2>/dev/null || true
        return 1
    fi

    python "$GUI360_EVAL/evaluation.py" \
        --root_dir "$GUI360_DATA" \
        --model_name "$MODEL_NAME" \
        --model_type "$MODEL_TYPE" \
        --type "action_prediction_ar" \
        --output_dir "$OUTPUT_DIR" \
        --api_url "http://localhost:8000/v1" \
        --threads 128

    echo "Shutting down vLLM for $MODEL_NAME..."
    kill $VLLM_PID 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    sleep 10
}

# 1. UI-S1-GRPO-Trained (Qwen2.5-VL)
eval_model \
    "$PROJECT_ROOT/checkpoints/UI-S1-GRPO-Trained" \
    "UI-S1-GRPO-Trained" \
    "$PROJECT_ROOT/outputs/ui_s1_grpo_gui360_test" \
    "qwen2.5_vl_7b"

# 2. UI-TARS-7B-DPO (UI-TARS architecture)
eval_model \
    "$PROJECT_ROOT/checkpoints/UI-TARS-7B-DPO" \
    "UI-TARS-7B-DPO" \
    "$PROJECT_ROOT/outputs/ui_tars_dpo_gui360_test" \
    "ui_tars"

echo ""
echo "=========================================="
echo "Extra baselines done: $(date)"
echo "=========================================="
