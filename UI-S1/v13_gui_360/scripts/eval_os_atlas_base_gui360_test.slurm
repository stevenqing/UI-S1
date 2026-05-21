#!/bin/bash
#SBATCH --job-name=atlas_base_test
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/eval_os_atlas_base_gui360_test_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/eval_os_atlas_base_gui360_test_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# OS-Atlas-Base-7B — GUI-360 Test Eval
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
MODEL_PATH="$PROJECT_ROOT/checkpoints/OS-Atlas-Base-7B"
GUI360_EVAL="$PROJECT_ROOT/train_GUI_360/GUI-360-eval"
GUI360_DATA="$PROJECT_ROOT/datasets/GUI-360"
MODEL_NAME="OS-Atlas-Base-7B"
OUTPUT_DIR="$PROJECT_ROOT/outputs/os_atlas_base_gui360_test"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$GUI360_EVAL:$PYTHONPATH"

echo "=========================================="
echo "OS-Atlas-Base-7B GUI-360 Test Eval"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"
mkdir -p "${PROJECT_ROOT}/v13_gui_360/scripts/logs"

echo "Starting vLLM server..."
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
VLLM_PID=$!

WAIT_COUNT=0
while [ $WAIT_COUNT -lt 600 ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM ready after ${WAIT_COUNT}s"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
done

if [ $WAIT_COUNT -ge 600 ]; then
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

echo "Running GUI-360 action_prediction_ar eval (test set)..."
python "$GUI360_EVAL/evaluation.py" \
    --root_dir "$GUI360_DATA" \
    --model_name "$MODEL_NAME" \
    --model_type "qwen2.5_vl_7b" \
    --type "action_prediction_ar" \
    --output_dir "$OUTPUT_DIR" \
    --api_url "http://localhost:8000/v1" \
    --threads 128

echo "Shutting down vLLM..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo "=========================================="
echo "Done: $(date)"
echo "Results: $OUTPUT_DIR"
echo "=========================================="
