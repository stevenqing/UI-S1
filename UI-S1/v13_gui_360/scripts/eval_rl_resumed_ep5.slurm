#!/bin/bash
#SBATCH --job-name=v13_eval_ep5
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/eval_rl_resumed_ep5_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/eval_rl_resumed_ep5_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# V13 Resumed Eval — epoch-5 only

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
TEST_DATA="$PROJECT_ROOT/v13_gui_360/data/gui360_test_968.jsonl"
COOP_DIR="$PROJECT_ROOT/checkpoints/v13_gui360_rl_resumed/epoch-5/cooperative"
MODEL_NAME="v13_gui360_rl_epoch-5-resumed"
OUTPUT_DIR="$PROJECT_ROOT/v13_gui_360/outputs/epoch-5-resumed"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=========================================="
echo "V13 Eval — epoch-5-resumed"
echo "Job ID: ${SLURM_JOB_ID} | Start: $(date)"
echo "=========================================="

mkdir -p "$OUTPUT_DIR"

python "$PROJECT_ROOT/v13_gui_360/serve_cooperative_direct.py" \
    --base_model "$BASE_MODEL" \
    --coop_checkpoint "$COOP_DIR" \
    --version v13 \
    --served-model-name "$MODEL_NAME" \
    --port 8000 \
    &
SERVER_PID=$!

WAIT_COUNT=0
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
    exit 1
fi

python "$PROJECT_ROOT/v13_gui_360/eval_cooperative.py" \
    --test_data "$TEST_DATA" \
    --api_url "http://localhost:8000/v1" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --threads 4

kill $SERVER_PID 2>/dev/null || true
pkill -f "serve_cooperative_direct" 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
