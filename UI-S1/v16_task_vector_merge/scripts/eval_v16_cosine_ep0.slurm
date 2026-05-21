#!/bin/bash
#SBATCH --job-name=ev_c_ep0
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v16_task_vector_merge/scripts/logs/eval_v16_cosine_ep0_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v16_task_vector_merge/scripts/logs/eval_v16_cosine_ep0_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
SERVE_SCRIPT="$PROJECT_ROOT/v16_task_vector_merge/serve_v16_direct.py"
EVAL_SCRIPT="$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

COOP_DIR="$PROJECT_ROOT/v16_task_vector_merge/output/dual_prompt_rl_full_cosine/epoch-0/cooperative"
MODEL_NAME="v16_cosine_ep0"
EVAL_DIR="$PROJECT_ROOT/v16_task_vector_merge/outputs/eval_v16_cosine_ep0"
mkdir -p "$EVAL_DIR"

echo "=========================================="
echo "V16 Cosine RL — epoch-0 (step-31)"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Start:  $(date)"
echo "=========================================="

python "$SERVE_SCRIPT" \
    --base_model "$BASE_MODEL" \
    --coop_checkpoint "$COOP_DIR" \
    --served-model-name "$MODEL_NAME" \
    --port 8000 \
    --num_gpus 4 \
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

python "$EVAL_SCRIPT" \
    --test_data "$TEST_DATA" \
    --api_url "http://localhost:8000/v1" \
    --model_name "$MODEL_NAME" \
    --output_dir "$EVAL_DIR" \
    --threads 4

echo ""
echo "Results:"
cat "$EVAL_DIR"/eval_summary_*.json 2>/dev/null || true

kill $SERVER_PID 2>/dev/null || true
pkill -f "serve_v16_direct.py" 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo "Done: $(date)"
