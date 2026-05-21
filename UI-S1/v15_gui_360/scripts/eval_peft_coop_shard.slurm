#!/bin/bash
#SBATCH --job-name=ev_coop_sh
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/eval_coop_shard_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/eval_coop_shard_%j.err
#SBATCH --time=3:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Cooperative LoRA eval - single shard
# Usage: sbatch --export=SHARD_START=0,SHARD_END=250 eval_peft_coop_shard.slurm
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
COOP_DIR="$PROJECT_ROOT/checkpoints/v15_peft_balanced_cooperative_r128"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
EVAL_DIR="$PROJECT_ROOT/v15_gui_360/outputs/eval_peft_balanced_cooperative_r128_shard_${SHARD_START}_${SHARD_END}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export TMPDIR="/tmp"

mkdir -p "$PROJECT_ROOT/v15_gui_360/scripts/logs"

echo "=========================================="
echo "Cooperative LoRA Eval Shard [${SHARD_START}:${SHARD_END}]"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

MODEL_NAME="v15_peft_bal_coop_r128"
mkdir -p "$EVAL_DIR"

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

python "$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py" \
    --test_data "$TEST_DATA" \
    --api_url "http://localhost:8000/v1" \
    --model_name "$MODEL_NAME" \
    --output_dir "$EVAL_DIR" \
    --threads 4 \
    --start "$SHARD_START" \
    --end "$SHARD_END"

echo "Shard [${SHARD_START}:${SHARD_END}] results:"
cat "$EVAL_DIR"/eval_summary_*.json 2>/dev/null || true

kill $SERVER_PID 2>/dev/null || true
pkill -f "serve_cooperative_direct" 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo "Done: $(date)"
