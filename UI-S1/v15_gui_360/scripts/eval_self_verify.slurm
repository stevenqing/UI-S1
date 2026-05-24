#!/bin/bash
#SBATCH --job-name=self_verify
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/self_verify_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/self_verify_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Self-Verification & Retry Experiment
#
# For each step:
#   1. Predict action (greedy T=0)
#   2. Verify: "does this coordinate target the right element?"
#   3. If WRONG → retry with verification feedback (up to 2 retries)
#
# Directly targets the binary grounding problem.
# Model: gui360_balanced_full_sft checkpoint-272
# GT history (teacher-forced)
#
# Estimated: ~3x baseline time (1 predict + 1 verify + possible retries)
# ~2 hours total
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CKPT_PATH="$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_balanced_full_sft/checkpoint-272"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
OUTPUT_BASE="$PROJECT_ROOT/v15_gui_360/outputs/preliminary_tests"
MODEL_NAME="balanced_sft_step272"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=========================================="
echo "Self-Verification & Retry Experiment"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "$CKPT_PATH" \
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

# ── Self-verification with max_retries=2 ──
echo ""
echo "=========================================="
echo "Self-Verify (max_retries=2)"
echo "Time: $(date)"
echo "=========================================="

VERIFY_DIR="$OUTPUT_BASE/self_verify_retry2"
mkdir -p "$VERIFY_DIR"

python "$PROJECT_ROOT/v15_gui_360/eval_self_verify.py" \
    --test_data "$TEST_DATA" \
    --api_url "http://localhost:8000/v1" \
    --model_name "$MODEL_NAME" \
    --output_dir "$VERIFY_DIR" \
    --threads 64 \
    --gt_history \
    --max_retries 2 \
    --match_threshold 0.5

echo "Self-verify done: $(date)"

# Shutdown
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
