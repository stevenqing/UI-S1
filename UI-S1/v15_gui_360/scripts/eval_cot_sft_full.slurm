#!/bin/bash
#SBATCH --job-name=ev_cot_sft
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/eval_cot_sft_full_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/eval_cot_sft_full_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Eval CoT SFT (full fine-tuning, epoch ~1)
#
# Checkpoint: checkpoint-550-epoch1-backup
# Test: gui360_test_1000_balanced.jsonl (1000 episodes)
# Script: eval_gui360_template.py (same as baseline SFT eval)
#
# The CoT model outputs reasoning before <tool_call>,
# but eval_gui360_template.py parses <tool_call> regardless
# of preceding text, so it works out of the box.
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CKPT_PATH="$PROJECT_ROOT/train_GUI_360/llamafactory/output/cot_sft_full/checkpoint-550-epoch1-backup"
MODEL_NAME="cot_sft_full_epoch1"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/v15_gui_360/outputs/eval_cot_sft_full_epoch1"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_ROOT/v15_gui_360/scripts/logs"

echo "=========================================="
echo "Eval: CoT SFT Full Fine-tuning (epoch ~1)"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "Checkpoint: $CKPT_PATH"
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

# Wait for server
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

# Run eval
python "$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py" \
    --test_data "$TEST_DATA" \
    --api_url "http://localhost:8000/v1" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --threads 128

# Shutdown server
echo "Shutting down vLLM..."
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

# Print summary
SUMMARY=$(ls "$OUTPUT_DIR"/eval_summary_*.json 2>/dev/null | tail -1)
if [ -n "$SUMMARY" ]; then
    echo ""
    echo "=========================================="
    echo "Result:"
    cat "$SUMMARY"
    echo ""
    echo "=========================================="
fi

echo "Done: $(date)"
