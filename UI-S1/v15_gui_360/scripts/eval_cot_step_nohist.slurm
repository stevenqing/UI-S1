#!/bin/bash
#SBATCH --job-name=ev_cot_nh
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/eval_cot_step_nohist_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/eval_cot_step_nohist_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Eval: cot_step with predicted history (no gt_history)
#
# Base SFT model + CoT prompt (single model, single pass)
# Fair comparison with baseline (both use predicted history)
#
# Previous result with gt_history: TSR=23.6%, Step SR=59.2%
# Baseline (no CoT, pred history): TSR=21.9%, Step SR=68.8%
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_SFT="$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_balanced_full_sft/checkpoint-272"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p "$PROJECT_ROOT/v15_gui_360/scripts/logs"

echo "=========================================="
echo "Eval: cot_step (no gt_history)"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

# Start vLLM server (base SFT, 4 GPU TP)
python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_SFT" \
    --served-model-name "base_sft" \
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
    echo "ERROR: vLLM failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Experiment 1: cot_step without gt_history (fair comparison)
OUTPUT_DIR_1="$PROJECT_ROOT/v15_gui_360/outputs/eval_cot_step_no_gt_history"
if ls "$OUTPUT_DIR_1"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[Exp 1] Already done."
else
    echo ""
    echo "=== cot_step (pred history) ==="
    mkdir -p "$OUTPUT_DIR_1"
    python "$PROJECT_ROOT/v15_gui_360/eval_cot_step.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "base_sft" \
        --output_dir "$OUTPUT_DIR_1" \
        --threads 128
fi

# Experiment 2: cot_step with gt_history (reproduce previous result)
OUTPUT_DIR_2="$PROJECT_ROOT/v15_gui_360/outputs/eval_cot_step_gt_history"
if ls "$OUTPUT_DIR_2"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[Exp 2] Already done."
else
    echo ""
    echo "=== cot_step (gt history) ==="
    mkdir -p "$OUTPUT_DIR_2"
    python "$PROJECT_ROOT/v15_gui_360/eval_cot_step.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "base_sft" \
        --output_dir "$OUTPUT_DIR_2" \
        --threads 128 \
        --gt_history
fi

# Shutdown
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

# Print summaries
echo ""
echo "=========================================="
for d in "$OUTPUT_DIR_1" "$OUTPUT_DIR_2"; do
    S=$(ls "$d"/eval_summary_*.json 2>/dev/null | tail -1)
    if [ -n "$S" ]; then
        echo "$(basename $(dirname $S)):"
        cat "$S"
        echo ""
    fi
done
echo "Done: $(date)"
