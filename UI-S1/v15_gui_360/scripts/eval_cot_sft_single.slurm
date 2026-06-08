#!/bin/bash
#SBATCH --job-name=ev_cot_s
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/eval_cot_sft_single_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/eval_cot_sft_single_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Eval: CoT SFT model in SINGLE-MODEL CoT mode
#
# Previous results:
#   Base SFT + standard prompt:  TSR=21.9%, Step SR=68.8%
#   Base SFT + CoT prompt:       TSR=23.1%, Step SR=46.9%
#   CoT SFT + standard prompt:   TSR=11.8%, Step SR=40.8%
#   CoT SFT + CoT prompt:        ???  ← THIS EXPERIMENT
#
# Hypothesis: CoT SFT model trained on reasoning data should
# generate BETTER reasoning than Base SFT, improving TSR further.
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
COT_SFT="$PROJECT_ROOT/train_GUI_360/llamafactory/output/cot_sft_full"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p "$PROJECT_ROOT/v15_gui_360/scripts/logs"

echo "=========================================="
echo "Eval: CoT SFT model (single-model CoT)"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

# Start vLLM server (CoT SFT model, 4 GPU TP)
python -m vllm.entrypoints.openai.api_server \
    --model "$COT_SFT" \
    --served-model-name "cot_sft" \
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

# Experiment 1: CoT SFT + CoT prompt (predicted history, fair comparison)
OUTPUT_DIR_1="$PROJECT_ROOT/v15_gui_360/outputs/eval_cot_sft_single_no_gt_history"
if ls "$OUTPUT_DIR_1"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[Exp 1] Already done."
else
    echo ""
    echo "=== CoT SFT + CoT prompt (pred history) ==="
    mkdir -p "$OUTPUT_DIR_1"
    python "$PROJECT_ROOT/v15_gui_360/eval_cot_step.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "cot_sft" \
        --output_dir "$OUTPUT_DIR_1" \
        --threads 128
fi

# Experiment 2: CoT SFT + CoT prompt (GT history)
OUTPUT_DIR_2="$PROJECT_ROOT/v15_gui_360/outputs/eval_cot_sft_single_gt_history"
if ls "$OUTPUT_DIR_2"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[Exp 2] Already done."
else
    echo ""
    echo "=== CoT SFT + CoT prompt (GT history) ==="
    mkdir -p "$OUTPUT_DIR_2"
    python "$PROJECT_ROOT/v15_gui_360/eval_cot_step.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "cot_sft" \
        --output_dir "$OUTPUT_DIR_2" \
        --threads 128 \
        --gt_history
fi

# Experiment 3: CoT SFT + standard prompt (reproduce 11.8% to confirm)
OUTPUT_DIR_3="$PROJECT_ROOT/v15_gui_360/outputs/eval_cot_sft_standard_prompt"
if ls "$OUTPUT_DIR_3"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[Exp 3] Already done."
else
    echo ""
    echo "=== CoT SFT + standard prompt (reproduce baseline) ==="
    mkdir -p "$OUTPUT_DIR_3"
    python "$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "cot_sft" \
        --output_dir "$OUTPUT_DIR_3" \
        --threads 128
fi

# Shutdown
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

# Print summaries
echo ""
echo "=========================================="
echo "Results Summary"
echo "=========================================="
for d in "$OUTPUT_DIR_1" "$OUTPUT_DIR_2" "$OUTPUT_DIR_3"; do
    S=$(ls "$d"/eval_summary_*.json 2>/dev/null | tail -1)
    if [ -n "$S" ]; then
        echo "$(basename $(dirname $S)):"
        cat "$S"
        echo ""
    fi
done
echo "Done: $(date)"
