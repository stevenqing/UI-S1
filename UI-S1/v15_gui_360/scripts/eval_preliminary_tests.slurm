#!/bin/bash
#SBATCH --job-name=prelim_test
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/preliminary_tests_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/preliminary_tests_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Preliminary Tests for Long-Horizon Reasoning
#
# Model: gui360_balanced_full_sft checkpoint-272
#
# Exp A: Planning Prompt (GT history + plan)
#   - Generate task plan at step 0, include in all steps
#   - Tests if explicit planning improves step accuracy
#
# Exp B: Autoregressive Error Recovery
#   - Autoregressive eval WITHOUT stop_on_error
#   - Tests if model can recover after errors
#
# (Exp C is offline analysis, already done)
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
echo "Preliminary Tests for Long-Horizon Reasoning"
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

# ============================================
# Exp A: Planning Prompt (GT history + plan)
# ============================================
EXP_A_DIR="$OUTPUT_BASE/exp_a_planning"
if ls "$EXP_A_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[Exp A] Already evaluated, skipping."
else
    echo ""
    echo "=========================================="
    echo "Exp A: Planning Prompt (GT history + plan)"
    echo "Time: $(date)"
    echo "=========================================="
    mkdir -p "$EXP_A_DIR"

    python "$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$EXP_A_DIR" \
        --threads 128 \
        --gt_history \
        --use_plan \
        --match_threshold 0.5

    echo "[Exp A] Done."
fi

# ============================================
# Exp B: Autoregressive WITHOUT stop_on_error
# ============================================
EXP_B_DIR="$OUTPUT_BASE/exp_b_autoreg_no_stop"
if ls "$EXP_B_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[Exp B] Already evaluated, skipping."
else
    echo ""
    echo "=========================================="
    echo "Exp B: Autoregressive without stop_on_error"
    echo "Time: $(date)"
    echo "=========================================="
    mkdir -p "$EXP_B_DIR"

    python "$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$EXP_B_DIR" \
        --threads 128 \
        --match_threshold 0.5

    echo "[Exp B] Done."
fi

# Shutdown server
echo ""
echo "Shutting down vLLM..."
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

# Print comparison
echo ""
echo "=========================================="
echo "Preliminary Test Results"
echo "=========================================="
python3 -c "
import json, glob

# Baseline from history ablation
baseline_dir = '$PROJECT_ROOT/v15_gui_360/outputs/history_ablation/gt_history_full'
baseline_files = sorted(glob.glob(f'{baseline_dir}/eval_summary_*.json'))

print(f'{\"Experiment\":<35} {\"TSR\":<8} {\"StepSR\":<8} {\"Progress\":<10} {\"MeanRwd\":<10}')
print('-' * 75)

if baseline_files:
    d = json.load(open(baseline_files[-1]))
    print(f'{\"Baseline (GT hist, no plan)\":<35} {d[\"tsr\"]:<8.3f} {d[\"step_sr\"]:<8.3f} {d[\"avg_progress\"]:<10.3f} {d[\"mean_reward\"]:<10.4f}')

for label, edir in [
    ('Exp A: Planning Prompt (GT hist)', '$OUTPUT_BASE/exp_a_planning'),
    ('Exp B: Autoreg no stop_on_error', '$OUTPUT_BASE/exp_b_autoreg_no_stop'),
]:
    summary_files = sorted(glob.glob(f'{edir}/eval_summary_*.json'))
    if summary_files:
        d = json.load(open(summary_files[-1]))
        print(f'{label:<35} {d[\"tsr\"]:<8.3f} {d[\"step_sr\"]:<8.3f} {d[\"avg_progress\"]:<10.3f} {d[\"mean_reward\"]:<10.4f}')
    else:
        print(f'{label:<35} {\"N/A\":<8}')
"

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
