#!/bin/bash
#SBATCH --job-name=hist_ablat
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/history_ablation_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/history_ablation_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# History Ablation Experiment
#
# Question: Does history length cause degradation on long trajectories?
#
# Model: gui360_balanced_full_sft checkpoint-272 (TSR=21.9%)
# All conditions use GT history (teacher-forced) to isolate
# history effects from error propagation.
#
# 3 conditions:
#   (a) full   - full GT history (baseline, already have results)
#   (b) none   - no history at all
#   (c) last_3 - only last 3 history entries
#
# Expected outcomes:
#   - If Step SR drops with more history → history explosion problem
#   - If Step SR similar with no history → model ignores history
#   - If last_3 > full → truncated history is optimal
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CKPT_PATH="$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_balanced_full_sft/checkpoint-272"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
OUTPUT_BASE="$PROJECT_ROOT/v15_gui_360/outputs/history_ablation"
MODEL_NAME="balanced_sft_step272"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=========================================="
echo "History Ablation Experiment"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Model:   $CKPT_PATH"
echo "Start:   $(date)"
echo "=========================================="

# Start vLLM server (shared across all conditions)
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

# Run all 3 conditions with GT history (teacher-forced)
for HIST_MODE in full none last_3; do
    OUTPUT_DIR="$OUTPUT_BASE/gt_history_${HIST_MODE}"

    if ls "$OUTPUT_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
        echo ""
        echo "[${HIST_MODE}] Already evaluated, skipping."
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Condition: history_mode=${HIST_MODE}"
    echo "Time: $(date)"
    echo "=========================================="

    mkdir -p "$OUTPUT_DIR"

    python "$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --threads 128 \
        --gt_history \
        --history_mode "$HIST_MODE" \
        --match_threshold 0.5

    echo "[${HIST_MODE}] Done."
done

# Shutdown server
echo ""
echo "Shutting down vLLM..."
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

# Print comparison
echo ""
echo "=========================================="
echo "History Ablation Results"
echo "=========================================="
python3 -c "
import json, glob

print(f'{\"Condition\":<15} {\"TSR\":<8} {\"StepSR\":<8} {\"Progress\":<10} {\"MeanRwd\":<10}')
print('-' * 55)

for mode in ['full', 'none', 'last_3']:
    output_dir = '$OUTPUT_BASE/gt_history_' + mode
    summary_files = sorted(glob.glob(f'{output_dir}/eval_summary_*.json'))
    if summary_files:
        d = json.load(open(summary_files[-1]))
        print(f'{mode:<15} {d[\"tsr\"]:<8.3f} {d[\"step_sr\"]:<8.3f} {d[\"avg_progress\"]:<10.3f} {d[\"mean_reward\"]:<10.4f}')
    else:
        print(f'{mode:<15} {\"N/A\":<8} {\"N/A\":<8} {\"N/A\":<10} {\"N/A\":<10}')
"

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
