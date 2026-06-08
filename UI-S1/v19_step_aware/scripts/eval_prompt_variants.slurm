#!/bin/bash
#SBATCH --job-name=ev_v19_prompts
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/eval_prompt_variants_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/eval_prompt_variants_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V19: Step-Aware Prompt Variant Evaluation
#
# 8 experiments: 4 variants x 2 history modes (pred/GT)
#
# Variants:
#   focused_cot  - History-anchored reasoning, anti-jump instruction
#   adaptive_cot - CoT for steps 0-1, standard for steps 2+
#   step_context - Standard + step-position metadata
#   type_focused - Forces explicit action-type reasoning
#
# Base SFT checkpoint-272, 4 GPU TP, 128 threads
# Expected: ~30 min per experiment, ~4 hours total
#
# Baselines (already measured):
#   standard (pred hist): TSR=21.9%
#   cot      (pred hist): TSR=23.1%
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

mkdir -p "$PROJECT_ROOT/v19_step_aware/scripts/logs"

echo "=========================================="
echo "V19: Step-Aware Prompt Variants"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

# ── Start vLLM server ──
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

# ── Run 8 experiments: 4 variants x 2 history modes ──
for variant in focused_cot adaptive_cot step_context type_focused; do
    for gt_flag in "" "--gt_history"; do
        if [ -z "$gt_flag" ]; then
            hist_mode="pred"
        else
            hist_mode="gt"
        fi

        OUTPUT_DIR="$PROJECT_ROOT/v19_step_aware/outputs/eval_${variant}_${hist_mode}"

        # Skip if already completed
        if ls "$OUTPUT_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
            echo "[SKIP] ${variant} (${hist_mode}) — already done"
            continue
        fi

        echo ""
        echo "=== ${variant} (${hist_mode} history) ==="
        echo "Start: $(date)"
        mkdir -p "$OUTPUT_DIR"

        python "$PROJECT_ROOT/v19_step_aware/eval_step_aware.py" \
            --prompt_variant "$variant" \
            --test_data "$TEST_DATA" \
            --api_url "http://localhost:8000/v1" \
            --model_name "base_sft" \
            --output_dir "$OUTPUT_DIR" \
            --threads 128 $gt_flag

        echo "Done: $(date)"
    done
done

# ── Shutdown vLLM ──
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

# ── Print all summaries ──
echo ""
echo "=========================================="
echo "ALL RESULTS SUMMARY"
echo "=========================================="
for variant in focused_cot adaptive_cot step_context type_focused; do
    for hist_mode in pred gt; do
        OUTPUT_DIR="$PROJECT_ROOT/v19_step_aware/outputs/eval_${variant}_${hist_mode}"
        S=$(ls "$OUTPUT_DIR"/eval_summary_*.json 2>/dev/null | tail -1)
        if [ -n "$S" ]; then
            echo ""
            echo "--- ${variant} (${hist_mode}) ---"
            python -c "
import json, sys
with open('$S') as f:
    d = json.load(f)
print(f\"  TSR:      {d['tsr']*100:.1f}%\")
print(f\"  Step SR:  {d['step_sr']*100:.1f}%\")
print(f\"  Progress: {d['avg_progress']*100:.1f}%\")
tl = d.get('task_length_metrics', {})
for k in sorted(tl.keys()):
    m = tl[k]
    print(f\"  {k:>5s}: TSR={m['tsr']*100:5.1f}% (n={m['num_episodes']})\")
"
        fi
    done
done

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
