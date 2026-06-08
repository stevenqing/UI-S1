#!/bin/bash
#SBATCH --job-name=ev_v21_thought
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v21_thought_sft/scripts/logs/eval_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v21_thought_sft/scripts/logs/eval_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V21: Thought-Augmented SFT Evaluation
#
# 4 experiments:
#   1. training actions + pred history  (format-matched, primary result)
#   2. training actions + GT history
#   3. eval actions + pred history      (comparable to prior baselines)
#   4. eval actions + GT history
#
# Auto-detects latest checkpoint from training output.
#
# Baselines:
#   Standard SFT-272 (pred): TSR=21.9%
#   CoT (pred):              TSR=23.1%
#   Type Focused (pred):     TSR=23.6%
#   Oracle subtask (pred):   TSR=35.2%
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CKPT_DIR="$PROJECT_ROOT/v21_thought_sft/output/thought_sft"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p "$PROJECT_ROOT/v21_thought_sft/scripts/logs"
mkdir -p "$PROJECT_ROOT/v21_thought_sft/outputs"

# Auto-detect latest checkpoint
LATEST_CKPT=$(ls -d "$CKPT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -z "$LATEST_CKPT" ]; then
    echo "ERROR: No checkpoints found in $CKPT_DIR"
    exit 1
fi
CKPT_NAME=$(basename "$LATEST_CKPT")

echo "=========================================="
echo "V21: Thought-Augmented SFT Eval"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Start:      $(date)"
echo "Checkpoint: $LATEST_CKPT ($CKPT_NAME)"
echo "=========================================="

# ── Start vLLM server ──
python -m vllm.entrypoints.openai.api_server \
    --model "$LATEST_CKPT" \
    --served-model-name "thought_sft" \
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

# ── Run 4 experiments ──
for actions_mode in training eval; do
    for gt_flag in "" "--gt_history"; do
        if [ -z "$gt_flag" ]; then
            hist_mode="pred"
        else
            hist_mode="gt"
        fi

        OUTPUT_DIR="$PROJECT_ROOT/v21_thought_sft/outputs/eval_${CKPT_NAME}_${actions_mode}_${hist_mode}"

        # Skip if already done
        if ls "$OUTPUT_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
            echo "[SKIP] ${actions_mode} actions, ${hist_mode} history — already done"
            continue
        fi

        echo ""
        echo "=== ${actions_mode} actions, ${hist_mode} history ==="
        echo "Start: $(date)"
        mkdir -p "$OUTPUT_DIR"

        python "$PROJECT_ROOT/v21_thought_sft/eval_thought_sft.py" \
            --test_data "$TEST_DATA" \
            --api_url "http://localhost:8000/v1" \
            --model_name "thought_sft" \
            --output_dir "$OUTPUT_DIR" \
            --actions_mode "$actions_mode" \
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
echo "ALL RESULTS SUMMARY (${CKPT_NAME})"
echo "=========================================="
for actions_mode in training eval; do
    for hist_mode in pred gt; do
        OUTPUT_DIR="$PROJECT_ROOT/v21_thought_sft/outputs/eval_${CKPT_NAME}_${actions_mode}_${hist_mode}"
        S=$(ls "$OUTPUT_DIR"/eval_summary_*.json 2>/dev/null | tail -1)
        if [ -n "$S" ]; then
            echo ""
            echo "--- ${actions_mode} actions, ${hist_mode} history ---"
            python -c "
import json
with open('$S') as f:
    d = json.load(f)
print(f\"  TSR:      {d['tsr']*100:.1f}%\")
print(f\"  Step SR:  {d['step_sr']*100:.1f}%\")
print(f\"  Progress: {d['avg_progress']*100:.1f}%\")
rs = d.get('reasoning_stats', {})
print(f\"  Reasoning: {rs.get('steps_with_reasoning',0)}/{rs.get('total_steps',0)} ({rs.get('reasoning_rate',0)*100:.1f}%)\")
tl = d.get('task_length_metrics', {})
for k in ['1','2-3','4-5','6+']:
    if k in tl:
        m = tl[k]
        print(f\"  {k:>5s}: TSR={m['tsr']*100:5.1f}% StepSR={m['step_sr']*100:5.1f}% (n={m['num_episodes']})\")
"
        fi
    done
done

echo ""
echo "=========================================="
echo "Baselines: standard=21.9%, CoT=23.1%, type_focused=23.6%, oracle=35.2%"
echo "Done: $(date)"
echo "=========================================="
