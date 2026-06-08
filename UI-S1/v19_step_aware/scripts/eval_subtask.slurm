#!/bin/bash
#SBATCH --job-name=eval_subtask
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/eval_subtask_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/eval_subtask_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Subtask (Oracle) Evaluation on Balanced Test Set
# Model: SFT checkpoint-272
# Uses GT thought as oracle step instruction
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
MODEL="$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_balanced_full_sft/checkpoint-272"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced_with_thought.jsonl"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p "$PROJECT_ROOT/v19_step_aware/scripts/logs"

echo "=========================================="
echo "Subtask (Oracle) Evaluation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start: $(date)"
echo "Model: $MODEL"
echo "Test data: $TEST_DATA"
echo "=========================================="

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
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
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# Exp 1: Subtask with pred history
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/eval_subtask_pred"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] subtask pred — already done"
else
    echo "=== Subtask (pred history) ==="
    mkdir -p "$OUTPUT"
    python "$PROJECT_ROOT/v19_step_aware/eval_step_aware.py" \
        --prompt_variant subtask \
        --test_data "$TEST_DATA" \
        --api_url http://localhost:8000/v1 \
        --model_name base_sft \
        --output_dir "$OUTPUT" \
        --threads 64
fi

# Exp 2: Subtask with GT history
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/eval_subtask_gt"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] subtask GT — already done"
else
    echo "=== Subtask (GT history) ==="
    mkdir -p "$OUTPUT"
    python "$PROJECT_ROOT/v19_step_aware/eval_step_aware.py" \
        --prompt_variant subtask \
        --test_data "$TEST_DATA" \
        --api_url http://localhost:8000/v1 \
        --model_name base_sft \
        --output_dir "$OUTPUT" \
        --gt_history \
        --threads 64
fi

# Cleanup
kill $VLLM_PID 2>/dev/null || true

echo ""
echo "=========================================="
echo "RESULTS"
echo "=========================================="
echo "Baselines: standard pred=21.9%, CoT pred=23.1%, type_focused pred=23.6%"
echo ""

for exp in eval_subtask_pred eval_subtask_gt; do
    S=$(ls "$PROJECT_ROOT/v19_step_aware/outputs/${exp}"/eval_summary_*.json 2>/dev/null | tail -1)
    if [ -n "$S" ]; then
        echo "--- ${exp} ---"
        python3 -c "
import json
with open('$S') as f:
    d = json.load(f)
print(f'  TSR={d[\"tsr\"]*100:.1f}%  StepSR={d[\"step_sr\"]*100:.1f}%  Progress={d[\"avg_progress\"]*100:.1f}%')
tlm = d.get('task_length_metrics', {})
for b in ['1','2-3','4-5','6+']:
    if b in tlm:
        m = tlm[b]
        print(f'  {b:>5s} steps: TSR={m[\"tsr\"]*100:5.1f}%  StepSR={m[\"step_sr\"]*100:5.1f}%  (n={m[\"num_episodes\"]})')
"
        echo ""
    fi
done

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
