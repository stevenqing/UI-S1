#!/bin/bash
#SBATCH --job-name=subtask_ablation
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/subtask_ablation_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/subtask_ablation_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Subtask Ablation: Isolate oracle thought vs compressed history
#
# Exp A: standard_compressed = standard prompt + compressed history (NO oracle)
#   → isolates the effect of compressed history format
#
# Exp B: subtask_verbose = subtask prompt (oracle thought) + verbose history
#   → isolates the effect of oracle thought
#
# Reference:
#   standard (verbose):     21.9% TSR  (no oracle, verbose hist)
#   subtask (compressed):   35.2% TSR  (oracle + compressed hist)
#
# Expected attribution:
#   If A >> 21.9%: compressed history is the main driver
#   If B >> 21.9%: oracle thought is the main driver
#   If both help: both contribute
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
echo "Subtask Ablation Experiments"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start: $(date)"
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

# Exp A: standard_compressed (standard prompt + compressed history, NO oracle)
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/eval_standard_compressed_pred"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] standard_compressed — already done"
else
    echo "=== Exp A: standard_compressed (pred history) ==="
    mkdir -p "$OUTPUT"
    python "$PROJECT_ROOT/v19_step_aware/eval_step_aware.py" \
        --prompt_variant standard_compressed \
        --test_data "$TEST_DATA" \
        --api_url http://localhost:8000/v1 \
        --model_name base_sft \
        --output_dir "$OUTPUT" \
        --threads 64
fi

# Exp B: subtask_verbose (subtask/oracle prompt + verbose history)
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/eval_subtask_verbose_pred"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] subtask_verbose — already done"
else
    echo "=== Exp B: subtask_verbose (pred history) ==="
    mkdir -p "$OUTPUT"
    python "$PROJECT_ROOT/v19_step_aware/eval_step_aware.py" \
        --prompt_variant subtask_verbose \
        --test_data "$TEST_DATA" \
        --api_url http://localhost:8000/v1 \
        --model_name base_sft \
        --output_dir "$OUTPUT" \
        --threads 64
fi

# Cleanup
kill $VLLM_PID 2>/dev/null || true

# Results
echo ""
echo "=========================================="
echo "ABLATION RESULTS"
echo "=========================================="
echo ""
echo "Factor isolation:"
echo "  standard (verbose, no oracle):     21.9% TSR  [baseline]"
echo "  subtask  (compressed + oracle):    35.2% TSR  [full]"
echo ""

for exp in eval_standard_compressed_pred eval_subtask_verbose_pred; do
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

echo "Attribution:"
echo "  Compressed history contribution = standard_compressed - 21.9%"
echo "  Oracle thought contribution     = subtask_verbose - 21.9%"
echo "  Interaction effect              = 35.2% - standard_compressed - subtask_verbose + 21.9%"
echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
