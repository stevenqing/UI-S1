#!/bin/bash
#SBATCH --job-name=ev_oracle
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/eval_oracle_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/eval_oracle_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Oracle Action-Type Experiment
#
# Hypothesis: Action type transitions are the
# bottleneck for long-horizon tasks.
#
# Experiments:
#   1. oracle_type (pred hist) — GT type hint
#   2. oracle_type (GT hist)  — upper bound
#   3. oracle_type_noisy (pred hist) — 80% correct
#   4. no_oracle (pred hist) — baseline
#
# Baselines:
#   standard:      TSR=21.9%
#   type_focused:  TSR=23.6%
#   6+ step TSR:   ~6% (all methods)
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
echo "Oracle Action-Type Experiment"
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

COMMON_ARGS="--test_data $TEST_DATA --api_url http://localhost:8000/v1 --model_name base_sft --threads 128"

# ── Exp 1: Oracle type, pred history (MAIN EXPERIMENT) ──
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/oracle_type_pred"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] oracle_type_pred — already done"
else
    echo ""
    echo "=== Exp 1: oracle_type, pred history ==="
    echo "Start: $(date)"
    mkdir -p "$OUTPUT"
    python "$PROJECT_ROOT/v19_step_aware/eval_oracle_type.py" \
        --mode oracle_type \
        $COMMON_ARGS --output_dir "$OUTPUT"
    echo "Done: $(date)"
fi

# ── Exp 2: Oracle type, GT history (UPPER BOUND) ──
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/oracle_type_gt"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] oracle_type_gt — already done"
else
    echo ""
    echo "=== Exp 2: oracle_type, GT history ==="
    echo "Start: $(date)"
    mkdir -p "$OUTPUT"
    python "$PROJECT_ROOT/v19_step_aware/eval_oracle_type.py" \
        --mode oracle_type --gt_history \
        $COMMON_ARGS --output_dir "$OUTPUT"
    echo "Done: $(date)"
fi

# ── Exp 3: Noisy oracle (80%), pred history ──
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/oracle_noisy_pred"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] oracle_noisy_pred — already done"
else
    echo ""
    echo "=== Exp 3: noisy oracle (80%), pred history ==="
    echo "Start: $(date)"
    mkdir -p "$OUTPUT"
    python "$PROJECT_ROOT/v19_step_aware/eval_oracle_type.py" \
        --mode oracle_type_noisy \
        $COMMON_ARGS --output_dir "$OUTPUT"
    echo "Done: $(date)"
fi

# ── Exp 4: No oracle baseline, pred history ──
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/oracle_none_pred"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] no_oracle_pred — already done"
else
    echo ""
    echo "=== Exp 4: no oracle, pred history ==="
    echo "Start: $(date)"
    mkdir -p "$OUTPUT"
    python "$PROJECT_ROOT/v19_step_aware/eval_oracle_type.py" \
        --mode no_oracle \
        $COMMON_ARGS --output_dir "$OUTPUT"
    echo "Done: $(date)"
fi

# ── Shutdown vLLM ──
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

# ── Print all summaries ──
echo ""
echo "=========================================="
echo "ALL ORACLE EXPERIMENT RESULTS"
echo "=========================================="
echo "Baselines: standard=21.9% | type_focused=23.6% | 6+ step TSR ≈ 6%"
echo ""
for exp in oracle_type_pred oracle_type_gt oracle_noisy_pred oracle_none_pred; do
    OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/${exp}"
    S=$(ls "$OUTPUT"/eval_summary_*.json 2>/dev/null | tail -1)
    if [ -n "$S" ]; then
        echo "--- ${exp} ---"
        python3 -c "
import json
with open('$S') as f:
    d = json.load(f)
print(f\"  TSR={d['tsr']*100:.1f}%  StepSR={d['step_sr']*100:.1f}%  Progress={d['avg_progress']*100:.1f}%\")
tl = d.get('task_length_metrics', {})
for k in ['1','2-3','4-5','6+']:
    if k in tl:
        m = tl[k]
        print(f\"  {k:>5s}: TSR={m['tsr']*100:5.1f}%  StepSR={m['step_sr']*100:5.1f}% (n={m['num_episodes']})\")
tm = d.get('transition_metrics', {})
if tm:
    print('  Transitions:')
    for t, m in sorted(tm.items(), key=lambda x: -x[1]['total'])[:6]:
        print(f\"    {t:>15s}: acc={m['accuracy']*100:5.1f}% (n={m['total']})\")
he = d.get('hint_effectiveness', {})
if he:
    print(f\"  Hint: correct_hint_acc={he['correct_hint_accuracy']*100:.1f}% wrong_hint_acc={he['wrong_hint_accuracy']*100:.1f}%\")
"
        echo ""
    fi
done

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
