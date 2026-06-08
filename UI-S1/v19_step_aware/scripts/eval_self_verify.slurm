#!/bin/bash
#SBATCH --job-name=ev_verify
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/eval_verify_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/eval_verify_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Self-Verification Diagnostic Experiment
#
# Can the model detect its own errors by
# comparing predicted action with the observed
# screen change?
#
# If TPR > 50%: self-correction is feasible
# and could dramatically improve long-horizon TSR
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
echo "Self-Verification Diagnostic"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

# ── Start vLLM server (allow 3 images per prompt for verification) ──
python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_SFT" \
    --served-model-name "base_sft" \
    --port 8000 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image": 3}' \
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

COMMON_ARGS="--test_data $TEST_DATA --api_url http://localhost:8000/v1 --model_name base_sft"

# ── Exp 1: Self-verify with pred history (main experiment) ──
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/self_verify_pred"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] self_verify_pred — already done"
else
    echo ""
    echo "=== Exp 1: Self-verify, pred history ==="
    echo "Start: $(date)"
    mkdir -p "$OUTPUT"
    python "$PROJECT_ROOT/v19_step_aware/eval_self_verify.py" \
        $COMMON_ARGS --output_dir "$OUTPUT" --threads 64
    echo "Done: $(date)"
fi

# ── Exp 2: Self-verify with GT history (upper bound) ──
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/self_verify_gt"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] self_verify_gt — already done"
else
    echo ""
    echo "=== Exp 2: Self-verify, GT history ==="
    echo "Start: $(date)"
    mkdir -p "$OUTPUT"
    python "$PROJECT_ROOT/v19_step_aware/eval_self_verify.py" \
        $COMMON_ARGS --output_dir "$OUTPUT" --threads 64 --gt_history
    echo "Done: $(date)"
fi

# ── Shutdown vLLM ──
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

# ── Print all summaries ──
echo ""
echo "=========================================="
echo "SELF-VERIFICATION RESULTS"
echo "=========================================="
for exp in self_verify_pred self_verify_gt; do
    OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/${exp}"
    S=$(ls "$OUTPUT"/eval_summary_*.json 2>/dev/null | tail -1)
    if [ -n "$S" ]; then
        echo "--- ${exp} ---"
        python3 -c "
import json
with open('$S') as f:
    d = json.load(f)
vm = d['verification_metrics']
sim = d['simulated_self_correction']
print(f\"  TSR={d['tsr']*100:.1f}%  StepSR={d['step_sr']*100:.1f}%\")
print(f\"  Detection Rate (TPR): {vm['detection_rate_TPR']*100:.1f}%\")
print(f\"  False Alarm (FPR):    {vm['false_alarm_FPR']*100:.1f}%\")
print(f\"  Precision:            {vm['precision']*100:.1f}%\")
print(f\"  Overall Accuracy:     {vm['accuracy']*100:.1f}%\")
print(f\"  Confusion: TP={vm['tp']} FN={vm['fn']} FP={vm['fp']} TN={vm['tn']}\")
print(f\"  Simulated: base_p={sim['base_step_accuracy']*100:.1f}% → 1retry_p={sim['effective_step_accuracy_1retry']*100:.1f}%\")
print(f\"  Simulated 6-step TSR: {sim['simulated_TSR_6step_base']*100:.1f}% → {sim['simulated_TSR_6step_1retry']*100:.1f}%\")
vt = d.get('verify_by_action_type', {})
for t in ['click','type','swipe']:
    if t in vt:
        m = vt[t]
        print(f\"  {t:>8s}: TPR={m['TPR']*100:5.1f}% FPR={m['FPR']*100:5.1f}% (wrong={m['wrong_actions']}, correct={m['correct_actions']})\")
"
        echo ""
    fi
done

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
