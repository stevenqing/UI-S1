#!/bin/bash
#SBATCH --job-name=ev_aisap
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/eval_aisap_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/eval_aisap_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# AISAP: Adaptive Inference-Time Scaling for
#        Sequential Action Prediction
#
# Ablation experiments (all pred history):
#   E1: N=5, single prompt (type_focused), voting
#   E2: N=5, diverse prompts (TF + std), voting
#   E3: same as E2 (consensus history is default in pred mode)
#   E4: E2 + trajectory repair
#   E5: adaptive N + diverse + repair
#
# Baselines (already measured):
#   single-pass standard:     TSR=21.9%
#   single-pass type_focused: TSR=23.6%
#
# Note: N=5 means 5× inference per step. With 128→32 threads
# to manage vLLM load, each experiment takes ~90-120 min.
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
echo "AISAP: Inference-Time Scaling Experiments"
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

COMMON_ARGS="--test_data $TEST_DATA --api_url http://localhost:8000/v1 --model_name base_sft"

# ── E1: N=5 voting, single prompt (type_focused only) ──
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/aisap_e1_voting_single"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] E1 — already done"
else
    echo ""
    echo "=== E1: N=5 voting, single prompt ==="
    echo "Start: $(date)"
    mkdir -p "$OUTPUT"
    python "$PROJECT_ROOT/v19_step_aware/eval_aisap.py" \
        --mode voting --n_samples 5 \
        $COMMON_ARGS --output_dir "$OUTPUT" --threads 32
    echo "Done: $(date)"
fi

# ── E2: N=5 voting, diverse prompts (TF + std) ──
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/aisap_e2_voting_diverse"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] E2 — already done"
else
    echo ""
    echo "=== E2: N=5 voting, diverse prompts ==="
    echo "Start: $(date)"
    mkdir -p "$OUTPUT"
    python "$PROJECT_ROOT/v19_step_aware/eval_aisap.py" \
        --mode voting --n_samples 5 --diverse_prompts \
        $COMMON_ARGS --output_dir "$OUTPUT" --threads 32
    echo "Done: $(date)"
fi

# ── E4: N=5 diverse + trajectory repair ──
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/aisap_e4_repair_diverse"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] E4 — already done"
else
    echo ""
    echo "=== E4: N=5 diverse + trajectory repair ==="
    echo "Start: $(date)"
    mkdir -p "$OUTPUT"
    python "$PROJECT_ROOT/v19_step_aware/eval_aisap.py" \
        --mode repair --n_samples 5 --diverse_prompts \
        --consensus_threshold 0.4 --repair_n_samples 10 --max_repair_rounds 2 \
        $COMMON_ARGS --output_dir "$OUTPUT" --threads 32
    echo "Done: $(date)"
fi

# ── E5: adaptive N + diverse + repair ──
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/aisap_e5_adaptive"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] E5 — already done"
else
    echo ""
    echo "=== E5: adaptive N + diverse + repair ==="
    echo "Start: $(date)"
    mkdir -p "$OUTPUT"
    python "$PROJECT_ROOT/v19_step_aware/eval_aisap.py" \
        --mode adaptive --diverse_prompts \
        --consensus_threshold 0.4 --repair_n_samples 10 --max_repair_rounds 2 \
        $COMMON_ARGS --output_dir "$OUTPUT" --threads 32
    echo "Done: $(date)"
fi

# ── Shutdown vLLM ──
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

# ── Print all summaries ──
echo ""
echo "=========================================="
echo "ALL AISAP RESULTS"
echo "=========================================="
echo "Baselines: single-pass standard=21.9% | single-pass type_focused=23.6%"
echo ""
for exp in e1_voting_single e2_voting_diverse e4_repair_diverse e5_adaptive; do
    OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/aisap_${exp}"
    S=$(ls "$OUTPUT"/eval_summary_*.json 2>/dev/null | tail -1)
    if [ -n "$S" ]; then
        echo "--- ${exp} ---"
        python3 -c "
import json
with open('$S') as f:
    d = json.load(f)
print(f\"  TSR={d['tsr']*100:.1f}%  StepSR={d['step_sr']*100:.1f}%  Progress={d['avg_progress']*100:.1f}%  AvgN={d.get('avg_samples_per_step',0):.1f}\")
ca = d.get('consensus_analysis', {})
if ca:
    print(f\"  Consensus: correct={ca.get('avg_consensus_correct',0)*100:.1f}%  wrong={ca.get('avg_consensus_wrong',0)*100:.1f}%\")
tl = d.get('task_length_metrics', {})
for k in ['1','2-3','4-5','6+']:
    if k in tl:
        m = tl[k]
        print(f\"  {k:>5s}: TSR={m['tsr']*100:5.1f}% (n={m['num_episodes']})\")
"
        echo ""
    fi
done

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
