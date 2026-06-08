#!/bin/bash
#SBATCH --job-name=best_of_n
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/best_of_n_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/best_of_n_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Best-of-N Evaluation with Verifier Selection
#
# Dual model: base_sft (TP=2, port 8000) + verifier (TP=2, port 8001)
# Experiments:
#   1. N=3, pred history (main experiment)
#   2. N=3, GT history (upper bound)
#   3. N=5, pred history (more candidates)
#   4. N=1 baseline (greedy only, for comparison)
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_SFT="$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_balanced_full_sft/checkpoint-272"
VERIFIER="$PROJECT_ROOT/v19_step_aware/output/verifier_hard_merged"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p "$PROJECT_ROOT/v19_step_aware/scripts/logs"

# Node-local HF cache
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_${SLURM_JOB_ID}
export HF_HOME=/tmp/hf_home_${SLURM_JOB_ID}
mkdir -p $HF_DATASETS_CACHE $HF_HOME

echo "=========================================="
echo "Best-of-N Evaluation"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

# ── Serve base_sft on port 8000 (TP=2, GPUs 0-1) ──
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_SFT" \
    --served-model-name "base_sft" \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image": 2}' \
    &
PRED_PID=$!

# ── Serve verifier on port 8001 (TP=2, GPUs 2-3) ──
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
    --model "$VERIFIER" \
    --served-model-name "verifier" \
    --port 8001 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image": 3}' \
    &
VERIFY_PID=$!

# Wait for both servers
for PORT in 8000 8001; do
    WAIT_COUNT=0
    while [ $WAIT_COUNT -lt 600 ]; do
        if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
            echo "Port ${PORT} ready after ${WAIT_COUNT}s"
            break
        fi
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
    done
    if [ $WAIT_COUNT -ge 600 ]; then
        echo "ERROR: Server on port ${PORT} failed to start"
        kill $PRED_PID $VERIFY_PID 2>/dev/null || true
        exit 1
    fi
done

# ═══════════════════════════════════════════════
# Exp 1: Best-of-3, pred history (main)
# ═══════════════════════════════════════════════
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/best_of_3_pred"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] best_of_3_pred — already done"
else
    echo ""
    echo "=== Exp 1: Best-of-3, pred history ==="
    echo "Start: $(date)"
    mkdir -p "$OUTPUT"
    python "$PROJECT_ROOT/v19_step_aware/eval_best_of_n.py" \
        --test_data "$TEST_DATA" \
        --predict_url http://localhost:8000/v1 \
        --predict_model base_sft \
        --verify_url http://localhost:8001/v1 \
        --verify_model verifier \
        --output_dir "$OUTPUT" \
        --n_candidates 3 \
        --sample_temperature 0.7 \
        --threads 32
    echo "Exp 1 done: $(date)"
fi

# ═══════════════════════════════════════════════
# Exp 2: Best-of-3, GT history (upper bound)
# ═══════════════════════════════════════════════
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/best_of_3_gt"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] best_of_3_gt — already done"
else
    echo ""
    echo "=== Exp 2: Best-of-3, GT history ==="
    echo "Start: $(date)"
    mkdir -p "$OUTPUT"
    python "$PROJECT_ROOT/v19_step_aware/eval_best_of_n.py" \
        --test_data "$TEST_DATA" \
        --predict_url http://localhost:8000/v1 \
        --predict_model base_sft \
        --verify_url http://localhost:8001/v1 \
        --verify_model verifier \
        --output_dir "$OUTPUT" \
        --n_candidates 3 \
        --sample_temperature 0.7 \
        --threads 32 \
        --gt_history
    echo "Exp 2 done: $(date)"
fi

# ═══════════════════════════════════════════════
# Exp 3: Best-of-5, pred history (more candidates)
# ═══════════════════════════════════════════════
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/best_of_5_pred"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] best_of_5_pred — already done"
else
    echo ""
    echo "=== Exp 3: Best-of-5, pred history ==="
    echo "Start: $(date)"
    mkdir -p "$OUTPUT"
    python "$PROJECT_ROOT/v19_step_aware/eval_best_of_n.py" \
        --test_data "$TEST_DATA" \
        --predict_url http://localhost:8000/v1 \
        --predict_model base_sft \
        --verify_url http://localhost:8001/v1 \
        --verify_model verifier \
        --output_dir "$OUTPUT" \
        --n_candidates 5 \
        --sample_temperature 0.7 \
        --threads 24
    echo "Exp 3 done: $(date)"
fi

# Cleanup
kill $PRED_PID $VERIFY_PID 2>/dev/null || true
wait $PRED_PID $VERIFY_PID 2>/dev/null || true

# ═══════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════
echo ""
echo "=========================================="
echo "BEST-OF-N RESULTS"
echo "=========================================="
echo "Baseline: TSR=23.3% (dual verify, no retry)"
echo ""

for exp in best_of_3_pred best_of_3_gt best_of_5_pred; do
    OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/${exp}"
    S=$(ls "$OUTPUT"/eval_summary_*.json 2>/dev/null | tail -1)
    if [ -n "$S" ]; then
        echo "--- ${exp} ---"
        python3 -c "
import json
with open('$S') as f:
    d = json.load(f)
sm = d['selection_metrics']
sim = d['simulated_tsr']
tlm = d.get('task_length_metrics', {})
print(f\"  TSR={d['tsr']*100:.1f}%  StepSR={d['step_sr']*100:.1f}%  Progress={d['avg_progress']*100:.1f}%\")
print(f\"  Greedy acc:   {sm['greedy_accuracy']*100:.1f}%\")
print(f\"  Oracle acc:   {sm['oracle_accuracy']*100:.1f}% (best of {d['n_candidates']})\")
print(f\"  Selected acc: {sm['selected_accuracy']*100:.1f}% (verifier-picked)\")
print(f\"  Net gain: {sm['net_gain']} steps (helped={sm['retry_helped']}, hurt={sm['retry_hurt']})\")
print(f\"  Sim 6-step TSR: greedy={sim['greedy_6step_TSR']*100:.1f}% → selected={sim['selected_6step_TSR']*100:.1f}% (oracle={sim['oracle_6step_TSR']*100:.1f}%)\")
for b in ['1','2-3','4-5','6+']:
    if b in tlm:
        m = tlm[b]
        print(f\"  {b:>5s} steps: TSR={m['tsr']*100:5.1f}%  StepSR={m['step_sr']*100:5.1f}%  (n={m['num_episodes']}, success={m['num_success']})\")
"
        echo ""
    fi
done

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
