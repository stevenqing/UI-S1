#!/bin/bash
#SBATCH --job-name=logprob_rank
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/logprob_rank_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/logprob_rank_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Diagnostic: logprob ranking vs first-YES
#
# Tests whether the verifier's logP(YES)-logP(NO) provides
# a useful ranking signal beyond binary YES/NO.
#
# Experiments:
#   1. N=3, logprob_rank, pred history
#   2. N=5, logprob_rank, pred history
#   3. N=3, logprob_rank, GT history
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

export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_${SLURM_JOB_ID}
export HF_HOME=/tmp/hf_home_${SLURM_JOB_ID}
mkdir -p $HF_DATASETS_CACHE $HF_HOME

echo "=========================================="
echo "Logprob Ranking Diagnostic"
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
# Exp 1: N=3, logprob_rank, pred history
# ═══════════════════════════════════════════════
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/logprob_n3_pred"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] logprob_n3_pred — already done"
else
    echo ""
    echo "=== Exp 1: N=3, logprob_rank, pred history ==="
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
        --selection_strategy logprob_rank \
        --threads 32
    echo "Exp 1 done: $(date)"
fi

# ═══════════════════════════════════════════════
# Exp 2: N=5, logprob_rank, pred history
# ═══════════════════════════════════════════════
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/logprob_n5_pred"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] logprob_n5_pred — already done"
else
    echo ""
    echo "=== Exp 2: N=5, logprob_rank, pred history ==="
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
        --selection_strategy logprob_rank \
        --threads 24
    echo "Exp 2 done: $(date)"
fi

# ═══════════════════════════════════════════════
# Exp 3: N=3, logprob_rank, GT history
# ═══════════════════════════════════════════════
OUTPUT="$PROJECT_ROOT/v19_step_aware/outputs/logprob_n3_gt"
if ls "$OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "[SKIP] logprob_n3_gt — already done"
else
    echo ""
    echo "=== Exp 3: N=3, logprob_rank, GT history ==="
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
        --selection_strategy logprob_rank \
        --threads 32 \
        --gt_history
    echo "Exp 3 done: $(date)"
fi

# Cleanup
kill $PRED_PID $VERIFY_PID 2>/dev/null || true
wait $PRED_PID $VERIFY_PID 2>/dev/null || true

# ═══════════════════════════════════════════════
# Summary: Compare logprob_rank vs first_yes
# ═══════════════════════════════════════════════
echo ""
echo "=========================================="
echo "LOGPROB RANKING vs FIRST-YES COMPARISON"
echo "=========================================="

print_result() {
    local label=$1
    local json_path=$2
    if [ -n "$json_path" ] && [ -f "$json_path" ]; then
        python3 -c "
import json
with open('$json_path') as f:
    d = json.load(f)
sm = d['selection_metrics']
sim = d['simulated_tsr']
tlm = d.get('task_length_metrics', {})
strategy = d.get('selection_strategy', '?')
print(f'$label [{strategy}]')
print(f'  TSR={d[\"tsr\"]*100:.1f}%  StepSR={d[\"step_sr\"]*100:.1f}%  Progress={d[\"avg_progress\"]*100:.1f}%')
print(f'  Selected acc: {sm[\"selected_accuracy\"]*100:.1f}%  Oracle acc: {sm[\"oracle_accuracy\"]*100:.1f}%')
print(f'  Net gain: {sm[\"net_gain\"]} (helped={sm[\"retry_helped\"]}, hurt={sm[\"retry_hurt\"]})')
t6 = tlm.get('6+', {})
if t6:
    print(f'  6+ steps: TSR={t6[\"tsr\"]*100:.1f}%  StepSR={t6[\"step_sr\"]*100:.1f}%  (n={t6[\"num_episodes\"]})')
print()
"
    fi
}

echo "--- N=3, Pred History ---"
print_result "first_yes" "$(ls $PROJECT_ROOT/v19_step_aware/outputs/best_of_3_pred/eval_summary_*.json 2>/dev/null | tail -1)"
print_result "logprob  " "$(ls $PROJECT_ROOT/v19_step_aware/outputs/logprob_n3_pred/eval_summary_*.json 2>/dev/null | tail -1)"

echo "--- N=5, Pred History ---"
print_result "first_yes" "$(ls $PROJECT_ROOT/v19_step_aware/outputs/best_of_5_pred/eval_summary_*.json 2>/dev/null | tail -1)"
print_result "logprob  " "$(ls $PROJECT_ROOT/v19_step_aware/outputs/logprob_n5_pred/eval_summary_*.json 2>/dev/null | tail -1)"

echo "--- N=3, GT History ---"
print_result "first_yes" "$(ls $PROJECT_ROOT/v19_step_aware/outputs/best_of_3_gt/eval_summary_*.json 2>/dev/null | tail -1)"
print_result "logprob  " "$(ls $PROJECT_ROOT/v19_step_aware/outputs/logprob_n3_gt/eval_summary_*.json 2>/dev/null | tail -1)"

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
