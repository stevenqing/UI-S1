#!/bin/bash
#SBATCH --job-name=v16_merge
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v16_task_vector_merge/scripts/logs/merge_and_eval_all_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v16_task_vector_merge/scripts/logs/merge_and_eval_all_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V16: Task Vector Merge (gui_grounding + gui_action)
#
# 12 configs: avg(1) + task_arith(4) + ties(3) + dare(4)
# For each: merge → vLLM serve → eval → kill → next
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
MODEL_A="$PROJECT_ROOT/checkpoints/gui_grounding"
MODEL_B="$PROJECT_ROOT/checkpoints/gui_action"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
MERGE_SCRIPT="$PROJECT_ROOT/v16_task_vector_merge/merge_task_vectors.py"
EVAL_SCRIPT="$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py"

CKPT_PREFIX="$PROJECT_ROOT/checkpoints/v16"
EVAL_PREFIX="$PROJECT_ROOT/v16_task_vector_merge/outputs"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p "$PROJECT_ROOT/v16_task_vector_merge/scripts/logs"
mkdir -p "$EVAL_PREFIX"

echo "=========================================="
echo "V16: Task Vector Merge - All Configs"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

# ── Reusable function: serve vLLM, eval, kill ────────────────

eval_merged_model() {
    local MODEL_DIR=$1
    local MODEL_NAME=$2
    local EVAL_DIR=$3

    mkdir -p "$EVAL_DIR"

    # Check if eval already done
    if ls "$EVAL_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
        echo "  Eval already done for $MODEL_NAME, skipping."
        return 0
    fi

    echo "  Serving $MODEL_NAME from $MODEL_DIR"
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_DIR" \
        --served-model-name "$MODEL_NAME" \
        --port 8000 \
        --tensor-parallel-size 4 \
        --max-model-len 32768 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --limit-mm-per-prompt '{"image": 2}' \
        &
    local VLLM_PID=$!

    # Wait for server
    local WAIT_COUNT=0
    while [ $WAIT_COUNT -lt 600 ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "  vLLM ready after ${WAIT_COUNT}s"
            break
        fi
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
    done

    if [ $WAIT_COUNT -ge 600 ]; then
        echo "  ERROR: vLLM server failed to start for $MODEL_NAME"
        kill $VLLM_PID 2>/dev/null || true
        return 1
    fi

    # Eval
    python "$EVAL_SCRIPT" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$EVAL_DIR" \
        --threads 128

    echo "  Results for $MODEL_NAME:"
    cat "$EVAL_DIR"/eval_summary_*.json 2>/dev/null || true

    # Kill vLLM
    echo "  Shutting down vLLM..."
    kill $VLLM_PID 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    sleep 10
}

# ── Define all 12 configurations ─────────────────────────────

run_config() {
    local SUFFIX=$1
    shift
    # Remaining args are passed to merge script

    local MERGED_DIR="${CKPT_PREFIX}_${SUFFIX}"
    local EVAL_DIR="${EVAL_PREFIX}/eval_${SUFFIX}"
    local MODEL_NAME="v16_${SUFFIX}"

    echo ""
    echo "=========================================="
    echo "Config: $SUFFIX"
    echo "Time:   $(date)"
    echo "=========================================="

    # Merge if not already done
    if [ -f "$MERGED_DIR/config.json" ]; then
        echo "  Merge already exists: $MERGED_DIR"
    else
        echo "  Merging..."
        python "$MERGE_SCRIPT" \
            --base_model "$BASE_MODEL" \
            --model_a "$MODEL_A" \
            --model_b "$MODEL_B" \
            --output_dir "$MERGED_DIR" \
            "$@"
    fi

    # Eval
    eval_merged_model "$MERGED_DIR" "$MODEL_NAME" "$EVAL_DIR"
}

# ── 1. Simple Average ────────────────────────────────────────

run_config "avg_05_05" \
    --method avg --alpha 0.5 --beta 0.5

# ── 2-5. Task Arithmetic ─────────────────────────────────────

run_config "ta_03_07" \
    --method task_arith --alpha 0.3 --beta 0.7

run_config "ta_05_05" \
    --method task_arith --alpha 0.5 --beta 0.5

run_config "ta_07_03" \
    --method task_arith --alpha 0.7 --beta 0.3

run_config "ta_10_10" \
    --method task_arith --alpha 1.0 --beta 1.0

# ── 6-8. TIES-Merging ────────────────────────────────────────

run_config "ties_k20_l10" \
    --method ties --ties_k 20 --ties_lambda 1.0

run_config "ties_k20_l15" \
    --method ties --ties_k 20 --ties_lambda 1.5

run_config "ties_k50_l10" \
    --method ties --ties_k 50 --ties_lambda 1.0

# ── 9-12. DARE ───────────────────────────────────────────────

run_config "dare_p03_l10" \
    --method dare --dare_p 0.3 --dare_lambda 1.0 --seed 42

run_config "dare_p05_l10" \
    --method dare --dare_p 0.5 --dare_lambda 1.0 --seed 42

run_config "dare_p07_l10" \
    --method dare --dare_p 0.7 --dare_lambda 1.0 --seed 42

run_config "dare_p05_l15" \
    --method dare --dare_p 0.5 --dare_lambda 1.5 --seed 42

# ── Summary ──────────────────────────────────────────────────

echo ""
echo "=========================================="
echo "ALL 12 CONFIGS COMPLETE"
echo "Time: $(date)"
echo ""
echo "=== RESULTS SUMMARY ==="
echo ""

CONFIGS=(
    "avg_05_05"
    "ta_03_07"
    "ta_05_05"
    "ta_07_03"
    "ta_10_10"
    "ties_k20_l10"
    "ties_k20_l15"
    "ties_k50_l10"
    "dare_p03_l10"
    "dare_p05_l10"
    "dare_p07_l10"
    "dare_p05_l15"
)

printf "%-20s %8s %8s %8s\n" "Config" "TSR" "StepSR" "Progress"
printf "%-20s %8s %8s %8s\n" "--------------------" "--------" "--------" "--------"

for cfg in "${CONFIGS[@]}"; do
    EVAL_DIR="${EVAL_PREFIX}/eval_${cfg}"
    SUMMARY=$(ls "$EVAL_DIR"/eval_summary_*.json 2>/dev/null | head -1)
    if [ -n "$SUMMARY" ]; then
        TSR=$(python -c "import json; d=json.load(open('$SUMMARY')); print(f\"{d.get('task_success_rate', d.get('TSR', 'N/A')):.1f}%\")" 2>/dev/null || echo "N/A")
        STEP=$(python -c "import json; d=json.load(open('$SUMMARY')); print(f\"{d.get('step_success_rate', d.get('StepSR', 'N/A')):.1f}%\")" 2>/dev/null || echo "N/A")
        PROG=$(python -c "import json; d=json.load(open('$SUMMARY')); print(f\"{d.get('progress', d.get('Progress', 'N/A')):.1f}%\")" 2>/dev/null || echo "N/A")
        printf "%-20s %8s %8s %8s\n" "$cfg" "$TSR" "$STEP" "$PROG"
    else
        printf "%-20s %8s %8s %8s\n" "$cfg" "FAIL" "FAIL" "FAIL"
    fi
done

echo ""
echo "--- Reference ---"
echo "gui_action alone:    TSR~22.2%"
echo "=========================================="
