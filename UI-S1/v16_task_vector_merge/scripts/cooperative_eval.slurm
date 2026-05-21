#!/bin/bash
#SBATCH --job-name=v16_coop
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v16_task_vector_merge/scripts/logs/cooperative_eval_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v16_task_vector_merge/scripts/logs/cooperative_eval_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V16: Cooperative LoRA Eval (gui_grounding + gui_action as two experts)
#
# 3 configs: coop_r128_attn, coop_r128_attn_nocomm, coop_r128_all7
# For each: convert (joint SVD) → serve_cooperative_direct → eval → kill → next
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

CONVERT_SCRIPT="$PROJECT_ROOT/v16_task_vector_merge/convert_fullsft_to_cooperative.py"
SERVE_SCRIPT="$PROJECT_ROOT/v13_gui_360/serve_cooperative_direct.py"
EVAL_SCRIPT="$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py"

CKPT_PREFIX="$PROJECT_ROOT/checkpoints/v16"
EVAL_PREFIX="$PROJECT_ROOT/v16_task_vector_merge/outputs"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p "$PROJECT_ROOT/v16_task_vector_merge/scripts/logs"
mkdir -p "$EVAL_PREFIX"

echo "=========================================="
echo "V16: Cooperative LoRA - Joint SVD + Eval"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

# ── Reusable function: convert + serve cooperative + eval + kill ────────

eval_cooperative_model() {
    local COOP_DIR=$1
    local MODEL_NAME=$2
    local EVAL_DIR=$3
    local EXTRA_FLAGS=$4  # e.g. "--no_comm"

    mkdir -p "$EVAL_DIR"

    # Check if eval already done
    if ls "$EVAL_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
        echo "  Eval already done for $MODEL_NAME, skipping."
        return 0
    fi

    echo "  Serving $MODEL_NAME from $COOP_DIR"
    python "$SERVE_SCRIPT" \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$COOP_DIR" \
        --version v13 \
        --served-model-name "$MODEL_NAME" \
        --port 8000 \
        --num_gpus 4 \
        $EXTRA_FLAGS &
    local SERVER_PID=$!

    # Wait for server
    local WAIT_COUNT=0
    while [ $WAIT_COUNT -lt 600 ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "  Server ready after ${WAIT_COUNT}s"
            break
        fi
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
    done

    if [ $WAIT_COUNT -ge 600 ]; then
        echo "  ERROR: Server failed to start for $MODEL_NAME"
        kill $SERVER_PID 2>/dev/null || true
        return 1
    fi

    # Eval
    python "$EVAL_SCRIPT" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$EVAL_DIR" \
        --threads 4

    echo "  Results for $MODEL_NAME:"
    cat "$EVAL_DIR"/eval_summary_*.json 2>/dev/null || true

    # Kill server
    echo "  Shutting down server..."
    kill $SERVER_PID 2>/dev/null || true
    pkill -f "serve_cooperative_direct.py" 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 10
}

# ── Config 1: coop_r128_attn (main config) ──────────────────────────────

SUFFIX="coop_r128_attn"
COOP_DIR="${CKPT_PREFIX}_${SUFFIX}"
EVAL_DIR="${EVAL_PREFIX}/eval_${SUFFIX}"
MODEL_NAME="v16_${SUFFIX}"

echo ""
echo "=========================================="
echo "Config 1: $SUFFIX"
echo "Time:     $(date)"
echo "=========================================="

if [ -f "$COOP_DIR/cooperative_config.json" ]; then
    echo "  Checkpoint already exists: $COOP_DIR"
else
    echo "  Converting (joint SVD)..."
    python "$CONVERT_SCRIPT" \
        --base_model "$BASE_MODEL" \
        --model_a "$MODEL_A" \
        --model_b "$MODEL_B" \
        --output_dir "$COOP_DIR" \
        --rank 128 --alpha 256 \
        --num_comm_rounds 2 \
        --target_modules "q_proj,k_proj,v_proj,o_proj"
fi

eval_cooperative_model "$COOP_DIR" "$MODEL_NAME" "$EVAL_DIR" ""

# ── Config 2: coop_r128_attn_nocomm (ablation: no communication) ────────

SUFFIX="coop_r128_attn_nocomm"
# Reuse same checkpoint as config 1 (communication disabled at serving time)
COOP_DIR="${CKPT_PREFIX}_coop_r128_attn"
EVAL_DIR="${EVAL_PREFIX}/eval_${SUFFIX}"
MODEL_NAME="v16_${SUFFIX}"

echo ""
echo "=========================================="
echo "Config 2: $SUFFIX"
echo "Time:     $(date)"
echo "=========================================="

eval_cooperative_model "$COOP_DIR" "$MODEL_NAME" "$EVAL_DIR" "--no_comm"

# ── Config 3: coop_r128_all7 (attn + MLP modules) ───────────────────────

SUFFIX="coop_r128_all7"
COOP_DIR="${CKPT_PREFIX}_${SUFFIX}"
EVAL_DIR="${EVAL_PREFIX}/eval_${SUFFIX}"
MODEL_NAME="v16_${SUFFIX}"

echo ""
echo "=========================================="
echo "Config 3: $SUFFIX"
echo "Time:     $(date)"
echo "=========================================="

if [ -f "$COOP_DIR/cooperative_config.json" ]; then
    echo "  Checkpoint already exists: $COOP_DIR"
else
    echo "  Converting (joint SVD, all 7 modules)..."
    python "$CONVERT_SCRIPT" \
        --base_model "$BASE_MODEL" \
        --model_a "$MODEL_A" \
        --model_b "$MODEL_B" \
        --output_dir "$COOP_DIR" \
        --rank 128 --alpha 256 \
        --num_comm_rounds 2 \
        --target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
fi

eval_cooperative_model "$COOP_DIR" "$MODEL_NAME" "$EVAL_DIR" ""

# ── Summary ──────────────────────────────────────────────────────────────

echo ""
echo "=========================================="
echo "ALL 3 COOPERATIVE CONFIGS COMPLETE"
echo "Time: $(date)"
echo ""
echo "=== RESULTS SUMMARY ==="
echo ""

CONFIGS=(
    "coop_r128_attn"
    "coop_r128_attn_nocomm"
    "coop_r128_all7"
)

printf "%-25s %8s %8s %8s\n" "Config" "TSR" "StepSR" "Progress"
printf "%-25s %8s %8s %8s\n" "-------------------------" "--------" "--------" "--------"

for cfg in "${CONFIGS[@]}"; do
    EVAL_DIR="${EVAL_PREFIX}/eval_${cfg}"
    SUMMARY=$(ls "$EVAL_DIR"/eval_summary_*.json 2>/dev/null | head -1)
    if [ -n "$SUMMARY" ]; then
        TSR=$(python -c "import json; d=json.load(open('$SUMMARY')); print(f\"{d.get('task_success_rate', d.get('TSR', 'N/A')):.1f}%\")" 2>/dev/null || echo "N/A")
        STEP=$(python -c "import json; d=json.load(open('$SUMMARY')); print(f\"{d.get('step_success_rate', d.get('StepSR', 'N/A')):.1f}%\")" 2>/dev/null || echo "N/A")
        PROG=$(python -c "import json; d=json.load(open('$SUMMARY')); print(f\"{d.get('progress', d.get('Progress', 'N/A')):.1f}%\")" 2>/dev/null || echo "N/A")
        printf "%-25s %8s %8s %8s\n" "$cfg" "$TSR" "$STEP" "$PROG"
    else
        printf "%-25s %8s %8s %8s\n" "$cfg" "FAIL" "FAIL" "FAIL"
    fi
done

echo ""
echo "--- Reference ---"
echo "gui_action alone:    TSR~22.2%"
echo "=========================================="
