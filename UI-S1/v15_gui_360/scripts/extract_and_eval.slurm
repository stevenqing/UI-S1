#!/bin/bash
#SBATCH --job-name=ev_v15_svd
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/extract_and_eval_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/extract_and_eval_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V15: SVD Extract Full-Param SFT → Cooperative LoRA + Eval
#
# 1. Extract cooperative LoRA from full-param SFT via SVD (~2-5 min)
# 2. Serve with serve_cooperative_direct.py (V13 format)
# 3. Eval on gui360_test_1000_balanced.jsonl
# 4. Target: close to full-param's TSR=22.2%
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
SFT_MODEL="$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_balanced_full_sft/checkpoint-250"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"

# Output dirs
EXTRACT_DIR="$PROJECT_ROOT/checkpoints/v15_svd_extracted_cooperative"
EVAL_OUTPUT="$PROJECT_ROOT/v15_gui_360/outputs/eval_svd_extracted"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p "$PROJECT_ROOT/v15_gui_360/scripts/logs"

echo "=========================================="
echo "V15: SVD Extract + Eval"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

# ── Step 1: SVD Extraction ──────────────────────────────────────

if [ -f "$EXTRACT_DIR/lora_weights.pt" ]; then
    echo "Extraction already done, skipping."
else
    echo ""
    echo "=========================================="
    echo "Step 1: SVD Extraction (Full-SFT → Cooperative)"
    echo "Time: $(date)"
    echo "=========================================="

    python "$PROJECT_ROOT/v15_gui_360/extract_fullsft_to_cooperative.py" \
        --sft_model "$SFT_MODEL" \
        --base_model "$BASE_MODEL" \
        --output "$EXTRACT_DIR" \
        --rank 128 \
        --alpha 256 \
        --num_comm_rounds 2 \
        --noise_scale 0.1

    echo "Extraction complete: $(date)"
fi

# ── Step 2: Serve + Eval ────────────────────────────────────────

if ls "$EVAL_OUTPUT"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "Eval already done, skipping."
else
    echo ""
    echo "=========================================="
    echo "Step 2: Serve + Eval (GUI-360 Template)"
    echo "Time: $(date)"
    echo "=========================================="

    MODEL_NAME="v15_svd_extracted"
    mkdir -p "$EVAL_OUTPUT"

    # Start cooperative server (4 GPU replicas)
    python "$PROJECT_ROOT/v13_gui_360/serve_cooperative_direct.py" \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$EXTRACT_DIR" \
        --version v13 \
        --served-model-name "$MODEL_NAME" \
        --port 8000 \
        &
    SERVER_PID=$!

    WAIT_COUNT=0
    while [ $WAIT_COUNT -lt 600 ]; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "Server ready after ${WAIT_COUNT}s"
            break
        fi
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
    done

    if [ $WAIT_COUNT -ge 600 ]; then
        echo "ERROR: Server failed to start"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi

    # Run evaluation
    python "$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$EVAL_OUTPUT" \
        --threads 4

    echo "Shutting down server..."
    kill $SERVER_PID 2>/dev/null || true
    pkill -f "serve_cooperative_direct" 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 10
fi

# ── Summary ─────────────────────────────────────────────────────

echo ""
echo "=========================================="
echo "Done: $(date)"
echo ""
if [ -f "$EVAL_OUTPUT"/eval_summary_*.json 2>/dev/null ]; then
    echo "Results:"
    cat "$EVAL_OUTPUT"/eval_summary_*.json
fi
echo "=========================================="
