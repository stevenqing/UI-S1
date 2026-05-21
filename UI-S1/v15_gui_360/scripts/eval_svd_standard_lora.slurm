#!/bin/bash
#SBATCH --job-name=ev_v15_stdl
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/eval_svd_standard_lora_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/eval_svd_standard_lora_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V15: SVD-Extracted Standard LoRA Eval
#
# Pipeline:
# 1. Convert cooperative → standard LoRA (average A_1, A_2)
# 2. Merge standard LoRA into base model
# 3. Serve with vLLM (TP=4)
# 4. Eval on gui360_test_1000_balanced.jsonl
#
# This gives a fair comparison: same SVD rank-128 approximation
# of ΔW, but without cooperative routing/communication overhead.
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
COOP_CHECKPOINT="$PROJECT_ROOT/checkpoints/v15_svd_extracted_cooperative"
STD_LORA_DIR="$PROJECT_ROOT/checkpoints/v15_svd_extracted_standard_lora"
MERGED_DIR="$PROJECT_ROOT/checkpoints/v15_svd_standard_lora_merged"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/v15_gui_360/outputs/eval_svd_standard_lora"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p "$PROJECT_ROOT/v15_gui_360/scripts/logs"

echo "=========================================="
echo "V15: SVD Standard LoRA Eval"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

# ── Step 1: Convert cooperative → standard LoRA ──────────────

if [ -f "$STD_LORA_DIR/lora_weights.pt" ]; then
    echo "Standard LoRA conversion already done, skipping."
else
    echo ""
    echo "=========================================="
    echo "Step 1: Convert cooperative → standard LoRA"
    echo "Time: $(date)"
    echo "=========================================="

    python "$PROJECT_ROOT/v15_gui_360/convert_cooperative_to_standard_lora.py" \
        --coop_checkpoint "$COOP_CHECKPOINT" \
        --output "$STD_LORA_DIR"
fi

# ── Step 2: Merge into base model ───────────────────────────

if [ -f "$MERGED_DIR/config.json" ]; then
    echo "Merge already done, skipping."
else
    echo ""
    echo "=========================================="
    echo "Step 2: Merge standard LoRA into base model"
    echo "Time: $(date)"
    echo "=========================================="

    python "$PROJECT_ROOT/v12_gui_360/merge_standard_lora.py" \
        --base_model "$BASE_MODEL" \
        --lora_checkpoint "$STD_LORA_DIR" \
        --output_dir "$MERGED_DIR"
fi

# ── Step 3: Serve + Eval ────────────────────────────────────

if ls "$OUTPUT_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "Eval already done, skipping."
else
    echo ""
    echo "=========================================="
    echo "Step 3: Serve + Eval (vLLM TP=4)"
    echo "Time: $(date)"
    echo "=========================================="

    MODEL_NAME="v15_svd_standard_lora"
    mkdir -p "$OUTPUT_DIR"

    python -m vllm.entrypoints.openai.api_server \
        --model "$MERGED_DIR" \
        --served-model-name "$MODEL_NAME" \
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

    python "$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --threads 128

    echo "Shutting down vLLM..."
    kill $VLLM_PID 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    sleep 10
fi

# ── Summary ─────────────────────────────────────────────────

echo ""
echo "=========================================="
echo "Done: $(date)"
echo ""
if ls "$OUTPUT_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "Results:"
    cat "$OUTPUT_DIR"/eval_summary_*.json
fi
echo "=========================================="
