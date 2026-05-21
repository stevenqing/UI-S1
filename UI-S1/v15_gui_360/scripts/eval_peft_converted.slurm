#!/bin/bash
#SBATCH --job-name=ev_v15_peft
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/eval_peft_converted_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/eval_peft_converted_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V15: PEFT Standard LoRA → Cooperative + Eval
#
# Source: extracted_lora_fullsft_r128 (PEFT, 7 modules, rank 128)
# Pipeline:
# 1. Convert PEFT → cooperative V13 format
# 2. Standard LoRA eval (merge + vLLM, fast)
# 3. Cooperative LoRA eval (serve_cooperative_direct, slow)
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
PEFT_CHECKPOINT="$PROJECT_ROOT/train_GUI_360/moe_rl/extracted_lora_fullsft_r128"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"

# Output dirs
COOP_DIR="$PROJECT_ROOT/checkpoints/v15_peft_to_cooperative_r128"
MERGED_DIR="$PROJECT_ROOT/train_GUI_360/moe_rl/extracted_lora_fullsft_r128_merged"
EVAL_STD="$PROJECT_ROOT/v15_gui_360/outputs/eval_peft_standard_r128"
EVAL_COOP="$PROJECT_ROOT/v15_gui_360/outputs/eval_peft_cooperative_r128"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p "$PROJECT_ROOT/v15_gui_360/scripts/logs"

echo "=========================================="
echo "V15: PEFT r128 Standard + Cooperative Eval"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "PEFT:    $PEFT_CHECKPOINT"
echo "=========================================="

# ── Step 1: Convert PEFT → cooperative ──────────────────────────

if [ -f "$COOP_DIR/lora_weights.pt" ]; then
    echo "Cooperative conversion already done, skipping."
else
    echo ""
    echo "=========================================="
    echo "Step 1: Convert PEFT → Cooperative V13"
    echo "Time: $(date)"
    echo "=========================================="

    python "$PROJECT_ROOT/v15_gui_360/convert_peft_to_cooperative.py" \
        --peft_checkpoint "$PEFT_CHECKPOINT" \
        --output "$COOP_DIR" \
        --num_comm_rounds 2 \
        --noise_scale 0.1
fi

# ── Step 2: Standard LoRA eval (merge PEFT + vLLM) ─────────────

if ls "$EVAL_STD"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "Standard LoRA eval already done, skipping."
else
    echo ""
    echo "=========================================="
    echo "Step 2: Standard LoRA Eval (PEFT merge + vLLM)"
    echo "Time: $(date)"
    echo "=========================================="

    # Use pre-existing merged model (PEFT r128 already merged)
    if [ ! -f "$MERGED_DIR/config.json" ]; then
        echo "ERROR: Merged model not found at $MERGED_DIR"
        exit 1
    fi
    echo "Using pre-merged model: $MERGED_DIR"

    MODEL_NAME="v15_peft_std_r128"
    mkdir -p "$EVAL_STD"

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
        --output_dir "$EVAL_STD" \
        --threads 128

    echo "Standard LoRA results:"
    cat "$EVAL_STD"/eval_summary_*.json 2>/dev/null || true

    echo "Shutting down vLLM..."
    kill $VLLM_PID 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    sleep 10
fi

# ── Step 3: Cooperative LoRA eval ──────────────────────────────

if ls "$EVAL_COOP"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "Cooperative eval already done, skipping."
else
    echo ""
    echo "=========================================="
    echo "Step 3: Cooperative LoRA Eval"
    echo "Time: $(date)"
    echo "=========================================="

    MODEL_NAME="v15_peft_coop_r128"
    mkdir -p "$EVAL_COOP"

    python "$PROJECT_ROOT/v13_gui_360/serve_cooperative_direct.py" \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$COOP_DIR" \
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

    python "$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$EVAL_COOP" \
        --threads 4

    echo "Cooperative LoRA results:"
    cat "$EVAL_COOP"/eval_summary_*.json 2>/dev/null || true

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
echo "--- Standard LoRA (PEFT r128, merged) ---"
cat "$EVAL_STD"/eval_summary_*.json 2>/dev/null || echo "Not available"
echo ""
echo "--- Cooperative LoRA (PEFT r128 → V13) ---"
cat "$EVAL_COOP"/eval_summary_*.json 2>/dev/null || echo "Not available"
echo ""
echo "--- Reference ---"
echo "Full-param SFT:              TSR=22.2%"
echo "SVD allmod standard r128:    TSR=17.9%, StepSR=65.0%"
echo "SVD attn-only standard r128: TSR=4.3%,  StepSR=48.2%"
echo "=========================================="
