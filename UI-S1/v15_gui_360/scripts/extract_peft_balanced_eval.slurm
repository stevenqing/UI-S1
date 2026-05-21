#!/bin/bash
#SBATCH --job-name=ev_peft_bal
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/extract_peft_balanced_eval_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/extract_peft_balanced_eval_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V15: PEFT Standard LoRA Extraction from Balanced SFT + Eval
#
# 1. Extract PEFT LoRA from balanced SFT checkpoint-250 (rank 128)
# 2. Merge PEFT LoRA into base model
# 3. Eval with vLLM (standard LoRA)
# 4. Convert to cooperative V13 + eval
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
PEFT_DIR="$PROJECT_ROOT/checkpoints/v15_peft_balanced_r128"
MERGED_DIR="$PROJECT_ROOT/checkpoints/v15_peft_balanced_r128_merged"
COOP_DIR="$PROJECT_ROOT/checkpoints/v15_peft_balanced_cooperative_r128"
EVAL_STD="$PROJECT_ROOT/v15_gui_360/outputs/eval_peft_balanced_standard_r128"
EVAL_COOP="$PROJECT_ROOT/v15_gui_360/outputs/eval_peft_balanced_cooperative_r128"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p "$PROJECT_ROOT/v15_gui_360/scripts/logs"

echo "=========================================="
echo "V15: PEFT Extract Balanced SFT + Eval"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "SFT:     $SFT_MODEL"
echo "=========================================="

# ── Step 1: PEFT Extraction ────────────────────────────────────

if [ -f "$PEFT_DIR/adapter_model.safetensors" ]; then
    echo "PEFT extraction already done, skipping."
else
    echo ""
    echo "=========================================="
    echo "Step 1: PEFT LoRA Extraction (rank 128, 7 modules)"
    echo "Time: $(date)"
    echo "=========================================="

    python "$PROJECT_ROOT/Option-incentivized-MoE/extract_fullsft_to_lora.py" \
        --sft_model "$SFT_MODEL" \
        --base_model "$BASE_MODEL" \
        --output "$PEFT_DIR" \
        --rank 128 \
        --alpha 256
fi

# ── Step 2: Standard LoRA Eval (merge + vLLM) ──────────────────

if ls "$EVAL_STD"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "Standard LoRA eval already done, skipping."
else
    echo ""
    echo "=========================================="
    echo "Step 2: Merge PEFT + vLLM Eval"
    echo "Time: $(date)"
    echo "=========================================="

    # Merge PEFT LoRA into base model
    if [ ! -f "$MERGED_DIR/config.json" ]; then
        python "$PROJECT_ROOT/v15_gui_360/merge_peft_lora.py" \
            --base_model "$BASE_MODEL" \
            --peft_checkpoint "$PEFT_DIR" \
            --output_dir "$MERGED_DIR"
    fi

    MODEL_NAME="v15_peft_bal_std_r128"
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

# ── Step 3: Convert to Cooperative + Eval ──────────────────────

if ls "$EVAL_COOP"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "Cooperative eval already done, skipping."
else
    echo ""
    echo "=========================================="
    echo "Step 3: Convert PEFT → Cooperative V13 + Eval"
    echo "Time: $(date)"
    echo "=========================================="

    # Convert PEFT → cooperative
    if [ ! -f "$COOP_DIR/lora_weights.pt" ]; then
        python "$PROJECT_ROOT/v15_gui_360/convert_peft_to_cooperative.py" \
            --peft_checkpoint "$PEFT_DIR" \
            --output "$COOP_DIR" \
            --num_comm_rounds 2 \
            --noise_scale 0.1
    fi

    MODEL_NAME="v15_peft_bal_coop_r128"
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
echo "--- Standard LoRA (PEFT balanced r128) ---"
cat "$EVAL_STD"/eval_summary_*.json 2>/dev/null || echo "Not available"
echo ""
echo "--- Cooperative LoRA (PEFT balanced r128 → V13) ---"
cat "$EVAL_COOP"/eval_summary_*.json 2>/dev/null || echo "Not available"
echo ""
echo "--- Reference ---"
echo "Full-param SFT:              TSR=22.2%"
echo "SVD allmod standard r128:    TSR=17.9%, StepSR=65.0%"
echo "=========================================="
