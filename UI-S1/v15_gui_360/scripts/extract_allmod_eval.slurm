#!/bin/bash
#SBATCH --job-name=ev_v15_all
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/extract_allmod_eval_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/extract_allmod_eval_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V15: SVD Extract (ALL 7 MODULES) + Eval
#
# Previous extraction only used attention (q/k/v/o) = 12% of ΔW energy.
# MLP modules (gate/up/down) account for 83% of ΔW energy.
# This version extracts all 7 modules.
#
# Pipeline:
# 1. Extract cooperative LoRA (all modules, rank 128) (~2 min)
# 2. Convert to standard LoRA + merge + vLLM eval (~30 min)
# 3. Cooperative serve + eval (~3 hours)
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

# Output dirs — use "_allmod" suffix to distinguish from attn-only runs
EXTRACT_DIR="$PROJECT_ROOT/checkpoints/v15_svd_cooperative_allmod"
STD_LORA_DIR="$PROJECT_ROOT/checkpoints/v15_svd_standard_lora_allmod"
MERGED_DIR="$PROJECT_ROOT/checkpoints/v15_svd_standard_lora_allmod_merged"
EVAL_STD="$PROJECT_ROOT/v15_gui_360/outputs/eval_svd_standard_lora_allmod"
EVAL_COOP="$PROJECT_ROOT/v15_gui_360/outputs/eval_svd_cooperative_allmod"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p "$PROJECT_ROOT/v15_gui_360/scripts/logs"

echo "=========================================="
echo "V15: SVD Extract ALL MODULES + Eval"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "Target modules: q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
echo "=========================================="

# ── Step 1: SVD Extraction (all 7 modules) ─────────────────────

if [ -f "$EXTRACT_DIR/lora_weights.pt" ]; then
    echo "Extraction already done, skipping."
else
    echo ""
    echo "=========================================="
    echo "Step 1: SVD Extraction (all 7 modules, rank 128)"
    echo "Time: $(date)"
    echo "=========================================="

    python "$PROJECT_ROOT/v15_gui_360/extract_fullsft_to_cooperative.py" \
        --sft_model "$SFT_MODEL" \
        --base_model "$BASE_MODEL" \
        --output "$EXTRACT_DIR" \
        --rank 128 \
        --alpha 256 \
        --num_comm_rounds 2 \
        --noise_scale 0.1 \
        --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj

    echo "Extraction complete: $(date)"
fi

# ── Step 2: Standard LoRA eval (fast — vLLM with 128 threads) ──

if ls "$EVAL_STD"/eval_summary_*.json 1>/dev/null 2>&1; then
    echo "Standard LoRA eval already done, skipping."
else
    echo ""
    echo "=========================================="
    echo "Step 2: Standard LoRA Eval (convert → merge → vLLM)"
    echo "Time: $(date)"
    echo "=========================================="

    # 2a. Convert cooperative → standard LoRA
    if [ ! -f "$STD_LORA_DIR/lora_weights.pt" ]; then
        python "$PROJECT_ROOT/v15_gui_360/convert_cooperative_to_standard_lora.py" \
            --coop_checkpoint "$EXTRACT_DIR" \
            --output "$STD_LORA_DIR"
    fi

    # 2b. Merge standard LoRA into base model
    if [ ! -f "$MERGED_DIR/config.json" ]; then
        python "$PROJECT_ROOT/v12_gui_360/merge_standard_lora.py" \
            --base_model "$BASE_MODEL" \
            --lora_checkpoint "$STD_LORA_DIR" \
            --output_dir "$MERGED_DIR"
    fi

    # 2c. Serve with vLLM + eval
    MODEL_NAME="v15_svd_std_allmod"
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
    echo "Step 3: Cooperative LoRA Eval (serve_cooperative_direct)"
    echo "Time: $(date)"
    echo "=========================================="

    MODEL_NAME="v15_svd_coop_allmod"
    mkdir -p "$EVAL_COOP"

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
echo "--- Standard LoRA (all modules, rank 128) ---"
cat "$EVAL_STD"/eval_summary_*.json 2>/dev/null || echo "Not available"
echo ""
echo "--- Cooperative LoRA (all modules, rank 128) ---"
cat "$EVAL_COOP"/eval_summary_*.json 2>/dev/null || echo "Not available"
echo ""
echo "--- Reference ---"
echo "Full-param SFT: TSR=22.2%"
echo "Attn-only SVD standard LoRA: TSR=4.3%, StepSR=48.2%"
echo "=========================================="
