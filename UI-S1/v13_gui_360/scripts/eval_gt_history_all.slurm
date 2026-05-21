#!/bin/bash
#SBATCH --job-name=ev_gt_hist
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/eval_gt_history_all_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/eval_gt_history_all_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# GT History Evaluation of ALL baselines
#
# Teacher-forced: GT actions in history, all steps evaluated.
# Gives step-level metrics comparable to training val_reward.
#
# Models served sequentially: start → eval → kill → next
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
EVAL_SCRIPT="$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py"
EVAL_PREFIX="$PROJECT_ROOT/v13_gui_360/outputs_gt_history"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

mkdir -p "$EVAL_PREFIX"
mkdir -p "$PROJECT_ROOT/v13_gui_360/scripts/logs"

echo "=========================================="
echo "GT History Evaluation — All Baselines"
echo "  Job ID: ${SLURM_JOB_ID}"
echo "  Start:  $(date)"
echo "=========================================="

# ── Function: serve merged model via vLLM, eval with GT history, kill ──

eval_vllm_model() {
    local MODEL_DIR=$1
    local MODEL_NAME=$2
    local EVAL_DIR="$EVAL_PREFIX/$MODEL_NAME"

    mkdir -p "$EVAL_DIR"

    # Skip if already done
    if ls "$EVAL_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
        echo "  [$MODEL_NAME] Already done, skipping."
        return 0
    fi

    echo ""
    echo "=========================================="
    echo "  Model: $MODEL_NAME"
    echo "  Path:  $MODEL_DIR"
    echo "  Time:  $(date)"
    echo "=========================================="

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
        echo "  ERROR: vLLM failed to start for $MODEL_NAME"
        kill $VLLM_PID 2>/dev/null || true
        return 1
    fi

    python "$EVAL_SCRIPT" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$EVAL_DIR" \
        --threads 4 \
        --gt_history

    echo "  Results ($MODEL_NAME):"
    python3 -c "
import json, glob
f = sorted(glob.glob('$EVAL_DIR/eval_summary_*.json'))[-1]
d = json.load(open(f))
print(f'  TSR:      {d[\"tsr\"]*100:.1f}%')
print(f'  StepSR:   {d[\"step_sr\"]*100:.1f}%')
print(f'  MeanRwd:  {d[\"mean_reward\"]:.4f}')
print(f'  Episodes: {d[\"num_episodes\"]}')
" 2>/dev/null || true

    kill $VLLM_PID 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    sleep 10
}

# ── Function: serve cooperative model, eval with GT history, kill ──

eval_cooperative_model() {
    local COOP_CKPT=$1
    local MODEL_NAME=$2
    local VERSION=$3  # v12 or v13
    local EVAL_DIR="$EVAL_PREFIX/$MODEL_NAME"
    local SERVE_SCRIPT="$PROJECT_ROOT/v13_gui_360/serve_cooperative_direct.py"

    mkdir -p "$EVAL_DIR"

    if ls "$EVAL_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
        echo "  [$MODEL_NAME] Already done, skipping."
        return 0
    fi

    echo ""
    echo "=========================================="
    echo "  Model: $MODEL_NAME (cooperative $VERSION)"
    echo "  Path:  $COOP_CKPT"
    echo "  Time:  $(date)"
    echo "=========================================="

    python "$SERVE_SCRIPT" \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$COOP_CKPT" \
        --version "$VERSION" \
        --served-model-name "$MODEL_NAME" \
        --port 8000 \
        --num_gpus 4 \
        &
    local SERVER_PID=$!

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

    python "$EVAL_SCRIPT" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$EVAL_DIR" \
        --threads 4 \
        --gt_history

    echo "  Results ($MODEL_NAME):"
    python3 -c "
import json, glob
f = sorted(glob.glob('$EVAL_DIR/eval_summary_*.json'))[-1]
d = json.load(open(f))
print(f'  TSR:      {d[\"tsr\"]*100:.1f}%')
print(f'  StepSR:   {d[\"step_sr\"]*100:.1f}%')
print(f'  MeanRwd:  {d[\"mean_reward\"]:.4f}')
print(f'  Episodes: {d[\"num_episodes\"]}')
" 2>/dev/null || true

    kill $SERVER_PID 2>/dev/null || true
    pkill -f "serve_cooperative_direct.py" 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 10
}

# ══════════════════════════════════════════════════════════════
# 1. Base model (zero-shot)
# ══════════════════════════════════════════════════════════════
eval_vllm_model \
    "$BASE_MODEL" \
    "base_model_gt"

# ══════════════════════════════════════════════════════════════
# 2. gui_action (full SFT, 97K data)
# ══════════════════════════════════════════════════════════════
eval_vllm_model \
    "$PROJECT_ROOT/checkpoints/gui_action" \
    "gui_action_gt"

# ══════════════════════════════════════════════════════════════
# 3. Full-param SFT step-250 (best, 22.2% TSR)
# ══════════════════════════════════════════════════════════════
eval_vllm_model \
    "$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_balanced_full_sft/checkpoint-250" \
    "fullsft_step250_gt"

# ══════════════════════════════════════════════════════════════
# 4. SVD Standard LoRA merged (17.9% TSR)
# ══════════════════════════════════════════════════════════════
eval_vllm_model \
    "$PROJECT_ROOT/checkpoints/v15_svd_standard_lora_allmod_merged" \
    "svd_standard_r128_gt"

# ══════════════════════════════════════════════════════════════
# 5. PEFT Standard r=128 merged (18.1% TSR)
# ══════════════════════════════════════════════════════════════
eval_vllm_model \
    "$PROJECT_ROOT/checkpoints/v15_peft_balanced_r128_merged" \
    "peft_standard_r128_gt"

# ══════════════════════════════════════════════════════════════
# 6. PEFT Cooperative r=128 (18.6% TSR)
# ══════════════════════════════════════════════════════════════
eval_cooperative_model \
    "$PROJECT_ROOT/checkpoints/v15_peft_balanced_cooperative_r128" \
    "peft_coop_r128_gt" \
    "v13"

# ══════════════════════════════════════════════════════════════
# 7. V15 Cooperative RL step-25 (20.8% TSR, best RL)
# ══════════════════════════════════════════════════════════════
eval_cooperative_model \
    "$PROJECT_ROOT/checkpoints/v15_rl_from_svd_extract/epoch-0_step-25/cooperative" \
    "v15_rl_svd_step25_gt" \
    "v13"

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
echo ""
echo "=========================================="
echo "GT HISTORY EVALUATION — SUMMARY"
echo "=========================================="
echo ""

for d in "$EVAL_PREFIX"/*/; do
    MODEL_NAME=$(basename "$d")
    python3 -c "
import json, glob, sys
files = sorted(glob.glob('${d}eval_summary_*.json'))
if not files:
    print(f'  {\"$MODEL_NAME\":30s} — NO RESULTS')
    sys.exit(0)
d = json.load(open(files[-1]))
print(f'  {\"$MODEL_NAME\":30s}  TSR={d[\"tsr\"]*100:5.1f}%  StepSR={d[\"step_sr\"]*100:5.1f}%  MeanRwd={d[\"mean_reward\"]:.4f}')
" 2>/dev/null || true
done

echo ""
echo "Done: $(date)"
