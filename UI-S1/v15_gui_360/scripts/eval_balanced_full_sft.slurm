#!/bin/bash
#SBATCH --job-name=ev_v15_fsft
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/eval_balanced_full_sft_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/eval_balanced_full_sft_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V15: Eval Full-Parameter SFT on GUI-360
#
# Uses the same SFT eval standard as cooperative SFT eval:
#   - eval_gui360_template.py (corrected prompt, stop_on_error)
#   - gui360_test_1000_balanced.jsonl (1000 episodes)
#   - vLLM server for fast inference
#
# Evaluates checkpoints: 100, 150, 200, 250, 272 (final)
# checkpoint-200 (epoch~3) has best eval_loss=0.0914
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
FULL_SFT_ROOT="$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_balanced_full_sft"
TEST_DATA="$PROJECT_ROOT/v12_gui_360/data/gui360_test_1000_balanced.jsonl"
OUTPUT_BASE="$PROJECT_ROOT/v15_gui_360/outputs"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=========================================="
echo "V15 Eval: Full-Parameter SFT (balanced data)"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

eval_checkpoint() {
    local CKPT_STEP="$1"
    local CKPT_PATH="$FULL_SFT_ROOT/checkpoint-${CKPT_STEP}"
    local MODEL_NAME="v15_full_sft_step${CKPT_STEP}"
    local OUTPUT_DIR="$OUTPUT_BASE/eval_full_sft_step-${CKPT_STEP}"

    if ls "$OUTPUT_DIR"/eval_summary_*.json 1>/dev/null 2>&1; then
        echo "[step-${CKPT_STEP}] Already evaluated, skipping."
        return 0
    fi

    if [ ! -f "$CKPT_PATH/config.json" ]; then
        echo "[step-${CKPT_STEP}] Checkpoint not found at $CKPT_PATH, skipping."
        return 0
    fi

    echo ""
    echo "=========================================="
    echo "Evaluating: full SFT step-${CKPT_STEP}"
    echo "Time: $(date)"
    echo "=========================================="

    mkdir -p "$OUTPUT_DIR"

    # Start vLLM server
    python -m vllm.entrypoints.openai.api_server \
        --model "$CKPT_PATH" \
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
            echo "vLLM ready after ${WAIT_COUNT}s"
            break
        fi
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
    done

    if [ $WAIT_COUNT -ge 600 ]; then
        echo "ERROR: vLLM server failed to start for step-${CKPT_STEP}"
        kill $VLLM_PID 2>/dev/null || true
        return 1
    fi

    # Run eval
    python "$PROJECT_ROOT/v13_gui_360/eval_gui360_template.py" \
        --test_data "$TEST_DATA" \
        --api_url "http://localhost:8000/v1" \
        --model_name "$MODEL_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --threads 128

    # Shutdown server
    echo "Shutting down vLLM..."
    kill $VLLM_PID 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    sleep 15

    # Print summary
    local SUMMARY=$(ls "$OUTPUT_DIR"/eval_summary_*.json 2>/dev/null | tail -1)
    if [ -n "$SUMMARY" ]; then
        echo "  Result: $(cat "$SUMMARY")"
    fi
}

# Eval all checkpoints
for STEP in 100 150 200 250 272; do
    eval_checkpoint "$STEP"
done

# Collect results summary
echo ""
echo "=========================================="
echo "Full SFT Eval Summary"
echo "=========================================="
python3 -c "
import json, glob

print(f'{\"Checkpoint\":<15} {\"Epoch\":<8} {\"TSR\":<8} {\"StepSR\":<8} {\"Progress\":<10}')
print('-' * 55)

for step in [100, 150, 200, 250, 272]:
    output_dir = '$OUTPUT_BASE/eval_full_sft_step-' + str(step)
    summary_files = glob.glob(f'{output_dir}/eval_summary_*.json')
    if summary_files:
        d = json.load(open(summary_files[-1]))
        # Approximate epoch from step
        epoch_map = {100: 1.47, 150: 2.21, 200: 2.95, 250: 3.68, 272: 4.00}
        ep = epoch_map.get(step, '?')
        print(f'step-{step:<9} {ep:<8.2f} {d[\"tsr\"]:<8.3f} {d[\"step_sr\"]:<8.3f} {d[\"avg_progress\"]:<10.3f}')
    else:
        print(f'step-{step:<9} {\"--\":<8} {\"N/A\":<8} {\"N/A\":<8} {\"N/A\":<10}')
"

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
