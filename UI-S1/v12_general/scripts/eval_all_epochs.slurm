#!/bin/bash
#SBATCH --job-name=v12_eval_all
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v12_general/scripts/logs/eval_all_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v12_general/scripts/logs/eval_all_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V12 Batch Eval — All epoch checkpoints
#
# For each checkpoint:
#   1) Extract cooperative weights (if HF Trainer format)
#   2) Merge r=0.5 + save correction adapters
#   3) Launch vLLM server with lossless V12 hooks
#   4) Run eval: stop-on-error mode
#   5) Run eval: no-stop mode
#   6) Kill vLLM, cleanup
#
# Checkpoints:
#   checkpoint-150 (~ep2.5)
#   checkpoint-200 (~ep3.3)
#   cooperative_final (ep4.0)
# ============================================

set -e

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
DATASET_PATH="$PROJECT_ROOT/datasets/android_control_evaluation_std.jsonl"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
mkdir -p "$PROJECT_ROOT/v12_general/scripts/logs"

# Fix image paths
FIXED_JSONL="/tmp/ac_eval_fixed_${SLURM_JOB_ID}.jsonl"
if [ ! -e /datasets ]; then
    ln -sf "$PROJECT_ROOT/datasets" /datasets 2>/dev/null || {
        sed "s|/datasets/|${PROJECT_ROOT}/datasets/|g" "$DATASET_PATH" > "$FIXED_JSONL"
        DATASET_PATH="$FIXED_JSONL"
        echo "Fixed JSONL paths -> $FIXED_JSONL"
    }
fi

echo "=========================================="
echo "V12 Batch Eval — All Epochs"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo "=========================================="

# Define checkpoints: (name, source_dir, is_hf_trainer)
declare -a CKPTS=(
    "ep2_5|checkpoints/v12_sft/checkpoint-150|hf"
    "ep3_3|checkpoints/v12_sft/checkpoint-200|hf"
    "ep4_0|checkpoints/v12_sft/cooperative_final|coop"
)

eval_one_checkpoint() {
    local CKPT_NAME="$1"
    local CKPT_DIR="$2"
    local CKPT_TYPE="$3"

    local MODEL_NAME="v12_sft_${CKPT_NAME}"
    local OUTPUT_DIR="$PROJECT_ROOT/outputs/v12_sft_${CKPT_NAME}_ac"
    local COOP_DIR
    local MERGED_MODEL="/tmp/v12_merged_${CKPT_NAME}_${SLURM_JOB_ID}"

    echo ""
    echo "=========================================="
    echo "Evaluating: $CKPT_NAME ($CKPT_TYPE)"
    echo "  Source: $CKPT_DIR"
    echo "  Output: $OUTPUT_DIR"
    echo "  Time:   $(date)"
    echo "=========================================="

    mkdir -p "$OUTPUT_DIR"

    # Step 1: Get cooperative weights
    if [ "$CKPT_TYPE" = "hf" ]; then
        COOP_DIR="/tmp/v12_coop_${CKPT_NAME}_${SLURM_JOB_ID}"
        echo "  Extracting cooperative weights from HF checkpoint..."
        python v12_general/extract_cooperative_ckpt.py \
            --hf_checkpoint "$CKPT_DIR" \
            --output_dir "$COOP_DIR"
    else
        COOP_DIR="$CKPT_DIR"
    fi

    # Step 2: Merge + save corrections
    echo "  Merging r=0.5 + corrections..."
    python v12_general/merge_to_hf.py \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$COOP_DIR" \
        --output_dir "$MERGED_MODEL"

    # Step 3: Start vLLM server
    echo "  Starting vLLM server..."
    python v12_general/serve_v12_vllm.py \
        --model "$MERGED_MODEL" \
        --served-model-name "$MODEL_NAME" \
        --port 8000 \
        --tensor-parallel-size 4 \
        --max-model-len 32768 \
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
        echo "  ERROR: vLLM server failed to start"
        kill $VLLM_PID 2>/dev/null || true
        rm -rf "$MERGED_MODEL" "$COOP_DIR" 2>/dev/null
        return 1
    fi

    # Step 4: Eval — stop-on-error
    echo "  Running eval (stop-on-error)..."
    python v12_general/eval_ac_vllm.py \
        --model_name "$MODEL_NAME" \
        --jsonl_file "$DATASET_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --n_history_image_limit 2 \
        --max_workers 128

    # Step 5: Eval — no-stop
    echo "  Running eval (no-stop)..."
    python v12_general/eval_ac_vllm.py \
        --model_name "$MODEL_NAME" \
        --jsonl_file "$DATASET_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --n_history_image_limit 2 \
        --max_workers 128 \
        --no_stop

    # Step 6: Cleanup
    echo "  Shutting down vLLM..."
    kill $VLLM_PID 2>/dev/null || true
    pkill -f "serve_v12_vllm" 2>/dev/null || true
    wait $VLLM_PID 2>/dev/null || true
    rm -rf "$MERGED_MODEL"
    [ "$CKPT_TYPE" = "hf" ] && rm -rf "$COOP_DIR"

    echo "  Done: $CKPT_NAME"
}

# Run all checkpoints sequentially
for entry in "${CKPTS[@]}"; do
    IFS='|' read -r NAME DIR TYPE <<< "$entry"
    eval_one_checkpoint "$NAME" "$DIR" "$TYPE" || true
done

# Cleanup
rm -f "$FIXED_JSONL" 2>/dev/null

# Print summary of all results
echo ""
echo "=========================================="
echo "ALL EVALUATIONS COMPLETE"
echo "=========================================="
for entry in "${CKPTS[@]}"; do
    IFS='|' read -r NAME DIR TYPE <<< "$entry"
    OUTPUT_DIR="$PROJECT_ROOT/outputs/v12_sft_${NAME}_ac"
    MODEL_NAME="v12_sft_${NAME}"
    echo ""
    echo "--- $NAME ---"
    for mode in stop nostop; do
        SUMMARY="$OUTPUT_DIR/${MODEL_NAME}_${mode}_summary.json"
        if [ -f "$SUMMARY" ]; then
            python -c "
import json
with open('$SUMMARY') as f:
    s = json.load(f)
print(f'  [{s[\"mode\"]:6s}] TSR={s[\"tsr\"]:.4f} ({s[\"success_count\"]}/{s[\"total_episodes\"]}) '
      f'SeqProg={s[\"avg_sequential_progress\"]:.4f} '
      f'Scattered={s[\"avg_scattered_progress\"]:.4f} '
      f'({s[\"elapsed_seconds\"]:.0f}s)')
"
        else
            echo "  [$mode] No results"
        fi
    done
done

echo ""
echo "Done: $(date)"
echo "=========================================="
