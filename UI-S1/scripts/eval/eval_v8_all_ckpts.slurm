#!/bin/bash
#SBATCH --job-name=eval_v8_all
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_v8_all_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_v8_all_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Evaluate ALL V8 checkpoints (40,50,70,80)
# Keep only the best checkpoint after eval
# vLLM with --max_workers 128 for fastest eval
# ============================================

set -e

source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="${PROJECT_DIR}/evaluation"
V8_CKPT_BASE="${PROJECT_DIR}/checkpoints/sp_gigpo_ac/sp_gigpo_spwa_k8_v8"
DATASET_PATH="${EVAL_DIR}/dataset/android_control_evaluation_fixed.jsonl"
OUTPUT_DIR="${EVAL_DIR}/results"
RESULTS_SUMMARY="${OUTPUT_DIR}/v8_eval_summary.txt"

STEPS="70 80"

cd $PROJECT_DIR
mkdir -p $OUTPUT_DIR /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs

echo "=========================================="
echo "V8 Full Checkpoint Evaluation"
echo "=========================================="
echo "Steps to evaluate: $STEPS"
echo "Dataset: $DATASET_PATH ($(wc -l < $DATASET_PATH) samples)"
echo "Start time: $(date)"
echo "=========================================="

# ============================================
# Step 1: Merge ALL checkpoints first
# ============================================
echo ""
echo "===== PHASE 1: Merge FSDP checkpoints ====="
for STEP in $STEPS; do
    CKPT_DIR="${V8_CKPT_BASE}/global_step_${STEP}/actor"
    MERGED_DIR="${V8_CKPT_BASE}/merged_step_${STEP}"

    if [ -d "$MERGED_DIR" ] && [ -f "$MERGED_DIR/config.json" ]; then
        echo "[step_${STEP}] Already merged, skipping."
        continue
    fi

    if [ ! -d "$CKPT_DIR" ]; then
        echo "[step_${STEP}] WARNING: Checkpoint dir not found: $CKPT_DIR, skipping."
        continue
    fi

    echo "[step_${STEP}] Merging FSDP -> HuggingFace..."
    python ${PROJECT_DIR}/scripts/model_merger.py merge \
        --backend fsdp \
        --local_dir "$CKPT_DIR" \
        --target_dir "$MERGED_DIR"

    if [ $? -ne 0 ]; then
        echo "[step_${STEP}] ERROR: Merge failed!"
        continue
    fi
    echo "[step_${STEP}] Merge done."
done

echo ""
echo "===== PHASE 2: Evaluate each checkpoint ====="

# Track best
BEST_STEP=""
BEST_ACC="-1"
> "$RESULTS_SUMMARY"

for STEP in $STEPS; do
    MERGED_DIR="${V8_CKPT_BASE}/merged_step_${STEP}"
    MODEL_NAME="sp_gigpo_v8_step${STEP}"

    if [ ! -d "$MERGED_DIR" ] || [ ! -f "$MERGED_DIR/config.json" ]; then
        echo "[step_${STEP}] No merged model, skipping eval."
        continue
    fi

    echo ""
    echo "=========================================="
    echo "[step_${STEP}] Starting vLLM server..."
    echo "=========================================="

    # Start vLLM server
    vllm serve "$MERGED_DIR" \
        --served-model-name $MODEL_NAME \
        --tensor-parallel-size 4 \
        --limit-mm-per-prompt image=2 \
        --port 8000 \
        --trust-remote-code \
        --disable-log-requests \
        &

    VLLM_PID=$!

    # Wait for server
    MAX_RETRIES=60
    RETRY_COUNT=0
    while ! curl -s http://localhost:8000/health > /dev/null 2>&1; do
        sleep 10
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
            echo "[step_${STEP}] ERROR: vLLM server failed to start"
            kill $VLLM_PID 2>/dev/null
            continue 2
        fi
    done
    echo "[step_${STEP}] vLLM server ready!"

    # Run evaluation with 128 workers
    echo "[step_${STEP}] Running evaluation (128 workers)..."
    EVAL_START=$(date +%s)

    cd $EVAL_DIR
    python eval_qwenvl.py \
        --model_name $MODEL_NAME \
        --jsonl_file $DATASET_PATH \
        --output_dir $OUTPUT_DIR \
        --n_history_image_limit 2 \
        --max_workers 128

    EVAL_EXIT=$?
    EVAL_END=$(date +%s)
    EVAL_TIME=$((EVAL_END - EVAL_START))
    cd $PROJECT_DIR

    # Kill vLLM server
    kill $VLLM_PID 2>/dev/null
    wait $VLLM_PID 2>/dev/null
    sleep 5  # let port release

    if [ $EVAL_EXIT -ne 0 ]; then
        echo "[step_${STEP}] Evaluation failed (exit=$EVAL_EXIT)"
        echo "step_${STEP}: FAILED" >> "$RESULTS_SUMMARY"
        continue
    fi

    # Parse results
    RESULT_FILE="${OUTPUT_DIR}/${MODEL_NAME}.jsonl"
    if [ -f "$RESULT_FILE" ]; then
        EVAL_METRICS=$(python3 -c "
import json
results = [json.loads(l) for l in open('$RESULT_FILE')]
total = len(results)
correct = sum(1 for r in results if r.get('correct', False) or r.get('task_success', False) or r.get('match', False))
acc = correct / total if total > 0 else 0

# Try to get more detailed metrics
type_matches = sum(1 for r in results if r.get('type_match', False))
extract_matches = sum(1 for r in results if r.get('extract_match', False))

print(f'{acc:.4f}|{correct}|{total}|{type_matches}|{extract_matches}')
" 2>/dev/null || echo "PARSE_ERROR")

        if [ "$EVAL_METRICS" != "PARSE_ERROR" ]; then
            ACC=$(echo $EVAL_METRICS | cut -d'|' -f1)
            CORRECT=$(echo $EVAL_METRICS | cut -d'|' -f2)
            TOTAL=$(echo $EVAL_METRICS | cut -d'|' -f3)
            TYPE_M=$(echo $EVAL_METRICS | cut -d'|' -f4)
            EXTRACT_M=$(echo $EVAL_METRICS | cut -d'|' -f5)
            echo "[step_${STEP}] acc=${ACC} (${CORRECT}/${TOTAL}) type_match=${TYPE_M} extract_match=${EXTRACT_M} time=${EVAL_TIME}s"
            echo "step_${STEP}: acc=${ACC} correct=${CORRECT}/${TOTAL} type_match=${TYPE_M} extract_match=${EXTRACT_M} time=${EVAL_TIME}s" >> "$RESULTS_SUMMARY"

            # Track best
            IS_BETTER=$(python3 -c "print(1 if float('$ACC') > float('$BEST_ACC') else 0)")
            if [ "$IS_BETTER" -eq 1 ]; then
                BEST_ACC=$ACC
                BEST_STEP=$STEP
            fi
        else
            echo "[step_${STEP}] Could not parse results from $RESULT_FILE"
            echo "step_${STEP}: PARSE_ERROR" >> "$RESULTS_SUMMARY"
        fi
    else
        echo "[step_${STEP}] No result file found"
        echo "step_${STEP}: NO_RESULT" >> "$RESULTS_SUMMARY"
    fi
done

# ============================================
# Step 3: Report and cleanup
# ============================================
echo ""
echo "=========================================="
echo "EVALUATION COMPLETE"
echo "=========================================="
cat "$RESULTS_SUMMARY"
echo ""
echo "BEST CHECKPOINT: step_${BEST_STEP} (acc=${BEST_ACC})"
echo "=========================================="

# Cleanup: remove non-best merged checkpoints (keep FSDP originals)
if [ -n "$BEST_STEP" ]; then
    echo ""
    echo "Cleaning up non-best merged checkpoints..."
    for STEP in $STEPS; do
        if [ "$STEP" != "$BEST_STEP" ]; then
            MERGED_DIR="${V8_CKPT_BASE}/merged_step_${STEP}"
            if [ -d "$MERGED_DIR" ]; then
                echo "  Removing merged_step_${STEP}..."
                rm -rf "$MERGED_DIR"
            fi
        fi
    done
    echo "Kept: merged_step_${BEST_STEP} (best)"
fi

echo ""
echo "Done at $(date)"
