#!/bin/bash
#SBATCH --job-name=eval_v8_%a
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_v8_step_%a_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_v8_step_%a_%j.log
#SBATCH --time=03:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# Usage: sbatch --array=90,100,110,120,130,140 eval_v8_single_step.slurm
# Each array task evaluates one checkpoint step

set -e

STEP=${SLURM_ARRAY_TASK_ID}
echo "=== Evaluating V8 step_${STEP} (Job ${SLURM_JOB_ID}, Array ${SLURM_ARRAY_TASK_ID}) ==="
echo "Start: $(date)"

source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="${PROJECT_DIR}/evaluation"
V8_CKPT_BASE="${PROJECT_DIR}/checkpoints/sp_gigpo_ac/sp_gigpo_spwa_k8_v8"
DATASET_PATH="${EVAL_DIR}/dataset/android_control_evaluation_fixed.jsonl"
OUTPUT_DIR="${EVAL_DIR}/results"

CKPT_DIR="${V8_CKPT_BASE}/global_step_${STEP}/actor"
MERGED_DIR="${V8_CKPT_BASE}/merged_step_${STEP}"
MODEL_NAME="sp_gigpo_v8_step${STEP}"

# Each job runs on its own node, so port 8000 is fine
PORT=8000

cd $PROJECT_DIR

# Phase 1: Merge
if [ -d "$MERGED_DIR" ] && [ -f "$MERGED_DIR/config.json" ]; then
    echo "[step_${STEP}] Already merged, skipping."
else
    if [ ! -d "$CKPT_DIR" ]; then
        echo "[step_${STEP}] ERROR: Checkpoint not found: $CKPT_DIR"
        exit 1
    fi
    echo "[step_${STEP}] Merging FSDP -> HuggingFace..."
    python ${PROJECT_DIR}/scripts/model_merger.py merge \
        --backend fsdp \
        --local_dir "$CKPT_DIR" \
        --target_dir "$MERGED_DIR"
    echo "[step_${STEP}] Merge done."
fi

# Phase 2: Start vLLM
echo "[step_${STEP}] Starting vLLM on port ${PORT}..."
vllm serve "$MERGED_DIR" \
    --served-model-name $MODEL_NAME \
    --tensor-parallel-size 4 \
    --limit-mm-per-prompt image=2 \
    --port $PORT \
    --trust-remote-code \
    --disable-log-requests \
    &
VLLM_PID=$!

MAX_RETRIES=60
RETRY_COUNT=0
while ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; do
    sleep 10
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "[step_${STEP}] ERROR: vLLM failed to start after 10min"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
done
echo "[step_${STEP}] vLLM ready!"

# Phase 3: Evaluate
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

# Kill vLLM
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

if [ $EVAL_EXIT -ne 0 ]; then
    echo "[step_${STEP}] FAILED (exit=$EVAL_EXIT)"
    exit 1
fi

# Parse results
RESULT_FILE="${OUTPUT_DIR}/${MODEL_NAME}.jsonl"
if [ -f "$RESULT_FILE" ]; then
    python3 -c "
import json
results = [json.loads(l) for l in open('$RESULT_FILE')]
total = len(results)
correct = sum(1 for r in results if r.get('correct', False) or r.get('task_success', False) or r.get('match', False))
acc = correct / total if total > 0 else 0
print(f'[step_${STEP}] acc={acc:.4f} ({correct}/{total}) time=${EVAL_TIME}s')
# Write summary line
with open('${OUTPUT_DIR}/v8_parallel_eval.txt', 'a') as f:
    f.write(f'step_${STEP}: acc={acc:.4f} correct={correct}/{total} time=${EVAL_TIME}s\n')
"
else
    echo "[step_${STEP}] No result file found!"
    exit 1
fi

echo "[step_${STEP}] Done at $(date)"
