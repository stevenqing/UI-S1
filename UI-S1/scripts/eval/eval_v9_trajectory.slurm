#!/bin/bash
#SBATCH --job-name=eval_v9_traj
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_v9_traj_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_v9_traj_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ============================================================
# Evaluate V9 step 40 on trajectory-level AR eval (eval_qwenvl.py)
# 1543 trajectories, stop-on-error, TSR metric
# For comparison with UI-S1 (16.59%) and V8 (15.36%)
# ============================================================

set -e
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="${PROJECT_DIR}/evaluation"
SCRIPT_DIR="${PROJECT_DIR}/scripts/eval"
DATASET_PATH="${EVAL_DIR}/dataset/android_control_evaluation_fixed.jsonl"
OUTPUT_DIR="${EVAL_DIR}/results"
MODEL_PATH="${PROJECT_DIR}/checkpoints/sp_gigpo_ac/sp_gigpo_spwa_k8_v9/merged_step40"
MODEL_NAME="v9_step40_traj"

mkdir -p $OUTPUT_DIR ${SCRIPT_DIR}/logs
cd $EVAL_DIR

echo "=========================================="
echo "Evaluating V9 Step 40 — Trajectory-Level AR Eval"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH (1543 trajectories)"
echo "Mode: stop-on-error (autoregressive)"
echo "Start: $(date)"
echo "=========================================="

# Start vLLM server
echo "Starting vLLM server..."
vllm serve "$MODEL_PATH" \
    --served-model-name $MODEL_NAME \
    --tensor-parallel-size 4 \
    --limit-mm-per-prompt image=2 \
    --port 8000 \
    --trust-remote-code \
    &
VLLM_PID=$!

# Wait for server
MAX_RETRIES=60
RETRY_COUNT=0
while ! curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 10
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: vLLM server failed to start"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
done
echo "vLLM server ready!"

# Run trajectory-level evaluation (1543 trajectories, stop-on-error)
echo "Starting trajectory-level evaluation ($(date))..."
python ${EVAL_DIR}/eval_qwenvl.py \
    --model_name $MODEL_NAME \
    --jsonl_file $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --n_history_image_limit 2 \
    --max_workers 128

EVAL_EXIT_CODE=$?

# Kill vLLM
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "=========================================="
echo "V9 Step 40 trajectory evaluation complete at $(date)"
echo "Exit code: $EVAL_EXIT_CODE"
echo "Results: $OUTPUT_DIR/${MODEL_NAME}.jsonl"
echo "=========================================="

# Print summary
if [ -f "$OUTPUT_DIR/${MODEL_NAME}.jsonl" ]; then
    TOTAL=$(wc -l < "$OUTPUT_DIR/${MODEL_NAME}.jsonl")
    SUCCESS=$(grep -c '"task_success": true' "$OUTPUT_DIR/${MODEL_NAME}.jsonl" || echo 0)
    echo ""
    echo "=== RESULTS SUMMARY ==="
    echo "Total tasks: $TOTAL"
    echo "Successful: $SUCCESS"
    python3 -c "print(f'TSR: {${SUCCESS}/${TOTAL}*100:.2f}%')" 2>/dev/null || true
fi

exit $EVAL_EXIT_CODE
