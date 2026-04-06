#!/bin/bash
#SBATCH --job-name=eval_uis1_vidm
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_uis1_vidm_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_uis1_vidm_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ============================================================
# Evaluate UI-S1-GRPO-Trained on eval_visual_idm.py
# (step-level type_match/extract_match, standard + hindsight)
# For comparison with V9 step 40 on the same framework
# ============================================================

set -e
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="${PROJECT_DIR}/evaluation"
SCRIPT_DIR="${PROJECT_DIR}/scripts/eval"
DATASET_PATH="${EVAL_DIR}/dataset/android_control_evaluation_fixed.jsonl"
OUTPUT_DIR="${EVAL_DIR}/results/v9_eval"
MODEL_PATH="${PROJECT_DIR}/checkpoints/UI-S1-GRPO-Trained"
MODEL_NAME="UI-S1-GRPO-Trained"
MAX_SAMPLES=500

mkdir -p $OUTPUT_DIR ${SCRIPT_DIR}/logs
cd $PROJECT_DIR

echo "=========================================="
echo "Evaluating UI-S1-GRPO-Trained on Visual IDM framework"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Output: $OUTPUT_DIR"
echo "Modes: standard + hindsight"
echo "Start: $(date)"
echo "=========================================="

# Start vLLM server (image=3 for hindsight mode)
echo "Starting vLLM server..."
vllm serve "$MODEL_PATH" \
    --served-model-name $MODEL_NAME \
    --tensor-parallel-size 4 \
    --limit-mm-per-prompt image=3 \
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

# Run standard + hindsight evaluation
for MODE in standard hindsight; do
    echo "Running mode: $MODE ($(date))"
    python ${SCRIPT_DIR}/eval_visual_idm.py \
        --model_name $MODEL_NAME \
        --jsonl_file $DATASET_PATH \
        --output_dir $OUTPUT_DIR \
        --mode $MODE \
        --max_samples $MAX_SAMPLES \
        --max_workers 128
    echo "Mode $MODE complete at $(date)"
done

# Kill vLLM
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "=========================================="
echo "UI-S1-GRPO-Trained Visual IDM evaluation complete at $(date)"
echo "=========================================="

# Print summary
echo ""
echo "=== RESULTS SUMMARY ==="
for f in ${OUTPUT_DIR}/*UI-S1*summary*.json; do
    echo "--- $(basename $f) ---"
    python -c "import json; d=json.load(open('$f')); print(f'  type_match: {d.get(\"overall_type_match\",\"N/A\"):.3f}, extract_match: {d.get(\"overall_extract_match\",\"N/A\"):.3f}, n={d.get(\"n_total\",\"N/A\")}')"
done
