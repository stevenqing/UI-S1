#!/bin/bash
#SBATCH --job-name=eval_oracle_abl
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/eval_oracle_abl_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/eval_oracle_abl_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ============================================
# Analysis A: Oracle Information Ablation
# Sweep: type_only, target_only, type_target
# (full oracle and baseline already done)
# ============================================

source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
MODEL_PATH="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
MODEL_NAME="Qwen2.5-VL-7B"
DATASET_PATH="$PROJECT_ROOT/datasets/android_control_evaluation_std.jsonl"
PORT=8000

echo "=========================================="
echo "Analysis A: Oracle Information Ablation"
echo "Model: $MODEL_PATH"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p $PROJECT_ROOT/scripts/eval/ac/logs

# Start vLLM server
vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --tensor-parallel-size 4 \
    --limit-mm-per-prompt image=2 \
    --port $PORT \
    --trust-remote-code \
    &
VLLM_PID=$!

MAX_RETRIES=60
RETRY_COUNT=0
while ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; do
    sleep 10
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Waiting for vLLM server... ($RETRY_COUNT/$MAX_RETRIES)"
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: vLLM server failed to start"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
done
echo "vLLM server is ready on port $PORT!"

cd $PROJECT_ROOT/scripts/eval/ac

# Run each ablation mode sequentially
for MODE in type_only target_only type_target; do
    OUTPUT_DIR="$PROJECT_ROOT/outputs/eval_oracle_ablation_${MODE}/${MODEL_NAME}"
    mkdir -p $OUTPUT_DIR

    echo "=========================================="
    echo "Running ablation mode=${MODE} at $(date)"
    echo "Output: $OUTPUT_DIR"
    echo "=========================================="

    python eval_oracle_ablation.py \
        --model_name $MODEL_NAME \
        --jsonl_file $DATASET_PATH \
        --output_dir $OUTPUT_DIR \
        --ablation_mode $MODE \
        --n_history_image_limit 2 \
        --max_workers 4

    echo "mode=${MODE} completed at $(date)"
    echo ""
done

# Cleanup
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "=========================================="
echo "All oracle ablation evaluations completed at $(date)"
echo "=========================================="

# Print summary of all results
echo ""
echo "=== RESULTS SUMMARY ==="
echo ""

# Include existing baseline and full oracle for comparison
echo "Reference conditions:"
for COND in eval_a_ac eval_context_subtask; do
    SUMMARY="$PROJECT_ROOT/outputs/${COND}/${MODEL_NAME}/summary.json"
    if [ -f "$SUMMARY" ]; then
        echo "  ${COND}:"
        cat "$SUMMARY" | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(f'    TSR={d[\"tsr\"]:.4f} AvgProg={d[\"avg_progress\"]:.4f} StepAcc={d.get(\"step_accuracy\",\"N/A\")} (n={d[\"n\"]})')
"
    fi
done

echo ""
echo "Ablation conditions:"
for MODE in type_only target_only type_target; do
    SUMMARY="$PROJECT_ROOT/outputs/eval_oracle_ablation_${MODE}/${MODEL_NAME}/summary.json"
    if [ -f "$SUMMARY" ]; then
        echo "  ${MODE}:"
        cat "$SUMMARY" | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(f'    TSR={d[\"tsr\"]:.4f} AvgProg={d[\"avg_progress\"]:.4f} StepAcc={d.get(\"step_accuracy\",0):.4f} (n={d[\"n\"]})')
for b in sorted(d.get('length_bucket_stats',{}).keys()):
    m=d['length_bucket_stats'][b]
    print(f'      {b}: TSR={m[\"tsr\"]:.3f} StepAcc={m.get(\"step_accuracy\",0):.3f} (n={m[\"n\"]})')
"
    fi
done
