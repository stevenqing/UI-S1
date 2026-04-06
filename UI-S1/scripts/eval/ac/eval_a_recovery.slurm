#!/bin/bash
#SBATCH --job-name=eval_a_recovery
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/eval_a_recovery_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/eval_a_recovery_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
MODEL_PATH="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
MODEL_NAME="Qwen2.5-VL-7B"
DATASET_PATH="$PROJECT_ROOT/datasets/android_control_evaluation_std.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/outputs/eval_a_ac/${MODEL_NAME}"

echo "=========================================="
echo "Eval A Recovery: Processing missing episodes"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p $OUTPUT_DIR
mkdir -p $PROJECT_ROOT/scripts/eval/ac/logs

# Start vLLM server
vllm serve $MODEL_PATH \
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
    echo "Waiting for vLLM server... ($RETRY_COUNT/$MAX_RETRIES)"
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: vLLM server failed to start"
        kill $VLLM_PID 2>/dev/null
        exit 1
    fi
done
echo "vLLM server is ready!"

cd $PROJECT_ROOT/scripts/eval/ac

python eval_a_recovery.py \
    --model_name $MODEL_NAME \
    --jsonl_file $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --n_history_image_limit 2 \
    --max_workers 4

EVAL_EXIT=$?

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

# Run all Phase 1 offline analyses on complete results
if [ $EVAL_EXIT -eq 0 ] && [ -f "$OUTPUT_DIR/trajectory_results.jsonl" ]; then
    echo "Running offline analyses..."
    RESULTS="$OUTPUT_DIR/trajectory_results.jsonl"
    OFFLINE="$OUTPUT_DIR/offline"

    python eval_b_step_analysis.py --input_file $RESULTS --output_dir $OFFLINE
    python eval_c_hard_cases.py --input_file $RESULTS --output_dir $OFFLINE
    python eval_c2_action_type_analysis.py --input_file $RESULTS --output_dir $OFFLINE
    python eval_c8_verifier_trigger.py --input_file $RESULTS --output_dir $OFFLINE
    python eval_d0_error_ceiling.py --input_file $RESULTS --output_dir $OFFLINE
    python eval_d10_action_diagnosis.py --input_file $RESULTS --output_dir $OFFLINE
    python eval_pretest3_counterfactual.py --input_file $RESULTS --output_dir $OFFLINE
    python eval_pretest1_shapley.py --baseline_file $RESULTS --output_dir $OFFLINE
    python eval_d4_d6_d7_offline.py --baseline_file $RESULTS --output_dir $OFFLINE

    echo "All offline analyses done."
fi

echo "=========================================="
echo "Eval A Recovery completed at $(date). Exit: $EVAL_EXIT"
echo "=========================================="
exit $EVAL_EXIT
