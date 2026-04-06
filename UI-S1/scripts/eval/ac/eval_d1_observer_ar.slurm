#!/bin/bash
#SBATCH --job-name=eval_d1_ac
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/eval_d1_ac_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/eval_d1_ac_%j.err
#SBATCH --time=24:00:00
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
MODEL_PATH="${MODEL_PATH:-$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct}"
MODEL_NAME="${MODEL_NAME:-Qwen2.5-VL-7B}"
OUTPUT_DIR="$PROJECT_ROOT/outputs/eval_d1_ac/${MODEL_NAME}"

mkdir -p $OUTPUT_DIR
mkdir -p $PROJECT_ROOT/scripts/eval/ac/logs

vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --tensor-parallel-size 4 \
    --limit-mm-per-prompt image=2 \
    --port 8000 \
    --trust-remote-code \
    &
VLLM_PID=$!

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
echo "vLLM server is ready!"

cd $PROJECT_ROOT/scripts/eval/ac

python eval_d1_observer_ar.py \
    --model_name $MODEL_NAME \
    --jsonl_file $PROJECT_ROOT/datasets/android_control_evaluation_std.jsonl \
    --output_dir $OUTPUT_DIR \
    --n_history_image_limit 2 \
    --max_workers 4

EVAL_EXIT=$?

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

# Run downstream offline analyses (need both Eval A and D1 results)
EVAL_A_RESULTS="$PROJECT_ROOT/outputs/eval_a_ac/${MODEL_NAME}/trajectory_results.jsonl"
D1_RESULTS="$OUTPUT_DIR/observer_results.jsonl"

if [ $EVAL_EXIT -eq 0 ] && [ -f "$D1_RESULTS" ]; then
    echo "Running D1 downstream analyses..."
    OFFLINE="$OUTPUT_DIR/offline"

    # E2: Observer value by length (needs Eval A + D1)
    if [ -f "$EVAL_A_RESULTS" ]; then
        python eval_e2_observer_length.py --baseline_file $EVAL_A_RESULTS --observer_file $D1_RESULTS --output_dir $OFFLINE
        python eval_d4_d6_d7_offline.py --baseline_file $EVAL_A_RESULTS --observer_file $D1_RESULTS --output_dir $OFFLINE
        python eval_pretest1_shapley.py --baseline_file $EVAL_A_RESULTS --observer_file $D1_RESULTS --output_dir $OFFLINE
    fi

    echo "D1 downstream analyses done."
fi

echo "=========================================="
echo "D1 Observer completed at $(date). Exit: $EVAL_EXIT"
echo "=========================================="
exit $EVAL_EXIT
