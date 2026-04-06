#!/bin/bash
#SBATCH --job-name=ctx_sub_v2
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/eval_ctx_subtask_v2_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/eval_ctx_subtask_v2_%j.err
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
MODEL_PATH="$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_full_sft_v2"
MODEL_NAME="gui360_full_sft_v2"
DATASET_PATH="$PROJECT_ROOT/datasets/android_control_evaluation_std.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/outputs/eval_context_subtask/${MODEL_NAME}"
PORT=8000

echo "=========================================="
echo "Eval: Context Subtask - AndroidControl (SFT v2)"
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
    --port $PORT \
    --trust-remote-code \
    &
VLLM_PID=$!

# Wait for server
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

python eval_context_subtask.py \
    --model_name $MODEL_NAME \
    --jsonl_file $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --n_history_image_limit 2 \
    --max_workers 4

EVAL_EXIT=$?

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "=========================================="
echo "Context Subtask v2 eval completed at $(date). Exit: $EVAL_EXIT"
echo "=========================================="
exit $EVAL_EXIT
