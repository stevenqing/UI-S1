#!/bin/bash
#SBATCH --job-name=eval_e4_replan
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/eval_e4_replan_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/eval_e4_replan_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ============================================
# E4: Midpoint Replanning Evaluation
# Supports both AC and GUI-360 via DATASET env
# Usage:
#   DATASET=ac    sbatch eval_e4_midpoint_replan.slurm
#   DATASET=gui360 sbatch eval_e4_midpoint_replan.slurm
# ============================================

DATASET="${DATASET:-ac}"
PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
SCRIPT_DIR="$PROJECT_ROOT/scripts/eval/ac"
EVAL_DIR="$PROJECT_ROOT/train_GUI_360/GUI-360-eval"

# Environment
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

# Dataset-specific config
if [ "$DATASET" = "ac" ]; then
    MODEL_PATH="$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct"
    MODEL_NAME="Qwen2.5-VL-7B"
    PORT=8000
    DATASET_ARGS="--jsonl_file $PROJECT_ROOT/datasets/android_control_evaluation_std.jsonl"
    WORKDIR="$SCRIPT_DIR"
    VLLM_EXTRA="--limit-mm-per-prompt image=2"
    MAX_WORKERS=128
else
    MODEL_PATH="$PROJECT_ROOT/train_GUI_360/llamafactory/output/gui360_full_sft_v3_grounding"
    MODEL_NAME="$MODEL_PATH"
    PORT=19815
    DATASET_ARGS="--root_dir $PROJECT_ROOT/datasets/GUI-360/test --api_url http://localhost:$PORT/v1"
    WORKDIR="$EVAL_DIR"
    VLLM_EXTRA="--max-model-len 32768 --gpu-memory-utilization 0.9"
    MAX_WORKERS=64
fi

OUTPUT_DIR="$PROJECT_ROOT/outputs/eval_e4_midpoint_replan/$DATASET"

echo "=========================================="
echo "E4: Midpoint Replanning Evaluation"
echo "Dataset: $DATASET"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p $OUTPUT_DIR
mkdir -p $SCRIPT_DIR/logs

# Start vLLM server
vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --tensor-parallel-size 4 \
    --port $PORT \
    --trust-remote-code \
    --disable-log-requests \
    --max-num-seqs 256 \
    $VLLM_EXTRA \
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

cd $WORKDIR

if [ "$DATASET" = "ac" ]; then
    python eval_e4_midpoint_replan.py \
        --dataset ac \
        --model_name $MODEL_NAME \
        --output_dir $OUTPUT_DIR \
        --max_workers $MAX_WORKERS \
        $DATASET_ARGS
else
    python $SCRIPT_DIR/eval_e4_midpoint_replan.py \
        --dataset gui360 \
        --model_name $MODEL_NAME \
        --output_dir $OUTPUT_DIR \
        --max_workers $MAX_WORKERS \
        $DATASET_ARGS
fi

EVAL_EXIT=$?

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "=========================================="
echo "E4 eval completed at $(date). Exit: $EVAL_EXIT"
echo "=========================================="
exit $EVAL_EXIT
