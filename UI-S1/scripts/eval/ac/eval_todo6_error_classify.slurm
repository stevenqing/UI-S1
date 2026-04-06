#!/bin/bash
#SBATCH --job-name=eval_todo6
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/eval_todo6_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/eval_todo6_%j.err
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
TRAJ_RESULTS="$PROJECT_ROOT/outputs/eval_a_ac/${MODEL_NAME}/trajectory_results.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/outputs/eval_todo6_error_classify/${MODEL_NAME}"

echo "=========================================="
echo "Eval TODO 6: LLM Error Classification"
echo "Model: $MODEL_PATH"
echo "Trajectory results: $TRAJ_RESULTS"
echo "Output: $OUTPUT_DIR"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p $OUTPUT_DIR
mkdir -p $PROJECT_ROOT/scripts/eval/ac/logs

# Check input file exists
if [ ! -f "$TRAJ_RESULTS" ]; then
    echo "ERROR: Trajectory results not found: $TRAJ_RESULTS"
    exit 1
fi

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

python eval_todo6_llm_error_classify.py \
    --model_name $MODEL_NAME \
    --trajectory_results $TRAJ_RESULTS \
    --output_dir $OUTPUT_DIR \
    --max_workers 4

EVAL_EXIT=$?

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "=========================================="
echo "TODO 6 Error Classification completed at $(date). Exit: $EVAL_EXIT"
echo "=========================================="
exit $EVAL_EXIT
