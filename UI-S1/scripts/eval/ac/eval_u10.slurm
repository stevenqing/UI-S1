#!/bin/bash
#SBATCH --job-name=eval_u10_ac
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/eval_u10_ac_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/logs/eval_u10_ac_%j.err
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
OUTPUT_DIR="$PROJECT_ROOT/outputs/eval_u10_ac/${MODEL_NAME}"

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

python eval_u10_history_verifier.py \
    --model_name $MODEL_NAME \
    --jsonl_file $PROJECT_ROOT/datasets/android_control_evaluation_std.jsonl \
    --output_dir $OUTPUT_DIR \
    --n_history_image_limit 2 \
    --max_workers 48 \
    --K_probe 3 --K_resample 5 \
    --probe_temperature 0.6 --resample_temperature 0.6 \
    --resume

EVAL_EXIT=$?

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "=========================================="
echo "U10 History-Aware Verifier completed at $(date). Exit: $EVAL_EXIT"
echo "Pre-registered prediction: TSR=16.61%, delta=+0.54pp over baseline"
echo "=========================================="
exit $EVAL_EXIT
