#!/bin/bash
#SBATCH --job-name=mv_ac_A
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/multiview_ac_A_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/multiview_ac_A_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

set -e
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="$PROJECT_DIR/evaluation"
LOG_DIR="$PROJECT_DIR/scripts/eval/logs"
AC_JSONL="$PROJECT_DIR/evaluation/dataset/android_control_evaluation_fixed.jsonl"
MODEL_PATH="$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct"
MODEL_NAME="$MODEL_PATH"
VLLM_PORT=8000
API_URL="http://localhost:${VLLM_PORT}/v1"
VARIANT="A"
WORKERS=256
OUTPUT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/results/multiview_ac_variants_v2_20260326_105231"

echo "=== AC Variant $VARIANT | Job ${SLURM_JOB_ID} | $(hostname) | $(date) ==="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export TMPDIR=/tmp/${SLURM_JOB_ID}
mkdir -p $TMPDIR
export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor
export TORCH_HOME=$TMPDIR/torch
export XDG_CACHE_HOME=$TMPDIR/cache
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TORCH_HOME $XDG_CACHE_HOME
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" --port $VLLM_PORT \
    --tensor-parallel-size 4 --max-model-len 32768 \
    --trust-remote-code --gpu-memory-utilization 0.9 \
    2>&1 | tee "$LOG_DIR/vllm_ac_${VARIANT}_${SLURM_JOB_ID}.log" &
VLLM_PID=$!

MAX_WAIT=600; WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    curl -s "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1 && echo "vLLM ready!" && break
    sleep 5; WAIT_COUNT=$((WAIT_COUNT + 5))
    echo "Waiting for vLLM... ($WAIT_COUNT/$MAX_WAIT)"
done
[ $WAIT_COUNT -ge $MAX_WAIT ] && echo "ERROR: vLLM failed" && kill $VLLM_PID 2>/dev/null && exit 1

cd "$EVAL_DIR"
python eval_multiview.py \
    --jsonl_file "$AC_JSONL" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --mode multiview \
    --variant $VARIANT \
    --api_url "$API_URL" \
    --max_workers $WORKERS \
    --n_history_image_limit 2 \
    --save_debug

kill $VLLM_PID 2>/dev/null || true
echo "=== Variant $VARIANT Complete | $(date) ==="
