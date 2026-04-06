#!/bin/bash
#SBATCH --job-name=eval_ba_ck600
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_ba_gui360_ck600_%j.out
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_ba_gui360_ck600_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
EVAL_DIR="$PROJECT_DIR/train_GUI_360/GUI-360-eval"
MERGED_MODEL="$PROJECT_DIR/train_GUI_360/llamafactory/output/bind_aux_r32/checkpoint-600_merged"
DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"
MODEL_TYPE="qwen2.5_vl_7b"
VLLM_PORT=19814
EVAL_THREADS=5

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$EVAL_DIR/results/bind_aux_r32_ck600_${TIMESTAMP}"

echo "GUI-360 Eval: bind_aux_r32 ck600 | Job ${SLURM_JOB_ID} | $(hostname) | $(date)"

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export PYTHONUNBUFFERED=1

if [ -f "$PROJECT_DIR/train/env_config.sh" ]; then
    source "$PROJECT_DIR/train/env_config.sh"
fi

# Check merged model exists
if [ ! -f "$MERGED_MODEL/model.safetensors.index.json" ]; then
    echo "ERROR: Merged model not found at $MERGED_MODEL"
    echo "Run merge_bind_aux_all.slurm first!"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model $MERGED_MODEL \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $PROJECT_DIR/scripts/eval/logs/vllm_ba_ck600_${SLURM_JOB_ID}.log &

VLLM_PID=$!
MAX_WAIT=600; WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    curl -s http://localhost:${VLLM_PORT}/health > /dev/null 2>&1 && echo "vLLM ready!" && break
    sleep 5; WAIT_COUNT=$((WAIT_COUNT + 5))
done
[ $WAIT_COUNT -ge $MAX_WAIT ] && echo "ERROR: vLLM timeout" && kill $VLLM_PID 2>/dev/null && exit 1

cd $EVAL_DIR
API_URL="http://localhost:${VLLM_PORT}/v1"

for EVAL_TYPE in grounding action_prediction action_prediction_a11y; do
    OUTPUT="$RESULTS_DIR/${EVAL_TYPE}_${TIMESTAMP}"
    mkdir -p $OUTPUT
    echo "Running $EVAL_TYPE..."
    python evaluation.py --root_dir $DATASET_ROOT --type $EVAL_TYPE --model_type $MODEL_TYPE --model_name $MERGED_MODEL --api_url $API_URL --threads $EVAL_THREADS --output_dir $OUTPUT
done

kill $VLLM_PID 2>/dev/null || true
echo "Done at $(date) | Results: $RESULTS_DIR"
cat $RESULTS_DIR/*/evaluation_summary_*.json 2>/dev/null || true
