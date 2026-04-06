#!/bin/bash
#SBATCH --job-name=eval_coop_ap
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_thought_v1_ap_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_thought_v1_ap_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
EVAL_DIR="$TRAIN_DIR/GUI-360-eval"
DATASET_ROOT="$PROJECT_DIR/datasets/GUI-360/test"
MERGED_MODEL="$PROJECT_DIR/checkpoints/cooperative_thought_v1_merged"
RESULTS_DIR="$EVAL_DIR/results/cooperative_thought_v1"
MODEL_TYPE="qwen2.5_vl_7b"
VLLM_PORT=19820
VLLM_HOST="localhost"
EVAL_THREADS=64

echo "=========================================="
echo "Cooperative Thought v1 - Action Prediction"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Threads: $EVAL_THREADS"
echo "Start: $(date)"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 3

python -m vllm.entrypoints.openai.api_server \
    --model "$MERGED_MODEL" \
    --port $VLLM_PORT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee "$PROJECT_DIR/scripts/exp_cooperative/logs/vllm_ap_${SLURM_JOB_ID}.log" &

VLLM_PID=$!

MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://${VLLM_HOST}:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "vLLM server ready!"
        break
    fi
    sleep 5
    WAIT_COUNT=$((WAIT_COUNT + 5))
done

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

mkdir -p "$RESULTS_DIR"
cd "$EVAL_DIR"

echo "Running action_prediction with $EVAL_THREADS threads..."
python evaluation.py \
    --root_dir "$DATASET_ROOT" \
    --type action_prediction \
    --model_type "$MODEL_TYPE" \
    --model_name "$MERGED_MODEL" \
    --api_url "http://${VLLM_HOST}:${VLLM_PORT}/v1" \
    --threads $EVAL_THREADS \
    --output_dir "${RESULTS_DIR}/action_prediction"

pkill -f "vllm.entrypoints.openai.api_server" || true

echo "=========================================="
echo "Results:"
cat $RESULTS_DIR/action_prediction/evaluation_summary_*.json 2>/dev/null || echo "(no summary)"
echo ""
echo "Done: $(date)"
echo "=========================================="
