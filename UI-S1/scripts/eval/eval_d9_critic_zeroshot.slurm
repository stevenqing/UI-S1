#!/bin/bash
#SBATCH --job-name=eval_d9
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_d9_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_d9_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:2

# ============================================
# Eval D9: Critic Zero-Shot Performance
# V2 on GPUs 0-1 (TP=2): Critic
# No V3 needed
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
LOG_DIR="$PROJECT_DIR/scripts/eval/logs"

echo "=========================================="
echo "Eval D9: Critic Zero-Shot Performance"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',')"
echo "Start time: $(date)"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

mkdir -p $LOG_DIR
cd $PROJECT_DIR

V2_PORT=19816

V2_MODEL="$PROJECT_DIR/train_GUI_360/llamafactory/output/gui360_full_sft_v2"

echo "Starting V2 vLLM on GPUs 0,1 (port $V2_PORT)..."
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model "$V2_MODEL" \
    --served-model-name sft_v2 \
    --port $V2_PORT \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    --dtype auto \
    &> $LOG_DIR/vllm_v2_d9_${SLURM_JOB_ID}.log &
V2_PID=$!

echo "Waiting for V2 vLLM to start..."
for i in $(seq 1 300); do
    if curl -s http://localhost:$V2_PORT/v1/models > /dev/null 2>&1; then
        echo "  Port $V2_PORT ready after ${i}s"
        break
    fi
    if [ $i -eq 300 ]; then
        echo "ERROR: Port $V2_PORT not ready after 300s"
        kill $V2_PID 2>/dev/null
        exit 1
    fi
    sleep 1
done

echo "V2 vLLM ready. Starting Critic evaluation..."
python -u scripts/eval/eval_d9_critic_zeroshot.py \
    --v2_endpoint http://localhost:$V2_PORT/v1 \
    --num_workers 8

echo "=========================================="
echo "Eval D9 Completed"
echo "End time: $(date)"
echo "=========================================="

kill $V2_PID 2>/dev/null
wait
