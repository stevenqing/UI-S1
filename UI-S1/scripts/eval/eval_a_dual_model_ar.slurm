#!/bin/bash
#SBATCH --job-name=eval_a_ar
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_a_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_a_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Eval A: V2+V3 Dual-Model AR Trajectory TSR Baseline
#
# Launches two vLLM servers (V2 on TP=2, V3 on TP=2) on 4 GPUs,
# then runs autoregressive trajectory evaluation with dual model.
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
LOG_DIR="$PROJECT_DIR/scripts/eval/logs"

V2_MODEL="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/llamafactory/output/gui360_full_sft_v2"
V3_MODEL="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/llamafactory/output/gui360_full_sft_v3_grounding"

V2_PORT=19816
V3_PORT=19815
V2_HOST="localhost"
V3_HOST="localhost"

echo "=========================================="
echo "Eval A: V2+V3 Dual-Model AR Trajectory TSR"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "V2 Model: $V2_MODEL"
echo "V3 Model: $V3_MODEL"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

mkdir -p $LOG_DIR

# Verify models exist
if [ ! -d "$V2_MODEL" ]; then
    echo "ERROR: V2 model not found at $V2_MODEL"
    exit 1
fi
if [ ! -d "$V3_MODEL" ]; then
    echo "ERROR: V3 model not found at $V3_MODEL"
    exit 1
fi

# Kill any existing vLLM servers
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

# ============================================
# Start V2 vLLM Server (GPUs 0-1, TP=2)
# ============================================
echo "Starting V2 vLLM Server on port $V2_PORT (GPUs 0-1)..."

CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model $V2_MODEL \
    --served-model-name sft_v2 \
    --port $V2_PORT \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $LOG_DIR/vllm_v2_${SLURM_JOB_ID}.log &
V2_PID=$!

# ============================================
# Start V3 vLLM Server (GPUs 2-3, TP=2)
# ============================================
echo "Starting V3 vLLM Server on port $V3_PORT (GPUs 2-3)..."

CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
    --model $V3_MODEL \
    --served-model-name sft_v3 \
    --port $V3_PORT \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee $LOG_DIR/vllm_v3_${SLURM_JOB_ID}.log &
V3_PID=$!

echo "V2 PID: $V2_PID, V3 PID: $V3_PID"

# ============================================
# Wait for both servers to be ready
# ============================================
wait_for_server() {
    local HOST=$1
    local PORT=$2
    local NAME=$3
    local MAX_WAIT=600
    local WAIT=0

    echo "Waiting for $NAME server ($HOST:$PORT)..."
    while [ $WAIT -lt $MAX_WAIT ]; do
        if curl -s http://${HOST}:${PORT}/health > /dev/null 2>&1; then
            echo "$NAME server is ready! (waited ${WAIT}s)"
            return 0
        fi
        sleep 10
        WAIT=$((WAIT + 10))
        echo "  Waiting for $NAME... ($WAIT/${MAX_WAIT}s)"
    done

    echo "ERROR: $NAME server failed to start within ${MAX_WAIT}s"
    return 1
}

wait_for_server $V2_HOST $V2_PORT "V2" || { kill $V2_PID $V3_PID 2>/dev/null; exit 1; }
wait_for_server $V3_HOST $V3_PORT "V3" || { kill $V2_PID $V3_PID 2>/dev/null; exit 1; }

echo "Both vLLM servers ready!"

# ============================================
# Run Dual-Model AR Evaluation
# ============================================
echo "=========================================="
echo "Starting Dual-Model AR Evaluation"
echo "=========================================="

cd $PROJECT_DIR

python -u scripts/eval/eval_a_dual_model_ar.py \
    --v2_endpoint "http://${V2_HOST}:${V2_PORT}/v1" \
    --v3_endpoint "http://${V3_HOST}:${V3_PORT}/v1" \
    --v2_model sft_v2 \
    --v3_model sft_v3 \
    --output_dir outputs/eval_a \
    --num_workers 3 \
    --stop_on_error

# ============================================
# Cleanup
# ============================================
echo "=========================================="
echo "Stopping vLLM Servers"
echo "=========================================="
kill $V2_PID $V3_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true

echo "=========================================="
echo "Eval A Completed"
echo "=========================================="
echo "End time: $(date)"
echo "Results: $PROJECT_DIR/outputs/eval_a/"
cat $PROJECT_DIR/outputs/eval_a/summary.json 2>/dev/null || echo "No summary found"
echo "=========================================="
