#!/bin/bash
#SBATCH --job-name=cog_base
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --output=scripts/exp2/logs/cog_base_%j.out
#SBATCH --error=scripts/exp2/logs/cog_base_%j.err

# ============================================================================
# Cognitive Interference — BASE MODEL (Qwen2.5-VL-7B-Instruct)
#
# vLLM server (4 GPU tensor parallel) + 128 concurrent workers
# Phase 1: Condition C (AR greedy, parallel across 202 trajectories)
# Phase 2: Conditions A+B+D (parallel across 1916 steps)
#
# Expected total time: ~15-30 minutes (vs 5 hours with HF Transformers)
# ============================================================================

set -uo pipefail

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="${PROJECT_ROOT}/checkpoints/Qwen2.5-VL-7B-Instruct"
DATA_ROOT="${PROJECT_ROOT}/datasets/GUI-360/test"
PATTERN_B_IDS="${PROJECT_ROOT}/scripts/exp2/pattern_b_ids.json"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${PROJECT_ROOT}/scripts/exp2/results/cognitive_interference_base_${TIMESTAMP}"
VLLM_PORT=19806
VLLM_HOST="localhost"
MAX_WORKERS=128

mkdir -p "${RESULTS_DIR}"
mkdir -p "${PROJECT_ROOT}/scripts/exp2/logs"

# Environment
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"

echo "================================================================"
echo "Cognitive Interference — BASE MODEL (vLLM)"
echo "Timestamp: ${TIMESTAMP}"
echo "Model: ${BASE_MODEL}"
echo "Workers: ${MAX_WORKERS}"
echo "vLLM: 4 GPU tensor parallel, port ${VLLM_PORT}"
echo "Output: ${RESULTS_DIR}"
echo "================================================================"

# ======================================================================
# Step 1: Start vLLM server
# ======================================================================
echo ""
echo "Step 1: Starting vLLM server..."

pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 2

python -m vllm.entrypoints.openai.api_server \
    --model "${BASE_MODEL}" \
    --port ${VLLM_PORT} \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --trust-remote-code \
    --gpu-memory-utilization 0.9 \
    2>&1 | tee "${RESULTS_DIR}/vllm_server.log" &

VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

# Wait for server
echo "Waiting for vLLM server..."
MAX_WAIT=600
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s http://${VLLM_HOST}:${VLLM_PORT}/health > /dev/null 2>&1; then
        echo "vLLM server ready! (${WAIT_COUNT}s)"
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

# ======================================================================
# Step 2: Run all conditions
# ======================================================================
echo ""
echo "Step 2: Running all conditions (Phase 1: C, Phase 2: A+B+D)..."

python scripts/exp2/cognitive_interference_vllm.py \
    --api_url "http://${VLLM_HOST}:${VLLM_PORT}/v1" \
    --model_name "${BASE_MODEL}" \
    --data_root "${DATA_ROOT}" \
    --trajectory_ids "${PATTERN_B_IDS}" \
    --output_dir "${RESULTS_DIR}" \
    --max_workers ${MAX_WORKERS}

echo ""
echo "Inference complete."

# ======================================================================
# Step 3: Analysis
# ======================================================================
echo ""
echo "Step 3: Running analysis..."

python scripts/exp2/analyze_cognitive_interference.py \
    --results_dir "${RESULTS_DIR}"

# ======================================================================
# Cleanup
# ======================================================================
echo ""
echo "Stopping vLLM server..."
kill $VLLM_PID 2>/dev/null || true
pkill -f "vllm.entrypoints.openai.api_server" || true

echo ""
echo "================================================================"
echo "Experiment complete!"
echo "Results: ${RESULTS_DIR}"
echo "Report: ${RESULTS_DIR}/COGNITIVE_INTERFERENCE_REPORT.md"
echo "================================================================"
