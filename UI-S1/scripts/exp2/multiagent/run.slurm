#!/bin/bash
#SBATCH --job-name=multiagent
#SBATCH --time=4:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --output=scripts/exp2/logs/multiagent_%j.out
#SBATCH --error=scripts/exp2/logs/multiagent_%j.err

# ============================================================================
# Multi-Agent Decomposition Experiment (Exp2d)
#
# Phase 1: Base model vLLM + base conditions (agent_v, agent_h, pass1, f5_pass1)
# Phase 2: SFT v2 vLLM + SFT conditions (f1, f2, f3, f4, f5, f6)
# Phase 3: Evaluation + Analysis (CPU only)
#
# Expected total time: ~20 minutes
# ============================================================================

set -uo pipefail

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="${PROJECT_ROOT}/checkpoints/Qwen2.5-VL-7B-Instruct"
SFT_MODEL="${PROJECT_ROOT}/train_GUI_360/llamafactory/output/gui360_full_sft_v2"
DATA_ROOT="${PROJECT_ROOT}/datasets/GUI-360/test"
PATTERN_B_IDS="${PROJECT_ROOT}/scripts/exp2/pattern_b_ids.json"
C0_RESULTS="${PROJECT_ROOT}/scripts/exp2/results/verifier_experiment/verifier_always_greedy_20260320_164952/verifier_always_greedy_results.json"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${PROJECT_ROOT}/scripts/exp2/results/multiagent_${TIMESTAMP}"
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
echo "Multi-Agent Decomposition Experiment (Exp2d)"
echo "Timestamp: ${TIMESTAMP}"
echo "Base model: ${BASE_MODEL}"
echo "SFT model: ${SFT_MODEL}"
echo "Workers: ${MAX_WORKERS}"
echo "Output: ${RESULTS_DIR}"
echo "================================================================"

# Helper function to start vLLM and wait for readiness
start_vllm() {
    local model_path="$1"
    local label="$2"

    echo ""
    echo "Starting vLLM server (${label})..."
    pkill -f "vllm.entrypoints.openai.api_server" || true
    sleep 3

    python -m vllm.entrypoints.openai.api_server \
        --model "${model_path}" \
        --port ${VLLM_PORT} \
        --tensor-parallel-size 4 \
        --max-model-len 32768 \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        2>&1 | tee "${RESULTS_DIR}/vllm_${label}.log" &

    VLLM_PID=$!
    echo "vLLM PID: ${VLLM_PID}"

    MAX_WAIT=600
    WAIT_COUNT=0
    while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
        if curl -s http://${VLLM_HOST}:${VLLM_PORT}/health > /dev/null 2>&1; then
            echo "vLLM server ready! (${WAIT_COUNT}s)"
            return 0
        fi
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
    done

    echo "ERROR: vLLM server failed to start"
    kill $VLLM_PID 2>/dev/null || true
    return 1
}

stop_vllm() {
    echo "Stopping vLLM server..."
    kill $VLLM_PID 2>/dev/null || true
    pkill -f "vllm.entrypoints.openai.api_server" || true
    sleep 5
}

# Common inference args
COMMON_ARGS="--api_url http://${VLLM_HOST}:${VLLM_PORT}/v1 \
    --data_root ${DATA_ROOT} \
    --trajectory_ids ${PATTERN_B_IDS} \
    --c0_results ${C0_RESULTS} \
    --output_dir ${RESULTS_DIR} \
    --max_workers ${MAX_WORKERS}"

# ======================================================================
# Phase 1: Base model conditions
# ======================================================================
echo ""
echo "================================================================"
echo "PHASE 1: Base Model Conditions"
echo "================================================================"

start_vllm "${BASE_MODEL}" "base" || exit 1

# Step 1a: Run agent_v, agent_h, pass1 in parallel (independent)
echo ""
echo "Running agent_v, agent_h, pass1 in parallel..."

python scripts/exp2/multiagent/inference.py \
    --condition agent_v \
    --model_name "${BASE_MODEL}" \
    ${COMMON_ARGS} &
PID_V=$!

python scripts/exp2/multiagent/inference.py \
    --condition agent_h \
    --model_name "${BASE_MODEL}" \
    ${COMMON_ARGS} &
PID_H=$!

python scripts/exp2/multiagent/inference.py \
    --condition pass1 \
    --model_name "${BASE_MODEL}" \
    ${COMMON_ARGS} &
PID_P1=$!

echo "Waiting for agent_v (PID=$PID_V), agent_h (PID=$PID_H), pass1 (PID=$PID_P1)..."
wait $PID_V $PID_H $PID_P1
echo "Phase 1a complete."

# Step 1b: Run f5_pass1 (depends on agent_v + agent_h)
echo ""
echo "Running f5_pass1 (depends on agent_v + agent_h)..."

python scripts/exp2/multiagent/inference.py \
    --condition f5_pass1 \
    --model_name "${BASE_MODEL}" \
    --aux_dir "${RESULTS_DIR}" \
    ${COMMON_ARGS}

echo "Phase 1b complete."

stop_vllm

# ======================================================================
# Phase 2: SFT v2 conditions
# ======================================================================
echo ""
echo "================================================================"
echo "PHASE 2: SFT v2 Conditions"
echo "================================================================"

start_vllm "${SFT_MODEL}" "sft" || exit 1

# Step 2a: Run f1, f2, f3, f4, f6 in parallel (all independent given aux_dir)
echo ""
echo "Running f1, f2, f3, f4, f6 in parallel..."

python scripts/exp2/multiagent/inference.py \
    --condition f1 \
    --model_name "${SFT_MODEL}" \
    --aux_dir "${RESULTS_DIR}" \
    ${COMMON_ARGS} &
PID_F1=$!

python scripts/exp2/multiagent/inference.py \
    --condition f2 \
    --model_name "${SFT_MODEL}" \
    --aux_dir "${RESULTS_DIR}" \
    ${COMMON_ARGS} &
PID_F2=$!

python scripts/exp2/multiagent/inference.py \
    --condition f3 \
    --model_name "${SFT_MODEL}" \
    --aux_dir "${RESULTS_DIR}" \
    ${COMMON_ARGS} &
PID_F3=$!

python scripts/exp2/multiagent/inference.py \
    --condition f4 \
    --model_name "${SFT_MODEL}" \
    --aux_dir "${RESULTS_DIR}" \
    ${COMMON_ARGS} &
PID_F4=$!

python scripts/exp2/multiagent/inference.py \
    --condition f6 \
    --model_name "${SFT_MODEL}" \
    --aux_dir "${RESULTS_DIR}" \
    ${COMMON_ARGS} &
PID_F6=$!

echo "Waiting for f1-f4, f6..."
wait $PID_F1 $PID_F2 $PID_F3 $PID_F4 $PID_F6
echo "Phase 2a complete."

# Step 2b: Run f5 (depends on f5_pass1 + agent_v + agent_h, all already computed)
echo ""
echo "Running f5 (depends on f5_pass1 + agent_v + agent_h)..."

python scripts/exp2/multiagent/inference.py \
    --condition f5 \
    --model_name "${SFT_MODEL}" \
    --aux_dir "${RESULTS_DIR}" \
    ${COMMON_ARGS}

echo "Phase 2b complete."

stop_vllm

# ======================================================================
# Phase 3: Evaluation + Analysis
# ======================================================================
echo ""
echo "================================================================"
echo "PHASE 3: Evaluation + Analysis"
echo "================================================================"

echo "Running evaluation..."
python scripts/exp2/multiagent/evaluate.py \
    --results_dir "${RESULTS_DIR}" \
    --c0_path "${C0_RESULTS}"

echo ""
echo "Running cross-experiment analysis..."
python scripts/exp2/multiagent/analyze.py \
    --results_dir "${RESULTS_DIR}" \
    --c0_path "${C0_RESULTS}"

# ======================================================================
# Done
# ======================================================================
echo ""
echo "================================================================"
echo "Experiment complete!"
echo "Results: ${RESULTS_DIR}"
echo "Report: ${RESULTS_DIR}/MULTIAGENT_EXPERIMENT_REPORT.md"
echo "Summary: ${RESULTS_DIR}/EVALUATION_SUMMARY.md"
echo "================================================================"
