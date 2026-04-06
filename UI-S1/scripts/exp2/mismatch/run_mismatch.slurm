#!/bin/bash
#SBATCH --job-name=mismatch
#SBATCH --time=2:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --output=scripts/exp2/logs/mismatch_%j.out
#SBATCH --error=scripts/exp2/logs/mismatch_%j.err

# ============================================================================
# Exp2e: History-Screenshot Mismatch Diagnostic Experiment
#
# Phase 1: Part A — CPU analysis (no GPU needed)
# Phase 2: Base model vLLM → oracle_agent_h
# Phase 3: SFT v2 vLLM → oracle + oracle_f4
# Phase 4: Evaluate all conditions
#
# Expected total time: ~15 minutes
# ============================================================================

set -uo pipefail

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="${PROJECT_ROOT}/checkpoints/Qwen2.5-VL-7B-Instruct"
SFT_MODEL="${PROJECT_ROOT}/train_GUI_360/llamafactory/output/gui360_full_sft_v2"
DATA_ROOT="${PROJECT_ROOT}/datasets/GUI-360/test"
PATTERN_B_IDS="${PROJECT_ROOT}/scripts/exp2/pattern_b_ids.json"
C0_RESULTS="${PROJECT_ROOT}/scripts/exp2/results/verifier_experiment/verifier_always_greedy_20260320_164952/verifier_always_greedy_results.json"
F4_RESULTS="${PROJECT_ROOT}/scripts/exp2/results/multiagent_20260321_072954/f4.jsonl"
AGENT_V_SOURCE="${PROJECT_ROOT}/scripts/exp2/results/multiagent_20260321_072954/agent_v.jsonl"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${PROJECT_ROOT}/scripts/exp2/results/mismatch_${TIMESTAMP}"
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
echo "Exp2e: History-Screenshot Mismatch Diagnostic"
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

COMMON_ARGS="--api_url http://${VLLM_HOST}:${VLLM_PORT}/v1 \
    --data_root ${DATA_ROOT} \
    --trajectory_ids ${PATTERN_B_IDS} \
    --c0_results ${C0_RESULTS} \
    --output_dir ${RESULTS_DIR} \
    --max_workers ${MAX_WORKERS}"

# ======================================================================
# Phase 1: Part A — CPU analysis (no GPU)
# ======================================================================
echo ""
echo "================================================================"
echo "PHASE 1: Part A — Mismatch Analysis (CPU only)"
echo "================================================================"

python scripts/exp2/mismatch/mismatch_analysis.py \
    --c0_results "${C0_RESULTS}" \
    --f4_results "${F4_RESULTS}" \
    --trajectory_ids "${PATTERN_B_IDS}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${RESULTS_DIR}"

echo "Phase 1 complete."

# ======================================================================
# Phase 2: Base model — oracle_agent_h
# ======================================================================
echo ""
echo "================================================================"
echo "PHASE 2: Base Model — oracle_agent_h"
echo "================================================================"

start_vllm "${BASE_MODEL}" "base" || exit 1

python scripts/exp2/mismatch/mismatch_inference.py \
    --condition oracle_agent_h \
    --model_name "${BASE_MODEL}" \
    ${COMMON_ARGS}

echo "Phase 2 complete."

stop_vllm

# ======================================================================
# Phase 3: SFT v2 — oracle + oracle_f4
# ======================================================================
echo ""
echo "================================================================"
echo "PHASE 3: SFT v2 — oracle + oracle_f4"
echo "================================================================"

start_vllm "${SFT_MODEL}" "sft" || exit 1

# Symlink agent_v from Exp2d (screenshot-only, history-independent)
ln -sf "${AGENT_V_SOURCE}" "${RESULTS_DIR}/agent_v.jsonl"
echo "Symlinked agent_v.jsonl from Exp2d results"

# Run oracle and oracle_f4 in parallel
# oracle_f4 depends on oracle_agent_h (done in Phase 2) + agent_v (symlinked)
python scripts/exp2/mismatch/mismatch_inference.py \
    --condition oracle \
    --model_name "${SFT_MODEL}" \
    ${COMMON_ARGS} &
PID_ORACLE=$!

python scripts/exp2/mismatch/mismatch_inference.py \
    --condition oracle_f4 \
    --model_name "${SFT_MODEL}" \
    --aux_dir "${RESULTS_DIR}" \
    ${COMMON_ARGS} &
PID_ORACLE_F4=$!

echo "Waiting for oracle (PID=$PID_ORACLE) and oracle_f4 (PID=$PID_ORACLE_F4)..."
wait $PID_ORACLE $PID_ORACLE_F4
echo "Phase 3 complete."

stop_vllm

# ======================================================================
# Phase 4: Evaluate all conditions
# ======================================================================
echo ""
echo "================================================================"
echo "PHASE 4: Evaluation"
echo "================================================================"

echo "Running evaluation on oracle conditions..."
python scripts/exp2/multiagent/evaluate.py \
    --results_dir "${RESULTS_DIR}" \
    --c0_path "${C0_RESULTS}"

# Append results to mismatch report
echo "" >> "${RESULTS_DIR}/MISMATCH_ANALYSIS_REPORT.md"
echo "---" >> "${RESULTS_DIR}/MISMATCH_ANALYSIS_REPORT.md"
echo "" >> "${RESULTS_DIR}/MISMATCH_ANALYSIS_REPORT.md"
echo "## Part B: Oracle History Inference Results" >> "${RESULTS_DIR}/MISMATCH_ANALYSIS_REPORT.md"
echo "" >> "${RESULTS_DIR}/MISMATCH_ANALYSIS_REPORT.md"

# Extract and append key metrics from eval_all.json
python3 -c "
import json, os

results_dir = '${RESULTS_DIR}'
eval_path = os.path.join(results_dir, 'eval_all.json')
if not os.path.exists(eval_path):
    print('eval_all.json not found, skipping report append')
    exit(0)

with open(eval_path) as f:
    evals = json.load(f)

report_path = os.path.join(results_dir, 'MISMATCH_ANALYSIS_REPORT.md')
with open(report_path, 'a') as f:
    f.write('| Condition | Step Acc | Func Match | Args Match | TSR | Seq Progress | Scat Progress |\n')
    f.write('|-----------|----------|------------|------------|-----|-------------|---------------|\n')

    for cond in ['c0', 'oracle', 'oracle_f4']:
        ev = evals.get(cond, {})
        sm = ev.get('step_metrics', {})
        tm = ev.get('trajectory_metrics', {})
        f.write(
            f'| {cond} | {sm.get(\"step_accuracy\", 0):.4f} | '
            f'{sm.get(\"function_match_rate\", 0):.4f} | '
            f'{sm.get(\"args_match_rate\", 0):.4f} | '
            f'{tm.get(\"tsr\", 0):.4f} | '
            f'{tm.get(\"avg_sequential_progress\", 0):.4f} | '
            f'{tm.get(\"avg_scattered_progress\", 0):.4f} |\n'
        )

    f.write('\n')

    # Agent quality for oracle_agent_h
    oah = evals.get('oracle_agent_h', {})
    aq = oah.get('agent_quality', {})
    if aq:
        f.write('### Agent Quality: oracle_agent_h\n\n')
        f.write(f'- Thought-hit rate: {aq.get(\"thought_hit_rate\", 0):.4f}\n')
        f.write(f'- Valid outputs: {aq.get(\"n_valid\", 0)}/{aq.get(\"n_steps\", 0)}\n')
        f.write(f'- Avg length: {aq.get(\"avg_length_chars\", 0):.0f} chars\n\n')

    # Compute deltas
    c0_acc = evals.get('c0', {}).get('step_metrics', {}).get('step_accuracy', 0)
    oracle_acc = evals.get('oracle', {}).get('step_metrics', {}).get('step_accuracy', 0)
    oracle_f4_acc = evals.get('oracle_f4', {}).get('step_metrics', {}).get('step_accuracy', 0)
    f.write('### Mismatch Tax\n\n')
    f.write(f'- oracle - C0 = {oracle_acc - c0_acc:+.4f} (mismatch tax on single model)\n')
    f.write(f'- oracle_f4 - F4 baseline = (compare with Exp2d F4 result)\n')
    f.write(f'- oracle step_acc = {oracle_acc:.4f}\n')
    f.write(f'- oracle_f4 step_acc = {oracle_f4_acc:.4f}\n')

print('Report appended successfully')
"

echo "Phase 4 complete."

# ======================================================================
# Done
# ======================================================================
echo ""
echo "================================================================"
echo "Experiment complete!"
echo "Results: ${RESULTS_DIR}"
echo "Report: ${RESULTS_DIR}/MISMATCH_ANALYSIS_REPORT.md"
echo "Eval: ${RESULTS_DIR}/EVALUATION_SUMMARY.md"
echo "================================================================"
