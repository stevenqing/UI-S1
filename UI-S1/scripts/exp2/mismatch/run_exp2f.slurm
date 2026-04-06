#!/bin/bash
#SBATCH --job-name=exp2f
#SBATCH --time=3:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --output=scripts/exp2/logs/exp2f_%j.out
#SBATCH --error=scripts/exp2/logs/exp2f_%j.err

# ============================================================================
# Exp2f: Agent_L Analysis + Reset Mechanism Inference
#
# Phase 1: Agent_L Quality Verification (CPU only, ~1 min)
# Phase 2: Reset Inference with L26 probe (GPU, HF Transformers, ~90 min)
# Phase 3: Summary & Report
#
# Expected total time: ~2 hours
# ============================================================================

set -uo pipefail

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
SFT_MODEL="${PROJECT_ROOT}/train_GUI_360/llamafactory/output/gui360_full_sft_v2"
DATA_ROOT="${PROJECT_ROOT}/datasets/GUI-360/test"
PATTERN_B_IDS="${PROJECT_ROOT}/scripts/exp2/pattern_b_ids.json"
PROBE_DATA_DIR="${PROJECT_ROOT}/scripts/exp2/results/layer_probing/sft_v2_20260320_130909"
PROBE_PATH="${PROBE_DATA_DIR}/correctness_probe_L26.pkl"
AGENT_V_RESULTS="${PROJECT_ROOT}/scripts/exp2/results/multiagent_20260321_072954/agent_v.jsonl"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${PROJECT_ROOT}/scripts/exp2/results/exp2f_${TIMESTAMP}"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/exp2/mismatch"

mkdir -p "${RESULTS_DIR}"
mkdir -p "${PROJECT_ROOT}/scripts/exp2/logs"

# Environment
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"

echo "================================================================"
echo "Exp2f: Agent_L Analysis + Reset Mechanism"
echo "Timestamp: ${TIMESTAMP}"
echo "SFT model: ${SFT_MODEL}"
echo "Probe: ${PROBE_PATH}"
echo "Output: ${RESULTS_DIR}"
echo "================================================================"


# ============================================================================
# Phase 1: Agent_L Quality Verification (CPU only)
# ============================================================================
echo ""
echo "================================================================"
echo "Phase 1: Agent_L Quality Verification (L15 Probes)"
echo "================================================================"
PHASE1_START=$(date +%s)

python "${SCRIPT_DIR}/agent_l_analysis.py" \
    --probe_data_dir "${PROBE_DATA_DIR}" \
    --output_dir "${RESULTS_DIR}" \
    --agent_v_results "${AGENT_V_RESULTS}"

PHASE1_EXIT=$?
PHASE1_END=$(date +%s)
PHASE1_DURATION=$((PHASE1_END - PHASE1_START))
echo "Phase 1 completed in ${PHASE1_DURATION}s (exit code: ${PHASE1_EXIT})"

if [ ${PHASE1_EXIT} -ne 0 ]; then
    echo "WARNING: Phase 1 failed, but continuing to Phase 2..."
fi


# ============================================================================
# Phase 2: Reset Mechanism Inference (GPU, HF Transformers)
# ============================================================================
echo ""
echo "================================================================"
echo "Phase 2: Reset Mechanism Inference"
echo "================================================================"
PHASE2_START=$(date +%s)

# Run reset mode (probe-monitored history reset)
echo ""
echo "--- Phase 2a: Reset mode (probe-monitored) ---"
python "${SCRIPT_DIR}/reset_inference.py" \
    --model_path "${SFT_MODEL}" \
    --data_root "${DATA_ROOT}" \
    --probe_path "${PROBE_PATH}" \
    --probe_data_dir "${PROBE_DATA_DIR}" \
    --trajectory_ids "${PATTERN_B_IDS}" \
    --mode reset \
    --output_dir "${RESULTS_DIR}" \
    --consecutive_limit 3

RESET_EXIT=$?
echo "Reset inference exit code: ${RESET_EXIT}"

PHASE2_END=$(date +%s)
PHASE2_DURATION=$((PHASE2_END - PHASE2_START))
echo "Phase 2 completed in ${PHASE2_DURATION}s"


# ============================================================================
# Phase 3: Summary & Report
# ============================================================================
echo ""
echo "================================================================"
echo "Phase 3: Summary"
echo "================================================================"

# Generate combined summary
python -c "
import json, os, glob

results_dir = '${RESULTS_DIR}'
print()
print('='*60)
print('Exp2f Results Summary')
print('='*60)

# Agent_L results
al_path = os.path.join(results_dir, 'agent_l_analysis.json')
if os.path.exists(al_path):
    with open(al_path) as f:
        al = json.load(f)
    tm = al['template_metrics']
    ui = al['ui_state_probe']
    te = al['target_element_probe']
    print()
    print('--- Agent_L (L15 Probes) ---')
    print(f'  UI State probe (L15, image): test_acc={ui[\"test_acc\"]:.4f}, F1={ui[\"test_f1\"]:.4f}')
    print(f'  Target Element probe (L15, last): test_acc={te[\"test_acc\"]:.4f}, F1={te[\"test_f1\"]:.4f}')
    print(f'  Both classes correct: {tm[\"category_correct_rate\"]:.4f}')
    if tm.get('agent_v_hit_rate'):
        print(f'  Agent V thought-hit: {tm[\"agent_v_hit_rate\"]:.4f}')
else:
    print('Agent_L results not found')

# Reset results
reset_path = os.path.join(results_dir, 'reset_reset_results.json')
if os.path.exists(reset_path):
    with open(reset_path) as f:
        reset = json.load(f)
    s = reset['statistics']
    print()
    print('--- Reset Mechanism ---')
    print(f'  Mode: {s[\"mode\"]}')
    print(f'  Step Accuracy: {s[\"step_success_rate\"]:.4f}')
    print(f'  TSR: {s[\"trajectory_success_rate\"]:.4f}')
    print(f'  Avg Progress: {s[\"avg_progress_rate\"]:.4f}')
    print(f'  Avg Scattered Progress: {s[\"avg_scattered_progress_rate\"]:.4f}')
    if 'total_resets' in s:
        print(f'  Total Resets: {s[\"total_resets\"]} ({s[\"avg_resets_per_trajectory\"]:.1f}/traj)')
    pa = s.get('probe_analysis', {})
    if pa:
        print(f'  Probe threshold: {pa.get(\"threshold\", \"N/A\")}')
        print(f'  Wrong detection recall: {pa.get(\"wrong_detection_recall\", \"N/A\")}')
        print(f'  Wrong detection precision: {pa.get(\"wrong_detection_precision\", \"N/A\")}')
        if pa.get('steps_after_reset_accuracy') is not None:
            print(f'  Post-reset accuracy: {pa[\"steps_after_reset_accuracy\"]:.4f} ({pa[\"n_steps_after_reset\"]} steps)')
    for d in sorted(s.get('by_domain', {}).keys()):
        ds = s['by_domain'][d]
        print(f'  {d}: step_acc={ds[\"step_accuracy\"]:.4f} (n={ds[\"n\"]})')
else:
    print('Reset results not found')

# Comparison with C0 baseline
print()
print('--- Comparison with C0 Baseline ---')
print('  C0 (Exp2e): step_acc=0.3653, TSR=0.0297')
if os.path.exists(reset_path):
    with open(reset_path) as f:
        reset = json.load(f)
    s = reset['statistics']
    delta_step = s['step_success_rate'] - 0.3653
    delta_tsr = s['trajectory_success_rate'] - 0.0297
    print(f'  Reset:       step_acc={s[\"step_success_rate\"]:.4f} (delta={delta_step:+.4f}), '
          f'TSR={s[\"trajectory_success_rate\"]:.4f} (delta={delta_tsr:+.4f})')
print('  Oracle (Exp2e): step_acc=0.4139, TSR=0.0743')
print()
"

echo ""
echo "================================================================"
echo "All phases complete!"
echo "Results: ${RESULTS_DIR}"
echo "================================================================"
echo ""
echo "Files:"
ls -la "${RESULTS_DIR}"
