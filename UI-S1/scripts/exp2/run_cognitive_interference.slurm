#!/bin/bash
#SBATCH --job-name=cog_interf
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --output=scripts/exp2/logs/cog_interf_%j.out
#SBATCH --error=scripts/exp2/logs/cog_interf_%j.err

# ============================================================================
# Cognitive Interference Hypothesis Experiment
#
# Runs conditions A, B, D on ALL 1916 Pattern B steps, sharded across 4 GPUs.
# Condition C is extracted from existing greedy results (no inference needed).
#
# Estimated time: ~1916 steps / 4 GPUs × ~5s/step (A+B+D per step) ≈ 40 min
# ============================================================================

set -uo pipefail

# Paths
export PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
MODEL_PATH="${PROJECT_ROOT}/train_GUI_360/llamafactory/output/gui360_full_sft_v2"
DATA_ROOT="${PROJECT_ROOT}/datasets/GUI-360/test"
GREEDY_RESULTS="${PROJECT_ROOT}/scripts/exp2/results/verifier_experiment/verifier_always_greedy_20260320_164952/verifier_always_greedy_results.json"
PATTERN_B_IDS="${PROJECT_ROOT}/scripts/exp2/pattern_b_ids.json"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${PROJECT_ROOT}/scripts/exp2/results/cognitive_interference_${TIMESTAMP}"
STEPS_FILE="${RESULTS_DIR}/all_steps.json"

mkdir -p "${RESULTS_DIR}"
mkdir -p "${PROJECT_ROOT}/scripts/exp2/logs"

# Environment — use direct PATH (same as verifier experiment)
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"

echo "================================================================"
echo "Cognitive Interference Experiment"
echo "Timestamp: ${TIMESTAMP}"
echo "Model: ${MODEL_PATH}"
echo "Data: ${DATA_ROOT}"
echo "Output: ${RESULTS_DIR}"
echo "================================================================"

# Step 1: Prepare all steps (CPU, fast)
echo ""
echo "Step 1: Preparing all steps..."
python scripts/exp2/sample_cognitive_interference.py \
    --data_root "${DATA_ROOT}" \
    --greedy_results "${GREEDY_RESULTS}" \
    --trajectory_ids "${PATTERN_B_IDS}" \
    --output "${STEPS_FILE}"

N_STEPS=$(python -c "import json; print(len(json.load(open('${STEPS_FILE}'))))")
echo "Total steps: ${N_STEPS}"

# Step 2: Run inference on 4 GPU shards in parallel
echo ""
echo "Step 2: Launching 4 GPU shards..."

N_SHARDS=4
PIDS=()

for SHARD_IDX in 0 1 2 3; do
    echo "  Starting shard ${SHARD_IDX}/${N_SHARDS} on GPU ${SHARD_IDX}..."
    CUDA_VISIBLE_DEVICES=${SHARD_IDX} python scripts/exp2/cognitive_interference_inference.py \
        --model_path "${MODEL_PATH}" \
        --steps_file "${STEPS_FILE}" \
        --greedy_results "${GREEDY_RESULTS}" \
        --conditions A B D \
        --output_dir "${RESULTS_DIR}" \
        --shard "${SHARD_IDX}/${N_SHARDS}" \
        > "${RESULTS_DIR}/shard_${SHARD_IDX}.log" 2>&1 &
    PIDS+=($!)
done

echo "All shards launched. PIDs: ${PIDS[*]}"
echo "Monitoring progress..."

# Monitor
while true; do
    ALL_DONE=true
    for i in "${!PIDS[@]}"; do
        if kill -0 "${PIDS[$i]}" 2>/dev/null; then
            ALL_DONE=false
        fi
    done
    if $ALL_DONE; then
        break
    fi

    # Print progress from logs
    echo ""
    echo "--- Progress at $(date +%H:%M:%S) ---"
    for i in 0 1 2 3; do
        LOG="${RESULTS_DIR}/shard_${i}.log"
        if [ -f "${LOG}" ]; then
            LAST=$(tail -1 "${LOG}" 2>/dev/null || echo "starting...")
            echo "  Shard ${i}: ${LAST}"
        fi
    done
    sleep 300  # Check every 5 min
done

# Collect exit codes
echo ""
echo "All shards finished. Checking exit codes..."
FAILED=0
for i in "${!PIDS[@]}"; do
    wait "${PIDS[$i]}" || {
        echo "  Shard ${i} FAILED (exit code $?)"
        FAILED=$((FAILED + 1))
    }
done

if [ $FAILED -gt 0 ]; then
    echo "WARNING: ${FAILED} shard(s) failed. Check logs in ${RESULTS_DIR}/shard_*.log"
fi

# Step 3: Merge shard results
echo ""
echo "Step 3: Merging shard results..."
python -c "
import json, os, glob

results_dir = '${RESULTS_DIR}'
all_results = []
for shard_file in sorted(glob.glob(os.path.join(results_dir, 'cognitive_interference_results_shard*.json'))):
    with open(shard_file) as f:
        data = json.load(f)
    all_results.extend(data['results'])
    print(f'  Loaded {len(data[\"results\"])} results from {os.path.basename(shard_file)}')

merged = {
    'config': {
        'conditions': ['A', 'B', 'C', 'D'],
        'n_samples': len(all_results),
        'n_shards': ${N_SHARDS},
    },
    'results': all_results,
}
merged_path = os.path.join(results_dir, 'cognitive_interference_results.json')
with open(merged_path, 'w') as f:
    json.dump(merged, f, indent=2, default=str)
print(f'Merged {len(all_results)} results → {merged_path}')
"

# Step 4: Run analysis
echo ""
echo "Step 4: Running analysis..."
python scripts/exp2/analyze_cognitive_interference.py \
    --results_dir "${RESULTS_DIR}"

echo ""
echo "================================================================"
echo "Experiment complete!"
echo "Results: ${RESULTS_DIR}"
echo "Report: ${RESULTS_DIR}/COGNITIVE_INTERFERENCE_REPORT.md"
echo "================================================================"
