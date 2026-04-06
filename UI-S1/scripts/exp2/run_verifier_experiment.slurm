#!/bin/bash
#SBATCH --job-name=verifier_exp
#SBATCH --time=08:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/verifier_exp_%j.out
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/verifier_exp_%j.err

set -uo pipefail

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
cd "$PROJECT_DIR"

# Environment
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export PYTHONUNBUFFERED=1

# Paths
MODEL_PATH="$PROJECT_DIR/train_GUI_360/llamafactory/output/gui360_full_sft_v2"
DATA_ROOT="$PROJECT_DIR/datasets/GUI-360/test"
PROBE_DATA_DIR="$PROJECT_DIR/scripts/exp2/results/layer_probing/sft_v2_20260320_130909"
PROBE_PATH="${PROBE_DATA_DIR}/correctness_probe_L26.pkl"
TRAJECTORY_IDS="$PROJECT_DIR/scripts/exp2/pattern_b_ids.json"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$PROJECT_DIR/scripts/exp2/results/verifier_experiment"

# Create dirs
mkdir -p "$PROJECT_DIR/scripts/exp2/logs"
mkdir -p "${RESULTS_DIR}"

echo "=== Verifier Validation Experiment ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Model: ${MODEL_PATH}"
echo "Data: ${DATA_ROOT}"
echo "Timestamp: ${TIMESTAMP}"
echo "Strategy: 3 conditions PARALLEL on GPUs 0/1/2"
echo ""

# Verify paths
for P in "$MODEL_PATH" "$DATA_ROOT" "$PROBE_DATA_DIR/layer_26_last.npy" "$TRAJECTORY_IDS"; do
    if [ ! -e "$P" ]; then
        echo "ERROR: Missing path: $P"
        exit 1
    fi
done
echo "All paths verified."

# Step 1: Save probe model (CPU, fast)
echo ""
echo "=== Step 1: Saving probe model ==="
python scripts/exp2/layer_probing/save_probe_model.py \
    --data_dir "${PROBE_DATA_DIR}" \
    --output_path "${PROBE_PATH}"
echo ""

# Step 2: Run 3 conditions in PARALLEL on separate GPUs
# 7B bf16 ≈ 14GB per model copy, each gets its own GPU
echo "=== Step 2: Launching 3 conditions in parallel ==="

# GPU 0: always_greedy (baseline)
OUTDIR_GREEDY="${RESULTS_DIR}/verifier_always_greedy_${TIMESTAMP}"
mkdir -p "${OUTDIR_GREEDY}"
echo "[GPU 0] Starting always_greedy → ${OUTDIR_GREEDY}"
CUDA_VISIBLE_DEVICES=0 python scripts/exp2/verifier_ar_inference.py \
    --model_path "${MODEL_PATH}" \
    --data_root "${DATA_ROOT}" \
    --trajectory_ids "${TRAJECTORY_IDS}" \
    --mode always_greedy \
    --output_dir "${OUTDIR_GREEDY}" \
    > "${OUTDIR_GREEDY}/stdout.log" 2>&1 &
PID_GREEDY=$!

# GPU 1: verifier (probe-guided) — fused generate+probe, single prefill
OUTDIR_VERIFIER="${RESULTS_DIR}/verifier_verifier_${TIMESTAMP}"
mkdir -p "${OUTDIR_VERIFIER}"
echo "[GPU 1] Starting verifier → ${OUTDIR_VERIFIER}"
CUDA_VISIBLE_DEVICES=1 python scripts/exp2/verifier_ar_inference.py \
    --model_path "${MODEL_PATH}" \
    --data_root "${DATA_ROOT}" \
    --probe_path "${PROBE_PATH}" \
    --trajectory_ids "${TRAJECTORY_IDS}" \
    --mode verifier \
    --output_dir "${OUTDIR_VERIFIER}" \
    > "${OUTDIR_VERIFIER}/stdout.log" 2>&1 &
PID_VERIFIER=$!

# GPU 2: always_temp1 (stochastic control)
OUTDIR_TEMP1="${RESULTS_DIR}/verifier_always_temp1_${TIMESTAMP}"
mkdir -p "${OUTDIR_TEMP1}"
echo "[GPU 2] Starting always_temp1 → ${OUTDIR_TEMP1}"
CUDA_VISIBLE_DEVICES=2 python scripts/exp2/verifier_ar_inference.py \
    --model_path "${MODEL_PATH}" \
    --data_root "${DATA_ROOT}" \
    --trajectory_ids "${TRAJECTORY_IDS}" \
    --mode always_temp1 \
    --output_dir "${OUTDIR_TEMP1}" \
    > "${OUTDIR_TEMP1}/stdout.log" 2>&1 &
PID_TEMP1=$!

echo "PIDs: greedy=${PID_GREEDY} verifier=${PID_VERIFIER} temp1=${PID_TEMP1}"
echo "Waiting for all 3 conditions to finish..."
echo ""

# Monitor progress every 5 min
while kill -0 $PID_GREEDY 2>/dev/null || kill -0 $PID_VERIFIER 2>/dev/null || kill -0 $PID_TEMP1 2>/dev/null; do
    echo "--- $(date) ---"
    for MODE in always_greedy verifier always_temp1; do
        LOGF="${RESULTS_DIR}/verifier_${MODE}_${TIMESTAMP}/stdout.log"
        if [ -f "$LOGF" ]; then
            LAST=$(tail -1 "$LOGF" 2>/dev/null || true)
            echo "  ${MODE}: ${LAST}"
        fi
    done
    sleep 300
done

# Collect exit codes
wait $PID_GREEDY;   EC_GREEDY=$?
wait $PID_VERIFIER; EC_VERIFIER=$?
wait $PID_TEMP1;    EC_TEMP1=$?

echo ""
echo "=== All conditions complete ==="
echo "Exit codes: greedy=${EC_GREEDY} verifier=${EC_VERIFIER} temp1=${EC_TEMP1}"

if [ $EC_GREEDY -ne 0 ] || [ $EC_VERIFIER -ne 0 ] || [ $EC_TEMP1 -ne 0 ]; then
    echo "WARNING: Some conditions failed! Check logs:"
    echo "  ${OUTDIR_GREEDY}/stdout.log"
    echo "  ${OUTDIR_VERIFIER}/stdout.log"
    echo "  ${OUTDIR_TEMP1}/stdout.log"
fi

# Step 3: Run analysis
echo ""
echo "=== Step 3: Running analysis ==="
python scripts/exp2/analyze_verifier_experiment.py \
    --results_dir "${RESULTS_DIR}" \
    --timestamp "${TIMESTAMP}"

echo ""
echo "=== Done: $(date) ==="
