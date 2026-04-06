#!/bin/bash
#SBATCH --job-name=layer_probe
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/layer_probe_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/layer_probe_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:1

# ============================================
# Layer-wise Probing Experiment
# Extract hidden states from all 28 layers of
# Qwen2.5-VL-7B (SFT v2) on GUI-360 test data,
# then train scene/progress/action probes.
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
SCRIPT_DIR="$PROJECT_DIR/scripts/exp2/layer_probing"
LOG_DIR="$PROJECT_DIR/scripts/exp2/logs"

MODEL_PATH="$PROJECT_DIR/train_GUI_360/llamafactory/output/gui360_full_sft_v2"
DATA_ROOT="$PROJECT_DIR/datasets/GUI-360/test"
PATTERN_B_FILE="$PROJECT_DIR/scripts/exp2/pattern_b_ids.json"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_DIR/scripts/exp2/results/layer_probing/sft_v2_${TIMESTAMP}"

echo "=========================================="
echo "Layer-wise Probing Experiment"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export PYTHONUNBUFFERED=1

mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

# Ensure Pattern B IDs exist
if [ ! -f "$PATTERN_B_FILE" ]; then
    echo "Generating Pattern B trajectory IDs..."
    python $PROJECT_DIR/scripts/exp2/dump_pattern_b_ids.py
fi
echo "Pattern B IDs: $(python -c "import json; print(len(json.load(open('$PATTERN_B_FILE'))))")"

# ── Step 1: Extract hidden states ───────────────────────────
echo ""
echo "=========================================="
echo "Step 1: Extracting hidden states (GPU)"
echo "=========================================="
echo "Start: $(date)"

python $SCRIPT_DIR/probe_layerwise.py \
    --mode extract \
    --model_path $MODEL_PATH \
    --data_root $DATA_ROOT \
    --pattern_b_file $PATTERN_B_FILE \
    --output_dir $OUTPUT_DIR \
    --max_samples 3000

echo "Extraction done: $(date)"
echo "Files in $OUTPUT_DIR:"
ls -lh $OUTPUT_DIR/*.npy 2>/dev/null | wc -l
echo "npy files"
du -sh $OUTPUT_DIR

# ── Step 2: Train probes ────────────────────────────────────
echo ""
echo "=========================================="
echo "Step 2: Training probes (CPU)"
echo "=========================================="
echo "Start: $(date)"

python $SCRIPT_DIR/probe_layerwise.py \
    --mode probe \
    --output_dir $OUTPUT_DIR \
    --pattern_b_file $PATTERN_B_FILE

echo "Probing done: $(date)"

# ── Step 3: Plot curves ─────────────────────────────────────
echo ""
echo "=========================================="
echo "Step 3: Plotting curves"
echo "=========================================="

python $SCRIPT_DIR/probe_layerwise.py \
    --mode plot \
    --output_dir $OUTPUT_DIR

echo ""
echo "=========================================="
echo "DONE"
echo "=========================================="
echo "End: $(date)"
echo "Results: $OUTPUT_DIR"
echo "  labels.json            — sample metadata"
echo "  probe_results.json     — probe accuracies per layer"
echo "  probing_curves.png     — main accuracy curves"
echo "  probing_curves_cv.png  — 5-fold CV curves"
echo "  cross_probe_heatmap.png — cross-probe analysis"
