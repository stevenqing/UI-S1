#!/bin/bash
#SBATCH --job-name=probe_train
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/probe_train_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/probe_train_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0

# Probe training + plotting only (extraction already done)

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
SCRIPT_DIR="$PROJECT_DIR/scripts/exp2/layer_probing"
OUTPUT_DIR="$PROJECT_DIR/scripts/exp2/results/layer_probing/sft_v2_20260320_130909"
PATTERN_B_FILE="$PROJECT_DIR/scripts/exp2/pattern_b_ids.json"

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Output: $OUTPUT_DIR"

# Step 2: Train probes
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

# Step 3: Plot curves
echo ""
echo "=========================================="
echo "Step 3: Plotting curves"
echo "=========================================="

python $SCRIPT_DIR/probe_layerwise.py \
    --mode plot \
    --output_dir $OUTPUT_DIR

echo ""
echo "=========================================="
echo "DONE: $(date)"
echo "=========================================="
