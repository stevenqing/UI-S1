#!/bin/bash
#SBATCH --job-name=probe_v2
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/probe_v2_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/probe_v2_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0

# Probe v2: Target Element / UI State / Correctness
# Reuses existing hidden state npy files, new labels

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
SCRIPT_DIR="$PROJECT_DIR/scripts/exp2/layer_probing"
OUTPUT_DIR="$PROJECT_DIR/scripts/exp2/results/layer_probing/sft_v2_20260320_130909"

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export PYTHONUNBUFFERED=1

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"

echo ""
echo "=========================================="
echo "Step 1: Training v2 probes (CPU)"
echo "=========================================="

python $SCRIPT_DIR/probe_v2.py \
    --mode probe \
    --output_dir $OUTPUT_DIR

echo "Probing done: $(date)"

echo ""
echo "=========================================="
echo "Step 2: Plotting v2 curves"
echo "=========================================="

python $SCRIPT_DIR/probe_v2.py \
    --mode plot \
    --output_dir $OUTPUT_DIR

echo ""
echo "=========================================="
echo "DONE: $(date)"
echo "=========================================="
