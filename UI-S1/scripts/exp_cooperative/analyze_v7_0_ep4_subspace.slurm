#!/bin/bash
#SBATCH --job-name=svd_v70e4
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/analyze_v7_0_ep4_subspace_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/analyze_v7_0_ep4_subspace_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0

# ============================================
# SVD subspace analysis: v7.0 (merge + diversity L_div = 0.01) @ epoch-4
# Final accuracy: 48.40% GUI-360 action_prediction.
#
# Question:
#   The div_loss training log reported cos(B_v, B_a) reaching ~-0.76.
#   Does a static SVD measurement on the final checkpoint confirm the
#   anti-correlated geometry, or was the training-time log an artifact
#   of the loss computation?
#
# Fills the v7.0 row of the 4-way table in section 26.3.
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

echo "=========================================="
echo "SVD subspace analysis: v7.0 @ epoch-4"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "=========================================="

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=72
export MKL_NUM_THREADS=72

cd "$PROJECT_DIR"

CKPT="train_GUI_360/llamafactory/output/cooperative_v7_0_merge_diversity/epoch-4"
OUT="evaluation/analysis_results_v3/v7_0_ep4_lora_subspace.json"

mkdir -p "$(dirname "$OUT")"

python -u evaluation/analyze_lora_subspace.py \
    --checkpoint "$CKPT" \
    --output "$OUT" \
    --top_k 64 \
    --keep_top_sigma 10

echo ""
echo "=========================================="
echo "Done: $(date)"
echo "Output: $OUT"
echo "=========================================="
