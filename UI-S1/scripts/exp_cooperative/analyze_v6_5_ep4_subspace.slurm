#!/bin/bash
#SBATCH --job-name=svd_v65e4
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/analyze_v6_5_ep4_subspace_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/analyze_v6_5_ep4_subspace_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0

# ============================================
# SVD subspace analysis: v6.5 (tanh, lora_B zero, ceiling+) @ epoch-4
# Best cooperative LoRA so far: 50.06% GUI-360 action_prediction.
#
# Question:
#   Does v6.5 ep4 sit in the same orthogonal regime as v6.4 ep4
#   (cos ~= 0, mean angle ~78°), or does its different init / tanh
#   gate put it somewhere else on the geometry spectrum?
#
# Fills the last row of the 4-way table in section 26.3.
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

echo "=========================================="
echo "SVD subspace analysis: v6.5 @ epoch-4"
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

CKPT="train_GUI_360/llamafactory/output/cooperative_v6_5_comm_thought/epoch-4"
OUT="evaluation/analysis_results_v3/v6_5_ep4_lora_subspace.json"

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
