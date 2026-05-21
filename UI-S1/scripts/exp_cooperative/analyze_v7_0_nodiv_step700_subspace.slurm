#!/bin/bash
#SBATCH --job-name=svd_v70n
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/analyze_v7_0_nodiv_step700_subspace_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/analyze_v7_0_nodiv_step700_subspace_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0

# ============================================
# SVD subspace analysis: v7.0 NO-diversity ablation @ step 700
# Checkpoint: cooperative_v7_0_merge_no_diversity/checkpoint-700/cooperative
#
# Question (the only one that matters for this run):
#   Without L_div, does merge mode still drive the two adapters into
#   the cos(B_v, B_a) ~= -0.76 anti-correlated state, or do they stay
#   near 0 (orthogonal / independent)?
#
# Outcome A:  cos << 0 -> anti-correlation is INHERENT to merge mode.
#             Diversity loss is icing on the cake. Strong paper claim.
# Outcome B:  cos ~= 0 -> two adapters live in independent random
#             subspaces; no specialization without L_div. Weak result.
# Outcome C:  cos ~= +1 -> mode collapse without explicit pressure.
#             Confirms L_div is essential.
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

echo "=========================================="
echo "SVD subspace analysis: v7.0 nodiv @ step 700"
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

CKPT="train_GUI_360/llamafactory/output/cooperative_v7_0_merge_no_diversity/checkpoint-700/cooperative"
OUT="evaluation/analysis_results_v3/v7_0_nodiv_step700_lora_subspace.json"

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
