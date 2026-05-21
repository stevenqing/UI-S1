#!/bin/bash
#SBATCH --job-name=svd_v70ne4
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/analyze_v7_0_nodiv_ep4_subspace_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/analyze_v7_0_nodiv_ep4_subspace_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0

# ============================================
# SVD subspace analysis: v7.0 NO-diversity ablation @ epoch-4 (final)
# Resume job 3746312 completed at epoch 4.0 without any L_div pressure.
#
# Question:
#   Step 700 (epoch 0.92) had frob_cos = +0.1646. Does the trajectory
#   saturate, continue toward mode collapse (+1), or settle?
#   Compare to v7.0 with L_div ep4 (frob_cos = +0.1837) and v6.x ep4
#   (frob_cos ~= 0).
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

echo "=========================================="
echo "SVD subspace analysis: v7.0 nodiv @ epoch-4"
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

CKPT="train_GUI_360/llamafactory/output/cooperative_v7_0_merge_no_diversity/epoch-4"
OUT="evaluation/analysis_results_v3/v7_0_nodiv_ep4_lora_subspace.json"

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
