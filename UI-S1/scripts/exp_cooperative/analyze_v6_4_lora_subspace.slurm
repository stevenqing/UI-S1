#!/bin/bash
#SBATCH --job-name=svd_v64
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/analyze_v6_4_lora_subspace_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/analyze_v6_4_lora_subspace_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0

# ============================================
# SVD subspace analysis for v6.4 epoch-4 cooperative LoRA
# CPU-only job (no GPU needed). Computes per-module:
#   - frob_cos(B_v A_v, B_a A_a)
#   - stable_rank V/A
#   - top-10 singular values V/A
#   - principal angles (col + row) between V and A subspaces (top-64)
# Aggregates by layer and by module type, prints summary, writes JSON.
# Expected runtime: ~5-10 min (196 modules, all on CPU).
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

echo "=========================================="
echo "SVD subspace analysis: v6.4 epoch-4"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "=========================================="

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1
# Force CPU SVD; PyTorch will pick up all available cores.
export OMP_NUM_THREADS=72
export MKL_NUM_THREADS=72

cd "$PROJECT_DIR"

CKPT="train_GUI_360/llamafactory/output/cooperative_v6_4_comm_thought/epoch-4"
OUT="evaluation/analysis_results_v3/v6_4_ep4_lora_subspace.json"

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
