#!/bin/bash
#SBATCH --job-name=v18_extract_k8
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v18_k_expert/scripts/logs/extract_k8_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v18_k_expert/scripts/logs/extract_k8_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:1

# ============================================
# V18: Extract K=8 Expert Cooperative LoRA from Full SFT via SVD
#
# Single GPU, ~10 min. Produces K=8 SVD-initialized checkpoint.
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

mkdir -p "${PROJECT_DIR}/v18_k_expert/scripts/logs"

echo "=========================================="
echo "V18: SVD Extraction K=8 Experts"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Start:   $(date)"
echo "=========================================="

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="${PROJECT_DIR}:$PYTHONPATH"
export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"

python v18_k_expert/extract_fullsft_to_k_expert.py \
    --sft_model train_GUI_360/llamafactory/output/gui360_balanced_full_sft/checkpoint-250 \
    --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
    --output checkpoints/v18_svd_k8_7mod \
    --rank 128 \
    --alpha 256 \
    --noise_scale 0.1 \
    --num_experts 8 \
    --comm_topology top2 \
    --num_comm_rounds 2 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
