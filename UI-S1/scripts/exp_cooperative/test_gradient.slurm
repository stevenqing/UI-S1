#!/bin/bash
#SBATCH --job-name=coop_grad_test
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/grad_test_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/grad_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:1

# ============================================
# Gradient Flow Test for Cooperative LoRA
# Tests G1 (both adapters get gradients) and
# G2 (LoRA_V gets gradients from both L_act
# and L_bind). Run BEFORE full training.
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

echo "=========================================="
echo "Cooperative LoRA Gradient Flow Test"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

mkdir -p "$PROJECT_DIR/scripts/exp_cooperative/logs"

cd "$PROJECT_DIR"

python evaluation/test_cooperative_gradient.py \
    --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
    --lora_r 4

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
