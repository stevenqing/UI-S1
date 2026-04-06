#!/bin/bash
#SBATCH --job-name=eval_coop_proper
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_proper_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/eval_proper_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Proper routing eval (no merge) — v1 checkpoint
# Tests if merge approximation was the problem
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

echo "=========================================="
echo "Cooperative LoRA v1 — Proper Routing Eval"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

cd "$PROJECT_DIR"

RESULTS_BASE="$PROJECT_DIR/train_GUI_360/GUI-360-eval/results/cooperative_thought_v1_proper"

# Quick test: 500 samples grounding to see if routing matters
python evaluation/eval_cooperative_proper.py \
    --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
    --coop_checkpoint train_GUI_360/llamafactory/output/cooperative_thought_v1/final \
    --eval_data_root datasets/GUI-360/test \
    --eval_type grounding \
    --output_dir "$RESULTS_BASE/grounding" \
    --max_samples 500

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
