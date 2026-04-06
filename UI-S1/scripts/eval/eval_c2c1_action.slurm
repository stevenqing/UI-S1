#!/bin/bash
#SBATCH --job-name=eval_c2c1
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_c2c1_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_c2c1_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ============================================
# Eval C2+C1: Action Type & Description Quality
# CPU only - analyzes K=10 grounding results
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
LOG_DIR="$PROJECT_DIR/scripts/eval/logs"

echo "=========================================="
echo "Eval C2+C1: Action Type Analysis"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

mkdir -p $LOG_DIR

cd $PROJECT_DIR
python -u scripts/eval/eval_c2c1_action_analysis.py

echo "=========================================="
echo "Eval C2+C1 Completed"
echo "End time: $(date)"
echo "=========================================="
