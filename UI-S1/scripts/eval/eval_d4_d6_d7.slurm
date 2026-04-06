#!/bin/bash
#SBATCH --job-name=eval_d4d6d7
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_d4d6d7_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/logs/eval_d4d6d7_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ============================================
# Eval D4+D6+D7: Offline analyses (CPU only)
# D4: Planner Oracle Ceiling
# D6: Failure Type Taxonomy
# D7: Length × Failure Type Cross-Analysis
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
LOG_DIR="$PROJECT_DIR/scripts/eval/logs"

echo "=========================================="
echo "Eval D4+D6+D7: Offline Analyses"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

mkdir -p $LOG_DIR

cd $PROJECT_DIR

python -u scripts/eval/eval_d4_d6_d7_offline.py

echo "=========================================="
echo "Eval D4+D6+D7 Completed"
echo "End time: $(date)"
echo "=========================================="
