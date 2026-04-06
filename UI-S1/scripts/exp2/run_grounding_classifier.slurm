#!/bin/bash
#SBATCH --job-name=grnd_class
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/grounding_classifier_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/grounding_classifier_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ============================================
# Grounding vs Planning Error Classifier
# CPU-only analysis job
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
cd "$PROJECT_DIR/scripts/exp2"

echo "Starting grounding vs planning classifier at $(date)"
echo "Host: $(hostname)"

python3 grounding_vs_planning_classifier.py

echo "Finished at $(date)"
