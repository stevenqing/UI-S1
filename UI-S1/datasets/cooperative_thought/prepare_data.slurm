#!/bin/bash
#SBATCH --job-name=prep_thought
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=datasets/cooperative_thought/logs/prep_%j.log
#SBATCH --error=datasets/cooperative_thought/logs/prep_%j.err

echo "=========================================="
echo "Prepare GUI-360 Thought-Augmented Data"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

PROJECT_DIR="/lus/lfs1aip2/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
cd "$PROJECT_DIR"

# Activate conda
source /home/a5l/shuqing.a5l/miniconda3/etc/profile.d/conda.sh
conda activate ui-s1

mkdir -p datasets/cooperative_thought/logs

python datasets/cooperative_thought/prepare_gui360_thought.py --val_trajectories 500

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
