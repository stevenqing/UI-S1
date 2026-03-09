#!/bin/bash
#SBATCH --job-name=analyze_routing
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/analyze_routing_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/analyze_routing_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

echo "=========================================="
echo "Routing Weight Analysis"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

source /scratch/a5l/shuqing.a5l/miniconda/etc/profile.d/conda.sh
conda activate qwen3-eval

cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1

python train_GUI_360/moe_sft/analyze_routing.py

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="
