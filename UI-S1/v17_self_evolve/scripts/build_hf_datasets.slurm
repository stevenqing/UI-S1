#!/bin/bash
#SBATCH --job-name=build_hf_ds
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v17_self_evolve/scripts/logs/build_hf_datasets_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v17_self_evolve/scripts/logs/build_hf_datasets_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export PYTHONPATH="${PROJECT_DIR}:$PYTHONPATH"

echo "=========================================="
echo "Build HF Datasets — GUI-360 Balanced"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start:  $(date)"
echo "=========================================="

cd "${PROJECT_DIR}"

python v17_self_evolve/build_hf_datasets.py \
    --balanced_train v12_gui_360/data/gui360_train_2000_balanced.jsonl \
    --balanced_test v12_gui_360/data/gui360_test_1000_balanced.jsonl \
    --processed_dir datasets/GUI-360/processed_data \
    --output_dir v17_self_evolve/data/hf_datasets \
    --push_to_hub \
    --hub_org Stevenshuqing

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
