#!/bin/bash
#SBATCH --job-name=v9_merge
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v9/scripts/logs/merge_v9_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v9/scripts/logs/merge_v9_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# tiny wrapper so the decode→train chain can be expressed as sbatch dependencies.
set -e
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
cd "$PROJECT_DIR"
bash v9/scripts/merge_credit_shards.sh
