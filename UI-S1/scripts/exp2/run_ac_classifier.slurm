#!/bin/bash
#SBATCH --job-name=ac_class
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/ac_classifier_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/ac_classifier_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -e
cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2

echo "Starting AC classifier at $(date)"
python3 ac_grounding_vs_planning.py
echo "Done at $(date)"
