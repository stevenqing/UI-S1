#!/bin/bash
#SBATCH --job-name=grnd_sftv2
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/grounding_classifier_sftv2_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/grounding_classifier_sftv2_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
cd "$PROJECT_DIR/scripts/exp2"

echo "Starting SFT v2 grounding vs planning classifier at $(date)"
echo "Host: $(hostname)"

python3 grounding_vs_planning_classifier_sftv2.py

echo "Finished at $(date)"
