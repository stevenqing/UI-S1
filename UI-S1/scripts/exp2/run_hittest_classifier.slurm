#!/bin/bash
#SBATCH --job-name=hittest
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/hittest_classifier_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/logs/hittest_classifier_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
cd "$PROJECT_DIR/scripts/exp2"

echo "========== Base Model =========="
python3 hittest_classifier.py --model base

echo ""
echo "========== SFT v2 =========="
python3 hittest_classifier.py --model sftv2

echo ""
echo "Done at $(date)"
