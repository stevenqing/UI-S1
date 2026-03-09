#!/bin/bash
#SBATCH --job-name=dl_qwen35_9b
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/dl_qwen35_9b_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/dl_qwen35_9b_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0

# Download Qwen3.5-9B from HuggingFace
set -e

CHECKPOINT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/checkpoints"
MODEL_ID="Qwen/Qwen3.5-9B"
TARGET_DIR="$CHECKPOINT_DIR/Qwen3.5-9B"

echo "=========================================="
echo "Downloading $MODEL_ID"
echo "Target: $TARGET_DIR"
echo "Start: $(date)"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
# export HF_TOKEN="your_token_here"  # Set your HF token

mkdir -p $TARGET_DIR

huggingface-cli download $MODEL_ID --local-dir $TARGET_DIR

echo "=========================================="
echo "Download completed at $(date)"
echo "Contents:"
ls -lh $TARGET_DIR/
echo "=========================================="
