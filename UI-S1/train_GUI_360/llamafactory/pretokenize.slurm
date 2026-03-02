#!/bin/bash
#SBATCH --job-name=gui360_tokenize
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/gui360_tokenize_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/gui360_tokenize_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:1

# ============================================
# Pre-tokenize GUI-360 dataset for LlamaFactory
# Single GPU, saves tokenized cache to disk
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

echo "=========================================="
echo "Pre-tokenize GUI-360 dataset"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

cd "$PROJECT_DIR"

# Tokenize and cache the dataset (no DeepSpeed, single GPU)
llamafactory-cli train \
    train_GUI_360/llamafactory/qwen25vl_gui360_pretokenize.yaml

echo "=========================================="
echo "Tokenization completed"
echo "End time: $(date)"
echo "Cache dir: $PROJECT_DIR/train_GUI_360/llamafactory/data/tokenized_cache"
echo "=========================================="
