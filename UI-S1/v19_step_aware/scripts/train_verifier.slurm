#!/bin/bash
#SBATCH --job-name=train_verify
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/train_verifier_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v19_step_aware/scripts/logs/train_verifier_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Train Self-Verification LoRA
#
# Task: Binary YES/NO — did the action produce
#       the observed screen change?
# Data: 30K samples (15K pos + 15K neg)
# Base: checkpoint-272 (GUI-360 SFT)
# Method: LoRA rank=128
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CONFIG_FILE="$PROJECT_DIR/v19_step_aware/train_verifier_lora.yaml"

echo "=========================================="
echo "Train Self-Verification LoRA"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start:  $(date)"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

if [ -f "$PROJECT_DIR/train/env_config.sh" ]; then
    source "$PROJECT_DIR/train/env_config.sh"
fi

mkdir -p "$PROJECT_DIR/v19_step_aware/scripts/logs"

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export MASTER_ADDR=localhost
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800

# Node-local HF cache
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_${SLURM_JOB_ID}
export HF_HOME=/tmp/hf_home_${SLURM_JOB_ID}
mkdir -p $HF_DATASETS_CACHE $HF_HOME

cd "$PROJECT_DIR"

FORCE_TORCHRUN=1 \
NNODES=1 \
NODE_RANK=0 \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
NPROC_PER_NODE=4 \
llamafactory-cli train "$CONFIG_FILE"

TRAIN_EXIT_CODE=$?

echo "=========================================="
echo "Training Completed"
echo "End:       $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "Output:    $PROJECT_DIR/v19_step_aware/output/verifier_lora_r128"
echo "=========================================="

exit $TRAIN_EXIT_CODE
