#!/bin/bash
#SBATCH --job-name=gui360_sft_v8_resume
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/gui360_sft_v8_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/gui360_sft_v8_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# GUI-360 SFT Training Script V8 - Resume
# ============================================
# Resume training from global_step_100 (v6)
# Fixed: Increased GLOO timeout to prevent checkpoint corruption
# ============================================

set -e

# ============================================
# Configuration
# ============================================
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
CONFIG_PATH="$TRAIN_DIR/config"
LOG_DIR="$TRAIN_DIR/logs"

# Training data paths
TRAIN_PARQUET="$TRAIN_DIR/data/gui360_train_sft_a11y_thinking.parquet"
EVAL_PARQUET="$TRAIN_DIR/data/gui360_eval_sft_a11y_thinking_100.parquet"

# Experiment settings
EXPERIMENT_NAME="gui360_sft_v8_resume"
CONFIG_NAME="gui360_sft_v8_resume"
CHECKPOINT_DIR="$TRAIN_DIR/checkpoints/${EXPERIMENT_NAME}"
RESUME_FROM="$TRAIN_DIR/checkpoints/gui360_sft_v8_resume/global_step_250"

# ============================================
# Environment Setup
# ============================================
echo "=========================================="
echo "GUI-360 SFT Training V8 - Resume"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Project Dir: $PROJECT_DIR"
echo "Experiment: $EXPERIMENT_NAME"
echo "Resume from: $RESUME_FROM"
echo "=========================================="

# Activate conda environment
source /home/a5l/shuqing.a5l/miniconda3/bin/activate qwen3-eval

# Source environment configuration
if [ -f "$PROJECT_DIR/train/env_config.sh" ]; then
    source $PROJECT_DIR/train/env_config.sh
fi

# Create directories
mkdir -p $LOG_DIR
mkdir -p $CHECKPOINT_DIR
mkdir -p $TRAIN_DIR/data

# Set distributed environment variables for multi-node
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export WORLD_SIZE=$((SLURM_NNODES * 4))
export NODE_RANK=$SLURM_NODEID

# ============================================
# FIX: Increased timeout settings
# ============================================
# NCCL timeout settings (increased to 3 hours)
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Gloo timeout settings (increased to 2 hours for large checkpoint saving)
# Previous value 1800 (30 min) was too short for 88GB checkpoints
export GLOO_TIMEOUT_SECONDS=7200
export TORCH_GLOO_TIMEOUT_SECONDS=7200

# Additional timeout for distributed operations
export TORCH_DISTRIBUTED_DEBUG=OFF

echo "Master node: $MASTER_ADDR"
echo "Total nodes: $SLURM_NNODES"
echo "Node rank: $NODE_RANK"
echo "World size: $WORLD_SIZE"
echo "GLOO timeout: $GLOO_TIMEOUT_SECONDS seconds"
echo "NCCL timeout: $NCCL_TIMEOUT seconds"

# ============================================
# Verify Resume Checkpoint
# ============================================
echo "=========================================="
echo "Verifying Resume Checkpoint"
echo "=========================================="

if [ ! -d "$RESUME_FROM" ]; then
    echo "ERROR: Resume checkpoint not found: $RESUME_FROM"
    exit 1
fi

echo "Checkpoint contents:"
ls -la $RESUME_FROM
echo "Checkpoint size:"
du -sh $RESUME_FROM

# Verify checkpoint is complete (should be ~88GB)
CKPT_SIZE=$(du -s $RESUME_FROM | cut -f1)
if [ "$CKPT_SIZE" -lt 80000000 ]; then
    echo "WARNING: Checkpoint size seems small ($CKPT_SIZE KB). Expected ~88GB"
fi

# ============================================
# Run Training
# ============================================
srun --ntasks-per-node=1 bash -c '
    source /home/a5l/shuqing.a5l/miniconda3/bin/activate qwen3-eval
    source '"$PROJECT_DIR"'/train/env_config.sh

    export MASTER_ADDR='"$MASTER_ADDR"'
    export MASTER_PORT='"$MASTER_PORT"'
    export NODE_RANK=$SLURM_NODEID

    cd '"$PROJECT_DIR"'

    torchrun \
        --nnodes=$SLURM_NNODES \
        --node_rank=$SLURM_NODEID \
        --nproc_per_node=4 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        -m verl.trainer.fsdp_sft_trainer \
        --config-path='"$CONFIG_PATH"' \
        --config-name='"$CONFIG_NAME"' \
        trainer.experiment_name='"$EXPERIMENT_NAME"'
'
TRAIN_EXIT_CODE=$?

# ============================================
# Completion
# ============================================
echo "=========================================="
echo "Training Completed"
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "=========================================="

exit $TRAIN_EXIT_CODE
