#!/bin/bash
#SBATCH --job-name=train_gui_sft
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_gui_sft_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_gui_sft_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# UI-S1 SFT Training Script for Qwen2.5-VL
# ============================================

set -e

# Environment setup
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CONFIG_PATH="$PROJECT_DIR/examples/qwen_gui_sft/config"
LOG_DIR="$PROJECT_DIR/train/logs"

# Activate conda environment
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
source $PROJECT_DIR/train/env_config.sh

# Create log directory
mkdir -p $LOG_DIR

# Training configuration
EXPERIMENT_NAME="qwenvl_uis1_full_v2_lowlr"
CONFIG_NAME="gui_sft_v2"  # Use improved config

echo "=========================================="
echo "Starting UI-S1 SFT Training"
echo "Project Dir: $PROJECT_DIR"
echo "Config: gui_sft"
echo "Experiment: $EXPERIMENT_NAME"
echo "Nodes: ${SLURM_NNODES}"
echo "GPUs per node: 4"
echo "Start time: $(date)"
echo "=========================================="

# Set distributed environment variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN

# Change to project directory
cd $PROJECT_DIR

# Launch training with torchrun
torchrun --nproc_per_node=4 --master_port=$MASTER_PORT \
    -m verl.trainer.fsdp_sft_trainer \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    2>&1 | tee $LOG_DIR/train_gui_sft_${SLURM_JOB_ID}.log

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo "=========================================="
echo "Training completed at $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

exit $TRAIN_EXIT_CODE
