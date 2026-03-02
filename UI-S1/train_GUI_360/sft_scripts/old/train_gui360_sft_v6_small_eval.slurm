#!/bin/bash
#SBATCH --job-name=gui360_sft_v6
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/gui360_sft_v6_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/gui360_sft_v6_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# GUI-360 SFT Training Script V6 - Small Eval
# ============================================
# Same as v4 but with 100-sample validation set for faster testing
# and fix for OmegaConf serialization issue in DCP checkpoint saving
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
EXPERIMENT_NAME="gui360_sft_v6_small_eval"
CONFIG_NAME="gui360_sft_v6_small_eval"
CHECKPOINT_DIR="$TRAIN_DIR/checkpoints/${EXPERIMENT_NAME}"

# ============================================
# Environment Setup
# ============================================
echo "=========================================="
echo "GUI-360 SFT Training V6 - Small Eval"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Project Dir: $PROJECT_DIR"
echo "Experiment: $EXPERIMENT_NAME"
echo "=========================================="
echo "Key features:"
echo "  - Same as v4 (A11y + Thinking)"
echo "  - Validation set: 100 samples (vs 26273)"
echo "  - LR: 1e-6"
echo "  - save_freq: 25"
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

# NCCL timeout settings
export NCCL_TIMEOUT=7200
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "Master node: $MASTER_ADDR"
echo "Total nodes: $SLURM_NNODES"
echo "Node rank: $NODE_RANK"
echo "World size: $WORLD_SIZE"

# ============================================
# Check Data
# ============================================
echo "=========================================="
echo "Checking Training Data"
echo "=========================================="

if [ ! -f "$TRAIN_PARQUET" ]; then
    echo "ERROR: Training data not found: $TRAIN_PARQUET"
    exit 1
fi

if [ ! -f "$EVAL_PARQUET" ]; then
    echo "ERROR: Eval data not found: $EVAL_PARQUET"
    exit 1
fi

python3 -c "
import pandas as pd
train = pd.read_parquet('$TRAIN_PARQUET')
eval_df = pd.read_parquet('$EVAL_PARQUET')
print(f'Training samples: {len(train)}')
print(f'Evaluation samples: {len(eval_df)}')
print(f'Columns: {train.columns.tolist()}')
print(f'enable_thinking: {train[\"enable_thinking\"].all() if \"enable_thinking\" in train.columns else \"N/A\"}')
"

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
