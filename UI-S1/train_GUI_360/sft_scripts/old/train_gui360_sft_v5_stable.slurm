#!/bin/bash
#SBATCH --job-name=gui360_sft_v5
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/gui360_sft_v5_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/gui360_sft_v5_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# GUI-360 SFT Training Script V5 - Stable Multi-Node
# ============================================
# Key improvements for multi-node stability:
#
# 1. Sequence Parallelism (SP):
#    - ulysses_sequence_parallel_size=4 (within node)
#    - Reduces cross-node communication significantly
#    - Most tensor operations happen locally
#
# 2. bfloat16 precision:
#    - 50% less communication data vs fp32
#    - Still maintains good numerical stability
#
# 3. Larger micro_batch_size:
#    - micro_batch_size_per_gpu=2
#    - Fewer gradient synchronization steps
#
# 4. 2 nodes (8 GPUs):
#    - Simpler communication topology
#    - Less chance of network issues
#
# Reference: NCCL multi-node best practices
# ============================================

set -e

# ============================================
# Configuration
# ============================================
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
CONFIG_PATH="$TRAIN_DIR/config"
LOG_DIR="$TRAIN_DIR/logs"

# Training data paths (A11y + Thinking)
TRAIN_PARQUET="$TRAIN_DIR/data/gui360_train_sft_a11y_thinking.parquet"
EVAL_PARQUET="$TRAIN_DIR/data/gui360_eval_sft_a11y_thinking.parquet"

# Experiment settings
EXPERIMENT_NAME="gui360_sft_v5_stable"
CONFIG_NAME="gui360_sft_v5_stable"
CHECKPOINT_DIR="$TRAIN_DIR/checkpoints/${EXPERIMENT_NAME}"

# ============================================
# Environment Setup
# ============================================
echo "=========================================="
echo "GUI-360 SFT Training V5 - Stable Multi-Node"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Project Dir: $PROJECT_DIR"
echo "Experiment: $EXPERIMENT_NAME"
echo "=========================================="
echo "Key stability improvements:"
echo "  - Sequence Parallelism: 4 (within node)"
echo "  - Precision: bfloat16 (vs fp32)"
echo "  - Micro batch size: 2 (vs 1)"
echo "  - Nodes: 2 (vs 4)"
echo "=========================================="
echo "Expected benefits:"
echo "  - Reduced cross-node communication"
echo "  - Fewer gradient sync operations"
echo "  - More stable multi-node training"
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
export WORLD_SIZE=$((SLURM_NNODES * 4))
export NODE_RANK=$SLURM_NODEID

# ============================================
# NCCL Communication Optimization
# ============================================
# Enable async error handling to detect hangs
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=INFO

# IB optimization (if available)
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5

# Socket optimization
export NCCL_SOCKET_IFNAME=^docker0,lo

# Blocking wait for better error detection
export NCCL_BLOCKING_WAIT=1

echo "Master node: $MASTER_ADDR"
echo "Total nodes: $SLURM_NNODES"
echo "Node rank: $NODE_RANK"
echo "World size: $WORLD_SIZE"

# ============================================
# Data Preparation
# ============================================
prepare_data() {
    echo "=========================================="
    echo "Checking Training Data"
    echo "=========================================="

    if [ -f "$TRAIN_PARQUET" ]; then
        echo "Training data exists: $TRAIN_PARQUET"
    else
        echo "ERROR: Training data not found: $TRAIN_PARQUET"
        exit 1
    fi

    if [ -f "$EVAL_PARQUET" ]; then
        echo "Eval data exists: $EVAL_PARQUET"
    else
        echo "ERROR: Eval data not found: $EVAL_PARQUET"
        exit 1
    fi

    # Print data stats
    python3 -c "
import pandas as pd
train = pd.read_parquet('$TRAIN_PARQUET')
eval_df = pd.read_parquet('$EVAL_PARQUET')
print(f'Training samples: {len(train)}')
print(f'Evaluation samples: {len(eval_df)}')
print(f'Columns: {train.columns.tolist()}')
print(f'enable_thinking: {train[\"enable_thinking\"].all() if \"enable_thinking\" in train.columns else \"N/A\"}')
"
}

# ============================================
# Execute Pipeline
# ============================================

# Step 1: Prepare data (only on master node)
if [ "$NODE_RANK" == "0" ] || [ -z "$NODE_RANK" ]; then
    prepare_data
fi

# Wait for all nodes to be ready
sleep 10

# Step 2: Run training on all nodes via srun
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
echo "Logs: $LOG_DIR"
echo "=========================================="

exit $TRAIN_EXIT_CODE
