#!/bin/bash
#SBATCH --job-name=gui360_sft_v3
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/gui360_sft_v3_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/gui360_sft_v3_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# GUI-360 SFT Training Script v3 - With Thinking
# ============================================
# Key changes from v2:
# - Training data includes reasoning/thinking from raw JSONL
# - Format: "Reasoning: {thought}\n\n<tool_call>...</tool_call>"
# - enable_thinking column set to True
# - Expected to reduce catastrophic forgetting
# - More frequent checkpointing (save_freq=25)
#
# Reference: train_GUI_360/data_preparation/ENABLE_THINKING_GUIDE.md
# ============================================

set -e

# ============================================
# Configuration
# ============================================
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
CONFIG_PATH="$TRAIN_DIR/config"
LOG_DIR="$TRAIN_DIR/logs"

# Training data paths (WITH THINKING)
TRAIN_PARQUET="$TRAIN_DIR/data/gui360_train_sft_with_thinking.parquet"
EVAL_PARQUET="$TRAIN_DIR/data/gui360_test_sft_with_thinking.parquet"

# Experiment settings
EXPERIMENT_NAME="gui360_sft_v3_thinking"
CONFIG_NAME="gui360_sft_v3_with_thinking"

# Checkpoint directory
CHECKPOINT_DIR="$TRAIN_DIR/checkpoints/${EXPERIMENT_NAME}"

# ============================================
# Environment Setup
# ============================================
echo "=========================================="
echo "GUI-360 SFT Training v3 - With Thinking"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Project Dir: $PROJECT_DIR"
echo "Experiment: $EXPERIMENT_NAME"
echo "=========================================="
echo "Key changes from v2:"
echo "  - Training data: WITH thinking/reasoning"
echo "  - Format: Reasoning: {thought} + <tool_call>"
echo "  - enable_thinking: True"
echo "  - save_freq: 25 (more checkpoints)"
echo "  - Expected: Better format retention"
echo "=========================================="
echo "Data files:"
echo "  - Train: $TRAIN_PARQUET"
echo "  - Eval: $EVAL_PARQUET"
echo "=========================================="

# Activate conda environment
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

# Source environment configuration
if [ -f "$PROJECT_DIR/train/env_config.sh" ]; then
    source $PROJECT_DIR/train/env_config.sh
fi

# Create directories
mkdir -p $LOG_DIR
mkdir -p $CHECKPOINT_DIR

# Set distributed environment variables for multi-node
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export WORLD_SIZE=$((SLURM_NNODES * 4))
export NODE_RANK=$SLURM_NODEID

echo "Master node: $MASTER_ADDR"
echo "Total nodes: $SLURM_NNODES"
echo "Node rank: $NODE_RANK"
echo "World size: $WORLD_SIZE"

# ============================================
# Verify Data Files
# ============================================
verify_data() {
    echo "=========================================="
    echo "Verifying Training Data"
    echo "=========================================="

    if [ ! -f "$TRAIN_PARQUET" ]; then
        echo "ERROR: Training parquet not found: $TRAIN_PARQUET"
        echo "Please run prepare_gui360_sft_with_thinking.py first!"
        exit 1
    fi

    if [ ! -f "$EVAL_PARQUET" ]; then
        echo "ERROR: Evaluation parquet not found: $EVAL_PARQUET"
        echo "Please run prepare_gui360_sft_with_thinking.py first!"
        exit 1
    fi

    # Print data info
    python3 -c "
import pandas as pd
train = pd.read_parquet('$TRAIN_PARQUET')
eval_df = pd.read_parquet('$EVAL_PARQUET')
print(f'Training samples: {len(train)}')
print(f'Evaluation samples: {len(eval_df)}')
print(f'Training columns: {train.columns.tolist()}')
print(f'enable_thinking: {train[\"enable_thinking\"].all() if \"enable_thinking\" in train.columns else \"N/A\"}')"

    echo "Data verification complete."
}

# ============================================
# Execute Pipeline
# ============================================

# Step 1: Verify data (only on master node)
if [ "$NODE_RANK" == "0" ] || [ -z "$NODE_RANK" ]; then
    verify_data
fi

# Wait for all nodes to be ready
sleep 5

# Step 2: Run training on all nodes via srun
srun --ntasks-per-node=1 bash -c '
    source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
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
echo ""
echo "Next steps:"
echo "1. Check training logs: tail -f $LOG_DIR/gui360_sft_v3_${SLURM_JOB_ID}.log"
echo "2. List checkpoints: ls -la $CHECKPOINT_DIR"
echo "3. Run evaluation: See train_GUI_360/eval/ for scripts"
echo "=========================================="

exit $TRAIN_EXIT_CODE
