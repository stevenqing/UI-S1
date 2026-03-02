#!/bin/bash
#SBATCH --job-name=gui360_sft_v4
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/gui360_sft_v4_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/gui360_sft_v4_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# GUI-360 SFT Training Script V4 - A11y + Thinking
# ============================================
# Combines the best of both approaches:
#
# From A11y (expected +45% grounding):
#   - SoM annotated screenshots with numbered labels
#   - Element list in prompt
#   - element_id based actions (not coordinates)
#
# From V3 Thinking (better format retention):
#   - Reasoning/thinking traces from raw data
#   - enable_thinking column
#   - Format: Reasoning: {thought}\n\n<tool_call>...</tool_call>
#
# Training optimizations (from v3 analysis):
#   - Lower LR (1e-6) - early checkpoints performed better
#   - Frequent saves (save_freq=25) for early stopping
#   - Increased max_length (12288) for element lists
#
# Reference:
#   - GUI-360 paper: A11y improves grounding 35%->82%
#   - V3 analysis: Best checkpoint at step 25
# ============================================

set -e

# ============================================
# Configuration
# ============================================
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
CONFIG_PATH="$TRAIN_DIR/config"
LOG_DIR="$TRAIN_DIR/logs"

# GUI-360 Dataset paths
GUI360_DIR="$PROJECT_DIR/datasets/GUI-360"

# Training data paths (A11y + Thinking)
TRAIN_PARQUET="$TRAIN_DIR/data/gui360_train_sft_a11y_thinking.parquet"
EVAL_PARQUET="$TRAIN_DIR/data/gui360_eval_sft_a11y_thinking.parquet"

# Experiment settings
EXPERIMENT_NAME="gui360_sft_v4_a11y_thinking"
CONFIG_NAME="gui360_sft_v4_a11y_thinking"
CHECKPOINT_DIR="$TRAIN_DIR/checkpoints/${EXPERIMENT_NAME}"

# ============================================
# Environment Setup
# ============================================
echo "=========================================="
echo "GUI-360 SFT Training V4 - A11y + Thinking"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Project Dir: $PROJECT_DIR"
echo "Experiment: $EXPERIMENT_NAME"
echo "=========================================="
echo "Key features:"
echo "  - A11y: SoM screenshots, element list, element_id actions"
echo "  - Thinking: Reasoning traces from raw data"
echo "  - LR: 1e-6 (lower, based on v3 analysis)"
echo "  - save_freq: 25 (early stopping support)"
echo "  - max_length: 12288 (for element lists)"
echo "=========================================="
echo "Expected improvements:"
echo "  - Grounding: 35% -> ~80% (from A11y)"
echo "  - Format retention: Better (from thinking)"
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

# NCCL timeout settings to prevent hanging during checkpoint save
export NCCL_TIMEOUT=7200  # 2 hours
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=INFO

echo "Master node: $MASTER_ADDR"
echo "Total nodes: $SLURM_NNODES"
echo "Node rank: $NODE_RANK"
echo "World size: $WORLD_SIZE"

# ============================================
# Data Preparation
# ============================================
prepare_data() {
    echo "=========================================="
    echo "Preparing A11y + Thinking Training Data"
    echo "=========================================="

    if [ ! -f "$TRAIN_PARQUET" ]; then
        echo "Generating training data with A11y + Thinking..."

        python $PROJECT_DIR/scripts/GUI_360/prepare_gui360_sft_a11y_with_thinking.py \
            --input-dir "$GUI360_DIR/train/data" \
            --output "$TRAIN_PARQUET" \
            --image-base-dir "$GUI360_DIR/train/images" \
            --apps word excel ppt

        echo "Training data generated: $TRAIN_PARQUET"
    else
        echo "Training data exists: $TRAIN_PARQUET"
    fi

    if [ ! -f "$EVAL_PARQUET" ]; then
        echo "Generating evaluation data with A11y + Thinking..."

        python $PROJECT_DIR/scripts/GUI_360/prepare_gui360_sft_a11y_with_thinking.py \
            --input-dir "$GUI360_DIR/test/data" \
            --output "$EVAL_PARQUET" \
            --image-base-dir "$GUI360_DIR/test/images" \
            --apps word excel ppt

        echo "Eval data generated: $EVAL_PARQUET"
    else
        echo "Eval data exists: $EVAL_PARQUET"
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
echo ""
echo "Next steps:"
echo "1. Check training logs: tail -f $LOG_DIR/gui360_sft_v4_${SLURM_JOB_ID}.log"
echo "2. List checkpoints: ls -la $CHECKPOINT_DIR"
echo "3. Run evaluation:"
echo "   cd $TRAIN_DIR/eval"
echo "   # Create evaluation scripts for v4 checkpoints"
echo "   # Expected significant improvement in grounding accuracy"
echo "=========================================="

exit $TRAIN_EXIT_CODE
