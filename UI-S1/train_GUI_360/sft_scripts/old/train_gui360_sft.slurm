#!/bin/bash
#SBATCH --job-name=gui360_sft
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/gui360_sft_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/gui360_sft_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# GUI-360 SFT Training Script for Qwen2.5-VL
# ============================================
# This script trains a vision-language model on GUI-360 dataset
# for computer-using agent tasks (Word, Excel, PowerPoint)
# ============================================

set -e

# ============================================
# Configuration
# ============================================
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRAIN_DIR="$PROJECT_DIR/train_GUI_360"
CONFIG_PATH="$TRAIN_DIR/config"
LOG_DIR="$TRAIN_DIR/logs"
# CHECKPOINT_DIR is set after EXPERIMENT_NAME below

# GUI-360 Dataset paths
GUI360_DIR="$PROJECT_DIR/datasets/GUI-360"
PROCESSED_DATA_DIR="$GUI360_DIR/processed_data"

# Training data paths (will be created if not exist)
TRAIN_PARQUET="$TRAIN_DIR/data/gui360_train_sft.parquet"
EVAL_PARQUET="$TRAIN_DIR/data/gui360_eval_sft.parquet"

# Experiment settings
EXPERIMENT_NAME="gui360_sft_qwenvl_matched_format"
CONFIG_NAME="gui360_sft_${SLURM_JOB_ID}"  # Unique config per job to avoid conflicts

# Use experiment name in checkpoint directory to avoid overlap
CHECKPOINT_DIR="$TRAIN_DIR/checkpoints/${EXPERIMENT_NAME}"

# ============================================
# Environment Setup
# ============================================
echo "=========================================="
echo "GUI-360 SFT Training"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Project Dir: $PROJECT_DIR"
echo "Experiment: $EXPERIMENT_NAME"
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
mkdir -p $TRAIN_DIR/data
mkdir -p $CONFIG_PATH

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
# Data Preparation (if needed)
# ============================================
prepare_data() {
    echo "=========================================="
    echo "Checking/Preparing Training Data"
    echo "=========================================="

    # Check if processed parquet files exist
    if [ ! -f "$TRAIN_PARQUET" ]; then
        echo "Training parquet not found. Preparing data..."

        if [ -f "$PROCESSED_DATA_DIR/action_prediction_train_resize/training_data.json" ]; then
            echo "Using pre-processed GUI-360 action prediction data..."
            python $PROJECT_DIR/scripts/GUI_360/prepare_gui360_sft_parquet.py \
                --input "$PROCESSED_DATA_DIR/action_prediction_train_resize/training_data.json" \
                --output "$TRAIN_PARQUET" \
                --image-base-dir "$PROCESSED_DATA_DIR/action_prediction_train_resize"
        else
            echo "ERROR: No GUI-360 training data found!"
            echo "Please run: python scripts/GUI_360/prepare_gui360_sft_parquet.py"
            exit 1
        fi
    else
        echo "Training parquet found: $TRAIN_PARQUET"
    fi

    # Check/create eval data
    if [ ! -f "$EVAL_PARQUET" ]; then
        echo "Evaluation parquet not found. Preparing..."

        if [ -d "$GUI360_DIR/test/data" ]; then
            python $PROJECT_DIR/scripts/GUI_360/prepare_gui360_eval_parquet.py \
                --input-dir "$GUI360_DIR/test/data" \
                --output "$EVAL_PARQUET"
        else
            echo "ERROR: No GUI-360 test data found!"
            echo "Please run: python scripts/GUI_360/prepare_gui360_eval_parquet.py"
            exit 1
        fi
    else
        echo "Evaluation parquet found: $EVAL_PARQUET"
    fi

    echo "Data preparation complete."
}

# ============================================
# Generate Config File
# ============================================
generate_config() {
    echo "=========================================="
    echo "Generating Training Configuration"
    echo "=========================================="

    # Calculate steps per epoch dynamically
    # With 4 nodes × 4 GPUs = 16 GPUs, micro_batch=2 → effective batch per step = 32
    # Use gradient accumulation to reach target batch size
    BATCH_SIZE=256  # Global batch size (will use grad accumulation: 256/32=8 steps)
    MICRO_BATCH_SIZE=2  # Per GPU micro batch size
    NUM_SAMPLES=$(python -c "import pandas as pd; print(len(pd.read_parquet('$TRAIN_PARQUET')))")
    STEPS_PER_EPOCH=$((NUM_SAMPLES / BATCH_SIZE))
    TEST_FREQ=$((STEPS_PER_EPOCH / 4))  # Validate 4 times per epoch

    echo "Training samples: $NUM_SAMPLES"
    echo "Batch size: $BATCH_SIZE"
    echo "Micro batch size per GPU: $MICRO_BATCH_SIZE"
    echo "Steps per epoch: $STEPS_PER_EPOCH"
    echo "Save frequency: $STEPS_PER_EPOCH (every epoch)"
    echo "Test frequency: $TEST_FREQ (4x per epoch)"

    cat > "$CONFIG_PATH/$CONFIG_NAME.yaml" << CONFIGEOF
hydra:
  searchpath:
    - file:///scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/verl/trainer/config

defaults:
  - sft_trainer
  - _self_

# ============================================
# GUI-360 SFT Configuration
# ============================================

# Data configuration
data:
  train_files: $TRAIN_PARQUET
  val_files: $EVAL_PARQUET
  train_batch_size: $BATCH_SIZE
  micro_batch_size_per_gpu: $MICRO_BATCH_SIZE
  max_length: 8192
  truncation: error
  balance_dp_token: False
  multiturn:
    enable: true
    messages_key: messages
    tools_key: tools
    enable_thinking_key: enable_thinking
  custom_cls:
    path: $PROJECT_DIR/verl/utils/dataset/gui_multiturn_sft_dataset.py
    name: GUIMultiTurnSFTDataset
  trust_remote_code: true

# Model configuration - Full fine-tuning with FSDP
model:
  partial_pretrain: $PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct
  strategy: fsdp2
  fsdp_config:
    model_dtype: fp32
    wrap_policy:
      min_num_params: 0
    cpu_offload: False
    offload_params: False
  enable_gradient_checkpointing: True
  trust_remote_code: True
  lora_rank: 0
  lora_alpha: 16
  target_modules: all-linear
  use_liger: False
  external_lib: null
  use_shm: False

# Optimizer configuration - Balanced for GUI-360
optim:
  lr: 2e-6
  betas: [0.9, 0.95]
  weight_decay: 0.1
  warmup_steps_ratio: 0.05
  clip_grad: 0.5
  lr_scheduler: cosine

# Training configuration
trainer:
  default_local_dir: $CHECKPOINT_DIR
  default_hdfs_dir: null
  resume_path: null
  project_name: gui360-sft
  experiment_name: gui360_sft_qwenvl
  total_epochs: 5
  total_training_steps: null
  logger: ['console', 'wandb']
  seed: 42
  save_freq: $STEPS_PER_EPOCH
  test_freq: $TEST_FREQ
  nnodes: $SLURM_NNODES
  n_gpus_per_node: 4
  max_ckpt_to_keep: 5

# Sequence parallelism
ulysses_sequence_parallel_size: 1
use_remove_padding: False
CONFIGEOF

    echo "Config file created: $CONFIG_PATH/$CONFIG_NAME.yaml"
}

# ============================================
# Main Training
# ============================================
run_training() {
    echo "=========================================="
    echo "Starting SFT Training (Multi-Node)"
    echo "=========================================="
    echo "Config: $CONFIG_NAME"
    echo "Nodes: $SLURM_NNODES"
    echo "GPUs per node: 4"
    echo "Total GPUs: $WORLD_SIZE"
    echo "=========================================="

    cd $PROJECT_DIR

    # Launch training with torchrun (multi-node)
    torchrun \
        --nnodes=$SLURM_NNODES \
        --node_rank=$NODE_RANK \
        --nproc_per_node=4 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        -m verl.trainer.fsdp_sft_trainer \
        --config-path=$CONFIG_PATH \
        --config-name=$CONFIG_NAME \
        trainer.experiment_name=$EXPERIMENT_NAME \
        2>&1 | tee $LOG_DIR/training_${SLURM_JOB_ID}_node${NODE_RANK}.log

    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    return $TRAIN_EXIT_CODE
}

# ============================================
# Execute Pipeline
# ============================================

# Step 1: Prepare data (only on master node)
if [ "$NODE_RANK" == "0" ] || [ -z "$NODE_RANK" ]; then
    prepare_data
    generate_config
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

exit $TRAIN_EXIT_CODE
