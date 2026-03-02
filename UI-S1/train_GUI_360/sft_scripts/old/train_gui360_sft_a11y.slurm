#!/bin/bash
#SBATCH --job-name=gui360_sft_a11y
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/gui360_sft_a11y_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/gui360_sft_a11y_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# GUI-360 SFT Training with A11y Support
# ============================================
# This script trains with accessibility information:
# - SoM annotated screenshots
# - Element list in prompt
# - element_id based actions
# Expected improvement: ~50% -> ~82% grounding accuracy
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

# Training data paths
TRAIN_PARQUET="$TRAIN_DIR/data/gui360_train_sft_a11y.parquet"
EVAL_PARQUET="$TRAIN_DIR/data/gui360_eval_sft_a11y.parquet"

# Experiment settings
EXPERIMENT_NAME="gui360_sft_a11y_qwenvl"
CONFIG_NAME="gui360_sft_a11y_${SLURM_JOB_ID}"
CHECKPOINT_DIR="$TRAIN_DIR/checkpoints/${EXPERIMENT_NAME}"

# ============================================
# Environment Setup
# ============================================
echo "=========================================="
echo "GUI-360 SFT Training with A11y Support"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Experiment: $EXPERIMENT_NAME"
echo "=========================================="

source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

if [ -f "$PROJECT_DIR/train/env_config.sh" ]; then
    source $PROJECT_DIR/train/env_config.sh
fi

mkdir -p $LOG_DIR
mkdir -p $CHECKPOINT_DIR
mkdir -p $TRAIN_DIR/data
mkdir -p $CONFIG_PATH

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export WORLD_SIZE=$((SLURM_NNODES * 4))
export NODE_RANK=$SLURM_NODEID

echo "Master node: $MASTER_ADDR"
echo "Total nodes: $SLURM_NNODES"

# ============================================
# Data Preparation
# ============================================
prepare_data() {
    echo "=========================================="
    echo "Preparing A11y Training Data"
    echo "=========================================="

    if [ ! -f "$TRAIN_PARQUET" ]; then
        echo "Generating A11y training data from raw trajectories..."

        python $PROJECT_DIR/scripts/GUI_360/prepare_gui360_sft_parquet_a11y.py \
            --input-dir "$GUI360_DIR/train/data" \
            --output "$TRAIN_PARQUET" \
            --image-base-dir "$GUI360_DIR/train/images" \
            --apps word excel ppt

        echo "Training data generated: $TRAIN_PARQUET"
    else
        echo "Training data exists: $TRAIN_PARQUET"
    fi

    if [ ! -f "$EVAL_PARQUET" ]; then
        echo "Generating A11y evaluation data..."

        python $PROJECT_DIR/scripts/GUI_360/prepare_gui360_sft_parquet_a11y.py \
            --input-dir "$GUI360_DIR/test/data" \
            --output "$EVAL_PARQUET" \
            --image-base-dir "$GUI360_DIR/test/images" \
            --apps word excel ppt

        echo "Eval data generated: $EVAL_PARQUET"
    else
        echo "Eval data exists: $EVAL_PARQUET"
    fi
}

# ============================================
# Generate Config
# ============================================
generate_config() {
    echo "=========================================="
    echo "Generating Training Configuration"
    echo "=========================================="

    BATCH_SIZE=256
    MICRO_BATCH_SIZE=2
    NUM_SAMPLES=$(python -c "import pandas as pd; print(len(pd.read_parquet('$TRAIN_PARQUET')))")
    STEPS_PER_EPOCH=$((NUM_SAMPLES / BATCH_SIZE))
    TEST_FREQ=$((STEPS_PER_EPOCH / 4))

    echo "Training samples: $NUM_SAMPLES"
    echo "Steps per epoch: $STEPS_PER_EPOCH"

    cat > "$CONFIG_PATH/$CONFIG_NAME.yaml" << CONFIGEOF
hydra:
  searchpath:
    - file:///scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/verl/trainer/config

defaults:
  - sft_trainer
  - _self_

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

# Model configuration
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
  use_liger: False

# Optimizer configuration
optim:
  lr: 1e-5
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
  project_name: gui360-sft-a11y
  experiment_name: $EXPERIMENT_NAME
  total_epochs: 3
  total_training_steps: null
  logger: ['console', 'wandb']
  seed: 42
  save_freq: $STEPS_PER_EPOCH
  test_freq: $TEST_FREQ
  nnodes: $SLURM_NNODES
  n_gpus_per_node: 4
  max_ckpt_to_keep: 5

ulysses_sequence_parallel_size: 1
use_remove_padding: False
CONFIGEOF

    echo "Config created: $CONFIG_PATH/$CONFIG_NAME.yaml"
}

# ============================================
# Execute
# ============================================
if [ "$NODE_RANK" == "0" ] || [ -z "$NODE_RANK" ]; then
    prepare_data
    generate_config
fi

sleep 5

# Run training
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

echo "=========================================="
echo "Training Completed"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "=========================================="

exit $TRAIN_EXIT_CODE
