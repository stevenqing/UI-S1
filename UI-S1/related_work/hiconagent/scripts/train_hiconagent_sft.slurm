#!/bin/bash
#SBATCH --job-name=hicon_sft
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/related_work/hiconagent/output/logs/train_hiconagent_sft_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/related_work/hiconagent/output/logs/train_hiconagent_sft_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# HiconAgent-Style SFT Training on GUI-360
# ============================================
# Base model: Qwen2.5-VL-7B-Instruct
# Data: GUI-360 in HiconAgent format (multi-image visual history + <think>/<action>)
# Method: Full-parameter fine-tuning with DeepSpeed ZeRO-3
# Note: cutoff_len=12288 to accommodate up to 3 images per sample
# ============================================

set -e

# ============================================
# Configuration
# ============================================
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
LLAMA_FACTORY_DIR="$PROJECT_DIR/train_GUI_360/LLaMA-Factory"
CONFIG_FILE="$PROJECT_DIR/related_work/hiconagent/scripts/qwen25vl_hiconagent_sft.yaml"
LOG_DIR="$PROJECT_DIR/related_work/hiconagent/output/logs"

# ============================================
# Environment Setup
# ============================================
echo "=========================================="
echo "HiconAgent-Style SFT Training on GUI-360"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: 4"
echo "=========================================="

# Activate conda environment
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

# Source environment configuration
if [ -f "$PROJECT_DIR/train/env_config.sh" ]; then
    source "$PROJECT_DIR/train/env_config.sh"
fi

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$PROJECT_DIR/related_work/hiconagent/output/hiconagent_sft"

# CUDA setup
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200
export TORCH_GLOO_TIMEOUT_SECONDS=7200

echo "Master node: $MASTER_ADDR"
echo "Config file: $CONFIG_FILE"

# ============================================
# Step 1: Prepare HiconAgent-format data (master only)
# ============================================
if [ "$SLURM_NODEID" = "0" ]; then
    echo "=========================================="
    echo "Preparing HiconAgent-format training data..."
    echo "=========================================="

    TRAIN_DATA="$PROJECT_DIR/related_work/hiconagent/data/gui360_hiconagent_train.jsonl"
    VAL_DATA="$PROJECT_DIR/related_work/hiconagent/data/gui360_hiconagent_val.jsonl"

    if [ ! -f "$TRAIN_DATA" ]; then
        cd "$PROJECT_DIR"

        # Convert training data
        python related_work/hiconagent/prepare_hiconagent_sft_data.py \
            --input_file v15_gui_360/data/gui360_balanced_train.jsonl \
            --output_file "$TRAIN_DATA" \
            --tau_history 2

        # Convert validation data
        python related_work/hiconagent/prepare_hiconagent_sft_data.py \
            --input_file v15_gui_360/data/gui360_balanced_val.jsonl \
            --output_file "$VAL_DATA" \
            --tau_history 2

        echo "HiconAgent data preparation done."
        echo "  Train: $(wc -l < $TRAIN_DATA) samples"
        echo "  Val:   $(wc -l < $VAL_DATA) samples"
    else
        echo "HiconAgent data already prepared."
        echo "  Train: $(wc -l < $TRAIN_DATA) samples"
        echo "  Val:   $(wc -l < $VAL_DATA) samples"
    fi
fi

# Wait for data preparation
sleep 10

# ============================================
# Step 2: Run Training
# ============================================
echo "=========================================="
echo "Starting HiconAgent SFT training..."
echo "=========================================="

cd "$PROJECT_DIR"

srun --ntasks-per-node=1 bash -c '
    export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
    export CONDA_DEFAULT_ENV=qwen3-eval
    export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
    export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    if [ -f '"$PROJECT_DIR"'/train/env_config.sh ]; then
        source '"$PROJECT_DIR"'/train/env_config.sh
    fi

    export MASTER_ADDR='"$MASTER_ADDR"'
    export MASTER_PORT='"$MASTER_PORT"'
    export NODE_RANK=$SLURM_NODEID

    # Use node-local /tmp for HF datasets cache to avoid NFS mmap Bus errors
    export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_${SLURM_JOB_ID}
    export HF_HOME=/tmp/hf_home_${SLURM_JOB_ID}
    mkdir -p $HF_DATASETS_CACHE $HF_HOME

    cd '"$PROJECT_DIR"'

    FORCE_TORCHRUN=1 \
    NNODES=$SLURM_NNODES \
    NODE_RANK=$SLURM_NODEID \
    MASTER_ADDR=$MASTER_ADDR \
    MASTER_PORT=$MASTER_PORT \
    NPROC_PER_NODE=4 \
    llamafactory-cli train '"$CONFIG_FILE"'
'
TRAIN_EXIT_CODE=$?

# ============================================
# Completion
# ============================================
echo "=========================================="
echo "HiconAgent SFT Training Completed"
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "Output dir: $PROJECT_DIR/related_work/hiconagent/output/hiconagent_sft"
echo "=========================================="

exit $TRAIN_EXIT_CODE
