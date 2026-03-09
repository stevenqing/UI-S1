#!/bin/bash
#SBATCH --job-name=moe_sft_v3
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/moe_sft_v3_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/moe_sft_v3_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# MoE SFT v3 Training — Context-Aware Router
# ============================================
# Key changes from v2:
#   - router_type: text_only → context_aware (step-level dynamic routing)
#   - top_k: 6 → 1 (hard routing for vLLM multi-LoRA compatibility)
#   - Train from scratch (no copy-init, ContextAwareRouter has different params)
#   - balance_weight=0.1 (entropy), z_loss_weight=0.01 (same as v2)
# ============================================

set -e

# ============================================
# Configuration
# ============================================
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CONFIG_FILE="$PROJECT_DIR/train_GUI_360/moe_sft/moe_sft_config_v3.yaml"
TRAIN_SCRIPT="$PROJECT_DIR/train_GUI_360/moe_sft/train_moe_sft.py"
LOG_DIR="$PROJECT_DIR/train_GUI_360/logs"

# ============================================
# Environment Setup
# ============================================
echo "=========================================="
echo "MoE SFT v3 Training (Context-Aware Router)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: 4"
echo "Config: $CONFIG_FILE"
echo "Training from scratch (no init checkpoint)"
echo "=========================================="

# Activate conda environment
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1

# Source environment configuration
if [ -f "$PROJECT_DIR/train/env_config.sh" ]; then
    source "$PROJECT_DIR/train/env_config.sh"
fi

# Create log directory
mkdir -p "$LOG_DIR"

# CUDA setup
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set distributed environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200
export TORCH_GLOO_TIMEOUT_SECONDS=7200

echo "Master node: $MASTER_ADDR"

# ============================================
# Run Training
# ============================================
echo "=========================================="
echo "Starting MoE SFT v3 training..."
echo "=========================================="

cd "$PROJECT_DIR"

srun --ntasks-per-node=1 bash -c '
    export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/ui-s1/bin:$PATH"
    export CONDA_DEFAULT_ENV=ui-s1
    export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
    export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    if [ -f '"$PROJECT_DIR"'/train/env_config.sh ]; then
        source '"$PROJECT_DIR"'/train/env_config.sh
    fi

    export MASTER_ADDR='"$MASTER_ADDR"'
    export MASTER_PORT='"$MASTER_PORT"'
    export NODE_RANK=$SLURM_NODEID

    # Use node-local /tmp for HF datasets cache
    export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_${SLURM_JOB_ID}
    export HF_HOME=/tmp/hf_home_${SLURM_JOB_ID}
    mkdir -p $HF_DATASETS_CACHE $HF_HOME

    cd '"$PROJECT_DIR"'

    torchrun \
        --nnodes=$SLURM_NNODES \
        --node_rank=$SLURM_NODEID \
        --nproc_per_node=4 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        '"$TRAIN_SCRIPT"' \
        --config '"$CONFIG_FILE"'
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
echo "Output dir: $PROJECT_DIR/train_GUI_360/moe_sft/output/moe_sft_v3"
echo "=========================================="

exit $TRAIN_EXIT_CODE
