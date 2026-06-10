#!/bin/bash
#SBATCH --job-name=har_sft_r
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/related_work/har/output/logs/train_har_sft_resume_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/related_work/har/output/logs/train_har_sft_resume_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# HAR SFT Resume from checkpoint-100
# ============================================
# Starts from checkpoint-100 weights, trains 3 more epochs
# (original was 4 epochs, ckpt-100 = ~0.37 epoch = ~1 epoch equivalent)
# Total effective training: ~4 epochs from base model
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CONFIG_FILE="$PROJECT_DIR/related_work/har/scripts/qwen25vl_har_sft_resume.yaml"
LOG_DIR="$PROJECT_DIR/related_work/har/output/logs"

echo "=========================================="
echo "HAR SFT Resume Training"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

if [ -f "$PROJECT_DIR/train/env_config.sh" ]; then
    source "$PROJECT_DIR/train/env_config.sh"
fi

mkdir -p "$LOG_DIR"
mkdir -p "$PROJECT_DIR/related_work/har/output/har_sft_resume"

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200
export TORCH_GLOO_TIMEOUT_SECONDS=7200

echo "Master node: $MASTER_ADDR"
echo "Config file: $CONFIG_FILE"
echo "Model: har_sft_checkpoint100 (continuing from step 100)"

# Ensure HAR data exists
TRAIN_DATA="$PROJECT_DIR/related_work/har/data/gui360_har_train.jsonl"
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found at $TRAIN_DATA"
    echo "Run data preparation first."
    exit 1
fi
echo "Train data: $(wc -l < $TRAIN_DATA) samples"

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

echo "=========================================="
echo "HAR SFT Resume Training Completed"
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "Output dir: $PROJECT_DIR/related_work/har/output/har_sft_resume"
echo "=========================================="

exit $TRAIN_EXIT_CODE
