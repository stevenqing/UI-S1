#!/bin/bash
#SBATCH --job-name=v15_full_sft
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/train_balanced_full_sft_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/train_balanced_full_sft_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V15: Full Parameter SFT on Balanced Data (2000 episodes)
#
# Upper bound baseline for V13 RL comparison.
# Starts from base Qwen2.5-VL-7B-Instruct (same starting point).
# Trains on the same balanced data used by V13 RL.
#
# eff_bs = 4 nodes * 4 gpu * 1 * 16 = 256
# lr=1e-5, 4 epochs
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
LLAMA_FACTORY_DIR="$PROJECT_DIR/train_GUI_360/LLaMA-Factory"
CONFIG_FILE="$PROJECT_DIR/train_GUI_360/llamafactory/qwen25vl_gui360_balanced_full_sft.yaml"
LOG_DIR="$PROJECT_DIR/v15_gui_360/scripts/logs"
mkdir -p "$LOG_DIR"

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

if [ -f "$PROJECT_DIR/train/env_config.sh" ]; then
    source "$PROJECT_DIR/train/env_config.sh"
fi

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200
export TORCH_GLOO_TIMEOUT_SECONDS=7200

echo "=========================================="
echo "V15: Full Parameter SFT on Balanced Data"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}  Master: ${MASTER_ADDR}"
echo "Config:  ${CONFIG_FILE}"
echo "Start:   $(date)"
echo "=========================================="

cd "$PROJECT_DIR"

srun --ntasks-per-node=1 bash -c '
    export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
    export CONDA_DEFAULT_ENV=qwen3-eval
    export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
    export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export PYTHONPATH="'"$PROJECT_DIR"':'"$LLAMA_FACTORY_DIR"'/src:$PYTHONPATH"
    export PYTHONUNBUFFERED=1
    export WANDB_MODE=disabled

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
TRAIN_EXIT=$?

echo "=========================================="
echo "Done: $(date)   exit=$TRAIN_EXIT"
echo "Output: $PROJECT_DIR/train_GUI_360/llamafactory/output/gui360_balanced_full_sft"
echo "=========================================="
exit $TRAIN_EXIT
