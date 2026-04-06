#!/bin/bash
#SBATCH --job-name=ss_p${P_AUTO}
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp3/logs/ss_p${P_AUTO}_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp3/logs/ss_p${P_AUTO}_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Exp 1: Scheduled Sampling SFT Training
# Tests: Does teacher forcing cause fragmentation?
# p_auto=0.3 or 0.7 (set via environment variable)
# Full-parameter SFT from Qwen2.5-VL-7B-Instruct
# ============================================

set -e

# Default p_auto if not set
P_AUTO=${P_AUTO:-0.3}

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
LOG_DIR="$PROJECT_DIR/scripts/exp3/logs"
OUTPUT_DIR="$PROJECT_DIR/train_GUI_360/llamafactory/output/scheduled_sampling_p${P_AUTO}"

echo "=========================================="
echo "Exp 1: Scheduled Sampling (p_auto=${P_AUTO})"
echo "=========================================="
echo "Job: ${SLURM_JOB_ID} | Node: $(hostname)"
echo "p_auto: ${P_AUTO}"
echo "Output: ${OUTPUT_DIR}"
echo "Start: $(date)"
echo "=========================================="

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

if [ -f "$PROJECT_DIR/train/env_config.sh" ]; then
    source "$PROJECT_DIR/train/env_config.sh"
fi

mkdir -p "$LOG_DIR"

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

    torchrun \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=4 \
        --node_rank=$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        evaluation/scheduled_sampling_train.py \
        --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
        --train_data train_GUI_360/llamafactory/data/gui360_mixed_train.json \
        --val_data train_GUI_360/llamafactory/data/gui360_mixed_val.json \
        --output_dir '"$OUTPUT_DIR"' \
        --p_auto '"$P_AUTO"' \
        --n_tf_tokens 3 \
        --num_epochs 2 \
        --per_device_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --learning_rate 1e-5 \
        --save_steps 50 \
        --eval_steps 100 \
        --logging_steps 10 \
        --deepspeed_config train_GUI_360/llamafactory/ds_z3_config.json
'
EXIT_CODE=$?

echo "=========================================="
echo "Exp 1: Scheduled Sampling Complete (p_auto=${P_AUTO})"
echo "End: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

exit $EXIT_CODE
