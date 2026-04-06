#!/bin/bash
#SBATCH --job-name=lora_baseline_r32
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --output=scripts/eval/logs/lora_baseline_r32_%j.out
#SBATCH --error=scripts/eval/logs/lora_baseline_r32_%j.err

# ============================================================================
# LoRA Baseline: Same setup as bind_aux_v2 but bind_weight=0.0
#
# L_total = L_act (no L_bind)
# 4 nodes × 4 GPUs = 16 GPUs
# Effective batch size: 2 × 16 GPUs × 4 grad_accum = 128
# ============================================================================

set -uo pipefail

# Paths
export PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
MODEL_PATH="${PROJECT_ROOT}/checkpoints/Qwen2.5-VL-7B-Instruct"
TRAIN_DATA="${PROJECT_ROOT}/train_GUI_360/llamafactory/data/gui360_train.json"
VAL_DATA="${PROJECT_ROOT}/train_GUI_360/llamafactory/data/gui360_val.json"
OUTPUT_DIR="${PROJECT_ROOT}/train_GUI_360/llamafactory/output/lora_baseline_r32"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${PROJECT_ROOT}/scripts/eval/logs"

# Environment
export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export PYTHONUNBUFFERED=1

# CUDA setup
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

if [ -f "${PROJECT_ROOT}/train/env_config.sh" ]; then
    source "${PROJECT_ROOT}/train/env_config.sh"
fi

# Multi-node setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200
export TORCH_GLOO_TIMEOUT_SECONDS=7200

# Wandb
export WANDB_PROJECT=gui360-bind-aux
export WANDB_RUN_NAME=lora_baseline_r32

cd "${PROJECT_ROOT}"

echo "================================================================"
echo "LoRA Baseline Training (bind_weight=0.0)"
echo "================================================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Nodes:     ${SLURM_NNODES}"
echo "Master:    ${MASTER_ADDR}"
echo "Model:     ${MODEL_PATH}"
echo "Train:     ${TRAIN_DATA}"
echo "Output:    ${OUTPUT_DIR}"
echo "GPUs:      4 × ${SLURM_NNODES} nodes = $((4 * SLURM_NNODES))"
echo "Batch:     2 × $((4 * SLURM_NNODES)) GPUs × 4 grad_accum = $((2 * 4 * SLURM_NNODES * 4)) effective"
echo "Lambda:    0.0 (pure SFT baseline)"
echo "LoRA:      r=32, α=64"
echo "Start:     $(date)"
echo "================================================================"
echo ""

srun --ntasks-per-node=1 bash -c '
    export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
    export CONDA_DEFAULT_ENV=qwen3-eval
    export PYTHONUNBUFFERED=1

    export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
    export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

    if [ -f '"${PROJECT_ROOT}"'/train/env_config.sh ]; then
        source '"${PROJECT_ROOT}"'/train/env_config.sh
    fi

    export MASTER_ADDR='"${MASTER_ADDR}"'
    export MASTER_PORT='"${MASTER_PORT}"'
    export NODE_RANK=$SLURM_NODEID

    export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_${SLURM_JOB_ID}
    export HF_HOME=/tmp/hf_home_${SLURM_JOB_ID}
    mkdir -p $HF_DATASETS_CACHE $HF_HOME

    cd '"${PROJECT_ROOT}"'

    torchrun \
        --nproc_per_node=4 \
        --nnodes=$SLURM_NNODES \
        --node_rank=$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        evaluation/bind_auxiliary_train.py \
        --model_path '"${MODEL_PATH}"' \
        --train_data '"${TRAIN_DATA}"' \
        --val_data '"${VAL_DATA}"' \
        --output_dir '"${OUTPUT_DIR}"' \
        --lora_rank 32 \
        --lora_alpha 64 \
        --bind_weight 0.0 \
        --per_device_batch_size 2 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1.5e-5 \
        --num_epochs 1.0 \
        --save_steps 200 \
        --eval_steps 200 \
        --logging_steps 1
'
TRAIN_EXIT_CODE=$?

echo ""
echo "================================================================"
echo "Training complete!"
echo "Exit code: ${TRAIN_EXIT_CODE}"
echo "Output: ${OUTPUT_DIR}"
echo "End:    $(date)"
echo "================================================================"

exit $TRAIN_EXIT_CODE
