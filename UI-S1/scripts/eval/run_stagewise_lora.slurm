#!/bin/bash
#SBATCH --job-name=sw_lora
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/sw_lora_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/sw_lora_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

# Required: CONDITION=A|B|C
CONDITION="${CONDITION:?'Set CONDITION=A|B|C'}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct}"
MAX_TRAIN="${MAX_TRAIN:-0}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_DIR/evaluation/results/stagewise_lora_${CONDITION}_${TIMESTAMP}"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200
export TORCH_GLOO_TIMEOUT_SECONDS=7200

echo "============================================================"
echo "Stage-wise LoRA Training: Condition ${CONDITION}"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NNODES} x 4 GPUs = $((SLURM_NNODES * 4)) total GPUs"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Start: $(date)"
echo "Model: $MODEL_PATH"
echo "Max train samples: $MAX_TRAIN"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

srun --ntasks-per-node=1 bash -c '
    export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
    export CONDA_DEFAULT_ENV=qwen3-eval
    export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
    export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

    export MASTER_ADDR='"$MASTER_ADDR"'
    export MASTER_PORT='"$MASTER_PORT"'
    export NODE_RANK=$SLURM_NODEID
    export PYTHONUNBUFFERED=1

    export TMPDIR=/tmp/${SLURM_JOB_ID}
    mkdir -p $TMPDIR 2>/dev/null || true
    if ! touch "$TMPDIR/.test_write" 2>/dev/null; then
        export TMPDIR=/scratch/a5l/shuqing.a5l/tmp/${SLURM_JOB_ID}
        mkdir -p $TMPDIR
    fi
    rm -f "$TMPDIR/.test_write" 2>/dev/null
    export TORCHINDUCTOR_CACHE_DIR=$TMPDIR/torchinductor
    export TORCH_HOME=$TMPDIR/torch
    export XDG_CACHE_HOME=$TMPDIR/cache
    mkdir -p $TORCHINDUCTOR_CACHE_DIR $TORCH_HOME $XDG_CACHE_HOME

    cd '"$PROJECT_DIR"'

    torchrun \
        --nproc_per_node=4 \
        --nnodes=$SLURM_NNODES \
        --node_rank=$SLURM_NODEID \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        evaluation/stagewise_lora_train.py \
        --condition '"$CONDITION"' \
        --model_path '"$MODEL_PATH"' \
        --output_dir '"$OUTPUT_DIR"' \
        --max_train_samples '"$MAX_TRAIN"' \
        --num_epochs 1.0 \
        --per_device_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1.5e-5 \
        --logging_steps 1 \
        --grad_record_every 50 \
        --save_steps 200 \
        --eval_steps 200
'

echo "Done at $(date) | Output: $OUTPUT_DIR"
