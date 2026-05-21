#!/bin/bash
#SBATCH --job-name=lora_r512_bl
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_lora_r512_thought_baseline_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_lora_r512_thought_baseline_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Vanilla LoRA r=512 baseline for cooperative v6.5 comparison
#
# Parameter-matched: cooperative v6.5 uses 2 × r=256 adapters (V + A)
# => r=512 single LoRA gives comparable main LoRA capacity.
# Note: v6.5 also has W_comm + gates (~25M extra); this baseline has
# no such overhead, so is slightly param-disadvantaged.
#
# Matched hparams from v6.5:
#   - Same data: gui360_train_thought.jsonl (97,647 samples)
#   - Same eff_bs: 8 nodes * 4 GPU * pdbs=1 * grad_accum=4 = 128
#   - Same lr/schedule: 1e-5 cosine, warmup=0.03, 4 epochs
#   - Same targets: q/k/v/o/gate/up/down_proj
#   - Same lora_alpha ratio: 2:1 (alpha=1024, r=512)
#   - Same cutoff: 4096
#   - Same freeze: vision tower + MM projector
#
# This gives us the critical question: does cooperative v6.5's
# specialization (two orthogonal adapters + comm) beat a single
# r=512 LoRA with the same parameter budget?
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
LLAMA_FACTORY_DIR="$PROJECT_DIR/train_GUI_360/LLaMA-Factory"
CONFIG_FILE="$PROJECT_DIR/train_GUI_360/llamafactory/qwen25vl_gui360_lora_r512_thought_baseline.yaml"
LOG_DIR="$PROJECT_DIR/scripts/exp_cooperative/logs"
mkdir -p "$LOG_DIR"

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

if [ -f "$PROJECT_DIR/train/env_config.sh" ]; then
    source "$PROJECT_DIR/train/env_config.sh"
fi

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29503
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200
export TORCH_GLOO_TIMEOUT_SECONDS=7200

echo "=========================================="
echo "Vanilla LoRA r=512 baseline (thought data, 4 epochs)"
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

    # Node-local HF cache to avoid NFS mmap SIGBUS errors
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
echo "Output: $PROJECT_DIR/train_GUI_360/llamafactory/output/gui360_lora_r512_thought_baseline"
echo "=========================================="
exit $TRAIN_EXIT
