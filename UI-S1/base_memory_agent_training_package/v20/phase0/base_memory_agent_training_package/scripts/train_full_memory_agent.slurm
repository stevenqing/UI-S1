#!/bin/bash
#SBATCH --job-name=full_mem_agent
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v20/phase0/base_memory_agent_training_package/logs/full_mem_agent_%x_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v20/phase0/base_memory_agent_training_package/logs/full_mem_agent_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1}"
PACKAGE_DIR="${PACKAGE_DIR:-$PROJECT_DIR/v20/phase0/base_memory_agent_training_package}"
RUN_KEY="${RUN_KEY:-type_recovery}"
LOG_DIR="$PACKAGE_DIR/logs"
mkdir -p "$LOG_DIR"

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

if [ -f "$PROJECT_DIR/train/env_config.sh" ]; then
    source "$PROJECT_DIR/train/env_config.sh"
fi

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6
export LD_LIBRARY_PATH=$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT="${MASTER_PORT:-29551}"
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200
export TORCH_GLOO_TIMEOUT_SECONDS=7200

echo "=========================================="
echo "Full-parameter memory-agent SFT"
echo "RUN_KEY: $RUN_KEY"
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}:${MASTER_PORT}"
echo "Start:   $(date)"
echo "=========================================="

cd "$PROJECT_DIR"

srun --ntasks-per-node=1 /bin/bash -c '
    export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
    export CONDA_DEFAULT_ENV=qwen3-eval
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
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
    ROOT_DIR='"$PROJECT_DIR"' PACKAGE_DIR='"$PACKAGE_DIR"' RUN_KEY='"$RUN_KEY"' NPROC_PER_NODE=4 NNODES=$SLURM_NNODES NODE_RANK=$SLURM_NODEID MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT \
      '"$PACKAGE_DIR"'/scripts/run_full_memory_agent_sft.sh
'
TRAIN_EXIT=$?

echo "=========================================="
echo "Done: $(date) exit=$TRAIN_EXIT"
echo "=========================================="
exit $TRAIN_EXIT
