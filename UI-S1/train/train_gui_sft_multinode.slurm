#!/bin/bash
#SBATCH --job-name=train_gui_sft_mn
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_gui_sft_mn_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_gui_sft_mn_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# UI-S1 SFT Multi-Node Training Script
# ============================================

set -e

# Environment setup
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CONFIG_PATH="$PROJECT_DIR/examples/qwen_gui_sft/config"
LOG_DIR="$PROJECT_DIR/train/logs"

# Training configuration
EXPERIMENT_NAME="qwenvl_uis1_full_finetune_multinode"

# Get node information
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

# Handle IPv6 addresses
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
fi

export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500

# Calculate world size
export WORLD_SIZE=$((SLURM_NNODES * 4))

echo "=========================================="
echo "Starting UI-S1 SFT Multi-Node Training"
echo "Project Dir: $PROJECT_DIR"
echo "Head Node: $head_node ($head_node_ip)"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Nodes: ${SLURM_NNODES}"
echo "GPUs per node: 4"
echo "Total GPUs: $WORLD_SIZE"
echo "Experiment: $EXPERIMENT_NAME"
echo "Start time: $(date)"
echo "=========================================="

# Activate environment and launch with torchrun
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
source $PROJECT_DIR/train/env_config.sh

cd $PROJECT_DIR

# Use srun to launch torchrun on each node
srun --ntasks-per-node=1 --exclusive \
    torchrun \
    --nnodes=${SLURM_NNODES} \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank=$SLURM_PROCID \
    -m verl.trainer.fsdp_sft_trainer \
    --config-path=$CONFIG_PATH \
    --config-name=gui_sft \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.nnodes=${SLURM_NNODES} \
    trainer.n_gpus_per_node=4

echo "=========================================="
echo "Training completed at $(date)"
echo "=========================================="
