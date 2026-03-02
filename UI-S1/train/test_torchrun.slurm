#!/bin/bash
#SBATCH --job-name=test_torchrun
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/test_torchrun_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/test_torchrun_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ============================================
# Test multi-node training with torchrun via srun
# This bypasses Ray entirely
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

# Get node information
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

echo "=========================================="
echo "Test torchrun multi-node training"
echo "Head Node: $head_node ($head_node_ip)"
echo "Nodes: ${SLURM_NNODES}, GPUs per node: 4"
echo "Total GPUs: $((SLURM_NNODES * 4))"
echo "=========================================="

# Load required modules for GPU access
module load cuda/12.6

# Activate environment
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
source $PROJECT_DIR/train/env_config.sh

export PYTHONPATH=$PROJECT_DIR
export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

# Use srun to launch torchrun on all nodes
# Each node runs 4 processes (one per GPU)
srun --nodes=$SLURM_NNODES --ntasks-per-node=4 --cpus-per-task=72 \
    bash -c "
        # Load CUDA module for GPU access
        module load cuda/12.6

        source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
        source $PROJECT_DIR/train/env_config.sh

        export MASTER_ADDR=$head_node_ip
        export MASTER_PORT=29500
        export WORLD_SIZE=\$SLURM_NTASKS
        export RANK=\$SLURM_PROCID
        export LOCAL_RANK=\$SLURM_LOCALID

        # Do NOT set CUDA_VISIBLE_DEVICES - let PyTorch select device based on LOCAL_RANK
        # The test script will use cuda:LOCAL_RANK

        echo \"[Node \$(hostname)] RANK=\$RANK LOCAL_RANK=\$LOCAL_RANK WORLD_SIZE=\$WORLD_SIZE\"

        python $PROJECT_DIR/train/test_nccl_minimal.py
    "

echo ""
echo "=========================================="
echo "Test completed at $(date)"
echo "=========================================="
