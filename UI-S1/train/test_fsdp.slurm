#!/bin/bash
#SBATCH --job-name=test_fsdp
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/test_fsdp_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/test_fsdp_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ============================================
# Test FSDP model loading with srun
# ============================================

set -e

# Load required modules
module load cuda/12.6

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

# Get node information
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

echo "=========================================="
echo "Test FSDP model loading with srun"
echo "Head Node: $head_node ($head_node_ip)"
echo "Nodes: ${SLURM_NNODES}, GPUs per node: 4"
echo "Total GPUs: $((SLURM_NNODES * 4))"
echo "=========================================="

# Launch test via srun
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
        export MODEL_PATH=$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct

        export PYTHONPATH=$PROJECT_DIR
        export PYTHONUNBUFFERED=1

        echo \"[Node \$(hostname)] RANK=\$RANK LOCAL_RANK=\$LOCAL_RANK starting FSDP test...\"

        python $PROJECT_DIR/train/test_fsdp_model.py
    "

echo ""
echo "=========================================="
echo "Test completed at $(date)"
echo "=========================================="
