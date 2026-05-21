#!/bin/bash
#SBATCH --job-name=test-fsdp2
#SBATCH --output=/lus/lfs1aip2/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v10/verl_recipe/logs/test_fsdp2_%j.log
#SBATCH --error=/lus/lfs1aip2/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v10/verl_recipe/logs/test_fsdp2_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --gres=gpu:2

set -e

PROJECT_DIR="/lus/lfs1aip2/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
cd "$PROJECT_DIR"

# NCCL config for Slingshot
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=LOC

# Activate environment
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

echo "Running FSDP2 + multi-adapter compatibility test..."
echo "Python: $(which python)"
echo "Torch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

torchrun --nproc_per_node=2 v10/verl_recipe/tests/test_dual_adapter_fsdp2.py

echo "Test complete!"
