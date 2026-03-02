#!/bin/bash
#SBATCH --job-name=nccl_test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/nccl_test_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/nccl_test_%j.log
#SBATCH --partition=workq
#SBATCH --gres=gpu:4

echo "============================================="
echo "NCCL Minimal Test - Job ID: $SLURM_JOB_ID"
echo "============================================="
echo "Nodes: $SLURM_NODELIST"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "GPUs per node: 4"
echo ""

# Activate conda environment
source /home/a5l/shuqing.a5l/miniconda3/etc/profile.d/conda.sh
conda activate ui-s1

# Set CUDA library paths
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

# NCCL configuration - Force socket transport
export NCCL_DEBUG=TRACE  # Maximum verbosity
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_DEBUG_FILE=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/nccl_debug_%j_rank%r.log
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME="hsn0,hsn1,hsn2,hsn3"
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_P2P_LEVEL=LOC
export NCCL_PROTO="^LL128"
export NCCL_CROSS_NIC=1

# Get master node and port
MASTER_NODE=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
MASTER_ADDR=$(host $MASTER_NODE | awk '/has address/ {print $4}')
MASTER_PORT=29500
WORLD_SIZE=$((SLURM_NNODES * 4))

echo "MASTER_NODE: $MASTER_NODE"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo ""

echo "NCCL environment variables:"
env | grep -E "^NCCL" | sort
echo ""

# Function to run on each rank
run_rank() {
    local NODE_ID=$1
    local LOCAL_RANK=$2
    local GLOBAL_RANK=$((NODE_ID * 4 + LOCAL_RANK))

    export RANK=$GLOBAL_RANK
    export LOCAL_RANK=$LOCAL_RANK
    export WORLD_SIZE=$WORLD_SIZE
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    export CUDA_VISIBLE_DEVICES=$LOCAL_RANK

    echo "[Node $NODE_ID] Starting rank $GLOBAL_RANK with LOCAL_RANK=$LOCAL_RANK, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

    python /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/test_nccl_minimal.py
}

export -f run_rank
export MASTER_ADDR MASTER_PORT WORLD_SIZE

# Use srun to launch across nodes
echo "Launching test across nodes..."
echo ""

# Launch all ranks using srun
srun --ntasks=$WORLD_SIZE --ntasks-per-node=4 bash -c '
    NODE_ID=$SLURM_NODEID
    LOCAL_RANK=$SLURM_LOCALID
    GLOBAL_RANK=$((NODE_ID * 4 + LOCAL_RANK))

    export RANK=$GLOBAL_RANK
    export LOCAL_RANK=$LOCAL_RANK
    export CUDA_VISIBLE_DEVICES=$LOCAL_RANK

    echo "[$(hostname)] Starting rank $GLOBAL_RANK with LOCAL_RANK=$LOCAL_RANK"

    python /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/test_nccl_minimal.py
'

echo ""
echo "Test completed!"
