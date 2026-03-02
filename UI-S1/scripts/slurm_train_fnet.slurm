#!/bin/bash
#SBATCH --job-name=train_fnet
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/outputs/fnet/train_fnet_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/outputs/fnet/train_fnet_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# ============================================================
# Task 3: Train Laplacian eigenfunction approximator (f_net)
#
# Trains per-app f_net on GUI-360 state transitions.
# Expected runtime: ~5-10 min total (3 apps × ~2 min each).
# ============================================================

# Activate environment
source /home/a5l/shuqing.a5l/miniconda3/bin/activate qwen3-eval

# Set CUDA library path
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

# Paths
BASE_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
TRANSITIONS_DIR="${BASE_DIR}/outputs/transitions/gui360_full"
OUTPUT_DIR="${BASE_DIR}/outputs/fnet/gui360"

echo "=========================================="
echo "Training f_net (Laplacian Eigenfunction)"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Transitions dir: ${TRANSITIONS_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "=========================================="

# Create output dir for logs
mkdir -p "${OUTPUT_DIR}"

# Run training (all 3 apps)
cd "${BASE_DIR}"
python scripts/train_eigenfunction.py \
    --transitions-dir "${TRANSITIONS_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --app all \
    --epochs 200 \
    --batch-size 2048 \
    --lr 1e-3 \
    --eta 1.0 \
    --percentile-k 30.0 \
    --hidden-dims 256 256 \
    --device auto

echo ""
echo "=========================================="
echo "Training complete at $(date)"
echo "Results in: ${OUTPUT_DIR}"
echo "=========================================="
