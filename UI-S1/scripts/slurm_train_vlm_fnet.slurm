#!/bin/bash
#SBATCH --job-name=vlm_fnet
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/outputs/vlm_fnet/vlm_fnet_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/outputs/vlm_fnet/vlm_fnet_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# ============================================================
# Task 3.1: Train VLM eigenfunction (Qwen2.5-VL + LoRA)
# Single-app training script. Submit 3 jobs in parallel:
#   sbatch --export=APP=excel scripts/slurm_train_vlm_fnet.slurm
#   sbatch --export=APP=word  scripts/slurm_train_vlm_fnet.slurm
#   sbatch --export=APP=ppt   scripts/slurm_train_vlm_fnet.slurm
#
# Expected runtime: ~5-8 hours per app (2 epochs, batch_size=16)
# Memory: ~95 GB GPU (7B model + LoRA + gradient checkpointing)
# ============================================================

# Validate APP variable
if [ -z "${APP}" ]; then
    echo "ERROR: APP variable not set. Use: sbatch --export=APP=excel scripts/slurm_train_vlm_fnet.slurm"
    exit 1
fi

# Activate environment
source /home/a5l/shuqing.a5l/miniconda3/bin/activate qwen3-eval

# Set CUDA library path
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

# Paths
BASE_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
DATA_DIR="${BASE_DIR}/outputs/vlm_fnet/data"
OUTPUT_DIR="${BASE_DIR}/outputs/vlm_fnet"
MODEL_PATH="${BASE_DIR}/checkpoints/Qwen2.5-VL-7B-Instruct"

echo "=========================================="
echo "Task 3.1: VLM Eigenfunction Training"
echo "App: ${APP}"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Model: ${MODEL_PATH}"
echo "=========================================="

cd "${BASE_DIR}"

python scripts/train_vlm_eigenfunction.py \
    --app "${APP}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --model-path "${MODEL_PATH}" \
    --input-mode screenshot_a11y \
    --lora-rank 16 \
    --lora-alpha 32 \
    --batch-size 16 \
    --epochs 2 \
    --lr 1e-4 \
    --eta 1.0 \
    --image-size 448 \
    --device auto

echo ""
echo "=========================================="
echo "Training complete for ${APP} at $(date)"
echo "Results in: ${OUTPUT_DIR}/${APP}"
echo "=========================================="
