#!/bin/bash
#SBATCH --job-name=v15_grad_diag
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/gradient_diagnostic_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/gradient_diagnostic_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# V15 Gradient Interference Diagnostic (LEAS-inspired)
# Measures gradient cosine similarity between Expert A and Expert B

set -e

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

PROJECT_ROOT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=========================================="
echo "V15 Gradient Interference Diagnostic"
echo "Job ID: ${SLURM_JOB_ID} | Start: $(date)"
echo "=========================================="

python "$PROJECT_ROOT/v15_gui_360/diagnose_gradient_interference.py" \
    --model_path "$PROJECT_ROOT/checkpoints/Qwen2.5-VL-7B-Instruct" \
    --coop_checkpoint "$PROJECT_ROOT/checkpoints/v15_rl_from_coop_sft/epoch-1_step-50/cooperative" \
    --train_data "$PROJECT_ROOT/v12_gui_360/data/gui360_train_2000_balanced.jsonl" \
    --num_samples 100 \
    --output "$PROJECT_ROOT/v15_gui_360/gradient_diagnostic.json"

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
