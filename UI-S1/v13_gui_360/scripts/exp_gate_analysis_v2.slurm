#!/bin/bash
#SBATCH --job-name=gate_v2
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/gate_v2_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/gate_v2_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:1

set -e
PROJECT="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/sbsa-linux/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

echo "Gate Analysis V2 — Per-layer | Job ${SLURM_JOB_ID} | $(date)"

cd "$PROJECT"

python v13_gui_360/exp_gate_analysis_perlayer.py \
    --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
    --coop_checkpoint checkpoints/v13_gui360_rl/epoch-3/cooperative \
    --test_data v13_gui_360/data/gui360_test_968.jsonl \
    --max_episodes 100 \
    --gpu 0

echo "Done: $(date)"
