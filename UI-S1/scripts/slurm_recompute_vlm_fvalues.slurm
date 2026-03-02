#!/bin/bash
#SBATCH --job-name=vlm_recompute
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/outputs/vlm_fnet/recompute_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/outputs/vlm_fnet/recompute_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Recompute f-values from saved VLM checkpoints (fix bf16→numpy bug)

source /home/a5l/shuqing.a5l/miniconda3/bin/activate qwen3-eval
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1
python scripts/recompute_vlm_fvalues.py --app all --device auto
