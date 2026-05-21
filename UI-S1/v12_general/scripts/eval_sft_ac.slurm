#!/bin/bash
#SBATCH --job-name=v12_eval
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v12_general/scripts/logs/eval_sft_ac_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v12_general/scripts/logs/eval_sft_ac_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V12 SFT Eval — AC trajectory (4 GPU shards)
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
COOP_CKPT="${PROJECT_DIR}/checkpoints/v12_sft/cooperative_final"
OUTPUT_DIR="${PROJECT_DIR}/outputs/v12_sft_ep4_ac"

mkdir -p "${PROJECT_DIR}/v12_general/scripts/logs"

echo "=========================================="
echo "V12 Eval — SFT epoch 4, AC trajectory"
echo "  Checkpoint: ${COOP_CKPT}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Start:      $(date)"
echo "=========================================="

srun --ntasks-per-node=1 bash -c '
    export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
    export CONDA_DEFAULT_ENV=ui-s1
    export PYTHONPATH="'"$PROJECT_DIR"':$PYTHONPATH"
    export PYTHONUNBUFFERED=1

    cd "'"$PROJECT_DIR"'"

    for GPU_ID in 0 1 2 3; do
        python v12_general/eval_ac_trajectory.py \
            --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
            --coop_checkpoint '"$COOP_CKPT"' \
            --output_dir '"$OUTPUT_DIR"' \
            --gpu_id $GPU_ID \
            --shard_id $GPU_ID \
            --num_shards 4 \
            --n_history_image_limit 2 \
            --max_new_tokens 512 \
            --no_stop &
    done
    wait
'

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
