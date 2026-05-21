#!/bin/bash
#SBATCH --job-name=merge_v6.5
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v10/scripts/logs/merge_v6_5_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v10/scripts/logs/merge_v6_5_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Merge v6.5 AC cooperative LoRA (LoRA_A only) into base model
# Result: a new base model that already knows open/system_button/etc.

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "=========================================="
echo "Merging v6.5 AC cooperative LoRA into base"
echo "=========================================="
echo "Start: $(date)"

cd "$PROJECT_DIR"

python evaluation/merge_cooperative_lora.py \
    --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
    --coop_checkpoint train_GUI_360/llamafactory/output/cooperative_v6_5_ac/epoch-4 \
    --output_dir checkpoints/Qwen2.5-VL-7B-v6.5-ac-merged \
    --merge_mode a_only

echo "=========================================="
echo "Done: $(date)"
echo "Output: checkpoints/Qwen2.5-VL-7B-v6.5-ac-merged"
echo "=========================================="
