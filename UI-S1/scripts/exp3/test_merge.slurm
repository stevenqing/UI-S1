#!/bin/bash
#SBATCH --job-name=test_merge
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp3/logs/test_merge_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp3/logs/test_merge_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Quick test: merge C2 MoE checkpoint into base model

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval

echo "=== Test Merge: C2 ==="
echo "Job: ${SLURM_JOB_ID} | Node: $(hostname) | Start: $(date)"

python "$PROJECT_DIR/evaluation/merge_moe_to_full.py" \
    --base_model "$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct" \
    --checkpoint_dir "$PROJECT_DIR/train_GUI_360/llamafactory/output/selective_moe_c2/final/moe_checkpoint" \
    --output_dir "$PROJECT_DIR/train_GUI_360/llamafactory/output/selective_moe_c2/merged_model"

echo ""
echo "=== Verifying merged model ==="
ls -lh "$PROJECT_DIR/train_GUI_360/llamafactory/output/selective_moe_c2/merged_model/"
echo ""
echo "Done: $(date)"
