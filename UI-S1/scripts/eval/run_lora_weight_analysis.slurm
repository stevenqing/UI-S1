#!/bin/bash
#SBATCH --job-name=lora_wt
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/lora_wt_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/logs/lora_wt_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
LORA_PATH="${LORA_PATH:-$PROJECT_DIR/train_GUI_360/moe_rl/extracted_lora_fullsft_r256}"
LORA_LABEL="${LORA_LABEL:-$(basename "$LORA_PATH")}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_DIR/evaluation/results/lora_analysis_${LORA_LABEL}_${TIMESTAMP}"

echo "LoRA Weight Analysis | Job ${SLURM_JOB_ID} | $(hostname) | $(date)"
echo "LoRA: $LORA_PATH"
echo "Output: $OUTPUT_DIR"

export PATH="/scratch/a5l/shuqing.a5l/miniconda/envs/qwen3-eval/bin:$PATH"
export CONDA_DEFAULT_ENV=qwen3-eval
export PYTHONUNBUFFERED=1

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"

python -u evaluation/lora_weight_analysis.py \
    --lora_path "$LORA_PATH" \
    --output_dir "$OUTPUT_DIR"

echo "Done at $(date)"
