#!/bin/bash
#SBATCH --job-name=eval_v10
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v10/scripts/logs/eval_v10_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v10/scripts/logs/eval_v10_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:1

# ============================================
# Evaluate v10 cooperative LoRA (S300 checkpoint)
# on AC test set — single GPU, greedy decoding
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CHECKPOINT="${CHECKPOINT:-$PROJECT_DIR/v10/output/v10_grpo_ddp/epoch-1_step-300}"
EVAL_DATA="${EVAL_DATA:-$PROJECT_DIR/datasets/cooperative_thought_ac/ac_val_thought.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/v10/output/eval_s300}"

mkdir -p "$PROJECT_DIR/v10/scripts/logs"

echo "=========================================="
echo "v10 Eval — Cooperative LoRA"
echo "=========================================="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Node:       $(hostname)"
echo "Checkpoint: $CHECKPOINT"
echo "Eval data:  $EVAL_DATA"
echo "Output:     $OUTPUT_DIR"
echo "Start:      $(date)"
echo "=========================================="

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

cd "$PROJECT_DIR"

python v10/eval_v10.py \
    --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
    --checkpoint "$CHECKPOINT" \
    --eval_data "$EVAL_DATA" \
    --output_dir "$OUTPUT_DIR"

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
