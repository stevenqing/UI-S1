#!/bin/bash
#SBATCH --job-name=coop_v2
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v2_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v2_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Cooperative LoRA v2 — Full capacity
# r=32, 7 target modules (~160M params)
# Matches LoRA baseline parameter budget
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

echo "=========================================="
echo "Cooperative Thought LoRA v2 — Full Capacity"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export WANDB_MODE=disabled

cd "$PROJECT_DIR"

OUTPUT_DIR="train_GUI_360/llamafactory/output/cooperative_thought_v2"

torchrun --nproc_per_node=4 --master_port=29501 train_cooperative.py \
    --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
    --train_data datasets/cooperative_thought/gui360_train_thought.jsonl \
    --val_data datasets/cooperative_thought/gui360_val_thought.jsonl \
    --output_dir "$OUTPUT_DIR" \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --bind_weight 0.0 \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1.5e-5 \
    --num_epochs 1.0 \
    --warmup_ratio 0.03 \
    --save_steps 200 \
    --eval_steps 200 \
    --logging_steps 10 \
    --max_length 4096

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
