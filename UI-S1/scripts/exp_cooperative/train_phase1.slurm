#!/bin/bash
#SBATCH --job-name=coop_lora_v1
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_phase1_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_phase1_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Phase 1: Token-Level Cooperative LoRA Training
# Image tokens → LoRA_V, text tokens → LoRA_A
# Loss = L_act + 0.1 * L_bind
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

echo "=========================================="
echo "Cooperative LoRA Phase 1 Training"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=========================================="

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

export WANDB_PROJECT="cooperative-lora"
export WANDB_RUN_NAME="phase1-r16-bind0.1-${SLURM_JOB_ID}"

mkdir -p "$PROJECT_DIR/scripts/exp_cooperative/logs"

cd "$PROJECT_DIR"

torchrun --nproc_per_node=4 --master_port=29500 \
    train_cooperative.py \
    --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
    --train_data train_GUI_360/llamafactory/data/gui360_train.json \
    --val_data train_GUI_360/llamafactory/data/gui360_val.json \
    --output_dir checkpoints/cooperative_v1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --target_modules q_proj k_proj v_proj o_proj \
    --bind_weight 0.1 \
    --bind_layer 27 \
    --per_device_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1.5e-5 \
    --warmup_ratio 0.03 \
    --num_epochs 1 \
    --save_steps 200 \
    --eval_steps 200 \
    --logging_steps 10

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
