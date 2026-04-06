#!/bin/bash
#SBATCH --job-name=coop_thought
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_thought_v1_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_thought_v1_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Cooperative LoRA + Thought-Augmented Training
# ============================================
# Primary signal: Thought CE — model learns <thought>visual_desc</thought>
# before <tool_call>, forcing LoRA_V (image) to encode visual semantics
# that LoRA_A (text) attends to for generating the thought.
#
# bind_weight=0: pure thought CE (no contrastive L_bind)
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

echo "=========================================="
echo "Cooperative LoRA + Thought Training v1"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Start: $(date)"
echo "=========================================="

export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export CONDA_DEFAULT_ENV=ui-s1
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

export WANDB_MODE="disabled"

mkdir -p "$PROJECT_DIR/scripts/exp_cooperative/logs"

cd "$PROJECT_DIR"

torchrun --nproc_per_node=4 --master_port=29500 \
    train_cooperative.py \
    --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
    --train_data datasets/cooperative_thought/gui360_train_thought.jsonl \
    --val_data datasets/cooperative_thought/gui360_val_thought.jsonl \
    --output_dir train_GUI_360/llamafactory/output/cooperative_thought_v1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --target_modules q_proj k_proj v_proj o_proj \
    --bind_weight 0.0 \
    --per_device_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1.5e-5 \
    --warmup_ratio 0.03 \
    --num_epochs 1 \
    --max_length 4096 \
    --save_steps 200 \
    --eval_steps 200 \
    --logging_steps 10

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
