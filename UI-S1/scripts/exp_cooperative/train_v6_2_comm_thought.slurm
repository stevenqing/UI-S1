#!/bin/bash
#SBATCH --job-name=coop_v62t
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v6_2_thought_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v6_2_thought_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Cooperative LoRA v6.2 — tanh gate + matched v3 capacity
# WITH thought data
#
# Delta vs v6.1 thought:
#   1. lora_r: 128 -> 256   (match v3 capacity, half the steps too)
#   2. lora_alpha: 256 -> 512
#   3. gate_type: sigmoid -> tanh
#        tanh(0)=0, gradient at 0 is 1.0 (4x sigmoid's 0.25)
#        bounded [-1,1], allows learning negative (anti) coupling
#        tanh(x) ≈ x near 0 (locally linear), but self-bounds at ±1
#   4. gate_init: 0.0 (same — stays in max-gradient region for tanh too)
#
# Kept from v6.1:
#   gate_lr_multiplier=10, gate_weight_decay=0
#   8 nodes x 4 GPUs = 32 GPUs, effective batch = 256
#   thought data (aligned with v3, v6.1)
#
# Schedule:
#   4 epochs x (97647/256 ~= 381 steps/epoch) = ~1525 total steps
#   Matches v3's 1525 steps exactly - clean capacity/step isolation vs v3.
#   Per-epoch persistent checkpoints saved to epoch-1/ .. epoch-4/
#   by CooperativeSaveCallback.on_epoch_end.
#
# Three-variable confound isolation: compared to v3 (r=256, 1525 steps,
# no comm), v6.2 has (a) same rank/capacity, (b) same number of steps,
# (c) cooperative_comm + tanh gate. v6.2 vs v3 is now a clean A/B for
# the comm mechanism.
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

# Multi-node setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200

echo "=========================================="
echo "Cooperative LoRA v6.2 — comm + thought, r=256, tanh gate"
echo "  gate_type=tanh, gate_init=0, lr_mult=10, wd=0"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}"
echo "Start:   $(date)"
echo "=========================================="

OUTPUT_DIR="train_GUI_360/llamafactory/output/cooperative_v6_2_comm_thought"

srun --ntasks-per-node=1 bash -c '
    export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
    export CONDA_DEFAULT_ENV=ui-s1
    export PYTHONPATH="'"$PROJECT_DIR"':$PYTHONPATH"
    export PYTHONUNBUFFERED=1
    export WANDB_MODE=disabled

    cd "'"$PROJECT_DIR"'"

    torchrun \
        --nproc_per_node=4 \
        --nnodes=$SLURM_NNODES \
        --node_rank=$SLURM_NODEID \
        --master_addr='"$MASTER_ADDR"' \
        --master_port='"$MASTER_PORT"' \
        train_cooperative.py \
        --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
        --train_data datasets/cooperative_thought/gui360_train_thought.jsonl \
        --val_data datasets/cooperative_thought/gui360_val_thought.jsonl \
        --output_dir '"$OUTPUT_DIR"' \
        --lora_r 256 \
        --lora_alpha 512 \
        --lora_dropout 0.05 \
        --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
        --num_agents 2 \
        --cooperative_comm \
        --gate_type tanh \
        --gate_init 0.0 \
        --gate_lr_multiplier 10.0 \
        --gate_weight_decay 0.0 \
        --bind_weight 0.0 \
        --per_device_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-5 \
        --num_epochs 4.0 \
        --warmup_ratio 0.03 \
        --save_steps 50 \
        --eval_steps 50 \
        --logging_steps 5 \
        --max_length 4096
'

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
