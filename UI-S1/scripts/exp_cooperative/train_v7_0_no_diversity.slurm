#!/bin/bash
#SBATCH --job-name=coop_v70_nodiv
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v7_0_no_diversity_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v7_0_no_diversity_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Cooperative LoRA v7.0 with diversity_loss_weight=0.0  (ABLATION of v7.0)
#
# Purpose:
#   v7.0 (the original, λ=0.01) showed cos(B_v, B_a) plunge from 0 to -0.76
#   within ~50 steps and stay there. Question: is this anti-correlation
#   driven by diversity loss, or is it an attractor of merge mode itself?
#
#   Ablation: same architecture (merge routing + per-layer comm + tanh
#   gate, r=256, eff_bs=128, lr=1e-5) but λ_div=0.0. The two adapters now
#   have NO explicit pressure to differentiate, but they still operate
#   under merge mode where every token sees 0.5*(delta_v + delta_a).
#
# Three possible outcomes (mapped to paper story):
#   A. cos still falls to ~-0.7 (or similar)  ->  anti-correlation is an
#      INHERENT property of merge mode + per-layer comm. Specialization
#      emerges from architecture alone, no auxiliary loss needed. This is
#      the strongest paper claim.
#   B. cos stays near 0 (random small fluctuations)  ->  the two adapters
#      live in independent random subspaces, no specialization, no mode
#      collapse. Weak result. Single LoRA would do as well.
#   C. cos -> +1 (mode collapse)  ->  without pressure the two adapters
#      become identical copies. Confirms diversity loss is essential.
#      Story becomes "you need an explicit objective to break symmetry".
#
# Why now:
#   We need this number BEFORE deciding whether to invest in v7.1 (the
#   information-bottleneck variant). v7.1's whole motivation is "use
#   capacity constraint instead of explicit loss to drive specialization".
#   If outcome B/C, v7.1 is the right path. If outcome A, v7.1 is icing
#   on the cake but not necessary.
#
# Schedule: same as v7.0
#   4 epochs x ~763 steps/epoch = ~3050 total steps, ~5h wall.
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
echo "Cooperative LoRA v7.0 — merge routing, NO diversity loss (lambda=0)"
echo "  Ablation of v7.0: is anti-correlation driven by L_div or by merge?"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}"
echo "Start:   $(date)"
echo "=========================================="

OUTPUT_DIR="train_GUI_360/llamafactory/output/cooperative_v7_0_merge_no_diversity"

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
        --routing_mode merge \
        --diversity_loss_weight 0.0 \
        --cooperative_comm \
        --gate_type tanh \
        --gate_init 0.0 \
        --gate_lr_multiplier 50.0 \
        --gate_weight_decay 0.0 \
        --bind_weight 0.0 \
        --per_device_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1e-5 \
        --num_epochs 4.0 \
        --warmup_ratio 0.03 \
        --save_steps 100 \
        --eval_steps 100 \
        --logging_steps 10 \
        --max_length 4096
'

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
