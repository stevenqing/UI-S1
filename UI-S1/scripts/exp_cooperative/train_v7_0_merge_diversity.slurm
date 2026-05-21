#!/bin/bash
#SBATCH --job-name=coop_v70t
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v7_0_thought_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v7_0_thought_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Cooperative LoRA v7.0 — Emergent Specialization (merge + diversity)
# WITH thought data, tanh gate, r=256, eff_bs=128
#
# PARADIGM FLIP (v6 → v7):
#   v6: designer routes tokens (image→V, text→A). Specialization is
#       *imposed*; the only question is how the two adapters communicate.
#   v7: NO routing. Every token gets 0.5·(delta_v + delta_a). The two
#       adapters start symmetric and must self-differentiate under an
#       auxiliary pressure signal — or collapse to identical copies.
#
# Delta vs v6.4:
#   1. routing_mode:         hard  -> merge
#        (forward: delta = 0.5 * (delta_v + delta_a), no torch.where)
#   2. diversity_loss_weight: 0.0  -> 0.01
#        L_div = λ · mean_m cos( flat(lora_A_v_m), flat(lora_A_a_m) )
#        pushes the two adapters apart in parameter space.
#        λ=0.01 is conservative — cos is bounded [-1,1] so gradient scale
#        is directly comparable to ce_loss scale (≈1-2 in this task).
#   3. cooperative_comm:      True -> True   (unchanged)
#        The per-layer W_av / W_va / gate_av / gate_va communication is
#        kept. In merge mode this becomes a differentiable "how much do
#        agents mix internally" dial rather than cross-routing.
#   4. Every other hparam identical to v6.4.
#
# What we expect:
#   A. Runaway anti-correlation: L_div dominates, cos → -1, ce_loss
#      explodes. Means λ=0.01 is too strong.
#   B. Mode collapse: cos → +1, two adapters become identical, no benefit
#      over single-LoRA. Means pressure is too weak and merge mode alone
#      can't break symmetry. Would drop to single-LoRA as the correct
#      architecture.
#   C. Stable differentiation: cos settles in ~[-0.3, +0.3], ce_loss
#      comparable to v6.4, AP accuracy comparable. Then we measure
#      per-module differentiation to see WHAT the agents specialize to
#      (not what we told them to).
#
# Primary diagnostic: div_cos in the logs (added in train_cooperative.py
# log()). We want to see it moving, not stuck at 1.0.
#
# Schedule: same as v6.4
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
echo "Cooperative LoRA v7.0 — merge routing + diversity loss"
echo "  routing_mode=merge, diversity_loss_weight=0.01"
echo "  r=256, tanh gate, lr_mult=50 (comm still enabled)"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}"
echo "Start:   $(date)"
echo "=========================================="

OUTPUT_DIR="train_GUI_360/llamafactory/output/cooperative_v7_0_merge_diversity"

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
        --diversity_loss_weight 0.01 \
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
