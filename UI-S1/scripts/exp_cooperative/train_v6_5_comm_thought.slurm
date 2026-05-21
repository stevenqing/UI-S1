#!/bin/bash
#SBATCH --job-name=coop_v65t
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v6_5_thought_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v6_5_thought_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Cooperative LoRA v6.5 — push gate / W_comm learning even harder
# WITH thought data, tanh gate, r=256, eff_bs=128 (same as v6.4)
#
# Delta vs v6.4:
#   1. gate_lr_multiplier: 50.0 -> 100.0
#        comm-group lr = base_lr * 100 = 1e-5 * 100 = 1e-3
#        (v6.4 comm-group lr = 5e-4)
#   Everything else identical to v6.4.
#
# Why this change:
#   v6.4 beats v6.3 at every epoch with a monotone growing delta:
#     ep1: +0.12   ep2: +0.16   ep3: +0.41   ep4: +0.99
#   v6.4 ep4 gate magnitudes (|tanh(gate)|):
#     attention mean 0.097, MLP mean 0.131, overall top 0.43
#   Still well within tanh linear regime (saturation |gate|>2).
#   k_proj median in v6.4 ep4 is 0.023 — attention comm was lr-starved in
#   v6.3 and only became "alive" in v6.4. v6.5 tests whether another 2x lr
#   continues the scaling curve or hits diminishing returns.
#
# Safety analysis:
#   - At lr*100 expected end-of-training |gate| ≈ 0.1-0.3 mean, max ~0.8.
#     tanh(0.8)=0.664 — still mostly linear, no saturation.
#   - W drift 4 epochs: expected ~40% (vs v6.4 ~20%, v6.3 ~4%).
#     Kaiming init scale is ~9, so final W norm ~12-13. LoRA A matrices have
#     norm ~200 in target modules — W comm is still a minority signal.
#   - weight_decay still 0 on comm group.
#
# Expected outcome:
#   - If v6.5 > v6.4: continue scaling, next step lr*200
#   - If v6.5 ≈ v6.4: found the plateau → publishable scaling curve
#   - If v6.5 < v6.4: v6.4 was near the peak, gate saturation affecting main loss
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
echo "Cooperative LoRA v6.5 — eff_bs=128, r=256, tanh gate, lr_mult=100"
echo "  grad_accum=4, lr=1e-5, gate_type=tanh, gate_init=0, lr_mult=100, wd=0"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}"
echo "Start:   $(date)"
echo "=========================================="

OUTPUT_DIR="train_GUI_360/llamafactory/output/cooperative_v6_5_comm_thought"

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
        --gate_lr_multiplier 100.0 \
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
