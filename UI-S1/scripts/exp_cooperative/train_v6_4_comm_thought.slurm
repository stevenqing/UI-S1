#!/bin/bash
#SBATCH --job-name=coop_v64t
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v6_4_thought_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v6_4_thought_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Cooperative LoRA v6.4 — push gate / W_comm learning harder
# WITH thought data, tanh gate, r=256, eff_bs=128 (same as v6.3)
#
# Delta vs v6.3:
#   1. gate_lr_multiplier: 10.0 -> 50.0
#        comm-group lr = base_lr * 50 = 1e-5 * 50 = 5e-4
#        (vs v6.3 comm-group lr = 1e-4)
#   Everything else identical to v6.3.
#
# Why this change:
#   v6.3 observation at epoch 3.17 (≈2400 optimizer steps):
#     - gate_av_std = 0.0248  (after tanh, basically linear in this regime)
#     - gate_va_std = 0.0388
#     - max |gate|  ≈ 0.095
#     - W_av norm   = 9.382 (kaiming init = 9.24, +1.5%)
#     - W_va norm   = 9.581 (+3.7%)
#   The cooperative communication machinery is barely moving relative to its
#   safe operating range (tanh saturates near ±1). The model is in a weak-comm
#   regime — the cross-feeding term ‖g·W·h_other‖ is only ~1-4% of ‖h_self‖.
#
#   Yet v6.3 epoch-2 already beats v3 epoch-2 by +1.48 pp (47.59 vs 46.11).
#   Hypothesis: stronger cooperative comm should help further. Bumping the
#   comm-group lr 5x lets gates and W matrices co-adapt more aggressively
#   without touching the main LoRA learning dynamics.
#
# Safety analysis:
#   - tanh saturates around |gate| > 2 (tanh(2)≈0.96). Even at 5x lr the
#     expected end-of-training |gate| is ≈0.05*5 = 0.25, well within linear
#     regime (tanh(0.25)≈0.245). No saturation risk.
#   - W matrices currently move <4% in 4 epochs. 5x bump → ~20% drift.
#     Still small relative to kaiming init scale.
#   - weight_decay still 0 on comm group → no implicit shrinkage.
#
# Schedule: same as v6.3
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
echo "Cooperative LoRA v6.4 — eff_bs=128, r=256, tanh gate, lr_mult=50"
echo "  grad_accum=4, lr=1e-5, gate_type=tanh, gate_init=0, lr_mult=50, wd=0"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}"
echo "Start:   $(date)"
echo "=========================================="

OUTPUT_DIR="train_GUI_360/llamafactory/output/cooperative_v6_4_comm_thought"

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
