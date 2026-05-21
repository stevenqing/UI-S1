#!/bin/bash
#SBATCH --job-name=coop_v63t
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v6_3_thought_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v6_3_thought_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Cooperative LoRA v6.3 — fix eff_bs to match v3
# WITH thought data, tanh gate, r=256
#
# Delta vs v6.2 thought:
#   1. gradient_accumulation_steps: 8 -> 4
#        eff_bs = 32 GPU x 1 x 4 = 128  (was 256)
#        Matches v3's eff_bs=128 exactly
#   Everything else identical to v6.2.
#
# Why this matters:
#   v6.2 (and v6.1) used eff_bs=256 with lr=1e-5 unchanged from v3 (eff_bs=128).
#   Without linear-scaling lr, larger batch -> half the optimizer updates per
#   epoch -> strictly LESS optimization at the same data exposure. This was a
#   silent confound in v6.1 and v6.2 and may explain why v6.2 epoch-1 did not
#   clearly beat v3 epoch-1 despite tanh+r=256.
#
#   Fix: keep lr=1e-5, halve grad_accum -> eff_bs=128. Now:
#     - Same eff_bs as v3
#     - Same lr as v3
#     - Same rank as v3
#     - Only difference: cooperative_comm + tanh gate
#   This is the CLEAN A/B test we should have run from the start.
#
# Schedule:
#   4 epochs x (97647/128 ~= 763 steps/epoch) = ~3050 total steps
#   Per-step wall time ~halves (grad_accum 8->4), so total wall time is
#   approximately unchanged (~5 hours for 4 epochs).
#
# Reference: v6.3 epoch-2 (steps ~1525) is the head-to-head with v3 epoch-2
#            (steps 1525). Same samples-seen, same updates, same eff_bs, same
#            lr, same rank. Only tanh+comm differs.
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
echo "Cooperative LoRA v6.3 — eff_bs=128 (v3-matched), r=256, tanh gate"
echo "  grad_accum=4, lr=1e-5, gate_type=tanh, gate_init=0, lr_mult=10, wd=0"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}"
echo "Start:   $(date)"
echo "=========================================="

OUTPUT_DIR="train_GUI_360/llamafactory/output/cooperative_v6_3_comm_thought"

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
