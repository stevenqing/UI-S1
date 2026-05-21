#!/bin/bash
#SBATCH --job-name=coop_v61t
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v6_1_thought_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v6_1_thought_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Cooperative LoRA v6.1 — Per-layer communication
# WITH thought data, stronger gate signal
#
# Delta vs v6 thought (job 3713229):
#   1. gate_init: -1.5 -> 0       (sigmoid(0)=0.5, max-gradient region)
#   2. gate_lr_multiplier: 1 -> 10 (lr 1e-5 * 10 = 1e-4 for comm params)
#   3. gate_weight_decay: 0.1 -> 0 (no WD dragging gate logit toward 0)
#
# Everything else identical to v6 thought:
#   r=128, 7 target modules, 2 agents, cooperative_comm on
#   8 nodes x 4 GPUs = 32 GPUs, effective batch = 256
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
echo "Cooperative LoRA v6.1 — comm + thought, r=128"
echo "  stronger gate signal: gate_init=0, lr_mult=10, wd=0"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}"
echo "Start:   $(date)"
echo "=========================================="

OUTPUT_DIR="train_GUI_360/llamafactory/output/cooperative_v6_1_comm_thought"

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
        --lora_r 128 \
        --lora_alpha 256 \
        --lora_dropout 0.05 \
        --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
        --num_agents 2 \
        --cooperative_comm \
        --gate_init 0.0 \
        --gate_lr_multiplier 10.0 \
        --gate_weight_decay 0.0 \
        --bind_weight 0.0 \
        --per_device_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-5 \
        --num_epochs 2.0 \
        --warmup_ratio 0.03 \
        --save_steps 50 \
        --eval_steps 50 \
        --logging_steps 5 \
        --max_length 4096
'

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
