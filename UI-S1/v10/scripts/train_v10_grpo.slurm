#!/bin/bash
#SBATCH --job-name=v10_grpo
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v10/scripts/logs/train_v10_grpo_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v10/scripts/logs/train_v10_grpo_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# v10 — Sequential Cooperative LoRA GRPO (DDP + PEFT)
#
# Two-pass generation:
#   Pass 1: Grounder (LoRA_grounder) → grounding description
#   Pass 2: Actor (LoRA_actor) → action (reads grounding)
#
# GRPO with dual rewards + KL penalty.
# DDP syncs gradients across 16 GPUs (4 nodes × 4).
# ~22GB/GPU (7B bf16 + 2×r=128 LoRA + optimizer). Fits 95GB GH200.
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/v10/output/v10_grpo_ddp}"
TRAIN_DATA="${TRAIN_DATA:-$PROJECT_DIR/datasets/cooperative_thought_ac/ac_train_thought.jsonl}"
VAL_DATA="${VAL_DATA:-$PROJECT_DIR/datasets/cooperative_thought_ac/ac_val_thought.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"

mkdir -p "$PROJECT_DIR/v10/scripts/logs"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29505

# NCCL config for hsn0 (Slingshot) — no InfiniBand
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=LOC
export NCCL_CROSS_NIC=1
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200

echo "=========================================="
echo "v10 GRPO Training (DDP + PEFT)"
echo "=========================================="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Nodes:      $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')"
echo "GPUs:       $((SLURM_NNODES * 4))"
echo "Master:     ${MASTER_ADDR}"
echo "Output:     $OUTPUT_DIR"
echo "Train data: $TRAIN_DATA"
echo "K samples:  $NUM_SAMPLES"
echo "Start:      $(date)"
echo "=========================================="

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
        v10/train_grpo.py \
        --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
        --train_data '"$TRAIN_DATA"' \
        --val_data '"$VAL_DATA"' \
        --output_dir '"$OUTPUT_DIR"' \
        --lora_r 128 \
        --lora_alpha 256 \
        --lora_dropout 0.05 \
        --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
        --num_samples '"$NUM_SAMPLES"' \
        --temperature 1.0 \
        --top_p 0.95 \
        --clip_range 0.2 \
        --kl_coef 0.001 \
        --max_grounder_tokens 256 \
        --max_actor_tokens 256 \
        --grounder_lr 1e-5 \
        --actor_lr 5e-6 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --num_epochs 4 \
        --gradient_accumulation_steps 2 \
        --logging_steps 5 \
        --save_steps 100 \
        --val_steps 50
'

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
