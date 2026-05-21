#!/bin/bash
#SBATCH --job-name=coop_v65t_re
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v6_5_thought_resume_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v6_5_thought_resume_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Cooperative LoRA v6.5 — RESUME from checkpoint-2300
#
# Why resume:
#   The original v6.5 run (job 3743502) hit a NCCL socket deadlock
#   between nodes nid011054 ↔ nid011032 sometime between epoch 3.0
#   and epoch 4.0. Last fully-written checkpoint is checkpoint-2300
#   at step 2300 / epoch 3.0144. checkpoint-2400 is partially written
#   (missing trainer_state.json, scheduler.pt, rng_state_0.pth).
#
#   epoch-3 LoRA-only snapshot was saved by CooperativeSaveCallback,
#   so we already have the ep3 eval point. The resume only needs to
#   produce epoch-4.
#
# Resume mechanism:
#   --resume_coop_checkpoint <ckpt-2300>/cooperative
#     -> loads lora_v.pt / lora_a.pt / lora_comm.pt into wrapper
#     -> train_cooperative.py line 744-753 also passes
#        checkpoint-2300/ to trainer.train(resume_from_checkpoint=...)
#     -> HF Trainer restores optimizer / scheduler / RNG / step
#
# Everything else identical to original v6.5 train script.
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
echo "Cooperative LoRA v6.5 — RESUME from checkpoint-2300 (epoch 3.01)"
echo "  Resuming v6.5 from step 2300, expected to finish ~step 3050 (epoch 4)"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}"
echo "Start:   $(date)"
echo "=========================================="

OUTPUT_DIR="train_GUI_360/llamafactory/output/cooperative_v6_5_comm_thought"
RESUME_COOP="${OUTPUT_DIR}/checkpoint-2300/cooperative"

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
        --resume_coop_checkpoint '"$RESUME_COOP"' \
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
