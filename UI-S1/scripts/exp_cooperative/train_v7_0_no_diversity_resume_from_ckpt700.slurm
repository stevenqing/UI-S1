#!/bin/bash
#SBATCH --job-name=coop_v70n_re
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v7_0_no_diversity_resume_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v7_0_no_diversity_resume_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=workq
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Cooperative LoRA v7.0 NO-diversity — RESUME from checkpoint-700
#
# Why resume:
#   Original v7.0 nodiv run (job 3745577) crashed at step 730 with
#   NCCL collective timeout (1800038 ms = 30 min) on rank 24
#   (nid010181). All ranks SIGABRT'd at 11:54:20.
#
#   Last fully-written checkpoint: checkpoint-700 (step 700, epoch 0.92).
#   This is the same NCCL deadlock pattern that hit v6.5 (job 3743502)
#   between epoch 3 and 4 — the cluster's HSN fabric appears to drop
#   socket connections sporadically under multi-node distributed loads.
#
# Resume mechanism: same as v6.5 resume.
#   --resume_coop_checkpoint <ckpt-700>/cooperative
#     -> loads lora_v.pt / lora_a.pt / lora_comm.pt
#     -> HF Trainer restores optimizer / scheduler / RNG / step
#     -> CooperativeTrainer._load_from_checkpoint override (added in
#        train_cooperative.py for the v6.5 resume) skips the model
#        weight reload check.
#
# Everything else identical to original v7.0 nodiv train script.
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
echo "Cooperative LoRA v7.0 NO-diversity — RESUME from checkpoint-700 (epoch 0.92)"
echo "  Resuming v7.0 nodiv from step 730 -> step 3052 (epoch 4)"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}"
echo "Start:   $(date)"
echo "=========================================="

OUTPUT_DIR="train_GUI_360/llamafactory/output/cooperative_v7_0_merge_no_diversity"
RESUME_COOP="${OUTPUT_DIR}/checkpoint-700/cooperative"

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
