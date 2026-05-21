#!/bin/bash
#SBATCH --job-name=coop_v65_ody
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v6_5_odyssey_thought_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v6_5_odyssey_thought_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
# Note: was nodes=8 for single-turn (101K samples), reduced to 4 for multi-turn (6.5K episodes)
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Cooperative LoRA v6.5 — GUI-Odyssey MULTI-TURN training
#
# Multi-turn trajectory format matching eval's gen_next_round() message structure.
# Each episode = system + N×(user+assistant) turns, avg 15.3 turns/episode.
#
# The wrapper / hyperparameters mirror v6.5 (tanh gate, r=256, lr_mult=100).
# Multi-turn data has 6,468 episodes (not 101K single-step samples).
#
#   nodes=4  ngpu/node=4  pdbs=1  grad_accum=4  ->  eff_bs = 4*4*1*4 = 64
#   steps/epoch ≈ 6468/64 ≈ 101,  total ≈ 404 steps over 4 epochs
#   max_length=16384, image_max_pixels=401408 (~506 vis tokens/image)
#   Odyssey eps avg 15+ steps. At default resolution (~4450 tok/img) that's
#   67K+ tokens — way over budget. Reducing to 401408 gives ~506 tok/img,
#   so 15 images ≈ 7590 + text ≈ 12K total. Turns that still overflow are
#   dropped from the beginning (keep most recent actions).
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29502
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200

echo "=========================================="
echo "Cooperative LoRA v6.5 (Odyssey) MULTI-TURN — eff_bs=64, r=256, tanh gate, lr_mult=100"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}"
echo "Start:   $(date)"
echo "=========================================="

OUTPUT_DIR="train_GUI_360/llamafactory/output/cooperative_v6_5_odyssey"

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
        --train_data datasets/cooperative_thought_odyssey/odyssey_train_multiturn.jsonl \
        --val_data datasets/cooperative_thought_odyssey/odyssey_val_multiturn.jsonl \
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
        --max_length 16384 \
        --image_max_pixels 401408
'

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
