#!/bin/bash
#SBATCH --job-name=coop_v65_ac_cr
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v6_5_ac_coord_routing_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v6_5_ac_coord_routing_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Cooperative LoRA v6.5 + coord_routing + cooperative reasoning — AndroidControl
#
# CHANGES vs v6.5:
#   1. --coord_routing: route coordinate/bbox digit tokens to LoRA_V
#      Gives LoRA_V direct CE loss on spatial tokens (~9.8% of assistant tokens)
#   2. --coop_reasoning_alpha 0.3: α-mix assistant tokens through V+A
#      Think/action tokens get 0.3·V + 0.7·A instead of pure A
#      LoRA_V effective CE weight: 0% → 36.9%
#
# Semantic division:
#   LoRA_V: image tokens (1.0) + coord/bbox digits (1.0) + reasoning (0.3)
#   LoRA_A: reasoning (0.7) + non-assistant tokens (1.0)
#
# All other hyperparameters identical to train_v6_5_ac_comm_thought.slurm
#   nodes=2  ngpu/node=4  pdbs=1  grad_accum=4  ->  eff_bs = 32
#   r=256, alpha=512, tanh gates, gate_lr_mult=100
#   max_length=16384, image_max_pixels=602112 (~900 vis tokens/image)
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200

echo "=========================================="
echo "Cooperative LoRA v6.5 + COORD_ROUTING + COOP_REASONING (AC) — eff_bs=32, r=256, tanh, lr_mult=100"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}"
echo "Start:   $(date)"
echo "=========================================="

OUTPUT_DIR="train_GUI_360/llamafactory/output/cooperative_v6_5_ac_coord_routing"

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
        --train_data datasets/cooperative_thought_ac/ac_train_multiturn.jsonl \
        --val_data datasets/cooperative_thought_ac/ac_val_multiturn.jsonl \
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
        --coord_routing \
        --coop_reasoning_alpha 0.3 \
        --bind_weight 0.0 \
        --per_device_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1e-5 \
        --num_epochs 4.0 \
        --warmup_ratio 0.03 \
        --save_steps 50 \
        --eval_steps 50 \
        --logging_steps 5 \
        --max_length 16384 \
        --image_max_pixels 602112
'

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
