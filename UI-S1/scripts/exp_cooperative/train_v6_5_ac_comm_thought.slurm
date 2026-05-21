#!/bin/bash
#SBATCH --job-name=coop_v65_ac
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v6_5_ac_thought_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v6_5_ac_thought_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Cooperative LoRA v6.5 — AndroidControl MULTI-TURN training
#
# Multi-turn trajectory format matching eval's gen_next_round() message structure.
# Each episode = system + N×(user+assistant) turns, avg 6.5 turns/episode.
#
# The wrapper / hyperparameters mirror v6.5 (tanh gate, r=256, lr_mult=100).
# Smaller eff_bs because multi-turn data has only 950 episodes (not 6.5K samples).
#
#   nodes=2  ngpu/node=4  pdbs=1  grad_accum=4  ->  eff_bs = 2*4*1*4 = 32
#   steps/epoch ≈ 950/32 ≈ 30,  total ≈ 120 steps over 4 epochs
#   max_length=16384, image_max_pixels=602112 (~900 vis tokens/image)
#   AC eps avg 6.5 turns. At default resolution (~3354 tok/img) that's
#   22K+ total. With 602112, ~900 tok/img × 6.5 ≈ 5850 + text ≈ 8K total.
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
echo "Cooperative LoRA v6.5 (AC) MULTI-TURN — eff_bs=32, r=256, tanh gate, lr_mult=100"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}"
echo "Start:   $(date)"
echo "=========================================="

OUTPUT_DIR="train_GUI_360/llamafactory/output/cooperative_v6_5_ac"

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
