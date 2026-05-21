#!/bin/bash
#SBATCH --job-name=ac_van_sft
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v6_5_ac_vanilla_sft_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v6_5_ac_vanilla_sft_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Vanilla SFT r=512 — AndroidControl
# Fair baseline: IDENTICAL to cooperative v6.5 AC except no cooperative mechanism.
#
# Same as train_v6_5_ac_comm_thought.slurm:
#   - Same multi-turn data (ac_train_multiturn.jsonl)
#   - Same chat template (processor.apply_chat_template)
#   - Same image_max_pixels=602112, max_length=16384
#   - Same eff_bs=32 (2 nodes × 4 GPU × pdbs=1 × grad_accum=4)
#   - Same lr=1e-5, warmup=0.03, weight_decay=0.1, cosine schedule
#   - Same 4 epochs
#
# Only difference: --vanilla_lora with r=512 (total-param match: 2×r=256 ≈ r=512)
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
echo "Vanilla SFT r=512 (AC) — Fair baseline for Cooperative v6.5"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}"
echo "Start:   $(date)"
echo "=========================================="

OUTPUT_DIR="train_GUI_360/llamafactory/output/v6_5_ac_vanilla_sft_r512"

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
        --vanilla_lora \
        --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
        --train_data datasets/cooperative_thought_ac/ac_train_multiturn.jsonl \
        --val_data datasets/cooperative_thought_ac/ac_val_multiturn.jsonl \
        --output_dir '"$OUTPUT_DIR"' \
        --lora_r 512 \
        --lora_alpha 1024 \
        --lora_dropout 0.05 \
        --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
        --per_device_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1e-5 \
        --num_epochs 4.0 \
        --warmup_ratio 0.03 \
        --save_steps 999999 \
        --logging_steps 5 \
        --max_length 16384 \
        --image_max_pixels 602112
'

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
