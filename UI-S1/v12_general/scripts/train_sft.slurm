#!/bin/bash
#SBATCH --job-name=v12_sft
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v12_general/scripts/logs/train_sft_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v12_general/scripts/logs/train_sft_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V12 Phase 1: Soft Cooperative LoRA SFT
#
# Soft-routed cooperative LoRA (no token masks, no side-channel comm)
# with per-layer learned routing vectors.
#
#   nodes=1  ngpu/node=4  pbs=1  grad_accum=4  ->  eff_bs = 1*4*1*4 = 16
#   lora_r=256, 28 layers × 4 modules (q/k/v/o), shared_B, ~308M params
#   route_lr = base_lr × 100 for fast routing convergence
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200

# Create log directory
mkdir -p "${PROJECT_DIR}/v12_general/scripts/logs"

echo "=========================================="
echo "V12 SFT — Soft Cooperative LoRA (r=256, 4 modules)"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}"
echo "Start:   $(date)"
echo "=========================================="

OUTPUT_DIR="${PROJECT_DIR}/checkpoints/v12_sft"

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
        v12_general/train_cooperative_sft.py \
        --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
        --train_data datasets/cooperative_thought_ac/ac_train_multiturn.jsonl \
        --output_dir '"$OUTPUT_DIR"' \
        --lora_r 256 \
        --lora_alpha 512 \
        --lora_dropout 0.05 \
        --target_modules q_proj k_proj v_proj o_proj \
        --balance_weight 0.01 \
        --route_lr_multiplier 100.0 \
        --per_device_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-5 \
        --weight_decay 0.01 \
        --num_epochs 4.0 \
        --warmup_ratio 0.03 \
        --save_steps 50 \
        --logging_steps 5 \
        --max_length 16384 \
        --image_max_pixels 602112
'

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
