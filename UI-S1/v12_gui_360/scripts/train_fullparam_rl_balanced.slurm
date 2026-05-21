#!/bin/bash
#SBATCH --job-name=fp_bal
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v12_gui_360/scripts/logs/train_fullparam_rl_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v12_gui_360/scripts/logs/train_fullparam_rl_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Full-Parameter RL Baseline
#
# Same RL as V12 but fine-tunes ALL model parameters (~7.6B)
# Uses ZeroRedundancyOptimizer (ZeRO-1) for optimizer state sharding:
#   ~91 GB optimizer states / 16 GPUs = ~5.7 GB/GPU
#   + ~15 GB model weights/GPU + ~15 GB gradients/GPU = ~36 GB/GPU
#   Fits in 80 GB with gradient checkpointing
#
# No LoRA, no routing, no balance loss, no KL penalty (kl_coef=0)
# lr=1e-6 (10x lower than LoRA due to full-param)
#
#   nodes=4  ngpu/node=4  grad_accum=4  ->  eff_bs = 4*4*4 = 64 episodes
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29503
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200

mkdir -p "${PROJECT_DIR}/v12_gui_360/scripts/logs"

echo "=========================================="
echo "Full-Parameter RL Baseline (~7.6B params)"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}"
echo "Start:   $(date)"
echo "=========================================="

OUTPUT_DIR="${PROJECT_DIR}/checkpoints/fullparam_gui360_rl_balanced"

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
        v12_gui_360/train_fullparam_rl.py \
        --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
        --train_data v12_gui_360/data/gui360_train_2000_balanced.jsonl \
        --val_data v12_gui_360/data/gui360_val_50.jsonl \
        --output_dir '"$OUTPUT_DIR"' \
        --image_max_pixels 602112 \
        --num_samples 8 \
        --temperature 1.0 \
        --top_p 0.95 \
        --clip_range 0.2 \
        --max_new_tokens 256 \
        --match_threshold 0.5 \
        --spwa_decay 0.5 \
        --dapo_threshold 0.0 \
        --step_adv_weight 0.5 \
        --w_format 0.1 \
        --w_type 0.2 \
        --w_content 0.7 \
        --lr 1e-6 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --num_epochs 3 \
        --gradient_accumulation_steps 4 \
        --logging_steps 1 \
        --save_steps 25 \
        --val_steps 25
'

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
