#!/bin/bash
#SBATCH --job-name=v13_gui360_rl
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/train_rl_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/scripts/logs/train_rl_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V13 GUI-360: Iterative Cooperative Reasoning
#
# Desktop GUI benchmark (Word/Excel/PowerPoint)
# From raw Qwen2.5-VL-7B (no SFT checkpoint):
#   - Iterative cooperative LoRA with T=2 communication rounds
#   - Per-token soft routing + gated expert communication in r-space
#   - K=8 rollouts with routing noise for exploration
#   - SP (Sequential Progress) as trajectory-level signal
#   - GiGPO cross-trajectory advantage normalization
#   - SPWA (Sequential Progress Weighted Advantage) step weighting
#   - PPO clipped loss + KL penalty + balance loss
#
#   nodes=2  ngpu/node=4  grad_accum=4  ->  eff_bs = 2*4*4 = 32 episodes
#   lora_lr=1e-5, route_lr=1e-3 (also used for comm params)
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29502
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200

# Create log directory
mkdir -p "${PROJECT_DIR}/v13_gui_360/scripts/logs"

echo "=========================================="
echo "V13 GUI-360 RL — Iterative Cooperative Reasoning"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}"
echo "Start:   $(date)"
echo "=========================================="

OUTPUT_DIR="${PROJECT_DIR}/checkpoints/v13_gui360_rl"

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
        v13_gui_360/train_trajectory_rl.py \
        --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
        --train_data v12_gui_360/data/gui360_train_2000.jsonl \
        --val_data v12_gui_360/data/gui360_val_50.jsonl \
        --output_dir '"$OUTPUT_DIR"' \
        --lora_r 128 \
        --lora_alpha 256 \
        --lora_dropout 0.05 \
        --target_modules q_proj k_proj v_proj o_proj \
        --num_comm_rounds 2 \
        --balance_weight 0.01 \
        --routing_noise_scale 0.5 \
        --image_max_pixels 602112 \
        --num_samples 8 \
        --temperature 1.0 \
        --top_p 0.95 \
        --clip_range 0.2 \
        --kl_coef 0.001 \
        --max_new_tokens 256 \
        --match_threshold 0.5 \
        --spwa_decay 0.5 \
        --dapo_threshold 0.0 \
        --step_adv_weight 0.5 \
        --w_format 0.1 \
        --w_type 0.2 \
        --w_content 0.7 \
        --lora_lr 1e-5 \
        --route_lr 1e-3 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --num_epochs 4 \
        --gradient_accumulation_steps 4 \
        --logging_steps 1 \
        --save_steps 25 \
        --val_steps 25
'

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
