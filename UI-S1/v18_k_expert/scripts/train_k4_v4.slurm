#!/bin/bash
#SBATCH --job-name=v18_k4_v4
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v18_k_expert/scripts/logs/train_k4_v4_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v18_k_expert/scripts/logs/train_k4_v4_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V18 K=4 Expert v4: Fix bfloat16 precision bug
#
# v1-v3 all showed div=1.0000 because:
#   Root cause: bfloat16 mixed precision kills SVD perturbations
#   - A_i = A_avg + δ_i, noise_scale=0.1
#   - Perturbation per element ≈ 1e-4 relative to A_avg
#   - bfloat16 precision ≈ 0.8% (7-bit mantissa)
#   - A_i.to(bfloat16) == A_j.to(bfloat16) for all i,j
#   - Expert outputs h_i = A_i @ x are BITWISE IDENTICAL
#   - cos_sim = 1.0 exactly, gradient = 0 exactly
#
# v4 fix: diversity loss on A matrices in float32
#   - compute_diversity_loss uses A params directly (float32)
#   - Subtract mean A to isolate expert-specific deviations
#   - cos_sim of (A_i - A_mean) has real values and gradients
#   - No output caching needed, no precision loss
#
# Diversity weight 0.1 (up from 0.01): A-matrix diversity
# starts near 0 (Gram-Schmidt orthogonal init), need
# stronger weight to maintain separation against RL gradient.
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29504
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200

mkdir -p "${PROJECT_DIR}/v18_k_expert/scripts/logs"

echo "=========================================="
echo "V18: K=4 Expert v4 (fix bf16 precision)"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}"
echo "Start:   $(date)"
echo "Fix: diversity on A matrices (float32)"
echo "=========================================="

OUTPUT_DIR="${PROJECT_DIR}/checkpoints/v18_rl_k4_v4"
SFT_CKPT="${PROJECT_DIR}/checkpoints/v18_svd_k4_7mod"

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
        v18_k_expert/train_k_expert_gspo.py \
        --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
        --train_data v12_gui_360/data/gui360_train_2000_balanced.jsonl \
        --val_data v12_gui_360/data/gui360_val_50.jsonl \
        --output_dir '"$OUTPUT_DIR"' \
        --sft_checkpoint '"$SFT_CKPT"' \
        --num_experts 4 \
        --comm_topology top2 \
        --lora_r 128 \
        --lora_alpha 256 \
        --lora_dropout 0.05 \
        --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
        --num_comm_rounds 2 \
        --balance_weight 0.0 \
        --diversity_weight 0.1 \
        --routing_noise_scale 1.0 \
        --route_init_std 0.01 \
        --image_max_pixels 602112 \
        --num_trajectories 8 \
        --temperature 1.0 \
        --top_p 0.95 \
        --clip_range 0.2 \
        --max_new_tokens 256 \
        --match_threshold 0.5 \
        --kl_coef 0.001 \
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
        --num_epochs 3 \
        --gradient_accumulation_steps 4 \
        --logging_steps 1 \
        --save_steps 3 \
        --val_steps 5
'

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
