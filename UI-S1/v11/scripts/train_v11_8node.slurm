#!/bin/bash
#SBATCH --job-name=v11_traj_grpo
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v11/scripts/logs/train_v11_8node_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v11/scripts/logs/train_v11_8node_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# v11: Cooperative LoRA Trajectory GRPO
#
# Architecture: CooperativeVLMWrapper (v6.5) with h-space communication
#   - LEARNED routing: per-layer nn.Linear(D,1)→sigmoid per-token soft blend
#   - Per-layer communication: h_v += gate * W_av @ h_a (and vice versa)
#   - Single forward pass, fully differentiable communication + routing
#   - gate_init=0.5 (warm-start communication, tanh(0.5)≈0.46)
#   - Balance loss (λ=0.01) prevents router collapse
#   - Auxiliary coord prediction head on image tokens → direct LoRA_V gradient
#
# Training: Trajectory GRPO
#   - Episode-level data: ~950 episodes, avg 6.5 steps/episode
#   - K=8 rollouts per step with dense multi-component rewards
#   - Discounted returns (γ=0.5) + dual-level advantage normalization
#   - DAPO filtering on collapsed advantages
#
# Infrastructure:
#   nodes=8  gpus/node=4  batch_size=1 (episode/gpu)  grad_accum=4
#   → eff_batch = 8*4*1*4 = 128 episodes per gradient step
#   → steps/epoch ≈ 950/128 ≈ 8
#   → total ≈ 32 steps over 4 epochs
#
# Optimizer (3-group):
#   lora_lr=1e-5  comm_lr=1e-3 (100× multiplier)
#   weight_decay=0.01 (LoRA only, comm has wd=0)
#
# Estimated time: ~1.8h for 4 epochs
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

# Create log directory
mkdir -p "${PROJECT_DIR}/v11/scripts/logs"

# ── NCCL / distributed config ──
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200

echo "=========================================="
echo "v11 Trajectory GRPO — Cooperative LoRA"
echo "=========================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Nodes:       ${SLURM_NNODES}"
echo "GPUs/node:   4"
echo "Total GPUs:  $((SLURM_NNODES * 4))"
echo "Master:      ${MASTER_ADDR}"
echo "Start:       $(date)"
echo "=========================================="

OUTPUT_DIR="${PROJECT_DIR}/v11/output/v11_traj_grpo"

# ── Step 1: Prepare episode data (run once on head node) ──
TRAIN_DATA="${PROJECT_DIR}/v11/data/ac_train_episodes.jsonl"
VAL_DATA="${PROJECT_DIR}/v11/data/ac_val_episodes.jsonl"

if [ ! -f "${TRAIN_DATA}" ]; then
    echo "Preparing episode data..."
    export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
    cd "${PROJECT_DIR}"
    python v11/prepare_episode_data.py \
        --input datasets/ui_s1_dataset/ui_s1_train_with_desc.jsonl \
        --train_output "${TRAIN_DATA}" \
        --val_output "${VAL_DATA}" \
        --val_episodes 50
    echo "Episode data prepared."
fi

# ── Step 2: Launch distributed training ──
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
        v11/train_trajectory_grpo.py \
        --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
        --train_data '"$TRAIN_DATA"' \
        --val_data '"$VAL_DATA"' \
        --output_dir '"$OUTPUT_DIR"' \
        --lora_r 256 \
        --lora_alpha 512 \
        --lora_dropout 0.05 \
        --target_modules q_proj k_proj v_proj o_proj \
        --gate_type tanh \
        --gate_init 0.5 \
        --routing_mode learned \
        --balance_weight 0.01 \
        --num_samples 8 \
        --temperature 1.0 \
        --top_p 0.95 \
        --clip_range 0.2 \
        --kl_coef 0.001 \
        --max_new_tokens 256 \
        --gamma 0.5 \
        --dapo_threshold 0.3 \
        --w_format 0.1 \
        --w_type 0.2 \
        --w_content 0.7 \
        --aux_coord_weight 0.1 \
        --lora_lr 1e-5 \
        --comm_lr 1e-3 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --num_epochs 4 \
        --gradient_accumulation_steps 4 \
        --logging_steps 1 \
        --save_steps 8 \
        --val_steps 8
'

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
