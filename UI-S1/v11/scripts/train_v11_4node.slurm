#!/bin/bash
#SBATCH --job-name=v11_traj_grpo
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v11/scripts/logs/train_v11_4node_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v11/scripts/logs/train_v11_4node_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# v11: Cooperative LoRA Trajectory GRPO (4-node)
#
# Architecture: Learned router + h-space communication
#   - Per-layer nn.Linear(D,1)→sigmoid per-token soft blend
#   - Communication: h_v += gate * W_av @ h_a (and vice versa)
#   - Aux coord head for direct LoRA_V gradient
#
# 4 nodes × 4 GPUs = 16 GPUs, grad_accum=8
# → eff_batch = 4*4*1*8 = 128 episodes per gradient step
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

mkdir -p "${PROJECT_DIR}/v11/scripts/logs"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=LOC
export NCCL_CROSS_NIC=1
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200

echo "=========================================="
echo "v11 Trajectory GRPO — Cooperative LoRA (4-node)"
echo "=========================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Nodes:       ${SLURM_NNODES}"
echo "Total GPUs:  $((SLURM_NNODES * 4))"
echo "Master:      ${MASTER_ADDR}"
echo "Start:       $(date)"
echo "=========================================="

OUTPUT_DIR="${PROJECT_DIR}/v11/output/v11_traj_grpo"

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
        --gradient_accumulation_steps 8 \
        --logging_steps 1 \
        --save_steps 8 \
        --val_steps 8
'

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
