#!/bin/bash
#SBATCH --job-name=v15_gspo
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/train_gspo_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/train_gspo_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V15 GUI-360: On-Policy RL + Per-Step Group-Relative Advantage
#
# Key differences from V14:
#   1. On-policy rollout: model uses its OWN predictions as history
#   2. Stop-on-error: type errors kill trajectory → implicit traj credit
#   3. Per-step group-relative advantage (not trajectory-level)
#   4. Token-level PPO clip=0.2 → routing gets fine-grained signal
#   5. V13 cooperative LoRA (shared B, proven)
#   6. K=8 full on-policy trajectories
#   7. temperature=0.7 (more stable than V14's 1.0)
#
# Why per-step not trajectory-level (GSPO):
#   - Routing needs per-step signal for action-type specialization
#   - Trajectory credit is IMPLICIT via stop-on-error:
#     type errors → no gradient for later steps
#   - Per-step group contrast: compare K trajs AT EACH STEP
#
# Memory estimate:
#   K=8 × T~3 steps = ~24 serial forward passes for generation per episode
#   Training: active_trajs × T steps → ~16 forward+backward passes
#   4 nodes × 4 GPUs = 16 GPUs, 1 episode per GPU, grad_accum=4
# ============================================

set -e

# Create logs directory
mkdir -p /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs

cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1

# -- Environment --
module load cpe/24.07 || true
source /home/a5l/shuqing.a5l/miniconda3/etc/profile.d/conda.sh
conda activate ui-s1

# -- NCCL --
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=LOC
export NCCL_CROSS_NIC=1

# -- Master node --
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Start: $(date)"

srun --ntasks-per-node=1 bash -c '
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  export WORLD_SIZE=$(($SLURM_JOB_NUM_NODES * 4))

  torchrun \
    --nproc_per_node=4 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    v15_gui_360/train_trajectory_gspo.py \
      --model_path checkpoints/gui_action \
      --train_data v12_gui_360/data/gui360_train_2000_balanced.jsonl \
      --val_data v12_gui_360/data/gui360_val_50.jsonl \
      --output_dir checkpoints/v15_onpolicy_gspo \
      --lora_r 128 \
      --lora_alpha 256 \
      --lora_dropout 0.05 \
      --target_modules q_proj k_proj v_proj o_proj \
      --num_comm_rounds 2 \
      --balance_weight 0.01 \
      --num_trajectories 8 \
      --temperature 0.7 \
      --top_p 0.9 \
      --clip_range 0.2 \
      --max_new_tokens 256 \
      --match_threshold 0.5 \
      --lora_lr 1e-5 \
      --route_lr 1e-3 \
      --weight_decay 0.01 \
      --max_grad_norm 1.0 \
      --num_epochs 3 \
      --gradient_accumulation_steps 4 \
      --logging_steps 1 \
      --save_steps 25 \
      --val_steps 25
'

echo "Done: $(date)"
