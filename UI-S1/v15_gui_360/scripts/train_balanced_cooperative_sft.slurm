#!/bin/bash
#SBATCH --job-name=v15_coop_sft
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/train_balanced_cooperative_sft_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/train_balanced_cooperative_sft_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V15: V13 Cooperative LoRA SFT on Balanced Data
#
# Tests if V13's cooperative architecture helps even without RL.
# Same architecture as V13 RL (r=128, T=2 comm rounds, 4 modules).
# Same base model (Qwen2.5-VL-7B-Instruct), same data (balanced 2000 eps).
#
# eff_bs = 2 nodes * 4 gpu * 1 * 16 = 128
# lora_lr=1e-5, route_lr=1e-3, 4 epochs
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
OUTPUT_DIR="${PROJECT_DIR}/checkpoints/v15_balanced_cooperative_sft"
LOG_DIR="$PROJECT_DIR/v15_gui_360/scripts/logs"
mkdir -p "$LOG_DIR"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29502
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200

echo "=========================================="
echo "V15: Cooperative LoRA SFT on Balanced Data"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}"
echo "Start:   $(date)"
echo "=========================================="

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
        v15_gui_360/train_cooperative_sft.py \
        --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
        --train_data v15_gui_360/data/gui360_balanced_train.jsonl \
        --val_data v15_gui_360/data/gui360_balanced_val.jsonl \
        --output_dir '"$OUTPUT_DIR"' \
        --lora_r 128 \
        --lora_alpha 256 \
        --lora_dropout 0.05 \
        --target_modules q_proj k_proj v_proj o_proj \
        --num_comm_rounds 2 \
        --balance_weight 0.01 \
        --image_max_pixels 1003520 \
        --cutoff_len 8192 \
        --lora_lr 1e-5 \
        --route_lr 1e-3 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --warmup_ratio 0.03 \
        --num_epochs 4 \
        --gradient_accumulation_steps 16 \
        --logging_steps 10 \
        --save_steps 50 \
        --val_steps 50
'

echo "=========================================="
echo "Done: $(date)"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
