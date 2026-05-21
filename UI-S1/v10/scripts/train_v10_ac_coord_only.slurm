#!/bin/bash
#SBATCH --job-name=v10_ac
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v10/scripts/logs/train_v10_ac_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v10/scripts/logs/train_v10_ac_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# v10 — coord_only_routing SFT for cooperative LoRA on AC
#
# Key change from v6.5/v9:
#   - v6.5: image→V, coord→V, text→A
#   - v10:  coord→V, everything else→A (including image tokens)
#
# LoRA_V becomes a pure "grounder" that only generates coordinates.
# LoRA_A handles everything: image understanding, thought, action type.
# LoRA_V reads A's thought + image representations via self-attention.
#
# Warm-start from v6.5 AC epoch-4 checkpoint.
# Uses standard cooperative SFT (no credit reweighting).
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
WARM_START="$PROJECT_DIR/train_GUI_360/llamafactory/output/cooperative_v6_5_ac/epoch-4"
COMM="${COMM:-0}"

mkdir -p "$PROJECT_DIR/v10/scripts/logs"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29505
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200

if [ "$COMM" = "1" ]; then
    OUTPUT_DIR="$PROJECT_DIR/v10/output/v10_ac_comm"
    COMM_FLAG="--cooperative_comm"
    LABEL="comm=ON"
else
    OUTPUT_DIR="$PROJECT_DIR/v10/output/v10_ac_nocomm"
    COMM_FLAG=""
    LABEL="comm=OFF"
fi

echo "=========================================="
echo "v10 coord_only_routing SFT (AC, $LABEL)"
echo "Warm-start: $WARM_START"
echo "Output:     $OUTPUT_DIR"
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Master:     ${MASTER_ADDR}"
echo "Start:      $(date)"
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
        train_cooperative.py \
        --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
        --train_data datasets/cooperative_thought_ac/ac_train_thought.jsonl \
        --val_data datasets/cooperative_thought_ac/ac_val_thought.jsonl \
        --output_dir '"$OUTPUT_DIR"' \
        --resume_coop_checkpoint '"$WARM_START"' \
        --lora_r 256 \
        --lora_alpha 512 \
        --lora_dropout 0.05 \
        --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
        --num_agents 2 \
        --gate_type tanh \
        --gate_init 0.0 \
        --gate_lr_multiplier 100.0 \
        --gate_weight_decay 0.0 \
        --routing_mode hard \
        --coord_only_routing \
        --bind_weight 0.0 \
        --per_device_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 5e-6 \
        --num_epochs 4.0 \
        --warmup_ratio 0.03 \
        --save_steps 100 \
        --logging_steps 5 \
        --max_length 8192 \
        '"$COMM_FLAG"'
'

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
