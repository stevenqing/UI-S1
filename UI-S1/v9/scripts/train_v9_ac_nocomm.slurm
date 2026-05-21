#!/bin/bash
#SBATCH --job-name=v9_ac_nocomm
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v9/scripts/logs/train_v9_ac_nocomm_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v9/scripts/logs/train_v9_ac_nocomm_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# v9 step-1 — Reweighted-SFT for cooperative LoRA, AC single-turn, comm=OFF.
#
# Identical to train_v9_ac_comm.slurm except --cooperative_comm is omitted.
# This is the head-to-head baseline that isolates the contribution of the
# cross-agent communication gates under v9 credit-weighted SFT.
#
# Same warm-start, same credit cache, same hyperparameters → only the gate
# variable differs.
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
WARM_START="$PROJECT_DIR/train_GUI_360/llamafactory/output/cooperative_v6_5_ac/epoch-4"
CREDIT_FILE="$PROJECT_DIR/v9/credits/v6_5_ac_ep4/credit_cache.jsonl"

mkdir -p "$PROJECT_DIR/v9/scripts/logs"

if [ ! -f "$CREDIT_FILE" ]; then
    echo "ERROR: credit cache not found at $CREDIT_FILE" >&2
    echo "Run v9/scripts/decode_v9_ac_credits.slurm then merge_credit_shards.sh." >&2
    exit 1
fi

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29504
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200

OUTPUT_DIR="$PROJECT_DIR/v9/output/v9_ac_nocomm"

echo "=========================================="
echo "v9 step-1 (AC, comm=OFF) — warm-start from v6.5 ep4"
echo "Credit:  $CREDIT_FILE"
echo "Output:  $OUTPUT_DIR"
echo "Job ID:  ${SLURM_JOB_ID}"
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
        v9/train_v9.py \
        --credit_file '"$CREDIT_FILE"' \
        --credit_fallback_w 1.0 \
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
        --coord_routing \
        --bind_weight 0.0 \
        --per_device_batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 5e-6 \
        --num_epochs 4.0 \
        --warmup_ratio 0.03 \
        --save_steps 100 \
        --logging_steps 5 \
        --max_length 8192
'

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
