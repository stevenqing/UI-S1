#!/bin/bash
#SBATCH --job-name=v17_sft_warmup
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v17_self_evolve/scripts/logs/sft_warmup_v17_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v17_self_evolve/scripts/logs/sft_warmup_v17_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V17.1 SFT Warmup — Teach Dual-Aux Format
#
# Trains cooperative LoRA on template-based dual-aux data:
#   <aux_a>perspective A</aux_a><aux_b>perspective B</aux_b><decision>action</decision>
#
# Only 0.5 epoch — teaches format structure, NOT content.
# Uses V15 train_cooperative_sft.py (V13 wrapper, compatible weights).
#
# Steps:
#   1. Generate SFT data with prepare_dual_aux_sft.py
#   2. Train with train_cooperative_sft.py (0.5 epoch ≈ half the data)
#
# nodes=2  ngpu/node=4  grad_accum=8  ->  ~64 eff_bs
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29504
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200

mkdir -p "${PROJECT_DIR}/v17_self_evolve/scripts/logs"
mkdir -p "${PROJECT_DIR}/v17_self_evolve/data"

echo "=========================================="
echo "V17.1 SFT Warmup — Dual-Aux Format Teaching"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}"
echo "Start:   $(date)"
echo "=========================================="

# Step 1: Prepare dual-aux SFT data (runs on single node, fast)
echo "Preparing SFT data..."
export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
export PYTHONPATH="${PROJECT_DIR}:$PYTHONPATH"

python "${PROJECT_DIR}/v17_self_evolve/prepare_dual_aux_sft.py" \
    --input_data "${PROJECT_DIR}/v12_gui_360/data/gui360_train_2000_balanced.jsonl" \
    --output_data "${PROJECT_DIR}/v17_self_evolve/data/dual_aux_sft_train.jsonl" \
    --output_val "${PROJECT_DIR}/v17_self_evolve/data/dual_aux_sft_val.jsonl" \
    --val_episodes 50

echo "SFT data prepared."

# Step 2: Train cooperative SFT (0.5 epoch)
OUTPUT_DIR="${PROJECT_DIR}/checkpoints/v17_dual_aux_sft_warmup"

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
        --train_data v17_self_evolve/data/dual_aux_sft_train.jsonl \
        --val_data v17_self_evolve/data/dual_aux_sft_val.jsonl \
        --output_dir '"$OUTPUT_DIR"' \
        --lora_r 128 \
        --lora_alpha 256 \
        --lora_dropout 0.05 \
        --target_modules q_proj k_proj v_proj o_proj \
        --num_comm_rounds 2 \
        --balance_weight 0.01 \
        --image_max_pixels 602112 \
        --lora_lr 2e-5 \
        --route_lr 1e-3 \
        --weight_decay 0.01 \
        --max_grad_norm 1.0 \
        --warmup_ratio 0.03 \
        --num_epochs 1 \
        --gradient_accumulation_steps 8 \
        --logging_steps 10 \
        --save_steps 200 \
        --val_steps 100
'

echo "=========================================="
echo "SFT Warmup Done: $(date)"
echo "Checkpoint: ${OUTPUT_DIR}"
echo "=========================================="
