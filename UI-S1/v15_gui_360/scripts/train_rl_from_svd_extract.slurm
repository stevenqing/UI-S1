#!/bin/bash
#SBATCH --job-name=v15_rl_svd
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/train_rl_from_svd_extract_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v15_gui_360/scripts/logs/train_rl_from_svd_extract_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4
# ============================================
# V15 GUI-360: Cooperative RL from SVD-Extracted LoRA
#
# Warmstart from PEFT SVD extraction of balanced full-param SFT
# (checkpoint-250, TSR=22.2%) converted to cooperative V13 format.
#
# Key difference from train_rl_from_coop_sft.slurm:
#   - 7 target modules (q/k/v/o + gate/up/down) vs 4 (attn only)
#   - SVD-extracted init: TSR=18.6% cooperative, 18.1% standard
#   - down_proj uses fixed r=0.5 routing (dim mismatch handled)
#
#   nodes=4  ngpu/node=4  grad_accum=4  ->  eff_bs = 64 episodes
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
SFT_CKPT="$PROJECT_DIR/checkpoints/v15_peft_balanced_cooperative_r128"
OUTPUT_DIR="$PROJECT_DIR/checkpoints/v15_rl_from_svd_extract"

# Verify checkpoint
if [ ! -f "$SFT_CKPT/lora_weights.pt" ]; then
    echo "ERROR: lora_weights.pt not found at $SFT_CKPT"
    exit 1
fi

echo "=========================================="
echo "V15 RL from SVD-Extracted Cooperative LoRA"
echo "=========================================="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Nodes:      ${SLURM_NNODES}"
echo "Start:      $(date)"
echo "Checkpoint: $SFT_CKPT"
echo "Output:     $OUTPUT_DIR"
echo "Modules:    q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
echo "=========================================="

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29502
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200

srun --ntasks-per-node=1 bash -c '
    export PATH="/home/a5l/shuqing.a5l/miniconda3/envs/ui-s1/bin:$PATH"
    export CONDA_DEFAULT_ENV=ui-s1
    export PYTHONPATH="'"$PROJECT_DIR"':$PYTHONPATH"
    export PYTHONUNBUFFERED=1
    export WANDB_MODE=disabled
    export TMPDIR=/tmp

    cd "'"$PROJECT_DIR"'"

    torchrun \
        --nproc_per_node=4 \
        --nnodes=$SLURM_NNODES \
        --node_rank=$SLURM_NODEID \
        --master_addr='"$MASTER_ADDR"' \
        --master_port='"$MASTER_PORT"' \
        v13_gui_360/train_trajectory_rl.py \
        --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
        --sft_checkpoint '"$SFT_CKPT"' \
        --train_data v12_gui_360/data/gui360_train_2000_balanced.jsonl \
        --val_data v12_gui_360/data/gui360_val_50.jsonl \
        --output_dir '"$OUTPUT_DIR"' \
        --lora_r 128 \
        --lora_alpha 256 \
        --lora_dropout 0.05 \
        --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
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
        --num_epochs 3 \
        --gradient_accumulation_steps 4 \
        --logging_steps 1 \
        --save_steps 25 \
        --val_steps 25
'

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
