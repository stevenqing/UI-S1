#!/bin/bash
#SBATCH --job-name=v17_dualaux_rl
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v17_self_evolve/scripts/logs/train_v17_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v17_self_evolve/scripts/logs/train_v17_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V17.1: Dual-Aux RL — Phase-Conditional Hard Routing
#
# Three-phase generation: <aux_a>...<aux_b>...<decision>
#   - Phase-conditional hard routing: aux_a→Expert A, aux_b→Expert B
#   - Diversity loss on r-space expert outputs
#   - Custom autoregressive generation with mid-sequence routing switch
#   - SFT warmup teaches format; RL learns aux content
#
#   nodes=4  ngpu/node=4  grad_accum=4  ->  eff_bs = 4*4*4 = 64 episodes
#   lora_lr=1e-5, route_lr=1e-3
#
# SFT checkpoint: V17.1 dual-aux warmup (0.5 epoch)
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29503
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200

mkdir -p "${PROJECT_DIR}/v17_self_evolve/scripts/logs"

echo "=========================================="
echo "V17.1 Dual-Aux RL — Phase-Conditional Hard Routing"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}"
echo "Start:   $(date)"
echo "=========================================="

OUTPUT_DIR="${PROJECT_DIR}/checkpoints/v17_dual_aux_rl"

# SFT checkpoint from V17.1 dual-aux warmup
SFT_CKPT="${PROJECT_DIR}/checkpoints/v17_dual_aux_sft_warmup"

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
        v17_self_evolve/train_aux_decision_rl.py \
        --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
        --train_data v12_gui_360/data/gui360_train_2000_balanced.jsonl \
        --val_data v12_gui_360/data/gui360_val_50.jsonl \
        --output_dir '"$OUTPUT_DIR"' \
        --sft_checkpoint '"$SFT_CKPT"' \
        --lora_r 128 \
        --lora_alpha 256 \
        --lora_dropout 0.05 \
        --target_modules q_proj k_proj v_proj o_proj \
        --num_comm_rounds 2 \
        --balance_weight 0.01 \
        --routing_noise_scale 0.5 \
        --image_max_pixels 602112 \
        --max_aux_tokens 150 \
        --min_aux_tokens 20 \
        --aux_len_bonus 0.01 \
        --aux_a_hard_route 0.9 \
        --aux_b_hard_route 0.1 \
        --diversity_weight 0.03 \
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
