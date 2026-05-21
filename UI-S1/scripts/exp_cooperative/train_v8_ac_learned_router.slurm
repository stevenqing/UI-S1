#!/bin/bash
#SBATCH --job-name=coop_v8_ac_lr
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v8_ac_learned_router_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp_cooperative/logs/train_v8_ac_learned_router_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# Cooperative LoRA v8: LEARNED ROUTER on AndroidControl
#
# Core idea:
#   Replace fixed token-type routing with per-layer learned router.
#   - One nn.Linear(D, 1) per transformer layer (28 routers)
#   - Output: w = sigmoid(router(x)) ∈ [0, 1]
#   - Blend:  delta = w * delta_WHERE + (1-w) * delta_WHAT
#
# Motivation (from hidden-state verification, job 3833920):
#   - 100% linear probe accuracy: hidden states encode token modality for free
#   - Routing is dynamic across layers (think tokens → IMG in mid, ACT at out)
#   - Hard routing (v6.5) wastes this information
#
# Training strategy:
#   - Warm-start routers on 16 samples (zero-shot ~90/10 logit for WHERE/WHAT)
#   - Balance loss λ=0.01: push mean router_w per layer toward 0.5
#   - No coord_routing / coop_reasoning_alpha (those were hand-coded hacks)
#   - No binding loss (v6.5 confirmed it's not useful on AC)
#
# Hyperparameters: single-node variant (eff_bs matches v6.5 coord_routing)
#   nodes=1  ngpu/node=4  pdbs=1  grad_accum=8  ->  eff_bs = 32
#   num_epochs=3 (v6.5 plateaus by ep2-3; 1-node wall clock ~2x slower,
#                 3 epochs keeps us well under 24h limit)
#   r=256, alpha=512, tanh gates, gate_lr_mult=100
#   max_length=16384, image_max_pixels=602112
#   shared_B=True: single B matrix per module, blend in r-space. Saves ~14 GiB
#                  forward memory vs double-tower, lets 1-node fit comfortably.
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200
# Reduce fragmentation on GH200 95 GiB — error log from 3848479 explicitly suggested this.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "Cooperative LoRA v8 LEARNED ROUTER (AC) — 1-node, eff_bs=32, r=256, balance=0.01, 3 epochs"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}"
echo "Start:   $(date)"
echo "=========================================="

OUTPUT_DIR="train_GUI_360/llamafactory/output/cooperative_v8_ac_learned_router"

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
        --train_data datasets/cooperative_thought_ac/ac_train_multiturn.jsonl \
        --val_data datasets/cooperative_thought_ac/ac_val_multiturn.jsonl \
        --output_dir '"$OUTPUT_DIR"' \
        --lora_r 256 \
        --lora_alpha 512 \
        --lora_dropout 0.05 \
        --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
        --num_agents 2 \
        --cooperative_comm \
        --gate_type tanh \
        --gate_init 0.0 \
        --gate_lr_multiplier 100.0 \
        --gate_weight_decay 0.0 \
        --routing_mode learned \
        --shared_b \
        --balance_weight 0.01 \
        --router_warmstart_samples 16 \
        --bind_weight 0.0 \
        --per_device_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-5 \
        --num_epochs 3.0 \
        --warmup_ratio 0.03 \
        --save_steps 50 \
        --eval_steps 50 \
        --logging_steps 5 \
        --max_length 16384 \
        --image_max_pixels 602112
'

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
