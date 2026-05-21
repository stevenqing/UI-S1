#!/bin/bash
#SBATCH --job-name=v16_dual_rl
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v16_task_vector_merge/scripts/logs/train_dual_prompt_rl_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v16_task_vector_merge/scripts/logs/train_dual_prompt_rl_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V16 Dual-Prompt Cooperative RL — Hidden-State Fusion
#
# Merged checkpoint: Expert 1 = grounder, Expert 2 = actor
# Experts communicate in LATENT SPACE (not text):
#   Expert 1: A_1 @ grounder_hidden_states  (grounder features)
#   Expert 2: A_2 @ actor_input             (actor features)
#   Communication: comm_W_12/W_21 + gates in r-space
#
# Per step (has_coordinate=True):
#   1. LoRA-enabled forward (expert_1_only) on grounder prompt
#      → cache per-layer hidden states (detached, no grad)
#      → features include Expert 1 grounding knowledge
#      → co-evolve passively as LoRA weights are updated by RL
#   2. Actor forward (normal routing): Expert 1 uses cached
#      grounder features, Expert 2 uses actor input
# Per step (has_coordinate=False):
#   Both experts use actor input (standard cooperative LoRA)
#
# 4 nodes x 4 GPUs = 16 GPUs, grad_accum=4, eff_bs=64.
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

# Multi-node setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200

echo "=========================================="
echo "V16 Dual-Prompt RL — Hidden-State Fusion"
echo "  4 nodes x 4 GPUs, K=8, grad_accum=4"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}"
echo "Start:   $(date)"
echo "=========================================="

# Create log directory
mkdir -p "${PROJECT_DIR}/v16_task_vector_merge/scripts/logs"

# ── Step 1: Prepare dual-prompt data (if not already done) ──
TRAIN_DATA="${PROJECT_DIR}/v16_task_vector_merge/data/gui360_dual_prompt_train.jsonl"
VAL_DATA="${PROJECT_DIR}/v16_task_vector_merge/data/gui360_dual_prompt_val.jsonl"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Preparing dual-prompt data..."
    python "${PROJECT_DIR}/v16_task_vector_merge/prepare_dual_prompt_data.py" \
        --train_input "${PROJECT_DIR}/v12_gui_360/data/gui360_train_2000_balanced.jsonl" \
        --val_input "${PROJECT_DIR}/v12_gui_360/data/gui360_val_50.jsonl" \
        --output_dir "${PROJECT_DIR}/v16_task_vector_merge/data"
    echo "Data prepared."
fi

# ── Step 2: Merge cooperative checkpoints (if not already done) ──
MERGED_CKPT="${PROJECT_DIR}/checkpoints/v16_merged_grounder_actor_cooperative"

if [ ! -f "${MERGED_CKPT}/lora_weights.pt" ]; then
    echo "Merging cooperative expert checkpoints..."
    python "${PROJECT_DIR}/v16_task_vector_merge/merge_cooperative_experts.py" \
        --grounder_checkpoint "${PROJECT_DIR}/checkpoints/v16_peft_grounding_cooperative_r128" \
        --actor_checkpoint "${PROJECT_DIR}/checkpoints/v15_peft_balanced_cooperative_r128" \
        --output_dir "${MERGED_CKPT}" \
        --num_comm_rounds 2
    echo "Merge complete."
fi

# ── Step 3: Train ──
OUTPUT_DIR="v16_task_vector_merge/output/dual_prompt_rl"

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
        v16_task_vector_merge/train_dual_prompt_rl.py \
        --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
        --sft_checkpoint '"$MERGED_CKPT"' \
        --train_data '"$TRAIN_DATA"' \
        --val_data '"$VAL_DATA"' \
        --output_dir '"$OUTPUT_DIR"' \
        --lora_r 128 \
        --lora_alpha 256 \
        --lora_dropout 0.05 \
        --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
        --freeze_lora \
        --balance_weight 0.01 \
        --num_comm_rounds 2 \
        --routing_noise_scale 0.5 \
        --num_samples 8 \
        --temperature 1.0 \
        --top_p 0.95 \
        --clip_range 0.2 \
        --kl_coef 0.0 \
        --max_new_tokens 512 \
        --advantage_mode sp \
        --match_threshold 0.5 \
        --spwa_decay 0.5 \
        --dapo_threshold 0.1 \
        --step_adv_weight 0.5 \
        --route_lr 1e-3 \
        --comm_lr 1e-3 \
        --max_grad_norm 1.0 \
        --num_epochs 3 \
        --gradient_accumulation_steps 4 \
        --logging_steps 1 \
        --save_steps 50 \
        --val_steps 25
'

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="
