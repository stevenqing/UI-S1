#!/bin/bash
#SBATCH --job-name=v16_full_rl
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v16_task_vector_merge/scripts/logs/train_dual_prompt_rl_full_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v16_task_vector_merge/scripts/logs/train_dual_prompt_rl_full_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# V16 Dual-Prompt RL — Full LoRA Training
#
# SAME as train_dual_prompt_rl.slurm EXCEPT:
#   --no_freeze_lora  (trains A_1, A_2, B + comm + route = ~142M params)
#   --lora_lr 1e-5    (same as V15 RL SVD)
#   --kl_coef 0.001   (meaningful now since LoRA changes distribution)
#
# Comparison: frozen-LoRA (1.9M) vs full-LoRA (142M)
# Hypothesis: routing will actually learn when experts are genuinely
# different (grounder A_1 vs actor A_2, cosine sim ≈ -0.01)
#
# V15 showed routing=0.5 didn't budge with nearly-identical experts.
# With orthogonal experts + full LoRA training, routing should differentiate.
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"

# Multi-node setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29501
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=10800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export GLOO_TIMEOUT_SECONDS=7200

echo "=========================================="
echo "V16 Dual-Prompt RL — Full LoRA (unfreeze)"
echo "  4 nodes x 4 GPUs, K=8, grad_accum=4"
echo "=========================================="
echo "Job ID:  ${SLURM_JOB_ID}"
echo "Nodes:   ${SLURM_NNODES}"
echo "Master:  ${MASTER_ADDR}"
echo "Start:   $(date)"
echo "=========================================="

mkdir -p "${PROJECT_DIR}/v16_task_vector_merge/scripts/logs"

TRAIN_DATA="${PROJECT_DIR}/v16_task_vector_merge/data/gui360_dual_prompt_train.jsonl"
VAL_DATA="${PROJECT_DIR}/v16_task_vector_merge/data/gui360_dual_prompt_val.jsonl"
MERGED_CKPT="${PROJECT_DIR}/checkpoints/v16_merged_grounder_actor_cooperative"
OUTPUT_DIR="v16_task_vector_merge/output/dual_prompt_rl_full"

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
        --no_freeze_lora \
        --balance_weight 0.01 \
        --num_comm_rounds 2 \
        --routing_noise_scale 0.5 \
        --num_samples 8 \
        --temperature 1.0 \
        --top_p 0.95 \
        --clip_range 0.2 \
        --kl_coef 0.001 \
        --max_new_tokens 512 \
        --advantage_mode sp \
        --match_threshold 0.5 \
        --spwa_decay 0.5 \
        --dapo_threshold 0.1 \
        --step_adv_weight 0.5 \
        --lora_lr 1e-5 \
        --route_lr 1e-3 \
        --comm_lr 1e-3 \
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
