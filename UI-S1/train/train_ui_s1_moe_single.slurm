#!/bin/bash
#SBATCH --job-name=train_ui_s1_moe_test
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_ui_s1_moe_test_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_ui_s1_moe_test_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# UI-S1 MoE (Mixture of Experts) Test Script
# Single-Node Version for Quick Validation
#
# This script is for testing the MoE training setup
# before running full multi-node training.
#
# Note: For single-node, we let Python manage Ray directly
# (don't use ray start separately to avoid conflicts)
#
# Usage:
#   sbatch train_ui_s1_moe_single.slurm
#
# ============================================

set -e

# Set paths
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CONFIG_PATH="$PROJECT_DIR/examples/qwen_gui_moe/config"
LOG_DIR="$PROJECT_DIR/train/logs"

# Create log directory
mkdir -p $LOG_DIR

# ============================================
# MoE Configuration (same as multi-node)
# ============================================
NUM_EXPERTS=4
EXPERT_LORA_R=16
EXPERT_LORA_ALPHA=32
TARGET_MODULES='[q_proj,v_proj]'
ROUTER_HIDDEN=256
ROUTER_TEMPERATURE=1.0
BALANCE_WEIGHT=0.1
BALANCE_TYPE="mse"
POOLING_STRATEGY="mean"

# Training configuration (reduced for testing)
ENGINE="vllm"
MODE="mean_std_norm"
GAMMA=0.5
EXPERIMENT_NAME="qwenvl_uis1_MoE_test_${NUM_EXPERTS}experts"

# Single node batch size
BATCH_SIZE=4

echo "=========================================="
echo "Starting UI-S1 MoE Test (Single-Node)"
echo "=========================================="
echo "Project Dir: $PROJECT_DIR"
echo "GPUs: 4"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "=========================================="
echo "MoE Configuration:"
echo "  Number of Experts: $NUM_EXPERTS"
echo "  Expert LoRA Rank: $EXPERT_LORA_R"
echo "  Expert LoRA Alpha: $EXPERT_LORA_ALPHA"
echo "  Balance Weight: $BALANCE_WEIGHT"
echo "=========================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Start time: $(date)"
echo "=========================================="

# Activate environment
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
source $PROJECT_DIR/train/env_config.sh

# Set triton cache
export TRITON_CACHE_DIR=/scratch/a5l/shuqing.a5l/tmp/triton_cache_${SLURM_JOB_ID}
mkdir -p $TRITON_CACHE_DIR
export TRITON_HOME=$TRITON_CACHE_DIR

# Stop any existing Ray processes (cleanup only)
ray stop 2>/dev/null || true

# IMPORTANT: Don't start Ray here - let Python manage it
# This avoids the conflict between ray.init() with num_cpus and existing cluster

export PYTHONPATH=$PROJECT_DIR
cd $PROJECT_DIR

echo '=========================================='
echo 'Starting MoE Test Training...'
echo '(Ray will be started by Python)'
echo '=========================================='

python3 -m verl.trainer.main_dapo \
    --config-path=$CONFIG_PATH \
    --config-name=traj_grpo_moe \
    algorithm.adv_estimator=uis1 \
    data.train_files=$PROJECT_DIR/datasets/ui_s1_train.jsonl \
    data.val_files=$PROJECT_DIR/evaluation/dataset/android_control_evaluation_std.jsonl \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=$((4 * BATCH_SIZE)) \
    data.max_prompt_length=4096 \
    data.max_response_length=256 \
    data.truncation=error \
    actor_rollout_ref.model.path=$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.model.moe.enabled=true \
    actor_rollout_ref.model.moe.num_experts=$NUM_EXPERTS \
    actor_rollout_ref.model.moe.top_k=1 \
    actor_rollout_ref.model.moe.expert_lora_r=$EXPERT_LORA_R \
    actor_rollout_ref.model.moe.expert_lora_alpha=$EXPERT_LORA_ALPHA \
    "actor_rollout_ref.model.moe.target_modules=$TARGET_MODULES" \
    actor_rollout_ref.model.moe.router_hidden=$ROUTER_HIDDEN \
    actor_rollout_ref.model.moe.router_temperature=$ROUTER_TEMPERATURE \
    actor_rollout_ref.model.moe.balance_weight=$BALANCE_WEIGHT \
    actor_rollout_ref.model.moe.balance_type=$BALANCE_TYPE \
    actor_rollout_ref.model.moe.pooling_strategy=$POOLING_STRATEGY \
    actor_rollout_ref.model.moe.log_routing_matrix_freq=10 \
    actor_rollout_ref.actor.optim.lr=1e-4 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \
    actor_rollout_ref.actor.use_fixed_num_mini_batches=false \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    'actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra]' \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.limit_images=2 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=$GAMMA \
    algorithm.uis1.step_advantage_w=1.0 \
    algorithm.uis1.mode=$MODE \
    algorithm.filter_groups.enable=False \
    trainer.critic_warmup=0 \
    'trainer.logger=[console]' \
    trainer.project_name=gui_traj_grpo_moe_test \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    ray_init.num_cpus=${SLURM_CPUS_PER_TASK}

TRAIN_EXIT_CODE=$?

# Stop Ray (cleanup)
ray stop 2>/dev/null || true

echo "=========================================="
echo "MoE Test Training completed at $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "SUCCESS: MoE training test passed!"
    echo "You can now run the full multi-node training:"
    echo "  sbatch train_ui_s1_moe.slurm"
else
    echo ""
    echo "ERROR: MoE training test failed!"
    echo "Check the log for details: $LOG_DIR/train_ui_s1_moe_test_${SLURM_JOB_ID}.log"
fi

exit $TRAIN_EXIT_CODE
