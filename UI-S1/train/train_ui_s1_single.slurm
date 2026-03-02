#!/bin/bash
#SBATCH --job-name=train_ui_s1
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_ui_s1_single_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_ui_s1_single_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ============================================
# UI-S1 Training Script with Ray + GRPO (Single-Node)
# ============================================

set -e

# Set paths
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CONFIG_PATH="$PROJECT_DIR/examples/qwen_gui_static_grpo/config"
LOG_DIR="$PROJECT_DIR/train/logs"

# Create log directory
mkdir -p $LOG_DIR

# Training configuration
ENGINE="vllm"
MODE="mean_std_norm"
GAMMA=0.5
DAPO=False
DAPO_THRESHOLD=0.3
# Note: DAPO filtering disabled to prevent training failures when reward variance is low
PATCH_THRESHOLD=2
EXPERIMENT_NAME="qwenvl_uis1_DAPO_${DAPO}_${DAPO_THRESHOLD}_patch_${PATCH_THRESHOLD}_gamma_${GAMMA}_single"

BATCH_SIZE=8  # Single node with 4 GPUs

echo "=========================================="
echo "Starting UI-S1 Training (Single-Node)"
echo "Project Dir: $PROJECT_DIR"
echo "Node: $(hostname)"
echo "GPUs: 4"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "Experiment: $EXPERIMENT_NAME"
echo "Start time: $(date)"
echo "=========================================="

# Activate environment
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

# Source shared environment config
source $PROJECT_DIR/train/env_config.sh

export PYTHONPATH=$PROJECT_DIR
cd $PROJECT_DIR

# Stop any existing Ray processes
ray stop 2>/dev/null || true

echo 'Starting training (Ray will initialize automatically)...'
python3 -m verl.trainer.main_dapo \
    --config-path=$CONFIG_PATH \
    --config-name=traj_grpo \
    algorithm.adv_estimator=uis1 \
    data.train_files=$PROJECT_DIR/datasets/ui_s1_train.jsonl \
    data.val_files=$PROJECT_DIR/evaluation/dataset/android_control_evaluation_std.jsonl \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=$((8 * BATCH_SIZE)) \
    data.max_prompt_length=12288 \
    data.max_response_length=512 \
    data.truncation=error \
    actor_rollout_ref.model.path=$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \
    actor_rollout_ref.actor.use_fixed_num_mini_batches=true \
    actor_rollout_ref.actor.fixed_num_mini_batches=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    'actor_rollout_ref.actor.checkpoint.contents=[model,hf_model,optimizer,extra]' \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.max_model_len=32678 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.limit_images=2 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=$GAMMA \
    algorithm.uis1.step_advantage_w=1.0 \
    algorithm.uis1.mode=$MODE \
    algorithm.patch_threshold=$PATCH_THRESHOLD \
    algorithm.filter_groups.enable=$DAPO \
    algorithm.filter_groups.metric=seq_future_reward \
    algorithm.filter_groups.std_threshold=$DAPO_THRESHOLD \
    trainer.critic_warmup=0 \
    'trainer.logger=[console,wandb]' \
    trainer.project_name=gui_traj_grpo \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.total_epochs=3 2>&1 | tee $LOG_DIR/train_${SLURM_JOB_ID}.log

TRAIN_EXIT_CODE=$?

# Stop Ray
ray stop 2>/dev/null || true

echo "=========================================="
echo "Training completed at $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

exit $TRAIN_EXIT_CODE
