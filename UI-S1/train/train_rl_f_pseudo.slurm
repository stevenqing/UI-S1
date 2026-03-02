#!/bin/bash
#SBATCH --job-name=rl_fpseudo
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/rl_fpseudo_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/rl_fpseudo_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# f_pseudo RL Comparison Experiment
# 3 runs: lambda = 0 (baseline), 0.1, 0.3
#
# Usage:
#   sbatch --export=LAMBDA=0.0 train/train_rl_f_pseudo.slurm   # baseline
#   sbatch --export=LAMBDA=0.1 train/train_rl_f_pseudo.slurm   # conservative
#   sbatch --export=LAMBDA=0.3 train/train_rl_f_pseudo.slurm   # aggressive
# ============================================

set -e

# ---- Configurable ----
LAMBDA=${LAMBDA:-0.1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
LR=${LR:-1e-6}
N_ROLLOUTS=${N_ROLLOUTS:-4}
BATCH_SIZE=${BATCH_SIZE:-4}
GAMMA=${GAMMA:-0.5}
# ---- End Configurable ----

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CONFIG_PATH="$PROJECT_DIR/train_GUI_360/config"
LOG_DIR="$PROJECT_DIR/train/logs"

mkdir -p $LOG_DIR

# Format lambda for experiment name (replace . with p)
LAMBDA_TAG=$(echo $LAMBDA | sed 's/\./_/')
EXPERIMENT_NAME="f_pseudo_lambda${LAMBDA_TAG}_${SLURM_JOB_ID}"

echo "=========================================="
echo "f_pseudo RL Training"
echo "=========================================="
echo "Node: $(hostname)"
echo "GPUs: 4"
echo "Lambda: $LAMBDA"
echo "LR: $LR"
echo "Batch Size: $BATCH_SIZE"
echo "Rollouts: $N_ROLLOUTS"
echo "Epochs: $TOTAL_EPOCHS"
echo "Experiment: $EXPERIMENT_NAME"
echo "Start time: $(date)"
echo "=========================================="

# Activate environment
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

# Source shared environment config
source $PROJECT_DIR/train/env_config.sh

# Ray temp dir — /local/user/ is not writable on compute nodes
export RAY_TMPDIR=/tmp/ray_${USER}_${SLURM_JOB_ID}
export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}
mkdir -p $RAY_TMPDIR $TMPDIR

export PYTHONPATH=$PROJECT_DIR
cd $PROJECT_DIR

# Stop any existing Ray processes
ray stop 2>/dev/null || true

echo "Starting f_pseudo RL training (lambda=$LAMBDA)..."
python3 -m verl.trainer.main_dapo \
    --config-path=$CONFIG_PATH \
    --config-name=gui360_rl_f_pseudo \
    data.train_files=$PROJECT_DIR/datasets/ui_s1_dataset/ui_s1_train.jsonl \
    data.val_files=$PROJECT_DIR/evaluation/dataset/android_control_evaluation_std.jsonl \
    data.train_batch_size=$BATCH_SIZE \
    data.max_prompt_length=8192 \
    data.max_response_length=512 \
    data.truncation=error \
    actor_rollout_ref.model.path=$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \
    actor_rollout_ref.actor.use_fixed_num_mini_batches=true \
    actor_rollout_ref.actor.fixed_num_mini_batches=2 \
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
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.limit_images=2 \
    actor_rollout_ref.rollout.n=$N_ROLLOUTS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.adv_estimator=uis1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=$GAMMA \
    algorithm.uis1.step_advantage_w=1.0 \
    algorithm.uis1.episode_advantage_w=1.0 \
    algorithm.uis1.mode=mean_std_norm \
    algorithm.patch_threshold=2 \
    algorithm.filter_groups.enable=False \
    reward_model.reward_manager=f_pseudo_dapo \
    reward_model.reward_kwargs.f_pseudo_path=$PROJECT_DIR/outputs/f_pseudo/f_pseudo_map.json \
    reward_model.reward_kwargs.f_pseudo_lambda=$LAMBDA \
    +trainer.trainer_cls=traj_dapo \
    trainer.project_name=gui360_rl_f_pseudo \
    trainer.experiment_name=$EXPERIMENT_NAME \
    'trainer.logger=[console,wandb]' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=10 \
    trainer.val_before_train=False \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.critic_warmup=0 \
    +trainer.checkpoint_dir=$PROJECT_DIR/checkpoints/f_pseudo_lambda${LAMBDA_TAG}_${SLURM_JOB_ID} \
    2>&1 | tee $LOG_DIR/rl_fpseudo_lambda${LAMBDA_TAG}_${SLURM_JOB_ID}.log

TRAIN_EXIT_CODE=$?

# Stop Ray
ray stop 2>/dev/null || true

echo ""
echo "=========================================="
echo "f_pseudo RL Training completed at $(date)"
echo "Lambda: $LAMBDA"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

exit $TRAIN_EXIT_CODE
