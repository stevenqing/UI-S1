#!/bin/bash
#SBATCH --job-name=train_ui_s1_test
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_ui_s1_test_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_ui_s1_test_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ============================================
# UI-S1 Training Script - TEST VERSION (Single Node)
# Reduced parameters for debugging
# ============================================

set -e

# Set paths
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CONFIG_PATH="$PROJECT_DIR/examples/qwen_gui_static_grpo/config"
LOG_DIR="$PROJECT_DIR/train/logs"

# Create log directory
mkdir -p $LOG_DIR

# Training configuration - REDUCED FOR TESTING
ENGINE="vllm"
MODE="mean_std_norm"
GAMMA=0.5
DAPO=False
DAPO_THRESHOLD=0.3
PATCH_THRESHOLD=2
EXPERIMENT_NAME="qwenvl_uis1_TEST_single_node"

# Single node setup
head_node_ip=$(hostname --ip-address)

# Handle IPv6 addresses
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
    echo "IPV6 address detected. Using IPV4 address: $head_node_ip"
fi

port=6379

# Reduced batch size for testing
BATCH_SIZE=4
N_ROLLOUTS=4

echo "=========================================="
echo "Starting UI-S1 Training (TEST - Single Node)"
echo "Project Dir: $PROJECT_DIR"
echo "Head Node IP: $head_node_ip"
echo "GPUs: 4"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "N_ROLLOUTS: $N_ROLLOUTS"
echo "Experiment: $EXPERIMENT_NAME"
echo "Start time: $(date)"
echo "=========================================="

source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
source $PROJECT_DIR/train/env_config.sh

# Stop any existing Ray processes
ray stop 2>/dev/null || true

# Start Ray head
echo 'Starting Ray head node...'
ray start --head --node-ip-address=$head_node_ip --port=$port \
    --num-cpus ${SLURM_CPUS_PER_TASK} --num-gpus 4 \
    --disable-usage-stats --include-dashboard=false

# Wait for Ray to fully initialize
echo 'Waiting for Ray to fully initialize...'
sleep 10

# Check Ray cluster status
echo 'Ray cluster status:'
ray status

# Set RAY_ADDRESS
export RAY_ADDRESS="$head_node_ip:$port"
echo "RAY_ADDRESS set to: $RAY_ADDRESS"

export PYTHONPATH=$PROJECT_DIR
cd $PROJECT_DIR

echo 'Starting training...'
python3 -m verl.trainer.main_dapo \
    --config-path=$CONFIG_PATH \
    --config-name=traj_grpo \
    algorithm.adv_estimator=uis1 \
    data.train_files=$PROJECT_DIR/datasets/ui_s1_train.jsonl \
    data.val_files=$PROJECT_DIR/evaluation/dataset/android_control_evaluation_std.jsonl \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=512 \
    data.truncation=left \
    actor_rollout_ref.model.path=$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
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
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=32678 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.limit_images=2 \
    actor_rollout_ref.rollout.n=$N_ROLLOUTS \
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
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    trainer.total_training_steps=10

TRAIN_EXIT_CODE=$?

# Stop Ray
ray stop

echo "=========================================="
echo "Training completed at $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

exit $TRAIN_EXIT_CODE
