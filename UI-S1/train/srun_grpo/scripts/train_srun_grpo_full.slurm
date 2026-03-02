#!/bin/bash
#SBATCH --job-name=train_grpo_full
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/srun_grpo/logs/train_grpo_full_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/srun_grpo/logs/train_grpo_full_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ============================================
# UI-S1 GRPO Full Training with srun
# Complete implementation matching Ray version:
# - Multi-round generation
# - UI-S1 advantage estimation
# - DAPO filtering
# - Reference policy for KL penalty
# - Validation loop
# ============================================

set -e

# Load required modules
module load cuda/12.6

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CONFIG_PATH="$PROJECT_DIR/examples/qwen_gui_static_grpo/config"
LOG_DIR="$PROJECT_DIR/train/srun_grpo/logs"
SCRIPT_DIR="$PROJECT_DIR/train/srun_grpo/scripts"

mkdir -p $LOG_DIR

# ============================================
# Training Configuration
# ============================================

# UI-S1 Algorithm parameters (matching paper settings)
GAMMA=0.5                    # Step-level discount factor
STEP_ADV_W=1.0              # Step-level advantage weight
EPISODE_ADV_W=1.0           # Episode-level advantage weight
ADV_MODE="mean_std_norm"    # Advantage normalization mode

# GRPO parameters
N_ROLLOUTS=4                # Number of rollouts per prompt
CLIP_RANGE=0.2              # PPO clip range
KL_COEF=0.0001              # KL penalty coefficient
KL_LOSS_TYPE="low_var_kl"   # KL loss type

# DAPO filtering
DAPO_ENABLED=True           # Enable DAPO filtering
DAPO_THRESHOLD=0.3          # Std threshold for filtering
DAPO_METRIC="seq_future_reward"

# Training parameters
TOTAL_EPOCHS=3
LR=1e-6
PARAM_OFFLOAD=False         # CPU offload for actor
MAX_PROMPT_LENGTH=4096      # Reduced for memory efficiency
MAX_RESPONSE_LENGTH=512

# Checkpointing
SAVE_FREQ=5
TEST_FREQ=10
VAL_BEFORE_TRAIN=False

EXPERIMENT_NAME="grpo_full_uis1_gamma${GAMMA}_dapo${DAPO_ENABLED}_${SLURM_JOB_ID}"

# Get node information
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

BATCH_SIZE=$((SLURM_NNODES * 8))

echo "=========================================="
echo "UI-S1 GRPO Full Training with srun"
echo "=========================================="
echo "Head Node: $head_node ($head_node_ip)"
echo "Nodes: ${SLURM_NNODES}, GPUs: $((SLURM_NNODES * 4))"
echo "Batch Size: $BATCH_SIZE"
echo ""
echo "UI-S1 Parameters:"
echo "  gamma=$GAMMA"
echo "  step_advantage_w=$STEP_ADV_W"
echo "  episode_advantage_w=$EPISODE_ADV_W"
echo "  mode=$ADV_MODE"
echo ""
echo "GRPO Parameters:"
echo "  n_rollouts=$N_ROLLOUTS"
echo "  clip_range=$CLIP_RANGE"
echo "  kl_coef=$KL_COEF"
echo "  kl_loss_type=$KL_LOSS_TYPE"
echo ""
echo "DAPO Filtering:"
echo "  enabled=$DAPO_ENABLED"
echo "  threshold=$DAPO_THRESHOLD"
echo "  metric=$DAPO_METRIC"
echo ""
echo "Start time: $(date)"
echo "=========================================="

# Launch training via srun
# Each node runs 4 processes (one per GPU)
srun --nodes=$SLURM_NNODES --ntasks-per-node=4 --cpus-per-task=72 \
    bash -c "
        # Load CUDA module
        module load cuda/12.6

        source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
        source $PROJECT_DIR/train/env_config.sh

        # Distributed training environment
        export MASTER_ADDR=$head_node_ip
        export MASTER_PORT=29500
        export WORLD_SIZE=\$SLURM_NTASKS
        export RANK=\$SLURM_PROCID
        export LOCAL_RANK=\$SLURM_LOCALID
        export LOCAL_WORLD_SIZE=4

        # Python path
        export PYTHONPATH=$PROJECT_DIR
        export PYTHONUNBUFFERED=1
        SCRIPT_DIR=$PROJECT_DIR/train/srun_grpo/scripts

        echo \"[Node \$(hostname)] RANK=\$RANK LOCAL_RANK=\$LOCAL_RANK starting GRPO full training...\"

        cd $PROJECT_DIR

        # Run GRPO full training
        python \$SCRIPT_DIR/train_srun_grpo_full.py \\
            --config-path=$CONFIG_PATH \\
            --config-name=traj_grpo \\
            algorithm.adv_estimator=uis1 \\
            algorithm.gamma=$GAMMA \\
            algorithm.uis1.step_advantage_w=$STEP_ADV_W \\
            algorithm.uis1.episode_advantage_w=$EPISODE_ADV_W \\
            algorithm.uis1.mode=$ADV_MODE \\
            algorithm.filter_groups.enable=$DAPO_ENABLED \\
            algorithm.filter_groups.metric=$DAPO_METRIC \\
            algorithm.filter_groups.std_threshold=$DAPO_THRESHOLD \\
            data.train_files=$PROJECT_DIR/datasets/ui_s1_train.jsonl \\
            data.val_files=$PROJECT_DIR/evaluation/dataset/android_control_evaluation_std.jsonl \\
            data.train_batch_size=$BATCH_SIZE \\
            data.max_prompt_length=$MAX_PROMPT_LENGTH \\
            data.max_response_length=$MAX_RESPONSE_LENGTH \\
            actor_rollout_ref.model.path=$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct \\
            actor_rollout_ref.actor.optim.lr=$LR \\
            actor_rollout_ref.actor.clip_ratio=$CLIP_RANGE \\
            actor_rollout_ref.actor.use_kl_loss=True \\
            actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \\
            actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \\
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \\
            actor_rollout_ref.model.enable_gradient_checkpointing=True \\
            actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD \\
            actor_rollout_ref.rollout.n=$N_ROLLOUTS \\
            trainer.project_name=gui_traj_grpo_srun_full \\
            trainer.experiment_name=$EXPERIMENT_NAME \\
            'trainer.logger=[console,wandb]' \\
            trainer.n_gpus_per_node=4 \\
            trainer.nnodes=${SLURM_NNODES} \\
            trainer.save_freq=$SAVE_FREQ \\
            trainer.test_freq=$TEST_FREQ \\
            trainer.val_before_train=$VAL_BEFORE_TRAIN \\
            trainer.total_epochs=$TOTAL_EPOCHS \\
            +trainer.checkpoint_dir=$PROJECT_DIR/checkpoints/grpo_full_${SLURM_JOB_ID}
    "

TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "GRPO Full Training completed at $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

exit $TRAIN_EXIT_CODE
