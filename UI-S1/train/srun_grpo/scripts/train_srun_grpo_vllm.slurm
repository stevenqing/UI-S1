#!/bin/bash
#SBATCH --job-name=train_grpo_vllm
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/srun_grpo/logs/train_grpo_vllm_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/srun_grpo/logs/train_grpo_vllm_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ============================================
# UI-S1 GRPO Training with vLLM Rollout
#
# Features:
# - vLLM high-throughput generation
# - FSDP-vLLM weight synchronization
# - UI-S1 step + episode advantage
# - DAPO filtering
# - Reference policy KL penalty
# ============================================

set -e

module load cuda/12.6

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CONFIG_PATH="$PROJECT_DIR/examples/qwen_gui_static_grpo/config"
LOG_DIR="$PROJECT_DIR/train/srun_grpo/logs"
SCRIPT_DIR="$PROJECT_DIR/train/srun_grpo/scripts"

mkdir -p $LOG_DIR

# ============================================
# Configuration
# ============================================

# UI-S1 Algorithm
GAMMA=0.5
STEP_ADV_W=1.0
EPISODE_ADV_W=1.0
ADV_MODE="mean_std_norm"

# GRPO
N_ROLLOUTS=4
CLIP_RANGE=0.2
KL_COEF=0.0001
KL_LOSS_TYPE="low_var_kl"

# DAPO
DAPO_ENABLED=True
DAPO_THRESHOLD=0.3

# vLLM
VLLM_TP_SIZE=1
VLLM_GPU_UTIL=0.7
VLLM_MAX_MODEL_LEN=8192
VLLM_ENFORCE_EAGER=True

# Training
TOTAL_EPOCHS=3
LR=1e-6
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=512
SAVE_FREQ=5

EXPERIMENT_NAME="grpo_vllm_uis1_gamma${GAMMA}_${SLURM_JOB_ID}"

# Node setup
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

BATCH_SIZE=$((SLURM_NNODES * 8))

echo "=========================================="
echo "UI-S1 GRPO Training with vLLM Rollout"
echo "=========================================="
echo "Head Node: $head_node ($head_node_ip)"
echo "Nodes: ${SLURM_NNODES}, GPUs: $((SLURM_NNODES * 4))"
echo "vLLM TP Size: $VLLM_TP_SIZE"
echo "vLLM GPU Utilization: $VLLM_GPU_UTIL"
echo "Start time: $(date)"
echo "=========================================="

# Launch training
srun --nodes=$SLURM_NNODES --ntasks-per-node=4 --cpus-per-task=72 \
    bash -c "
        module load cuda/12.6

        source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
        source $PROJECT_DIR/train/env_config.sh

        export MASTER_ADDR=$head_node_ip
        export MASTER_PORT=29500
        export WORLD_SIZE=\$SLURM_NTASKS
        export RANK=\$SLURM_PROCID
        export LOCAL_RANK=\$SLURM_LOCALID
        export LOCAL_WORLD_SIZE=4

        export PYTHONPATH=$PROJECT_DIR
        export PYTHONUNBUFFERED=1
        SCRIPT_DIR=$PROJECT_DIR/train/srun_grpo/scripts

        echo \"[Node \$(hostname)] RANK=\$RANK LOCAL_RANK=\$LOCAL_RANK starting GRPO with vLLM...\"

        cd $PROJECT_DIR

        python \$SCRIPT_DIR/train_srun_grpo_vllm.py \\
            --config-path=$CONFIG_PATH \\
            --config-name=traj_grpo \\
            algorithm.adv_estimator=uis1 \\
            algorithm.gamma=$GAMMA \\
            algorithm.uis1.step_advantage_w=$STEP_ADV_W \\
            algorithm.uis1.episode_advantage_w=$EPISODE_ADV_W \\
            algorithm.uis1.mode=$ADV_MODE \\
            algorithm.filter_groups.enable=$DAPO_ENABLED \\
            algorithm.filter_groups.std_threshold=$DAPO_THRESHOLD \\
            data.train_files=$PROJECT_DIR/datasets/ui_s1_train.jsonl \\
            data.train_batch_size=$BATCH_SIZE \\
            data.max_prompt_length=$MAX_PROMPT_LENGTH \\
            data.max_response_length=$MAX_RESPONSE_LENGTH \\
            actor_rollout_ref.model.path=$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct \\
            actor_rollout_ref.actor.optim.lr=$LR \\
            actor_rollout_ref.actor.clip_ratio=$CLIP_RANGE \\
            actor_rollout_ref.actor.use_kl_loss=True \\
            actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \\
            actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \\
            actor_rollout_ref.rollout.n=$N_ROLLOUTS \\
            actor_rollout_ref.rollout.tensor_model_parallel_size=$VLLM_TP_SIZE \\
            actor_rollout_ref.rollout.gpu_memory_utilization=$VLLM_GPU_UTIL \\
            actor_rollout_ref.rollout.max_model_len=$VLLM_MAX_MODEL_LEN \\
            actor_rollout_ref.rollout.enforce_eager=$VLLM_ENFORCE_EAGER \\
            trainer.project_name=gui_traj_grpo_vllm \\
            trainer.experiment_name=$EXPERIMENT_NAME \\
            'trainer.logger=[console,wandb]' \\
            trainer.n_gpus_per_node=4 \\
            trainer.nnodes=${SLURM_NNODES} \\
            trainer.save_freq=$SAVE_FREQ \\
            trainer.total_epochs=$TOTAL_EPOCHS \\
            +trainer.checkpoint_dir=$PROJECT_DIR/checkpoints/grpo_vllm_${SLURM_JOB_ID}
    "

echo ""
echo "=========================================="
echo "GRPO vLLM Training completed at $(date)"
echo "=========================================="
