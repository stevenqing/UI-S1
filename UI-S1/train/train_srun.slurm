#!/bin/bash
#SBATCH --job-name=train_srun
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_srun_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_srun_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ============================================
# UI-S1 Training with srun (bypassing Ray)
# Uses PyTorch native distributed training
# ============================================

set -e

# Load required modules
module load cuda/12.6

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CONFIG_PATH="$PROJECT_DIR/examples/qwen_gui_static_grpo/config"
LOG_DIR="$PROJECT_DIR/train/logs"

mkdir -p $LOG_DIR

# Training configuration
# NOTE: param_offload=False because FSDP CPU offloading is incompatible with
# gradient checkpointing (causes "storage of size 0" errors during backward pass)
# NOTE: max_prompt_length=4096 (reduced from 12288) to fit in GPU memory with FSDP
# We rely on gradient checkpointing + FSDP sharding for memory efficiency
PARAM_OFFLOAD=False
OPTIMIZER_OFFLOAD=False
ENGINE="vllm"
MODE="mean_std_norm"
GAMMA=0.5
DAPO=False
DAPO_THRESHOLD=0.3
PATCH_THRESHOLD=2
TOTAL_EPOCHS=3
EXPERIMENT_NAME="srun_full_train_${SLURM_JOB_ID}"

# Get node information
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

BATCH_SIZE=$((SLURM_NNODES * 8))

echo "=========================================="
echo "UI-S1 Training with srun"
echo "=========================================="
echo "Head Node: $head_node ($head_node_ip)"
echo "Nodes: ${SLURM_NNODES}, GPUs: $((SLURM_NNODES * 4))"
echo "Batch Size: $BATCH_SIZE"
echo "Start time: $(date)"
echo "=========================================="

# Launch training via srun
# Each node runs 4 processes (one per GPU)
srun --nodes=$SLURM_NNODES --ntasks-per-node=4 --cpus-per-task=72 \
    bash -c "
        # Load CUDA module for GPU access
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

        # Extra settings
        export PYTHONPATH=$PROJECT_DIR
        export PYTHONUNBUFFERED=1

        echo \"[Node \$(hostname)] RANK=\$RANK LOCAL_RANK=\$LOCAL_RANK WORLD_SIZE=\$WORLD_SIZE starting training...\"

        # Change to project directory for hydra config resolution
        cd $PROJECT_DIR

        # Use torchrun-style training entry point
        python $PROJECT_DIR/train/train_srun_worker.py \
            --config-path=$CONFIG_PATH \
            --config-name=traj_grpo \
            algorithm.adv_estimator=uis1 \
            data.train_files=$PROJECT_DIR/datasets/ui_s1_train.jsonl \
            data.val_files=$PROJECT_DIR/evaluation/dataset/android_control_evaluation_std.jsonl \
            data.train_batch_size=$BATCH_SIZE \
            data.max_prompt_length=4096 \
            data.max_response_length=512 \
            actor_rollout_ref.model.path=$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD \
            actor_rollout_ref.rollout.name=$ENGINE \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
            actor_rollout_ref.rollout.n=4 \
            algorithm.gamma=$GAMMA \
            algorithm.filter_groups.enable=$DAPO \
            trainer.project_name=gui_traj_grpo_srun \
            trainer.experiment_name=$EXPERIMENT_NAME \
            'trainer.logger=[console,wandb]' \
            trainer.n_gpus_per_node=4 \
            trainer.nnodes=${SLURM_NNODES} \
            trainer.save_freq=5 \
            trainer.test_freq=10 \
            trainer.val_before_train=False \
            trainer.total_epochs=$TOTAL_EPOCHS
    "

TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training completed at $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

exit $TRAIN_EXIT_CODE
