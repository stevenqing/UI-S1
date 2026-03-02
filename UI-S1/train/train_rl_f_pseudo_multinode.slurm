#!/bin/bash
#SBATCH --job-name=rl_fpseudo
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/rl_fpseudo_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/rl_fpseudo_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# f_pseudo RL Comparison Experiment (Multi-Node)
# 8 nodes × 4 GPUs = 32 GPUs total
#
# Usage:
#   sbatch --export=LAMBDA=0.0 train/train_rl_f_pseudo_multinode.slurm   # baseline
#   sbatch --export=LAMBDA=0.1 train/train_rl_f_pseudo_multinode.slurm   # conservative
#   sbatch --export=LAMBDA=0.3 train/train_rl_f_pseudo_multinode.slurm   # aggressive
#
# Override nodes/batch:
#   sbatch --export=LAMBDA=0.1,BATCH_SIZE=64 --nodes=8 train/train_rl_f_pseudo_multinode.slurm
# ============================================

set -e

# Load required modules
module load cuda/12.6

# ---- Configurable ----
LAMBDA=${LAMBDA:-0.1}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
LR=${LR:-1e-6}
N_ROLLOUTS=${N_ROLLOUTS:-4}
BATCH_SIZE=${BATCH_SIZE:-32}
GAMMA=${GAMMA:-0.5}
# ---- End Configurable ----

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CONFIG_PATH="$PROJECT_DIR/train_GUI_360/config"
LOG_DIR="$PROJECT_DIR/train/logs"

mkdir -p $LOG_DIR

# Format lambda for experiment name (replace . with _)
LAMBDA_TAG=$(echo $LAMBDA | sed 's/\./_/')
EXPERIMENT_NAME="f_pseudo_lambda${LAMBDA_TAG}_${SLURM_JOB_ID}"

# Get node information
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" /usr/bin/hostname --ip-address | awk '{print $1}')

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
ip_head=$head_node_ip:$port
export ip_head
export head_node_ip
export port

echo "=========================================="
echo "f_pseudo RL Training (Multi-Node)"
echo "=========================================="
echo "Head Node: $head_node ($head_node_ip)"
echo "Nodes: ${SLURM_NNODES}, GPUs: $((SLURM_NNODES * 4))"
echo "Lambda: $LAMBDA"
echo "LR: $LR"
echo "Batch Size: $BATCH_SIZE"
echo "Rollouts: $N_ROLLOUTS"
echo "Epochs: $TOTAL_EPOCHS"
echo "Experiment: $EXPERIMENT_NAME"
echo "Start time: $(date)"
echo "=========================================="

# Create worker startup script
WORKER_SCRIPT="$LOG_DIR/worker_${SLURM_JOB_ID}.sh"
cat > $WORKER_SCRIPT << WORKER_EOF
#!/bin/bash
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
source $PROJECT_DIR/train/env_config.sh

# Ray temp dir
export RAY_TMPDIR=/tmp/ray_\${USER}_${SLURM_JOB_ID}
export TMPDIR=/tmp/\${USER}_${SLURM_JOB_ID}
mkdir -p \$RAY_TMPDIR \$TMPDIR

ray stop 2>/dev/null || true
echo "Starting Ray worker on \$(hostname), connecting to \$1"
ray start --address \$1 --num-cpus \$2 --num-gpus 4 --disable-usage-stats --block
WORKER_EOF
chmod +x $WORKER_SCRIPT

# Start Ray worker nodes in background
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting Ray WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        /bin/bash $WORKER_SCRIPT $ip_head ${SLURM_CPUS_PER_TASK} &
done

# Start Ray head and training on head node
echo "Starting Ray HEAD and training at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    /bin/bash -c "
        source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
        source $PROJECT_DIR/train/env_config.sh

        # Ray temp dir
        export RAY_TMPDIR=/tmp/ray_\${USER}_${SLURM_JOB_ID}
        export TMPDIR=/tmp/\${USER}_${SLURM_JOB_ID}
        mkdir -p \$RAY_TMPDIR \$TMPDIR

        # Stop any existing Ray processes
        ray stop 2>/dev/null || true

        # Start Ray head
        echo 'Starting Ray head node...'
        ray start --head --node-ip-address=$head_node_ip --port=$port \
            --num-cpus ${SLURM_CPUS_PER_TASK} --num-gpus 4 \
            --disable-usage-stats --include-dashboard=false

        # Wait for Ray to initialize
        echo 'Waiting for Ray to initialize...'
        sleep 10

        # Wait for workers to connect
        echo 'Waiting for Ray workers to connect (need ${SLURM_NNODES} nodes)...'
        for attempt in {1..60}; do
            node_count=\$(ray status 2>/dev/null | grep -c 'node_' || echo '0')
            if [ \"\$node_count\" -ge \"${SLURM_NNODES}\" ]; then
                echo \"Ray cluster has \$node_count nodes - ready!\"
                break
            fi
            echo \"Waiting for Ray nodes... attempt \$attempt/60 (found \$node_count/${SLURM_NNODES} nodes)\"
            sleep 5
        done

        # Check Ray cluster status
        echo 'Ray cluster status:'
        ray status

        export RAY_ADDRESS=$head_node_ip:$port
        echo \"RAY_ADDRESS set to: \$RAY_ADDRESS\"

        # Wait for workers to stabilize
        sleep 15

        export PYTHONPATH=$PROJECT_DIR
        cd $PROJECT_DIR

        echo 'Starting f_pseudo RL training (lambda=$LAMBDA)...'
        python3 -m verl.trainer.main_dapo \
            --config-path=$CONFIG_PATH \
            --config-name=gui360_rl_f_pseudo \
            data.train_files=$PROJECT_DIR/datasets/GUI-360/rl_data/gui360_train.jsonl \
            data.val_files=$PROJECT_DIR/datasets/GUI-360/rl_data/gui360_val_small.jsonl \
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
            trainer.nnodes=${SLURM_NNODES} \
            trainer.save_freq=100 \
            trainer.test_freq=50 \
            trainer.val_before_train=False \
            trainer.total_epochs=$TOTAL_EPOCHS \
            trainer.critic_warmup=0 \
            +trainer.checkpoint_dir=$PROJECT_DIR/checkpoints/gui360_rl_f_pseudo/f_pseudo_lambda${LAMBDA_TAG}_${SLURM_JOB_ID}

        TRAIN_EXIT_CODE=\$?

        # Stop Ray
        ray stop 2>/dev/null || true

        exit \$TRAIN_EXIT_CODE
    " 2>&1 | tee $LOG_DIR/rl_fpseudo_lambda${LAMBDA_TAG}_${SLURM_JOB_ID}.log

TRAIN_EXIT_CODE=$?

# Clean up
rm -f $WORKER_SCRIPT

echo ""
echo "=========================================="
echo "f_pseudo RL Training completed at $(date)"
echo "Lambda: $LAMBDA"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

exit $TRAIN_EXIT_CODE
