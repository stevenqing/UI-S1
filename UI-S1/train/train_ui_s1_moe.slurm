#!/bin/bash
#SBATCH --job-name=train_ui_s1_moe
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_ui_s1_moe_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_ui_s1_moe_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# UI-S1 MoE (Mixture of Experts) Training Script
# with Ray + GRPO (Multi-Node)
#
# This script trains a GUI Agent with:
# - 4 Expert LoRAs (click, type, navigate, scroll)
# - Text-only Router for instruction-based expert selection
# - Load balance loss to prevent expert collapse
#
# Usage:
#   sbatch train_ui_s1_moe.slurm
#
# ============================================

set -e

# Set paths
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CONFIG_PATH="$PROJECT_DIR/examples/qwen_gui_moe/config"
LOG_DIR="$PROJECT_DIR/train/logs"

# Create log directory
mkdir -p $LOG_DIR

# Memory monitoring interval (seconds)
MEMORY_MONITOR_INTERVAL=30

# Create memory monitoring script
MEMORY_MONITOR_SCRIPT="$LOG_DIR/memory_monitor_${SLURM_JOB_ID}.sh"
cat > $MEMORY_MONITOR_SCRIPT << 'MONITOR_EOF'
#!/bin/bash
# Memory monitoring script - logs CPU and GPU memory usage
LOG_FILE=$1
INTERVAL=$2
HOSTNAME=$(hostname)

echo "=== Memory Monitor Started on $HOSTNAME at $(date) ===" >> $LOG_FILE
echo "Logging every ${INTERVAL}s to $LOG_FILE" >> $LOG_FILE

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # CPU Memory (in GB)
    MEM_TOTAL=$(free -g | awk '/^Mem:/{print $2}')
    MEM_USED=$(free -g | awk '/^Mem:/{print $3}')
    MEM_FREE=$(free -g | awk '/^Mem:/{print $4}')
    MEM_AVAILABLE=$(free -g | awk '/^Mem:/{print $7}')
    MEM_PERCENT=$(free | awk '/^Mem:/{printf "%.1f", $3/$2 * 100}')

    # GPU Memory
    GPU_MEM=$(nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "N/A")

    echo "[$TIMESTAMP] [$HOSTNAME] CPU_MEM: ${MEM_USED}G/${MEM_TOTAL}G (${MEM_PERCENT}%) | Available: ${MEM_AVAILABLE}G" >> $LOG_FILE

    if [ "$GPU_MEM" != "N/A" ]; then
        echo "$GPU_MEM" | while IFS=',' read -r idx used total free util; do
            echo "[$TIMESTAMP] [$HOSTNAME] GPU$idx: ${used}MB/${total}MB (Free: ${free}MB) | Util: ${util}%" >> $LOG_FILE
        done
    fi

    echo "---" >> $LOG_FILE
    sleep $INTERVAL
done
MONITOR_EOF
chmod +x $MEMORY_MONITOR_SCRIPT

# ============================================
# MoE Configuration
# ============================================
# Number of expert LoRAs
NUM_EXPERTS=4

# Expert LoRA settings (total params ≈ single LoRA with rank 64)
EXPERT_LORA_R=16
EXPERT_LORA_ALPHA=32

# Target modules for Qwen2.5-VL
TARGET_MODULES='[q_proj,v_proj]'

# Router settings
ROUTER_HIDDEN=256
# CHANGED: 1.0 → 0.5 for more confident routing decisions
ROUTER_TEMPERATURE=0.5

# Load balance loss weight (higher = more uniform, may hurt specialization)
# CHANGED: 0.1 → 0.2 to better prevent expert collapse
BALANCE_WEIGHT=0.2
BALANCE_TYPE="mse"  # mse, switch, entropy

# Z-loss for training stability (critical for MoE!)
# CHANGED: 0.0 → 0.01 to stabilize router logits
Z_LOSS_WEIGHT=0.01

# Feature extraction strategy
POOLING_STRATEGY="mean"  # mean, last, first, max

# Training configuration
ENGINE="vllm"
MODE="mean_std_norm"
GAMMA=0.5
DAPO=False
DAPO_THRESHOLD=0.3
PATCH_THRESHOLD=2
# Top-k for soft routing (CHANGED: 2 → 4, use all experts)
TOP_K=4

# Use vectorized routing for soft expert combination
USE_VECTORIZED_ROUTING=true

# Actor training parameters (conservative settings)
# CHANGED: 5e-5 → 1e-5 for more stable updates
LR=1e-5
# CHANGED: 0.01 → 0.1 to strongly penalize KL divergence
KL_LOSS_COEF=0.1
# CHANGED: 0.2 → 0.1 for more conservative policy updates
CLIP_RATIO=0.1
# CHANGED: 1.0 → 0.5 for stricter gradient clipping
GRAD_CLIP=0.5

EXPERIMENT_NAME="qwenvl_uis1_MoE_${NUM_EXPERTS}experts_r${EXPERT_LORA_R}_allexperts_topk${TOP_K}_gamma${GAMMA}"

# Get node information
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

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

# MoE uses similar memory as LoRA, reduced batch size for top_k=4
BATCH_SIZE=$((SLURM_NNODES * 1))

echo "=========================================="
echo "Starting UI-S1 MoE Training (Multi-Node)"
echo "=========================================="
echo "Project Dir: $PROJECT_DIR"
echo "Head Node: $head_node ($head_node_ip)"
echo "IP Head: $ip_head"
echo "Nodes: ${SLURM_NNODES}"
echo "GPUs per node: 4"
echo "Total GPUs: $((SLURM_NNODES * 4))"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "=========================================="
echo "MoE Configuration (Conservative):"
echo "  Number of Experts: $NUM_EXPERTS"
echo "  Top-K: $TOP_K (soft routing)"
echo "  Use Vectorized Routing: $USE_VECTORIZED_ROUTING"
echo "  Expert LoRA Rank: $EXPERT_LORA_R"
echo "  Expert LoRA Alpha: $EXPERT_LORA_ALPHA"
echo "  Target Modules: $TARGET_MODULES"
echo "  Router Hidden: $ROUTER_HIDDEN"
echo "  Router Temperature: $ROUTER_TEMPERATURE"
echo "  Balance Weight: $BALANCE_WEIGHT"
echo "  Balance Type: $BALANCE_TYPE"
echo "  Z-Loss Weight: $Z_LOSS_WEIGHT"
echo "  Pooling Strategy: $POOLING_STRATEGY"
echo "=========================================="
echo "Actor Training (Conservative):"
echo "  Learning Rate: $LR"
echo "  KL Loss Coef: $KL_LOSS_COEF"
echo "  Clip Ratio: $CLIP_RATIO"
echo "  Grad Clip: $GRAD_CLIP"
echo "=========================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Memory monitor interval: ${MEMORY_MONITOR_INTERVAL}s"
echo "Memory logs: $LOG_DIR/memory_*_${SLURM_JOB_ID}.log"
echo "Start time: $(date)"
echo "=========================================="

# Create worker startup script
WORKER_SCRIPT="$LOG_DIR/worker_${SLURM_JOB_ID}.sh"
cat > $WORKER_SCRIPT << WORKER_EOF
#!/bin/bash
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
source $PROJECT_DIR/train/env_config.sh

# Set triton cache to scratch to avoid permission issues
export TRITON_CACHE_DIR=/scratch/a5l/shuqing.a5l/tmp/triton_cache_${SLURM_JOB_ID}
mkdir -p \$TRITON_CACHE_DIR
export TRITON_HOME=\$TRITON_CACHE_DIR

# Start memory monitor in background
MEMORY_LOG="$LOG_DIR/memory_\$(hostname)_${SLURM_JOB_ID}.log"
bash $MEMORY_MONITOR_SCRIPT \$MEMORY_LOG $MEMORY_MONITOR_INTERVAL &
MONITOR_PID=\$!
echo "Memory monitor started with PID \$MONITOR_PID, logging to \$MEMORY_LOG"

# Cleanup function
cleanup() {
    echo "Stopping memory monitor (PID \$MONITOR_PID)..."
    kill \$MONITOR_PID 2>/dev/null || true
}
trap cleanup EXIT

ray stop 2>/dev/null || true
echo "Starting Ray worker, connecting to \$1"
ray start --address \$1 --num-cpus \$2 --num-gpus 4 --disable-usage-stats --block
WORKER_EOF
chmod +x $WORKER_SCRIPT

# Start Ray worker nodes in background
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting Ray WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        bash $WORKER_SCRIPT $ip_head ${SLURM_CPUS_PER_TASK} &
done

# Start Ray head and training on head node (this will block until training completes)
echo "Starting Ray HEAD and training at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "
        source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
        source $PROJECT_DIR/train/env_config.sh

        # Set triton cache to scratch to avoid permission issues
        export TRITON_CACHE_DIR=/scratch/a5l/shuqing.a5l/tmp/triton_cache_${SLURM_JOB_ID}
        mkdir -p \$TRITON_CACHE_DIR
        export TRITON_HOME=\$TRITON_CACHE_DIR

        # Start memory monitor in background on head node
        MEMORY_LOG=\"$LOG_DIR/memory_\$(hostname)_${SLURM_JOB_ID}.log\"
        bash $MEMORY_MONITOR_SCRIPT \$MEMORY_LOG $MEMORY_MONITOR_INTERVAL &
        MONITOR_PID=\$!
        echo \"Memory monitor started with PID \$MONITOR_PID, logging to \$MEMORY_LOG\"

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

        # Wait for workers to connect
        echo 'Waiting for Ray workers to connect...'
        for attempt in {1..30}; do
            node_count=\$(ray status 2>/dev/null | grep -c 'node_' || echo '0')
            if [ \"\$node_count\" -ge \"2\" ]; then
                echo \"Ray cluster has \$node_count nodes - ready!\"
                break
            fi
            echo \"Waiting for Ray nodes... attempt \$attempt/30 (found \$node_count nodes)\"
            sleep 5
        done

        # Check Ray cluster status
        echo 'Ray cluster status:'
        ray status

        # Set RAY_ADDRESS for Python to connect to existing cluster
        export RAY_ADDRESS=\"\$head_node_ip:\$port\"
        echo \"RAY_ADDRESS set to: \$RAY_ADDRESS\"

        # Additional wait for workers to fully initialize before FSDP
        echo 'Waiting for worker processes to stabilize...'
        sleep 15

        export PYTHONPATH=$PROJECT_DIR
        cd $PROJECT_DIR

        echo '=========================================='
        echo 'Starting MoE Training...'
        echo '=========================================='

        python3 -m verl.trainer.main_dapo \
            --config-path=$CONFIG_PATH \
            --config-name=traj_grpo_moe \
            algorithm.adv_estimator=uis1 \
            data.train_files=$PROJECT_DIR/datasets/ui_s1_train.jsonl \
            data.val_files=$PROJECT_DIR/evaluation/dataset/android_control_evaluation_std.jsonl \
            data.train_batch_size=$BATCH_SIZE \
            data.val_batch_size=$((2 * BATCH_SIZE)) \
            data.max_prompt_length=8192 \
            data.max_response_length=512 \
            data.truncation=error \
            actor_rollout_ref.model.path=$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct \
            actor_rollout_ref.model.moe.enabled=true \
            actor_rollout_ref.model.moe.num_experts=$NUM_EXPERTS \
            actor_rollout_ref.model.moe.top_k=$TOP_K \
            actor_rollout_ref.model.moe.expert_lora_r=$EXPERT_LORA_R \
            actor_rollout_ref.model.moe.expert_lora_alpha=$EXPERT_LORA_ALPHA \
            \"actor_rollout_ref.model.moe.target_modules=$TARGET_MODULES\" \
            actor_rollout_ref.model.moe.router_hidden=$ROUTER_HIDDEN \
            actor_rollout_ref.model.moe.router_temperature=$ROUTER_TEMPERATURE \
            actor_rollout_ref.model.moe.balance_weight=$BALANCE_WEIGHT \
            actor_rollout_ref.model.moe.balance_type=$BALANCE_TYPE \
            actor_rollout_ref.model.moe.pooling_strategy=$POOLING_STRATEGY \
            actor_rollout_ref.model.moe.z_loss_weight=$Z_LOSS_WEIGHT \
            actor_rollout_ref.model.moe.use_vectorized_routing=$USE_VECTORIZED_ROUTING \
            actor_rollout_ref.model.moe.log_routing_matrix_freq=100 \
            actor_rollout_ref.actor.optim.lr=$LR \
            actor_rollout_ref.actor.grad_clip=$GRAD_CLIP \
            actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
            actor_rollout_ref.actor.use_torch_compile=False \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \
            actor_rollout_ref.actor.use_fixed_num_mini_batches=false \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.actor.use_kl_loss=True \
            actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.actor.entropy_coeff=0.0 \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.param_offload=False \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
            'actor_rollout_ref.actor.checkpoint.contents=[model,hf_model,optimizer,extra]' \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            actor_rollout_ref.rollout.name=$ENGINE \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
            actor_rollout_ref.rollout.max_model_len=16384 \
            actor_rollout_ref.rollout.enable_chunked_prefill=False \
            actor_rollout_ref.rollout.enforce_eager=True \
            actor_rollout_ref.rollout.free_cache_engine=True \
            actor_rollout_ref.rollout.limit_images=2 \
            actor_rollout_ref.rollout.n=4 \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.ref.fsdp_config.param_offload=False \
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
            trainer.project_name=gui_traj_grpo_moe \
            trainer.experiment_name=$EXPERIMENT_NAME \
            trainer.n_gpus_per_node=4 \
            trainer.nnodes=${SLURM_NNODES} \
            trainer.save_freq=20 \
            trainer.test_freq=10 \
            trainer.val_before_train=False \
            trainer.total_epochs=5

        TRAIN_EXIT_CODE=\$?

        # Stop memory monitor
        echo \"Stopping memory monitor (PID \$MONITOR_PID)...\"
        kill \$MONITOR_PID 2>/dev/null || true

        # Stop Ray
        ray stop

        exit \$TRAIN_EXIT_CODE
    " 2>&1 | tee $LOG_DIR/train_moe_${SLURM_JOB_ID}.log

TRAIN_EXIT_CODE=$?

echo "=========================================="
echo "MoE Training completed at $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

# Print MoE summary if training completed successfully
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "MoE Training Summary:"
    echo "  - Experts: $NUM_EXPERTS"
    echo "  - Expert LoRA Rank: $EXPERT_LORA_R"
    echo "  - Balance Weight: $BALANCE_WEIGHT"
    echo "  - Checkpoints saved to: $PROJECT_DIR/checkpoints/gui_traj_grpo_moe/$EXPERIMENT_NAME/"
    echo ""
    echo "To analyze routing patterns, check wandb logs for:"
    echo "  - moe/expert_*_utilization"
    echo "  - moe/load_balance_coefficient"
    echo "  - moe/routing_entropy_mean"
    echo "  - moe/routing_matrix/* (logged every $((100)) steps)"
fi

# Clean up scripts
rm -f $WORKER_SCRIPT
rm -f $MEMORY_MONITOR_SCRIPT

echo "Memory logs saved to: $LOG_DIR/memory_*_${SLURM_JOB_ID}.log"
echo "Training logs saved to: $LOG_DIR/train_moe_${SLURM_JOB_ID}.log"

exit $TRAIN_EXIT_CODE
