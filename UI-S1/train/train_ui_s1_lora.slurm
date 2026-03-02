n#!/bin/bash
#SBATCH --job-name=train_ui_s1_lora
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_ui_s1_lora_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_ui_s1_lora_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# UI-S1 LoRA Training Script with Ray + GRPO (Multi-Node)
# ============================================

set -e

# Set paths
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CONFIG_PATH="$PROJECT_DIR/examples/qwen_gui_static_grpo/config"
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
# LoRA Configuration
# ============================================
LORA_RANK=64
LORA_ALPHA=128
# Target modules for Qwen2.5-VL (attention + MLP layers)
TARGET_MODULES='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'

# Training configuration
ENGINE="vllm"
MODE="mean_std_norm"
GAMMA=0.5
DAPO=False
DAPO_THRESHOLD=0.3
PATCH_THRESHOLD=2
EXPERIMENT_NAME="qwenvl_uis1_LORA_r${LORA_RANK}_a${LORA_ALPHA}_gamma_${GAMMA}"

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

# LoRA uses less memory, can use larger batch size per GPU
BATCH_SIZE=$((SLURM_NNODES * 16))

echo "=========================================="
echo "Starting UI-S1 LoRA Training (Multi-Node)"
echo "Project Dir: $PROJECT_DIR"
echo "Head Node: $head_node ($head_node_ip)"
echo "IP Head: $ip_head"
echo "Nodes: ${SLURM_NNODES}"
echo "GPUs per node: 4"
echo "Total GPUs: $((SLURM_NNODES * 4))"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "LoRA Rank: $LORA_RANK"
echo "LoRA Alpha: $LORA_ALPHA"
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

        # Set RAY_ADDRESS for Python to connect to existing cluster (use GCS address, not ray client)
        export RAY_ADDRESS=\"\$head_node_ip:\$port\"
        echo \"RAY_ADDRESS set to: \$RAY_ADDRESS\"

        # Additional wait for workers to fully initialize before FSDP
        echo 'Waiting for worker processes to stabilize...'
        sleep 15

        export PYTHONPATH=$PROJECT_DIR
        cd $PROJECT_DIR

        echo 'Starting LoRA training...'
        python3 -m verl.trainer.main_dapo \
            --config-path=$CONFIG_PATH \
            --config-name=traj_grpo \
            algorithm.adv_estimator=uis1 \
            data.train_files=$PROJECT_DIR/datasets/ui_s1_train.jsonl \
            data.val_files=$PROJECT_DIR/evaluation/dataset/android_control_evaluation_std.jsonl \
            data.train_batch_size=$BATCH_SIZE \
            data.val_batch_size=$((8 * BATCH_SIZE)) \
            data.max_prompt_length=8192 \
            data.max_response_length=512 \
            data.truncation=error \
            actor_rollout_ref.model.path=$PROJECT_DIR/checkpoints/Qwen2.5-VL-7B-Instruct \
            actor_rollout_ref.model.lora_rank=$LORA_RANK \
            actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
            actor_rollout_ref.model.target_modules='$TARGET_MODULES' \
            actor_rollout_ref.actor.optim.lr=1e-4 \
            actor_rollout_ref.actor.use_torch_compile=False \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \
            actor_rollout_ref.actor.use_fixed_num_mini_batches=false \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
            actor_rollout_ref.actor.use_kl_loss=True \
            actor_rollout_ref.actor.kl_loss_coef=0.0001 \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.actor.entropy_coeff=0.0 \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.param_offload=False \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
            'actor_rollout_ref.actor.checkpoint.contents=[model,hf_model,optimizer,extra]' \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            actor_rollout_ref.rollout.name=$ENGINE \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
            actor_rollout_ref.rollout.max_model_len=16384 \
            actor_rollout_ref.rollout.enable_chunked_prefill=False \
            actor_rollout_ref.rollout.enforce_eager=True \
            actor_rollout_ref.rollout.free_cache_engine=True \
            actor_rollout_ref.rollout.limit_images=2 \
            actor_rollout_ref.rollout.n=4 \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
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
            trainer.nnodes=${SLURM_NNODES} \
            trainer.save_freq=5 \
            trainer.test_freq=10 \
            trainer.val_before_train=False \
            trainer.total_epochs=3

        TRAIN_EXIT_CODE=\$?

        # Stop memory monitor
        echo \"Stopping memory monitor (PID \$MONITOR_PID)...\"
        kill \$MONITOR_PID 2>/dev/null || true

        # Stop Ray
        ray stop

        exit \$TRAIN_EXIT_CODE
    " 2>&1 | tee $LOG_DIR/train_lora_${SLURM_JOB_ID}.log

TRAIN_EXIT_CODE=$?

echo "=========================================="
echo "LoRA Training completed at $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

# Clean up scripts
rm -f $WORKER_SCRIPT
rm -f $MEMORY_MONITOR_SCRIPT

echo "Memory logs saved to: $LOG_DIR/memory_*_${SLURM_JOB_ID}.log"

exit $TRAIN_EXIT_CODE
