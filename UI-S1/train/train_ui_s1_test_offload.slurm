#!/bin/bash
#SBATCH --job-name=test_offload
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/test_offload_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/test_offload_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ============================================
# Test Training with param_offload DISABLED
# 用于验证关闭 offload 是否解决卡住问题
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CONFIG_PATH="$PROJECT_DIR/examples/qwen_gui_static_grpo/config"
LOG_DIR="$PROJECT_DIR/train/logs"

mkdir -p $LOG_DIR

# ============================================
# 关键改动：关闭 offload
# ============================================
PARAM_OFFLOAD=False      # 原来是 True
OPTIMIZER_OFFLOAD=False  # 原来是 True/False

ENGINE="vllm"
MODE="mean_std_norm"
GAMMA=0.5
DAPO=False
DAPO_THRESHOLD=0.3
PATCH_THRESHOLD=2
EXPERIMENT_NAME="test_no_offload_${SLURM_JOB_ID}"

# Get node information
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head head_node_ip port

BATCH_SIZE=$((SLURM_NNODES * 8))

echo "=========================================="
echo "Test Training - Offload DISABLED"
echo "=========================================="
echo "param_offload: $PARAM_OFFLOAD"
echo "optimizer_offload: $OPTIMIZER_OFFLOAD"
echo "Head Node: $head_node ($head_node_ip)"
echo "Nodes: ${SLURM_NNODES}, GPUs: $((SLURM_NNODES * 4))"
echo "Start time: $(date)"
echo "=========================================="

# Worker script
WORKER_SCRIPT="$LOG_DIR/worker_${SLURM_JOB_ID}.sh"
cat > $WORKER_SCRIPT << 'WORKER_EOF'
#!/bin/bash
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
source $2/train/env_config.sh
ray stop 2>/dev/null || true
ray start --address $1 --num-cpus $3 --num-gpus 4 --disable-usage-stats --block
WORKER_EOF
chmod +x $WORKER_SCRIPT

# Start workers
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting Ray WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        bash $WORKER_SCRIPT $ip_head $PROJECT_DIR ${SLURM_CPUS_PER_TASK} &
done

# Start head and training
srun --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "
        source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
        source $PROJECT_DIR/train/env_config.sh

        # 额外的 NCCL 诊断设置
        export NCCL_DEBUG=INFO
        export NCCL_DEBUG_SUBSYS=INIT,NET,COLL
        export TORCH_DISTRIBUTED_DEBUG=DETAIL
        # Disable Ray log deduplication to see all worker messages
        export RAY_DEDUP_LOGS=0
        # Set GPUs per node for local_rank calculation
        export VERL_N_GPUS_PER_NODE=4

        ray stop 2>/dev/null || true
        ray start --head --node-ip-address=$head_node_ip --port=$port \
            --num-cpus ${SLURM_CPUS_PER_TASK} --num-gpus 4 \
            --disable-usage-stats --include-dashboard=false

        sleep 10

        for attempt in {1..30}; do
            node_count=\$(ray status 2>/dev/null | grep -c 'node_' || echo '0')
            if [ \"\$node_count\" -ge \"2\" ]; then
                echo \"Ray cluster has \$node_count nodes - ready!\"
                break
            fi
            echo \"Waiting for Ray nodes... attempt \$attempt/30\"
            sleep 5
        done

        ray status
        export RAY_ADDRESS=\"\$head_node_ip:\$port\"
        sleep 15

        export PYTHONPATH=$PROJECT_DIR
        cd $PROJECT_DIR

        # Use file-based rendezvous for better multi-node reliability
        export VERL_RENDEZVOUS_FILE=$PROJECT_DIR/train/logs/rendezvous_${SLURM_JOB_ID}
        # Clean up any stale rendezvous file
        rm -f \$VERL_RENDEZVOUS_FILE 2>/dev/null || true

        echo ''
        echo '=========================================='
        echo 'Starting training with OFFLOAD DISABLED'
        echo 'Rendezvous file: '\$VERL_RENDEZVOUS_FILE
        echo '=========================================='
        echo ''

        # 只训练很短时间用于测试
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
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.rollout.name=$ENGINE \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
            actor_rollout_ref.rollout.n=4 \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.ref.fsdp_config.param_offload=$PARAM_OFFLOAD \
            algorithm.gamma=$GAMMA \
            algorithm.filter_groups.enable=$DAPO \
            trainer.project_name=gui_traj_grpo_test \
            trainer.experiment_name=$EXPERIMENT_NAME \
            trainer.n_gpus_per_node=4 \
            trainer.nnodes=${SLURM_NNODES} \
            trainer.save_freq=999 \
            trainer.test_freq=999 \
            trainer.val_before_train=False \
            trainer.total_epochs=1

        TRAIN_EXIT_CODE=\$?
        ray stop
        exit \$TRAIN_EXIT_CODE
    " 2>&1 | tee $LOG_DIR/test_offload_${SLURM_JOB_ID}.log

TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Test completed at $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "RESULT: Training with offload=False SUCCEEDED"
    echo "This confirms that param_offload is causing the hang."
else
    echo "RESULT: Training FAILED even with offload=False"
    echo "The issue may not be related to offload."
fi
echo "=========================================="

# Collect NCCL debug logs from all nodes
echo ""
echo "=========================================="
echo "Collecting NCCL debug logs..."
echo "=========================================="
for node in $nodes; do
    echo "Collecting logs from $node..."
    srun --nodes=1 --ntasks=1 -w "$node" bash -c "
        mkdir -p $LOG_DIR/nccl_${SLURM_JOB_ID}
        cp /tmp/nccl_debug_rank*.log $LOG_DIR/nccl_${SLURM_JOB_ID}/ 2>/dev/null || echo 'No NCCL logs on $node'
    " 2>/dev/null || true
done

rm -f $WORKER_SCRIPT
exit $TRAIN_EXIT_CODE
