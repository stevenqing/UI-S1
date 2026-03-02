#!/bin/bash
#SBATCH --job-name=train_diag
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_diag_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_diag_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

# ============================================
# Training with Diagnosis Logging Enabled
# 用于定位具体卡住位置
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CONFIG_PATH="$PROJECT_DIR/examples/qwen_gui_static_grpo/config"
LOG_DIR="$PROJECT_DIR/train/logs"

mkdir -p $LOG_DIR

ENGINE="vllm"
MODE="mean_std_norm"
GAMMA=0.5
DAPO=False
EXPERIMENT_NAME="train_with_diag_${SLURM_JOB_ID}"

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
echo "Training with Diagnosis Logging"
echo "=========================================="
echo "Head Node: $head_node ($head_node_ip)"
echo "Nodes: ${SLURM_NNODES}, GPUs: $((SLURM_NNODES * 4))"
echo "Start time: $(date)"
echo "=========================================="

# Enable diagnosis logging in sharding manager
echo "Enabling diagnosis logging..."
cd $PROJECT_DIR
python3 scripts/add_diagnosis_logging.py --enable --file vllm 2>&1 || echo "Warning: Could not enable diagnosis logging"

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

        # 完整的 NCCL 诊断
        export NCCL_DEBUG=INFO
        export NCCL_DEBUG_SUBSYS=INIT,NET,COLL
        export TORCH_DISTRIBUTED_DEBUG=DETAIL
        export NCCL_TIMEOUT=1800

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

        echo ''
        echo '=========================================='
        echo 'Starting training with diagnosis logging'
        echo 'Watch for [DIAG] prefixed lines'
        echo '=========================================='
        echo ''

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
            actor_rollout_ref.actor.fsdp_config.param_offload=True \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.rollout.name=$ENGINE \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
            actor_rollout_ref.rollout.n=8 \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            algorithm.gamma=$GAMMA \
            algorithm.filter_groups.enable=$DAPO \
            trainer.project_name=gui_traj_grpo_diag \
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
    " 2>&1 | tee $LOG_DIR/train_diag_${SLURM_JOB_ID}.log

TRAIN_EXIT_CODE=$?

# Disable diagnosis logging
echo "Disabling diagnosis logging..."
cd $PROJECT_DIR
python3 scripts/add_diagnosis_logging.py --disable --file vllm 2>&1 || echo "Warning: Could not disable diagnosis logging"

echo ""
echo "=========================================="
echo "Training completed at $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo ""
echo "To analyze diagnosis logs, search for [DIAG] in:"
echo "  $LOG_DIR/train_diag_${SLURM_JOB_ID}.log"
echo ""
echo "Example analysis commands:"
echo "  grep '\\[DIAG\\]' $LOG_DIR/train_diag_${SLURM_JOB_ID}.log"
echo "  grep '\\[DIAG\\].*Rank 0' $LOG_DIR/train_diag_${SLURM_JOB_ID}.log"
echo "=========================================="

rm -f $WORKER_SCRIPT
exit $TRAIN_EXIT_CODE
