#!/bin/bash
#SBATCH --job-name=grpo_sftv2_${EXP_NAME}
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp3/logs/grpo_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp3/logs/grpo_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# GRPO on SFT v2 — Exp 2 (binary) or Exp 3 (dense)
#
# Usage:
#   EXP_NAME=exp2 CONFIG_NAME=traj_grpo_sftv2_exp2 sbatch scripts/exp3/train_grpo_sftv2.slurm
#   EXP_NAME=exp3 CONFIG_NAME=traj_grpo_sftv2_exp3 sbatch scripts/exp3/train_grpo_sftv2.slurm
# ============================================

set -e

EXP_NAME=${EXP_NAME:-exp2}
CONFIG_NAME=${CONFIG_NAME:-traj_grpo_sftv2_exp2}

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CONFIG_PATH="$PROJECT_DIR/train_GUI_360/moe_rl"
LOG_DIR="$PROJECT_DIR/scripts/exp3/logs"

mkdir -p $LOG_DIR

# Node setup
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

BATCH_SIZE=$((SLURM_NNODES * 2))

echo "=========================================="
echo "GRPO on SFT v2 — ${EXP_NAME}"
echo "=========================================="
echo "Head Node: $head_node ($head_node_ip)"
echo "Nodes: ${SLURM_NNODES}, Total GPUs: $((SLURM_NNODES * 4))"
echo "Config: $CONFIG_NAME"
echo "Batch Size: $BATCH_SIZE"
echo "Start: $(date)"
echo "=========================================="

# Worker startup script
WORKER_SCRIPT="$LOG_DIR/worker_${EXP_NAME}_${SLURM_JOB_ID}.sh"
cat > $WORKER_SCRIPT << WORKER_EOF
#!/bin/bash
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
source $PROJECT_DIR/train/env_config.sh

export TRITON_CACHE_DIR=/scratch/a5l/shuqing.a5l/tmp/triton_cache_${SLURM_JOB_ID}
mkdir -p \$TRITON_CACHE_DIR
export TRITON_HOME=\$TRITON_CACHE_DIR

ray stop 2>/dev/null || true
echo "Starting Ray worker, connecting to \$1"
ray start --address \$1 --num-cpus \$2 --num-gpus 4 --disable-usage-stats --block
WORKER_EOF
chmod +x $WORKER_SCRIPT

# Start workers
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting Ray WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        bash $WORKER_SCRIPT $ip_head ${SLURM_CPUS_PER_TASK} &
done

# Head node: Ray head + training
echo "Starting Ray HEAD and training at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "
        source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
        source $PROJECT_DIR/train/env_config.sh

        export TRITON_CACHE_DIR=/scratch/a5l/shuqing.a5l/tmp/triton_cache_${SLURM_JOB_ID}
        mkdir -p \$TRITON_CACHE_DIR
        export TRITON_HOME=\$TRITON_CACHE_DIR

        ray stop 2>/dev/null || true

        echo 'Starting Ray head node...'
        ray start --head --node-ip-address=$head_node_ip --port=$port \
            --num-cpus ${SLURM_CPUS_PER_TASK} --num-gpus 4 \
            --disable-usage-stats --include-dashboard=false

        echo 'Waiting for Ray to initialize...'
        sleep 10

        echo 'Waiting for Ray workers...'
        for attempt in {1..30}; do
            node_count=\$(ray status 2>/dev/null | grep -c 'node_' || echo '0')
            if [ \"\$node_count\" -ge \"2\" ]; then
                echo \"Ray cluster has \$node_count nodes - ready!\"
                break
            fi
            echo \"Waiting... attempt \$attempt/30 (found \$node_count nodes)\"
            sleep 5
        done

        ray status

        export RAY_ADDRESS=\"\$head_node_ip:\$port\"
        sleep 15

        export PYTHONPATH=$PROJECT_DIR
        cd $PROJECT_DIR

        echo '=========================================='
        echo 'Starting GRPO on SFT v2 (${EXP_NAME})...'
        echo '=========================================='

        python3 -m verl.trainer.main_dapo \
            --config-path=$CONFIG_PATH \
            --config-name=$CONFIG_NAME \
            data.train_batch_size=$BATCH_SIZE \
            trainer.nnodes=${SLURM_NNODES} \
            trainer.n_gpus_per_node=4

        TRAIN_EXIT_CODE=\$?
        ray stop
        exit \$TRAIN_EXIT_CODE
    " 2>&1 | tee $LOG_DIR/grpo_${EXP_NAME}_${SLURM_JOB_ID}.log

TRAIN_EXIT_CODE=$?

echo "=========================================="
echo "GRPO ${EXP_NAME} completed at $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

rm -f $WORKER_SCRIPT

exit $TRAIN_EXIT_CODE
