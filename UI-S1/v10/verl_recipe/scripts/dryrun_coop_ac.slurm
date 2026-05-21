#!/bin/bash
#SBATCH --job-name=dryrun-coop
#SBATCH --output=/lus/lfs1aip2/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v10/verl_recipe/logs/dryrun_%j.log
#SBATCH --error=/lus/lfs1aip2/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v10/verl_recipe/logs/dryrun_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=workq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --gres=gpu:4

set -e

PROJECT_DIR="/lus/lfs1aip2/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
cd "$PROJECT_DIR"

# NCCL config for Slingshot
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=LOC
export NCCL_CROSS_NIC=1
export NCCL_TIMEOUT=10800

# ── Multi-node setup ─────────────────────────────────────────────
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

# Paths
export MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
export TRAIN_PARQUET="data/ac_train.parquet"
export VAL_PARQUET="data/ac_val.parquet"
export OUTPUT_DIR="output/dryrun_coop_ac"
export GPUS_PER_NODE=4
export NNODES=2

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_DIR/v10/verl_recipe/logs"

echo "=== Dry-run: Cooperative GRPO on AndroidControl ==="
echo "Head Node: $head_node ($head_node_ip)"
echo "Nodes: ${SLURM_NNODES}, Total GPUs: $((SLURM_NNODES * 4))"
echo "Model: $MODEL_PATH"
echo "Data: $TRAIN_PARQUET"

# ── Worker startup script ────────────────────────────────────────
WORKER_SCRIPT="$PROJECT_DIR/v10/verl_recipe/logs/worker_${SLURM_JOB_ID}.sh"
cat > $WORKER_SCRIPT << 'WORKER_EOF'
#!/bin/bash
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=LOC
export NCCL_CROSS_NIC=1
export NCCL_TIMEOUT=10800
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

ray stop 2>/dev/null || true
echo "Starting Ray worker, connecting to $1"
ray start --address $1 --num-cpus $2 --num-gpus 4 --disable-usage-stats --block
WORKER_EOF
chmod +x $WORKER_SCRIPT

# Start workers on non-head nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting Ray WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        bash $WORKER_SCRIPT $ip_head ${SLURM_CPUS_PER_TASK} &
done

# ── Head node: Ray head + training ───────────────────────────────
srun --nodes=1 --ntasks=1 -w "$head_node" \
    bash -c "
        source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
        export NCCL_SOCKET_IFNAME=hsn0
        export GLOO_SOCKET_IFNAME=hsn0
        export NCCL_NET=Socket
        export NCCL_IB_DISABLE=1
        export NCCL_P2P_LEVEL=LOC
        export NCCL_CROSS_NIC=1
        export NCCL_TIMEOUT=10800
        export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:\${LD_LIBRARY_PATH:-}\"

        ray stop 2>/dev/null || true

        echo 'Starting Ray head node...'
        ray start --head --node-ip-address=$head_node_ip --port=$port \
            --num-cpus ${SLURM_CPUS_PER_TASK} --num-gpus 4 \
            --disable-usage-stats --include-dashboard=false

        echo 'Waiting for Ray workers...'
        sleep 10
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

        export RAY_ADDRESS=\"$head_node_ip:$port\"
        export PYTHONPATH=$PROJECT_DIR
        cd $PROJECT_DIR

        python v10/verl_recipe/main_coop_grpo.py \
            actor_rollout_ref.model.path=\"$MODEL_PATH\" \
            data.train_files=\"$TRAIN_PARQUET\" \
            data.val_files=\"$VAL_PARQUET\" \
            data.train_batch_size=4 \
            data.filter_overlong_prompts=true \
            trainer.n_gpus_per_node=$GPUS_PER_NODE \
            trainer.nnodes=$NNODES \
            trainer.total_epochs=1 \
            trainer.save_freq=9999 \
            trainer.test_freq=9999 \
            trainer.val_before_train=false \
            trainer.logger=console \
            trainer.default_local_dir=\"$OUTPUT_DIR\" \
            trainer.experiment_name=\"dryrun_\${SLURM_JOB_ID}\" \
            actor_rollout_ref.actor.ppo_mini_batch_size=4 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.rollout.n=2 \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1

        echo 'Dry-run complete!'
        ray stop
    "

wait
