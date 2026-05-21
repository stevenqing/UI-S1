#!/bin/bash
#SBATCH --job-name=coop-grpo-ac
#SBATCH --output=/lus/lfs1aip2/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v10/verl_recipe/logs/train_%j.log
#SBATCH --error=/lus/lfs1aip2/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v10/verl_recipe/logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=8
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
# Note: expandable_segments:True is incompatible with vLLM CuMemAllocator

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
export OUTPUT_DIR="output/coop_grpo_ac"
export GPUS_PER_NODE=4
export NNODES=8

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_DIR/v10/verl_recipe/logs"

echo "=== Cooperative GRPO Training on AndroidControl ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Head Node: $head_node ($head_node_ip)"
echo "Nodes: ${SLURM_NNODES}, Total GPUs: $((SLURM_NNODES * 4))"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"

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
# Note: expandable_segments:True is incompatible with vLLM CuMemAllocator
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
        # Note: expandable_segments:True is incompatible with vLLM CuMemAllocator
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
        export WANDB_PROJECT=coop_grpo_ac
        cd $PROJECT_DIR

        python v10/verl_recipe/main_coop_grpo.py \
            trainer.n_gpus_per_node=$GPUS_PER_NODE \
            trainer.nnodes=$NNODES \
            trainer.logger=console \
            trainer.experiment_name=\"v10_coop_ac_\${SLURM_JOB_ID}\"

        echo 'Training complete!'
        ray stop
    "

wait
