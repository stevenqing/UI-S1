#!/bin/bash
#SBATCH --job-name=moe_rl_uis1
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/moe_rl_v5/logs/moe_rl_uis1_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/moe_rl_v5/logs/moe_rl_uis1_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# UI-S1 MoE RL: Random-Init, Soft Routing
#
# 4 experts (r=32), top-k=4 (soft, all experts), 7 target modules
# Random initialization — no SVD warm-start
# 8 nodes (32 GPUs), 1K trajectories
#
# Key fix vs old collapsed run (r16/topk4/2modules/no-entropy):
#   r=32, 7 modules, entropy_coeff=0.01
#
# Usage:
#   sbatch train/moe_rl_v5/train_moe_rl_uis1.slurm
# ============================================

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CONFIG_PATH="$PROJECT_DIR/train/moe_rl_v5"
LOG_DIR="$CONFIG_PATH/logs"

mkdir -p $LOG_DIR

# ============================================
# MoE Configuration
# ============================================
NUM_EXPERTS=4
TOP_K=4       # Soft routing: use all 4 experts with continuous weights
EXPERT_LORA_R=32
EXPERT_LORA_ALPHA=64
TARGET_MODULES='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'
ROUTER_HIDDEN=256
ROUTER_TEMPERATURE=0.5
BALANCE_WEIGHT=0.2
BALANCE_TYPE="mse"
Z_LOSS_WEIGHT=0.01
POOLING_STRATEGY="mean"
USE_VECTORIZED_ROUTING=true

# Actor training — higher KL since random init is close to base
LR=1e-5
KL_LOSS_COEF=0.1
CLIP_RATIO=0.1
GRAD_CLIP=0.5
TOTAL_EPOCHS=5

# RL settings
ENGINE="vllm"
MODE="mean_std_norm"
GAMMA=0.5

BATCH_SIZE=$((SLURM_NNODES * 1))

EXPERIMENT_NAME="moe_${NUM_EXPERTS}exp_r${EXPERT_LORA_R}_topk${TOP_K}_soft_randinit"

# ============================================
# Node setup
# ============================================
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

echo "=========================================="
echo "UI-S1 MoE RL — Soft Routing, Random Init r32"
echo "=========================================="
echo "Head Node: $head_node ($head_node_ip)"
echo "Nodes: ${SLURM_NNODES}, GPUs/node: 4, Total GPUs: $((SLURM_NNODES * 4))"
echo "Batch size: $BATCH_SIZE"
echo "------------------------------------------"
echo "MoE Config:"
echo "  Experts: $NUM_EXPERTS, Top-K: $TOP_K"
echo "  LoRA rank: $EXPERT_LORA_R, alpha: $EXPERT_LORA_ALPHA"
echo "  Target modules: $TARGET_MODULES"
echo "  Balance: $BALANCE_WEIGHT ($BALANCE_TYPE), Z-loss: $Z_LOSS_WEIGHT"
echo "  Routing: SOFT (top_k=4, all experts)"
echo "  Init: RANDOM (no warm-start)"
echo "------------------------------------------"
echo "Actor: lr=$LR, kl=$KL_LOSS_COEF, clip=$CLIP_RATIO, grad_clip=$GRAD_CLIP"
echo "  optimizer_offload=false, micro_batch=2 (r=32 fits in memory)"
echo "  entropy_coeff=0.01 (encourage exploration)"
echo "Epochs: $TOTAL_EPOCHS"
echo "Experiment: $EXPERIMENT_NAME"
echo "Start: $(date)"
echo "=========================================="

# ============================================
# Worker startup script
# ============================================
WORKER_SCRIPT="$LOG_DIR/worker_uis1_${SLURM_JOB_ID}.sh"
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

# ============================================
# Head node: Ray head + training
# ============================================
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
        echo 'Starting UI-S1 MoE RL Training (Random Init r32)...'
        echo '=========================================='

        python3 -m verl.trainer.main_dapo \
            --config-path=$CONFIG_PATH \
            --config-name=traj_grpo_moe_uis1 \
            algorithm.adv_estimator=uis1 \
            data.train_files=$PROJECT_DIR/datasets/ui_s1_dataset/ui_s1_train.jsonl \
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
            actor_rollout_ref.model.moe.z_loss_weight=$Z_LOSS_WEIGHT \
            actor_rollout_ref.model.moe.pooling_strategy=$POOLING_STRATEGY \
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
            actor_rollout_ref.actor.entropy_coeff=0.01 \
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
            algorithm.patch_threshold=2 \
            algorithm.filter_groups.enable=False \
            trainer.critic_warmup=0 \
            'trainer.logger=[console,wandb]' \
            trainer.project_name=ui_s1_moe_rl \
            trainer.experiment_name=$EXPERIMENT_NAME \
            trainer.n_gpus_per_node=4 \
            trainer.nnodes=${SLURM_NNODES} \
            trainer.save_freq=10 \
            +trainer.max_ckpt_to_keep=5 \
            +trainer.hf_repo_id=Stevenshuqing/ui-s1-moe-rl \
            +trainer.hf_private=true \
            trainer.test_freq=10 \
            trainer.val_before_train=False \
            trainer.total_epochs=$TOTAL_EPOCHS

        TRAIN_EXIT_CODE=\$?
        ray stop
        exit \$TRAIN_EXIT_CODE
    " 2>&1 | tee $LOG_DIR/train_moe_rl_uis1_${SLURM_JOB_ID}.log

TRAIN_EXIT_CODE=$?

echo "=========================================="
echo "UI-S1 MoE RL Training completed at $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "UI-S1 MoE RL Training Summary:"
    echo "  Experts: $NUM_EXPERTS (r=$EXPERT_LORA_R, 7 modules, random init)"
    echo "  Top-K: $TOP_K (sparse routing)"
    echo "  KL coef: $KL_LOSS_COEF, entropy: 0.01"
    echo "  Checkpoints: $PROJECT_DIR/checkpoints/ui_s1_moe_rl/$EXPERIMENT_NAME/"
    echo ""
    echo "Next steps:"
    echo "  1. Check wandb for routing patterns and reward curves"
    echo "  2. Run evaluation: sbatch eval_scripts/eval_moe_rl_uis1.slurm"
fi

rm -f $WORKER_SCRIPT

exit $TRAIN_EXIT_CODE
