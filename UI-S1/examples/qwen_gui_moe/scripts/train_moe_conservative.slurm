#!/bin/bash
#SBATCH --job-name=moe_gui_fix
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_moe_%j.out
#SBATCH --error=logs/train_moe_%j.err

#
# MoE GUI Agent 训练 Slurm 脚本
#
# 使用保守配置进行训练，防止崩塌
#
# 提交方式:
#   sbatch scripts/train_moe_conservative.slurm
#
# 监控:
#   squeue -u $USER
#   tail -f logs/train_moe_<JOB_ID>.out
#

set -e  # 遇到错误立即退出

# ============================================================================
# 环境设置
# ============================================================================

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# 配置文件
CONFIG_FILE="${PROJECT_ROOT}/examples/qwen_gui_moe/config/traj_grpo_moe.yaml"

# 训练配置
EXPERIMENT_NAME="moe_4experts_r16_conservative_topk2_fix"
PROJECT_NAME="gui_traj_grpo_moe"

# 获取 Slurm 分配的资源
NNODES=${SLURM_JOB_NUM_NODES:-2}
NGPUS_PER_NODE=4  # 从 #SBATCH --gres=gpu:4

# 获取主节点地址
MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
MASTER_PORT=29500

echo "=========================================="
echo "MoE 训练 Slurm 任务"
echo "=========================================="
echo "任务ID: ${SLURM_JOB_ID}"
echo "节点数: ${NNODES}"
echo "每节点GPU数: ${NGPUS_PER_NODE}"
echo "主节点: ${MASTER_ADDR}:${MASTER_PORT}"
echo "=========================================="

# ============================================================================
# 验证配置
# ============================================================================

echo ""
echo "验证配置文件..."
python "${PROJECT_ROOT}/examples/qwen_gui_moe/scripts/validate_config.py" \
    --config "${CONFIG_FILE}"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 配置验证失败！请检查配置文件。"
    exit 1
fi

echo ""
echo "✅ 配置验证通过！"

# ============================================================================
# 设置环境变量
# ============================================================================

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_TIMEOUT=50
export NCCL_NET_GDR_LEVEL=5

# Ray 配置
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# 项目根目录
cd "${PROJECT_ROOT}"

# ============================================================================
# 启动 Ray 集群 (多节点)
# ============================================================================

if [ ${NNODES} -gt 1 ]; then
    echo ""
    echo "启动 Ray 集群..."

    # 在所有节点上启动 Ray
    srun --ntasks=${NNODES} --ntasks-per-node=1 \
        ray start --head --address=${MASTER_ADDR}:${MASTER_PORT} --block &

    # 等待 Ray 集群启动
    sleep 30

    echo "✅ Ray 集群已启动"
fi

# ============================================================================
# 启动训练
# ============================================================================

echo ""
echo "=========================================="
echo "关键训练配置"
echo "=========================================="
echo "kl_loss_coef: 0.1         (1000x 增加)"
echo "lr: 1e-5                   (降低 10x)"
echo "grad_clip: 0.5            (降低 50%)"
echo "clip_ratio: 0.1           (降低 50%)"
echo "balance_weight: 0.2       (MoE 负载均衡)"
echo "z_loss_weight: 0.01       (MoE 稳定性)"
echo "total_epochs: 10          (增加训练轮数)"
echo "=========================================="

echo ""
echo "启动训练..."
echo "实验名称: ${EXPERIMENT_NAME}"
echo ""

# 构建训练命令
TRAIN_CMD="python -m verl.trainer.main_dapo \
    --config ${CONFIG_FILE} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}"

# 如果是多节点，添加 Ray 地址配置
if [ ${NNODES} -gt 1 ]; then
    TRAIN_CMD="${TRAIN_CMD} \
        ray_init.address=${MASTER_ADDR}:${MASTER_PORT}"
fi

echo "执行命令:"
echo "${TRAIN_CMD}"
echo ""

# 只在主节点执行训练
if [ "${SLURM_NODEID}" -eq 0 ]; then
    # 创建日志目录
    mkdir -p logs

    # 执行训练
    ${TRAIN_CMD} 2>&1 | tee logs/train_moe_${SLURM_JOB_ID}.log

    # 训练完成
    echo ""
    echo "=========================================="
    echo "训练完成！"
    echo "=========================================="
    echo "Checkpoint 保存在:"
    echo "  checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/"
    echo ""
    echo "Wandb 监控:"
    echo "  https://wandb.ai/${USER}/${PROJECT_NAME}"
    echo "=========================================="
fi

# ============================================================================
# 清理
# ============================================================================

if [ ${NNODES} -gt 1 ]; then
    echo ""
    echo "停止 Ray 集群..."
    srun --ntasks=${NNODES} --ntasks-per-node=1 ray stop --force
fi
