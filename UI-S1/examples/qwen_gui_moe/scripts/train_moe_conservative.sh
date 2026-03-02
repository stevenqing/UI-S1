#!/bin/bash
#
# MoE GUI Agent 训练启动脚本
#
# 使用保守配置进行训练，防止崩塌
#
# 使用方式:
#   bash scripts/train_moe_conservative.sh [nnodes] [gpus_per_node]
#
# 示例:
#   bash scripts/train_moe_conservative.sh 1 8    # 单机 8 卡
#   bash scripts/train_moe_conservative.sh 2 4    # 双机 4 卡
#

set -e  # 遇到错误立即退出

# ============================================================================
# 配置参数
# ============================================================================

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# 配置文件
CONFIG_FILE="${PROJECT_ROOT}/examples/qwen_gui_moe/config/traj_grpo_moe.yaml"

# 默认资源配置
NNODES=${1:-1}
NGPUS_PER_NODE=${2:-8}

# 训练配置
EXPERIMENT_NAME="moe_4experts_r16_conservative_topk2_fix"
PROJECT_NAME="gui_traj_grpo_moe"

# ============================================================================
# 验证配置
# ============================================================================

echo "=========================================="
echo "MoE 训练启动脚本"
echo "=========================================="
echo "项目根目录: ${PROJECT_ROOT}"
echo "配置文件: ${CONFIG_FILE}"
echo "节点数: ${NNODES}"
echo "每节点GPU数: ${NGPUS_PER_NODE}"
echo "=========================================="

# 检查配置文件是否存在
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "❌ 配置文件不存在: ${CONFIG_FILE}"
    exit 1
fi

# 验证配置
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
# 打印关键配置
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

# ============================================================================
# 启动训练
# ============================================================================

echo ""
echo "启动训练..."
echo "实验名称: ${EXPERIMENT_NAME}"
echo ""

# 进入项目根目录
cd "${PROJECT_ROOT}"

# 构建训练命令
TRAIN_CMD="python -m verl.trainer.main_dapo \
    --config ${CONFIG_FILE} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}"

# 如果是多节点，添加 Ray 地址配置
if [ ${NNODES} -gt 1 ]; then
    # 假设 Ray 集群已通过 ray start 启动
    TRAIN_CMD="${TRAIN_CMD} \
        ray_init.address=auto"
fi

echo "执行命令:"
echo "${TRAIN_CMD}"
echo ""

# 执行训练
${TRAIN_CMD}

# ============================================================================
# 训练完成
# ============================================================================

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
