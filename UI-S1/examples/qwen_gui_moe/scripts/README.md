# MoE GUI Agent 训练脚本

## 概述

这些脚本用于运行 MoE (Mixture of Experts) GUI Agent 训练，使用保守配置防止训练崩塌。

## 关键配置修复

| 参数 | 旧值 (导致崩塌) | 新值 (保守配置) | 改进幅度 |
|------|----------------|---------------|---------|
| `kl_loss_coef` | 0.0001 | **0.1** | +1000x ⭐ |
| `lr` | 1e-4 | **1e-5** | -10x |
| `grad_clip` | 1.0 | **0.5** | -50% |
| `clip_ratio` | 0.2 | **0.1** | -50% |
| `balance_weight` | 0.1 | **0.2** | +100% |
| `z_loss_weight` | 0.0 | **0.01** | 新增 |
| `total_epochs` | 3-5 | **10** | +2-3x |

## 文件说明

### 1. `validate_config.py` - 配置验证脚本

验证 MoE 配置文件是否包含正确的保守设置。

```bash
python examples/qwen_gui_moe/scripts/validate_config.py
```

### 2. `train_moe_conservative.sh` - 训练启动脚本

用于手动启动训练的 Shell 脚本。

```bash
# 单机 8 卡
bash examples/qwen_gui_moe/scripts/train_moe_conservative.sh 1 8

# 双机 4 卡
bash examples/qwen_gui_moe/scripts/train_moe_conservative.sh 2 4
```

### 3. `train_moe_conservative.slurm` - Slurm 批处理脚本

用于在 Slurm 集群上提交训练任务。

```bash
sbatch examples/qwen_gui_moe/scripts/train_moe_conservative.slurm
```

## 使用流程

### 步骤 1: 验证配置

```bash
cd /lus/lfs1aip2/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1
python examples/qwen_gui_moe/scripts/validate_config.py
```

预期输出：
```
✅ actor_rollout_ref.actor.kl_loss_coef: 0.1
✅ actor_rollout_ref.actor.optim.lr: 1e-05
✅ actor_rollout_ref.actor.grad_clip: 0.5
✅ actor_rollout_ref.actor.clip_ratio: 0.1
✅ actor_rollout_ref.model.moe.balance_weight: 0.2
✅ actor_rollout_ref.model.moe.z_loss_weight: 0.01
✅ actor_rollout_ref.model.moe.top_k: 2
✅ trainer.total_epochs: 10

✅ 所有配置验证通过！
```

### 步骤 2: 启动训练

#### 方式 A: 使用 Slurm (推荐)

```bash
# 直接提交
sbatch examples/qwen_gui_moe/scripts/train_moe_conservative.slurm

# 监控任务
squeue -u $USER

# 查看日志
tail -f logs/train_moe_<JOB_ID>.out
```

#### 方式 B: 手动启动

```bash
# 在交互节点上运行
bash examples/qwen_gui_moe/scripts/train_moe_conservative.sh 1 8
```

### 步骤 3: 监控训练

```bash
# Wandb 监控
# https://wandb.ai/<USER>/gui_traj_grpo_moe

# 关键指标
# - actor/ppo_kl: 应该 < 0.05
# - critic/score/std: 应该 > 0.05 (不再是 0.00)
# - critic/advantages/std: 应该 > 0.1
# - val-core/gui_traj_action_match/task_acc: 应该上升
```

## 期望结果

### 训练稳定性指标

| 指标 | 崩塌训练 | 修复后训练 |
|------|---------|-----------|
| 格式错误率 | 10.6% | < 5% |
| Click 过拟合 | +32.4% | < +10% |
| 奖励方差 | ~0.00 | > 0.05 |
| 任务成功率 | 8.8% | > 10% |

### MoE 特定指标

| 指标 | 目标值 |
|------|--------|
| 专家利用率 | 每个专家 20-30% |
| 路由熵 | > 1.0 |
| 专家输出多样性 | > 0.1 |

## 故障排查

### 问题 1: 配置验证失败

```
❌ actor_rollout_ref.actor.kl_loss_coef:
   期望: 0.1
   实际: 0.0001
```

**解决方案**: 确认使用的是最新的 `traj_grpo_moe.yaml` 配置文件。

### 问题 2: 训练中 actor/ppo_kl 过高

```
actor/ppo_kl: 0.15  # 应该 < 0.05
```

**解决方案**:
1. 进一步降低学习率: `lr: 1e-5` → `5e-6`
2. 增加 KL 系数: `kl_loss_coef: 0.1` → `0.2`

### 问题 3: 奖励方差仍然为零

```
critic/score/std: 0.00
```

**解决方案**: 这说明奖励函数需要改进。参考 `docs/training_collapse_solutions.md` 中的"阶段 2B"。

## 相关文档

- `docs/training_collapse_analysis.md` - 训练崩塌分析
- `docs/training_collapse_solutions.md` - 所有解决方案
- `docs/diagnosis_summary_20260210.md` - 诊断总结

## 更新历史

- **2026-02-10**: 创建修复脚本
  - 添加配置验证脚本
  - 添加训练启动脚本
  - 添加 Slurm 批处理脚本
  - 使用保守配置 (kl_loss_coef=0.1, lr=1e-5, etc.)
