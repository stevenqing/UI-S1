# GUI Agent 训练崩塌解决方案

**文档版本**: v1.0
**更新日期**: 2026-02-10
**状态**: 设计阶段

---

## 目录

1. [问题背景](#1-问题背景)
2. [已有分析总结](#2-已有分析总结)
3. [方案分类](#3-方案分类)
4. [详细方案设计](#4-详细方案设计)
5. [实施优先级](#5-实施优先级)
6. [评估指标](#6-评估指标)

---

## 1. 问题背景

### 1.1 核心问题

GUI Agent 训练过程中观察到典型的"先上后下"崩塌模式：

```
Performance
    ^
    |      ╭──╮
    |     ╱    ╲
    |    ╱      ╲
    |   ╱        ╲___________
    |  ╱
    |─────────────────────────> Steps
       5   10   15   20   25   30
```

### 1.2 关键证据

| 证据类型 | 数据 | 说明 |
|---------|------|------|
| 格式退化 | 2.3% → 10.6% | 双括号错误增长 360% |
| Click 过拟合 | 51.7% → 84.1% | Action 分布严重偏移 |
| Terminate 退化 | 15.3% → 0.3% | 关键能力丧失 |
| 奖励常数化 | score=0.10 (恒定) | 无学习信号 |
| 优势为零 | advantages=0.00 | 无法改进策略 |

---

## 2. 已有分析总结

### 2.1 崩塌原因排序

| 排序 | 原因 | 影响程度 | 可修复性 |
|------|------|---------|---------|
| **1** | 奖励信号稀疏 | 极高 | 需重新设计 |
| **2** | 训练数据不平衡 | 高 | 数据增强 |
| **3** | KL 惩罚不足 | 高 | 调参 |
| **4** | 序列误差累积 | 中 | 算法改进 |
| **5** | MoE 路由不稳定 | 中 | 训练技巧 |

### 2.2 MoE 特有问题

在 MoE (Mixture of Experts) 训练中观察到：
- 专家退化到相似行为（多个专家都只学 click）
- 输出崩塌（生成无效 action: "appershatch", "click離れ"）
- 路由集中（所有样本都路由到同一专家）

---

## 3. 方案分类

```
解决方案
├── A. 参数调整（快速修复）
├── B. 数据层面改进
├── C. 奖励函数重设计
├── D. MoE 专家正则化 ⭐ 新增
│   ├── D1. 专家输出多样性 KL
│   ├── D2. 条件专家多样性 KL
│   ├── D3. Router-Expert 一致性
│   └── D4. LoRA 权重正交化
└── E. 算法架构改进
```

---

## 4. 详细方案设计

---

## A. 参数调整（快速修复）

### A1. 增加全局 KL 惩罚

**问题**: `kl_loss_coef: 0.001` 太低，允许策略大幅偏离参考模型

**解决方案**:
```yaml
# 当前配置
kl_loss_coef: 0.001

# 建议配置
kl_loss_coef: 0.1  # 增加 100 倍
```

**预期效果**: 限制策略偏离，保持预训练模型的泛化能力

---

### A2. 降低学习率

**问题**: `lr: 1e-4` 过高，可能导致训练不稳定

**解决方案**:
```yaml
# 当前配置
lr: 1e-4

# 建议配置
lr: 1e-5  # 降低 10 倍
# 或使用 cosine decay
lr_scheduler: cosine
warmup_steps_ratio: 0.1
```

---

### A3. 更保守的梯度裁剪

```yaml
# 当前配置
clip_ratio: 0.2
grad_clip: 1.0

# 建议配置
clip_ratio: 0.1  # 减半
grad_clip: 0.5   # 减半
```

---

### A4. Early Stopping

```python
early_stopping:
  metric: val-core/gui_traj_action_match/task_acc
  patience: 3  # 连续 3 个 checkpoint 下降
  mode: max
```

---

## B. 数据层面改进

### B1. 数据重采样

**问题**: 训练数据中 click 占 51.7%，但评估任务需要多样化操作

**解决方案**:
```python
sample_weights = {
    'click': 0.5,      # 降低权重
    'type': 2.0,       # 提高权重
    'open': 2.0,       # 提高权重
    'navigate': 2.0,   # 提高权重
    'swipe': 1.0,
    'wait': 1.0,
}
```

---

### B2. 数据增强

针对稀有操作进行增强：
- 对 `open` 和 `type` 操作进行过采样
- 构造更多以这些操作开始的训练样本
- 使用指令改写增加多样性

---

## C. 奖励函数重设计

### C1. 分步加权奖励

**问题**: 当前奖励只在 episode 结束时给予，中间步骤信号弱

**解决方案**:
```python
def compute_reward(pred_action, gt_action, step_idx, total_steps):
    # 基础分数
    type_match = 0.3 if pred_action['action'] == gt_action['action'] else 0
    detail_match = 0.7 if check_action_detail(pred_action, gt_action) else 0

    # 步骤权重（早期步骤更重要）
    step_weight = 1.0 + 0.1 * (total_steps - step_idx)

    # 操作类型权重（稀有操作加权）
    action_weight = ACTION_IMPORTANCE.get(gt_action['action'], 1.0)

    return (type_match + detail_match) * step_weight * action_weight

ACTION_IMPORTANCE = {
    'open': 2.0,      # 任务入口，极其重要
    'type': 1.5,      # 输入操作，重要
    'navigate': 1.3,  # 导航操作，重要
    'click': 1.0,     # 通用操作
    'swipe': 1.0,
    'wait': 0.8,
}
```

---

### C2. 形状奖励 (Shape Reward)

```python
# 奖励部分完成任务的轨迹
def compute_shape_reward(traj_actions, gt_actions):
    reward = 0
    for i, (pred, gt) in enumerate(zip(traj_actions, gt_actions)):
        if pred['action'] == gt['action']:
            reward += 0.1 * (1 - i / len(gt_actions))  # 递减权重
    return reward
```

---

## D. MoE 专家正则化 ⭐ 核心创新

### 背景：MoE 崩塌问题

当前 MoE 训练出现：
- 专家退化：多个专家都只学 click
- 输出崩塌：生成无效 action
- 路由集中：所有样本路由到同一专家

**核心思想**: 通过专家间的 KL 正则化，强制不同专家保持差异化和专业化

---

### D1. 专家输出多样性 KL (Expert Output Diversity)

**目标**: 鼓励不同专家对相同输入产生不同的输出分布

**实现**:
```python
def compute_expert_diversity_loss(expert_outputs):
    """
    expert_outputs: [num_experts, batch_size, vocab_size]
    返回: 标量损失（越大表示专家越相似，需要最小化）
    """
    num_experts = expert_outputs.shape[0]
    kl_divergence = 0.0
    count = 0

    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            # 计算 KL(expert_i || expert_j)
            kl = F.kl_div(
                F.log_softmax(expert_outputs[i], dim=-1),
                F.softmax(expert_outputs[j], dim=-1),
                reduction='batchmean'
            )
            kl_divergence += kl
            count += 1

    return kl_divergence / count if count > 0 else 0.0

# 使用
diversity_loss = compute_expert_diversity_loss(expert_outputs)
total_loss = policy_loss + diversity_loss * diversity_weight
```

**参数建议**: `diversity_weight: 0.01 - 0.1`

**优点**: 简单直接，防止专家退化
**缺点**: 可能过度鼓励差异，影响任务性能

---

### D2. 条件专家多样性 KL (Conditional Expert Diversity) ⭐ 推荐

**核心思想**: 让专家在它们"应该处理"的 action 类型上保持相似，在其他 action 上保持差异

**实现**:
```python
def compute_conditional_expert_kl(expert_outputs, action_type_labels, expert_specializations):
    """
    expert_specializations: 定义每个专家专长的 action 类型
    {
        'expert_0': ['click', 'long_press'],
        'expert_1': ['type', 'answer', 'key'],
        'expert_2': ['open', 'system_button'],
        'expert_3': ['swipe', 'wait'],
    }
    """
    loss = 0.0
    num_experts = len(expert_specializations)

    for i in range(num_experts):
        for j in range(i + 1, num_experts):
            types_i = set(expert_specializations[f'expert_{i}'])
            types_j = set(expert_specializations[f'expert_{j}'])

            # 重叠类型：鼓励相似性（minimize KL）
            overlap_types = types_i & types_j
            if overlap_types:
                for action_type in overlap_types:
                    mask = (action_type_labels == action_type)
                    if mask.sum() > 0:
                        kl = F.kl_div(
                            F.log_softmax(expert_outputs[i][mask], dim=-1),
                            F.softmax(expert_outputs[j][mask], dim=-1),
                            reduction='batchmean'
                        )
                        loss -= kl  # 负号：最小化 KL（相似）

            # 独有类型：鼓励差异性（maximize KL）
            unique_types_i = types_i - types_j
            unique_types_j = types_j - types_i

            for action_type in unique_types_i:
                mask_i = (action_type_labels == action_type)
                mask_j = (action_type_labels == list(unique_types_j)[0]) if unique_types_j else None
                if mask_i.sum() > 0 and mask_j is not None and mask_j.sum() > 0:
                    kl = F.kl_div(
                        F.log_softmax(expert_outputs[i][mask_i], dim=-1),
                        F.softmax(expert_outputs[j][mask_j], dim=-1),
                        reduction='batchmean'
                    )
                    loss += kl  # 正号：最大化 KL（差异）

    return loss
```

**参数建议**:
```yaml
conditional_kl_weight: 0.05
overlap_kl_sign: -1  # 重叠类型最小化 KL
unique_kl_sign: 1    # 独有类型最大化 KL
```

**优点**:
- 结合领域知识（action 类型）
- 平衡专家差异化和任务性能
- 理论上更合理

**缺点**: 实现复杂，需要定义专家专长

---

### D3. Router-Expert 一致性 KL

**目标**: 确保 router 的分配与专家的实际能力一致

**实现**:
```python
def compute_router_expert_consistency_loss(router_weights, expert_outputs, gt_actions):
    """
    router_weights: [batch_size, num_experts] - router 的软分配
    expert_outputs: [num_experts, batch_size, action_dim]
    gt_actions: [batch_size] - ground truth actions
    """
    batch_size = router_weights.shape[0]
    loss = 0.0

    for i in range(batch_size):
        # Router 的软分配
        routing = router_weights[i]  # [num_experts]

        # 每个 expert 在这个样本上的输出质量
        expert_qualities = []
        for exp_id in range(len(expert_outputs)):
            expert_output = expert_outputs[exp_id][i]
            # 计算该 expert 输出与 ground truth 的匹配度
            quality = compute_action_match_score(expert_output, gt_actions[i])
            expert_qualities.append(quality)

        expert_qualities = torch.tensor(expert_qualities)

        # KL 散度：router 应该倾向于分配给表现好的 expert
        target_routing = F.softmax(expert_qualities / temperature, dim=0)
        loss += F.kl_div(
            F.log_softmax(routing, dim=0),
            target_routing,
            reduction='batchmean'
        )

    return loss / batch_size

def compute_action_match_score(expert_output, gt_action):
    """计算 expert 输出与 gt 的匹配分数"""
    # 根据具体任务实现
    # 例如：action type 正确性 + 坐标相似性
    return match_score
```

**参数建议**:
```yaml
router_consistency_weight: 0.02
temperature: 1.0  # 软化 target distribution
```

**优点**: 使 router 学习到更好的分配策略
**缺点**: 需要准确计算 expert 输出质量

---

### D4. LoRA 权重正交化

**目标**: 在参数层面鼓励不同 LoRA 专家的权重正交

**实现**:
```python
def compute_lora_orthogonality_loss(lora_weights_dict):
    """
    lora_weights_dict: {
        'expert_0': {
            'q_proj': (lora_A, lora_B),
            'v_proj': (lora_A, lora_B),
            ...
        },
        ...
    }
    """
    loss = 0.0
    lora_matrices = []

    # 收集所有 LoRA 的有效权重 (B @ A)
    for expert_name, expert_dict in lora_weights_dict.items():
        for module_name, (lora_A, lora_B) in expert_dict.items():
            # LoRA 的有效权重是 B @ A
            lora_weight = lora_B @ lora_A
            lora_matrices.append(lora_weight.flatten())

    # 计算两两之间的余弦相似度，最小化
    count = 0
    for i in range(len(lora_matrices)):
        for j in range(i + 1, len(lora_matrices)):
            cos_sim = F.cosine_similarity(
                lora_matrices[i].unsqueeze(0),
                lora_matrices[j].unsqueeze(0)
            )
            loss += cos_sim  # 最小化相似度 = 最大化正交性
            count += 1

    return loss / count if count > 0 else 0.0
```

**参数建议**:
```yaml
lora_orthogonality_weight: 0.001  # 通常较小
```

**优点**: 在参数层面直接约束，不依赖输出
**缺点**: 可能过于严格，限制表达能力

---

### D5. 综合方案：层级化 KL 正则 ⭐ 最推荐

**组合多个 KL 损失**:

```python
class MoETrainerWithKLMetrics:
    def __init__(self, config):
        # KL 权重配置
        self.kl_weights = {
            'global_policy': config.kl_loss_coef,        # 全局策略 KL（原有）
            'expert_diversity': config.diversity_weight,    # 专家输出多样性
            'router_consistency': config.router_kl_weight,  # Router 一致性
            'lora_orthogonality': config.ortho_weight,      # 参数正交性
            'load_balance': config.balance_weight,          # Load balance（原有）
        }

    def compute_moe_loss(self, policy_loss, expert_outputs, router_weights,
                         lora_weights, ref_policy, gt_actions):
        """
        综合计算所有损失
        """
        losses = {}
        total_loss = policy_loss

        # 1. 全局策略 KL（原有）
        if ref_policy is not None:
            policy_kl = compute_policy_kl(policy, ref_policy)
            total_loss += policy_kl * self.kl_weights['global_policy']
            losses['policy_kl'] = policy_kl

        # 2. 专家输出多样性
        diversity_loss = compute_expert_diversity_loss(expert_outputs)
        total_loss += diversity_loss * self.kl_weights['expert_diversity']
        losses['expert_diversity_loss'] = diversity_loss

        # 3. Router-Expert 一致性
        if router_weights is not None and gt_actions is not None:
            router_loss = compute_router_expert_consistency_loss(
                router_weights, expert_outputs, gt_actions
            )
            total_loss += router_loss * self.kl_weights['router_consistency']
            losses['router_consistency_loss'] = router_loss

        # 4. LoRA 权重正交性
        if lora_weights is not None:
            ortho_loss = compute_lora_orthogonality_loss(lora_weights)
            total_loss += ortho_loss * self.kl_weights['lora_orthogonality']
            losses['lora_orthogonality_loss'] = ortho_loss

        # 5. Load balance（原有）
        balance_loss = compute_load_balance_loss(router_weights)
        total_loss += balance_loss * self.kl_weights['load_balance']
        losses['load_balance_loss'] = balance_loss

        return total_loss, losses
```

**推荐配置**:
```yaml
# MoE KL 正则化权重
kl_loss_coef: 0.1           # 全局策略 KL（提高）
diversity_weight: 0.05      # 专家多样性
router_kl_weight: 0.02      # Router 一致性
ortho_weight: 0.001         # 参数正交性（很小）
balance_weight: 0.1         # Load balance
```

**监控指标**:
```python
# WandB 指标
wandb.log({
    'loss/expert_diversity': diversity_loss,
    'loss/router_consistency': router_loss,
    'loss/lora_orthogonality': ortho_loss,
    'loss/load_balance': balance_loss,

    # 专家利用率
    'moe/expert_0_utilization': (router_weights[:, 0] > 0.5).float().mean(),
    'moe/expert_1_utilization': (router_weights[:, 1] > 0.5).float().mean(),
    'moe/expert_2_utilization': (router_weights[:, 2] > 0.5).float().mean(),
    'moe/expert_3_utilization': (router_weights[:, 3] > 0.5).float().mean(),

    # 专家间平均 KL（越大越差异）
    'moe/avg_expert_kl': avg_expert_kl,

    # 路由熵（越高越均匀）
    'moe/routing_entropy': routing_entropy,
})
```

---

## E. 算法架构改进

### E1. 课程学习 (Curriculum Learning)

```
Phase 1: 只训练 1-2 步任务
Phase 2: 加入 3-4 步任务
Phase 3: 加入 5+ 步任务
```

---

### E2. 切换到 DPO

- DPO 对稀疏奖励更鲁棒
- 使用对比样本构造训练信号
- 不依赖优势函数估计

---

### E3. 辅助任务

```python
loss = policy_loss +
       0.1 * action_type_classification_loss +  # 辅助：预测动作类型
       0.1 * grounding_loss                      # 辅助：UI 元素定位
```

---

## 5. 实施优先级

### Phase 1: 快速验证（1-2天）

```
优先级 1: A1 + A2 + A3 (参数调整)
├── 增加全局 KL 惩罚 (0.001 → 0.1)
├── 降低学习率 (1e-4 → 1e-5)
└── 更保守的梯度裁剪

预期：减缓崩塌，观察训练稳定性
```

### Phase 2: MoE 优化（3-5天）

```
优先级 2: D1 + D4 (MoE 基础正则)
├── D1. 专家输出多样性 KL
└── D4. LoRA 权重正交化

预期：专家开始分化，输出质量改善
```

### Phase 3: 高级 MoE（5-7天）

```
优先级 3: D2 + D3 (条件 KL + Router 一致性)
├── D2. 条件专家多样性 KL ⭐
└── D3. Router-Expert 一致性

预期：专家专业化，任务性能提升
```

### Phase 4: 系统改进（1-2周）

```
优先级 4: B + C (数据 + 奖励)
├── B1. 数据重采样
├── B2. 数据增强
└── C1. 奖励函数改进

预期：从根本上解决崩塌问题
```

### Phase 5: 架构升级（长期）

```
优先级 5: E (算法改进)
├── 课程学习
├── DPO 替代 GRPO
└── 辅助任务

预期：系统性提升
```

---

## 6. 评估指标

### 必须监控

| 指标 | 健康值 | 警告阈值 | 说明 |
|------|--------|---------|------|
| `critic/score/mean` | 有方差 | std < 0.01 | 奖励稀疏度 |
| `critic/advantages/std` | > 0.1 | → 0 | 学习信号强度 |
| `actor/ppo_kl` | < 0.05 | > 0.1 | 策略偏离度 |
| `val-core/task_acc` | 上升 | 连续下降 | 主要任务指标 |
| `loss/expert_diversity` | > 0.1 | → 0 | 专家差异度 |
| `moe/routing_entropy` | > 1.0 | < 0.5 | 路由均匀性 |

### 崩塌预警系统

```python
def check_training_health(metrics):
    warnings = []

    # 奖励稀疏警告
    if metrics['critic/score/std'] < 0.01:
        warnings.append("⚠️ 奖励信号过于稀疏！考虑调整奖励函数")

    # 优势为零警告
    if abs(metrics['critic/advantages/mean']) < 0.001:
        warnings.append("⚠️ 优势函数为零，无法学习！检查奖励计算")

    # KL 过大警告
    if metrics['actor/ppo_kl'] > 0.1:
        warnings.append("⚠️ 策略偏离过大，考虑增加 KL 惩罚")

    # 专家退化警告
    if metrics['loss/expert_diversity'] < 0.01:
        warnings.append("⚠️ 专家输出趋同，正在退化！")

    # 路由集中警告
    if metrics['moe/routing_entropy'] < 0.5:
        warnings.append("⚠️ 路由过度集中，专家利用率不均！")

    return warnings
```

---

## 附录

### A. 相关论文

1. **Switch Transformer** (Google, 2021) - MoE 负载均衡
2. **ST-MoE** (Google, 2022) - Z-Loss 防止路由崩塌
3. **Expert Diversity Loss** - 多专家系统正则化
4. **GRPO** (Group Relative Policy Optimization) - 当前训练算法
5. **DPO** (Direct Preference Optimization) - 替代方案

### B. 代码位置

| 模块 | 路径 |
|------|------|
| MoE Router | `verl/models/moe/router.py` |
| Expert LoRA | `verl/models/moe/expert_lora.py` |
| MoE Trainer | `verl/trainer/ppo/moe_dapo_trainer.py` |
| FSDP Workers | `verl/workers/fsdp_workers.py` |
| DP Actor | `verl/workers/actor/dp_actor.py` |

### C. 实验检查清单

- [ ] Phase 1: 参数调整实验
- [ ] Phase 2: MoE 基础正则实验
- [ ] Phase 3: MoE 高级正则实验
- [ ] Phase 4: 数据 + 奖励改进实验
- [ ] Phase 5: 架构升级实验

---

*文档维护：请在实施每个方案后更新结果*
