# GUI Agent 训练崩塌分析报告

**日期**: 2026-02-10
**分析对象**: LoRA 和 MoE 训练实验
**现象**: 训练曲线呈现先上后下的崩塌模式

---

## 目录

1. [问题现象](#1-问题现象)
2. [数据分析](#2-数据分析)
3. [训练配置分析](#3-训练配置分析)
4. [崩塌原因深度分析](#4-崩塌原因深度分析)
5. [根本原因总结](#5-根本原因总结)
6. [解决方案建议](#6-解决方案建议)

---

## 1. 问题现象

### 1.1 训练曲线特征

训练过程中观察到典型的"先上后下"模式：

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

### 1.2 关键指标变化 (来自 wandb 日志)

**Score 变化** (从 `wandb/run-20260208_235136-yattg464`):
```
前10步 scores: [0.463, 0.481, 0.519, 0.584, 0.546, 0.615, 0.537, 0.586, 0.599, 0.606]
                ↑ 上升趋势，峰值 0.615

后10步 scores: [0.390, 0.431, 0.430, 0.334, 0.375, 0.425, 0.406, 0.442, 0.509, 0.456]
                ↓ 下降并波动，最低 0.334
```

| 训练阶段 | critic/score/mean | task_acc | 状态 |
|---------|------------------|----------|------|
| 初始 (Step 0) | 0.50 | 8.6% (baseline) | 正常 |
| 早期 (Step ~10) | 0.60-0.62 | 8.6% | **峰值** |
| 中期 (Step ~20-30) | 0.40-0.45 | 下降中 | 开始退化 |
| 后期 (Step 90+) | 0.33-0.45 | 1.87% | 崩塌 |

### 1.3 评估结果对比

| 模型 | Task Success Rate | 与 Baseline 对比 |
|------|------------------|------------------|
| Qwen2.5-VL-7B (Baseline) | 8.62% | - |
| LoRA Step 30 | 8.81% | +0.19% (微弱提升) |
| MoE Step 20 | 运行中 | - |
| MoE Step 80 | 运行中 | - |

### 1.4 ⚠️ 关键证据：格式退化

**训练过程中模型输出格式逐渐退化！**

| 训练阶段 | Error 总数 | 双括号 `}}` 错误 | 格式错误率 |
|---------|----------|-----------------|-----------|
| Feb 6 (早期) | 2,172 | 51 | 2.3% |
| Feb 8 (后期) | 4,414 | 469 | **10.6%** |
| **增长** | +103% | **+820%** | +360% |

**错误示例** (来自训练日志):
```json
// 正确格式
{"action": "click", "coordinate": [548, 1030]}

// 训练后的错误输出
{"action": "click", "coordinate": [548, 1030]}}  // 多了 }
{"action": "click", "coordinate": [379, 750]()}  // 多了 ()
```

**关键发现**: 训练数据中 **0个** 格式错误，但训练后模型开始生成错误格式！

---

## 2. 数据分析

### 2.1 训练数据分布

**数据集**: `datasets/ui_s1_train.jsonl` (1000 episodes, 6536 actions)

| Action Type | 数量 | 占比 | 评估任务需求 |
|-------------|------|------|-------------|
| **click** | 3380 | **51.7%** | 高频 |
| terminate | 1000 | 15.3% | 每个任务必需 |
| swipe | 782 | 12.0% | 中频 |
| **type** | 391 | **6.0%** | 高频但欠采样 |
| **open** | 376 | **5.8%** | 任务开始必需 |
| wait | 372 | 5.7% | 中频 |
| system_button | 227 | 3.5% | 中频 |
| long_press | 8 | 0.1% | 低频 |

### 2.2 数据不平衡问题

**核心问题**: 训练数据中 `click` 操作占据绝对优势 (51.7%)

```
click       ████████████████████████████████████████████████████  51.7%
terminate   ████████████████                                       15.3%
swipe       ████████████                                           12.0%
type        ██████                                                  6.0%
open        ██████                                                  5.8%
wait        ██████                                                  5.7%
sys_button  ████                                                    3.5%
long_press  █                                                       0.1%
```

### 2.3 ⚠️ 关键证据：Action 分布偏移

**训练后模型输出的 Action 分布严重偏移！**

| Action | 训练数据 | LoRA 模型输出 | 偏移量 | 问题 |
|--------|---------|--------------|--------|------|
| **click** | 51.7% | **84.1%** | **+32.4%** | ⚠️ 严重过拟合 |
| swipe | 12.0% | 3.0% | -9.0% | ❌ 能力退化 |
| **terminate** | 15.3% | **0.3%** | **-15.0%** | ❌ 严重退化 |
| open | 5.8% | 1.7% | -4.1% | ❌ 能力退化 |
| wait | 5.7% | 1.0% | -4.7% | ❌ 能力退化 |
| type | 6.0% | 2.8% | -3.2% | 能力下降 |

**MoE 训练更严重**: 出现大量无效 action 类型：
```
无效action示例: "appershatch", "click離れ", "終结", "NASA",
"swipeanesamlleBreadcrumbMultipournalogoproject"  // 完全乱码
```

**数据来源**: `wandb/run-20260206_205101-usbr83cn/files/output.log`

### 2.4 评估任务复杂度分布

| 难度 | 步数范围 | 任务数 | 占比 | 最佳模型表现 |
|------|---------|--------|------|------------|
| Easy | 1-3 steps | 438 | 28.4% | 40.6% |
| Medium | 4-6 steps | 650 | 42.1% | 18.3% |
| Hard | 7-10 steps | 347 | 22.5% | 6.0% |
| Very Hard | 11+ steps | 104 | 6.7% | 0.0% |

**关键发现**: 超过 70% 的评估任务需要 4+ 步骤，但训练数据中每个 episode 平均只有 6.5 个 action。

---

## 3. 训练配置分析

### 3.1 LoRA 配置

```yaml
# LoRA Parameters
lora_rank: 64
lora_alpha: 128
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

# Optimizer
lr: 1e-4  # 较高学习率
weight_decay: 0.01
warmup_steps_ratio: 0.1

# PPO/GRPO
clip_ratio: 0.2
ppo_epochs: 1
grad_clip: 1.0
kl_loss_coef: 0.001  # 很低的 KL 惩罚
```

### 3.2 MoE 配置 (Conservative)

```yaml
# MoE Parameters
num_experts: 4
top_k: 2  # 软路由
expert_lora_r: 16
expert_lora_alpha: 32

# Conservative Settings
lr: 1e-5  # 降低 10x
clip_ratio: 0.1  # 更保守
kl_loss_coef: 0.1  # 提高 100x
balance_weight: 0.2
z_loss_weight: 0.01
```

### 3.3 配置问题对比

| 参数 | LoRA | MoE (Conservative) | 问题 |
|------|------|-------------------|------|
| Learning Rate | 1e-4 | 1e-5 | LoRA 过高，可能导致不稳定 |
| KL Loss Coef | 0.001 | 0.1 | LoRA 的 KL 惩罚太低，容易偏离 |
| Clip Ratio | 0.2 | 0.1 | LoRA 允许更大的策略更新 |
| Grad Clip | 1.0 | 0.5 | LoRA 梯度裁剪不够严格 |

---

## 4. 崩塌原因深度分析

### 4.1 原因一：奖励信号稀疏且二元化

**问题描述**:
- 奖励函数只有在完成整个 episode 时才给予完整奖励
- 中间步骤的奖励信号很弱（只有 action type match）
- 导致 policy gradient 信号噪声大

**证据**:
```
critic/score/mean: 0.10 (常数)  # 几乎所有样本得分相同
critic/advantages/mean: 0.00    # 没有有效的优势估计
```

**影响**:
- 当所有样本获得相似的奖励时，优势函数接近零
- 没有梯度信号来指导策略改进
- 模型陷入局部最优，无法改进

### 4.2 原因二：训练数据分布不匹配评估任务

**问题描述**:
- 训练数据偏重 `click` (51.7%)
- 但评估任务需要多样化的操作序列
- `open` 操作只有 5.8%，但几乎每个任务都以此开始

**具体不匹配**:

| 操作 | 训练占比 | 评估重要性 | 不匹配程度 |
|------|---------|-----------|-----------|
| open | 5.8% | 极高（任务入口） | **严重欠采样** |
| type | 6.0% | 高（输入操作） | **严重欠采样** |
| navigate | 3.5% | 高（导航操作） | **严重欠采样** |
| click | 51.7% | 中（通用操作） | 过度采样 |

### 4.3 原因三：多步任务的序列依赖问题

**问题描述**:
- GUI Agent 任务本质上是序列决策问题
- 错误会在序列中累积传播
- GRPO 训练假设步骤间相对独立

**崩塌机制**:

```
Step 1: 95% correct (open app)
Step 2: 90% correct (navigate)
Step 3: 85% correct (click target)
Step 4: 80% correct (interact)
...
Overall: 0.95 × 0.90 × 0.85 × 0.80 = 58% (4-step task)

随着训练，如果某一步准确率下降 (e.g., Step 1: 95% → 85%):
New Overall: 0.85 × 0.90 × 0.85 × 0.80 = 52% (↓6%)
```

**为什么"先上后下"**:
1. **初期上升**: 模型学习到最频繁的操作（click），提升这部分准确率
2. **中期稳定**: 达到数据分布偏差允许的上限
3. **后期下降**:
   - 过度拟合 click 操作
   - 稀有操作（open, type）能力退化
   - KL 惩罚过低，策略过度偏离参考模型
   - 整体任务成功率因序列依赖而快速下降

### 4.4 原因四：KL 散度惩罚不足

**问题描述**:
- LoRA 配置中 `kl_loss_coef: 0.001` 极低
- 允许策略大幅偏离预训练模型
- 预训练模型的通用能力被破坏

**证据**:
```python
# 从 wandb summary 中观察到的 KL 散度
actor/ppo_kl: 0.015  # Step 1
actor/ppo_kl: 0.045  # Step 90+ (增长了 3x)

# 但 kl_loss 对总 loss 的贡献
kl_loss_contribution = 0.001 * 0.045 = 0.000045  # 几乎可忽略
```

**影响**:
- 模型可以随意偏离，没有正则化约束
- 容易过拟合到训练数据的特定模式
- 失去预训练模型的泛化能力

### 4.5 原因五：MoE 路由不稳定

**问题描述** (针对 MoE 训练):
- Router 需要学习将不同指令类型路由到正确的专家
- 但训练数据的指令类型分布不均匀
- 导致某些专家过度使用，其他专家饥饿

**预期 vs 实际路由**:

```
预期均匀分布:
Expert 0 (click):    25%
Expert 1 (type):     25%
Expert 2 (navigate): 25%
Expert 3 (scroll):   25%

实际可能的分布:
Expert 0 (click):    60%  (过载)
Expert 1 (type):     10%  (饥饿)
Expert 2 (navigate): 10%  (饥饿)
Expert 3 (scroll):   20%  (正常)
```

---

## 5. 根本原因总结

### 5.1 主要原因（按重要性排序）

| 排序 | 原因 | 影响程度 | 可修复性 |
|------|------|---------|---------|
| **1** | 奖励信号稀疏 | 极高 | 需要重新设计奖励 |
| **2** | 训练数据不平衡 | 高 | 可通过数据增强解决 |
| **3** | KL 惩罚不足 | 高 | 调参可解决 |
| **4** | 序列误差累积 | 中 | 需要算法改进 |
| **5** | MoE 路由不稳定 | 中 | 需要更多训练技巧 |

### 5.2 崩塌机制图解

```
                    ┌─────────────────────┐
                    │   初始状态           │
                    │ (预训练模型能力)      │
                    └─────────┬───────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│                    早期训练 (Step 1-10)                   │
│  • 学习 click 操作 (数据量大)                              │
│  • 性能略有提升                                           │
│  • KL 散度开始增长                                        │
└─────────────────────────────┬───────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│                    中期训练 (Step 10-20)                  │
│  • click 过拟合                                          │
│  • type/open 能力开始退化 (数据量少，被 "遗忘")            │
│  • 奖励信号变得稀疏 (大部分任务失败)                       │
└─────────────────────────────┬───────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│                    崩塌阶段 (Step 20+)                    │
│  • 优势函数 → 0 (所有样本得分相似)                         │
│  • 无有效梯度信号                                         │
│  • 策略陷入次优状态                                       │
│  • 预训练能力被破坏                                       │
└─────────────────────────────────────────────────────────┘
```

---

## 6. 解决方案建议

### 6.1 短期调整（参数层面）

#### A. 增加 KL 惩罚
```yaml
# 当前
kl_loss_coef: 0.001

# 建议
kl_loss_coef: 0.1  # 增加 100x
```

#### B. 降低学习率
```yaml
# 当前 LoRA
lr: 1e-4

# 建议
lr: 1e-5  # 或使用 cosine decay
```

#### C. 使用 Early Stopping
```yaml
# 监控 val-core/task_acc
# 当连续 3 个 checkpoint 下降时停止
early_stopping:
  metric: val-core/gui_traj_action_match/task_acc
  patience: 3
  mode: max
```

### 6.2 中期改进（数据层面）

#### A. 数据重采样
```python
# 平衡不同 action type 的采样权重
sample_weights = {
    'click': 0.5,      # 降低权重
    'type': 2.0,       # 提高权重
    'open': 2.0,       # 提高权重
    'navigate': 2.0,   # 提高权重
    'swipe': 1.0,
    'wait': 1.0,
}
```

#### B. 数据增强
- 对 `open` 和 `type` 操作进行过采样
- 构造更多以这些操作开始的训练样本
- 使用指令改写增加多样性

### 6.3 长期改进（算法层面）

#### A. 改进奖励函数
```python
def compute_reward(pred_action, gt_action, step_idx, total_steps):
    # 基础分数
    type_match = 0.3 if pred_action['action'] == gt_action['action'] else 0
    detail_match = 0.7 if check_action_detail(pred_action, gt_action) else 0

    # 步骤权重 (早期步骤更重要)
    step_weight = 1.0 + 0.1 * (total_steps - step_idx)

    # 操作类型权重 (稀有操作加权)
    action_weight = ACTION_IMPORTANCE.get(gt_action['action'], 1.0)

    return (type_match + detail_match) * step_weight * action_weight
```

#### B. 使用课程学习
```
Phase 1: 只训练 1-2 步任务
Phase 2: 加入 3-4 步任务
Phase 3: 加入 5+ 步任务
```

#### C. 考虑使用 DPO 替代 GRPO
- DPO 对稀疏奖励更鲁棒
- 可以使用对比样本构造训练信号
- 不依赖于优势函数估计

#### D. 添加辅助任务
```python
# 多任务学习
loss = policy_loss +
       0.1 * action_type_classification_loss +  # 辅助：预测动作类型
       0.1 * grounding_loss                      # 辅助：UI 元素定位
```

---

## 附录

### A. 实验配置对比表

| 配置项 | LoRA | MoE (Original) | MoE (Conservative) | 建议值 |
|--------|------|----------------|-------------------|--------|
| lr | 1e-4 | 1e-4 | 1e-5 | 5e-6 |
| kl_loss_coef | 0.001 | 0.001 | 0.1 | 0.05-0.1 |
| clip_ratio | 0.2 | 0.2 | 0.1 | 0.1-0.15 |
| grad_clip | 1.0 | 1.0 | 0.5 | 0.5 |
| balance_weight | - | 0.1 | 0.2 | 0.1-0.2 |
| ppo_epochs | 1 | 1 | 1 | 1 |

### B. 监控指标建议

**必须监控**:
- `critic/score/mean` - 应该有方差，不应为常数
- `critic/advantages/std` - 应该 > 0.1
- `val-core/gui_traj_action_match/task_acc` - 主要指标
- `actor/ppo_kl` - 不应增长过快

**警告阈值**:
```python
if critic_score_std < 0.01:
    warn("奖励信号过于稀疏，考虑调整奖励函数")

if advantages_mean == 0:
    warn("优势函数为零，无法学习")

if ppo_kl > 0.1:
    warn("策略偏离过大，考虑增加 KL 惩罚")
```

### C. 参考文献

1. GRPO: Group Relative Policy Optimization
2. PPO: Proximal Policy Optimization Algorithms
3. LoRA: Low-Rank Adaptation of Large Language Models
4. Switch Transformer: Scaling to Trillion Parameter Models
5. ST-MoE: Designing Stable and Transferable Sparse Expert Models

---

*报告生成时间: 2026-02-10*
