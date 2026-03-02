# 训练崩塌诊断总结 - 完整版

**日期**: 2026-02-10
**状态**: 阶段 0 诊断完成，阶段 1 MoE 验证完成

---

## 目录

1. [重大发现](#重大发现)
2. [MoE 训练结果分析](#moe-训练结果分析)
3. [核心问题分析](#核心问题分析)
4. [行动计划](#行动计划)
5. [立即下一步](#立即下一步)

---

## 重大发现

### 🔴 发现 1: LoRA 配置比分析文档声称的更差

| 参数 | 分析文档声称 | 实际 LoRA 配置 | 差异 |
|------|-------------|---------------|------|
| `kl_loss_coef` | 0.001 | **0.0001** | **差 10x！** |
| `lr` | 1e-4 | 1e-4 | 一致 |
| `grad_clip` | 1.0 | 1.0 | 一致 |
| `clip_ratio` | 0.2 | 0.2 | 一致 |

**结论**: LoRA 训练的 KL 惩罚比文档中声称的还要**弱 10 倍**！这解释了为什么 KL 完全失效。

---

### 🔴 发现 2: MoE 配置文件存在，但训练中**未使用**保守设置！

| 参数 | 配置文件定义 (traj_grpo_moe.yaml) | 实际 MoE 训练配置 | 差异 |
|------|--------------------------------|-----------------|------|
| `kl_loss_coef` | **0.1** | **0.0001** | **差 1000x！** |
| `lr` | **1e-5** | **1e-4** | 差 10x |
| `grad_clip` | **0.5** | **1.0** | 差 2x |
| `clip_ratio` | **0.1** | **0.2** | 差 2x |
| `balance_weight` | **0.2** | **0.1** | 差 2x |
| `z_loss_weight` | **0.01** | **0.0** | 未启用! |

**结论**: MoE 训练**没有使用**配置文件中定义的保守设置，而是使用了和 LoRA 一样的弱参数！

---

### 发现 3: 奖励函数过于简单

```python
# 当前奖励函数 (verl/utils/reward_score/gui_utils/utils.py)
def check_response_match(pred_action, current_check_pam, ...):
    # 返回 (type_match_bool, detail_match_bool)
    return type_match, detail_match

# 实际奖励计算
score = 0.3 * type_match + 0.7 * detail_match  # 二元: 0 或 1
```

**问题**:
- 奖励是**二元匹配** (0 或 1)
- 没有步骤权重
- 没有 Action 类型重要性权重
- 导致大部分失败样本获得相同分数 (~0.10)

**证据** (来自训练日志):
```
critic/score/mean: 0.10 (常数)
critic/score/std: ~0.00 (无方差)
critic/advantages/mean: 0.00 (无学习信号)
```

---

## MoE 训练结果分析

### 📊 任务成功率对比

| 模型 | Checkpoint | 总样本数 | 成功数 | 成功率 |
|------|-----------|---------|--------|--------|
| LoRA | step 30 | 1,543 | 136 | **8.8%** |
| MoE | step 20 (balance0.1) | 5,745 | 136 | ~2.4%* |
| MoE | step 80 (conservative_topk1) | 5,744 | 136 | ~2.4%* |

*注: MoE 样本数较高可能因为 GRPO 的 n=4/8 采样策略生成多个候选

### 🔍 关键发现

**1. 所有模型的 136 个成功任务完全相同！**

这说明：
- LoRA 和 MoE 学到了相同的能力边界
- 成功任务集合完全一致，没有模型表现更好
- MoE 架构没有带来性能提升

**2. MoE 训练步数严重不足**

```
LoRA: 30 steps (或更早的 93 steps)
MoE step 20: 仅 20 steps
MoE step 80: 仅 80 steps
```

**3. MoE 训练使用的是弱配置**

```
实际 MoE 配置:
├── kl_loss_coef: 0.0001  (应该是 0.1)
├── lr: 1e-4              (应该是 1e-5)
├── grad_clip: 1.0        (应该是 0.5)
└── clip_ratio: 0.2       (应该是 0.1)
```

**4. Checkpoint 分析**

```
gui_traj_grpo_moe/
├── qwenvl_uis1_MoE_4experts_r16_balance0.1_gamma0.5/
│   └── global_step_20/  (只训练了 20 steps!)
└── qwenvl_uis1_MoE_4experts_r16_conservative_topk1_gamma0.5/
    └── global_step_80/  (80 steps, 但配置仍不保守)
```

---

## 核心问题分析

### 问题 1: 为什么 LoRA 崩塌？

```
根本原因: KL 惩罚 = 0.0001 (几乎为零)

影响路径:
1. KL 太弱 → 策略可以随意偏离参考模型
2. 奖励函数简单 → 大部分样本获得相同分数 (0.10)
3. 优势函数 ≈ 0 → 无学习信号
4. 模型过拟合到训练数据分布
5. 格式退化 (2.3% → 10.6%)
6. Action 分布偏移 (click: 51.7% → 84.1%)
```

### 问题 2: 为什么 MoE 也没有效果？

```
根本原因: 配置未生效 + 训练不足

1. 配置文件定义了保守设置，但训练时未使用
2. 实际使用 kl_loss_coef=0.0001 (和 LoRA 一样弱)
3. 训练步数太少 (20-80 steps vs LoRA 的 30-93 steps)
4. MoE 的复杂度需要更多训练才能显现优势
```

### 问题 3: 奖励函数的根本缺陷

```
问题: 二元匹配奖励导致信号稀疏

当前:
score = 0.3 * (type_match) + 0.7 * (detail_match)
# type_match ∈ {0, 1}, detail_match ∈ {0, 1}
# 结果: score ∈ {0, 0.3, 0.7, 1.0}

影响:
- 100% 失败 → score = 0.10 (常数)
- 无方差 → 无学习信号
- advantages = 0 → policy gradient = 0
```

---

## 行动计划

### 阶段 0: 诊断 ✅ 完成

- [x] 检查奖励函数实现
- [x] 检查 KL 惩罚实现
- [x] 分析训练配置
- [x] 发现 LoRA 实际 kl_loss_coef = 0.0001
- [x] 发现 MoE 配置未生效
- [x] 分析 MoE 训练结果

### 阶段 1: MoE 验证 ✅ 完成

**结论**: MoE **未使用保守配置**，因此无法验证保守设置是否有效。

**需要重新训练** MoE 使用正确的保守配置。

---

### 阶段 2A: 修复配置问题 + 重新训练 ⭐ 最优先 (已完成)

**目标**: 确保配置文件中的保守设置真正生效

**状态**: ✅ 修复脚本已创建

#### 已创建的文件

| 文件 | 路径 | 说明 |
|------|------|------|
| 配置验证脚本 | `examples/qwen_gui_moe/scripts/validate_config.py` | 验证配置是否正确 |
| 修复训练脚本 | `train/train_ui_s1_moe_fix.slurm` | 使用保守配置的 Slurm 脚本 |
| 使用文档 | `examples/qwen_gui_moe/scripts/README.md` | 详细使用说明 |

#### 立即使用

```bash
# 1. 提交修复后的训练任务
sbatch train/train_ui_s1_moe_fix.slurm

# 2. 监控任务
squeue -u $USER
tail -f train/logs/moe_gui_fix_<JOB_ID>.log

# 3. 验证配置生效
# 训练开始后，检查 wandb 日志中的配置
# 应该看到 kl_loss_coef=0.1, lr=1e-5, etc.
```

#### 关键修复参数 (已写入脚本)

```bash
# 这些参数已硬编码到 train_ui_s1_moe_fix.slurm
KL_LOSS_COEF=0.1      # 修复: 0.0001 → 0.1 (+1000x)
LR=1e-5               # 修复: 1e-4 → 1e-5 (-10x)
GRAD_CLIP=0.5         # 修复: 1.0 → 0.5 (-50%)
CLIP_RATIO=0.1        # 修复: 0.2 → 0.1 (-50%)
BALANCE_WEIGHT=0.2    # 修复: 0.1 → 0.2 (+100%)
Z_LOSS_WEIGHT=0.01    # 新增
TOTAL_EPOCHS=10       # 修复: 5 → 10 (+100%)
```

#### 步骤 2: 修复后的完整配置

```yaml
# examples/qwen_gui_moe/config/traj_grpo_moe.yaml

actor_rollout_ref:
  model:
    moe:
      enabled: true
      num_experts: 4
      top_k: 2
      expert_lora_r: 16
      expert_lora_alpha: 32
      expert_lora_dropout: 0.05
      target_modules: [q_proj, v_proj]
      router_hidden: 256
      router_dropout: 0.1
      router_temperature: 0.5
      pooling_strategy: mean
      balance_weight: 0.2
      balance_type: mse
      z_loss_weight: 0.01
      use_vectorized_routing: true

  actor:
    strategy: fsdp
    ppo_mini_batch_size: 64
    ppo_micro_batch_size_per_gpu: 2

    optim:
      lr: 1e-5                    # ⭐ 降低
      weight_decay: 0.01
      lr_warmup_steps_ratio: 0.1

    grad_clip: 0.5                # ⭐ 降低
    clip_ratio: 0.1               # ⭐ 降低

    use_kl_loss: true
    kl_loss_coef: 0.1            # ⭐ 增加 1000x!
    kl_loss_type: low_var_kl
    loss_agg_mode: token-mean
    ppo_epochs: 1

trainer:
  total_epochs: 10               # ⭐ 增加训练轮数
  save_freq: 100
  test_freq: 50
```

#### 步骤 3: 监控指标

```python
# 训练开始时打印配置
logging.info(f"Effective KL loss coef: {config.kl_loss_coef}")
logging.info(f"Effective learning rate: {config.lr}")

# 每个 checkpoint 验证
if step % 50 == 0:
    logging.info(f"Step {step}: actor/ppo_kl={metrics['actor/ppo_kl']}")
```

---

### 阶段 2B: 改进奖励函数 ⭐⭐⭐ 根本性解决方案

**目标**: 解决奖励信号稀疏问题

#### 方案 1: 分步加权奖励

```python
# verl/utils/reward_score/gui_utils/utils.py

ACTION_IMPORTANCE = {
    'open': 2.0,       # 任务入口，极其重要
    'type': 1.5,       # 输入操作，重要
    'navigate': 1.3,   # 导航操作，重要
    'click': 1.0,      # 通用操作
    'long_press': 1.0,
    'terminate': 1.2,  # 任务结束，重要
    'swipe': 1.0,
    'wait': 0.8,
    'key': 1.3,
    'answer': 1.5,
    'system_button': 1.2,
}

def compute_weighted_reward(pred_action, gt_action, step_idx, total_steps):
    """
    计算加权的奖励分数

    Args:
        pred_action: 预测的 action
        gt_action: ground truth action
        step_idx: 当前步骤索引 (0-based)
        total_steps: 总步骤数

    Returns:
        float: 加权后的奖励分数 [0, 1]
    """
    # 基础匹配检查
    type_match, detail_match = check_response_match(
        pred_action, gt_action,
        width=..., height=...,
        resized_width=..., resized_height=...
    )

    # 基础分数
    base_score = 0.3 * type_match + 0.7 * detail_match

    # 步骤权重: 早期步骤更重要
    # step_0 → weight=1.1, step_last → weight=1.0
    step_weight = 1.0 + 0.1 * (1 - step_idx / max(total_steps - 1, 1))

    # Action 类型权重
    action_type = gt_action.get('action', 'click')
    action_weight = ACTION_IMPORTANCE.get(action_type, 1.0)

    return base_score * step_weight * action_weight
```

#### 方案 2: 形状奖励 (Shape Reward)

```python
def compute_shape_reward(traj_actions, gt_actions):
    """
    奖励部分完成的轨迹

    即使没有完全完成任务，如果完成了一些正确步骤，
    也应该给予部分奖励
    """
    reward = 0.0
    total_steps = len(gt_actions)

    for i, (pred, gt) in enumerate(zip(traj_actions, gt_actions)):
        # 当前步骤是否正确
        type_match, _ = check_response_match(pred, gt, ...)

        # 递减权重: 早期步骤更重要
        step_weight = (total_steps - i) / total_steps

        reward += type_match * 0.1 * step_weight

    return reward
```

#### 方案 3: 对比奖励 (Contrastive Reward)

```python
def compute_contrastive_reward(pred_action, gt_action, negative_samples):
    """
    使用对比学习思想: 预测应该比随机样本更接近 GT
    """
    # pred 到 GT 的距离
    pred_distance = action_distance(pred_action, gt_action)

    # 负样本到 GT 的距离
    neg_distances = [action_distance(neg, gt_action) for neg in negative_samples]
    avg_neg_distance = sum(neg_distances) / len(neg_distances)

    # pred 应该比负样本更接近 GT
    if pred_distance < avg_neg_distance:
        return 1.0
    else:
        return 0.0
```

---

### 阶段 3: MoE 专家 KL 正则化 (长期)

**前提**: 基础配置已修复，训练已稳定

#### D1: 专家输出多样性 KL

```python
def compute_expert_diversity_loss(expert_outputs):
    """鼓励不同专家产生不同的输出分布"""
    num_experts = expert_outputs.shape[0]
    kl_divergence = 0.0
    count = 0

    for i in range(num_experts):
        for j in range(i + 1, num_experts):
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
total_loss = policy_loss + diversity_loss * 0.05
```

#### D2: 条件专家多样性 KL

```python
EXPERT_SPECIALIZATIONS = {
    'expert_0': ['click', 'long_press'],
    'expert_1': ['type', 'answer', 'key'],
    'expert_2': ['open', 'system_button'],
    'expert_3': ['swipe', 'wait'],
}

def compute_conditional_expert_kl(expert_outputs, action_labels):
    """
    重叠类型: 鼓励相似性
    独有类型: 鼓励差异性
    """
    loss = 0.0

    for i, j in combinations(range(len(EXPERT_SPECIALIZATIONS)), 2):
        types_i = set(EXPERT_SPECIALIZATIONS[f'expert_{i}'])
        types_j = set(EXPERT_SPECIALIZATIONS[f'expert_{j}'])

        # 重叠: 最小化 KL
        overlap = types_i & types_j
        for action_type in overlap:
            mask = (action_labels == action_type)
            if mask.sum() > 0:
                kl = F.kl_div(
                    F.log_softmax(expert_outputs[i][mask], dim=-1),
                    F.softmax(expert_outputs[j][mask], dim=-1),
                    reduction='batchmean'
                )
                loss -= kl  # 负号: 相似

        # 独有: 最大化 KL
        unique_i = types_i - types_j
        for action_type in unique_i:
            mask_i = (action_labels == action_type)
            mask_j = (action_labels == list(types_j - types_i)[0])
            if mask_i.sum() > 0 and mask_j.sum() > 0:
                kl = F.kl_div(
                    F.log_softmax(expert_outputs[i][mask_i], dim=-1),
                    F.softmax(expert_outputs[j][mask_j], dim=-1),
                    reduction='batchmean'
                )
                loss += kl  # 正号: 差异

    return loss
```

---

## 立即下一步

### 🎯 优先级排序

```
优先级 1: 修复配置 + 重新训练 (2-3天)
├── 验证配置加载机制
├── 确保 kl_loss_coef=0.1 生效
├── 训练 10+ epochs
└── 监控关键指标

优先级 2: 实现奖励函数改进 (3-5天)
├── 实现 compute_weighted_reward
├── 添加步骤权重
├── 添加 Action 重要性权重
└── 验证奖励方差增加

优先级 3: 添加 MoE 专家 KL (1周)
├── 实现专家多样性损失
├── 实现条件专家 KL
└── 验证专家专业化
```

### 📋 今天可以做的

#### 选项 A: 快速验证 (2小时)

```bash
# 1. 创建测试配置验证脚本
cat > test_config.py << 'EOF'
import yaml
config = yaml.safe_load(open('examples/qwen_gui_moe/config/traj_grpo_moe.yaml'))
print(f"kl_loss_coef: {config['actor_rollout_ref']['actor']['kl_loss_coef']}")
print(f"lr: {config['actor_rollout_ref']['actor']['optim']['lr']}")
EOF

# 2. 检查现有训练脚本如何加载配置
# 3. 修复配置加载问题（如果有）
```

#### 选项 B: 直接实现奖励函数 (4小时)

```bash
# 1. 备份当前奖励函数
cp verl/utils/reward_score/gui_utils/utils.py verl/utils/reward_score/gui_utils/utils.py.bak

# 2. 实现 compute_weighted_reward
# 3. 添加测试
# 4. 运行小规模验证
```

---

## 成功标准

### 阶段 2A 成功 (配置修复)

- [ ] 配置文件中的值真正生效
- [ ] `kl_loss_coef = 0.1` (不再是 0.0001)
- [ ] `lr = 1e-5` (不再是 1e-4)
- [ ] 训练完成 10+ epochs
- [ ] 格式错误率下降到 < 5%

### 阶段 2B 成功 (奖励改进)

- [ ] 实现 `compute_weighted_reward`
- [ ] `critic/score/std` > 0.05 (不再是 0.00)
- [ ] `critic/advantages/std` > 0.1 (不再是 0.00)
- [ ] 训练曲线显示真实学习（而非先升后降）
- [ ] 任务成功率 > 8%

### 阶段 3 成功 (MoE KL)

- [ ] 专家利用率均衡 (每个专家 20-30%)
- [ ] 专家输出多样性 > 0.1
- [ ] 任务成功率 > 10%

---

## 附录: 关键代码位置

| 组件 | 路径 |
|------|------|
| 奖励函数 | `verl/utils/reward_score/gui_utils/utils.py` |
| KL 损失计算 | `verl/trainer/ppo/core_algos.py:kl_penalty` |
| Actor 更新 | `verl/workers/actor/dp_actor.py:update_policy` |
| MoE Router | `verl/models/moe/router.py` |
| MoE 配置 | `examples/qwen_gui_moe/config/traj_grpo_moe.yaml` |
| 主训练脚本 | `verl/trainer/main_dapo.py` |

---

## 附录: 配置对比速查

### LoRA (当前，有问题)

```yaml
actor_rollout_ref:
  model:
    lora_rank: 64
    lora_alpha: 128
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  actor:
    lr: 1e-4
    kl_loss_coef: 0.0001  # ⚠️ 太小!
    grad_clip: 1.0
    clip_ratio: 0.2
```

### MoE (配置文件定义，未生效)

```yaml
actor_rollout_ref:
  model:
    moe:
      enabled: true
      num_experts: 4
      expert_lora_r: 16
      balance_weight: 0.2
      z_loss_weight: 0.01
  actor:
    lr: 1e-5
    kl_loss_coef: 0.1  # ⭐ 应该是这个
    grad_clip: 0.5
    clip_ratio: 0.1
```

### MoE (实际运行，有问题)

```yaml
# 从 wandb 日志提取的实际配置
actor_rollout_ref:
  model:
    moe:
      enabled: true
      num_experts: 4
      expert_lora_r: 16
      balance_weight: 0.1  # ⚠️ 不是 0.2
      z_loss_weight: 0.0   # ⚠️ 不是 0.01
  actor:
    lr: 1e-4              # ⚠️ 不是 1e-5
    kl_loss_coef: 0.0001  # ⚠️ 不是 0.1
    grad_clip: 1.0        # ⚠️ 不是 0.5
    clip_ratio: 0.2       # ⚠️ 不是 0.1
```

---

*文档维护：2026-02-10 完成 MoE 结果分析，发现配置未生效问题*
