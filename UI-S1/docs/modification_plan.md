# GUI Agent 训练崩塌问题修改方案

**日期**: 2026-02-10
**基于分析**: `training_collapse_analysis.md`

---

## 1. 概述

本文档详细描述了针对 GUI Agent 训练崩塌问题的修改方案。根据分析报告，主要问题包括：
1. 奖励信号稀疏且二元化
2. 训练数据分布不平衡
3. KL 散度惩罚不足
4. 序列误差累积

---

## 2. 修改方案概览

| 优先级 | 修改项 | 复杂度 | 预期效果 | 文件 |
|--------|--------|--------|----------|------|
| **P0** | 改进奖励函数 | 中 | 高 | `verl/utils/reward_score/gui_traj.py` |
| **P0** | 调整训练参数 | 低 | 高 | `verl/trainer/config/ppo_trainer.yaml` |
| **P1** | 数据重采样机制 | 中 | 中 | `verl/utils/dataset/rl_dataset.py` |
| **P1** | 添加监控警告 | 低 | 中 | `verl/trainer/ppo/dapo_ray_trainer.py` |
| **P2** | Early Stopping | 中 | 中 | `verl/trainer/ppo/dapo_ray_trainer.py` |
| **P3** | 辅助损失函数 | 高 | 中 | `verl/workers/actor/dp_actor.py` |

---

## 3. 详细修改方案

### 3.1 改进奖励函数 (P0)

**问题**: 当前奖励函数只在完全匹配时给予奖励，导致信号稀疏。

**当前实现** (`gui_traj.py:16-61`):
```python
def gui_action_match_compute_score(solution_str, ground_truth, extra_info=None):
    ...
    # 只在完全匹配时给予奖励
    step_reward = 1/num_steps if extract_match else 0.0
    return {
        "score": format_score * 0.1 + action_score * 0.9,
        "step_reward": step_reward
    }
```

**修改方案**: 创建新的改进版本 `gui_traj_enhanced.py`

```python
# 新文件: verl/utils/reward_score/gui_traj_enhanced.py

ACTION_IMPORTANCE = {
    'open': 2.0,       # 任务入口，最重要
    'type': 1.8,       # 输入操作，高优先级
    'system_button': 1.5,  # 导航操作
    'swipe': 1.2,      # 滑动操作
    'click': 1.0,      # 基础操作
    'long_press': 1.0,
    'wait': 0.8,
    'terminate': 1.5,  # 任务完成
}

def compute_enhanced_reward(solution_str, ground_truth, extra_info=None, step_idx=0, total_steps=1):
    """
    改进的奖励函数，具有以下特性：
    1. 步骤权重：早期步骤权重更高
    2. 操作类型权重：稀有操作获得更高权重
    3. 部分匹配奖励：即使在部分匹配时也给予奖励
    """
    ground_truth, num_steps = ground_truth['check_options'], ground_truth['num_steps']

    # 基础分数
    format_score = 0.0
    action_score = 0.0
    type_match = False
    extract_match = False

    try:
        result = fm.parse_response(solution_str)
        think_str = result['think']
        format_score = 1.0 if think_str and think_str.strip() else 0.0

        if 'action_content' in result:
            pred_action = result['action_content']
            gt_action = ground_truth
            action_type = gt_action.get('action', 'click')

            # 类型匹配奖励（30%）
            type_match = pred_action.get('action') == action_type
            type_reward = 0.3 if type_match else 0.0

            # 细节匹配奖励（70%）
            if type_match:
                if check_response_match(pred_action, ground_truth, width, height, ...):
                    detail_reward = 0.7
                    extract_match = True
                else:
                    # 部分匹配：坐标接近时给予部分奖励
                    detail_reward = compute_partial_match_reward(pred_action, ground_truth)
            else:
                detail_reward = 0.0

            action_score = type_reward + detail_reward
        else:
            action_score = 0.0

    except Exception as e:
        traceback.print_exc()
        action_score = 0.0

    # 步骤权重：早期步骤更重要
    step_weight = 1.0 + 0.1 * (total_steps - step_idx) / max(total_steps, 1)

    # 操作类型权重
    action_type = ground_truth.get('action', 'click')
    action_weight = ACTION_IMPORTANCE.get(action_type, 1.0)

    # 计算最终奖励
    base_reward = format_score * 0.1 + action_score * 0.9
    enhanced_reward = base_reward * step_weight * action_weight

    return {
        "score": enhanced_reward,
        "format_score": format_score,
        "type_match": type_match,
        "extract_match": extract_match,
        "step_reward": enhanced_reward / num_steps if extract_match else enhanced_reward * 0.1,
        "step_weight": step_weight,
        "action_weight": action_weight,
    }


def compute_partial_match_reward(pred_action, ground_truth, tolerance=50):
    """
    计算部分匹配奖励：当预测坐标接近真实坐标时给予部分奖励
    """
    if 'coordinate' not in pred_action or 'coordinate' not in ground_truth:
        return 0.0

    pred_coord = pred_action['coordinate']
    gt_coord = ground_truth['coordinate']

    if len(pred_coord) < 2 or len(gt_coord) < 2:
        return 0.0

    # 计算欧氏距离
    distance = ((pred_coord[0] - gt_coord[0])**2 + (pred_coord[1] - gt_coord[1])**2)**0.5

    # 距离越近，奖励越高
    if distance <= tolerance:
        return 0.5 * (1 - distance / tolerance)
    return 0.0
```

**集成方案**: 在 `verl/utils/reward_score/__init__.py` 中添加配置选项：
```python
def get_reward_function(use_enhanced=False):
    if use_enhanced:
        from .gui_traj_enhanced import compute_enhanced_reward
        return compute_enhanced_reward
    else:
        from .gui_traj import gui_action_match_compute_score
        return gui_action_match_compute_score
```

---

### 3.2 调整训练参数 (P0)

**修改文件**: `verl/trainer/config/ppo_trainer.yaml`

**当前问题参数**:
```yaml
actor_rollout_ref:
  actor:
    kl_loss_coef: 0.001  # 太低
    clip_ratio: 0.2      # 太宽松
    grad_clip: 1.0       # 裁剪不够
  optim:
    lr: 1e-6             # 可能需要调整
```

**建议修改**: 创建新的配置文件 `ppo_trainer_gui_fixed.yaml`

```yaml
# 基于默认配置，只覆盖需要修改的参数
actor_rollout_ref:
  actor:
    # GRPO 使用 KL loss 而非 KL reward
    use_kl_loss: true

    # 增加 KL 惩罚 100x
    kl_loss_coef: 0.1  # 从 0.001 -> 0.1

    # 更保守的 clip ratio
    clip_ratio: 0.1  # 从 0.2 -> 0.1

    # 更严格的梯度裁剪
    grad_clip: 0.5  # 从 1.0 -> 0.5

    # 降低学习率或使用余弦衰减
    optim:
      lr: 5e-6  # 从 1e-6 -> 5e-6
      warmup_style: cosine  # 使用余弦预热
      lr_warmup_steps_ratio: 0.1  # 10% 预热

    # 优势估计器使用 GRPO
  rollout:
    n: 4  # GRPO 需要多个采样

algorithm:
  # 使用 GRPO 优势估计
  adv_estimator: grpo

  # GRPO 配置
  norm_adv_by_std_in_grpo: true

  # 不使用 KL reward（使用 KL loss）
  use_kl_in_reward: false

# 训练器配置
trainer:
  total_epochs: 30
  test_freq: 5  # 每 5 步验证一次
  save_freq: 5  # 每 5 步保存一次
```

---

### 3.3 数据重采样机制 (P1)

**问题**: `click` 操作占比 51.7%，导致模型过拟合。

**修改文件**: `verl/utils/dataset/rl_dataset.py`

**方案**: 在 `TrajDataset` 类中添加重采样逻辑

```python
# 在 TrajDataset 类中添加
class TrajDataset(Qwen25VLNoRolloutDataset):
    def __init__(self, data_files, tokenizer, config, processor=None):
        super().__init__(data_files, tokenizer, config, processor)

        # 新增：采样权重配置
        self.action_weights = config.get("action_sample_weights", {
            'click': 0.5,      # 降低权重
            'type': 2.0,       # 提高权重
            'open': 2.0,       # 提高权重
            'system_button': 1.5,
            'swipe': 1.0,
            'wait': 1.0,
            'terminate': 1.0,
            'long_press': 1.0,
        })

        # 计算每个 episode 的采样权重
        self._compute_sample_weights()

    def _compute_sample_weights(self):
        """
        根据每个 episode 中的 action 类型分布计算采样权重
        """
        import numpy as np

        self.sample_weights = []

        for idx in range(len(self.dataframe)):
            line = self.dataframe[idx]
            steps = line.get('steps', [])

            # 统计该 episode 中各 action 类型的数量
            action_counts = defaultdict(int)
            for step in steps:
                action_type = step.get('action_content', {}).get('action', 'click')
                action_counts[action_type] += 1

            # 计算该 episode 的平均权重
            if action_counts:
                total_actions = sum(action_counts.values())
                weighted_sum = sum(
                    count * self.action_weights.get(action_type, 1.0)
                    for action_type, count in action_counts.items()
                )
                # 权重越高，采样概率越低
                weight = total_actions / (weighted_sum + 1e-6)
            else:
                weight = 1.0

            self.sample_weights.append(weight)

        self.sample_weights = np.array(self.sample_weights)

    def __getitem__(self, item):
        # 使用加权随机采样替代随机采样
        if self.index_remap:
            # 使用权重进行有放回采样
            probabilities = self.sample_weights[self.index_remap]
            probabilities = probabilities / probabilities.sum()
            item = np.random.choice(self.index_remap, p=probabilities)

        return super().__getitem__(item)
```

**配置文件更新** (`traj_grpo.yaml`):
```yaml
data:
  # 添加采样权重配置
  action_sample_weights:
    click: 0.5
    type: 2.0
    open: 2.0
    system_button: 1.5
    swipe: 1.0
    wait: 1.0
    terminate: 1.0
    long_press: 1.0
```

---

### 3.4 监控指标和警告系统 (P1)

**修改文件**: `verl/trainer/ppo/dapo_ray_trainer.py`

**方案**: 在 `fit()` 方法中添加监控和警告

```python
class RayTrajDAPOTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 新增：监控阈值配置
        self.monitor_thresholds = {
            'critic_score_std_min': 0.01,
            'advantages_mean_abs': 0.001,
            'ppo_kl_max': 0.1,
            'task_acc_decrease_patience': 3,
        }

        # 新增：历史指标跟踪
        self.metric_history = defaultdict(list)

    def _check_training_health(self, metrics, step):
        """
        检查训练健康状态，发出警告
        """
        warnings = []

        # 1. 检查奖励信号是否过于稀疏
        if 'critic/score/std' in metrics:
            score_std = metrics['critic/score/std']
            if score_std < self.monitor_thresholds['critic_score_std_min']:
                warnings.append(
                    f"⚠️ 奖励信号过于稀疏！score_std={score_std:.4f} < "
                    f"{self.monitor_thresholds['critic_score_std_min']}"
                )

        # 2. 检查优势函数是否为零
        if 'critic/advantages/mean' in metrics:
            adv_mean = metrics['critic/advantages/mean']
            if abs(adv_mean) < self.monitor_thresholds['advantages_mean_abs']:
                warnings.append(
                    f"⚠️ 优势函数接近零！advantages_mean={adv_mean:.4f}"
                )

        # 3. 检查 KL 散度是否过大
        if 'actor/ppo_kl' in metrics:
            ppo_kl = metrics['actor/ppo_kl']
            if ppo_kl > self.monitor_thresholds['ppo_kl_max']:
                warnings.append(
                    f"⚠️ 策略偏离过大！ppo_kl={ppo_kl:.4f} > "
                    f"{self.monitor_thresholds['ppo_kl_max']}"
                )

        # 4. 检查任务准确率是否连续下降
        if 'val-core/gui_traj_action_match/task_acc' in metrics:
            task_acc = metrics['val-core/gui_traj_action_match/task_acc']
            self.metric_history['task_acc'].append(task_acc)

            if len(self.metric_history['task_acc']) >= 3:
                recent = self.metric_history['task_acc'][-3:]
                if all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
                    warnings.append(
                        f"⚠️ 任务准确率连续下降！{recent}"
                    )

        # 打印警告
        for warning in warnings:
            print(f"[Step {step}] {warning}", flush=True)

        return len(warnings) == 0  # 返回是否健康

    def fit(self):
        """修改后的 fit 方法，添加健康检查"""
        # ... 原有代码 ...

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                # ... 原有训练代码 ...

                # 在日志记录后添加健康检查
                is_healthy = self._check_training_health(metrics, self.global_steps)

                # 如果训练不健康，可以采取行动（如降低学习率）
                if not is_healthy:
                    self._handle_unhealthy_training(metrics)
```

---

### 3.5 Early Stopping 机制 (P2)

**修改文件**: `verl/trainer/ppo/dapo_ray_trainer.py`

```python
class RayTrajDAPOTrainer(RayPPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Early stopping 配置
        self.early_stopping = {
            'enabled': True,
            'patience': 3,  # 容忍连续下降次数
            'min_delta': 0.001,  # 最小改善幅度
            'best_metric': -float('inf'),
            'wait_count': 0,
        }

    def _should_early_stop(self, metrics):
        """
        检查是否应该提前停止训练
        """
        if not self.early_stopping['enabled']:
            return False

        metric_key = 'val-core/gui_traj_action_match/task_acc'
        if metric_key not in metrics:
            return False

        current_metric = metrics[metric_key]
        best_metric = self.early_stopping['best_metric']

        # 检查是否有改善
        if current_metric > best_metric + self.early_stopping['min_delta']:
            self.early_stopping['best_metric'] = current_metric
            self.early_stopping['wait_count'] = 0
            return False
        else:
            self.early_stopping['wait_count'] += 1
            print(
                f"[Early Stopping] No improvement for "
                f"{self.early_stopping['wait_count']}/"
                f"{self.early_stopping['patience']} checks. "
                f"Best: {best_metric:.4f}, Current: {current_metric:.4f}"
            )
            return self.early_stopping['wait_count'] >= self.early_stopping['patience']

    def fit(self):
        """修改后的 fit 方法，添加 early stopping"""
        # ... 原有代码 ...

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                # ... 原有训练代码 ...

                # 在验证后检查是否提前停止
                if self._should_early_stop(metrics):
                    print(f"Early stopping triggered at step {self.global_steps}")
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    return
```

---

### 3.6 辅助损失函数 (P3)

**修改文件**: `verl/workers/actor/dp_actor.py`

**方案**: 添加动作类型分类损失和格式正确性损失

```python
class ActorRolloutWorker:
    def compute_auxiliary_loss(self, batch):
        """
        计算辅助损失，防止能力退化
        """
        # 1. 动作类型分类损失
        action_type_loss = self._compute_action_type_loss(batch)

        # 2. 格式正确性损失
        format_loss = self._compute_format_loss(batch)

        return {
            'action_type_loss': action_type_loss,
            'format_loss': format_loss,
        }

    def _compute_action_type_loss(self, batch):
        """
        计算动作类型分类损失，确保模型不会遗忘稀有动作
        """
        # 从 batch 中提取预测的动作类型和真实动作类型
        # 使用交叉熵损失
        pass

    def _compute_format_loss(self, batch):
        """
        计算格式正确性损失，防止格式退化
        """
        # 检查输出是否符合 JSON 格式
        # 检查是否有多余字符
        pass
```

---

## 4. 实施计划

### 阶段 1: 快速修复 (1-2天)

| 任务 | 文件 | 预期时间 |
|------|------|----------|
| 更新训练配置参数 | `ppo_trainer_gui_fixed.yaml` | 0.5天 |
| 添加监控和警告 | `dapo_ray_trainer.py` | 0.5天 |
| 实现 Early Stopping | `dapo_ray_trainer.py` | 0.5天 |

### 阶段 2: 核心改进 (3-5天)

| 任务 | 文件 | 预期时间 |
|------|------|----------|
| 实现改进的奖励函数 | `gui_traj_enhanced.py` | 1天 |
| 实现数据重采样机制 | `rl_dataset.py` | 1天 |
| 测试和调优 | - | 1-2天 |

### 阶段 3: 高级优化 (可选，5-7天)

| 任务 | 文件 | 预期时间 |
|------|------|----------|
| 实现辅助损失函数 | `dp_actor.py` | 2天 |
| 课程学习框架 | 新文件 | 2天 |
| DPO 替代方案探索 | 新文件 | 2天 |

---

## 5. 验证方案

### 5.1 指标监控

**核心指标**:
- `val-core/gui_traj_action_match/task_acc` - 任务成功率
- `critic/score/mean` 和 `critic/score/std` - 奖励分布
- `actor/ppo_kl` - KL 散度
- `critic/advantages/std` - 优势函数方差

**行为分布指标**:
- 各 action 类型的输出分布
- 格式错误率变化

### 5.2 预期改善

| 指标 | 当前 | 目标 | 改善幅度 |
|------|------|------|----------|
| Task Acc | 1.87% | >8% | +300% |
| Score Std | ~0 | >0.1 | 有效信号 |
| 格式错误率 | 10.6% | <3% | -70% |
| Action 分布偏差 | ±32% | ±10% | 更平衡 |

---

## 6. 回滚方案

如果新方案效果不佳，可以：
1. 使用原始配置文件
2. 恢复原始奖励函数
3. 禁用数据重采样

所有修改都通过配置开关控制，不需要修改核心代码。

---

## 7. 附录

### 7.1 配置文件对比

| 参数 | 原始值 | 建议值 | 变化 |
|------|--------|--------|------|
| `kl_loss_coef` | 0.001 | 0.1 | +100x |
| `clip_ratio` | 0.2 | 0.1 | -50% |
| `grad_clip` | 1.0 | 0.5 | -50% |
| `lr` | 1e-6 | 5e-6 | +5x |
| `rollout.n` | 1 | 4 | +4x |

### 7.2 文件修改清单

- [ ] `verl/utils/reward_score/gui_traj_enhanced.py` (新建)
- [ ] `verl/utils/reward_score/__init__.py` (修改)
- [ ] `verl/trainer/config/ppo_trainer_gui_fixed.yaml` (新建)
- [ ] `verl/utils/dataset/rl_dataset.py` (修改)
- [ ] `verl/trainer/ppo/dapo_ray_trainer.py` (修改)
- [ ] `examples/qwen_gui_static_grpo/config/traj_grpo_fixed.yaml` (新建)

---

*文档版本: 1.0*
*最后更新: 2026-02-10*
