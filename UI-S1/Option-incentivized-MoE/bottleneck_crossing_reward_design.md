# Bottleneck Crossing Reward (BCR) 设计文档

> 基于 5,461 对 success/fail 轨迹的 contrastive analysis，设计一种基于 "crossing value" 而非简单 "difficulty weight" 的 reward shaping 方案。

---

## 1. 问题回顾：为什么 f_pseudo 无法帮助 agent 跨越 bottleneck

f_pseudo 的三个致命问题：
1. **无方向性**: `f_pseudo(t) = f(s_t) - f(s_{t+1})`，沿轨迹累加 ≈ 0（望远镜求和）
2. **信号稀疏**: 54.6% 为零值（self-loop），67.2% 接近零
3. **bottleneck 定义错位**: f_net 捕捉的是图连通性（Fiedler vector），而模型的实际 failure 是坐标精度（82.1%）和 error cascade

**核心矛盾**: f_net 说"状态 A→B 是图的瓶颈"，但模型实际失败原因是"在状态 A 点错了坐标 100px"——完全不同的 bottleneck。

---

## 2. 数据分析：真正的 Bottleneck 在哪里

### 2.1 数据源

| 数据集 | 轨迹数 | 说明 |
|-------|:------:|------|
| `datasets/GUI-360/train/data/` (success) | 13,750 | 成功轨迹，mean 7.7 steps |
| `datasets/GUI-360/fail/data/` (fail) | 62,236 | 失败轨迹，mean 20.1 steps |
| 其中 paired (same request) | 5,461 | 同一任务的 success/fail 对 |

Fail 轨迹分布：excel 23,104 / ppt 20,050 / word 19,082

### 2.2 Contrastive 分叉分析 (1,000 对)

对每对 success/fail 轨迹（same task），逐步比较动作：
- **Function 不匹配** → 分叉
- **Click/drag 坐标偏差 > 80px** → 分叉

**分叉位置分布**:
```
pos=0.0 (step 1):     4.8%  ████
pos=0.1 (first 10%): 33.5%  █████████████████████████████████
pos=0.2:             24.5%  ████████████████████████
pos=0.3:             10.6%  ██████████
pos=0.4:              5.1%  █████
pos=0.5:              6.2%  ██████
pos=0.6:              2.2%  ██
pos=0.7:              2.9%  ██
pos=0.8:              3.5%  ███
pos=0.9:              0.2%
pos=1.0:              6.6%  ██████
```

**关键发现**: **58% 的分叉发生在轨迹前 20%**。早期一个错误导致整条轨迹失败。

**分叉原因**:
| 原因 | 占比 |
|------|:----:|
| 坐标偏差 >80px | **66%** |
| Function 类型不匹配 | 30% |
| 轨迹长度差异 | 4% |

### 2.3 Survival Curve

**P(fail 轨迹与 success 轨迹在位置 t 仍匹配)**:

```
pos=0.0: 49% ████████████████████████
pos=0.1: 42% █████████████████████
pos=0.2: 33% ████████████████
pos=0.3: 26% ████████████
pos=0.4: 20% █████████
pos=0.5: 20% █████████
pos=0.6: 16% ████████
pos=0.7: 14% ███████
pos=0.8: 15% ███████
pos=0.9: 14% ██████
pos=1.0:  8% ████
```

**解读**: 一半的 fail 轨迹从**第一步**就偏离了成功路径。到位置 0.3，只有 26% 还在跟随成功路径。

### 2.4 Sub-Score Completion vs 分叉位置

| 分叉位置 | Sub-score 完成度 | 含义 |
|---------|:--------------:|------|
| 0.0-0.1 (极早期) | 0.28 | 早期偏离 → 任务几乎无进展 |
| 0.2-0.3 | 0.39 | 完成了部分子目标 |
| 0.7-0.8 (晚期) | 0.55 | 完成了大部分，仅最后阶段失败 |

**因果关系清晰**: 越早偏离 → 最终完成度越低 → error cascade 越严重。

---

## 3. 从 Difficulty 到 Crossing Value：为什么要更深入

### 3.1 Simple Difficulty Weight 的问题

朴素方案: `difficulty_bonus = base_score * fail_rate(domain, action_type) * λ`

问题：
1. **不区分位置**: step 1 的 click 和 step 10 的 click 获得相同 difficulty weight，但 step 1 的 click 对轨迹成功的影响远大于 step 10
2. **不捕捉 cascade**: 难度是独立的，但 failure 是 cascading 的 —— step 1 的错误导致 step 2-N 全部失败
3. **不区分"做对了很重要"vs"做对了不重要"**: difficulty 只说明 step 有多难，不说明做对这个 step 对最终成功有多大贡献

### 3.2 Crossing Value 的核心思想

**Crossing Value ≠ 这个 step 有多难**

**Crossing Value = 做对这个 step 解锁了多少未来的成功可能性**

具体地：
- `crossing_importance(t)` = P(这个位置是 success/fail 分叉点)
- `Future Crossing Value (FCV(t))` = Σ crossing_importance(t') for all t' ≥ t

FCV 捕捉的是：**如果 agent 在位置 t 做对了，它还需要跨越多少后续的 bottleneck？**

### 3.3 FCV 曲线

```
pos=0.0: FCV=1.000  ██████████████████████████████████████████████████
pos=0.1: FCV=0.953  ███████████████████████████████████████████████
pos=0.2: FCV=0.618  ██████████████████████████████
pos=0.3: FCV=0.373  ██████████████████
pos=0.4: FCV=0.267  █████████████
pos=0.5: FCV=0.216  ██████████
pos=0.6: FCV=0.154  ███████
pos=0.7: FCV=0.132  ██████
pos=0.8: FCV=0.102  █████
pos=0.9: FCV=0.068  ███
pos=1.0: FCV=0.066  ███
```

**解读**:
- Step@0.1 的 FCV=0.95：做对这步几乎决定了整条轨迹的成败
- Step@0.5 的 FCV=0.22：只有约 22% 的未来 bottleneck 需要跨越
- Step@0.8 的 FCV=0.10：接近终点，大部分 bottleneck 已经跨过

**对比**:
- FCV approach: step@0.1 获得 **9.3x** 更多权重 vs step@0.8
- Simple difficulty: 所有步都是同等权重（无位置感知）

---

## 4. Bottleneck Crossing Reward 设计

### 4.1 总体 Reward 公式

```
total_reward(t) = base_score(t) × (1 + λ_c × FCV(t)) + λ_p × progress_delta(t)
```

| 组分 | 作用 | 信号来源 |
|------|------|---------|
| `base_score(t)` | 动作正确性 (0-1) | gui360_compute_score |
| `λ_c × FCV(t) × base_score(t)` | **Crossing bonus**: 关键位置正确动作获得放大奖励 | Contrastive 分析 |
| `λ_p × progress_delta(t)` | **Progress bonus**: 方向性信号 | Progress estimator |

### 4.2 Signal 1: Crossing Bonus (multiplicative)

```python
crossing_bonus(t) = base_score(t) × FCV(domain, action_type, normalized_pos(t)) × λ_c
```

**性质**:
- **Multiplicative**: 与 base_score 相乘 → 只有做对了才能获得 bonus
- **Position-aware**: 早期步骤获得更大的 bonus（FCV 高）
- **Domain-specific**: 可按 domain 计算独立的 FCV 曲线（excel vs ppt vs word 的 bottleneck 位置可能不同）

**效果** (λ_c = 0.5):
| 位置 | FCV | Reward 放大倍数 |
|------|:---:|:--------------:|
| 0.0 | 1.000 | 1.50x |
| 0.1 | 0.953 | 1.48x |
| 0.2 | 0.618 | 1.31x |
| 0.5 | 0.216 | 1.11x |
| 0.8 | 0.102 | 1.05x |
| 1.0 | 0.066 | 1.03x |

### 4.3 Signal 2: Progress Delta (additive)

```python
progress_delta(t) = p(s_{t+1}) - p(s_t)
progress_bonus(t) = λ_p × progress_delta(t)
```

**性质**:
- **Additive**: 独立于 base_score → 即使动作不完美，方向正确也有微小奖励
- **Directional**: 正值 = 朝成功推进，负值 = 退步
- 与 f_pseudo 的关键区别：progress 是 **有方向的**（基于 success trajectory 的进度定义），f_pseudo 是 **无方向的**（cumsum ≈ 0）

### 4.4 两个信号的交互

| 场景 | Crossing | Progress | 总体效果 |
|------|:--------:|:--------:|---------|
| 关键位置 + 做对 + 前进 | ↑↑ | ↑ | 大奖励 — agent 成功跨越 bottleneck |
| 关键位置 + 做对 + 原地 | ↑↑ | ~ | 中等奖励 — 做对了但没推进 |
| 关键位置 + 做错 + 任何方向 | 0 | ↑/↓ | 几乎无奖励 — 没跨过 bottleneck |
| 常规位置 + 做对 + 前进 | ↑ | ↑ | 正常奖励 |
| 常规位置 + 做错 | 0 | ~ | 低奖励 |

---

## 5. 实施方案

### Task 1: `scripts/build_crossing_map.py` — 构建 Crossing Value Map

**输入**:
- `datasets/GUI-360/train/data/` (success trajectories)
- `datasets/GUI-360/fail/data/` (fail trajectories)
- `datasets/GUI-360/rl_data/gui360_train.jsonl` (RL 训练数据)

**处理流程**:
1. 按 `request` 文本匹配 success/fail 轨迹对
2. 对每对：找分叉步，记录 `(domain, action_type, normalized_position)`
3. 计算 `crossing_importance(domain, action_type, pos_bucket)` → 分叉频率分布
4. 计算 `FCV(pos)` → 累积未来 crossing value
5. 遍历 gui360_train.jsonl，为每个 step 查 FCV 值
6. 输出 `{execution_id: {step_id_str: fcv_value}}`

**输出**: `outputs/crossing_map/crossing_value_map.json`

### Task 2: 修改 `verl/workers/reward_manager/f_pseudo_dapo.py`

新增参数:
- `crossing_map_path` / `crossing_lambda` (default 0.5)
- `progress_map_path` / `progress_lambda` (default 0.1)

新增 reward computation:
```python
# Crossing bonus
crossing_bonus = score * self.crossing_lambda * fcv_value

# Progress bonus
progress_bonus = self.progress_lambda * progress_value

reward += crossing_bonus + progress_bonus
```

### Task 3: `scripts/train_progress_estimator.py` — 训练进度网络

- MLP 43→256→256→1 + Sigmoid
- Success 轨迹: label = step_id / total_steps
- Fail 轨迹: label = step_id / total_steps × sub_score_completion
- Loss: MSE

### Task 4: `scripts/precompute_progress_bonus.py` — 预计算进度 bonus

- 加载 progress_net
- 遍历 transitions.jsonl
- `progress_bonus(t) = p(s_{t+1}) - p(s_t)`
- 输出: `{execution_id: {step_id_str: progress_delta}}`

### Task 5: Training Config

```yaml
reward_kwargs:
    f_pseudo_lambda: 0.0            # 关闭旧的 f_pseudo
    crossing_map_path: outputs/crossing_map/crossing_value_map.json
    crossing_lambda: 0.5
    progress_map_path: outputs/progress_bonus/progress_bonus_map.json
    progress_lambda: 0.1
```

---

## 6. 为什么这比 f_pseudo 和 simple difficulty 更好

| 维度 | f_pseudo | Simple Difficulty | **BCR (本方案)** |
|------|---------|-----------------|-----------------|
| Bottleneck 定义 | 图连通性 | per-step 失败率 | **轨迹分叉点** (因果) |
| 方向性 | 无 (cumsum≈0) | 无 | **有** (progress delta) |
| 位置感知 | 无 | 无 | **有** (FCV 曲线) |
| Cascade 建模 | 无 | 无 | **有** (FCV = 累积未来值) |
| 数据来源 | f_net eigenfunction | fail_rate 统计 | **5,461 对 contrastive** |
| 信号类型 | f(s)-f(s') | fail_rate × score | **score × FCV + progress** |

**核心区别**: BCR 回答的问题不是"这步有多难"，而是"做对这步对最终成功有多重要"。

---

## 7. 执行顺序与依赖

```
Task 1 (crossing map)  ──┐
                         ├──→ Task 5 (config) → Training
Task 2 (reward manager) ──┘
                              ↑
Task 3 (progress net) → Task 4 (progress map) ─┘
```

Task 1 和 Task 2 可并行。Task 3→4 是串行的。Task 5 依赖 1 和 4 的输出。

---

## 8. 预期效果

- **Medium trajectories (6-15 steps)**: 提升最大，这些轨迹的 bottleneck 在前 2-3 步
- **Long trajectories (16+ steps)**: 有帮助但有限，error cascade 过长仍难以恢复
- **Short trajectories (1-5 steps)**: 帮助 agent 在关键的 1-2 步上更精确
- **Coordinate precision**: crossing bonus 在早期步骤放大 reward → agent 在关键位置投入更多学习

**定量预期** (保守估计):
- Step-level SR: +2-5%
- TSR (medium): 32.79% → 38-42%
- TSR (overall): 16.21% → 19-22%
