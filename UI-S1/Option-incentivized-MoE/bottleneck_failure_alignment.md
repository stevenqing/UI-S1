# Bottleneck 与 Model Failure 对齐分析

> 核心问题：当前 f_net 定义的 bottleneck（图连通性瓶颈）与 SFT v2 / Base model 的实际 failure（坐标精度、error cascade）完全不对齐。本文档分析差异根因，提出系统性对齐方案。

---

## 目录

1. [SFT v2 / Base Model 的实际 Failure 分析](#1-sft-v2--base-model-的实际-failure-分析)
2. [f_net Bottleneck 回顾](#2-f_net-bottleneck-回顾)
3. [Failure 与 Bottleneck 的对比：为什么不对齐](#3-failure-与-bottleneck-的对比为什么不对齐)
4. [重新定义 Bottleneck：从图结构到能力差距](#4-重新定义-bottleneck从图结构到能力差距)
5. [对齐方案：三阶段实施](#5-对齐方案三阶段实施)
6. [长轨迹 (16+ steps) 专项对齐](#6-长轨迹-16-steps-专项对齐)
7. [完整 Pipeline 设计](#7-完整-pipeline-设计)
8. [实施优先级与路线图](#8-实施优先级与路线图)

---

## 1. SFT v2 / Base Model 的实际 Failure 分析

### 1.1 Step-Level 性能

| 指标 | Base Model | SFT v2 (1ep) | SFT v2 (2ep) | Paper SFT |
|------|:----------:|:------------:|:------------:|:---------:|
| Grounding | 42.47% | 70.56% | 70.77% | 82.30% |
| Action (Visual) | 18.05% | 46.90% | 49.37% | 50.08% |
| Action (A11y) | 14.53% | 17.51% | N/A | 25.78% |

### 1.2 Trajectory-Level 性能 (Semi-Online AR)

| 指标 | Base Model | SFT v2 |
|------|:----------:|:------:|
| Step-Level SR | 22.10% | 55.28% |
| **Trajectory SR (TSR)** | **1.64%** | **16.21%** |
| Avg Progress | 12.30% | 36.70% |

| 长度分桶 | Base TSR | SFT v2 TSR | SFT v2 Progress |
|---------|:--------:|:----------:|:---------------:|
| Short (1-5) | 1.64% | 15.89% | 35.68% |
| Medium (6-15) | 0.00% | 32.79% | 89.64% |
| **Long (16+)** | **0.00%** | **0.00%** | **0.00%** |

### 1.3 Failure 类型分解

| Failure 类型 | 占失败轨迹比例 | 说明 |
|-------------|:------------:|------|
| **坐标/参数错误** | **82.1%** | 模型空间定位精度不足，是 **主导 failure** |
| 函数类型错误 | 42.5% | 预测了错误的 action type (如 drag vs click) |
| 状态判断错误 | 22.6% | CONTINUE / FINISH 混淆 |

### 1.4 Step-Level 指标分解 (SFT v2)

| 指标 | 值 | 解读 |
|------|:---:|------|
| Func Match | 86.18% | 高——模型基本能判断 action 类型 |
| **Args Match** | **11.28%** | **极低——坐标精度是真正的瓶颈** |
| Status Match | 95.31% | 高——CONTINUE/FINISH 判断基本正确 |

**结论：82.1% 的 failure 是坐标精度问题，func match 86% 但 args match 仅 11%。模型知道做什么，但不知道在哪里做。**

### 1.5 Error Cascade 分析

| 指标 | 值 |
|------|:---:|
| 第一个错误的平均归一化位置 | 0.4369 (~step 1.81) |
| 错误后准确率 | 41.54% |
| 恢复率（错误后有正确步骤） | 65.96% |
| Step 位置准确率变化 | Step 1: 48.8% → Step 2: 59.9% → Step 6+: 38-42% |

### 1.6 Per-Domain 分解 (SFT v2)

| Domain | TSR | Progress | 特点 |
|--------|:---:|:--------:|------|
| PPT | 18.27% | 46.44% | 最好，UI 元素较大 |
| Word | 15.49% | 35.37% | 中等 |
| Excel | 15.42% | 30.09% | 最差，单元格密集，坐标困难 |

---

## 2. f_net Bottleneck 回顾

### 2.1 当前定义

f_net 通过近似图拉普拉斯的第二特征向量（Fiedler vector），识别 **UI 状态转移图中连通性最差的状态**：

```
f_net: state_embedding (43-dim) → scalar f-value
bottleneck_states = {s : f_value(s) < 30th percentile}
```

每个 app 约 450 个 bottleneck 状态（out of ~1,400-3,500 total）。

### 2.2 Bottleneck 状态的实际内容

| App | Bottleneck 示例 | 特征 |
|-----|----------------|------|
| Excel | 对话框（Format Cells, Excel Options）、Afrikaans 本地化 UI | 稀有 UI 配置 |
| Word | 表格编辑模式（双 Layout tab）、Developer 模式 | 非标准视图 |
| PPT | 嵌入式对象编辑、缺少 Home tab 的视图 | 嵌套状态 |

### 2.3 f_pseudo 信号质量

| 问题 | 严重程度 | 数据 |
|------|:--------:|------|
| **累积和 ≈ 0** | **致命** | f(s_start) - f(s_end) 均值 < 0.003，无方向性 |
| **54.6% 零值** | 高 | click/type 不改变 screenshot hash → 自环 |
| **长轨迹退化** | 高 | 16+ 步: 68.9% 符号一致性 < 0.3，信号为纯噪声 |
| Off-by-one Bug | 已修复 | step_id+1 对齐 |

详细数据见 [f_pseudo_effectiveness_analysis.md](f_pseudo_effectiveness_analysis.md)。

---

## 3. Failure 与 Bottleneck 的对比：为什么不对齐

### 3.1 两种 "Bottleneck" 的本质差异

| 维度 | f_net Bottleneck (图结构) | Model Failure (能力缺陷) |
|------|--------------------------|-------------------------|
| **抽象层次** | 状态级（哪些 UI 状态难到达） | 动作级（哪些 action 执行不对） |
| **根因** | 图拓扑的连通性 | 模型的视觉理解 + 坐标预测 |
| **时间尺度** | 静态（图结构不变） | 动态（随轨迹位置/长度变化） |
| **主要模式** | 对话框、非标准视图、本地化 UI | 坐标偏移 (82.1%)、error cascade |
| **长轨迹解释力** | f(start)≈f(end)，无方向性 | 误差累积导致状态偏离 |
| **可操作性** | 低（无法指导模型改进） | 高（直接指向需要提升的能力） |

### 3.2 不对齐的三个根本原因

#### 原因 1：f_net 回答的不是正确的问题

```
f_net 回答:  "哪些 UI 状态在转移图中连通性差？"
实际需要:    "模型在哪些 (状态, 动作) 对上会失败？"
```

82.1% 的失败是坐标精度问题——模型可能已经到达了正确的 UI 状态（包括 f_net 认为的 bottleneck 状态），但无法精确点击目标元素。这和图连通性完全无关。

#### 原因 2：f_pseudo 在最需要帮助的地方退化为噪声

| 轨迹长度 | TSR (SFT v2) | f_pseudo 高一致性 (≥0.8) | f_pseudo 低一致性 (<0.3) | 矛盾 |
|---------|:-----------:|:---------------------:|:---------------------:|------|
| Short (1-5) | 15.89% | 43.5% | 35.4% | 短轨迹不太需要额外信号，信号质量最好 |
| Medium (6-15) | 32.79% | 15.5% | 51.1% | 信号质量下降 |
| **Long (16+)** | **0%** | **10.0%** | **68.9%** | **最需要帮助，信号完全退化** |

#### 原因 3：Bottleneck ≠ Difficulty

f_net 的 bottleneck 是**可达性**概念——某些 UI 状态在图中距离远。但模型的 difficulty 是**执行精度**概念——在任何 UI 状态（包括常见状态）都可能坐标不准。

反例：
- Excel 普通表格视图（f_net：不是 bottleneck）→ 模型因单元格密集而点错（高 failure）
- Word 对话框（f_net：是 bottleneck）→ 大按钮反而更容易点击（低 failure）

### 3.3 f_net Bottleneck 与 MoE Router 的断连

```
当前 Pipeline:

  f_net → f_pseudo → reward shaping (辅助奖励信号)
           ↑ 完全独立 ↓
  Router → Expert LoRAs → action output (模型路由)

两者没有任何连接:
- f_pseudo 不影响 routing 决策
- Router 不感知 bottleneck 状态
- Expert 分工不基于 failure type
```

---

## 4. 重新定义 Bottleneck：从图结构到能力差距

### 4.1 新定义

```
旧: bottleneck(s) = {1 if f_value(s) < 30th percentile, 0 otherwise}
    含义: 状态 s 在转移图中连通性差 → 难以到达

新: bottleneck(s, a) = 1 - P(model correctly executes action a in state s)
    含义: 模型在状态 s 执行动作 a 的预期失败概率
```

### 4.2 新定义如何覆盖所有 failure 类型

| Failure (82.1% 坐标) | 旧定义能否捕获 | 新定义如何捕获 |
|----------------------|:------------:|--------------|
| Dense UI 中 click 坐标偏移 | ✗ | dense UI 的 state 有高 bottleneck score |
| 小按钮点击不准 | ✗ | 小目标的 (state, click) 有高 failure rate |
| Drag 的起/终点坐标错 | ✗ | drag action 的 args match 低 → 高 bottleneck |

| Failure (42.5% 函数类型) | 旧定义能否捕获 | 新定义如何捕获 |
|-------------------------|:------------:|--------------|
| Click vs drag 混淆 | ✗ | 容易混淆的 (state, action_pair) 有高 bottleneck |
| Scroll 方向错误 | ✗ | wheel/swipe 的 function match 失败率 → 高 bottleneck |

| Failure (长轨迹 TSR=0%) | 旧定义能否捕获 | 新定义如何捕获 |
|-------------------------|:------------:|--------------|
| Error cascade | ✗ (无方向性) | 每步独立评估，不依赖累积 |
| State divergence | ✗ (退化为噪声) | 不需要轨迹级累积信号 |
| Context loss | ✗ | step position 作为 bottleneck 的调制因子 |

### 4.3 三层 Bottleneck 体系

#### Layer 1: Step-Level Capability Bottleneck（动作级）

```
定义: 在什么条件下模型的 step success rate 最低？

维度:
  - action_type: click vs type vs drag vs wheel vs ...
  - UI complexity: 元素密度、嵌套深度、target 大小
  - domain: Excel (密集) vs PPT (稀疏)

识别方法:
  对所有评估步骤做 failure analysis
  按 (action_type, UI_state_features) 聚类
  找出 success rate 最低的聚类
```

#### Layer 2: Trajectory-Level Bottleneck（轨迹级）

```
定义: 什么类型的轨迹最容易失败？

维度:
  - trajectory_length: Short / Medium / Long
  - first_error_position: 越早 → 越严重
  - task_complexity: 子目标数量
  - state_diversity: 跨越多少不同 UI 状态

识别方法:
  对所有轨迹做回归分析
  找出哪些特征最能预测 trajectory failure
```

#### Layer 3: Error Cascade Bottleneck（级联级）

```
定义: 哪些错误最容易导致后续步骤全部失败？

维度:
  - error_type at failure point
  - state_deviation after error
  - recovery_probability

识别方法:
  分析 error cascade 数据
  找出 "不可恢复错误" 的特征
```

---

## 5. 对齐方案：三阶段实施

### 阶段一：从 Eval 数据构建 Failure Map

**目标**: 用现有 SFT v2 trajectory evaluation 数据，构建每个 (state_type, action_type) 的 failure rate。

**数据来源**:
- `train_GUI_360/GUI-360-eval/results/trajectory_analysis/base_vs_sft_v2/`
- Step-level evaluation results

**构建方法**:

```python
# 伪代码：从 trajectory_analysis 结果提取 failure map
failure_map = {}

for trajectory in eval_results:
    for step in trajectory.steps:
        key = (step.ui_state_type, step.action_type)
        # e.g., ("excel_ribbon", "click"), ("word_dialog", "type")

        if key not in failure_map:
            failure_map[key] = {"success": 0, "total": 0}
        failure_map[key]["total"] += 1
        if step.is_correct:
            failure_map[key]["success"] += 1

# Bottleneck score = failure rate
bottleneck_score = {
    k: 1 - v["success"] / v["total"]
    for k, v in failure_map.items()
}
```

**需要补充的分析维度**（从现有 eval 数据即可提取）:

| 维度 | 数据来源 | 预期发现 |
|------|---------|---------|
| per action_type success rate | step-level eval results | click/drag 的 args match 远低于 type |
| per UI_complexity success rate | state_registry.json | 对话框、dense spreadsheet 失败率更高 |
| per step_position success rate | trajectory analysis | step 6+ 准确率急剧下降 |
| per domain success rate | 已有：PPT > Word > Excel | Excel 坐标最难 |

### 阶段二：Difficulty-Weighted Reward 替代 f_pseudo

**目标**: 用 failure map 计算的 difficulty weight 替代 f_pseudo 的 reward shaping。

**修改文件**: `verl/workers/reward_manager/f_pseudo_dapo.py`

```python
# ======== 当前 (f_pseudo_dapo.py ~line 178) ========
r_total = r_action_match + lambda * f_pseudo(t)
# f_pseudo ≈ noise for long trajectories

# ======== 新方案 ========
difficulty_weight = failure_map.get(
    (ui_state_type, action_type),
    default=0.5
)
r_total = r_action_match * (1 + lambda_d * difficulty_weight)
```

**设计思路**:

| 场景 | difficulty_weight | r_action_match | r_total | 效果 |
|------|:-----------------:|:--------------:|:-------:|------|
| 简单步做对 | 0.2 | 0.9 | 0.9 × 1.02 = 0.918 | 基本不变 |
| 困难步做对 | 0.8 | 0.9 | 0.9 × 1.08 = 0.972 | 额外奖励 |
| 困难步做错 | 0.8 | 0.1 | 0.1 × 1.08 = 0.108 | 基本不变 |
| 简单步做错 | 0.2 | 0.1 | 0.1 × 1.02 = 0.102 | 基本不变 |

**关键优势**:

| | f_pseudo (旧) | difficulty_weight (新) |
|---|---|---|
| 信号来源 | 图拉普拉斯特征向量 | 模型 eval 数据 |
| 方向性 | 无（累积 ≈ 0） | 不需要方向性（每步独立） |
| 稀疏性 | 54.6% 零值 | 0%——每步都有 weight |
| 长轨迹表现 | 退化为噪声 | 不退化——每步独立计算 |
| 可解释性 | "穿越图瓶颈" | "在困难步骤做对了" |
| 测试集覆盖 | 0%（仅训练数据） | 100%（基于 state/action 类型泛化） |

### 阶段三：Failure Taxonomy 接入 MoE Expert Routing

**目标**: 让 bottleneck 不仅影响 reward，还指导 expert 分工。

**当前 Router 的问题**:

```python
# router.py: TextOnlyRouter
# 只看 instruction text，不看 UI state，不知道当前步骤的难点
instruction_features = self.feature_extractor(hidden_states, instruction_mask)
router_output = self.router(instruction_features)  # [B, num_experts]
```

同一个 instruction（如 "Format the cell as bold"）无论在什么 UI 状态下都路由到同一个 expert。但实际上：
- 在简单 UI（Home tab 可见）→ 只需一次 click → 应路由到 **坐标精度 expert**
- 在复杂 UI（需先导航到 Home tab）→ 多步操作 → 应路由到 **导航 expert**

**新 Router: FailureAwareRouter**

```python
class FailureAwareRouter(nn.Module):
    """
    路由策略: 根据当前步骤最可能的 failure type 选择 expert。

    Expert 分工:
      Expert 0: Coordinate Precision (click/drag 坐标精度)
      Expert 1: Text Manipulation (type/select_text 文本处理)
      Expert 2: Navigation / State Change (tab 切换/dialog 关闭)
      Expert 3: Long-horizon Recovery (从错误状态恢复)
    """

    def __init__(self, hidden_size, num_experts=4, router_hidden=256):
        super().__init__()

        # Vision features (UI 状态复杂度)
        self.vision_proj = nn.Linear(hidden_size, router_hidden)

        # Text features (instruction 意图)
        self.text_proj = nn.Linear(hidden_size, router_hidden)

        # Step context (轨迹位置信息 — 影响 error cascade 处理)
        self.step_embed = nn.Embedding(64, router_hidden)

        # Router MLP
        self.router = nn.Sequential(
            nn.Linear(router_hidden * 3, router_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(router_hidden, num_experts),
        )

    def forward(self, vision_features, text_features, step_id):
        v = self.vision_proj(vision_features)
        t = self.text_proj(text_features)
        s = self.step_embed(step_id.clamp(max=63))

        combined = torch.cat([v, t, s], dim=-1)
        logits = self.router(combined)
        return F.softmax(logits / self.temperature, dim=-1)
```

**Expert 分工对齐 Failure 数据**:

| Expert | 目标 Failure Mode | 对应 Eval 数据 | 激活条件 |
|--------|-----------------|---------------|---------|
| Expert 0 (坐标) | args_match=11.28%（最弱） | 82.1% trajectory failure | UI 元素密集、click/drag |
| Expert 1 (文本) | type action 的 text matching | type 自环率高 (2,532) | action = type/select_text |
| Expert 2 (导航) | 状态转移失败 | dialog/非标准 tab 状态 | 需要多步 navigation |
| Expert 3 (恢复) | error cascade (step 6+ accuracy 38%) | 长轨迹 TSR=0% | step_id > 5, post-error |

---

## 6. 长轨迹 (16+ steps) 专项对齐

### 6.1 为什么长轨迹是独立的问题

长轨迹 TSR=0%（SFT v2 和 Base 均如此），且 f_pseudo 在此完全退化：

| 指标 | Long (16+ steps) |
|------|:----------------:|
| TSR | 0% (SFT v2), 0% (Base) |
| f_pseudo 高一致性 | 10.0% |
| f_pseudo 低一致性（噪声） | 68.9% |
| f_pseudo 零值比例 | 72.6% |

### 6.2 长轨迹 = 组合多个 Bottleneck 类型

```
Long trajectory (16+ steps) 示例:
  Step  1-3:  Navigation bottleneck → 找到正确的 tab/menu
  Step  4-8:  Coordinate bottleneck → 精确点击目标
  Step  9-12: Text bottleneck → 输入正确内容
  Step 13-16: Recovery bottleneck → 从之前的小错误中恢复

每段需要不同的 expert → FailureAwareRouter + step_embed 使动态路由成为可能
```

### 6.3 Progress Estimator 替代 Eigenfunction

**目的**: 为长轨迹提供方向性信号（解决 f_pseudo 累积≈0 的致命问题）。

```python
class ProgressEstimator(nn.Module):
    """
    替代 eigenfunction 的进度估计器。

    训练目标: p(s_t) ≈ t / T (归一化进度)

    保证:
    - p(s_start) ≈ 0, p(s_end) ≈ 1
    - 累积: Σ(p(s_{t+1}) - p(s_t)) ≈ 1 > 0 (有净方向性!)

    对比 eigenfunction:
    - f(s_start) ≈ f(s_end) → 累积 ≈ 0 (无方向性)
    """

    def __init__(self, input_dim=43, hidden_dims=[256, 256]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())  # 输出 [0, 1]
        self.net = nn.Sequential(*layers)

    def forward(self, state_embedding):
        return self.net(state_embedding)
```

**训练数据**: 直接从现有成功轨迹构造，不需要额外数据收集：

```python
# 用 outputs/transitions/gui360_full/ 的数据
# 每条成功轨迹的每个状态:
#   state_embedding (43-dim) → target = step_id / total_steps
```

**与 Difficulty Estimator 正交**:

| 组件 | 预测 | 用途 |
|------|------|------|
| Difficulty Net d(s) | 模型在状态 s 的平均 failure rate | 困难步额外奖励 |
| Progress Net p(s) | 任务完成进度 | 长轨迹方向性信号 |

**结合的 reward shaping**:

```python
reward_bonus = lambda_d * difficulty(s_t) * action_score(t)    # 困难步奖励
             + lambda_p * (progress(s_{t+1}) - progress(s_t))  # 进度奖励
```

### 6.4 改造 f_net 而非废弃

不需要完全丢弃 f_net 框架，而是改变学习目标：

```python
# 旧目标 (graph_analysis.py 的 eigenfunction_loss):
# G(f) = 0.5 * E[(f(s) - f(s'))²] + eta * [(E[f²]-1)² + (E[f])²]
# 学习图结构 → 没有方向性

# 新目标: Difficulty Estimator
# Loss: MSE(d_net(s), failure_rate(s))
#
# 复用:
# - 43-dim state embedding pipeline (collect_transitions.py, state_representation.py)
# - EigenfunctionNet 网络结构 (43 → 256 → 256 → 1)
# - 训练框架 (train_eigenfunction.py, 改 loss 即可)
```

---

## 7. 完整 Pipeline 设计

### 7.1 架构图

```
                    ┌──────────────────────────────────────┐
                    │         Failure-Aligned Pipeline      │
                    └──────────────────┬───────────────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
   Difficulty Net                Progress Net               FailureAware Router
   d(s) ∈ [0,1]                p(s) ∈ [0,1]              vision+text+step → expert
   "状态难度"                    "任务进度"                "哪个 expert 处理"
         │                             │                             │
         │    ┌────────────────────────┘                             │
         ▼    ▼                                                      ▼
   Reward Shaping:                                            Expert Selection:
   r = action_score                                           Expert 0: 坐标精度
     × (1 + λ_d × d(s))     ← 困难步奖励                     Expert 1: 文本处理
     + λ_p × (p(s')-p(s))   ← 进度奖励                       Expert 2: 导航
                                                              Expert 3: 错误恢复
```

### 7.2 与现有代码的对应关系

| 新组件 | 替换/修改的现有代码 | 改动量 |
|--------|-------------------|:------:|
| Failure Map | 新增分析脚本 (从 eval results 提取) | 新脚本 ~200 行 |
| Difficulty Net | `graph_analysis.py` 的 loss 换成 MSE | 小（改 loss） |
| Progress Net | 新增，复用 `EigenfunctionNet` 结构 | 小（~100 行） |
| Difficulty-Weighted Reward | `f_pseudo_dapo.py` 的 reward 计算 | 中（改 reward） |
| FailureAware Router | 扩展 `ContextAwareRouter` 加 `step_embed` | 小（加 embedding） |
| Expert 分工 | 不需要代码改动——通过 RL 训练自然分化 | 无 |

### 7.3 与 FULL_SFT_TO_MOE_DESIGN 方案的兼容性

本方案与 `FULL_SFT_TO_MOE_DESIGN.md` 中的方案二（SFT as Base）完全兼容：

```yaml
# traj_grpo_moe_v5_sftbase.yaml 改动:
actor_rollout_ref:
  model:
    path: .../gui360_full_sft_v2   # SFT v2 as frozen base (不变)
    moe:
      # 新增 router 配置:
      router_type: failure_aware   # 替代 text_only / context_aware

      # 新增 reward shaping 配置:
      difficulty_map_path: outputs/failure_map/difficulty_map.json
      progress_estimator_path: outputs/progress_net/progress_net_final.pt
      difficulty_lambda: 0.1
      progress_lambda: 0.05
```

---

## 8. 实施优先级与路线图

### P0: 确认 f_pseudo 当前效果（前提验证）

| | 详情 |
|---|------|
| **做什么** | 运行 ablation: λ=0 vs λ=0.1 对比 TSR |
| **改动** | 只改 1 行 config (`f_pseudo_lambda: 0.0`) |
| **工作量** | 1 天 (training run) |
| **预期** | A ≈ B → 确认 f_pseudo 无效，放心弃用 |

### P1: 构建 Failure Map（数据分析）

| | 详情 |
|---|------|
| **做什么** | 从 eval 数据提取 per-(state_type, action_type) failure rate |
| **输入** | `GUI-360-eval/results/trajectory_analysis/` |
| **输出** | `outputs/failure_map/difficulty_map.json` |
| **工作量** | 2 天 |
| **价值** | 定量理解 failure 分布，为后续所有改进提供数据基础 |

### P2: Difficulty-Weighted Reward 替代 f_pseudo（最小可行改动）

| | 详情 |
|---|------|
| **做什么** | 用 difficulty_weight 替代 f_pseudo |
| **改动文件** | `verl/workers/reward_manager/f_pseudo_dapo.py` |
| **工作量** | 1 天 |
| **预期** | 困难步额外奖励 → 模型在 args match 上提升 |

### P3: 训练 Progress Estimator（长轨迹方向信号）

| | 详情 |
|---|------|
| **做什么** | 训 p(s)∈[0,1] 预测完成进度 |
| **复用** | `EigenfunctionNet` 结构 + `train_eigenfunction.py` 框架 |
| **工作量** | 2 天 |
| **预期** | 长轨迹有净方向性信号（累积≈1 vs 当前≈0） |

### P4: 扩展 Router 加 step_embed（Router 感知轨迹位置）

| | 详情 |
|---|------|
| **做什么** | 在 ContextAwareRouter 中加 step position embedding |
| **改动文件** | `verl/models/moe/router.py` |
| **工作量** | 1 天 |
| **预期** | Router 能在不同 step 动态选择 expert |

### P5: 端到端 MoE RL 训练验证

| | 详情 |
|---|------|
| **做什么** | 方案二 (SFT base) + difficulty reward + progress estimator + failure-aware router |
| **工作量** | 3 天 (training) |
| **评估** | TSR by trajectory length, per-domain, per-action-type |

### 路线图

```
Week 1:  P0 (ablation) + P1 (failure map)
         → 确认 f_pseudo 无效 + 定量 failure 分析

Week 2:  P2 (difficulty reward) + P3 (progress estimator)
         → 替代 f_pseudo 的新 reward shaping

Week 3:  P4 (router) + P5 (端到端训练)
         → 完整 failure-aligned MoE pipeline
```

---

## 附录: 文件索引

| 文件 | 说明 |
|------|------|
| `Option-incentivized-MoE/bottleneck_failure_alignment.md` | 本文档 |
| `Option-incentivized-MoE/f_pseudo_effectiveness_analysis.md` | f_pseudo 信号质量详细分析 |
| `Option-incentivized-MoE/f_pseudo_tldr.md` | f_pseudo 分析 TL;DR |
| `Option-incentivized-MoE/FULL_SFT_TO_MOE_DESIGN.md` | Full SFT → MoE 方案设计 |
| `train_GUI_360/GUI_360_all_eval_results.md` | 所有模型评估结果汇总 |
| `verl/workers/reward_manager/f_pseudo_dapo.py` | 当前 f_pseudo reward shaping（待改） |
| `verl/models/moe/graph_analysis.py` | eigenfunction 训练（待改 loss） |
| `verl/models/moe/router.py` | MoE Router（待扩展 step_embed） |
| `verl/models/moe/moe_wrapper.py` | MoE VLM Wrapper |
| `verl/utils/reward_score/gui360/reward.py` | GUI-360 soft coordinate scoring |

---

*分析日期: 2026-03-12*
*关联分析: f_pseudo_effectiveness_analysis.md, FULL_SFT_TO_MOE_DESIGN.md*
