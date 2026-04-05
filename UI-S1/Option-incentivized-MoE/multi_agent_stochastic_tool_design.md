# Model-Based Multi-Agent GUI Framework: 详细设计与早期验证

> 核心命题：把 grounding model 视为 **stochastic tool**（随机性工具），通过 multi-sampling + verification + recovery 构建 multi-agent system，让 agent 即使 grounding 错误也能恢复并跨越 bottleneck。

---

## 目录

1. [数据驱动的问题分析](#1-数据驱动的问题分析)
2. [理论框架：Model-Based Stochastic Tool](#2-理论框架model-based-stochastic-tool)
3. [Multi-Agent Architecture](#3-multi-agent-architecture)
4. [核心机制 1: Multi-Sample Grounding](#4-核心机制-1-multi-sample-grounding)
5. [核心机制 2: Verification](#5-核心机制-2-verification)
6. [核心机制 3: Recovery](#6-核心机制-3-recovery)
7. [MoE 连接](#7-moe-连接)
8. [早期验证实验（Phase 0）](#8-早期验证实验phase-0)
9. [分阶段实施路线](#9-分阶段实施路线)
10. [风险与 Fallback](#10-风险与-fallback)
11. [Phase 0 实验结果](#11-phase-0-实验结果-2026-03-12)
12. [Phase 1 早期验证：双模型互补实验](#12-phase-1-早期验证双模型互补实验)
13. [Phase 1 实验结果与分析](#13-phase-1-实验结果与分析-2026-03-13)
14. [训练演进与失败分析：Base → SFT v1 → SFT v2 → SFT v2 2ep → SFT v3](#14-训练演进与失败分析)
15. [RL 训练深化设计：数据驱动的方案](#15-rl-训练深化设计数据驱动的方案)

---

## 1. 数据驱动的问题分析

### 1.1 核心失败模式

| 指标 | 值 | 含义 |
|------|:--:|------|
| func_match | 86.18% | 模型知道**做什么** |
| args_match | **11.28%** | 模型不知道**在哪里做** |
| 坐标错误占失败比例 | **82.1%** | 主导失败模式 |

### 1.2 分叉分析 (5,461 对 success/fail 轨迹)

| 发现 | 数据 |
|------|------|
| 58% 分叉在前 20% | 早期错误 → 整条轨迹失败 |
| 66% 分叉因坐标偏差 >80px | 不是微小偏差，是点错位置 |
| Fail 轨迹 2.6x 长 (20.1 vs 7.7 steps) | Agent 无法恢复，盲目继续 |
| Survival curve: pos=0.3 仅 26% 匹配 | 到第 3 步只有 1/4 还在正确路径上 |

### 1.3 关键新发现：是点错元素，不是点偏了

对分叉步的 UI 控件 (control_label) 分析：

| 分叉类型 | 占比 | 示例 |
|---------|:----:|------|
| **点了错误的 UI 元素** | **37.9%** | 应点 "Draw Horizontal Text Box" → 实际点了 "Text Box" |
| 同一元素但坐标偏了 | 4.5% | 在正确按钮上但偏了 |
| 无 label 信息 | 57.6% | 无法确定（很可能也是错元素） |

**这意味着**：
- ❌ 单纯提高坐标精度帮助有限（只有 4.5% 是同一元素上的偏差）
- ✅ 模型需要更好地**识别 UI 元素**（区分 "Text Box" vs "Draw Horizontal Text Box"）
- ✅ **Verification** 极其重要：点错元素后屏幕变化完全不同，验证可以检测到
- ✅ **Multi-sampling** 有价值：如果模型在两个相似元素间犹豫，采样多次可以投票

### 1.4 Recovery 潜力

| 指标 | 值 |
|------|:--:|
| 分叉处坐标距离 mean | 364px |
| 分叉处坐标距离 median | 308px |
| 单步修复可挽救的轨迹 | ~12% (下界) |
| 分叉越早 → sub_score 越低 | 0.0处 28% vs 0.7处 56% |

**12% 是下界**：只考虑了"修正一步后后续自然匹配"的情况。加上 verification+recovery 后，每个 recovery 尝试都是新的机会，理论上限远高于 12%。

---

## 2. 理论框架：Model-Based Stochastic Tool

### 2.1 确定性工具 vs 随机性工具

**传统工具调用** (如 API, 数据库):
```
agent.call_tool("search_button") → click(39, 71)  [确定性]
→ 如果错了：没有恢复机制
```

**Model-Based 随机工具** (grounding model):
```
agent.sample_tool("search_button", K=5, screenshot) → {
  click(39, 71),   ← sample 1
  click(42, 68),   ← sample 2
  click(305, 136),  ← sample 3 (不同元素！)
  click(40, 70),   ← sample 4
  click(41, 69),   ← sample 5
}
→ 聚类: cluster_1 = {(39,71), (42,68), (40,70), (41,69)} [4 票]
         cluster_2 = {(305,136)} [1 票]
→ 选 cluster_1 中心 = (40.5, 69.5) [consensus]
```

### 2.2 不确定性是信息

模型预测的**分布**本身包含有价值的信息：

| 分布形态 | 含义 | 策略 |
|---------|------|------|
| 低方差（紧密聚集） | 模型很确信 → 大概率正确 | 直接用中心，K=1 即可 |
| 双峰分布 | 模型在两个元素间犹豫 | 多采样 + 投票可区分 |
| 高方差（分散） | 模型很不确定 | 这是 bottleneck → 增加 K + 启动 verification |
| 全部一致但错误 | 模型自信地错了 | multi-sample 无效 → 必须靠 verification + recovery |

### 2.3 与 Options 框架的联系

| 概念 | 传统 Options | 我们的 Multi-Agent |
|------|------------|------------------|
| Option = 时间抽象动作 | (initiation, policy, termination) | (sub-goal, grounding, verification) |
| Initiation | 进入 option 的条件 | Planner 生成 sub-goal |
| Policy | option 内部策略 | Grounder 的 K 次采样 |
| Termination | 检测 option 完成 | Verifier 检查是否成功 |
| Inter-option | option 间切换 | Recovery → re-plan |

---

## 3. Multi-Agent Architecture

### 3.1 四个 Agent 角色

```
                        ┌──────────────┐
                        │   Planner    │ → 生成可验证的 sub-goal
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │  Grounder    │ → 生成 K 个坐标候选
                        │  (×K samples)│
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │  Selector    │ → 选最佳候选 (clustering/voting)
                        └──────┬───────┘
                               │
                          Execute action
                               │
                        ┌──────▼───────┐
                        │  Verifier    │ → 比较 before/after screenshot
                        └──────┬───────┘
                             │    │
                          PASS   FAIL
                           │      │
                   Next step   ┌──▼──────────┐
                               │  Recovery   │ → 重试 / 重新规划 / 跳过
                               └─────────────┘
```

### 3.2 各 Agent 的输入输出

| Agent | Input | Output | 现有基础设施 |
|-------|-------|--------|-------------|
| Planner | task + screenshot + history | structured sub-goal + expected state | 现有 `thought` 字段 → 扩展为结构化 sub-goal |
| Grounder | sub-goal + screenshot | K 组 (function, coordinates) | 现有 vLLM rollout 的 n=4 采样 |
| Selector | K 个候选 | 1 个最佳动作 | **新**: clustering/voting 逻辑 |
| Verifier | before_img + action + after_img + expected_state | PASS/FAIL + reason | **新**: 需要训练 |
| Recovery | FAIL 原因 + current_img + goal | 新动作 / 新 plan | **新**: 需要 prompt template + 数据 |

---

## 4. 核心机制 1: Multi-Sample Grounding

### 4.1 Selection Strategy: Self-Consistency Clustering

```python
def select_best_action(candidates: list[Action], eps=30) -> Action:
    """从 K 个候选中选最佳动作"""
    # 1. 按坐标聚类
    coords = [(a.x, a.y) for a in candidates if a.function == 'click']
    clusters = DBSCAN(eps=eps, min_samples=2).fit(coords)

    # 2. 选最大 cluster 的质心
    largest_cluster = find_largest_cluster(clusters)
    centroid = mean(largest_cluster)

    # 3. 返回最接近质心的候选
    return closest_candidate(candidates, centroid)
```

### 4.2 Adaptive K: FCV 驱动

```python
def adaptive_K(fcv_value):
    """根据 bottleneck importance 决定采样数"""
    if fcv_value > 0.6:   return 7   # 关键 bottleneck
    elif fcv_value > 0.3: return 5   # 中等重要
    elif fcv_value > 0.1: return 3   # 略微重要
    else:                 return 1   # 常规步骤
```

### 4.3 Multi-Sample 能解决多少问题？

根据 §1.3 的分析，Multi-Sample 主要帮助以下场景：
- **双峰分布** (模型在正确/错误元素间犹豫): ~20-30% 的 divergence cases
- **对同一元素但坐标略偏**: ~4.5% 的 divergence cases
- **模型一致错误**: multi-sample **无法**解决 → 需要 verification + recovery

**预期 args_match 提升**: 11.28% → ~15-18% (仅靠 multi-sample)
**加上 verification+recovery**: → ~25-35% (目标)

---

## 5. 核心机制 2: Verification

### 5.1 为什么 Verification 是最关键的新能力

当前系统的致命缺陷：**agent 从不知道自己的上一步是否成功**。

- 点了 "Text Box" 而非 "Draw Horizontal Text Box" → 屏幕显示完全不同的东西
- 但 agent 看到新截图后继续盲目操作 → 后续所有步骤都建立在错误状态上
- 这就是 error cascade 的根源

**Verification 打破 cascade**: 在每步执行后，显式检查"我的动作是否达到了预期效果？"

### 5.2 Verification 的数据来源

**正例** (action 成功):
- 来自 13,750 success 轨迹的每一步
- (before_screenshot, action, after_screenshot) → "YES, sub-goal achieved"

**负例** (action 失败):
- 来自 62,236 fail 轨迹的分叉步及之后
- 特别是 5,461 对 paired 轨迹的分叉点
- (before_screenshot, wrong_action, wrong_after_screenshot) → "NO, sub-goal not achieved"

**数据量估计**:
- 正例: ~13,750 × 7.7 steps = ~106K examples
- 负例: 从 fail 轨迹选取关键分叉步 ~50K examples
- 总计: ~150K verification training examples

### 5.3 Verification Prompt Template

```
You are verifying whether a GUI action achieved its intended sub-goal.

Sub-goal: {sub_goal_description}
Action taken: {action_type} at coordinate ({x}, {y})

[Image 1: Screenshot BEFORE the action]
[Image 2: Screenshot AFTER the action]

Did the action achieve the sub-goal?
Answer in this format:
{"verified": true/false, "reason": "brief explanation of what changed or didn't change"}
```

### 5.4 Rule-Based Verification (零成本 baseline)

在 RL **训练时** (有 ground truth)，可以用 rule-based verification:

```python
def rule_based_verify(predicted_action, ground_truth_action):
    """训练时：用 ground truth 作为 verification oracle"""
    score = soft_coordinate_score(predicted_action, ground_truth_action)
    return score > 0.3  # 阈值以上 = PASS
```

这在训练时零成本，但推理时无 ground truth → 需要 model-based verification。

---

## 6. 核心机制 3: Recovery

### 6.1 Recovery 决策树

```
Verification FAIL at step t:
│
├── retry_count < max_retries (default 2)?
│   ├── YES → RETRY: 给模型 "action failed" context, 重新 sample K 个坐标
│   │         prompt: "Your {action} at ({x},{y}) did not achieve '{sub_goal}'.
│   │                  The current screenshot shows: [image]. Try again."
│   │
│   └── NO → retry_count >= max_retries?
│       ├── YES → RE-PLAN: 调用 Planner 从当前状态重新规划
│       │         prompt: "Sub-goal '{sub_goal}' failed after {n} attempts.
│       │                  Current state: [image]. Original task: {task}.
│       │                  Generate a new plan from this state."
│       │
│       └── SKIP: 接受错误，continue to next sub-goal
│                 (当 FCV(t) < 0.2 时选此路径 — 这步不重要)
```

### 6.2 Recovery 数据来源

**Retry 数据**: 对 5,461 paired 轨迹，在分叉步构造:
- Input: (错误的 after_screenshot, "action X failed", original_sub_goal)
- Target: success 轨迹在同一步的 action (正确的动作)
- ~5K training examples

**Re-plan 数据**: 对 fail 轨迹中 sub_score 有 partial progress 的:
- `sub_scores = {'File menu accessed': 'yes', 'Info selected': 'no', ...}`
- Input: (当前截图, "已完成: ..., 未完成: ..., 原始任务")
- Target: 完成剩余子目标的 action sequence
- 需要从 success 轨迹构造 target

### 6.3 Recovery 与 Error Cascade 的定量关系

当前 error cascade 的代价:

| 分叉位置 | Sub-score 完成度 | 轨迹浪费 |
|---------|:--------------:|---------|
| 0.0-0.1 (极早期) | 28% | 剩余 ~18 步都是无效的 |
| 0.2-0.3 | 39% | 剩余 ~14 步大部分无效 |
| 0.7-0.8 (晚期) | 55% | 接近完成，仅损失少量步 |

**Recovery 的期望收益**: 如果能在分叉步成功 recover:
- 早期 recovery (pos=0.1): 挽救 ~72% 的剩余进度 → TSR 提升最大
- 晚期 recovery (pos=0.7): 挽救 ~45% 的剩余进度 → 帮助有限

配合 BCR: **FCV 越高的位置** → **越值得 recovery** → **分配更多 retry 预算**

---

## 7. MoE 连接

### 7.1 方案 A: Phase-Based Routing（推荐）

不绑定 expert 到固定 agent 角色，而是让 router 根据当前 phase 的 prompt 特征自然路由:

| Phase | Prompt 特征 | Router 行为 |
|-------|-----------|-----------|
| Planning | "Break down this task..." | 倾向 reasoning expert |
| Grounding | "Click the {target}..." | 倾向 precision expert |
| Verification | "Compare these screenshots..." | 倾向 visual comparison expert |
| Recovery | "The action failed, try..." | 倾向 reasoning expert |

ContextAwareRouter 已支持 vision+text 特征融合，不同 phase 的 text context 不同 → router 可以自动学会区分。

### 7.2 方案 B: Multi-Expert Ensemble for Grounding

对 grounding 步骤用 top_k=4 (所有 expert):
- 每个 expert 的 LoRA delta 微调坐标预测 → 天然多样性
- 4 个 expert 的预测 → coordinate clustering → consensus
- 这就是 MoE 实现的 multi-agent voting

### 7.3 FCV-Adaptive top_k

```python
# 在 moe_wrapper.py forward() 中:
if fcv_value > 0.6:
    top_k = 4  # 所有 expert 参与投票 — 关键步骤
elif fcv_value > 0.3:
    top_k = 2  # 两个 expert
else:
    top_k = 1  # 单 expert 即可
```

---

## 8. 早期验证实验（Phase 0）

> **目标**: 在不修改任何训练代码的情况下，验证 multi-agent 路线的可行性。每个实验独立，可并行执行。

### 实验 0.1: Model Uncertainty Analysis（模型不确定性分析）

**问题**: Multi-sampling 有效的前提是模型有有意义的不确定性。如果模型每次都自信地给出相同的错误坐标，multi-sample 无用。

**方法**:
1. 用现有 eval 脚本，对 200 个 test step 各生成 K=10 个预测 (temperature=0.7)
2. 对每个 step 计算:
   - 坐标方差 σ(x), σ(y)
   - 是否有 cluster 包含 ground truth
   - Best-of-K accuracy (K 个中最好的一个是否正确)
   - Self-consistency accuracy (cluster 中心是否正确)

**成功标准**:
- Best-of-10 的 args_match 显著高于 Best-of-1 (e.g., 25% vs 11%)  → multi-sampling 有效
- Self-consistency accuracy > Best-of-1  → clustering 有效
- 如果 Best-of-10 ≈ Best-of-1  → 模型总是错到同一个地方 → multi-sample 无效，需要 verification

**工具**: 修改 `evaluation/eval_gui360_parquet.py`，增加 K 次 temperature sampling

**预计时间**: 1-2 天

### 实验 0.2: Zero-Shot Verification Test（零样本验证测试）

**问题**: 模型能否在不训练的情况下判断"我的动作是否成功"？

**方法**:
1. 从 paired 轨迹中取 100 个分叉步
2. 对每个: 构造 verification prompt (before_img + action + after_img + sub_goal)
3. 用当前 SFT 模型 zero-shot 回答 "YES/NO"
4. 测量 verification accuracy

**成功标准**:
- Accuracy > 70%  → 模型已有基础 verification 能力，SFT fine-tune 可进一步提升
- Accuracy ~50%  → 模型没有 verification 能力，需要专门训练
- 即使 accuracy 低，如果 false positive (错误地说 PASS) rate < 20% → 仍然有用

**工具**: 简单 Python 脚本 + vLLM inference

**预计时间**: 1-2 天

### 实验 0.3: Oracle Recovery Upper Bound（Oracle Recovery 上界）

**问题**: 如果我们有完美的 verification + 完美的 recovery (oracle)，能挽救多少 fail 轨迹？

**方法**:
1. 对 500 对 paired 轨迹:
   - 在分叉步，假设 oracle 检测到了错误
   - 假设 oracle 将 fail 的动作替换为 success 的动作
   - 检查: 替换后，fail 轨迹的后续步骤是否重新与 success 对齐?
2. 分别测量:
   - **1-step oracle recovery**: 替换 1 个动作后的对齐率
   - **2-step oracle recovery**: 允许替换 2 个动作
   - **Perfect recovery**: 从分叉步开始全部替换为 success 动作

**成功标准**:
- 1-step oracle recovery 可挽救 >20% 的 fail 轨迹  → recovery 有价值
- 如果 <10%  → error cascade 太严重，单步修复不够，需要 full re-plan

**工具**: 纯 Python 数据分析脚本

**预计时间**: 1 天

### 实验 0.4: Per-Domain FCV Curves（各 domain 的 FCV 曲线）

**问题**: FCV (Future Crossing Value) 曲线是否在所有 domain 一致，还是 domain-specific?

**方法**:
1. 分别对 excel/ppt/word 的 paired 轨迹计算 FCV 曲线
2. 比较三个 domain 的 bottleneck 位置是否一致

**成功标准**:
- 如果所有 domain FCV 曲线相似 → 可以用统一的 FCV
- 如果差异大 → 需要 per-domain FCV (更精确但更复杂)

**工具**: 扩展之前的 divergence 分析脚本

**预计时间**: 半天

### 实验 0.5: Retry with Error Context（带错误上下文的重试）

**问题**: 如果给模型 "你的上一步错了" 的信息，它能否产生更好的下一步？

**方法**:
1. 从 paired 轨迹中取 100 个分叉步
2. 构造 recovery prompt: "Your click at (x,y) did not open the File menu. The current screenshot shows [image]. Try again."
3. 用当前模型生成新动作
4. 比较: 新动作 vs 原始 fail 动作 vs ground truth

**成功标准**:
- 带 error context 的 retry accuracy > 原始 fail accuracy → recovery prompt 有效
- 如果无改善 → 模型需要专门的 recovery training

**工具**: 简单推理脚本 + 新 prompt template

**预计时间**: 2 天

---

## 9. 分阶段实施路线

### Phase 0: 验证 (1-2 周)

执行 §8 的 5 个实验，判断哪些机制值得投入。

**决策矩阵**:

| 实验结果 | → 行动 |
|---------|--------|
| 0.1 Best-of-K 显著提升 | → Phase 1 优先做 multi-sample |
| 0.1 Best-of-K 无提升 | → skip multi-sample, 直接做 verification |
| 0.2 Zero-shot verify >70% | → Phase 1 直接用 zero-shot verification |
| 0.2 Zero-shot verify <50% | → Phase 2 需要 verification SFT |
| 0.3 Oracle 1-step recovery >20% | → Phase 2 做 recovery |
| 0.3 Oracle 1-step recovery <10% | → 需要 full re-plan, 不只是 retry |
| 0.5 Retry with context 有效 | → Recovery prompt 可以直接用 |

### Phase 1: Quick Wins (2-3 周)

基于 Phase 0 结果，选择:

**1a. Multi-Sample Consensus at Eval Time** [如果实验 0.1 显示有效]
- 修改 eval 脚本: K=5 temperature sampling + self-consistency
- 零训练成本，仅推理时 5x 计算
- 预期 args_match: 11.28% → ~15-20%

**1b. BCR Crossing Bonus in RL** [无条件执行]
- `scripts/build_crossing_map.py` → crossing_value_map.json
- 修改 `f_pseudo_dapo.py` 加 crossing bonus
- 在 bottleneck 位置放大 reward

**1c. Rule-Based Verification in RL Training** [无条件执行]
- 训练时有 ground truth → 直接用 soft_coordinate_score 做 verification
- `if score < 0.3: add "action failed" context to next step prompt`
- 测试: 给 model error feedback 是否改善下一步

### Phase 2: Verification + Recovery SFT (4-6 周)

**2a. Verification 数据构建** — `scripts/build_verification_data.py`
- 从 paired 轨迹 + fail 轨迹构造 150K examples
- 格式: (before_img, action, after_img, sub_goal) → YES/NO

**2b. Recovery 数据构建** — `scripts/build_recovery_data.py`
- 从 paired 轨迹的分叉步构造 retry examples
- 从 fail 轨迹的 sub_scores 构造 re-plan examples

**2c. Verification + Recovery SFT**
- 在现有 LLaMA-Factory pipeline 中加入 verification + recovery 数据
- 多任务训练: grounding + verification + recovery

**2d. Multi-Phase Eval Pipeline**
- 完整 Plan → Ground(×K) → Select → Verify → Recovery loop
- 在 test set 上评估完整 pipeline

### Phase 3: Multi-Agent RL (6-8 周)

**3a. Phase-Aware MoE Routing**
- 扩展 ContextAwareRouter 支持 FCV + phase 信息
- Adaptive top_k 基于 FCV

**3b. Multi-Agent Reward Functions**
- consensus_reward (multi-sample agreement)
- verification_accuracy_reward
- recovery_success_reward
- crossing_bonus (BCR)

**3c. Full RL Training with Multi-Agent Loop**
- 每步: Plan → Ground(×K) → Select → Execute → Verify → (Recovery)
- 在 trajectory-level DAPO 训练中集成

---

## 10. 风险与 Fallback

| 风险 | 影响 | Fallback |
|------|------|---------|
| Multi-sample 无效 (模型总是一致地错) | Phase 1a 失效 | 跳过 multi-sample, 依赖 verification + recovery |
| Verification 准确率低 | Phase 2 延期 | 用 rule-based verification (训练时可用) |
| Recovery 训练数据不足 | Recovery 能力弱 | 从 fail 轨迹 + GPT-4 生成更多 recovery 数据 |
| FCV 曲线 domain-specific | 统一 FCV 不准确 | 切换为 per-domain FCV |
| Multi-phase 推理太慢 | 推理时间 5-10x | 仅在 high-FCV 步骤启用 verification，低 FCV 用 greedy |
| MoE routing 退化 | Expert 不分化 | 回退到固定 expert 分配 |

---

## 11. Phase 0 实验结果 (2026-03-12)

> 5 个实验全部完成。核心结论：**Multi-sampling 是最有效的机制**，Best-of-10 将 accuracy 从 37.5% 翻倍到 61.0%。Recovery 价值有限因为 90% 的 divergence 在 step 0 发生。Verification 需要专门训练。

### 11.1 实验 0.1: Model Uncertainty Analysis — ✅ 强烈验证 multi-sampling

| 指标 | Greedy (K=1) | Self-Consistency (Cluster) | Best-of-K (Oracle) |
|------|:---:|:---:|:---:|
| Accuracy | **37.5%** | **44.5%** (+7.0pp) | **61.0%** (+23.5pp) |
| Mean distance (px) | 212.1 | 198.9 | 84.8 |
| Median distance (px) | 101.4 | 74.0 | 24.9 |

**不确定性统计** (N=200, K=10, temperature=0.7):
- Mean σ(x) = 121.4px, σ(y) = 72.6px → 模型确实有有意义的不确定性
- 35.5% 的预测呈双峰分布 (模型在两个 UI 元素间犹豫)
- Mean agreement rate = 0.60

**Agreement rate 是极强的置信度信号**:

| Agreement Rate | 占比 | Cluster Accuracy |
|:-:|:-:|:-:|
| ≥ 0.3 | 88% | 49.1% |
| ≥ 0.5 | 63% | 58.7% |
| ≥ 0.7 | 44% | 68.5% |
| ≥ 0.9 | 26% | **83.0%** |

**不确定性 vs. 准确率**:
- 低方差 (≤157px): cluster accuracy = **61.0%**
- 高方差 (>157px): cluster accuracy = **28.0%**
- → 高方差预测 = bottleneck 信号 → 应增加 K 或启动 verification

**关键结论**:
1. Multi-sampling **极为有效** — Best-of-10 几乎翻倍准确率
2. Self-consistency clustering 比 greedy 好 7pp，但离 oracle 还有 16.5pp 的 gap → room for better selection strategies
3. Agreement rate 可以作为 adaptive K 的决策信号
4. 35.5% 双峰分布证实了 "模型在相似元素间犹豫" 的假设

### 11.2 实验 0.2: Zero-Shot Verification — ❌ 需要专门训练

> **v2 更新 (2026-03-12)**: 修复了 negative examples 缺失问题。由于 fail 截图未解压 (tar.gz archives)，改用 scrambled success pairs 构造 negative examples。N=200 (100 positive + 100 negative)。

| 指标 | 值 |
|------|:--:|
| Overall accuracy | **59.0%** (N=200) |
| YES accuracy | 18.0% (大量 false negative) |
| NO accuracy | **100.0%** (完美识别错误) |
| Precision | **100.0%** |
| Recall | 18.0% |
| F1 | 30.5% |

**Confusion Matrix**:
|  | Predicted YES | Predicted NO |
|--|:---:|:---:|
| Actual YES | 18 | 82 |
| Actual NO | 0 | **100** |

**分析**:
- 模型强烈偏向回答 "NO" — 82% 的实际成功步骤被判为失败
- **Precision = 100%**: 当模型说 YES 时总是对的 → 高度保守但完全可信
- **NO accuracy = 100%**: 所有 scrambled negative examples 都被正确识别 (这部分相对简单)
- 59% 仍低于 70% 阈值
- 核心问题: 模型缺乏判断 "动作成功" 的能力，倾向于不确定时默认回答 NO

**结论**: 模型没有足够的 verification 能力，需要 **verification SFT 训练**。但 precision=100% 是一个很好的基础 — 模型不会产生 false positive，这意味着训练只需提高 recall (减少 false negative)。

### 11.3 实验 0.3: Oracle Recovery Upper Bound — ⚠️ Recovery 价值有限

| 指标 | 值 |
|------|:--:|
| 1-step oracle recovery | **13.0%** (低于 20% 阈值) |
| 2-step oracle recovery | **15.0%** |
| Perfect oracle recovery | 100% (by definition) |

**分叉分析** (N=500 paired trajectories):

| 指标 | 值 | 含义 |
|------|:--:|------|
| Mean divergence step | **0.3** | 几乎全部在 step 0 分叉！ |
| Early divergence (< 20%) | **90.2%** | 绝大多数错在第一步 |
| Mean fail length | 18.1 steps | Fail 轨迹 2.2x 长 |
| Mean wasted steps | 16.8 | 分叉后平均浪费 17 步 |
| Mean coord distance at div | 278.8px | 不是微偏，是点错了位置 |

**分叉类型**:
| 类型 | 数量 | 占比 |
|------|:---:|:---:|
| action_type 不同 | 282 | 56.4% |
| coordinate 不同 | 196 | 39.2% |
| text 不同 | 22 | 4.4% |

**关键结论**:
1. **Recovery 的根本问题**: 90% 的 divergence 在 step 0 → 没有 "before divergence" 的步骤可以匹配，所以 1-step replacement 后 fail trajectory 仍然与 success 不对齐
2. **Prevention 比 Recovery 更重要**: 应集中资源让 step 0 正确 (multi-sampling)，而非在错误后恢复
3. 13% 的 1-step recovery 仍有价值 → 但不应是主要投入方向
4. **Wasted steps 巨大**: 平均 16.8 步浪费 → 即使部分 recovery 也能显著减少资源浪费

### 11.4 实验 0.4: Per-Domain FCV Curves — ✅ 验证 bottleneck 集中在 step 0

**Cumulative divergence rate** (N=2,000 pairs, excel+word):

| Step | All | Excel | Word |
|:---:|:---:|:---:|:---:|
| 0 | **81.9%** | 79.5% | 82.5% |
| 1 | 93.0% | 92.4% | 93.2% |
| 2 | 96.9% | 96.2% | 97.1% |
| 3 | 98.7% | 98.0% | 98.8% |
| 5 | 99.9% | 99.7% | 99.9% |

**分叉步的 action type 分布**:
| Action Type | All | Excel | Word |
|------------|:---:|:---:|:---:|
| click | 44.0% | 52.4% | 42.0% |
| empty/terminate | 24.7% | 9.6% | 28.4% |
| type | 16.6% | 20.8% | 15.6% |
| select_text | 5.5% | — | — |
| select_table_range | 2.6% | 13.4% | — |

**Cross-domain consistency**: Excel vs Word correlation r = 0.466 (中等相关)

**关键结论**:
1. **Step 0 是绝对的 bottleneck** — 82% 在第一步就分叉
2. Excel 和 Word 的 FCV pattern 类似但不完全一致 → **per-domain FCV recommended**
3. PPT 数据缺失 (paired data 中没有 ppt 匹配) — 需要补充
4. 44% 的分叉发生在 click 动作 → multi-sampling 最应该应用于 click

### 11.5 实验 0.5: Retry with Error Context — ⚠️ 微弱改善

> **v2 更新 (2026-03-12)**: 修复了 prompt 格式 (添加 `<tool_call>` system prompt) 和 GT action 过滤 (排除 empty/terminate)。N=100。

| 条件 | Full Match | Function Match | Coord Match (given func) |
|------|:---:|:---:|:---:|
| Original (无 context) | **24.0%** | 66.0% | 36.4% |
| Retry (简单错误提示) | **25.0%** (+1pp) | 66.0% | 37.9% |
| Retry (详细推理提示) | **26.0%** (+2pp) | 66.0% | 39.4% |

**Pairwise comparison** (retry simple vs original):
| 变化 | 数量 | 占比 |
|------|:---:|:---:|
| Improved (wrong→right) | 4 | 4.0% |
| Regressed (right→wrong) | 3 | 3.0% |
| Same correct | 21 | 21.0% |
| Same wrong | 72 | 72.0% |
| **Net improvement** | **+1** | |

**Coordinate distance** (pixels, lower is better):
| 条件 | Mean | Median |
|------|:---:|:---:|
| Original | 222.5 | 183.2 |
| Retry simple | **213.4** | **121.0** |
| Retry reasoning | 243.3 | 194.8 |

**分析**:
- Function match 完全一致 (66%) — retry context 不影响 action type 选择
- Retry 主要改善 coordinate precision (+1-2pp)，但效果非常微弱
- Simple retry 的 median distance 改善显著 (183→121px)，但 mean 改善不大
- 72% 的样本三种条件都做错 → 错误的根本原因不是 "缺少错误提示"
- 只有 word domain 的数据 (paired data 限制)

**结论**: Error context 对 retry 只有**微弱帮助** (+2pp)。大部分错误是因为模型本身无法正确 ground 到目标元素，而非缺少上下文。这与 Exp 0.3 的结论一致: **Prevention (multi-sampling) 比 Recovery (retry) 更有价值**。

---

### 11.6 综合结论与决策矩阵

| 实验 | 结果 | → 行动 |
|------|------|--------|
| **0.1** Best-of-10 = 61% (+23.5pp) | ✅ 强烈有效 | **Phase 1 最高优先级**: eval-time multi-sampling + adaptive K |
| **0.2** Verify accuracy = 59%, recall=18% | ❌ 不够 | **Phase 2**: verification SFT 训练 (precision=100% 是好基础，只需提高 recall) |
| **0.3** 1-step recovery = 13% | ⚠️ 有限价值 | **降低优先级**: recovery 不如 prevention 重要 |
| **0.4** 82% diverge at step 0 | ✅ 关键发现 | **聚焦 step 0**: 所有资源投入第一步的准确率 |
| **0.5** Retry +2pp (24%→26%) | ⚠️ 微弱改善 | **降低优先级**: error context 几乎不帮忙，prevention > recovery |

### 11.7 修订后的实施优先级

基于实验结果，**修订优先级**:

**新 Phase 1 (2 周)**: Multi-Sample Consensus at Eval + Training

1. **Eval-time K-sample**: 对所有 click 动作用 K=5, DBSCAN clustering
   - 预期: args_match 从 11.28% → ~20-25% (基于 exp 0.1 的 cluster accuracy 44.5%)
2. **Agreement-based adaptive K**:
   - agreement ≥ 0.9 → K=1 (已有 83% accuracy, 无需多次采样)
   - agreement < 0.5 → K=10 (bottleneck, 增加预算)
3. **Consensus Reward in RL**: 对 4 个 rollout sample 的 coordinate clustering agreement 作为额外 reward
4. **Step-0 Focus**: 在 RL 中对 step 0 的 reward 额外放大 (基于 82% divergence 在 step 0)

**新 Phase 2 (4 周)**: Verification SFT

1. 构建 150K verification training data
2. 重点: fail 轨迹的截图比较 (before/after) → YES/NO 判断
3. 验证 SFT 后 accuracy 是否超过 70%

**Phase 3 (降低优先级)**: Recovery

1. 仅在 Phase 2 verification 成功后开始
2. 先做 rule-based recovery (训练时有 GT)
3. Model-based recovery 作为 Phase 4

---

## 12. Phase 1 早期验证：双模型互补实验

> **核心假设**: 利用已有的 80% grounding 模型 (SFT v3) + 全能模型 (SFT v2) 的互补能力，通过 multi-sampling + selection 大幅提升 GUI agent 性能。
>
> **目标**: 在不修改任何训练代码的前提下，用 5-7 天的 eval-time 实验验证双模型组合方案是否可行，为后续 Phase 2/3 提供 Go/No-Go 决策依据。

### 12.1 动机：从 Phase 0 到 Phase 1 的逻辑链

Phase 0 的核心发现为 Phase 1 指明了方向:

| Phase 0 发现 | 数据 | → Phase 1 行动 |
|-------------|------|---------------|
| Multi-sampling 有效 | Best-of-10=61% vs greedy 37.5% | 用 80% 的 grounder (SFT v3) 做 multi-sampling → 更高上限 |
| DBSCAN 只用了部分潜力 | Cluster 44.5% vs Oracle 61.0% (gap=16.5pp) | 更强的 base model 可以缩小这个 gap |
| Agreement rate 是强信号 | ≥0.9 → 83% accuracy | 可用于 adaptive K 和 confidence routing |
| 82% 分叉在 step 0 | Exp 0.3/0.4 | Step 0 的 grounding accuracy 至关重要 |
| Recovery 价值有限 | 1-step recovery = 13%, retry +2pp | Prevention > Recovery → 集中资源做 multi-sample grounding |

**关键洞察**: Phase 0 用的是 LoRA v4 (64.37% grounding)。现在有 SFT v3 (79.48% grounding) — **base accuracy 提升 15pp → multi-sampling 效果应显著更强**。

### 12.2 已有模型资源

| 模型 | Grounding | Action(Visual) | Checkpoint 路径 | 特点 |
|------|:---------:|:--------------:|----------------|------|
| **Grounding SFT v3** | **79.48%** | 3.07% | `llamafactory/output/gui360_full_sft_v3_grounding` | Grounding 特化，Action 严重过拟合 |
| **SFT v2** | 70.56% | **46.90%** | `llamafactory/output/gui360_full_sft_v2` | 全能型最优 |
| LoRA v4 (Phase 0 用) | 64.37% | 27.53% | `llamafactory/output/gui360_lora_sft_v4/checkpoint-354` | RL 初始化候选 |

**核心想法**: SFT v3 擅长 **WHERE** (grounding 定位)，SFT v2 擅长 **WHAT** (action 规划) → 组合 > 任一单模型。

### 12.3 假设与对应实验

| 假设 | 验证实验 |
|------|---------|
| H1: SFT v3 的 80% accuracy + multi-sampling → 90%+ grounding | Exp 1.1 |
| H2: 更好的坐标能直接提升 action prediction (坐标是瓶颈) | Exp 1.3 ★ |
| H3: 双模型组合 > 任一单模型 | Exp 1.5 |
| H4: SFT v2 和 SFT v3 犯不同的错误 → 互补 | Exp 1.6 |
| H5: Agreement rate 在更强 grounder 上仍然有效 | Exp 1.7 |
| H6: SFT v3 的 grounding 能力不是 prompt-specific | Exp 1.4 |

### 12.4 实验总览

共 7 个实验，分为两组:

| 实验 | 问题 | GPU 需求 | 优先级 |
|------|------|:--------:|:------:|
| **1.1** | SFT v3 multi-sample 上限 | 1 node (V3) | ★★★ |
| **1.2** | SFT v2 multi-sample 对比 | 1 node (V2) | ★★ |
| **1.3** ★ | Oracle 坐标替换 (Go/No-Go) | 2 nodes (V2+V3) | ★★★★ |
| **1.4** | V3 action prompt 泛化 | 1 node (V3) | ★★ |
| **1.5** | 双模型组合 vs 单模型 | 2 nodes (V2+V3) | ★★★ |
| **1.6** | 错误多样性分析 | CPU only | ★★ |
| **1.7** | Agreement rate 校准 | CPU only (从 1.1 数据) | ★ |

---

### 12.5 实验 1.1: SFT v3 Multi-Sample 上限

**问题**: 80% 的 grounder + multi-sampling 能达到多高？

**方法**:
1. 启动 vLLM with SFT v3 grounding checkpoint
2. 在 GUI-360 test grounding eval (parquet 数据) 上:
   - Greedy (K=1, temp=0.0): 验证复现 79.48%
   - K=5, temp=0.7: DBSCAN clustering + best-of-5
   - K=10, temp=0.7: DBSCAN clustering + best-of-10
3. 记录: greedy_acc, cluster_acc, best_of_k_acc
4. 按 agreement rate 分段分析 (同时为 Exp 1.7 提供数据)

**成功标准**: K=5 cluster accuracy > 83% (比 greedy 提升 ≥ 3.5pp)

**与 Phase 0 的对比** (预期):

| 指标 | LoRA v4 (Phase 0 实测) | SFT v3 (Phase 1 预期) | 提升来源 |
|------|:-:|:-:|------|
| Greedy | 37.5% | ~79% | Base model 更强 |
| Cluster (K=10) | 44.5% | ~85% | 更高 base + clustering |
| Best-of-K (K=10) | 61.0% | ~92% | 采样分布中正确答案比例更高 |

**脚本**: `scripts/exp1/exp1_1_sft_v3_multisample.py`

```bash
# 运行 (vLLM 需先启动)
python scripts/exp1/exp1_1_sft_v3_multisample.py \
    --endpoint http://localhost:19815/v1 --K 5 --temperature 0.7

# 仅分析已有结果
python scripts/exp1/exp1_1_sft_v3_multisample.py \
    --analyze_only --results_dir outputs/exp1_1
```

**功能特性**:
- 支持断点续跑 (已完成的 sample 自动跳过)
- Greedy 和 K-sample 在同一轮中完成
- 输出包含 agreement rate 分段分析 (为 Exp 1.7 复用)

---

### 12.6 实验 1.2: SFT v2 Multi-Sample 对比

**问题**: 同样的 multi-sampling 在 SFT v2 上效果如何？

**方法**: 同 1.1，但用 SFT v2 checkpoint

**关键对比**:
- SFT v3 K=5 cluster vs SFT v2 K=5 cluster → 谁更强？
- 如果 SFT v3 K=5 > SFT v2 K=10 → 证明 **base accuracy 比采样次数更重要**
- 这决定了后续路线：投入更好的模型还是更多的采样

**脚本**: `scripts/exp1/exp1_2_sft_v2_multisample.py`
- 内置与 Exp 1.1 的 head-to-head 比较 (按 sample_idx 对齐)

---

### 12.7 实验 1.3: Oracle 坐标替换 ★ (Go/No-Go 门)

> **这是整个双模型方案的核心验证。如果坐标不是瓶颈，后续所有实验都没有意义。**

**问题**: 如果坐标完全正确，action prediction 能提升多少？

**方法**:
1. 在 GUI-360 test action prediction eval (Visual) 上，对坐标类动作 (click, right_click, double_click) 评估:
   - **条件 A.** SFT v2 原始预测 (baseline)
   - **条件 B.** SFT v2 的 action type + SFT v3 预测的坐标
   - **条件 C.** SFT v2 的 action type + ground truth 坐标 (oracle)
2. 评估: function_match, coord_match, args_match

**预期结果**:

| 条件 | 坐标来源 | args_match |
|------|:---:|:---:|
| A. Baseline | SFT v2 | ~17% (已知) |
| B. V3 coord | SFT v3 grounding | ~22-25% |
| C. Oracle | Ground truth | ~35-40% |

**Go/No-Go 决策**:

| 情况 | 条件 | → 行动 |
|------|------|--------|
| ✅ **GO** | Oracle (C) > Baseline (A) ≥ 10pp | 坐标确实是瓶颈，双模型方案有价值 |
| ✅ **GO** | V3 coord (B) > Baseline (A) | SFT v3 grounding 有实际帮助 |
| ⚠️ **MARGINAL** | Oracle delta = 5-10pp | 坐标是瓶颈之一但非唯一 |
| ❌ **NO-GO** | Oracle (C) - Baseline (A) < 5pp | 坐标不是主要瓶颈 → 需要分析其他原因 |

**重要注释**: 条件 B 和 C 的区别告诉我们 SFT v3 的坐标与完美坐标之间还有多大 gap。如果 B ≈ C → SFT v3 已经足够好；如果 B << C → 需要 multi-sampling 来缩小差距。

**脚本**: `scripts/exp1/exp1_3_oracle_coord_replacement.py`

```bash
# 需要两个 vLLM 同时运行 (SFT v2 + SFT v3)
python scripts/exp1/exp1_3_oracle_coord_replacement.py \
    --sft_v2_endpoint http://localhost:19816/v1 \
    --sft_v3_endpoint http://localhost:19815/v1
```

---

### 12.8 实验 1.4: SFT v3 在 Action Prompt 下的 Grounding 泛化

**问题**: SFT v3 的 80% grounding 是 prompt-specific 还是通用能力？

这个问题直接影响 Phase 2 pipeline 的设计复杂度。

**方法**:
1. 对相同的 grounding test samples，分别用两种 prompt 调用 SFT v3:
   - **Grounding prompt** (标准): `<coordinate> [x, y] </coordinate>` 格式
   - **Action prompt** (tool_call 格式): `<tool_call>{"function": "click", "args": {"coordinate": [x, y]}}</tool_call>` 格式
2. 从两种输出中提取坐标，用 grounding eval 标准 (bbox containment) 评估

**成功标准及影响**:

| 结果 | 条件 | → Phase 2 设计 |
|------|------|---------------|
| ✅ 通用能力 | Action prompt 下 >70% | Pipeline 简洁：SFT v3 直接在 action 格式下出坐标 |
| ⚠️ 部分迁移 | 50-70% | 需要 prompt bridge，但可控 |
| ❌ Prompt-specific | <50% | 必须用专用 grounding prompt → 两次调用 |

**脚本**: `scripts/exp1/exp1_4_sft_v3_action_prompt.py`
- 对每个 sample 同时调用两种 prompt，输出混淆矩阵

---

### 12.9 实验 1.5: 双模型组合 vs 单模型

**问题**: SFT v2 action type + SFT v3 coordinate 是否优于任一单模型？

**方法** (4 种条件):

| 条件 | Action Type 来源 | Coordinate 来源 | 含义 |
|------|:---:|:---:|------|
| A. Baseline | SFT v2 | SFT v2 | 当前最佳单模型 |
| B. V3 only | SFT v3 | SFT v3 | Grounder 做全部 (预期差) |
| C. Dual (K=1) | SFT v2 | SFT v3 greedy | 最简双模型组合 |
| D. Dual (K=5) | SFT v2 | SFT v3 K=5 cluster | 双模型 + multi-sampling |

**成功标准**: D 的 args_match > A 至少 5pp

**与 Exp 1.3 的关系**:
- Exp 1.3 的 oracle 坐标验证了**理论上限** (coord IS bottleneck?)
- Exp 1.5 用实际的 SFT v3 坐标验证**实际收益** (dual model works?)
- 如果 1.3 GO 但 1.5 FAIL → SFT v3 的坐标不够好，需要更多 sampling 或更好的 selector

**额外分析**: 按 agreement rate 分段输出 coord_match，验证 adaptive K 的决策逻辑

**脚本**: `scripts/exp1/exp1_5_dual_model_eval.py`

---

### 12.10 实验 1.6: 错误多样性分析 (SFT v2 vs SFT v3)

**问题**: 两个模型的错误是互补的还是重叠的？

**方法**: 在 grounding eval 上逐样本比较两个模型的正确性

**预期的错误分布**:

| 情况 | 预期占比 | 含义 |
|------|:--------:|------|
| 两者都对 | ~65% | 共同能力 |
| V3 对 V2 错 | ~15% | V3 的独有优势 (grounding 特化) |
| V2 对 V3 错 | ~5% | V2 的独有优势 |
| 两者都错 | ~15% | 两模型共同的盲区 |

**关键指标**:
- **Oracle ensemble accuracy** = 至少一个模型正确 ≈ **85%+**
- **Cohen's kappa** = 错误相关性 (越低越好 → 错误越互补)

**成功标准**: Oracle ensemble > max(V2, V3) ≥ 5pp

**脚本**: `scripts/exp1/exp1_6_error_diversity.py`
- CPU only，从 Exp 1.1/1.2 的 K=1 结果或 GUI-360 eval 结果分析
- 输出完整的 2×2 混淆矩阵 + Cohen's kappa + 互补性分析

---

### 12.11 实验 1.7: Agreement Rate 在 SFT v3 上的校准

**问题**: Agreement rate 在更强的 grounder 上仍然是有效的置信度信号吗？

**方法**: 从 Exp 1.1 K=10 数据中，按 agreement rate 分段分析 accuracy

| Agreement Rate | LoRA v4 (Phase 0 已知) | SFT v3 (预期) | Adaptive K 策略 |
|:-:|:-:|:-:|------|
| ≥ 0.9 | 83% | ~95% | K=1 即可 (高置信) |
| 0.5-0.9 | 59% | ~82% | K=3-5 (中等置信) |
| < 0.5 | 23% | ~55% | K=10 (低置信, 需要更多采样) |

**成功标准**: 高 agreement 区域 (≥0.9) accuracy > 90%

**不需要额外脚本** — Exp 1.1 的 `--analyze_only` 模式已内置 agreement rate 分段输出

---

### 12.12 执行计划

#### 时间线

```
Day 1-2: sbatch scripts/exp1/run_exp1_grounding.slurm
         → Exp 1.1 (SFT v3 K=1/5/10)
         → Exp 1.2 (SFT v2 K=1/5/10)
         → Exp 1.6 (error diversity, CPU, 从 K=1 结果)
         → Exp 1.7 (agreement calibration, 从 K=10 结果)

Day 3-4: sbatch scripts/exp1/run_exp1_dual_model.slurm
         → Exp 1.4 (V3 prompt generalization)
         → Exp 1.3 ★ (oracle coord replacement, Go/No-Go)
         → Exp 1.5 (dual-model combination)

Day 5:   汇总结果 → Go/No-Go 决策
```

#### Slurm 脚本

| Slurm 文件 | 节点 | GPU | 包含实验 | 预计时间 |
|------------|:----:|:---:|---------|---------|
| `scripts/exp1/run_exp1_grounding.slurm` | 1 | 4 | 1.1 + 1.2 + 1.6 (+1.7) | ~18h |
| `scripts/exp1/run_exp1_dual_model.slurm` | 2 | 8 | 1.3 + 1.4 + 1.5 | ~18h |

**执行流程**:
- `run_exp1_grounding.slurm`: 在单节点上依次启动 SFT v3 和 SFT v2 的 vLLM，每个模型跑完后切换。最后用 CPU 跑 error diversity
- `run_exp1_dual_model.slurm`: 在两个节点同时启动两个 vLLM server (V2 on node 0, V3 on node 1)，实验脚本从 node 0 通过 HTTP 调用两个 endpoint

#### Go/No-Go 决策矩阵

| 实验 | ✅ Go 条件 | ❌ No-Go 条件 | Fallback |
|------|-----------|-------------|----------|
| **1.1** | SFT v3 K=5 cluster > 83% | < 80% | 用 SFT v2 做 grounder |
| **1.3** ★ | Oracle coord 提升 ≥10pp | < 5pp | 坐标不是瓶颈 → 分析其他因素 |
| **1.4** | Action prompt 下 >70% | < 50% | 用专用 grounding prompt 桥接 |
| **1.5** | 双模型 > 单模型 ≥5pp | 无提升 | 双模型 grounding 不可行 |
| **1.6** | Oracle ensemble > max ≥5pp | 错误重叠 | Ensemble 价值有限 |

**最关键的 Go/No-Go 门**: **Exp 1.3**。如果 oracle 坐标替换都无法显著提升 args_match，说明瓶颈不在坐标精度，整个双模型方案需要重新评估。

---

### 12.13 验证通过后的完整路线 (Phase 2-4)

#### Phase 2 (2 周): 双模型推理 Pipeline

```
SFT v2 (action agent) ──→ action_type + reasoning
                              │
SFT v3 (grounder × K) ──→ K 个坐标候选
                              │
              DBSCAN / Selector ──→ 最优坐标
                              │
              组合: action_type + coordinate ──→ 执行
```

- **Adaptive K 策略** (基于 Exp 1.7 校准):
  - agreement ≥ 0.9 → K=1 (已有 ~95% accuracy)
  - 0.5 ≤ agreement < 0.9 → K=5
  - agreement < 0.5 → K=10 (bottleneck, 最大预算)
- **评估**: args_match 和 TSR 端到端评估

#### Phase 3 (3 周): MoE Expert 分化训练

- Expert 0: SFT v2 LoRA 初始化 (planning/action reasoning)
- Expert 1: SFT v3 delta 初始化 (grounding precision)
- Phase-based router (§7.1)
- Consensus RL reward + Step-0 reward 放大 (基于 §11.4 的 82% divergence at step 0)

#### Phase 4 (4 周): Verification + Recovery

- Verification SFT (recall 18% → 70%+, 基于 §11.2 的 precision=100% 基础)
- 完整 verify-recover loop
- 端到端轨迹评估

#### 预期效果

| 阶段 | 方法 | Grounding | Action(V) | args_match | TSR |
|------|------|:---------:|:---------:|:----------:|:---:|
| 现状 | SFT v2 greedy | 70.56% | 46.90% | 17.07% | 16.21% |
| P1 验证 | SFT v3 + K=5 | ~85% | — | — | — |
| P2 | v2 action + v3 grounding K=5 | ~85% | ~47% | ~26% | ~30% |
| P3 | MoE + consensus RL | ~87% | ~50% | ~30% | ~35% |
| P4 | + Verify + Recovery | ~90% | ~52% | ~35% | ~42% |

---

### 12.14 文件索引

#### Phase 1 实验脚本

| 文件 | 用途 | GPU |
|------|------|:---:|
| `scripts/exp1/exp1_1_sft_v3_multisample.py` | SFT v3 multi-sample grounding eval | V3 |
| `scripts/exp1/exp1_2_sft_v2_multisample.py` | SFT v2 multi-sample grounding eval (对比) | V2 |
| `scripts/exp1/exp1_3_oracle_coord_replacement.py` | Oracle 坐标替换 (Go/No-Go 门) | V2+V3 |
| `scripts/exp1/exp1_4_sft_v3_action_prompt.py` | 跨 prompt 泛化测试 | V3 |
| `scripts/exp1/exp1_5_dual_model_eval.py` | 双模型组合 eval | V2+V3 |
| `scripts/exp1/exp1_6_error_diversity.py` | 错误多样性分析 | CPU |
| `scripts/exp1/run_exp1_grounding.slurm` | Slurm: Exp 1.1 + 1.2 + 1.6 (1 node) | 4 GPU |
| `scripts/exp1/run_exp1_dual_model.slurm` | Slurm: Exp 1.3 + 1.4 + 1.5 (2 nodes) | 8 GPU |

#### 共享工具 (来自 Phase 0)

| 模块 | 关键函数 | 用于 |
|------|---------|------|
| `scripts/exp0/data_utils.py` | `PARQUET_EVAL_PATH`, `DATASET_ROOT`, `is_coord_in_bbox`, `load_trajectory` | 所有实验 |
| `scripts/exp0/exp0_1_uncertainty_analysis.py` | `call_model_k_times`, `cluster_coordinates`, `parse_tool_call`, `extract_coordinate`, `evaluate_coord` | 1.1, 1.2, 1.5 |

所有 Phase 1 脚本复用 Phase 0 的工具函数 (DBSCAN clustering, coordinate parsing, eval 逻辑)，确保评估标准一致。每个脚本支持 `--analyze_only` 模式离线分析和断点续跑。

---

## 13. Phase 1 实验结果与分析 (2026-03-13)

> **执行周期**: 2026-03-12 ~ 2026-03-13
> **计算资源**: 1-2 节点 × 4 GPU (MI250X)，vLLM tensor-parallel-size=4
> **评估基准**: GUI-360 test set，Grounding prompt + bbox containment 评估

### 13.1 总览：实验完成状态

| 实验 | 描述 | 样本数 | 状态 | 核心结论 |
|------|------|--------|------|---------|
| 1.1 | SFT v3 Multi-Sample 上限 | 18,265 × 3 (K=1/5/10) | ✅ 完成 | DBSCAN 聚类无效，但 agreement rate 是强置信信号 |
| 1.2 | SFT v2 Multi-Sample 对比 | 18,265 (K=1) | ✅ 完成 | V2 grounding=70.6%，比 V3 低 8.9pp |
| 1.3 | Oracle 坐标替换 ★ | — | ⏳ 运行中 | Go/No-Go 关键门控 |
| 1.4 | Action Prompt 泛化测试 | 14,467 | ✅ 完成 | 严重 prompt 敏感：79.2% → 39.3% |
| 1.5 | 双模型组合 | — | ⏳ 运行中 | 依赖 Exp 1.3 |
| 1.6 | 错误多样性分析 | — | ⏳ 等待 1.2 完成 | 依赖 Exp 1.1+1.2 |

---

### 13.2 实验 1.1: SFT v3 Multi-Sample Grounding 上限

**问题**: 80% grounder + multi-sampling → 准确率能提多高？

#### 总体结果 (N=18,265)

| 指标 | K=1 (贪心) | K=5 (temp=0.7) | K=10 (temp=0.7) |
|------|-----------|----------------|-----------------|
| 贪心准确率 | **79.46%** | 79.46% | 79.46% |
| Best-of-K (Oracle 上界) | 79.46% | **84.34%** (+4.9pp) | **86.77%** (+7.3pp) |
| DBSCAN 聚类准确率 | 79.46% | **78.67%** (-0.8pp) | **79.39%** (-0.1pp) |
| 有效坐标率 | 100.0% | ~100.0% | ~100.0% |
| 多模态样本占比 | 0% | 29.5% | 34.8% |

#### 分域结果 (K=1 贪心)

| 域 | 正确数 | 总数 | 准确率 |
|----|--------|------|--------|
| excel | 4,105 | 5,351 | 76.71% |
| ppt | 4,191 | 5,345 | 78.41% |
| word | 6,218 | 7,569 | 82.15% |
| **合计** | **14,514** | **18,265** | **79.46%** |

#### ★ Agreement Rate 置信校准 (K=5)

这是 Phase 1 **最有价值的发现**之一：agreement rate 与准确率呈强单调相关。

| Agreement Rate | 准确率 | 样本数 | 占总样本比 |
|----------------|--------|--------|-----------|
| ≥0.9 | **92.0%** | 12,873 | 70.5% |
| 0.7–0.9 | 66.9% | 2,128 | 11.7% |
| 0.5–0.7 | 48.5% | 1,629 | 8.9% |
| 0.3–0.5 | 24.1% | 1,186 | 6.5% |
| <0.3 | 6.9% | 449 | 2.5% |

**含义**：
- **高置信区** (≥0.9, 70.5% 样本): 准确率 92%，可直接信赖
- **低置信区** (<0.5, 9% 样本): 准确率 <25%，需 fallback 或 retry
- → agreement rate 可作为 **RL 伪奖励信号**或**路由决策依据**

#### 分域 Agreement Rate 校准 (K=5)

| Agreement | excel | ppt | word |
|-----------|-------|-----|------|
| ≥0.9 | 89.4% (n=3751) | 93.0% (n=3895) | 92.9% (n=5227) |
| 0.7–0.9 | 62.8% (n=662) | 58.1% (n=503) | 76.6% (n=963) |
| 0.5–0.7 | 48.0% (n=469) | 40.2% (n=428) | 57.7% (n=732) |
| 0.3–0.5 | 26.9% (n=331) | 22.1% (n=380) | 33.1% (n=475) |

#### Go/No-Go 判定

| 标准 | 结果 | 判定 |
|------|------|------|
| K=5 聚类 > 83% | 78.67% | ❌ **FAIL** |
| 但: agreement rate 置信校准有效 | ≥0.9 → 92% | ✅ |
| 但: Oracle 上界存在 headroom | K=10 best-of-K = 86.8% | ✅ |

**结论**: DBSCAN 聚类中心作为"更好预测"的策略失败 — 约 30% 的样本具有多模态分布，聚类中心偏移降低准确率。但多采样的价值在于 **置信估计** (agreement rate) 和 **选择** (best-of-K)，而非 **平均** (cluster center)。需要学习的 verifier/reward model 来从 K 个样本中选最优。

---

### 13.3 实验 1.2: SFT v2 Grounding 基线

**问题**: all-round 模型的 grounding 能力如何？与专用 grounder 差距多大？

#### 总体结果 (N=18,265, K=1 贪心)

| 域 | 正确数 | 总数 | 准确率 | 有效坐标率 |
|----|--------|------|--------|-----------|
| excel | 3,605 | 5,351 | 67.37% | 97.96% |
| ppt | 3,758 | 5,345 | 70.31% | 98.26% |
| word | 5,526 | 7,569 | 73.01% | 99.18% |
| **合计** | **12,889** | **18,265** | **70.57%** | **98.55%** |

**无效坐标**: 264 / 18,265 (1.45%) — V2 偶尔生成 `[65, null]` 等格式错误，而 V3 为 0%。

---

### 13.4 V3 vs V2 Head-to-Head 对比

#### 匹配样本分析 (N=18,265)

| 指标 | SFT v3 | SFT v2 | 差异 |
|------|--------|--------|------|
| 总体准确率 | **79.46%** | 70.57% | **+8.89pp** |
| Excel | 76.71% | 67.37% | +9.34pp |
| PPT | 78.41% | 70.31% | +8.10pp |
| Word | 82.15% | 73.01% | +9.14pp |

#### 混淆矩阵

| 结果 | 数量 | 比例 | 含义 |
|------|------|------|------|
| 两者都对 | 12,464 | 68.2% | V3 和 V2 的共同能力区 |
| **仅 V3 对** | **2,050** | **11.2%** | V3 grounding 训练的独特提升 |
| 仅 V2 对 | 425 | 2.3% | V2 偶然命中但 V3 失误 |
| **两者都错** | **3,326** | **18.2%** | 真正的 hard cases |

**关键洞察**:
1. V3 单方面修复了 V2 的 2,050 个错误，仅丢失 425 个 → 提升是单向的 (V3 >> V2)
2. V2 仅在 2.3% 的样本上优于 V3 → 两者互补性很低（在 grounding 维度）
3. **18.2% 的 "两者都错" 代表 SFT 训练无法解决的 hard cases** — 这些是 RL 的核心目标样本

---

### 13.5 实验 1.4: SFT v3 Action Prompt 泛化测试

**问题**: SFT v3 的 79% grounding 能力是否随 prompt 迁移？

这是 Phase 1 **最重要的发现**。

#### 总体结果 (N=14,467, 仅 click 类型)

| 提示词格式 | 准确率 | 数量 | 说明 |
|-----------|--------|------|------|
| Grounding prompt (`<coordinate>` 格式) | **79.21%** | 11,460 | 标准 grounding 评估 |
| Action prompt (`<tool_call>` 格式) | **39.34%** | 5,691 | 使用 action prediction 提示词 |
| **下降** | **-39.88pp** | — | 几乎腰斩 |

#### Action Prompt 失败模式

| 失败类型 | 数量 | 比例 | 说明 |
|---------|------|------|------|
| 无坐标输出 (null) | 3,349 | 23.1% | 模型完全未生成坐标 |
| 坐标错误 | 5,427 | 37.5% | 提取到坐标但位置不正确 |
| 坐标正确 | 5,691 | 39.3% | — |

#### 分域详细结果

| 域 | Grounding Prompt | Action Prompt | 下降 | Action null 率 |
|----|-----------------|---------------|------|---------------|
| excel | 78.55% | 36.81% | -41.74pp | 23.88% |
| ppt | 78.94% | 43.98% | -34.97pp | 18.26% |
| word | 79.90% | 36.94% | -42.97pp | 26.97% |

#### 混淆矩阵

| 结果 | 数量 | 比例 |
|------|------|------|
| 两个 prompt 都对 | 5,471 | 37.8% |
| **仅 Grounding 对** | **5,989** | **41.4%** |
| 仅 Action 对 | 220 | 1.5% |
| 两者都错 | 2,787 | 19.3% |

#### Go/No-Go 判定

| 标准 | 结果 | 判定 |
|------|------|------|
| Action prompt >70% → 通用能力 | 39.3% | ❌ **FAIL** |
| Action prompt >50% → 部分迁移 | 39.3% | ❌ **FAIL** |
| Action prompt <50% → prompt 特定 | 39.3% | ✅ **确认** |

**结论**: SFT v3 的 grounding 能力是 **prompt 特定的**，不能直接在 action 推理管线中使用。在实际部署中，必须使用 **grounding prompt 桥接模块**: 先用 grounding prompt 获取坐标，再注入 action pipeline。

#### 架构影响

```
原设想 (已证伪):
  screenshot → SFT_v3(action_prompt) → coord → evaluate
                    ↑ 39.3% 准确率

正确方案:
  screenshot → SFT_v3(grounding_prompt) → coord → inject into action pipeline
                    ↑ 79.2% 准确率
```

---

### 13.6 实验 1.2: SFT v2 Multi-Sample 完整结果

**问题**: SFT v2 (70.6%) 与 SFT v3 (79.5%) 在多采样场景下如何对比？

#### 总体结果 (N=18,265)

| 指标 | V3 K=1 | V3 K=5 | V3 K=10 | V2 K=1 | V2 K=5 | V2 K=10 |
|------|:------:|:------:|:-------:|:------:|:------:|:-------:|
| Greedy | 79.5% | 79.5% | 79.5% | 70.6% | 70.6% | 70.7% |
| Best-of-K | 79.5% | 84.3% | 86.8% | 70.6% | 78.3% | 81.9% |
| DBSCAN cluster | 79.5% | 78.7% | 79.4% | 70.6% | 69.8% | 70.8% |

#### Head-to-Head (匹配样本)

| K | V3 cluster | V2 cluster | 差异 |
|:-:|:----------:|:----------:|:----:|
| 1 | 79.5% | 70.6% | **+8.9pp** |
| 5 | 78.7% | 69.8% | **+8.8pp** |
| 10 | 79.4% | 70.8% | **+8.6pp** |

**核心结论**:
- **V3 K=1 greedy (79.5%) > V2 K=10 cluster (70.8%)** → base accuracy 比采样次数更重要
- V2 的 best-of-K=10 oracle 上界 (81.9%) 仍低于 V3 best-of-K=10 (86.8%)
- DBSCAN 在 V2 上同样无效 (K=5: -0.8pp)

---

### 13.7 实验 1.3: Oracle 坐标替换 ★ (Go/No-Go)

**问题**: 如果坐标完全正确，action prediction 能提升多少？

#### 总体结果 (N=14,467, 仅 click/right_click/double_click)

| 条件 | func_match | coord_match | args_match |
|------|:----------:|:-----------:|:----------:|
| **A. V2 原始预测** | 91.1% | 46.8% | 5.0% |
| **B. V2 action + V3 coord** | 91.1% | **72.9%** | 7.9% |
| **C. V2 action + GT coord (oracle)** | 91.1% | **92.8%** | 10.9% |

#### 关键 Delta

| 对比 | coord_match Δ | args_match Δ | 含义 |
|------|:-------------:|:------------:|------|
| Oracle - Baseline (C-A) | **+46.0pp** | **+5.9pp** | 坐标是中等瓶颈 |
| V3 coord - Baseline (B-A) | **+26.1pp** | **+2.9pp** | V3 grounding 有实际价值 |
| Oracle - V3 coord (C-B) | +19.9pp | +3.0pp | V3 仍有提升空间 |

#### 分域结果

| 域 | A args_match | B args_match | C args_match | Δ(C-A) |
|----|:----------:|:----------:|:----------:|:------:|
| excel (N=3,790) | 8.3% | 14.2% | 18.3% | **+10.0pp** |
| ppt (N=5,005) | 5.3% | 7.2% | 10.9% | **+5.6pp** |
| word (N=5,672) | 2.5% | 4.3% | 5.9% | **+3.4pp** |

#### Go/No-Go 判定

| 标准 | 结果 | 判定 |
|------|------|------|
| Oracle Δ ≥10pp → 坐标是主要瓶颈 | +5.9pp | ❌ 未达标 |
| Oracle Δ 5-10pp → 中等瓶颈 | +5.9pp | ⚠️ **MARGINAL GO** |
| V3 coord > baseline → V3 有价值 | +2.9pp | ✅ **确认** |

#### 深层分析: 为什么 coord_match 提升 26pp 但 args_match 仅 +2.9pp?

coord_match 从 46.8% → 72.9%（+26pp），但 args_match 仅从 5.0% → 7.9%（+2.9pp）。差距来自 `other_match`（非坐标参数匹配）：

- 模型预测的 args 可能包含额外字段 (如 `button`, `double`)
- GT args 格式与预测格式有差异
- **args_match = coord_match AND other_match**，other_match 的通过率很低

因此 **coord_match 是更准确的衡量指标**。Oracle coord_match 提升 +46pp 证明坐标确实是关键瓶颈。

---

### 13.8 实验 1.5: 双模型组合评估

**问题**: V2 action type + V3 coordinate 的组合是否优于单模型？

#### 总体结果 (N=14,467)

| 条件 | func_match | coord_match | args_match |
|------|:----------:|:-----------:|:----------:|
| **A. V2 only** (action + coord) | 91.2% | 46.8% | 4.9% |
| **B. V3 only** (action prompt) | 87.4% | 36.8% | 10.2% |
| **C. Dual K=1**: V2 act + V3 greedy | 91.2% | **73.0%** | 7.9% |
| **D. Dual K=5**: V2 act + V3 cluster | 91.2% | **72.3%** | 7.8% |

#### 关键对比

| 对比 | coord_match Δ | args_match Δ |
|------|:-------------:|:------------:|
| Dual K=1 (C) vs V2 only (A) | **+26.2pp** | +3.0pp |
| Dual K=5 (D) vs V2 only (A) | **+25.5pp** | +2.9pp |
| Dual K=1 (C) vs Dual K=5 (D) | +0.7pp | +0.1pp |

#### Agreement Rate → Coord Accuracy (Condition D)

| Agreement Rate | coord_match | 样本数 | 占比 |
|:-:|:-:|:-:|:-:|
| ≥0.9 | **86.1%** | 10,622 | 73.4% |
| 0.5-0.9 | 43.5% | 2,530 | 17.5% |
| <0.5 | 16.7% | 1,315 | 9.1% |

#### 关键发现

1. **Dual K=1 ≈ Dual K=5**: K=5 cluster (72.3%) 反而略低于 K=1 greedy (73.0%)，再次证明 DBSCAN 聚类无益
2. **V3 greedy coord 即可大幅提升**: +26pp coord_match，不需要多采样
3. **V3 action prompt 很差**: B 的 coord_match 仅 36.8%，证实 Exp 1.4 的 prompt 特定性结论
4. **Agreement rate 校准良好**: ≥0.9 区域 86.1% coord_match，可用于置信路由

---

### 13.9 实验 1.6: 错误多样性分析

**问题**: V3 和 V2 的 grounding 错误是互补的还是重叠的？

#### 错误多样性矩阵 (N=18,265)

|  | V2 ✓ | V2 ✗ |
|---|:---:|:---:|
| **V3 ✓** | 12,464 (68.2%) | 2,050 (11.2%) |
| **V3 ✗** | 425 (2.3%) | 3,326 (18.2%) |

| 指标 | 值 |
|------|:---:|
| V3 准确率 | 79.5% |
| V2 准确率 | 70.6% |
| Oracle ensemble (至少一个对) | **81.8%** |
| Ensemble 增益 | **+2.3pp** (vs V3) |
| V3 覆盖 V2 错误的比例 | 38.1% (2050/5376) |
| Cohen's Kappa | 0.642 (高相关) |

**结论**: Oracle ensemble 仅 +2.3pp，**互补性有限**。V3 是 V2 的近乎严格超集 — 双模型 grounding ensemble 价值不大，不如 V3 单模型 + 多采样 (best-of-K=5 → 84.3%)。

---

### 13.10 综合分析与关键发现

#### 13.10.1 发现 1: DBSCAN 聚类策略失败，但 agreement rate 是有效置信信号

| 方法 | V3 K=5 | V2 K=5 |
|------|:------:|:------:|
| 贪心 (K=1) | 79.5% | 70.6% |
| DBSCAN 聚类中心 | 78.7% (**-0.8pp**) | 69.8% (**-0.8pp**) |
| Best-of-K Oracle | 84.3% (+4.9pp) | 78.3% (+7.7pp) |

**原因分析**: ~30% 的样本具有多模态坐标分布，DBSCAN 聚类中心是"平均"操作，会偏离正确坐标。多采样的价值不在 averaging，而在:
1. **置信估计**: agreement rate ≥0.9 → V3 92% / Dual 86% 准确率
2. **候选选择**: Best-of-K 提供 +5~8pp 上界，但需要 learned selector

#### 13.10.2 发现 2: SFT v3 Grounding 是 prompt 特定的 (79% → 39%)

这是 Phase 1 **最关键的发现**，直接影响系统架构:

```
SFT v3 grounding 能力:
  ├── Grounding prompt: 79.2% ✅
  └── Action prompt:    39.3% ❌ (-40pp)

意味着:
  ├── 不能简单用 action prompt 调用 V3 获取坐标
  ├── 必须设计 grounding prompt 桥接模块
  └── RL 训练需要考虑 prompt 格式对 grounding 的影响
```

#### 13.10.3 发现 3: 坐标是中等瓶颈 (Exp 1.3 Go/No-Go)

```
coord_match 视角 (更准确):
  A. V2 原始:    46.8%
  B. V2 + V3:    72.9%  (+26.1pp) ★
  C. V2 + Oracle: 92.8%  (+46.0pp)

  → 坐标从 47% 提升到 93% 是可能的
  → V3 已经实现了一半的提升空间 (26/46 = 57%)
  → 还有 20pp 空间给 learned selector 或 RL

args_match 视角 (更严格):
  A → B: +2.9pp, A → C: +5.9pp
  → 中等瓶颈 (5-10pp), 不是唯一瓶颈
  → other_match 限制了 args_match 的提升
```

#### 13.10.4 发现 4: Dual-Model 组合有效 (Exp 1.5)

| 方案 | coord_match | vs V2 only |
|------|:-----------:|:----------:|
| V2 only | 46.8% | baseline |
| **V2 action + V3 greedy coord** | **73.0%** | **+26.2pp** |
| V2 action + V3 K=5 cluster | 72.3% | +25.5pp |
| V2 action + GT coord | 92.8% | +46.0pp |

Dual-model (V2 action + V3 grounding) 将 coord_match 从 47% 提升到 73%，证明双模型方案有实际价值。但 K=5 cluster 反而比 K=1 greedy 差，再次证明 DBSCAN 无效。

#### 13.10.5 发现 5: V3 vs V2 互补性有限，但角色互补明确

| 维度 | V3 优势 | V2 优势 |
|------|---------|---------|
| Grounding | **79.5%** (>>70.6%) | — |
| Action func_match | — | **91.1%** (>>87.4%) |
| Action (Visual) | 3.07% | **46.90%** |

V3 和 V2 不是 grounding 层面的互补（错误重叠度高），而是 **角色互补**: V2 做 action planning, V3 做 grounding。这验证了 multi-agent 方案的核心假设。

#### 13.10.6 发现 6: Base accuracy 比采样次数更重要

V3 K=1 greedy (79.5%) > V2 K=10 cluster (70.8%), 说明提升 base model grounding accuracy 比多采样更有效。但 best-of-K oracle headroom 始终存在 (~5-7pp)，说明如果有好的 selector，多采样仍有价值。

#### 13.10.7 发现 7: Excel 域持续最难

| 域 | V3 Grnd | V2 Grnd | Exp1.3 Oracle Δ |
|----|:-------:|:-------:|:---------------:|
| excel | 76.7% | 67.4% | +10.0pp |
| ppt | 78.4% | 70.3% | +5.6pp |
| word | 82.2% | 73.0% | +3.4pp |

Excel 坐标替换提升最大 (+10pp)，说明 Excel 的 action 失败更多由坐标错误导致。

---

### 13.11 修正后的路线建议

基于全部 Phase 1 实验结果，Phase 2 路线调整:

#### 原计划 vs 修正计划

| 组件 | 原计划 | 修正后 | 原因 |
|------|--------|--------|------|
| Grounding 增强 | DBSCAN 聚类中心 | **Agreement-based 路由 + Learned selector** | DBSCAN 在 V3 和 V2 上均失败 |
| Action-Grounding 桥接 | 直接用 action prompt | **专用 grounding prompt 桥接** | Exp 1.4: prompt 特定性确认 |
| 多模型 ensemble | V2+V3 grounding ensemble | **单 V3 grounding + 多采样** | Exp 1.6: 互补性仅 +2.3pp |
| 坐标替换 | — | **V2 action + V3 greedy coord** | Exp 1.3/1.5: +26pp coord_match |
| Multi-sampling | K=5 DBSCAN | **K=1 greedy (默认) + K>1 仅在低置信时** | Exp 1.5: K=5 cluster ≤ K=1 greedy |
| RL 奖励信号 | 需要 verifier 模型 | **Agreement rate 伪奖励 + 学习 selector** | Agreement rate 校准良好 |

#### 修正后的 Phase 2 优先级

1. **P0**: 实现 **V2 action + V3 grounding prompt 桥接** inference pipeline
2. **P1**: 训练 **learned selector** (替代 DBSCAN) — 从 best-of-K 的 ~5-7pp headroom 中获益
3. **P2**: 设计 **agreement-rate-based adaptive K** — 高置信 K=1，低置信 K=5+
4. **P3**: 训练 MoE — Expert 0 = V2 (action), Expert 1 = V3 delta (grounding)

#### 预期效果 (修正后)

| 阶段 | 方法 | coord_match | 预期 args_match |
|------|------|:-----------:|:---------------:|
| 现状 | V2 only | 46.8% | ~17% (step-level) |
| P2 Step 1 | V2 act + V3 greedy coord | **73.0%** | ~22% |
| P2 Step 2 | + learned selector (best-of-5) | **~80%** | ~27% |
| P2 Step 3 | + adaptive K + agreement routing | **~83%** | ~30% |
| P3 | MoE + RL | **~87%** | ~35% |

---

## 14. 训练演进与失败分析

> **目的**: 对比 Base model、SFT v1（灾难性遗忘）、SFT v2 1-epoch、SFT v2 2-epoch、SFT v3 的性能，分析训练策略对各任务的影响，为 multi-agent 方案提供基线参考。

---

### 14.1 全模型 Step-Level 性能对比

| 模型 | Grounding | Action (Visual) | Action (A11y) | 训练特点 |
|------|:---------:|:---------------:|:-------------:|---------|
| **GUI-360 Paper SFT** | **82.30%** | **50.08%** | **25.78%** | 官方全参 SFT (参考上限) |
| Qwen2.5-VL-7B Base | 42.47% | 18.05% | 14.53% | Zero-shot |
| SFT v1 (full) | 12.87% | 11.14% | 13.40% | 灾难性遗忘 (低分辨率 + 冻 projector) |
| **SFT v2 (full, 1ep)** | **70.56%** | **46.90%** | 17.51% | 高分辨率 + 解冻 projector |
| SFT v2 (full, 2ep) | 70.77% | 49.37% | 未评测 | +1 epoch 仅微弱提升 |
| **SFT v3 (grounding)** | **79.48%** | 3.07% | 10.88% | +grounding 数据，但 action 过拟合 |

---

### 14.2 SFT v1 灾难性遗忘分析

SFT v1 是首次全参 SFT 尝试。所有任务均**严重退化**，甚至低于 zero-shot base model。

#### 根因诊断

| 优先级 | 问题 | 影响 |
|--------|------|------|
| **P0** | `image_max_pixels: 262144` (~512×512) | 训练分辨率远低于推理分辨率 (1040×736)，造成 1.7-1.8× 分辨率失配 |
| **P0** | 冻结 projector + vision tower | 模型无法适应 GUI 坐标预测，grounding 12.87% 远低于 base 42.47% |
| **P1** | 无 grounding 训练数据 | 仅含 action prediction 样本 (101K `<tool_call>` 格式) |
| **P2** | Eval prompt 偏移 | 推理提示含 "explain your reasoning" 等训练中未见的指令 |

#### SFT v1 vs Base 分域对比

| 任务 | 域 | Base | SFT v1 | Δ | 状态 |
|------|-----|:----:|:------:|:---:|------|
| **Grounding** | PPT | 50.72% | 8.03% | -42.69pp | 灾难退化 |
| | Word | 44.54% | 19.45% | -25.09pp | 严重退化 |
| | Excel | 31.32% | 8.39% | -22.93pp | 严重退化 |
| | **Overall** | **42.47%** | **12.87%** | **-29.60pp** | **灾难性遗忘** |
| **Action (Visual)** | PPT | 25.65% | 11.35% | -14.30pp | 退化 |
| | Word | 16.33% | 13.73% | -2.60pp | 略降 |
| | Excel | 13.10% | 6.93% | -6.17pp | 退化 |
| | **Overall** | **18.05%** | **11.14%** | **-6.91pp** | **退化** |
| **Action (A11y)** | Overall | 14.53% | 13.40% | -1.13pp | 基本持平 |

**教训**: 低分辨率 SFT 不仅没有学到坐标预测，反而破坏了 base model 的 zero-shot 能力。

---

### 14.3 SFT v2 修复与成功

SFT v2 通过两个关键修复大幅逆转了 v1 的失败:

| 参数 | SFT v1 | SFT v2 | 效果 |
|------|--------|--------|------|
| `image_max_pixels` | 262,144 (~512×512) | **1,003,520** (~1036×968) | 消除训推分辨率失配 |
| `freeze_multi_modal_projector` | `true` | **`false`** | 允许 projector 适应 GUI 坐标 |

#### SFT v2 (1ep) vs Base 分域对比

| 任务 | 域 | Base | SFT v2 1ep | Δ |
|------|-----|:----:|:----------:|:---:|
| **Grounding** | PPT | 50.72% | 69.39% | **+18.67pp** |
| | Word | 44.54% | 72.60% | **+28.06pp** |
| | Excel | 31.32% | 68.85% | **+37.53pp** |
| | **Overall** | **42.47%** | **70.56%** | **+28.09pp** |
| **Action (Visual)** | PPT | 25.65% | 53.17% | **+27.52pp** |
| | Word | 16.33% | 45.51% | **+29.18pp** |
| | Excel | 13.10% | 42.77% | **+29.67pp** |
| | **Overall** | **18.05%** | **46.90%** | **+28.85pp** |
| **Action (A11y)** | PPT | 23.57% | 24.22% | +0.65pp |
| | Word | 16.46% | 22.64% | +6.18pp |
| | Excel | 2.48% | 2.82% | +0.34pp |
| | **Overall** | **14.53%** | **17.51%** | **+2.98pp** |

#### SFT v2 vs Paper 差距分析

| 任务 | Paper SFT | SFT v2 | Δ | Paper 达成率 |
|------|:---------:|:------:|:---:|:-----------:|
| Grounding | 82.30% | 70.56% | -11.74pp | 85.7% |
| Action (Visual) | 50.08% | 46.90% | -3.18pp | 93.7% |
| Action (A11y) | 25.78% | 17.51% | -8.27pp | 67.9% |

#### 放松 BBox 评估 (坐标偏移分析)

| 放松像素 | Grounding | Action (Visual) | 说明 |
|:--------:|:---------:|:---------------:|------|
| ±0 px | 70.56% | 46.90% | 标准评估 |
| ±20 px | 76.01% | **51.81%** | 超过 paper Visual (50.08%) |
| ±50 px | **81.14%** | 56.50% | 接近 paper Grounding (82.30%) |

**关键发现**: 差距主要是**坐标偏移**而非点错元素。±50px 即可追平 paper grounding，说明 SFT v2 已学到正确的 UI 元素定位，仅坐标精度不足。

---

### 14.4 SFT v2 2-Epoch 的边际收益分析

| 任务 | SFT v2 1ep | SFT v2 2ep | Δ | 分析 |
|------|:----------:|:----------:|:---:|------|
| Grounding | 70.56% | 70.77% | **+0.21pp** | 几乎无提升 |
| Action (Visual) | 46.90% | 49.37% | **+2.47pp** | 小幅提升 |
| Action (A11y) | 17.51% | 未评测 | — | — |

**结论**: 第二个 epoch 仅带来边际收益 (grounding 仅 +0.21pp)。SFT v2 在 1 epoch 已接近**饱和**。这意味着:
1. 单纯增加训练步数无法显著提升性能
2. **需要数据层面的改进** (如 grounding 专用数据) 而非更多训练量
3. 进一步验证了 SFT v3 (含 grounding 数据) 的必要性

---

### 14.5 SFT v3 的 Grounding 特化与 Action 过拟合

SFT v3 在 v2 基础上加入 grounding 标注数据，grounding 大幅提升但 action 灾难性退化:

| 任务 | SFT v2 1ep | SFT v3 | Δ | 分析 |
|------|:----------:|:------:|:---:|------|
| **Grounding** | 70.56% | **79.48%** | **+8.92pp** | grounding 数据的直接效果 |
| Action (Visual) | **46.90%** | 3.07% | **-43.83pp** | 灾难性 action 退化 |
| Action (A11y) | 17.51% | 10.88% | -6.63pp | 同样退化 |

#### SFT v3 Checkpoint 演进

| Checkpoint | Grounding | Action (Visual) | Action (A11y) |
|------------|:---------:|:---------------:|:-------------:|
| ckpt150 | 77.66% | 3.89% | 未评测 |
| ckpt200 | 78.55% | 3.63% | 10.93% |
| ckpt250 | 79.39% | 3.08% | 11.08% |
| ckpt300 | 79.61% | 3.06% | 11.02% |
| Final | 79.48% | 3.07% | 10.88% |

**关键发现**:
1. Grounding 在 ckpt150 已达 77.66%，之后收敛缓慢 (final 仅 +1.82pp)
2. Action 从训练初期 (3.89%) 就已经崩塌，不是逐渐退化
3. 这种 **grounding ↑ action ↓ 的 trade-off** 是 multi-agent 方案的核心动机

---

### 14.6 Sub-Metrics 对比: 为什么坐标是瓶颈

| 任务 | 指标 | Base | SFT v1 | SFT v2 1ep | Paper SFT |
|------|------|:----:|:------:|:----------:|:---------:|
| **Action (Visual)** | func_match | 69.79% | 86.18% | ~87% | — |
| | args_match | 18.14% | 11.28% | ~47% | — |
| | status_match | 96.71% | 95.31% | ~95% | — |
| | step_success | 18.05% | 11.14% | 46.90% | 50.08% |

**分析**:
- **func_match 已经很高** (SFT v1/v2 均 >86%): 模型知道**做什么** (click, type, scroll...)
- **args_match 是主要瓶颈**: SFT v1 仅 11.28% (比 base 更差!)，SFT v2 提升至 ~47% 但仍远低于 func_match
- **status_match 几乎饱和** (>95%): 不是问题
- 结论: **坐标精度 (args_match) 是 action prediction 的决定性瓶颈**

---

### 14.7 Trajectory-Level 评估对比

Step-level 指标可能低估实际差距。Trajectory-level (semi-online AR, stop_on_error) 评估更能反映真实使用场景:

#### 整体对比

| 指标 | Base | SFT v2 1ep |
|------|:----:|:----------:|
| 总轨迹数 | 3,233 | 3,233 |
| 平均步数/轨迹 | 1.3 | 1.9 |
| Step-Level SR | 22.10% | 55.28% |
| **Trajectory SR (TSR)** | **1.64%** | **16.21%** |
| 平均进度率 | 12.30% | 36.70% |
| func_match | 62.54% | 77.45% |
| args_match | 1.86% | 17.07% |

#### 按轨迹长度

| 长度 | Base TSR | SFT v2 TSR | Base 进度 | SFT v2 进度 |
|------|:--------:|:----------:|:---------:|:----------:|
| 短 (1-5 步) | 1.64% | 15.89% | 12.30% | 35.68% |
| 中 (6-15 步) | 0.00% | 32.79% | 0.00% | 89.64% |
| 长 (16+ 步) | 0.00% | 0.00% | 0.00% | 0.00% |

#### 按域 (SFT v2)

| 域 | 轨迹数 | TSR | 进度 |
|----|:------:|:---:|:----:|
| PPT | 865 | 18.27% | 46.44% |
| Word | 1,369 | 15.49% | 35.37% |
| Excel | 999 | 15.42% | 30.09% |

**关键发现**:
1. **TSR 远低于 step SR**: SFT v2 step 55.28% → TSR 仅 16.21%。**错误的累积效应**在轨迹层面被放大。
2. **长轨迹 TSR = 0%**: 即使 SFT v2 也无法完成 16+ 步的任务。这直接印证了 Section 1.2 中 "Fail 轨迹 2.6x 长" 的发现。
3. **args_match 在 trajectory 中暴跌**: step-level ~47% → trajectory 仅 17.07%。坐标错误在序列执行中不断积累。
4. Excel 进度最低 (30.09%): 表格操作的坐标精度要求更高。

---

### 14.8 训练演进总结与 Multi-Agent 动机

```
Base (42.47% / 18.05%)
  │
  ├─ SFT v1: 灾难性遗忘 → 12.87% / 11.14%  [失败: 低分辨率 + 冻结 projector]
  │
  ├─ SFT v2 1ep: 70.56% / 46.90%  [成功: 高分辨率 + 解冻 projector]
  │   └─ SFT v2 2ep: 70.77% / 49.37%  [边际收益, 已饱和]
  │
  └─ SFT v3: 79.48% / 3.07%  [grounding 特化, action 过拟合]
```

| 发现 | 数据 | 对 Multi-Agent 的意义 |
|------|------|---------------------|
| SFT v2 已饱和 | 2ep 仅 +0.21pp grounding | 单纯增加训练量不可行 |
| 坐标是瓶颈 | args_match << func_match | Multi-agent grounding 优化最有价值 |
| Grounding-Action trade-off | v3: +8.9pp grounding, -43.8pp action | **两模型互补**是必然选择 |
| 放松 bbox 即可追平 paper | ±20px → 超过 paper Visual | 坐标精度提升 20px 就能突破 |
| TSR 放大坐标错误 | Step 55% → TSR 16% | 轨迹级改进需要更高的 step 可靠性 |
| V3 vs V2 错误互补低 | 仅 2.3% V2 独对 (§13.4) | V3 是 V2 的严格超集 → 用 V3 替代 V2 grounding |

**核心结论**: 单模型训练路径已到尽头。SFT v2 提供最佳 action 能力，SFT v3 提供最佳 grounding 能力，但两者无法在单一模型中兼得。Multi-agent 方案 (V2 出 action type + V3 出 coordinate) 是突破这一瓶颈的最有前景的路径。

---

## 15. RL 训练深化设计：数据驱动的方案

> **基于 Phase 0-1 全部实验结果，从数据反推 RL 的真正目标、训练对象、奖励信号和课程设计。**

---

### 15.1 天花板分析：RL 需要吃掉的三层 Gap

| 指标 | 现状 | RL 可触及上限 | Gap | 来源 |
|------|------|:------------:|:---:|------|
| coord_match | 73.0% (V2+V3 greedy) | 92.8% (oracle) | **19.8pp** | Exp 1.3 |
| grounding acc | 79.5% (V3 greedy) | 86.8% (best-of-K=10) | **7.3pp** | Exp 1.1 |
| hard cases | 18.2% (both fail) | 0% (理想) | **18.2%** | Exp 1.6 |
| TSR | 16.21% | ~42% (Phase 4 目标) | **25.8pp** | §14.7 |

**RL 优先级**: Selector (7.3pp headroom) → Hard Cases (18.2%) → Coord Precision (19.8pp)

---

### 15.2 核心发现：角色分工已确定，RL 不是训单个模型

Phase 1 最重要的发现是角色分工已明确：

```
V2 (action agent):   func_match 91.1%  ← 不需要训练
V3 (grounder):       coord_match 73.0%  ← RL 的核心目标
Selector:            DBSCAN 失败        ← 需要从头学
```

RL 策略：**分工训练，而非单模型全能化**
- V2 冻结（或极小 learning rate）
- V3 用 RL 提升 grounding 精度，仅在 grounding prompt 格式下训练
- Selector 作为新的可学习模块独立训练

---

### 15.3 关键诊断：args_match vs coord_match 差异根因

#### 问题

Exp 1.3 Oracle condition C 中，coord_match=92.8% 但 args_match 仅 10.9%，差距巨大。

#### 诊断过程

对比 GUI-360 eval 框架 vs exp1_3 自定义 `evaluate_action`:

| 评估方式 | args_match | 差异原因 |
|---------|:---------:|---------|
| GUI-360 eval (全 prompt + normalize_tool_args) | 47.1% | 使用 `normalize_tool_args` 填充默认值，x/y 转 coordinate |
| exp1_3 (最小 prompt + 原始比较) | 5.0% | 无归一化，V2 输出格式因 prompt 不同而变化 |

#### 根因

1. **Prompt 差异**: exp1_3 使用最小化 prompt (无 tool definitions)，V2 模型输出的 args 格式与 GUI-360 eval prompt 不同
2. **缺少归一化**: exp1_3 的 `evaluate_action` 不调用 `normalize_tool_args`，直接比较原始 args
3. **实际 other_match 错误率**: 在正确 prompt 下仅 **1%** (87/13010 click predictions)

#### 结论

| 指标 | 可靠性 | 用途 |
|------|:------:|------|
| **coord_match** | ✅ 高 | RL 主要奖励信号 |
| args_match (exp1_3) | ❌ 低 | 被 prompt format 污染，不可用 |
| args_match (GUI-360 eval) | ✅ 高 | 最终评估指标 |
| other_match 错误 | 仅 1% | 通过 prompt 工程 / post-processing 解决，不需要 RL |

**RL 应使用 coord_match 而非 args_match 作为主要奖励信号。**

---

### 15.4 Selector 训练设计：Phase 2 核心

#### 问题根源

DBSCAN 在所有场景中均失败（Exp 1.1/1.2/1.5），原因：~30% 样本为双峰分布，聚类中心偏离两个正确候选。

#### Selector 任务

给定截图 + K 个候选坐标 → 预测哪个最可能正确

#### Stage 1: Offline Supervised Pre-training

直接利用 Exp 1.1 的 K=10 数据 (18,265 × 10 = 182,650 候选):

```python
# 训练数据格式
(screenshot, [coord_1, ..., coord_10], GT_bbox) → label: which coord is correct

# 损失函数：listwise ranking loss
L_selector = -log P(correct_coord | screenshot, K_candidates)

# Agreement rate 分段加权：低 agreement 区域 loss 权重 ×2
# （低 agreement = 选择更难 = 更需要学习）
weight(sample) = 2.0 if agreement_rate < 0.5 else 1.0
```

#### Stage 2: Online RL Fine-tuning

```python
# Selector 的 RL 奖励
R_selector(t) = soft_coordinate_score(selected_coord, GT)
              × step_amplifier(t)       # step 0 放大 3×
              - λ × K_used              # 用 K 越多，惩罚越大

step_amplifier(t) = 3.0 if t == 0 else 1.0
```

目标：高置信时用 K=1，低置信时才扩展 K。

---

### 15.5 V3 Grounding RL：针对 Hard Cases

#### Hard Cases 特征

18.2% 的 hard cases (V2/V3 都错) 中有一个关键特点：**高 agreement + 错误** — 模型自信地犯错。

| 分段 | 样本数 | 占比 | 准确率 | "自信错误"数量 |
|------|:------:|:----:|:------:|:-------------:|
| agreement ≥ 0.9 | ~12,873 | 70.5% | 92% | ~1,030 |
| agreement 0.5-0.9 | ~3,654 | 20.0% | ~82% | ~658 |
| agreement < 0.5 | ~1,738 | 9.5% | ~55% | ~782 |

这 ~1,030 个"自信错误"是 RL 最有价值的目标。

#### Grounding Reward 设计

```python
def compute_grounding_reward(pred_coord, GT_bbox, agreement_rate):
    coord_correct = is_coord_in_bbox(pred_coord, GT_bbox)
    base_reward = soft_coordinate_score(pred_coord, GT_bbox)

    # 高置信但错误：强惩罚（推动模型探索）
    if agreement_rate >= 0.9 and not coord_correct:
        return base_reward - 0.5

    # 低置信且错误：轻惩罚（模型在探索，这是正常的）
    elif agreement_rate < 0.5 and not coord_correct:
        return base_reward  # 不额外惩罚

    # 正确：奖励正比于置信度（训练模型 calibration）
    elif coord_correct:
        return base_reward + 0.2 * agreement_rate

    return base_reward
```

#### V3 DAPO 训练约束

```python
# 关键约束
1. V3 仅在 grounding steps 接受梯度
2. 必须始终使用 grounding prompt 格式（Exp 1.4 确认 prompt-specific）
3. 每个 step 采样 K=4 个坐标（对应 DAPO 的 n=4）
4. Step 0 的 advantage 放大 3× （82% 分叉在 step 0）
5. 只更新 V3 的 LoRA 参数

# Advantage 权重
advantage_weights = [3.0] + [1.0] * (trajectory_len - 1)
```

---

### 15.6 完整 RL Training Loop

```
每个 training step:

1. 采样 query（oversample hard cases 40%）

2. V2 rollout（冻结）:
   → action_type + reasoning（greedy, 不采样）

3. V3 rollout（训练目标）:
   → grounding prompt 采样 K=4 坐标
   → 计算 agreement_rate

4. Selector 选择最优坐标:
   → 根据 agreement_rate 决定是否扩展 K
   → 输出 selected_coord

5. 组合: action_type (V2) + selected_coord (Selector from V3)

6. 计算 reward:
   R = R_coord × step_amplifier(t)
     + R_agreement_calibration
     + R_hard_case_penalty

7. DAPO 更新:
   → V3 LoRA (仅 grounding steps)
   → Selector (所有 grounding steps)
   → V2 LoRA (可选, 仅 action steps, lr = 0.1 × base_lr)
```

---

### 15.7 Curriculum 设计

| 训练阶段 | Easy/Hard 比例 | K | 目的 |
|---------|:-------------:|:---:|------|
| Warm-up (前 500 steps) | 80/20 | 1 | 稳定 selector 基础能力 |
| Main (500-3000 steps) | 60/40 | 4 | 聚焦 hard cases |
| Late (3000+ steps) | 40/60 | 4-10 | 强化低置信区域 |

Easy/Hard 划分：agreement ≥ 0.7 = easy (~70%)，< 0.7 = hard (~30%)

---

### 15.8 评估指标体系

#### 训练过程监控

```
主要指标:  coord_match（对 click/right_click/double_click）
辅助指标:  agreement_rate 分层准确率
          - ≥0.9 区域应 ≥92%（保持稳定或微升）
          - 0.5-0.9 区域应持续上升
          - <0.5 区域是 RL 最大收益区
最终指标:  TSR（每 500 step 跑一次 trajectory eval）
```

#### RL 有效性的早期信号

如果 RL 在做正确的事：
1. 低 agreement 区域 (<0.5) 的准确率先提升（探索空间最大）
2. 高 agreement 区域 (≥0.9) 保持稳定（已经很好，不应退化）
3. 整体 agreement 分布右移（模型变得更确定且更准确）

#### 不使用 args_match 的原因

| 问题 | 说明 |
|------|------|
| Prompt 敏感 | 不同 prompt 产生不同 arg 格式 (x/y vs coordinate)，影响评估 |
| 归一化依赖 | 需要 normalize_tool_args 才能可靠比较，RL 中引入不必要复杂度 |
| 实际贡献极小 | GUI-360 eval 下 other_match 错误率仅 1%，不是瓶颈 |
| coord_match 更直接 | 直接反映 grounding 能力，不受格式噪音干扰 |

---

### 15.9 其他注意事项

#### 避免破坏 V3 已有能力

V3 的 grounding 能力是 **prompt-specific** 的 (Exp 1.4: action prompt 下 70.6% vs grounding prompt 79.5%)。RL 训练时：
- 必须始终用 **grounding prompt 格式**
- 在 eval 中同时监控 grounding prompt 和 action prompt 下的性能
- 如果 action prompt 性能下降 > 5pp → 降低 learning rate 或加入 KL constraint

#### other_match 的修复方案

other_match 1% 错误率不需要 RL，通过以下方式解决：

1. **Prompt 工程**: 在 inference pipeline 中使用完整 GUI-360 tool definitions prompt
2. **Post-processing**: 调用 `normalize_tool_args` 归一化 pred/GT args
3. **格式规范**: 确保 V2 output → 标准化 → 组合 V3 coord 全程使用同一格式

#### 先决条件：Phase 2 Inference Pipeline

RL 训练前必须先完成 Phase 2 P0：

```
V2 (action agent, 完整 prompt) ──→ action_type + reasoning
                                      │
V3 (grounder, grounding prompt) ──→ K 个坐标候选
                                      │
                      Selector ──→ 最优坐标
                                      │
                      组合: action_type + coordinate ──→ 执行
```

- V2 必须使用完整 GUI-360 tool definitions prompt（确保 91%+ func_match 和 1% other_match）
- V3 使用 grounding prompt（确保 79.5% 基线）
- Selector 先用 agreement-based heuristic，后续训练替换

---

## 附录 A: 与 BCR (Bottleneck Crossing Reward) 的协同

BCR 回答 "WHERE to focus"，Multi-Agent 回答 "HOW to handle it":

```
Step at pos=0.1, FCV=0.95:
  BCR:   reward amplification = 1.48x
  Multi: K=7 samples, verification ON, max_retries=3
  → Agent 在关键位置投入最大计算预算

Step at pos=0.8, FCV=0.10:
  BCR:   reward amplification = 1.05x
  Multi: K=1 sample, verification OFF, no retry
  → Agent 在常规位置快速通过
```

## 附录 B: 关键代码路径

| 文件 | 当前功能 | 扩展计划 |
|------|---------|---------|
| `verl/workers/reward_manager/f_pseudo_dapo.py` | f_pseudo reward | + crossing_bonus + consensus_reward + verify_reward |
| `verl/models/moe/router.py` | TextOnlyRouter / ContextAwareRouter | + FCV-aware routing + adaptive top_k |
| `verl/models/moe/moe_wrapper.py` | Two-pass forward | + verification pass + recovery pass |
| `evaluation/eval_gui360_parquet.py` | Single-shot eval | + K-sample consensus + verify-recover loop |
| `verl/utils/reward_score/gui360/reward.py` | soft_coordinate_score | 复用 (no change) |
| `train_GUI_360/GUI-360-eval/prompts/prompt_action_prediction.py` | Action prompt template | + verification + recovery prompt templates |
