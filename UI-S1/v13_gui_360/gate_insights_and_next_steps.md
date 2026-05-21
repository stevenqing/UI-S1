# Gate Analysis 深度洞察与下一步方向

**Date**: 2026-04-27

---

## 核心发现汇总

### 1. Gate 编码了三层信息

| 维度 | 发现 | 实验 |
|------|------|------|
| **Modality** (image vs text) | L10: image +0.044, L18: text +0.067 | V3 |
| **Reasoning Phase** (planning→coordinate) | L10 递减 0.622→0.516, L18 递增 0.416→0.504 | V4 |
| **Input-dependent** | Cross-episode std: L10=0.023, L18=0.023 | V4 |

### 2. 两层 "X 形" 交叉模式

```
Gate value
0.62 │  L10 ↘
     │        ↘
0.55 │          ↘───────
     │     L18 ↗
0.50 │  ──↗
     │↗
0.42 │
     ├────────────────────
     planning  action  coordinate
```

- **L10**: 从 planning (0.622) 到 coordinate (0.516) 单调递减 → 视觉-语义融合需求逐步降低
- **L18**: 从 planning (0.416) 到 coordinate (0.504) 单调递增 → 空间推理需求逐步升高
- **L27**: action_start 时骤降 (0.490→0.495→0.517) → 高层在格式切换时调整

### 3. Planning phase 方差最大（input-dependent）

| Phase | L10 cross-ep std | L18 cross-ep std |
|-------|-----------------|-----------------|
| planning | **0.0233** | **0.0232** |
| action_type | 0.0079 | 0.0120 |
| coordinate | 0.0107 | 0.0225 |

Planning 阶段对不同 episode 的响应最敏感——不同的 UI 截图和 goal 导致不同的 communication 强度。
而 action_type 阶段几乎恒定（std=0.008）——格式化输出不需要 input-dependent 通信。

### 4. L10/L18 反相关

`corr(L10_planning, L10_coordinate) = -0.251`
`corr(L18_planning, L18_coordinate) = +0.413`

L10 planning 高的 episode，coordinate 反而低 → 视觉理解充分后，定位时不再需要强通信。
L18 planning 高的 episode，coordinate 也高 → 某些 episode 始终需要更多空间推理。

---

## 核心问题

当前 V13 的瓶颈不在 gate/communication（这部分工作正常），而在：
1. **Type/swipe 100% 预测为 click** — SP+SPWA 无法给 type/swipe 有效梯度
2. **84% 坐标完全点错** — 不是精度问题，是没理解该点什么
3. **训练震荡** — ep3 后 TSR 不再稳定上升

Gate 分析告诉我们：communication 已经 emergent 出了有意义的 modality-aware + phase-aware 模式。
问题是如何**利用**这些 insights 来解决上述瓶颈。

---

## 可行方向

### 方向 A: Phase-Aware Reward — 分阶段奖励

**洞察**: Gate 自然将生成分成 planning / action_type / coordinate 三个阶段，
每个阶段的 expert 协作模式不同。但当前 reward 是 per-step 的（一个 step 一个分数），
没有区分阶段。

**做法**:
- 把 step reward 拆成 phase-level reward：
  - R_planning: 是否正确识别了目标 UI 元素（可通过 planning text 中是否提及正确元素来判断）
  - R_action_type: 动作类型是否正确
  - R_coordinate: 坐标是否正确
- 对每个 phase 的 tokens 用对应的 reward 计算 advantage
- 关键：coordinate phase 的 L18 communication 最活跃，但 84% 完全错误 →
  在 coordinate phase 给更强的 reward signal

**预期**: 更精确的 credit assignment → 各 phase 独立优化 → coordinate 和 type 分别改善

**实施难度**: 中。需要修改 advantage 计算，按 token position 分段。

### 方向 B: Communication-Guided Exploration — 利用 gate 方差做探索

**洞察**: Planning 阶段 gate 方差大（0.023），说明不同 episode 的通信需求不同。
但当前 RL 的 K=8 rollouts 是 i.i.d.（同一输入 K 次独立采样），通信模式完全相同。

**做法**:
- 在 K 次 rollout 中，给不同 rollout **不同的 communication bias**：
  - Rollout 1-4: 正常 gate
  - Rollout 5-6: gate 偏移 +δ（更多 communication）
  - Rollout 7-8: gate 偏移 -δ（更少 communication）
- 不同 communication 强度 → 可能产生不同的 action type 预测 → 打破 "100% click" 的僵局
- 如果 high-communication rollout 产生了 type 预测且正确 → 正 advantage → 学到 type

**关键假设**: Gate 偏移能导致不同的 action 输出。需要先验证（做一个 ablation）。

**预期**: 打破 type/swipe 探索死局，+3-5% TSR

**实施难度**: 低。只需在 rollout 时给 gate 加 bias。

### 方向 C: Layer-Specific Communication Curriculum

**洞察**: L10 在 planning 最强，L18 在 coordinate 最强。
但当前训练中所有层的 communication 参数用同一个 learning rate 更新。

**做法**:
- L10 (visual-semantic alignment): 用更大的 lr / 更多的训练信号
- L18 (spatial reasoning): 特别关注 coordinate phase 的 loss，加大 lr
- L27 (high-level switching): 保持较小 lr，避免过度适应

或更激进地：
- **冻结** L10/L27 的 comm weights（已经学到有用的模式）
- **只训练** L18 的 comm weights（需要更多空间推理能力提升）
- 结合 coordinate-specific reward

**预期**: 减少震荡（少改已学好的层），集中改善空间推理

**实施难度**: 低。改 optimizer 的 param groups。

### 方向 D: Gate-Conditioned Auxiliary Head — 坐标预测辅助头

**洞察**: Coordinate 阶段 L18 communication 最活跃（0.474 vs planning 的 0.445），
说明模型在此阶段确实在做空间推理。但 84% 坐标完全错误。

**做法**:
- 在 L18 层的 communication output 后面加一个轻量的 coordinate regression head
- 训练时：同时用 LM loss（生成坐标 token）+ regression loss（直接预测坐标）
- 推理时：可以用 regression head 的输出来 refine 生成的坐标，或作为 reranking signal

**预期**: 直接改善坐标准确率，+3-5% TSR

**实施难度**: 中~高。需要修改模型架构。

### 方向 E: "Think More" on Hard Episodes — 自适应 Communication Rounds

**洞察**: 不同 episode 的 planning gate 方差很大（L10: 0.504~0.639）。
Gate 高的 episode = 需要更多 cross-expert 融合。

**做法**:
- 训练时自适应 communication rounds T：
  - 如果 gate > threshold → 增加一轮 communication (T=3)
  - 如果 gate < threshold → 保持 T=2
- 或者：用 gate magnitude 来决定是否做额外的 "thinking" step
- Hard episode（gate 高）获得更多计算 → 更好的 planning → 更准确的 action

**预期**: 在保持简单 episode 速度的同时，hard episode 获得更多推理能力

**实施难度**: 中。需要 dynamic T 的实现。

---

## 验证实验结果 (2026-04-27)

### ❌ 方向 B 被否定 — Gate Perturbation 无法改变 Action Type

Gate perturbation (±0.5 on all/per-layer) 对 action type **完全无影响**：
- 所有 delta 下 99.8-100% 预测 click
- 坐标会移动（85% change at d=-0.5），但随机移动不会变准
- **结论**: "100% click" 锁在 A/B 投影或 base model，不在 communication gates

### ✅ 方向 A 验证通过 — Planning Gates 预测 Success

Gate Signature Analysis (offline, 968 episodes):
- **L10 planning high → 65% correct** vs low → 51% correct (14% gap!)
- **L18 planning low → 66% correct** vs high → 51% correct (15% gap!)
- Correct episodes: planning 阶段更多 L10 通信 + 更少 L18 通信
- Correct episodes: coordinate 阶段反而**更少**通信 → 决策在 planning 阶段完成
- Gate variance 无差异 → 不是通信"多少"的问题，是通信"方向对不对"

### ✅ Phase-Conditional Ablation — 通信是结构性必需的

| Mode | Click% | Coord <50px |
|------|--------|------------|
| full | 99.5% | 49.3% |
| no_comm | 7.7% | 31.4% |
| planning_only | 26.7% | 38.7% |
| coord_only | 7.7% | 31.3% |

- 关掉通信 → 模型崩溃（99.5% → 7.7%）→ 通信是**结构性必需的**
- Planning 阶段通信恢复最多（26.7%）但不够 → 全程通信才能正常工作
- coord_only ≈ no_comm → coordinate 阶段通信依赖 planning 阶段的基础

### ✅ Forced-Prefix — 模型"不选"而非"不会"

- 强制选 type 后，**54.3% 能正确输出文本内容** (mean similarity=0.59)
- Logit gap: left_click vs type = **18.9** (GT=type) vs **21.9** (GT=click)，差 3.0
- **→ 模型隐约知道该选 type（gap 更小），但 18.9 的 gap 太大，永远不会自发选择**

### ✅ Base Model — RL 从零教会格式，同时创造了 click bias

- Base model 98.6% 输出没有 `<action>` tag → 不是 "collapse 多样性"
- RL 训练同时教会了 action 格式 + 强化了 click → 两者纠缠

---

## 最终诊断总结 (2026-04-27)

| 问题 | 答案 | 证据 |
|------|------|------|
| 通信必要吗？ | **结构性必需** | no_comm → 7.7% click (从 99.5%) |
| 哪个阶段最重要？ | **Planning** | planning_only 恢复到 26.7% |
| Gates 预测 success？ | **是，planning 阶段** | L10 high → 65% vs 51% (14% gap) |
| 模型会 type/swipe 吗？ | **会，但不选** | 强制后 54% 正确 |
| Click bias 多强？ | **巨大** (18-22 logit gap) | 但 GT=type 时 gap 小 3.0 |
| RL collapse 了多样性？ | **否** — base model 无格式 | RL 从零教会格式 + click |

---

## 最终优先级排序

| 优先级 | 方向 | 预期收益 | 难度 | 状态 |
|--------|------|---------|------|------|
| **P0** | A: Phase-Aware Reward | +3-5% | 中 | ✅ 验证通过 |
| **P0** | Type/Swipe: Reward shaping | +3-5% | 中 | ✅ 确认"不选"→ reward 引导可行 |
| **P1** | C: Layer-Specific Curriculum | +2-3% | 低 | ⚠️ planning 重要但需全程通信 |
| ~~P0~~ | ~~B: Comm-Guided Exploration~~ | — | — | ❌ 否定 |
| **P1** | D: Coordinate Aux Head | +3-5% | 中~高 | 仍有价值 |
| **P2** | E: Adaptive Comm Rounds | +1-3% | 中 | 优先级降低 |
