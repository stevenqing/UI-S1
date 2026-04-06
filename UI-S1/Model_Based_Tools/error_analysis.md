# Cross-Dataset Long-Horizon Bottleneck Analysis

> **Datasets**: AndroidControl (AC, mobile, 1543 trajectories) + GUI-360 (G360, desktop, 3233 trajectories)
> **Model**: Qwen2.5-VL-7B-Instruct (base) + gui360_full_sft_v2
> **Goal**: 找到影响 long-horizon task 的关键 features，并通过实验验证

---

## 1. Three Common Bottlenecks

通过 17+ controlled evaluations，在两个数据集上识别出三个共同的 long-horizon bottleneck：

### 1.1 Error Non-Independence (Error Cascading)

> 一旦犯错，后续步骤犯错概率显著升高，且效应随 trajectory 长度放大。

| Evidence | AC | G360 |
|----------|-----|------|
| P(correct \| prev_correct) − P(correct \| prev_error) | **12.5pp** | **11.5pp** |
| Clustering gap (short traj) | 7.4pp | 7.3pp |
| Clustering gap (long traj) | **28.0pp** | **15.5pp** |
| Error propagation duration | 3-4 steps | 3-4 steps |

**Root Cause: Context Explosion** — 通过 AC 的有/无历史对照实验证明：

| Evidence | Key Number |
|----------|-----------|
| History value 随 step 递减 | Step 3: **+43pp**, Step 13+: **−18pp** (history 成为净负担) |
| 无历史时 error clustering 降低一半 | 12.8pp → **5.8pp** |
| Late-stage clustering 83% 来自 context | 有历史 23.1pp vs 无历史 4.0pp |

→ 错误预测被 feed back 到历史 context → 噪声历史稀释 goal 信息 → 偏置后续预测 → 效应随长度放大

### 1.2 Phase-0 Initialization Failure

| Dataset | Phase 0 Completion | Phase 1 Completion |
|---------|-------------------|-------------------|
| AC | **38.1%** | 64.5% |
| G360 | **35.9%** | 37.2% |

→ 两个数据集 Phase 0 完成率都仅 **~37%**，是最大单一瓶颈
→ 长任务的 step_0 更难（AC: 47%→36%, G360: 64%→58%），任务复杂度与长度正相关
→ 初始化失败 + error cascading = trajectory 从一开始就进入 Error State

### 1.3 Action Type Transition Difficulty

| Dataset | Short Traj Gap | Long Traj Gap |
|---------|---------------|---------------|
| AC | 2.9pp | **10.3pp** |
| G360 | ~15pp | ~16pp |

→ Action type 切换点（click→type, click→scroll）始终比 continuation 更难
→ 更多 unique action types = 更低准确率 (G360: 1 type 68.7% → 4 types 54.0%)

### 1.4 NOT a Common Bottleneck: Positional Degradation

AC accuracy **下降** (83%→67%), G360 accuracy **上升** (57%→75%) → 位置降级不是 universal pattern

---

## 2. Experimental Validation: Planning vs Grounding Decomposition

通过 summary/subtask context 实验，将 error 分解为 **planning failure** 和 **grounding failure**。

**实验设计:**
- **AR Baseline**: 标准 autoregressive rollout，累积完整历史
- **Summary Context**: 压缩历史为简短 action 描述（如 "Step 1: click at [x,y]"）
- **Subtask Context (oracle)**: 提供 GT step-level instruction + 压缩历史
  - AC: 人工标注的 `step_instruction`（如 "Click on the Settings icon"）
  - G360: GT `thought` 字段（如 "I need to click on the File tab..."）
  - 本质相同：告诉模型"这一步该做什么"，测试 grounding 上限

**Metrics**: TSR (全部正确=成功), AvgProg (完成比例均值), StepAcc (单步正确率)

### 2.1 Results

**AndroidControl (Qwen2.5-VL-7B base, 1543 episodes)**

| Condition | TSR | StepAcc |
|-----------|-----|---------|
| AR Baseline | 16.1% | 55.5% |
| Summary | 17.3% (+1.2pp) | 57.2% (+1.7pp) |
| **Subtask (oracle)** | **34.7% (+18.6pp)** | **75.5% (+20.0pp)** |

**GUI-360 Base (Qwen2.5-VL-7B, 3233 trajectories)**

| Condition | TSR | StepAcc |
|-----------|-----|---------|
| AR Baseline | 1.64% | 22.10% |
| Summary | 1.98% (+0.34pp) | 24.57% (+2.47pp) |
| **Subtask (oracle)** | **5.41% (+3.77pp)** | **43.33% (+21.23pp)** |

**GUI-360 SFT v2 (gui360_full_sft_v2, 3233 trajectories)**

| Condition | TSR | StepAcc |
|-----------|-----|---------|
| AR Baseline | 16.21% | 55.28% |
| Summary | 17.07% (+0.86pp) | 56.22% (+0.94pp) |
| **Subtask (oracle)** | **23.48% (+7.27pp)** | **67.82% (+12.54pp)** |

### 2.2 Error Decomposition

> Subtask StepAcc = grounding ceiling (给了完美指令仍做不对的比例)
> Baseline StepAcc − Subtask StepAcc = planning error

| Model | Grounding Error | Planning Error | Total Error |
|-------|-----------------|----------------|-------------|
| AC Base | **24.5%** | **20.0pp** | 44.5% |
| G360 Base | **56.67%** | **21.23pp** | 77.90% |
| G360 SFT v2 | **32.18%** | **12.54pp** | 44.72% |

**AC Per-Length Decomposition (关键发现):**

| Length | Grounding Error | Planning Error | Planning 占比 |
|--------|-----------------|----------------|--------------|
| short(1-3) | 22.3% | 19.8pp | 47% |
| medium(4-7) | 24.5% | 18.6pp | 43% |
| long(8-15) | 26.4% | **24.6pp** | **48%** |
| vlong(16+) | 23.3% | **25.0pp** | **52%** |

→ **Grounding error 恒定 (~23-26%)**，不随 trajectory 长度变化
→ **Planning error 随长度增加** (20pp → 25pp)，是 long-horizon 降级的直接驱动因素
→ Subtask 的 StepAcc **不随长度衰减** (77.7% → 76.7%)，证明 oracle planning 消除了长度效应

### 2.3 Cross-Dataset Key Findings

| Finding | Evidence |
|---------|---------|
| **Summary 压缩效果有限** | 所有实验 TSR 提升均 <2pp，单纯减少 token 不解决问题 |
| **Task decomposition 是最高杠杆** | 两个数据集 subtask 都有大幅提升 (AC +18.6pp TSR, G360 SFT +7.27pp TSR) |
| **Planning error 跨数据集一致** | AC 20.0pp ≈ G360 21.23pp，是 model-intrinsic 的瓶颈 |
| **Grounding 是 G360 主要瓶颈** | G360 base 56.67% vs AC 24.5%，桌面 UI 更难 |
| **SFT 大幅改善 grounding** | G360 grounding error: 56.67% → 32.18% (−24.49pp) |
| **SFT 对 planning 改善有限** | G360 planning error: 21.23pp → 12.54pp (−8.69pp) |

---

## 3. Solution Priority

```
Priority 1 (最高杠杆): Task Decomposition / Planner Module
  → AC: TSR 16.1% → 34.7% (2.15x), StepAcc +20pp
  → G360 SFT v2: TSR 16.21% → 23.48% (1.45x), StepAcc +12.5pp
  → Planning error 跨数据集一致 (~20pp base, ~12pp after SFT)
  → 关键: Subtask StepAcc 不随 trajectory 长度衰减
  → 实现: 训练 planner 或用 LLM 生成 step-level instruction

Priority 2: Domain-Specific Grounding Training (SFT)
  → GUI-360 grounding error: 56.67% → 32.18% (−24.49pp)
  → 对 desktop domain 尤为重要（复杂 UI, dense 菜单）

Priority 3: Context Window Management (配合 Task Decomposition)
  → 单独使用仅 +0.3~1.2pp TSR（两个数据集均验证）
  → 结合 task decomposition 可进一步减少 noise

Priority 4: Self-Consistency Decoding
  → G360 oracle 64.5% vs greedy 11.2% = 6x gap
  → 推理时即可使用，无需训练

Priority 5: Step-0 Specialized Training
  → Phase-0 ≈ 37% 完成率，最大单一瓶颈
```

---

## 4. From Symptoms to Root Cause: Lack of Task Progress State

Section 1-3 的三个 bottleneck（error propagation、phase-0 failure、transition difficulty）是**现象**，不是原因。核心问题是：**为什么这些现象在 long-horizon 任务中出现，而在 short-horizon 中不严重？**

### 4.1 Core Claim

Context explosion 是症状。真正的根因是：

> **模型没有关于"任务执行状态"的 explicit representation，被迫从原始 context 中 implicitly 推断"我在哪里、做了什么、还剩什么"。**

这个推断在短任务里可以靠 local context 完成；在长任务里，这个推断本身成为最难的子问题。

### 4.2 Evidence Reinterpretation

**Evidence 1：Subtask oracle 几乎消除了位置降级**

```
baseline StepAcc:  57.9% → 51.7%  (short → vlong, -6.2pp)
oracle StepAcc:    77.7% → 76.7%  (short → vlong, -1.0pp)
```

→ 模型的**执行能力（grounding）不随长度衰减**，衰减的是**状态估计能力**（知道该做什么）。

**Evidence 2：History 在 step 13+ 成为净负担（-17.9pp）**

→ 不是因为 context 太长，而是因为：随着执行推进，历史 context 对"当前应该做什么"的信噪比趋近于零——正确步骤记录对下一步的帮助，远小于错误步骤对预测的污染。

**Evidence 3：Planning error 跨数据集一致（~20pp），与 domain 无关**

→ Planning failure 是 **model-intrinsic** 的，是当前 autoregressive 范式的系统性缺陷，不是数据问题。

### 4.3 Planning Failure 的三种机制

当前报告把 planning error 定义为"给了 oracle instruction 后的 StepAcc 提升"，但这是**量化**，不是**解释**。Planning failure 至少有三种不同机制，需要**不同的解决方案**：

| 类型 | 描述 | 位置分布 | 解决方案 |
|------|------|----------|----------|
| **Goal Misparse** | 初始就对任务理解错误 | 前 20% (Phase-0) | 更好的 task understanding |
| **Progress Misestimation** | 知道 goal，但不知道"做到哪了" | 后 30% (terminate-too-early) | Explicit state tracking |
| **Subgoal Selection Failure** | 知道"做到哪了"，但不知道"下一步该做什么" | Action type boundary 处 | Hierarchical planner |

→ 目前的诊断没有区分这三者。这是第一个需要填补的分析空白。

### 4.4 Reactive vs. Proactive Planning

这是最深层的问题。

```
Reactive (当前):   a_t = π(o_t, h_t, G)         → 每步重新"猜测"自己在哪里
Proactive (目标):  a_t = π(o_t, s_t, plan_t)     → 维护并更新 task progress state
```

Subtask oracle 实验证明：如果外部给出 plan_t（oracle instruction），性能大幅提升。但这相当于把 planning 外包给了 oracle。**真正的研究问题是：模型能否自主维护和更新 s_t？**

---

## 5. Proposed Deep Analyses

### 5.1 Planning Failure 三分类实验 ✅

对 AC baseline 1292 failed episodes (stop-on-error) 做三分类，共 1292 first-error steps：

**分类结果：**

| Failure Type | Count | Proportion |
|-------------|-------|------------|
| **Goal Misparse** (step-0 error) | 765 | **59.2%** |
| **Progress Misestimation** (premature terminate) | 85 | **6.6%** |
| **Subgoal Selection Failure** (at action type boundary) | 249 | **19.3%** |
| Pure Grounding Error (known type continuation) | 193 | **14.9%** |

**位置分布（验证了预期的分离）：**

- **Goal Misparse**: 100% 集中在 step 0（by definition）
- **Progress Misestimation**: 66% 在 trajectory 后 25%（[75-100%] range），符合"快完成时误判为已完成"
- **Subgoal Selection**: 分布在整个 trajectory 中，在 action type transition 处

**按 trajectory 长度的变化（关键发现）：**

| Length | Goal Misparse | Progress Misest. | Subgoal Selection | Pure Grounding |
|--------|--------------|------------------|-------------------|----------------|
| short(1-3) | 61.0% | 3.3% | 12.6% | 23.1% |
| medium(4-7) | 57.2% | 8.1% | 22.1% | 12.6% |
| long(8-15) | 57.7% | 8.5% | 23.2% | 10.6% |
| vlong(16+) | 60.7% | 3.6% | **32.1%** | 3.6% |

→ **Goal Misparse ~60% 恒定**，是 trajectory 长度无关的 baseline failure
→ **Subgoal Selection 随长度显著增加** (12.6% → 32.1%)，是 long-horizon 特有的 planning 难题
→ **Pure Grounding 随长度减少** (23.1% → 3.6%)，长任务中 planning 主导失败
→ 三类失败位置分离已验证，支持 specialist 分工

### 5.2 State Information Decay 曲线 ✅

**实验**: AC 数据集上，比较不同历史窗口大小对 StepAcc 的影响。

**Overall Results (1543 episodes):**

| Condition | TSR | AvgProg | StepAcc |
|-----------|-----|---------|---------|
| K=0 (no history) | 17.30% | 27.78% | 57.40% |
| **K=1** | **18.02%** | **28.39%** | **58.05%** |
| K=3 | 17.37% | 27.78% | 57.53% |
| K=5 | 17.24% | 27.80% | 57.47% |
| AR full (K=all) | 16.07% | 26.36% | ~55.5% |
| Summary (compressed) | 17.30% | 27.75% | ~57.2% |
| **Subtask (oracle)** | **34.67%** | **45.47%** | **~75.5%** |

**关键发现:**

1. **K=1 是最优窗口**：仅保留最近 1 步历史比无历史好 (+0.65pp StepAcc)，但更长窗口不再帮助
2. **历史窗口的边际收益递减极快**：K=1 → K=3 已经从正变负 (58.05% → 57.53%)
3. **Full history 最差** (55.5%)，证实了 context explosion hypothesis
4. **K=0 ≈ Summary**: 无历史 (57.40%) ≈ 压缩历史 (57.2%)，说明压缩不解决本质问题

**Per-Length StepAcc:**

| Length | K=0 | K=1 | K=3 | K=5 |
|--------|-----|-----|-----|-----|
| short(1-3) | 60.2% | **61.3%** | 60.3% | 60.4% |
| medium(4-7) | 57.7% | **58.9%** | 58.4% | 58.3% |
| long(8-15) | **52.9%** | 53.1% | 51.1% | 51.3% |

→ **短任务**: K=1 最优 (+1.1pp over K=0)
→ **长任务**: K=0 和 K=1 接近最优，K=3/5 开始有害 (-1.8pp)
→ 更长 trajectory → 更小的最优窗口

**Per-Step Position StepAcc (stop-on-error):**

```
Step:    0      1      2      3      4      5
K=0:   44.6%  69.2%  72.5%  71.9%  71.6%  67.3%
K=1:   44.7%  72.6%  71.2%  71.6%  72.9%  68.5%
K=3:   44.5%  71.3%  72.2%  68.1%  70.8%  70.8%
K=5:   44.3%  72.0%  73.4%  66.2%  73.4%  63.3%
```

→ **Step 0 完全相同** (~44.5%) — 无历史可用，纯 task understanding
→ **Step 1: K=1 最高** (72.6% vs K=0 69.2%)，1 步历史有价值
→ **Step 3+: K=5 开始波动**，噪声累积影响 later steps

**实际形状（与预期对比）:**

```
StepAcc
   ↑
   │  ·····················  Oracle (~75%)
   │  ═══════════════════   K=1 ≈ K=0 ≈ K=3 ≈ K=5 (~57%)  ← 全部重叠!
   │  ─────────────────     AR full (~55.5%)
   │
   │  ← 巨大 gap (~18pp) →
   └──────────────────────→ Trajectory Position
```

→ **预期被推翻**: 不同 K 之间差异极小 (<1pp)，真正的 gap 在 window vs oracle (18pp)
→ **结论: 历史信息数量不是瓶颈，历史信息质量才是** — 任何基于原始 action history 的方法（无论窗口大小或压缩策略）都无法接近 oracle
→ 需要的是 **semantic state representation** (如 oracle instruction)，而非更好的 history management

### 5.3 Uncertainty as Planning Difficulty Signal ✅

用 GUI-360 base model 的 K=10 采样数据 (19,046 steps)，分析 uncertainty 与 planning difficulty 的关系。

**检验 1: Uncertainty 在 error steps 是否高于 correct steps？**

→ **反直觉结果**: 错误步骤 entropy **更低** (0.152) 而非更高 (correct: 0.319)
→ 原因：82.2% 预测是 unanimous (entropy~0)，但其中 **91.4% 是错误的**
→ **模型是"自信地犯错"** — 在 GUI-360 base 上，低 entropy 意味着模型自信但不意味着正确

**检验 2: Uncertainty 在 action type boundaries 处是否升高？** ✅

| Position | Entropy | Greedy Acc |
|----------|---------|------------|
| Boundary | 0.324 | 7.9% |
| Continuation | 0.123 | 12.5% |

→ Boundary entropy **2.6x** 于 continuation，且 accuracy 更低

**检验 3: Uncertainty 是否随 trajectory 位置增加？** ✅ (弱趋势)

```
Position   Entropy
0-10%      0.178
20-30%     0.136
50-60%     0.179
80-90%     0.193
90-100%    0.256  ← 尾端显著升高
```

**关键发现: Agreement Rate 是强预测信号**

| Agreement | Greedy Acc | Oracle Acc | n |
|-----------|-----------|------------|---|
| 0.8-1.0 | **91.3%** | 96.2% | 1,017 |
| 0.6-0.8 | **58.2%** | 93.6% | 981 |
| 0.4-0.6 | 22.7% | 93.0% | 1,849 |
| 0.2-0.4 | 4.2% | 91.8% | 4,497 |
| <0.2 | 0.2% | 42.5% | 10,702 |

→ Agreement > 0.8 时 greedy accuracy 91.3%，几乎不需要干预
→ Agreement < 0.2 时 greedy accuracy 0.2%，但 oracle 仍有 42.5%，说明正确答案在采样集中存在
→ **Agreement rate 是目前最强的无监督 reliability signal**

**结论**: Entropy 不适合做 error detector（模型自信地犯错），但 **agreement rate** 是强信号。高 agreement (>0.8) 的步骤可以信任 greedy；低 agreement (<0.4) 需要 planner 介入或 self-consistency 机制。

### 5.4 Cross-Condition Rescue Analysis ✅

**方法**: 用 stop-on-error 数据比较 baseline vs subtask vs summary 在相同 episode 上的表现，分析 oracle planning 在哪些情况下能"rescue"baseline failures。

**Rescue 总览 (1295 baseline failures):**

| Condition | Episodes Rescued | Rescue Rate |
|-----------|-----------------|-------------|
| **Subtask (oracle)** | 508 | **39.2%** |
| Summary | 156 | 12.0% |

→ Oracle planning 能修复 39.2% 的 baseline failures = pure planning errors
→ 60.8% 即使给了 oracle instruction 仍然失败 = grounding errors

**Rescue 按 failure 位置:**

| Position | Rescued/Failed | Rate |
|----------|---------------|------|
| 0-25% (early) | 371/1011 | 36.7% |
| 25-50% | 69/129 | **53.5%** |
| 50-75% | 51/115 | 44.3% |
| 75-100% (late) | 17/40 | 42.5% |

→ Mid-trajectory (25-75%) rescue rate 最高 (~45-53%)，说明这些步骤主要是 planning failure
→ Early failures (0-25%) rescue rate 较低 (36.7%)，因为大量 step-0 errors 是 grounding failures

**Rescue 按 trajectory 长度:**

| Length | Rescued/Failed | Rate |
|--------|---------------|------|
| short(1-3) | 135/285 | **47.4%** |
| medium(4-7) | 267/695 | 38.4% |
| long(8-15) | 98/287 | 34.1% |
| vlong(16+) | 8/28 | 28.6% |

→ **短任务 rescue rate 更高** (47% vs 29%)
→ 长任务中更多是 grounding failures（即使知道该做什么也做不对）

**Grounding Limit: 即使有 oracle 也修不了的 action types:**

| Action Type | Count | Proportion |
|-------------|-------|------------|
| open | 468 | **47.9%** |
| click | 248 | 25.4% |
| system_button | 114 | 11.7% |
| wait | 110 | 11.2% |

→ `open` 是最大的 grounding bottleneck（47.9%），即使告诉模型"打开某 app"也经常失败

**Cross-Reference: 5.1 Failure Types × Subtask Rescue Rate**

| Failure Type | Rescue Rate | Interpretation |
|-------------|------------|----------------|
| Progress Misestimation | **57.6%** | Oracle instruction 最能修复 premature termination |
| Pure Grounding (right type) | 53.4% | Oracle 帮助校准 coordinate 等参数 |
| Pure Grounding (wrong type) | 46.9% | 部分可通过更明确指令修复 |
| Subgoal Selection (boundary) | 42.4% | Oracle 在 transition 处提供方向 |
| Goal Misparse (step 0) | **34.5%** | 最难修复：初始理解错误是 deep grounding 问题 |

→ Planning failure 的不同子类有不同的 oracle 修复率
→ **Progress Misestimation 最容易通过 planner 修复** (57.6%)，因为只需要"告诉模型还没完"
→ **Step-0 Goal Misparse 最难修复** (34.5%)，是 grounding 和 task understanding 的共同瓶颈

---

## 6. Deep Mechanism Analyses

### 6.1 Analysis B: Step-0 Failure Anatomy ✅

**问题**: Step-0 是最大 failure source (60% of all failures)，但 oracle rescue rate 仅 34.5%。65.5% 不被 rescue 的部分到底是什么？

**结果 (896 step-0 failures):**

| Component | Count | Proportion |
|-----------|-------|------------|
| Rescued by oracle (planning failure) | 309 | **34.5%** |
| Not rescued (grounding failure) | 587 | **65.5%** |

**Grounding failure 按 GT action type 分解:**

| GT Action Type | Grounding Failure Rate | Count |
|---------------|----------------------|-------|
| wait | **100.0%** | 6/6 |
| **open** | **88.6%** | 460/519 |
| system_button | 58.4% | 66/113 |
| click | 26.4% | 53/201 |
| swipe | 3.8% | 2/53 |

→ **`open` 是 step-0 grounding failure 的主要来源 (460/587 = 78.4%)**
→ 模型不是"不知道该开什么 app"，而是**不知道 `open` 这个 action type**
→ 最常见的错误模式: 模型预测 system_button (205), click (165), swipe (137) 代替 open

**Step-0 failure 构成:**

| 类型 | Count | Proportion |
|------|-------|------------|
| Wrong action type | 765 | **85.4%** |
| Right type, wrong args | 131 | 14.6% |

→ **85% 的 step-0 error 是 action type 就错了**——模型在 "理解该做什么" 这一步就失败
→ 意味着 step-0 的 "Goal Misparse" 标签是准确的：不是 grounding 精度问题，而是根本性的 action type 选择错误

**Step-0 failure rate 随 trajectory 长度增加:**

| Length | Step-0 Fail Rate |
|--------|-----------------|
| short(1-3) | 47.3% |
| medium(4-7) | 60.4% |
| long(8-15) | **67.5%** |

→ 长任务 step-0 更难，因为长任务通常从 "open app" 开始（需要先 navigate），而短任务可能直接在已打开的界面操作

**关键结论**: Step-0 failure 主要是 `open` action 的 grounding failure。模型缺乏 "open app by name" 的 action representation。这是一个 **可直接通过训练修复** 的问题（SFT 阶段上采样 open action）。

### 6.2 Analysis C: Subgoal Selection Mechanism Decomposition ✅

**问题**: Subgoal Selection failure 从 12.6% (short) 增加到 32.1% (vlong)，是数量效应、难度效应、还是累积效应？

**Metric 1: Transition Density**

| Trajectory Type | Mean Transition Density |
|----------------|----------------------|
| Successful | 0.159 |
| Failed | 0.064 |

→ 成功 trajectory 反而有**更高**的 transition density
→ 说明 transition density 本身不是 failure predictor

**Metric 2: Transition Pair Difficulty (top failure pairs)**

| Transition Pair | Failure Count |
|----------------|--------------|
| click → wait | 53 |
| click → swipe | 45 |
| type → click | 30 |
| click → type | 17 |
| system_button → click | 17 |

→ **click→wait** 是最难的 transition：模型完成 click 后不知道该 wait
→ **click→swipe** 次之：从精确点击切换到滑动方向

**Metric 3: Historical Transitions (累积效应检验)**

| Prior Transitions | Failure Count |
|------------------|--------------|
| 0 | **145 (65.3%)** |
| 1 | 53 (23.9%) |
| 2+ | 24 (10.8%) |

→ **65% 的 subgoal selection failure 发生在第一次 transition**（0 prior transitions）
→ **不是累积效应**——历史中的 transition 数量不预测失败

**Per-Length Transition Failure Rate (控制 opportunity):**

| Length | Failures / Transitions | Rate |
|--------|----------------------|------|
| short(1-3) | 55/69 | **79.7%** |
| medium(4-7) | 123/233 | 52.8% |
| long(8-15) | 40/104 | 38.5% |
| vlong(16+) | 4/14 | 28.6% |

→ **短任务的 per-transition failure rate 反而更高** (79.7% vs 28.6%)
→ 这意味着 5.1 中观察到的 "subgoal selection 随长度增加" 是**数量效应**（长任务有更多 transition opportunities），而非 per-transition 难度增加

**模型在 transition 处的预测模式:**

| Pattern | Proportion |
|---------|-----------|
| Correct GT type but wrong args | 24.8% |
| Premature terminate (gt=wait/swipe) | **22.5%** |
| **Repeat previous action type** | **22.5%** |
| Other wrong type | 30.2% |

→ 22.5% 是 **action inertia** — 模型"惯性"地重复上一步的 action type
→ 22.5% 是 premature termination — 在需要 wait/swipe 时错误地 terminate

**结论**:
- Subgoal Selection failure 的长度效应是**数量效应**（更多 transition opportunities），不是难度效应或累积效应
- 每个 transition 的难度实际上在长任务中更低（可能因为长任务到达 transition 时已经通过了 easier steps）
- 主要失败模式：action inertia (22.5%) 和 premature termination (22.5%)
- **解决方案指向**：不需要 hierarchical planner，而是需要 action type diversity training + anti-termination bias

### 6.3 Analysis A: Oracle Information Ablation ✅

**目标**: 拆解 oracle 的 StepAcc 增益来自哪些信息组件。

**完整结果:**

| Condition | TSR | StepAcc | Δ StepAcc vs Baseline |
|-----------|-----|---------|----------------------|
| AR Baseline (K=all) | 16.1% | 55.5% | — |
| K=0 (no history) | 17.3% | 57.4% | +1.9pp |
| **type_only** | **26.5%** | **69.3%** | **+13.8pp** |
| **target_only** | **27.8%** | **69.4%** | **+13.9pp** |
| Full oracle (NL) | 34.7% | 75.5% | +20.0pp |
| **type_target** | **36.9%** | **78.3%** | **+22.8pp** |

**Per-Length StepAcc:**

| Length | Baseline | type_only | target_only | Full oracle | type_target |
|--------|----------|-----------|-------------|-------------|-------------|
| short(1-3) | 58.0% | 72.4% | 73.7% | 77.7% | 79.2% |
| medium(4-7) | 54.8% | 68.6% | 69.5% | 75.5% | 78.1% |
| long(8-15) | 52.7% | 68.1% | 65.7% | 73.6% | 77.7% |
| vlong(16+) | 51.7% | 68.5% | 60.0% | 76.7% | **81.5%** |

**关键发现:**

1. **type_only ≈ target_only** (69.3% ≈ 69.4%): 两种信号单独几乎等效，信息高度冗余
2. **type_target > Full oracle** (78.3% > 75.5%, +2.8pp):
   → 结构化指令 ("click" + "Settings icon") **超越**自然语言指令 ("Click on the Settings icon")
   → 自然语言的冗余/歧义反而**损害**了性能
   → 模型更擅长执行结构化 command 而非理解自然语言描述
3. **信息分解 (加法性检验)**:
   → type_only: +13.8pp, target_only: +13.9pp → 如果独立: 27.7pp
   → type_target 实际: +22.8pp → **互信息 ≈ 4.9pp** (两者共享的信息)
   → 但 22.8pp > 13.8pp，说明信息组合仍有互补价值 (+9.0pp beyond single)
4. **长 trajectory 上 target_only 衰减 (73.7%→60.0%)，type_only 不衰减 (72.4%→68.5%)**
   → type_target 在 vlong 上最好 (81.5%)，因为 type 稳定了 target 的使用
5. **type_target 的超 oracle 表现说明**: NL instruction 不是最优的 planning signal
   → 最优 planner 应输出 structured (action_type, target) pair，而非 NL sentence

**信息流图:**

```
                    +13.8pp (action type selection)
                   ┌──────────────────────────────┐
                   │                              │
Baseline ──────────┤     互信息 4.9pp              ├── type_target (+22.8pp)
(55.5%)            │    (type ↔ target 共享)       │    (78.3%)
                   │                              │
                   └──────────────────────────────┘
                    +13.9pp (target identification)

vs. Full oracle NL: +20.0pp (75.5%) — 自然语言格式损失 2.8pp
```

### 6.4 Q2: Oracle Error Decomposition — Grounding Ceiling Analysis ✅

**问题**: Oracle 24.5% grounding error 中，多少是 irreducible，多少是 model capacity？

**结果 (1006 oracle failures):**

| Category | Count | Proportion |
|----------|-------|------------|
| **Action type error** (instruction says X, model does Y) | 864 | **85.9%** |
| Coordinate error (right type, wrong location) | 133 | 13.2% |
| Text/app/button error | 9 | 0.9% |

→ **85.9% 的 oracle failures 是 action type error** — 即使 instruction 说 "open the app"，模型仍然输出 click/swipe/system_button
→ 这不是指令理解失败，而是 **action vocabulary failure** — 模型的 action space 缺少某些 action types

**Oracle failure rate 按 GT action type:**

| Action Type | Oracle Fail Rate | N |
|-------------|-----------------|---|
| **open** | **77.1%** | 607 |
| **wait** | **68.7%** | 163 |
| system_button | 45.7% | 256 |
| click | 11.6% | 2344 |
| swipe | 5.9% | 547 |
| type | 2.7% | 183 |

→ **`open` (77.1%) 和 `wait` (68.7%)** 是两个 "model doesn't know this action type" 的 action
→ `click`, `swipe`, `type` 的 oracle failure rate 都很低 (<12%)
→ 如果修复 open + wait，oracle accuracy 从 75.5% → **~89.6%** (理论 ceiling)

**Click coordinate precision (oracle failures with correct type):**

```
N=132 click errors, median distance = 490px, mean = 673px
Within 100px: 3.8%, within 200px: 18.9%
```

→ 不是精度问题——当 click 失败时，模型点在完全错误的位置（490px 偏差）
→ 真正的 irreducible error (屏幕歧义等) 估计 < 5% of all steps

**结论:**
- Oracle 的 24.5% grounding error 几乎全部是 **model capacity failure**，irreducible error 极小
- 修复 `open` + `wait` action type 就能将 oracle ceiling 从 75.5% 提升到 ~90%
- Coordinate errors 是 visual grounding 问题（看到了错误的元素），不是精度问题

---

## 7. Mechanism Framework: Beyond Diagnostics

> 从"哪里出错"到"为什么出错"——AR model 在 sequential decision making 中的结构性局限

### 7.1 核心问题

> **Long-horizon failure 是 representational limit (信息不在模型内部) 还是 inference limit (信息在但没被使用)?**

### 7.2 Q1: Task Progress State Probing ✅ → **Hypothesis B CONFIRMED**

**假设:**
- **Hypothesis A**: 模型 hidden states 中不存在 task progress state 表示 → 需要 explicit state module
- **Hypothesis B**: 信息存在但 autoregressive decoding 没有访问机制 → fine-tuning 可解决

**实验**: 提取 oracle 条件下 200 episodes (1063 samples) 每步最后 token 的 hidden states (layers 0/7/14/21/27)，训练线性 probe (Ridge/LogisticRegression, 5-fold CV)。

**结果:**

| Layer | Position R² | Action Type Acc | Boundary Acc | Completion R² |
|-------|-------------|-----------------|--------------|---------------|
| 0 (embedding) | ~0 | 57.6% (=majority) | 58.8% (=majority) | ~0 |
| 7 | 0.157 | 73.9% | 74.2% | 0.431 |
| 14 | 0.417 | 88.7% | 76.9% | 0.589 |
| 21 | **0.508** | **94.0%** | **88.5%** | **0.650** |
| 27 (last) | **0.528** | **94.2%** | **87.3%** | **0.677** |

Baselines: Action type majority=57.6%, Boundary majority=58.8%, 7 action types (click 57.6%, swipe, type, open, system_button, wait, long_press)

**关键发现:**

1. **Action type 94.2% 线性可解码** vs 模型实际输出 ~57% accuracy → **37pp internal-vs-output gap**
   - 模型 hidden state 几乎完美地编码了当前步该做什么 action type
   - 但 autoregressive decoder 没有利用这个信息
   - 这是最直接的 **Hypothesis B 证据**

2. **Boundary detection 88.5%** vs 58.8% majority → 模型知道何时 action type 要切换
   - 但在 Analysis C 中我们看到 22.5% 的失败是 action inertia (重复上一步 action type)
   - 说明 boundary 信号存在于 hidden states，但 decoder 偏好惯性路径

3. **Position 和 Completion 线性编码** (R²=0.53/0.68) → 模型有 task progress 的内部表示
   - 但在 5.1 中我们看到 progress misestimation 导致 premature termination
   - 信息存在但在 decoding 时没有被正确使用

4. **Layer-wise progressive encoding**: 信息从 layer 7 开始出现，在 layer 21 达到最高
   - Layer 0 ≈ random → embedding 层不编码 task state
   - 中间层 (14-21) 已经有丰富的 task state 表示
   - 最后一层不总是最好的 (boundary acc 在 layer 21 最高: 88.5% vs layer 27: 87.3%)

**结论: Long-horizon failure 是 inference limit，不是 representational limit**

→ 模型的 hidden states 已经"知道"正确的 action type (94%)、当前位置 (R²=0.53)、是否在边界 (88.5%)
→ 但 autoregressive decoding 无法有效利用这些信息
→ 这意味着 **不需要新的架构组件**，而是需要：
  - 更好的 decoding strategy (e.g., self-consistency, chain-of-thought)
  - Fine-tuning 帮助 decoder 访问已有的内部表示
  - 或者 lightweight adapter 把 hidden state 信号路由到 output
→ **这是本分析最重要的发现之一**：解决方案在 fine-tuning/decoding 层面，不在架构层面

### 7.3 Q3: Signal Degradation — 94% 内部信号如何退化为 57% 输出 (Phase A) ✅

**问题**: Q1 发现 layer 21 probe accuracy 94%，但模型实际输出 accuracy 仅 ~57%。这 37pp 的 gap 是**解码失败**（信息在但 decoder 用不了）还是**表示失败**（信息不在）？

**方法**: 将 probe 的 cross-validated 预测与 oracle trajectory 实际输出逐样本匹配 (503 matched samples)，分类每个 oracle output failure：
- **Decoding failure**: probe 预测正确 + 模型输出错误 → 信息存在但 decoder 未利用
- **Representation failure**: probe 预测错误 + 模型输出错误 → 信息本身缺失

**结果:**

| Layer | Probe CV Acc | Output Failures | Decoding Failure | Representation Failure |
|-------|-------------|-----------------|------------------|----------------------|
| 0 | 57.6% | 119 (23.7%) | 16 (13.4%) | 103 (86.6%) |
| 7 | 73.7% | 119 (23.7%) | 90 (75.6%) | 29 (24.4%) |
| 14 | 88.7% | 119 (23.7%) | 107 (89.9%) | 12 (10.1%) |
| **21** | **94.0%** | **119 (23.7%)** | **108 (90.8%)** | **11 (9.2%)** |
| **27** | **94.2%** | **119 (23.7%)** | **111 (93.3%)** | **8 (6.7%)** |

**关键发现:**

1. **Layer 21: 90.8% 的 oracle 输出失败是 decoding failure** — probe 正确预测了 action type，但模型输出仍然错误
   - 仅 9.2% 是 representation failure（信息确实不在 hidden state 中）
   - **这是 Hypothesis B 的最强证据: 瓶颈在 decoding，不在 representation**

2. **Layer-wise 信号出现过程**: Layer 0 几乎全是 representation failure (86.6%)，到 layer 7 已翻转为 decoding failure 主导 (75.6%)，layer 14+ 稳定在 ~90% decoding failure
   - 信息从 layer 7 开始被编码，但从未被 decoder 有效利用

3. **Boundary failures**: 在 14 个 boundary failure 中，layer 21 有 10 个 (71.4%) 是 decoding failure → 模型在 transition 处也"知道"正确答案

**结论**: Oracle 24.5% grounding error 中，绝大多数 (>90%) 的失败是**模型知道正确答案但 decoder 无法输出**。这进一步强化了"inference limit, not representational limit"的结论。

### 7.3.1 Q3 Phase B: Gap Decomposition Table ✅

Q6 SLURM 完成后，构建完整的 gap 分解表:

| Layer | Oracle Probe (A) | Baseline Probe (B) | Cross-Apply (C) | Repr Gap (A−B) | Decoding Gap (B−output) |
|-------|-----------------|-------------------|-----------------|----------------|------------------------|
| 0 | 57.6% | 57.6% | 57.6% | 0.0pp | 0.6pp |
| 7 | 73.9% | 62.4% | 69.1% | **11.6pp** | 5.4pp |
| 14 | 88.7% | 69.8% | 53.0% | **18.9pp** | 12.8pp |
| **21** | **94.0%** | **79.0%** | **63.9%** | **15.0pp** | **22.0pp** |
| **27** | **94.2%** | **78.2%** | **73.3%** | **16.0pp** | **21.2pp** |

**三个 gap 的解释:**
- **Repr Gap (A−B)**: Oracle prompt 向 hidden states 注入的额外信息量 → layer 14 最大 (18.9pp)
- **Decoding Gap (B−output)**: 即使无 oracle，baseline hidden state 也编码了 79% 的 action type，但 output 仅 57% → **22pp 纯 decoding loss**
- **Cross-Apply (C)**: Oracle 和 baseline 的表示**部分共享**同一线性子空间，但 action type 维度差异较大 (layer 21: 63.9% vs self 94.0%)

**核心发现:**

1. **Baseline probe 79.0% >> output 57.0%** → 即使没有 oracle，模型内部也编码了比输出高 22pp 的 action type 信息
2. **Oracle 额外贡献 ~15pp** (94.0% − 79.0%)，主要在 layer 7-14 注入
3. **37pp total gap = 15pp repr gap + 22pp decoding gap** → decoding gap 是 repr gap 的 1.5x，是更大的瓶颈
4. **Cross-apply 在 layer 27 最高 (73.3%)** → 最后一层的表示在两种条件间最一致

### 7.4 Q4: Action Inertia 机制分析 ✅

**问题**: Q1 显示 boundary detection 88.5% 线性可解码，但 6.2 节中 22.5% 的失败是 action inertia（重复上一步 action type）。为什么检测到了 boundary 却仍然重复？

**数据**: Baseline 2905 steps (588 boundary steps) + Oracle 4106 steps (1034 boundary steps) + 完整 GT 序列

**核心数据:**

| 指标 | Baseline | Oracle | 差值 |
|------|----------|--------|------|
| Boundary type error rate | **26.0%** | **14.7%** | −11.3pp |
| Non-boundary type error rate | 14.4% | 8.4% | −6.0pp |
| Boundary/non-boundary 比值 | 1.80x | 1.75x | 基本相同 |

→ Oracle 降低了两种 error rate，但 **boundary 始终比 non-boundary 难约 1.8x**，oracle 没有改变这个比例

**分析 1: Inertia 与 run length 关系 (baseline)**

| 前序连续相同 action 步数 | Boundary 总数 | Inertia Rate | Type Error Rate |
|--------------------------|--------------|--------------|-----------------|
| 1 | 449 | 8.9% | 23.6% |
| 2 | 83 | **16.9%** | 34.9% |
| 3 | 35 | 14.3% | 25.7% |
| 4+ | 21 | 9.5% | 42.9% |

→ **Run length = 2 时 inertia 最高 (16.9%)**，但 4+ 时 type error rate 最高 (42.9%)，说明长 run 后的 boundary 虽然 inertia 不严重，但其他类型错误增加

**分析 2: Transition Pair Inertia (高频 pairs, n≥10)**

| Transition Pair | Baseline Inertia | Oracle Inertia | Oracle 改善 |
|----------------|------------------|----------------|------------|
| click→swipe | **29.3%** (n=82) | 4.5% (n=156) | **−24.8pp** ✅ |
| swipe→click | **27.0%** (n=37) | 2.5% (n=118) | **−24.5pp** ✅ |
| click→wait | 21.3% (n=75) | **38.9%** (n=126) | **+17.6pp** ❌ |
| system_button→click | 9.6% (n=52) | 2.6% (n=77) | −7.0pp |
| click→type | 2.6% (n=116) | 0.5% (n=183) | −2.1pp |
| type→click | 0.0% (n=91) | 0.6% (n=170) | 无变化 |

**最重要的发现:**

> **click→wait 是唯一 oracle 反而加重 inertia 的 transition** (21.3% → 38.9%)

→ Oracle 告诉模型 "wait"，但模型仍然 click — 这不是 planning failure，是 **action vocabulary failure**（模型不知道 wait 这个 action）
→ 与 6.4 节结果一致: wait 的 oracle failure rate = 68.7%
→ click→swipe 和 swipe→click 的 inertia 被 oracle 几乎消除 (−24pp)，说明这些才是 planning-fixable 的 inertia

**分析 3: Inertia 与 trajectory position**

| Position | Baseline Inertia | Oracle Inertia |
|----------|-----------------|----------------|
| 0-20% | 9.1% | 4.5% |
| 20-40% | 10.1% | 4.8% |
| 40-60% | **12.8%** | 4.6% |
| 60-80% | 6.6% | 5.9% |
| 80-100% | 11.7% | **10.0%** |

→ Baseline inertia 在 40-60% 位置最高，trajectory 末端 (80-100%) 在 oracle 条件下仍然偏高 (10.0%)
→ 末端的 inertia 可能与 premature termination 机制重叠

**结论:**
- **Action inertia 有两种机制**: (a) Planning-fixable inertia (click↔swipe)，oracle 可以消除; (b) Vocabulary-driven inertia (click→wait)，oracle 反而加重，因为模型根本不认识 wait action
- Boundary 的额外难度 (1.8x error rate) 是 **恒定的**，不随 oracle 条件变化
- Inertia 与 run length 呈弱正相关，但 **transition pair type 是更强的预测因子**

#### 7.4.1 Q4 GUI-360 Replication ✅ (200 episodes 初步结果; 全量 3439 episodes 评测中 SLURM #3162440)

> **数据**: baseline.json (200 eps, 2201 steps, type_acc=46.8%) vs sft.json (200 eps, 2201 steps, type_acc=61.8%)
> **Action space**: ~16 types (click 64.8%, type 16.4%, select_text 2.7%, drag 1.5%, summary 1.0%, select_table_range 1.0%, wheel_mouse_input 1.0%, 其余 <1%; 另有 236 步 gt_type 为空已过滤)
> **GT 来源**: gui360_train.jsonl (eval 200 eps 匹配 train split，非 test split)
> **对比**: SFT vs baseline (替代 AC 的 oracle vs baseline)

**运行脚本**: `python scripts/eval/ac/analysis_q4_inertia_gui360.py`
**输出**: `outputs/analysis_q4_inertia_gui360/q4_results.json`

**结果 (200 eps 初步)**:

| 指标 | AC Baseline | AC Oracle | GUI-360 Baseline | GUI-360 SFT |
|------|------------|-----------|------------------|-------------|
| Boundary error rate | 26.0% | 14.7% | **66.2%** | **57.2%** |
| Non-boundary error rate | 14.4% | 8.4% | **41.9%** | **32.0%** |
| Boundary/non-boundary ratio | 1.80x | 1.75x | **1.58x** | **1.79x** |
| Total boundary steps | 588 / 588 | | 556 | 556 |
| Skipped (empty gt_type) | 0 | | 236 | 236 |

→ **GUI-360 boundary/non-boundary ratio (1.58x-1.79x) 与 AC (1.75x-1.80x) 高度一致**
→ GUI-360 整体 error rate 更高 (baseline 66.2% vs AC 26.0%)，反映 desktop action space 更难
→ SFT 降低 boundary error 9pp (66.2%→57.2%)，降低 non-boundary error 10pp (41.9%→32.0%)

**分析 1: Inertia by Run Length (Baseline)**

| 前序连续相同 action 步数 | Boundary 总数 | Inertia Rate | Type Error Rate |
|--------------------------|--------------|--------------|-----------------|
| 1 | 370 | 18.6% | 65.4% |
| 2 | 62 | **33.9%** | 62.9% |
| 3 | 48 | 29.2% | 58.3% |
| 4 | 32 | 31.2% | 59.4% |
| 5+ | 44 | **34.1%** | **90.9%** |

→ **Run length ≥ 2 时 inertia 显著跳升 (18.6% → 33%+)**，与 AC 趋势一致
→ GUI-360 inertia 整体高于 AC (18-34% vs 9-17%)，desktop 更多重复点击序列

**分析 2: Transition Pair Inertia (n≥10, Baseline vs SFT)**

| Transition Pair | Baseline Inertia | SFT Inertia | 改善 |
|----------------|------------------|-------------|------|
| type→click | **39.8%** (n=133) | **18.8%** (n=133) | **−21.1pp** ✅ |
| click→summary | **43.8%** (n=16) | **43.8%** (n=16) | 0pp ❌ |
| type→select_text | 34.3% (n=35) | 28.6% (n=35) | −5.7pp |
| click→type | 20.1% (n=169) | 20.1% (n=169) | 0pp |
| click→drag | 20.8% (n=24) | 20.8% (n=24) | 0pp |
| drag→click | 15.4% (n=13) | **38.5%** (n=13) | **+23.1pp** ❌ |

**关键发现:**

> **type→click 是 SFT 显著改善的 transition** (39.8% → 18.8%, −21pp)，类似 AC 中 click→swipe 被 oracle 消除
> **click→summary 和 drag→click 是 SFT 无法修复甚至加重的 transition**，类似 AC 中 click→wait 的 vocabulary failure
> drag→click inertia SFT 反而加重 (+23pp)，说明 drag 操作后模型倾向于继续 drag

→ **GUI-360 同样呈现两种 inertia 机制**: (a) Planning-fixable (type↔click)，SFT 可改善; (b) Vocabulary/rare-action driven (click→summary, drag→click)，SFT 无效或加重

**Cross-Dataset Q4 对比**:
- ✅ Boundary/non-boundary difficulty ratio **跨数据集一致** (AC 1.8x ≈ G360 1.6-1.8x)
- ✅ **两种 inertia 机制跨数据集成立**: planning-fixable + vocabulary-driven
- ✅ Run length 与 inertia 正相关跨数据集一致
- 差异: GUI-360 整体 error rate 更高，inertia rate 更高 — desktop action space 更复杂

### 7.5 Q5: Pattern Memorization vs Compositional Generalization ✅

**问题**: ~20pp 的 planning error 是因为模型只记住了训练中见过的 action 模式（memorization），还是因为无法组合新的 action 序列（compositional failure）？

**方法**: 对 1543 episodes 计算 action bigram novelty score（每个 episode 的 action type bigram 的平均 log-inverse frequency），按 novelty 分组比较 baseline 和 oracle accuracy。

**分析 1: App 提取**
- 正则匹配覆盖 80.2% episodes (1237/1543)，553 个独特 app
- Top apps: gmail (42), pinterest (33), artsy (32), culture (28)

**分析 2: Novelty-binned Accuracy**

| Novelty 分位 | 平均 Novelty | Baseline StepAcc | Oracle StepAcc | Gap (Δ) |
|-------------|-------------|------------------|----------------|---------|
| Q1 (最常见) | 0.59 | **60.1%** | **80.4%** | 20.3pp |
| Q2 | 1.60 | 28.6% | 47.9% | 19.3pp |
| Q3 | 2.09 | 22.5% | 39.9% | 17.4pp |
| Q4 | 2.54 | 25.0% | 44.7% | 19.8pp |
| Q5 (最新颖) | 3.30 | 25.7% | 42.3% | **16.6pp** |

→ **Oracle-baseline gap 在各 novelty 分位间基本恒定 (~17-20pp)**，没有随 novelty 增加而扩大
→ 粗看不支持"新颖组合更难 plan"的假说
→ **但**: Q1 (常见模式) 的绝对 accuracy 远高于 Q2-Q5，说明 **frequency 影响 baseline accuracy，但不影响 oracle gap**

**分析 3: Difficulty-Controlled Analysis (关键)**

控制 oracle accuracy（排除 intrinsic difficulty），在相同 oracle 难度内比较高/低 novelty:

| Oracle 难度层 | Low Novelty Baseline | High Novelty Baseline | Gap 差异 |
|-------------|---------------------|----------------------|---------|
| Low oracle (0%-0%) | 5.6% | 5.1% | 无意义 (全部失败) |
| Mid oracle (0%-100%) | 45.4% | 37.8% | Low novelty gap=25.3pp, High=23.1pp |
| **High oracle (100%)** | **60.9%** | **38.7%** | **Low novelty gap=27.3pp, High=47.3pp** |

> **在 high-oracle-accuracy episodes（oracle 全对的容易任务）中: 高 novelty 的 baseline accuracy 大幅下降 (60.9% → 38.7%)**

→ **Gap 从 27.3pp 扩大到 47.3pp** — 在容易任务上，新颖 action 组合的 planning failure 是常见模式的 **1.7x**
→ 这是 **compositional failure 的确凿证据**: 对于 intrinsically easy tasks，模型在 novel compositions 上表现显著更差
→ Q5 总体 gap 看似恒定，是因为 novel episodes 本身更难 (oracle accuracy 也更低)，混淆了 difficulty 和 novelty

**分析 4: Action Type Diversity**

| 独特 Action Types 数 | Baseline StepAcc | Oracle StepAcc | Gap |
|---------------------|------------------|----------------|-----|
| 1 | 59.1% | 80.9% | 21.8pp |
| 2 | 35.9% | 62.8% | **26.9pp** |
| 3 | 19.3% | 35.3% | 16.1pp |
| 4 | 15.7% | 20.0% | 4.3pp |
| 5 | 23.3% | 24.9% | 1.5pp |

→ **2 types 时 gap 最大 (26.9pp)** — 单一 transition 是 planning 最难的点
→ 4-5 types 时 gap 几乎消失 — 这些 episode 对 oracle 也很难 (intrinsic difficulty dominates)

**分析 5: App Frequency**
- log(app_frequency) vs baseline_accuracy 相关系数: r = −0.024（几乎无相关）
→ App 频率不影响 accuracy，**不支持 memorization 假说**

**结论:**
- **在控制 difficulty 后，compositional failure 确实存在**: 容易任务上 novel composition gap 是常见模式的 1.7x
- **但 memorization 不是瓶颈**: app frequency 与 accuracy 无关
- **核心机制**: 模型在训练中学会了常见 action bigram 的 transition 规则，但对少见的 bigram 组合（如 open→wait→swipe）缺乏泛化能力
- **解决方案指向**: 增加 action sequence diversity 的训练数据，或使用 structured (type, target) planner 绕过 composition 问题

#### 7.5.1 Q5 GUI-360 Replication ✅ (200 episodes 初步结果; 全量评测运行中)

> **数据**: baseline.json + sft.json, GT from gui360_train.jsonl (13750 eps 用于 bigram 频率计算)
> **Domain**: 100% 匹配 — word 5633, excel 4348, ppt 3769
> **Bigram space**: 136 unique bigrams (vs AC: ~20), 76264 total
> **对比**: SFT vs baseline

**运行脚本**: `python scripts/eval/ac/analysis_q5_compositional_gui360.py`
**输出**: `outputs/analysis_q5_compositional_gui360/q5_results.json`

**分析 1: Novelty-binned Accuracy**

| Novelty 分位 | 平均 Novelty | Baseline StepAcc | SFT StepAcc | Gap (Δ) |
|-------------|-------------|------------------|-------------|---------|
| Q1 (最常见) | 0.49 | 18.5% | 37.9% | **19.4pp** |
| Q2 | 0.81 | 11.6% | 35.6% | **24.0pp** |
| Q3 | 1.44 | 17.1% | 35.2% | 18.0pp |
| Q4 | 2.10 | 17.3% | 34.9% | 17.6pp |
| Q5 (最新颖) | 3.41 | 21.9% | 38.1% | **16.2pp** |

→ **SFT-baseline gap 从 Q1=19.4pp 逐渐下降到 Q5=16.2pp**，微弱趋势
→ AC 中 oracle-baseline gap 也是基本恒定 (~17-20pp)
→ **注意**: Q5 (最新颖) 的 baseline accuracy (21.9%) 反而略高于 Q2 (11.6%)，说明 novelty 和 difficulty 混淆

**分析 2: Difficulty-Controlled Analysis (关键)**

控制 SFT accuracy（排除 intrinsic difficulty），在相同 SFT 难度内比较高/低 novelty:

| SFT 难度层 | Low Novelty BL | High Novelty BL | Low Novelty Gap | High Novelty Gap |
|-----------|---------------|----------------|----------------|-----------------|
| Low SFT (0.14-0.57) | 17.6% | 18.2% | 2.9pp | 9.2pp |
| Mid SFT (0.57-0.83) | 14.3% | 17.0% | 18.0pp | 23.7pp |
| **High SFT (0.83-1.00)** | **14.7%** | **21.8%** | **28.9pp** | **30.9pp** |

→ **在 high-SFT-accuracy 的容易任务上**: high novelty gap (30.9pp) > low novelty gap (28.9pp)
→ 差异 ~2pp，远小于 AC 的 1.7x (27pp→47pp)
→ **初步结论: GUI-360 上 compositional failure 信号较弱**，可能因为 desktop action space 中 action type diversity 较低（大多数 episode 仅 1-2 types）

**分析 3: Domain Frequency**

| Domain | Eval Episodes | Dataset Count | Baseline StepAcc | SFT StepAcc | Gap |
|--------|-------------|--------------|------------------|-------------|-----|
| Excel | 61 | 4348 | 14.3% | 33.4% | 19.2pp |
| Word | 78 | 5633 | 20.3% | 37.3% | 17.0pp |
| PPT | 61 | 3769 | 16.4% | 37.9% | 21.5pp |

→ Domain frequency 与 baseline accuracy 相关 r=0.751（word 最多也最准），但 sample size 太小（3 domains）不具统计意义
→ AC 中 app frequency 与 accuracy 无关 (r=−0.024)

**分析 4: Action Type Diversity**

| 独特 Action Types 数 | n_episodes | Baseline StepAcc | SFT StepAcc | Gap |
|---------------------|-----------|------------------|-------------|-----|
| 1 | 55 | 16.2% | 38.4% | **22.2pp** |
| 2 | 88 | 17.5% | 38.1% | **20.6pp** |
| 3 | 48 | 18.0% | 31.2% | 13.3pp |
| 4 | 9 | 17.7% | 32.9% | 15.2pp |

→ **1-2 types 时 gap 最大 (~20-22pp)**，3+ types 时 gap 下降
→ 与 AC 模式一致: 2 types gap 最大 (AC: 26.9pp)

**Cross-Dataset Q5 对比**:
- ✅ **Novelty-binned gap 跨数据集基本恒定** (AC ~17-20pp, G360 ~16-24pp)
- ⚠️ **Difficulty-controlled compositional failure 在 G360 上信号较弱** (gap ratio ~1.1x vs AC 1.7x)
  - 可能原因: GUI-360 action type diversity 较低 (73% episodes 仅 1-2 types)，限制了 compositional challenge
- ✅ **Action type diversity 与 gap 的关系跨数据集一致**: 2 types 最难，4+ types gap 消失
- ⚠️ Domain frequency 仅 3 个 domain，无法充分检验 memorization 假说

### 7.6 Q6: Baseline Probing — 模型是否不依赖 Oracle 也"知道"? ✅

**问题**: Q1 的 94% probe accuracy 是在 oracle 条件下（prompt 包含 step_instruction）获得的。是 oracle prompt 赋予了这个信息，还是模型固有就编码了 action type？

**实验设计**: Fork `probe_hidden_states.py`，唯一改变是 `build_baseline_messages()` 省略 step_instruction。三个 mode:
1. **extract**: 无 oracle 的 forward pass，提取 5 层 hidden states
2. **probe**: 在 baseline hidden states 上训练 probe (5-fold CV)
3. **cross_apply**: 用 oracle 训练的 probe 直接应用于 baseline hidden states

**结果:**

| Layer | Oracle Probe Acc | Baseline Probe Acc | Cross-Apply Acc | Baseline Boundary Acc |
|-------|-----------------|-------------------|-----------------|----------------------|
| 0 | 57.6% (=majority) | 57.6% (=majority) | 57.6% | 58.8% (=majority) |
| 7 | 73.9% | 62.4% | 69.1% | 73.1% |
| 14 | 88.7% | 69.8% | 53.0% | 74.1% |
| **21** | **94.0%** | **79.0%** | **63.9%** | **78.6%** |
| **27** | **94.2%** | **78.2%** | **73.3%** | **77.7%** |

**关键发现:**

1. **Baseline probe 79.0% — 模型固有能力显著高于 output (57%)**
   - 即使没有 oracle step_instruction，模型 hidden state 已编码 79% 的 action type
   - 这 79% 中有 22pp 被 decoder 丢失 (79% → 57% output)
   - Oracle 在此基础上额外注入 ~15pp (79% → 94%)

2. **Oracle 贡献的 15pp 信息主要在 layer 7-14 注入**
   - Layer 7: 差距 11.6pp (73.9% vs 62.4%)
   - Layer 14: 差距最大 18.9pp (88.7% vs 69.8%)
   - Layer 21+: 差距缩小到 15.0pp (baseline 在 deep layers 追赶)

3. **Cross-apply 揭示表示空间差异**
   - Layer 14 cross-apply 仅 53.0% (甚至低于 baseline probe 69.8%) → 两种条件下 action type 编码在**不同的线性子空间**
   - Layer 27 cross-apply 73.3% (最高) → 最后一层的表示在两种条件间最一致
   - 这意味着 oracle prompt 不仅增强信号，还**重组了中间层的表示几何**

4. **Boundary detection 也有 baseline 能力 (78.6% at layer 21)**
   - Oracle: 88.5%, Baseline: 78.6% → oracle 提升 ~10pp
   - 模型固有就能检测 ~79% 的 boundary

**结论: 模型固有编码 79% action type (vs output 57%)，oracle 额外注入 15pp**

→ 37pp total gap 分解: **22pp decoding gap** (baseline probe → output) + **15pp repr gap** (oracle → baseline probe)
→ **Decoding gap (22pp) > Repr gap (15pp)**: 最大瓶颈是 decoder 无法利用自身已有的表示，而非缺少信息
→ 但 15pp repr gap 也不小: oracle prompt 确实在 hidden states 中注入了额外的规划信号
→ **最优解决方案**: 同时攻击两个 gap — (1) CoT/adapter 解锁 22pp decoding gap, (2) planner 补充 15pp repr gap

#### 7.6.1 Q6 + Q3 GUI-360 Replication (待 GPU 数据)

> **状态**: 需要先完成全量 eval (SLURM #3162440)，再提交 probe extraction (probe_gui360_extract.slurm)
> **Probe 脚本**: `scripts/eval/ac/probe_gui360_extract.py` (4 modes: extract_oracle, extract_baseline, probe, cross_apply)
> **SLURM**: `sbatch scripts/eval/ac/probe_gui360_extract.slurm` (6h, 2 GPU)
> **Q3 分析**: `python scripts/eval/ac/analysis_q3_signal_degradation_gui360.py --phase AB`

**执行依赖链**:
1. ✅ 分析脚本已就绪
2. 🔄 全量 eval 运行中 (SLURM #3162440, baseline + SFT on 3439 test episodes)
3. ⏳ Probe extraction 待提交 (eval 完成后)
4. ⏳ Q3 分析 (probe 完成后)

**Probe Results** (probe SLURM 完成后填充):

| Layer | Oracle Probe Acc | Baseline Probe Acc | Cross-Apply Acc | Repr Gap | Decoding Gap |
|-------|-----------------|-------------------|-----------------|----------|-------------|
| 0 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 7 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 14 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **21** | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **27** | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Q3 Signal Degradation** (probe 完成后填充):
- Decoding failure fraction: _TBD_ (AC: 90.8%)
- Representation failure fraction: _TBD_ (AC: 9.2%)

**Cross-Dataset Q3/Q6 对比**:

| 指标 | AC | GUI-360 |
|------|-----|---------|
| Oracle probe (L21) | 94.0% | _TBD_ |
| Baseline probe (L21) | 79.0% | _TBD_ |
| Output accuracy | 57% | 46.8% |
| Decoding gap | 22pp | _TBD_ |
| Repr gap | 15pp | _TBD_ |
| Decoding failure % | 90.8% | _TBD_ |

→ 验证 "decoding gap > repr gap" 是否跨数据集成立

### 7.7 Evidence Summary

| Finding | Implication |
|---------|------------|
| **🔑 Hidden states 编码 action type at 94%** | **Hypothesis B: inference limit, not representational limit** |
| **🔑 Internal-vs-output gap: 94% → 57%** | **Decoder 无法访问自身的内部表示，是核心瓶颈** |
| **🔑 90.8% oracle failures 是 decoding failure (Q3)** | **信息存在于 layer 21 但 decoder 未利用** |
| **🔑 37pp gap = 22pp decoding + 15pp repr (Q6)** | **Decoding gap > repr gap, 最大瓶颈在 decoder** |
| **🔑 Baseline probe 79.0% without oracle (Q6)** | **模型固有编码远超 output，oracle 额外贡献 15pp** |
| **🔑 click→wait inertia: oracle 反而加重 (Q4)** | **wait 是 vocabulary failure, 不是 planning failure** |
| **🔑 Difficulty-controlled compositional gap: 27pp→47pp (Q5)** | **Novel compositions 的 planning failure 是常见模式的 1.7x** |
| Action type 贡献 20pp gap 中的 13.8pp (69%) | Planner 核心任务是 action type prediction |
| **type_only ≈ target_only (69.3% ≈ 69.4%)** | Action type 和 target 信息高度冗余 |
| **type_target (78.3%) > Full oracle NL (75.5%)** | **结构化指令优于自然语言，planner 应输出 structured pair** |
| **type_only 在长 traj 不衰减，target_only 衰减** | Action type 是更稳定的 planning signal |
| Oracle 85.9% failures 是 action type error | 模型 action vocabulary 不完整 |
| `open` + `wait` 占 oracle failure 的 57.7% | 修复两个 action type → oracle ceiling ~90% |
| Boundary 26.0% error vs non-boundary 14.4% (Q4) | Boundary 恒定比 non-boundary 难 1.8x |
| Boundary error rate: baseline 26.0% → oracle 14.7% (Q4) | Oracle 降低 inertia 但不改变 boundary/non-boundary 比例 |
| App frequency 与 accuracy 无关 r=−0.024 (Q5) | Memorization 不是瓶颈 |
| Completion R²=0.68 vs progress misestimation 6.6% | 模型知道完成度但仍 premature terminate |
| History quantity 不影响 (<1pp K=0~K=5) | 需要 semantic state，非 history management |
| Agreement rate 是最强 reliability signal | 可用于 confidence routing |

### 7.8 Revised Solution Priority (based on Q1-Q5 Analyses)

```
Priority 1: Decoder-Internal Representation Utilization (最高杠杆 — Q1+Q3+Q6)
  → Q6 证实: baseline (无 oracle) hidden state 已编码 79% action type，但 output 仅 57%
  → 22pp 纯 decoding loss 是最大的可解锁空间（不需要任何外部 planner）
  → Q3 证实: 90.8% 的 oracle 输出失败是 decoding failure
  → 方案:
    a) Chain-of-thought fine-tuning: 让模型先输出 action type 判断，再执行
    b) Lightweight adapter: 从 layer 21 hidden state route action type signal
    c) Self-consistency decoding: 多次采样 → majority vote (已验证 agreement 有效)
  → 预期: StepAcc 57% → 65-70% (不需要外部 planner)

Priority 2: Action Type Vocabulary Training (Q4 验证)
  → type_only 已带来 +13.8pp StepAcc (与 target_only 等效)
  → Q4 发现: click→wait inertia 在 oracle 下反而加重 (21%→39%)，因为模型不认识 wait
  → 核心: 上采样 open/wait action (这两个 type 占 oracle failure 的 57.7%)
  → 预期: oracle ceiling 75.5% → ~90%

Priority 3: Compositional Generalization Training (Q5 新发现)
  → Q5 证实: 在控制 difficulty 后，novel compositions gap 是常见模式的 1.7x
  → 增加 action sequence diversity 的训练数据
  → 或使用 structured planner 绕过 composition 问题

Priority 4: Structured (type, target) Planner
  → type_target (78.3%) > full oracle NL (75.5%) → 结构化输出优于 NL
  → Planner 输出 structured pair: (action_type, target_description)
  → 比 NL step_instruction 更好，因为减少了歧义
  → 在 vlong trajectory 上尤其突出 (81.5% vs 76.7%)

Priority 5: Confidence-Aware Routing
  → Agreement rate routing: high agreement (>0.8) → trust greedy (91.3% acc)
  → Low agreement (<0.4) → trigger planner/resample
  → 推理时即可使用，无需训练
```

**核心洞察: 从 "model doesn't know" 到 "model knows but can't output"**

Q1-Q6 综合结论：
- Q1: 模型内部编码了 94% action type accuracy (oracle 条件)
- Q6: **即使无 oracle，baseline probe 也达 79%** → 模型固有能力远超 output (57%)
- Q3: **90.8% 的输出失败是 decoding failure**; **37pp gap = 22pp decoding + 15pp repr**
- Q4: Action inertia 有两种机制——planning-fixable (click↔swipe) 和 vocabulary-driven (click→wait)
- Q5: **Compositional failure 在容易任务上确实存在** (gap 1.7x)，但 memorization 不是瓶颈
- **最大瓶颈是 22pp decoding gap**（decoder 无法利用自身已有的 79% 内部表示），其次是 15pp repr gap（可通过 planner 补充）
- 三个最高杠杆干预: **CoT/adapter 解锁 decoding gap** + **action vocabulary 扩充 (open/wait)** + **composition diversity training**

### 7.9 Open Questions (Mechanism Level)

1. ~~**Q1 Probing**: Hidden states 是否编码了 task progress?~~ ✅ **YES — Hypothesis B confirmed**
2. ~~**Oracle ablation**: type_target 结果~~ ✅ **type_target (78.3%) > full oracle NL (75.5%)**
3. ~~**Q3: Signal Degradation**~~ ✅ **90.8% decoding failure; 37pp gap = 22pp decoding + 15pp repr**
4. ~~**Q4: Action Inertia Mechanism**~~ ✅ **两种机制: planning-fixable vs vocabulary-driven**
5. ~~**Q5: Compositional Generalization**~~ ✅ **Difficulty-controlled 证实 compositional failure (gap 1.7x)，memorization 不是瓶颈**
6. ~~**Q6: Baseline Probing**~~ ✅ **Baseline probe 79.0% (vs oracle 94%, output 57%) → 22pp decoding gap + 15pp repr gap**

---

## Appendix: AC-Specific Mechanisms

以下机制仅在 AC 中显著，GUI-360 中因 output format inconsistency 无法观测：

| Mechanism | Evidence |
|-----------|---------|
| **Premature Termination** | terminate rate: 0.9%→40% with position |
| **Action Repetition Loop** | loop rate: 0.8%→34.7% with traj length |
| **Goal Drift** | 5+步正确后首次错误 42.6% 是 terminate |

统一解释：**Progress Estimation Failure** — Agent 缺乏 explicit progress tracking，将"任务进展"误判为"任务完成"。

---

## 8. Adaptive Replanning Evaluation (E1-E4) — AC Results

核心假设：在 subgoal boundary 处，planner 重新激活并生成 structured message，executor 重置 context 到 O(1)，从而缓解 context degradation 和 action inertia。

所有实验使用 teacher-forced GT screenshots + agent-predicted action history。

### 8.1 E4: Midpoint Replanning (最重要)

**设计**: 对所有 8+ 步的长 trajectory，在 T//2 处触发一次 adaptive replanning (structured protocol + context reset)。

- **Baseline (A)**: 标准 AR，growing context，无 replanning
- **Adaptive (B)**: 在 step T//2 处 planner 生成 structured message，executor 重置 context

**数据**: 317 long trajectories (289 long(8-15) + 28 vlong(16+)), 3316 total steps

| 指标 | Baseline | Adaptive | Δ |
|------|----------|----------|---|
| **Overall StepAcc** | 53.8% | **60.2%** | **+6.4pp** |
| **Overall TypeAcc** | 64.9% | **70.5%** | **+5.6pp** |
| **Second-Half StepAcc** | 51.8% | **64.0%** | **+12.2pp** |
| Second-Half TypeAcc | 62.8% | **73.6%** | **+10.8pp** |

**Per-Length Breakdown (Second-Half StepAcc)**:

| Length Bucket | n | Baseline | Adaptive | Δ |
|--------------|---|----------|----------|---|
| long(8-15) | 289 | 54.3% | **66.5%** | **+12.2pp** |
| vlong(16+) | 28 | 37.9% | **50.0%** | **+12.1pp** |

**Paired Comparison** (per-trajectory second-half StepAcc):

| 结果 | 数量 |
|------|------|
| Adaptive better | **155** (48.9%) |
| Baseline better | 55 (17.4%) |
| Tied | 107 (33.8%) |

> **关键发现**: 仅一次 midpoint replanning 即可将 second-half StepAcc 提升 +12.2pp (51.8%→64.0%)。Adaptive 在 48.9% 的 trajectory 上优于 baseline，仅 17.4% 劣于 baseline (2.8:1 win ratio)。效果在 long 和 vlong trajectories 上高度一致 (+12pp)。

### 8.2 E2: Communication Protocol Ablation

**设计**: 固定 oracle boundary detection，测试 5 种 planner communication protocol 对 executor 的影响。

**数据**: 1543 trajectories, 8444 steps, oracle 平均 3.2 planner calls/traj

| Protocol | StepAcc | TypeAcc | TSR | Boundary Acc | Within-Phase Acc |
|----------|---------|---------|-----|-------------|-----------------|
| **structured** | **76.2%** | **83.3%** | **32.8%** | **79.5%** | 71.4% |
| nl_instruction | 73.2% | 80.2% | 29.1% | 74.3% | 71.6% |
| type_only | 71.5% | **84.0%** | 26.1% | 69.8% | **73.9%** |
| structured_progress | 71.4% | 79.1% | 27.9% | 71.1% | 71.7% |
| none (reset only) | 58.2% | 69.0% | 17.2% | 53.7% | 64.5% |

**Per-Length StepAcc**:

| Protocol | short(1-3) | medium(4-7) | long(8-15) | vlong(16+) |
|----------|-----------|-------------|------------|-----------|
| structured | 76.6% | 77.8% | **76.0%** | 63.4% |
| nl_instruction | 75.2% | 73.8% | 73.0% | **65.1%** |
| type_only | 72.9% | 72.2% | 71.9% | 60.2% |
| structured_progress | 73.9% | 72.6% | 70.4% | 62.0% |
| none | 59.7% | 60.0% | 56.7% | 48.0% |

> **关键发现**:
> 1. **Structured > NL instruction** (76.2% vs 73.2%, +3.0pp) — 与 7.2 节 oracle ablation 结论一致 (type_target 78.3% > full NL 75.5%)
> 2. **Context reset alone (none) 也有价值** — 58.2% vs no_boundary baseline 62.1%: none 略低于不做 boundary 的 baseline，说明 context reset 不加 planner 信号会丢失信息
> 3. **type_only 的 TypeAcc (84.0%) 最高** — 因为直接提供了 action type，但 StepAcc 低于 structured，说明 target 信息对 grounding 很重要
> 4. **structured_progress 不如 structured** (71.4% vs 76.2%) — progress summary 可能引入噪声

### 8.3 E1: Boundary Detection Comparison

**设计**: 固定 structured protocol，比较 3 种 boundary detection 策略。

**数据**: 1543 trajectories, 8444 steps

| Condition | StepAcc | TypeAcc | TSR | Planner Calls/Traj | Boundary Acc |
|-----------|---------|---------|-----|--------------------|--------------|
| **oracle** | **76.3%** | **83.3%** | **33.4%** | 3.2 | **79.7%** |
| fixed (K=3) | 69.8% | 78.1% | 25.4% | 2.2 | 77.2% |
| no_boundary | 62.1% | 72.0% | 22.2% | 1.0 | 63.8% |

**Per-Length StepAcc**:

| Condition | short(1-3) | medium(4-7) | long(8-15) | vlong(16+) |
|-----------|-----------|-------------|------------|-----------|
| oracle | **77.5%** | **78.0%** | **75.6%** | **64.5%** |
| fixed (K=3) | 67.1% | 70.7% | 70.0% | 66.3% |
| no_boundary | 67.5% | 65.1% | 58.4% | 47.4% |

> **关键发现**:
> 1. **Oracle >> fixed >> no_boundary**: 14.2pp gap (oracle vs no_boundary)，说明 boundary detection 是 adaptive replanning 的关键
> 2. **Fixed(K=3) 在 vlong 上接近 oracle** (66.3% vs 64.5%) — 长 trajectory 中 fixed interval 是 reasonable 的近似
> 3. **Short trajectories**: oracle 和 no_boundary 差距较小 (77.5% vs 67.5%, 10pp)，长 trajectory 差距更大 (64.5% vs 47.4%, 17pp)
> 4. **Oracle 的 TSR 优势最大** (33.4% vs 22.2%) — boundary detection 对完成整个任务尤其重要

### 8.4 E3: Replanning Frequency Sweep

**设计**: 对 8+ 步 long trajectories，sweep fixed interval (K=1,2,3,5) + oracle boundary。分析 StepAcc vs planner call count 的 Pareto 关系。

**数据**: 317 long trajectories (289 long + 28 vlong), 3316 steps

| Condition | StepAcc | TypeAcc | TSR | Avg Planner Calls | Efficiency |
|-----------|---------|---------|-----|-------------------|-----------|
| **fixed_1 (every step)** | **85.2%** | **89.1%** | **26.5%** | 10.5 | 0.081 |
| oracle | 74.4% | 82.5% | 9.5% | 5.1 | 0.146 |
| fixed_2 | 73.9% | 81.8% | 6.0% | 5.4 | — |
| fixed_3 | 69.1% | 77.5% | 4.7% | 3.8 | 0.182 |
| fixed_5 | 63.2% | 73.1% | 2.8% | 2.4 | 0.259 |

**Pareto Frontier** (StepAcc vs planner calls, 效率递减):

```
fixed_5 (2.4 calls, 63.2%) → fixed_3 (3.8, 69.1%) → oracle (5.1, 74.4%) → fixed_1 (10.5, 85.2%)
```

**Per-Length StepAcc**:

| Condition | long(8-15) | vlong(16+) |
|-----------|-----------|-----------|
| fixed_1 | **85.4%** | **83.6%** |
| oracle | 75.7% | 67.1% |
| fixed_2 | 74.2% | 72.7% |
| fixed_3 | 69.6% | 65.9% |
| fixed_5 | 63.9% | 59.5% |

> **关键发现**:
> 1. **fixed_1 (每步 replan) 远超 oracle boundary** (85.2% vs 74.4%, +10.8pp) — 完全消除 context degradation 的效果极为显著
> 2. **Oracle 并非 Pareto optimal**: fixed_2 用 5.4 calls 达到 73.9%，oracle 用 5.1 calls 达到 74.4%，效率接近
> 3. **Efficiency 最高的是 fixed_5** (0.259 StepAcc/call)，但绝对 StepAcc 最低
> 4. **fixed_1 在 vlong 上的优势最大** (83.6% vs oracle 67.1%, +16.5pp) — 越长的 trajectory 越需要频繁 replan
> 5. **StepAcc vs Planner Calls 近似线性**: 每增加 1 planner call ≈ +2.7pp StepAcc

### 8.5 Cross-Experiment Consistency Check

| Metric | E1 (oracle+structured) | E2 (oracle+structured) | E3 (oracle, long only) |
|--------|----------------------|----------------------|----------------------|
| StepAcc | 76.3% | 76.2% | 74.4% |
| TypeAcc | 83.3% | 83.3% | 82.5% |

→ E1 和 E2 的 oracle+structured 条件高度一致 (76.3% ≈ 76.2%)，验证实验可靠性。E3 稍低因为只包含 8+ 步的长 trajectory (更难)。

### 8.6 Summary & Implications

**核心结论**:

1. **Adaptive replanning 显著有效**: 仅 midpoint 一次 replan 即 +12.2pp second-half StepAcc (E4)
2. **Structured protocol 最优**: structured > NL > type_only > none (E2)，与 7.2 节 oracle ablation 结论完全一致
3. **Oracle boundary 显著优于 fixed**: +6.5pp StepAcc (E1)，但 fixed interval 在长 trajectory 上是 reasonable 近似
4. **频率越高越好**: fixed_1 达 85.2% (E3)，但 cost 最高 (10.5 calls/traj)；Pareto optimal 的实用选择是 fixed_3 (69.1%, 3.8 calls)
5. **长 trajectory 受益最大**: vlong 上 adaptive vs baseline 差距始终 >12pp

**与 Section 7 分析的关联**:

| Section 7 发现 | E1-E4 验证 |
|---------------|-----------|
| 22pp decoding gap (Q3/Q6) | E3 fixed_1 消除大部分 gap (85.2%) |
| Structured > NL (7.2 oracle ablation) | E2 再次确认 (76.2% vs 73.2%) |
| Boundary 1.8x harder (Q4) | E1 oracle boundary 79.7% vs within-phase 71.5% |
| Action inertia at boundary (Q4) | E4 midpoint replan 消除 second-half degradation |
| Context degradation with length | E3 fixed_1 在 vlong 上优势最大 (+16.5pp vs oracle) |
