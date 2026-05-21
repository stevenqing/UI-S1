# Cooperative LoRA for GUI Agents: From V12 to V14

## Beamer Presentation Source Material

请用以下内容生成一个学术风格的 Beamer 演示文稿。每个 section 对应一组 slides。

---

## Section 1: Problem & Motivation

### GUI Agent 任务

- 任务：给定截图 + 指令，预测下一步操作（click/type/swipe/terminate）
- 评估：GUI-360 桌面基准（Word/Excel/PowerPoint），968 条测试轨迹
- 指标：TSR（Trajectory Success Rate）= 所有步骤都正确的轨迹比例
- 难点：自回归评估（stop-on-error），错误会跨步骤累积

### 现有方法的局限

- 全参数 SFT（gui_action, 7.6B params）: TSR = 17.1%
- 标准 LoRA + GRPO（132M params）: TSR = 17.3%
- 全参数 RL（7.6B params）: TSR = 10.5-12.3%（更大模型 ≠ 更好 RL）

---

## Section 2: Method — Cooperative LoRA Architecture

### V12: Soft Cooperative LoRA

- 两个 A 矩阵（A1, A2）= 两个"专家槽位"
- Per-token sigmoid routing: `r = sigmoid(x @ w_route)`
- r-space 混合: `h_blend = r * h1 + (1-r) * h2`
- 共享 B 矩阵投影回全维度
- 参数量 ~132M（与标准 LoRA 相当）

### V13: Iterative Cooperative LoRA（核心贡献）

在 V12 基础上，混合前加入 **T=2 轮门控通信**：

```
for t in range(T):
    g_12 = sigmoid(h1 @ gate_12[t])    # 输入相关门控
    h1 = h1 + g_12 * (h2 @ W_12[t])    # Expert 1 接收 Expert 2
    g_21 = sigmoid(h2 @ gate_21[t])
    h2 = h2 + g_21 * (h1 @ W_21[t])    # Expert 2 接收更新后的 Expert 1
```

- 通信在 r-space（128x128），不在 d-space（3584x3584），开销可忽略
- 额外参数：~1.85M（仅占总参数 1.4%）
- 当 gates=0 时退化为 V12（graceful fallback）

### 训练方法：SP + GiGPO + SPWA

- **SP**（Sequential Progress）: `SP_k = first_error_step / total_steps`
- **GiGPO**: 跨 K 条轨迹归一化 SP 作为 advantage
- **SPWA**: 第一个错误后步骤权重指数衰减（decay=0.5）
- PPO clipped loss + KL penalty + routing balance loss

---

## Section 3: Results — 2x2 Ablation + Cooperative

### 主结果表（968-episode test, best epoch）

| Architecture | Params | Standard GRPO | Our SP+SPWA |
|-------------|--------|---------------|-------------|
| Standard LoRA | 132M | 17.3% | 16.0% |
| Full-Parameter | 7.6B | 10.5% | 12.3% |
| V12 Cooperative | 132M | 16.5% | 15.6% |
| **V13 Iterative Coop** | **132M** | 16.0% | **18.7%** |

- V13+SP 18.7% > 所有其他组合，包括全参数 SFT baseline（17.1%）
- V13 > V12: +3.1% TSR，确认迭代通信的价值
- 全参数模型反而更差：更大 ≠ 更好（RL 场景下）

### 训练曲线（V13+SP）

| Epoch | TSR | Progress | Step SR |
|-------|-----|----------|---------|
| 0 | 15.1% | 25.7% | 55.9% |
| 1 | 16.5% | 30.3% | 61.4% |
| 2 | 18.2% | 30.8% | 62.1% |
| 3 | 18.7% | 31.9% | 63.3% |
| 5-resumed | 18.9% | 33.0% | 64.2% |

### 按轨迹长度分析

| Length | V13+SP | V12+SP | StdLoRA+GRPO |
|--------|--------|--------|--------------|
| Short (1-3 steps) | 37.0% | 34.5% | 35.3% |
| Medium (4-7 steps) | **12.0%** | 7.4% | 9.0% |
| Long (8+ steps) | 2.4% | 2.4% | 3.6% |

V13 的核心优势在中等长度轨迹：+4.6% absolute (+62% relative) over V12。多步推理需要专家协作。

### Baselines 对比

| Model | TSR | Type |
|-------|-----|------|
| Qwen2.5-VL-7B (base) | 1.55% | No fine-tuning |
| OS-Atlas-Pro-7B | 2.07% | Qwen2-VL |
| UI-TARS-7B-DPO | 1.98% | DPO |
| gui_action (SFT) | 17.1% | Full-param SFT |
| **V13 Coop+SP (ours)** | **18.7%** | LoRA RL |

---

## Section 4: Gate Analysis — 通信门控到底学到了什么？

### 四级递进实验

| 实验 | 粒度 | 发现 |
|------|------|------|
| V1: 全局平均 | 所有层/token | Gate ≈ 0.51，无差异 |
| V2: 逐层 | 每层每模块 | 路由 r 差异大，但 gate 仍无 action-type 差异 |
| V3: Token 级 | Image vs Text | **Gates 编码模态**: L10 偏图像 +0.044, L18 偏文本 -0.067 |
| V4: 推理阶段 | Planning/Action/Coord | **Gates 编码阶段**: L10↓ L18↑ "X-crossing" 模式 |

### X-Crossing Pattern（核心发现）

```
Gate value
0.62 |  L10 ╲
     |        ╲
0.55 |          ╲───────
     |     L18 ╱
0.50 |  ──╱
     |╱
0.42 |
     ├────────────────────
     Planning  Action  Coordinate
```

- L10: Planning 阶段最活跃（0.594），Coordinate 最低（0.542）→ 视觉语义理解前置
- L18: Action_type 最低（0.433），Coordinate 最高（0.474）→ 空间推理后置
- Within-generation std: L10=0.066, L18=0.069 → 门控逐 token 动态适应

### Gate 预测成功率（Gate Signature Analysis, 968 episodes）

| 分割方式 | 正确率 |
|----------|--------|
| L10 planning gate > median | **65%** |
| L10 planning gate < median | 51% |
| L18 planning gate < median | **66%** |
| L18 planning gate > median | 51% |

Planning 阶段门控差异可预测 14-15% 的准确率差距（p < 1e-10）。

---

## Section 5: Diagnostic Experiments — 瓶颈在哪？

### 5.1 Phase-Conditional Ablation

| 模式 | Click% | Coord <50px |
|------|--------|------------|
| full (正常) | 99.5% | 49.3% |
| no_comm (关闭通信) | 7.7% | 31.4% |
| planning_only | 26.7% | 38.7% |
| coord_only | 7.7% | 31.3% |

- 关闭通信 → 模型崩溃（99.5% → 7.7% click）→ 通信是结构性必需的
- Planning 阶段通信单独恢复最多 → Planning 是核心驱动力
- Coord/type 阶段通信 ≈ 无通信 → 这些阶段依赖 planning 建立的表示

### 5.2 Forced-Prefix + Logit Gap

模型**"不愿"而非"不能"**预测 type/swipe：

| GT Type | P(click) | Gap(click-type) |
|---------|----------|-----------------|
| click | 97.5% | 21.9 |
| type | 93.5% | 18.9 |
| swipe | 82.8% | 18.3 |

- 强制 type 前缀后，54.3% 能正确输出文本内容（mean similarity=0.59）
- GT=type 时 gap 比 GT=click 小 3.0 → 模型隐约知道应该选 type，但 18.9 的 gap 太大

### 5.3 Gate Perturbation

Gate 扰动 ±0.5 对 action type **零影响**（99.8-100% 仍预测 click）。坐标会移动（85% 变化），但随机方向。**"100% click" bias 在 A/B 权重中，不在通信门控中。**

### 5.4 Base Model 分析

Base Qwen2.5-VL 98.6% 输出无 `<action>` 标签 → RL 从零教会格式，同时创造了 click bias。不是"多样性坍缩"而是"从未有过多样性"。

### 5.5 错误类型分析

| 错误类型 | 占比 |
|----------|------|
| Format error | 0% |
| Type mismatch | 26.6% |
| Coordinate error | 73.4% |

73% 的错误是坐标点击位置错误，其中 84% 是完全错误（content_reward=0）。Type/swipe 100% 被误预测为 click。

---

## Section 6: V14 CoPDA — 基于 Gate 分析的 Credit Assignment

### 动机

V13 的问题：每个 step 所有 token 共享同一个 scalar advantage。

当 "type 对了但坐标错了" 时：
- Planning tokens（决定做什么）本应被奖励
- Coordinate tokens（决定在哪做）本应被惩罚
- 但两者得到相同的混合信号 → Credit assignment 瓶颈

### Phase Signal: Cross-Layer Gate Variance

利用 X-crossing 模式的本质：**planning 时各层 gate 差异大，coordinate 时各层 gate 趋同**。

```
gate_std(i) = std across layers [g_L0(i), g_L1(i), ..., g_L27(i)]
phi(i) = sigmoid(z-normalize(gate_std(i)))

phi -> 1: Planning token (高跨层方差)
phi -> 0: Coordinate token (低跨层方差)
```

量化示例：
- Planning: L10=0.622, L18=0.416, L27=0.507 → std ≈ 0.10 → phi ≈ 0.8
- Coordinate: L10=0.516, L18=0.504, L27=0.511 → std ≈ 0.006 → phi ≈ 0.2

### Per-Token Advantage

```
R_what = format_reward + type_reward    # 决定做什么
R_where = content_reward               # 做得准不准

A(k,s,i) = GiGPO(SP_k) * SPWA(k,s) * [phi(i)*A_what(k,s) + (1-phi(i))*A_where(k,s)]
```

**性质**：
- phi=0.5 时退化为标准 advantage（graceful fallback）
- 无新超参：z-normalize 自适应
- Planning tokens 对 type 负责，coordinate tokens 对坐标负责

### 实现

- Gate 录制：LoRA forward +9 行，backward compatible
- Phase score API：wrapper +25 行
- CoPDA 核心：copda.py 63 行
- Training loop 改动：~85 行
- **总新增代码：~180 行**

---

## Section 7: Summary

### 主要贡献

1. **Iterative Cooperative LoRA (V13)**：在 r-space 中加入门控通信，仅 +1.85M 参数，TSR 15.6% → 18.7% (+3.1%)

2. **Gate Analysis Campaign**：9 个实验系统性揭示通信门控的工作机制
   - 门控编码模态（image vs text）和推理阶段（planning vs coordinate）
   - L10-L18 X-crossing 模式：视觉理解前置，空间推理后置
   - Planning 阶段门控可预测成功率（14-15% gap）
   - 通信是结构性必需的，关闭后模型崩溃

3. **CoPDA (V14)**：利用 gate 的跨层方差作为天然 credit assignment 信号
   - Per-token phase-aware advantage 分解
   - Planning tokens 对 action type 负责，coordinate tokens 对空间精度负责
   - 零新超参，graceful fallback

### 关键数字

| 指标 | 数值 |
|------|------|
| V13 最佳 TSR | 18.7% (ep3) / 18.9% (ep5-resumed) |
| V13 vs V12 增益 | +3.1% absolute |
| V13 vs SFT baseline | +1.6% (18.7% vs 17.1%) |
| 通信额外参数 | 1.85M (1.4% of total) |
| Planning gate 预测力 | 14-15% accuracy gap |
| 关闭通信后崩溃 | click 99.5% → 7.7% |
| 模型能做 type 但不选 | 54.3% forced-prefix accuracy |
| Click 偏差根源 | RL 从零学习格式同时创造 |
