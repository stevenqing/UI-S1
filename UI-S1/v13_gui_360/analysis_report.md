# V13 Detailed Analysis & Optimization Directions

**Date**: 2026-04-26

---

## 1. 各方法 Best-Epoch 按 App 拆解

| Method | Word | Excel | PPT | Total |
|--------|------|-------|-----|-------|
| **V13+SP ep3** | 16.8% (62/368) | **23.5% (62/264)** | 17.0% (57/336) | **18.7%** |
| V13+SP ep3-resumed | 16.8% (62/368) | 23.1% (61/264) | 15.5% (52/336) | 18.1% |
| V12+SP ep3 | 14.4% (53/368) | 19.7% (52/264) | 14.9% (50/336) | 16.0% |
| StdLoRA+GRPO ep3 | 16.0% (59/368) | 20.8% (55/264) | 15.8% (53/336) | 17.3% |
| StdLoRA+SP ep3 | 13.6% (50/368) | 20.8% (55/264) | 14.9% (50/336) | 16.0% |
| V12+GRPO ep2 | 13.0% (48/368) | 19.7% (52/264) | 14.3% (48/336) | 15.3% |
| FullParam+SP ep3 | 8.7% (32/368) | 17.8% (47/264) | 11.9% (40/336) | 12.3% |
| FullParam+GRPO ep3 | 7.1% (26/368) | 17.0% (45/264) | 9.2% (31/336) | 10.5% |

**发现**:
- V13 在 **Excel** 上优势最大: +3.8% over V12, +2.7% over StdLoRA+GRPO
- Word 是所有方法的瓶颈（step-1 准确率最低）
- Excel 普遍表现好于 Word/PPT — 可能因为操作更结构化

---

## 2. 各方法按轨迹长度拆解

| Method | Short(1-3步) | Medium(4-7步) | Long(8+步) | Total |
|--------|-------------|--------------|-----------|-------|
| **V13+SP ep3** | 37.0% (131/354) | **12.0% (44/366)** | 2.4% (6/248) | 18.7% |
| V12+SP ep3 | 34.5% (122/354) | 7.4% (27/366) | 2.4% (6/248) | 16.0% |
| StdLoRA+GRPO ep3 | 35.3% (125/354) | 9.0% (33/366) | **3.6% (9/248)** | 17.3% |
| StdLoRA+SP ep3 | 34.2% (121/354) | 8.5% (31/366) | 1.2% (3/248) | 16.0% |
| FullParam+SP ep3 | 27.4% (97/354) | 5.2% (19/366) | 1.2% (3/248) | 12.3% |
| FullParam+GRPO ep3 | 25.4% (90/354) | 3.0% (11/366) | 0.4% (1/248) | 10.5% |

**关键发现**:
- **V13 的核心优势在 Medium (4-7步) 轨迹**: 12.0% vs V12的7.4% (+4.6%绝对值, +62%相对)
- Short 轨迹各方法差距不大（都在 25-37% 之间），V13 只领先 2-3%
- Long 轨迹所有方法都 <4% — 基本不可能做对，这是当前的天花板
- **Medium 轨迹是 V13 Cooperative 沟通机制发挥最大作用的地方** — 多步推理需要 expert 协作

---

## 3. V13 训练过程：每个 Epoch 做对了什么？

| 过渡 | 新增做对 | 新增做错 | 净增 |
|------|---------|---------|------|
| ep0 → ep1 | +50 | -36 | +14 |
| ep1 → ep2 | +31 | -15 | +16 |
| ep2 → ep3 | +18 | -13 | +5 |

### 新增做对的轨迹特征

| 过渡 | Word | Excel | PPT | Short | Med | Long | 典型步数 |
|------|------|-------|-----|-------|-----|------|---------|
| ep0→1 | 22 | 9 | 19 | 35 | 14 | 1 | 主要是 2-3 步 |
| ep1→2 | 6 | **14** | 11 | 15 | **14** | 2 | 开始攻克 4-6 步 |
| ep2→3 | 8 | 4 | 6 | 7 | **11** | 0 | 主要是 4-7 步 |

**训练趋势**:
- 早期 (ep0→1): 主要是简单短轨迹受益（35/50 是 short）
- 中期 (ep1→2): Excel 大幅提升，medium 轨迹开始做对
- 后期 (ep2→3): 增速放缓，主要在 medium 上持续微调，但 short 开始出现"遗忘"（lost 7个）
- **训练进入瓶颈**: net gain 从 +14 → +16 → +5，gained/lost 比例恶化

---

## 4. V13 vs V12 vs StdLoRA+GRPO 头对头

| 对比 | 共同做对 | 仅前者 | 仅后者 |
|------|---------|--------|--------|
| V13+SP vs V12+SP | 131 | **50** | 24 |
| V13+SP vs StdLoRA+GRPO | 126 | **55** | 41 |
| V12+SP vs StdLoRA+GRPO | 129 | 26 | 38 |

- V13 相比 V12 额外做对 50 个，V12 有 24 个 V13 做不对 → **V13 不是 V12 的严格超集**
- V13 相比 StdLoRA+GRPO 额外做对 55 个，StdLoRA+GRPO 有 41 个 V13 做不对 → 互补性较强
- 所有方法加起来能做对 **272/968 (28.1%)** 个轨迹

### V13 独有优势（相比 V12 额外做对的 50 个）
- 平均步数 = 4.9（V12 独有的平均步数 = 3.7）→ **V13 在较长轨迹上优势更明显**
- Excel 贡献 20 个（最多），Word 14 个，PPT 16 个
- 典型例子：「Delete content in cell G7」「Display the ruler in Word」「Open Slide Master view」

---

## 5. 错误类型分析

### 5.1 首次错误分类

| 错误类型 | V13+SP ep3 | V12+SP ep3 | StdLoRA+GRPO ep3 |
|---------|-----------|-----------|-----------------|
| Format error (格式错误) | 0 (0%) | 9 (1.1%) | 25 (3.1%) |
| Type mismatch (动作类型错误) | 209 (26.6%) | 196 (24.1%) | 185 (23.1%) |
| Content wrong (坐标/内容错误) | **578 (73.4%)** | **608 (74.8%)** | **591 (73.8%)** |

**核心问题**: ~73% 的错误是 **坐标点击位置错误**（动作类型对了但位置不对）。

### 5.2 坐标错误的严重程度

Click 错误的 content_reward 分布（V13 ep3, n=578）:
- **content_reward = 0.0**: 488 (84.4%) — 完全点错位置
- content_reward 0~0.3: 90 (15.6%) — 点得接近但不够准确
- content_reward > 0.3: 0 (0%)

→ **84% 的点击错误是"完全点错"**，不是"差一点"。说明模型没有理解应该点哪个 UI 元素，而非精度问题。

### 5.3 Type/Swipe 动作

| GT 动作类型 | V13 成功率 | V13 错误时预测 |
|------------|-----------|--------------|
| click | 67.7% (1213/1791) | — |
| type (输入文字) | 45.3% (145/320) | **100% 预测为 click** |
| swipe (滑动) | **0.0%** (0/34) | **100% 预测为 click** |

→ V13 **完全不会 swipe**，type 动作也有一半情况错误预测为 click。训练数据中 click 动作占绝大多数，导致模型过度偏向 click。

### 5.4 首次错误位置

| 位置（%轨迹） | V13 | V12 | StdLoRA+GRPO |
|-------------|-----|-----|-------------|
| 0-25% (前 1/4) | 333 (42.3%) | 357 (43.9%) | 365 (45.6%) |
| 25-50% | 241 (30.6%) | 253 (31.1%) | 253 (31.6%) |
| 50-75% | 74 (9.4%) | 60 (7.4%) | 62 (7.7%) |
| 75-100% (快做完了) | 139 (17.7%) | 143 (17.6%) | 121 (15.1%) |

→ V13 在 50-75% 和 75-100% 区间的错误更多 → 说明 V13 能走得更远才出错，但到了后半段也容易出错。

---

## 6. 训练效果

### Step-1 准确率（V13 ep0 → ep3）

| App | ep0 | ep3 | 提升 |
|-----|-----|-----|------|
| Word | 37.0% | 51.6% | **+14.6%** |
| Excel | 47.3% | 54.9% | +7.6% |
| PPT | 58.9% | 68.2% | +9.3% |

Word 提升最大但仍然最低。

### Medium (4-7步) 轨迹的深度到达率

| 方法 | TSR | Avg Progress | 过Step2 | 过Step3 |
|------|-----|-------------|---------|---------|
| V13+SP ep3 | 12.0% | 31.8% | 43.2% | 28.4% |
| V12+SP ep3 | 7.4% | 26.9% | 36.9% | 23.2% |
| StdLoRA+GRPO ep3 | 9.0% | 27.2% | 37.2% | 23.8% |

→ V13 在 medium 轨迹上各个深度都领先，过 Step3 的比例 28.4% vs V12 的 23.2%。

---

## 7. 326 个"零进度"轨迹（所有方法全失败）

| App | 数量 |
|-----|------|
| Word | **152** (46.6%) |
| Excel | 93 (28.5%) |
| PPT | 81 (24.8%) |

典型的硬任务:
- 「在 Excel 中输入公式」（需要 type 动作，模型全部预测为 click）
- 「在 Word 中插入特殊符号」（需要多步 UI 导航，step 1 就错）
- 「创建模板」（open-ended 文字输入任务）

---

## 8. Gate 分析：V13 Communication Mechanism 到底学到了什么？

### 实验设计

三轮递进实验：
1. **V1 (全局平均)**: 对所有层、所有 token 的 gate 值取均值 → 结果 ~0.51，看似 dead
2. **V2 (per-layer)**: 区分高 gate-norm 层 vs 低 gate-norm 层 → 仍然按 action type 无差异
3. **V3 (token-level)**: 区分 image tokens vs text tokens → **发现显著差异**

### 核心发现

**Gates 编码的是 modality (image vs text)，不是 action type**：

| Layer | Image tokens gate | Text tokens gate | Diff |
|-------|------------------|-----------------|------|
| L10 (early) | 0.5945 ± 0.009 | 0.5510 ± 0.003 | **+0.044** |
| L18 (middle) | 0.4273 ± 0.006 | 0.4939 ± 0.004 | **-0.067** |
| L27 (late) | 0.5033 ± 0.002 | 0.5135 ± 0.002 | -0.010 |

**关键观察**:
- 方向相反：L10 对 image 多通信，L18 对 text 多通信
- g_12 vs g_21 不对称：L10 expert 2→1 方向更强 (0.66 vs 0.53)
- Gate range 实际很大（0.22 ~ 0.80），只是平均后看起来接近 0.5
- Routing 也有 modality 差异：L10 image 用更多 Expert 2 (r=0.87 vs txt r=0.97)

### 解释

V13 相比 V12 的 +3.1% 提升来自 **modality-aware cross-expert communication**：
- **早期层** (L10): Image tokens 需要更多 expert 间信息交换 → 视觉-语义对齐
- **中期层** (L18): Text tokens 需要更多 expert 间信息交换 → 指令理解
- **晚期层** (L27): 差异极小，experts 已经 converge

这说明 V13 的架构优势是 **结构性的**（W_12/W_21 提供了跨 expert 的信息融合能力），
gates 起的是 **调节器** 作用（根据 token modality 调节通信量），而非开关作用。

### V4: Reasoning Path — Gate 在生成过程中的动态行为

**实验**: 968 episodes 全量，用 `model.generate()` 逐 token 收集 gate，按生成内容分 4 个 phase。

**Gates 在不同 reasoning phase 有显著差异**:

| Phase | L10 | L18 | L27 |
|-------|-----|-----|-----|
| **planning** (自然语言推理) | **0.594** | 0.445 | 0.507 |
| **action_start** (`<action>{`) | 0.550 | 0.467 | **0.467** |
| **action_type** (`"action":"click"`) | 0.577 | **0.433** | 0.496 |
| **coordinate** (`[x, y]`) | 0.542 | **0.474** | 0.511 |

**关键发现**:
- **L10**: planning 阶段通信最强 (0.594)，coordinate 阶段最弱 (0.542)，Δ=0.052
  → 视觉理解和决策规划需要最多的 cross-expert 融合
- **L18**: action_type 最弱 (0.433)，coordinate 最强 (0.474)，方向与 L10 相反
  → 空间定位需要不同的 expert 协作模式
- **L27**: action_start 时明显下降 (0.467 vs 0.507)
  → 从自然语言切换到格式化输出时，高层 communication 减少
- **Within-generation std**: L10=0.066, L18=0.069，gate 在生成过程中动态变化

**结论**: Gates emergent 出了 phase-dependent reasoning patterns，不同生成阶段使用不同的 expert 通信策略。这说明 V13 的 communication mechanism 不仅是 modality-aware，还是 **reasoning-stage-aware** 的。

### 对下一步的启示

1. Gate 不能做 action-type 级别的 exploration signal（Thought 3 被否定）
2. 但 modality-aware + reasoning-stage-aware communication 是有效的
3. V13 + GRPO: ep0=13.7%, ep1=16.5%, ep2=15.6%（已下降）— GRPO 对 V13 的提升有限

---

## 9. 优化方向建议

### 方向 1: 🔑 提升 Type/Swipe 动作能力 (预期 +3~5%)

**问题**: Swipe 0% 准确率，Type 45.3% 准确率，100% 的错误是"预测为 click"
**原因**: 训练数据中 click 占比过高，RL 探索不到 type/swipe 的正确策略

**方案**:
- A) **混合 SFT + RL**: 先用少量 type/swipe 标注数据做 SFT warm-up，再 RL fine-tune
- B) **奖励加权**: 在 SP+SPWA 中对 type/swipe step 给更高权重，或 type_reward 从 0.2 提高到 0.4
- C) **类型平衡采样**: 训��数据中 up-sample 含 type/swipe 的轨迹

**预期效果**: 34 个 swipe + ~175 个 type step，即使只修复一半也能多做对 ~30-50 个 step，间接提升 TSR 3-5%。

### 方向 2: 🔑 提升坐标定位准确率 (预期 +3~5%)

**问题**: 84% 的 click 错误是 content_reward=0 (完全点错位置)
**原因**: 模型不理解应该点哪个 UI 元素（不是精度问题）

**方案**:
- A) **图像分辨率提升**: 当前 image_max_pixels=602112 (~776×776)，提高到 ~1M pixels 可以看清更多 UI 细节
- B) **Grounding 预训练**: 用 GUI grounding 数据（如 ScreenSpot）做预训练，增强 UI 元素定位能力
- C) **负样本学习**: 在 RL 的 K=8 rollout 中，同一 step 有些 click 对有些错，可以强化 contrastive 信号
- D) **坐标回归 head**: 当前用 text token 生成坐标，考虑加一个轻量的坐标预测头

### 方向 3: 🔑 延长训练 + Learning Rate Schedule (预期 +2~3%)

**证据**: V13 ep5-resumed 在半程已经达到 ~21.4%，比 ep3 的 18.7% 提升显著
**问题**: ep2→ep3 的 gained/lost 比例恶化 (18/13)，有遗忘趋势

**方案**:
- A) **Cosine LR decay**: 当前恒定 lr=1e-5，后期应降低以减少 loss spike 和遗忘
- B) **训练到 8-10 epochs**: ep5 暂时还在涨，但需要 LR schedule 防止后期震荡
- C) **KL penalty 逐渐增加**: 防止 policy 偏离 base model 太远导致 format/type 退化

### 方向 4: Medium 轨迹专项优化 (预期 +2~3%)

**证据**: Medium (4-7步) 是 V13 优势区域但 TSR 只有 12%，有很大提升空间
**过 Step3 只��� 28.4%** → 后半段容易崩溃

**方案**:
- A) **Multi-step reward bonus**: 对连续正确 step 给指数递增的奖励，鼓励"一鼓作气"
- B) **Curriculum learning**: 先只训练 short，再加入 medium，最后 long
- C) **Expert communication 增加轮次**: 对 medium/long 轨迹的后半段 step，增加 communication rounds (T=3 或 4)

### 方向 5: 集成/投票策略 (预期 +3~5%)

**证据**: 所有方法并集能做对 272/968 (28.1%)，但单个最好方法只有 18.7%
**V13 和 StdLoRA+GRPO 互补性强**: 41 个轨迹 StdLoRA+GRPO 能做对但 V13 不行

**方案**:
- 推理时用多个 checkpoint 投票（majority vote across K rollouts from different models）
- 或训练一个 router 选择用哪个模型

### 优先级排序

| 优先级 | 方向 | 预期收益 | 实施难度 |
|--------|------|---------|---------|
| **P0** | 延长训练 + LR schedule | +2~3% | 低（改 hyperparams 即可） |
| **P0** | Type/Swipe 动作修复 | +3~5% | 中（需要 SFT 数据或 reward 调整） |
| **P1** | 坐标定位优化 | +3~5% | 中~高 |
| **P1** | Medium 轨迹专项 | +2~3% | 中 |
| **P2** | 集成投票 | +3~5% | 低（推理时间增加） |

**最快获得增益**: P0 方向 — 继续训练到 ep8 + cosine LR + type/swipe reward 加权，预期可到 **23-25% TSR**。
