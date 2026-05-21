# Cooperative LoRA 架构根本性重设计分析

## 1. 现有架构的核心问题

### 当前分工：按输入modality分
- **LoRA_V**: image tokens → 仅影响 image hidden states 的质量
- **LoRA_A**: text tokens (think + action) → 承担几乎所有的 text generation

### 问题诊断

**问题1：LoRA_V 的贡献路径过于间接**
- Image tokens 只在 prefill 阶段被 encode 一次
- LoRA_V 改善 image hidden states → 通过 attention 传递给后续 tokens
- 但这个传递经过 28 层 attention，信号逐层衰减
- LoRA_V 没有直接的 CE loss，只能通过 attention gradient 间接获得梯度

**问题2：两个 adapter 做的是 orthogonal 的事**
- LoRA_V: 学习 "如何更好地 encode image pixels"（改变 linear projection 对 image token 的映射）
- LoRA_A: 学习 "如何更好地生成 text"（改变 linear projection 对 text token 的映射）
- 这两个功能空间的 weight change 是正交的，不能混用（α-mixing 失败的根因）

**问题3：在 AC 上，visual reasoning 不是瓶颈**
- AC 每步只有 1 张 screenshot，识别 UI 元素是 base model 已经能做的
- 瓶颈在于：理解 goal → 选择正确 action → 输出正确 coordinate
- 92% 的 assistant tokens 是 think（描述图片），8% 是 action

### 实验证据

| Config | Best TSR |
|--------|---------|
| old v6.5 (image→V, text→A) | 10.11% (ep4) |
| + coord_routing + α=0.3 | 9.40% (ep2) |
| old v6.5 CA ablation: t_only | 10.33% |
| old v6.5 CA ablation: v_only | 7.26% |
| old v6.5 CA = hard - max(v_only, t_only) | -0.22% |

**结论**：LoRA_V 不但没帮忙，反而是负担。t_only（不用 V）比 hard routing 还好。

## 2. 根本性重设计方向

### 方向 A: Functional Split（按功能分工）

**Think-Action Split**:
- LoRA_R (Reasoning): 处理 think tokens — 负责视觉推理和决策
- LoRA_E (Execution): 处理 action tokens — 负责动作生成和坐标输出
- Image tokens: 不加 adapter（base model 已足够好）

**优势**:
- 两个 adapter 都直接参与 CE loss（都生成 text）
- 分工有实际意义：推理能力 vs 执行能力
- Think tokens 负责理解"做什么"，Action tokens 负责"怎么做"

**风险**:
- Think 和 Action 之间的界限比 image/text 模糊
- 可能退化为 "两个 LoRA 做同一件事"

### 方向 B: Depth Split（按层深度分工）

**Shallow-Deep Split**:
- LoRA_S (Shallow, layers 0-13): 所有 tokens — 负责 feature extraction
- LoRA_D (Deep, layers 14-27): 所有 tokens — 负责 task-specific generation

**优势**:
- 不需要 token routing，两个 adapter 自然分层
- Shallow layers 学通用 feature，Deep layers 学 task-specific
- 类似 prefix-tuning 的思路

**风险**:
- 这等价于两个不同深度的 standard LoRA
- 可能没有 cooperative 的效果

### 方向 C: Projection Split（按 attention projection 分工）

**QK-VO Split**:
- LoRA_QK: 只加在 q_proj, k_proj — 负责 "看哪里"
- LoRA_VO: 只加在 v_proj, o_proj — 负责 "输出什么"

**优势**:
- 功能分工明确：QK 决定 attention pattern，VO 决定 output content
- 不需要 token-level routing
- 两个 adapter 自然 cooperate（QK 选择 attention targets，VO 基于 attended info 生成 output）

**风险**:
- 这和标准 LoRA 选择不同 target_modules 没什么区别
- 可能没有 cooperative 的优势

### 方向 D: Shared Trunk + Specialized Heads

**一个共享 lora_A (down-projection) + 两个专用 lora_B (up-projection)**:
- Shared lora_A: r → low-rank feature extraction (shared across all tokens)
- lora_B_vision: low-rank → output (only for image tokens)
- lora_B_action: low-rank → output (only for text tokens)

**优势**:
- Shared trunk 保证信息共享
- Specialized heads 允许不同 token type 有不同的 output mapping
- 参数更少（只需要一个 lora_A）
- lora_B 的专业化更自然（因为 output mapping 确实是 task-specific 的）

**风险**:
- 可能退化为 standard LoRA + 一个 adapter switch

### 方向 E: 放弃 Vision-Action Split，改为 MoE-style

**Shared Input + Router + Dual Expert**:
- 一个 learned router（per-token）决定 soft routing weight
- Expert_1, Expert_2: 两个等价的 LoRA
- Router based on hidden state, not token type

**优势**:
- 让模型自己学习什么时候用哪个 expert
- Token type 只是一个 prior，真正有用的分工由数据驱动
- 更灵活

**风险**:
- Router 可能 collapse（两个 expert 学成一样的）
- 训练更复杂

## 3. 推荐方向

考虑到 AC 的实验证据（t_only > hard，CA 为负），最根本的问题是：
**按 modality 分工在 AC 上是错误的先验**。

推荐优先级：
1. **方向 D (Shared Trunk + Specialized Heads)** — 最小改动，保留 cooperative 结构
2. **方向 A (Think-Action Split)** — 如果 AC 的 think/action 分工有意义
3. **方向 E (Learned Router)** — 最灵活但最复杂

但更重要的问题是：**cooperative LoRA 在什么任务上真正有用？**
- AC: single image, simple visual reasoning → cooperative 不needed
- GUI-Odyssey: multi-step, multi-image → 可能 cooperative 更有意义
- 需要的是一个 visual reasoning 是真正瓶颈的任务
