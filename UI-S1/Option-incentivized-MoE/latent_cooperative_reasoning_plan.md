# Latent Cooperative Reasoning: Emergent Specialization and Communication in VLM Agents

> **Project**: MA-GUI / MABelief
> **Date**: 2026-04-02
> **Status**: Plan
> **Predecessor**: `gradient_conflict_results.md` (v3, 2026-03-29)

---

## TL;DR

单模型 VLM 的 attention routing（q/k projection）无法同时服务 binding 和 action selection 两个 competency——这是 representational fragmentation 的机制根源。我们提出训练两个 LoRA-specialized agent（Visual Agent V + Action Agent A），共享 pretrained backbone，通过 **可微的 latent hidden states** 通信，end-to-end joint training 让 V 学会以 A 能利用的形式表达 binding 结果。核心 claim：**routing separation + latent communication + cooperative training = bridging fragmentation**。

---

## 目录

1. [Problem: Routing Conflict](#1-problem-routing-conflict)
2. [Idea: Latent Cooperative Agents](#2-idea-latent-cooperative-agents)
3. [Architecture](#3-architecture)
4. [Research Questions](#4-research-questions)
5. [Experimental Plan](#5-experimental-plan)
6. [Implementation Considerations](#6-implementation-considerations)
7. [Theoretical Contribution](#7-theoretical-contribution)
8. [Risk and Contingency](#8-risk-and-contingency)
9. [Timeline](#9-timeline)
10. [Evidence Chain Summary](#appendix-evidence-chain-summary)

---

## 1. Problem: Routing Conflict

VLMs performing GUI tasks must simultaneously resolve two coupled sub-problems:

| Sub-problem | What it does | Attention demand |
|---|---|---|
| **Binding** | 哪个 UI element 对应 task instruction？ | image tokens 被 task-relevant queries 找到 |
| **Action selection** | 对该 element 执行什么操作？ | last token 聚合 action-relevant context |

这两个 competency 需要不同的 attention routing。单一的 attention 参数（尤其是 k_proj）无法同时最优服务两种 routing 需求。

### 1.1 诊断证据

| 证据 | 来源 | 发现 |
|---|---|---|
| 信息存在但未被使用 | Exp F: concat +0.09 AUC | Binding info 编码在 value content 中 |
| Attention 找不到它 | Exp E: target rank 90-190 | Routing 没有指向正确 element |
| k_proj 是冲突载体 | Step 1b: L24 k_proj = -0.099 | Key projection 携带 binding vs action 的对立梯度 |
| v_proj 是合作的 | Step 1b: L24 v_proj = +0.011 | Value content 共享，没有竞争 |
| 改善 binding 不帮助 action | Step 2α: Probe C -0.05→+0.42 但 action ≈ unchanged | Binding subspace 改善不传导到 action pathway |

### 1.2 核心洞察

**瓶颈不是信息，而是信息路由。** 单模型的 attention mechanism 是一个容量有限的共享通道，无法在所有 28 层上同时服务竞争的 routing 需求。

这连接到 cooperative AI 的基本问题：**当信息分布在正交子空间中时，cooperation 成为必要**——不是因为 "一个模型装不下"，而是因为 single forward pass 只能遍历一条计算路径。

---

## 2. Idea: Latent Cooperative Agents

训练两个 specialized agent——**Visual Agent (V)** 和 **Action Agent (A)**——共享 pretrained VLM backbone，通过各自的 LoRA adapter 发展互补 specialization。V 专注 binding，A 专注 action selection。它们通过 **latent representations**（连续 hidden states，不是 text）通信，communication protocol 从 end-to-end gradient flow 中 emerge。

### 2.1 为什么是 Latent Communication？

| 通信类型 | 优势 | 劣势 |
|---|---|---|
| **Text** (V → text → A) | 干净分离；易于解释 | 离散化的信息瓶颈；两次独立 forward pass；A→V 无梯度流 |
| **Latent** (V → hidden states → A) | 丰富的连续信息；端到端可微；V 学会传达 A 需要的东西 | 难以解释；需要表示对齐 |

Text communication 把两个 agent 当作独立系统。Latent communication 把它们当作 **cooperative system**——V 不只是找到 target，它学会以最大化帮助 A 的形式表达发现。这只有通过 end-to-end gradient flow 才能实现。

### 2.2 与 Emergent Communication 文献的联系

Emergent communication 研究中，agents 从零开始在 joint task 上学会 communication protocol。我们的 setting 有两点独特：

1. **Shared pretrained backbone as common ground.** 两个 agent 共享 pretrained VLM 的 representation space，提供 "共同语言"。Protocol 是在此基础上 specialize，不是从零发明。好处：emergence 更容易。风险：两个 agent 太 similar，难以 diverge。

2. **Communication content 是 perceptual grounding.** 不是 abstract symbols，而是 visual binding 信息——"哪个像素区域对应 task instruction"。这种 communication 有 ground truth（GT bounding box），可以直接测量 quality 和 content。

---

## 3. Architecture

### 3.1 Overview

一个 base model（Qwen2.5-VL-7B-Instruct），两套 LoRA adapter（LoRA_V, LoRA_A）。

```
┌──────────────────────────────────────────────────┐
│                  Shared Input                     │
│           screenshot + instruction                │
└──────────┬───────────────────────────┬────────────┘
           │                           │
           ▼                           │
┌──────────────────────┐               │
│   Visual Agent (V)   │               │
│   Base + LoRA_V      │               │
│                      │               │
│   Layers 0 → L_comm  │               │
│   Objective: binding │               │
│                      │               │
│   Output: M (latent  │               │
│   binding tokens)    │               │
└──────────┬───────────┘               │
           │                           │
           │  M (differentiable)       │
           │                           │
           ▼                           ▼
┌──────────────────────────────────────────────────┐
│                Action Agent (A)                   │
│                Base + LoRA_A                      │
│                                                   │
│   Layers 0 → L_comm: process screenshot+instr    │
│   At L_comm: inject M as additional tokens        │
│   Layers L_comm → 27: attend to M + own context  │
│                                                   │
│   Objective: action prediction                    │
│   Output: action (function + params)              │
└──────────────────────────────────────────────────┘
```

**关键设计决策**:
- V 只 forward 到 L_comm（不需要完整 28 层），节省计算
- M 不 detach → gradient 从 A 流经 M 到 V，实现 cooperative training
- 两套 LoRA 共享同一 base model 参数

### 3.2 Forward Pass

```python
# Step 1: V forward (with LoRA_V) — partial, 到 L_comm 为止
v_hidden = base_model.forward(
    screenshot + instruction,
    lora=LoRA_V,
    output_hidden_states=True,
    up_to_layer=L_comm
)

# Step 2: 从 V 的 hidden states 中选择 binding message tokens
binding_scores = compute_binding_relevance(v_hidden, task_tokens)
M = select_top_k(v_hidden, binding_scores, K=8)  # K 个最相关的 tokens

# Step 3: A forward (with LoRA_A) — partial, 到 L_comm 为止
a_hidden = base_model.forward(
    screenshot + instruction,
    lora=LoRA_A,
    up_to_layer=L_comm
)

# Step 4: 在 L_comm 处注入 M 作为额外 tokens
a_hidden_augmented = concat(a_hidden, M)

# Step 5: A 从 L_comm 继续 forward
action_output = base_model.forward(
    a_hidden_augmented,
    lora=LoRA_A,
    from_layer=L_comm
)
```

### 3.3 Training Objective

```
Loss = L_act(action_output, GT_action) + λ · L_bind(v_hidden, GT_target)
```

**Gradient flow 分析**:

| Loss | 流向 | 作用 |
|---|---|---|
| L_act → LoRA_A | 直接 | A 学习 action selection |
| L_act → M → LoRA_V | 经 M 间接 | V 学习以 A 能利用的形式通信（cooperative signal） |
| L_bind → LoRA_V | 直接 | V 学习 binding（grounding supervision） |

V 收到两个训练信号——不只是做 binding，而是做 **communicable binding**。这是 joint training 的核心价值。

---

## 4. Research Questions

### Q1: Emergent Specialization

> 从同一个 pretrained model 出发，V 和 A 能否发展出 complementary specialization？

**测量方法**:
- LoRA_V 与 LoRA_A 的 weight cosine similarity：训练后是否 diverge？
- 各自在哪些 module (q/k/v/o/FFN) 上变化最大？
  - 预测：V 的 q/k 变化侧重 binding routing，A 的 q/k 侧重 action routing
- V 单独 grounding eval vs A 单独 action eval：各自在 specialty 上是否更强？

**Risk**: shared backbone 导致太 similar，无法 diverge → 退化为 single model。
**Mitigation**: 监控 LoRA weight divergence；若太低加 diversity regularization。

### Q2: Communication Effectiveness

> V 传给 A 的 latent message M 是否携带有用的 binding 信息？A 是否真的依赖 M？

**消融实验**:

| 消融 | 操作 | 预期（若 communication effective） |
|---|---|---|
| M → zero | 移除通信 | A accuracy 显著下降 |
| M → random | 噪声替代 | A accuracy 显著下降 |
| M → wrong-target | V 输出错误 element 的 hidden states | accuracy 下降且 error pattern 改变 |
| M → GT-target (oracle) | V 输出 GT target 的 hidden states | A accuracy 上升（upper bound） |

**深入分析**:
- 对 M 做 linear probe → decode target (x, y) coordinate
- A 对 M tokens 的 attention weight 分布 → binding 信息在 A 的哪些层被 consumed

### Q3: Joint vs Sequential Training

> End-to-end joint training 是否优于 sequential training？

| Training mode | V 收到的 signal | 预测 |
|---|---|---|
| **Joint** | L_bind + L_act via M | V 学到 A-useful representation |
| **Sequential** | L_bind only（freeze V 后训 A） | V 的表示可能不是 A 能利用的形式 |
| **A-only** | No V training | M from pretrained V，信息量低 |

- Joint >> Sequential → end-to-end gradient 对 communication emergence 是 **必要的**
- Joint ≈ Sequential → V 的 binding representation 天然对 A 有用，不需要 cooperative signal

### Q4: Parameter Sharing Granularity

> V 和 A 之间应该 share 多少参数？

| 条件 | Shared | Separate | 测试什么 |
|---|---|---|---|
| P1 | Base + v/o/FFN LoRA | q/k LoRA only | Routing-only separation（基于 k_proj conflict 理论） |
| P2 | Base model only | All LoRA modules | Full adapter separation |
| P3 | Nothing | Full model × 2 | Complete separation（upper bound） |

**理论预测**（基于 Step 1b k_proj conflict, v_proj cooperative）：P1 (routing-only separation) 应该 sufficient。若 P1 ≈ P2 → routing 是唯一 bottleneck。若 P2 >> P1 → content separation 也 matters。

### Q5: Communication Layer

> M 在哪一层 inject 最有效？

| L_comm | 含义 | 预测 |
|---|---|---|
| 9 (early) | V 只做 early visual processing | 信息量不够：binding 还没完成 |
| 19 (mid) | V 完成大部分 binding | 可能最佳：A 有足够 remaining layers 整合 |
| 24 (late) | V 几乎完成全部处理 | M 信息丰富，但 A 只剩 3 层整合 |

基于诊断数据（k_proj conflict 从 L21 开始 consistently negative），**L_comm ≈ 19 是 theoretically motivated 的 starting point**。

---

## 5. Experimental Plan

### Phase 1: Proof of Concept

**目标**: 验证 latent cooperative training 是否 work。最简配置，最快验证。

**配置**:

| Parameter | Value | Rationale |
|---|---|---|
| Base model | Qwen2.5-VL-7B-Instruct | 现有 checkpoint |
| LoRA rank | r=16 each | 与现有实验一致 |
| Sharing mode | P2 (base shared, all LoRA separate) | 最大自由度 |
| L_comm | 19 | 理论预测 |
| K (message tokens) | 8 | 起步值 |
| Token selection | Attention-weighted (task text → image tokens) | 最便宜 |
| λ (L_bind weight) | 0.1 | binding 作为辅助 |
| Data | GUI-360 train set | 已有 |
| Training | 1 epoch, cosine lr, batch 128 | 与 §7 一致 |

**Baselines**:

| ID | 描述 | 对照意义 |
|---|---|---|
| B0 | Single model, LoRA r=16, L_act only | = §7 Condition A，无 binding |
| B1 | Single model, LoRA r=16, L_act + L_bind | = Step 2α，same total capacity |
| B2 | Sequential: train V (L_bind) → freeze V → train A (L_act + M) | 无 cooperative signal |
| B_text | Text communication: V outputs text grounding → A receives as text | 离散通信 baseline |

**Success Criteria (决定是否进入 Phase 2)**:

| # | Criterion | Threshold |
|---|---|---|
| S1 | Joint > B0 | ≥ 3pp action accuracy |
| S2 | Joint > B1 | > 0（separate routing better than auxiliary loss） |
| S3 | M → zero ablation | accuracy drop ≥ 3pp（communication effective） |
| S4 | cos(LoRA_V, LoRA_A) for q/k < cos for v/o | routing diverges more than content |

**Decision gate**: S1 + S3 必须满足。S2、S4 是 supportive evidence。

---

### Phase 2: Mechanism Analysis

> **前提**: Phase 1 成功（S1 + S3 满足）

**Exp A: Specialization structure**
- LoRA_V vs LoRA_A 在 q/k/v/o/FFN 上的 Frobenius norm 和 effective rank
- 预测：V 的 q/k norm > A 的 q/k norm（V 更需要改变 routing）
- 在 joint-trained model 上重跑 gradient conflict analysis：routing conflict 是否被 resolve？

**Exp B: Communication content analysis**
- 对 M 训 linear probe → decode target (x,y), element type, action type
- M 的 information content 与 V 的 L_bind accuracy 的关系
- A 对 M tokens 的 attention weight 分布

**Exp C: Parameter sharing ablation**
- P1 (q/k only separate) vs P2 (all LoRA separate)
- 若 P1 ≈ P2 → routing separation 是 sufficient mechanism

**Exp D: Communication layer sweep**
- L_comm ∈ {9, 14, 19, 24}
- 与 Probe C 的 per-layer binding gap 做 correlation

---

### Phase 3: Comparison and Validation

**Exp E: Training paradigm comparison**
- Joint vs Sequential vs Text communication
- Joint > Sequential → cooperative signal 是 necessary
- Joint > Text → latent communication 优于 discretized communication

**Exp F: Capacity control**
- Joint (2 × LoRA r=16, 总 r=32) vs Single model (LoRA r=32)
- 相同总参数量。若 Joint > Single → benefit 不来自 capacity，来自 specialization

**Exp G: Cross-dataset generalization**
- 在 AndroidControl 上重复 Phase 1
- 验证 framework 不是 GUI-360 specific

---

## 6. Implementation Considerations

### 6.1 Memory Budget

| 组件 | 相对成本 | 说明 |
|---|---|---|
| V forward (L0→L19) | ~0.68× | 19/28 layers |
| A forward (L0→L27) | 1.0× | Full forward |
| V activation memory | ~0.68× | 为 backward 保留（不 detach M） |
| **Total** | **~1.7×** | vs single model forward |

Gradient checkpointing 可以管理 V 的 activation memory。

**估算**: 单 GH200 96GB，batch size 1-2 可行。多节点可用 DeepSpeed ZeRO-2。

### 6.2 LoRA Switching

HuggingFace PEFT 支持 multiple adapters：

```python
model.load_adapter(lora_v_path, adapter_name="visual")
model.load_adapter(lora_a_path, adapter_name="action")

# V forward
model.set_adapter("visual")
v_out = model.forward(input, up_to_layer=L_comm)

# A forward
model.set_adapter("action")
a_out = model.forward(input_with_M, from_layer=L_comm)
```

**需要验证**: PEFT 是否支持在同一 forward-backward cycle 内 switch adapter 且保持 gradient flow。不支持的话需要 custom implementation——手动管理 LoRA weight 的 add/remove。

**Fallback**: 不用 PEFT 的 adapter switching，而是维护两份 LoRA weight dict，在 forward 时手动 apply：

```python
def apply_lora(base_weight, lora_A, lora_B, scaling):
    return base_weight + scaling * (lora_B @ lora_A)
```

### 6.3 Partial Forward (up_to_layer / from_layer)

Qwen2.5-VL 的 `model.forward()` 可能不原生支持 partial forward。需要：

1. 修改 model 的 decoder loop，支持 `start_layer` 和 `end_layer` 参数
2. 或者手动遍历 `model.model.layers[start:end]`

这是 implementation 中最重要的工程任务之一。

### 6.4 Token Selection for M

| 方式 | 优点 | 缺点 | Phase |
|---|---|---|---|
| Attention weight (task text → image tokens) | 便宜；反映 cross-modal attention | 可能跟 binding 不完全对应 | **Phase 1** |
| L_bind gradient magnitude | 直接反映 binding contribution | 需要额外 backward | Phase 2 |
| Learned selection network | End-to-end optimal | 额外参数；可能不稳定 | Phase 3 |
| Fixed: GT bbox region tokens | Oracle selection (upper bound) | 需要 GT；非 learned | Diagnostic |

Phase 1 先用 attention-weighted selection（最便宜），搭配 GT-region 的 oracle upper bound 做对照。

---

## 7. Theoretical Contribution

### 7.1 Core Claim

**Representational fragmentation** ——不同 competency 的信息驻留在正交子空间中——是单模型多模态推理的根本限制。这种 fragmentation 集中在 **attention routing** (q/k projections)，而非 **value content** (v/o projections)。Multi-agent cooperation 通过给每个 competency 分配独立的 routing pathway 来 resolve this，同时 latent communication 保留了共享的 value content。

### 7.2 Novelty

| 方面 | 已有工作 | 我们的不同 |
|---|---|---|
| Multi-agent GUI agent | Planner + actor, text 通信, 独立训练 | **Latent 通信, joint 训练, shared backbone** |
| Emergent communication | Tabula rasa agents, discrete symbols, toy games | **Pretrained common ground, continuous latent, real VLM task** |
| Multi-task conflict | Gradient conflict analysis (PCGrad, CAGrad) | **Per-module decomposition → routing vs content 不对称性** |
| VLM probing | Layer-level representation analysis | **Projection-level (q/k/v/o) analysis linked to functional role** |

### 7.3 Generalizability

Representational fragmentation 不只是 GUI agent 的问题。任何需要跨 competency joint reasoning 的 VLM task 都可能出现：

- **Visual QA**: object recognition + question understanding + spatial reasoning
- **Embodied navigation**: scene understanding + instruction grounding + action planning
- **Document understanding**: layout parsing + text reading + semantic linking

诊断方法（DIBV: Diagnose → Isolate → Bridge → Validate）+ latent cooperative architecture 可以 generalize 到这些 settings。

---

## 8. Risk and Contingency

| Risk | 可能性 | 后果 | Contingency |
|---|---|---|---|
| Joint training 不 converge | Medium | V 和 A gradient 互相干扰 | 降低 λ；先 warm-up V 几步 (L_bind only) 再 joint；gradient clipping on M |
| Specialization 不 emerge | Medium | LoRA_V ≈ LoRA_A，退化为 single model | 加 diversity loss: penalize cos(LoRA_V, LoRA_A)；增大 LoRA rank 给更多 divergence 空间 |
| A 忽略 M | Medium | 学会不看 V 的 message，M 不携带信息 | Learned gate 让 M 更 salient；curriculum: 先用 GT-M 训 A 依赖 M，再换 V 的 predicted M |
| Latent 不如 text | Low-Med | 离散化反而帮助 communication | 这本身是有价值的 finding：与 emergent comm "discrete is better" 一致 |
| PEFT 不支持 same-cycle adapter switch with gradient flow | Medium | 需要 custom implementation | 手动 LoRA weight management（不依赖 PEFT 的 set_adapter） |
| Partial forward 改动过大 | Low | 需要 hack model internals | 手动遍历 `model.model.layers[start:end]` |

---

## 9. Timeline

| Week | Task | Deliverable | Decision Gate |
|---|---|---|---|
| **W1** | Implementation: partial forward + LoRA switching + M injection + training loop | 可运行的 training script | — |
| **W1-2** | Phase 1: proof of concept training + eval + ablation | Joint vs baselines 结果 + M ablation | S1+S3 满足? → Phase 2 |
| **W2-3** | Phase 2: mechanism analysis (specialization, communication content, parameter sharing) | LoRA divergence, probe on M, P1 vs P2 | Routing separation sufficient? |
| **W3-4** | Phase 3: comparison + capacity control + cross-dataset | Joint vs Sequential vs Text; capacity-matched control | Joint > Sequential? Latent > Text? |
| **W4-5** | Paper writing | — | — |

---

## Appendix: Evidence Chain Summary

从 "发现问题" 到 "设计方案" 的完整逻辑链：

```
Phase 0: Naive multi-agent (hard partition) 无效
    → 不能简单按 modality 拆分

Phase 1 Probing:
    Probe A: visual features OK (AUC 0.80-0.88)
    Probe C: binding gap 全层为负
    Exp F: info exists in concat (+0.09 AUC)
    → 信息存在但 action pathway 看不到

Step 1: Gradient conflict per-layer = cos ≈ 0
    → 不是 gradient interference，是 orthogonal subspaces

Step 1b: Per-module decomposition
    → k_proj conflict (-0.099), v_proj cooperative (+0.011)
    → Routing 冲突, content 协作
    → Bottleneck 不是信息，是信息路由

Step 2α: Auxiliary binding loss
    → Probe C -0.05→+0.42 but action ≈ unchanged
    → Binding subspace 改善不传导到 action pathway
    → Confirms: 改善一个子空间而不桥接 routing 无效

    ↓ Therefore:

Solution: Two agents with separate routing (q/k), shared content (v/o)
    → V specializes in binding routing
    → A specializes in action routing
    → Latent communication bridges them (differentiable, end-to-end)
    → Joint training ensures V communicates what A needs
```

### Evidence → Design Decision Mapping

| 诊断发现 | 直接推导的设计决策 |
|---|---|
| k_proj conflict, v_proj cooperative | → 分离 routing (LoRA), 共享 content (base model) |
| Binding gap 全层为负 | → 需要 explicit communication channel (M) |
| Auxiliary loss 不传导 | → Communication 必须 differentiable (latent, not text) |
| cos ≈ 0 (not negative) | → 不是 gradient interference → 不需要 gradient surgery → 需要 structural separation |
| k_proj conflict 从 L21 consistently negative | → L_comm ≈ 19 (在 conflict 开始前 inject) |
| Concat probe +0.09 AUC | → V 的 value content 有信息 → M 应该携带 hidden states (not just attention weights) |
