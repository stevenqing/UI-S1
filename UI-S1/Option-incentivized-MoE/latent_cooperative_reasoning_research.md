# Latent Cooperative Reasoning: Beyond Routing — Towards True Multi-Agent Cooperation in VLMs

> **Project**: MA-GUI / Cooperative LoRA
> **Date**: 2026-04-07
> **Status**: Research Proposal
> **Predecessors**: `latent_cooperative_implementation.md` (v3-v5 experiments), `latent_cooperative_reasoning_plan.md` (original plan)

---

## TL;DR

v3-v5 的实验揭示了一个根本性问题：**cooperation 不能仅靠 routing/architecture 来实现**。Hard routing（v3/v4）导致 specialization 和 cooperation 不可兼得；soft routing（v5）的 SFT gradient 信号不足以学会有意义的 cooperation boundary。我们需要从 research idea 层面重新思考：(1) **什么是正确的 cooperative reasoning mechanism**，(2) **什么 training signal 能驱动 cooperation emergence**。

本文提出三个互补的 research direction，核心思想是：**将 text-based thought 替换为 latent cooperative thought + 用 counterfactual RL 训练 cooperation + iterative latent debate 实现 active cooperation**。这三者共同填补了当前 multi-agent latent cooperative reasoning 文献的关键空白。

> **🔥 当前优先方向：Per-Layer Cooperative Communication（Section 12）**
>
> 基于 v5 soft routing 的思路，我们提出在 LoRA 的 low-rank space 中实现 **每一层的双向 cooperative communication**。V 和 A agent 在每一层通过学习的线性投影交换信息，在单次 forward pass 中完成 joint reasoning。这是当前最可行的 SFT-based 方案，无需 thought augmentation、无需 RL、无需 multi-turn 格式，直接使用原始 GUI-360 数据训练。

---

## 目录

1. [Diagnosis: Why v3-v5 All Failed](#1-diagnosis-why-v3-v5-all-failed)
2. [Related Work Landscape](#2-related-work-landscape)
3. [Research Direction A: Latent Cooperative Thought](#3-research-direction-a-latent-cooperative-thought)
4. [Research Direction B: Counterfactual Cooperative RL](#4-research-direction-b-counterfactual-cooperative-rl)
5. [Research Direction C: Iterative Latent Debate](#5-research-direction-c-iterative-latent-debate)
6. [Unified Framework: A + B + C](#6-unified-framework-a--b--c)
7. [Positioning Against Literature](#7-positioning-against-literature)
8. [Experimental Plan](#8-experimental-plan)
9. [Paper Framing](#9-paper-framing)
10. [Critical Data Construction Analysis](#10-critical-data-construction-analysis)
11. [Data Construction for New Research Directions](#11-data-construction-for-new-research-directions)
12. **[🔥 Per-Layer Cooperative Communication (v6)](#12-per-layer-cooperative-communication-v6)** ← 当前优先方向

---

## 1. Diagnosis: Why v3-v5 All Failed

### 1.1 Empirical Evidence Summary

| Version | Architecture | Result | Root Cause |
|---------|-------------|--------|------------|
| **v3** (2-agent hard) | LoRA_V + LoRA_A, thought+action → LoRA_A | 46.1% (vs SVD 47.4%) | Text thought 污染 LoRA_A 的 function prior：click→select_text +79, click→wheel +71 |
| **v4** (3-agent hard) | LoRA_V + LoRA_T + LoRA_A, hard separation | **5.1% AP** (catastrophic) | 完全切断 T→A 信息流；batch thought state machine bug 加剧 |
| **v5** (soft routing) | LoRA_V + LoRA_A, learnable sep per layer | sep 几乎未移动（~0.5） | SFT loss 的梯度对 196 个 sep scalar 信号太弱，无法学会 cooperation/specialization boundary |

### 1.2 The Fundamental Pattern

```
v3:  Too much coupling    → thought contaminates action prior
v4:  Too much separation  → destroys cooperation entirely
v5:  Learnable boundary   → insufficient training signal to learn it
```

**核心诊断**：这三个版本都把 cooperation 当作 **routing problem**（哪个 token 由哪个 adapter 处理），但真正缺失的是：

1. **Cooperation Mechanism**: Passive attention（shared KV cache）不够 expressive，agents 无法主动向对方传递有用信息
2. **Cooperation Training Signal**: SFT loss 是 monolithic 的，不区分各 agent 的贡献，无法 incentivize cooperation
3. **Communication Protocol**: Text-based thought 与 action 共享词表空间，latent thought 从未被实现

### 1.3 Key Data Points Supporting the Diagnosis

**Complementarity is real**（cooperation potential exists）:
- Oracle ensemble: **55.5%** vs individual 47.4%/46.1%
- Coop-unique wins: 1,531 samples（8.0%）— mostly coordinate localization
- SVD-unique wins: 1,785 samples（9.4%）— mostly function selection

**Text thought is a double-edged sword**:
- Helps structured operations: set_font +17.7pp, select_table +20.7pp, select_paragraph +10.2pp
- Hurts common operations: click -1.6pp, type -2.5pp
- Net negative: thought helps 8.0% of samples, hurts 9.4%

**The contamination is in the vocabulary space**:
- 75% of wrong_function errors: thought doesn't mention any function name, but LoRA_A's prior is shifted
- Function confusion is systematic: click→{select_text, wheel, select_table_range, drag} are all **Coop-only** confusions
- 87.5% of both-fail wrong_function predict the SAME wrong function → base model limitation, not adapter issue

---

## 2. Related Work Landscape

### 2.1 Latent Communication Between Agents

| Paper | Venue | Key Idea | Agents | Communication | Limitation |
|-------|-------|----------|--------|---------------|------------|
| **COCONUT** | ICLR 2025 | Chain of Continuous Thought — LLM's last hidden state as latent "thought" | Single agent | Self-recurrent latent | No multi-agent cooperation |
| **LatentMAS** | arXiv 2511.20639 | Training-free latent collaboration via shared working memory | Separate models (Qwen3 4B/8B/14B) | Shared latent working memory | Inter-model (N× cost), sequential chain |
| **Interlat** | arXiv 2511.09149 | Agents communicate entirely in latent space via learned adapter | Separate models | Compressed latent states via adapter | Requires adapter training, pairwise |
| **Vision Wormhole** | arXiv 2602.15382 | Visual pathway as inter-agent communication channel | Heterogeneous VLMs | Universal visual codec | Requires VLM architecture |
| **System 1&2 Comm** | arXiv 2510.00494 | Dual-architecture latent reasoning (Base + Coprocessor) | Dual architecture | Latent message exchange | Joint finetuning ≈ unified soft-embedding baseline |

**Gap**: 所有工作都是 **inter-model** latent communication（多个独立模型之间通信）。没有工作研究 **intra-model** latent cooperation（单个模型内的 specialized adapters 之间通信）。

### 2.2 Multi-Agent Cooperative Training with RL

| Paper | Venue | Key Idea | Training | Cooperation Mechanism |
|-------|-------|----------|----------|----------------------|
| **MAPoRL** | ACL 2025 | Multi-agent post-co-training via RL with discussion quality reward | RL co-training | Multi-turn text discussion + verifier |
| **MAGRPO** | NeurIPS 2025 WS | Dec-POMDP formulation with centralized group-relative advantages | MARL (CTDE) | Centralized training, decentralized execution |
| **M-GRPO** | arXiv 2511.13288 | Hierarchical GRPO for main agent + sub-agent | Decoupled RL | Trajectory alignment across servers |
| **Evolving Orchestration** | NeurIPS 2025 | Learned orchestrator dynamically routes between agents | RL-trained orchestrator | Adaptive agent sequencing |

**Gap**: 所有工作都用 **separate models** 做 multi-agent RL。没有工作研究 **intra-model adapter-level** cooperative RL with credit assignment。

### 2.3 MoE and Cooperative Expert Reasoning

| Paper | Venue | Key Idea | Routing | Cooperation |
|-------|-------|----------|---------|-------------|
| **GraphMoE** | arXiv 2501.07890 | Expert pseudo-graph with recurrent routing | Per-token, multi-round | Inter-expert via virtual hub node |
| **ReMoE** | ICLR 2025 | Fully differentiable ReLU routing (variable # experts per token) | Per-token, continuous | Soft expert combination |
| **MoA** | ICLR 2025 Spotlight | Layered multi-LLM architecture | Per-layer, all agents | Previous layer outputs as auxiliary info |
| **Self-MoA** | arXiv 2502.00674 | Single top LLM self-ensemble outperforms multi-model MoA | Self-ensemble | Multiple samples from same model |

**Gap**: MoE 是 per-token routing（每个 token 选不同 expert），不是 per-skill routing。没有工作研究 **按功能子技能（perception/reasoning/action）路由**，且 agents 通过 learned latent protocol 主动通信。

### 2.4 Multi-Agent Debate

| Paper | Venue | Finding |
|-------|-------|---------|
| **MAD** | ICML 2024 | Multi-agent text debate improves factuality |
| **Should We Be Going MAD?** | ICML 2024 | MAD doesn't reliably outperform other prompting strategies |
| **Self-MoA** | arXiv 2502.00674 | Single-model ensemble > multi-model ensemble |

**Lesson**: Naive aggregation of multiple agents/responses does NOT reliably help. **Quality of cooperation and training for cooperation matter more than diversity.**

### 2.5 Positioning Summary

```
                        Inter-model              Intra-model
                    ┌─────────────────┬─────────────────────────┐
  Text-based        │ MAPoRL, MAD,    │ v3 (text thought)       │
  communication     │ MoA             │ ← contamination problem │
                    ├─────────────────┼─────────────────────────┤
  Latent-based      │ LatentMAS,      │ *** OUR PROPOSAL ***    │
  communication     │ Interlat,       │ Latent cooperative      │
                    │ Vision Wormhole │ thought + counterfactual │
                    │                 │ RL + latent debate       │
                    └─────────────────┴─────────────────────────┘

                    N× model cost         1× model + adapters
```

**Our unique position**: Intra-model + latent communication + cooperative training. 兼具 MAS 的 specialization 优势和 single model 的 efficiency 优势。

---

## 3. Research Direction A: Latent Cooperative Thought

### 3.1 Core Problem

v3 的 text thought（`<thought>I should use select_text...</thought>`）和 action tokens 共享词表空间。当模型生成 thought 中的 "select_text" 时，LoRA_A 的 function prior 被 shift。

但 thought 本身是有用的——coop 在 structured operations 上显著优于 SVD（set_font +17.7pp, select_table +20.7pp）。问题不是 thought 没用，而是 **text-based thought 的副作用**。

### 3.2 Idea: Replace Text Thought with Latent Thought

```
v3 (text thought):
  [image] [instruction] → <thought>text tokens</thought> → <action>JSON</action>
                           ↑ 共享词表空间
                           ↑ "select_text" 等词汇 shift function prior
                           ↑ LoRA_A 看到 thought tokens → contamination

Proposed (latent thought):
  [image] [instruction] → [latent₁][latent₂]...[latentₖ] → <action>JSON</action>
                           ↑ 连续隐状态，不经过词表
                           ↑ LoRA_T 处理这些 latent tokens
                           ↑ LoRA_A 通过 cross-attention 读取 latent tokens
                           ↑ 不可能出现 "select_text" → 不可能污染 function prior
```

### 3.3 Why This Solves the Contamination Problem

| Mechanism | Text Thought (v3) | Latent Thought (proposed) |
|-----------|-------------------|--------------------------|
| Thought representation | Discrete tokens from vocabulary | Continuous hidden states |
| Function names in thought | Yes（"select_text" explicitly appears） | **Impossible**（no vocabulary decoding） |
| Information density | Low（natural language is redundant） | **High**（COCONUT shows latent > text for reasoning） |
| LoRA_A exposure to thought | Full（shares vocabulary space） | **None**（latent space is separate from vocabulary） |
| Cooperation mechanism | Passive（A reads text thought tokens） | **Active**（A attends to latent thought via cross-attention） |

### 3.4 Why This Is Better Than v4

v4 的失败是因为 hard separation 完全切断了 T→A 的信息流。Latent thought approach 中：
- LoRA_T 产生 latent thought tokens（不是 text）
- LoRA_A **explicitly attend to** latent thought tokens via cross-attention
- Cooperation 是 by design（cross-attention）, 不依赖 passive shared KV cache
- 但 latent space 与 vocabulary space 分离 → 不可能有 function prior contamination

### 3.5 Architecture

```
[image tokens] [instruction tokens]
        │               │
        ▼               ▼
   ┌─────────────────────────────┐
   │  Base Model + LoRA_V/A      │
   │  Layers 0 → L_think         │  ← V/A 共同处理到第 L_think 层
   └──────────┬──────────────────┘
              │
              ▼
   ┌─────────────────────────────┐
   │  Latent Thought Generation  │
   │  LoRA_T processes K rounds  │  ← 产生 K 个 latent thought tokens
   │  of recurrent latent tokens │  ← 每一轮: h_{t+1} = f_T(h_t, context)
   │  (continuous, no vocab)     │  ← 类似 COCONUT 的 recurrent latent
   └──────────┬──────────────────┘
              │ [latent₁]...[latentₖ]
              ▼
   ┌─────────────────────────────┐
   │  Action Generation          │
   │  Base Model + LoRA_A        │
   │  Layers L_think → 27        │  ← LoRA_A attends to latent thoughts
   │  Cross-attn to latent       │  ← via injected KV from latent tokens
   │  Output: <action>JSON       │
   └─────────────────────────────┘
```

### 3.6 Training Protocol

**Phase 1: Curriculum Warm-up**（类似 COCONUT）

逐步将 text thought 替换为 latent thought：

```
Stage 1: Full text thought    → <thought>text₁ text₂ ... textₙ</thought> → action
Stage 2: Partial replacement  → <thought>text₁ [latent] ... [latent]</thought> → action
Stage 3: Full latent thought  → [latent₁][latent₂]...[latentₖ] → action
```

每个 stage 用上一个 stage 的 checkpoint 初始化。Text thought 的 training data 提供 warm-start signal。

**Phase 2: RL Fine-tuning**（见 Direction B）

用 action success 作为 reward 训练 LoRA_T 产生对 LoRA_A 有用的 latent thoughts。

### 3.7 Comparison with COCONUT

| Aspect | COCONUT | Ours |
|--------|---------|------|
| Agent structure | Single model, single adapter | **Multi-agent**: LoRA_T (thought) + LoRA_A (action) |
| Latent reasoning | Self-recurrent (h → h → h) | **Cooperative**: T produces, A consumes |
| Communication | Self-loop (within same model) | **Inter-agent**: T's latent → injected into A's attention |
| Training | Curriculum from text CoT | Curriculum + **cooperative RL** |
| Application | Logical reasoning (toy tasks) | **GUI agents** (grounded multimodal actions) |

核心区别：COCONUT 是 single-agent latent reasoning（模型自己跟自己通信）。我们是 **multi-agent latent cooperation**（specialized thought agent 和 action agent 通过 latent channel 通信）。

---

## 4. Research Direction B: Counterfactual Cooperative RL

### 4.1 Core Problem

SFT loss 无法训练 cooperation：
- SFT 不区分 "V 的 grounding 好但 A 的 function 选错了" 和 "V 的 grounding 差导致 A 选了错误的位置"
- 所有 agents 收到同样的 gradient signal → 没有 **credit assignment**
- v5 的 sep 参数几乎没移动 → SFT gradient 对 cooperation boundary 的信号太弱

### 4.2 Idea: Counterfactual-Based Per-Agent Credit Assignment

```python
For each training sample (image, instruction, GT_action):

  # Step 1: Full cooperation forward
  latent_thought = LoRA_T.generate_latent(image, instruction)
  action_full = LoRA_A.generate(image, instruction, latent_thought)
  reward_full = R(action_full, GT_action)

  # Step 2: Ablate thought agent (counterfactual)
  action_no_T = LoRA_A.generate(image, instruction, zeros_like(latent_thought))
  reward_no_T = R(action_no_T, GT_action)

  # Step 3: Per-agent credit assignment
  T_cooperation_value = reward_full - reward_no_T  # T 的边际贡献

  # Step 4: Differentiated training signals
  if T_cooperation_value > 0:
      # T's thought helped → reinforce T's current output
      loss_T = -T_cooperation_value * log_prob(latent_thought)
  else:
      # T's thought hurt → penalize T's current output
      loss_T = -T_cooperation_value * log_prob(latent_thought)  # negative reward → penalty

  # A always gets full reward signal
  loss_A = GRPO_loss(action_full, reward_full)
```

### 4.3 Why This Solves the Precision-Recall Tradeoff

v3 的核心问题是 **thought 在 structured operations 上有用（set_font +17.7pp），但在 common operations 上有害（click -1.6pp）**。

Counterfactual credit assignment 直接解决这个问题：

| Scenario | reward_full | reward_no_T | T_cooperation_value | Training Signal |
|----------|-------------|-------------|---------------------|-----------------|
| T correctly suggests select_text (rare) | 1.0 | 0.0 | **+1.0** | Strongly reinforce |
| T incorrectly suggests select_text (should be click) | 0.0 | 1.0 | **-1.0** | Strongly penalize |
| T's thought is neutral (click task, thought doesn't help) | 0.8 | 0.8 | **0.0** | No signal → T learns to stay neutral |

**结果**：T 自然学会 **selective cooperation** —— 只在有帮助的时候（structured operations）提供 informative thought，其他时候（common click/type）保持 neutral。

### 4.4 Comparison with MAPoRL (ACL 2025)

| Aspect | MAPoRL | Ours |
|--------|--------|------|
| Agent architecture | Separate LLMs | **Intra-model** LoRA adapters |
| Cooperation mechanism | Multi-turn text discussion | **Latent thought injection** |
| Credit assignment | Verifier evaluates discussion quality | **Counterfactual ablation** (self-contained) |
| External requirement | Trained verifier model | **None** (reward from task success) |
| Efficiency | N× model cost, multi-turn | 1× model, single forward (+ ablation forward) |

### 4.5 Comparison with MAGRPO (NeurIPS 2025 WS)

| Aspect | MAGRPO | Ours |
|--------|--------|------|
| Credit assignment | Group-relative advantage (cross-trajectory) | **Counterfactual** (within-sample ablation) |
| Granularity | Per-trajectory comparison | **Per-sample, per-agent** comparison |
| Can distinguish agent contributions? | No (shared group reward) | **Yes** (ablate each agent separately) |
| Applicable to intra-model agents? | No (designed for separate models) | **Yes** (adapters can be ablated by zeroing output) |

### 4.6 Efficient Implementation

**Challenge**: 每个 sample 需要 full + ablated forward → 2× cost。

**Solutions**:
1. **Mini-batch ablation**: 每个 batch 中，50% samples 做 full forward，50% 做 ablated forward。用 batch statistics 估计 counterfactual。
2. **Learned value function**: 训练一个 lightweight value network V(state) 预测 reward_no_T，避免实际跑 ablated forward。类似 COMA (Foerster et al., 2018) 的 counterfactual baseline。
3. **Periodic ablation**: 每 N steps 做一次 full counterfactual evaluation，其他 steps 用 cached cooperation values。

---

## 5. Research Direction C: Iterative Latent Debate

### 5.1 Core Problem

所有 v3-v5 版本都是 **single-pass cooperation** — agents 处理各自的 tokens 一次，通过 attention 被动共享信息。但真正的 cooperation 需要 **iterative refinement**：
- V 的 binding 应该根据 A 的需求调整（A 需要点击按钮 → V 应该 focus on 按钮区域）
- A 的 action plan 应该根据 V 的 binding 调整（V 发现没有按钮 → A 应该考虑其他操作）

### 5.2 Idea: Latent Debate Within a Single Forward Pass

在模型的特定层插入 communication rounds，agents 在 latent space 中 iterative refinement：

```
Layer 0-14:   Normal processing
              V processes image tokens with LoRA_V
              A processes text tokens with LoRA_A
                              │
                              ▼
Layer 15:     ═══ Communication Round 1 ═══
              V → extracts binding state: h_V (which UI elements are relevant?)
              A → extracts action intent: h_A (what operation might I need?)

              Cross-attend:
                V sees h_A → refines binding to match A's intent
                A sees h_V → refines intent based on available elements
                              │
                              ▼
Layer 16-20:  Process with refined representations
                              │
                              ▼
Layer 21:     ═══ Communication Round 2 ═══
              V → refined binding: "Save button at (400, 200)"
              A → refined intent: "Need to click this specific button"

              Cross-attend: Final alignment between binding and intent
                              │
                              ▼
Layer 22-27:  Generate action with aligned V+A representations
```

### 5.3 Why This Is Better Than Passive Attention

| Mechanism | Passive Attention (v3-v5) | Iterative Latent Debate |
|-----------|--------------------------|------------------------|
| V knows what A needs? | **No** (A's query is implicit) | **Yes** (A sends explicit intent h_A to V) |
| A knows what V found? | **Partially** (through attention) | **Yes** (V sends explicit binding h_V to A) |
| Iterative refinement? | **No** (single pass) | **Yes** (multiple communication rounds) |
| Communication direction | **One-way** (V's KV → A's query) | **Bidirectional** (V↔A exchange) |
| Failure mode | A misinterprets V's representation | **Self-correcting**: V adjusts based on A's confusion |

### 5.4 Comparison with Multi-Agent Debate (MAD)

| Aspect | MAD (ICML 2024) | Latent Debate (Ours) |
|--------|-----------------|---------------------|
| Communication medium | Natural language text | **Continuous latent states** |
| Rounds | Multi-turn generation (expensive) | **Intra-forward cross-attention** (cheap) |
| Model cost | N× separate LLMs | **1× model + lightweight cross-attention** |
| Information loss | Discretization bottleneck | **Lossless** (continuous states) |
| Trainability | Difficult to backprop through text | **Fully differentiable** |

### 5.5 Comparison with GraphMoE

| Aspect | GraphMoE | Latent Debate (Ours) |
|--------|----------|---------------------|
| Cooperation unit | Per-token expert selection | **Per-agent** (V and A as wholes) |
| Routing | Recurrent per-token routing through expert graph | **Cross-attention between agent states** |
| Graph structure | Pseudo-graph with virtual hub node | **Bidirectional V↔A communication channel** |
| Semantic level | Token-level (no task decomposition) | **Skill-level** (perception ↔ action) |

### 5.6 Implementation: Lightweight Cross-Attention Module

在 communication layer 插入一个 lightweight cross-attention module：

```python
class LatentDebateModule(nn.Module):
    def __init__(self, d_model, n_heads=4, r=16):
        # Low-rank cross-attention (LoRA-style for efficiency)
        self.q_proj = LoRALinear(d_model, d_model, r=r)  # project agent's query
        self.kv_proj = LoRALinear(d_model, 2*d_model, r=r)  # project other agent's KV
        self.out_proj = LoRALinear(d_model, d_model, r=r)
        self.gate = nn.Parameter(torch.zeros(1))  # learnable gate, init=0 (no change)

    def forward(self, agent_hidden, other_agent_hidden):
        q = self.q_proj(agent_hidden)
        k, v = self.kv_proj(other_agent_hidden).chunk(2, dim=-1)
        attn_out = scaled_dot_product_attention(q, k, v)
        return agent_hidden + torch.sigmoid(self.gate) * self.out_proj(attn_out)
```

**Properties**:
- Gate 初始化为 0 → `sigmoid(0) = 0.5` → 模型可以学会增加或减少 communication
- LoRA-style low-rank → 每个 module 只增加 ~3×r×d_model 参数（r=16, d=3584 → ~172K per module）
- 2 communication rounds × 2 directions = 4 modules → total ~688K additional parameters（negligible）

---

## 6. Unified Framework: A + B + C

### 6.1 The Combined Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: [image] [instruction]               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Shared Processing (Layers 0 → L₁)                 │
│  LoRA_V processes image tokens, LoRA_A processes text tokens │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Latent Debate Round 1 (Layer L₁)         [Dir C]  │
│  V sends binding state → A                                   │
│  A sends action intent → V                                   │
│  Both refine their representations                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: Latent Thought Generation (Layer L₁ → L₂) [Dir A] │
│  LoRA_T generates K latent thought tokens                    │
│  Recurrent: h_{t+1} = f_T(h_t, refined_context)             │
│  Output: [latent₁]...[latentₖ] (continuous, no vocabulary)  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 4: Latent Debate Round 2 (Layer L₂)         [Dir C]  │
│  V sends refined binding → A                                 │
│  A sends refined intent (informed by latent thoughts) → V    │
│  Final alignment                                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 5: Action Generation (Layers L₂ → 27)                │
│  LoRA_A attends to:                                          │
│    - Original visual/text context                            │
│    - Latent thought tokens from Stage 3                      │
│    - Refined binding from Stage 4                            │
│  Output: <action>JSON</action>                               │
└─────────────────────────────────────────────────────────────┘

Training:  Counterfactual Cooperative RL (Direction B)
           - Full cooperation reward vs ablated (no thought) reward
           - Per-agent credit assignment
           - T learns when to provide informative thought vs stay neutral
```

### 6.2 Training Pipeline

```
Phase 1: SFT Warm-up (Weeks 1-2)
├── Train LoRA_V + LoRA_A with text thought (v3-style) → establish basic skills
├── Curriculum: gradually replace text thought with latent thought tokens
└── Objective: L_act + λ·L_bind

Phase 2: Cooperative RL (Weeks 3-4)
├── Freeze latent thought curriculum → full latent thought mode
├── Train with GRPO + counterfactual credit assignment
│   ├── Sample: full cooperation → reward_full
│   ├── Ablate: zero out latent thought → reward_no_T
│   └── Credit: T_value = reward_full - reward_no_T
├── LoRA_T: learns selective cooperation (help when useful, neutral otherwise)
├── LoRA_A: learns to leverage latent thoughts for action generation
└── Latent debate modules: learn when to communicate and what to share

Phase 3: Analysis & Ablation (Week 5)
├── Cooperation Topology Map: analyze learned gate values per layer
├── Counterfactual importance: which samples benefit most from cooperation?
├── Latent thought probing: what information do latent thoughts encode?
└── Cross-dataset transfer: AndroidControl, WebArena
```

### 6.3 How Each Direction Addresses a Specific v3-v5 Failure

| v3-v5 Failure | Direction A (Latent Thought) | Direction B (Counterfactual RL) | Direction C (Latent Debate) |
|---------------|----------------------------|-------------------------------|---------------------------|
| Function prior contamination (v3) | **Solves**: latent thought 不在词表空间 → 无法污染 function prior | — | — |
| Destroyed cooperation (v4) | **Solves**: cross-attention ensures T→A information flow | — | **Solves**: bidirectional communication maintains cooperation |
| Insufficient training signal (v5) | — | **Solves**: counterfactual RL provides strong per-agent gradient | — |
| Passive attention not expressive enough | — | — | **Solves**: explicit cross-attention enables active cooperation |
| T doesn't know when to help | — | **Solves**: T_cooperation_value teaches selective cooperation | — |

---

## 7. Positioning Against Literature

### 7.1 Unique Contributions

| Dimension | Existing Best | Our Approach | Novelty |
|-----------|--------------|--------------|---------|
| **Agent granularity** | Per-model (LatentMAS) or per-token (MoE) | **Per-skill** (V/T/A) | New decomposition level |
| **Communication cost** | Adapter training (Interlat) or text (MAPoRL) | **Zero** (shared attention + lightweight cross-attn) | Eliminates communication overhead |
| **Communication mechanism** | One-way (most MAS) or passive (attention) | **Bidirectional iterative latent debate** | Active cooperation |
| **Thought representation** | Text CoT (all prior work) or self-recurrent latent (COCONUT) | **Multi-agent latent cooperative thought** | Cooperation without contamination |
| **Training signal** | Shared reward (MAGRPO) or discussion verifier (MAPoRL) | **Counterfactual per-agent credit** | Fine-grained cooperation incentive |
| **Efficiency** | N× models (all inter-model MAS) | **1× model + adapters** | Parameter efficient |
| **Interpretability** | Black-box (most) | **Gate values = cooperation topology** | Reveals VLM internal structure |

### 7.2 Theoretical Contributions

1. **Latent Cooperative Thought as solution to vocabulary contamination**: We identify a new failure mode (text thought contaminates function prior through shared vocabulary space) and show that latent thought eliminates this by design.

2. **Counterfactual credit assignment for intra-model agents**: We adapt multi-agent RL credit assignment to the setting where agents are adapter modules within a single model, using output ablation as the counterfactual intervention.

3. **Cooperation Topology Map**: The learned gate values across layers and modules reveal which layers of VLMs need cooperation vs specialization between visual and action processing — providing empirical guidance for VLM architecture design.

4. **Selective cooperation principle**: Through counterfactual RL, the thought agent learns to provide informative latent thoughts only when they help the action agent, automatically resolving the precision-recall tradeoff seen in v3.

---

## 8. Experimental Plan

### 8.1 Phase 1: Latent Thought Proof-of-Concept

**Goal**: Verify that latent thought avoids function prior contamination while preserving thought benefit.

| Condition | Description | Expected |
|-----------|-------------|----------|
| A1 | SVD LoRA r=256 (baseline, no thought) | 47.4% |
| A2 | v3 text thought (reproduction) | 46.1% |
| A3 | Latent thought (K=4 tokens, curriculum) | **≥47.4%** (no contamination) |
| A4 | Latent thought + text thought oracle | Upper bound |

**Key metrics**:
- Overall action accuracy
- Function confusion matrix: does click→select_text false positive disappear?
- Per-function breakdown: does structured operation benefit persist (set_font, select_table)?

**Success criterion**: A3 ≥ A1 (latent thought at least matches no-thought baseline) AND A3's function confusion matrix closer to A1's than A2's (no contamination).

### 8.2 Phase 2: Counterfactual RL

**Goal**: Train selective cooperation through per-agent credit assignment.

| Condition | Description | Expected |
|-----------|-------------|----------|
| B1 | Latent thought + SFT only | Baseline from Phase 1 |
| B2 | Latent thought + GRPO (shared reward) | Slight improvement |
| B3 | Latent thought + GRPO + counterfactual credit | **Best** — T learns selective cooperation |
| B4 | Text thought + GRPO + counterfactual credit | Ablation: does latent matter with RL? |

**Key metrics**:
- T_cooperation_value distribution: should be positive for structured operations, near-zero for click/type
- Per-function improvement: structured operations should improve WITHOUT hurting common operations

### 8.3 Phase 3: Latent Debate

**Goal**: Verify that bidirectional iterative communication improves cooperation quality.

| Condition | Description | Expected |
|-----------|-------------|----------|
| C1 | No debate (A3 from Phase 1) | Baseline |
| C2 | 1 debate round (Layer 15) | Improvement |
| C3 | 2 debate rounds (Layers 15, 21) | **Best** |
| C4 | 3 debate rounds | Diminishing returns |

**Key metrics**:
- Gate values: do debate modules learn non-trivial gating?
- Coordinate accuracy: does V's binding improve after seeing A's intent?
- A's function accuracy: does A make better function choices after seeing V's binding?

### 8.4 Phase 4: Full Framework + Analysis

**Goal**: Combine A + B + C and analyze cooperation mechanisms.

| Analysis | Method | Expected Finding |
|----------|--------|-----------------|
| Cooperation Topology Map | Plot gate values across layers × modules | Early layers share, late layers specialize |
| Latent thought content | Linear probe on latent thoughts | Encodes target location + operation type |
| Selective cooperation | Histogram of T_cooperation_value by function type | Bimodal: positive for structured ops, zero for click |
| Complementarity | Oracle ensemble analysis | Approaching 55.5% (oracle upper bound) |
| Cross-dataset transfer | AndroidControl evaluation | Framework generalizes beyond GUI-360 |

### 8.5 Baselines Comparison

| Method | Category | Expected AP | Notes |
|--------|----------|-------------|-------|
| SVD LoRA r=256 | Single adapter, no thought | 47.4% | Current best |
| Coop v3 (text thought) | Intra-model, text cooperation | 46.1% | Contamination problem |
| Self-MoA (sample 5, pick best) | Inference-time scaling | ~49% (est.) | N× inference cost |
| Full framework (A+B+C) | Intra-model latent cooperation | **≥49%** (target) | 1× inference cost |

---

## 9. Paper Framing

### 9.1 Title Candidates

1. **"Latent Cooperative Thought: Training Intra-Model Agents to Reason Together Without Words"**
2. **"Beyond Routing: Counterfactual Cooperative Training for Specialized VLM Agents"**
3. **"Think Together in Latent Space: Multi-Agent Cooperative Reasoning Within a Single Vision-Language Model"**

### 9.2 Story Arc

```
Section 1: Introduction
  - VLMs for GUI agents face a fundamental tension: visual grounding vs action generation
  - Multi-agent cooperation could resolve this, but inter-model MAS is expensive
  - We propose intra-model latent cooperation: specialized adapters that cooperate
    through latent representations, trained with counterfactual RL

Section 2: The Contamination Problem (Motivation)
  - Text-based cooperative thought helps structured operations (+17.7pp on set_font)
  - But hurts common operations (-1.6pp on click) via function prior contamination
  - Root cause: text thought shares vocabulary space with action tokens
  - Hard separation (v4) destroys cooperation entirely (5.1% AP)
  - Soft routing (v5) doesn't learn meaningful boundaries (insufficient training signal)
  → Need: (1) contamination-free thought, (2) proper cooperation training signal

Section 3: Method
  3.1: Latent Cooperative Thought — continuous thought tokens avoid vocabulary contamination
  3.2: Counterfactual Cooperative RL — per-agent credit drives selective cooperation
  3.3: Iterative Latent Debate — bidirectional communication enables active cooperation

Section 4: Experiments
  - Phase 1: Latent thought eliminates contamination (function confusion matrix)
  - Phase 2: Counterfactual RL enables selective cooperation (T helps only when useful)
  - Phase 3: Latent debate improves cooperation quality (bidirectional refinement)
  - Phase 4: Full framework analysis (Cooperation Topology Map, cross-dataset transfer)

Section 5: Analysis
  - Cooperation Topology Map: which VLM layers need cooperation vs specialization?
  - Selective cooperation: T learns bimodal behavior (active for structured, neutral for common)
  - Latent thought content: linear probes reveal encoded information

Section 6: Related Work
  - Position against LatentMAS, COCONUT, MAPoRL, MAGRPO, GraphMoE, MAD
  - Unique: intra-model + latent + counterfactual + iterative

Section 7: Conclusion
  - Cooperation is not a routing problem — it requires proper mechanism + training signal
  - Latent cooperative thought + counterfactual RL + iterative debate
  - Broader impact: general framework for decomposing VLM tasks into cooperating sub-skills
```

### 9.3 Target Venues

- **NeurIPS 2026** (deadline ~May 2026): Best fit — covers ML methodology + multi-agent + VLM
- **ICLR 2027** (deadline ~Oct 2026): If we need more time for experiments
- **ACL 2026** (deadline ~Jan 2026, passed): Alternative if framed as NLP/agent task

### 9.4 Anticipated Reviewer Concerns

| Concern | Response |
|---------|----------|
| "这不就是 MoE 吗？" | MoE is per-token routing; we are per-skill routing + learned latent communication + counterfactual RL training. MoE experts don't actively communicate. |
| "为什么不用更大的 single LoRA？" | Oracle ensemble 55.5% shows the ceiling of complementarity. Larger single adapter doesn't create complementary skills — structural inductive bias is needed. |
| "Counterfactual ablation 太贵？" | (1) Mini-batch ablation reduces cost to ~1.5×, (2) learned value function eliminates extra forward, (3) periodic ablation amortizes cost. |
| "只在 GUI-360 上测试？" | Framework is general for any VLM task with sub-skill decomposition. We validate on AndroidControl and WebArena. |
| "Latent thought 没有 interpretability？" | (1) Linear probes decode latent content, (2) gate values provide cooperation topology, (3) this is a fundamental tradeoff — LatentMAS/COCONUT make the same tradeoff and gain efficiency. |
| "v3 的 gap 只有 1.3%，值得这么复杂的 framework？" | 1.3% is after contamination. Thought helps 8% of samples, hurts 9.4%. The true potential is 55.5% (oracle). Our framework targets this potential by eliminating contamination + adding proper cooperation training. |

---

## Appendix: v3-v5 Key Data for Paper

### A.1 Function Confusion (v3 vs SVD) — Contamination Evidence

| GT → Predicted | Coop v3 | SVD | Diff | Pattern |
|---|---|---|---|---|
| click → select_text | 113 | 34 | **+79** | Coop-only over-prediction |
| click → wheel_mouse | 95 | 24 | **+71** | Coop-only over-prediction |
| click → select_table_range | 71 | 25 | **+46** | Coop-only over-prediction |
| click → drag | 43 | 5 | **+38** | Coop-only over-prediction |
| type → select_text | 62 | 17 | **+45** | Coop-only over-prediction |

### A.2 Structured Operation Benefit — Cooperation Potential

| Function | N | SVD% | Coop% | Delta |
|---|---|---|---|---|
| set_font | 96 | 36.5% | 54.2% | **+17.7%** |
| select_table | 29 | 17.2% | 37.9% | **+20.7%** |
| select_paragraph | 49 | 8.2% | 18.4% | **+10.2%** |
| select_text | 496 | 44.0% | 49.6% | **+5.6%** |

### A.3 Oracle Ensemble — Complementarity Ceiling

| Metric | Value |
|---|---|
| SVD accuracy | 47.4% |
| Coop v3 accuracy | 46.1% |
| Both correct | 38.1% |
| Either correct (oracle) | **55.5%** |
| Only SVD correct | 9.4% |
| Only Coop correct | 8.0% |
| Both wrong | 44.5% |

### A.4 v4 Catastrophe — Separation Destroys Cooperation

| Model | AP |
|---|---|
| SVD LoRA r=256 | 47.4% |
| Coop v3 (2-agent) | 46.1% |
| Coop v4 (3-agent hard) | **5.1%** |

### A.5 v5 Stagnation — SFT Cannot Learn Cooperation

| Metric | v5 Condition B (init sep=0) |
|---|---|
| sep values after 0.92 epochs | **~0.5 (barely moved from init)** |
| Expected learning | sep → different values per layer/module |
| Actual | Uniform ~0.5 everywhere |

---

## 10. Critical Data Construction Analysis

### 10.1 BPE Tokenization Bug: `</thought>` Bigram Detection Failure

**发现时间**: 2026-04-07

**问题**: v4 的 `_build_3way_mask()` 使用 bigram pattern 检测 `<thought>` 和 `</thought>` 标签：
- `<thought>` → 期望 bigram `(13708, 2450)` = `("<th", "ought")`
- `</thought>` → 期望 bigram `(522, 60565)` = `("</", "thought")`

**实测结果**: 在 1000 个训练样本中检测 `</thought>` 的 bigram：

| 结果 | 数量 | 比例 |
|------|------|------|
| `</thought>` bigram (522, 60565) **匹配成功** | 1 | **0.1%** |
| `</thought>` bigram **匹配失败** | 848 | **99.9%** |
| 无 thought | 151 | — |

**根因**: BPE tokenization 是 context-dependent 的。Thought 内容几乎总是以标点（句号/叹号/逗号）结尾，BPE 把标点和 `</` 合并成单个 token：

```
"value.</thought>"  → [957:"value", 3918:"./", 60565:"thought", 29:">"]
                                    ↑ BPE 把 "." + "</" 合并！
                                    ↑ 3918 ≠ 522 → bigram 检测失败

"cell</thought>"    → [5873:"cell", 522:"</", 60565:"thought", 29:">"]
                                    ↑ 无前置标点 → bigram 正确
                                    ↑ 但这种情况极其罕见（0.1%）
```

**所有导致失败的上下文**:

| 前置字符 | BPE 合并结果 | token id | 期望 522? |
|----------|-------------|----------|-----------|
| `.` | `".</"` | 3918 | **否** |
| `!` | `"!</"` | 18685 | **否** |
| ` ` (空格) | `" </"` | 690 | **否** |
| `)` + `,` | `"),"` + `"</"` | 701, 522 | 是（但前面有逗号时才行） |
| 无标点 | `"</"` | 522 | **是**（极罕见） |

**`<thought>` 不受影响**: `<thought>` 始终在 assistant turn 的开头，前面是 `\n`（Qwen2.5-VL chat template 的 assistant marker），BPE 不会把 `\n` 和 `<th` 合并：

```
"\n<thought>To use"  → [198:"\n", 13708:"<th", 2450:"ought", 93376:">To", ...]
                                    ↑ bigram 始终正确
```

### 10.2 Bug 的影响范围

| Version | 是否受影响 | 原因 | 后果 |
|---------|-----------|------|------|
| **v3 (2-agent)** | **不受影响** | 2-agent mask 只看 `IMAGE_PAD_ID`，不做 thought 检测 | 无 |
| **v4 (3-agent) 训练** | **严重受影响** | `_build_3way_mask` 检测到 `<thought>` 后 `in_thought=True`，永远不关闭 | **全部 action tokens 误标为 LoRA_T** |
| **v4 (3-agent) 推理** | **严重受影响** | 同样使用 bigram 检测（thought state machine） | **action tokens 由 LoRA_T 处理而非 LoRA_A** |
| **v5 (soft routing)** | **不受影响** | 2-agent，不做 thought 检测 | 无 |

**v4 的 5.1% AP 完全可以用这个 bug 解释**：

```
训练时实际发生的事情:

正确的 mask (期望):
  [system+instruction: LoRA_A] [image: LoRA_V] [thought: LoRA_T] [action: LoRA_A]
                                                                   ↑ LoRA_A 学习 action

实际的 mask (因 bug):
  [system+instruction: LoRA_A] [image: LoRA_V] [thought: LoRA_T] [action: LoRA_T !!!]
                                                                   ↑ LoRA_A 从未见过 action tokens
                                                                   ↑ LoRA_T 同时学习 thought + action
                                                                   ↑ 推理时 action tokens 由未训练的 LoRA_A 处理 → 崩溃
```

### 10.3 当前数据格式的结构性问题

即使修复了 BPE bug，当前数据格式仍有根本性问题：

**问题 1: Thought 和 Action 在同一个 assistant message 中拼接**

```json
{
  "from": "assistant",
  "value": "<thought>{thought_text}</thought>\n<tool_call>{action_json}</tool_call>"
}
```

这意味着：
- 在 2-agent 模式下，LoRA_A 同时处理 thought 和 action tokens → **contamination**
- 在 3-agent 模式下，需要用 fragile 的 bigram 检测来分离 → **BPE bug**
- Tokenizer 的 BPE 会在 thought/action 边界产生跨边界 tokens → **无法干净分割**

**问题 2: Thought 内容是 image 的文本描述**

Thought 描述了图像中的 UI 状态（"I see a Save button at the top right"），这些文本包含：
- UI 元素名称（"Save button", "Editing menu"）
- 操作动词（"click", "select", "type"）
- 空间描述（"top right", "below the ribbon"）

当 LoRA_A 在 2-agent 模式下处理这些 thought tokens 时，它的词汇先验（function prior）被 shift：
- 看到 "select" → 增加 select_text 的 prior（即使 GT 是 click）
- 看到 "drag" → 增加 drag 的 prior
- 这就是 v3 的 function confusion 的根源

**问题 3: 训练 label 覆盖了 thought + action**

```python
labels[:prompt_len] = -100   # system + user (包括 image) → 不训练
labels[prompt_len:] = ids    # assistant (thought + action) → 全部训练
```

- LoRA_V 只处理 image tokens（在 prompt 部分），但 image tokens 的 label 是 -100 → **LoRA_V 没有直接的 CE loss 训练信号**
- LoRA_V 的训练信号完全来自 attention backprop（LoRA_A 的 CE loss 通过 cross-attention 传播到 LoRA_V）
- 这意味着 LoRA_V 的 specialization 完全依赖于 indirect gradient → 训练效率低

### 10.4 Token 序列的完整结构

基于实际 tokenization 验证的完整 token 序列：

```
Position  Token ID    Decoded          Mask(2-agent)  Mask(3-agent)  Label
────────  ─────────   ───────          ─────────────  ─────────────  ─────
0         151644      <|im_start|>     LoRA_A         LoRA_A         -100
1-14      ...         system\n...      LoRA_A         LoRA_A         -100
15        151655      <|image_pad|>    LoRA_V         LoRA_V         -100
...       151655      <|image_pad|>    LoRA_V         LoRA_V         -100
976       151655      <|image_pad|>    LoRA_V         LoRA_V         -100
977-2680  ...         instruction...   LoRA_A         LoRA_A         -100
────────── PROMPT END / ASSISTANT START ──────────
2681      13708       <th              LoRA_A         LoRA_T ✓       TRAIN ←┐
2682      2450        ought            LoRA_A         LoRA_T ✓       TRAIN  │
2683      93376       >To              LoRA_A         LoRA_T ✓       TRAIN  │ Thought
...       ...         thought text     LoRA_A         LoRA_T ✓       TRAIN  │ (正确标记)
2737      897         " value"         LoRA_A         LoRA_T ✓       TRAIN  │
2738      3918        ".</"            LoRA_A         LoRA_T ✗ BUG   TRAIN ←┘
2739      60565       "thought"        LoRA_A         LoRA_T ✗       TRAIN
2740      397         ">\n"            LoRA_A         LoRA_T ✗       TRAIN
2741      151657      <tool_call>      LoRA_A         LoRA_T ✗ BUG   TRAIN ←┐
2742      198         "\n"             LoRA_A         LoRA_T ✗       TRAIN  │ Action
...       ...         action JSON      LoRA_A         LoRA_T ✗       TRAIN  │ (全部误标为 T!)
2804      151658      </tool_call>     LoRA_A         LoRA_T ✗       TRAIN  │
2805      151645      <|im_end|>       LoRA_A         LoRA_T ✗       TRAIN ←┘
```

---

## 11. Data Construction for New Research Directions

### 11.1 核心原则

基于上述分析，新数据格式需要满足：

1. **Thought 和 Action 必须在 token level 干净分离** — 不能依赖 fragile 的 tag bigram 检测
2. **Thought 内容不应包含 function 名称** — 避免 vocabulary-level contamination
3. **LoRA_V 需要有直接的训练信号** — 不能完全依赖 indirect gradient
4. **支持 latent thought** — 数据格式需要兼容 text thought → latent thought 的 curriculum 过渡

### 11.2 方案 A: Multi-Turn 分离（推荐）

**核心思想**: 把 thought 和 action 放在不同的 assistant turn 中，利用 Qwen2.5-VL 的 multi-turn chat template 自然分割。

```json
{
  "conversations": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "path/to/screenshot.png"},
        {"type": "text", "text": "instruction + history + action schema..."}
      ]
    },
    {
      "role": "assistant",
      "content": "Analyze: {visual_description_without_function_names}"
    },
    {
      "role": "user",
      "content": "Based on your analysis, what action should be taken?"
    },
    {
      "role": "assistant",
      "content": "<tool_call>{action_json}</tool_call>"
    }
  ]
}
```

**Token 序列**:
```
[im_start]system\n...[im_end]
[im_start]user\n[IMAGE_TOKENS] instruction...[im_end]
[im_start]assistant\nAnalyze: visual description...[im_end]     ← Turn 1: Thought
[im_start]user\nBased on your analysis...[im_end]
[im_start]assistant\n<tool_call>{action}</tool_call>[im_end]     ← Turn 2: Action
```

**优势**:
- Thought 和 action 被 `<|im_end|>` + `<|im_start|>` 天然分隔 — 无需 bigram 检测
- Label 可以分别控制：Turn 1 只训练 thought，Turn 2 只训练 action
- Multi-turn 分割对 BPE 完全鲁棒（不依赖任何自定义 tag）
- LoRA routing 可以按 turn 分配：Turn 1 → LoRA_T，Turn 2 → LoRA_A

**Mask building (简化)**:
```python
def build_multi_turn_mask(input_ids, turn_boundaries):
    """
    turn_boundaries: list of (start, end, agent) tuples
    e.g., [(0, prompt_end, 'A'), (thought_start, thought_end, 'T'), (action_start, action_end, 'A')]
    """
    mask = torch.zeros_like(input_ids, dtype=torch.int8)  # default: LoRA_A
    mask[input_ids == IMAGE_PAD_ID] = 1                    # LoRA_V for image
    for start, end, agent in turn_boundaries:
        if agent == 'T':
            mask[start:end] = 2                            # LoRA_T for thought turn
    return mask
```

**Label masking per-agent**:
```python
# Turn 1 (thought): only thought tokens have labels
labels_thought = input_ids.clone()
labels_thought[:thought_start] = -100
labels_thought[thought_end:] = -100

# Turn 2 (action): only action tokens have labels
labels_action = input_ids.clone()
labels_action[:action_start] = -100
labels_action[action_end:] = -100
```

### 11.3 方案 B: Special Token 分隔

**核心思想**: 在 Qwen2.5-VL tokenizer 中添加特殊 token 来分隔 thought 和 action。

```python
# 添加特殊 token
special_tokens = {
    "additional_special_tokens": [
        "<|thought_start|>",   # 单个 token，不会被 BPE 拆分
        "<|thought_end|>",
    ]
}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))
```

**数据格式**:
```json
{
  "from": "assistant",
  "value": "<|thought_start|>{thought_text}<|thought_end|><tool_call>{action}</tool_call>"
}
```

**优势**: 特殊 token 是 single token（不会被 BPE 拆分） → mask 检测 100% 可靠
**劣势**: 需要 resize embedding → 影响 pretrained weights；需要重新训练

### 11.4 方案 C: Thought Content Restructuring（配合方案 A/B）

**核心问题**: 当前 thought 内容包含 function 名称（"I will use select_text API"），即使 token-level 分离了 thought 和 action，LoRA_T 产生的 hidden states（被 LoRA_A attend to）仍可能通过 attention 间接传递 function prior。

**解决方案**: 重构 thought 内容，移除 function 名称，只保留 visual grounding 信息：

```
当前 thought（包含 function 名称）:
  "I will use the 'select_text' shortcut API to select the first paragraph"
  "I need to click the 'Editing' menu item in the ribbon"
  "I will increase the font size using the set_font API"

重构后的 thought（纯视觉描述）:
  "The first paragraph starts at the top of the document content area"
  "The 'Editing' menu item is located in the ribbon at the top right, label 64"
  "The font size control is in the Home tab ribbon, currently showing 11pt"
```

**实现**: 修改 `prepare_gui360_thought.py`，在 `recover_thought()` 后添加 function name filtering：

```python
FUNCTION_NAMES = {
    'click', 'type', 'drag', 'wheel_mouse_input', 'select_text',
    'select_table', 'select_paragraph', 'set_font', 'insert_table',
    'select_table_range', 'set_cell_value', 'auto_fill', ...
}

def clean_thought(thought_text):
    """Remove function name mentions from thought to prevent vocabulary contamination."""
    for fn in FUNCTION_NAMES:
        # Remove patterns like "use the select_text API", "call select_text"
        thought_text = re.sub(
            rf"\b(use|call|invoke|apply)\s+(the\s+)?['\"]?{fn}['\"]?\s*(API|function|shortcut)?",
            "perform the appropriate action",
            thought_text,
            flags=re.IGNORECASE
        )
        # Remove standalone function name mentions
        thought_text = re.sub(
            rf"['\"]?{fn}['\"]?(\s+API|\s+function|\s+shortcut)?",
            "the appropriate operation",
            thought_text,
            flags=re.IGNORECASE
        )
    return thought_text
```

### 11.5 方案 D: Latent Thought Data Construction（for Direction A）

**目标**: 为 COCONUT-style latent thought 训练构建 curriculum 数据。

**Curriculum 阶段**:

**Stage 1**: Full text thought（=当前数据，但用方案 A 的 multi-turn 格式）
```
User: [image] [instruction]
Assistant Turn 1: "The Save button is at (400, 200) in the ribbon area"     ← text thought
User: "What action?"
Assistant Turn 2: <tool_call>{"function": "click", "coordinate": [400, 200]}</tool_call>  ← action
```

**Stage 2**: Partial latent（thought 的前半部分是 text，后半部分替换为 latent tokens）
```
User: [image] [instruction]
Assistant Turn 1: "The Save button is" [latent₁] [latent₂] [latent₃]      ← 混合 thought
User: "What action?"
Assistant Turn 2: <tool_call>{"function": "click", "coordinate": [400, 200]}</tool_call>
```

**Stage 3**: Full latent（thought 完全是 latent tokens）
```
User: [image] [instruction]
Assistant Turn 1: [latent₁] [latent₂] ... [latentₖ]                       ← 纯 latent thought
User: "What action?"
Assistant Turn 2: <tool_call>{"function": "click", "coordinate": [400, 200]}</tool_call>
```

**Latent token 实现**:
```python
# 添加 K 个 latent placeholder tokens
LATENT_TOKENS = [f"<|latent_{i}|>" for i in range(K)]
tokenizer.add_special_tokens({"additional_special_tokens": LATENT_TOKENS})

# Stage 2: 替换 thought 的后 50% tokens 为 latent placeholders
thought_ids = tokenize(thought_text)
n_replace = len(thought_ids) // 2
thought_ids[-n_replace:] = [tokenizer.convert_tokens_to_ids(f"<|latent_{i}|>") for i in range(n_replace)]

# Training: latent token 的 embedding 由 LoRA_T 的 recurrent mechanism 产生
# 而不是从 embedding table lookup
```

### 11.6 数据构建 Action Plan

| 步骤 | 任务 | 文件 | 优先级 |
|------|------|------|--------|
| **D1** | 修复 `</thought>` bigram bug — 使用 robust detection（tokenize `</thought>` in context, 或改用 special tokens） | `cooperative_wrapper.py` | **P0 (blocker)** |
| **D2** | 构建 multi-turn 格式数据（方案 A）| `prepare_gui360_thought_v2.py` | **P0** |
| **D3** | Clean thought content — 移除 function name mentions（方案 C）| `prepare_gui360_thought_v2.py` | **P1** |
| **D4** | 实现 turn-based mask building（替代 bigram detection）| `cooperative_wrapper.py` | **P0** |
| **D5** | 验证：用修复后的 3-agent mask 重新训练 v4 | `train_cooperative.py` | **P1** |
| **D6** | 构建 latent thought curriculum 数据（方案 D）| `prepare_latent_thought.py` | **P2** |

### 11.7 BPE Bug Fix 的具体方案

**Option 1: Robust token-level detection（推荐，最快）**

不依赖 bigram，而是检测 `</thought>` 中唯一可靠的 token — `60565`（"thought"）：

```python
def _build_3way_mask_robust(self, input_ids):
    """Robust 3-way mask using token-60565 anchor + context check."""
    mask = torch.zeros_like(input_ids, dtype=torch.int8)
    mask[input_ids == IMAGE_PAD_ID] = 1  # LoRA_V

    THOUGHT_TOKEN = 60565  # "thought" — unique anchor
    OPEN_PREV = 13708      # "<th" — only appears before "ought" in <thought>
    # Close detection: look for "thought" preceded by any token containing "</"

    for b in range(input_ids.shape[0]):
        ids = input_ids[b]
        in_thought = False
        i = 0
        while i < len(ids):
            tid = ids[i].item()

            # Detect <thought> opening: still use bigram (reliable at turn start)
            if i + 1 < len(ids) and tid == 13708 and ids[i+1].item() == 2450:
                in_thought = True
                mask[b, i] = 2
                mask[b, i+1] = 2
                i += 2
                continue

            # Detect </thought> closing: look for "thought" token (60565)
            # preceded by a token whose decoded text ends with "</"
            if in_thought and tid == THOUGHT_TOKEN and i > 0:
                prev_decoded = self.tokenizer.decode([ids[i-1].item()])
                if prev_decoded.endswith('</') or prev_decoded.endswith('</'):
                    # Mark the preceding token, current token, and next token (">")
                    mask[b, i-1] = 2  # ".</" or "</"
                    mask[b, i] = 2    # "thought"
                    if i + 1 < len(ids):
                        mask[b, i+1] = 2  # ">" or ">\n"
                    in_thought = False
                    i += 2
                    continue

            if in_thought and mask[b, i] != 1:
                mask[b, i] = 2
            i += 1

    return mask
```

**Option 2: Multi-turn detection（配合方案 A，最干净）**

用 `<|im_end|>` token 作为 turn boundary，完全不需要检测 thought tags：

```python
def _build_turn_based_mask(self, input_ids, thought_turn_ranges):
    """Turn-based mask: thought turns → LoRA_T, action turns → LoRA_A."""
    mask = torch.zeros_like(input_ids, dtype=torch.int8)
    mask[input_ids == IMAGE_PAD_ID] = 1
    for b, (t_start, t_end) in enumerate(thought_turn_ranges):
        mask[b, t_start:t_end] = 2
    return mask
```

---

## 12. 🔥 Per-Layer Cooperative Communication (v6)

> **Status**: 当前优先实现方向
> **思路来源**: v5 soft routing 的自然延伸 — 从 scalar gating 进化到 learned communication
> **核心理念**: Cooperation 不是 routing problem，而是 **communication problem**。V 和 A 需要在每一层交换信息来实现 joint reasoning。

### 12.1 Motivation: Why Communication > Routing

v3-v5 的核心教训：

| Version | 做法 | 问题 |
|---------|------|------|
| v3 (2-agent) | Hard routing: image→V, text→A | V 和 A 通过 shared attention 被动合作 → contamination |
| v4 (3-agent) | Hard routing: thought→T, action→A | 完全隔离 → 合作被切断（+ BPE bug） |
| v5 (soft routing) | Learnable scalar `sep` 控制 V/A 混合比例 | SFT gradient 太弱 → sep 不学 |

**根本问题**: 以上所有版本都在回答 "how to route tokens to agents"，但从未回答 **"how agents communicate with each other"**。

在真正的 multi-agent system 中，agents 的强大之处在于 **communication**（信息交换），而不是 **isolation**（独立处理）。我们需要的是：

```
当前做法（v3-v5）: Route tokens → Agents process independently → Merge results
需要的做法（v6）:   Route tokens → Agents process → Exchange info → Process again → ...
                                                    ↑ Per-layer communication!
```

### 12.2 Architecture: Per-Layer Bidirectional Communication in Low-Rank Space

**核心创新**: 在 LoRA 的 low-rank bottleneck（dimension r）中实现 V 和 A 之间的双向通信。

#### 12.2.1 回顾 v5 的 forward pass

```python
# v5: Soft routing (scalar sep)
h_v = lora_A_v(x) → lora_B_v(·)   # V's contribution: delta_v
h_a = lora_A_a(x) → lora_B_a(·)   # A's contribution: delta_a
output = x + sep * delta_v + (1 - sep) * delta_a   # scalar mixing
```

**问题**: sep 只能控制"用多少 V vs A"，不能让 V 和 A **交换信息**。

#### 12.2.2 v6: Per-Layer Cooperative Communication

```python
# v6: Bidirectional communication in low-rank space
h_v = lora_A_v(x)                     # V's low-rank representation (dim = r)
h_a = lora_A_a(x)                     # A's low-rank representation (dim = r)

# ===== Bidirectional Communication =====
# V sees what A is thinking (A→V communication)
h_v_coop = h_v + gate_av * W_av(h_a)   # V enriched by A's representation
# A sees what V is seeing (V→A communication)
h_a_coop = h_a + gate_va * W_va(h_v)   # A enriched by V's representation

# ===== Output with communication =====
delta_v = lora_B_v(h_v_coop)   # V's output informed by A
delta_a = lora_B_a(h_a_coop)   # A's output informed by V

# Token routing (same as v3)
output = x + torch.where(mask == V, delta_v, delta_a)
```

#### 12.2.3 Architecture Diagram

```
Input x (dim = d_model = 3584)
│
├─── lora_A_v ──→ h_v (dim = r)  ←──── gate_av · W_av ←─── h_a
│                     │                                        ↑
│                     ▼                                        │
│               h_v_coop (dim = r)                             │
│                     │                                        │
│                lora_B_v                                      │
│                     │                                        │
│                     ▼                                        │
│               delta_v (dim = d_model)                        │
│                                                              │
├─── lora_A_a ──→ h_a (dim = r)  ←──── gate_va · W_va ←─── h_v
│                     │
│                     ▼
│               h_a_coop (dim = r)
│                     │
│                lora_B_a
│                     │
│                     ▼
│               delta_a (dim = d_model)
│
└─── Token Routing: output = x + where(mask==V, delta_v, delta_a)
```

**关键设计选择**:

1. **Communication 在 low-rank space (r 维)**: 而非 full d_model 维度。因为 low-rank space 是 LoRA 压缩后的 "essential information"，在这里通信更高效，参数开销也更小。

2. **Communication 在 lora_A 之后、lora_B 之前**: 这样 V 可以看到 A 对同一个 token 的理解（in compressed form），然后结合这个信息来产生自己的 output。

3. **每一层都有 communication**: 而非只在特定层。因为 VLM 的不同层编码不同层级的信息（early layers: low-level features; late layers: high-level semantics），每一层的 cooperative需求可能不同。

4. **Gate 初始化为 0 或接近 0**: 保证训练初期 v6 退化为 v3（各自独立），只有当 cooperation 真正有用时才学会打开 communication。

#### 12.2.4 完整 Module 实现

```python
class CooperativeLoRALinear(nn.Module):
    """Per-layer cooperative communication in low-rank space."""

    def __init__(self, in_features, out_features, r=128, alpha=256, comm_r=None):
        super().__init__()
        # Standard LoRA components (V agent)
        self.lora_A_v = nn.Linear(in_features, r, bias=False)
        self.lora_B_v = nn.Linear(r, out_features, bias=False)
        # Standard LoRA components (A agent)
        self.lora_A_a = nn.Linear(in_features, r, bias=False)
        self.lora_B_a = nn.Linear(r, out_features, bias=False)

        # Cooperative communication projections
        comm_r = comm_r or r  # communication dimension, default = r
        # A→V: project A's representation for V to consume
        self.W_av = nn.Linear(r, comm_r, bias=False)
        # V→A: project V's representation for A to consume
        self.W_va = nn.Linear(r, comm_r, bias=False)

        # If comm_r != r, need projection back
        if comm_r != r:
            self.proj_av_back = nn.Linear(comm_r, r, bias=False)
            self.proj_va_back = nn.Linear(comm_r, r, bias=False)
        else:
            self.proj_av_back = nn.Identity()
            self.proj_va_back = nn.Identity()

        # Learnable gates (per-module, per-direction)
        # Initialized to small negative value → sigmoid ≈ 0 → no communication initially
        self.gate_av = nn.Parameter(torch.tensor(-3.0))  # sigmoid(-3) ≈ 0.047
        self.gate_va = nn.Parameter(torch.tensor(-3.0))

        # Scaling factor
        self.scaling = alpha / r

        # Initialize LoRA weights (same as standard LoRA)
        nn.init.kaiming_uniform_(self.lora_A_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_v.weight)
        nn.init.kaiming_uniform_(self.lora_A_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_a.weight)

        # Initialize communication weights small
        nn.init.normal_(self.W_av.weight, std=0.01)
        nn.init.normal_(self.W_va.weight, std=0.01)

    def forward(self, x, base_output, mask):
        """
        Args:
            x: input hidden states [batch, seq_len, d_model]
            base_output: output from frozen base model linear layer
            mask: agent routing mask [batch, seq_len]
                  0 = LoRA_A (action/text tokens)
                  1 = LoRA_V (image tokens)
        Returns:
            output: base_output + cooperative LoRA delta
        """
        # Step 1: Compute low-rank representations
        h_v = self.lora_A_v(x)      # [batch, seq_len, r]
        h_a = self.lora_A_a(x)      # [batch, seq_len, r]

        # Step 2: Bidirectional communication
        g_av = torch.sigmoid(self.gate_av)   # A→V gate
        g_va = torch.sigmoid(self.gate_va)   # V→A gate

        comm_av = self.proj_av_back(self.W_av(h_a))  # A's info for V: [batch, seq_len, r]
        comm_va = self.proj_va_back(self.W_va(h_v))  # V's info for A: [batch, seq_len, r]

        h_v_coop = h_v + g_av * comm_av   # V enriched by A
        h_a_coop = h_a + g_va * comm_va   # A enriched by V

        # Step 3: Compute agent outputs from enriched representations
        delta_v = self.lora_B_v(h_v_coop) * self.scaling   # [batch, seq_len, d_model]
        delta_a = self.lora_B_a(h_a_coop) * self.scaling

        # Step 4: Route by mask
        # mask: [batch, seq_len] → [batch, seq_len, 1]
        mask_3d = (mask == 1).unsqueeze(-1).float()  # 1 for V tokens, 0 for A tokens
        delta = mask_3d * delta_v + (1 - mask_3d) * delta_a

        return base_output + delta
```

### 12.3 Cooperative Communication 的直觉理解

**为什么 V 需要看 A 的信息？**

V agent 处理 image tokens，负责 visual grounding（找到目标 UI 元素的位置）。但 V 不知道 A 要做什么操作：
- 如果 A 打算做 `click` → V 只需要定位一个点
- 如果 A 打算做 `select_text` → V 需要定位 start 和 end 两个点
- 如果 A 打算做 `drag` → V 需要定位 source 和 target

通过 V→A communication，V 可以在 low-rank space 中"感知"到 A 的 action intent，从而调整自己的 visual grounding 策略。

**为什么 A 需要看 V 的信息？**

A agent 处理 text tokens（instruction + action），负责决定执行什么操作。但 A 通过 attention 看到的 image 信息是 base model 的 shared representation，没有经过 V 的专业处理：
- V 可能已经识别出 "这是一个 font size 控件" → A 应该选择 `set_font`
- V 可能已经定位了表格区域 → A 应该选择 `select_table_range`

通过 A→V communication，A 可以利用 V 的 specialized visual understanding 来做出更好的 function 选择。

**直觉类比**: 这就像一个 radiologist（V）和 surgeon（A）的合作：
- Radiologist 需要知道 surgeon 打算做什么手术来决定扫描哪个区域
- Surgeon 需要 radiologist 的专业影像解读来决定手术方案
- 他们在每一个步骤（每一层）都在交流，而不是各自做完才合并

### 12.4 参数开销分析

**Communication 新增参数（per cooperative LoRA module）**:

| Component | Shape | Parameters |
|-----------|-------|------------|
| `W_av` | r × r | r² |
| `W_va` | r × r | r² |
| `gate_av` | scalar | 1 |
| `gate_va` | scalar | 1 |
| **Per module total** | | **2r² + 2** |

**全模型开销**:

| r | Per module | Modules (Qwen2.5-VL-7B: 28 layers × 4 targets) | Total | vs Base LoRA |
|---|-----------|------------------------------------------------|-------|-------------|
| 16 | 514 | 112 | **57.6K** | +0.09% |
| 64 | 8.2K | 112 | **918K** | +0.36% |
| 128 | 32.8K | 112 | **3.67M** | +0.72% |

注：Base Cooperative LoRA (2-agent, r=128) 总参数量 ≈ 2 × 128 × 3584 × 2 × 112 = **205M**。
Communication 开销仅为 0.72%（r=128）到 0.09%（r=16），**可忽略不计**。

**可选: 低维 communication（comm_r < r）**:
- 如果 r=128 但 comm_r=16，每个 module 只增加 2×(128×16) = 4096 参数
- Total: 4096 × 112 = **459K**（0.22%）
- 可以进一步降低开销同时保留 communication 能力

### 12.5 Gate Initialization Strategy

Gate 的初始化是关键设计选择：

| Strategy | Init Value | sigmoid | 含义 | 适用场景 |
|----------|-----------|---------|------|----------|
| **Zero-init (推荐)** | `gate = -3.0` | ≈ 0.047 | 几乎无 communication → 退化为独立 LoRA | 最安全；让模型自己学会何时需要 communication |
| Warm-init | `gate = 0.0` | = 0.5 | 中等 communication | 适合 communication 几乎总是有用的场景 |
| Full-init | `gate = 3.0` | ≈ 0.953 | 几乎全 communication | 不推荐：可能一开始就破坏 LoRA 的独立学习 |

**推荐 Zero-init** 的理由：
1. 训练初期，两个 LoRA 还没学到有意义的 representation，此时 communication 是 noise
2. 随着训练进行，LoRA 学到了 specialization，communication 变得有信息量
3. Gate 会自然地从 ≈0 增长到有意义的值
4. 如果某一层不需要 communication，gate 会保持在低值 → 自动适应

**可分析的指标**: 训练完成后，每一层每个 module 的 gate 值构成一个 **Cooperation Topology Map**，揭示 VLM 内部哪些层需要 V-A cooperation。

### 12.6 Data Construction: 最简方案

**核心变化**: v6 不需要 thought augmentation。

由于 V 和 A 在每一层都能直接通信，它们的 cooperative reasoning 发生在 latent space 中（通过 W_av 和 W_va 的学习），不需要显式的 text thought 作为中介。

**数据格式**: 直接使用原始 GUI-360 数据，无需任何修改。

```json
{
  "conversations": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "path/to/screenshot.png"},
        {"type": "text", "text": "instruction + history + action schema..."}
      ]
    },
    {
      "role": "assistant",
      "content": "<tool_call>{\"function\": \"click\", \"coordinate\": [400, 200]}</tool_call>"
    }
  ]
}
```

**Mask building**: 与 v3 完全相同（2-agent mask），只需区分 image tokens vs text/action tokens：

```python
def build_2agent_mask(input_ids):
    """Same as v3: image tokens → V, everything else → A."""
    mask = torch.zeros_like(input_ids, dtype=torch.int8)
    mask[input_ids == IMAGE_PAD_ID] = 1  # LoRA_V
    return mask
```

**训练 Loss**:

```python
# L_action: 标准 next-token prediction on action tokens
loss_action = cross_entropy(logits[action_tokens], labels[action_tokens])

# L_bind (optional): Contrastive binding loss for visual grounding
# 使用 gt_coords 作为监督信号
loss_bind = contrastive_binding_loss(h_v_pooled, gt_coord_embedding)

# Total
loss = loss_action + lambda_bind * loss_bind
```

**gt_coords 来源**: GUI-360 数据中已经包含 ground truth coordinate annotations，直接使用。

### 12.7 与 v3/v5 的对比

| Dimension | v3 (2-agent hard) | v5 (soft routing) | **v6 (cooperative comm)** |
|-----------|-------------------|-------------------|--------------------------|
| Cooperation mechanism | Shared attention (passive) | Scalar weighting | **Bidirectional learned comm** |
| Communication | None (implicit via attention) | None | **Per-layer, per-module** |
| V sees A's info? | Only via shared attention | No | **Yes: W_av in low-rank space** |
| A sees V's info? | Only via shared attention | No | **Yes: W_va in low-rank space** |
| Per-layer adaptation? | No (same routing everywhere) | Scalar per module | **Learned gate per module** |
| Additional params | 0 | 28×4 scalars = 112 | **~3.7M (0.72%)** |
| Data requirement | Thought-augmented | 2-agent, no thought | **Original GUI-360 (simplest)** |
| Training | SFT | SFT | **SFT** |
| Expected benefit | Function confusion (+79 click→select) | sep doesn't learn | **Joint reasoning → better function + coordinate** |

### 12.8 Training Plan

#### Phase 1: Baseline Reproduction (Week 1)

| Step | Description | Expected |
|------|-------------|----------|
| 1a | 复现 v3 2-agent (r=128) 作为 baseline | ~46.1% AP |
| 1b | 训练 v6 (r=128, gate init=-3) 对比 | **≥46.1%** (with communication) |

#### Phase 2: Ablation Study (Week 2)

| Condition | Description | Purpose |
|-----------|-------------|---------|
| v6-A | gate_av only (A→V, V 看不到 A) | V 的 grounding 是否因 A 的 intent 而改善？ |
| v6-B | gate_va only (V→A, A 看不到 V) | A 的 function choice 是否因 V 的 binding 而改善？ |
| v6-C | gate_av + gate_va (full) | 双向 communication 是否优于单向？ |
| v6-D | comm_r=16 (低维 communication) | Communication 需要多少维度？ |
| v6-E | Communication 只在特定层 (e.g., 14-27) | 低层是否不需要 communication？ |

#### Phase 3: Analysis (Week 3)

| Analysis | Method | Expected Finding |
|----------|--------|-----------------|
| **Cooperation Topology Map** | Plot `sigmoid(gate_av)` and `sigmoid(gate_va)` for all 28 layers × 4 modules | 不同层的 communication 需求不同 |
| **Function accuracy breakdown** | Per-function AP: v3 vs v6 | v6 在 structured operations 上提升，click/type 不退化 |
| **Coordinate accuracy** | Distance to GT coordinate | V→A comm 让 A 更准确地使用 V 的 grounding |
| **Communication content** | Probe W_av, W_va 的 activation patterns | 理解 V 和 A 交换了什么信息 |
| **Gate dynamics** | Plot gate values during training | Communication 何时开始变得有用？ |

#### Phase 4: Optional Extensions

| Extension | Description |
|-----------|-------------|
| + L_bind | 添加 contrastive binding loss，看是否加速 V 的 specialization |
| + RL (future) | Phase 1-3 完成后，用 counterfactual RL (Section 4) 进一步优化 cooperation |
| + Latent thought | 如果 v6 验证了 communication 的价值，可以在此基础上加入 latent thought (Section 3) |

### 12.9 Implementation Roadmap

| Step | Task | Files to Modify/Create | Priority |
|------|------|----------------------|----------|
| **I1** | 实现 `CooperativeLoRALinear` with communication | `cooperative_lora.py` | **P0** |
| **I2** | 修改 `CooperativeModelWrapper` 以支持新 module | `cooperative_wrapper.py` | **P0** |
| **I3** | 准备训练数据（原始 GUI-360 格式，无 thought）| data scripts | **P0** |
| **I4** | 配置训练脚本（SFT, 2-agent mask）| `train_cooperative.py` | **P0** |
| **I5** | 训练 v6 baseline + ablations | SLURM scripts | **P1** |
| **I6** | Gate analysis 和 visualization | analysis scripts | **P1** |
| **I7** | 完整 evaluation 和 error analysis | eval scripts | **P1** |

### 12.10 Key Research Questions

1. **Communication 的 emergence**: Gate 是否能从 near-zero 自然学到有意义的值？还是需要 warm-up？
2. **Layer-wise cooperation pattern**: 是否存在 "cooperation layers" 和 "specialization layers" 的分化？
3. **Communication vs attention**: Per-layer low-rank communication 相比 shared attention 提供了什么额外信息？
4. **Communication dimensionality**: 在 low-rank space 中需要交换多少维度的信息才能实现有效 cooperation？
5. **Cooperation Topology Map**: 这个 map 是否在不同数据集/任务上保持一致？（structural property of VLM）

### 12.11 与 Related Work 的定位

| Approach | Communication Location | Communication Content | Learning |
|----------|----------------------|----------------------|----------|
| MoE routing (DeepSeek, Switch) | Router level | Gating weights | Token-level routing |
| GraphMoE | Expert level | Graph attention scores | Topology learning |
| ReMoE | Expert level | Soft routing probabilities | Routing distribution |
| Multi-Agent Debate (MAD) | Text output level | Text arguments | Separate models |
| LatentMAS | Adapter level | Latent vectors | Adapter training |
| **v6 (Ours)** | **LoRA low-rank bottleneck** | **Compressed agent representations** | **Per-layer learned comm gates** |

**Novelty**:
1. **Communication granularity**: 在 LoRA 的 low-rank space 中通信，而非 full representation space — 这是一个新的 communication abstraction level
2. **Per-layer learned communication**: 每一层独立学习是否需要 communication 以及 communication 的内容 — 自动发现 VLM 的 cooperation topology
3. **Zero-overhead at init**: Gate 初始化使得 communication 开始时为零 — 不会破坏预训练的 representation，只在有用时才 emerge
4. **Intra-model multi-agent**: 在单个模型内部实现了 multi-agent communication，无需多个模型的开销

### 12.12 Implementation Status (2026-04-07)

> **Status**: v6 代码实现完成，待训练

#### 修改/创建的文件

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 1 | `verl/models/cooperative/cooperative_lora.py` | **修改** | 添加 `cooperative_comm`, `gate_init` 构造参数；创建 `W_av`, `W_va` (r×r 零初始化), `gate_av`, `gate_va` (init=-3.0)；修改 `forward()` 在 lora_A 之后 lora_B 之前插入双向 communication |
| 2 | `verl/models/cooperative/cooperative_wrapper.py` | **修改** | `__init__` 添加 `cooperative_comm`, `gate_init`；`_replace_target_modules` 传参；`save_cooperative_checkpoint` 保存 `lora_comm.pt` + gate 值到 config；`load_cooperative_checkpoint` 加载 comm 参数 |
| 3 | `train_cooperative.py` | **修改** | CLI 添加 `--cooperative_comm`, `--gate_init`；传参到 wrapper；`CooperativeTrainer.log()` 输出 `gate_av_mean/max`, `gate_va_mean/max` |
| 4 | `evaluation/eval_cooperative_batch.py` | **修改** | `load_model()` 从 `cooperative_config.json` 读取 `cooperative_comm`, `gate_init` 并传给 wrapper |
| 5 | `datasets/cooperative_thought/prepare_gui360_no_thought.py` | **新建** | 从 thought JSONL 去除 `<thought>...</thought>` 生成 no-thought 训练数据 |
| 6 | `scripts/exp_cooperative/train_v6_comm_thought.slurm` | **新建** | v6 + thought data, r=128, 4 nodes |
| 7 | `scripts/exp_cooperative/train_v6_comm_nothought.slurm` | **新建** | v6 + no-thought data, r=128, 4 nodes |

#### 实际实现 vs 计划的关键差异

**实现采用了简化版设计**（对应 12.2.2 的方案，而非 12.2.4 的 `comm_r` 版本）：

```python
# 实际实现 (cooperative_lora.py)
if self.cooperative_comm and hasattr(self, 'W_av'):
    g_av = torch.sigmoid(self.gate_av)
    g_va = torch.sigmoid(self.gate_va)
    h_v = h_v + g_av * F.linear(h_a, self.W_av.to(dtype))  # V sees A
    h_a = h_a + g_va * F.linear(h_v, self.W_va.to(dtype))  # A sees V (注意: h_v 此时已包含 A 的信息)
```

**设计决策**：
1. **W 零初始化**（而非 12.2.4 中的 `normal(std=0.01)`）: W=0 + gate≈0 → 初始 forward 输出与 v3 **完全一致**，更安全
2. **comm_r = r**（无低维投影）: 简化实现，先验证机制有效，comm_r<r 可作为后续 ablation
3. **顺序 communication**: V 先吸收 A 的信息，然后 A 吸收已更新的 V（含 A 信息）→ 轻微不对称，但工程上更简单
4. **num_agents==2 only**: Communication 仅在 2-agent 模式下启用，3-agent 暂不支持

#### 参数量验证

以 r=128, 7 target modules, 28 layers 为例：
- Per module: `W_av` (128×128) + `W_va` (128×128) + 2 scalars = **32,770**
- 总 modules: 28 × 7 = 196
- **Communication 总参数: 6.42M** (≈ 1.0% of base cooperative LoRA ~640M)

#### 两个实验条件

| 条件 | 数据 | SLURM | 目的 |
|------|------|-------|------|
| **v6-thought** | `gui360_train_thought.jsonl` | `train_v6_comm_thought.slurm` | 验证 communication 能否减轻 thought contamination |
| **v6-nothought** | `gui360_train_nothought.jsonl` | `train_v6_comm_nothought.slurm` | 验证 communication 在无 thought（纯 action SFT）下的效果 |

对照基线：
- v3 (2-agent hard, thought, r=256): 46.1%
- SVD LoRA r=256 (no thought): 47.4%

**Expected**: v6-nothought ≥ 47.4%（communication 提供 V-A joint reasoning 的能力，不受 thought contamination 影响）

#### 训练超参

```
r=128, alpha=256, dropout=0.05
target_modules: q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
cooperative_comm=True, gate_init=-3.0
bind_weight=0.0
batch_size=1, grad_accum=8, 16 GPUs → effective batch 128
lr=1e-5, cosine, warmup 3%
epochs=2
```

r=128（vs v3 的 r=256）：快速迭代验证 communication 机制，训练时间减半。

#### 下一步

1. 运行 `prepare_gui360_no_thought.py` 生成 no-thought 数据
2. 提交两个 SLURM job
3. 监控 gate 值变化曲线（关键指标）
4. Epoch-1 checkpoint 做快速 eval，确认无 regression
5. 完整 eval + error analysis，与 v3/SVD 对比 function confusion matrix
