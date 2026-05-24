# V17: 双专家自进化协同 RL — 通用辅助生成框架

## 背景与动机

### 当前状态

协同 LoRA (V15) 使用两个 Expert 在 r-space 中通过迭代通信协作：
- **18.6% TSR** (SFT) → **20.8% TSR** (RL)
- 两个 Expert 看到**完全相同的输入**，角色分化完全依赖隐状态级的 soft routing
- Expert 间的信息交换仅通过 r-space 通信矩阵 W₁₂/W₂₁ 实现

### V16 的教训

V16 尝试在**输入层面**区分 Expert：Expert 1 用 grounder prompt，Expert 2 用 actor prompt。
- 需要两次 forward pass（grounder cache + actor inference）
- 训练时 grounder cache 与推理时不一致
- 最终只有 **13.9% TSR**
- **教训**：输入级的显式角色分化增加了系统复杂度，反而有害

### V17 核心问题

> 如何让两个 Expert 在**不预设角色**的前提下，通过 RL 自进化出更好的协作模式？

关键观察：当前两个 Expert 的协作仅发生在**隐状态空间**（r-space 通信）。V17 的核心思路是在**输出空间**引入一个**辅助生成阶段 (auxiliary generation)**，让模型先生成辅助 tokens（思考/分析/提示），再生成决策 tokens。辅助生成的质量通过 RL 奖励信号**隐式自进化**。

---

## 文献综述：Self-Evolve 中的 "Hint" 机制

### 三大范式对比

| 维度 | Hint-based (智能体间提示) | Thought-based (自我推理) | Latent Communication (潜空间通信) |
|------|--------------------------|------------------------|--------------------------------|
| **代表工作** | MAE, SAGE, DIAL | Quiet-STaR, STaR, ReAct | LatentMAS, Coconut, **我们的 V15** |
| **辅助信息来源** | 来自其他智能体 | 来自自身 | 来自所有智能体共享空间 |
| **信息形式** | 显式文本/结构化提示 | 显式推理链/隐式 token | 连续向量/隐状态 |
| **训练信号** | 多智能体联合 RL | REINFORCE / NTP loss | 端到端梯度 / RL |
| **计算开销** | 中（多次推理） | 高（生成推理链）→ 低（Quiet-STaR） | 低（隐状态传递） |
| **信息损失** | 大（文本瓶颈） | 中（文本压缩） | 小（连续表征） |

### 关键文献

| 文献 | 关键启示 |
|------|---------|
| **Quiet-STaR** (2024) | 辅助推理不需要外部监督，仅靠下游任务奖励就能自进化 |
| **Fast Quiet-STaR** (EMNLP 2025) | 辅助生成最终可以被**内化**到模型权重中 |
| **Coconut** (Meta, 2024) | 辅助推理**不必是文本形式**——连续表征可能更强大 |
| **STaR** (NeurIPS 2022) | 可以用"逆向工程"方式为 Expert 生成辅助信息训练数据 |
| **MAE** (2025) | Agent 间的辅助信息需要与接收方能力匹配，否则无效 |
| **LatentMAS** (ICML 2026 Spotlight) | **潜空间通信 >> 文本通信**，与我们的 r-space 通信高度一致 |
| **RAGEN / StarPO** (2025) | **没有细粒度推理感知奖励，推理不会通过多轮 RL 自发涌现** |
| **MAPoRL** (ACL 2025) | 显式奖励辅助生成质量可以加速自进化 |
| **GEPA** (ICLR 2026 Oral) | Prompt 级自进化**互补于**权重级 RL 训练 |

---

## V17.0: 单 Aux + 软路由（已失败）

### 设计

模型输出分为两个阶段，**不预设任何 Expert 角色**：

```
模型输出: <aux>辅助信息</aux><decision>最终决策</decision>
          ├── 辅助生成阶段 ──┤├──── 决策生成阶段 ────┤
          两个 Expert 均参与     两个 Expert 均参与
          routing 自然分化        routing 自然分化
```

核心特性：
- 双层通信: 潜空间 (r-space) + 显式空间 (aux tokens) 互补
- Prompt 不指定 `<aux>` 中应包含什么内容，让 RL 自己发现
- 长度限制: `max_aux_tokens = 150`
- 停止条件: `stop = ["</decision>"]`

### Prompt 模板

```
...write your analysis inside <aux></aux> tags...
Then output your action inside <decision></decision> tags...

<aux>
```

### 自进化方案组合路径

| 阶段 | 方法 | 说明 |
|------|------|------|
| **V17.0** | A (隐式 RL) | 基础版：验证 aux 生成是否通过 RL 自进化 |
| **V17.1** | A + B (AT-GRPO) | 更清晰的 aux vs decision 梯度分离 |
| **V17.2** | A + B + C (Aux 质量奖励) | 为 aux 提供直接信号 |
| **V17.3** | A + B + C + D (GEPA) | 每 epoch prompt 反思进化 |

### 实验结果 ❌

V17.0 RL 训练**完全失败**：
- **模型从未生成 aux 内容**：`aux_len=0` throughout entire training
- V15 两个 expert 因对称初始化退化为近似相同函数
- Soft routing 在 aux 和 decision 阶段的行为完全相同
- 没有任何机制迫使 expert 在不同阶段承担不同角色

### 失败根因分析

**核心问题**：V15 软路由 + 相同输入/输出 mask → expert 异质性无法涌现。

具体原因：
1. **对称初始化陷阱**：两个 A 矩阵 kaiming_uniform 初始化后差异极小，B=0 意味着初始梯度也近似相同
2. **软路由无差异**：`r = sigmoid(x @ w_route)` 对 aux 和 decision tokens 产生相同的 routing pattern
3. **Aux 无直接奖励**：aux tokens 只有间接折扣 advantage，信号太弱
4. **无格式先验**：模型没有见过 `<aux>...</aux>` 的格式，RL 无法从零开始学会结构

**关键教训**：需要**结构性非对称**来打破 expert 对称性，而不是依赖 RL 自发涌现。

---

## V17.1: Dual-Aux + Phase-Conditional Hard Routing（当前实现）

### 设计动机

V17.1 解决 V17.0 的核心问题：用 **Dual-Aux 作为 heterogeneity engine**，通过 phase-conditional hard routing **强制**两个 expert 在各自 aux 阶段获得结构性非对称梯度，同时在 decision 阶段通过 KV cache 实现 token 空间的 expert 通信。

### 输出格式

三阶段生成：

```
<aux_a>[Expert A perspective]</aux_a>
<aux_b>[Expert B perspective]</aux_b>
<decision><tool_call>{action}</tool_call></decision>
```

### Routing Schedule

| Phase | Expert A (r) | Expert B (1-r) | 机制 |
|-------|-------------|----------------|------|
| `<aux_a>` 阶段 | **0.9** (硬路由) | 0.1 | Expert A 主导 |
| `<aux_b>` 阶段 | 0.1 | **0.9** (硬路由) | Expert B 主导 |
| `<decision>` 阶段 | learned | learned | 软路由 (sigmoid) |

**效果**：
- aux_a 阶段：Expert A 接收 90% 的梯度，Expert B 仅 10% → A 被训练为某种 "视角"
- aux_b 阶段：Expert B 接收 90% 的梯度，同时看到 Expert A 的输出（通过 KV cache）→ B 被训练为互补视角
- decision 阶段：两个 expert 自由混合，综合两个视角做决策

### 关键设计决策

1. **顺序固定 aux_a → aux_b**：不随机交换。aux_b 看得到 aux_a（通过 KV cache），形成 "Expert B builds on Expert A" 的非对称设计
2. **生成用自定义 loop + per-sample phase routing**：HF `model.generate()` 不支持 mid-sequence routing switch。用 step-by-step autoregressive loop，每步检查 phase transition 并切换路由。K 个 sample 并行，**每个 sample 独立追踪 phase**（用 `[B]` phase tensor 而非 majority vote，避免少数 sample 获得错误路由）
3. **训练用 per-token mask**：完整 sequence 一次 forward，通过 `phase_mask: [B, S]` 在每层内逐 token 应用硬路由
4. **Aux 长度激励用 reward bonus（不用 advantage discount）**：AT-GRPO per-phase normalization 会消掉 advantage 上的任何正标量乘子。长度激励改用 `reward += min(aux_len, min_t)/min_t * bonus`，直接影响 K 个 sample 的相对排序，不被 normalization 消掉
5. **SFT warmup 教格式（不泄露 GT）**：V17.0 不做 warmup 导致 aux_len=0。SFT 只教 `<aux_a>...<aux_b>...<decision>` 的结构，aux 模板使用纯观察性描述（不包含 GT action type），避免信息泄露
6. **AT-GRPO per-phase advantage normalization**：在 policy loss 中对 aux_a、aux_b、decision 三个 phase 的 advantage 分别标准化（zero mean, unit variance），防止 decision 阶段梯度淹没 aux 学习信号
7. **Diversity loss 保留计算图**：h1/h2 不 detach，cosine similarity 梯度回流到 lora_A_1/lora_A_2，实际推动 A 矩阵分化

### Prompt 模板

```
...
In <aux_a></aux_a>, analyze from perspective A.
In <aux_b></aux_b>, analyze from perspective B, considering perspective A's analysis.
Then output your action inside <decision></decision> tags, with a <tool_call> block:
...

<aux_a>
```

**Prompt 设计原则**：
- 对两个 perspective **不预设角色语义**（不指定 grounder/actor），只标注 A/B
- B 可参考 A（non-symmetric by design）
- RL 自发分化两个 perspective 的功能

### Loss 函数

```
L_total = L_ppo + kl_coef * L_kl + balance_weight * L_balance + diversity_weight * L_div
```

各项说明：

| Loss | 说明 | 默认权重 |
|------|------|---------|
| `L_ppo` | PPO clipped policy loss，per-token phase-weighted advantage | 1.0 |
| `L_kl` | KL penalty vs ref model (初始权重快照) | 0.001 |
| `L_balance` | Routing entropy regularization → 推 mean(r) 向 0.5 | 0.01 |
| `L_div` | **新增**: Diversity loss — 在 **decision phase tokens** 比较 Expert A 和 Expert B 的 r-space 输出 cosine similarity，推低以增加异质性。h1/h2 **保留计算图**（不 detach），梯度回流到 lora_A_1/lora_A_2 | 0.03 |

**Advantage 设计：AT-GRPO + Reward Bonus（二层分离）**:

> **关键数学性质**：per-phase normalization `normalize(c * x) = normalize(x)` 对任意正标量 c。因此 advantage 层面的 discount/ramp 会被 AT-GRPO 完全消掉。长度激励必须通过 **reward bonus** 实现。

```python
# 第一层：Reward Bonus（在 trajectory reward 中，影响 K 个 sample 的相对排序）
# 不会被 AT-GRPO normalization 消掉
if aux_len_bonus > 0:
    aux_a_tok_len = (phase_mask == 0).sum()
    aux_b_tok_len = (phase_mask == 1).sum()
    bonus = (min(aux_a_tok_len, min_t) / min_t
             + min(aux_b_tok_len, min_t) / min_t) * aux_len_bonus
    reward += bonus  # max bonus = 2 * aux_len_bonus = 0.02

# 第二层：AT-GRPO Per-Phase Normalization（在 policy loss 中，平衡三阶段梯度）
# advantage 对所有 phase uniform（不做 discount）
adv_weighted = torch.full_like(mask, advantage)  # uniform

# 在 compute_policy_loss 中 per-phase 标准化
for phase_id in (0, 1, 2):
    phase_tokens = (phase_mask == phase_id) & (mask > 0)
    if phase_tokens.sum() > 1:
        vals = advantages[phase_tokens]
        advantages[phase_tokens] = (vals - vals.mean()) / vals.std()
```

**Diversity loss 计算**:
```python
def compute_diversity_loss(phase_mask):
    for each LoRA module:
        h1, h2 = module._last_h1, module._last_h2  # [B, S, r], 保留计算图
        # 在 decision tokens 比较（相同输入 → 衡量 expert 函数差异）
        dec_mask = (phase_mask == 2)
        h1_dec = mean_pool(h1, where dec_mask)  # Expert A at decision
        h2_dec = mean_pool(h2, where dec_mask)  # Expert B at decision
        sim += cosine_similarity(h1_dec, h2_dec)
    return sim / num_modules  # 推低 → expert 函数更不同
    # 梯度: ∂sim/∂lora_A_1, ∂sim/∂lora_A_2 → 推两个 A 矩阵学不同变换
```

> **设计取舍 1：为什么不在 aux_a vs aux_b 比较？**
> 因为 aux_a 和 aux_b 的输入 token 完全不同，cosine similarity 主要反映输入分布差异而非 expert 函数差异。在 decision phase，两个 expert 处理相同输入，similarity 的变化才反映 expert 学到了不同的变换函数。

> **设计取舍 2：为什么 h1/h2 不 detach？**
> Detach 后 diversity loss 变成纯 logging 指标（梯度为零），不影响训练。保留计算图让 `∂L_div/∂lora_A_1` 和 `∂L_div/∂lora_A_2` 有效，推动两个 expert 的 A 矩阵分化。

> **设计取舍 3：为什么不用 advantage discount？**
> Per-phase normalization 数学上消除任何正标量乘子。长度激励改用 reward bonus（影响 sample 间排序），不受 normalization 影响。

---

## 实现细节

### 文件结构

```
v17_self_evolve/
├── __init__.py
├── v17_design.md                          # 本文档
├── phase_aware_cooperative_lora.py        # V17.0: PhaseAwareLoRALinear (已弃用)
├── dual_aux_cooperative_lora.py           # V17.1: DualAuxLoRALinear + Wrapper ★
├── train_aux_decision_rl.py               # V17.1: 主训练脚本 ★
├── serve_v17_direct.py                    # 推理服务（支持 dual-aux）
├── prepare_dual_aux_sft.py                # SFT warmup 数据准备
├── data/
│   ├── dual_aux_sft_train.jsonl           # 生成的 SFT 训练数据
│   └── dual_aux_sft_val.jsonl             # 生成的 SFT 验证数据
└── scripts/
    ├── sft_warmup_v17.slurm               # SFT warmup SLURM 脚本
    ├── train_v17.slurm                    # RL 训练 SLURM 脚本
    ├── eval_v17.slurm                     # 评估 SLURM 脚本
    └── logs/
```

### 1. `dual_aux_cooperative_lora.py` — LoRA 层 + Wrapper

#### DualAuxLoRALinear

继承 `IterativeCooperativeLoRALinear`（V13），新增 per-token phase-conditional hard routing：

```python
class DualAuxLoRALinear(IterativeCooperativeLoRALinear):
    """LoRA layer with per-token phase-conditional hard routing."""

    # 额外属性：
    _phase_mask: Optional[Tensor]      # [B, S] int: 0=aux_a, 1=aux_b, 2=decision
    _global_phase_id: Optional[int|Tensor]  # Generation: int (全局) 或 [B] tensor (per-sample)
    _aux_a_route: float = 0.9          # 硬路由值
    _aux_b_route: float = 0.1
    _capture_expert_h: bool = False    # 是否捕获 h1/h2 用于 diversity loss
    _last_h1, _last_h2: Optional[Tensor]  # 捕获的 expert 输出

    def forward(self, x):
        # 1. 计算 soft routing: r_soft = sigmoid(x @ w_route)
        # 2. 根据 phase_mask 或 global_phase_id 覆盖为硬路由：
        #    phase==0 → r=0.9 (Expert A 主导)
        #    phase==1 → r=0.1 (Expert B 主导)
        #    phase==2 → r=r_soft (学习路由)
        # 3. 如果 _capture_expert_h: 保存 h1, h2 (保留计算图，不 detach)
        # 4. 迭代通信 (V13 不变)
        # 5. h_blend = r * h1 + (1-r) * h2
        # 6. delta = B @ h_blend * scaling
```

**Training vs Generation 两种模式**：
- **Training**: 一次 forward 整个 sequence。设置 `_phase_mask: [B, S]`，每个 token 位置独立路由
- **Generation**: token-by-token autoregressive。设置 `_global_phase_id: Tensor[B]`，每个 sample 独立追踪 phase（支持 batch 内不同 sample 处于不同 phase）

#### DualAuxCooperativeVLMWrapper

继承 `IterativeCooperativeVLMWrapper`（V13），替换模块为 `DualAuxLoRALinear`：

```python
class DualAuxCooperativeVLMWrapper(IterativeCooperativeVLMWrapper):
    # 核心方法：
    set_phase_mask(mask: [B,S])           # Training: per-token phase mask
    clear_phase_mask()                     # 清除 mask 恢复 soft routing
    set_phase_mask_global(phase_id: int|Tensor[B])  # Generation: 全局或 per-sample phase
    set_capture_expert_h(enabled: bool)    # 开关 diversity loss 数据采集
    compute_diversity_loss(phase_mask) → Tensor  # r-space diversity loss (在 decision tokens 比较)
    save_cooperative(save_dir)             # 保存，type="dual_aux_cooperative_v17"
    load_cooperative(load_dir)             # 加载（兼容 V13 权重格式）
```

**权重兼容性**：DualAux 的 LoRA 参数结构与 V13 完全相同（lora_A_1, lora_A_2, lora_B, route_weights, comm_*），因此可以：
- 用 V13/V15 的 SFT checkpoint 初始化
- 用 `v15_gui_360/train_cooperative_sft.py`（V13 wrapper）训练 warmup，然后加载到 DualAux wrapper

### 2. `train_aux_decision_rl.py` — RL 训练器

基于 V17.0 训练器完全重写。

#### 主要改动一览

| 模块 | V17.0 | V17.1 |
|------|-------|-------|
| Prompt | 单 `<aux>` | Dual `<aux_a>` + `<aux_b>` |
| 生成 | `model.generate()` 一次调用 | 自定义 phase-aware autoregressive loop (per-sample routing) |
| Phase masks | 2-phase (aux/decision) | 3-phase (aux_a=0, aux_b=1, decision=2) |
| Advantage | 2-phase weighting | Uniform + AT-GRPO per-phase norm + aux length reward bonus |
| Diversity loss | 无 | `L_div` (decision tokens r-space cosine sim, weight=0.03) |
| 日志 | `aux_len` | `aux_a_len`, `aux_b_len`, `diversity_loss` |
| Wrapper | V13 或 PhaseAware | DualAuxCooperativeVLMWrapper |
| Stop strings | `["</decision>"]` | `["</decision>"]` (不变) |

#### Tag 解析

```python
AUX_A_OPEN = "<aux_a>"
AUX_A_CLOSE = "</aux_a>"
AUX_B_OPEN = "<aux_b>"
AUX_B_CLOSE = "</aux_b>"
DECISION_OPEN = "<decision>"
DECISION_CLOSE = "</decision>"

def parse_dual_aux_decision(text) -> (aux_a_text, aux_b_text, decision_text):
    # Regex 提取三段内容
    # Fallback: 如果无标签，整段视为 decision
```

#### 3-Phase Mask 计算

```python
def compute_three_phase_mask_fast(token_ids, tokenizer, prompt_len, tag_ids_cache):
    """对 response tokens 计算 3-phase mask。

    返回 [resp_len] int tensor: 0=aux_a, 1=aux_b, 2=decision
    使用 token subsequence matching（预编码 tag IDs），O(n) 复杂度。
    """
    # 1. 找 <aux_a>.....</aux_a> → 标记为 phase 0
    # 2. 找 <aux_b>.....</aux_b> → 标记为 phase 1
    # 3. 其余 → phase 2 (decision)
```

#### Phase-Aware Generation (自定义 autoregressive loop)

```python
def _generate_k_samples_phase_aware(self, messages, image, K, max_new_tokens):
    """K 个 sample 并行生成，每步检测 phase 转换并切换路由。"""
    # 1. Repeat prompt K 次
    # 2. 初始 forward 获取 past_key_values, phase=0 (aux_a)
    # 3. 逐 token 循环:
    #    a. Sample next token (temperature + top_p)
    #    b. 对每个 sample 解码最新 token，检测 phase 转换:
    #       "</aux_a>" 出现 → phase 1 (aux_b)
    #       "</aux_b>" 出现 → phase 2 (decision)
    #       "</decision>" 出现 → finished
    #    c. 构建 per-sample phase tensor: [B] (每个 sample 独立)
    #    d. set_phase_mask_global(phase_tensor)  # [B] tensor
    #    e. Forward only new token with past_key_values
    # 4. 返回 [K, seq_len] output_ids
```

**为什么 per-sample 而非 majority vote**：
- Majority vote 导致少数 sample 获得**错误路由**（如某 sample 仍在 aux_a 但被强制用 aux_b 路由）
- 这使 RL 的 on-policy 数据产生系统性偏差
- Per-sample phase tensor `[B]` 确保每个 sample 获得正确的 phase routing

**为什么不用 HF generate()**：
- `model.generate()` 的 LogitsProcessor 在 forward 之后才被调用
- 我们需要在 forward **之前**设置 routing phase
- 自定义 loop 允许每步切换路由

#### AT-GRPO + Aux Length Reward Bonus

```python
def _compute_phase_weighted_advantage(self, advantage, phase_mask, mask):
    """Uniform advantage — AT-GRPO handles gradient balancing。"""
    return torch.full_like(mask, advantage)
    # 不做 discount/ramp: normalize(c*x) = normalize(x), 标量乘子无效

def compute_policy_loss(..., phase_mask=None):
    """PPO loss + AT-GRPO per-phase normalization。"""
    if phase_mask is not None:
        for phase_id in (0, 1, 2):
            phase_tokens = (phase_mask == phase_id) & (mask > 0)
            if phase_tokens.sum() > 1:
                vals = advantages[phase_tokens]
                advantages[phase_tokens] = (vals - vals.mean()) / vals.std()

# Aux 长度激励通过 reward bonus 实现（不被 normalization 消掉）
# 在 generate_episode_rollouts 中：
bonus = (min(aux_a_tok_len, min_t)/min_t + min(aux_b_tok_len, min_t)/min_t) * aux_len_bonus
reward += bonus  # max bonus = 2 * 0.01 = 0.02
```

> **AT-GRPO 为什么消掉 discount**：per-phase normalization 将每个 phase 的 advantage 标准化为 mean=0, std=1。乘以正标量 c 后 normalize(cx) = normalize(x)，因此 discount 和 ramp 在数学上被完全消除。长度激励必须通过 reward（影响 K 个 sample 的相对排序）而非 advantage scaling 实现。

#### 训练 Step 中的 Diversity Loss

```python
def train_step(self, batch_rollouts):
    # 1. 开启 expert h 捕获
    self.model.set_capture_expert_h(True)

    for each sample:
        # 2. Forward with phase_mask (硬路由)
        tok_lp, mask, _ = self._compute_token_log_probs(
            ..., phase_mask=phase_mask  # ← 新增: 设置 phase mask
        )

        # 3. 计算 diversity loss
        div_loss = self.model.compute_diversity_loss(full_phase_mask)

        # 4. 总 loss
        loss = (pg_loss + kl_coef * kl_loss
                + balance_weight * bal_loss
                + diversity_weight * div_loss) / total_seqs
        loss.backward()

    # 5. 关闭 expert h 捕获
    self.model.set_capture_expert_h(False)
```

#### CLI 新增参数

```python
# V17.1 Dual-Aux 特有
--min_aux_tokens 20       # Aux 长度 reward bonus 的 ramp 目标
--aux_len_bonus 0.01      # 每个 aux 达到 min_aux 的 reward bonus（max total = 0.02）
--aux_a_hard_route 0.9    # aux_a 阶段 Expert A 的 routing 值
--aux_b_hard_route 0.1    # aux_b 阶段 Expert A 的 routing 值
--diversity_weight 0.03   # diversity loss 权重（在 decision tokens 比较，有梯度）
```

### 3. `prepare_dual_aux_sft.py` — SFT Warmup 数据

将 `v12_gui_360/data/gui360_train_2000_balanced.jsonl`（episode 格式）转换为 ShareGPT JSONL。

**Template-based aux 内容**（故意通用、低信息量、**不泄露 GT action type**）：
- `aux_a`: "I observe the current screen. The task requires {goal}. I need to identify the relevant UI element and determine the appropriate action."
- `aux_b`: "Considering perspective A's analysis, I examine the screen layout and identify the target element for the next action."

**输出格式**：
```json
{
  "conversations": [
    {"from": "human", "value": "<prompt with dual-aux template>"},
    {"from": "gpt", "value": "<aux_a content>\n</aux_a>\n<aux_b>\n<aux_b content>\n</aux_b>\n<decision>\n<tool_call>..."}
  ],
  "images": ["/path/to/screenshot.png"]
}
```

**设计原则**：
- 模板故意通用 → 模型学**结构**但不 overfit **内容**
- SFT 只做 0.5~1 epoch，足够学格式
- **不包含 GT action type**（避免 SFT 数据信息泄露导致 RL 阶段 aux 不进化）

### 4. `serve_v17_direct.py` — 推理服务

修改内容：
- 新增 `_AUX_A_RE`, `_AUX_B_RE` 正则匹配 dual-aux tags
- `_do_inference()` 返回 4 个值: `text_out, aux_content, aux_a_content, aux_b_content`
- `ChatCompletionResponse` 新增 `aux_a_content`, `aux_b_content` 字段
- 模型加载支持 `type="dual_aux_cooperative_v17"`，实例化 `DualAuxCooperativeVLMWrapper`
- Strip 逻辑: 同时移除 `<aux>`, `<aux_a>`, `<aux_b>` tags（兼容 V17.0 和 V17.1）

### 5. SLURM 脚本

#### `scripts/sft_warmup_v17.slurm`

```
nodes=2, ngpu=4/node, grad_accum=8
两步:
  1. python prepare_dual_aux_sft.py → 生成 SFT 数据
  2. torchrun train_cooperative_sft.py → 训练 1 epoch（≈0.5 有效 epoch）
     使用 V13 wrapper（权重与 DualAux 兼容）
     lora_lr=2e-5, route_lr=1e-3
输出: checkpoints/v17_dual_aux_sft_warmup/
```

#### `scripts/train_v17.slurm`

```
nodes=4, ngpu=4/node, grad_accum=4
新增参数:
  --aux_a_hard_route 0.9
  --aux_b_hard_route 0.1
  --diversity_weight 0.03
  --min_aux_tokens 20
  --aux_len_bonus 0.01
  --dapo_threshold 0.0
SFT 起点: checkpoints/v17_dual_aux_sft_warmup/
输出: checkpoints/v17_dual_aux_rl/
```

---

## 复用代码

| 来源 | 复用内容 |
|------|---------|
| `v13_gui_360/iterative_cooperative_lora.py` | 基类: `IterativeCooperativeLoRALinear`（通信、路由、门控） |
| `v13_gui_360/iterative_cooperative_wrapper.py` | 基类: `IterativeCooperativeVLMWrapper`（模块替换、参数管理） |
| `v12_gui_360/reward.py` | `compute_step_reward`（不变，只看 decision 内容） |
| `v15_gui_360/train_cooperative_sft.py` | SFT warmup 训练器（不变，V13 wrapper 权重兼容） |
| `v12_gui_360/data/gui360_train_2000_balanced.jsonl` | SFT warmup 数据源 |

---

## 超参数

```python
# 继承 V15/V13
lora_r = 128, lora_alpha = 256
target_modules = [q_proj, k_proj, v_proj, o_proj]
num_comm_rounds = 2
K = 8, temperature = 1.0, top_p = 0.95
balance_weight = 0.01
routing_noise_scale = 0.5

# V17.1 特有
max_aux_tokens = 150          # 辅助生成最大总长度
min_aux_tokens = 20           # Aux 长度 reward bonus 的 ramp 目标
aux_len_bonus = 0.01          # 每个 aux 达到 min_aux 的 reward bonus（max total=0.02）
aux_a_hard_route = 0.9        # aux_a 硬路由值
aux_b_hard_route = 0.1        # aux_b 硬路由值
diversity_weight = 0.03       # Diversity loss 权重（decision tokens, 有梯度）
max_new_tokens = 256          # Decision 阶段最大长度
kl_coef = 0.001
clip_range = 0.2
lora_lr = 1e-5
route_lr = 1e-3
```

---

## 与 V15/V16/V17.0 对比

| 方面 | V15 | V16 | V17.0 | **V17.1** |
|------|-----|-----|-------|-----------|
| Expert 输入 | 相同 | 不同 (2次fwd) | 相同 | 相同 |
| Forward 次数 | 1 | 2 | 1 | 1 |
| 辅助信息 | 无 | Grounder cache | `<aux>` (空) | `<aux_a>` + `<aux_b>` |
| Expert 路由 | 软 (sigmoid) | N/A | 软 (sigmoid) | **硬 (aux) + 软 (decision)** |
| 角色分化 | 仅 r-space | Prompt 级 | 无（失败） | **Phase-conditional hard routing** |
| SFT warmup | 有 | 有 | 无 | **有 (0.5 epoch 格式教学)** |
| Diversity loss | 无 | 无 | 无 | **有 (decision tokens r-space cosine sim)** |
| 自进化 | 无 | 无 | 失败 | RL 进化 aux 内容 |
| TSR | 20.8% | 13.9% | 失败 | **TBD** |

---

## V17.1 Code Review 修复记录

实现完成后经过详细 code review，发现并修复了 6 个问题（3 个关键，3 个重要）：

### Critical Fix 1: Diversity Loss 测量对象错误

**问题**：原实现比较 h1@aux_a vs h2@aux_b（不同输入位置），衡量的是输入分布差异，不是 expert 函数差异。
**修复**：改为比较 h1 vs h2 **在 decision tokens**（phase=2），此时两个 expert 处理相同输入，cosine similarity 反映的是 expert 变换函数的差异。

### Critical Fix 2: 生成时 Majority Vote 路由导致 On-Policy 数据偏差

**问题**：K 个 sample 并行生成时，用 majority vote 决定全局 phase。少数 sample 可能处于不同 phase，获得错误路由。
**修复**：`_global_phase_id` 改为支持 `Tensor[B]`（per-sample phase），每个 sample 独立追踪并设置正确的 phase routing。LoRA layer 的 forward 中增加 `isinstance(pid, torch.Tensor)` 分支处理 per-sample 路由。

### Critical Fix 3: SFT 模板泄露 GT Action Type

**问题**：`prepare_dual_aux_sft.py` 的 aux 模板包含 GT action type（"next logical step is to {click/type/drag}"），导致 SFT 后模型在 aux 中已经 "知道" 答案，RL 阶段 aux 不会进化新内容。
**修复**：移除 action_type 参数，aux 模板改为纯观察性描述："I need to identify the relevant UI element and determine the appropriate action."

### Fix 4: min_aux_tokens 硬阈值 → 软线性 Ramp ~~→ **已被 Fix 8 supersede**~~

**问题**：`aux_len < 20 → discount=0` 创造 reward hacking 激励。
**原修复**：线性 ramp `discount = base_discount * min(aux_len / min_aux, 1.0)`。
**Superseded by Fix 8**：AT-GRPO normalization 消掉任何 advantage 层面的标量乘子，soft ramp 无效。长度激励改用 reward bonus 实现。

### Fix 5: AT-GRPO Per-Phase Advantage Normalization

**问题**：原实现只有简单的 phase discount，没有真正的 per-phase normalization。Decision 阶段的 advantage 量级远大于 aux，导致 aux 梯度被淹没。
**修复**：在 `compute_policy_loss()` 中增加 `phase_mask` 参数，对每个 phase 的 advantage tokens 独立做 zero-mean/unit-variance 标准化。

### Fix 6: Diversity Weight 0.1 → 0.03

**问题**：diversity_weight=0.1 可能主导 loss landscape，干扰 PPO 学习。
**修复**：降低到 0.03（slurm 脚本 + CLI 默认值同步更新）。

### Fix 7: Diversity Loss 的 h1/h2 被 detach — 梯度为零

**问题**：`_last_h1 = h_1.detach()` 和 `_last_h2 = h_2.detach()`。Detach 后 diversity loss 对 lora_A_1/lora_A_2 的梯度为零，diversity_weight=0.03 实际等于 0（纯 logging）。
**修复**：去掉 `.detach()`，保留计算图。`compute_diversity_loss` 的 cosine similarity 梯度能回流到 lora_A_1/lora_A_2，推动两个 A 矩阵学不同变换。

### Fix 8: AT-GRPO Normalization 消掉 Advantage Discount/Ramp

**问题**：数学性质 `normalize(c * x) = normalize(x)`。Per-phase normalization 后任何正标量乘子（aux_adv_discount, soft ramp）都被消除。`aux_adv_discount=0.5` 和 `1.0` 训练效果完全等价。
**修复方案**：采用 **(a) + (c)** 组合：
- **(a)** PPO 层：uniform advantage（所有 phase 相同），AT-GRPO per-phase normalization 负责梯度平衡
- **(c)** Reward 层：aux length bonus `min(aux_len, min_t)/min_t * 0.01` 加到 trajectory reward，影响 K 个 sample 的相对排序，不被 normalization 消掉。量级故意很小（max total = 0.02），避免在全 K 失败时 bonus 主导 advantage 排序
- 移除 `--aux_adv_discount`（deprecated），新增 `--aux_len_bonus 0.01`

---

## 执行顺序

1. ✅ 更新 `v17_design.md` — 将完整设计写入
2. ✅ 新建 `dual_aux_cooperative_lora.py` — LoRA layer + wrapper
3. ✅ 重写 `train_aux_decision_rl.py` — 训练器主体
4. ✅ 新建 `prepare_dual_aux_sft.py` — 数据转换
5. ✅ 修改 `serve_v17_direct.py` — 支持 dual-aux tag parsing
6. ✅ 修改 `train_v17.slurm` + 新建 `sft_warmup_v17.slurm`
7. ✅ Code review round 1：修复 6 个问题（diversity loss 比较位置、per-sample routing、SFT 模板泄露、soft ramp、AT-GRPO、diversity weight）
8. ✅ Code review round 2：修复 2 个数学层面问题（diversity loss detach bug、AT-GRPO 消掉 discount/ramp → 改用 reward bonus）
9. ⬜ **SFT warmup** (0.5~1 epoch, ~1h on 2 nodes)
10. ⬜ **RL 训练** (4 nodes, train_v17.slurm)

## 验证清单

1. **数据**: 抽样确认 dual-aux SFT 格式正确，aux_a/aux_b 非空
2. **SFT 后**: 用 serve 推理确认模型生成 `<aux_a>...<aux_b>...<decision>` 格式
3. **RL 训练**: 监控 `aux_a_len > 0`, `aux_b_len > 0`, `diversity_loss` 下降
4. **异质性度量**: 分别用 `expert_1_only` / `expert_2_only` 推理，比较 per-action-type accuracy 差异
5. **TSR**: 对比 V15 baseline (20.8%)

## Ablation 计划

| 实验 | 配置 | 目的 |
|------|------|------|
| V15 baseline | 无 aux | 基线 20.8% TSR |
| Single aux 无硬路由 | V17.0 (已失败) | 证明 soft routing 不足 |
| Dual-aux 无硬路由 | dual format + soft routing | 隔离硬路由的贡献 |
| Dual-aux + 硬路由 无 diversity loss | `diversity_weight=0` | 隔离 diversity loss 的贡献 |
| **完整 V17.1** | dual-aux + 硬路由 + diversity loss | 完整系统 |

---

## 参考文献

- [Quiet-STaR](https://arxiv.org/abs/2403.09629) (2024) — 模型学习在说话前思考
- [Fast Quiet-STaR](https://arxiv.org/abs/2505.17746) (EMNLP 2025) — 无显式思维 token 的思考
- [Coconut](https://arxiv.org/abs/2412.06769) (Meta, 2024) — 连续潜空间推理
- [STaR](https://arxiv.org/abs/2203.14465) (NeurIPS 2022) — 自举推理
- [LatentMAS](https://arxiv.org/abs/2511.20639) (ICML 2026 Spotlight) — 纯潜空间多智能体协作
- [RAGEN / StarPO](https://arxiv.org/abs/2504.20073) (2025) — 轨迹级智能体 RL 框架
- [MAE](https://arxiv.org/abs/2510.23595) (2025) — 多智能体共进化
- [SAGE](https://arxiv.org/abs/2603.15255) (2025) — 四智能体自进化 LLM 推理
- [Stronger-MAS / AT-GRPO](https://arxiv.org/abs/2510.11062) (ICLR 2026) — Agent-Turn-wise GRPO
- [MAPoRL](https://arxiv.org/abs/2502.18439) (ACL 2025) — 多 Agent 后训练协作
- [GEPA](https://arxiv.org/abs/2507.19457) (ICLR 2026 Oral) — Prompt 反思进化
- [AgentEvolver](https://arxiv.org/abs/2511.10395) — 自进化 Agent 系统
- [MobileRL / ADAGRPO](https://arxiv.org/abs/2509.18119) — GUI Agent 在线 RL
- [Inner Monologue](https://arxiv.org/abs/2207.05608) (CoRL 2022) — 具身推理语言规划
- [ReAct](https://arxiv.org/abs/2210.17527) (2023) — 推理与行动协同

---

## V17.2: Standard Generation + Phase-Gradient Routing + Multi-Reward

### V17.1 失败分析

V17.1 训练 23 步，所有 loss=0, reward=0, 零梯度。根因：自定义 autoregressive generation loop 的 per-token hard routing 在推理时强制单 Expert 输出，破坏了 SFT warmup 后（随机初始化）的输出质量。表现为双峰输出长度（3-19 tokens EOS 或 406 tokens 截断）。

### 核心设计改变

**关键洞察**："不同的 reward 去更新 module 的不同部分" — 分离生成质量（用标准生成）和梯度特化（用 phase routing 在训练 backward），增加细粒度 reward 路由到不同模块。

### 三层架构

#### 第一层：标准生成（解决输出崩溃）

用 `model.generate()` 替换自定义 autoregressive loop：
- 推理时不使用 phase routing — cooperative LoRA 使用自然学习的 soft routing
- 两个 Expert 自然参与所有 token 生成 → 连贯输出
- 停止条件：`eos_token_id` + `</tool_call>` (ID 151658) + `max_new_tokens=512`
- 生成后解析 aux_a/aux_b/decision 三阶段

#### 第二层：Phase-Gradient Routing（Expert 特化通过 backward pass）

训练 forward 时保持 phase_mask hard routing（已有实现）：
- 生成后从生成的 token 计算 phase mask
- 训练 forward：`model.set_phase_mask(full_mask)` → hard routing r=0.9/0.1
- 梯度自然非对称流动：aux_a tokens → 90% 梯度到 Expert A，aux_b → 90% 到 Expert B

**效果**：生成是 cooperative（两个 Expert），但学习是 specialized（每个 Expert 主要从 "自己的" aux phase 学习）。

#### 第三层：Multi-Reward（不同 reward → 不同模块）

三个 reward 通道通过 phase-specific advantage 影响不同参数：

| Reward | Signal | Magnitude | Target |
|--------|--------|-----------|--------|
| `r_decision` | action correctness (format+type+coord) | 0-1 | All params (base advantage) |
| `r_structure` | valid 3-phase format, both aux non-empty | 0 or 0.1 | All params (added to total reward) |
| `r_aux_utility` | aux predicts correct action type | 0-0.3 | Expert A/B (via phase-specific advantage boost) |

**Phase-specific advantage（核心创新）**：

```python
# GRPO 标准 advantage
base_advantage = grpo_advantage(total_rewards_across_K)  # [K]

# Per-token phase-specific advantage
for token at position t:
    if phase_mask[t] in (0, 1):  # aux_a or aux_b tokens
        advantage[t] = base_advantage + aux_utility_weight * aux_utility_bonus
    else:  # decision tokens (phase 2)
        advantage[t] = base_advantage
```

当 backward 流过 phase-masked forward 时：
- aux_a tokens 的 **boosted advantage** → **9x 梯度到 Expert A** (r=0.9)
- aux_b tokens 的 **boosted advantage** → **9x 梯度到 Expert B** (r=0.1)
- decision tokens 的 base advantage → 按 learned routing 分配

### Aux Utility Reward

```python
def compute_aux_utility(aux_a_text, aux_b_text, gt_action):
    gt_type = gt_action["action"]
    bonus = 0.0
    for aux_text in [aux_a_text, aux_b_text]:
        lower = aux_text.lower()
        if gt_type == "click" and "click" in lower: bonus += 0.15
        elif gt_type == "type" and ("type" in lower or "input" in lower): bonus += 0.15
        elif gt_type in ("swipe","drag") and any(w in lower for w in ["swipe","scroll","drag"]): bonus += 0.15
    return min(bonus, 0.3)
```

**非 GT 泄露**：(1) reward 在生成后计算，不输入模型，(2) RL 本身就使用 GT 计算 reward。

### 修改文件

| 文件 | 改动 |
|------|------|
| `train_aux_decision_rl.py` | 替换生成方法、多 reward、phase-specific advantage、截断处理、logging |
| `scripts/train_v17.slurm` | `--max_new_tokens 512`, `--aux_utility_weight 0.15`, 移除 `--max_aux_tokens` |
| `v17_design.md` | 新增本节 |

### 不修改的文件

- `dual_aux_cooperative_lora.py` — phase_mask routing 已正常工作
- `v12_gui_360/reward.py` — 不变
- SFT warmup checkpoint — 复用 `checkpoints/v17_dual_aux_sft_warmup/epoch-0`

### 超参数

```python
# V17.2 新增/修改
max_new_tokens = 512          # 全响应最大长度（aux + decision）
aux_utility_weight = 0.15     # aux_utility advantage boost 权重
# 移除: max_aux_tokens, min_aux_tokens, aux_len_bonus（V17.1 遗留，不再使用）
```

### 验证计划

1. RL 训练前几步应该看到：非零 reward（模型输出可解析 action — debug job 已确认）、非零 advantage → 非零 loss → grad_norm > 0
2. 训练 25 步后：保存 checkpoint，评估，对比 V15 baseline (20.8% TSR)
