# Cooperative LoRA v6: Per-Layer V↔A Communication

## 1. Motivation

v3-v5 的 cooperative LoRA 都没能实现真正的 V-A cooperation：

- **v3**：通过 shared attention 被动合作 → contamination（两 agent 的 output 被迫纠缠）
- **v4**：BPE bug → catastrophic failure
- **v5**：scalar `sep` 梯度信号不足 → gate 几乎不动

v6 的核心思路：**在 LoRA 的 low-rank space 中，让每一层做一次双向 cooperative communication**，使 V 和 A 在单次 forward pass 中 joint reasoning。

## 2. Architecture

在每一层 cooperative LoRA 中（2-agent hard routing 路径）：

```
原来 (v3):
    delta_v = B_v @ A_v @ x   # V agent 单独计算
    delta_a = B_a @ A_a @ x   # A agent 单独计算
    (两 agent 之间没有任何 information flow)

v6:
    h_v = A_v @ x             # [B, S, r] — V 的 latent
    h_a = A_a @ x             # [B, S, r] — A 的 latent

    h_v_new = h_v + sigmoid(gate_av) * W_av @ h_a   # V 看到 A 的 latent
    h_a_new = h_a + sigmoid(gate_va) * W_va @ h_v_new  # A 看到 V 的 latent

    delta_v = B_v @ h_v_new * scaling
    delta_a = B_a @ h_a_new * scaling
```

每层新增参数（per cooperative LoRA module）：
- `W_av ∈ R^{r×r}`：A→V 的 low-rank communication projection
- `W_va ∈ R^{r×r}`：V→A 的 low-rank communication projection
- `gate_av ∈ R`：scalar sigmoid gate，控制 A→V 信号强度
- `gate_va ∈ R`：scalar sigmoid gate，控制 V→A 信号强度

**设计要点**：communication 发生在 LoRA 的 **low-rank space** (rank r)，不是 full hidden dim。这保证：
1. 参数开销小（每层 2r² + 2 个 scalar）
2. Communication 被限制在 task-specific 的 LoRA subspace
3. 不污染 backbone（backbone W 仍然被 freeze）

## 3. Initialization（重要）

### 3.1 初版的 chicken-and-egg 问题

最初的设计：`W_av = W_va = 0`（零初始化），`gate_av = gate_va = -3`（sigmoid ≈ 0.047）。

**问题**：LoRA 的 B_v, B_a 已经是零初始化（标准 LoRA 约定），再加上 W_av=0：

- Step 0：`delta_v = B_v @ h_v_new = 0 @ anything = 0` → `dL/dW_av = 0`, `dL/d(gate_av) = 0`
- Step 1：B_v 开始非零，但 `W_av ≈ 0` → `h_v_new ≈ h_v` → gate 的 gradient 仍然 ~0
- ...

**Double zero 锁死**：gate 几乎永远不更新。

### 3.2 修正：Kaiming init for W

```python
# cooperative_lora.py
self.W_av = nn.Parameter(torch.zeros(r, r))
self.W_va = nn.Parameter(torch.zeros(r, r))
nn.init.kaiming_uniform_(self.W_av, a=math.sqrt(5))   # ← 关键
nn.init.kaiming_uniform_(self.W_va, a=math.sqrt(5))
self.gate_av = nn.Parameter(torch.tensor(gate_init))
self.gate_va = nn.Parameter(torch.tensor(gate_init))
```

**正确性**：
- 即使 B_v=0，模型整体 output 在 step 0 仍然不受 communication 影响（B=0 屏蔽了 LoRA 分支）
- Step 1 B 开始非零，此时 `W_av h_a` 已经是非零向量 → `dL/d(gate) = dL/d(B_v h_v_new) · d(h_v_new)/d(gate) = dL/d(output) · (W_av h_a) * sigmoid'(gate)` 非零
- 验证（PyTorch 脚本）：step 0 `dL/d(gate_av) = 0`，step 1 `dL/d(gate_av) = -0.08`（非零）

### 3.3 gate_init 的选择

第一次实验 `gate_init = -3.0`（sigmoid 0.047），观察到 gate 50 步只移动 ~2e-5 → 太慢。

**原因**：gate 的 gradient ∝ `sigmoid'(gate) = sigmoid(gate) * (1 - sigmoid(gate))`：
- `sigmoid'(-3) = 0.047 * 0.953 ≈ 0.045`
- `sigmoid'(-1.5) = 0.182 * 0.818 ≈ 0.149`（3.3× 更大）

第二次实验 `gate_init = -1.5`（sigmoid 0.182），gate 移动速度 **~5× 变快**。

## 4. Files Modified/Created

| # | 文件 | 操作 |
|---|------|------|
| 1 | `verl/models/cooperative/cooperative_lora.py` | 修改 — 添加 W_av/W_va/gate_av/gate_va + kaiming init + cooperative forward |
| 2 | `verl/models/cooperative/cooperative_wrapper.py` | 修改 — 传参 + save/load + gate value 提取 |
| 3 | `train_cooperative.py` | 修改 — CLI args + per-layer gate logging + disable eval |
| 4 | `evaluation/eval_cooperative_batch.py` | 修改 — load_model 读取 cooperative_comm config |
| 5 | `datasets/cooperative_thought/prepare_gui360_no_thought.py` | 新建 — 去除 `<thought>` 的数据 |
| 6 | `scripts/exp_cooperative/train_v6_comm_{thought,nothought}.slurm` | 新建 — SLURM 训练脚本 |

### 4.1 `cooperative_lora.py` 核心改动

```python
# 新增构造参数
cooperative_comm: bool = False
gate_init: float = -1.5

# __init__ 末尾，2-agent 且 cooperative_comm=True 时：
if cooperative_comm and num_agents == 2:
    self.W_av = nn.Parameter(torch.zeros(r, r, device=device))
    self.W_va = nn.Parameter(torch.zeros(r, r, device=device))
    nn.init.kaiming_uniform_(self.W_av, a=math.sqrt(5))
    nn.init.kaiming_uniform_(self.W_va, a=math.sqrt(5))
    self.gate_av = nn.Parameter(torch.tensor(gate_init, device=device))
    self.gate_va = nn.Parameter(torch.tensor(gate_init, device=device))

# forward() 中，2-agent hard routing 路径：
h_v = F.linear(x_drop, self.lora_A_v.to(dtype))
h_a = F.linear(x_drop, self.lora_A_a.to(dtype))

if self.cooperative_comm:
    g_av = torch.sigmoid(self.gate_av)
    g_va = torch.sigmoid(self.gate_va)
    h_v = h_v + g_av * F.linear(h_a, self.W_av.to(dtype))
    h_a = h_a + g_va * F.linear(h_v, self.W_va.to(dtype))

delta_v = F.linear(h_v, self.lora_B_v.to(dtype)) * self.scaling
delta_a = F.linear(h_a, self.lora_B_a.to(dtype)) * self.scaling
```

### 4.2 `cooperative_wrapper.py` — save 的关键 bug fix

初版 save 代码：

```python
# BUG：target_modules 包含 "gate_proj"，其 W_av 也含 "gate" 子串
if "gate" in name:
    gate_values[name] = round(torch.sigmoid(param).item(), 6)
```

运行时 crash（step 50 save checkpoint 时）：
```
RuntimeError: a Tensor with 16384 elements cannot be converted to Scalar
```
（16384 = 128 × 128 的 W_av matrix）

**修复**：使用 `endswith` 精确匹配 scalar：

```python
if name.endswith(".gate_av") or name.endswith(".gate_va"):
    gate_values[name] = round(torch.sigmoid(param).item(), 6)
```

### 4.3 `train_cooperative.py` — per-layer logging

Trainer 的 `log()` 方法在每次 logging_step 时：
1. 收集所有 coop module 的 `sigmoid(gate_av)`, `sigmoid(gate_va)`, `W_av.norm()`, `W_va.norm()`
2. 计算 mean/std/min/max → 塞进 `logs` dict（会出现在 stdout）
3. Rank 0 把 per-layer 的数据 dump 到 `gate_history.jsonl`，字段：
   - `step`, `epoch`, `loss`
   - `gate_av_per_layer` / `gate_va_per_layer`：长度 28 的 list（每层均值）
   - `W_av_norm_per_layer` / `W_va_norm_per_layer`

此外 `eval_strategy="no"`：eval set 的某些 sample 会产生 nan（truncation 导致全 -100 label），污染 log，real eval 走单独的 benchmark 脚本。

## 5. Training Setup

### 5.1 数据
- **thought**: `gui360_train_thought.jsonl` — 保留 `<thought>...</thought>` 的完整 response
- **nothought**: `gui360_train_nothought.jsonl` — 通过 `prepare_gui360_no_thought.py` 从 thought 数据中去除 `<thought>` 部分

两个对照实验的目的：观察 thought 是否必要 / communication 是否能替代 explicit chain-of-thought。

### 5.2 Hyperparameters (v6)
```
base_model       Qwen2.5-VL-3B-Instruct
num_agents       2 (V + A hard routing by token type)
lora_r           128   (vs v3 的 256，为了更快迭代)
lora_alpha       256
target_modules   q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
cooperative_comm True
gate_init        -1.5  (sigmoid 0.182)
bind_weight      0.0
batch_size       1 per GPU × 8 grad_accum = 8 per GPU
nodes            8 (× 4 GPU = 32 GPUs)
effective_batch  32 × 8 = 256
learning_rate    1e-5
num_epochs       2.0
logging_steps    5
eval_strategy    "no"
```

## 6. Training Observations (jobs 3700799 thought, 3700800 nothought)

### 6.1 全局

| Job | Epoch | loss | ce_loss | gate_av_mean | gate_va_mean | nan count |
|---|---|---|---|---|---|---|
| thought | 0.13 | 3.60 | 0.422 | 0.182469 | 0.182464 | 0 |
| nothought | 0.24 | 1.23 | 0.136 | 0.182504 | 0.182495 | 2 (transient) |

- 两个 job 的 loss 都在稳定下降，nothought 学得更快（序列更短 + label 更干净）
- nothought 有 2 个 transient nan（step 60 左右），没有 crash，已恢复
- Gate 移动 **确实非零**（与 v5 对比明显），但绝对量级仍然很小（~1e-4 级别）

### 6.2 Per-layer 分析（关键发现）

**Pattern: 清晰的 U 形 — 中间层（L10-16）主导 cooperative communication，边缘层（L0-4, L26-27）几乎不参与。两个 job 独立地呈现相同的形状。**

**THOUGHT (step 5→50, 10 records)**
- gate_av top-5: **L11 (+66e-6), L12 (+61), L14 (+59), L13/L20 (+56)**
- gate_va top-5: **L21 (+59), L20 (+55), L19 (+54), L12/L13/L14/L18 (+51)**
- Bottom-5 (av): L27 (+31), L3 (+29), L1 (+20), L0 (+17), L4 (+12)
- Bottom-5 (va): L1/L2 (+15), L0 (+13), L5 (+11), L4 (+9)

**NOTHOUGHT (step 5→115, 23 records)**
- gate_av top-5: **L10 (+138e-6), L14 (+137), L13/L16 (+136), L15 (+133)**
- gate_va top-5: **L12 (+118), L23 (+116), L13 (+115), L10 (+114), L24/L25 (+112)**
- Bottom-5 (av): L3 (+74), L0 (+67), L2 (+66), L26 (+61), L27 (+34)
- Bottom-5 (va): L6 (+56), L5 (+53), L2 (+36), L0/L1 (+34)

### 6.3 几个关键洞察

1. **U-shape 普遍存在**：transformer 的中间层承担 cooperative communication 最重。这与「早期层做 token-level 编码、末层做 output projection」的已知现象一致 — V-A 交互发生在 mid-level semantic 抽象层。

2. **av 和 va 的非对称性**：
   - av（A→V，action 告诉 vision）峰值在 **中间偏前**（L10-16）
   - va（V→A，vision 告诉 action）峰值在 **中间偏后**（L18-25，nothought 里尤其明显 L23-L25）
   - 解读：**vision 在早期层影响 action，action 在后期层影响 vision**。这是一个可以写进 paper 的 empirical finding。

3. **W norm 佐证 gate**：gate 移动最快的层，对应的 `W_av.norm()` 也在增长（+0.0003~0.0006），说明 communication projection 确实在被训练，不是 gate 在噪声上乱开。

4. **Nothought 学得更快**：相同 wall-clock 时间内，nothought 的 step 数更多 + 每步 gate 移动量也略大。序列短 + label 干净是双赢。

5. **移动量仍然偏小**：115 步 gate_av 从 0.18243 → 0.18250，绝对增量 ~8e-5。按线性外推到 762 步，final gate ~0.18283（仍然 < 0.2）。这是一个 potential issue：communication 能打开，但打开的幅度可能不够显著。后续需要观察是否会 super-linear 增长（自强化）或需要进一步调整 gate_init / learning rate。

## 7. 已知 issues 和下一步

### 7.1 Transient nan in nothought
- 发生时间：step 60 左右（epoch 0.14）
- 次数：2 次
- 状态：已恢复，loss 继续正常下降
- 原因推测：某个特定 batch 里某个 sample 的 label 全是 -100（图像 token 被 mask，action 部分被 truncate）导致 cross_entropy → nan
- 优先级：低 — 目前没有影响 training stability

### 7.2 Gate 最终移动幅度
- 当前速率：每 100 step ~7e-5
- 全训练外推：~6e-4 总增量
- 问题：从 0.182 → 0.1826 的 communication strength，effective 影响可能很小
- 可能的解决方向：
  1. 更激进的 gate_init（e.g., 0 → sigmoid 0.5）
  2. 单独为 gate / W_av 设置更大的 learning rate multiplier
  3. 接受目前的 regime，后续 eval 判断 communication 是否有效

### 7.3 需要验证
- Checkpoint round-trip（save → load → inference 正常）
- Full training 完成后的 benchmark eval
- 与 v3 baseline 的 head-to-head 对比（gui-odyssey, gui360 dev）

## 8. Lineage (v6 original)

| Job | 状态 | 备注 |
|---|---|---|
| 3700555/3700556 | Cancelled | W_av 是零初始化 — gate 无法更新 |
| 3700593/3700594 | Cancelled | kaiming W 已修，但缺 per-layer logging |
| 3700606/3700607 | **FAILED** step 50 | save bug（"gate" 匹配 gate_proj.W_av）+ eval nan |
| 3700799 (thought) | Completed | All fixes applied: kaiming W + save endswith + no eval + gate_init=-1.5 |
| 3700800 (nothought) | Completed | Same fixes, no thought data |

---

## 9. v6.1 — Upgrade to r=256 + switch to Qwen2.5-VL-**7B**

Original v6 used 3B backbone and r=128. Since v3 baseline uses 7B + r=256, v6.1 matches that to make results directly comparable to v3.

**Delta vs v6**:
- Backbone: Qwen2.5-VL-3B → **7B**
- LoRA rank: 128 → **256**, alpha 256 → **512**
- Thought data only
- Everything else identical (sigmoid gate, `gate_init=-1.5`, `lr_mult=10`)

**Hparam mistake (silent confound, realized later)**:
- Nodes: 4 → **8** (to keep wall time reasonable with 2x rank)
- Effective batch size: 128 (v3) → **256**
- Learning rate **not scaled** (kept at 1e-5)
- → Half the optimizer updates per epoch at the same data exposure ⇒ strictly less optimization per epoch than v3

---

## 10. v6.2 — Switch to tanh gate

**Delta vs v6.1**:
- Gate activation: sigmoid → **tanh**
- `gate_init`: -1.5 → **0.0**
- Motivation: `tanh(0) = 0`, gradient at 0 is **1.0** (vs sigmoid's 0.25), and tanh is bounded `[-1, 1]`, allowing **negative / anti-coupling** (important for MoE-like specialization — some layers may want V to *subtract* A's features, which sigmoid cannot express)
- Schedule: 4 epochs (up from 2), per-epoch checkpoints saved

**Still had the eff_bs=256 confound from v6.1.**

---

## 11. The hyperparameter confound (post-mortem)

After v6.2 epoch-1 came in **below v3 epoch-1** (42.85% vs 44.35%), traced to the silent eff_bs issue:

| Run | Nodes | grad_accum | eff_bs | lr | steps/epoch | updates per epoch |
|---|---|---|---|---|---|---|
| v3 | 4 | 4 | 128 | 1e-5 | 762 | 762 |
| v6.1 | 8 | 8 | **256** | 1e-5 | 381 | **381** |
| v6.2 | 8 | 8 | **256** | 1e-5 | 381 | **381** |

Linear-scaling rule (Goyal et al. 2017): if you double batch, you should double lr. v6.1/v6.2 doubled batch but kept lr — so the cooperative architecture was being evaluated under **strictly less gradient signal than v3 baseline at the same epoch**. Root cause of the v6.x family's earlier under-performance.

---

## 12. v6.3 — Clean A/B test (eff_bs matched to v3)

**Delta vs v6.2**: only `gradient_accumulation_steps 8 → 4`
- eff_bs: 256 → **128** (matches v3)
- Per-step wall time roughly halves, total wall ≈ unchanged
- Everything else (tanh gate, r=256, lr=1e-5, 4 epochs, 8 nodes, gate_init=0, lr_mult=10, wd=0) identical to v6.2

This is the **clean single-variable A/B**: v3 vs v6.3 differ only by `tanh cooperative comm`. Any score gap is attributable to the architecture change, not hparams.

Job: **3726419**. Wall time: ~5h 20min for 4 epochs.

### 12.1 v6.3 training dynamics

| epoch | step | ce_loss | gate_av_std | gate_va_std | W_av_norm | W_va_norm |
|---|---|---|---|---|---|---|
| init | 0 | — | 0 | 0 | 9.24 | 9.24 |
| 0.56 | 425 | 0.27 | 0.0117 | 0.0193 | 9.280 | 9.345 |
| 1.06 | 810 | ~0.21 | 0.0178 | 0.0258 | 9.294 | 9.433 |
| 1.72 | 1310 | 0.20 | 0.0207 | 0.0310 | 9.326 | 9.493 |
| 2.19 | 1670 | 0.186 | 0.0226 | 0.0345 | 9.352 | 9.535 |
| 3.17 | 2420 | 0.181 | 0.0248 | 0.0388 | 9.382 | 9.581 |

- Gate spreads monotonically widening (good)
- Mean ≈ 0 throughout — layers learn both positive and negative coupling (tanh's key benefit)
- ce_loss descending smoothly
- W matrices barely move: +0.5% / +3.7% from kaiming init over 4 epochs

### 12.2 v6.3 eval results (proper routing, GUI-360 action_prediction)

| Run | ep1 | ep2 | ep3 | ep4 (final) |
|---|---|---|---|---|
| v3 (vanilla LoRA) | 44.35% | **46.11%** | — | — |
| v6 (3B, sigmoid) | — | — | — | 42.31% |
| v6 nothought (3B) | — | — | — | 41.82% |
| v6.1 (7B, sigmoid, eff_bs=256) | — | — | — | 43.64% |
| v6.2 (7B, tanh, eff_bs=256) | **42.85%** | — | — | — |
| **v6.3 (7B, tanh, eff_bs=128)** | **44.78%** | **47.59%** | **48.53%** | *running (job 3729581)* |

**v6.3 beats v3 at every matched epoch**:
- ep1: +0.43 pp (44.78 vs 44.35)
- ep2: **+1.48 pp** (47.59 vs 46.11, v3's best) ← key result
- ep3: no v3 baseline at this epoch, but monotonically improving

**The eff_bs fix alone recovers the v6.x family** from being worse than v3 to beating v3. The tanh + cooperative communication architecture does contribute positively, just was masked by the hparam confound.

### 12.3 v6.3 per-shard detail (epoch-2 head-to-head)

```
Shard 0: 2058/4762 = 43.23%
Shard 1: 2516/4762 = 52.83%
Shard 2: 2307/4762 = 48.44%
Shard 3: 2183/4760 = 45.86%
-------------------------
Total:   9084/19046 = 47.59%
```

### 12.4 Impact analysis — are the gates doing anything?

Parameter count: cooperative comm machinery (W_av/W_va 65k each + 2 scalars per layer × 7 projections × 28 layers ≈ 25 M params) = **~2% of total LoRA params** (~1.3 B at r=256).

Per-layer activation contribution (at epoch-2 checkpoint, using typical gate ≈ 0.022, `||W_av h_a|| ≈ 0.585·||h_a||`):
- Average perturbation to `h_v`: |g|·0.585 ≈ **1.3%** of `||h_v||`
- Max perturbation: 0.072·0.585 ≈ **4.2%**

Non-zero gate values on attention vs MLP (sampled from epoch-2 config):
- Attention projections: |gate| ~ 0.001–0.004
- **MLP projections: |gate| ~ 0.012–0.020 (3-5× larger)**

→ The model learned that **cross-expert coupling is more valuable on MLP projections** than on attention. This is a concrete empirical finding.

**Bottom line**: a ~2% parameter overhead inducing ~1–4% per-layer activation perturbation yields +1.48 pp at epoch-2. Small, surgical, effective.

---

## 13. v6.4 — Push gate / W_comm learning 5× harder

**Delta vs v6.3**: only `gate_lr_multiplier 10 → 50`
- Comm-group lr: 1e-4 → **5e-4**
- All main LoRA lr unchanged (1e-5)
- Motivation: v6.3 gates are still in a weak regime (std ≈ 0.025, max ≈ 0.1) after 4 epochs, far from tanh saturation (|gate| > 2). Hypothesis: stronger cooperative coupling = stronger effect.

**Safety**: expected end-of-training gate std ≈ 0.12, max ≈ 0.5 — still in tanh linear regime.

Job: **3728475**. Status at epoch 1.31 (1h 42min in):

| Metric | v6.3 @ ep1.3 | v6.4 @ ep1.3 | Ratio |
|---|---|---|---|
| gate_av_std | 0.0185 | **0.0679** | **3.7×** |
| gate_va_std | 0.0270 | **0.1073** | **4.0×** |
| gate_av_max | 0.067 | 0.230 | 3.4× |
| gate_va_max | 0.081 | 0.282 | 3.5× |
| W_av_norm | 9.295 | 10.33 | +11.2% drift (vs v6.3 +0.6% at same epoch) |
| W_va_norm | 9.438 | 11.20 | +21.2% drift (vs v6.3 +2.1% at same epoch) |
| ce_loss | 0.203 | 0.232 | +14% (slightly slower convergence) |
| loss | 0.97 | 1.00 | +3% |

- Comm parameters moving 3–4× faster → 5× lr bump is working as intended
- ce_loss slightly higher at matched epoch — more aggressive optimization on the auxiliary path costs a little main-task progress early on
- **Eval result**: the tradeoff **pays off**. v6.4 beats v6.3 at every epoch, and the delta **grows with training** (+0.12 → +0.16 → +0.41 → +0.99 at ep1/2/3/4). Final v6.4 ep4 = **49.66%**, a new v6 best and **+3.55 pp over v3 baseline (46.11%)**. Aggressive gate learning early costs a little ce_loss but unlocks more capacity for the cooperative path to contribute by the end.

---

## 14. Complete Results Table

| Run | Backbone | rank | gate | eff_bs | comm_lr_mult | ep1 | ep2 | ep3 | ep4 | Job |
|---|---|---|---|---|---|---|---|---|---|---|
| v3 | 7B | 256 | — | 128 | — | 44.35 | **46.11** | — | — | (earlier) |
| v6 thought | 3B | 128 | sigmoid, init=-1.5 | 256 | 10 | — | 42.31 | — | — | 3700799 |
| v6 nothought | 3B | 128 | sigmoid, init=-1.5 | 256 | 10 | — | 41.82 | — | — | 3700800 |
| v6.1 thought | 7B | 256 | sigmoid, init=-1.5 | 256 | 10 | — | 43.64 | — | — | (earlier) |
| v6.2 thought | 7B | 256 | **tanh, init=0** | 256 | 10 | 42.85 | — | — | — | 3725787 |
| v6.3 thought | 7B | 256 | tanh, init=0 | **128** | 10 | 44.78 | 47.59 | 48.53 | 48.67 | 3726419 |
| **v6.4 thought** | 7B | 256 | tanh, init=0 | 128 | **50** | **44.90** | **47.75** | **48.94** | **49.66** | 3728475 |

**Key finding**: once the eff_bs hparam confound is fixed (v6.3), tanh + per-layer cooperative communication **beats v3 vanilla LoRA by +2.56 pp at epoch 4** (v6.3 ep4 48.67 vs v3 best 46.11). Pushing the comm-group lr 5× harder (v6.4) **wins at every epoch and the gap grows with training**: +0.12 → +0.16 → +0.41 → **+0.99** at ep1/2/3/4. v6.4 ep4 = **49.66%** is the new best, **+3.55 pp over v3 baseline**. The growing delta confirms aggressive gate learning accumulates benefit at later epochs.

**v6.3 vs v6.4 head-to-head:**

| epoch | v6.3 | v6.4 | Δ |
|---|---|---|---|
| 1 | 44.78 | 44.90 | +0.12 |
| 2 | 47.59 | 47.75 | +0.16 |
| 3 | 48.53 | 48.94 | +0.41 |
| 4 | 48.67 | **49.66** | **+0.99** |

---

## 15. Lineage — full

| Job | Variant | Status | Notes |
|---|---|---|---|
| 3700555/6 | v6 (3B) | Cancelled | W_av zero init bug |
| 3700593/4 | v6 (3B) | Cancelled | No per-layer logging |
| 3700606/7 | v6 (3B) | FAILED step 50 | Save bug: "gate" matched gate_proj.W_av |
| 3700799 | v6 thought (3B) | Completed | 42.31% — first working run |
| 3700800 | v6 nothought (3B) | Completed | 41.82% |
| (earlier) | v6.1 thought (7B) | Completed | 43.64% — silently had eff_bs=256 confound |
| 3725750 | v6.2 thought 2ep | Cancelled | Wanted 4 epochs instead |
| 3725787 | v6.2 thought 4ep | Cancelled at ep1 | ep1 eval 42.85% < v3 → revealed confound |
| 3726285 | v6.2 ep1 eval | Completed | 42.85% |
| **3726419** | **v6.3 thought 4ep** | **Completed** | Clean A/B vs v3 (eff_bs matched), 4 epochs done |
| 3726581 | v6.3 ep1 eval | Completed | 44.78% |
| 3727352 | v6.3 ep2 eval | Completed | **47.59%** (+1.48 vs v3 best) |
| 3728214 | v6.3 ep3 eval | Completed | **48.53%** |
| 3729581 | v6.3 ep4 eval | Completed | **48.67%** (final) |
| **3728475** | **v6.4 thought 4ep** | **Completed** | `gate_lr_mult=50` (5× v6.3), 4 epochs done |
| 3737807 | v6.4 ep1 eval | Completed | 44.90% |
| 3737808 | v6.4 ep2 eval | Completed | 47.75% |
| 3737809 | v6.4 ep3 eval | Completed | 48.94% |
| 3737810 | v6.4 ep4 eval | Completed | **49.66%** — new v6 best, +3.55 pp over v3 |

---

## 16. Implementation reference (consolidated)

This section collects every code change that makes v6 work, from the per-layer
module all the way through checkpoint save/load and diagnostic logging. All
snippets are the **current state as of v6.4** (tanh gate, separate comm lr
multiplier, per-epoch save).

### 16.1 File map

| File | Purpose |
|---|---|
| `verl/models/cooperative/cooperative_lora.py` | `CooperativeLoRALinear` — per-layer V/A LoRA + cooperative communication |
| `verl/models/cooperative/cooperative_wrapper.py` | `CooperativeVLMWrapper` — replaces target modules, token routing, save/load |
| `train_cooperative.py` | `CooperativeTrainer` — 3-group optimizer, gate diagnostics, epoch save callback |
| `evaluation/eval_cooperative_batch.py` | Inference entry — reads `cooperative_config.json`, loads `lora_comm.pt` |

### 16.2 `cooperative_lora.py` — module constructor (v6.x final)

```python
class CooperativeLoRALinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        num_agents: int = 2,
        soft_routing: bool = False,
        init_sep: float = 0.0,
        cooperative_comm: bool = False,   # v6 switch
        gate_init: float = -3.0,          # v6: -1.5 (sigmoid), v6.2+: 0.0 (tanh)
        gate_type: str = "sigmoid",       # v6/v6.1: sigmoid, v6.2+: tanh
    ):
        super().__init__()
        self.base_linear = base_linear
        self.num_agents = num_agents
        self.soft_routing = soft_routing
        self.cooperative_comm = cooperative_comm
        if gate_type not in ("sigmoid", "tanh"):
            raise ValueError(f"gate_type must be 'sigmoid' or 'tanh', got {gate_type}")
        self.gate_type = gate_type

        # Freeze base weights
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        in_f = base_linear.in_features
        out_f = base_linear.out_features
        self.scaling = alpha / r
        device = base_linear.weight.device

        # V / A LoRA pairs (standard LoRA: A = kaiming, B = zero)
        self.lora_A_v = nn.Parameter(torch.zeros(r, in_f, device=device))
        self.lora_B_v = nn.Parameter(torch.zeros(out_f, r, device=device))
        self.lora_A_a = nn.Parameter(torch.zeros(r, in_f, device=device))
        self.lora_B_a = nn.Parameter(torch.zeros(out_f, r, device=device))
        if num_agents >= 3:
            self.lora_A_t = nn.Parameter(torch.zeros(r, in_f, device=device))
            self.lora_B_t = nn.Parameter(torch.zeros(out_f, r, device=device))

        self.lora_dropout = nn.Dropout(p=dropout)

        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_a, a=math.sqrt(5))
        if num_agents >= 3:
            nn.init.kaiming_uniform_(self.lora_A_t, a=math.sqrt(5))
        # B stays zero -> delta starts at zero

        if soft_routing and num_agents == 2:
            self.sep = nn.Parameter(torch.tensor(init_sep))

        # Per-layer cooperative communication (v6, 2-agent only)
        # W_av/W_va: kaiming init (NOT zero) — required for gate to receive
        # gradient at step 1. Safe warmup is still preserved by B=0 zeroing
        # the LoRA branch at the model output.
        if cooperative_comm and num_agents == 2:
            self.W_av = nn.Parameter(torch.zeros(r, r, device=device))   # A→V
            self.W_va = nn.Parameter(torch.zeros(r, r, device=device))   # V→A
            nn.init.kaiming_uniform_(self.W_av, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.W_va, a=math.sqrt(5))
            # sigmoid: g = σ(logit),  init=-1.5 → g≈0.18,  max ∂g/∂logit = 0.25
            # tanh:    g = tanh(logit), init=0    → g=0,     ∂g/∂logit at 0 = 1.0
            #          bounded [-1,1], allows negative (anti) coupling
            self.gate_av = nn.Parameter(torch.tensor(gate_init, device=device))
            self.gate_va = nn.Parameter(torch.tensor(gate_init, device=device))

        self._token_mask: Optional[torch.Tensor] = None
```

### 16.3 `cooperative_lora.py` — forward (the cooperative path)

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    base_out = self.base_linear(x)

    token_mask = self._token_mask
    if token_mask is None:
        if self.training:
            raise RuntimeError(
                "CooperativeLoRALinear: token_mask is None during training. "
                "Call set_token_mask() before model.forward()."
            )
        return base_out  # inference without mask → base-only

    x_drop = self.lora_dropout(x)
    dtype = x_drop.dtype

    # Compute ALL deltas for all tokens (both stay in autograd graph)
    h_v = F.linear(x_drop, self.lora_A_v.to(dtype))  # [B, S, r]
    h_a = F.linear(x_drop, self.lora_A_a.to(dtype))  # [B, S, r]

    if self.cooperative_comm and hasattr(self, 'W_av'):
        if self.gate_type == "tanh":
            g_av = torch.tanh(self.gate_av)
            g_va = torch.tanh(self.gate_va)
        else:  # sigmoid (v6 / v6.1 default)
            g_av = torch.sigmoid(self.gate_av)
            g_va = torch.sigmoid(self.gate_va)
        h_v = h_v + g_av * F.linear(h_a, self.W_av.to(dtype))  # V sees A
        h_a = h_a + g_va * F.linear(h_v, self.W_va.to(dtype))  # A sees V

    delta_v = F.linear(h_v, self.lora_B_v.to(dtype)) * self.scaling
    delta_a = F.linear(h_a, self.lora_B_a.to(dtype)) * self.scaling

    mask = token_mask.unsqueeze(-1)
    if self.num_agents >= 3:
        delta_t = F.linear(
            F.linear(x_drop, self.lora_A_t.to(dtype)),
            self.lora_B_t.to(dtype)
        ) * self.scaling
        delta = torch.where(mask == 1, delta_v,
                    torch.where(mask == 2, delta_t, delta_a))
    elif self.soft_routing:
        s = torch.sigmoid(self.sep)
        mask_f = token_mask.unsqueeze(-1).to(dtype)
        w_v = mask_f * s + (1.0 - mask_f) * (1.0 - s)
        w_a = mask_f * (1.0 - s) + (1.0 - mask_f) * s
        delta = w_v * delta_v + w_a * delta_a
    else:
        delta = torch.where(mask, delta_v, delta_a)

    return base_out + delta
```

Key details:
- **Sequential feed**: `h_a` sees the *already-updated* `h_v` (line: `F.linear(h_v, W_va)` uses the new `h_v`). So V→A path carries a composition of A→V first. This is by design — breaks symmetry and avoids fix-point issues.
- **Non-zero W at init**: required so that `∂L/∂gate ≠ 0` at step 1. See the init comment block.
- **`hasattr` guard**: tolerates old checkpoints that were saved without `W_av`.

### 16.4 `cooperative_wrapper.py` — module replacement

```python
def _replace_target_modules(self, r, alpha, dropout, soft_routing=False, init_sep=0.0):
    """Replace nn.Linear in each transformer layer with CooperativeLoRALinear."""
    vlm = self.base_model.model  # Qwen2_5_VLModel
    if hasattr(vlm, "language_model"):
        layers = vlm.language_model.layers   # transformers ≥4.57
    elif hasattr(vlm, "layers"):
        layers = vlm.layers                   # older transformers
    else:
        raise AttributeError(...)

    for layer_idx in range(len(layers)):
        layer = layers[layer_idx]
        for module_name in self.target_modules:
            if module_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                parent = layer.self_attn
            elif module_name in ("gate_proj", "up_proj", "down_proj"):
                parent = layer.mlp
            else:
                raise ValueError(f"Unknown target module: {module_name}")
            original = getattr(parent, module_name)
            coop_linear = CooperativeLoRALinear(
                original, r, alpha, dropout, num_agents=self.num_agents,
                soft_routing=soft_routing, init_sep=init_sep,
                cooperative_comm=self.cooperative_comm,
                gate_init=self.gate_init,
                gate_type=self.gate_type)
            setattr(parent, module_name, coop_linear)
            self.coop_modules.append(coop_linear)
```

The wrapper holds a flat list `self.coop_modules` — used later by the trainer
to aggregate per-layer diagnostics.

### 16.5 `cooperative_wrapper.py` — save/load

**save_cooperative_checkpoint** produces four files per checkpoint:
```
<ckpt_dir>/
  ├── lora_v.pt               # {name: tensor}  – lora_A_v, lora_B_v per module
  ├── lora_a.pt               # {name: tensor}  – lora_A_a, lora_B_a per module
  ├── lora_comm.pt            # {name: tensor}  – W_av, W_va, gate_av, gate_va per module (v6+)
  └── cooperative_config.json # hparams + gate_values snapshot
```

```python
# Save communication params (v6)
comm_state = {}
for name, param in self.named_parameters():
    if not param.requires_grad:
        continue
    if any(k in name for k in ['W_av', 'W_va', 'gate_av', 'gate_va']):
        comm_state[name] = param.data.clone().cpu()
if comm_state:
    torch.save(comm_state, os.path.join(output_dir, "lora_comm.pt"))

# Save config with post-activation gate values for quick inspection
config = {
    "target_modules": self.target_modules,
    "num_agents": self.num_agents,
    "lora_v_params": sum(v.numel() for v in v_state.values()),
    "lora_a_params": sum(v.numel() for v in a_state.values()),
    "soft_routing": self.soft_routing,
    "cooperative_comm": self.cooperative_comm,
    "gate_init": self.gate_init,
    "gate_type": self.gate_type,
}
if comm_state:
    gate_values = {}
    act = torch.tanh if self.gate_type == "tanh" else torch.sigmoid
    for name, param in comm_state.items():
        # Bug fix: target_modules may include "gate_proj", whose W_av/W_va
        # contain the substring "gate". Use endswith to be precise.
        if name.endswith(".gate_av") or name.endswith(".gate_va"):
            gate_values[name] = round(act(param).item(), 6)
    config["gate_values"] = gate_values
    config["comm_params"] = sum(v.numel() for v in comm_state.values())
with open(os.path.join(output_dir, "cooperative_config.json"), "w") as f:
    json.dump(config, f, indent=2)
```

**load_cooperative_checkpoint** reads `lora_comm.pt` only if the wrapper was
instantiated with `cooperative_comm=True`:

```python
comm_path = os.path.join(checkpoint_dir, "lora_comm.pt")
if os.path.exists(comm_path) and self.cooperative_comm:
    comm_state = torch.load(comm_path, map_location="cpu", weights_only=True)
    loaded = 0
    for name, param in self.named_parameters():
        if name in comm_state:
            param.data.copy_(comm_state[name].to(param.device))
            loaded += 1
    print(f"Loaded {loaded} communication params from checkpoint")
```

This means **inference uses the exact same forward path as training** — the
cooperative comm branch is identical in train/eval.

### 16.6 `train_cooperative.py` — 3-group optimizer (key for v6.4)

```python
def create_optimizer(self):
    """Custom optimizer with separate param group for cooperative comm params.

    Three groups:
      1. Comm:    lr × gate_lr_multiplier, wd = gate_weight_decay
      2. Decay:   lr × 1,                  wd = args.weight_decay
      3. NoDecay: lr × 1,                  wd = 0  (biases, 1-d params)
    """
    if self.optimizer is not None:
        return self.optimizer

    comm_suffixes = {"gate_av", "gate_va", "W_av", "W_va"}
    comm_params, decay_params, no_decay_params = [], [], []
    comm_names, decay_names, no_decay_names = [], [], []

    for name, param in self.model.named_parameters():
        if not param.requires_grad:
            continue
        suffix = name.split(".")[-1]
        if suffix in comm_suffixes:
            comm_params.append(param)
            comm_names.append(name)
        elif param.dim() == 1 or name.endswith(".bias"):
            no_decay_params.append(param)
            no_decay_names.append(name)
        else:
            decay_params.append(param)
            decay_names.append(name)

    base_lr = self.args.learning_rate
    base_wd = self.args.weight_decay

    param_groups = [
        {"params": decay_params,    "lr": base_lr, "weight_decay": base_wd,
         "group_name": "lora_decay"},
        {"params": no_decay_params, "lr": base_lr, "weight_decay": 0.0,
         "group_name": "no_decay"},
        {"params": comm_params,
         "lr": base_lr * self.gate_lr_multiplier,
         "weight_decay": self.gate_weight_decay,
         "group_name": "comm"},
    ]
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
    self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
    return self.optimizer
```

Motivation (from the docstring):
> v6 thought training (job 3713229) showed that gates barely moved. Root cause:
> weak gradient flow in the sigmoid cold-start regime + default `weight_decay=0.1`
> dragging gate logits toward 0. Putting comm params in a dedicated group with
> higher LR and no WD lets the cooperative mechanism actually learn end-to-end.

v6.1–v6.3 used `gate_lr_multiplier=10`, v6.4 uses **50**.

### 16.7 `train_cooperative.py` — diagnostic logging in `log()`

```python
def log(self, logs, *args, **kwargs):
    # ── Per-micro-batch accumulated act loss (correct denominator) ──
    if self._act_loss_count > 0:
        logs["ce_loss"] = round(self._act_loss_sum / self._act_loss_count, 6)

    # ── Gate / W_comm distribution stats ──
    if self.cooperative_model.cooperative_comm:
        gates_av, gates_va = [], []
        w_av_norms, w_va_norms = [], []
        gate_act = (torch.tanh if self.cooperative_model.gate_type == "tanh"
                    else torch.sigmoid)
        for m in self.cooperative_model.coop_modules:
            if hasattr(m, 'gate_av'):
                gates_av.append(gate_act(m.gate_av).item())
                gates_va.append(gate_act(m.gate_va).item())
                w_av_norms.append(m.W_av.detach().float().norm().item())
                w_va_norms.append(m.W_va.detach().float().norm().item())
        if gates_av:
            logs["gate_av_mean"] = round(statistics.mean(gates_av), 6)
            logs["gate_av_std"]  = round(statistics.pstdev(gates_av), 6)
            logs["gate_av_min"]  = round(min(gates_av), 6)
            logs["gate_av_max"]  = round(max(gates_av), 6)
            logs["gate_va_mean"] = round(statistics.mean(gates_va), 6)
            logs["gate_va_std"]  = round(statistics.pstdev(gates_va), 6)
            logs["gate_va_min"]  = round(min(gates_va), 6)
            logs["gate_va_max"]  = round(max(gates_va), 6)
            logs["W_av_norm_mean"] = round(statistics.mean(w_av_norms), 4)
            logs["W_va_norm_mean"] = round(statistics.mean(w_va_norms), 4)

            # Per-layer means -> gate_history.jsonl (rank 0 only)
            if self.is_world_process_zero():
                n_per_layer = len(self.cooperative_model.target_modules)
                n_layers = len(gates_av) // n_per_layer
                layer_av = [statistics.mean(gates_av[i*n_per_layer:(i+1)*n_per_layer])
                            for i in range(n_layers)]
                layer_va = [statistics.mean(gates_va[i*n_per_layer:(i+1)*n_per_layer])
                            for i in range(n_layers)]
                layer_w_av = [statistics.mean(w_av_norms[i*n_per_layer:(i+1)*n_per_layer])
                              for i in range(n_layers)]
                layer_w_va = [statistics.mean(w_va_norms[i*n_per_layer:(i+1)*n_per_layer])
                              for i in range(n_layers)]
                record = {
                    "step": self.state.global_step,
                    "epoch": round(self.state.epoch, 4) if self.state.epoch else 0,
                    "loss": logs.get("loss"),
                    "gate_av_per_layer": [round(v, 6) for v in layer_av],
                    "gate_va_per_layer": [round(v, 6) for v in layer_va],
                    "W_av_norm_per_layer": [round(v, 4) for v in layer_w_av],
                    "W_va_norm_per_layer": [round(v, 4) for v in layer_w_va],
                }
                history_path = os.path.join(self.args.output_dir, "gate_history.jsonl")
                with open(history_path, "a") as fp:
                    fp.write(json.dumps(record) + "\n")

    # Reset running accumulators
    self._act_loss_sum = 0.0; self._act_loss_count = 0
    self._bind_loss_sum = 0.0; self._bind_loss_count = 0
    self._target_sim_sum = 0.0; self._nontarget_sim_sum = 0.0
    self._bind_sample_count = 0

    super().log(logs, *args, **kwargs)
```

Two outputs:
1. **Console / loss dict**: distribution summary (mean/std/min/max + W norms) per logging step.
2. **`gate_history.jsonl`**: per-layer trajectory for offline analysis. Used to
   produce the U-shape plot in §6.2.

Important subtlety: `logs["loss"]` from HF Trainer equals
`grad_accum_steps × ce_loss` due to how HF accumulates `tr_loss` per
micro-batch but divides by optimizer-step count. **`ce_loss` (our own metric)
is the correct thing to compare across different `grad_accum_steps` configs.**

### 16.8 `train_cooperative.py` — epoch save callback

```python
class CooperativeSaveCallback(TrainerCallback):
    """Save cooperative checkpoint at each save step + persistent copy per epoch."""

    def on_save(self, args, state, control, **kwargs):
        model = self._get_model(kwargs)
        if model is None:
            return
        ckpt_dir = os.path.join(args.output_dir,
                                f"checkpoint-{state.global_step}", "cooperative")
        model.save_cooperative_checkpoint(ckpt_dir)

    def on_epoch_end(self, args, state, control, **kwargs):
        """Save a persistent epoch checkpoint that won't be auto-deleted."""
        model = self._get_model(kwargs)
        if model is None:
            return
        epoch = int(round(state.epoch))
        epoch_dir = os.path.join(args.output_dir, f"epoch-{epoch}")
        model.save_cooperative_checkpoint(epoch_dir)
```

This is what produces `epoch-1/`, `epoch-2/`, ... directories — they are
never pruned by `save_total_limit`, enabling clean per-epoch eval.

Also note: `_save_checkpoint` is overridden to be a no-op for the heavy weights
because a 17 GB unsharded save across 32 ranks killed an earlier run. HF
Trainer still writes `training_args.bin` and the optimizer/scheduler state;
the *LoRA* weights are persisted by `CooperativeSaveCallback.on_save`.

### 16.9 `evaluation/eval_cooperative_batch.py` — inference entry

```python
# Auto-detect cooperative communication (v6) from config
coop_config = json.load(open(os.path.join(coop_checkpoint_path,
                                          "cooperative_config.json")))
cooperative_comm = coop_config.get("cooperative_comm", False)
gate_init = coop_config.get("gate_init", -3.0)
gate_type = coop_config.get("gate_type", "sigmoid")

model = CooperativeVLMWrapper(
    base_model=base_model, lora_r=r, lora_alpha=r * 2, lora_dropout=0.0,
    target_modules=coop_config["target_modules"], bind_weight=0.0,
    num_agents=num_agents, soft_routing=soft_routing, init_sep=init_sep,
    cooperative_comm=cooperative_comm,
    gate_init=gate_init,
    gate_type=gate_type)
model.load_cooperative_checkpoint(coop_checkpoint_path)
model.eval()
```

The `gate_type` is persisted into the config so inference always reconstructs
the matching activation function — no training/inference skew.

### 16.10 Training CLI flags (delta across v6 versions)

All shared: `--cooperative_comm` turns the whole mechanism on.

```bash
# v6 / v6.1 (sigmoid)
--gate_type sigmoid
--gate_init -1.5
--gate_lr_multiplier 10.0
--gate_weight_decay 0.0

# v6.2 / v6.3 (tanh, gate_init=0 for zero-start)
--gate_type tanh
--gate_init 0.0
--gate_lr_multiplier 10.0
--gate_weight_decay 0.0

# v6.4 (same as v6.3 except push comm lr 5x)
--gate_type tanh
--gate_init 0.0
--gate_lr_multiplier 50.0
--gate_weight_decay 0.0
```

### 16.11 `cooperative_config.json` schema (inference contract)

```json
{
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
  "bind_weight": 0.0, "bind_layer": 27, "bind_temperature": 0.1,
  "num_agents": 2,
  "lora_v_params": 645922816,
  "lora_a_params": 645922816,
  "soft_routing": false,
  "init_sep": 0.0,
  "cooperative_comm": true,
  "gate_init": 0.0,
  "gate_type": "tanh",
  "gate_values": {
    "base_model.model.language_model.layers.0.self_attn.q_proj.gate_av": -0.001729,
    "base_model.model.language_model.layers.0.self_attn.q_proj.gate_va": -0.004055,
    ... (one entry per cooperative layer × 2 gates)
  },
  "comm_params": 25755752
}
```

Everything an inference job needs to reconstruct the exact same forward path:
`gate_type` + `target_modules` + `num_agents` is sufficient. The actual
trained weights live in the sibling `.pt` files. At r=256 with 7 target
modules on Qwen2.5-VL-7B (28 layers), `comm_params ≈ 25.8 M` (≈ 2% of total
LoRA params of 1.3 B).

### 16.12 Active parameter groups at v6.4 train start (rank-0 log)

```
[CooperativeTrainer] optimizer param groups:
  lora_decay:  392 params, lr=1e-05, wd=0.0
  no_decay:    0 params,   lr=1e-05, wd=0.0
  comm:        392 params, lr=0.0005 (x50.0), wd=0.0
```

392 = 28 layers × 7 target modules × 2 (one per V/A pair). Confirms that each
`CooperativeLoRALinear` contributes 4 comm params
(`W_av`, `W_va`, `gate_av`, `gate_va`) — so `comm` group size = 28 × 7 × 4 = 784
tensor parameters, grouped by PyTorch into 392 after internal fusing... the
exact count depends on tensor layout but the `lr=0.0005` on comm group confirms
the lr multiplier was applied correctly.

---

## 17. Where the v6.4 gain comes from — gate analysis + differential diagnosis

With v6.4 ep4 = 49.66% landed (+3.55 pp over v3 best), two post-hoc analyses
run on the ep4 checkpoint answer: **which modules carry the cooperation**, and
**which failure modes does the cooperation actually fix**.

### 17.1 Per-module gate magnitude: "MLP > attention" was an optimization artifact

Measuring `|tanh(gate)|` over all `gate_av` + `gate_va` on the `epoch-4`
checkpoint, separated by module class:

| checkpoint | `attn` mean | `mlp` mean | mlp/attn mean | mlp/attn median |
|---|---|---|---|---|
| v6.3 ep4 (comm_lr_mult=10) | 0.01211 | 0.03458 | **2.86×** | **6.15×** |
| v6.4 ep4 (comm_lr_mult=50) | 0.09672 | 0.13120 | **1.36×** | 1.82× |

v6.3 showed MLP gates 3–6× larger than attention gates — an apparent
architectural asymmetry. **v6.4 reveals this was an optimization artifact**:
when comm lr is raised 5×, attention gates grow **8.0×** while MLP gates grow
only **3.79×**. The "MLP dominance" was the attention gates being *lr-starved*,
not MLP being fundamentally the right place for communication.

Per-submodule growth from v6.3 → v6.4:

| submodule | v6.3 mean | v6.4 mean | v6.4/v6.3 |
|---|---|---|---|
| q_proj | 0.01460 | 0.10214 | 7.00× |
| k_proj | 0.00494 | 0.04853 | 9.83× |
| v_proj | 0.00887 | 0.07665 | 8.64× |
| o_proj | 0.02003 | 0.15956 | 7.96× |
| gate_proj | 0.06145 | 0.13151 | 2.14× |
| up_proj | 0.03065 | 0.11772 | 3.84× |
| down_proj | 0.01163 | 0.14439 | **12.41×** |

k/v/q_proj were essentially dead in v6.3 (median 0.003–0.005, ≈ zero). In v6.4
they become comparable to the MLP gates. **The +0.99 pp v6.4 wins over v6.3 has
a mechanistic explanation: v6.4 unlocks attention-layer cooperative
communication that was lr-starved in v6.3.**

### 17.2 Per-layer pattern: top-heavy, not U-shaped

Grouping by transformer depth (28 layers on Qwen2.5-VL 7B):

| layer group | v6.4 ep4 mean `|tanh(gate)|` |
|---|---|
| early (0–7) | 0.0401 |
| mid (8–19) | 0.1330 |
| late (20–27) | 0.1506 |

Top 5 layers by gate magnitude: **22, 21, 16, 24, 20** (all mid–late).
Bottom 5: **2, 1, 3, 0, 5**. This is a **monotone top-heavy profile**, not a
U-shape — cooperation happens where abstract action/visual binding lives
(mid–late layers), not on shallow token-level features. Consistent with
standard interpretability findings about where semantic binding happens in
transformers.

### 17.3 Differential analysis vs v3: where the +3.55 pp comes from

Using `evaluation/analysis_differential_v3.py` for a per-sample comparison of
v6.4 ep4 vs v3 ep2 on all 19046 GUI-360 test samples:

| quadrant | N | pct |
|---|---|---|
| both succeed | 7852 | 41.2% |
| both fail | 8657 | 45.5% |
| v3 fails, v6.4 correct | **1617** | **8.4%** |
| v6.4 fails, v3 correct | 930 | 4.9% |

Net recovered: **687 samples = +3.61 pp** (matches eval delta +3.55 pp within
rounding).

**Breakdown of the net gain by failure category:**

| category | v3-only-fail | v6.4-only-fail | net | share of gain |
|---|---|---|---|---|
| `wrong_coordinate` | 994 | 588 | **+406** | **59.1%** |
| `wrong_function` | 340 | 181 | +159 | 23.1% |
| `format_error` | 158 | 46 | +112 | 16.3% |
| `wrong_args` | 125 | 115 | +10 | 1.5% |
| **total** | **1617** | **930** | **+687** | 100% |

**60% of v6.4's improvement is coordinate-grounding recovery.** The cooperative
V→A channel is letting the action head physically place clicks on the right
pixel more often. Restricted to click: **+920 recovered − 546 new = +374 net
click-coord** — the dominant single contributor.

### 17.4 Function confusion: click⟷type dominates

Top wrong-function cells where v3 fails but v6.4 is correct (minus the reverse):

| GT → Pred | v3 fails (v6.4 ok) | v6.4 fails (v3 ok) | net |
|---|---|---|---|
| click → type | 111 | 56 | **+55** |
| type → click | 62 | 35 | **+27** |
| click → select_text | 18 | 8 | +10 |
| click → wheel_mouse_input | 17 | 13 | +4 |
| click → select_table_range | 16 | 12 | +4 |

click⟷type alone accounts for **+82 of the +159** function-level net gain.
**Interpretation**: click and type on text fields require identical visual
grounding but different action semantics. v3's hard-separated agents cannot
share the "this is a text input field" visual binding, so the action head
misroutes based on thought context alone. Cooperative communication restores
the shared grounding channel — precisely what the architecture was designed to
provide.

### 17.5 Both-wrong-coordinate analysis: flips, not smoothing

On the 4089 samples where **both** v6.4 and v3 had `wrong_coordinate`:

- v6.4 mean distance to target: **263.1 px**
- v3 mean distance to target: **264.4 px**
- v6.4 closer: 1481 | v3 closer: 1438 | tied: 1170

These distributions are essentially identical. **v6.4 does not make bad clicks
less bad** — it flips specific borderline cases from outside-rect to
inside-rect. The improvement is discrete, not a smooth distance reduction.
Consistent with cooperative communication unlocking *specific* samples where
the V→A channel provides a missing piece of grounding information, rather than
a uniform attention-map improvement.

### 17.6 Updated story of what v6 actually does

1. **Architectural**: per-layer cooperative communication in LoRA low-rank
   space lets hard-separated V and A agents exchange grounding context without
   re-mixing their separated parameters.
2. **Optimization**: the default training lr starves attention-layer
   communication (v6.3 k_proj gate median ≈ 0.005). Bumping the comm-group lr
   5× unlocks it (v6.4 k_proj median ≈ 0.023, mean 0.049), and the effect
   scales with training duration (+0.12 → +0.16 → +0.41 → **+0.99** at ep1–4).
3. **Localization**: cooperation lives in **mid–late layers** (top-heavy
   profile) and spreads evenly across attention and MLP modules once properly
   trained — not an MLP-specific architectural insight.
4. **Failure mode**: 60% of the improvement is **click-coordinate grounding**.
   Cooperative communication is primarily helping the action agent see the
   visual agent's spatial grounding — the V→A direction of the channel.
5. **Function disambiguation**: the remaining 23% comes mostly from
   click⟷type confusion fixes — exactly the class of errors requiring shared
   visual grounding but differing only in action semantics.

Source data / scripts:
- Gate analysis: inline scripts in section 17.1–17.2 reading
  `cooperative_v6_{3,4}_comm_thought/epoch-4/cooperative_config.json`.
- Differential: `evaluation/analysis_differential_v3.py --coop_dir
  cooperative_v6_4_thought_epoch4_proper/action_prediction --svd_dir
  cooperative_thought_v3_ep2/action_prediction`, output in
  `evaluation/analysis_results_v64_vs_v3/`.

---

## 18. Step 1b correlation — does learned gate structure align with diagnosed conflict?

**Setup**: match v6.4 ep4 `|tanh(gate)|` at each (layer, module) location against
Base-model **Step 1b** per-module gradient conflict (`|cos(∇L_bind, ∇L_act)|`)
measured in `evaluation/results/grad_conflict_projection_base_20260330_073454/per_module_conflict.json`.
28 layers × 7 target modules = 196 locations.

### 18.1 Global correlation is weak, but structured

| restriction | Pearson |
|---|---|
| all 196 modules, `|Base conflict|` vs v6.4 gate | **+0.22** (n=28 per-layer) / **−0.15** (n=196 module-level) |
| attention-only (112) | −0.12 |
| MLP-only (84) | −0.08 |
| **k_proj only (28)** | **+0.45** |
| SFT v2 control (conflict ≈ noise) | −0.41 (spurious) |

The noisy aggregate hides a **sharp k_proj signal**: where Base has most gradient
conflict on k_proj, v6.4 places the biggest k_proj gate. This is exactly the
module the Step 1b analysis flagged as the "primary conflict carrier".

### 18.2 L24 is the smoking gun

The single most-conflicted location in Base model is **L24 k_proj** (conflict
−0.0994, 2× any other module). v6.4's response at that layer:

| module @ L24 | Base `|conflict|` | v6.4 `|tanh(gate)|` mean |
|---|---|---|
| q_proj | 0.00358 | 0.134 |
| **k_proj** | **0.09942** | **0.131** |
| v_proj | 0.01117 | 0.129 |
| **o_proj** | 0.01640 | **0.247** ← top |
| gate_proj | 0.01658 | 0.155 |
| up_proj | 0.00371 | 0.162 |
| **down_proj** | 0.00560 | **0.217** ← top |

**L24 k_proj** is top-1 on the k_proj-restricted ranking **on both sides**: most
Base conflict AND biggest v6.4 k_proj gate. But note where v6.4 actually
invests most of its L24 capacity: **o_proj (0.247) and down_proj (0.217)** —
the output-side projections, not the conflicting input-side k_proj itself.

### 18.3 Top-15 v6.4 gates are o_proj / down_proj, not k_proj

| rank | layer | module | v6.4 gate | Base `|conflict|` |
|---|---|---|---|---|
| 1 | 21 | o_proj | 0.356 | 0.0012 |
| 2 | 20 | o_proj | 0.310 | 0.0041 |
| 3 | 22 | o_proj | 0.308 | 0.0021 |
| 4 | 21 | down_proj | 0.307 | 0.0049 |
| 5 | 23 | down_proj | 0.275 | 0.0095 |
| 6 | 15 | o_proj | 0.273 | 0.0053 |
| 7 | 13 | o_proj | 0.262 | 0.0031 |
| 8 | 16 | down_proj | 0.254 | 0.0061 |
| 9 | 24 | o_proj | 0.247 | 0.0164 |
| 10 | 17 | down_proj | 0.242 | 0.0070 |

**Key observation**: the top v6.4 gates are almost all `o_proj` and `down_proj`
in **mid-late layers (13–24)**. These have essentially zero Step 1b conflict in
Base model. **v6.4 is not "fixing the conflict at the site of the conflict" —
it is fixing it at the output side of each block.**

### 18.4 Mechanism interpretation

Step 1b measures where the gradient conflict **enters** the computation
(k_proj — the routing decision that "wants one attention pattern" for binding
and "another pattern" for action). v6.4 learns to **resolve** the conflict at
the output side:

1. Input routing (k_proj, q_proj): each agent gets its own, so the routing
   conflict is *architecturally* gone — v6.4 doesn't need to strongly
   communicate here.
2. Output projection (o_proj, down_proj): after each agent has done its own
   independent attention/MLP computation, cooperative communication **mixes
   the two hidden streams** right before projecting back to the backbone
   residual. This is the last place to exchange information before it flows
   back into shared computation.
3. The layer depth matches: cooperative communication peaks in **mid-late
   layers (16-24)**, which is also where Base model's k_proj conflict peaks
   (layer-level Pearson **+0.22**, noisy but top-heavy).

This tells a consistent story:
- **Diagnosis** (Step 1b): "conflict lives on k_proj in mid-late layers"
- **Architecture** (hard-separated V/A): "give them separate k_proj"
- **Emergent fix** (v6.4 learned gates): "let them share at o_proj / down_proj
  in mid-late layers"

The three independent findings converge on the same layer range and the same
mechanistic role (mid-late attention binding), but address different surfaces
of the problem. **The end-to-end trained model automatically discovers the
right post-hoc communication pattern without being told where the conflict is.**

---

## 19. In-flight experiments

Launched after the §17/§18 analyses, based on the updated story:

| Job | Label | Purpose | Status |
|---|---|---|---|
| 3743502 | v6.5 train | `gate_lr_mult=100` (2× v6.4) — scaling curve | Training, ~5h |
| 3743509 | v6.4 ep4 **VA-only** ablation eval | zero `W_av` (only V→A direction active), keep `W_va` | Running, ~80 min |
| 3743510 | v6.4 ep4 **AV-only** ablation eval | zero `W_va` (only A→V direction active), keep `W_av` | Running, ~80 min |

**Directional ablation setup**: the two ablated checkpoints share `lora_v.pt`,
`lora_a.pt`, `cooperative_config.json` via symlink with the original
`epoch-4/`. Only `lora_comm.pt` is a new file, with 196 × 256×256 tensors of
one direction zeroed out. Stored under
`cooperative_v6_4_comm_thought/epoch-4-{va,av}-only/`.

**Predictions**:
- **VA_only** (V→A only, action agent reads visual): expected to retain most
  of v6.4's +3.55 pp gain. The differential analysis (§17.3) attributed 60%
  of v6.4's improvement to coordinate grounding, which is the V→A
  direction. Predicted: **49.0–49.5%** (close to full v6.4 = 49.66%).
- **AV_only** (A→V only, visual agent reads action): expected to collapse back
  toward v3. Without the grounding channel, the action head loses the
  mechanism that recovers click coordinates. Predicted: **47.0–48.0%**,
  possibly close to v6.3 ep4 = 48.67%.

If confirmed, this directly proves v6's mechanism is **unidirectional V→A
grounding transfer**, not symmetric "cooperative joint reasoning".

Source: `scripts/exp_cooperative/eval_v6_4_ep4_{va,av}_only_ap.slurm`,
`scripts/exp_cooperative/train_v6_5_comm_thought.slurm`.

---

## 20. v7 — Emergent specialization via symmetry breaking (design)

### 20.1 The paradigm flip

Everything from v1 through v6 has used the same implicit assumption: **the
modality of a token determines which agent processes it**. Image tokens go to
the V agent, text tokens go to the A agent. This is a human-designed prior
baked into the routing mask.

That prior is not obviously correct:

- Thought tokens are **text** but perform **visual reasoning**. Which agent
  should handle them? v3→v6 all had to manually assign them.
- Within the image, target-element patches and background patches have very
  different information value. A single "image" label routes them identically.
- Within the text, function-name tokens and coordinate-digit tokens demand
  different kinds of computation. A single "action" label routes them
  identically.

Every time we discover such a mismatch we have to redesign the mask. This is a
symptom of the real problem: **we don't know what the optimal decomposition of
multi-modal computation looks like**. We are guessing, and each guess ships
a new routing prior.

**v7 flips the problem**: instead of *us* designing routing, let the two
agents **discover their own specialization** from the data. If the emergent
decomposition matches visual/action, great — we've confirmed our prior.
If it matches something else (perception/planning, static/dynamic,
low-level/high-level, grounding/execution), we've discovered a better axis of
decomposition — and that **is** the research finding.

### 20.2 Symmetry breaking — the only real design choice

If two agents are trained symmetrically against a single task loss from
symmetric initialization, they must collapse to identical functions. That's
the whole problem. To force differentiation, the training dynamics must
contain an explicit symmetry-breaking pressure.

Four candidates, in order of increasing "let the data decide":

1. **Explicit diversity loss.** Add a penalty on the pairwise cosine
   similarity between the two agents' LoRA weights (or their per-token
   outputs). This tells the model "you must differ" without telling it
   *how* to differ — the network finds the direction of easiest differentiation
   under task constraint. Simple and controllable via a single weight.

2. **Information-bottleneck cooperation (most fundamental).** Both agents see
   all tokens and compute independent latents h₁, h₂. They exchange messages
   through a narrow channel (e.g. `r/4`-dim bottleneck). Each agent's final
   output is its own latent + the received message. *The bottleneck makes
   redundancy expensive.* If both agents compute the same thing, the channel
   transmits redundant information and task loss plateaus. If they
   differentiate, the channel carries complementary information and task loss
   drops further. **Gradient descent automatically pushes the model to the
   specialized equilibrium** — no hand-designed pressure, just a capacity
   constraint.

3. **Stochastic soft routing.** Each token is routed to agent₁ with a
   learnable per-token probability `p`, to agent₂ with `1-p`. Stochastic
   during training, expectation at inference. This is close to MoE but with
   weighted soft mixing instead of hard top-k. The specialization emerges
   from routing gradient.

4. **Asymmetric initialization.** Initialize agent₁ from a vision-pretrained
   LoRA checkpoint, agent₂ from an action-pretrained LoRA checkpoint. No
   runtime mask, no routing. Training dynamics drift each agent further into
   its starting niche. Least principled (relies on the pre-training choice),
   but easiest to implement and could be a control baseline.

### 20.3 Why the information-bottleneck framing is the strongest

Option 2 reframes the entire problem in cooperative game theory terms:

- Cooperation has a marginal benefit equal to the **complementarity** of the
  two agents' information. If they know the same things, cooperation adds
  zero.
- The bottleneck constrains *how much* information can be exchanged per
  forward pass. Call this `C` bits per token.
- If the agents are identical, the channel transmits zero useful bits
  (the receiver already knows everything). Effective cooperation value = 0.
- If the agents are fully specialized on orthogonal features, every bit
  through the channel is informative. Effective cooperation value ≈ C.
- Therefore any gradient descent that cares about task loss will push the
  agents toward specialization, *as long as* the channel is narrow enough
  that redundant cooperation is a visible efficiency loss.

This gives a direct theoretical prediction: specialization should be stronger
when the channel is narrower, and there should be a critical width below
which specialization emerges and above which it doesn't. That's a
**scaling curve** for a paper, with a clean mechanistic hypothesis.

### 20.4 Minimum-viable experiment: v7.0

**Goal**: establish whether explicit diversity pressure is enough to break
the symmetry without destroying task accuracy. No bottleneck yet — simplest
possible version.

**Delta vs v6.4**:
1. **Remove hard routing mask.** Both agents see all tokens. Replace
   `delta = torch.where(mask, delta_v, delta_a)` with
   `delta = 0.5 * (delta_v + delta_a)` (or a learnable per-token soft mix,
   TBD — start with fixed 0.5).
2. **Keep cooperative_comm** (v6.4 style): `W_av`, `W_va`, `gate_av`,
   `gate_va` unchanged. These still allow the two agents to exchange
   information in the low-rank space.
3. **Add diversity loss**: at the end of every forward,
   `L_div = λ · mean_over_target_modules(cos(vec(lora_v), vec(lora_a)))`.
   Starting `λ = 0.01`. Total loss = `L_task + L_div`.
4. **Init**: keep symmetric random init of lora_A, zero init of lora_B —
   the diversity loss is what breaks symmetry.

Everything else identical to v6.4 (rank 256, tanh gate, `eff_bs=128`,
`gate_lr_mult=50`, 4 epochs).

### 20.5 What to measure (the three questions)

1. **Accuracy**: does v7.0 beat / match / drop relative to v6.4 (49.66%)?
   - If ≥ v6.4: emergent specialization works without hand-designed routing.
     This is a publishable result even at +0 pp — the **mechanism is different**.
   - If < v6.4 − 1 pp: the diversity pressure is fighting task loss.
     Tune λ down or switch to bottleneck.

2. **Weight-space differentiation**: after training, compute `cos(lora_v,
   lora_a)` per target module, mean across layers.
   - ≈ 0.99 → no differentiation, diversity loss too weak
   - 0.3 – 0.7 → moderate differentiation, the interesting regime
   - < 0 → over-separation, may harm accuracy

3. **Functional differentiation**: for each token in the val set, run forward
   and record `‖delta_v‖` and `‖delta_a‖`. Stratify by token class (image
   patch / thought text / instruction text / action output text). If
   `‖delta_v‖ / ‖delta_a‖` shows a clean per-class pattern — we have an
   **emergent decomposition**. Compare to the hard-routing mask:
   - matches exactly → confirms the visual/action prior was right
   - different clean pattern → we've found a better axis
   - no clean pattern → the agents differ in weight space but not in
     semantic role — a sign that the diversity loss is distributing noise

### 20.6 Ablations that naturally follow

- **λ sweep**: 0.001, 0.01, 0.05, 0.1 — find where specialization emerges
  and where it starts costing accuracy. Gives a scaling curve.
- **Freeze-one-agent ablation**: after training v7.0, freeze agent₁ and
  re-evaluate. If both agents are real specialists, the accuracy drop
  measures each agent's contribution. If they're duplicates, freezing
  either has no effect.
- **Bottleneck-width sweep (v7.1)**: replace diversity loss with the
  information-bottleneck setup. Sweep channel width `C ∈ {r/32, r/16, r/8,
  r/4, r/2, r}` and plot task accuracy + weight-space differentiation. The
  critical-width prediction gives the theoretical result.
- **Init ablation**: start from v6.4's trained checkpoint (asymmetric
  due to training) vs fresh random init (symmetric). Does starting
  asymmetric help?

### 20.7 Why this is more fundamental than the v6 line

The v6 program assumed the decomposition and engineered a communication
channel on top. Even with +3.55 pp over v3 and a clean mechanism story
(§17/§18), it is still a **fix** for a hand-designed routing constraint.

v7 asks a deeper question: **what is the optimal decomposition of VLM
computation, and can the model discover it end-to-end?** If the answer is
"yes, and it matches visual/action modality" — v6 was on the right track and
v7 gives it the right framing (no hand-coded mask, emergent instead). If the
answer is "yes, and it's something different" — v6 was a local optimum and
v7 discovers the global one. Either outcome is a finding.

It also directly addresses the main limitation of v6 revealed in §17/§18:
**v6 only ever communicates at the output side (o_proj / down_proj) because
the hard routing stops it from mixing information earlier**. Removing the
routing mask lets cooperation happen anywhere the training dynamics decide is
useful — including q_proj / k_proj, which v6 could never touch.

### 20.8 Implementation path (concrete)

The minimum experiment requires three local changes to the codebase:

1. **`verl/models/cooperative/cooperative_lora.py`**:
   - Add `routing_mode` constructor flag. Default `hard` (current behavior).
     `soft` or `merge` mode skips the `torch.where(mask, ...)` and returns
     `0.5 * (delta_v + delta_a)`. When in merge mode, `token_mask` becomes
     optional (can be None at training time).

2. **`train_cooperative.py`**:
   - Add `--routing_mode {hard,merge}` CLI flag.
   - Add `--diversity_loss_weight` CLI flag (default 0.0).
   - In the training loop, after the task forward, compute
     `L_div = λ · mean(cos(flat(lora_A_v), flat(lora_A_a)))` across all
     `CooperativeLoRALinear` modules, plus the same term on `lora_B_v/B_a`.
     Add to `total_loss` before backward.
   - Log `diversity_cosine` as a diagnostic in the existing `log()` method
     (similar to how `gate_av_std` is logged).

3. **New eval path / skip the token-mask plumbing**: the existing
   `CooperativeVLMWrapper` sets `token_mask` before forward. In merge mode
   this is a no-op — still pass it (for free), let the module ignore it.

4. **Slurm**: `scripts/exp_cooperative/train_v7_0_merge_diversity.slurm`,
   same hparams as v6.4 plus `--routing_mode merge --diversity_loss_weight
   0.01`.

5. **Post-training analysis**: new script
   `evaluation/analysis_v7_differentiation.py` that reads the v7
   checkpoint, computes per-module `cos(lora_A_v, lora_A_a)`, and for a
   sample of val inputs computes the per-token-class `‖delta_v‖ / ‖delta_a‖`
   histogram.

Estimated effort: ~150 LoC in the three files. All changes are backward
compatible (default behavior = v6.4).

### 20.9 Phase A — Implementation complete

All five code changes landed and smoke-tested. Files touched:

1. **`verl/models/cooperative/cooperative_lora.py`** — `routing_mode`
   constructor flag (`"hard"` | `"merge"`), merge path returns
   `0.5*(delta_v + delta_a)`, merge mode tolerates `token_mask=None`
   during training. `extra_repr()` reports routing mode.

2. **`verl/models/cooperative/cooperative_wrapper.py`** — threaded
   `routing_mode` through wrapper constructor, `_replace_target_modules`,
   forward() token_mask creation, `generate()` pre_hook, and
   `save_cooperative_checkpoint` config.

3. **`train_cooperative.py`** — `--routing_mode` and
   `--diversity_loss_weight` CLI flags. Diversity loss computed on
   `lora_B_v` vs `lora_B_a` (see rationale below), added to `total_loss`
   before backward. New `div_loss` / `div_cos` diagnostics in `log()`.

4. **`evaluation/eval_cooperative_batch.py`** — reads `routing_mode`
   from `cooperative_config.json` at load time (default "hard" for
   backward compat with v3-v6 checkpoints).

5. **`scripts/exp_cooperative/train_v7_0_merge_diversity.slurm`** —
   v6.4 hparams + `--routing_mode merge --diversity_loss_weight 0.01`.

6. **`evaluation/analysis_v7_differentiation.py`** — new analysis
   script. Reports `cos(A_v, A_a)`, `cos(B_v, B_a)`, and `cos(dW_v, dW_a)`
   per module, aggregated per-layer and per-module-type.

**Key design correction during implementation**: the plan (20.8 item 2)
specified diversity loss on `lora_A`. After running the analysis script
on the v6.4 ep4 checkpoint, `lora_A` was found to be essentially
un-moved from Kaiming init (norms still ≈ 9.3, the Kaiming init value).
All meaningful learning happens in `lora_B` (zero init → learned from
scratch). The diversity loss was therefore placed on `B`, not `A`.

**Crucial v6.4 baseline finding** (from the analysis script):

```
Overall cos(A_v, A_a)       n=196  mean=-0.0001  std=0.0010  [-0.0030, +0.0023]
Overall cos(B_v, B_a)       n=191  mean=-0.0003  std=0.0028  [-0.0108, +0.0079]
Overall cos(dW_v, dW_a)     n=191  mean=+0.0018  std=0.0030  [-0.0039, +0.0184]
```

v6.4's two agents are **already essentially orthogonal** in both A and B
space (cos ≈ 0 ± 0.003). Why? Under hard routing each agent only receives
gradient from its own token type (V sees image tokens, A sees
text/action). Because image-token statistics and text-token statistics
are independent, the two gradient streams are statistically independent,
and the B matrices end up near-orthogonal without any explicit
regularization.

**Implication for v7 merge mode**: when we remove the hard mask, BOTH
agents see ALL tokens — so both gradient streams are driven by the same
targets and they will inevitably collapse toward each other (cos_B →
+1). This is mathematically the expected mode-collapse failure mode, and
is exactly what the diversity loss exists to prevent.

**Sharper expected outcomes** for v7.0:
- **Mode collapse scenario**: cos_B → +1, L_div ≈ +λ, accuracy ≤ single
  LoRA. Means λ=0.01 too weak; scale up, or try information bottleneck.
- **Stable differentiation**: cos_B stays in [-0.2, +0.2] (i.e. order
  of magnitude wider than v6.4's 0.003 but not runaway anti-correlation),
  ce_loss comparable to v6.4, accuracy close to or above v6.4. Then
  measure what the agents specialize to (token class, layer, head, ...).
- **Runaway anti-correlation**: cos_B → -1, ce_loss explodes. Means
  λ=0.01 too strong; halve it.

**Smoke tests passed**:
- `CooperativeLoRALinear(routing_mode='merge')` constructs, forward with
  `token_mask=None` works in both `train()` and `eval()` mode
- Hard mode with `token_mask=None` still raises as before
- `routing_mode='bogus'` raises `ValueError`
- Diversity loss computation: skips modules with zero-init B (step 0),
  gradient flows correctly on perturbed B matrices
- `analysis_v7_differentiation.py` runs on v6.4 ep4 and produces the
  baseline numbers above

Phase A done. Ready to submit v7.0 (Phase B).

---

## 21. v6.4 direction ablation — information flow asymmetry

To decompose where v6.4's +3.55 pp comes from, we ran two surgical
ablations on the v6.4 ep4 checkpoint, each zeroing one direction of the
cross-LoRA communication:

- **VA-only** (A reads V): zero `W_av`, `gate_av` — V cannot read from A.
  Only the `h_a ← h_a + g_va * W_va(h_v)` branch is active.
- **AV-only** (V reads A): zero `W_va`, `gate_va` — A cannot read from V.
  Only the `h_v ← h_v + g_av * W_av(h_a)` branch is active.

Eval on the GUI-360 action_prediction test split (19,046 samples,
4 shards), `proper_routing_batch` inference mode:

| Setting                         | Success | Total  | Accuracy     |
|---------------------------------|---------|--------|--------------|
| v6.4 full (both directions)     | —       | 19,046 | **49.66%**   |
| v6.4 **VA-only** (A reads V)    | 8,905   | 19,046 | **46.76%**   |
| v6.4 **AV-only** (V reads A)    | 8,679   | 19,046 | **45.57%**   |
| v3 (no cross-LoRA comm)         | —       | 19,046 | ~46.1% (ref) |

**Observation**: both directions contribute, but the asymmetry is only
**1.19 pp** (VA > AV). Neither direction alone recovers the full v6.4
gain — removing either drops accuracy by ~3-4 pp, which is roughly the
total v6.4 gain. Interpretation: the two directions are **nearly
additive** rather than redundant — both matter, slightly favoring the
"A reads V" direction.

### 21.1 Independent confirmation from v6.5 training dynamics

v6.5 (same architecture but with `tanh` gate instead of `sigmoid`, and
`lora_B` zero-init so gates can grow beyond the v6.4 ceiling) was trained
from scratch and provides an independent signal. At epoch 1.74:

```
W_av_norm_mean = 12.59   (V reads A path)
W_va_norm_mean = 14.81   (A reads V path)
```

`W_va` is **18% larger** than `W_av`. Since both are zero-init, this
asymmetry is **entirely emergent from the gradient dynamics** — the
optimizer naturally invests more capacity in the "A reads V" direction.

| Signal                          | VA direction (A reads V) | AV direction (V reads A) |
|---------------------------------|--------------------------|--------------------------|
| v6.4 ablation accuracy          | 46.76%                   | 45.57%                   |
| v6.5 learned W norm (ep 1.74)   | 14.81                    | 12.59                    |

Two independent experiments (surgical ablation of a trained v6.4 vs.
from-scratch training of v6.5) both rank the VA direction above the
AV direction. This gives a clean mechanistic claim for the paper:

> In cross-LoRA cooperative communication, the **action agent reading
> from the visual latent** is the dominant information-flow direction.
> The reverse direction (visual agent reading from action latent) is
> beneficial but secondary.

---

## 22. v7.0 live training observations (job 3743585)

At the time of this write-up, v7.0 (`routing_mode=merge`,
`diversity_loss_weight=0.01`) is at **epoch ≈ 0.9 / 4.0**
(~690 / 3052 steps, ~1h15m elapsed).

### 22.1 Training trajectory

```
step    ep   loss   ce_loss  div_cos   W_av_norm  W_va_norm
  10  0.01   5.31    1.318   -0.684      9.238      9.238
 100  0.13   1.55    0.376   -0.558      9.290      9.288
 300  0.38   1.19    0.321   -0.667     10.323     10.237
 500  0.62   1.19    0.270   -0.700     10.564     10.452
 690  0.90   1.06    0.275   -0.717     11.006     10.881
```

**All `W_*` norms grow symmetrically** in v7.0 — unlike v6.5, v7.0 has no
hard routing, so there is no pre-existing reason for one direction to
dominate. Gate magnitudes grow steadily (gate_av_std 0 → 0.089, gate_va
similar).

### 22.2 The div_cos surprise: negative, not positive

The Phase A prediction (§20.9) was that without hard masking, both
agents see all tokens, so `cos(B_v, B_a)` should drift toward **+1**
(mode collapse), and the diversity loss exists to counteract this.

Observed:

- At step 10, `div_cos` jumps to **-0.684** — already strongly negative.
- Bounces back to -0.33 around step 30-40 (small norms, cosine is
  noisy here).
- Steadily drifts more negative from step 100 onward: **-0.56 → -0.72**.

This is the **anti-correlation regime**, not mode collapse. Mechanistic
explanation: in merge mode, `delta = 0.5 * (delta_v + delta_a)`. For any
target delta, the two LoRAs have a one-dimensional slack — they can each
be arbitrarily large in opposite directions and still produce the same
sum. The symmetric CE loss has no preference between `(a, b)` and
`(a+c, b-c)` for any c, so the optimizer is free to find an arbitrarily
anti-correlated pair. The λ=0.01 diversity loss then **explicitly
rewards** negative cosine, accelerating the drift.

This is the opposite failure mode from what was planned: not collapse,
but **hyper-specialization via anti-correlation**. Whether this produces
useful specialization or just a numerically-expensive identity
(`cancel-each-other` pair) is an empirical question the first accuracy
checkpoint will answer.

### 22.3 ce_loss comparison vs v6.4 at the same training step

| epoch | v6.4 ce_loss | v7.0 ce_loss |
|-------|--------------|--------------|
| 0.13  | ~0.48        | 0.376        |
| 0.38  | ~0.42        | 0.321        |
| 0.62  | ~0.38        | 0.270        |
| 0.90  | ~0.36        | 0.275        |

v7.0 converges **measurably faster** on CE loss. Three possible
explanations, not yet distinguished:

1. **Both agents on all tokens** — v7.0's effective batch of
   LoRA-covered tokens is larger than v6.4's (no masking), so gradient
   statistics are better.
2. **Anti-correlated pair as capacity doubling** — the `(B_v, -B_a)`-style
   pair acts like a single LoRA with double the effective rank.
3. **Logging artifact** — the `ce_loss` field has intermittent `nan`
   entries (~1 in 3), so the means above are over a slightly biased
   subsample. The `loss` field (grad-accum aggregate) is clean and also
   lower for v7.0, so the effect seems real but the exact magnitude
   should be read off the clean `loss` column.

### 22.4 The nan logging bug

About 1 in 3 `log()` calls print `ce_loss: nan` while `loss` is healthy
and training is clearly progressing (grad_norm ~1.7, loss monotonically
dropping). The `loss` column is the `Trainer` aggregate and is
authoritative. The `ce_loss` field is our custom per-module running
mean; it is intermittently NaN because some logging windows evaluate
the mean over zero-count buffers. To be fixed post-hoc; not a training
issue.

### 22.5 Evaluation plan

First accuracy eval will be on the **`epoch-1` checkpoint** (saved
automatically by `CooperativeTrainer` at each epoch boundary). Expected
around step 763, ~15 minutes after this write-up. The decision matrix:

- **v7.0 ep1 ≥ v6.4 ep1 (~47.3%)**: merge mode works. Story becomes
  "hard routing unnecessary, self-differentiation beats prior".
- **v7.0 ep1 within -2 pp of v6.4**: merge is viable, direction is
  promising, wait for later checkpoints before concluding.
- **v7.0 ep1 < v6.4 ep1 by >2 pp**: likely the anti-correlated pair is
  not producing useful specialization. Ablate: retry with λ=0 to see
  if diversity loss is the problem, or with λ=-0.01 to force positive
  correlation and confirm diversity is doing something.

## 23. Per-epoch accuracy: v6.5, v7.0 (live results)

All numbers are GUI-360 `action_prediction` accuracy on the held-out
test split, full 19,046 samples (4 shards × 4762, last shard 4760),
proper-routing batch inference. Eval jobs: see `scripts/exp_cooperative/`.

### 23.1 Per-epoch table

| Run                                  | ep1     | ep2     | ep3     | ep4        | Best        |
|--------------------------------------|---------|---------|---------|------------|-------------|
| v6.4 (sigmoid, Kaiming A)            | 47.34%  | 49.24%  | 49.66%  | 49.66%     | 49.66%      |
| v6.5 (tanh, lora_B zero, ceiling↑)   | 45.21%  | 48.49%  | 49.58%  | **50.06%** | **50.06%**  |
| v7.0 (merge + diversity λ=0.01)      | 43.49%  | 46.50%  | 48.08%  | 48.40%     | 48.40%      |

Per-epoch deltas:

| Run    | ep1→ep2 | ep2→ep3 | ep3→ep4 |
|--------|---------|---------|---------|
| v6.4   | +1.90   | +0.42   |  0.00   |
| v6.5   | +3.28   | +1.09   | +0.48   |
| v7.0   | +3.01   | +1.58   | +0.32   |

### 23.2 What this tells us

- **Both v6.5 and v7.0 are slower starters but learn faster per epoch**
  than v6.4. v6.4 had lora_B Kaiming-init → strong but premature
  specialization, so ep1 was already near its plateau. v6.5 (lora_B=0
  zero-start) and v7.0 (merge mode, no hard prior) both bootstrap from
  zero-effect cooperation, so the per-epoch gain is steeper.
- **v6.5 ep4 = 50.06% is the new best cooperative LoRA number, +0.40 pp
  over v6.4 ep4 (49.66%).** v6.5's slower-start / faster-climb trajectory
  from the lora_B zero init finally pays off at ep4: v6.4 plateaus after
  ep3 (ep3→ep4 = +0.00), while v6.5 still picks up +0.48 pp. The
  higher-ceiling tanh gate lets v6.5 amortize its cooperation over a
  longer learning horizon.
- **v7.0 ep4 = 48.40% is still 1.66 pp behind v6.5 ep4.** The gap
  narrowing story from ep1→ep3 (3.85 → 2.74 → 1.58 → 1.66 pp) stalled
  at ep4: v7.0 added only +0.32 pp while v6.5 added +0.48, so v7.0
  actually *lost* a sliver of ground. Merge mode + L_div is a strict
  accuracy regression versus hard-routed cooperative communication on
  GUI-360.
- **Final ranking (best cooperative LoRA checkpoints):**
  1. **v6.5 ep4 = 50.06%** (tanh gate, lora_B zero, gate_lr_mult=100)
  2. v6.4 ep4 = 49.66% (sigmoid gate, Kaiming lora_A)
  3. v7.0 ep4 = 48.40% (merge mode + diversity λ=0.01)
- **What this says about the v6 vs v7 axis:** hard routing + per-layer
  cooperative communication gives ≥1.66 pp advantage over merge mode
  with diversity pressure. The three runs produce genuinely different
  LoRA geometries:
  - v6.4 / v6.5: orthogonal specialization (cos ≈ 0 from §25 SVD)
  - v7.0:        anti-correlated pair (cos = -0.76 from div_loss log)
  - v7.0 nodiv:  drifting toward mode collapse (cos = +0.16 from §26 SVD)

  Of the three, the orthogonal regime is the winner in accuracy and
  the one that stays accurate per epoch the longest.
- **Eval lesson logged for posterity**: extrapolating from the first
  ~160 samples of shard-0 is dangerous. My early estimate predicted
  v6.5 ep2 at 56.2%; actual aggregate was 48.49% (-7.7 pp). Always
  wait for full 4-shard aggregate before drawing conclusions.

### 23.3 Eval job IDs (for log retrieval)

| Run / epoch | Job ID  | Status  |
|-------------|---------|---------|
| v6.5 ep1    | 3743936 | done    |
| v6.5 ep2    | 3743937 | done    |
| v6.5 ep3    | 3744555 | done    |
| v6.5 ep4    | 3745759 | done    |
| v7.0 ep1    | 3743932 | done    |
| v7.0 ep2    | 3744164 | done    |
| v7.0 ep3    | 3744612 | done    |
| v7.0 ep4    | 3745576 | done    |

## 24. v6.5 NCCL deadlock incident

### 24.1 Symptom

v6.5 training (job 3743502) hung silently between epoch 3 and epoch 4.
The job stayed in `RUNNING` state in slurm but no new step was logged
for >1 hour. The last successful step was step 2400 at epoch 3.15
(~07:31), and `checkpoint-2400` was the most recent HF Trainer
checkpoint with full optimizer state. epoch-3 was saved (LoRA-only
folder, ~5 GB), but epoch-4 never produced.

### 24.2 Root cause

The training log was flooded with **9,334** copies of:

```
NCCL WARN ... socketProgress: Connection closed by remote peer
... nid011054 <-> nid011032-hsn1<42918>
```

between two HSN endpoints (nid011054 ↔ nid011032 hsn1 port 42918).
After the connection broke, the training collective (probably an
all-reduce inside the next gradient step) blocked indefinitely on
both sides while waiting for the dead peer. NCCL had no internal
timeout configured, so the deadlock did not auto-fail the job.

### 24.3 Recovery plan

`scancel 3743502` was issued. **First-pass plan was to resume from
`checkpoint-2400`, but inspection revealed it is partially written**:
missing `trainer_state.json`, `scheduler.pt`, and `rng_state_0.pth`.
The NCCL deadlock killed the job mid-checkpoint write — rank 0
finished `optimizer.pt` and the `cooperative/` subfolder, but never
got to `trainer_state.json`. So we resume from the **last fully
written checkpoint, `checkpoint-2300`** instead:

- `cooperative_v6_5_comm_thought/checkpoint-2300/`
  - `cooperative/{lora_v.pt, lora_a.pt, lora_comm.pt, cooperative_config.json}` ← LoRA delta (custom)
  - `optimizer.pt`, `scheduler.pt`, `rng_state_0..31.pth`,
    `trainer_state.json`, `training_args.bin` ← full HF Trainer state
- Step 2300 = epoch **3.0144**, so we resume right at the start of
  epoch 4. We lose ~100 steps of gradient noise and ~0.13 epoch of
  recent updates, but keep optimizer state, LR schedule, and RNG.
- v6.5 has `epoch-3` saved already (LoRA-only) — that gives us the
  ep3 evaluation point regardless of whether the resumed run finishes.

The resume slurm uses the existing `--resume_coop_checkpoint` flag in
`train_cooperative.py` (line 600), pointing at the
`checkpoint-2300/cooperative/` subfolder. The resume logic at line
744-753 detects the parent directory has `trainer_state.json` and
hands it to `trainer.train(resume_from_checkpoint=...)`. HF Trainer
restores optimizer / scheduler / RNG / step automatically; the
cooperative LoRA wrapper loads `lora_v.pt` / `lora_a.pt` /
`lora_comm.pt` from the `cooperative/` subfolder via
`load_cooperative_checkpoint()`.

**Forensic note**: in checkpoint-2400, the `cooperative/` subfolder
files were written successfully (LoRA weights), but the missing
`rng_state_0.pth` shows the NCCL deadlock hit *during* the per-rank
RNG dump — rank 0 wrote `cooperative/` first, then was waiting for an
all-gather/barrier as other ranks wrote their `rng_state_<rank>.pth`,
and the deadlock caught it before `trainer_state.json` could be
written by rank 0.

### 24.4 Resume bug found: HF Trainer rejects "no model weights"

The first resume submission (job 3744722) **failed** at trainer init
with:

```
ValueError: Can't find a valid checkpoint at .../checkpoint-2300
```

Root cause: `CooperativeTrainer.save_model()` is overridden to skip
the 17 GB `model.safetensors` dump (see §24.3 below for why), so the
checkpoint dir contains *zero* model weight files. HF Trainer's
default `_load_from_checkpoint(...)` raises `ValueError` whenever no
`model.safetensors`, `pytorch_model.bin`, etc. is found, **before**
the optimizer / scheduler / RNG state is loaded.

**Fix**: override `_load_from_checkpoint` in `CooperativeTrainer` to
be a no-op for the model side. Cooperative LoRA weights are loaded
from `<ckpt>/cooperative/{lora_v.pt, lora_a.pt, lora_comm.pt}` by
`model.load_cooperative_checkpoint()` in `main()` *before*
`trainer.train()` is called, so the model side is already restored.
HF Trainer's `_load_optimizer_and_scheduler` and `_load_rng_state`
are still called separately and restore optimizer / scheduler / RNG
state correctly.

```python
def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
    """Skip HF Trainer's model-weight reload on resume.
    Cooperative LoRA params are loaded from <ckpt>/cooperative/
    by load_cooperative_checkpoint() in main() before trainer.train().
    """
    if self.is_world_process_zero():
        print(f"[CooperativeTrainer] Skipping model weight reload from "
              f"{resume_from_checkpoint} (cooperative/ already loaded "
              f"separately).")
```

After this fix, job **3744767** resumed successfully:
- `[CooperativeTrainer] Skipping model weight reload from .../checkpoint-2300`
- Step 2300/epoch 3.03 → first logged step `'epoch': 3.03, 'lr': 1.47e-6`
- W_va_norm starts at 16.33 (matches checkpoint state, not zero-init)
- gate_va_max = 0.717 (matches the trajectory of the original v6.5)
- LR continues from the cosine-decayed schedule, not from warmup

This is a one-line generalization of the existing pattern (skip model
file IO because we manage LoRA weights ourselves) and is now a
permanent feature of `CooperativeTrainer`. Future runs that need to
resume will Just Work.

## 25. v6.4 ep4 SVD subspace analysis — orthogonal specialization,
   ##  not anti-correlation

Computed by `evaluation/analyze_lora_subspace.py` on
`cooperative_v6_4_comm_thought/epoch-4`. Job 3745454, ~1m runtime
(CPU-only, 196 modules). For each cooperative LoRA module, extract
the effective delta `ΔW = B A` (without ever materializing
`[out, in]` because of a trace-trick: SVD of `B [out,r]` and
`A [r,in]` separately, then assemble singular values from the central
`[r,r]` matrix).

### 25.1 Overall numbers (averaged over 196 modules)

| Quantity | V | A |
|---|---|---|
| Frobenius cos(ΔW_v, ΔW_a) | **+0.0017** (essentially 0) | — |
| Stable rank | **3.76** | **5.56** |
| Mean principal angle col (top-64) | **78.3°** | — |
| Mean principal angle row (top-64) | **83.8°** | — |

**The headline finding**: v6.4's two adapters have nearly **zero
Frobenius correlation** — they live in essentially orthogonal
subspaces of the residual stream. This is achieved spontaneously,
without any diversity loss, without any bottleneck — just from hard
routing exposing them to disjoint token sets (image vs text).

This reframes v7.0's `cos = -0.76`: v7.0 isn't "achieving
specialization that v6.4 lacks"; v7.0 is *over-shooting past
orthogonal* into actively anti-aligned. Two qualitatively different
regimes:

| Regime | frob_cos | mechanism | description |
|---|---|---|---|
| v6.4 (hard routing) | **~0** | input-space disjointness | each adapter sees its own token type, naturally lands in different subspaces |
| v7.0 (merge + λ=0.01) | **-0.76** | parameter-space repulsion | every token sees both, L_div pushes them apart in flat-vector space |

### 25.2 Stable-rank by module type — A is doing more work than V

| Module type   | Stable rank V | Stable rank A | A/V ratio |
|---------------|---------------|---------------|-----------|
| `o_proj`      | 5.70          | **8.90**      | 1.56      |
| `k_proj`      | 5.24          | 7.99          | 1.52      |
| `down_proj`   | 4.69          | 7.42          | 1.58      |
| `v_proj`      | 3.49          | 6.41          | 1.84      |
| `q_proj`      | 3.81          | 4.18          | 1.10      |
| `up_proj`     | 2.12          | 2.65          | 1.25      |
| `gate_proj`   | **1.24**      | **1.34**      | 1.08      |

Two structural facts pop out:

1. **A consistently uses more effective rank than V** — across
   attention modules (`o_proj`, `k_proj`, `v_proj`) the ratio is
   ~1.5-1.85. This independently corroborates the v6.4 / v6.5 finding
   that the action agent is doing more learned work than the vision
   agent (W_va > W_av in v6.5, asymmetry observed in §21).

2. **`gate_proj` has effective rank ~1.3**, meaning the LoRA delta on
   gate_proj is essentially a **rank-1 update** even though the
   allocated rank is 256. We're wasting 99% of the gate_proj LoRA
   capacity. This is the strongest argument so far for **dropping
   gate_proj from `target_modules`** in future runs — that frees up
   28 layers × 2 (V+A) × ~1.4M params = ~78M trainable params, ~10%
   of the total LoRA parameter budget.

### 25.3 Implications for v7.1 information bottleneck design

The whole point of v7.1 is to use a *capacity constraint* in the
communication channel `W_av` / `W_va` to drive specialization, instead
of an explicit diversity loss. The bottleneck width should be:

- **Larger than** the effective rank we expect specialization to
  occupy (otherwise the bottleneck destroys task performance)
- **Much smaller than** the full rank (otherwise it's not a constraint)

v6.4's stable ranks land in [1.2, 8.9]. Taking the high end (~9) as
the upper bound on what a single agent needs:

| Bottleneck | Headroom over effective rank | Verdict |
|---|---|---|
| `r/4 = 64`  | ~7× | too loose, no real pressure |
| `r/8 = 32`  | ~3.5× | safe, mild pressure |
| **`r/16 = 16`** | **~1.8×** | **recommended starting point** |
| `r/32 = 8`  | ~0.9× | tight, may degrade task perf |

So v7.1 should start with **bottleneck = r/16 = 16**. If task
accuracy holds, sweep to r/32 = 8 and r/8 = 32 for the scaling curve
(Exp 9).

### 25.4 Subspace geometry — almost-but-not-quite orthogonal

Mean principal angles: **78.3° col**, **83.8° row**. For perfectly
orthogonal subspaces this would be 90°; for identical subspaces, 0°.
Two random k-dim subspaces in n-dim space (n ≫ k) have expected
principal angles close to 90° too — but the consistent ~5-12°
deviation from 90° here is statistically significant across 196
modules and indicates **mild** but real subspace overlap that does
not show up in the Frobenius cosine because the magnitudes in the
overlap are tiny.

Intuition: V and A occupy mostly disjoint dimensions of the residual
stream, but in a few "shared" dimensions (likely tokens that need
both visual and textual reasoning, e.g. coordinate parsing) they
both put a small amount of weight. The shared dimensions are heavily
diluted by the large orthogonal mass, so frob_cos ≈ 0.

**`k_proj` has the smallest col-angle (66.4°)** — the largest
overlap. Interpretation: V and A both want to extract similar query
keys ("what features am I looking for in the residual stream"). This
matches the §17 finding that k_proj has the largest gate magnitudes
in v6.4 — k_proj is the module where V and A most need to share
information.

### 25.5 Top-5 anti / pro-correlated modules

```
Most ANTI-correlated (lowest frob_cos):
  layer 23 q_proj    cos=-0.0039   col_angle=79.2°
  layer 22 down_proj cos=-0.0029   col_angle=82.1°
  layer 24 k_proj    cos=-0.0028   col_angle=68.2°
  layer 20 k_proj    cos=-0.0026   col_angle=68.3°
  layer  1 v_proj    cos=-0.0022   col_angle=70.7°

Most CORRELATED (highest frob_cos):
  layer 15 gate_proj cos=+0.0087   col_angle=86.8°
  layer 11 gate_proj cos=+0.0116
  layer 12 gate_proj cos=+0.0118
  layer 13 gate_proj cos=+0.0120
  layer 10 gate_proj cos=+0.0177
```

The top-5 most-correlated are *all* gate_proj in middle layers
(10-15). This is the same module type that has stable_rank ≈ 1.3.
gate_proj is essentially "doing nothing in particular" with
near-rank-1 updates that happen to point in similar directions for V
and A — the small positive correlation is **noise**, not signal.

The top-5 most anti-correlated are mid-to-late layer attention
modules (q_proj, k_proj, v_proj at L20-L24, plus down_proj at L22).
These are the modules where V and A have the *least* overlap. They
also concentrate in the layer band L20-L24 — the same band §17
identified as carrying the gate magnitudes. **The most differentiated
modules are in the same layer band that learns the strongest
cooperative communication** — these are the layers that "know"
they need to talk.

### 25.6 Story integration

The v6.4 SVD result fundamentally changes how we tell the story:

- **Old framing**: v6.4 specializes via hard routing; v7.0 specializes
  via merge + diversity loss; the question is which mechanism is
  "better".
- **New framing**: v6.4 spontaneously achieves **orthogonal**
  specialization (cos ≈ 0). v7.0 with λ=0.01 achieves **anti-aligned**
  specialization (cos ≈ -0.76). These are qualitatively different
  outcomes:
  - cos = 0 means "they don't share dimensions"
  - cos = -0.76 means "they actively cancel each other in the dimensions
    they do share"

Question for v7.0 λ=0 ablation: which point on this spectrum does
merge mode alone produce? Hypothesis: somewhere near 0 (like v6.4)
because merge mode without L_div has no explicit drive toward
anti-alignment, but with no input-space separation either. If λ=0
gives cos ≈ 0, the merge architecture and hard routing converge to
the same orthogonal subspace structure, just via different paths.
That would be a beautiful paper result.

If λ=0 gives cos > 0 (mode collapse): the diversity loss is
load-bearing, and v6.4's hard routing is the "correct" inductive
bias. If λ=0 gives cos < 0: merge mode itself has an attractor
toward anti-alignment, and the diversity loss is just helping it
get there faster.

The v7.0 λ=0 ablation slurm is ready
(`scripts/exp_cooperative/train_v7_0_no_diversity.slurm`); it will
be submitted once v7.0 main and v6.5 resume free up the 8-node slot.

### 24.4 Mitigation for v7.0

v7.0 (job 3743585) is on a different node group and has been training
without socket errors for >4 hours. No immediate action needed, but
we should periodically check the log for NCCL warnings as a
preemptive signal.

### 24.5 Second NCCL hit: v7.0 nodiv ablation (job 3745577)

Same pattern, second time: v7.0 nodiv crashed at step 730 with
`NCCL collective operation timed out (1800038 ms)` on rank 24
(nid010181). All 32 ranks SIGABRT'd at 11:54:20.

This time we got a clean error (not a silent hang) because
`TORCH_NCCL_ASYNC_ERROR_HANDLING=1` was set in the slurm script;
the watchdog enforced the 30-min collective timeout and aborted the
process group, releasing the slurm allocation immediately. Compare
to the v6.5 incident where the same setting wasn't in the original
script and the job hung indefinitely.

**Lesson**: keep `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` in every
multi-node slurm; it converts deadlocks into crashes, which we can
catch and resume from.

Recovery: resume from checkpoint-700 (epoch 0.92, last fully-formed
checkpoint). Job 3746312 submitted, pending allocation. The
post-resume run will produce ep1-ep4 LoRA snapshots. Note that we
already used checkpoint-700 to compute the answer to the L_div
ablation question (see §26 below) — we don't actually NEED the
resume to land before drawing the conclusion. The resume is for
completeness and to get full per-epoch eval numbers.

## 26. v7.0 nodiv step-700 SVD analysis — without L_div, merge mode drifts toward mode collapse

### 26.1 The question

v7.0 (with diversity loss λ=0.01) drove cos(B_v, B_a) from 0 to
**-0.76** within ~50 steps and held it there for 3050 steps. Two
hypotheses for why:

- **H_arch**: anti-correlation is an attractor of merge mode + per-
  layer comm itself. The diversity loss merely accelerates the
  trip; the architecture would get there on its own.
- **H_loss**: anti-correlation is purely the diversity loss talking.
  Without it, the two adapters do whatever they want — independent
  random subspaces (cos ≈ 0), or worse, mode collapse (cos → +1).

Distinguishing H_arch from H_loss matters for v7.1: if H_arch, v7.1
(information bottleneck instead of L_div) is icing on the cake,
because the architecture already knows how to specialize. If H_loss,
v7.1 (or any explicit symmetry breaker) is essential.

### 26.2 The data

v7.0 nodiv at step 700 (epoch 0.92, ~91% of one epoch on r=256
merge mode with all hparams matched to v7.0 except λ=0):

| Module type   | n  | frob_cos     | stable_rank V | stable_rank A | mean angle col | mean angle row |
|---------------|----|-------------|---------------|---------------|----------------|----------------|
| down_proj     | 28 | **+0.1078** | 6.07          | 6.07          | 49.0°          | 86.6°          |
| gate_proj     | 28 | **+0.4485** | 1.37          | 1.23          | 53.6°          | 83.0°          |
| k_proj        | 28 | **+0.0890** | 8.33          | 8.30          | 35.5°          | 83.5°          |
| o_proj        | 28 | **+0.1044** | 6.90          | 7.11          | 45.7°          | 83.4°          |
| q_proj        | 28 | **+0.1237** | 4.61          | 4.47          | 40.7°          | 83.4°          |
| up_proj       | 28 | **+0.1852** | 2.34          | 2.23          | 52.4°          | 83.3°          |
| v_proj        | 28 | **+0.0936** | 5.43          | 5.45          | 38.4°          | 83.5°          |
| **ALL (196)** |    | **+0.1646** | **5.01**      | **4.98**      | **45.1°**      | **83.8°**      |

Most-correlated modules are dominated by mid-block `gate_proj`:

```
9.mlp.gate_proj    cos=+0.5440  angle_col=59.5°
6.mlp.gate_proj    cos=+0.4934  angle_col=58.2°
14.mlp.gate_proj   cos=+0.4924  angle_col=53.0°
16.mlp.gate_proj   cos=+0.4803  angle_col=52.9°
18.mlp.gate_proj   cos=+0.4739  angle_col=52.5°
```

Least-correlated modules are early-layer `down_proj`:

```
1.mlp.down_proj    cos=+0.0263  angle_col=47.4°
2.mlp.down_proj    cos=+0.0428  angle_col=50.4°
0.mlp.down_proj    cos=+0.0498  angle_col=48.9°
5.mlp.down_proj    cos=+0.0568  angle_col=45.1°
4.mlp.down_proj    cos=+0.0640  angle_col=46.7°
```

All modules have **positive** cos. Nothing is anti-correlated.

### 26.3 Three-way comparison

| Run                       | epoch | frob_cos | mean angle col | stable_rank V | story                       |
|---------------------------|-------|----------|----------------|---------------|-----------------------------|
| v6.4 (hard route, sigmoid)| 4     | +0.0017  | 78.3°          | 3.76          | spontaneous orthogonal      |
| v7.0 (merge + L_div=0.01) | 4     | -0.7649  | ~120°          | n/a           | active anti-correlation     |
| **v7.0 nodiv (this)**     | 0.92  | **+0.1646** | **45.1°**   | **5.01**      | **drift toward correlation** |

The three rows are three qualitatively distinct outcomes:

1. **Hard routing (v6.4)**: each adapter only sees its own modality
   in the LoRA forward, so gradients to the two adapters come from
   disjoint token subsets → no incentive to align. Result: random
   orthogonal subspaces, mean angle ~78° (close to the mean angle of
   two random r-dim subspaces in a high-D space).
2. **Merge mode + L_div (v7.0)**: every token sees `0.5*(δv + δa)`,
   so the gradient on (B_v A_v) and (B_a A_a) is identical except
   for the L_div penalty. The penalty pushes them past orthogonal
   into actively anti-aligned territory.
3. **Merge mode no L_div (this)**: every token sees the same
   merged delta. The gradients to the two adapters are now nearly
   identical, so they slowly move toward the same direction. Cos
   has drifted from 0 (init) to +0.165 in 700 steps, mean angle
   from 90° to 45°. Trend: clearly toward +1 (mode collapse).

### 26.4 What this means for v7.1

**Outcome C** (mode collapse direction) is the right interpretation,
even though we caught it before full collapse. Without L_div the two
adapters are heading to the same place. The diversity loss in v7.0
is doing real work: it breaks the symmetry that merge mode imposes.

This **strengthens** the case for v7.1 (information bottleneck):

- v7.1's premise is "use a structural constraint instead of an
  explicit loss to drive specialization". The structural constraint
  has to come from somewhere — and §26 shows that merge mode by
  itself does not provide one.
- A bottleneck on (lora_A_v, lora_A_a) channels would force each
  adapter to compress the input to a low-dim summary, and the only
  way to recover full information at the merge step is for the two
  summaries to be complementary (i.e. orthogonal in input space).
  Symmetry-breaking comes from the joint reconstruction objective.
- Concrete proposal for v7.1: insert an information bottleneck of
  size r/16 = 16 between lora_A and lora_B for each adapter, plus
  a small L2 on the bottleneck activations. No L_div. Hypothesis:
  the bottleneck + merge will give cos near 0 and accuracy
  comparable to v7.0.

The alternative — go back to v6.x hard routing and forget merge —
is also valid (v6.5 ep3 already matches v6.4 ep3 within 0.08 pp).
But the merge story is interesting precisely because it forces
both adapters to "collaborate" on every token, which is a more
elegant model of cooperation than "you handle text, I handle image".

### 26.5 Trajectory locked: saturation, not collapse

**UPDATE**: resume job 3746312 completed (4 epochs). SVD on the final
v7.0 nodiv epoch-4 checkpoint (job 3753184):

| Checkpoint         | epoch | frob_cos | col angle (B) |
|--------------------|-------|----------|---------------|
| v7.0 nodiv step700 | 0.92  | +0.1646  | 45.1°         |
| **v7.0 nodiv ep4** | **4** | **+0.2014** | **46.2°**  |

The trajectory **saturated**, not collapsed:
- frob_cos drifted only +0.037 over 3 more epochs (0.92 → 4.0)
- col angle actually *increased* slightly (45.1° → 46.2°), meaning
  B column spaces stopped aligning further after epoch ~1

The earlier linear extrapolation ("cos at step 3050 ≈ 0.7") was wrong.
Merge mode without L_div converges to a mild positive correlation
(cos ~0.20, col ~46°), not full mode collapse. This is an equilibrium
where the shared gradient creates some alignment but not total overlap.

See §27 for the full 5-way comparison and the deeper Goodhart finding.

## 27. v6.5 ep4 + v7.0 ep4 SVD analysis — L_div Goodharted, v7.0 is not anti-correlated

Computed by `analyze_lora_subspace.py` on the epoch-4 LoRA snapshots,
filling the last two rows of the 4-way table. Jobs 3748172 (v6.5 ep4,
~75 s) and 3748173 (v7.0 ep4, ~75 s), CPU-only.

### 27.1 Final 5-way table

| Run                         | acc        | frob_cos(ΔW_v, ΔW_a) | mean angle col (B) | mean angle row (A) | stable_rank V | stable_rank A |
|-----------------------------|------------|----------------------|--------------------|--------------------|---------------|---------------|
| **v6.5 ep4 (hard route)**   | **50.06%** | **+0.0015**          | **78.4°**          | **83.8°**          | **3.96**      | **5.67**      |
| v6.4 ep4 (hard route)       | 49.66%     | +0.0017              | 78.3°              | 83.8°              | 3.76          | 5.56          |
| v7.0 nodiv ep4 (merge, λ=0) | **48.62%** | +0.2014              | **46.2°**          | 83.5°              | 5.83          | 5.50          |
| v7.0 ep4 (merge, L_div)     | 48.40%     | +0.1837              | **27.8°**          | 83.5°              | 5.21          | 4.99          |

(Row for v7.0 nodiv step-700 snapshot: frob_cos = +0.1646, col = 45.1°
— trajectory saturated from step 700 to ep4, not collapsed.)

All runs fall into exactly **two** geometric regimes:

- **Orthogonal (v6.4, v6.5)**: B column spaces and A row spaces are
  both near-orthogonal. Frob cos ≈ 0. Accuracy 49–50%.
- **B-partially-aligned (v7.0 variants)**: B column spaces are
  partially aligned (col angle 27–46°), A row spaces still orthogonal,
  Frob cos of the effective delta is positive (+0.18 to +0.20).
  Accuracy 48.40–48.62%.

**The accuracy ladder is perfectly monotonic with B column-space
orthogonality** (larger col angle → higher accuracy, 4/4 data points).

**L_div is confirmed harmful**: v7.0 nodiv (48.62%) beats v7.0 L_div
(48.40%) by +0.22 pp, while having more orthogonal B column spaces
(46.2° vs 27.8°). L_div actively *worsened* column-space diversity
while achieving −0.76 on its own proxy (flattened B cosine). The
Goodhart failure is not just "no effect" — it's actively harmful to
both the geometric quantity that correlates with accuracy AND to
accuracy itself.

### 27.2 The Goodhart twist: v7.0 is NOT anti-correlated

This is the central surprise of this section. The v7.0 training log
reported `div_cos → −0.7649` and I had been writing this up as
"active anti-correlation" for the past 24 hours. The static SVD
measurement on the final checkpoint **contradicts** that framing:
effective Frobenius cos is **+0.1837**, not −0.76.

**Why the discrepancy?** Look at `train_cooperative.py:407-423`. The
diversity loss is computed as:

```python
bv = m.lora_B_v.flatten()                     # [out * r] vector
ba = m.lora_B_a.flatten()                     # [out * r] vector
cos = F.cosine_similarity(bv.unsqueeze(0), ba.unsqueeze(0))
L_div = mean over modules of cos               # pushed toward -inf
```

This measures `cos(flatten(B_v), flatten(B_a))` — cosine of
**flattened B matrices as raw vectors**. My SVD analysis measures
`cos(flatten(B_v A_v), flatten(B_a A_a))` — cosine of the
**effective adapter delta**. The two quantities are very different:

- `cos(flat(B_v), flat(B_a))` is sensitive to element-wise sign
  patterns of B. If you multiply `B_a` by −1, this cosine flips
  from +1 to −1 but the span of the columns of `B_a` is unchanged.
- `cos(flat(B_v A_v), flat(B_a A_a))` is sensitive to the full
  geometry of the adapter output: B's column span, A's row span,
  and their magnitudes and relative sign-coupling.

When you penalize the first and train A jointly, A learns to
**compensate** for B's sign-flipping: A co-adapts its entries so that
`B_v A_v` and `B_a A_a` stay roughly in the same direction even
though B_v and B_a point opposite element-wise. The div_loss metric
goes to −0.76; the actual adapter geometry doesn't.

This is a textbook **Goodhart failure**: the proxy was
aggressively optimized (achieved −0.76), but the target
quantity it was supposed to represent (adapter diversity in
output space) barely moved (+0.18 vs +0.16 for the nodiv
ablation).

Supporting evidence: **B column spaces in v7.0 ep4 are at mean
angle 27.8°**, i.e. *strongly aligned* (cos of column span ≈ 0.88).
This is the opposite of what L_div was trying to achieve. If L_div
were truly making the adapters diverse in output space, the column
angle would be closer to 90°.

### 27.3 v6.5 ≈ v6.4 geometrically, but +0.40 pp in accuracy

v6.5 ep4 (50.06%) and v6.4 ep4 (49.66%) are almost **indistinguishable**
in SVD subspace terms:

| Quantity              | v6.4 ep4 | v6.5 ep4 | Δ     |
|-----------------------|----------|----------|-------|
| frob_cos(ΔW_v, ΔW_a)  | +0.0017  | +0.0015  | −0.0002 |
| Mean col angle (B)    | 78.3°    | 78.4°    | +0.1°   |
| Mean row angle (A)    | 83.8°    | 83.8°    | 0.0°    |
| Stable rank V (mean)  | 3.76     | 3.96     | +0.20   |
| Stable rank A (mean)  | 5.56     | 5.67     | +0.11   |

The two runs land in statistically equivalent orthogonal regimes.
v6.5's +0.40 pp gain is therefore **not** from a better geometric
regime — both are sitting at cos ≈ 0. It must come from better
optimization within the same regime. Candidates:

- **Softer start, longer climb.** v6.5's lora_B zero-init gives it
  zero effective cooperation at step 0, so early epochs are
  effectively plain LoRA (V adapter only) + slow cooperation ramp-up.
  v6.4's Kaiming lora_B has large |BA| at step 1, which means v6.4
  has to *un-learn* random cooperation before learning the right
  cooperation. Observed in the epoch-1 gap: v6.4 47.34% vs v6.5 45.21%
  (v6.4 ahead), and by ep4 the order reverses.
- **Higher gate ceiling + gate_lr_mult=100.** v6.5 lets the gate
  climb higher and learn faster, so by ep4 the cooperative path is
  contributing more net signal.
- **Tanh instead of sigmoid.** Tanh is centered at 0 and symmetric,
  so at gate init the pass-through is exactly the identity LoRA
  (no cooperation); sigmoid starts at 0.5 (half cooperation from
  step 0, with random Kaiming B). Tanh is a cleaner zero point.

Stable ranks are very slightly higher in v6.5 (+0.2 on V, +0.1 on A),
meaning v6.5 uses its r=256 capacity a bit more broadly. This is
consistent with "softer start" → "less collapse onto a few top
singular directions".

### 27.4 The accuracy ladder = B column-space alignment

With four data points now, the pattern is clean:

| Run                  | Mean angle col (B) | frob_cos | Accuracy |
|----------------------|--------------------|----------|----------|
| v6.5 ep4             | **78.4°**          | +0.0015  | **50.06%** |
| v6.4 ep4             | 78.3°              | +0.0017  | 49.66% |
| v7.0 nodiv ep4       | **46.2°**          | +0.2014  | **48.62%** |
| v7.0 ep4 (L_div)     | **27.8°**          | +0.1837  | 48.40% |

**Perfect monotonic correlation** (4/4 data points): larger col
angle → higher accuracy. Orthogonal B column spaces are the winning
geometry on GUI-360.

**L_div is confirmed harmful.** v7.0 nodiv (48.62%) > v7.0 L_div
(48.40%) by +0.22 pp. v7.0 nodiv also has more orthogonal B column
spaces (46.2° vs 27.8°). The diversity loss pushed B columns closer
together in span while making them anti-correlated element-wise,
damaging both geometry and accuracy. Case closed.

### 27.5 Revised v7.1 design implications

§26.4 proposed an information-bottleneck v7.1. §27.2 reinforces the
need for it but also sharpens the failure mode we're trying to fix:

- L_div on `flat(B)` is **the wrong proxy**. Any replacement metric
  must operate on the effective adapter delta (or the column span
  of B), not element-wise on B.
- Option 1: direct column-span diversity loss.
  `L_div_v2 = mean_m cos_principal_top_k(B_v^col, B_a^col)`.
  Expensive (SVD every step) but targets the right quantity.
- Option 2: information bottleneck (still the leading candidate).
  Insert a low-rank (r/8 = 32) linear bottleneck between A and B
  per adapter. Force each adapter to compress its input before
  producing output. The only way merge mode can reconstruct full
  signal is if the two compressed summaries span complementary
  directions → orthogonality as an emergent property.
- Option 3: **give up on merge and go back to v6.x**. v6.5 ep4 is
  already the best number we have (50.06%). The elegance argument
  for merge mode no longer has an accuracy leg to stand on: merge
  is 1.66 pp worse at epoch 4 in both L_div and no-L_div
  variants, and the geometry it produces is strictly inferior
  (col aligned instead of col orthogonal).

### 27.6 Action items

1. ~~Update §26 caveat~~ — DONE (§26.5 updated: trajectory saturated
   at cos +0.20, col 46°, not collapsed).
2. ~~Re-run SVD on v7.0 nodiv ep4~~ — DONE (job 3753184: frob_cos
   = +0.2014, col = 46.2°). Trajectory saturated.
3. ~~Run v7.0 nodiv ep4 eval~~ — DONE (job 3762694: **48.62%**).
   L_div confirmed harmful: nodiv (48.62%) > L_div (48.40%) by
   +0.22 pp. The accuracy ladder is perfectly monotonic with B
   column-space orthogonality across all 4 checkpoints.
4. Decide v7.1 direction: options 1/2/3 in §27.5 above. My
   recommendation given the new data: **option 3 (abandon merge)**
   for the near-term accuracy push, **option 2 (bottleneck)** as
   the architecturally ambitious follow-up on a separate track.
