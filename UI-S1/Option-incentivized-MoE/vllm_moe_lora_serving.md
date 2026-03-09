# vLLM MoE-LoRA Serving: Per-Sample Routing to Multiple LoRA Experts

> 调研时间：2026-03-05
> 目标：调研如何使用 vLLM 实现 per-sample routing 到多个 LoRA expert，评估现有支持与扩展方案

---

## 1. 背景

我们的 MoE-LoRA 架构（`verl/models/moe/`）在 base VLM 上叠加多个 LoRA expert，通过 router 对每个 request 生成 routing weights `[B, num_experts]`，在 forward pass 中加权组合所有 expert 的输出：

```python
# MoELoRALinear.forward() 核心逻辑
base_out = self.base_linear(x)
all_deltas = torch.stack([lora(x) for lora in self.expert_loras], dim=0)  # [E, B, S, D]
weighted_delta = (all_deltas * routing_weights).sum(dim=expert_dim)       # [B, S, D]
return base_out + weighted_delta
```

使用 HuggingFace 原生推理（`serve_moe.py`）每个 sample 需要 ~2.7s，完整评估需要 ~41 小时，无法接受。需要调研 vLLM 是否能支持这种 MoE-LoRA 推理。

---

## 2. vLLM LoRA 架构分析

### 2.1 核心设计：One LoRA Per Request

vLLM 的 LoRA 系统基于 Punica/S-LoRA 论文，设计目标是**多租户 LoRA serving**（不同 request 用不同 LoRA），而非**单 request 多 LoRA 混合**。

```python
# vllm/lora/request.py
class LoRARequest(msgspec.Struct):
    lora_name: str
    lora_int_id: int       # 全局唯一整数 ID，每个 request 恰好一个
    lora_path: str = ""
    # 没有 weight 字段，没有多 adapter 支持
```

### 2.2 Token-to-LoRA Mapping：纯整数索引

```python
# vllm/adapter_commons/layers.py
@dataclass
class AdapterMapping:
    index_mapping: Tuple[int, ...]    # 每个 token 一个整数 LoRA ID
    prompt_mapping: Tuple[int, ...]   # 每个 sampled token 一个 LoRA ID
```

`convert_mapping()` 函数（`vllm/lora/punica_wrapper/utils.py`）将 mapping 转换为 index tensor：

```python
# token_lora_mapping: [num_tokens], dtype=torch.int32
# 每个 token 映射到一个 LoRA slot index，-1 表示无 LoRA
# 没有 float weight，没有多 index 支持
```

`LoRAKernelMeta`（`vllm/lora/ops/triton_ops/lora_kernel_metadata.py`）进一步确认：

```python
@dataclass
class LoRAKernelMeta:
    token_lora_mapping: torch.Tensor        # [num_tokens], int32
    token_indices_sorted_by_lora_ids: ...   # 按 LoRA ID 排序的 token indices
    active_lora_ids: torch.Tensor           # [max_loras + 1]
    num_tokens_per_lora: torch.Tensor       # 每个 LoRA 的 token 数量
    lora_token_start_loc: torch.Tensor      # cumsum offsets
```

### 2.3 LoRA 权重存储

```python
# vllm/lora/layers.py - BaseLinearLayerWithLoRA.create_lora_weights()
self.lora_a_stacked = torch.zeros(max_loras, 1, rank, input_size, ...)     # [max_loras, 1, r, D_in]
self.lora_b_stacked = torch.zeros(max_loras, 1, output_size, rank, ...)    # [max_loras, 1, D_out, r]
```

第一维是 `max_loras`（adapter slot 数量），通过 `set_lora(index, ...)` 写入特定 slot。

### 2.4 Triton Kernel：每个 token 只用一个 LoRA

**Shrink kernel**（`vllm/lora/ops/triton_ops/lora_shrink.py`）：

```python
grid = (
    SPLIT_K * cdiv(M, BLOCK_M) * cdiv(N, BLOCK_N),
    NUM_SLICES,
    MAX_LORAS,      # <-- 每个 LoRA 一组 thread blocks
)

# 内部逻辑：
lora_id = tl.load(lora_ids + lora_idx)
if lora_id == -1:
    return   # 该 slot 无 LoRA，直接返回
# 只处理属于该 LoRA 的 tokens
```

**Expand kernel**（`vllm/lora/ops/triton_ops/kernel_utils.py`）：

```python
# 使用 lora_index（单个整数）索引权重 buffer
b_ptr = cur_lora_ptr + cur_lora_d0_stride * lora_index
# output[token] += input[token] @ lora_b[lora_id]
# 没有 routing weight 乘法，没有多 expert 累加
```

### 2.5 结论

| 维度 | vLLM LoRA | 我们的 MoE-LoRA | Gap |
|------|-----------|----------------|-----|
| LoRA/request | 恰好 1 个 | N 个加权组合 | **根本性差异** |
| Token mapping | 整数索引 (int32) | Float weights [B, E] | 数据类型 + 语义 |
| Kernel 计算 | 1 次 matmul/LoRA，加到 output | E 次 matmul，加权求和 | 缺少 weight 乘法 |
| Routing | 无（固定 mapping） | Router MLP 前向传播 | 不在 serving 路径中 |
| 两遍 forward | 不支持 | Pass 1 routing + Pass 2 inference | 需要 prefill hook |

**vLLM 原生不支持 weighted multi-LoRA combination。** 没有现有 PR 或 issue 讨论 MoE-LoRA 支持。

---

## 3. 实现方案

### 方案 1：Hard Routing + vLLM Multi-LoRA（最简单，推荐先用）

**思路：** 用 argmax 硬路由选择单个 expert，利用 vLLM 原生 multi-LoRA serving。

**实现步骤：**

1. **Router 作为独立服务/预处理**：Router 是一个 2 层 MLP（256 hidden），计算量可忽略（<0.1ms）
2. **预加载所有 expert 到 vLLM**：我们的 checkpoint 已经是 PEFT 格式（`experts/expert_{i}/adapter_model.bin`）
3. **每个 request 路由到 top-1 expert**：

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# 预加载
llm = LLM(model="base_model_path", enable_lora=True, max_loras=6)
expert_lora_requests = [
    LoRARequest(f"expert_{i}", i+1, f"checkpoint/experts/expert_{i}")
    for i in range(6)
]

# 推理
expert_idx = router.forward(instruction_features).argmax().item()
output = llm.generate(prompt, sampling_params, lora_request=expert_lora_requests[expert_idx])
```

**优点：**
- 零 vLLM 修改，今天就能用
- 完整 Triton kernel 性能
- 不同 request 可以路由到不同 expert，batch 内并行

**缺点：**
- 丢失 soft routing（只用 top-1 expert）
- Router 需要在 vLLM 外部运行

**适用场景：** 如果 router confidence 高（dominant expert weight >> 0.5），top-1 近似损失很小。我们的 `top_k=6` 配置使用全部 expert，所以这个方案会有质量损失。

**预计工作量：** 1-2 天

---

### 方案 2：Per-Request 权重合并（Soft Routing，推荐）

**思路：** 每个 request 先计算 routing weights，然后在 GPU 上将多个 expert LoRA 权重合并为一个等效 LoRA，再交给 vLLM。

**数学等价性：**

```
output = base(x) + Σ_i w_i * (B_i @ A_i @ x)
       = base(x) + (Σ_i w_i * B_i) @ (A_i) @ x    # 如果 A_i 相同
       ≈ base(x) + B_merged @ A_merged @ x           # 近似
```

注意：严格来说 `Σ w_i (B_i @ A_i)` ≠ `(Σ w_i B_i) @ (Σ w_i A_i)`，但可以直接合并 `delta_W = Σ w_i (B_i @ A_i)` 然后做 SVD 重新分解为 rank-r，或者直接合并到 base weight。

**实现方案 2a：合并到 base weight（最精确）**

```python
import torch

def merge_experts_to_base(base_model, expert_loras, routing_weights, scaling):
    """将加权 expert LoRA 直接合并到 base weight"""
    merged_model = copy.deepcopy(base_model)
    for layer_name in target_modules:  # q_proj, v_proj
        base_weight = get_weight(merged_model, layer_name)  # [D_out, D_in]
        delta = torch.zeros_like(base_weight)
        for i, (lora_A, lora_B) in enumerate(expert_loras):
            # lora_A: [r, D_in], lora_B: [D_out, r]
            delta += routing_weights[i] * (lora_B @ lora_A) * scaling
        base_weight.add_(delta)
    return merged_model
```

**合并开销估算：**
- q_proj + v_proj × 28 layers = 56 个模块
- 每个模块：6 个 expert × (B[3584,16] @ A[16,3584]) = 6 × 3584² ≈ 77M FLOPs
- 总计：56 × 77M ≈ 4.3G FLOPs ≈ **~2ms on GPU**
- 但需要 deepcopy base model weights ≈ 14GB，不可接受

**实现方案 2b：合并 LoRA A/B 矩阵（推荐）**

```python
def merge_expert_lora_weights(expert_checkpoints, routing_weights):
    """合并为一个等效 LoRA adapter"""
    merged = {}
    for key in expert_checkpoints[0].keys():
        # 对 A 和 B 矩阵分别加权求和
        merged[key] = sum(
            w * ckpt[key]
            for w, ckpt in zip(routing_weights, expert_checkpoints)
            if w > 1e-6
        )
    return merged
```

注意：`Σ w_i B_i @ Σ w_i A_i ≠ Σ w_i (B_i @ A_i)`，这只是近似。但如果所有 expert 从同一个初始化 copy 出发（我们的 v2 就是），early training 阶段 A_i 和 B_i 差异小，近似误差也小。

**更精确的做法：** 合并 delta_W = Σ w_i (B_i @ A_i)，然后做 truncated SVD 分解回 rank-r：

```python
delta_W = sum(w * (B_i @ A_i) for w, B_i, A_i in zip(weights, Bs, As))  # [D_out, D_in]
U, S, Vt = torch.linalg.svd(delta_W, full_matrices=False)
# 截断到 rank r
merged_B = U[:, :r] * S[:r].sqrt()    # [D_out, r]
merged_A = Vt[:r, :] * S[:r].sqrt().unsqueeze(1)  # [r, D_in]
```

SVD 开销：[3584, 3584] 矩阵 SVD ≈ 几十 ms，56 个模块 ≈ 1-2s per request。可以接受但不理想。

**优点：**
- 保留 soft routing 语义（精确或高精度近似）
- 不修改 vLLM kernel

**缺点：**
- Per-request 合并开销（方案 2b 可控，方案精确需要 SVD）
- 不能跨 request 共享合并后的 adapter（除非 routing weights 相同）
- 需要管理 vLLM 的临时 LoRA slot

**预计工作量：** 2-3 天

---

### 方案 3：修改 vLLM Triton Kernel（最优性能，工程量大）

**思路：** 修改 Triton kernel 支持 weighted multi-LoRA summation。

**需要修改的文件和内容：**

#### 3.1 新的 Mapping 数据结构

```python
# 替换 token_lora_mapping: [num_tokens] (int32)
# 为：
token_expert_indices: [num_tokens, top_k]    # int32, 每个 token 的 top-k expert indices
token_expert_weights: [num_tokens, top_k]    # bfloat16, 对应的 routing weights
```

#### 3.2 修改 LoRAKernelMeta

```python
# vllm/lora/ops/triton_ops/lora_kernel_metadata.py
@dataclass
class MoELoRAKernelMeta:
    token_expert_indices: torch.Tensor    # [num_tokens, top_k]
    token_expert_weights: torch.Tensor    # [num_tokens, top_k]
    # 不再按 LoRA ID 排序，而是每个 token 独立处理 top_k experts
```

#### 3.3 修改 Shrink Kernel

```python
# 当前：buffer = x @ lora_a[lora_id] * scale
# 修改为：buffer = Σ_k w_k * (x @ lora_a[expert_k]) * scale

@triton.jit
def _moe_lora_shrink_kernel(
    input_ptr, lora_a_ptr, output_ptr,
    expert_indices_ptr, expert_weights_ptr,  # 新增
    TOP_K: tl.constexpr, ...
):
    token_idx = ...
    accumulator = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(TOP_K):
        expert_id = tl.load(expert_indices_ptr + token_idx * TOP_K + k)
        weight = tl.load(expert_weights_ptr + token_idx * TOP_K + k)
        if expert_id >= 0:
            # lora_a for this expert
            a_ptr = lora_a_ptr + expert_id * lora_a_stride
            partial = matmul(input_ptr, a_ptr, ...)
            accumulator += weight * partial

    tl.store(output_ptr + ..., accumulator * scale)
```

#### 3.4 修改 Expand Kernel

类似修改，在 `kernel_utils.py` 的 `_lora_expand` 中加入 top_k 循环和 weight 乘法。

#### 3.5 修改 PunicaWrapperGPU

在 `add_shrink()` 和 `add_expand()` 中传递新的 multi-index metadata。

#### 3.6 新增 MoELoRARequest

```python
class MoELoRARequest:
    expert_lora_ids: List[int]       # 多个 LoRA adapter IDs
    expert_weights: List[float]      # 对应 routing weights
```

**优点：**
- Kernel 级别完整 soft routing，最大性能
- 支持 batch 内不同 request 有不同 routing weights

**缺点：**
- 工程量大（~2-4 周）
- 每个 token 做 top_k 次 matmul，LoRA 部分吞吐量降低 top_k 倍
- 与上游 vLLM 分叉，维护成本高

**预计工作量：** 2-4 周

---

### 方案 4：Custom Model Class + Module Replacement（中等方案）

**思路：** 不使用 vLLM 的 LoRA 基础设施，直接将 `MoELoRALinear` 注入到 vLLM 加载的模型中。

**实现思路：**

```python
# 1. 注册自定义模型类
from vllm import ModelRegistry

class Qwen2VLMoE(Qwen2VLForConditionalGeneration):
    def __init__(self, config, ...):
        super().__init__(config, ...)
        # 替换 q_proj, v_proj 为 MoELoRALinear
        self._replace_with_moe_lora(moe_config, checkpoint_path)

    def _replace_with_moe_lora(self, moe_config, checkpoint_path):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and name.endswith(('q_proj', 'v_proj')):
                moe_linear = MoELoRALinear(module, moe_config)
                replace_module(self, name, moe_linear)
        # 加载 expert weights + router weights

ModelRegistry.register_model("Qwen2VLMoE", Qwen2VLMoE)
```

**优点：**
- 复用训练时的 MoELoRALinear 架构
- 保留完整 soft routing

**缺点：**
- 绕过 vLLM 的 LoRA 优化（Triton kernel, multi-tenant）
- Python loop 计算 expert deltas 会很慢（需要 kernel fusion）
- 与 vLLM 的 torch.compile 和 CUDAGraph 冲突
- Router 的两遍 forward 与 vLLM 的 pipeline 不兼容

**预计工作量：** 1-2 周

---

## 4. 方案对比

| 方案 | Soft Routing | 性能 | 工程量 | vLLM 修改 | 推荐度 |
|------|-------------|------|--------|-----------|--------|
| 1. Hard Routing | ❌ top-1 only | ⭐⭐⭐⭐⭐ | 1-2天 | 无 | 🔥 先用这个 |
| 2. Per-Request 合并 | ✅ 精确/近似 | ⭐⭐⭐⭐ | 2-3天 | 无 | 🔥 质量优先 |
| 3. Custom Triton Kernel | ✅ 精确 | ⭐⭐⭐⭐⭐ | 2-4周 | 大量 | 长期投资 |
| 4. Module Replacement | ✅ 精确 | ⭐⭐ | 1-2周 | 中等 | 不推荐 |

---

## 5. 推荐路线

### 短期（评估 MoE v2）

使用**方案 1（Hard Routing）** 快速跑通评估：
1. Router 作为独立预处理（加载 router checkpoint，对每个 sample 计算 top-1 expert）
2. 用 vLLM multi-LoRA serving（`enable_lora=True, max_loras=6`）
3. 每个 request 附带对应 expert 的 `LoRARequest`
4. 预计推理速度与单 LoRA vLLM 相同（~0.3s/sample）

### 中期（质量保证）

使用**方案 2（Per-Request 合并）** 保留 soft routing：
1. 对每个 request，router 计算 routing weights
2. 将 6 个 expert LoRA 按权重合并为 1 个等效 LoRA
3. 动态加载合并后的 LoRA 到 vLLM
4. 合并开销 ~2ms/request，可接受

### 长期（如果 MoE-LoRA 成为核心方案）

考虑**方案 3（Custom Kernel）** 或贡献上游 vLLM。

---

## 6. 关键文件索引

### vLLM 源码（v0.8.5）

| 组件 | 路径 |
|------|------|
| LoRA Request | `vllm/lora/request.py` |
| LoRA Layers (linear forward) | `vllm/lora/layers.py` |
| Token-to-LoRA Mapping | `vllm/adapter_commons/layers.py` |
| Mapping → Index Tensors | `vllm/lora/punica_wrapper/utils.py` |
| Punica GPU Wrapper | `vllm/lora/punica_wrapper/punica_gpu.py` |
| Kernel Metadata | `vllm/lora/ops/triton_ops/lora_kernel_metadata.py` |
| Triton Shrink Kernel | `vllm/lora/ops/triton_ops/lora_shrink.py` |
| Triton Expand Kernel | `vllm/lora/ops/triton_ops/lora_expand.py` |
| Kernel Utils | `vllm/lora/ops/triton_ops/kernel_utils.py` |
| LoRA Worker Manager | `vllm/lora/worker_manager.py` |

### 我们的 MoE 实现

| 组件 | 路径 |
|------|------|
| MoE Wrapper (two-pass forward) | `verl/models/moe/moe_wrapper.py` |
| MoELoRALinear (expert computation) | `verl/models/moe/expert_lora.py` |
| TextOnlyRouter (MLP router) | `verl/models/moe/router.py` |
| MoE Checkpoint Format | `moe_sft/output/*/experts/expert_{i}/adapter_model.bin` |

---

## 7. 附录：vLLM Triton Kernel 详细分析

### Shrink Kernel 流程

```
Input: x [num_tokens, D_in]
LoRA A: lora_a_stacked [max_loras, 1, r, D_in]
Output: buffer [num_tokens, r]

Grid: (tiles, NUM_SLICES, MAX_LORAS)

Per thread block:
1. 确定当前处理的 LoRA slot (lora_idx = pid_lora)
2. 加载该 slot 的 LoRA ID，如果 == -1 则 return
3. 加载该 LoRA 的 token 数量和起始位置
4. 对属于该 LoRA 的 tokens 做 matmul: buffer[token] = x[token] @ A[lora_id]
```

### Expand Kernel 流程

```
Input: buffer [num_tokens, r]  (来自 shrink 的输出)
LoRA B: lora_b_stacked [max_loras, 1, D_out, r]
Output: output [num_tokens, D_out]  (累加到 base output 上)

类似流程，对每个 LoRA 的 tokens: output[token] += buffer[token] @ B[lora_id]
```

### 关键限制

1. **Grid 的 MAX_LORAS 维度**：每个 LoRA 独立的 thread block 组，token 集合不重叠
2. **Additive output**：`output += ...`，没有 weight 乘法
3. **Integer indexing**：通过 `lora_index` 整数索引权重 buffer，不支持 interpolation

---

## 8. 实现：Step-Level Dynamic Router + vLLM Multi-LoRA Serving

> 实现时间：2026-03-07
> 基于方案 1（Hard Routing）扩展，解决 TextOnlyRouter 无法区分同一 trajectory 不同 step 的问题

### 8.1 问题

当前 `TextOnlyRouter` 仅用 instruction text 做 routing，同一 trajectory 的所有 step 共享相同的 instruction，导致 routing 结果完全相同。在 agentic GUI 场景中，不同 step 面对不同 UI 状态（截图不同、历史动作不同），应该路由到不同 expert LoRA。

### 8.2 整体架构

```
                           ┌─────────────────────────────────────────────┐
                           │          Phase 1: 训练 (ContextAwareRouter)  │
                           │                                             │
  (screenshot, instruction,│   Pass 1 (no LoRA)        Pass 2 (LoRA)    │
   action history)────────►│   ─────────────►          ─────────────►    │
                           │   hidden_states          logits + loss     │
                           │        │                                    │
                           │   ┌────┴────┐                               │
                           │   │ vision  │  mean pool vision tokens      │
                           │   │ mask    │──────────┐                    │
                           │   └─────────┘          │                    │
                           │   ┌─────────┐   ┌──────┴──────┐            │
                           │   │ text    │   │ContextAware │──► routing │
                           │   │ mask    │──►│ Router      │   weights  │
                           │   └─────────┘   └─────────────┘            │
                           └─────────────────────────────────────────────┘

                           ┌─────────────────────────────────────────────┐
                           │          Phase 2: 蒸馏 (StandaloneRouter)   │
                           │                                             │
  ContextAwareRouter ──────│──► routing labels ──► CE + KL loss          │
  decisions (teacher)      │                            │                │
                           │                    ┌───────┴───────┐       │
  screenshot ──────────────│──────────────────► │ SigLIP Router │       │
                           │                    │ (~100M params) │       │
                           │                    └───────────────┘       │
                           └─────────────────────────────────────────────┘

                           ┌─────────────────────────────────────────────┐
                           │          Phase 3: Serving (vLLM)            │
                           │                                             │
  screenshot ──────►┌──────────────┐                                     │
                    │Standalone    │ expert_idx                           │
                    │Router (<10ms)│──────────┐                          │
                    └──────────────┘          │                          │
                                      ┌──────┴──────┐                   │
  prompt ────────────────────────────►│ vLLM Engine  │──► response      │
                                      │ multi-LoRA   │                   │
                                      └─────────────┘                   │
                           └─────────────────────────────────────────────┘
```

### 8.3 向后兼容保证

所有改动通过 `router_type` 配置项切换，默认值 `'text_only'` 保证旧 config 零影响：

| 现有组件 | 是否修改 | 说明 |
|---------|---------|------|
| `TextOnlyRouter` | 不改 | 完整保留 |
| `InstructionFeatureExtractor` | 不改 | 复用于两种 router |
| `MoELoRALinear` | 不改 | routing weights 接口不变 |
| `MoEVLMWrapper` 现有逻辑 | 不改 | `router_type='text_only'` 走原有代码路径 |
| `create_instruction_mask*` | 不改 | 新增独立 mask 函数 |
| `serve_moe.py` | 不改 | 新建独立 serving 脚本 |
| 现有 config (v1/v2) | 不改 | `router_type` 默认 `'text_only'` |
| Expert checkpoint 格式 | 不改 | PEFT 格式通用于两种 router |

---

## 9. Phase 1: Context-Aware Router（训练用）

### 9.1 ContextAwareRouter — `verl/models/moe/router.py`（追加，不改已有代码）

利用 Pass 1 的 hidden states 同时提取 vision tokens 和 text tokens 的特征：

- **Vision features**: mean pool `<|vision_start|>` ~ `<|vision_end|>` tokens（截图特征，每步不同）
- **Text features**: mean pool `<|vision_end|>` 之后所有文本 tokens（instruction + history，每步不同）

```python
class ContextAwareRouter(nn.Module):
    """Router using vision + text features for step-level routing.
    Coexists with TextOnlyRouter, selected via MoEConfig.router_type."""

    def __init__(self, hidden_size, num_experts, router_hidden=256,
                 top_k=1, dropout=0.1, temperature=1.0, noise_std=0.0):
        super().__init__()
        self.vision_proj = nn.Linear(hidden_size, router_hidden)
        self.text_proj = nn.Linear(hidden_size, router_hidden)
        self.router = nn.Sequential(
            nn.Linear(router_hidden * 2, router_hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(router_hidden, num_experts),
        )

    def forward(self, vision_features, text_features) -> RouterOutput:
        """Takes two feature vectors, returns RouterOutput (same interface)."""
        v = self.vision_proj(vision_features)   # [B, router_hidden]
        t = self.text_proj(text_features)       # [B, router_hidden]
        combined = torch.cat([v, t], dim=-1)    # [B, router_hidden*2]
        router_logits = self.router(combined)   # [B, num_experts]

        # Add noise during training
        if self.training and self.noise_std > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.noise_std

        scaled_logits = router_logits / self.temperature
        routing_weights = F.softmax(scaled_logits, dim=-1)
        top_k_weights, top_k_indices = routing_weights.topk(self.top_k, dim=-1)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-10)

        return RouterOutput(
            routing_weights=routing_weights,
            top_k_weights=top_k_weights,
            top_k_indices=top_k_indices,
            router_logits=router_logits,
        )
```

### 9.2 Mask 函数 — `verl/models/moe/router.py`（追加到文件末尾）

现有 `create_instruction_mask` 和 `create_instruction_mask_from_text` 不动。

```python
def create_vision_mask(input_ids, tokenizer) -> torch.Tensor:
    """Mark <|vision_start|> to <|vision_end|> tokens (inclusive).

    For Qwen2.5-VL:
        <|vision_start|><|image_pad|>...<|vision_end|>
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ marked True
    """
    # 遍历 token IDs，在 vision_start 和 vision_end 之间设置 True
    # 支持 batch，每个样本独立处理


def create_text_context_mask(input_ids, tokenizer) -> torch.Tensor:
    """Mark all text tokens after last <|vision_end|> until <|im_end|>.

    Covers instruction + action history for the current turn.
    Uses LAST vision_end (for multi-turn scenarios).
    """
    # 找到最后一个 vision_end 的位置，标记从那里到 im_end 的所有 token
```

### 9.3 MoEConfig 扩展 — `verl/models/moe/moe_wrapper.py`

仅新增一个字段，默认值保证向后兼容：

```python
@dataclass
class MoEConfig:
    # ... 所有现有字段不变 ...

    # Router type: 'text_only' (original) or 'context_aware' (vision+text)
    router_type: str = 'text_only'

    def to_dict(self) -> dict:
        """使用 __dataclass_fields__ 自动序列化所有字段。"""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "MoEConfig":
        """只取已知字段，忽略未知 key（向后兼容）。"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
```

**向后兼容验证**：

```python
# V2 config dict（没有 router_type）能正常加载，默认为 text_only
v2_dict = {'num_experts': 6, 'top_k': 6, 'expert_lora_r': 16}
cfg = MoEConfig.from_dict(v2_dict)
assert cfg.router_type == 'text_only'  # ✅

# 带 unknown key 也能正常加载
cfg = MoEConfig.from_dict({'num_experts': 4, 'unknown_key': 'foo'})
assert cfg.num_experts == 4  # ✅
```

### 9.4 MoEVLMWrapper 条件分支 — `verl/models/moe/moe_wrapper.py`

#### `_init_moe_components()`:

```python
def _init_moe_components(self):
    config = self.moe_config

    # Router (selected by router_type)
    if config.router_type == 'context_aware':
        self.router = ContextAwareRouter(
            hidden_size=self.hidden_size,
            num_experts=config.num_experts,
            router_hidden=config.router_hidden,
            top_k=config.top_k,
            dropout=config.router_dropout,
            temperature=config.router_temperature,
        )
    else:
        # Default: text_only (original behavior, unchanged)
        self.router = TextOnlyRouter(...)

    # Feature Extractors
    self.feature_extractor = InstructionFeatureExtractor(
        pooling_strategy=config.pooling_strategy,
    )
    if config.router_type == 'context_aware':
        # Separate extractor for vision features
        self.vision_feature_extractor = InstructionFeatureExtractor(
            pooling_strategy=config.pooling_strategy,
        )

    # MoE Loss + Module Replacement ... (unchanged)
```

#### `forward()` / `generate()` — 条件分支:

```python
# Pass 1 后的 routing 逻辑
if self.moe_config.router_type == 'context_aware':
    # === 新增分支 ===
    vision_mask = create_vision_mask(input_ids, self.tokenizer)
    text_mask = create_text_context_mask(input_ids, self.tokenizer)
    vision_features = self.vision_feature_extractor(hidden_states, vision_mask)
    text_features = self.feature_extractor(hidden_states, text_mask)
    router_output = self.router(vision_features, text_features)
else:
    # === 完全保留现有逻辑（text_only）===
    instruction_features = self.feature_extractor(hidden_states, instruction_mask)
    router_output = self.router(instruction_features)
```

`text_only` 分支的代码路径完全不变。

### 9.5 Checkpoint 保存/加载

- `save_moe_checkpoint()`: `router_type` 随 `moe_config.json` 自动保存（`to_dict()` 已覆盖）
- `load_moe_checkpoint()`: `router.pt` 的 state_dict key 不同（`TextOnlyRouter` 只有 `router.*`，`ContextAwareRouter` 有 `vision_proj.*`, `text_proj.*`, `router.*`），`load_state_dict` 自然处理

### 9.6 训练 Config — `train_GUI_360/moe_sft/moe_sft_config_v3.yaml`

```yaml
# Key changes from v2:
#   - router_type: text_only → context_aware
#   - top_k: 6 → 1 (hard routing for vLLM compatibility)

moe:
  num_experts: 6
  top_k: 1                         # hard routing for vLLM compatibility
  router_type: "context_aware"     # NEW: vision+text step-level routing
  expert_lora_r: 16
  expert_lora_alpha: 32
  target_modules: ["q_proj", "v_proj"]
  router_hidden: 256
  balance_weight: 0.1
  balance_type: "entropy"
  z_loss_weight: 0.01
  # 其余参数继承 v2
```

---

## 10. Phase 2: Standalone Router 蒸馏（Serving 用）

### 10.1 原理

ContextAwareRouter 依赖 7B base model 的 hidden states（需要 Pass 1），不适合直接在 vLLM serving 中使用。蒸馏到独立的轻量 SigLIP router (~100M params)，仅需截图即可预测 expert。

### 10.2 生成蒸馏标签 — `train_GUI_360/moe_sft/generate_routing_labels.py`

```bash
python train_GUI_360/moe_sft/generate_routing_labels.py \
    --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
    --moe_checkpoint train_GUI_360/moe_sft/output/moe_sft_v3/final \
    --data_file train_GUI_360/llamafactory/data/gui360_train.json \
    --output routing_labels.jsonl
```

流程：
1. 加载训练好的 MoE 模型（含 ContextAwareRouter）
2. 对训练样本跑 Pass 1，记录 hard routing label + soft routing distribution
3. 输出 JSONL:

```json
{
    "image_path": "path/to/screenshot.png",
    "instruction_text": "Click on the search button",
    "expert_label": 3,
    "routing_weights": [0.01, 0.02, 0.85, 0.05, 0.04, 0.03]
}
```

### 10.3 StandaloneRouter — `verl/models/moe/standalone_router.py`

```python
class StandaloneRouter(nn.Module):
    """Lightweight SigLIP-based router for vLLM serving.
    Independent of base model. ~100M params."""

    def __init__(self, num_experts, vision_model="google/siglip-base-patch16-224"):
        self.vision_encoder = SiglipVisionModel.from_pretrained(vision_model)  # frozen
        self.vision_proj = nn.Linear(vision_hidden, 256)
        self.classifier = nn.Sequential(
            nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, num_experts),
        )

    def predict(self, image: PIL.Image) -> int:
        """Returns expert index for a single screenshot. <10ms on GPU."""
        pixel_values = self.processor(image)
        features = self.vision_encoder(pixel_values).pooler_output
        logits = self.classifier(self.vision_proj(features))
        return logits.argmax(dim=-1).item()

    def predict_with_distribution(self, image) -> tuple:
        """Returns (expert_idx, routing_weights) for analysis."""

    def save(self, save_dir):
        """Save router_head.pt + config.json (vision encoder loaded from HF)."""

    @classmethod
    def load(cls, load_dir, device='cpu') -> "StandaloneRouter":
        """Load from checkpoint."""
```

### 10.4 蒸馏训练 — `train_GUI_360/moe_sft/train_standalone_router.py`

```bash
python train_GUI_360/moe_sft/train_standalone_router.py \
    --labels routing_labels.jsonl \
    --output_dir standalone_router_checkpoint \
    --vision_model google/siglip-base-patch16-224 \
    --num_experts 6 \
    --epochs 10 \
    --lr 1e-4 \
    --batch_size 32 \
    --kl_weight 0.5
```

Loss 设计：
- **CE loss**: 对 hard labels (argmax) 做分类
- **KL divergence**: 对 soft routing distribution 做知识蒸馏
- `total_loss = CE + kl_weight * KL`

目标 agreement rate >90%。

---

## 11. Phase 3: vLLM Multi-LoRA Serving

### 11.1 Serving 脚本 — `train_GUI_360/moe_sft/serve_vllm_moe.py`

现有 `serve_moe.py`（HF 推理版）保留不动。新建独立 vLLM serving 脚本。

```bash
python train_GUI_360/moe_sft/serve_vllm_moe.py \
    --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
    --moe_checkpoint train_GUI_360/moe_sft/output/moe_sft_v3/final \
    --standalone_router standalone_router_checkpoint/best \
    --port 19810
```

核心逻辑：

```python
# 1. 加载 standalone router (SigLIP, ~100M params, 独立于 vLLM)
standalone_router = StandaloneRouter.load("checkpoint/standalone_router/")

# 2. 启动 vLLM with multi-LoRA (现有 expert checkpoint 直接用)
llm = LLM(model="base_model_path", enable_lora=True, max_loras=8)
expert_lora_requests = [
    LoRARequest(f"expert_{i}", i+1, f"checkpoint/experts/expert_{i}")
    for i in range(6)
]

# 3. Per-step routing: 每个 request 动态选择 LoRA
@app.post("/v1/chat/completions")
async def generate(request):
    screenshot = extract_screenshot(request)
    expert_idx = standalone_router.predict(screenshot)  # <10ms
    output = llm.generate(prompt, params,
                          lora_request=expert_lora_requests[expert_idx])
    return output
```

### 11.2 API 兼容性

与 `serve_moe.py` 使用相同的 OpenAI-compatible API：

```
GET  /health              → {"status": "ok", "engine": "vllm"}
GET  /v1/models           → 列出 base model + 所有 expert LoRA
POST /v1/chat/completions → 生成 (自动 per-step routing)
```

响应额外包含 `routing_info`:

```json
{
    "choices": [{"message": {"content": "..."}}],
    "routing_info": {
        "expert_idx": 3,
        "routing_weights": [0.01, 0.02, 0.85, 0.05, 0.04, 0.03]
    }
}
```

### 11.3 性能预期

| 指标 | HF serve_moe.py | vLLM serve_vllm_moe.py |
|------|-----------------|------------------------|
| 推理速度 | ~2.7s/sample | ~0.3s/sample |
| Router 延迟 | ~50ms (Pass 1 of 7B) | <10ms (SigLIP) |
| 完整评估 | ~41 小时 | ~4.5 小时 |
| GPU 内存 | ~28GB (7B × 2 pass) | ~16GB (vLLM + LoRA) |

---

## 12. 完整文件清单

| 操作 | 文件 | 说明 |
|------|------|------|
| 追加代码 | `verl/models/moe/router.py` | `ContextAwareRouter` + `create_vision_mask` + `create_text_context_mask` |
| 小幅修改 | `verl/models/moe/moe_wrapper.py` | `MoEConfig` 加 `router_type`, `_init_moe_components`/`forward`/`generate` 加条件分支 |
| 小幅修改 | `verl/models/moe/__init__.py` | 导出新类 |
| 新建 | `verl/models/moe/standalone_router.py` | SigLIP standalone router (~100M params) |
| 新建 | `train_GUI_360/moe_sft/moe_sft_config_v3.yaml` | Context-aware router 训练配置 |
| 新建 | `train_GUI_360/moe_sft/generate_routing_labels.py` | 生成蒸馏标签 |
| 新建 | `train_GUI_360/moe_sft/train_standalone_router.py` | 蒸馏训练 standalone router |
| 新建 | `train_GUI_360/moe_sft/serve_vllm_moe.py` | vLLM multi-LoRA per-step routing server |

---

## 13. 验证方案

### 13.1 回归验证

用现有 v2 config (`router_type='text_only'`) 跑 forward，确认输出与修改前完全一致：

```python
cfg = MoEConfig.from_dict({'num_experts': 6, 'top_k': 6, 'expert_lora_r': 16})
assert cfg.router_type == 'text_only'
# forward 走完全相同的代码路径 ✅
```

### 13.2 Phase 1 验证

训练后检查同一 trajectory 内不同 step 的 routing diversity：

```python
# 同一 instruction，不同 screenshot → routing 应该不同
step1_routing = router(vision_feat_step1, text_feat_step1)
step2_routing = router(vision_feat_step2, text_feat_step2)
assert step1_routing.top_k_indices != step2_routing.top_k_indices  # 大概率不同
```

### 13.3 Phase 2 验证

Standalone router 与 context-aware router agreement rate >90%：

```python
agreement = (standalone_labels == context_aware_labels).float().mean()
assert agreement > 0.9
```

### 13.4 Phase 3 验证

vLLM 端到端评估，速度目标 ~0.3s/sample：

```bash
# 启动 vLLM server
python serve_vllm_moe.py --base_model ... --moe_checkpoint ... --standalone_router ...

# 评估
python eval_scripts/eval_moe_v3.py --api_url http://localhost:19810
```

---

## 13.5 训推不匹配（Train-Inference Mismatch）分析与缓解

> 更新时间：2026-03-07

v3 架构在训练（SFT/RL 使用 `MoEVLMWrapper`）和推理（vLLM multi-LoRA + `StandaloneRouter`）之间存在 3 个潜在不匹配，如果不加处理可能导致性能下降。

### 13.5.1 Mismatch 1: Soft Routing（训练）vs Hard Routing（推理）

**问题描述：**

```
训练时 (MoELoRALinear.forward()):
  output = base_out + Σ_i(weight_i * expert_i(x))   # 6 个 expert 全部参与
  weight_i 来自完整 softmax 分布，e.g. [0.05, 0.72, 0.08, 0.05, 0.05, 0.05]

推理时 (vLLM LoRARequest):
  output = base_out + expert_top1(x)                  # 只有 1 个 expert
  等价于 weight = [0, 1, 0, 0, 0, 0]（hard argmax）
```

**影响程度：** 取决于 routing 分布的 sharpness：
- 如果 top-1 weight > 0.9（sharp routing）：其余 expert 贡献 < 10%，影响小
- 如果 top-1 weight ≈ 0.3（flat routing）：丢失 70% 的 expert 贡献，影响大

**当前缓解（已部署）：**
- `balance_type: entropy` + `balance_weight: 0.1`：entropy balance loss 惩罚均匀分布，鼓励 router 产生 sharp（low-entropy）routing decisions
- `top_k: 1` 配置：虽然训练时实际使用 soft routing，但 config 语义上表达了 "期望 hard routing" 的意图

**额外缓解方案（可选）：**

| 方案 | 描述 | 开销 | 效果 |
|------|------|------|------|
| **Gumbel-Softmax Anneal** | 训练后期逐步从 soft routing 过渡到 hard routing（temperature τ: 1.0→0.1）| 需要 annealing schedule + 额外超参 | 理论最优，但增加训练复杂度 |
| **Top-1 Straight-Through** | 训练时 forward 用 top-1 hard routing，backward 用 soft gradient（STE） | 修改 `MoELoRALinear.forward()` | 训练推理完全一致，但梯度近似可能不稳定 |
| **训练后 calibration** | 训练完成后在验证集上测量 soft vs hard 差异，如果差异小则无需处理 | 一次性评估开销 | 最简单，先做 Step 2 HF 评估时顺带测量 |

**推荐策略：** 先不做额外处理。在 Step 2（HF 评估）时同时测量：
1. 每个样本的 top-1 routing weight 分布（直方图）
2. soft-routing 推理结果 vs hard-routing 推理结果的一致率

如果一致率 > 95%，无需额外处理。如果 < 90%，考虑 Gumbel-Softmax Anneal 重新训练。

### 13.5.2 Mismatch 2: ContextAwareRouter（训练）vs StandaloneRouter（推理）

**问题描述：**

```
训练时 (ContextAwareRouter):
  输入: base model hidden states (3584-dim)
  ├── vision_features: mean pool <|vision_start|>~<|vision_end|> tokens
  │   → 包含模型理解的语义视觉特征
  └── text_features: mean pool text tokens after <|vision_end|>
      → 包含 instruction + action history 的完整上下文

推理时 (StandaloneRouter):
  输入: raw screenshot pixels → SigLIP encoder (768-dim)
  └── 仅有视觉特征，无文本上下文
```

**影响分析：**
- **信息丢失**：StandaloneRouter 无法利用 instruction text 和 action history。如果 ContextAwareRouter 的 routing 决策强依赖 text 信息（e.g. "搜索 xxx" vs "打开 xxx" → 不同 expert），StandaloneRouter 无法复制这些决策
- **特征空间差异**：ContextAwareRouter 使用 7B 模型的 hidden states（更丰富的语义表示），StandaloneRouter 使用 SigLIP 的 pooled output（更偏向视觉表面特征）
- **预期差距**：对于纯视觉区分（不同 app/界面 → 不同 expert），StandaloneRouter 应能很好复制；对于上下文相关决策（同一界面、不同指令 → 不同 expert），可能存在差距

**当前缓解（Phase 2 蒸馏设计中已包含）：**
- CE loss on hard labels + KL divergence on soft distribution
- 目标 agreement rate > 90%

**额外缓解方案（可选）：**

| 方案 | 描述 | 开销 | 效果 |
|------|------|------|------|
| **Multi-modal StandaloneRouter** | SigLIP vision + SigLIP text encoder（输入 instruction text） | 增加 ~20M params，需要文本输入 | 能部分恢复 text context 信息 |
| **Unfreeze SigLIP** | 蒸馏时 fine-tune SigLIP vision encoder（`--no_freeze_vision`，lr × 0.1） | 训练时间 ×2-3 | 让视觉编码器学习 routing-relevant 特征 |
| **增加蒸馏数据** | 用更多 screenshots（含 augmentation）增加蒸馏样本 | 数据准备开销 | 提升 StandaloneRouter 泛化能力 |
| **Ensemble fallback** | 推理时如果 StandaloneRouter confidence < threshold，fallback 到 HF 模式的 ContextAwareRouter | 部分请求延迟增加 | 高置信度请求用 vLLM 快速推理，低置信度用 HF 精确推理 |

**推荐策略：** 先用 Phase 2 标准蒸馏。Step 4 完成后测量 agreement rate：
- \> 95%：无需额外处理
- 90%-95%：尝试 unfreeze SigLIP
- < 90%：考虑 multi-modal StandaloneRouter（加 text encoder）

### 13.5.3 Mismatch 3: Scaling Factor 差异

**问题描述：**

```
训练时 (MoELoRALinear):
  delta = Σ_i weight_i * (lora_B_i @ lora_A_i) * (alpha / r) * x
        = Σ_i weight_i * (lora_B_i @ lora_A_i) * scaling * x
  其中 weight_i 来自 routing，scaling = alpha / r = 32 / 16 = 2.0

推理时 (vLLM PEFT adapter):
  delta = lora_B_top1 @ lora_A_top1 * peft_scaling * x
  其中 peft_scaling 由 adapter_config.json 中的 lora_alpha / r 决定
```

**潜在问题：**
- 训练时 expert delta 被 `weight_i` 缩放（e.g. weight_top1 = 0.72），所以模型学到的 base + 0.72 × delta 的表现
- 推理时 vLLM 应用整个 LoRA delta（等价于 weight = 1.0），delta 的贡献变为 1.0 / 0.72 ≈ 1.39 倍

**影响评估：**
- 如果 soft routing 下 top-1 weight 接近 1.0：影响极小（1.0 / 0.95 ≈ 1.05 倍）
- 如果 top-1 weight 较低（0.5-0.7）：影响可能较大（1.0 / 0.6 ≈ 1.67 倍放大）
- 这与 Mismatch 1 相关：soft routing 越 sharp，scaling 差异越小

**缓解方案：**

| 方案 | 描述 | 开销 | 效果 |
|------|------|------|------|
| **适配 vLLM LoRA scaling** | 推理时将 `lora_alpha` 乘以训练时的平均 top-1 weight | 修改 `adapter_config.json` | 简单有效，但是全局 scaling，不够精确 |
| **Per-expert calibration** | 对每个 expert 单独测量平均 routing weight，写入各自的 `adapter_config.json` | 需要统计 + 修改 config | 更精确 |
| **训练端解决（推荐）** | 确保训练时 entropy balance 使 top-1 weight → 0.9+，则 scaling 差异 < 11% | 已有 entropy balance | 最自然的方式 |

**推荐策略：** 训练端通过 entropy balance loss 鼓励 sharp routing（top-1 weight > 0.9），使 scaling 差异自然缩小。Step 2 评估时验证 top-1 weight 分布。

### 13.5.4 三个 Mismatch 的交互关系

```
                    Mismatch 1              Mismatch 3
                 (soft vs hard routing)   (scaling factor)
                          │                     │
                          │    top-1 weight      │
                          │    越接近 1.0         │
                          │    两者影响都越小      │
                          └────────┬─────────────┘
                                   │
                          entropy balance loss
                          (已配置 weight=0.1)
                                   │
                                   ▼
                          训练出 sharp routing → 同时缓解 1 和 3

                    Mismatch 2
                 (ContextAware vs Standalone Router)
                          │
                          │    独立于 1 和 3
                          │    通过蒸馏 agreement rate 衡量
                          │
                          ▼
                    KL + CE 蒸馏 → 目标 agreement > 90%
```

**核心观察**：Mismatch 1 和 3 的严重程度都由 routing distribution 的 sharpness 决定。entropy balance loss 同时缓解两者。Mismatch 2 是独立问题，由蒸馏质量决定。

### 13.5.5 验证清单

在执行计划各 Step 中需要检查的 mismatch 指标：

| Step | 验证内容 | 通过标准 | 不通过时的处理 |
|------|---------|---------|--------------|
| Step 1（SFT 训练） | WandB 观察 `routing/expert_max_weight` | 训练末期 mean > 0.8 | 增大 `balance_weight` 或改用 Gumbel anneal |
| Step 2（HF 评估） | 同时跑 soft-routing 和 hard-routing 推理，对比结果 | 一致率 > 95% | 考虑训练时加 top-1 STE 或 Gumbel anneal |
| Step 2（HF 评估） | 统计 top-1 routing weight 分布 | P50 > 0.85, P10 > 0.6 | 增大 entropy balance weight |
| Step 4（蒸馏训练） | Standalone Router agreement rate | > 90% | unfreeze SigLIP；或加 text encoder |
| Step 5（vLLM 评估） | 对比 vLLM(hard+standalone) vs HF(soft+context-aware) | 分数差距 < 3% | 根据主要来源分别处理 1/2/3 |

---

## 14. 详细执行计划

> 更新时间：2026-03-07

### 当前状态

| 阶段 | 代码实现 | 训练/执行 |
|------|---------|----------|
| Phase 1: ContextAwareRouter | ✅ 已完成 | ❌ 未训练 |
| Phase 2: Standalone Router 蒸馏 | ✅ 已完成 | ❌ 未执行（依赖 Phase 1 产物） |
| Phase 3: vLLM Serving | ✅ 已完成 | ❌ 未执行（依赖 Phase 2 产物） |

**关键发现**：现有 MoE v1/v2 训练产物（final checkpoint）已不在磁盘上，`moe_sft_v1_copy_init/` 也不存在。v3 的 ContextAwareRouter 参数结构与 TextOnlyRouter 完全不同（`vision_proj.*`, `text_proj.*` vs `router.*`），因此 **v3 router 必须从零训练**。Expert LoRA 可以从零开始（Kaiming/zeros 初始化），配合 `balance_weight=0.1` + `z_loss_weight=0.01` 避免 v1 的 collapse 问题。

---

### Step 1: 创建 v3 训练 Slurm 脚本

**文件**：`train_GUI_360/moe_sft/train_moe_sft_v3.slurm`

**改动点**（相对 `train_moe_sft_v2.slurm`）：
- `CONFIG_FILE` → `moe_sft_config_v3.yaml`
- **去掉** `MOE_INIT_CHECKPOINT` 和 copy-init 验证（从零训练）
- **去掉** torchrun 的 `--moe_init_checkpoint` 参数
- job-name → `moe_sft_v3`
- 日志文件名 → `moe_sft_v3_*.log`

**资源**：4 nodes × 4 GPUs = 16 GPUs，24h，partition=workq（与 v2 相同）

**产物**：
```
train_GUI_360/moe_sft/output/moe_sft_v3/
├── checkpoint-200/
├── checkpoint-400/
├── ...
└── final/
    ├── router.pt              # ContextAwareRouter state_dict
    ├── moe_config.json        # 含 router_type: "context_aware"
    └── experts/
        ├── expert_0/adapter_model.bin
        └── ...expert_5/adapter_model.bin
```

**预计耗时**：12-24h（3 epochs，与 v2 相同数据量）

**验证要点**：
- WandB 观察 `routing_entropy` > 0.5（v1 collapse 时 ≈ 0.0002）
- 6 个 expert 的 utilization 均 > 5%（不是全集中在一个 expert）
- 同一 trajectory 不同 step 的 `top_k_indices` 有变化（v3 的核心目标）

---

### Step 2: 评估 v3 训练效果（用 HF serve_moe.py）

**目的**：在蒸馏之前，先确认 ContextAwareRouter + Expert LoRA 的质量

**文件**：`train_GUI_360/moe_sft/eval_moe_v3.slurm`（新建，基于 `eval_moe_v2.slurm`）

**改动点**（相对 `eval_moe_v2.slurm`）：
- `MOE_CHECKPOINT` → `output/moe_sft_v3/final`
- `RESULTS_DIR` → `output/eval_moe_v3`
- job-name → `eval_moe_v3`
- 其余不变（仍用 `serve_moe.py`，因为它已支持 `router_type='context_aware'`）

**流程**：
1. 启动 `serve_moe.py`（HF 推理，含两遍 forward）
2. 跑 grounding / action_prediction / action_prediction_a11y 三个评估
3. 结果对比 v1 / v2 / SFT-398 baseline

**资源**：1 node × 4 GPUs，24h（eval 环境用 `qwen3-eval` conda env）

**预计耗时**：~41h（HF 推理慢，~2.7s/sample）。但作为 **一次性质量验证**，可接受。

**Gate**：如果 v3 评估质量不如 v2 或 SFT baseline，需要回到 Step 1 调整超参（增大 top_k、调整 balance_weight 等）。

---

### Step 3: 生成蒸馏标签

**前置条件**：Step 2 评估通过，确认 v3 训练质量 OK

**文件**：`train_GUI_360/moe_sft/generate_routing_labels.slurm`（新建）

**命令**：
```bash
python train_GUI_360/moe_sft/generate_routing_labels.py \
    --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
    --moe_checkpoint train_GUI_360/moe_sft/output/moe_sft_v3/final \
    --data_file train_GUI_360/llamafactory/data/gui360_train.json \
    --output train_GUI_360/moe_sft/output/routing_labels_v3.jsonl
```

**资源**：1 node × 4 GPUs（需加载 7B 模型跑 Pass 1），4-8h

**产物**：
```
train_GUI_360/moe_sft/output/routing_labels_v3.jsonl
# 每行: {"image_path": ..., "expert_label": 3, "routing_weights": [...]}
```

**验证**：
- 检查 expert 分布：6 个 expert 的样本数应大致均匀（最少的 > 5%）
- 检查 routing confidence：大部分样本 top-1 weight > 0.5（router 有明确决策）
- 样本数应 ≈ 训练集大小

---

### Step 4: 训练 Standalone Router（蒸馏）

**前置条件**：Step 3 生成完标签

**预检查**：确认集群环境有 SigLIP 模型访问

```bash
# 在 slurm job 中预下载（如果没有 HF 网络访问）
python -c "from transformers import SiglipVisionModel; SiglipVisionModel.from_pretrained('google/siglip-base-patch16-224')"
```

如果集群无法下载，需先在有网络的机器下载到 `checkpoints/siglip-base-patch16-224/`，然后改 `--vision_model` 为本地路径。

**文件**：`train_GUI_360/moe_sft/train_standalone_router.slurm`（新建）

**命令**：
```bash
python train_GUI_360/moe_sft/train_standalone_router.py \
    --labels train_GUI_360/moe_sft/output/routing_labels_v3.jsonl \
    --output_dir train_GUI_360/moe_sft/output/standalone_router_v3 \
    --vision_model google/siglip-base-patch16-224 \
    --num_experts 6 \
    --epochs 10 \
    --lr 1e-4 \
    --batch_size 32 \
    --kl_weight 0.5
```

**资源**：1 node × 1 GPU 足够（SigLIP ~100M params），1-2h

**产物**：
```
train_GUI_360/moe_sft/output/standalone_router_v3/
├── best/
│   ├── router_head.pt
│   └── config.json
├── final/
│   ├── router_head.pt
│   └── config.json
└── training_summary.json    # 含 best_val_acc
```

**验证**：
- `best_val_acc` > 90%（standalone router 与 ContextAwareRouter 决策一致性）
- 如果 < 90%，考虑：unfreeze SigLIP（`--no_freeze_vision`）、增大 epochs、降低 lr

---

### Step 5: vLLM Multi-LoRA 端到端评估

**前置条件**：Step 4 standalone router 训练完成，agreement > 90%

**预检查**：确认 eval 环境有 vLLM

```bash
conda run -n qwen3-eval python -c "import vllm; print(vllm.__version__)"
```

如果没有 vLLM，需要安装：`pip install vllm`

**文件**：`train_GUI_360/moe_sft/eval_moe_v3_vllm.slurm`（新建）

**流程**：
1. 启动 `serve_vllm_moe.py`（vLLM + standalone router）
2. 等待 server ready（检查 `/health`）
3. 跑 grounding / action_prediction / action_prediction_a11y
4. 记录推理速度

**命令**：
```bash
# Server
python train_GUI_360/moe_sft/serve_vllm_moe.py \
    --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
    --moe_checkpoint train_GUI_360/moe_sft/output/moe_sft_v3/final \
    --standalone_router train_GUI_360/moe_sft/output/standalone_router_v3/best \
    --port 19810 \
    --gpu_memory_utilization 0.9

# Eval (与 v2 相同的 evaluation.py)
cd train_GUI_360/GUI-360-eval
python evaluation.py \
    --root_dir $DATASET_ROOT \
    --type grounding \
    --model_type qwen2.5_vl_7b \
    --model_name moe_v3_vllm \
    --api_url http://localhost:19810/v1 \
    --threads 4    # vLLM 可以并发，比 HF 模式的 threads=1 快
```

**资源**：1 node × 4 GPUs，4-8h（vLLM ~0.3s/sample，比 HF 快 ~9x）

**产物**：
```
train_GUI_360/moe_sft/output/eval_moe_v3_vllm/
├── grounding_*/
├── action_prediction_*/
└── action_prediction_a11y_*/
```

**验证**：
- v3 (vLLM) 评估分数 ≈ v3 (HF serve_moe.py) 评估分数（验证 standalone router 蒸馏没有明显掉点）
- 推理速度 < 0.5s/sample（vLLM multi-LoRA）
- 完整评估 < 6h

---

### 依赖关系与时间线

```
                    ┌──────────┐
                    │  Step 1  │  创建 v3 训练 slurm
                    │  (~30min)│  train_moe_sft_v3.slurm
                    └────┬─────┘
                         │ sbatch
                         ▼
                    ┌──────────┐
                    │ 训练 v3   │  4 nodes × 4 GPUs × 24h
                    │ (12-24h) │
                    └────┬─────┘
                         │ output/moe_sft_v3/final/
                    ┌────┴─────┐
                    │          │
               ┌────▼────┐    │
               │ Step 2  │    │
               │ HF 评估  │    │
               │ (24-41h)│    │
               └────┬────┘    │
                    │         │
                    │ GATE: 质量 OK?
                    │ ├─ YES ─┤
                    │         │
                    │    ┌────▼────┐
                    │    │ Step 3  │  生成蒸馏标签
                    │    │ (4-8h)  │
                    │    └────┬────┘
                    │         │ routing_labels_v3.jsonl
                    │    ┌────▼────┐
                    │    │ Step 4  │  训练 standalone router
                    │    │ (1-2h)  │
                    │    └────┬────┘
                    │         │ standalone_router_v3/best/
                    │    ┌────▼────┐
                    │    │ Step 5  │  vLLM 端到端评估
                    │    │ (4-8h)  │
                    │    └────┬────┘
                    │         │
                    │    评估对比:
                    │    v3(vLLM) ≈ v3(HF) ≈ v2 ?
                    │
               ├─ NO → 调参回到 Step 1
```

**乐观路径总耗时**：~2-3 天
- Day 1: Step 1 (30min) → 提交训练（12-24h 排队+运行）
- Day 2: Step 2 (HF 评估) 与 Step 3 (标签生成) 可并行提交
- Day 3: Step 4 (1-2h) → Step 5 (4-8h)

**注意**：Step 2（HF 评估）和 Step 3（标签生成）可以 **并行提交**，因为它们都只依赖 Step 1 产物且都是只读操作。

---

### 需要新建的文件清单

| 文件 | 目的 | 基于模板 |
|------|------|---------|
| `train_GUI_360/moe_sft/train_moe_sft_v3.slurm` | v3 训练 | `train_moe_sft_v2.slurm`，去掉 copy-init |
| `train_GUI_360/moe_sft/eval_moe_v3.slurm` | v3 HF 评估 | `eval_moe_v2.slurm`，改路径 |
| `train_GUI_360/moe_sft/generate_routing_labels.slurm` | 标签生成 | 新建，1 node × 4 GPU |
| `train_GUI_360/moe_sft/train_standalone_router.slurm` | 蒸馏训练 | 新建，1 node × 1 GPU |
| `train_GUI_360/moe_sft/eval_moe_v3_vllm.slurm` | vLLM 评估 | `eval_moe_v2.slurm`，改用 serve_vllm_moe.py |

---

### 风险与备选方案

| 风险 | 影响 | 备选方案 |
|------|------|---------|
| v3 训练 router collapse | Step 2 评估差 | 增大 `balance_weight` 到 0.2-0.5；或改 `top_k=2`（多 expert 参与） |
| v3 训练质量不如 v2 | Step 2 评估差 | top_k=1 可能太激进；改回 `top_k=6` soft routing，Phase 3 用方案 2（per-request 权重合并）替代 hard routing |
| SigLIP 模型下载失败 | Step 4 阻塞 | 提前在有网络的节点下载到 `checkpoints/` |
| Standalone router agreement < 90% | Step 5 掉点 | unfreeze SigLIP fine-tune；或增加 text encoder（SigLIP text branch）做 multi-modal 蒸馏 |
| vLLM 不支持 Qwen2.5-VL LoRA | Step 5 阻塞 | 检查 vLLM 版本 ≥ 0.4.0；或退回用 `serve_moe.py`（HF 推理）+ batch 优化 |
| 集群排队时间长 | 整体延迟 | Step 2 & 3 并行提交；减少 Step 2 的 eval 范围（只跑 grounding） |

---

## 15. Phase 4: RL 训练集成 — MoE + f_pseudo Expert Specialization

> 更新时间：2026-03-07

### 15.1 背景与动机

MoE 的核心价值在于 **expert 特化**：不同 expert 擅长处理不同类型的 UI 状态/任务。但纯 SFT 训练只能通过数据分布隐式地引导 expert 特化，缺乏显式信号告诉 router "哪些 step 是关键的、应该路由到特定 expert"。

**f_pseudo 的关键角色**：`f_pseudo(t) = f(s_t) - f(s_{t+1})` 来自图谱分析的 eigenfunction，提供了每个 step 的 **状态转移重要性信号**：
- `f_pseudo >> 0`：瓶颈跨越步（从高 f 区域到低 f 区域），跨越 connectivity bottleneck
- `f_pseudo ≈ 0`：区域内常规移动步
- `f_pseudo << 0`：远离目标方向的步

**这个信号天然适合做 MoE expert 特化的 reward shaping**：
- 瓶颈跨越步是 long-horizon 任务的关键，需要特定 expert 专门处理
- f_pseudo bonus 使得被 router 选中处理瓶颈步的 expert 获得更强的正向梯度
- 随着训练推进，某些 expert 自然特化为"瓶颈跨越 expert"，其余 expert 特化为"区域内导航 expert"

**目标**：
1. 将 ContextAwareRouter 集成到 RL 训练
2. 利用 f_pseudo reward 引导 expert 特化（不同 expert 擅长不同类型的状态转移）
3. Router 学会识别瓶颈步并路由到对应的特化 expert

### 15.1.1 f_pseudo 如何驱动 Expert 特化

```
                    ┌──────────────────────────────────────────────────────┐
                    │     f_pseudo → Expert Specialization 机制             │
                    │                                                      │
                    │  Step t: screenshot_t + action_history               │
                    │     │                                                │
                    │     ▼                                                │
                    │  ContextAwareRouter(vision_feat, text_feat)          │
                    │     │                                                │
                    │     ▼ 选择 expert_k                                  │
                    │                                                      │
                    │  Expert_k 生成 action                                │
                    │     │                                                │
                    │     ▼                                                │
                    │  Reward = r_action_match + λ * f_pseudo(t)           │
                    │           ~~~~~~~~~~~~~~~   ~~~~~~~~~~~~~~~~         │
                    │           base reward        expert 特化信号          │
                    │                                                      │
                    │  梯度流向:                                            │
                    │  ┌─────────────────────────────────────────────────┐ │
                    │  │ f_pseudo > 0 (瓶颈跨越步)                       │ │
                    │  │   → reward ↑ → expert_k 得到正向强化             │ │
                    │  │   → router 学会: 遇到类似截图 → 选 expert_k     │ │
                    │  │                                                  │ │
                    │  │ f_pseudo ≈ 0 (常规步)                            │ │
                    │  │   → reward ≈ r_base → 正常训练                   │ │
                    │  │                                                  │ │
                    │  │ f_pseudo < 0 (错误方向步)                         │ │
                    │  │   → reward ↓ → expert_k 受到抑制                 │ │
                    │  │   → router 学会: 类似截图 → 换别的 expert        │ │
                    │  └─────────────────────────────────────────────────┘ │
                    │                                                      │
                    │  训练收敛后的预期 expert 分工:                         │
                    │                                                      │
                    │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │
                    │  │Expert 0  │ │Expert 1  │ │Expert 2  │ ...        │
                    │  │常规导航   │ │瓶颈跨越   │ │文本输入   │            │
                    │  │click/    │ │多步复杂   │ │type/     │            │
                    │  │scroll    │ │导航序列   │ │answer    │            │
                    │  │f_pseudo≈0│ │f_pseudo>0 │ │f_pseudo≈0│            │
                    │  └──────────┘ └──────────┘ └──────────┘            │
                    └──────────────────────────────────────────────────────┘
```

与纯 SFT MoE 对比：

| 方面 | SFT MoE (Phase 1) | RL MoE + f_pseudo (Phase 4) |
|------|-------------------|----------------------------|
| Expert 特化信号 | 隐式（数据分布） | **显式**（f_pseudo reward shaping） |
| Router 优化信号 | CE loss on ground truth | **Policy gradient + f_pseudo** |
| 能否识别瓶颈步 | 不能（SFT 无此概念） | **能**（f_pseudo 标记了瓶颈位置） |
| Expert 分工来源 | 数据中不同 UI pattern | **图谱分析 + RL 探索** |
| 长 trajectory 处理 | 每步独立，无全局视角 | **f_pseudo 提供 trajectory-level 信号** |

### 15.2 两套 MoE 架构对比

代码库中存在两套不同的 MoE 实现架构：

| 方面 | SFT 路径 | RL 路径 |
|------|---------|---------|
| **核心文件** | `verl/models/moe/moe_wrapper.py` | `verl/trainer/ppo/moe_dapo_trainer.py` |
| **LoRA 注入方式** | Module Replacement（`MoELoRALinear` 替换 `nn.Linear`） | Hook-based（`ExpertLoRACollection` + `MoEExpertApplier` via hooks） |
| **Forward 流程** | Two-pass: Pass 1 (base) → router → Pass 2 (base + LoRA delta) | Router 在 driver，Expert 应用在 worker（通过 Ray） |
| **Router 位置** | 集成在 `MoEVLMWrapper` 内 | 独立在 `MoERayTrajDAPOTrainer` driver 上 |
| **支持的 Router** | TextOnlyRouter + ContextAwareRouter（已实现） | **仅 TextOnlyRouter**（待扩展） |
| **Config 类** | `MoEConfig`（dataclass） | `MoETrainerConfig`（dataclass） |

**关键差异**：RL 的 `MoERayTrajDAPOTrainer` 使用 hook-based 架构是因为 VERL 框架的 Ray 分布式设计——模型在 worker 上，router 在 driver 上。这种分离式设计在 RL 场景中是必要的。

### 15.3 需要修改的文件

#### 15.3.1 `MoETrainerConfig` — 新增 `router_type` 字段

```python
# verl/trainer/ppo/moe_dapo_trainer.py

@dataclass
class MoETrainerConfig:
    # ... 现有字段不变 ...

    # NEW: Router type selection
    router_type: str = 'text_only'  # 'text_only' | 'context_aware'
```

默认 `'text_only'` 保证向后兼容。

#### 15.3.2 `_lazy_init_moe()` — 条件创建 Router

```python
def _lazy_init_moe(self, hidden_size, num_layers, device='cuda'):
    if self._moe_initialized:
        return

    from verl.models.moe import (
        TextOnlyRouter,
        ContextAwareRouter,          # NEW import
        ExpertLoRACollection,
        MoEExpertApplier,
        InstructionFeatureExtractor,
        MoELoss,
    )

    # Router (conditional on router_type)
    if self.moe_config.router_type == 'context_aware':
        self._router = ContextAwareRouter(
            hidden_size=hidden_size,
            num_experts=self.moe_config.num_experts,
            router_hidden=self.moe_config.router_hidden,
            top_k=self.moe_config.top_k,
            dropout=self.moe_config.router_dropout,
            temperature=self.moe_config.router_temperature,
        ).to(device)
        # Vision feature extractor (for screenshot tokens)
        self._vision_feature_extractor = InstructionFeatureExtractor(
            pooling_strategy=self.moe_config.pooling_strategy,
        ).to(device)
    else:
        self._router = TextOnlyRouter(
            hidden_size=hidden_size,
            num_experts=self.moe_config.num_experts,
            router_hidden=self.moe_config.router_hidden,
            top_k=self.moe_config.top_k,
            dropout=self.moe_config.router_dropout,
            temperature=self.moe_config.router_temperature,
        ).to(device)

    # Feature extractor (shared, used for text features in both modes)
    self._feature_extractor = InstructionFeatureExtractor(
        pooling_strategy=self.moe_config.pooling_strategy,
    ).to(device)

    # ... Expert LoRA, Applier, Loss 不变 ...
```

#### 15.3.3 `compute_moe_routing()` — 双模式 routing

```python
def compute_moe_routing(
    self,
    hidden_states: torch.Tensor,
    instruction_mask: torch.Tensor,
    vision_mask: torch.Tensor = None,      # NEW
    text_context_mask: torch.Tensor = None, # NEW
) -> Dict[str, torch.Tensor]:

    if self.moe_config.router_type == 'context_aware':
        # Context-aware: use vision + text features
        assert vision_mask is not None and text_context_mask is not None, \
            "context_aware router requires vision_mask and text_context_mask"
        vision_features = self._vision_feature_extractor(hidden_states, vision_mask)
        text_features = self._feature_extractor(hidden_states, text_context_mask)
        router_output = self._router(vision_features, text_features)
    else:
        # Text-only: original behavior
        instruction_features = self._feature_extractor(hidden_states, instruction_mask)
        router_output = self._router(instruction_features)

    return {
        'routing_weights': router_output.routing_weights,
        'top_k_indices': router_output.top_k_indices,
        'top_k_weights': router_output.top_k_weights,
        'router_logits': router_output.router_logits,
    }
```

#### 15.3.4 新增 mask 计算方法

```python
def _compute_vision_mask(self, batch: DataProto) -> torch.Tensor:
    """Compute vision token mask from batch for ContextAwareRouter."""
    from verl.models.moe.router import create_vision_mask
    input_ids = batch.batch['input_ids']
    return create_vision_mask(input_ids, self.tokenizer)

def _compute_text_context_mask(self, batch: DataProto) -> torch.Tensor:
    """Compute text context mask (instruction + action history)."""
    from verl.models.moe.router import create_text_context_mask
    input_ids = batch.batch['input_ids']
    return create_text_context_mask(input_ids, self.tokenizer)
```

#### 15.3.5 `save_moe_checkpoint()` — 保存 `router_type`

```python
config_dict = {
    # ... 现有字段 ...
    'router_type': self.moe_config.router_type,  # NEW
}
```

#### 15.3.6 `load_moe_checkpoint()` — 根据 `router_type` 加载

现有代码调用 `_lazy_init_moe()` 时只传 `hidden_size` 和 `num_layers`。需确保在 lazy init 之前已正确设置 `self.moe_config.router_type`。如果从 checkpoint 加载的 `moe_config.json` 包含 `router_type`，应以 checkpoint 中的为准。

#### 15.3.7 `get_moe_trainable_params()` — 包含新增参数

```python
def get_moe_trainable_params(self) -> List[nn.Parameter]:
    params = list(self._router.parameters())
    params.extend(list(self._expert_collection.parameters()))
    # NEW: include vision feature extractor if context_aware
    if hasattr(self, '_vision_feature_extractor') and self._vision_feature_extractor is not None:
        params.extend(list(self._vision_feature_extractor.parameters()))
    return params
```

### 15.4 RL Config — `traj_grpo_moe_v2_fpseudo.yaml`

关键设计：同时启用 MoE (ContextAwareRouter) 和 f_pseudo reward shaping。

```yaml
# 基于 traj_grpo_moe.yaml + gui360_rl_f_pseudo.yaml 合并
actor_rollout_ref:
  model:
    moe:
      enabled: true
      num_experts: 6                    # 与 SFT v3 一致（之前 RL 用 4）
      top_k: 1                          # hard routing for vLLM compat
      router_type: "context_aware"      # NEW: step-level dynamic routing
      expert_lora_r: 16
      expert_lora_alpha: 32
      target_modules: ["q_proj", "v_proj"]
      router_hidden: 256
      balance_weight: 0.2
      balance_type: entropy             # entropy balance 更稳定
      z_loss_weight: 0.01
      use_vectorized_routing: false     # hard routing, no need for vectorized

trainer:
  trainer_class: verl.trainer.ppo.moe_dapo_trainer.MoERayTrajDAPOTrainer

# f_pseudo reward shaping for expert specialization
reward_model:
  enable: false
  reward_manager: f_pseudo_dapo         # 已实现的 FPseudoDAPORewardManager
  reward_kwargs:
    f_pseudo_path: outputs/f_pseudo/f_pseudo_map.json
    f_pseudo_lambda: 0.1                # 可调：控制 expert 特化强度

algorithm:
  adv_estimator: uis1                   # trajectory-level advantage
  uis1:
    episode_advantage_w: 1.0
    step_advantage_w: 1.0
    mode: mean_norm
```

**与之前两个 config 的关系：**

| Config | MoE | f_pseudo | 目的 |
|--------|-----|----------|------|
| `traj_grpo_moe.yaml` | TextOnlyRouter, 4 experts | 无 | 原始 MoE RL |
| `gui360_rl_f_pseudo.yaml` | 无 MoE | f_pseudo reward | 瓶颈跨越 RL |
| **`traj_grpo_moe_v2_fpseudo.yaml`** | **ContextAwareRouter, 6 experts** | **f_pseudo reward** | **MoE expert 特化 RL** |

### 15.5 训练流程：SFT → RL (MoE + f_pseudo) Pipeline

```
┌───────────────────────────────────────────────────────────────────┐
│           Complete MoE + f_pseudo Training Pipeline                │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ Stage 0: f_pseudo 预计算 (一次性, CPU, ~3秒)                  │  │
│  │                                                              │  │
│  │ - 输入: outputs/fnet/gui360/{app}/f_values.npz              │  │
│  │         + transitions.jsonl (91,618 条)                      │  │
│  │ - 输出: outputs/f_pseudo/f_pseudo_map.json                  │  │
│  │ - 已完成 ✅                                                  │  │
│  └─────────────────────────┬───────────────────────────────────┘  │
│                             │ f_pseudo_map.json (离线 reward 信号) │
│                             │                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ Stage 1: SFT 预训练 (Phase 1)                                │  │
│  │                                                              │  │
│  │ - Config: moe_sft_config_v3.yaml                            │  │
│  │ - Router: ContextAwareRouter (from scratch)                  │  │
│  │ - Experts: 6 × LoRA r=16 (from scratch)                    │  │
│  │ - Data: GUI-360 SFT dataset                                 │  │
│  │ - Reward: CE loss only (SFT, 无 f_pseudo)                   │  │
│  │ - 目标: 学会基本动作预测 + router 初步分工                    │  │
│  │ - 产物: output/moe_sft_v3/final/                            │  │
│  └─────────────────────────┬───────────────────────────────────┘  │
│                             │ SFT checkpoint (router + experts)   │
│                             ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ Stage 2: RL 微调 + f_pseudo Expert 特化 (Phase 4)            │  │
│  │                                                              │  │
│  │ - Config: traj_grpo_moe_v2_fpseudo.yaml                     │  │
│  │ - Trainer: MoERayTrajDAPOTrainer                             │  │
│  │ - Reward: FPseudoDAPORewardManager                           │  │
│  │   r_total = r_action_match + λ * f_pseudo(t)                │  │
│  │                                                              │  │
│  │ - 从 SFT checkpoint 初始化 router + experts                  │  │
│  │ - f_pseudo 信号引导 expert 特化:                              │  │
│  │   * 瓶颈步 (f_pseudo>0) → 强化被选中 expert 的瓶颈处理能力   │  │
│  │   * 常规步 (f_pseudo≈0) → 正常 action match reward          │  │
│  │   * router 学会: 瓶颈状态 → 路由到擅长跨越的 expert          │  │
│  │ - Data: UI-S1 trajectory dataset                             │  │
│  │ - 产物: checkpoints/gui_traj_grpo_moe/final/moe/            │  │
│  └─────────────────────────┬───────────────────────────────────┘  │
│                             │ RL-optimized checkpoint              │
│                             │ (experts 已按 f_pseudo 特化)         │
│                             ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ Stage 3: 蒸馏 + Serving (Phase 2-3)                          │  │
│  │                                                              │  │
│  │ - 用 RL checkpoint 生成 routing labels                       │  │
│  │ - 蒸馏到 StandaloneRouter                                    │  │
│  │ - vLLM multi-LoRA serving                                    │  │
│  └─────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

### 15.5.1 f_pseudo 在 MoE RL 中的梯度流

```
  Step t: screenshot_t, instruction, action_history
      │
      ▼
  ContextAwareRouter(vision_feat_t, text_feat_t)
      │
      ▼ 选择 expert_k (hard routing, top-1)
      │
  Expert_k(input) → action prediction
      │
      ▼
  Environment / Ground Truth → r_action_match(t)
      │
  f_pseudo_map[exec_id][step_id] → f_pseudo(t)
      │
      ▼
  r_total(t) = r_action_match(t) + λ * f_pseudo(t)
      │
      ▼
  UIS1 Advantage Estimator:
    A(t) = episode_advantage + step_advantage
    step_advantage 包含了 f_pseudo bonus
      │
      ▼
  Policy Gradient: ∇θ = A(t) * ∇θ log π(action|state; expert_k)
      │
      │ 梯度分别流向:
      │
      ├──→ Expert_k parameters (被选中的 expert 获得全部梯度)
      │    - f_pseudo > 0 + r_base 高 → 强正向梯度 → expert_k 强化瓶颈处理
      │    - f_pseudo > 0 + r_base 低 → 混合信号 → expert_k 在瓶颈步还需提升
      │
      ├──→ ContextAwareRouter parameters (vision_proj, text_proj, router MLP)
      │    - 当 expert_k 在 f_pseudo>0 步骤表现好 → router 增强 "类似截图→expert_k" 映射
      │    - 当 expert_k 在 f_pseudo>0 步骤表现差 → router 减弱此映射，尝试其他 expert
      │
      └──→ MoE balance loss (不受 f_pseudo 影响，防止 collapse)
```

**为什么 f_pseudo 对 MoE 比对单模型更有效：**

| 场景 | 单模型 + f_pseudo | MoE + f_pseudo |
|------|------------------|----------------|
| 瓶颈跨越步 (14.2%) | 单一模型同时学习所有技能，梯度可能冲突 | 特定 expert 专注瓶颈跨越，参数不与其他任务冲突 |
| 常规导航步 (85.8%) | 同上 | 其他 experts 专注导航，不被瓶颈梯度干扰 |
| 训练稳定性 | f_pseudo 的 [-3, +3] 范围可能导致梯度波动 | 每个 expert 只接收自己被路由到的步的梯度，更稳定 |
| Expert 特化 | N/A | f_pseudo 提供了自然的"任务类型"划分信号 |

### 15.6 SFT → RL Checkpoint 转换

SFT 和 RL 使用不同的 MoE 架构（module replacement vs hooks），checkpoint 格式需要转换：

| 组件 | SFT 格式 (MoEVLMWrapper) | RL 格式 (MoERayTrajDAPOTrainer) |
|------|--------------------------|--------------------------------|
| Router | `router.pt` (同结构) | `router.pt` (同结构) ✅ 直接用 |
| Experts | `experts/expert_{i}/adapter_model.bin` (PEFT) | `experts/expert_{i}/adapter_model.bin` (PEFT) ✅ 直接用 |
| Feature Extractor | `feature_extractor` (无参数，仅 pooling) | `feature_extractor` (无参数) ✅ 不需要 |

**好消息**：Router 和 Expert LoRA 的 checkpoint 格式在两套架构之间是一致的（都是 `router.pt` + PEFT format），**无需格式转换**。RL trainer 的 `load_moe_checkpoint()` 可以直接加载 SFT 训练的 checkpoint。

### 15.7 SFT → RL 加载实现

在 `MoERayTrajDAPOTrainer` 中，SFT checkpoint 加载需要一个小改动——支持从非 `moe/` 子目录加载（SFT checkpoint 的结构是 `output/moe_sft_v3/final/`，没有 `moe/` 前缀）：

```python
def load_moe_checkpoint(self, load_dir: str):
    """Load MoE components from checkpoint.
    Supports both RL format (load_dir/moe/) and SFT format (load_dir/)."""
    moe_dir = os.path.join(load_dir, 'moe')
    if not os.path.exists(moe_dir):
        # Try SFT format: config at load_dir/moe_config.json
        if os.path.exists(os.path.join(load_dir, 'moe_config.json')):
            moe_dir = load_dir
        else:
            print(f"[MoE] No checkpoint found at {moe_dir}")
            return

    # ... rest of loading logic (same) ...
```

### 15.8 RL 阶段的特殊考虑

#### 15.8.1 Hidden States 获取

RL 训练中，hidden states 来自 worker 上运行的 actor model forward pass。当前 `_compute_instruction_mask()` 使用 heuristic（最后 50 个 token）。对于 `context_aware` router，需要精确的 `vision_mask` 和 `text_context_mask`，这可以通过 `create_vision_mask()` 和 `create_text_context_mask()` 函数从 `input_ids` 中精确提取。

**注意**：RL 训练的 `input_ids` 包含 prompt + generated response。Vision tokens 在 prompt 部分，text context 也在 prompt 部分。需要确保 mask 函数正确处理较长的 RL sequence。

#### 15.8.2 Router 梯度传播

在 RL 训练中，router 梯度通过 policy gradient 传播：
- Forward: hidden_states → router → expert_idx → expert LoRA → output → reward
- Backward: reward → REINFORCE gradient → router parameters

`MoEExpertApplier`（hook-based）将 routing weights 作为 expert 输出的权重系数，梯度自然传播到 router。对于 `ContextAwareRouter`，额外的 `vision_proj` 和 `text_proj` 参数也会收到梯度更新。

#### 15.8.3 与 vLLM Rollout 的交互

VERL 的 RL 训练使用 vLLM 做 rollout（生成 trajectory），但 rollout 阶段 **不需要 MoE routing**——因为 rollout 用的是 base model（无 LoRA）或者固定策略。MoE routing 只在 **actor training forward** 阶段使用，此时模型在 FSDP worker 上运行，可以正常获取 hidden states。

### 15.9 修订后的执行计划

将原 Step 1-5（纯 SFT 路径）扩展为 Step 1-7（SFT + RL + f_pseudo 路径）：

```
  ┌──────────────────────────────────────────────────────────────────┐
  │ 前置: f_pseudo_map.json 已生成 ✅ (outputs/f_pseudo/)            │
  │       f_net 已训练完成 ✅ (outputs/fnet/gui360/)                 │
  └──────────────────────────────────────────┬───────────────────────┘
                                             │
                    ┌──────────┐             │
                    │  Step 1  │  创建 v3 SFT 训练 slurm
                    │  (~30min)│
                    └────┬─────┘
                         │ sbatch
                         ▼
                    ┌──────────┐
                    │ SFT 训练  │  4 nodes × 4 GPUs × 24h
                    │ (12-24h) │  (无 f_pseudo，纯 CE loss)
                    └────┬─────┘
                         │ output/moe_sft_v3/final/
                    ┌────┴─────┐
                    │          │
               ┌────▼────┐    │
               │ Step 2  │    │
               │ SFT 评估 │    │
               │ (24-41h)│    │
               └────┬────┘    │
                    │         │
                    │ GATE: SFT 质量 OK?
                    │ ├─ YES ─┤
                    │         │
                    │    ┌────▼──────────────────────────────────┐
                    │    │ Step 3: RL + f_pseudo Code Changes     │
                    │    │ (~2-4h)                                │
                    │    │ - 修改 moe_dapo_trainer.py             │
                    │    │   (ContextAwareRouter 支持)            │
                    │    │ - 创建 traj_grpo_moe_v2_fpseudo.yaml  │
                    │    │   (MoE + f_pseudo_dapo 联合配置)       │
                    │    └────┬──────────────────────────────────┘
                    │         │
                    │    ┌────▼──────────────────────────────────┐
                    │    │ Step 4: RL + f_pseudo 训练             │
                    │    │ (24-48h, 8 GPUs)                      │
                    │    │ - 从 SFT checkpoint 初始化             │
                    │    │ - FPseudoDAPORewardManager             │
                    │    │ - r = r_base + λ * f_pseudo            │
                    │    │ - 观察 expert 特化 pattern              │
                    │    └────┬──────────────────────────────────┘
                    │         │ RL-optimized checkpoint
                    │         │ (experts 按 f_pseudo 特化)
                    │    ┌────▼────┐
                    │    │ Step 5  │  RL 模型评估 (HF)
                    │    │ (24-41h)│  对比: SFT-only vs SFT+RL+f_pseudo
                    │    └────┬────┘
                    │         │
                    │    GATE: RL + f_pseudo 提升了？
                    │    ├─ YES ──────────────────────────────┐
                    │    │                                     │
                    │    │    ┌────▼────┐                      │
                    │    │    │ Step 6  │  蒸馏 (标签生成       │
                    │    │    │ (5-10h) │  + standalone router │
                    │    │    └────┬────┘  训练)               │
                    │    │         │                            │
                    │    │    ┌────▼────┐                      │
                    │    │    │ Step 7  │  vLLM 端到端评估     │
                    │    │    │ (4-8h)  │                      │
                    │    │    └─────────┘                      │
                    │    │                                     │
                    │    ├─ NO → 用 SFT checkpoint 直接走 Step 6-7
                    │
               ├─ NO → 调参回到 Step 1
```

**修订后总耗时估算**：
- Day 1: Step 1 (~30min) → 提交 SFT 训练 (12-24h)
- Day 2: Step 2 (SFT 评估) 与 Step 3 (RL 代码改动) 并行进行
- Day 3: Step 4 (RL + f_pseudo 训练, 24-48h)
- Day 4-5: Step 5 (RL 评估) → Step 6 (蒸馏) → Step 7 (vLLM 评估)

**总计 ~5-7 天**（乐观路径）。

### 15.10 需要新建/修改的文件（Phase 4 新增）

| 操作 | 文件 | 说明 |
|------|------|------|
| 修改 | `verl/trainer/ppo/moe_dapo_trainer.py` | `MoETrainerConfig` 加 `router_type`；`_lazy_init_moe` / `compute_moe_routing` / `save_moe_checkpoint` / `load_moe_checkpoint` / `get_moe_trainable_params` 加 context-aware 分支 |
| 新建 | `examples/qwen_gui_moe/config/traj_grpo_moe_v2_fpseudo.yaml` | RL config: MoE (ContextAwareRouter) + f_pseudo reward |
| 新建 | `train_GUI_360/moe_sft/train_rl_moe_v2_fpseudo.slurm` | RL 训练 slurm 脚本 |
| 新建 | `train_GUI_360/moe_sft/eval_rl_moe_v2.slurm` | RL 模型评估 slurm 脚本 |
| 已有 | `verl/workers/reward_manager/f_pseudo_dapo.py` | FPseudoDAPORewardManager ✅ 已实现 |
| 已有 | `outputs/f_pseudo/f_pseudo_map.json` | f_pseudo 预计算值 ✅ 已生成 |

### 15.11 风险与备选方案

| 风险 | 影响 | 备选方案 |
|------|------|---------|
| RL + f_pseudo 导致 router collapse | Expert 全部特化为瓶颈步或全忽略瓶颈步 | 增大 `balance_weight`；降低 `f_pseudo_lambda`；freeze router 前 N steps |
| f_pseudo λ 太大导致梯度不稳定 | 训练发散 | 降低 λ (0.1→0.05→0.01)；f_pseudo 范围 [-3,+3] × 0.01 = [-0.03,+0.03] 更安全 |
| 瓶颈步太少 (14.2%) 导致特化 expert 训练不充分 | Expert 没有足够数据特化 | 增大 λ 补偿样本数少；或对瓶颈步做 oversampling |
| f_pseudo 与 router 决策正交 | f_pseudo bonus 均匀分散到所有 expert，无特化效果 | 需验证: 同一类型瓶颈是否倾向于被同一 expert 处理。如果不是，说明 router 需要更多 epoch 收敛 |
| SFT→RL checkpoint 不兼容 | 加载失败 | `load_moe_checkpoint()` 已处理两种格式（见 15.7） |
| Hidden states 在 RL 中的 vision mask 不准确 | Routing 质量差 | RL 的 input_ids 包含 response，但 vision tokens 仍在 prompt 部分，mask 函数应能正确处理 |

### 15.12 关键验证指标（MoE + f_pseudo 特有）

训练过程中需监控的 WandB metrics：

```
# Expert 特化指标
moe/expert_{i}_utilization         # 各 expert 使用率（均匀 ~16.7% 每个）
moe/routing_entropy_mean           # routing 熵（太低=collapse，太高=无特化）
moe/load_balance_coefficient       # 负载均衡系数（>0.8 为好）

# f_pseudo 与 expert 关联指标（新增，需要在 trainer 中记录）
moe/f_pseudo_by_expert_{i}_mean   # 每个 expert 被选中时的平均 f_pseudo 值
                                   # 预期: 至少 1 个 expert 的值显著 > 0
                                   # → 说明该 expert 被路由到瓶颈步

moe/bottleneck_expert_usage       # f_pseudo > 0.5 的步中，哪些 expert 被选中
                                   # 预期: 集中在 1-2 个 expert

# 整体性能
reward/f_pseudo_mean               # 平均 f_pseudo bonus（应接近 0）
reward/base_reward_mean            # action match 基础分
reward/total_reward_mean           # 总 reward = base + λ*f_pseudo
```

**Expert 特化成功的判据：**

1. **Expert 分工可见**：至少 1 个 expert 的 `f_pseudo_by_expert_mean` 显著 > 0（被选中处理瓶颈步）
2. **Router 区分能力**：f_pseudo > 0.5 步中的 expert 分布不均匀（不是每个 expert 各 16.7%）
3. **性能提升**：瓶颈跨越步的 action match accuracy 提升
4. **不collapse**：所有 6 个 expert 的 utilization > 5%

### 15.13 与之前 f_pseudo RL 实验的对比

TASK_LIST.md 中 Task 6 已经启动了 3 组 f_pseudo RL 对比实验（Job 2523626/2523627/2523628），但那些实验 **没有 MoE**。

| 实验 | MoE | Router | f_pseudo | 预期效果 |
|------|-----|--------|----------|---------|
| Task 6 实验 (已提交) | 无 | 无 | λ=0.1 | 单模型学习瓶颈跨越 |
| **Phase 4 (本计划)** | **6 experts** | **ContextAwareRouter** | **λ=0.1** | **expert 按瓶颈类型特化** |

Phase 4 的核心假设是：**MoE 的 expert 特化 + f_pseudo 的瓶颈信号** 比单模型 + f_pseudo 更有效，因为：
- 单模型中瓶颈跨越梯度会与常规导航梯度冲突（参数共享）
- MoE 中不同 expert 各自负责不同类型的步骤，梯度不冲突
- f_pseudo 信号在 MoE 中自然成为 expert 分工的引导信号

**验证方法**：Step 5 评估时，对比 Phase 4 结果与 Task 6 结果，重点关注长 trajectory 场景中的瓶颈跨越成功率。
