# MoE Tool Agent 架构设计

## 1. 系统架构总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MoE Tool Agent System                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Input: (screenshot, instruction)                                       │
│                    │                                                     │
│                    ▼                                                     │
│   ┌────────────────────────────────────────────────────────────────┐    │
│   │                    Qwen2.5-VL Base Model (Frozen)               │    │
│   │                                                                 │    │
│   │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │    │
│   │   │   Vision    │    │   Text      │    │   Fusion    │        │    │
│   │   │   Encoder   │───▶│   Encoder   │───▶│   Layers    │        │    │
│   │   └─────────────┘    └─────────────┘    └──────┬──────┘        │    │
│   │                                                 │               │    │
│   └─────────────────────────────────────────────────┼───────────────┘    │
│                                                     │                    │
│                    ┌────────────────────────────────┼────────────┐       │
│                    │                                │            │       │
│                    ▼                                ▼            │       │
│         ┌──────────────────┐           ┌─────────────────────┐   │       │
│         │      Router      │           │    Expert LoRAs     │   │       │
│         │                  │           │                     │   │       │
│         │  instruction     │  weights  │  ┌───┐ ┌───┐ ┌───┐ │   │       │
│         │  features ──────▶│──────────▶│  │E0 │ │E1 │ │E2 │ │   │       │
│         │                  │           │  │   │ │   │ │   │ │   │       │
│         │  Linear + Softmax│           │  │E3 │ │   │ │   │ │   │       │
│         └──────────────────┘           │  └─┬─┘ └─┬─┘ └─┬─┘ │   │       │
│                                        │    └──┬──┴──┬──┘   │   │       │
│                                        │       │ weighted   │   │       │
│                                        │       ▼ sum        │   │       │
│                                        │  combined_output   │   │       │
│                                        └─────────┬──────────┘   │       │
│                                                  │              │       │
│                                                  ▼              │       │
│                                        ┌─────────────────┐      │       │
│                                        │   LM Head       │      │       │
│                                        │  (Generation)   │      │       │
│                                        └────────┬────────┘      │       │
│                                                 │               │       │
│                                                 ▼               │       │
│                                          Action Output          │       │
│                                                                 │       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心组件定义

### 2.1 组件清单

| 组件 | 类名 | 文件位置 | 可训练参数 |
|------|------|---------|-----------|
| Base VLM | `Qwen2_5_VLForCausalLM` | `verl/models/transformers/qwen2_5_vl.py` | ❌ Frozen |
| Router | `TextOnlyRouter` | `verl/models/moe/router.py` | ✅ |
| Expert LoRAs | `ExpertLoRACollection` | `verl/models/moe/expert_lora.py` | ✅ |
| MoE Wrapper | `MoEVLMWrapper` | `verl/models/moe/moe_wrapper.py` | - |

### 2.2 参数量对比

```python
# 设计原则: MoE 总参数量 ≈ Single LoRA 参数量 (公平对比)

# MoE 配置
num_experts = 4
expert_rank = 16
expert_alpha = 32
router_params = hidden_size * 256 + 256 * 4  # ~260K

# Single LoRA 配置 (baseline)
single_rank = 64  # 4 * 16 = 64
single_alpha = 128

# 参数量计算 (以 Qwen2.5-VL-7B hidden_size=3584 为例)
# 每个 expert LoRA (只适配 q_proj, v_proj):
#   - lora_A: 3584 * 16 = 57,344
#   - lora_B: 16 * 3584 = 57,344
#   - 每层: 114,688 * 2 (q, v) = 229,376
#   - 总计 (28 层): 229,376 * 28 ≈ 6.4M per expert
# 4 experts total: 25.6M

# Single LoRA (rank=64):
#   - 每层: 3584 * 64 * 2 * 2 = 917,504
#   - 总计 (28 层): 917,504 * 28 ≈ 25.7M
```

---

## 3. 数据流设计

### 3.1 训练阶段数据流

```python
# 输入格式
batch = {
    'input_ids': Tensor[B, seq_len],           # tokenized (image + instruction)
    'attention_mask': Tensor[B, seq_len],
    'pixel_values': Tensor[B, C, H, W],        # screenshot
    'labels': Tensor[B, seq_len],              # target action sequence
    'instruction_type': List[str],             # ['click', 'type', ...] 用于分析
}

# MoE Forward Pass
def forward(self, batch):
    # 1. Base model encoding (frozen)
    with torch.no_grad():
        base_outputs = self.base_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            pixel_values=batch['pixel_values'],
            output_hidden_states=True,
        )

    hidden_states = base_outputs.hidden_states[-1]  # [B, seq_len, hidden_size]

    # 2. Router: 提取 instruction 部分的 features 进行路由
    instruction_features = self._extract_instruction_features(hidden_states, batch)
    routing_weights = self.router(instruction_features)  # [B, num_experts]

    # 3. Top-k expert selection
    top_k_weights, top_k_indices = routing_weights.topk(self.top_k, dim=-1)

    # 4. Apply expert LoRAs
    expert_outputs = self._apply_experts(hidden_states, top_k_indices, top_k_weights)

    # 5. LM head for generation
    logits = self.lm_head(expert_outputs)

    return {
        'logits': logits,
        'routing_weights': routing_weights,
        'top_k_indices': top_k_indices,
    }
```

### 3.2 推理阶段数据流 (vLLM)

```python
# vLLM 推理时的策略:
# 方案 A: 预先确定 expert (基于 instruction prefix)
# 方案 B: 动态 merge weights (训练完成后)

# 方案 A 实现
class MoEInferenceWrapper:
    def __init__(self, base_model_path, expert_lora_paths):
        self.engine = LLM(
            model=base_model_path,
            enable_lora=True,
            max_loras=4,
        )
        self.expert_loras = expert_lora_paths
        self.router = self._load_router()  # 加载训练好的 router

    def generate(self, screenshot, instruction):
        # 1. 使用 router 确定最佳 expert
        expert_idx = self._route(instruction)

        # 2. 使用对应的 LoRA adapter
        outputs = self.engine.generate(
            prompts=...,
            lora_request=LoRARequest(
                lora_name=f"expert_{expert_idx}",
                lora_path=self.expert_loras[expert_idx],
            )
        )
        return outputs
```

---

## 4. 模块接口定义

### 4.1 Router 接口

```python
# verl/models/moe/router.py

class TextOnlyRouter(nn.Module):
    """
    基于 instruction text features 进行路由决策

    Args:
        hidden_size: Base model hidden dimension
        num_experts: Number of expert LoRAs
        router_hidden: Router MLP hidden dimension

    Input:
        instruction_features: [B, hidden_size] - pooled instruction representation

    Output:
        routing_weights: [B, num_experts] - softmax normalized weights
    """

    def __init__(self, hidden_size: int, num_experts: int = 4, router_hidden: int = 256):
        ...

    def forward(self, instruction_features: torch.Tensor) -> torch.Tensor:
        ...
```

### 4.2 Expert LoRA 接口

```python
# verl/models/moe/expert_lora.py

class ExpertLoRACollection(nn.Module):
    """
    管理多个 Expert LoRA adapters

    Args:
        base_model: Frozen base VLM
        num_experts: Number of experts
        lora_r: LoRA rank per expert
        lora_alpha: LoRA alpha scaling
        target_modules: Which modules to apply LoRA

    Methods:
        apply_expert(hidden_states, expert_idx): Apply single expert
        apply_experts_weighted(hidden_states, indices, weights): Weighted combination
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_experts: int = 4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        target_modules: List[str] = ['q_proj', 'v_proj'],
    ):
        ...
```

### 4.3 MoE Wrapper 接口

```python
# verl/models/moe/moe_wrapper.py

class MoEVLMWrapper(nn.Module):
    """
    包装 Base VLM + Router + Expert LoRAs

    与 verl 训练框架兼容的完整 MoE 模型
    """

    def __init__(
        self,
        base_model_name_or_path: str,
        num_experts: int = 4,
        top_k: int = 1,
        expert_lora_r: int = 16,
        expert_lora_alpha: int = 32,
        router_hidden: int = 256,
        balance_weight: float = 0.1,
    ):
        ...

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> MoEOutput:
        """
        Returns:
            MoEOutput with fields:
                - loss: total loss (LM + balance)
                - logits: [B, seq_len, vocab_size]
                - routing_weights: [B, num_experts]
                - top_k_indices: [B, top_k]
        """
        ...
```

---

## 5. 与现有代码的集成点

### 5.1 模型层集成

```
verl/models/
├── transformers/
│   └── qwen2_5_vl.py         # 现有 - Base VLM
├── moe/                       # 新增
│   ├── __init__.py
│   ├── router.py             # Router 实现
│   ├── expert_lora.py        # Expert LoRA 实现
│   └── moe_wrapper.py        # 整合 wrapper
└── registry.py               # 注册 MoE 模型
```

### 5.2 训练层集成

```
verl/trainer/ppo/
├── dapo_ray_trainer.py       # 现有 - 扩展支持 MoE
├── core_algos.py             # 现有 - 添加 balance loss
├── moe_loss.py               # 新增 - MoE 特定 loss
└── moe_dapo_trainer.py       # 新增 - MoE trainer wrapper
```

### 5.3 推理层集成

```
verl/workers/
├── sharding_manager/
│   ├── fsdp_vllm.py          # 现有 - 扩展支持多 LoRA
│   └── fsdp_vllm_moe.py      # 新增 - MoE 特定管理
└── rollout/
    └── vllm_rollout/
        └── vllm_rollout.py   # 现有 - 扩展 LoRA 选择逻辑
```

---

## 6. 配置示例

```yaml
# examples/qwen_gui_moe/config/traj_grpo_moe.yaml

model:
  base_model: Qwen/Qwen2.5-VL-7B-Instruct

  # MoE 配置
  moe:
    enabled: true
    num_experts: 4
    top_k: 1
    expert_lora_r: 16
    expert_lora_alpha: 32
    target_modules: ['q_proj', 'v_proj']
    router_hidden: 256
    balance_weight: 0.1

trainer:
  # 继承现有 DAPO 配置
  algorithm: dapo

  # MoE 特定配置
  moe_loss:
    balance_weight: 0.1
    balance_type: mse  # or 'switch_transformer'

rollout:
  engine: vllm
  # vLLM LoRA 配置
  enable_lora: true
  max_loras: 4

data:
  # 添加 instruction_type 用于分析
  include_instruction_type: true
```

---

## 7. 下一步

详细实现请参考：
- [02_router_implementation.md](./02_router_implementation.md) - Router 详细实现
- [03_expert_lora.md](./03_expert_lora.md) - Expert LoRA 详细实现
