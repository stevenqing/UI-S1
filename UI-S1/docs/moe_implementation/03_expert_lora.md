# Expert LoRA 模块实现

## 1. 设计决策

### 1.1 Expert 结构选择

| 方案 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **独立 LoRA** (选择) | 每个 expert 有完全独立的 LoRA 参数 | 最大分化潜力 | 参数量 × N |
| 共享 Down + 独立 Up | 共享 lora_A，独立 lora_B | 减少参数 | 分化受限 |
| 共享大部分 | 只有最后一层不同 | 参数最少 | 分化能力弱 |

### 1.2 Target Modules

```python
# Qwen2.5-VL-7B 结构
# 每个 transformer block 包含:
# - self_attn: q_proj, k_proj, v_proj, o_proj
# - mlp: gate_proj, up_proj, down_proj

# Pilot 实验选择 (参数量适中)
target_modules = ['q_proj', 'v_proj']

# 完整实验选择 (更强表达力)
target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
```

### 1.3 参数量计算

```
Qwen2.5-VL-7B:
- hidden_size = 3584
- num_layers = 28
- 每层 q_proj/v_proj: 3584 × 3584 (实际 attention heads 分割)

单个 Expert LoRA (r=16, target=['q_proj', 'v_proj']):
- 每层: 2 × (3584 × 16 + 16 × 3584) = 229,376
- 28 层: 28 × 229,376 = 6,422,528 ≈ 6.4M

4 Experts 总计: 4 × 6.4M = 25.6M 参数

对比 Single LoRA (r=64):
- 每层: 2 × (3584 × 64 + 64 × 3584) = 917,504
- 28 层: 28 × 917,504 = 25,690,112 ≈ 25.7M
```

---

## 2. 完整实现代码

### 2.1 Expert LoRA Collection

```python
# verl/models/moe/expert_lora.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class ExpertOutput:
    """Expert forward 输出"""
    hidden_states: torch.Tensor        # [B, seq_len, hidden_size]
    expert_contributions: Dict[int, torch.Tensor]  # 每个 expert 的贡献 (用于分析)


class LoRALayer(nn.Module):
    """
    单个 LoRA adapter layer

    LoRA: y = Wx + (alpha/r) * B @ A @ x

    其中:
    - W: 原始权重 (frozen)
    - A: down projection [hidden_size, r]
    - B: up projection [r, hidden_size]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算 LoRA delta

        Args:
            x: [B, seq_len, in_features]

        Returns:
            delta: [B, seq_len, out_features]
        """
        # x @ A^T @ B^T * scaling
        x = self.lora_dropout(x)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return delta


class SingleExpertLoRA(nn.Module):
    """
    单个 Expert 的 LoRA adapter 集合

    包含多个层的 LoRA adapters
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        target_modules: List[str],
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.target_modules = target_modules
        self.r = r

        # 为每层的每个 target module 创建 LoRA
        self.lora_layers = nn.ModuleDict()

        for layer_idx in range(num_layers):
            for module_name in target_modules:
                key = f"layer_{layer_idx}_{module_name}"
                self.lora_layers[key] = LoRALayer(
                    in_features=hidden_size,
                    out_features=hidden_size,  # 简化：假设 in = out
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                )

    def get_lora_delta(
        self,
        layer_idx: int,
        module_name: str,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        获取特定层特定模块的 LoRA delta

        Args:
            layer_idx: transformer layer index
            module_name: e.g., 'q_proj', 'v_proj'
            x: input tensor [B, seq_len, hidden_size]

        Returns:
            delta: [B, seq_len, hidden_size]
        """
        key = f"layer_{layer_idx}_{module_name}"
        if key in self.lora_layers:
            return self.lora_layers[key](x)
        return torch.zeros_like(x)


class ExpertLoRACollection(nn.Module):
    """
    管理多个 Expert LoRA adapters

    支持:
    - Top-k expert 选择
    - 加权组合多个 experts
    - 单 expert 推理 (for vLLM)
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_experts: int = 4,
        target_modules: List[str] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        """
        Args:
            hidden_size: Model hidden dimension
            num_layers: Number of transformer layers
            num_experts: Number of expert LoRAs
            target_modules: Which modules to apply LoRA
            lora_r: LoRA rank
            lora_alpha: LoRA scaling factor
            lora_dropout: Dropout rate
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.target_modules = target_modules or ['q_proj', 'v_proj']

        # 创建 experts
        self.experts = nn.ModuleList([
            SingleExpertLoRA(
                hidden_size=hidden_size,
                num_layers=num_layers,
                target_modules=self.target_modules,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            for _ in range(num_experts)
        ])

    def apply_single_expert(
        self,
        expert_idx: int,
        layer_idx: int,
        module_name: str,
        base_output: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        应用单个 expert 的 LoRA delta

        Args:
            expert_idx: which expert to use
            layer_idx: transformer layer index
            module_name: target module name
            base_output: output from base model [B, seq_len, hidden_size]
            x: input to the module [B, seq_len, hidden_size]

        Returns:
            modified_output: base_output + lora_delta
        """
        delta = self.experts[expert_idx].get_lora_delta(layer_idx, module_name, x)
        return base_output + delta

    def apply_experts_weighted(
        self,
        layer_idx: int,
        module_name: str,
        base_output: torch.Tensor,
        x: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        加权应用多个 experts

        Args:
            layer_idx: transformer layer index
            module_name: target module name
            base_output: [B, seq_len, hidden_size]
            x: [B, seq_len, hidden_size]
            top_k_indices: [B, top_k] selected expert indices
            top_k_weights: [B, top_k] normalized weights

        Returns:
            modified_output: base_output + weighted_sum(lora_deltas)
        """
        batch_size = base_output.size(0)
        top_k = top_k_indices.size(1)

        # 计算加权 delta
        weighted_delta = torch.zeros_like(base_output)

        for b in range(batch_size):
            for k in range(top_k):
                expert_idx = top_k_indices[b, k].item()
                weight = top_k_weights[b, k]

                delta = self.experts[expert_idx].get_lora_delta(
                    layer_idx, module_name, x[b:b+1]
                )
                weighted_delta[b:b+1] += weight * delta

        return base_output + weighted_delta

    def get_expert_state_dict(self, expert_idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个 expert 的 state dict (用于保存/加载)

        Args:
            expert_idx: which expert

        Returns:
            state_dict for this expert
        """
        return self.experts[expert_idx].state_dict()

    def load_expert_state_dict(self, expert_idx: int, state_dict: Dict[str, torch.Tensor]):
        """
        加载单个 expert 的 state dict

        Args:
            expert_idx: which expert
            state_dict: state dict to load
        """
        self.experts[expert_idx].load_state_dict(state_dict)

    def save_experts_separately(self, save_dir: str):
        """
        分别保存每个 expert (用于 vLLM 加载)

        Args:
            save_dir: directory to save experts
        """
        import os

        os.makedirs(save_dir, exist_ok=True)

        for i in range(self.num_experts):
            expert_path = os.path.join(save_dir, f"expert_{i}")
            os.makedirs(expert_path, exist_ok=True)

            # 转换为 PEFT 格式
            peft_state_dict = self._convert_to_peft_format(i)
            torch.save(peft_state_dict, os.path.join(expert_path, "adapter_model.bin"))

            # 保存 config
            self._save_peft_config(expert_path)

    def _convert_to_peft_format(self, expert_idx: int) -> Dict[str, torch.Tensor]:
        """
        将 expert 权重转换为 PEFT 格式

        PEFT 格式: base_model.model.model.layers.{i}.self_attn.{module}.lora_A.weight
        """
        expert = self.experts[expert_idx]
        peft_state_dict = {}

        for layer_idx in range(self.num_layers):
            for module_name in self.target_modules:
                key = f"layer_{layer_idx}_{module_name}"
                if key in expert.lora_layers:
                    lora_layer = expert.lora_layers[key]

                    # 构建 PEFT 格式的 key
                    if module_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                        peft_prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{module_name}"
                    else:  # mlp modules
                        peft_prefix = f"base_model.model.model.layers.{layer_idx}.mlp.{module_name}"

                    peft_state_dict[f"{peft_prefix}.lora_A.weight"] = lora_layer.lora_A.data
                    peft_state_dict[f"{peft_prefix}.lora_B.weight"] = lora_layer.lora_B.data

        return peft_state_dict

    def _save_peft_config(self, save_dir: str):
        """保存 PEFT adapter_config.json"""
        import json

        config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": self.experts[0].r,
            "lora_alpha": self.experts[0].r * 2,  # 假设 alpha = 2r
            "target_modules": self.target_modules,
            "lora_dropout": 0.05,
            "bias": "none",
        }

        with open(os.path.join(save_dir, "adapter_config.json"), "w") as f:
            json.dump(config, f, indent=2)


class MoEExpertApplier(nn.Module):
    """
    在模型 forward 过程中应用 MoE experts

    这是一个 hook-based 实现，可以注入到现有模型中
    """

    def __init__(
        self,
        expert_collection: ExpertLoRACollection,
        router,  # TextOnlyRouter
    ):
        super().__init__()

        self.expert_collection = expert_collection
        self.router = router

        # 缓存当前 batch 的 routing 结果
        self._current_top_k_indices = None
        self._current_top_k_weights = None

    def set_routing(
        self,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ):
        """
        设置当前 batch 的 routing 结果

        在每个 forward 开始时调用
        """
        self._current_top_k_indices = top_k_indices
        self._current_top_k_weights = top_k_weights

    def create_forward_hook(self, layer_idx: int, module_name: str):
        """
        创建 forward hook 来应用 LoRA

        Usage:
            module.register_forward_hook(applier.create_forward_hook(layer_idx, 'q_proj'))
        """
        def hook(module, input, output):
            if self._current_top_k_indices is None:
                return output

            # input[0] 是 module 的输入
            x = input[0] if isinstance(input, tuple) else input

            return self.expert_collection.apply_experts_weighted(
                layer_idx=layer_idx,
                module_name=module_name,
                base_output=output,
                x=x,
                top_k_indices=self._current_top_k_indices,
                top_k_weights=self._current_top_k_weights,
            )

        return hook

    def clear_routing(self):
        """清除 routing 缓存"""
        self._current_top_k_indices = None
        self._current_top_k_weights = None
```

---

## 3. 与现有 PEFT 集成

### 3.1 加载现有 LoRA checkpoint

```python
def load_from_peft_checkpoint(
    expert_collection: ExpertLoRACollection,
    expert_idx: int,
    peft_checkpoint_path: str,
):
    """
    从 PEFT checkpoint 加载到指定 expert

    Args:
        expert_collection: ExpertLoRACollection instance
        expert_idx: which expert to load into
        peft_checkpoint_path: path to PEFT checkpoint
    """
    import os

    # 加载 PEFT state dict
    adapter_path = os.path.join(peft_checkpoint_path, "adapter_model.bin")
    if os.path.exists(adapter_path):
        peft_state_dict = torch.load(adapter_path)
    else:
        # SafeTensors 格式
        from safetensors.torch import load_file
        adapter_path = os.path.join(peft_checkpoint_path, "adapter_model.safetensors")
        peft_state_dict = load_file(adapter_path)

    # 转换 keys
    expert = expert_collection.experts[expert_idx]

    for peft_key, value in peft_state_dict.items():
        # 解析 PEFT key
        # 格式: base_model.model.model.layers.{i}.self_attn.{module}.lora_{A/B}.weight
        parts = peft_key.split('.')

        try:
            layer_idx = int(parts[4])  # layers.{i}

            if 'self_attn' in peft_key:
                module_name = parts[6]  # q_proj, v_proj, etc.
            elif 'mlp' in peft_key:
                module_name = parts[6]  # gate_proj, up_proj, down_proj

            lora_type = parts[-2]  # lora_A or lora_B

            # 构建 expert key
            expert_key = f"layer_{layer_idx}_{module_name}"

            if expert_key in expert.lora_layers:
                lora_layer = expert.lora_layers[expert_key]
                if lora_type == 'lora_A':
                    lora_layer.lora_A.data.copy_(value)
                elif lora_type == 'lora_B':
                    lora_layer.lora_B.data.copy_(value)

        except (IndexError, ValueError) as e:
            print(f"Skipping key {peft_key}: {e}")
            continue

    print(f"Loaded PEFT checkpoint into expert {expert_idx}")
```

### 3.2 与 verl FSDP 集成

```python
# 在 verl/workers/sharding_manager/fsdp_vllm.py 中集成

def collect_moe_lora_params(module: nn.Module) -> Dict[int, OrderedDict]:
    """
    收集 MoE 中所有 experts 的 LoRA 参数

    Returns:
        Dict[expert_idx, state_dict]
    """
    expert_params = {}

    # 假设 module 是 MoEVLMWrapper
    if hasattr(module, 'expert_collection'):
        collection = module.expert_collection
        for i in range(collection.num_experts):
            expert_params[i] = collection.get_expert_state_dict(i)

    return expert_params
```

---

## 4. 测试代码

```python
# tests/test_expert_lora.py

import torch
from verl.models.moe.expert_lora import (
    LoRALayer,
    SingleExpertLoRA,
    ExpertLoRACollection,
)


def test_lora_layer():
    """测试单个 LoRA 层"""
    lora = LoRALayer(in_features=256, out_features=256, r=16, alpha=32)

    x = torch.randn(2, 10, 256)
    delta = lora(x)

    assert delta.shape == x.shape
    print("LoRA layer test passed!")


def test_single_expert():
    """测试单个 Expert"""
    expert = SingleExpertLoRA(
        hidden_size=256,
        num_layers=4,
        target_modules=['q_proj', 'v_proj'],
        r=16,
    )

    x = torch.randn(2, 10, 256)

    for layer_idx in range(4):
        for module_name in ['q_proj', 'v_proj']:
            delta = expert.get_lora_delta(layer_idx, module_name, x)
            assert delta.shape == x.shape

    print("Single expert test passed!")


def test_expert_collection():
    """测试 Expert Collection"""
    collection = ExpertLoRACollection(
        hidden_size=256,
        num_layers=4,
        num_experts=4,
        target_modules=['q_proj', 'v_proj'],
        lora_r=16,
    )

    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 256)
    base_output = torch.randn(batch_size, seq_len, 256)

    # 测试单 expert
    output = collection.apply_single_expert(
        expert_idx=0,
        layer_idx=0,
        module_name='q_proj',
        base_output=base_output,
        x=x,
    )
    assert output.shape == base_output.shape

    # 测试加权组合
    top_k_indices = torch.tensor([[0, 1], [2, 3]])
    top_k_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4]])

    output = collection.apply_experts_weighted(
        layer_idx=0,
        module_name='q_proj',
        base_output=base_output,
        x=x,
        top_k_indices=top_k_indices,
        top_k_weights=top_k_weights,
    )
    assert output.shape == base_output.shape

    print("Expert collection test passed!")


def test_parameter_count():
    """验证参数量计算"""
    collection = ExpertLoRACollection(
        hidden_size=3584,  # Qwen2.5-VL-7B
        num_layers=28,
        num_experts=4,
        target_modules=['q_proj', 'v_proj'],
        lora_r=16,
    )

    total_params = sum(p.numel() for p in collection.parameters())
    expected = 4 * 28 * 2 * (3584 * 16 + 16 * 3584)  # 4 experts, 28 layers, 2 modules

    print(f"Total parameters: {total_params:,}")
    print(f"Expected: {expected:,}")
    assert abs(total_params - expected) < 1000, "Parameter count mismatch!"
    print("Parameter count test passed!")


if __name__ == "__main__":
    test_lora_layer()
    test_single_expert()
    test_expert_collection()
    test_parameter_count()
    print("\nAll expert LoRA tests passed!")
```

---

## 5. 下一步

实现 Expert LoRA 后，继续：
- [04_training_integration.md](./04_training_integration.md) - 集成到 verl 训练框架
