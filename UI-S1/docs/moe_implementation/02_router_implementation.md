# Router 模块实现

## 1. 设计决策

### 1.1 Router 类型选择

| Router 类型 | 输入 | 优点 | 缺点 |
|------------|------|------|------|
| **Text-Only** (选择) | instruction text features | 简单、可解释、验证假设 | 忽略 visual context |
| Multimodal | fused features | 更多信息 | 过拟合风险、不可解释 |
| Token-level | 每个 token | 细粒度 | 复杂、难以分析 |

**我们选择 Text-Only Router**，原因：
1. Pilot 实验目标是验证 expert 按 instruction 类型分化
2. Text-only 使得 routing 完全可解释
3. 如果 text-only 有效，说明 instruction 本身足以决定需要什么 expert

### 1.2 Routing 策略

| 策略 | 描述 | 选择 |
|------|------|------|
| Soft Routing | 所有 experts 加权求和 | ❌ |
| **Top-k Hard Routing** | 只选择 top-k experts | ✅ k=1 或 k=2 |

选择 Top-k 的原因：
- 更清晰的分化信号
- 易于分析每个 expert 的专长
- 推理时计算效率高

---

## 2. 完整实现代码

### 2.1 文件结构

```
verl/models/moe/
├── __init__.py
├── router.py          ← 本文档
├── expert_lora.py
└── moe_wrapper.py
```

### 2.2 Router 实现

```python
# verl/models/moe/router.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class RouterOutput:
    """Router 输出结构"""
    routing_weights: torch.Tensor      # [B, num_experts] softmax weights
    top_k_weights: torch.Tensor        # [B, top_k] renormalized weights
    top_k_indices: torch.Tensor        # [B, top_k] selected expert indices
    router_logits: torch.Tensor        # [B, num_experts] raw logits (for analysis)


class TextOnlyRouter(nn.Module):
    """
    基于 instruction text features 进行路由决策

    设计原则:
    - 只使用 instruction 部分的 text features
    - 不使用 screenshot visual features
    - 输出可解释的 routing 决策

    Architecture:
        instruction_features [B, hidden_size]
                    │
                    ▼
            ┌───────────────┐
            │   Linear      │  hidden_size → router_hidden
            │   GELU        │
            │   Dropout     │
            │   Linear      │  router_hidden → num_experts
            └───────────────┘
                    │
                    ▼
            ┌───────────────┐
            │   Softmax     │
            └───────────────┘
                    │
                    ▼
            routing_weights [B, num_experts]
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 4,
        router_hidden: int = 256,
        top_k: int = 1,
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        """
        Args:
            hidden_size: Base model hidden dimension (e.g., 3584 for Qwen2.5-VL-7B)
            num_experts: Number of expert LoRAs
            router_hidden: Hidden dimension of router MLP
            top_k: Number of experts to select
            dropout: Dropout rate
            temperature: Softmax temperature (lower = sharper distribution)
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature

        # Router MLP
        self.router = nn.Sequential(
            nn.Linear(hidden_size, router_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden, num_experts),
        )

        # 初始化：使用较小的权重，初始时接近均匀分布
        self._init_weights()

    def _init_weights(self):
        """初始化 router 权重，确保初始时 routing 接近均匀"""
        for module in self.router:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        instruction_features: torch.Tensor,
        return_all: bool = True,
    ) -> RouterOutput:
        """
        计算 routing weights

        Args:
            instruction_features: [B, hidden_size] pooled instruction representation
            return_all: 是否返回完整的 RouterOutput

        Returns:
            RouterOutput with routing weights and top-k selection
        """
        # 计算 router logits
        router_logits = self.router(instruction_features)  # [B, num_experts]

        # 应用 temperature scaling
        scaled_logits = router_logits / self.temperature

        # Softmax to get routing weights
        routing_weights = F.softmax(scaled_logits, dim=-1)  # [B, num_experts]

        # Top-k selection
        top_k_weights, top_k_indices = routing_weights.topk(self.top_k, dim=-1)

        # Renormalize top-k weights to sum to 1
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        return RouterOutput(
            routing_weights=routing_weights,
            top_k_weights=top_k_weights,
            top_k_indices=top_k_indices,
            router_logits=router_logits,
        )

    def get_routing_distribution(self, instruction_features: torch.Tensor) -> torch.Tensor:
        """
        获取完整的 routing 分布 (用于分析)

        Returns:
            routing_weights: [B, num_experts]
        """
        with torch.no_grad():
            router_logits = self.router(instruction_features)
            return F.softmax(router_logits / self.temperature, dim=-1)


class InstructionFeatureExtractor(nn.Module):
    """
    从 VLM hidden states 中提取 instruction 部分的 features

    策略:
    1. 使用 instruction tokens 的 mean pooling
    2. 或使用最后一个 instruction token (类似 CLS)
    """

    def __init__(self, pooling_strategy: str = 'mean'):
        """
        Args:
            pooling_strategy: 'mean', 'last', or 'first'
        """
        super().__init__()
        self.pooling_strategy = pooling_strategy

    def forward(
        self,
        hidden_states: torch.Tensor,
        instruction_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        从 hidden states 中提取 instruction features

        Args:
            hidden_states: [B, seq_len, hidden_size]
            instruction_mask: [B, seq_len] - 1 表示 instruction token, 0 表示其他

        Returns:
            instruction_features: [B, hidden_size]
        """
        if self.pooling_strategy == 'mean':
            # Mean pooling over instruction tokens
            mask_expanded = instruction_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
            sum_features = (hidden_states * mask_expanded).sum(dim=1)  # [B, hidden_size]
            num_tokens = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]
            return sum_features / num_tokens

        elif self.pooling_strategy == 'last':
            # 使用每个样本最后一个 instruction token
            batch_size = hidden_states.size(0)
            features = []
            for i in range(batch_size):
                instruction_indices = instruction_mask[i].nonzero(as_tuple=True)[0]
                if len(instruction_indices) > 0:
                    last_idx = instruction_indices[-1]
                    features.append(hidden_states[i, last_idx])
                else:
                    # Fallback: 使用序列最后一个 token
                    features.append(hidden_states[i, -1])
            return torch.stack(features, dim=0)

        elif self.pooling_strategy == 'first':
            # 使用每个样本第一个 instruction token
            batch_size = hidden_states.size(0)
            features = []
            for i in range(batch_size):
                instruction_indices = instruction_mask[i].nonzero(as_tuple=True)[0]
                if len(instruction_indices) > 0:
                    first_idx = instruction_indices[0]
                    features.append(hidden_states[i, first_idx])
                else:
                    features.append(hidden_states[i, 0])
            return torch.stack(features, dim=0)

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")


def create_instruction_mask(
    input_ids: torch.Tensor,
    tokenizer,
    image_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    创建 instruction mask，标记哪些 tokens 属于 instruction 文本

    对于 Qwen2.5-VL 格式:
    <|im_start|>user
    <|vision_start|><|image_pad|>...<|vision_end|>
    Click on the search button
    <|im_end|>

    我们想要标记 "Click on the search button" 部分

    Args:
        input_ids: [B, seq_len]
        tokenizer: Qwen2.5-VL tokenizer
        image_token_id: image token ID (if known)

    Returns:
        instruction_mask: [B, seq_len]
    """
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)

    # 获取特殊 token IDs
    vision_start_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
    im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')

    for i in range(batch_size):
        ids = input_ids[i].tolist()

        # 找到 vision_end 的位置
        vision_end_pos = -1
        for j, token_id in enumerate(ids):
            if token_id == vision_end_id:
                vision_end_pos = j
                break

        # 找到 im_end 的位置 (在 vision_end 之后)
        im_end_pos = seq_len
        for j in range(vision_end_pos + 1, len(ids)):
            if ids[j] == im_end_id:
                im_end_pos = j
                break

        # 标记 instruction 区域 (vision_end 到 im_end 之间)
        if vision_end_pos >= 0:
            mask[i, vision_end_pos + 1:im_end_pos] = True

    return mask
```

---

## 3. Load Balancing Loss

### 3.1 为什么需要 Load Balancing

问题：如果没有平衡约束，所有样本可能都路由到同一个 expert（坍塌）

```
Without Balance Loss:
Expert 0: 95% ████████████████████
Expert 1:  2% █
Expert 2:  2% █
Expert 3:  1%

With Balance Loss:
Expert 0: 28% ██████
Expert 1: 24% █████
Expert 2: 26% █████
Expert 3: 22% ████
```

### 3.2 Balance Loss 实现

```python
# verl/trainer/ppo/moe_loss.py

import torch
import torch.nn.functional as F
from typing import Dict, Literal


def compute_load_balance_loss(
    routing_weights: torch.Tensor,
    num_experts: int,
    balance_type: Literal['mse', 'switch', 'entropy'] = 'mse',
) -> torch.Tensor:
    """
    计算 load balancing loss

    Args:
        routing_weights: [B, num_experts] - softmax routing weights
        num_experts: number of experts
        balance_type: loss type
            - 'mse': MSE to uniform distribution
            - 'switch': Switch Transformer style
            - 'entropy': Maximize routing entropy

    Returns:
        balance_loss: scalar tensor
    """
    # 平均 routing 分布
    avg_routing = routing_weights.mean(dim=0)  # [num_experts]

    if balance_type == 'mse':
        # MSE to uniform distribution
        target = torch.ones_like(avg_routing) / num_experts
        loss = F.mse_loss(avg_routing, target)

    elif balance_type == 'switch':
        # Switch Transformer: aux_loss = sum(f_i * P_i) * num_experts
        # f_i = fraction of tokens to expert i
        # P_i = mean routing probability for expert i
        f = (routing_weights.argmax(dim=-1, keepdim=True) == torch.arange(
            num_experts, device=routing_weights.device
        )).float().mean(dim=0)
        P = avg_routing
        loss = num_experts * (f * P).sum()

    elif balance_type == 'entropy':
        # 最大化 routing 分布的熵
        # 高熵 = 均匀分布
        entropy = -(avg_routing * torch.log(avg_routing + 1e-10)).sum()
        max_entropy = torch.log(torch.tensor(num_experts, dtype=torch.float, device=routing_weights.device))
        loss = 1.0 - entropy / max_entropy  # 归一化到 [0, 1]

    else:
        raise ValueError(f"Unknown balance_type: {balance_type}")

    return loss


def compute_router_z_loss(router_logits: torch.Tensor, z_loss_weight: float = 0.001) -> torch.Tensor:
    """
    Router Z-loss: 惩罚过大的 router logits，提高训练稳定性

    From: ST-MoE (https://arxiv.org/abs/2202.08906)

    Args:
        router_logits: [B, num_experts] raw router logits
        z_loss_weight: weight for z-loss

    Returns:
        z_loss: scalar tensor
    """
    # z-loss = mean(log(sum(exp(logits))))^2
    log_z = torch.logsumexp(router_logits, dim=-1)  # [B]
    z_loss = z_loss_weight * (log_z ** 2).mean()
    return z_loss


class MoELoss:
    """
    MoE 训练的完整 loss 计算
    """

    def __init__(
        self,
        num_experts: int,
        balance_weight: float = 0.1,
        balance_type: str = 'mse',
        z_loss_weight: float = 0.001,
    ):
        self.num_experts = num_experts
        self.balance_weight = balance_weight
        self.balance_type = balance_type
        self.z_loss_weight = z_loss_weight

    def __call__(
        self,
        lm_loss: torch.Tensor,
        routing_weights: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算总 loss

        Args:
            lm_loss: Language modeling loss from base model
            routing_weights: [B, num_experts] softmax weights
            router_logits: [B, num_experts] raw logits

        Returns:
            Dict with 'total_loss' and individual components
        """
        # Balance loss
        balance_loss = compute_load_balance_loss(
            routing_weights,
            self.num_experts,
            self.balance_type,
        )

        # Z-loss
        z_loss = compute_router_z_loss(router_logits, self.z_loss_weight)

        # Total loss
        total_loss = lm_loss + self.balance_weight * balance_loss + z_loss

        return {
            'total_loss': total_loss,
            'lm_loss': lm_loss,
            'balance_loss': balance_loss,
            'z_loss': z_loss,
        }
```

---

## 4. 超参数调优指南

### 4.1 Balance Weight

```
balance_weight = 0.01  → 可能坍塌 (一个 expert 主导)
                 观察: routing 分布严重倾斜
                 行动: 增加 balance_weight

balance_weight = 0.1   → 适中起点 (推荐)
                 观察: 应该看到逐渐分化
                 行动: 根据分化效果微调

balance_weight = 1.0   → 可能过于均匀
                 观察: routing 几乎完全均匀，没有分化
                 行动: 减少 balance_weight
```

### 4.2 Temperature

```
temperature = 0.5  → 更 sharp 的分布，更明确的 expert 选择
temperature = 1.0  → 默认
temperature = 2.0  → 更 soft 的分布，多个 experts 贡献
```

### 4.3 Top-k 选择

```
top_k = 1  → 每个样本只用一个 expert
           优点: 分化信号最清晰，推理最快
           缺点: 梯度只流向选中的 expert

top_k = 2  → 每个样本用两个 experts
           优点: 更平滑的梯度，可能更好的泛化
           缺点: 分化可能不够明显
```

---

## 5. 测试代码

```python
# tests/test_router.py

import torch
from verl.models.moe.router import TextOnlyRouter, InstructionFeatureExtractor

def test_router_output_shape():
    """测试 Router 输出形状"""
    batch_size = 8
    hidden_size = 3584
    num_experts = 4

    router = TextOnlyRouter(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=1,
    )

    # 模拟 instruction features
    instruction_features = torch.randn(batch_size, hidden_size)

    output = router(instruction_features)

    assert output.routing_weights.shape == (batch_size, num_experts)
    assert output.top_k_weights.shape == (batch_size, 1)
    assert output.top_k_indices.shape == (batch_size, 1)
    print("Router output shape test passed!")


def test_router_distribution():
    """测试 Router 输出是有效的概率分布"""
    router = TextOnlyRouter(hidden_size=256, num_experts=4)
    features = torch.randn(4, 256)

    output = router(features)

    # 检查 softmax 属性
    assert torch.allclose(output.routing_weights.sum(dim=-1), torch.ones(4))
    assert (output.routing_weights >= 0).all()
    assert (output.routing_weights <= 1).all()
    print("Router distribution test passed!")


def test_balance_loss():
    """测试 Balance Loss 行为"""
    from verl.trainer.ppo.moe_loss import compute_load_balance_loss

    # 完全均匀的分布 - loss 应该接近 0
    uniform = torch.ones(8, 4) / 4
    loss_uniform = compute_load_balance_loss(uniform, 4, 'mse')
    assert loss_uniform < 0.01, f"Uniform distribution loss should be ~0, got {loss_uniform}"

    # 完全坍塌的分布 - loss 应该较大
    collapsed = torch.zeros(8, 4)
    collapsed[:, 0] = 1.0
    loss_collapsed = compute_load_balance_loss(collapsed, 4, 'mse')
    assert loss_collapsed > 0.1, f"Collapsed distribution loss should be large, got {loss_collapsed}"

    print(f"Balance loss test passed! uniform={loss_uniform:.4f}, collapsed={loss_collapsed:.4f}")


if __name__ == "__main__":
    test_router_output_shape()
    test_router_distribution()
    test_balance_loss()
    print("\nAll router tests passed!")
```

---

## 6. 下一步

实现 Router 后，继续：
- [03_expert_lora.md](./03_expert_lora.md) - Expert LoRA 实现
