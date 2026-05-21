"""V12 Soft Cooperative LoRA Linear Layer.

Drop-in nn.Linear replacement with dual A matrices (A_1, A_2), shared B matrix,
and per-token sigmoid routing. No hard token-type routing, no side-channel
communication — cooperation emerges through attention.

Routing: r = sigmoid(x @ w_route + noise), delta = B @ (r * A_1(x) + (1-r) * A_2(x)) * scaling
- w_route is a per-layer routing vector (dim=in_features), set by the wrapper
- noise enables structured exploration during RL rollouts
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftCooperativeLoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with soft-routed dual LoRA adapters.

    Two A matrices (A_1, A_2) provide two "specialization slots".
    A shared B matrix projects the blended low-rank output back to full dim.
    Per-token sigmoid routing decides the blend, driven by a routing vector
    shared across q/k/v/o modules within the same transformer layer.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 128,
        alpha: int = 256,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        in_f = base_linear.in_features
        out_f = base_linear.out_features
        self.scaling = alpha / r

        device = base_linear.weight.device

        # Dual A matrices — two specialization slots
        self.lora_A_1 = nn.Parameter(torch.zeros(r, in_f, device=device))
        self.lora_A_2 = nn.Parameter(torch.zeros(r, in_f, device=device))

        # Shared B matrix
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=device))

        self.lora_dropout = nn.Dropout(p=dropout)

        # Init: A = kaiming_uniform, B = zeros (starts as identity)
        nn.init.kaiming_uniform_(self.lora_A_1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_2, a=math.sqrt(5))
        # B stays zero -> delta starts at zero

        # Routing weight vector set by wrapper (NOT registered as parameter here)
        # Shape: [in_features] — shared across q/k/v/o in same layer
        object.__setattr__(self, "_route_weight", None)

        # Routing noise std for exploration during RL rollouts
        self._routing_noise_std: float = 0.0

        # Cache last routing weights for balance loss computation
        self._last_routing_weights: Optional[torch.Tensor] = None

    def set_route_weight(self, w: Optional[nn.Parameter]):
        """Attach the per-layer routing vector (owned by wrapper)."""
        object.__setattr__(self, "_route_weight", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with soft routing.

        Args:
            x: [B, seq_len, D] or [seq_len, D]

        Returns:
            output: same shape as x, base + soft-routed LoRA delta
        """
        base_out = self.base_linear(x)

        # If no route weight set, pass through base only (ref model mode)
        if self._route_weight is None:
            self._last_routing_weights = None
            return base_out

        x_drop = self.lora_dropout(x)
        dtype = x_drop.dtype

        # Compute routing weights: r = sigmoid(x @ w_route)
        w = self._route_weight.to(dtype)
        logits = F.linear(x_drop, w.unsqueeze(0))  # [B, S, 1]

        # Add noise for exploration during RL rollouts
        if self._routing_noise_std > 0 and self.training:
            noise = torch.randn_like(logits) * self._routing_noise_std
            logits = logits + noise

        r = torch.sigmoid(logits)  # [B, S, 1]
        self._last_routing_weights = r.detach()

        # Dual low-rank projections
        h_1 = F.linear(x_drop, self.lora_A_1.to(dtype))  # [B, S, r]
        h_2 = F.linear(x_drop, self.lora_A_2.to(dtype))  # [B, S, r]

        # Blend in r-space, single B matmul
        h_blend = r * h_1 + (1 - r) * h_2  # [B, S, r]
        delta = F.linear(h_blend, self.lora_B.to(dtype)) * self.scaling

        return base_out + delta

    def extra_repr(self) -> str:
        in_f = self.base_linear.in_features
        out_f = self.base_linear.out_features
        r = self.lora_A_1.shape[0]
        has_route = self._route_weight is not None
        return (f"in={in_f}, out={out_f}, r={r}, scaling={self.scaling:.2f}, "
                f"route={'attached' if has_route else 'none'}, "
                f"noise_std={self._routing_noise_std}")
