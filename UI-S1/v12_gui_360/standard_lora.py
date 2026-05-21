"""Standard LoRA Linear Layer — baseline for V12 cooperative comparison.

Single A/B matrices per module, no routing, no dual experts.
This is the classic LoRA architecture (Hu et al., 2022).

delta = B @ A(x) * scaling
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardLoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with standard LoRA adapter."""

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 210,
        alpha: int = 420,
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

        self.lora_A = nn.Parameter(torch.zeros(r, in_f, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=device))
        self.lora_dropout = nn.Dropout(p=dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B = 0 -> delta starts at zero

        self._enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_linear(x)

        if not self._enabled:
            return base_out

        x_drop = self.lora_dropout(x)
        dtype = x_drop.dtype
        h = F.linear(x_drop, self.lora_A.to(dtype))
        delta = F.linear(h, self.lora_B.to(dtype)) * self.scaling
        return base_out + delta

    def extra_repr(self) -> str:
        in_f = self.base_linear.in_features
        out_f = self.base_linear.out_features
        r = self.lora_A.shape[0]
        return f"in={in_f}, out={out_f}, r={r}, scaling={self.scaling:.2f}, enabled={self._enabled}"
