"""
Token-Level Cooperative LoRA Linear Layer.

Replaces nn.Linear in transformer layers with a multi-adapter module.
Supports 2-agent (V, A) or 3-agent (V, T, A) routing:
  - LoRA_V: Image tokens (binding-optimized)
  - LoRA_T: Thought tokens (reasoning-optimized) — only when num_agents=3
  - LoRA_A: Instruction + action tokens (action-optimized)

Routing is determined by a fixed token_mask, not learned.

The key insight: attention is inherently cross-agent. When an action token's
query (LoRA_A) attends to an image token's key (LoRA_V), the attention
computation naturally combines two specializations. No explicit communication
channel needed.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CooperativeLoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with cooperative LoRA adapters.

    Supports 2-agent (V, A) or 3-agent (V, T, A) routing:

    2-agent (num_agents=2, default):
      Token mask is bool: True = LoRA_V, False = LoRA_A

    3-agent (num_agents=3):
      Token mask is int8: 0 = LoRA_A, 1 = LoRA_V, 2 = LoRA_T

    All deltas are computed for all tokens so that gradients flow to all
    adapters. ``torch.where`` selects which delta applies per token.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        num_agents: int = 2,
        soft_routing: bool = False,
        init_sep: float = 0.0,
        cooperative_comm: bool = False,
        gate_init: float = -3.0,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.num_agents = num_agents
        self.soft_routing = soft_routing
        self.cooperative_comm = cooperative_comm
        # Freeze base weights
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        in_f = base_linear.in_features
        out_f = base_linear.out_features
        self.scaling = alpha / r

        # Create LoRA params on same device as base linear
        device = base_linear.weight.device

        # LoRA_V — applied to image tokens (binding-optimized)
        self.lora_A_v = nn.Parameter(torch.zeros(r, in_f, device=device))
        self.lora_B_v = nn.Parameter(torch.zeros(out_f, r, device=device))

        # LoRA_A — applied to instruction/action tokens (action-optimized)
        self.lora_A_a = nn.Parameter(torch.zeros(r, in_f, device=device))
        self.lora_B_a = nn.Parameter(torch.zeros(out_f, r, device=device))

        # LoRA_T — applied to thought tokens (reasoning-optimized), 3-agent only
        if num_agents >= 3:
            self.lora_A_t = nn.Parameter(torch.zeros(r, in_f, device=device))
            self.lora_B_t = nn.Parameter(torch.zeros(out_f, r, device=device))

        self.lora_dropout = nn.Dropout(p=dropout)

        # Init: A = kaiming_uniform, B = zeros (starts as identity)
        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_a, a=math.sqrt(5))
        if num_agents >= 3:
            nn.init.kaiming_uniform_(self.lora_A_t, a=math.sqrt(5))
        # B stays zero -> delta starts at zero

        # Learnable separation parameter for soft routing (2-agent only)
        if soft_routing and num_agents == 2:
            self.sep = nn.Parameter(torch.tensor(init_sep))

        # Per-layer cooperative communication (v6, 2-agent only)
        if cooperative_comm and num_agents == 2:
            self.W_av = nn.Parameter(torch.zeros(r, r, device=device))   # A→V projection
            self.W_va = nn.Parameter(torch.zeros(r, r, device=device))   # V→A projection
            self.gate_av = nn.Parameter(torch.tensor(gate_init, device=device))  # sigmoid(-3)≈0.05
            self.gate_va = nn.Parameter(torch.tensor(gate_init, device=device))

        # Token mask set externally before forward
        self._token_mask: Optional[torch.Tensor] = None

    def set_token_mask(self, mask: Optional[torch.Tensor]):
        """Set the token routing mask.

        Args:
            mask: For num_agents=2: [B, seq_len] bool tensor (True = LoRA_V).
                  For num_agents=3: [B, seq_len] int8 tensor (0=A, 1=V, 2=T).
                  None clears the mask (base-only mode).
        """
        self._token_mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, D] or [seq_len, D] (unbatched)

        Returns:
            output: same shape as x, with per-token LoRA delta applied.
        """
        base_out = self.base_linear(x)

        # Resolve mask
        token_mask = self._token_mask
        if token_mask is None:
            if self.training:
                raise RuntimeError(
                    "CooperativeLoRALinear: token_mask is None during training. "
                    "Call set_token_mask() before model.forward()."
                )
            # Inference without mask -> base-only
            return base_out

        x_drop = self.lora_dropout(x)

        # Cast LoRA params to input dtype (base model may be bf16/fp16)
        dtype = x_drop.dtype

        # Compute ALL deltas for all tokens (all stay in autograd graph)
        h_v = F.linear(x_drop, self.lora_A_v.to(dtype))    # [B, S, r]
        h_a = F.linear(x_drop, self.lora_A_a.to(dtype))    # [B, S, r]

        if self.cooperative_comm and hasattr(self, 'W_av'):
            g_av = torch.sigmoid(self.gate_av)
            g_va = torch.sigmoid(self.gate_va)
            h_v = h_v + g_av * F.linear(h_a, self.W_av.to(dtype))  # V sees A
            h_a = h_a + g_va * F.linear(h_v, self.W_va.to(dtype))  # A sees V

        delta_v = F.linear(h_v, self.lora_B_v.to(dtype)) * self.scaling
        delta_a = F.linear(h_a, self.lora_B_a.to(dtype)) * self.scaling

        # Expand mask to match hidden dim: [B, seq_len] -> [B, seq_len, 1]
        mask = token_mask.unsqueeze(-1)

        if self.num_agents >= 3:
            # 3-agent: int8 mask (0=A, 1=V, 2=T)
            delta_t = F.linear(
                F.linear(x_drop, self.lora_A_t.to(dtype)),
                self.lora_B_t.to(dtype)
            ) * self.scaling
            delta = torch.where(mask == 1, delta_v,
                        torch.where(mask == 2, delta_t, delta_a))
        elif self.soft_routing:
            # 2-agent soft routing: learnable weighted sum
            s = torch.sigmoid(self.sep)
            mask_f = token_mask.unsqueeze(-1).to(dtype)  # [B, seq, 1]
            # Image tokens: w_v=s, w_a=1-s  (at s=1: pure V)
            # Text tokens:  w_v=1-s, w_a=s  (at s=1: pure A)
            w_v = mask_f * s + (1.0 - mask_f) * (1.0 - s)
            w_a = mask_f * (1.0 - s) + (1.0 - mask_f) * s
            delta = w_v * delta_v + w_a * delta_a
        else:
            # 2-agent: bool mask (True=V, False=A)
            delta = torch.where(mask, delta_v, delta_a)

        return base_out + delta

    def extra_repr(self) -> str:
        in_f = self.base_linear.in_features
        out_f = self.base_linear.out_features
        r = self.lora_A_v.shape[0]
        parts = [f"in={in_f}, out={out_f}, r={r}, scaling={self.scaling:.2f}, "
                 f"num_agents={self.num_agents}"]
        if self.soft_routing and hasattr(self, "sep"):
            s = torch.sigmoid(self.sep).item()
            parts.append(f", soft_routing=True, s={s:.4f}")
        if self.cooperative_comm and hasattr(self, "gate_av"):
            g_av = torch.sigmoid(self.gate_av).item()
            g_va = torch.sigmoid(self.gate_va).item()
            parts.append(f", comm=True, g_av={g_av:.4f}, g_va={g_va:.4f}")
        return "".join(parts)
