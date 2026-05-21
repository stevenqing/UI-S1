"""V14 Cooperative LoRA Linear Layer — Separate B Matrices.

Key change from V13:
  Separate B matrices (B_1, B_2) — each expert projects to its own output subspace.
  This gives routing actual differentiating power: r can select between two
  genuinely different output directions, not just two points in span(B).
  Communication stays always-on (same as V13) — it was never the problem.

V13: delta = B @ (r * h₁ᵀ + (1-r) * h₂ᵀ) * scaling           (shared B)
V14: delta = r * (B₁ @ h₁ᵀ) + (1-r) * (B₂ @ h₂ᵀ) * scaling   (separate B)

Parameter increase: +25M (one extra B matrix at r=128 × 28 layers × 4 projections).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class IterativeCooperativeLoRALinear(nn.Module):
    """Drop-in nn.Linear replacement with iterative cooperative LoRA.

    Two A matrices (A_1, A_2) provide two "specialization slots".
    Two B matrices (B_1, B_2) project each expert to its own output subspace.
    Per-token sigmoid routing decides the blend in output space.
    Iterative communication in r-space before projection (set by wrapper).
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

        # Separate B matrices — each expert has its own output projection
        self.lora_B_1 = nn.Parameter(torch.zeros(out_f, r, device=device))
        self.lora_B_2 = nn.Parameter(torch.zeros(out_f, r, device=device))

        self.lora_dropout = nn.Dropout(p=dropout)

        # Init: A = kaiming_uniform, B = zeros (starts as identity)
        nn.init.kaiming_uniform_(self.lora_A_1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_2, a=math.sqrt(5))
        # B stays zero -> delta starts at zero (safe init)

        # Routing weight vector set by wrapper (NOT registered as parameter here)
        object.__setattr__(self, "_route_weight", None)

        # Communication params set by wrapper (NOT registered as parameter here)
        # When set: {'T': int, 'W_12': list[Param], 'W_21': list[Param],
        #            'gate_12': list[Param], 'gate_21': list[Param]}
        object.__setattr__(self, "_comm_params", None)

        # Routing noise std for exploration during RL rollouts
        self._routing_noise_std: float = 0.0

        # Cache last routing weights for balance loss computation
        self._last_routing_weights: Optional[torch.Tensor] = None

        # Gate recording for CoPDA phase scores
        self._record_gates: bool = False

        # Inference mode override: None (normal), "expert_1_only", "expert_2_only"
        self._inference_mode: Optional[str] = None

        # Disable communication (for ablation)
        self._disable_comm: bool = False

    def set_route_weight(self, w: Optional[nn.Parameter]):
        """Attach the per-layer routing vector (owned by wrapper)."""
        object.__setattr__(self, "_route_weight", w)

    def set_comm_params(self, params):
        """Attach per-layer communication params (owned by wrapper).

        Args:
            params: dict with keys 'T', 'W_12', 'W_21', 'gate_12', 'gate_21'
                    where W_12[t] is [r, r] and gate_12[t] is [r].
                    Set to None to disable communication.
        """
        object.__setattr__(self, "_comm_params", params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with soft routing, iterative communication, and separate B.

        Args:
            x: [B, seq_len, D] or [seq_len, D]

        Returns:
            output: same shape as x, base + cooperative LoRA delta
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

        # Iterative communication in r-space (always-on, same as V13)
        if self._comm_params is not None and not self._disable_comm:
            T = self._comm_params['T']
            gate_accum = 0
            for t in range(T):
                # Expert 1 receives from expert 2
                g_12 = torch.sigmoid(
                    F.linear(h_1, self._comm_params['gate_12'][t].to(dtype).unsqueeze(0))
                )  # [B, S, 1]
                h_1 = h_1 + g_12 * F.linear(h_2, self._comm_params['W_12'][t].to(dtype))

                # Expert 2 receives from (updated) expert 1
                g_21 = torch.sigmoid(
                    F.linear(h_2, self._comm_params['gate_21'][t].to(dtype).unsqueeze(0))
                )  # [B, S, 1]
                h_2 = h_2 + g_21 * F.linear(h_1, self._comm_params['W_21'][t].to(dtype))

                if self._record_gates:
                    gate_accum = gate_accum + g_12.detach() + g_21.detach()

            if self._record_gates:
                self._last_gate_mean = gate_accum / (2 * T)  # [B, S, 1]

        # Separate B projections → output-space blend
        if self._inference_mode == "expert_1_only":
            delta = F.linear(h_1, self.lora_B_1.to(dtype)) * self.scaling
        elif self._inference_mode == "expert_2_only":
            delta = F.linear(h_2, self.lora_B_2.to(dtype)) * self.scaling
        else:
            delta_1 = F.linear(h_1, self.lora_B_1.to(dtype)) * self.scaling
            delta_2 = F.linear(h_2, self.lora_B_2.to(dtype)) * self.scaling
            delta = r * delta_1 + (1 - r) * delta_2  # [B, S, out_f]

        return base_out + delta

    def extra_repr(self) -> str:
        in_f = self.base_linear.in_features
        out_f = self.base_linear.out_features
        r = self.lora_A_1.shape[0]
        has_route = self._route_weight is not None
        has_comm = self._comm_params is not None
        T = self._comm_params['T'] if has_comm else 0
        return (f"in={in_f}, out={out_f}, r={r}, scaling={self.scaling:.2f}, "
                f"route={'attached' if has_route else 'none'}, "
                f"comm_rounds={T}, "
                f"noise_std={self._routing_noise_std}")
