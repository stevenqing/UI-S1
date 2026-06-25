"""V13 Iterative Cooperative LoRA Linear Layer.

Extension of V12's SoftCooperativeLoRALinear with iterative message passing
in the shared low-rank space (r=128). Before blending, the two experts
"deliberate" through gated communication rounds.

V12: delta = B @ (r * A₁(x) + (1-r) * A₂(x)) * scaling
V13: delta = B @ (r * h₁ᵀ + (1-r) * h₂ᵀ) * scaling
     where h₁ᵀ, h₂ᵀ are outputs of T rounds of gated communication:
       g₁₂ = sigmoid(h₁ @ gate_12[t])  # input-dependent gate
       h₁  = h₁ + g₁₂ * (h₂ @ W₁₂[t])
       g₂₁ = sigmoid(h₂ @ gate_21[t])
       h₂  = h₂ + g₂₁ * (h₁ @ W₂₁[t])  # uses updated h₁

Key properties:
  - Input-dependent gates: communication amount adapts per-token
  - Sequential update: expert 2 sees expert 1's updated state
  - Negligible overhead: communication is r×r (128²), not d×d (3584²)
  - Reduces to V12 when gates=0 (graceful fallback)
  - Safe init: B=0 means output is zero regardless of communication
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class IterativeCooperativeLoRALinear(nn.Module):
    """Drop-in nn.Linear replacement with iterative cooperative LoRA.

    Two A matrices (A_1, A_2) provide two "specialization slots".
    A shared B matrix projects the blended low-rank output back to full dim.
    Per-token sigmoid routing decides the blend.
    Iterative communication in r-space before blending (set by wrapper).
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
        object.__setattr__(self, "_route_weight", None)

        # Communication params set by wrapper (NOT registered as parameter here)
        # When set: {'T': int, 'W_12': list[Param], 'W_21': list[Param],
        #            'gate_12': list[Param], 'gate_21': list[Param]}
        object.__setattr__(self, "_comm_params", None)

        # Routing noise std for exploration during RL rollouts
        self._routing_noise_std: float = 0.0

        # Cache last routing weights for balance loss computation
        self._last_routing_weights: Optional[torch.Tensor] = None
        self._last_routing_weights_for_loss: Optional[torch.Tensor] = None

        # Gate recording for CoPDA phase scores (V14)
        self._record_gates: bool = False
        self._last_comm_gate_records = None
        self._last_comm_gate_records_for_loss = None

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
        """Forward pass with soft routing and iterative communication.

        Args:
            x: [B, seq_len, D] or [seq_len, D]

        Returns:
            output: same shape as x, base + iterative cooperative LoRA delta
        """
        base_out = self.base_linear(x)

        # If no route weight set, pass through base only (ref model mode)
        if self._route_weight is None:
            self._last_routing_weights = None
            self._last_routing_weights_for_loss = None
            self._last_comm_gate_records = None
            self._last_comm_gate_records_for_loss = None
            return base_out

        x_drop = self.lora_dropout(x)
        dtype = x_drop.dtype

        # Compute routing weights: r = sigmoid(x @ w_route)
        w = self._route_weight.to(dtype)
        if w.shape[0] == x_drop.shape[-1]:
            logits = F.linear(x_drop, w.unsqueeze(0))  # [B, S, 1]

            # Add noise for exploration during RL rollouts
            if self._routing_noise_std > 0 and self.training:
                noise = torch.randn_like(logits) * self._routing_noise_std
                logits = logits + noise

            r = torch.sigmoid(logits)  # [B, S, 1]
        else:
            # Dimension mismatch (e.g. down_proj input=intermediate_size vs
            # route_weight=hidden_size): fall back to equal blend r=0.5
            r = torch.full(
                (*x_drop.shape[:-1], 1), 0.5,
                device=x_drop.device, dtype=dtype,
            )
        self._last_routing_weights = r.detach()
        self._last_routing_weights_for_loss = r if r.requires_grad else None
        self._last_comm_gate_records = None
        self._last_comm_gate_records_for_loss = None

        # Dual low-rank projections
        h_1 = F.linear(x_drop, self.lora_A_1.to(dtype))  # [B, S, r]
        h_2 = F.linear(x_drop, self.lora_A_2.to(dtype))  # [B, S, r]

        # Iterative communication in r-space
        if self._comm_params is not None and not self._disable_comm:
            T = self._comm_params['T']
            gate_accum = 0
            gate_records = []
            gate_records_for_loss = []
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
                    gate_records.append({
                        "round": t,
                        "g_12": g_12.detach(),
                        "g_21": g_21.detach(),
                    })
                    if g_12.requires_grad or g_21.requires_grad:
                        gate_records_for_loss.append({
                            "round": t,
                            "g_12": g_12,
                            "g_21": g_21,
                        })

            if self._record_gates:
                self._last_gate_mean = gate_accum / (2 * T)  # [B, S, 1]
                self._last_comm_gate_records = gate_records
                self._last_comm_gate_records_for_loss = gate_records_for_loss

        # Blend in r-space, single B matmul
        if self._inference_mode == "expert_1_only":
            h_blend = h_1
        elif self._inference_mode == "expert_2_only":
            h_blend = h_2
        else:
            h_blend = r * h_1 + (1 - r) * h_2  # [B, S, r]
        delta = F.linear(h_blend, self.lora_B.to(dtype)) * self.scaling

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
