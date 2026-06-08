"""V18 K-Expert Cooperative LoRA Linear Layer.

Extension of V13's IterativeCooperativeLoRALinear from 2 experts to K experts
with softmax routing and configurable communication topology.

V13: delta = B @ (r * A₁(x) + (1-r) * A₂(x)) * scaling
V18: delta = B @ (Σ_k w_k * h_k) * scaling
     where w = softmax(x @ W_route), h_k after T rounds of topology-dependent comm

Key changes from V13:
  - 2 → K experts (lora_A ParameterList)
  - sigmoid → softmax routing
  - pairwise comm → configurable topology (none, top2, shared, full)
  - Caches expert outputs for diversity loss

Communication topologies:
  - none: No communication between experts
  - top2: Only top-2 activated experts communicate (shared W)
  - shared: All pairs share same W; expert i receives avg msg from all j≠i
  - full: Each directed pair has its own W and gate
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class KExpertCooperativeLoRALinear(nn.Module):
    """Drop-in nn.Linear replacement with K-expert cooperative LoRA.

    K A matrices provide K "specialization slots".
    A shared B matrix projects the blended low-rank output back to full dim.
    Per-token softmax routing decides the K-way blend.
    Topology-dependent communication in r-space before blending (set by wrapper).
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 128,
        alpha: int = 256,
        dropout: float = 0.05,
        num_experts: int = 4,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        in_f = base_linear.in_features
        out_f = base_linear.out_features
        self.scaling = alpha / r
        self.num_experts = num_experts

        device = base_linear.weight.device

        # K A matrices — K specialization slots
        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.zeros(r, in_f, device=device))
            for _ in range(num_experts)
        ])

        # Shared B matrix
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=device))

        self.lora_dropout = nn.Dropout(p=dropout)

        # Init: A = kaiming_uniform, B = zeros (starts as identity)
        for i in range(num_experts):
            nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))
        # B stays zero -> delta starts at zero

        # Routing weight matrix set by wrapper (NOT registered as parameter here)
        # Shape: [hidden_size, K] for softmax routing
        object.__setattr__(self, "_route_weight", None)

        # Communication params set by wrapper (NOT registered as parameter here)
        # Format depends on topology — see set_comm_params docstring
        object.__setattr__(self, "_comm_params", None)

        # Routing noise std for exploration during RL rollouts
        self._routing_noise_std: float = 0.0

        # Cache last routing weights for balance loss computation
        # Shape: [B, S, K] after softmax
        self._last_routing_weights: Optional[torch.Tensor] = None

        # Cache last expert outputs for diversity loss computation
        # List of K tensors, each [B, S, r]
        self._last_expert_outputs: Optional[list] = None

        # Whether to cache expert outputs (only during training)
        self._cache_expert_outputs: bool = False

    def set_route_weight(self, w: Optional[nn.Parameter]):
        """Attach the per-layer routing matrix (owned by wrapper).

        Args:
            w: Parameter of shape [hidden_size, K] or None to disable.
        """
        object.__setattr__(self, "_route_weight", w)

    def set_comm_params(self, params):
        """Attach per-layer communication params (owned by wrapper).

        Args:
            params: dict with topology-dependent structure:
                Common keys: 'T' (num rounds), 'topology' (str)

                topology='none': no additional keys

                topology='top2':
                    'W_comm': list[Param[r,r]] per round
                    'gate': list[Param[r]] per round

                topology='shared':
                    'W_comm': list[Param[r,r]] per round
                    'gate': list[Param[r]] per round

                topology='full':
                    'W_comm': list[list[Param[r,r]]] per round, K*(K-1) pairs
                    'gate': list[list[Param[r]]] per round, K*(K-1) pairs
                    'pair_map': list of (src, dst) tuples

                Set to None to disable communication.
        """
        object.__setattr__(self, "_comm_params", params)

    def _apply_communication(self, h: list, weights: torch.Tensor) -> list:
        """Apply topology-dependent communication between experts.

        Args:
            h: list of K tensors, each [B, S, r]
            weights: routing weights [B, S, K]

        Returns:
            h: list of K tensors after communication
        """
        if self._comm_params is None:
            return h

        topology = self._comm_params.get('topology', 'none')
        if topology == 'none':
            return h

        T = self._comm_params['T']
        dtype = h[0].dtype
        K = self.num_experts

        for t in range(T):
            if topology == 'top2':
                h = self._comm_top2(h, weights, t, dtype)
            elif topology == 'shared':
                h = self._comm_shared(h, weights, t, dtype)
            elif topology == 'full':
                h = self._comm_full(h, weights, t, dtype)

        return h

    def _comm_top2(self, h: list, weights: torch.Tensor, t: int, dtype) -> list:
        """Top-2 communication: only the two highest-weight experts communicate.

        Uses shared W_comm and gate across all pairs.
        """
        W = self._comm_params['W_comm'][t].to(dtype)
        gate_vec = self._comm_params['gate'][t].to(dtype)
        K = self.num_experts

        # Find top-2 experts per token: [B, S, 2]
        _, top2_idx = weights.topk(2, dim=-1)

        # For efficiency, apply shared communication to all experts but
        # mask to only top-2. This avoids complex scatter/gather.
        h_new = list(h)  # shallow copy
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                # Mask: both i and j must be in top-2
                i_in_top2 = (top2_idx == i).any(dim=-1)  # [B, S]
                j_in_top2 = (top2_idx == j).any(dim=-1)  # [B, S]
                pair_mask = (i_in_top2 & j_in_top2).unsqueeze(-1).float()  # [B, S, 1]

                # Message from j to i
                g = torch.sigmoid(F.linear(h[i], gate_vec.unsqueeze(0)))  # [B, S, 1]
                msg = F.linear(h[j], W)  # [B, S, r]
                h_new[i] = h_new[i] + pair_mask * g * msg

        return h_new

    def _comm_shared(self, h: list, weights: torch.Tensor, t: int, dtype) -> list:
        """Shared communication: all pairs use same W; expert i receives avg msg from j≠i."""
        W = self._comm_params['W_comm'][t].to(dtype)
        gate_vec = self._comm_params['gate'][t].to(dtype)
        K = self.num_experts

        h_new = []
        for i in range(K):
            # Gate based on receiver's state
            g = torch.sigmoid(F.linear(h[i], gate_vec.unsqueeze(0)))  # [B, S, 1]

            # Average message from all other experts
            msg_sum = torch.zeros_like(h[i])
            for j in range(K):
                if j != i:
                    msg_sum = msg_sum + F.linear(h[j], W)
            avg_msg = msg_sum / (K - 1)

            h_new.append(h[i] + g * avg_msg)

        return h_new

    def _comm_full(self, h: list, weights: torch.Tensor, t: int, dtype) -> list:
        """Full communication: each directed pair (j→i) has its own W and gate."""
        pair_map = self._comm_params['pair_map']
        W_list = self._comm_params['W_comm'][t]
        gate_list = self._comm_params['gate'][t]
        K = self.num_experts

        # Accumulate messages for each expert
        h_new = [hi.clone() for hi in h]

        for pair_idx, (src, dst) in enumerate(pair_map):
            W = W_list[pair_idx].to(dtype)
            gate_vec = gate_list[pair_idx].to(dtype)

            g = torch.sigmoid(F.linear(h[dst], gate_vec.unsqueeze(0)))  # [B, S, 1]
            msg = F.linear(h[src], W)  # [B, S, r]
            h_new[dst] = h_new[dst] + g * msg

        return h_new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with softmax routing and topology-dependent communication.

        Args:
            x: [B, seq_len, D] or [seq_len, D]

        Returns:
            output: same shape as x, base + K-expert cooperative LoRA delta
        """
        base_out = self.base_linear(x)

        # If no route weight set, pass through base only (ref model mode)
        if self._route_weight is None:
            self._last_routing_weights = None
            self._last_expert_outputs = None
            return base_out

        x_drop = self.lora_dropout(x)
        dtype = x_drop.dtype
        K = self.num_experts

        # 1. Softmax routing
        w = self._route_weight.to(dtype)  # [hidden_size, K]
        if w.shape[0] == x_drop.shape[-1]:
            logits = F.linear(x_drop, w.t())  # [B, S, K]

            # Add noise for exploration during RL rollouts
            # Apply in both train and eval: during generation (eval mode),
            # noise ensures different routing → different trajectories → RL signal
            if self._routing_noise_std > 0:
                noise = torch.randn_like(logits) * self._routing_noise_std
                logits = logits + noise

            weights = F.softmax(logits, dim=-1)  # [B, S, K]
        else:
            # Dimension mismatch: fall back to uniform 1/K
            weights = torch.full(
                (*x_drop.shape[:-1], K), 1.0 / K,
                device=x_drop.device, dtype=dtype,
            )
        self._last_routing_weights = weights.detach()

        # 2. K expert projections
        h = [F.linear(x_drop, self.lora_A[i].to(dtype)) for i in range(K)]  # K × [B, S, r]

        # Cache expert outputs before communication for diversity loss
        # NOTE: do NOT detach here — diversity loss needs gradients to flow
        # back through A matrices to push experts apart.
        if self._cache_expert_outputs and self.training:
            self._last_expert_outputs = list(h)

        # 3. Communication (topology-dependent)
        h = self._apply_communication(h, weights)

        # 4. Weighted blend
        h_stack = torch.stack(h, dim=2)  # [B, S, K, r]
        h_blend = (weights.unsqueeze(-1) * h_stack).sum(dim=2)  # [B, S, r]

        # 5. Shared B projection
        delta = F.linear(h_blend, self.lora_B.to(dtype)) * self.scaling

        return base_out + delta

    def extra_repr(self) -> str:
        in_f = self.base_linear.in_features
        out_f = self.base_linear.out_features
        r = self.lora_A[0].shape[0]
        has_route = self._route_weight is not None
        has_comm = self._comm_params is not None
        T = self._comm_params['T'] if has_comm else 0
        topology = self._comm_params.get('topology', 'none') if has_comm else 'none'
        return (f"in={in_f}, out={out_f}, r={r}, K={self.num_experts}, "
                f"scaling={self.scaling:.2f}, "
                f"route={'attached' if has_route else 'none'}, "
                f"topology={topology}, comm_rounds={T}, "
                f"noise_std={self._routing_noise_std}")
