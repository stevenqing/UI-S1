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
        gate_type: str = "sigmoid",
        routing_mode: str = "hard",
        shared_B: bool = False,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.num_agents = num_agents
        self.soft_routing = soft_routing
        self.cooperative_comm = cooperative_comm
        # Shared-B mode (v8+):
        #   Instead of per-agent B_v, B_a (double-tower), use a single shared B.
        #   Agents are differentiated by their A matrices (low-rank summaries).
        #   Blend happens in r-dim space (tiny), then a single B matmul emits
        #   the out_f tensor once.
        #   Memory: up_proj at seq=12288, bf16 → saves ~450 MiB / layer
        #   Params: up_proj r=256 → 6.7 M (shared) vs 11.5 M (double-tower), −42%
        #   Semantic: both agents write to the SAME output basis (residual stream)
        self.shared_B = shared_B
        if gate_type not in ("sigmoid", "tanh"):
            raise ValueError(f"gate_type must be 'sigmoid' or 'tanh', got {gate_type}")
        self.gate_type = gate_type
        # Routing mode:
        #   "hard":    original token-type routing via torch.where(mask, delta_v, delta_a).
        #              Requires _token_mask to be set before forward() during training.
        #   "merge":   no routing — delta = 0.5 * (delta_v + delta_a). Agents have
        #              NO structural specialization; any differentiation must come
        #              from training-time pressure (e.g. diversity loss).
        #   "learned": per-token soft routing from a shared linear router.
        #              w = sigmoid(router(x)); delta = w * delta_v + (1-w) * delta_a.
        #              router is a nn.Linear(D, 1) passed from the wrapper
        #              (shared across q/k/v/o modules within the same layer).
        #              No token_mask needed — the router reads from hidden states directly.
        if routing_mode not in ("hard", "merge", "learned"):
            raise ValueError(
                f"routing_mode must be 'hard', 'merge', or 'learned', got {routing_mode}")
        self.routing_mode = routing_mode
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

        # LoRA_A — applied to instruction/action tokens (action-optimized)
        self.lora_A_a = nn.Parameter(torch.zeros(r, in_f, device=device))

        # LoRA_T — applied to thought tokens (reasoning-optimized), 3-agent only
        if num_agents >= 3:
            self.lora_A_t = nn.Parameter(torch.zeros(r, in_f, device=device))

        # B matrices: shared (one B) or per-agent (B_v, B_a[, B_t])
        if shared_B:
            self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=device))
        else:
            self.lora_B_v = nn.Parameter(torch.zeros(out_f, r, device=device))
            self.lora_B_a = nn.Parameter(torch.zeros(out_f, r, device=device))
            if num_agents >= 3:
                self.lora_B_t = nn.Parameter(torch.zeros(out_f, r, device=device))

        self.lora_dropout = nn.Dropout(p=dropout)

        # Init: A = kaiming_uniform, B = zeros (starts as identity)
        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_a, a=math.sqrt(5))
        if num_agents >= 3:
            nn.init.kaiming_uniform_(self.lora_A_t, a=math.sqrt(5))
        # B stays zero -> delta starts at zero (whether shared or per-agent)

        # Learnable separation parameter for soft routing (2-agent only)
        if soft_routing and num_agents == 2:
            self.sep = nn.Parameter(torch.tensor(init_sep))

        # Per-layer cooperative communication (v6, 2-agent only)
        # W_av/W_va: kaiming init (NOT zero) — required for gate to receive
        # gradient at step 1. Safe warmup is still preserved by B=0 zeroing
        # the LoRA branch at the model output.
        if cooperative_comm and num_agents == 2:
            self.W_av = nn.Parameter(torch.zeros(r, r, device=device))   # A→V projection
            self.W_va = nn.Parameter(torch.zeros(r, r, device=device))   # V→A projection
            nn.init.kaiming_uniform_(self.W_av, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.W_va, a=math.sqrt(5))
            # gate_type="sigmoid": g = σ(logit),  init=-1.5 → g≈0.18,  max gradient 0.25
            # gate_type="tanh":    g = tanh(logit), init=0    → g=0,     gradient at 0 is 1.0
            #                      bounded [-1,1], allows learning negative (anti) coupling
            self.gate_av = nn.Parameter(torch.tensor(gate_init, device=device))
            self.gate_va = nn.Parameter(torch.tensor(gate_init, device=device))

        # Token mask set externally before forward
        self._token_mask: Optional[torch.Tensor] = None

        # Inference-only routing override: 'hard' (default) | 'v_only' | 't_only' | 'merge'
        self._inference_mode: str = "hard"

        # Shared per-layer router (v8, learned routing). Passed in from wrapper
        # via set_router(). A nn.Linear(in_f, 1) whose output is sigmoid-squashed
        # to produce w ∈ [0, 1] per token. Multiple CooperativeLoRALinear modules
        # within the same transformer layer share the SAME router instance so the
        # routing decision is tied to the semantic state of that layer, not to
        # each individual projection.
        #
        # IMPORTANT: stored via object.__setattr__ to bypass nn.Module's
        # submodule registration — the router is owned by the wrapper
        # (as wrapper.routers[layer_idx]). If it were registered here as a
        # submodule, its parameters would appear twice in self.parameters()
        # and get double-counted / double-broadcast in DDP.
        object.__setattr__(self, "router", None)
        # Cache of most-recent router weights, used by the wrapper to compute
        # balance loss without re-running the router.
        self._last_router_w: Optional[torch.Tensor] = None

    def set_token_mask(self, mask: Optional[torch.Tensor]):
        """Set the token routing mask.

        Args:
            mask: For num_agents=2: [B, seq_len] bool tensor (True = LoRA_V).
                  For num_agents=3: [B, seq_len] int8 tensor (0=A, 1=V, 2=T).
                  None clears the mask (base-only mode).
        """
        self._token_mask = mask

    def set_router(self, router: Optional[nn.Linear]):
        """Attach a shared per-layer learned router (v8).

        Args:
            router: nn.Linear(in_features, 1) module or None. Forward pass will
                    use `w = sigmoid(router(x))` as the per-token routing weight.

        Note: uses object.__setattr__ to bypass nn.Module submodule registration
        so the router's parameters are NOT double-counted. The wrapper owns the
        router in wrapper.routers[layer_idx]; we just keep a reference for use
        during forward.
        """
        object.__setattr__(self, "router", router)

    def _route(self, v_t: torch.Tensor, a_t: torch.Tensor,
               t_t: Optional[torch.Tensor], x: torch.Tensor,
               dtype: torch.dtype, token_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply routing logic to blend two/three input tensors.

        Routing operates identically on per-agent h (shared_B) or delta
        (double-tower). v_t/a_t/t_t are either [B,S,r] (shared_B path) or
        [B,S,out_f] (double-tower path).
        """
        if self.routing_mode == "merge":
            # v7: uniform 50/50 merge
            return torch.lerp(a_t, v_t, 0.5)

        if self.routing_mode == "learned":
            # v8: learned per-token router
            # Auto-fallback to 50/50 when router is missing or dim mismatches
            # (e.g. down_proj whose in_features = intermediate_size).
            if self.router is None or self.router.in_features != x.shape[-1]:
                self._last_router_w = None
                return torch.lerp(a_t, v_t, 0.5)
            router_weight = self.router.weight.to(dtype)
            router_bias = self.router.bias.to(dtype) if self.router.bias is not None else None
            w = torch.sigmoid(F.linear(x, router_weight, router_bias))  # [B, S, 1]
            self._last_router_w = w
            return torch.lerp(a_t, v_t, w)

        # Hard / soft routing — require token_mask
        mask = token_mask.unsqueeze(-1)  # [B, S, 1]

        if self.num_agents >= 3:
            # 3-agent: int8 mask (0=A, 1=V, 2=T)
            return torch.where(mask == 1, v_t,
                        torch.where(mask == 2, t_t, a_t))

        if self.soft_routing:
            # 2-agent soft routing: learnable weighted sum
            s = torch.sigmoid(self.sep)
            mask_f = token_mask.unsqueeze(-1).to(dtype)  # [B, S, 1]
            # Image tokens: w_v=s, w_a=1-s;  Text: w_v=1-s, w_a=s
            w_v = mask_f * s + (1.0 - mask_f) * (1.0 - s)
            w_a = mask_f * (1.0 - s) + (1.0 - mask_f) * s
            return w_v * v_t + w_a * a_t

        # 2-agent hard routing
        if not self.training and self._inference_mode != "hard":
            if self._inference_mode == "v_only":
                return v_t
            if self._inference_mode == "t_only":
                return a_t
            if self._inference_mode == "merge":
                return torch.lerp(a_t, v_t, 0.5)

        if mask.dtype == torch.bool:
            return torch.where(mask, v_t, a_t)
        # Float mask: cooperative reasoning with α-mixing (0=A, α=coop, 1=V)
        mask_f = mask.to(dtype)
        return torch.lerp(a_t, v_t, mask_f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, D] or [seq_len, D] (unbatched)

        Returns:
            output: same shape as x, with per-token LoRA delta applied.
        """
        base_out = self.base_linear(x)

        # Resolve mask (not required in merge or learned modes)
        token_mask = self._token_mask
        if token_mask is None and self.routing_mode not in ("merge", "learned"):
            if self.training:
                raise RuntimeError(
                    "CooperativeLoRALinear: token_mask is None during training. "
                    "Call set_token_mask() before model.forward()."
                )
            # Inference without mask -> base-only
            return base_out

        # Note: in learned mode, a missing router (or dim mismatch) triggers
        # merge-mode fallback inside _route() — legal for modules like down_proj
        # whose in_features differs from hidden_size.

        x_drop = self.lora_dropout(x)
        # Cast LoRA params to input dtype (base model may be bf16/fp16)
        dtype = x_drop.dtype

        # ── Low-rank projection (h-space, [B,S,r]) ──
        h_v = F.linear(x_drop, self.lora_A_v.to(dtype))
        h_a = F.linear(x_drop, self.lora_A_a.to(dtype))

        # ── Cooperative communication in h-space (v6) ──
        if self.cooperative_comm and hasattr(self, 'W_av'):
            if self.gate_type == "tanh":
                g_av = torch.tanh(self.gate_av)
                g_va = torch.tanh(self.gate_va)
            else:
                g_av = torch.sigmoid(self.gate_av)
                g_va = torch.sigmoid(self.gate_va)
            h_v = h_v + g_av * F.linear(h_a, self.W_av.to(dtype))  # V sees A
            h_a = h_a + g_va * F.linear(h_v, self.W_va.to(dtype))  # A sees V (uses updated h_v)

        h_t = None
        if self.num_agents >= 3:
            h_t = F.linear(x_drop, self.lora_A_t.to(dtype))

        if self.shared_B:
            # ── Shared-B: blend in r-space, single big matmul ──
            # Saves ~450 MiB/layer on up_proj at seq=12288 vs double-tower
            # (2× [B,S,out_f] collapse to 1×).
            h_blend = self._route(h_v, h_a, h_t, x, dtype, token_mask)
            delta = F.linear(h_blend, self.lora_B.to(dtype)) * self.scaling
        else:
            # ── Double-tower: compute per-agent deltas, blend at output ──
            delta_v = F.linear(h_v, self.lora_B_v.to(dtype)) * self.scaling
            delta_a = F.linear(h_a, self.lora_B_a.to(dtype)) * self.scaling
            delta_t = None
            if h_t is not None:
                delta_t = F.linear(h_t, self.lora_B_t.to(dtype)) * self.scaling
            delta = self._route(delta_v, delta_a, delta_t, x, dtype, token_mask)

        return base_out + delta

    def extra_repr(self) -> str:
        in_f = self.base_linear.in_features
        out_f = self.base_linear.out_features
        r = self.lora_A_v.shape[0]
        parts = [f"in={in_f}, out={out_f}, r={r}, scaling={self.scaling:.2f}, "
                 f"num_agents={self.num_agents}, routing={self.routing_mode}, "
                 f"shared_B={self.shared_B}"]
        if self.soft_routing and hasattr(self, "sep"):
            s = torch.sigmoid(self.sep).item()
            parts.append(f", soft_routing=True, s={s:.4f}")
        if self.cooperative_comm and hasattr(self, "gate_av"):
            if self.gate_type == "tanh":
                g_av = torch.tanh(self.gate_av).item()
                g_va = torch.tanh(self.gate_va).item()
            else:
                g_av = torch.sigmoid(self.gate_av).item()
                g_va = torch.sigmoid(self.gate_va).item()
            parts.append(f", comm=True, gate_type={self.gate_type}, g_av={g_av:.4f}, g_va={g_va:.4f}")
        if self.routing_mode == "learned":
            parts.append(
                f", router={'attached' if self.router is not None else 'MISSING'}")
        return "".join(parts)
