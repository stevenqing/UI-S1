"""V17.1 Dual-Aux Cooperative LoRA — Phase-Conditional Hard Routing.

Extension of V13 iterative cooperative LoRA with per-token phase-conditional
hard routing. During aux_a phase (phase=0), Expert A gets r=0.9; during
aux_b phase (phase=1), Expert B gets r=0.1; during decision phase (phase=2),
routing is learned (sigmoid).

Key properties:
  - Phase mask is a [B, S] int tensor: 0=aux_a, 1=aux_b, 2=decision
  - Hard routing forces structural asymmetric gradients per expert
  - KV cache enables token-space expert communication (aux_b sees aux_a)
  - Diversity loss on r-space expert outputs encourages heterogeneity
  - Falls back to standard V13 when phase_mask is None

Usage:
  wrapper = DualAuxCooperativeVLMWrapper(base_model, ...)
  wrapper.set_phase_mask(mask)   # [B, S] int tensor for training
  wrapper.forward(...)
  wrapper.clear_phase_mask()

  # Generation: set global phase for next token
  wrapper.set_phase_mask_global(0)  # aux_a phase
  wrapper.forward(...)
"""

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from v13_gui_360.iterative_cooperative_lora import IterativeCooperativeLoRALinear
from v13_gui_360.iterative_cooperative_wrapper import IterativeCooperativeVLMWrapper


class DualAuxLoRALinear(IterativeCooperativeLoRALinear):
    """LoRA layer with per-token phase-conditional hard routing.

    Phase mask values:
        0 = aux_a → Expert A dominant (r=0.9)
        1 = aux_b → Expert B dominant (r=0.1)
        2 = decision → learned routing (sigmoid)
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 128,
        alpha: int = 256,
        dropout: float = 0.05,
    ):
        super().__init__(base_linear, r, alpha, dropout)

        # Phase mask: [B, S] int tensor, set by wrapper
        object.__setattr__(self, "_phase_mask", None)

        # Global phase id for autoregressive generation (single int or [B] tensor)
        object.__setattr__(self, "_global_phase_id", None)

        # Hard routing values (set by wrapper)
        self._aux_a_route: float = 0.9
        self._aux_b_route: float = 0.1

        # Expert h capture for diversity loss
        self._capture_expert_h: bool = False
        self._last_h1: Optional[torch.Tensor] = None
        self._last_h2: Optional[torch.Tensor] = None

    def set_phase_mask(self, mask: Optional[torch.Tensor]):
        """Set per-token [B, S] phase mask."""
        object.__setattr__(self, "_phase_mask", mask)

    def set_global_phase_id(self, phase_id: Optional[int]):
        """Set global phase id for generation (applies to all tokens)."""
        object.__setattr__(self, "_global_phase_id", phase_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with phase-conditional hard routing."""
        base_out = self.base_linear(x)

        if self._route_weight is None:
            self._last_routing_weights = None
            return base_out

        x_drop = self.lora_dropout(x)
        dtype = x_drop.dtype

        # Compute soft routing as usual
        w = self._route_weight.to(dtype)
        if w.shape[0] == x_drop.shape[-1]:
            logits = F.linear(x_drop, w.unsqueeze(0))  # [B, S, 1]

            # Add noise for exploration during RL rollouts
            if self._routing_noise_std > 0 and self.training:
                noise = torch.randn_like(logits) * self._routing_noise_std
                logits = logits + noise

            r_soft = torch.sigmoid(logits)  # [B, S, 1]
        else:
            r_soft = torch.full(
                (*x_drop.shape[:-1], 1), 0.5,
                device=x_drop.device, dtype=dtype,
            )

        # Apply phase-conditional hard routing
        if self._phase_mask is not None:
            pm = self._phase_mask  # [B, S]
            # Handle shape: pm may be shorter/longer than x_drop seq dim
            seq_len = x_drop.shape[-2] if x_drop.dim() >= 2 else 1
            if pm.dim() == 2 and pm.shape[1] != seq_len:
                # Pad or truncate mask to match sequence length
                if pm.shape[1] < seq_len:
                    # Pad with 2 (decision phase) — for prompt tokens before response
                    pad = torch.full(
                        (pm.shape[0], seq_len - pm.shape[1]),
                        2, device=pm.device, dtype=pm.dtype,
                    )
                    pm = torch.cat([pad, pm], dim=1)
                else:
                    pm = pm[:, -seq_len:]

            pm_expanded = pm.unsqueeze(-1)  # [B, S, 1]
            r = torch.where(
                pm_expanded == 0,
                torch.tensor(self._aux_a_route, device=x_drop.device, dtype=dtype),
                torch.where(
                    pm_expanded == 1,
                    torch.tensor(self._aux_b_route, device=x_drop.device, dtype=dtype),
                    r_soft,
                ),
            )
        elif self._global_phase_id is not None:
            # Generation mode: per-sample or global phase
            pid = self._global_phase_id
            if isinstance(pid, torch.Tensor):
                # Per-sample phase: pid is [B] int tensor
                pid_expanded = pid.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
                r = torch.where(
                    pid_expanded == 0,
                    torch.tensor(self._aux_a_route, device=x_drop.device, dtype=dtype),
                    torch.where(
                        pid_expanded == 1,
                        torch.tensor(self._aux_b_route, device=x_drop.device, dtype=dtype),
                        r_soft,
                    ),
                )
            elif pid == 0:
                r = torch.full_like(r_soft, self._aux_a_route)
            elif pid == 1:
                r = torch.full_like(r_soft, self._aux_b_route)
            else:
                r = r_soft
        else:
            r = r_soft

        self._last_routing_weights = r.detach()

        # Dual low-rank projections
        h_1 = F.linear(x_drop, self.lora_A_1.to(dtype))  # [B, S, r]
        h_2 = F.linear(x_drop, self.lora_A_2.to(dtype))  # [B, S, r]

        # Capture for diversity loss — keep graph alive for gradient flow
        # (detach would make diversity loss gradient-free / logging-only)
        if self._capture_expert_h:
            self._last_h1 = h_1
            self._last_h2 = h_2

        # Iterative communication in r-space (same as V13)
        if self._comm_params is not None and not self._disable_comm:
            T = self._comm_params['T']
            gate_accum = 0
            for t in range(T):
                g_12 = torch.sigmoid(
                    F.linear(h_1, self._comm_params['gate_12'][t].to(dtype).unsqueeze(0))
                )
                h_1 = h_1 + g_12 * F.linear(h_2, self._comm_params['W_12'][t].to(dtype))

                g_21 = torch.sigmoid(
                    F.linear(h_2, self._comm_params['gate_21'][t].to(dtype).unsqueeze(0))
                )
                h_2 = h_2 + g_21 * F.linear(h_1, self._comm_params['W_21'][t].to(dtype))

                if self._record_gates:
                    gate_accum = gate_accum + g_12.detach() + g_21.detach()

            if self._record_gates:
                self._last_gate_mean = gate_accum / (2 * T)

        # Blend in r-space
        if self._inference_mode == "expert_1_only":
            h_blend = h_1
        elif self._inference_mode == "expert_2_only":
            h_blend = h_2
        else:
            h_blend = r * h_1 + (1 - r) * h_2

        delta = F.linear(h_blend, self.lora_B.to(dtype)) * self.scaling
        return base_out + delta


class DualAuxCooperativeVLMWrapper(IterativeCooperativeVLMWrapper):
    """Wrapper with per-token phase-conditional hard routing for dual-aux.

    Replaces target modules with DualAuxLoRALinear instead of
    IterativeCooperativeLoRALinear. Adds phase mask management and
    diversity loss computation.
    """

    def __init__(
        self,
        base_model: nn.Module,
        lora_r: int = 128,
        lora_alpha: int = 256,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        balance_weight: float = 0.01,
        num_comm_rounds: int = 2,
        aux_a_hard_route: float = 0.9,
        aux_b_hard_route: float = 0.1,
    ):
        # Store hard route values before parent init (which calls _replace)
        self._aux_a_hard_route = aux_a_hard_route
        self._aux_b_hard_route = aux_b_hard_route

        # We need to override the module replacement to use DualAuxLoRALinear.
        # Call grandparent init, then do our own setup.
        nn.Module.__init__(self)
        self.base_model = base_model
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.balance_weight = balance_weight
        self.num_comm_rounds = num_comm_rounds

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.target_modules = target_modules

        self.config = getattr(base_model, "config", None)

        # Freeze ALL base model parameters
        for param in base_model.parameters():
            param.requires_grad = False

        layers = self._get_transformer_layers()
        num_layers = len(layers)
        hidden_size = layers[0].self_attn.q_proj.in_features
        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype

        # Per-layer routing vectors
        self.route_weights = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype))
            for _ in range(num_layers)
        ])

        # Per-layer communication parameters
        T = num_comm_rounds
        self.comm_W_12 = nn.ParameterList([
            nn.Parameter(torch.zeros(lora_r, lora_r, device=device, dtype=dtype))
            for _ in range(num_layers * T)
        ])
        self.comm_W_21 = nn.ParameterList([
            nn.Parameter(torch.zeros(lora_r, lora_r, device=device, dtype=dtype))
            for _ in range(num_layers * T)
        ])
        self.comm_gate_12 = nn.ParameterList([
            nn.Parameter(torch.zeros(lora_r, device=device, dtype=dtype))
            for _ in range(num_layers * T)
        ])
        self.comm_gate_21 = nn.ParameterList([
            nn.Parameter(torch.zeros(lora_r, device=device, dtype=dtype))
            for _ in range(num_layers * T)
        ])

        for i in range(num_layers * T):
            nn.init.kaiming_uniform_(self.comm_W_12[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.comm_W_21[i], a=math.sqrt(5))

        # Replace target modules with DualAuxLoRALinear
        self.coop_modules: List[DualAuxLoRALinear] = []
        self._module_to_layer: List[int] = []
        self._replace_target_modules_dual_aux(lora_r, lora_alpha, lora_dropout)

        # Attach communication params
        self._attach_comm_params()

        print(f"[DualAuxCooperativeVLMWrapper] {num_layers} layers, "
              f"{len(self.coop_modules)} replaced modules, "
              f"target_modules={target_modules}, "
              f"num_comm_rounds={num_comm_rounds}, "
              f"aux_a_route={aux_a_hard_route}, aux_b_route={aux_b_hard_route}")

    def _replace_target_modules_dual_aux(self, r: int, alpha: int, dropout: float):
        """Replace nn.Linear with DualAuxLoRALinear."""
        layers = self._get_transformer_layers()

        for layer_idx in range(len(layers)):
            layer = layers[layer_idx]
            for module_name in self.target_modules:
                if module_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    parent = layer.self_attn
                elif module_name in ("gate_proj", "up_proj", "down_proj"):
                    parent = layer.mlp
                else:
                    raise ValueError(f"Unknown target module: {module_name}")

                original = getattr(parent, module_name)
                coop_linear = DualAuxLoRALinear(
                    original, r, alpha, dropout,
                )
                coop_linear.set_route_weight(self.route_weights[layer_idx])
                coop_linear._aux_a_route = self._aux_a_hard_route
                coop_linear._aux_b_route = self._aux_b_hard_route
                setattr(parent, module_name, coop_linear)
                self.coop_modules.append(coop_linear)
                self._module_to_layer.append(layer_idx)

    # -- Phase mask management ------------------------------------------------

    def set_phase_mask(self, mask: Optional[torch.Tensor]):
        """Set per-token [B, S] phase mask on all LoRA modules."""
        for module in self.coop_modules:
            module.set_phase_mask(mask)

    def clear_phase_mask(self):
        """Clear phase mask (fall back to soft routing)."""
        self.set_phase_mask(None)
        for module in self.coop_modules:
            module.set_global_phase_id(None)

    def set_phase_mask_global(self, phase_id):
        """Set global phase id for autoregressive generation.

        Args:
            phase_id: int (0=aux_a, 1=aux_b, 2=decision),
                      torch.Tensor [B] for per-sample phase,
                      or None for soft routing
        """
        for module in self.coop_modules:
            module.set_phase_mask(None)
            module.set_global_phase_id(phase_id)

    # -- Diversity loss -------------------------------------------------------

    def set_capture_expert_h(self, enabled: bool):
        """Enable/disable expert h capture for diversity loss."""
        for module in self.coop_modules:
            module._capture_expert_h = enabled

    def compute_diversity_loss(self, phase_mask: torch.Tensor) -> torch.Tensor:
        """Compute r-space diversity loss between experts at decision tokens.

        Measures cosine similarity between Expert A and Expert B outputs at
        decision phase tokens (phase=2), where both experts see the SAME input.
        This captures actual expert function difference, not input distribution
        difference (which would happen if we compared at aux_a vs aux_b tokens).

        Args:
            phase_mask: [B, S] int tensor (0=aux_a, 1=aux_b, 2=decision)

        Returns:
            Scalar similarity loss (push toward 0 for diversity).
        """
        total_sim = 0.0
        count = 0
        device = next(self.parameters()).device

        for module in self.coop_modules:
            if module._last_h1 is None or module._last_h2 is None:
                continue

            h1, h2 = module._last_h1, module._last_h2  # [B, S, r]

            # Match sequence lengths
            h_seq = h1.shape[1]
            pm = phase_mask
            if pm.shape[1] != h_seq:
                if pm.shape[1] < h_seq:
                    pad = torch.full(
                        (pm.shape[0], h_seq - pm.shape[1]),
                        2, device=pm.device, dtype=pm.dtype,
                    )
                    pm = torch.cat([pad, pm], dim=1)
                else:
                    pm = pm[:, -h_seq:]

            # Decision phase tokens — both experts see same input here
            dec_mask = (pm == 2).unsqueeze(-1).float()  # [B, S, 1]

            # Skip if no decision tokens
            if dec_mask.sum() < 1:
                continue

            # Mean-pool both expert outputs over decision phase
            h1_dec = (h1 * dec_mask).sum(dim=1) / dec_mask.sum(dim=1).clamp(min=1)  # [B, r]
            h2_dec = (h2 * dec_mask).sum(dim=1) / dec_mask.sum(dim=1).clamp(min=1)  # [B, r]

            sim = F.cosine_similarity(h1_dec, h2_dec, dim=-1).mean()
            total_sim += sim
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=device)

        return total_sim / count

    # -- Save/Load ------------------------------------------------------------

    def save_cooperative(self, save_dir: str):
        """Save cooperative LoRA weights and config."""
        os.makedirs(save_dir, exist_ok=True)

        lora_state = {}
        route_state = {}
        comm_state = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "route_weights" in name:
                route_state[name] = param.data.cpu()
            elif "comm_" in name:
                comm_state[name] = param.data.cpu()
            elif "lora_" in name:
                lora_state[name] = param.data.cpu()

        torch.save(lora_state, os.path.join(save_dir, "lora_weights.pt"))
        torch.save(route_state, os.path.join(save_dir, "route_weights.pt"))
        torch.save(comm_state, os.path.join(save_dir, "comm_weights.pt"))

        config = {
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "balance_weight": self.balance_weight,
            "num_comm_rounds": self.num_comm_rounds,
            "aux_a_hard_route": self._aux_a_hard_route,
            "aux_b_hard_route": self._aux_b_hard_route,
            "type": "dual_aux_cooperative_v17",
        }
        with open(os.path.join(save_dir, "cooperative_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def load_cooperative(self, load_dir: str, device=None):
        """Load cooperative LoRA weights."""
        for fname in ("lora_weights.pt", "route_weights.pt", "comm_weights.pt"):
            fpath = os.path.join(load_dir, fname)
            if os.path.exists(fpath):
                state = torch.load(fpath, map_location=device or "cpu",
                                   weights_only=True)
                missing, unexpected = self.load_state_dict(state, strict=False)
                print(f"  Loaded {fname}: {len(state)} tensors "
                      f"(missing={len(missing)}, unexpected={len(unexpected)})")
