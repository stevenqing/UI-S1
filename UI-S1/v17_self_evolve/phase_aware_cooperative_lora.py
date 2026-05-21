"""V17 Phase-Aware Cooperative LoRA + Wrapper.

Optional extension of V13 iterative cooperative LoRA that adds learnable
phase embeddings to the routing mechanism. During aux phase (phase_id=0)
and decision phase (phase_id=1), the routing computation receives a
different bias signal — enabling experts to naturally differentiate
their behavior across phases.

Key properties:
  - Phase embedding is zero-initialized → starts identical to V13 (safe init)
  - Phase signal is additive to the routing input, not multiplicative
  - RL learns optimal phase-dependent expert allocation automatically
  - Falls back to standard V13 when phase_id is not set (default=None)

Usage:
  wrapper = PhaseAwareCooperativeVLMWrapper(base_model, ...)
  wrapper.set_phase_id(0)  # aux phase
  wrapper.forward(...)
  wrapper.set_phase_id(1)  # decision phase
  wrapper.forward(...)
  wrapper.set_phase_id(None)  # normal (no phase signal)
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


class PhaseAwareLoRALinear(IterativeCooperativeLoRALinear):
    """Extension of IterativeCooperativeLoRALinear with phase embedding.

    Adds a learnable phase embedding to the routing computation:
        route_input = sigmoid(x @ w_route + phase_embed[phase_id])

    When phase_id is None, behaves identically to parent class.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 128,
        alpha: int = 256,
        dropout: float = 0.05,
    ):
        super().__init__(base_linear, r, alpha, dropout)

        # Phase embedding: [2, 1] — one scalar bias per phase
        # Zero init → sigmoid(x @ w + 0) = sigmoid(x @ w) → same as V13
        device = base_linear.weight.device
        self._phase_embed = None  # Set by wrapper
        self._phase_id = None    # None = no phase signal

    def set_phase_embed(self, embed: Optional[nn.Parameter]):
        """Attach phase embedding parameter (owned by wrapper)."""
        object.__setattr__(self, "_phase_embed", embed)

    def set_phase_id(self, phase_id: Optional[int]):
        """Set current phase: 0=aux, 1=decision, None=normal."""
        object.__setattr__(self, "_phase_id", phase_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with optional phase-dependent routing bias."""
        base_out = self.base_linear(x)

        if self._route_weight is None:
            self._last_routing_weights = None
            return base_out

        x_drop = self.lora_dropout(x)
        dtype = x_drop.dtype

        # Compute routing weights with phase bias
        w = self._route_weight.to(dtype)
        if w.shape[0] == x_drop.shape[-1]:
            logits = F.linear(x_drop, w.unsqueeze(0))  # [B, S, 1]

            # Add phase embedding bias
            if (self._phase_embed is not None
                    and self._phase_id is not None
                    and 0 <= self._phase_id < self._phase_embed.shape[0]):
                phase_bias = self._phase_embed[self._phase_id].to(dtype)
                logits = logits + phase_bias  # broadcast [1] to [B, S, 1]

            # Add noise for exploration during RL rollouts
            if self._routing_noise_std > 0 and self.training:
                noise = torch.randn_like(logits) * self._routing_noise_std
                logits = logits + noise

            r = torch.sigmoid(logits)
        else:
            r = torch.full(
                (*x_drop.shape[:-1], 1), 0.5,
                device=x_drop.device, dtype=dtype,
            )
        self._last_routing_weights = r.detach()

        # Dual low-rank projections
        h_1 = F.linear(x_drop, self.lora_A_1.to(dtype))
        h_2 = F.linear(x_drop, self.lora_A_2.to(dtype))

        # Iterative communication in r-space (same as parent)
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


class PhaseAwareCooperativeVLMWrapper(IterativeCooperativeVLMWrapper):
    """Extension of V13 wrapper with per-layer phase embeddings.

    Adds a learnable phase_embed[layer, 2] parameter that biases the
    routing computation depending on the generation phase (aux vs decision).
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
    ):
        # We need to override the module replacement to use PhaseAwareLoRALinear.
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

        # Per-layer phase embeddings: [2] scalar per layer (zero-init = safe)
        self.phase_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(2, device=device, dtype=dtype))
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

        # Replace target modules with PhaseAwareLoRALinear
        self.coop_modules: List[PhaseAwareLoRALinear] = []
        self._module_to_layer: List[int] = []
        self._replace_target_modules_phase_aware(lora_r, lora_alpha, lora_dropout)

        # Attach communication params and phase embeds
        self._attach_comm_params()
        self._attach_phase_embeds()

        print(f"[PhaseAwareCooperativeVLMWrapper] {num_layers} layers, "
              f"{len(self.coop_modules)} replaced modules, "
              f"target_modules={target_modules}, "
              f"num_comm_rounds={num_comm_rounds}, "
              f"phase_embeds=True")

    def _replace_target_modules_phase_aware(self, r: int, alpha: int, dropout: float):
        """Replace nn.Linear with PhaseAwareLoRALinear."""
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
                coop_linear = PhaseAwareLoRALinear(
                    original, r, alpha, dropout,
                )
                coop_linear.set_route_weight(self.route_weights[layer_idx])
                setattr(parent, module_name, coop_linear)
                self.coop_modules.append(coop_linear)
                self._module_to_layer.append(layer_idx)

    def _attach_phase_embeds(self):
        """Attach phase embedding params to all LoRA modules."""
        for module, layer_idx in zip(self.coop_modules, self._module_to_layer):
            module.set_phase_embed(self.phase_embeds[layer_idx])

    def set_phase_id(self, phase_id: Optional[int]):
        """Set generation phase for all modules.

        Args:
            phase_id: 0=aux, 1=decision, None=normal (no phase signal)
        """
        for module in self.coop_modules:
            module.set_phase_id(phase_id)

    def count_trainable_params(self) -> Dict[str, int]:
        """Count trainable parameters by category."""
        lora_n = 0
        route_n = 0
        comm_n = 0
        phase_n = 0
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "route_weights" in name:
                route_n += param.numel()
            elif "phase_embed" in name:
                phase_n += param.numel()
            elif "comm_" in name:
                comm_n += param.numel()
            elif "lora_" in name:
                lora_n += param.numel()
        return {
            "lora": lora_n,
            "route_weights": route_n,
            "comm": comm_n,
            "phase_embed": phase_n,
            "total": lora_n + route_n + comm_n + phase_n,
        }

    def save_cooperative(self, save_dir: str):
        """Save cooperative LoRA weights including phase embeds."""
        os.makedirs(save_dir, exist_ok=True)

        lora_state = {}
        route_state = {}
        comm_state = {}
        phase_state = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "route_weights" in name:
                route_state[name] = param.data.cpu()
            elif "phase_embed" in name:
                phase_state[name] = param.data.cpu()
            elif "comm_" in name:
                comm_state[name] = param.data.cpu()
            elif "lora_" in name:
                lora_state[name] = param.data.cpu()

        torch.save(lora_state, os.path.join(save_dir, "lora_weights.pt"))
        torch.save(route_state, os.path.join(save_dir, "route_weights.pt"))
        torch.save(comm_state, os.path.join(save_dir, "comm_weights.pt"))
        torch.save(phase_state, os.path.join(save_dir, "phase_weights.pt"))

        config = {
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "balance_weight": self.balance_weight,
            "num_comm_rounds": self.num_comm_rounds,
            "type": "phase_aware_cooperative_v17",
        }
        with open(os.path.join(save_dir, "cooperative_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def load_cooperative(self, load_dir: str, device=None):
        """Load cooperative LoRA weights including phase embeds."""
        for fname in ("lora_weights.pt", "route_weights.pt",
                      "comm_weights.pt", "phase_weights.pt"):
            fpath = os.path.join(load_dir, fname)
            if os.path.exists(fpath):
                state = torch.load(fpath, map_location=device or "cpu",
                                   weights_only=True)
                missing, unexpected = self.load_state_dict(state, strict=False)
                print(f"  Loaded {fname}: {len(state)} tensors "
                      f"(missing={len(missing)}, unexpected={len(unexpected)})")
