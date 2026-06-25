"""V13 Iterative Cooperative VLM Wrapper.

Extension of V12's SoftCooperativeVLMWrapper with per-layer iterative
communication parameters. Communication params (W_12, W_21, gate_12, gate_21)
are owned by the wrapper and shared across q/k/v/o modules within the same
transformer layer — the deliberation protocol is a property of the layer,
not individual projections.

Parameter overhead (T=2, 28 layers):
  Per layer per round: W_12[r,r] + W_21[r,r] + gate_12[r] + gate_21[r] = 33,024
  Total: 33,024 * 2 * 28 = 1,849,344 (~1.85M additional params)
"""

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from v13_gui_360.iterative_cooperative_lora import IterativeCooperativeLoRALinear


class IterativeCooperativeVLMWrapper(nn.Module):
    """Wrap a Qwen2.5-VL model with iterative cooperative LoRA."""

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
        super().__init__()
        self.base_model = base_model
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.balance_weight = balance_weight
        self.num_comm_rounds = num_comm_rounds

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.target_modules = target_modules

        # Expose base model's config for HF Trainer compatibility
        self.config = getattr(base_model, "config", None)

        # Freeze ALL base model parameters
        for param in base_model.parameters():
            param.requires_grad = False

        # Get transformer layers info
        layers = self._get_transformer_layers()
        num_layers = len(layers)
        hidden_size = layers[0].self_attn.q_proj.in_features
        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype

        # Per-layer routing vectors (dim=hidden_size, init=zeros -> sigmoid(0)=0.5)
        self.route_weights = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype))
            for _ in range(num_layers)
        ])

        # Per-layer communication parameters (shared across q/k/v/o within layer)
        # Indexed as: layer_idx * T + t
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

        # Init: W_12, W_21 = kaiming_uniform; gate vectors = zeros
        # sigmoid(0) = 0.5 -> moderate initial communication; safe because B=0
        for i in range(num_layers * T):
            nn.init.kaiming_uniform_(self.comm_W_12[i], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.comm_W_21[i], a=math.sqrt(5))
            # gate vectors stay zero

        # Replace target modules with IterativeCooperativeLoRALinear
        self.coop_modules: List[IterativeCooperativeLoRALinear] = []
        self._module_to_layer: List[int] = []
        self._replace_target_modules(lora_r, lora_alpha, lora_dropout)

        # Attach communication params to each LoRA module
        self._attach_comm_params()

        print(f"[IterativeCooperativeVLMWrapper] {num_layers} layers, "
              f"{len(self.coop_modules)} replaced modules, "
              f"target_modules={target_modules}, "
              f"num_comm_rounds={num_comm_rounds}")

    # -- Module replacement ------------------------------------------------

    def _get_transformer_layers(self):
        """Locate the transformer layer ModuleList."""
        vlm = self.base_model.model
        if hasattr(vlm, "language_model"):
            return vlm.language_model.layers
        elif hasattr(vlm, "layers"):
            return vlm.layers
        else:
            raise AttributeError(
                f"Cannot find transformer layers in {type(vlm).__name__}. "
                f"Children: {[n for n, _ in vlm.named_children()]}"
            )

    def _replace_target_modules(self, r: int, alpha: int, dropout: float):
        """Replace nn.Linear in each transformer layer with IterativeCooperativeLoRALinear."""
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
                coop_linear = IterativeCooperativeLoRALinear(
                    original, r, alpha, dropout,
                )
                # Attach shared per-layer routing vector
                coop_linear.set_route_weight(self.route_weights[layer_idx])
                setattr(parent, module_name, coop_linear)
                self.coop_modules.append(coop_linear)
                self._module_to_layer.append(layer_idx)

    def _build_comm_dict(self, layer_idx: int) -> dict:
        """Build communication param dict for a specific layer."""
        T = self.num_comm_rounds
        base = layer_idx * T
        return {
            'T': T,
            'W_12': [self.comm_W_12[base + t] for t in range(T)],
            'W_21': [self.comm_W_21[base + t] for t in range(T)],
            'gate_12': [self.comm_gate_12[base + t] for t in range(T)],
            'gate_21': [self.comm_gate_21[base + t] for t in range(T)],
        }

    def _attach_comm_params(self):
        """Attach communication params to all LoRA modules."""
        for module, layer_idx in zip(self.coop_modules, self._module_to_layer):
            module.set_comm_params(self._build_comm_dict(layer_idx))

    # -- Gradient checkpointing --------------------------------------------

    def gradient_checkpointing_enable(self, **kwargs):
        self.base_model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.base_model.gradient_checkpointing_disable()

    # -- LoRA enable/disable (for ref model computation) -------------------

    def disable_lora(self):
        """Disable LoRA by clearing route weights and comm params."""
        for module in self.coop_modules:
            module.set_route_weight(None)
            module.set_comm_params(None)

    def enable_lora(self):
        """Re-enable LoRA by restoring route weights and comm params."""
        for module, layer_idx in zip(self.coop_modules, self._module_to_layer):
            module.set_route_weight(self.route_weights[layer_idx])
            module.set_comm_params(self._build_comm_dict(layer_idx))

    def enable_lora_layers(self, active_layers: list):
        """Enable LoRA only for specified layers, disable for all others.

        Args:
            active_layers: list of layer indices (0-27) to keep active.
                           All other layers revert to base model behavior.
        """
        active_set = set(active_layers)
        for module, layer_idx in zip(self.coop_modules, self._module_to_layer):
            if layer_idx in active_set:
                module.set_route_weight(self.route_weights[layer_idx])
                module.set_comm_params(self._build_comm_dict(layer_idx))
            else:
                module.set_route_weight(None)
                module.set_comm_params(None)

    # -- Gate recording for CoPDA (V14) ------------------------------------

    def enable_gate_recording(self):
        """Enable gate recording on all cooperative modules."""
        for m in self.coop_modules:
            m._record_gates = True

    def disable_gate_recording(self):
        """Disable gate recording and clean up cached gate means."""
        for m in self.coop_modules:
            m._record_gates = False
            if hasattr(m, '_last_gate_mean'):
                del m._last_gate_mean

    def get_phase_scores(self) -> torch.Tensor:
        """Per-token phase scores from cross-layer gate variance. Returns [B, S]."""
        layer_gates = {}
        for m, l in zip(self.coop_modules, self._module_to_layer):
            if hasattr(m, '_last_gate_mean'):
                layer_gates.setdefault(l, []).append(m._last_gate_mean)

        per_layer = []
        for l in sorted(layer_gates.keys()):
            per_layer.append(torch.stack(layer_gates[l]).mean(0))  # [B, S, 1]

        gate_stack = torch.stack(per_layer, dim=0)    # [L, B, S, 1]
        gate_std = gate_stack.std(dim=0).squeeze(-1)  # [B, S]

        mu = gate_std.mean(dim=-1, keepdim=True)
        sigma = gate_std.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return torch.sigmoid((gate_std - mu) / sigma)  # [B, S]

    # -- Inference mode (expert-only ablation) ------------------------------

    def set_inference_mode(self, mode: Optional[str]):
        """Override routing for all modules.

        Args:
            mode: None (normal routing), "expert_1_only", or "expert_2_only".
        """
        assert mode in (None, "expert_1_only", "expert_2_only"), \
            f"Invalid inference mode: {mode}"
        for module in self.coop_modules:
            module._inference_mode = mode

    # -- Disable communication (ablation) ----------------------------------

    def set_disable_comm(self, disable: bool):
        """Disable iterative communication for all modules (ablation)."""
        for module in self.coop_modules:
            module._disable_comm = disable

    # -- Routing noise for RL exploration ----------------------------------

    def set_routing_noise(self, std: float):
        """Set routing noise std for all modules (structured exploration)."""
        for module in self.coop_modules:
            module._routing_noise_std = std

    # -- Generation --------------------------------------------------------

    @torch.no_grad()
    def generate(self, input_ids, **kwargs):
        """Generate — routing and communication are automatic in forward."""
        return self.base_model.generate(input_ids=input_ids, **kwargs)

    # -- Forward (delegates to base model) ---------------------------------

    def forward(self, **kwargs):
        """Forward pass. Routing and communication happen inside each LoRA module."""
        return self.base_model(**kwargs)

    # -- Balance loss ------------------------------------------------------

    def compute_balance_loss(self) -> Tuple[torch.Tensor, float]:
        """Routing entropy regularization — pushes mean routing toward 0.5."""
        if self.balance_weight <= 0:
            return torch.tensor(0.0, device=next(self.parameters()).device), 0.5

        balance_terms = []
        w_sum = 0.0
        w_count = 0
        eps = 1e-6
        for m in self.coop_modules:
            if m._last_routing_weights is None:
                continue
            w = m._last_routing_weights.mean()
            neg_entropy = w * torch.log(w + eps) + (1 - w) * torch.log(1 - w + eps)
            balance_terms.append(neg_entropy)
            w_sum += w.item()
            w_count += 1

        if not balance_terms:
            return torch.tensor(0.0, device=next(self.parameters()).device), 0.5

        loss = torch.stack(balance_terms).mean()
        mean_w = w_sum / w_count if w_count > 0 else 0.5
        return loss, mean_w

    def get_route_values_for_loss(self) -> Optional[torch.Tensor]:
        """Return differentiable mean token routing values for route loss.

        Shape is [batch, seq_len]. Modules whose routing fallback is a constant
        0.5 are ignored because they do not carry gradient to route weights.
        """
        route_tensors = []
        for module in self.coop_modules:
            route = getattr(module, "_last_routing_weights_for_loss", None)
            if route is not None and route.requires_grad:
                route_tensors.append(route.squeeze(-1))
        if not route_tensors:
            return None
        return torch.stack(route_tensors, dim=0).mean(dim=0)

    # -- Trainable parameter count -----------------------------------------

    def count_trainable_params(self) -> Dict[str, int]:
        """Count trainable parameters by category."""
        lora_n = 0
        route_n = 0
        comm_n = 0
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "route_weights" in name:
                route_n += param.numel()
            elif "comm_" in name:
                comm_n += param.numel()
            elif "lora_" in name:
                lora_n += param.numel()
        return {
            "lora": lora_n,
            "route_weights": route_n,
            "comm": comm_n,
            "total": lora_n + route_n + comm_n,
        }

    # -- Save/Load ---------------------------------------------------------

    def save_cooperative(self, save_dir: str):
        """Save cooperative LoRA weights, route weights, comm weights, and config."""
        os.makedirs(save_dir, exist_ok=True)

        lora_state = {}
        route_state = {}
        comm_state = {}
        for name, param in self.named_parameters():
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
            "type": "iterative_cooperative_v13",
        }
        with open(os.path.join(save_dir, "cooperative_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def load_cooperative(self, load_dir: str, device=None):
        """Load cooperative LoRA weights, route weights, and comm weights."""
        for fname in ("lora_weights.pt", "route_weights.pt", "comm_weights.pt"):
            fpath = os.path.join(load_dir, fname)
            if os.path.exists(fpath):
                state = torch.load(fpath, map_location=device or "cpu",
                                   weights_only=True)
                missing, unexpected = self.load_state_dict(state, strict=False)
                print(f"  Loaded {fname}: {len(state)} tensors "
                      f"(missing={len(missing)}, unexpected={len(unexpected)})")
