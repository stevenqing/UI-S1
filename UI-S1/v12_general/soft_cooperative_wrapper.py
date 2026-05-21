"""V12 Soft Cooperative VLM Wrapper.

Wraps a Qwen2.5-VL model with soft-routed cooperative LoRA:
  1. Replaces target nn.Linear modules with SoftCooperativeLoRALinear
  2. Per-layer routing via learned routing vectors (init=zeros -> sigmoid(0)=0.5)
  3. No token masks, no side-channel comm — routing is purely input-dependent
  4. Cooperation emerges through attention: q(slot1) @ k(slot2)

Key simplifications vs CooperativeVLMWrapper:
  - No token mask code, no 3-agent, no hard/merge modes
  - No bind loss, no coord routing, no thought state machine
  - generate() is trivial — just self.base_model.generate() (routing is automatic)
  - disable_lora()/enable_lora() for ref model computation
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from v12_general.soft_cooperative_lora import SoftCooperativeLoRALinear


class SoftCooperativeVLMWrapper(nn.Module):
    """Wrap a Qwen2.5-VL model with soft-routed cooperative LoRA."""

    def __init__(
        self,
        base_model: nn.Module,
        lora_r: int = 128,
        lora_alpha: int = 256,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        balance_weight: float = 0.01,
    ):
        super().__init__()
        self.base_model = base_model
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.balance_weight = balance_weight

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
        # One per layer, shared across q/k/v/o modules
        self.route_weights = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype))
            for _ in range(num_layers)
        ])

        # Replace target modules with SoftCooperativeLoRALinear
        self.coop_modules: List[SoftCooperativeLoRALinear] = []
        self._module_to_layer: List[int] = []
        self._replace_target_modules(lora_r, lora_alpha, lora_dropout)

        print(f"[SoftCooperativeVLMWrapper] {num_layers} layers, "
              f"{len(self.coop_modules)} replaced modules, "
              f"target_modules={target_modules}")

    # ── Module replacement ──────────────────────────────────────────

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
        """Replace nn.Linear in each transformer layer with SoftCooperativeLoRALinear."""
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
                coop_linear = SoftCooperativeLoRALinear(
                    original, r, alpha, dropout,
                )
                # Attach shared per-layer routing vector
                coop_linear.set_route_weight(self.route_weights[layer_idx])
                setattr(parent, module_name, coop_linear)
                self.coop_modules.append(coop_linear)
                self._module_to_layer.append(layer_idx)

    # ── Gradient checkpointing ──────────────────────────────────────

    def gradient_checkpointing_enable(self, **kwargs):
        self.base_model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.base_model.gradient_checkpointing_disable()

    # ── LoRA enable/disable (for ref model computation) ─────────────

    def disable_lora(self):
        """Disable LoRA by clearing route weights -> forward returns base_out only."""
        for module in self.coop_modules:
            module.set_route_weight(None)

    def enable_lora(self):
        """Re-enable LoRA by restoring route weights."""
        for module, layer_idx in zip(self.coop_modules, self._module_to_layer):
            module.set_route_weight(self.route_weights[layer_idx])

    # ── Routing noise for RL exploration ────────────────────────────

    def set_routing_noise(self, std: float):
        """Set routing noise std for all modules (structured exploration)."""
        for module in self.coop_modules:
            module._routing_noise_std = std

    # ── Generation ──────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, input_ids, **kwargs):
        """Generate — routing is automatic in forward, no hooks needed."""
        return self.base_model.generate(input_ids=input_ids, **kwargs)

    # ── Forward (delegates to base model) ───────────────────────────

    def forward(self, **kwargs):
        """Forward pass. Routing happens inside each SoftCooperativeLoRALinear."""
        return self.base_model(**kwargs)

    # ── Balance loss ────────────────────────────────────────────────

    def compute_balance_loss(self) -> Tuple[torch.Tensor, float]:
        """Routing entropy regularization — pushes mean routing toward 0.5.

        Returns (loss, mean_routing_weight).
        """
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

    # ── Trainable parameter count ───────────────────────────────────

    def count_trainable_params(self) -> Dict[str, int]:
        """Count trainable parameters by category."""
        lora_n = 0
        route_n = 0
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "route_weights" in name:
                route_n += param.numel()
            elif "lora_" in name:
                lora_n += param.numel()
        return {
            "lora": lora_n,
            "route_weights": route_n,
            "total": lora_n + route_n,
        }

    # ── Save/Load ───────────────────────────────────────────────────

    def save_cooperative(self, save_dir: str):
        """Save cooperative LoRA weights and config."""
        os.makedirs(save_dir, exist_ok=True)

        lora_state = {}
        route_state = {}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "route_weights" in name:
                route_state[name] = param.data.cpu()
            elif "lora_" in name:
                lora_state[name] = param.data.cpu()

        torch.save(lora_state, os.path.join(save_dir, "lora_weights.pt"))
        torch.save(route_state, os.path.join(save_dir, "route_weights.pt"))

        config = {
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "balance_weight": self.balance_weight,
            "type": "soft_cooperative_v12",
        }
        with open(os.path.join(save_dir, "cooperative_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def load_cooperative(self, load_dir: str, device=None):
        """Load cooperative LoRA weights."""
        for fname in ("lora_weights.pt", "route_weights.pt"):
            fpath = os.path.join(load_dir, fname)
            if os.path.exists(fpath):
                state = torch.load(fpath, map_location=device or "cpu",
                                   weights_only=True)
                missing, unexpected = self.load_state_dict(state, strict=False)
                print(f"  Loaded {fname}: {len(state)} tensors "
                      f"(missing={len(missing)}, unexpected={len(unexpected)})")
