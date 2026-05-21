"""Standard LoRA VLM Wrapper — baseline for V12 cooperative comparison.

Wraps a Qwen2.5-VL model with standard LoRA (single A/B per module, no routing).
Matched parameter count to V12 cooperative LoRA via larger rank.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from v12_gui_360.standard_lora import StandardLoRALinear


class StandardLoRAWrapper(nn.Module):
    """Wrap a Qwen2.5-VL model with standard LoRA adapters."""

    def __init__(
        self,
        base_model: nn.Module,
        lora_r: int = 210,
        lora_alpha: int = 420,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.target_modules = target_modules

        self.config = getattr(base_model, "config", None)

        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False

        # Replace target modules
        self.lora_modules: List[StandardLoRALinear] = []
        self._replace_target_modules(lora_r, lora_alpha, lora_dropout)

        print(f"[StandardLoRAWrapper] {len(self.lora_modules)} replaced modules, "
              f"r={lora_r}, target_modules={target_modules}")

    def _get_transformer_layers(self):
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
                lora_linear = StandardLoRALinear(original, r, alpha, dropout)
                setattr(parent, module_name, lora_linear)
                self.lora_modules.append(lora_linear)

    # ── Gradient checkpointing ──

    def gradient_checkpointing_enable(self, **kwargs):
        self.base_model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        self.base_model.gradient_checkpointing_disable()

    # ── LoRA enable/disable (for ref model computation) ──

    def disable_lora(self):
        for m in self.lora_modules:
            m._enabled = False

    def enable_lora(self):
        for m in self.lora_modules:
            m._enabled = True

    # ── Generation ──

    @torch.no_grad()
    def generate(self, input_ids, **kwargs):
        return self.base_model.generate(input_ids=input_ids, **kwargs)

    def forward(self, **kwargs):
        return self.base_model(**kwargs)

    # ── Trainable parameter count ──

    def count_trainable_params(self) -> Dict[str, int]:
        lora_n = 0
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_" in name:
                lora_n += param.numel()
        return {"lora": lora_n, "total": lora_n}

    # ── Save/Load ──

    def save_lora(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        lora_state = {}
        for name, param in self.named_parameters():
            if param.requires_grad and "lora_" in name:
                lora_state[name] = param.data.cpu()

        torch.save(lora_state, os.path.join(save_dir, "lora_weights.pt"))

        config = {
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "type": "standard_lora_baseline",
        }
        with open(os.path.join(save_dir, "lora_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def load_lora(self, load_dir: str, device=None):
        fpath = os.path.join(load_dir, "lora_weights.pt")
        if os.path.exists(fpath):
            state = torch.load(fpath, map_location=device or "cpu",
                               weights_only=True)
            missing, unexpected = self.load_state_dict(state, strict=False)
            print(f"  Loaded lora_weights.pt: {len(state)} tensors "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
