# Copyright 2024 UI-S1 Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Expert LoRA Implementation for MoE GUI Agent.

This module implements the Expert LoRA collection that provides multiple
specialized LoRA adapters for different instruction types.

Design Decisions:
- Independent LoRAs: Each expert has completely independent LoRA parameters
  for maximum specialization potential
- Modular Design: Can apply single expert or weighted combination
- PEFT Compatible: Can export to PEFT format for vLLM inference

Architecture:
    ExpertLoRACollection
        ├── Expert 0 (SingleExpertLoRA)
        │   ├── Layer 0: {q_proj: LoRALayer, v_proj: LoRALayer}
        │   ├── Layer 1: {q_proj: LoRALayer, v_proj: LoRALayer}
        │   └── ...
        ├── Expert 1 (SingleExpertLoRA)
        │   └── ...
        ├── Expert 2 (SingleExpertLoRA)
        │   └── ...
        └── Expert 3 (SingleExpertLoRA)
            └── ...
"""

from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ExpertLoRAConfig:
    """Configuration for Expert LoRA collection."""

    num_experts: int = 4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ['q_proj', 'v_proj']

    @property
    def scaling(self) -> float:
        return self.lora_alpha / self.lora_r

    def to_dict(self) -> dict:
        return {
            'num_experts': self.num_experts,
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'target_modules': self.target_modules,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExpertLoRAConfig":
        return cls(**d)


class LoRALayer(nn.Module):
    """
    Single LoRA adapter layer.

    Implements the LoRA decomposition:
        y = Wx + (alpha/r) * B @ A @ x

    Where:
    - W: Original weight (frozen, not stored here)
    - A: Down projection [r, in_features]
    - B: Up projection [out_features, r]
    - alpha/r: Scaling factor

    Args:
        in_features: Input dimension
        out_features: Output dimension
        r: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout probability

    Example:
        >>> lora = LoRALayer(3584, 3584, r=16, alpha=32)
        >>> x = torch.randn(8, 512, 3584)
        >>> delta = lora(x)  # [8, 512, 3584]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # LoRA parameters: A is down-projection, B is up-projection
        # A: [r, in_features] - projects input to low-rank space
        # B: [out_features, r] - projects from low-rank space to output
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Dropout (applied to input)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """
        Initialize LoRA weights.

        - A: Kaiming uniform initialization
        - B: Zero initialization (LoRA starts as identity)
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute LoRA delta.

        Args:
            x: Input tensor [*, in_features]

        Returns:
            delta: LoRA output [*, out_features]
        """
        # Apply dropout
        x = self.lora_dropout(x)

        # Compute: delta = x @ A^T @ B^T * scaling
        # Equivalent to: (B @ A @ x^T)^T * scaling
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B)

        return delta * self.scaling

    def merge_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Merge LoRA into original weight.

        Args:
            weight: Original weight [out_features, in_features]

        Returns:
            merged: Weight with LoRA merged [out_features, in_features]
        """
        # delta_W = B @ A * scaling
        delta_W = (self.lora_B @ self.lora_A) * self.scaling
        return weight + delta_W

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, r={self.r}, alpha={self.alpha}"


class SingleExpertLoRA(nn.Module):
    """
    Single Expert's LoRA adapter collection.

    Contains LoRA adapters for all target modules across all layers.

    Args:
        num_layers: Number of transformer layers
        hidden_size: Model hidden dimension
        target_modules: List of module names to adapt (e.g., ['q_proj', 'v_proj'])
        r: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout probability
        module_dims: Optional dict of {module_name: (in_dim, out_dim)} for non-standard dims

    Example:
        >>> expert = SingleExpertLoRA(
        ...     num_layers=28,
        ...     hidden_size=3584,
        ...     target_modules=['q_proj', 'v_proj'],
        ...     r=16,
        ... )
        >>> delta = expert.get_lora_delta(layer_idx=0, module_name='q_proj', x=x)
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        target_modules: List[str],
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        module_dims: Optional[Dict[str, Tuple[int, int]]] = None,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.target_modules = target_modules
        self.r = r
        self.alpha = alpha
        self.dropout = dropout

        # Default: all modules have same dimension
        if module_dims is None:
            module_dims = {m: (hidden_size, hidden_size) for m in target_modules}
        self.module_dims = module_dims

        # Create LoRA layers for each (layer, module) combination
        self.lora_layers = nn.ModuleDict()

        for layer_idx in range(num_layers):
            for module_name in target_modules:
                key = self._make_key(layer_idx, module_name)
                in_dim, out_dim = module_dims.get(module_name, (hidden_size, hidden_size))

                self.lora_layers[key] = LoRALayer(
                    in_features=in_dim,
                    out_features=out_dim,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                )

    def _make_key(self, layer_idx: int, module_name: str) -> str:
        """Create key for lora_layers dict."""
        return f"layer_{layer_idx}_{module_name}"

    def get_lora_delta(
        self,
        layer_idx: int,
        module_name: str,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get LoRA delta for a specific layer and module.

        Args:
            layer_idx: Transformer layer index
            module_name: Target module name (e.g., 'q_proj')
            x: Input tensor [B, seq_len, hidden_size]

        Returns:
            delta: LoRA delta [B, seq_len, hidden_size]
        """
        key = self._make_key(layer_idx, module_name)

        if key not in self.lora_layers:
            # Return zeros if this module is not adapted
            return torch.zeros_like(x)

        return self.lora_layers[key](x)

    def get_all_lora_deltas(
        self,
        layer_idx: int,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Get LoRA deltas for all target modules in a layer.

        Args:
            layer_idx: Transformer layer index
            inputs: Dict of {module_name: input_tensor}

        Returns:
            deltas: Dict of {module_name: delta_tensor}
        """
        deltas = {}
        for module_name, x in inputs.items():
            if module_name in self.target_modules:
                deltas[module_name] = self.get_lora_delta(layer_idx, module_name, x)
        return deltas

    def num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class ExpertLoRACollection(nn.Module):
    """
    Collection of Expert LoRA adapters.

    Manages multiple SingleExpertLoRA instances, one per expert.
    Supports:
    - Single expert application
    - Weighted combination of experts
    - Export to PEFT format for vLLM

    Args:
        num_layers: Number of transformer layers
        hidden_size: Model hidden dimension
        num_experts: Number of expert LoRAs (default: 4)
        target_modules: Which modules to apply LoRA
        lora_r: LoRA rank per expert
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout rate
        module_dims: Optional dimension overrides

    Example:
        >>> collection = ExpertLoRACollection(
        ...     num_layers=28,
        ...     hidden_size=3584,
        ...     num_experts=4,
        ...     target_modules=['q_proj', 'v_proj'],
        ...     lora_r=16,
        ... )
        >>> # Apply single expert
        >>> output = collection.apply_single_expert(
        ...     expert_idx=0,
        ...     layer_idx=5,
        ...     module_name='q_proj',
        ...     base_output=base_out,
        ...     x=input_x,
        ... )
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_experts: int = 4,
        target_modules: Optional[List[str]] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        module_dims: Optional[Dict[str, Tuple[int, int]]] = None,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.target_modules = target_modules or ['q_proj', 'v_proj']
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # Store config for serialization
        self.config = ExpertLoRAConfig(
            num_experts=num_experts,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=self.target_modules,
        )

        # Create expert LoRAs
        self.experts = nn.ModuleList([
            SingleExpertLoRA(
                num_layers=num_layers,
                hidden_size=hidden_size,
                target_modules=self.target_modules,
                r=lora_r,
                alpha=lora_alpha,
                dropout=lora_dropout,
                module_dims=module_dims,
            )
            for _ in range(num_experts)
        ])

    def apply_single_expert(
        self,
        expert_idx: int,
        layer_idx: int,
        module_name: str,
        base_output: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply single expert's LoRA delta to base output.

        Args:
            expert_idx: Which expert to use (0 to num_experts-1)
            layer_idx: Transformer layer index
            module_name: Target module name
            base_output: Output from base model [B, seq_len, hidden_size]
            x: Input to the module [B, seq_len, hidden_size]

        Returns:
            modified_output: base_output + lora_delta
        """
        if expert_idx < 0 or expert_idx >= self.num_experts:
            raise ValueError(f"expert_idx must be in [0, {self.num_experts}), got {expert_idx}")

        delta = self.experts[expert_idx].get_lora_delta(layer_idx, module_name, x)
        return base_output + delta

    def apply_experts_weighted(
        self,
        layer_idx: int,
        module_name: str,
        base_output: torch.Tensor,
        x: torch.Tensor,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply weighted combination of experts.

        For each sample in batch, applies selected experts with their weights.

        Args:
            layer_idx: Transformer layer index
            module_name: Target module name
            base_output: [B, seq_len, hidden_size] Base model output
            x: [B, seq_len, hidden_size] Input to the module
            top_k_indices: [B, top_k] Selected expert indices
            top_k_weights: [B, top_k] Normalized weights for selected experts

        Returns:
            modified_output: base_output + weighted_sum(lora_deltas)
        """
        batch_size = base_output.size(0)
        top_k = top_k_indices.size(1)
        device = base_output.device

        # Compute weighted delta
        weighted_delta = torch.zeros_like(base_output)

        for b in range(batch_size):
            for k in range(top_k):
                expert_idx = top_k_indices[b, k].item()
                weight = top_k_weights[b, k]

                # Get delta from this expert
                delta = self.experts[expert_idx].get_lora_delta(
                    layer_idx, module_name, x[b:b+1]
                )
                weighted_delta[b:b+1] += weight * delta

        return base_output + weighted_delta

    def apply_experts_weighted_vectorized(
        self,
        layer_idx: int,
        module_name: str,
        base_output: torch.Tensor,
        x: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorized weighted application (more efficient for soft routing).

        Computes all expert deltas and combines with weights.

        Args:
            layer_idx: Transformer layer index
            module_name: Target module name
            base_output: [B, seq_len, hidden_size]
            x: [B, seq_len, hidden_size]
            routing_weights: [B, num_experts] Full routing distribution

        Returns:
            modified_output: base_output + weighted expert deltas
        """
        batch_size, seq_len, hidden_size = base_output.shape

        # Compute all expert deltas: [num_experts, B, seq_len, hidden_size]
        all_deltas = torch.stack([
            self.experts[i].get_lora_delta(layer_idx, module_name, x)
            for i in range(self.num_experts)
        ], dim=0)

        # Reshape weights for broadcasting: [B, num_experts, 1, 1]
        weights = routing_weights.unsqueeze(-1).unsqueeze(-1)  # [B, num_experts, 1, 1]

        # Weighted sum: [B, seq_len, hidden_size]
        # all_deltas: [num_experts, B, seq_len, hidden_size] -> [B, num_experts, seq_len, hidden_size]
        all_deltas = all_deltas.permute(1, 0, 2, 3)
        weighted_delta = (all_deltas * weights).sum(dim=1)

        return base_output + weighted_delta

    def get_expert(self, expert_idx: int) -> SingleExpertLoRA:
        """Get single expert by index."""
        return self.experts[expert_idx]

    def get_expert_state_dict(self, expert_idx: int) -> Dict[str, torch.Tensor]:
        """
        Get state dict for single expert.

        Args:
            expert_idx: Which expert

        Returns:
            State dict for this expert
        """
        return self.experts[expert_idx].state_dict()

    def load_expert_state_dict(
        self,
        expert_idx: int,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
    ):
        """
        Load state dict into single expert.

        Args:
            expert_idx: Which expert
            state_dict: State dict to load
            strict: Whether to strictly enforce that keys match
        """
        self.experts[expert_idx].load_state_dict(state_dict, strict=strict)

    def save_experts_separately(self, save_dir: str, save_format: str = 'peft'):
        """
        Save each expert separately.

        Args:
            save_dir: Directory to save experts
            save_format: 'peft' for PEFT format, 'raw' for raw state dict
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save config
        config_path = os.path.join(save_dir, 'moe_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        for i in range(self.num_experts):
            expert_dir = os.path.join(save_dir, f'expert_{i}')
            os.makedirs(expert_dir, exist_ok=True)

            if save_format == 'peft':
                # Convert to PEFT format
                peft_state_dict = self._convert_to_peft_format(i)
                torch.save(peft_state_dict, os.path.join(expert_dir, 'adapter_model.bin'))
                self._save_peft_config(expert_dir)
            else:
                # Save raw state dict
                torch.save(
                    self.experts[i].state_dict(),
                    os.path.join(expert_dir, 'expert.pt')
                )

        print(f"Saved {self.num_experts} experts to {save_dir}")

    def load_experts_separately(self, load_dir: str, load_format: str = 'peft'):
        """
        Load experts from separate directories.

        Args:
            load_dir: Directory containing experts
            load_format: 'peft' or 'raw'
        """
        for i in range(self.num_experts):
            expert_dir = os.path.join(load_dir, f'expert_{i}')

            if not os.path.exists(expert_dir):
                print(f"Warning: Expert {i} not found at {expert_dir}")
                continue

            if load_format == 'peft':
                self._load_from_peft_format(i, expert_dir)
            else:
                state_dict = torch.load(os.path.join(expert_dir, 'expert.pt'))
                self.experts[i].load_state_dict(state_dict)

        print(f"Loaded experts from {load_dir}")

    def _convert_to_peft_format(self, expert_idx: int) -> Dict[str, torch.Tensor]:
        """
        Convert expert weights to PEFT format.

        PEFT format: base_model.model.model.layers.{i}.self_attn.{module}.lora_{A/B}.weight
        """
        expert = self.experts[expert_idx]
        peft_state_dict = {}

        for layer_idx in range(self.num_layers):
            for module_name in self.target_modules:
                key = f"layer_{layer_idx}_{module_name}"

                if key not in expert.lora_layers:
                    continue

                lora_layer = expert.lora_layers[key]

                # Build PEFT format key
                if module_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    peft_prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{module_name}"
                elif module_name in ['gate_proj', 'up_proj', 'down_proj']:
                    peft_prefix = f"base_model.model.model.layers.{layer_idx}.mlp.{module_name}"
                else:
                    peft_prefix = f"base_model.model.model.layers.{layer_idx}.{module_name}"

                peft_state_dict[f"{peft_prefix}.lora_A.weight"] = lora_layer.lora_A.data.clone()
                peft_state_dict[f"{peft_prefix}.lora_B.weight"] = lora_layer.lora_B.data.clone()

        return peft_state_dict

    def _save_peft_config(self, save_dir: str):
        """Save PEFT adapter_config.json."""
        config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "target_modules": self.target_modules,
            "lora_dropout": self.lora_dropout,
            "bias": "none",
            "modules_to_save": None,
        }

        with open(os.path.join(save_dir, 'adapter_config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def _load_from_peft_format(self, expert_idx: int, peft_dir: str):
        """Load expert from PEFT format checkpoint."""
        # Try different file formats
        adapter_path = os.path.join(peft_dir, 'adapter_model.bin')
        if os.path.exists(adapter_path):
            peft_state_dict = torch.load(adapter_path, map_location='cpu')
        else:
            # Try safetensors
            safetensor_path = os.path.join(peft_dir, 'adapter_model.safetensors')
            if os.path.exists(safetensor_path):
                from safetensors.torch import load_file
                peft_state_dict = load_file(safetensor_path)
            else:
                raise FileNotFoundError(f"No adapter found in {peft_dir}")

        expert = self.experts[expert_idx]

        # Parse PEFT keys and load into expert
        for peft_key, value in peft_state_dict.items():
            try:
                # Parse: base_model.model.model.layers.{i}.self_attn.{module}.lora_{A/B}.weight
                parts = peft_key.split('.')

                # Find layer index
                layer_idx = None
                for j, part in enumerate(parts):
                    if part == 'layers' and j + 1 < len(parts):
                        layer_idx = int(parts[j + 1])
                        break

                if layer_idx is None:
                    continue

                # Find module name
                if 'self_attn' in peft_key:
                    attn_idx = parts.index('self_attn')
                    module_name = parts[attn_idx + 1]
                elif 'mlp' in peft_key:
                    mlp_idx = parts.index('mlp')
                    module_name = parts[mlp_idx + 1]
                else:
                    continue

                # Find lora type
                if 'lora_A' in peft_key:
                    lora_type = 'A'
                elif 'lora_B' in peft_key:
                    lora_type = 'B'
                else:
                    continue

                # Load into expert
                key = f"layer_{layer_idx}_{module_name}"
                if key in expert.lora_layers:
                    lora_layer = expert.lora_layers[key]
                    if lora_type == 'A':
                        lora_layer.lora_A.data.copy_(value)
                    else:
                        lora_layer.lora_B.data.copy_(value)

            except (IndexError, ValueError) as e:
                print(f"Skipping key {peft_key}: {e}")
                continue

    def num_parameters(self) -> int:
        """Get total number of parameters across all experts."""
        return sum(p.numel() for p in self.parameters())

    def num_parameters_per_expert(self) -> int:
        """Get number of parameters per expert."""
        return self.experts[0].num_parameters()


class MoEExpertApplier(nn.Module):
    """
    Applies MoE expert LoRAs during model forward pass.

    This class manages the routing state and provides hooks that can be
    registered with base model modules to inject LoRA deltas.

    Usage:
        1. Create applier with expert collection
        2. Register hooks with target modules
        3. Before each forward: call set_routing() with router output
        4. After forward: call clear_routing()

    Args:
        expert_collection: ExpertLoRACollection instance
        use_vectorized: Use vectorized (soft routing) or loop-based (hard routing)

    Example:
        >>> applier = MoEExpertApplier(expert_collection)
        >>> # Register hooks
        >>> for layer_idx, layer in enumerate(model.layers):
        ...     layer.self_attn.q_proj.register_forward_hook(
        ...         applier.create_forward_hook(layer_idx, 'q_proj')
        ...     )
        >>> # During forward
        >>> applier.set_routing(top_k_indices, top_k_weights)
        >>> output = model(inputs)
        >>> applier.clear_routing()
    """

    def __init__(
        self,
        expert_collection: ExpertLoRACollection,
        use_vectorized: bool = False,
    ):
        super().__init__()

        self.expert_collection = expert_collection
        self.use_vectorized = use_vectorized

        # Routing state (set before forward, cleared after)
        self._top_k_indices: Optional[torch.Tensor] = None
        self._top_k_weights: Optional[torch.Tensor] = None
        self._routing_weights: Optional[torch.Tensor] = None

        # Track registered hooks
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

    def set_routing(
        self,
        top_k_indices: torch.Tensor,
        top_k_weights: torch.Tensor,
        routing_weights: Optional[torch.Tensor] = None,
    ):
        """
        Set routing state for current forward pass.

        Args:
            top_k_indices: [B, top_k] Selected expert indices
            top_k_weights: [B, top_k] Weights for selected experts
            routing_weights: [B, num_experts] Full routing distribution (for vectorized)
        """
        self._top_k_indices = top_k_indices
        self._top_k_weights = top_k_weights
        self._routing_weights = routing_weights

    def clear_routing(self):
        """Clear routing state after forward pass."""
        self._top_k_indices = None
        self._top_k_weights = None
        self._routing_weights = None

    @property
    def has_routing(self) -> bool:
        """Check if routing state is set."""
        return self._top_k_indices is not None

    def create_forward_hook(
        self,
        layer_idx: int,
        module_name: str,
    ) -> Callable:
        """
        Create a forward hook for a specific layer and module.

        The hook adds LoRA delta to the module output.

        Args:
            layer_idx: Transformer layer index
            module_name: Target module name

        Returns:
            Hook function to register with module
        """
        def hook(module: nn.Module, input: Tuple, output: torch.Tensor) -> torch.Tensor:
            if not self.has_routing:
                return output

            # Get input to module
            x = input[0] if isinstance(input, tuple) else input

            # Apply expert LoRAs
            if self.use_vectorized and self._routing_weights is not None:
                modified = self.expert_collection.apply_experts_weighted_vectorized(
                    layer_idx=layer_idx,
                    module_name=module_name,
                    base_output=output,
                    x=x,
                    routing_weights=self._routing_weights,
                )
            else:
                modified = self.expert_collection.apply_experts_weighted(
                    layer_idx=layer_idx,
                    module_name=module_name,
                    base_output=output,
                    x=x,
                    top_k_indices=self._top_k_indices,
                    top_k_weights=self._top_k_weights,
                )

            return modified

        return hook

    def register_hooks(self, model: nn.Module, layer_pattern: str = "model.layers"):
        """
        Register hooks with model layers.

        Args:
            model: The base model
            layer_pattern: Pattern to find layers (e.g., "model.layers")
        """
        # Get layers
        layers = None
        for name, module in model.named_modules():
            if name == layer_pattern:
                layers = module
                break

        if layers is None:
            # Try to find layers
            for name, module in model.named_modules():
                if 'layers' in name and isinstance(module, nn.ModuleList):
                    layers = module
                    break

        if layers is None:
            raise ValueError(f"Could not find layers in model with pattern '{layer_pattern}'")

        # Register hooks for each layer and target module
        target_modules = self.expert_collection.target_modules

        for layer_idx, layer in enumerate(layers):
            for module_name in target_modules:
                # Find the target module
                target_module = None

                if module_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    if hasattr(layer, 'self_attn'):
                        target_module = getattr(layer.self_attn, module_name, None)
                elif module_name in ['gate_proj', 'up_proj', 'down_proj']:
                    if hasattr(layer, 'mlp'):
                        target_module = getattr(layer.mlp, module_name, None)

                if target_module is not None:
                    hook = self.create_forward_hook(layer_idx, module_name)
                    handle = target_module.register_forward_hook(hook)
                    self._hooks.append(handle)

        print(f"Registered {len(self._hooks)} hooks for MoE expert application")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()


def compute_expert_parameter_count(
    num_layers: int,
    hidden_size: int,
    num_experts: int,
    lora_r: int,
    target_modules: List[str],
) -> Dict[str, int]:
    """
    Compute parameter counts for MoE expert LoRAs.

    Args:
        num_layers: Number of transformer layers
        hidden_size: Model hidden dimension
        num_experts: Number of experts
        lora_r: LoRA rank
        target_modules: List of target modules

    Returns:
        Dict with parameter counts
    """
    # Per-module LoRA params: A [r, hidden] + B [hidden, r]
    params_per_module = 2 * hidden_size * lora_r

    # Per-layer params
    params_per_layer = params_per_module * len(target_modules)

    # Per-expert params
    params_per_expert = params_per_layer * num_layers

    # Total
    total_params = params_per_expert * num_experts

    return {
        'per_module': params_per_module,
        'per_layer': params_per_layer,
        'per_expert': params_per_expert,
        'total': total_params,
        'total_millions': total_params / 1e6,
    }


if __name__ == "__main__":
    # Quick tests
    print("Testing Expert LoRA modules...")
    print()

    # Test LoRALayer
    print("Test 1: LoRALayer...")
    lora = LoRALayer(in_features=256, out_features=256, r=16, alpha=32)
    x = torch.randn(4, 128, 256)
    delta = lora(x)
    assert delta.shape == x.shape, f"Expected {x.shape}, got {delta.shape}"
    print("  PASSED")

    # Test SingleExpertLoRA
    print("Test 2: SingleExpertLoRA...")
    expert = SingleExpertLoRA(
        num_layers=4,
        hidden_size=256,
        target_modules=['q_proj', 'v_proj'],
        r=16,
    )
    delta = expert.get_lora_delta(layer_idx=0, module_name='q_proj', x=x)
    assert delta.shape == x.shape
    print("  PASSED")

    # Test ExpertLoRACollection
    print("Test 3: ExpertLoRACollection...")
    collection = ExpertLoRACollection(
        num_layers=4,
        hidden_size=256,
        num_experts=4,
        target_modules=['q_proj', 'v_proj'],
        lora_r=16,
    )

    base_output = torch.randn(4, 128, 256)
    output = collection.apply_single_expert(
        expert_idx=0,
        layer_idx=0,
        module_name='q_proj',
        base_output=base_output,
        x=x,
    )
    assert output.shape == base_output.shape
    print("  PASSED")

    # Test weighted application
    print("Test 4: Weighted expert application...")
    top_k_indices = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
    top_k_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.8, 0.2], [0.5, 0.5]])

    output = collection.apply_experts_weighted(
        layer_idx=0,
        module_name='q_proj',
        base_output=base_output,
        x=x,
        top_k_indices=top_k_indices,
        top_k_weights=top_k_weights,
    )
    assert output.shape == base_output.shape
    print("  PASSED")

    # Test vectorized application
    print("Test 5: Vectorized expert application...")
    routing_weights = F.softmax(torch.randn(4, 4), dim=-1)
    output = collection.apply_experts_weighted_vectorized(
        layer_idx=0,
        module_name='q_proj',
        base_output=base_output,
        x=x,
        routing_weights=routing_weights,
    )
    assert output.shape == base_output.shape
    print("  PASSED")

    # Test parameter count
    print("Test 6: Parameter count...")
    params = compute_expert_parameter_count(
        num_layers=28,
        hidden_size=3584,
        num_experts=4,
        lora_r=16,
        target_modules=['q_proj', 'v_proj'],
    )
    print(f"  Total parameters: {params['total_millions']:.2f}M")
    print("  PASSED")

    print()
    print("=== All tests passed! ===")
