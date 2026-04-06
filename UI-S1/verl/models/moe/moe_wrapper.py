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
MoE VLM Wrapper for GUI Agent.

This module wraps a base VLM (Qwen2.5-VL) with MoE routing and expert LoRAs.
It uses **module replacement** (not hooks) to inject expert LoRA deltas into
the base model's Linear layers, ensuring correct gradient flow during training.

Architecture:
    Input: (screenshot, instruction)
              │
              ▼
    ┌─────────────────────────────────────────┐
    │         Base VLM (Frozen)                │
    │   target Linear layers replaced with     │
    │   MoELoRALinear (frozen base + LoRA)     │
    └─────────────────────────────────────────┘
              │
    Pass 1: LoRA disabled (routing_weights=None)
              │ → hidden_states → Router → routing_weights
              │
    Pass 2: LoRA enabled (routing_weights set on MoELoRALinear modules)
              │ → logits + loss
              ▼
         Action Output
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from verl.models.moe.router import (
    TextOnlyRouter,
    ContextAwareRouter,
    RouterOutput,
    InstructionFeatureExtractor,
    create_instruction_mask,
    create_vision_mask,
    create_text_context_mask,
)
from verl.models.moe.expert_lora import (
    MoELoRALinear,
    LoRALayer,
)
from verl.models.moe.moe_loss import MoELoss, MoELossOutput

logger = logging.getLogger(__name__)


@dataclass
class MoEConfig:
    """Configuration for MoE VLM Wrapper."""

    # Expert configuration
    num_experts: int = 4
    top_k: int = 1

    # LoRA configuration
    expert_lora_r: int = 16
    expert_lora_alpha: int = 32
    expert_lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ['q_proj', 'v_proj']
    )

    # Selective MoE: which target_modules get MoE treatment (None = all)
    moe_modules: Optional[List[str]] = None
    # Standard LoRA config for non-MoE modules (used in hybrid mode)
    standard_lora_r: int = 32
    standard_lora_alpha: int = 64

    # Router configuration
    router_hidden: int = 256
    router_dropout: float = 0.1
    router_temperature: float = 1.0

    # Feature extraction
    pooling_strategy: str = 'mean'

    # Loss configuration
    balance_weight: float = 0.1
    balance_type: str = 'mse'
    z_loss_weight: float = 0.0

    # Inference mode
    use_vectorized_routing: bool = False

    # Router type: 'text_only' (original) or 'context_aware' (vision+text)
    router_type: str = 'text_only'

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: dict) -> "MoEConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class MoEOutput:
    """Output from MoE VLM forward pass."""

    # Main outputs
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None

    # Routing information
    routing_weights: Optional[torch.Tensor] = None
    top_k_indices: Optional[torch.Tensor] = None
    top_k_weights: Optional[torch.Tensor] = None
    router_logits: Optional[torch.Tensor] = None

    # Loss breakdown
    lm_loss: Optional[torch.Tensor] = None
    balance_loss: Optional[torch.Tensor] = None
    z_loss: Optional[torch.Tensor] = None

    # Hidden states (optional)
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging."""
        result = {}
        if self.loss is not None:
            result['loss'] = self.loss.item()
        if self.lm_loss is not None:
            result['lm_loss'] = self.lm_loss.item()
        if self.balance_loss is not None:
            result['balance_loss'] = self.balance_loss.item()
        if self.z_loss is not None:
            result['z_loss'] = self.z_loss.item()
        return result


class MoEVLMWrapper(nn.Module):
    """
    MoE Wrapper for Vision-Language Model using module replacement.

    Wraps a base VLM (frozen) with:
    - Text-only router for expert selection
    - MoELoRALinear modules replacing target nn.Linear layers

    Unlike hook-based approaches, module replacement ensures correct gradient
    flow to expert LoRA parameters during training. DDP works naturally since
    LoRA params are proper children in the module tree.

    Args:
        base_model: Pre-trained VLM (will be frozen)
        moe_config: MoE configuration
        tokenizer: Optional tokenizer for instruction mask creation

    Example:
        >>> from transformers import Qwen2VLForConditionalGeneration
        >>> base = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B")
        >>> config = MoEConfig(num_experts=6, expert_lora_r=16)
        >>> moe_model = MoEVLMWrapper(base, config)
        >>> outputs = moe_model(input_ids, attention_mask, pixel_values, labels=labels)
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        moe_config: MoEConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        super().__init__()

        self.moe_config = moe_config
        self.tokenizer = tokenizer

        # Store base model (will be frozen)
        self.base_model = base_model

        # Get model dimensions
        self.hidden_size = self._get_hidden_size()
        self.num_layers = self._get_num_layers()

        # Freeze base model
        self._freeze_base_model()

        # Initialize MoE components (router, loss, module replacement)
        self._init_moe_components()

    def _get_hidden_size(self) -> int:
        """Get hidden size from base model."""
        if hasattr(self.base_model, 'config'):
            if hasattr(self.base_model.config, 'hidden_size'):
                return self.base_model.config.hidden_size
            if hasattr(self.base_model.config, 'd_model'):
                return self.base_model.config.d_model
        raise ValueError("Cannot determine hidden_size from base model")

    def _get_num_layers(self) -> int:
        """Get number of layers from base model."""
        if hasattr(self.base_model, 'config'):
            if hasattr(self.base_model.config, 'num_hidden_layers'):
                return self.base_model.config.num_hidden_layers
            if hasattr(self.base_model.config, 'num_layers'):
                return self.base_model.config.num_layers
        raise ValueError("Cannot determine num_layers from base model")

    def _freeze_base_model(self):
        """Freeze all base model parameters."""
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

    def _init_moe_components(self):
        """Initialize router, MoE loss, and replace target modules with MoELoRALinear."""
        config = self.moe_config

        # 1. Router (selected by router_type)
        if config.router_type == 'context_aware':
            self.router = ContextAwareRouter(
                hidden_size=self.hidden_size,
                num_experts=config.num_experts,
                router_hidden=config.router_hidden,
                top_k=config.top_k,
                dropout=config.router_dropout,
                temperature=config.router_temperature,
            )
        else:
            # Default: text_only (original behavior)
            self.router = TextOnlyRouter(
                hidden_size=self.hidden_size,
                num_experts=config.num_experts,
                router_hidden=config.router_hidden,
                top_k=config.top_k,
                dropout=config.router_dropout,
                temperature=config.router_temperature,
            )

        # 2. Feature Extractors
        self.feature_extractor = InstructionFeatureExtractor(
            pooling_strategy=config.pooling_strategy,
        )
        if config.router_type == 'context_aware':
            # Separate extractor for vision features
            self.vision_feature_extractor = InstructionFeatureExtractor(
                pooling_strategy=config.pooling_strategy,
            )

        # 3. MoE Loss
        self.moe_loss = MoELoss(
            num_experts=config.num_experts,
            balance_weight=config.balance_weight,
            balance_type=config.balance_type,
            z_loss_weight=config.z_loss_weight,
        )

        # 4. Replace target modules with MoELoRALinear (and standard LoRA for hybrid)
        self._moe_linear_modules: List[MoELoRALinear] = []
        self._std_linear_modules: List[MoELoRALinear] = []
        self._replace_target_modules()

    def _find_layers(self) -> nn.ModuleList:
        """Find the transformer layers in the base model."""
        # Try common patterns
        for pattern in ["model.layers", "model.model.layers", "language_model.model.layers"]:
            parts = pattern.split(".")
            obj = self.base_model
            found = True
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    found = False
                    break
            if found and isinstance(obj, nn.ModuleList):
                return obj

        # Fallback: find any ModuleList named "layers"
        for name, module in self.base_model.named_modules():
            if name.endswith('layers') and isinstance(module, nn.ModuleList):
                return module

        raise ValueError("Could not find transformer layers in base model")

    def _replace_target_modules(self):
        """Replace target nn.Linear modules with MoELoRALinear.

        In hybrid mode (moe_modules is set), MoE-designated modules get
        MoELoRALinear with num_experts experts and moe_r rank. Standard modules
        get MoELoRALinear with 1 expert and standard_r rank (mathematically
        identical to standard LoRA, but uses the same code path).
        """
        config = self.moe_config
        layers = self._find_layers()

        # Determine which modules get MoE vs standard LoRA
        # None = all modules are MoE (original behavior), [] = no MoE modules
        if config.moe_modules is not None:
            moe_module_set = set(config.moe_modules)
        else:
            moe_module_set = set(config.target_modules)
        replaced_moe = 0
        replaced_std = 0

        for layer_idx, layer in enumerate(layers):
            for module_name in config.target_modules:
                # Find parent module and target
                if module_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    parent = getattr(layer, 'self_attn', None)
                elif module_name in ['gate_proj', 'up_proj', 'down_proj']:
                    parent = getattr(layer, 'mlp', None)
                else:
                    parent = layer

                if parent is None:
                    continue

                original_linear = getattr(parent, module_name, None)
                if original_linear is None or not isinstance(original_linear, nn.Linear):
                    continue

                is_moe = module_name in moe_module_set

                if is_moe:
                    moe_linear = MoELoRALinear(
                        base_linear=original_linear,
                        num_experts=config.num_experts,
                        r=config.expert_lora_r,
                        alpha=config.expert_lora_alpha,
                        dropout=config.expert_lora_dropout,
                    )
                    setattr(parent, module_name, moe_linear)
                    self._moe_linear_modules.append(moe_linear)
                    replaced_moe += 1
                else:
                    # Standard LoRA: 1 expert, higher rank
                    std_linear = MoELoRALinear(
                        base_linear=original_linear,
                        num_experts=1,
                        r=config.standard_lora_r,
                        alpha=config.standard_lora_alpha,
                        dropout=config.expert_lora_dropout,
                    )
                    setattr(parent, module_name, std_linear)
                    self._std_linear_modules.append(std_linear)
                    replaced_std += 1

        logger.info(
            f"Replaced {replaced_moe} MoE modules "
            f"({config.num_experts} experts, r={config.expert_lora_r}) + "
            f"{replaced_std} standard LoRA modules "
            f"(1 expert, r={config.standard_lora_r})"
        )

    def _set_routing_weights(self, routing_weights: torch.Tensor):
        """Set routing weights on all MoELoRALinear modules.

        MoE modules get the full [B, num_experts] routing weights.
        Standard LoRA modules get constant ones [B, 1] (always-on single expert).
        """
        for m in self._moe_linear_modules:
            m.set_routing_weights(routing_weights)

        if self._std_linear_modules:
            batch_size = routing_weights.shape[0]
            ones = torch.ones(batch_size, 1, device=routing_weights.device,
                              dtype=routing_weights.dtype)
            for m in self._std_linear_modules:
                m.set_routing_weights(ones)

    def _clear_routing_weights(self):
        """Clear routing weights (MoELoRALinear acts as plain Linear)."""
        for m in self._moe_linear_modules:
            m.set_routing_weights(None)
        for m in self._std_linear_modules:
            m.set_routing_weights(None)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        instruction_mask: Optional[torch.Tensor] = None,
        instruction_texts: Optional[List[str]] = None,
        return_routing_info: bool = True,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> MoEOutput:
        """
        Forward pass with MoE routing.

        Pass 1 (no_grad, LoRA disabled): Extract hidden states for routing.
        Pass 2 (with grad, LoRA enabled): Compute logits and loss with expert deltas.

        Args:
            input_ids: [B, seq_len] Input token IDs
            attention_mask: [B, seq_len] Attention mask
            pixel_values: Vision inputs
            labels: [B, seq_len] Labels for LM loss
            instruction_mask: [B, seq_len] Boolean mask for instruction tokens
            instruction_texts: List of instruction strings (alternative to mask)
            return_routing_info: Whether to return routing details
            output_hidden_states: Whether to return all hidden states

        Returns:
            MoEOutput with logits, loss, and routing information
        """
        # ---- Pass 1: Get hidden states for routing (no LoRA, no grad) ----
        self._clear_routing_weights()
        with torch.no_grad():
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )

        # Get last hidden state
        if hasattr(base_outputs, 'hidden_states') and base_outputs.hidden_states:
            hidden_states = base_outputs.hidden_states[-1]
        elif hasattr(base_outputs, 'last_hidden_state'):
            hidden_states = base_outputs.last_hidden_state
        else:
            raise ValueError("Cannot get hidden states from base model")

        # ---- Compute routing ----
        if self.moe_config.router_type == 'context_aware':
            # Context-aware routing: use vision + text features
            vision_mask = create_vision_mask(input_ids, self.tokenizer)
            text_mask = create_text_context_mask(input_ids, self.tokenizer)
            vision_features = self.vision_feature_extractor(hidden_states, vision_mask)
            text_features = self.feature_extractor(hidden_states, text_mask)
            router_output = self.router(vision_features, text_features)
        else:
            # Original text_only routing (unchanged)
            if instruction_mask is None:
                if instruction_texts is not None and self.tokenizer is not None:
                    from verl.models.moe.router import create_instruction_mask_from_text
                    instruction_mask = create_instruction_mask_from_text(
                        input_ids, self.tokenizer, instruction_texts
                    )
                elif self.tokenizer is not None:
                    instruction_mask = create_instruction_mask(input_ids, self.tokenizer)
                else:
                    instruction_mask = torch.ones_like(input_ids, dtype=torch.bool)
            instruction_features = self.feature_extractor(hidden_states, instruction_mask)
            router_output = self.router(instruction_features)

        # ---- Pass 2: Forward with expert LoRAs enabled ----
        self._set_routing_weights(router_output.routing_weights)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        # ---- Compute loss ----
        loss = None
        lm_loss = None
        balance_loss = None
        z_loss = None

        if labels is not None and hasattr(outputs, 'loss') and outputs.loss is not None:
            lm_loss = outputs.loss
            loss_output = self.moe_loss(
                lm_loss=lm_loss,
                routing_weights=router_output.routing_weights,
                router_logits=router_output.router_logits,
            )
            loss = loss_output.total_loss
            balance_loss = loss_output.balance_loss
            z_loss = loss_output.z_loss

        # ---- Build output ----
        moe_output = MoEOutput(
            logits=outputs.logits,
            loss=loss,
            lm_loss=lm_loss,
            balance_loss=balance_loss,
            z_loss=z_loss,
        )

        if return_routing_info:
            moe_output.routing_weights = router_output.routing_weights
            moe_output.top_k_indices = router_output.top_k_indices
            moe_output.top_k_weights = router_output.top_k_weights
            moe_output.router_logits = router_output.router_logits

        if output_hidden_states and hasattr(outputs, 'hidden_states'):
            moe_output.hidden_states = outputs.hidden_states

        return moe_output

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        instruction_mask: Optional[torch.Tensor] = None,
        instruction_texts: Optional[List[str]] = None,
        max_new_tokens: int = 100,
        **generate_kwargs,
    ) -> Tuple[torch.Tensor, RouterOutput]:
        """
        Generate with MoE routing.

        Args:
            input_ids: [B, seq_len] Input token IDs
            attention_mask: [B, seq_len] Attention mask
            pixel_values: Vision inputs
            image_grid_thw: Image grid for Qwen2.5-VL
            instruction_mask: [B, seq_len] Instruction token mask
            instruction_texts: Alternative to instruction_mask
            max_new_tokens: Maximum tokens to generate
            **generate_kwargs: Additional generation arguments

        Returns:
            generated_ids: [B, seq_len + generated_len]
            router_output: Routing information
        """
        # Pass 1: Get hidden states for routing (no LoRA)
        self._clear_routing_weights()
        with torch.no_grad():
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = base_outputs.hidden_states[-1]

        # Compute routing
        if self.moe_config.router_type == 'context_aware':
            vision_mask = create_vision_mask(input_ids, self.tokenizer)
            text_mask = create_text_context_mask(input_ids, self.tokenizer)
            vision_features = self.vision_feature_extractor(hidden_states, vision_mask)
            text_features = self.feature_extractor(hidden_states, text_mask)
            router_output = self.router(vision_features, text_features)
        else:
            # Original text_only routing (unchanged)
            if instruction_mask is None:
                if instruction_texts is not None and self.tokenizer is not None:
                    from verl.models.moe.router import create_instruction_mask_from_text
                    instruction_mask = create_instruction_mask_from_text(
                        input_ids, self.tokenizer, instruction_texts
                    )
                elif self.tokenizer is not None:
                    instruction_mask = create_instruction_mask(input_ids, self.tokenizer)
                else:
                    instruction_mask = torch.ones_like(input_ids, dtype=torch.bool)
            instruction_features = self.feature_extractor(hidden_states, instruction_mask)
            router_output = self.router(instruction_features)

        # Set routing for generation (LoRA enabled)
        self._set_routing_weights(router_output.routing_weights)

        # Generate
        generated_ids = self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

        # Clear routing after generation
        self._clear_routing_weights()

        return generated_ids, router_output

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (router + MoE expert LoRAs + standard LoRAs)."""
        params = []
        params.extend(list(self.router.parameters()))
        for m in self._moe_linear_modules:
            for lora in m.expert_loras:
                params.extend(list(lora.parameters()))
        for m in self._std_linear_modules:
            for lora in m.expert_loras:
                params.extend(list(lora.parameters()))
        return params

    def num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_parameters())

    def num_total_parameters(self) -> int:
        """Get total number of parameters (including frozen)."""
        return sum(p.numel() for p in self.parameters())

    def train(self, mode: bool = True):
        """Set training mode (only affects MoE components, base stays frozen)."""
        self.base_model.eval()
        self.router.train(mode)
        self.feature_extractor.train(mode)
        if hasattr(self, 'vision_feature_extractor'):
            self.vision_feature_extractor.train(mode)
        for m in self._moe_linear_modules:
            for lora in m.expert_loras:
                lora.train(mode)
        for m in self._std_linear_modules:
            for lora in m.expert_loras:
                lora.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    # ── Proxy methods for HF Trainer compatibility ──

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Proxy to base_model for HF Trainer compatibility."""
        self.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """Proxy to base_model for HF Trainer compatibility."""
        self.base_model.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        """Proxy to base_model for HF Trainer compatibility."""
        self.base_model.enable_input_require_grads()

    @property
    def config(self):
        """Proxy to base_model.config for HF Trainer compatibility."""
        return self.base_model.config

    @property
    def device(self):
        """Proxy to base_model.device for HF Trainer compatibility."""
        return self.base_model.device

    def save_moe_checkpoint(self, save_dir: str):
        """
        Save MoE components (router + expert LoRAs) in PEFT-compatible format.

        Creates:
        - router.pt: Router state dict
        - experts/expert_{i}/adapter_model.bin: Per-expert PEFT weights
        - moe_config.json: Configuration

        Args:
            save_dir: Directory to save checkpoint
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save router
        torch.save(
            self.router.state_dict(),
            os.path.join(save_dir, 'router.pt')
        )

        # Save experts in PEFT format (per expert)
        config = self.moe_config
        experts_dir = os.path.join(save_dir, 'experts')
        os.makedirs(experts_dir, exist_ok=True)

        # Collect expert LoRA weights organized by expert
        layers = self._find_layers()
        for expert_idx in range(config.num_experts):
            peft_state_dict = {}
            module_idx = 0

            for layer_idx in range(len(layers)):
                for module_name in config.target_modules:
                    if module_idx >= len(self._moe_linear_modules):
                        break
                    moe_linear = self._moe_linear_modules[module_idx]
                    lora_layer = moe_linear.expert_loras[expert_idx]

                    # Build PEFT key
                    if module_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                        prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{module_name}"
                    elif module_name in ['gate_proj', 'up_proj', 'down_proj']:
                        prefix = f"base_model.model.model.layers.{layer_idx}.mlp.{module_name}"
                    else:
                        prefix = f"base_model.model.model.layers.{layer_idx}.{module_name}"

                    peft_state_dict[f"{prefix}.lora_A.weight"] = lora_layer.lora_A.data.clone()
                    peft_state_dict[f"{prefix}.lora_B.weight"] = lora_layer.lora_B.data.clone()
                    module_idx += 1

            expert_dir = os.path.join(experts_dir, f'expert_{expert_idx}')
            os.makedirs(expert_dir, exist_ok=True)
            torch.save(peft_state_dict, os.path.join(expert_dir, 'adapter_model.bin'))

            # Save PEFT config
            peft_config = {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": config.expert_lora_r,
                "lora_alpha": config.expert_lora_alpha,
                "target_modules": config.target_modules,
                "lora_dropout": config.expert_lora_dropout,
                "bias": "none",
            }
            with open(os.path.join(expert_dir, 'adapter_config.json'), 'w') as f:
                json.dump(peft_config, f, indent=2)

        # Save MoE config
        with open(os.path.join(save_dir, 'moe_config.json'), 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

        logger.info(f"Saved MoE checkpoint to {save_dir}")

    def load_moe_checkpoint(self, load_dir: str):
        """
        Load MoE components from checkpoint.

        Args:
            load_dir: Directory containing checkpoint
        """
        # Load router
        router_path = os.path.join(load_dir, 'router.pt')
        if os.path.exists(router_path):
            state_dict = torch.load(router_path, map_location='cpu')
            self.router.load_state_dict(state_dict)
            logger.info(f"Loaded router from {router_path}")

        # Load experts
        config = self.moe_config
        experts_dir = os.path.join(load_dir, 'experts')
        if not os.path.exists(experts_dir):
            return

        self._load_experts_from_peft_dir(experts_dir)

    def _load_experts_from_peft_dir(self, experts_dir: str):
        """Load expert LoRA weights from a directory of PEFT checkpoints.

        Each expert is stored in experts_dir/expert_{i}/ with adapter_model.bin
        or adapter_model.safetensors.

        Args:
            experts_dir: Directory containing expert_{0..N-1} subdirectories
        """
        config = self.moe_config
        layers = self._find_layers()

        for expert_idx in range(config.num_experts):
            expert_dir = os.path.join(experts_dir, f'expert_{expert_idx}')

            # Try .bin then .safetensors
            adapter_path = os.path.join(expert_dir, 'adapter_model.bin')
            if os.path.exists(adapter_path):
                peft_state_dict = torch.load(adapter_path, map_location='cpu')
            else:
                safetensor_path = os.path.join(expert_dir, 'adapter_model.safetensors')
                if os.path.exists(safetensor_path):
                    from safetensors.torch import load_file
                    peft_state_dict = load_file(safetensor_path)
                else:
                    logger.warning(f"Expert {expert_idx} not found at {expert_dir}")
                    continue

            module_idx = 0
            loaded = 0
            for layer_idx in range(len(layers)):
                for module_name in config.target_modules:
                    if module_idx >= len(self._moe_linear_modules):
                        break
                    moe_linear = self._moe_linear_modules[module_idx]
                    lora_layer = moe_linear.expert_loras[expert_idx]

                    if module_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                        prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{module_name}"
                    elif module_name in ['gate_proj', 'up_proj', 'down_proj']:
                        prefix = f"base_model.model.model.layers.{layer_idx}.mlp.{module_name}"
                    else:
                        prefix = f"base_model.model.model.layers.{layer_idx}.{module_name}"

                    a_key = f"{prefix}.lora_A.weight"
                    b_key = f"{prefix}.lora_B.weight"
                    if a_key in peft_state_dict:
                        lora_layer.lora_A.data.copy_(peft_state_dict[a_key])
                        loaded += 1
                    if b_key in peft_state_dict:
                        lora_layer.lora_B.data.copy_(peft_state_dict[b_key])
                        loaded += 1
                    module_idx += 1

            logger.info(f"Expert {expert_idx}: loaded {loaded} weight tensors")

        logger.info(f"Loaded experts from {experts_dir}")

    def load_moe_warmstart(self, checkpoint_dir: str):
        """
        Load MoE warm-start checkpoint (converted from SFT LoRA).

        This loads pre-trained expert weights and router from the format
        created by convert_sft_lora_to_moe.py:
            checkpoint_dir/
            ├── router.pt
            ├── moe_config.json
            └── experts/expert_{0-3}/adapter_model.bin

        Args:
            checkpoint_dir: Path to warm-start checkpoint directory
        """
        if not os.path.exists(checkpoint_dir):
            logger.warning(f"Warm-start checkpoint not found: {checkpoint_dir}")
            return

        logger.info(f"Loading MoE warm-start from {checkpoint_dir}")

        # Load router weights
        router_path = os.path.join(checkpoint_dir, 'router.pt')
        if os.path.exists(router_path):
            router_state = torch.load(router_path, map_location='cpu')
            self.router.load_state_dict(router_state)
            logger.info(f"Loaded router from warm-start checkpoint")

        # Load expert weights
        experts_dir = os.path.join(checkpoint_dir, 'experts')
        if os.path.exists(experts_dir):
            self._load_experts_from_peft_dir(experts_dir)
        else:
            logger.warning(f"No experts directory found at {experts_dir}")

        # Log config info if available
        config_path = os.path.join(checkpoint_dir, 'moe_config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                warmstart_config = json.load(f)
            logger.info(
                f"Warm-start config: r={warmstart_config.get('expert_lora_r')}, "
                f"alpha={warmstart_config.get('expert_lora_alpha')}, "
                f"modules={warmstart_config.get('target_modules')}, "
                f"source={warmstart_config.get('source_checkpoint', 'unknown')}"
            )

    def save_selective_checkpoint(self, save_dir: str):
        """
        Save hybrid MoE checkpoint as flat state_dict.

        Saves router state_dict + all LoRA weights (MoE and standard) as a
        single flat state_dict, plus moe_config.json for reconstruction.

        Args:
            save_dir: Directory to save checkpoint
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save router
        torch.save(
            self.router.state_dict(),
            os.path.join(save_dir, 'router.pt')
        )

        # Save all LoRA weights as flat state_dict
        lora_state_dict = {}
        config = self.moe_config
        layers = self._find_layers()
        moe_module_set = set(config.moe_modules) if config.moe_modules is not None else set(config.target_modules)

        moe_idx = 0
        std_idx = 0
        for layer_idx in range(len(layers)):
            for module_name in config.target_modules:
                is_moe = module_name in moe_module_set

                if is_moe:
                    if moe_idx >= len(self._moe_linear_modules):
                        continue
                    moe_linear = self._moe_linear_modules[moe_idx]
                    for expert_idx, lora_layer in enumerate(moe_linear.expert_loras):
                        prefix = f"moe.layer_{layer_idx}.{module_name}.expert_{expert_idx}"
                        lora_state_dict[f"{prefix}.lora_A"] = lora_layer.lora_A.data.clone()
                        lora_state_dict[f"{prefix}.lora_B"] = lora_layer.lora_B.data.clone()
                    moe_idx += 1
                else:
                    if std_idx >= len(self._std_linear_modules):
                        continue
                    std_linear = self._std_linear_modules[std_idx]
                    lora_layer = std_linear.expert_loras[0]
                    prefix = f"std.layer_{layer_idx}.{module_name}"
                    lora_state_dict[f"{prefix}.lora_A"] = lora_layer.lora_A.data.clone()
                    lora_state_dict[f"{prefix}.lora_B"] = lora_layer.lora_B.data.clone()
                    std_idx += 1

        torch.save(lora_state_dict, os.path.join(save_dir, 'lora_weights.pt'))

        # Save config
        with open(os.path.join(save_dir, 'moe_config.json'), 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

        logger.info(
            f"Saved selective checkpoint to {save_dir}: "
            f"{moe_idx} MoE modules, {std_idx} standard modules, "
            f"{len(lora_state_dict)} weight tensors"
        )

    def load_selective_checkpoint(self, load_dir: str):
        """
        Load hybrid MoE checkpoint from flat state_dict.

        Args:
            load_dir: Directory containing checkpoint
        """
        # Load router
        router_path = os.path.join(load_dir, 'router.pt')
        if os.path.exists(router_path):
            state_dict = torch.load(router_path, map_location='cpu')
            self.router.load_state_dict(state_dict)
            logger.info(f"Loaded router from {router_path}")

        # Load LoRA weights
        lora_path = os.path.join(load_dir, 'lora_weights.pt')
        if not os.path.exists(lora_path):
            logger.warning(f"No lora_weights.pt found in {load_dir}")
            return

        lora_state_dict = torch.load(lora_path, map_location='cpu')
        config = self.moe_config
        layers = self._find_layers()
        moe_module_set = set(config.moe_modules) if config.moe_modules is not None else set(config.target_modules)

        loaded = 0
        moe_idx = 0
        std_idx = 0
        for layer_idx in range(len(layers)):
            for module_name in config.target_modules:
                is_moe = module_name in moe_module_set

                if is_moe:
                    if moe_idx >= len(self._moe_linear_modules):
                        continue
                    moe_linear = self._moe_linear_modules[moe_idx]
                    for expert_idx, lora_layer in enumerate(moe_linear.expert_loras):
                        prefix = f"moe.layer_{layer_idx}.{module_name}.expert_{expert_idx}"
                        a_key = f"{prefix}.lora_A"
                        b_key = f"{prefix}.lora_B"
                        if a_key in lora_state_dict:
                            lora_layer.lora_A.data.copy_(lora_state_dict[a_key])
                            loaded += 1
                        if b_key in lora_state_dict:
                            lora_layer.lora_B.data.copy_(lora_state_dict[b_key])
                            loaded += 1
                    moe_idx += 1
                else:
                    if std_idx >= len(self._std_linear_modules):
                        continue
                    std_linear = self._std_linear_modules[std_idx]
                    lora_layer = std_linear.expert_loras[0]
                    prefix = f"std.layer_{layer_idx}.{module_name}"
                    a_key = f"{prefix}.lora_A"
                    b_key = f"{prefix}.lora_B"
                    if a_key in lora_state_dict:
                        lora_layer.lora_A.data.copy_(lora_state_dict[a_key])
                        loaded += 1
                    if b_key in lora_state_dict:
                        lora_layer.lora_B.data.copy_(lora_state_dict[b_key])
                        loaded += 1
                    std_idx += 1

        logger.info(f"Loaded {loaded} weight tensors from {load_dir}")

    def get_routing_statistics(
        self,
        routing_weights: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute routing statistics for analysis.

        Args:
            routing_weights: [B, num_experts] Routing probabilities

        Returns:
            Dict with statistics
        """
        from verl.models.moe.router import compute_routing_entropy, compute_routing_diversity
        from verl.models.moe.moe_loss import compute_expert_utilization, compute_load_balance_coefficient

        stats = {}

        utilization = compute_expert_utilization(
            routing_weights, self.moe_config.num_experts
        )
        for i, u in enumerate(utilization.tolist()):
            stats[f'expert_{i}_utilization'] = u

        stats['load_balance_coefficient'] = compute_load_balance_coefficient(utilization)

        entropy = compute_routing_entropy(routing_weights)
        stats['routing_entropy_mean'] = entropy.mean().item()
        stats['routing_diversity'] = compute_routing_diversity(routing_weights)

        return stats


def create_moe_wrapper(
    base_model_path: str,
    moe_config: Optional[MoEConfig] = None,
    device: str = 'auto',
    **model_kwargs,
) -> MoEVLMWrapper:
    """
    Factory function to create MoE wrapper.

    Args:
        base_model_path: Path or name of base model
        moe_config: MoE configuration (uses defaults if None)
        device: Device to load model ('auto', 'cuda', 'cpu')
        **model_kwargs: Additional arguments for model loading

    Returns:
        MoEVLMWrapper instance
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=device if device != 'auto' else 'auto',
        trust_remote_code=True,
        **model_kwargs,
    )

    if moe_config is None:
        moe_config = MoEConfig()

    wrapper = MoEVLMWrapper(
        base_model=model,
        moe_config=moe_config,
        tokenizer=tokenizer,
    )

    return wrapper


if __name__ == "__main__":
    print("Testing MoE VLM Wrapper (module replacement)...")
    print()

    # Create a simple mock base model for testing
    class MockConfig:
        hidden_size = 256
        num_hidden_layers = 4
        vocab_size = 1000

    class MockOutput:
        def __init__(self, hidden_states, logits, loss=None):
            self.hidden_states = hidden_states
            self.logits = logits
            self.loss = loss
            self.last_hidden_state = hidden_states[-1]

    class MockBaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = MockConfig()
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    'self_attn': nn.ModuleDict({
                        'q_proj': nn.Linear(256, 256),
                        'v_proj': nn.Linear(256, 256),
                    })
                })
                for _ in range(4)
            ])
            self.lm_head = nn.Linear(256, 1000)

        def forward(self, input_ids, attention_mask=None, pixel_values=None,
                    labels=None, output_hidden_states=False, return_dict=True, **kwargs):
            batch_size, seq_len = input_ids.shape
            hidden = torch.randn(batch_size, seq_len, 256)

            # Pass through layers (so MoELoRALinear modules get called)
            for layer in self.layers:
                q_out = layer['self_attn']['q_proj'](hidden)
                v_out = layer['self_attn']['v_proj'](hidden)
                hidden = hidden + q_out + v_out  # simplified

            hidden_states = tuple([hidden] * 5)
            logits = self.lm_head(hidden)

            loss = None
            if labels is not None:
                loss = F.cross_entropy(
                    logits.view(-1, 1000),
                    labels.view(-1),
                    ignore_index=-100,
                )

            return MockOutput(hidden_states, logits, loss)

        def generate(self, input_ids, **kwargs):
            batch_size = input_ids.size(0)
            new_tokens = torch.randint(0, 1000, (batch_size, 10))
            return torch.cat([input_ids, new_tokens], dim=1)

    # Create mock model
    base_model = MockBaseModel()

    # Create MoE config
    config = MoEConfig(
        num_experts=4,
        top_k=4,  # all experts
        expert_lora_r=8,
        expert_lora_alpha=16,
        target_modules=['q_proj', 'v_proj'],
        balance_weight=0.0,
        use_vectorized_routing=True,
    )

    # Test 1: Creation
    print("Test 1: Creating MoE wrapper with module replacement...")
    wrapper = MoEVLMWrapper(base_model, config)
    print(f"  MoELoRALinear modules: {len(wrapper._moe_linear_modules)}")
    print(f"  Trainable params: {wrapper.num_trainable_parameters():,}")
    print(f"  Total params: {wrapper.num_total_parameters():,}")

    # Verify module replacement happened
    assert isinstance(base_model.layers[0]['self_attn']['q_proj'], MoELoRALinear)
    assert isinstance(base_model.layers[0]['self_attn']['v_proj'], MoELoRALinear)
    print("  Module replacement verified ✓")
    print("  PASSED")

    # Test 2: Forward pass
    print("\nTest 2: Forward pass...")
    batch_size = 4
    seq_len = 32

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    instruction_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    instruction_mask[:, 5:15] = True

    output = wrapper(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        instruction_mask=instruction_mask,
    )

    print(f"  Logits shape: {output.logits.shape}")
    print(f"  Loss: {output.loss.item():.4f}")
    print(f"  Routing weights shape: {output.routing_weights.shape}")
    print("  PASSED")

    # Test 3: Gradient flow
    print("\nTest 3: Gradient flow (the critical test)...")
    wrapper.train()
    optimizer = torch.optim.Adam(wrapper.get_trainable_parameters(), lr=1e-4)
    optimizer.zero_grad()

    output = wrapper(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        instruction_mask=instruction_mask,
    )

    output.loss.backward()

    # Check router gradients
    router_grads = sum(
        1 for p in wrapper.router.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    print(f"  Router params with grad: {router_grads}")

    # Check LoRA gradients
    lora_grads = 0
    lora_total = 0
    for m in wrapper._moe_linear_modules:
        for lora in m.expert_loras:
            for p in lora.parameters():
                lora_total += 1
                if p.grad is not None and p.grad.abs().sum() > 0:
                    lora_grads += 1
    print(f"  LoRA params with grad: {lora_grads}/{lora_total}")

    assert lora_grads > 0, "FAILED: No LoRA gradients! Module replacement didn't fix gradient flow."
    assert router_grads > 0, "FAILED: No router gradients!"
    print("  PASSED — Gradients flow correctly ✓")

    optimizer.step()

    # Test 4: Save/Load
    print("\nTest 4: Save/Load checkpoint...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        wrapper.save_moe_checkpoint(tmpdir)

        # Verify files exist
        assert os.path.exists(os.path.join(tmpdir, 'router.pt'))
        assert os.path.exists(os.path.join(tmpdir, 'experts', 'expert_0', 'adapter_model.bin'))
        assert os.path.exists(os.path.join(tmpdir, 'moe_config.json'))

        # Create new wrapper and load
        base_model2 = MockBaseModel()
        wrapper2 = MoEVLMWrapper(base_model2, config)
        wrapper2.load_moe_checkpoint(tmpdir)

        # Verify router weights match
        for (n1, p1), (n2, p2) in zip(
            wrapper.router.named_parameters(),
            wrapper2.router.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Router mismatch in {n1}"

    print("  PASSED")

    print()
    print("=== All tests passed! Module replacement works correctly. ===")
