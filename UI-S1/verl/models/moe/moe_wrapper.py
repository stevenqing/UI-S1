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
It integrates:
1. Frozen base VLM for encoding
2. Text-only router for expert selection
3. Expert LoRA collection for specialization
4. MoE loss computation

Architecture:
    Input: (screenshot, instruction)
              │
              ▼
    ┌─────────────────────────────────────────┐
    │         Base VLM (Frozen)                │
    │   [Vision Encoder + Text Encoder]        │
    │              │                           │
    │              ▼                           │
    │       hidden_states                      │
    └─────────────────────────────────────────┘
              │                    │
              ▼                    ▼
    ┌─────────────────┐    ┌─────────────────┐
    │     Router      │    │  Expert LoRAs   │
    │ (instruction    │───▶│ (weighted by    │
    │  features)      │    │  routing)       │
    └─────────────────┘    └─────────────────┘
                                   │
                                   ▼
                            ┌─────────────┐
                            │   LM Head   │
                            └─────────────┘
                                   │
                                   ▼
                             Action Output
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from verl.models.moe.router import (
    TextOnlyRouter,
    RouterOutput,
    InstructionFeatureExtractor,
    create_instruction_mask,
)
from verl.models.moe.expert_lora import (
    ExpertLoRACollection,
    ExpertLoRAConfig,
    MoEExpertApplier,
)
from verl.models.moe.moe_loss import MoELoss, MoELossOutput


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

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'num_experts': self.num_experts,
            'top_k': self.top_k,
            'expert_lora_r': self.expert_lora_r,
            'expert_lora_alpha': self.expert_lora_alpha,
            'expert_lora_dropout': self.expert_lora_dropout,
            'target_modules': self.target_modules,
            'router_hidden': self.router_hidden,
            'router_dropout': self.router_dropout,
            'router_temperature': self.router_temperature,
            'pooling_strategy': self.pooling_strategy,
            'balance_weight': self.balance_weight,
            'balance_type': self.balance_type,
            'z_loss_weight': self.z_loss_weight,
            'use_vectorized_routing': self.use_vectorized_routing,
        }

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
    MoE Wrapper for Vision-Language Model.

    Wraps a base VLM (frozen) with:
    - Text-only router for expert selection
    - Expert LoRA collection for parameter-efficient specialization
    - Hook-based LoRA injection

    The base model remains frozen. Only router and expert LoRAs are trained.

    Args:
        base_model: Pre-trained VLM (will be frozen)
        moe_config: MoE configuration
        tokenizer: Optional tokenizer for instruction mask creation

    Example:
        >>> from transformers import Qwen2VLForConditionalGeneration
        >>> base = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B")
        >>> config = MoEConfig(num_experts=4, expert_lora_r=16)
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

        # Initialize MoE components
        self._init_moe_components()

        # Track if hooks are registered
        self._hooks_registered = False

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

        # Ensure base model is in eval mode for consistent behavior
        # (but we'll still call train() on MoE components)
        self.base_model.eval()

    def _init_moe_components(self):
        """Initialize router, expert LoRAs, and applier."""
        config = self.moe_config

        # 1. Router
        self.router = TextOnlyRouter(
            hidden_size=self.hidden_size,
            num_experts=config.num_experts,
            router_hidden=config.router_hidden,
            top_k=config.top_k,
            dropout=config.router_dropout,
            temperature=config.router_temperature,
        )

        # 2. Expert LoRA Collection
        self.expert_collection = ExpertLoRACollection(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            num_experts=config.num_experts,
            target_modules=config.target_modules,
            lora_r=config.expert_lora_r,
            lora_alpha=config.expert_lora_alpha,
            lora_dropout=config.expert_lora_dropout,
        )

        # 3. Feature Extractor
        self.feature_extractor = InstructionFeatureExtractor(
            pooling_strategy=config.pooling_strategy,
        )

        # 4. Expert Applier (manages hooks)
        self.expert_applier = MoEExpertApplier(
            expert_collection=self.expert_collection,
            use_vectorized=config.use_vectorized_routing,
        )

        # 5. MoE Loss
        self.moe_loss = MoELoss(
            num_experts=config.num_experts,
            balance_weight=config.balance_weight,
            balance_type=config.balance_type,
            z_loss_weight=config.z_loss_weight,
        )

    def register_hooks(self):
        """Register forward hooks with base model layers."""
        if self._hooks_registered:
            return

        self.expert_applier.register_hooks(self.base_model)
        self._hooks_registered = True

    def remove_hooks(self):
        """Remove all registered hooks."""
        self.expert_applier.remove_hooks()
        self._hooks_registered = False

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

        Steps:
        1. Get hidden states from base model (frozen)
        2. Extract instruction features
        3. Compute routing weights
        4. Apply expert LoRAs via hooks
        5. Compute loss if labels provided

        Args:
            input_ids: [B, seq_len] Input token IDs
            attention_mask: [B, seq_len] Attention mask
            pixel_values: Vision inputs (format depends on model)
            labels: [B, seq_len] Labels for LM loss
            instruction_mask: [B, seq_len] Boolean mask for instruction tokens
            instruction_texts: List of instruction strings (alternative to mask)
            return_routing_info: Whether to return routing details
            output_hidden_states: Whether to return all hidden states

        Returns:
            MoEOutput with logits, loss, and routing information
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Ensure hooks are registered
        if not self._hooks_registered:
            self.register_hooks()

        # Step 1: Get hidden states from base model (first pass, no LoRA)
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
        else:
            # Fallback if hidden_states not available
            hidden_states = base_outputs.last_hidden_state if hasattr(base_outputs, 'last_hidden_state') else None
            if hidden_states is None:
                raise ValueError("Cannot get hidden states from base model")

        # Step 2: Create instruction mask if not provided
        if instruction_mask is None:
            if instruction_texts is not None and self.tokenizer is not None:
                from verl.models.moe.router import create_instruction_mask_from_text
                instruction_mask = create_instruction_mask_from_text(
                    input_ids, self.tokenizer, instruction_texts
                )
            elif self.tokenizer is not None:
                instruction_mask = create_instruction_mask(input_ids, self.tokenizer)
            else:
                # Fallback: use all tokens
                instruction_mask = torch.ones_like(input_ids, dtype=torch.bool)

        # Step 3: Extract instruction features
        instruction_features = self.feature_extractor(hidden_states, instruction_mask)

        # Step 4: Compute routing
        router_output = self.router(instruction_features)

        # Step 5: Set routing for expert applier
        self.expert_applier.set_routing(
            top_k_indices=router_output.top_k_indices,
            top_k_weights=router_output.top_k_weights,
            routing_weights=router_output.routing_weights if self.moe_config.use_vectorized_routing else None,
        )

        # Step 6: Forward with expert LoRAs (second pass)
        # This time, the hooks will apply LoRA deltas
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        # Step 7: Clear routing state
        self.expert_applier.clear_routing()

        # Step 8: Compute MoE loss if labels provided
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

        # Build output
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

        Routes to experts first, then generates using selected experts.

        Args:
            input_ids: [B, seq_len] Input token IDs
            attention_mask: [B, seq_len] Attention mask
            pixel_values: Vision inputs
            image_grid_thw: Image grid (temporal, height, width) for Qwen2.5-VL
            instruction_mask: [B, seq_len] Instruction token mask
            instruction_texts: Alternative to instruction_mask
            max_new_tokens: Maximum tokens to generate
            **generate_kwargs: Additional generation arguments

        Returns:
            generated_ids: [B, seq_len + generated_len]
            router_output: Routing information
        """
        batch_size = input_ids.size(0)

        # Ensure hooks are registered
        if not self._hooks_registered:
            self.register_hooks()

        # Get hidden states for routing
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

        # Create instruction mask if needed
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

        # Extract features and compute routing
        instruction_features = self.feature_extractor(hidden_states, instruction_mask)
        router_output = self.router(instruction_features)

        # Set routing for generation
        self.expert_applier.set_routing(
            top_k_indices=router_output.top_k_indices,
            top_k_weights=router_output.top_k_weights,
            routing_weights=router_output.routing_weights if self.moe_config.use_vectorized_routing else None,
        )

        # Generate with expert LoRAs
        generated_ids = self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

        # Clear routing
        self.expert_applier.clear_routing()

        return generated_ids, router_output

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (router + expert LoRAs)."""
        params = []
        params.extend(list(self.router.parameters()))
        params.extend(list(self.expert_collection.parameters()))
        return params

    def num_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_parameters())

    def num_total_parameters(self) -> int:
        """Get total number of parameters (including frozen)."""
        return sum(p.numel() for p in self.parameters())

    def train(self, mode: bool = True):
        """Set training mode (only affects MoE components, base model stays frozen)."""
        # Base model always in eval
        self.base_model.eval()

        # MoE components follow mode
        self.router.train(mode)
        self.expert_collection.train(mode)
        self.feature_extractor.train(mode)

        return self

    def eval(self):
        """Set evaluation mode."""
        return self.train(False)

    def save_moe_checkpoint(self, save_dir: str):
        """
        Save MoE components (router + experts).

        Creates:
        - router.pt: Router state dict
        - experts/: Directory with PEFT-format experts
        - moe_config.json: Configuration

        Args:
            save_dir: Directory to save checkpoint
        """
        import json

        os.makedirs(save_dir, exist_ok=True)

        # Save router
        torch.save(
            self.router.state_dict(),
            os.path.join(save_dir, 'router.pt')
        )

        # Save experts in PEFT format
        experts_dir = os.path.join(save_dir, 'experts')
        self.expert_collection.save_experts_separately(experts_dir, save_format='peft')

        # Save config
        config_path = os.path.join(save_dir, 'moe_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.moe_config.to_dict(), f, indent=2)

        print(f"Saved MoE checkpoint to {save_dir}")

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
            print(f"Loaded router from {router_path}")

        # Load experts
        experts_dir = os.path.join(load_dir, 'experts')
        if os.path.exists(experts_dir):
            self.expert_collection.load_experts_separately(experts_dir, load_format='peft')
            print(f"Loaded experts from {experts_dir}")

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

        # Expert utilization
        utilization = compute_expert_utilization(
            routing_weights, self.moe_config.num_experts
        )
        for i, u in enumerate(utilization.tolist()):
            stats[f'expert_{i}_utilization'] = u

        # Load balance coefficient
        stats['load_balance_coefficient'] = compute_load_balance_coefficient(utilization)

        # Routing entropy
        entropy = compute_routing_entropy(routing_weights)
        stats['routing_entropy_mean'] = entropy.mean().item()

        # Routing diversity
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

    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map=device if device != 'auto' else 'auto',
        trust_remote_code=True,
        **model_kwargs,
    )

    # Create config
    if moe_config is None:
        moe_config = MoEConfig()

    # Create wrapper
    wrapper = MoEVLMWrapper(
        base_model=model,
        moe_config=moe_config,
        tokenizer=tokenizer,
    )

    return wrapper


if __name__ == "__main__":
    print("Testing MoE VLM Wrapper (mock base model)...")
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
            # Mock hidden states
            hidden = torch.randn(batch_size, seq_len, 256)
            hidden_states = tuple([hidden] * 5)  # 4 layers + embedding
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
            # Mock generation
            batch_size = input_ids.size(0)
            new_tokens = torch.randint(0, 1000, (batch_size, 10))
            return torch.cat([input_ids, new_tokens], dim=1)

    # Create mock model
    base_model = MockBaseModel()

    # Create MoE config
    config = MoEConfig(
        num_experts=4,
        top_k=1,
        expert_lora_r=8,
        expert_lora_alpha=16,
        target_modules=['q_proj', 'v_proj'],
        balance_weight=0.1,
    )

    # Create wrapper
    print("Test 1: Creating MoE wrapper...")
    wrapper = MoEVLMWrapper(base_model, config)
    print(f"  Trainable params: {wrapper.num_trainable_parameters():,}")
    print(f"  Total params: {wrapper.num_total_parameters():,}")
    print("  PASSED")

    # Test forward pass
    print("\nTest 2: Forward pass...")
    batch_size = 4
    seq_len = 32

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    instruction_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    instruction_mask[:, 5:15] = True  # Instructions at positions 5-15

    output = wrapper(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        instruction_mask=instruction_mask,
    )

    print(f"  Logits shape: {output.logits.shape}")
    print(f"  Loss: {output.loss.item():.4f}")
    print(f"  LM Loss: {output.lm_loss.item():.4f}")
    print(f"  Balance Loss: {output.balance_loss.item():.6f}")
    print(f"  Routing weights shape: {output.routing_weights.shape}")
    print(f"  Top-k indices: {output.top_k_indices.squeeze().tolist()}")
    print("  PASSED")

    # Test gradient flow
    print("\nTest 3: Gradient flow...")
    wrapper.train()
    optimizer = torch.optim.Adam(wrapper.get_trainable_parameters(), lr=1e-4)

    output = wrapper(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        instruction_mask=instruction_mask,
    )

    output.loss.backward()
    optimizer.step()

    # Check gradients exist
    has_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                    for p in wrapper.get_trainable_parameters())
    assert has_grads, "No gradients computed"
    print("  PASSED")

    # Test routing statistics
    print("\nTest 4: Routing statistics...")
    stats = wrapper.get_routing_statistics(output.routing_weights)
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")
    print("  PASSED")

    # Test save/load
    print("\nTest 5: Save/Load checkpoint...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        wrapper.save_moe_checkpoint(tmpdir)

        # Create new wrapper and load
        wrapper2 = MoEVLMWrapper(MockBaseModel(), config)
        wrapper2.load_moe_checkpoint(tmpdir)

        # Check weights match
        for (n1, p1), (n2, p2) in zip(
            wrapper.router.named_parameters(),
            wrapper2.router.named_parameters()
        ):
            assert torch.allclose(p1, p2), f"Mismatch in {n1}"

    print("  PASSED")

    print()
    print("=== All tests passed! ===")
