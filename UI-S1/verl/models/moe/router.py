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
MoE Router Implementation for GUI Agent.

This module implements a text-only router that routes instructions to
appropriate expert LoRAs based on instruction text features.

Design Decisions:
- Text-Only Router: Only uses instruction text features (not screenshot)
  - Simpler and more interpretable
  - Validates hypothesis that instruction alone determines expert
- Top-k Hard Routing: Selects top-k experts for clearer specialization
- Load Balancing: Prevents all samples routing to single expert

Architecture:
    instruction_features [B, hidden_size]
                │
                ▼
        ┌───────────────┐
        │   Linear      │  hidden_size → router_hidden
        │   GELU        │
        │   Dropout     │
        │   Linear      │  router_hidden → num_experts
        └───────────────┘
                │
                ▼
        ┌───────────────┐
        │   Softmax     │
        └───────────────┘
                │
                ▼
        routing_weights [B, num_experts]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizer


@dataclass
class RouterOutput:
    """
    Router output containing routing weights and expert selection.

    Attributes:
        routing_weights: [B, num_experts] Softmax normalized routing probabilities
        top_k_weights: [B, top_k] Renormalized weights for selected experts
        top_k_indices: [B, top_k] Indices of selected experts
        router_logits: [B, num_experts] Raw router logits (for analysis/loss)
    """
    routing_weights: torch.Tensor
    top_k_weights: torch.Tensor
    top_k_indices: torch.Tensor
    router_logits: torch.Tensor

    def to(self, device: Union[str, torch.device]) -> "RouterOutput":
        """Move all tensors to specified device."""
        return RouterOutput(
            routing_weights=self.routing_weights.to(device),
            top_k_weights=self.top_k_weights.to(device),
            top_k_indices=self.top_k_indices.to(device),
            router_logits=self.router_logits.to(device),
        )

    def detach(self) -> "RouterOutput":
        """Detach all tensors from computation graph."""
        return RouterOutput(
            routing_weights=self.routing_weights.detach(),
            top_k_weights=self.top_k_weights.detach(),
            top_k_indices=self.top_k_indices.detach(),
            router_logits=self.router_logits.detach(),
        )


class TextOnlyRouter(nn.Module):
    """
    Text-only router for MoE GUI Agent.

    Routes instructions to appropriate experts based solely on instruction
    text features. This design choice is intentional:
    1. Validates that instruction type determines routing (not visual context)
    2. Makes routing decisions interpretable
    3. Simplifies the routing mechanism

    Args:
        hidden_size: Base model hidden dimension (e.g., 3584 for Qwen2.5-VL-7B)
        num_experts: Number of expert LoRAs (default: 4 for click/type/navigate/scroll)
        router_hidden: Hidden dimension of router MLP (default: 256)
        top_k: Number of experts to select per sample (default: 1)
        dropout: Dropout rate in router MLP (default: 0.1)
        temperature: Softmax temperature - lower means sharper distribution (default: 1.0)
        noise_std: Standard deviation of noise added during training (default: 0.0)

    Example:
        >>> router = TextOnlyRouter(hidden_size=3584, num_experts=4)
        >>> features = torch.randn(8, 3584)  # batch of 8
        >>> output = router(features)
        >>> print(output.top_k_indices)  # [8, 1] - selected expert per sample
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 4,
        router_hidden: int = 256,
        top_k: int = 1,
        dropout: float = 0.1,
        temperature: float = 1.0,
        noise_std: float = 0.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        self.noise_std = noise_std

        # Validate top_k
        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})"
            )

        # Router MLP: hidden_size -> router_hidden -> num_experts
        self.router = nn.Sequential(
            nn.Linear(hidden_size, router_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden, num_experts),
        )

        # Initialize with small weights for uniform initial distribution
        self._init_weights()

    def _init_weights(self):
        """
        Initialize router weights for near-uniform initial routing.

        Uses Xavier initialization with small gain to ensure:
        - Initial routing is approximately uniform
        - Gradients can flow to learn meaningful routing
        """
        for module in self.router:
            if isinstance(module, nn.Linear):
                # Small gain for near-uniform initial routing
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        instruction_features: torch.Tensor,
        return_all: bool = True,
    ) -> RouterOutput:
        """
        Compute routing weights and select top-k experts.

        Args:
            instruction_features: [B, hidden_size] Pooled instruction representations
            return_all: Whether to return full RouterOutput (default: True)

        Returns:
            RouterOutput containing:
            - routing_weights: [B, num_experts] Full routing distribution
            - top_k_weights: [B, top_k] Renormalized weights for selected experts
            - top_k_indices: [B, top_k] Selected expert indices
            - router_logits: [B, num_experts] Raw logits
        """
        batch_size = instruction_features.size(0)

        # Ensure dtype matches router weights
        router_dtype = next(self.router.parameters()).dtype
        if instruction_features.dtype != router_dtype:
            instruction_features = instruction_features.to(router_dtype)

        # Compute router logits
        router_logits = self.router(instruction_features)  # [B, num_experts]

        # Add noise during training for exploration (optional)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise

        # Apply temperature scaling
        scaled_logits = router_logits / self.temperature

        # Softmax to get routing probabilities
        routing_weights = F.softmax(scaled_logits, dim=-1)  # [B, num_experts]

        # Top-k selection
        top_k_weights, top_k_indices = routing_weights.topk(self.top_k, dim=-1)

        # Renormalize top-k weights to sum to 1
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-10)

        return RouterOutput(
            routing_weights=routing_weights,
            top_k_weights=top_k_weights,
            top_k_indices=top_k_indices,
            router_logits=router_logits,
        )

    def get_routing_distribution(
        self,
        instruction_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get full routing distribution without top-k selection.

        Useful for analysis and visualization.

        Args:
            instruction_features: [B, hidden_size]

        Returns:
            routing_weights: [B, num_experts] Softmax probabilities
        """
        with torch.no_grad():
            router_logits = self.router(instruction_features)
            return F.softmax(router_logits / self.temperature, dim=-1)

    def get_hard_routing(
        self,
        instruction_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get hard routing decision (argmax).

        Args:
            instruction_features: [B, hidden_size]

        Returns:
            expert_indices: [B] Index of selected expert for each sample
        """
        with torch.no_grad():
            router_logits = self.router(instruction_features)
            return router_logits.argmax(dim=-1)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f"temperature={self.temperature}"
        )


class InstructionFeatureExtractor(nn.Module):
    """
    Extract instruction features from VLM hidden states.

    Strategies:
    - 'mean': Mean pooling over instruction tokens
    - 'last': Use last instruction token (similar to CLS)
    - 'first': Use first instruction token
    - 'max': Max pooling over instruction tokens

    Args:
        pooling_strategy: Pooling strategy to use (default: 'mean')
        hidden_size: Optional projection dimension (if different from input)

    Example:
        >>> extractor = InstructionFeatureExtractor(pooling_strategy='mean')
        >>> hidden_states = torch.randn(8, 512, 3584)  # [B, seq_len, hidden]
        >>> instruction_mask = torch.zeros(8, 512, dtype=torch.bool)
        >>> instruction_mask[:, 100:150] = True  # instruction at positions 100-150
        >>> features = extractor(hidden_states, instruction_mask)
        >>> print(features.shape)  # [8, 3584]
    """

    VALID_STRATEGIES = {'mean', 'last', 'first', 'max'}

    def __init__(
        self,
        pooling_strategy: str = 'mean',
        hidden_size: Optional[int] = None,
        output_size: Optional[int] = None,
    ):
        super().__init__()

        if pooling_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"pooling_strategy must be one of {self.VALID_STRATEGIES}, "
                f"got {pooling_strategy}"
            )

        self.pooling_strategy = pooling_strategy

        # Optional projection layer
        self.projection = None
        if hidden_size is not None and output_size is not None:
            self.projection = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        instruction_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract instruction features from hidden states.

        Args:
            hidden_states: [B, seq_len, hidden_size] VLM hidden states
            instruction_mask: [B, seq_len] Boolean mask where True = instruction token

        Returns:
            instruction_features: [B, hidden_size] Pooled features
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        if self.pooling_strategy == 'mean':
            features = self._mean_pooling(hidden_states, instruction_mask)

        elif self.pooling_strategy == 'last':
            features = self._last_token_pooling(hidden_states, instruction_mask)

        elif self.pooling_strategy == 'first':
            features = self._first_token_pooling(hidden_states, instruction_mask)

        elif self.pooling_strategy == 'max':
            features = self._max_pooling(hidden_states, instruction_mask)

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        # Apply optional projection
        if self.projection is not None:
            features = self.projection(features)

        return features

    def _mean_pooling(
        self,
        hidden_states: torch.Tensor,
        instruction_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pooling over instruction tokens."""
        mask_expanded = instruction_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
        sum_features = (hidden_states * mask_expanded).sum(dim=1)  # [B, hidden_size]
        num_tokens = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]
        return sum_features / num_tokens

    def _last_token_pooling(
        self,
        hidden_states: torch.Tensor,
        instruction_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Use last instruction token."""
        batch_size = hidden_states.size(0)
        features = []

        for i in range(batch_size):
            instruction_indices = instruction_mask[i].nonzero(as_tuple=True)[0]
            if len(instruction_indices) > 0:
                last_idx = instruction_indices[-1]
                features.append(hidden_states[i, last_idx])
            else:
                # Fallback: use last token of sequence
                features.append(hidden_states[i, -1])

        return torch.stack(features, dim=0)

    def _first_token_pooling(
        self,
        hidden_states: torch.Tensor,
        instruction_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Use first instruction token."""
        batch_size = hidden_states.size(0)
        features = []

        for i in range(batch_size):
            instruction_indices = instruction_mask[i].nonzero(as_tuple=True)[0]
            if len(instruction_indices) > 0:
                first_idx = instruction_indices[0]
                features.append(hidden_states[i, first_idx])
            else:
                # Fallback: use first token of sequence
                features.append(hidden_states[i, 0])

        return torch.stack(features, dim=0)

    def _max_pooling(
        self,
        hidden_states: torch.Tensor,
        instruction_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Max pooling over instruction tokens."""
        # Set non-instruction tokens to -inf for max pooling
        mask_expanded = instruction_mask.unsqueeze(-1)  # [B, seq_len, 1]
        masked_hidden = hidden_states.masked_fill(~mask_expanded, float('-inf'))
        max_features, _ = masked_hidden.max(dim=1)  # [B, hidden_size]

        # Handle case where all tokens are masked
        all_masked = ~instruction_mask.any(dim=1, keepdim=True)  # [B, 1]
        if all_masked.any():
            fallback = hidden_states[:, -1]  # Use last token as fallback
            max_features = torch.where(all_masked.expand_as(max_features), fallback, max_features)

        return max_features


def create_instruction_mask(
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    instruction_start_token: Optional[str] = None,
    instruction_end_token: Optional[str] = None,
) -> torch.Tensor:
    """
    Create instruction mask marking which tokens belong to instruction text.

    For Qwen2.5-VL format:
        <|im_start|>user
        <|vision_start|><|image_pad|>...<|vision_end|>
        Click on the search button
        <|im_end|>

    We want to mark "Click on the search button" part.

    Args:
        input_ids: [B, seq_len] Token IDs
        tokenizer: Tokenizer with special token mappings
        instruction_start_token: Token marking instruction start (auto-detected if None)
        instruction_end_token: Token marking instruction end (auto-detected if None)

    Returns:
        instruction_mask: [B, seq_len] Boolean mask
    """
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    device = input_ids.device

    # Try to get special token IDs
    try:
        vision_end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
        im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')

        # Handle case where tokens don't exist
        if vision_end_id == tokenizer.unk_token_id:
            vision_end_id = None
        if im_end_id == tokenizer.unk_token_id:
            im_end_id = None

    except Exception:
        vision_end_id = None
        im_end_id = None

    for i in range(batch_size):
        ids = input_ids[i].tolist()

        # Find instruction region
        instruction_start = 0
        instruction_end = seq_len

        # Find vision_end position (instruction starts after this)
        if vision_end_id is not None:
            for j, token_id in enumerate(ids):
                if token_id == vision_end_id:
                    instruction_start = j + 1
                    break

        # Find im_end position (instruction ends before this)
        if im_end_id is not None:
            for j in range(instruction_start, len(ids)):
                if ids[j] == im_end_id:
                    instruction_end = j
                    break

        # Mark instruction region
        if instruction_start < instruction_end:
            mask[i, instruction_start:instruction_end] = True

    return mask


def create_instruction_mask_from_text(
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    instruction_texts: List[str],
) -> torch.Tensor:
    """
    Create instruction mask by matching instruction text tokens.

    More reliable than position-based detection for various formats.

    Args:
        input_ids: [B, seq_len] Token IDs
        tokenizer: Tokenizer
        instruction_texts: List of instruction strings for each sample

    Returns:
        instruction_mask: [B, seq_len] Boolean mask
    """
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for i in range(batch_size):
        if i >= len(instruction_texts) or not instruction_texts[i]:
            continue

        # Tokenize instruction
        instruction_ids = tokenizer.encode(
            instruction_texts[i],
            add_special_tokens=False,
        )

        if not instruction_ids:
            continue

        # Find instruction in input_ids using sliding window
        input_list = input_ids[i].tolist()
        instr_len = len(instruction_ids)

        for j in range(len(input_list) - instr_len + 1):
            if input_list[j:j + instr_len] == instruction_ids:
                mask[i, j:j + instr_len] = True
                break

    return mask


# Utility functions for routing analysis

def compute_routing_entropy(routing_weights: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy of routing distribution.

    Higher entropy = more uniform distribution.

    Args:
        routing_weights: [B, num_experts] or [num_experts]

    Returns:
        entropy: Scalar or [B] tensor
    """
    # Add small epsilon for numerical stability
    log_weights = torch.log(routing_weights + 1e-10)
    entropy = -(routing_weights * log_weights).sum(dim=-1)
    return entropy


def compute_routing_diversity(routing_weights: torch.Tensor) -> float:
    """
    Compute diversity of routing across batch.

    Returns ratio of unique expert selections to batch size.

    Args:
        routing_weights: [B, num_experts]

    Returns:
        diversity: float in [0, 1]
    """
    dominant_experts = routing_weights.argmax(dim=-1)
    unique_experts = len(dominant_experts.unique())
    return unique_experts / routing_weights.size(1)


class ContextAwareRouter(nn.Module):
    """
    Context-aware router using vision + text features for step-level routing.

    Unlike TextOnlyRouter which uses only instruction text, this router
    combines visual features (from screenshot tokens) with text features
    (instruction + action history) to make per-step routing decisions.

    This enables different routing for different steps in the same trajectory,
    since each step has a different screenshot and accumulated history.

    Coexists with TextOnlyRouter; selected via MoEConfig.router_type.

    Args:
        hidden_size: Base model hidden dimension (e.g., 3584 for Qwen2.5-VL-7B)
        num_experts: Number of expert LoRAs
        router_hidden: Hidden dimension of router MLP (default: 256)
        top_k: Number of experts to select per sample (default: 1)
        dropout: Dropout rate in router MLP (default: 0.1)
        temperature: Softmax temperature (default: 1.0)
        noise_std: Training noise standard deviation (default: 0.0)
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 4,
        router_hidden: int = 256,
        top_k: int = 1,
        dropout: float = 0.1,
        temperature: float = 1.0,
        noise_std: float = 0.0,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        self.noise_std = noise_std

        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) cannot be greater than num_experts ({num_experts})"
            )

        # Separate projections for vision and text features
        self.vision_proj = nn.Linear(hidden_size, router_hidden)
        self.text_proj = nn.Linear(hidden_size, router_hidden)

        # Router MLP: concatenated features -> expert logits
        self.router = nn.Sequential(
            nn.Linear(router_hidden * 2, router_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden, num_experts),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights for near-uniform initial routing."""
        for module in [self.vision_proj, self.text_proj]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        for module in self.router:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        return_all: bool = True,
    ) -> RouterOutput:
        """
        Compute routing from vision + text features.

        Args:
            vision_features: [B, hidden_size] Pooled vision token representations
            text_features: [B, hidden_size] Pooled text token representations
            return_all: Whether to return full RouterOutput (default: True)

        Returns:
            RouterOutput (same interface as TextOnlyRouter)
        """
        # Ensure dtype matches router weights
        router_dtype = self.vision_proj.weight.dtype
        if vision_features.dtype != router_dtype:
            vision_features = vision_features.to(router_dtype)
        if text_features.dtype != router_dtype:
            text_features = text_features.to(router_dtype)

        # Project vision and text features
        v = self.vision_proj(vision_features)   # [B, router_hidden]
        t = self.text_proj(text_features)       # [B, router_hidden]

        # Combine and route
        combined = torch.cat([v, t], dim=-1)    # [B, router_hidden * 2]
        router_logits = self.router(combined)   # [B, num_experts]

        # Add noise during training for exploration
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise

        # Apply temperature scaling
        scaled_logits = router_logits / self.temperature

        # Softmax to get routing probabilities
        routing_weights = F.softmax(scaled_logits, dim=-1)  # [B, num_experts]

        # Top-k selection
        top_k_weights, top_k_indices = routing_weights.topk(self.top_k, dim=-1)

        # Renormalize top-k weights
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-10)

        return RouterOutput(
            routing_weights=routing_weights,
            top_k_weights=top_k_weights,
            top_k_indices=top_k_indices,
            router_logits=router_logits,
        )

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f"temperature={self.temperature}"
        )


def create_vision_mask(
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
) -> torch.Tensor:
    """
    Create mask marking vision tokens (<|vision_start|> to <|vision_end|>).

    For Qwen2.5-VL format:
        <|vision_start|><|image_pad|>...<|vision_end|>

    Marks all tokens between (inclusive) vision_start and vision_end.

    Args:
        input_ids: [B, seq_len] Token IDs
        tokenizer: Tokenizer with special token mappings

    Returns:
        vision_mask: [B, seq_len] Boolean mask
    """
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)

    try:
        vision_start_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
        vision_end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')

        if vision_start_id == tokenizer.unk_token_id:
            vision_start_id = None
        if vision_end_id == tokenizer.unk_token_id:
            vision_end_id = None
    except Exception:
        vision_start_id = None
        vision_end_id = None

    if vision_start_id is None or vision_end_id is None:
        return mask

    for i in range(batch_size):
        ids = input_ids[i].tolist()
        in_vision = False
        for j, token_id in enumerate(ids):
            if token_id == vision_start_id:
                in_vision = True
            if in_vision:
                mask[i, j] = True
            if token_id == vision_end_id:
                in_vision = False

    return mask


def create_text_context_mask(
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
) -> torch.Tensor:
    """
    Create mask for all text tokens after <|vision_end|> until <|im_end|>.

    Covers instruction text + action history for the current turn.
    This differs from create_instruction_mask in that it captures ALL text
    context (not just instruction), enabling the router to consider
    accumulated action history.

    Args:
        input_ids: [B, seq_len] Token IDs
        tokenizer: Tokenizer with special token mappings

    Returns:
        text_context_mask: [B, seq_len] Boolean mask
    """
    batch_size, seq_len = input_ids.shape
    mask = torch.zeros_like(input_ids, dtype=torch.bool)

    try:
        vision_end_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
        im_end_id = tokenizer.convert_tokens_to_ids('<|im_end|>')

        if vision_end_id == tokenizer.unk_token_id:
            vision_end_id = None
        if im_end_id == tokenizer.unk_token_id:
            im_end_id = None
    except Exception:
        vision_end_id = None
        im_end_id = None

    for i in range(batch_size):
        ids = input_ids[i].tolist()

        # Find last vision_end (for multi-turn, use the last one)
        text_start = 0
        if vision_end_id is not None:
            for j in range(len(ids) - 1, -1, -1):
                if ids[j] == vision_end_id:
                    text_start = j + 1
                    break

        # Find im_end after text_start
        text_end = seq_len
        if im_end_id is not None:
            for j in range(text_start, len(ids)):
                if ids[j] == im_end_id:
                    text_end = j
                    break

        if text_start < text_end:
            mask[i, text_start:text_end] = True

    return mask


if __name__ == "__main__":
    # Quick test
    print("Testing TextOnlyRouter...")

    batch_size = 8
    hidden_size = 3584
    num_experts = 4

    router = TextOnlyRouter(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=1,
    )

    # Test forward pass
    features = torch.randn(batch_size, hidden_size)
    output = router(features)

    print(f"  routing_weights shape: {output.routing_weights.shape}")
    print(f"  top_k_weights shape: {output.top_k_weights.shape}")
    print(f"  top_k_indices shape: {output.top_k_indices.shape}")
    print(f"  routing_weights sum: {output.routing_weights.sum(dim=-1)}")

    # Verify outputs
    assert output.routing_weights.shape == (batch_size, num_experts)
    assert output.top_k_weights.shape == (batch_size, 1)
    assert output.top_k_indices.shape == (batch_size, 1)
    assert torch.allclose(output.routing_weights.sum(dim=-1), torch.ones(batch_size))

    print("  All tests passed!")

    # Test InstructionFeatureExtractor
    print("\nTesting InstructionFeatureExtractor...")

    extractor = InstructionFeatureExtractor(pooling_strategy='mean')

    hidden_states = torch.randn(batch_size, 512, hidden_size)
    instruction_mask = torch.zeros(batch_size, 512, dtype=torch.bool)
    instruction_mask[:, 100:150] = True  # Instruction at positions 100-150

    extracted = extractor(hidden_states, instruction_mask)
    print(f"  extracted shape: {extracted.shape}")
    assert extracted.shape == (batch_size, hidden_size)

    print("  All tests passed!")

    print("\nRouter module ready for use!")
