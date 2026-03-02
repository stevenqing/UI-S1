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
VLM-based eigenfunction approximation for option discovery on GUI-360.

Replaces the 43-dim MLP (Task 3) with Qwen2.5-VL-7B + LoRA + regression head,
training directly on screenshots (+ optional a11y tree text).

Task 3.1 in the Option-Incentivized MoE pipeline.

Supports two input modes:
  1. "screenshot" — image-only with fixed prompt
  2. "screenshot_a11y" — image + compressed a11y tree text as prompt

References:
  - Wu et al. (2019), "The Laplacian in RL", ICLR 2019.
  - Jinnai et al. (2020), "Exploration in RL with Deep Covering Options", ICLR 2020.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class VLMEigenfunctionConfig:
    """Configuration for VLM-based eigenfunction training."""
    # Model
    model_path: str = "checkpoints/Qwen2.5-VL-7B-Instruct"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Input mode: "screenshot" or "screenshot_a11y"
    input_mode: str = "screenshot_a11y"
    image_size: int = 448  # Fixed resolution → ~256 visual tokens

    # Eigenfunction loss
    eta: float = 1.0  # Weight of repulsive term

    # Training
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 5
    gradient_checkpointing: bool = True
    bf16: bool = True

    # Bottleneck identification
    percentile_k: float = 30.0

    # Logging & checkpointing
    log_every: int = 10  # Log every N batches
    save_every: int = 1  # Save every N epochs

    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "input_mode": self.input_mode,
            "image_size": self.image_size,
            "eta": self.eta,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "num_epochs": self.num_epochs,
            "gradient_checkpointing": self.gradient_checkpointing,
            "bf16": self.bf16,
            "percentile_k": self.percentile_k,
        }


# ============================================================================
# Regression Head
# ============================================================================

class RegressionHead(nn.Module):
    """Projects VLM hidden states to scalar f-value.

    LayerNorm(hidden_size) → Linear(hidden_size, 1)
    """

    def __init__(self, hidden_size: int = 3584):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map pooled hidden states to scalar f-values.

        Args:
            x: (batch_size, hidden_size) pooled VLM outputs.

        Returns:
            (batch_size, 1) f-values.
        """
        return self.linear(self.norm(x))


# ============================================================================
# VLM Eigenfunction Model
# ============================================================================

class VLMEigenfunctionModel(nn.Module):
    """Qwen2.5-VL + LoRA + RegressionHead for eigenfunction approximation.

    Wraps a pretrained VLM with LoRA adapters and a regression head that
    maps mean-pooled hidden states to scalar f-values.
    """

    def __init__(self, config: VLMEigenfunctionConfig):
        super().__init__()
        self.config = config
        self._build_model()

    def _build_model(self):
        """Load Qwen2.5-VL, apply LoRA, and add regression head."""
        from transformers import Qwen2_5_VLForConditionalGeneration
        from peft import LoraConfig, get_peft_model

        logger.info(f"Loading base model from {self.config.model_path}")

        # Load kwargs
        load_kwargs = {
            "dtype": torch.bfloat16 if self.config.bf16 else torch.float32,
        }

        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model_path, **load_kwargs
        )

        # Freeze all parameters before applying LoRA
        for param in self.vlm.parameters():
            param.requires_grad = False

        # Apply LoRA
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.vlm = get_peft_model(self.vlm, lora_config)

        # Enable gradient checkpointing AFTER LoRA so inputs have requires_grad=True
        if self.config.gradient_checkpointing:
            self.vlm.enable_input_require_grads()
            self.vlm.gradient_checkpointing_enable()

        # Get hidden size from model config
        hidden_size = self.vlm.config.hidden_size
        logger.info(f"VLM hidden size: {hidden_size}")

        # Add regression head (always trainable, match VLM dtype)
        dtype = torch.bfloat16 if self.config.bf16 else torch.float32
        self.regression_head = RegressionHead(hidden_size).to(dtype)

        # Log parameter counts
        lora_params = sum(
            p.numel() for p in self.vlm.parameters() if p.requires_grad
        )
        head_params = sum(p.numel() for p in self.regression_head.parameters())
        total_params = sum(p.numel() for p in self.vlm.parameters())
        logger.info(
            f"Parameters: LoRA={lora_params:,} ({lora_params/total_params*100:.2f}%), "
            f"Head={head_params:,}, Total base={total_params:,}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: VLM → mean pool → regression head → scalar f-values.

        Args:
            input_ids: (B, seq_len) token IDs.
            attention_mask: (B, seq_len) attention mask.
            pixel_values: (N, C, H, W) pixel values for visual inputs.
            image_grid_thw: (N, 3) grid sizes for visual inputs.

        Returns:
            (B, 1) scalar f-values.
        """
        # Run VLM forward without LM head
        outputs = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
        )

        # Get last hidden states: (B, seq_len, hidden_size)
        hidden_states = outputs.hidden_states[-1]

        # Mean-pool over sequence (respecting attention mask)
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)  # (B, seq_len, 1)
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        # Regression head → (B, 1)
        f_values = self.regression_head(pooled)
        return f_values

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Return list of trainable parameters (LoRA + regression head)."""
        params = [p for p in self.vlm.parameters() if p.requires_grad]
        params.extend(self.regression_head.parameters())
        return params

    def save_adapter(self, output_dir: str | Path):
        """Save LoRA adapter and regression head separately."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter
        self.vlm.save_pretrained(output_dir / "lora_adapter")

        # Save regression head
        torch.save(
            self.regression_head.state_dict(),
            output_dir / "regression_head.pt",
        )
        logger.info(f"Saved adapter and head to {output_dir}")

    @classmethod
    def load_adapter(
        cls,
        adapter_dir: str | Path,
        config: VLMEigenfunctionConfig,
        device: str = "cpu",
    ) -> "VLMEigenfunctionModel":
        """Load a trained model from adapter directory."""
        from peft import PeftModel
        from transformers import Qwen2_5_VLForConditionalGeneration

        adapter_dir = Path(adapter_dir)
        model = cls.__new__(cls)
        nn.Module.__init__(model)
        model.config = config

        dtype = torch.bfloat16 if config.bf16 else torch.float32
        base_vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_path, dtype=dtype
        )
        model.vlm = PeftModel.from_pretrained(
            base_vlm, adapter_dir / "lora_adapter"
        )

        hidden_size = model.vlm.config.hidden_size
        model.regression_head = RegressionHead(hidden_size).to(dtype)
        model.regression_head.load_state_dict(
            torch.load(adapter_dir / "regression_head.pt", map_location=device, weights_only=True)
        )

        model.to(device)
        model.eval()
        return model


# ============================================================================
# Datasets
# ============================================================================

class VLMTransitionDataset(Dataset):
    """Dataset of (src_hash, dst_hash) transition pairs.

    Loads from transition_pairs.json produced by prepare_vlm_eigenfunction_data.py.
    """

    def __init__(self, transitions_path: str | Path):
        with open(transitions_path) as f:
            self.transitions = json.load(f)
        logger.info(f"Loaded {len(self.transitions)} transition pairs")

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx) -> tuple[str, str]:
        t = self.transitions[idx]
        return t["src_hash"], t["dst_hash"]


class VLMStateDataset(Dataset):
    """Dataset of all unique states for random sampling in repulsive term."""

    def __init__(self, manifest_path: str | Path):
        with open(manifest_path) as f:
            self.states = json.load(f)
        self.hashes = [s["hash"] for s in self.states]
        logger.info(f"Loaded {len(self.states)} unique states")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx) -> str:
        return self.hashes[idx]

    def sample(self, n: int) -> list[str]:
        """Sample n random state hashes."""
        return random.choices(self.hashes, k=n)


# ============================================================================
# Collator — builds VLM inputs from state hashes
# ============================================================================

SCREENSHOT_PROMPT = "Describe this UI state."

A11Y_PREFIX = "UI state: "


def compress_a11y_tree(a11y_text: str, max_chars: int = 200) -> str:
    """Compress a11y text to fit within max_chars.

    Already-compressed format from prepare script:
      Window: Excel | Tab: Data,Insert,... | Dialog: Format Cells | Controls: Button:3,Edit:1,...

    If longer, truncate intelligently.
    """
    if len(a11y_text) <= max_chars:
        return a11y_text
    return a11y_text[:max_chars - 3] + "..."


class VLMCollator:
    """Collates state hashes into VLM-ready batches.

    For each state hash, loads the screenshot and (optionally) a11y text,
    then processes through the Qwen2.5-VL processor.

    Args:
        manifest: Dict mapping hash → state info (screenshot_path, a11y_text).
        processor: Qwen2.5-VL processor (AutoProcessor).
        input_mode: "screenshot" or "screenshot_a11y".
        image_size: Target image resolution.
    """

    def __init__(
        self,
        manifest: dict[str, dict],
        processor,
        input_mode: str = "screenshot_a11y",
        image_size: int = 448,
    ):
        self.manifest = manifest
        self.processor = processor
        self.input_mode = input_mode
        self.image_size = image_size

    def __call__(self, hashes: list[str]) -> dict:
        """Build a VLM batch from a list of state hashes.

        Returns dict with: input_ids, attention_mask, pixel_values, image_grid_thw
        """
        from PIL import Image

        messages_batch = []
        images_batch = []

        for h in hashes:
            state_info = self.manifest[h]
            img_path = state_info["screenshot_path"]

            # Load and resize image
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
            images_batch.append(img)

            # Build prompt
            if self.input_mode == "screenshot_a11y":
                a11y = state_info.get("a11y_text", "")
                a11y_compressed = compress_a11y_tree(a11y)
                text_prompt = A11Y_PREFIX + a11y_compressed
            else:
                text_prompt = SCREENSHOT_PROMPT

            # Build message in Qwen2.5-VL chat format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": text_prompt},
                    ],
                }
            ]
            messages_batch.append(messages)

        # Process all messages through the processor
        # Use apply_chat_template to format, then process
        texts = []
        all_images = []
        for msgs, img in zip(messages_batch, images_batch):
            text = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            all_images.append(img)

        batch = self.processor(
            text=texts,
            images=all_images,
            padding=True,
            return_tensors="pt",
        )

        return batch


# ============================================================================
# Loss Function
# ============================================================================

def vlm_eigenfunction_loss(
    f_src: torch.Tensor,
    f_dst: torch.Tensor,
    f_rand: torch.Tensor,
    eta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute eigenfunction loss on pre-computed f-values.

    Same formulation as graph_analysis.eigenfunction_loss but operates on
    pre-computed scalars rather than calling the network.

    G_tilde(f) = 0.5 * E[(f(s) - f(s'))^2]
                 + eta * [(E[f^2] - 1)^2 + (E[f])^2]

    Args:
        f_src: (B, 1) f-values for source states.
        f_dst: (B, 1) f-values for destination states.
        f_rand: (B, 1) f-values for random states.
        eta: Weight of repulsive term.

    Returns:
        (loss, metrics_dict)
    """
    # Smoothness: connected states should have similar f values
    smoothness = 0.5 * ((f_src - f_dst) ** 2).mean()

    # Repulsive: enforce E[f^2] = 1 (normalization) and E[f] = 0 (orthogonality)
    norm_penalty = (f_rand.pow(2).mean() - 1).pow(2)
    ortho_penalty = f_rand.mean().pow(2)
    repulsive = norm_penalty + ortho_penalty

    loss = smoothness + eta * repulsive

    metrics = {
        "loss": loss.item(),
        "smoothness": smoothness.item(),
        "repulsive": repulsive.item(),
        "norm_penalty": norm_penalty.item(),
        "ortho_penalty": ortho_penalty.item(),
        "f_src_mean": f_src.mean().item(),
        "f_src_std": f_src.std().item(),
        "f_rand_mean": f_rand.mean().item(),
        "f_rand_std": f_rand.std().item(),
    }

    return loss, metrics
