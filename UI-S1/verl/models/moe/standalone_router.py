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
Standalone Router for vLLM Multi-LoRA Serving.

Lightweight SigLIP-based router (~100M params) that predicts which expert LoRA
to use based solely on a screenshot. Independent of the 7B base model, enabling
fast per-step routing during vLLM inference.

This router is trained via distillation from the ContextAwareRouter's decisions
(see train_standalone_router.py).

Architecture:
    Screenshot (PIL.Image)
           │
           ▼
    ┌──────────────────┐
    │  SigLIP Vision   │  (~90M params, frozen or fine-tuned)
    │  Encoder         │
    └──────────────────┘
           │ pooler_output [vision_hidden]
           ▼
    ┌──────────────────┐
    │  vision_proj     │  vision_hidden → 256
    │  GELU            │
    │  classifier      │  256 → 128 → num_experts
    └──────────────────┘
           │
           ▼
    expert_index (int)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class StandaloneRouter(nn.Module):
    """
    Lightweight SigLIP-based router for vLLM serving.

    Predicts expert index from a screenshot. Independent of the base VLM.
    Designed for <10ms inference on GPU.

    Args:
        num_experts: Number of expert LoRAs to route to
        vision_model: SigLIP model name or path
        freeze_vision: Whether to freeze the vision encoder (default: True)
        proj_hidden: Projection hidden dimension (default: 256)
        classifier_hidden: Classifier hidden dimension (default: 128)

    Example:
        >>> router = StandaloneRouter(num_experts=6)
        >>> from PIL import Image
        >>> img = Image.open("screenshot.png")
        >>> expert_idx = router.predict(img)
        >>> print(f"Route to expert {expert_idx}")
    """

    def __init__(
        self,
        num_experts: int = 6,
        vision_model: str = "google/siglip-base-patch16-224",
        freeze_vision: bool = True,
        proj_hidden: int = 256,
        classifier_hidden: int = 128,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.vision_model_name = vision_model
        self.freeze_vision = freeze_vision

        # Lazy-load vision encoder (avoids import at module level)
        self.vision_encoder = None
        self.processor = None
        self._vision_hidden_size = None
        self._vision_model_name = vision_model

        # These will be initialized after vision encoder is loaded
        self.proj_hidden = proj_hidden
        self.classifier_hidden = classifier_hidden

        # Placeholder layers (will be re-initialized in _load_vision_encoder)
        self.vision_proj = None
        self.classifier = None

    def _load_vision_encoder(self):
        """Lazily load SigLIP vision encoder and processor."""
        if self.vision_encoder is not None:
            return

        try:
            from transformers import SiglipVisionModel, AutoImageProcessor
        except ImportError:
            from transformers import AutoModel, AutoImageProcessor
            logger.warning("SiglipVisionModel not available, using AutoModel")
            SiglipVisionModel = None

        if SiglipVisionModel is not None:
            self.vision_encoder = SiglipVisionModel.from_pretrained(
                self._vision_model_name
            )
        else:
            self.vision_encoder = AutoModel.from_pretrained(
                self._vision_model_name
            )

        self.processor = AutoImageProcessor.from_pretrained(
            self._vision_model_name
        )

        # Get vision hidden size
        if hasattr(self.vision_encoder.config, 'hidden_size'):
            self._vision_hidden_size = self.vision_encoder.config.hidden_size
        else:
            self._vision_hidden_size = 768  # SigLIP-base default

        # Freeze vision encoder if requested
        if self.freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            self.vision_encoder.eval()

        # Initialize projection and classifier layers
        self.vision_proj = nn.Linear(self._vision_hidden_size, self.proj_hidden)
        self.classifier = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.proj_hidden, self.classifier_hidden),
            nn.GELU(),
            nn.Linear(self.classifier_hidden, self.num_experts),
        )

        # Initialize with small weights
        nn.init.xavier_uniform_(self.vision_proj.weight, gain=0.1)
        nn.init.zeros_(self.vision_proj.bias)
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: pixel_values -> expert logits.

        Args:
            pixel_values: [B, C, H, W] Preprocessed image tensors

        Returns:
            logits: [B, num_experts] Expert routing logits
        """
        self._load_vision_encoder()

        # Get vision features
        if self.freeze_vision:
            with torch.no_grad():
                vision_output = self.vision_encoder(pixel_values=pixel_values)
        else:
            vision_output = self.vision_encoder(pixel_values=pixel_values)

        # Use pooler_output if available, else mean pool last_hidden_state
        if hasattr(vision_output, 'pooler_output') and vision_output.pooler_output is not None:
            features = vision_output.pooler_output  # [B, vision_hidden]
        else:
            features = vision_output.last_hidden_state.mean(dim=1)  # [B, vision_hidden]

        # Project and classify
        projected = self.vision_proj(features)      # [B, proj_hidden]
        logits = self.classifier(projected)          # [B, num_experts]

        return logits

    @torch.no_grad()
    def predict(self, image) -> int:
        """
        Predict expert index for a single screenshot.

        Args:
            image: PIL.Image or preprocessed pixel_values tensor

        Returns:
            expert_idx: int, selected expert index
        """
        self._load_vision_encoder()
        self.eval()

        if isinstance(image, torch.Tensor):
            pixel_values = image
            if pixel_values.dim() == 3:
                pixel_values = pixel_values.unsqueeze(0)
        else:
            # PIL Image
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"]

        # Move to same device as model
        device = next(self.vision_proj.parameters()).device
        pixel_values = pixel_values.to(device=device)

        logits = self.forward(pixel_values)
        return logits.argmax(dim=-1).item()

    @torch.no_grad()
    def predict_batch(self, images: list) -> list:
        """
        Predict expert indices for a batch of screenshots.

        Args:
            images: List of PIL.Image objects

        Returns:
            expert_indices: List of int
        """
        self._load_vision_encoder()
        self.eval()

        inputs = self.processor(images=images, return_tensors="pt")
        device = next(self.vision_proj.parameters()).device
        pixel_values = inputs["pixel_values"].to(device=device)

        logits = self.forward(pixel_values)
        return logits.argmax(dim=-1).tolist()

    @torch.no_grad()
    def predict_with_distribution(self, image) -> tuple:
        """
        Predict expert with full routing distribution.

        Args:
            image: PIL.Image or pixel_values tensor

        Returns:
            (expert_idx, routing_weights) tuple
        """
        self._load_vision_encoder()
        self.eval()

        if isinstance(image, torch.Tensor):
            pixel_values = image
            if pixel_values.dim() == 3:
                pixel_values = pixel_values.unsqueeze(0)
        else:
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"]

        device = next(self.vision_proj.parameters()).device
        pixel_values = pixel_values.to(device=device)

        logits = self.forward(pixel_values)
        weights = F.softmax(logits, dim=-1)
        expert_idx = logits.argmax(dim=-1).item()

        return expert_idx, weights[0].cpu()

    def save(self, save_dir: str):
        """
        Save standalone router checkpoint.

        Saves:
        - router_head.pt: vision_proj + classifier state dict
        - config.json: Router configuration

        Args:
            save_dir: Directory to save checkpoint
        """
        self._load_vision_encoder()
        os.makedirs(save_dir, exist_ok=True)

        # Save router head (not the vision encoder — it can be loaded from HF)
        head_state = {
            'vision_proj': self.vision_proj.state_dict(),
            'classifier': self.classifier.state_dict(),
        }
        torch.save(head_state, os.path.join(save_dir, 'router_head.pt'))

        # Save config
        config = {
            'num_experts': self.num_experts,
            'vision_model': self._vision_model_name,
            'freeze_vision': self.freeze_vision,
            'proj_hidden': self.proj_hidden,
            'classifier_hidden': self.classifier_hidden,
            'vision_hidden_size': self._vision_hidden_size,
        }
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved standalone router to {save_dir}")

    @classmethod
    def load(cls, load_dir: str, device: str = 'cpu') -> "StandaloneRouter":
        """
        Load standalone router from checkpoint.

        Args:
            load_dir: Directory containing checkpoint
            device: Device to load to

        Returns:
            StandaloneRouter instance
        """
        config_path = os.path.join(load_dir, 'config.json')
        with open(config_path) as f:
            config = json.load(f)

        router = cls(
            num_experts=config['num_experts'],
            vision_model=config['vision_model'],
            freeze_vision=config.get('freeze_vision', True),
            proj_hidden=config.get('proj_hidden', 256),
            classifier_hidden=config.get('classifier_hidden', 128),
        )

        # Initialize vision encoder + head layers
        router._load_vision_encoder()

        # Load router head weights
        head_path = os.path.join(load_dir, 'router_head.pt')
        head_state = torch.load(head_path, map_location=device)
        router.vision_proj.load_state_dict(head_state['vision_proj'])
        router.classifier.load_state_dict(head_state['classifier'])

        router.to(device)
        logger.info(f"Loaded standalone router from {load_dir}")

        return router
