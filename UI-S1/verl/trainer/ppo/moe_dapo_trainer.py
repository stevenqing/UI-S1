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
MoE-enhanced DAPO Trainer for GUI Agent.

This trainer extends RayTrajDAPOTrainer with MoE (Mixture of Experts) support:
1. Text-only Router for instruction-based expert selection
2. Expert LoRA collection for parameter-efficient specialization
3. Load balance loss to prevent expert collapse
4. MoE-specific metrics logging and checkpointing

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     MoERayTrajDAPOTrainer                            │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │   ┌─────────────────┐     ┌──────────────────────────────────────┐  │
    │   │   Base DAPO     │     │        MoE Components                 │  │
    │   │   Trainer       │     │                                       │  │
    │   │                 │     │  - Router: TextOnlyRouter             │  │
    │   │  - PPO/GRPO     │────▶│  - Experts: ExpertLoRACollection      │  │
    │   │  - Multi-round  │     │  - Loss: MoELoss                      │  │
    │   │  - Trajectory   │     │  - Applier: MoEExpertApplier          │  │
    │   └─────────────────┘     └──────────────────────────────────────┘  │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘

Usage:
    # In config YAML:
    model:
      moe:
        enabled: true
        num_experts: 4
        expert_lora_r: 16
        balance_weight: 0.1

    trainer:
      trainer_class: verl.trainer.ppo.moe_dapo_trainer.MoERayTrajDAPOTrainer
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from verl import DataProto
from verl.trainer.ppo.dapo_ray_trainer import RayTrajDAPOTrainer


@dataclass
class MoETrainerConfig:
    """Configuration for MoE trainer components."""

    # Core MoE settings
    enabled: bool = False
    num_experts: int = 4
    top_k: int = 1

    # Expert LoRA settings
    expert_lora_r: int = 16
    expert_lora_alpha: int = 32
    expert_lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ['q_proj', 'v_proj']
    )

    # Router settings
    router_hidden: int = 256
    router_dropout: float = 0.1
    router_temperature: float = 1.0

    # Feature extraction
    pooling_strategy: str = 'mean'

    # Loss settings
    balance_weight: float = 0.1
    balance_type: str = 'mse'  # 'mse', 'switch', 'entropy'
    z_loss_weight: float = 0.0

    # Training settings
    use_vectorized_routing: bool = False
    freeze_router_epochs: int = 0  # Optionally freeze router for initial epochs

    # Warm-start: path to converted SFT LoRA checkpoint for expert initialization
    moe_checkpoint: Optional[str] = None

    # Logging settings
    log_routing_matrix_freq: int = 100  # Log routing matrix every N steps
    log_expert_grads: bool = False

    @classmethod
    def from_config(cls, config: DictConfig) -> "MoETrainerConfig":
        """Create from OmegaConf config."""
        # MoE config is at actor_rollout_ref.model.moe
        moe_config = config.actor_rollout_ref.model.get('moe', {})
        if isinstance(moe_config, DictConfig):
            moe_config = OmegaConf.to_container(moe_config)

        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in moe_config.items() if k in valid_fields}

        return cls(**filtered_config)


class MoEMetricsTracker:
    """Tracks MoE-specific metrics across training."""

    def __init__(self, num_experts: int, instruction_types: Optional[List[str]] = None):
        self.num_experts = num_experts
        self.instruction_types = instruction_types or ['click', 'type', 'navigate', 'scroll']

        # Cumulative tracking
        self.total_samples = 0
        self.expert_counts = np.zeros(num_experts)
        self.routing_entropy_sum = 0.0

        # Per-instruction-type tracking
        self.type_to_expert_counts = defaultdict(lambda: np.zeros(num_experts))

        # History for visualization
        self.routing_history = []
        self.loss_history = []

    def update(
        self,
        routing_weights: torch.Tensor,
        instruction_types: Optional[List[str]] = None,
        balance_loss: Optional[float] = None,
    ):
        """Update metrics with new batch."""
        batch_size = routing_weights.size(0)
        self.total_samples += batch_size

        # Expert selection counts
        dominant_experts = routing_weights.argmax(dim=-1).cpu().numpy()
        for expert_idx in dominant_experts:
            self.expert_counts[expert_idx] += 1

        # Routing entropy
        entropy = -(routing_weights * torch.log(routing_weights + 1e-10)).sum(dim=-1)
        self.routing_entropy_sum += entropy.sum().item()

        # Per-type tracking
        if instruction_types is not None:
            for i, instr_type in enumerate(instruction_types):
                if i < batch_size:
                    expert_idx = dominant_experts[i]
                    self.type_to_expert_counts[instr_type][expert_idx] += 1

        # History
        self.routing_history.append({
            'routing_weights': routing_weights.detach().cpu().numpy(),
            'dominant_experts': dominant_experts,
        })

        if balance_loss is not None:
            self.loss_history.append(balance_loss)

    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        if self.total_samples == 0:
            return {}

        metrics = {}

        # Expert utilization
        utilization = self.expert_counts / self.total_samples
        for i, u in enumerate(utilization):
            metrics[f'moe/expert_{i}_utilization'] = u

        # Load balance coefficient
        ideal = 1.0 / self.num_experts
        variance = ((utilization - ideal) ** 2).mean()
        max_variance = (self.num_experts - 1) / (self.num_experts ** 2)
        if max_variance > 0:
            balance_coef = 1.0 - (variance / max_variance)
        else:
            balance_coef = 1.0
        metrics['moe/load_balance_coefficient'] = balance_coef

        # Average routing entropy
        metrics['moe/routing_entropy_mean'] = self.routing_entropy_sum / self.total_samples

        # Average balance loss
        if self.loss_history:
            metrics['moe/balance_loss_mean'] = np.mean(self.loss_history)

        return metrics

    def get_routing_matrix(self) -> np.ndarray:
        """Get routing matrix [num_instruction_types x num_experts]."""
        matrix = np.zeros((len(self.instruction_types), self.num_experts))

        for i, instr_type in enumerate(self.instruction_types):
            counts = self.type_to_expert_counts[instr_type]
            total = counts.sum()
            if total > 0:
                matrix[i] = counts / total

        return matrix

    def reset(self):
        """Reset all metrics."""
        self.total_samples = 0
        self.expert_counts = np.zeros(self.num_experts)
        self.routing_entropy_sum = 0.0
        self.type_to_expert_counts = defaultdict(lambda: np.zeros(self.num_experts))
        self.routing_history = []
        self.loss_history = []


class MoERayTrajDAPOTrainer(RayTrajDAPOTrainer):
    """
    MoE-enhanced DAPO Trainer for GUI Agent.

    Extends RayTrajDAPOTrainer with:
    - MoE routing based on instruction text
    - Expert LoRA application
    - Load balance loss
    - MoE-specific metrics and checkpointing

    The MoE components are managed on the driver process, while the
    actual model forward/backward is done on workers via Ray.

    Usage:
        trainer = MoERayTrajDAPOTrainer(config, tokenizer, ...)
        trainer.init_workers()
        trainer.fit()
    """

    def __init__(self, config: DictConfig, *args, **kwargs):
        # Extract MoE configuration before parent init
        self.moe_config = MoETrainerConfig.from_config(config)
        self.moe_enabled = self.moe_config.enabled

        if self.moe_enabled:
            print(f"[MoE] Initializing MoE trainer with {self.moe_config.num_experts} experts")
            self._validate_moe_config()

        # Initialize parent trainer
        super().__init__(config, *args, **kwargs)

        # Initialize MoE-specific components
        if self.moe_enabled:
            self._init_moe_components()

    def _validate_moe_config(self):
        """Validate MoE configuration."""
        if self.moe_config.num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {self.moe_config.num_experts}")

        if self.moe_config.top_k > self.moe_config.num_experts:
            raise ValueError(
                f"top_k ({self.moe_config.top_k}) cannot exceed "
                f"num_experts ({self.moe_config.num_experts})"
            )

        if self.moe_config.balance_weight < 0:
            raise ValueError(f"balance_weight must be >= 0, got {self.moe_config.balance_weight}")

    def _init_moe_components(self):
        """Initialize MoE components on driver process."""
        from verl.models.moe import (
            TextOnlyRouter,
            ExpertLoRACollection,
            MoEExpertApplier,
            InstructionFeatureExtractor,
            MoELoss,
        )

        # Get model hidden size from config
        # This will be set properly after model is loaded
        self._moe_hidden_size = None
        self._moe_num_layers = None

        # Lazy initialization - will be done when we know model dimensions
        self._router = None
        self._expert_collection = None
        self._expert_applier = None
        self._feature_extractor = None
        self._moe_loss_fn = None

        # Metrics tracker
        self._moe_metrics = MoEMetricsTracker(
            num_experts=self.moe_config.num_experts,
            instruction_types=['click', 'type', 'navigate', 'scroll'],
        )

        # Tracking
        self._moe_initialized = False
        self._current_epoch = 0

        print("[MoE] Components initialized (lazy)")

    def _lazy_init_moe(self, hidden_size: int, num_layers: int, device: str = 'cuda'):
        """Lazily initialize MoE components with model dimensions."""
        if self._moe_initialized:
            return

        from verl.models.moe import (
            TextOnlyRouter,
            ExpertLoRACollection,
            MoEExpertApplier,
            InstructionFeatureExtractor,
            MoELoss,
        )

        self._moe_hidden_size = hidden_size
        self._moe_num_layers = num_layers

        # Router
        self._router = TextOnlyRouter(
            hidden_size=hidden_size,
            num_experts=self.moe_config.num_experts,
            router_hidden=self.moe_config.router_hidden,
            top_k=self.moe_config.top_k,
            dropout=self.moe_config.router_dropout,
            temperature=self.moe_config.router_temperature,
        ).to(device)

        # Expert LoRA Collection
        self._expert_collection = ExpertLoRACollection(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_experts=self.moe_config.num_experts,
            target_modules=self.moe_config.target_modules,
            lora_r=self.moe_config.expert_lora_r,
            lora_alpha=self.moe_config.expert_lora_alpha,
            lora_dropout=self.moe_config.expert_lora_dropout,
        ).to(device)

        # Feature Extractor
        self._feature_extractor = InstructionFeatureExtractor(
            pooling_strategy=self.moe_config.pooling_strategy,
        ).to(device)

        # Expert Applier
        self._expert_applier = MoEExpertApplier(
            expert_collection=self._expert_collection,
            use_vectorized=self.moe_config.use_vectorized_routing,
        )

        # MoE Loss
        self._moe_loss_fn = MoELoss(
            num_experts=self.moe_config.num_experts,
            balance_weight=self.moe_config.balance_weight,
            balance_type=self.moe_config.balance_type,
            z_loss_weight=self.moe_config.z_loss_weight,
        )

        self._moe_initialized = True

        # Log parameter counts
        router_params = sum(p.numel() for p in self._router.parameters())
        expert_params = sum(p.numel() for p in self._expert_collection.parameters())
        print(f"[MoE] Initialized with hidden_size={hidden_size}, num_layers={num_layers}")
        print(f"[MoE] Router parameters: {router_params:,}")
        print(f"[MoE] Expert LoRA parameters: {expert_params:,}")
        print(f"[MoE] Total MoE parameters: {router_params + expert_params:,}")

        # Load warm-start checkpoint if configured
        if self.moe_config.moe_checkpoint:
            self._load_moe_warmstart(self.moe_config.moe_checkpoint)

    def _load_moe_warmstart(self, checkpoint_dir: str):
        """Load MoE warm-start weights from converted SFT checkpoint.

        This loads pre-trained expert LoRA weights and router into the
        already-initialized MoE components. The checkpoint is expected
        in the format produced by convert_sft_lora_to_moe.py.

        Args:
            checkpoint_dir: Path to warm-start checkpoint directory
        """
        if not os.path.exists(checkpoint_dir):
            print(f"[MoE] WARNING: Warm-start checkpoint not found: {checkpoint_dir}")
            return

        print(f"[MoE] Loading warm-start from {checkpoint_dir}")

        # Load router
        router_path = os.path.join(checkpoint_dir, 'router.pt')
        if os.path.exists(router_path):
            router_state = torch.load(router_path, map_location='cpu')
            self._router.load_state_dict(router_state)
            print(f"[MoE] Loaded router from warm-start")

        # Load expert weights
        experts_dir = os.path.join(checkpoint_dir, 'experts')
        if os.path.exists(experts_dir):
            self._expert_collection.load_experts_separately(experts_dir, load_format='peft')
            print(f"[MoE] Loaded {self.moe_config.num_experts} experts from warm-start")
        else:
            print(f"[MoE] WARNING: No experts directory at {experts_dir}")

        # Log source info
        config_path = os.path.join(checkpoint_dir, 'moe_config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                warmstart_config = json.load(f)
            print(f"[MoE] Warm-start source: {warmstart_config.get('source_checkpoint', 'unknown')}")
            print(f"[MoE] Warm-start perturbation: {warmstart_config.get('perturbation_std', 'unknown')}")

    def _compute_instruction_mask(
        self,
        batch: DataProto,
    ) -> torch.Tensor:
        """
        Compute instruction mask from batch.

        Identifies which tokens belong to the instruction (user query).
        """
        input_ids = batch.batch['input_ids']
        batch_size, seq_len = input_ids.shape

        # Try to get instruction mask from batch
        if 'instruction_mask' in batch.batch:
            return batch.batch['instruction_mask'].bool()

        # Create mask based on attention pattern
        # For Qwen VL format: instruction is between <|vision_end|> and <|im_end|>
        mask = torch.ones_like(input_ids, dtype=torch.bool)

        # If we have raw prompts, we can identify instruction tokens
        if 'raw_prompt_ids' in batch.non_tensor_batch:
            # Instruction tokens are at the end of the prompt
            for i in range(batch_size):
                prompt_len = len(batch.non_tensor_batch['raw_prompt_ids'][i])
                # Mark last 50 tokens of prompt as instruction (heuristic)
                instr_start = max(0, prompt_len - 50)
                mask[i, :instr_start] = False
                mask[i, prompt_len:] = False

        return mask

    def _extract_instruction_types(self, batch: DataProto) -> Optional[List[str]]:
        """Extract instruction types from batch for analysis."""
        if 'instruction_type' in batch.non_tensor_batch:
            return list(batch.non_tensor_batch['instruction_type'])

        # Try to infer from instruction text
        if 'instruction' in batch.non_tensor_batch:
            types = []
            for instr in batch.non_tensor_batch['instruction']:
                instr_lower = instr.lower() if isinstance(instr, str) else ''
                if 'click' in instr_lower or 'tap' in instr_lower:
                    types.append('click')
                elif 'type' in instr_lower or 'enter' in instr_lower or 'input' in instr_lower:
                    types.append('type')
                elif 'scroll' in instr_lower or 'swipe' in instr_lower:
                    types.append('scroll')
                elif 'navigate' in instr_lower or 'go to' in instr_lower or 'open' in instr_lower:
                    types.append('navigate')
                else:
                    types.append('unknown')
            return types

        return None

    def compute_moe_routing(
        self,
        hidden_states: torch.Tensor,
        instruction_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute MoE routing from hidden states.

        Args:
            hidden_states: [B, seq_len, hidden_size] Model hidden states
            instruction_mask: [B, seq_len] Boolean mask for instruction tokens

        Returns:
            Dict with routing_weights, top_k_indices, top_k_weights, router_logits
        """
        if not self.moe_enabled or not self._moe_initialized:
            return {}

        # Extract instruction features
        instruction_features = self._feature_extractor(hidden_states, instruction_mask)

        # Compute routing
        router_output = self._router(instruction_features)

        return {
            'routing_weights': router_output.routing_weights,
            'top_k_indices': router_output.top_k_indices,
            'top_k_weights': router_output.top_k_weights,
            'router_logits': router_output.router_logits,
        }

    def compute_moe_loss(
        self,
        lm_loss: torch.Tensor,
        routing_weights: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute MoE loss (LM loss + balance loss).

        Args:
            lm_loss: Language modeling loss
            routing_weights: [B, num_experts] Routing probabilities
            router_logits: [B, num_experts] Raw router logits

        Returns:
            Dict with total_loss, lm_loss, balance_loss, z_loss
        """
        if not self.moe_enabled or not self._moe_initialized:
            return {'total_loss': lm_loss}

        loss_output = self._moe_loss_fn(
            lm_loss=lm_loss,
            routing_weights=routing_weights,
            router_logits=router_logits,
        )

        return {
            'total_loss': loss_output.total_loss,
            'lm_loss': loss_output.lm_loss,
            'balance_loss': loss_output.balance_loss,
            'z_loss': loss_output.z_loss,
        }

    def _log_moe_metrics(self, metrics: Dict[str, Any], step: int):
        """Log MoE-specific metrics."""
        if not self.moe_enabled:
            return

        # Get accumulated metrics
        moe_metrics = self._moe_metrics.get_metrics()
        metrics.update(moe_metrics)

        # Log routing matrix periodically
        if step > 0 and step % self.moe_config.log_routing_matrix_freq == 0:
            routing_matrix = self._moe_metrics.get_routing_matrix()
            # Log as individual values
            for i, instr_type in enumerate(['click', 'type', 'navigate', 'scroll']):
                for j in range(self.moe_config.num_experts):
                    metrics[f'moe/routing_matrix/{instr_type}_to_expert_{j}'] = routing_matrix[i, j]

    def get_moe_trainable_params(self) -> List[nn.Parameter]:
        """Get list of MoE trainable parameters."""
        if not self.moe_enabled or not self._moe_initialized:
            return []

        params = []
        params.extend(list(self._router.parameters()))
        params.extend(list(self._expert_collection.parameters()))
        return params

    def save_moe_checkpoint(self, save_dir: str):
        """
        Save MoE components checkpoint.

        Saves:
        - router.pt: Router state dict
        - experts/: Expert LoRAs in PEFT format
        - moe_config.json: MoE configuration
        - moe_metrics.json: Training metrics
        """
        if not self.moe_enabled or not self._moe_initialized:
            return

        moe_dir = os.path.join(save_dir, 'moe')
        os.makedirs(moe_dir, exist_ok=True)

        # Save router
        router_path = os.path.join(moe_dir, 'router.pt')
        torch.save(self._router.state_dict(), router_path)

        # Save experts in PEFT format
        experts_dir = os.path.join(moe_dir, 'experts')
        self._expert_collection.save_experts_separately(experts_dir, save_format='peft')

        # Save config
        config_dict = {
            'num_experts': self.moe_config.num_experts,
            'top_k': self.moe_config.top_k,
            'expert_lora_r': self.moe_config.expert_lora_r,
            'expert_lora_alpha': self.moe_config.expert_lora_alpha,
            'target_modules': self.moe_config.target_modules,
            'router_hidden': self.moe_config.router_hidden,
            'hidden_size': self._moe_hidden_size,
            'num_layers': self._moe_num_layers,
        }
        config_path = os.path.join(moe_dir, 'moe_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Save metrics
        metrics_path = os.path.join(moe_dir, 'moe_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self._moe_metrics.get_metrics(), f, indent=2)

        print(f"[MoE] Saved checkpoint to {moe_dir}")

    def load_moe_checkpoint(self, load_dir: str):
        """Load MoE components from checkpoint."""
        if not self.moe_enabled:
            return

        moe_dir = os.path.join(load_dir, 'moe')
        if not os.path.exists(moe_dir):
            print(f"[MoE] No checkpoint found at {moe_dir}")
            return

        # Load config to get dimensions
        config_path = os.path.join(moe_dir, 'moe_config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                config_dict = json.load(f)

            # Lazy init if needed
            if not self._moe_initialized:
                self._lazy_init_moe(
                    hidden_size=config_dict['hidden_size'],
                    num_layers=config_dict['num_layers'],
                )

        # Load router
        router_path = os.path.join(moe_dir, 'router.pt')
        if os.path.exists(router_path):
            state_dict = torch.load(router_path, map_location='cpu')
            self._router.load_state_dict(state_dict)
            print(f"[MoE] Loaded router from {router_path}")

        # Load experts
        experts_dir = os.path.join(moe_dir, 'experts')
        if os.path.exists(experts_dir):
            self._expert_collection.load_experts_separately(experts_dir, load_format='peft')
            print(f"[MoE] Loaded experts from {experts_dir}")

    def _upload_moe_to_hf(self, moe_dir: str, step: int):
        """Upload MoE checkpoint to HuggingFace Hub.

        Reads from config:
          trainer.hf_repo_id: str  — e.g. "shuqing/ui-s1-moe-rl"
          trainer.hf_private: bool — default True
        """
        hf_repo = self.config.trainer.get('hf_repo_id', None)
        if not hf_repo:
            return

        try:
            from huggingface_hub import HfApi
            api = HfApi()

            private = self.config.trainer.get('hf_private', True)
            api.create_repo(repo_id=hf_repo, private=private, exist_ok=True)

            experiment = self.config.trainer.experiment_name
            path_in_repo = f"{experiment}/global_step_{step}/moe"
            api.upload_folder(
                folder_path=moe_dir,
                repo_id=hf_repo,
                path_in_repo=path_in_repo,
                repo_type="model",
            )
            print(f"[MoE] Uploaded checkpoint to HF: {hf_repo}/{path_in_repo}")
        except Exception as e:
            print(f"[MoE] WARNING: HF upload failed (training continues): {e}")

    def _save_checkpoint(self):
        """Override to include MoE checkpoint, HF upload, and cleanup old checkpoints."""
        # Call parent checkpoint saving
        super()._save_checkpoint()

        # Save MoE checkpoint
        if self.moe_enabled:
            checkpoint_dir = self.config.trainer.default_local_dir
            step_dir = os.path.join(checkpoint_dir, f'global_step_{self.global_steps}')
            self.save_moe_checkpoint(step_dir)

            # Upload MoE to HuggingFace (if hf_repo_id is set in config)
            moe_dir = os.path.join(step_dir, 'moe')
            self._upload_moe_to_hf(moe_dir, self.global_steps)

        # Clean up old global_step directories (actor + moe + data)
        max_ckpt = self.config.trainer.get("max_ckpt_to_keep", None)
        if max_ckpt and isinstance(max_ckpt, int) and max_ckpt > 0:
            import glob
            import shutil
            checkpoint_dir = self.config.trainer.default_local_dir
            step_dirs = sorted(glob.glob(os.path.join(checkpoint_dir, "global_step_*")),
                               key=lambda x: int(x.split("_")[-1]))
            if len(step_dirs) > max_ckpt:
                for old_dir in step_dirs[:-max_ckpt]:
                    print(f"[MoE] Removing old checkpoint: {old_dir}")
                    shutil.rmtree(old_dir, ignore_errors=True)

    def _load_checkpoint(self):
        """Override to include MoE checkpoint loading."""
        # Call parent checkpoint loading
        super()._load_checkpoint()

        # Load MoE checkpoint if exists
        if self.moe_enabled:
            checkpoint_dir = self.config.trainer.default_local_dir
            # Find latest checkpoint
            from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
            latest_path = find_latest_ckpt_path(checkpoint_dir)
            if latest_path:
                self.load_moe_checkpoint(latest_path)

    def fit(self):
        """
        The training loop with MoE support.

        Extends parent fit() to:
        - Initialize MoE components when model dimensions are known
        - Track MoE metrics
        - Log routing information
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # Log MoE config
        if self.moe_enabled:
            moe_config_dict = {
                'moe/num_experts': self.moe_config.num_experts,
                'moe/top_k': self.moe_config.top_k,
                'moe/expert_lora_r': self.moe_config.expert_lora_r,
                'moe/balance_weight': self.moe_config.balance_weight,
                'moe/balance_type': self.moe_config.balance_type,
            }
            logger.log(data=moe_config_dict, step=0)

        # Call parent fit
        # The actual MoE integration happens through the worker-side hooks
        # Here we just track metrics
        super().fit()

        # Final MoE metrics
        if self.moe_enabled and self._moe_initialized:
            final_metrics = self._moe_metrics.get_metrics()
            pprint(f"[MoE] Final routing metrics: {final_metrics}")


# Utility functions for MoE analysis

def analyze_routing_specialization(
    routing_matrix: np.ndarray,
    instruction_types: List[str],
) -> Dict[str, float]:
    """
    Analyze routing specialization from routing matrix.

    Args:
        routing_matrix: [num_types x num_experts] routing probabilities
        instruction_types: List of instruction type names

    Returns:
        Dict with specialization metrics
    """
    num_types, num_experts = routing_matrix.shape

    metrics = {}

    # Specialization score: average max probability per type
    max_probs = routing_matrix.max(axis=1)
    metrics['specialization_score'] = max_probs.mean()

    # Diversity: how many experts are used significantly (>0.1 probability)
    significant_experts = (routing_matrix > 0.1).sum(axis=1)
    metrics['avg_experts_per_type'] = significant_experts.mean()

    # Dominant expert per type
    dominant_experts = routing_matrix.argmax(axis=1)
    for i, instr_type in enumerate(instruction_types):
        metrics[f'{instr_type}_dominant_expert'] = int(dominant_experts[i])
        metrics[f'{instr_type}_dominant_prob'] = float(max_probs[i])

    # Expert overlap: do different types share experts?
    unique_dominant = len(set(dominant_experts))
    metrics['unique_dominant_experts'] = unique_dominant
    metrics['expert_overlap'] = 1.0 - (unique_dominant / min(num_types, num_experts))

    return metrics


def compute_expert_gradient_norms(
    expert_collection: nn.Module,
) -> Dict[str, float]:
    """
    Compute gradient norms for each expert.

    Useful for analyzing which experts are learning.
    """
    norms = {}

    for i, expert in enumerate(expert_collection.experts):
        total_norm = 0.0
        num_params = 0
        for p in expert.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
                num_params += 1
        if num_params > 0:
            norms[f'expert_{i}_grad_norm'] = (total_norm ** 0.5) / num_params

    return norms


if __name__ == "__main__":
    print("MoE DAPO Trainer module loaded.")
    print()
    print("Usage:")
    print("  1. Set moe.enabled=true in your config YAML")
    print("  2. Use trainer_class: verl.trainer.ppo.moe_dapo_trainer.MoERayTrajDAPOTrainer")
    print()
    print("Example config:")
    print("""
model:
  moe:
    enabled: true
    num_experts: 4
    top_k: 1
    expert_lora_r: 16
    expert_lora_alpha: 32
    target_modules: ['q_proj', 'v_proj']
    balance_weight: 0.1
    balance_type: mse
    """)
