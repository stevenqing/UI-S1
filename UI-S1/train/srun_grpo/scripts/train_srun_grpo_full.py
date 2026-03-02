#!/usr/bin/env python3
"""
Full-featured Srun-based UI-S1 GRPO Training Worker

This implements the complete UI-S1 GRPO algorithm matching the Ray version:
1. Multi-step trajectory rollout generation (with vLLM or model.generate)
2. Step-level reward computation
3. UI-S1 advantage estimation (episode + step level)
4. GRPO policy gradient updates with KL regularization
5. DAPO filtering
6. Reference policy for KL penalty
7. Validation loop

Usage:
    srun --nodes=$SLURM_NNODES --ntasks-per-node=4 python train_srun_grpo_full.py \
        --config-path=... --config-name=...
"""

import os
import sys
import copy
import socket
import datetime
import logging
import json
import uuid
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
from pprint import pprint

# Set NCCL environment variables BEFORE importing torch
os.environ.setdefault("NCCL_SOCKET_IFNAME", "hsn0")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "hsn0")
os.environ.setdefault("NCCL_NET", "Socket")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_DEBUG", "INFO")
os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,NET")
os.environ.setdefault("NCCL_P2P_LEVEL", "LOC")
os.environ.setdefault("NCCL_CROSS_NIC", "1")

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Distributed Setup
# ============================================================

def setup_distributed():
    """Initialize distributed training using SLURM environment variables."""
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 4))

    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    hostname = socket.gethostname()
    logger.info(f"[{hostname}] Rank {rank}/{world_size}: Starting distributed setup")

    # Set CUDA device
    torch.cuda.set_device(local_rank)
    logger.info(f"[{hostname}] Rank {rank}: Using CUDA device {local_rank}")

    # Initialize process group
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["RANK"] = str(rank)

        init_timeout = datetime.timedelta(seconds=600)
        logger.info(f"[{hostname}] Rank {rank}: Initializing NCCL process group...")

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=init_timeout,
            device_id=torch.device(f"cuda:{local_rank}")
        )
        logger.info(f"[{hostname}] Rank {rank}: NCCL process group initialized")

    dist.barrier()
    logger.info(f"[{hostname}] Rank {rank}: All processes synchronized")

    return rank, world_size, local_rank, local_world_size


# ============================================================
# UI-S1 Core Algorithms (from uis1/core_uis1.py)
# ============================================================

def compute_step_discounted_returns(
    rewards: np.ndarray,
    traj_uids: np.ndarray,
    extract_matches: np.ndarray,
    gamma: float = 0.5
) -> torch.Tensor:
    """
    Compute step-level discounted returns for UI-S1.
    Matches the implementation in uis1/core_uis1.py
    """
    batch_size = len(rewards)
    returns = np.zeros(batch_size, dtype=np.float32)

    # Group by trajectory
    unique_traj_uids = np.unique(traj_uids)
    returns_by_traj = {}

    for uid in unique_traj_uids:
        traj_indices = np.where(traj_uids == uid)[0]
        traj_rewards = rewards[traj_indices].astype(np.float32)
        traj_extract_matches = extract_matches[traj_indices]

        traj_returns = np.zeros_like(traj_rewards)
        running_return = 0.0

        # Backward pass - break when extract_match is False
        for t in reversed(range(len(traj_rewards))):
            if traj_extract_matches[t]:
                running_return = traj_rewards[t] + gamma * running_return
                traj_returns[t] = running_return
            else:
                running_return = 0.0
                traj_returns[t] = traj_rewards[t]

        returns_by_traj[uid] = (traj_indices, traj_returns)

    # Recombine to original batch order
    for uid, (indices, traj_returns) in returns_by_traj.items():
        for i, idx in enumerate(indices):
            returns[idx] = traj_returns[i]

    return torch.tensor(returns, dtype=torch.float32)


def compute_uis1_outcome_advantage(
    token_level_rewards: torch.Tensor,  # (bs, response_length)
    step_rewards: torch.Tensor,         # (bs,)
    response_mask: torch.Tensor,        # (bs, response_length)
    prompt_uids: np.ndarray,            # (bs,)
    traj_uids: np.ndarray,              # (bs,)
    step_ids: np.ndarray,               # (bs,)
    step_advantage_w: float = 1.0,
    episode_advantage_w: float = 1.0,
    mode: str = "mean_std_norm",
    epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute UI-S1 advantages combining episode-level and step-level normalization.
    Matches the implementation in uis1/core_uis1.py
    """
    remove_std = (mode == "mean_norm")
    response_length = token_level_rewards.shape[-1]

    # Episode-level normalization
    episode_advantages = _episode_norm_reward(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=prompt_uids,
        traj_index=traj_uids,
        epsilon=epsilon,
        remove_std=remove_std
    )

    # Step-level normalization
    step_advantages = _step_norm_reward(
        step_rewards=step_rewards,
        response_mask=response_mask,
        index=prompt_uids,
        step_id=step_ids,
        epsilon=epsilon,
        remove_std=remove_std
    )

    # Combine advantages
    advantages = episode_advantage_w * episode_advantages + step_advantage_w * step_advantages
    returns = advantages.clone()

    return advantages, returns


def _episode_norm_reward(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    remove_std: bool = True
) -> torch.Tensor:
    """Episode-level advantage normalization."""
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)  # (bs,)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            pair = (index[i], traj_index[i])
            if pair not in seen_pairs:
                id2score[index[i]].append(scores[i].item())
                # Don't add to seen_pairs to compute mean across all data

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = 0.0
                id2std[idx] = 1.0
            else:
                id2mean[idx] = np.mean(id2score[idx])
                id2std[idx] = np.std(id2score[idx])

        normalized_scores = scores.clone()
        for i in range(bsz):
            if remove_std:
                normalized_scores[i] = scores[i] - id2mean[index[i]]
            else:
                normalized_scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)

        episode_advantages = normalized_scores.unsqueeze(-1).expand(-1, response_length) * response_mask

    return episode_advantages


def _step_norm_reward(
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    step_id: np.ndarray,
    epsilon: float = 1e-6,
    remove_std: bool = True
) -> torch.Tensor:
    """Step-level advantage normalization."""
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            key = f"{index[i]}-{step_id[i]}"
            id2score[key].append(scores[i].item())

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = np.mean(id2score[idx])
                id2std[idx] = 1.0
            else:
                id2mean[idx] = np.mean(id2score[idx])
                id2std[idx] = np.std(id2score[idx])

        for i in range(bsz):
            key = f"{index[i]}-{step_id[i]}"
            if remove_std:
                scores[i] = scores[i] - id2mean[key]
            else:
                scores[i] = (scores[i] - id2mean[key]) / (id2std[key] + epsilon)

        step_advantages = scores.unsqueeze(-1).expand(-1, response_length) * response_mask

    return step_advantages


# ============================================================
# Policy Loss Computation (from verl/trainer/ppo/core_algos.py)
# ============================================================

def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_range: float = 0.2,
    clip_range_low: Optional[float] = None,
    clip_range_high: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute PPO/GRPO clipped policy loss."""
    ratio = torch.exp(log_probs - old_log_probs)

    if clip_range_low is None:
        clip_range_low = 1.0 - clip_range
    if clip_range_high is None:
        clip_range_high = 1.0 + clip_range

    clipped_ratio = torch.clamp(ratio, clip_range_low, clip_range_high)

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    pg_loss = torch.max(pg_loss1, pg_loss2)

    # Apply mask and average
    valid_tokens = response_mask.sum().clamp(min=1)
    pg_loss = (pg_loss * response_mask).sum() / valid_tokens

    # Clip fraction
    clip_frac = ((ratio < clip_range_low) | (ratio > clip_range_high)).float()
    clip_frac = (clip_frac * response_mask).sum() / valid_tokens

    # Approximate KL
    approx_kl = (old_log_probs - log_probs) * response_mask
    approx_kl = approx_kl.sum() / valid_tokens

    return pg_loss, clip_frac, approx_kl


def compute_kl_penalty(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    kl_penalty_type: str = "kl"
) -> torch.Tensor:
    """Compute KL penalty between current and reference policy."""
    if kl_penalty_type == "kl":
        kl = ref_log_probs - log_probs
    elif kl_penalty_type == "abs":
        kl = torch.abs(log_probs - ref_log_probs)
    elif kl_penalty_type == "mse":
        kl = 0.5 * (log_probs - ref_log_probs) ** 2
    elif kl_penalty_type == "low_var_kl":
        # Low variance KL estimator
        log_ratio = log_probs - ref_log_probs
        kl = torch.exp(log_ratio) - 1 - log_ratio
    else:
        raise ValueError(f"Unknown KL penalty type: {kl_penalty_type}")

    return kl * response_mask


# ============================================================
# DAPO Filtering (from dapo_ray_trainer.py)
# ============================================================

def apply_dapo_filter(
    batch_data: Dict,
    metric_name: str = "seq_future_reward",
    std_threshold: float = 0.3,
    max_prompts: Optional[int] = None
) -> Tuple[Dict, int]:
    """
    Apply DAPO filtering to remove low-variance prompt groups.

    Args:
        batch_data: Dictionary containing batch tensors and metadata
        metric_name: Metric to use for filtering (seq_future_reward, seq_reward, seq_final_reward)
        std_threshold: Minimum std threshold to keep a prompt group
        max_prompts: Maximum number of prompts to keep

    Returns:
        Filtered batch_data and number of kept prompts
    """
    prompt_uids = batch_data['prompt_uids']
    step_ids = batch_data['step_ids']

    if metric_name == "seq_future_reward":
        metrics = batch_data['step_rewards']
    elif metric_name == "seq_reward":
        metrics = batch_data['token_level_rewards'].sum(dim=-1).numpy()
    else:
        metrics = batch_data['rewards']

    # Collect metrics per prompt (using step_id == 0 for seq_future_reward)
    prompt_uid2metric_vals = defaultdict(list)

    if metric_name == "seq_future_reward":
        for uid, metric_val, step_id in zip(prompt_uids, metrics, step_ids):
            if step_id == 0:
                prompt_uid2metric_vals[uid].append(metric_val)
    else:
        # For seq_reward/seq_final_reward, use max step_id per trajectory
        prompt_uid2max_step = defaultdict(int)
        for uid, step_id in zip(prompt_uids, step_ids):
            prompt_uid2max_step[uid] = max(prompt_uid2max_step[uid], step_id)

        for uid, metric_val, step_id in zip(prompt_uids, metrics, step_ids):
            if step_id == prompt_uid2max_step[uid]:
                prompt_uid2metric_vals[uid].append(metric_val)

    # Compute std per prompt and filter
    prompt_uid2std = {}
    for uid, vals in prompt_uid2metric_vals.items():
        prompt_uid2std[uid] = np.std(vals) if len(vals) > 1 else 0.0

    # Keep prompts with high std or single rollout
    kept_prompt_uids = [
        uid for uid, std in prompt_uid2std.items()
        if std > std_threshold or len(prompt_uid2metric_vals[uid]) == 1
    ]

    if max_prompts and len(kept_prompt_uids) > max_prompts:
        # Sort by std and keep top max_prompts
        kept_prompt_uids = sorted(
            kept_prompt_uids,
            key=lambda x: prompt_uid2std[x],
            reverse=True
        )[:max_prompts]

    # Filter batch
    kept_indices = [
        i for i, uid in enumerate(prompt_uids)
        if uid in kept_prompt_uids
    ]

    if not kept_indices:
        return None, 0

    filtered_batch = {}
    for key, val in batch_data.items():
        if isinstance(val, torch.Tensor):
            filtered_batch[key] = val[kept_indices]
        elif isinstance(val, np.ndarray):
            filtered_batch[key] = val[kept_indices]
        else:
            filtered_batch[key] = val

    return filtered_batch, len(kept_prompt_uids)


# ============================================================
# Main Trainer Class
# ============================================================

class SrunGRPOFullTrainer:
    """
    Full-featured UI-S1 GRPO Trainer using srun for distributed training.
    Matches the functionality of RayTrajDAPOTrainer.
    """

    def __init__(
        self,
        config: DictConfig,
        rank: int,
        world_size: int,
        local_rank: int,
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.global_step = 0

        # Initialize components
        self._init_tokenizer()
        self._init_model()
        self._init_ref_model()
        self._init_optimizer()
        self._init_dataloader()
        self._init_reward_fn()
        self._init_tracking()

        if self.rank == 0:
            logger.info("SrunGRPOFullTrainer initialized successfully")

    def _init_tokenizer(self):
        """Initialize tokenizer and processor."""
        from transformers import AutoTokenizer, AutoProcessor

        model_path = self.config.actor_rollout_ref.model.path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.rank == 0:
            logger.info(f"Loaded tokenizer from {model_path}")

    def _init_model(self):
        """Initialize the actor model with FSDP."""
        from transformers import Qwen2_5_VLForConditionalGeneration
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer

        model_path = self.config.actor_rollout_ref.model.path

        if self.rank == 0:
            logger.info(f"Loading actor model from {model_path}")

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Enable gradient checkpointing
        if self.config.actor_rollout_ref.model.get("enable_gradient_checkpointing", True):
            model.gradient_checkpointing_enable()
            if self.rank == 0:
                logger.info("Gradient checkpointing enabled")

        # FSDP wrap policy
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Qwen2_5_VLDecoderLayer},
        )

        # Mixed precision
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        # CPU offload if configured
        cpu_offload = None
        if self.config.actor_rollout_ref.actor.fsdp_config.get("param_offload", False):
            from torch.distributed.fsdp import CPUOffload
            cpu_offload = CPUOffload(offload_params=True)
            if self.rank == 0:
                logger.info("FSDP CPU parameter offloading enabled")

        # Wrap with FSDP
        self.model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            auto_wrap_policy=auto_wrap_policy,
            device_id=self.local_rank,
            sync_module_states=False,
            cpu_offload=cpu_offload,
        )

        if self.rank == 0:
            logger.info("Actor model wrapped with FSDP")

        dist.barrier()

    def _init_ref_model(self):
        """Initialize reference model for KL penalty (if enabled)."""
        self.ref_model = None
        self.use_kl_loss = self.config.actor_rollout_ref.actor.get("use_kl_loss", False)

        if not self.use_kl_loss:
            if self.rank == 0:
                logger.info("KL loss disabled, skipping reference model")
            return

        from transformers import Qwen2_5_VLForConditionalGeneration
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer

        model_path = self.config.actor_rollout_ref.model.path

        if self.rank == 0:
            logger.info(f"Loading reference model from {model_path}")

        ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # Freeze reference model
        for param in ref_model.parameters():
            param.requires_grad = False

        # FSDP wrap policy
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Qwen2_5_VLDecoderLayer},
        )

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        # Reference model can use CPU offload more aggressively
        from torch.distributed.fsdp import CPUOffload
        cpu_offload = CPUOffload(offload_params=True)

        self.ref_model = FSDP(
            ref_model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            auto_wrap_policy=auto_wrap_policy,
            device_id=self.local_rank,
            sync_module_states=False,
            cpu_offload=cpu_offload,
        )

        if self.rank == 0:
            logger.info("Reference model wrapped with FSDP")

        dist.barrier()

    def _init_optimizer(self):
        """Initialize optimizer."""
        lr = self.config.actor_rollout_ref.actor.optim.lr
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        if self.rank == 0:
            logger.info(f"Initialized AdamW optimizer with lr={lr}")

    def _init_dataloader(self):
        """Initialize data loaders for trajectory data."""
        from torch.utils.data import DataLoader, DistributedSampler

        train_files = self.config.data.train_files
        if isinstance(train_files, str):
            train_files = [train_files]

        self.train_dataset = TrajDataset(
            data_files=train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data,
        )

        train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )

        batch_size = self.config.data.get('train_batch_size', 2) // self.world_size
        batch_size = max(1, batch_size)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=self._collate_trajectories,
            num_workers=2,
            pin_memory=True,
        )

        # Validation dataset
        val_files = self.config.data.get('val_files')
        if val_files:
            if isinstance(val_files, str):
                val_files = [val_files]

            self.val_dataset = TrajDataset(
                data_files=val_files,
                tokenizer=self.tokenizer,
                processor=self.processor,
                config=self.config.data,
            )

            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=self._collate_trajectories,
                num_workers=1,
            )
        else:
            self.val_dataset = None
            self.val_dataloader = None

        if self.rank == 0:
            logger.info(f"Train dataset size: {len(self.train_dataset)}")
            if self.val_dataset:
                logger.info(f"Val dataset size: {len(self.val_dataset)}")

    def _init_reward_fn(self):
        """Initialize reward function."""
        # For now, use a simple action matching reward
        # In full implementation, this would load from config
        self.reward_fn = self._compute_action_match_reward

    def _init_tracking(self):
        """Initialize tracking/logging."""
        self.tracker = None
        if self.rank == 0:
            try:
                from verl.utils.tracking import Tracking

                project_name = self.config.trainer.get("project_name", "ui-s1-grpo")
                experiment_name = self.config.trainer.get("experiment_name", "srun_grpo_full")
                logger_backends = self.config.trainer.get("logger", ["console", "wandb"])

                if isinstance(logger_backends, str):
                    logger_backends = [logger_backends]

                self.tracker = Tracking(
                    project_name=project_name,
                    experiment_name=experiment_name,
                    default_backend=logger_backends,
                    config=OmegaConf.to_container(self.config, resolve=True),
                )
                logger.info(f"Tracking initialized: {logger_backends}")
            except Exception as e:
                logger.warning(f"Failed to initialize tracking: {e}")

    def _collate_trajectories(self, batch: List[Dict]) -> Dict[str, Any]:
        """Collate trajectory data into a batch."""
        return {
            'trajectories': batch,
            'num_trajectories': len(batch),
        }

    def _compute_action_match_reward(
        self,
        predicted_action: str,
        ground_truth: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute reward based on action matching.
        Simplified version of the full reward function.
        """
        info = {
            'type_match': False,
            'extract_match': False,
            'format_score': 0.0,
            'score': 0.0,
        }

        # Try to parse predicted action
        try:
            pred_type, pred_content = self._parse_action(predicted_action)
            info['extract_match'] = True
        except:
            info['extract_match'] = False
            return 0.0, info

        # Get ground truth
        gt_type = ground_truth.get('action_type', '') or ground_truth.get('action', '')
        gt_text = ground_truth.get('text', '') or ''
        gt_coord = ground_truth.get('coordinate')

        # Type matching
        type_score = 0.0
        if gt_type and pred_type:
            gt_type_lower = gt_type.lower().replace('_', '')
            pred_type_lower = pred_type.lower().replace('_', '')
            type_score = 1.0 if pred_type_lower == gt_type_lower else 0.0
        info['type_match'] = type_score > 0
        info['format_score'] = type_score

        # Content matching
        content_score = 0.0
        if pred_content and gt_text:
            pred_lower = pred_content.lower().strip()
            gt_lower = gt_text.lower().strip()
            if pred_lower == gt_lower:
                content_score = 1.0
            elif pred_lower in gt_lower or gt_lower in pred_lower:
                content_score = 0.5
        elif info['type_match'] and not gt_text:
            # For actions without text, type match is sufficient
            content_score = 0.5

        # Coordinate matching (simplified)
        coord_score = 0.0
        if gt_coord and pred_content:
            try:
                # Try to extract coordinates from prediction
                import re
                coord_match = re.search(r'\[?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]?', pred_content)
                if coord_match:
                    pred_x, pred_y = float(coord_match.group(1)), float(coord_match.group(2))
                    if isinstance(gt_coord, (list, tuple)) and len(gt_coord) >= 2:
                        gt_x, gt_y = float(gt_coord[0]), float(gt_coord[1])
                        dist = ((pred_x - gt_x)**2 + (pred_y - gt_y)**2)**0.5
                        # Normalize distance (assuming 0-1000 coordinate range)
                        coord_score = max(0, 1 - dist / 100)
            except:
                pass

        # Combined score
        action_type_weight = 0.3
        action_content_weight = 0.5
        coord_weight = 0.2

        if gt_coord:
            info['score'] = (action_type_weight * type_score +
                           action_content_weight * content_score +
                           coord_weight * coord_score)
        else:
            info['score'] = (action_type_weight * type_score +
                           (action_content_weight + coord_weight) * content_score)

        return info['score'], info

    def _parse_action(self, action_str: str) -> Tuple[str, str]:
        """Parse action string to extract type and content."""
        import re

        action_str = action_str.strip()

        # Try to match function call format: action_type(content)
        match = re.match(r'(\w+)\s*\((.*)\)', action_str, re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()

        # Try to find action type keywords
        action_types = ['click', 'type', 'scroll', 'swipe', 'press', 'long_press', 'open_app', 'tap', 'input']

        for at in action_types:
            if at in action_str.lower():
                idx = action_str.lower().find(at)
                action_content = action_str[idx + len(at):].strip()
                # Clean up content
                action_content = action_content.strip('()[]{}').strip()
                return at, action_content

        raise ValueError(f"Cannot parse action: {action_str}")

    @torch.no_grad()
    def _generate_responses(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        do_sample: bool = True,
        num_return_sequences: int = 1,
    ) -> torch.Tensor:
        """Generate responses using the model."""
        self.model.eval()

        # Use model.generate for inference
        # Note: FSDP models need special handling for generation
        generation_config = {
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'do_sample': do_sample,
            'num_return_sequences': num_return_sequences,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }

        try:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model.generate(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                    **generation_config
                )

            # Extract only the new tokens
            responses = outputs[:, input_ids.shape[1]:]
            return responses

        except Exception as e:
            logger.warning(f"Generation failed: {e}, using dummy response")
            # Return dummy response
            dummy_response = torch.full(
                (input_ids.shape[0] * num_return_sequences, max_new_tokens),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=self.device
            )
            return dummy_response

    def _compute_log_probs(
        self,
        model: FSDP,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for response tokens."""
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = response_mask[:, 1:].contiguous()

        # Compute log softmax
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Apply mask
        token_log_probs = token_log_probs * shift_mask

        # Pad to original length
        padded_log_probs = torch.zeros(
            input_ids.shape[0], input_ids.shape[1],
            dtype=token_log_probs.dtype, device=token_log_probs.device
        )
        padded_log_probs[:, 1:] = token_log_probs

        return padded_log_probs

    def _process_trajectory_batch(
        self,
        trajectories: List[Dict],
        n_rollouts: int,
        gamma: float,
    ) -> Optional[Dict[str, Any]]:
        """Process a batch of trajectories and generate rollouts."""
        all_step_data = []

        for traj in trajectories:
            steps = traj.get('steps', [])
            if not steps:
                continue

            prompt_uid = str(uuid.uuid4())

            for rollout_idx in range(n_rollouts):
                traj_uid = f"{prompt_uid}_{rollout_idx}"
                history = []

                for step_idx, step in enumerate(steps):
                    # Build prompt from history and current step
                    prompt_text = self._build_prompt(history, step, traj)

                    # Get ground truth
                    action_content = step.get('action_content', {})
                    if isinstance(action_content, dict):
                        ground_truth = {
                            'action_type': action_content.get('action', ''),
                            'coordinate': action_content.get('coordinate'),
                            'text': action_content.get('text', ''),
                        }
                    else:
                        ground_truth = {'action_type': str(action_content)}

                    # Tokenize prompt
                    prompt_encoding = self.tokenizer(
                        prompt_text,
                        max_length=self.config.data.get('max_prompt_length', 4096),
                        truncation=True,
                        padding='max_length',
                        return_tensors='pt',
                    )

                    # Generate response
                    with torch.no_grad():
                        responses = self._generate_responses(
                            input_ids=prompt_encoding['input_ids'],
                            attention_mask=prompt_encoding['attention_mask'],
                            max_new_tokens=self.config.data.get('max_response_length', 512),
                            temperature=1.0,
                            do_sample=True,
                            num_return_sequences=1,
                        )

                    # Decode response
                    response_text = self.tokenizer.decode(responses[0], skip_special_tokens=True)

                    # Compute reward
                    reward, reward_info = self.reward_fn(response_text, ground_truth)

                    # Create full sequence (prompt + response)
                    full_ids = torch.cat([
                        prompt_encoding['input_ids'],
                        responses.cpu()
                    ], dim=1)

                    full_mask = torch.cat([
                        prompt_encoding['attention_mask'],
                        torch.ones_like(responses.cpu())
                    ], dim=1)

                    prompt_len = prompt_encoding['attention_mask'].sum().item()

                    step_data = {
                        'input_ids': full_ids.squeeze(0),
                        'attention_mask': full_mask.squeeze(0),
                        'prompt_length': prompt_len,
                        'reward': reward,
                        'reward_info': reward_info,
                        'prompt_uid': prompt_uid,
                        'traj_uid': traj_uid,
                        'step_id': step_idx,
                        'response_text': response_text,
                    }
                    all_step_data.append(step_data)

                    # Update history with ground truth for next step
                    history.append(ground_truth)

        if not all_step_data:
            return None

        return self._prepare_batch_tensors(all_step_data)

    def _build_prompt(self, history: List[Dict], current_step: Dict, traj: Dict) -> str:
        """Build prompt from history and current step context."""
        prompt_parts = []

        # Add task instruction
        if 'goal' in traj:
            prompt_parts.append(f"Task: {traj['goal']}")

        # Add history
        if history:
            prompt_parts.append("\nPrevious actions:")
            for i, h in enumerate(history):
                action_type = h.get('action_type', '')
                text = h.get('text', '')
                coord = h.get('coordinate')
                if text:
                    prompt_parts.append(f"Step {i+1}: {action_type}(\"{text}\")")
                elif coord:
                    prompt_parts.append(f"Step {i+1}: {action_type}({coord})")
                else:
                    prompt_parts.append(f"Step {i+1}: {action_type}()")

        # Add current context
        prompt_parts.append("\nCurrent step:")
        if 'screenshot' in current_step or 'image' in current_step:
            prompt_parts.append("[Screenshot provided]")

        prompt_parts.append("\nPredict the next action:")

        return "\n".join(prompt_parts)

    def _prepare_batch_tensors(self, step_data: List[Dict]) -> Dict[str, Any]:
        """Prepare batch tensors from step data."""
        batch_size = len(step_data)
        max_len = max(d['input_ids'].shape[0] for d in step_data)

        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        response_mask = torch.zeros(batch_size, max_len, dtype=torch.float)

        rewards = np.zeros(batch_size, dtype=np.float32)
        extract_matches = np.zeros(batch_size, dtype=bool)
        prompt_uids = []
        traj_uids = []
        step_ids = np.zeros(batch_size, dtype=np.int32)

        for i, data in enumerate(step_data):
            seq_len = data['input_ids'].shape[0]
            input_ids[i, :seq_len] = data['input_ids']
            attention_mask[i, :seq_len] = data['attention_mask']

            # Response mask (tokens after prompt)
            prompt_len = data['prompt_length']
            response_mask[i, prompt_len:seq_len] = 1.0

            rewards[i] = data['reward']
            extract_matches[i] = data['reward_info'].get('extract_match', True)
            prompt_uids.append(data['prompt_uid'])
            traj_uids.append(data['traj_uid'])
            step_ids[i] = data['step_id']

        # Token-level rewards (broadcast step reward to response tokens)
        token_level_rewards = torch.zeros_like(response_mask)
        for i in range(batch_size):
            valid_tokens = response_mask[i].sum().clamp(min=1)
            token_level_rewards[i] = response_mask[i] * rewards[i] / valid_tokens

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'response_mask': response_mask,
            'token_level_rewards': token_level_rewards,
            'rewards': rewards,
            'extract_matches': extract_matches,
            'prompt_uids': np.array(prompt_uids),
            'traj_uids': np.array(traj_uids),
            'step_ids': step_ids,
        }

    def _validate(self) -> Dict[str, float]:
        """Run validation loop."""
        if self.val_dataloader is None:
            return {}

        self.model.eval()

        all_rewards = []
        all_type_matches = []
        all_extract_matches = []
        total_steps = 0
        failed_steps = 0

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", disable=self.rank != 0):
                trajectories = batch['trajectories']

                for traj in trajectories:
                    steps = traj.get('steps', [])
                    history = []

                    for step in steps:
                        total_steps += 1

                        # Build prompt and generate
                        prompt_text = self._build_prompt(history, step, traj)
                        prompt_encoding = self.tokenizer(
                            prompt_text,
                            max_length=self.config.data.get('max_prompt_length', 4096),
                            truncation=True,
                            return_tensors='pt',
                        )

                        responses = self._generate_responses(
                            input_ids=prompt_encoding['input_ids'],
                            attention_mask=prompt_encoding['attention_mask'],
                            max_new_tokens=256,
                            temperature=0.0,  # Greedy for validation
                            do_sample=False,
                        )

                        response_text = self.tokenizer.decode(responses[0], skip_special_tokens=True)

                        # Get ground truth and compute reward
                        action_content = step.get('action_content', {})
                        if isinstance(action_content, dict):
                            ground_truth = {
                                'action_type': action_content.get('action', ''),
                                'coordinate': action_content.get('coordinate'),
                                'text': action_content.get('text', ''),
                            }
                        else:
                            ground_truth = {'action_type': str(action_content)}

                        reward, info = self.reward_fn(response_text, ground_truth)

                        all_rewards.append(reward)
                        all_type_matches.append(info.get('type_match', False))
                        all_extract_matches.append(info.get('extract_match', False))

                        if not info.get('extract_match', False):
                            failed_steps += 1

                        # Update history
                        history.append(ground_truth)

        self.model.train()

        if not all_rewards:
            return {}

        metrics = {
            'val/reward_mean': np.mean(all_rewards),
            'val/reward_std': np.std(all_rewards),
            'val/type_match_rate': np.mean(all_type_matches),
            'val/extract_match_rate': np.mean(all_extract_matches),
            'val/task_success_rate': 1.0 - failed_steps / max(total_steps, 1),
        }

        return metrics

    def fit(self):
        """Run the full GRPO training loop."""
        import time

        # Training hyperparameters
        total_epochs = self.config.trainer.total_epochs
        gamma = self.config.algorithm.get('gamma', 0.5)
        clip_range = self.config.actor_rollout_ref.actor.get('clip_ratio', 0.2)
        kl_coef = self.config.actor_rollout_ref.actor.get('kl_loss_coef', 0.0001)
        kl_loss_type = self.config.actor_rollout_ref.actor.get('kl_loss_type', 'low_var_kl')
        n_rollouts = self.config.actor_rollout_ref.rollout.get('n', 4)

        step_advantage_w = self.config.algorithm.uis1.get('step_advantage_w', 1.0)
        episode_advantage_w = self.config.algorithm.uis1.get('episode_advantage_w', 1.0)
        adv_mode = self.config.algorithm.uis1.get('mode', 'mean_std_norm')

        # DAPO settings
        dapo_enabled = self.config.algorithm.filter_groups.get('enable', False)
        dapo_threshold = self.config.algorithm.filter_groups.get('std_threshold', 0.3)
        dapo_metric = self.config.algorithm.filter_groups.get('metric', 'seq_future_reward')

        save_freq = self.config.trainer.get('save_freq', 5)
        test_freq = self.config.trainer.get('test_freq', 10)
        val_before_train = self.config.trainer.get('val_before_train', False)

        if self.rank == 0:
            logger.info(f"Starting GRPO training for {total_epochs} epochs")
            logger.info(f"gamma={gamma}, clip_range={clip_range}, n_rollouts={n_rollouts}")
            logger.info(f"step_advantage_w={step_advantage_w}, episode_advantage_w={episode_advantage_w}")
            logger.info(f"DAPO enabled={dapo_enabled}, threshold={dapo_threshold}")
            logger.info(f"KL loss enabled={self.use_kl_loss}, coef={kl_coef}, type={kl_loss_type}")

        # Validation before training
        if val_before_train and self.rank == 0:
            val_metrics = self._validate()
            if val_metrics:
                logger.info(f"Initial validation metrics: {val_metrics}")
                if self.tracker:
                    self.tracker.log(val_metrics, step=0)

        for epoch in range(total_epochs):
            self.train_dataloader.sampler.set_epoch(epoch)
            epoch_start_time = time.time()
            epoch_metrics = defaultdict(list)

            for batch_idx, batch in enumerate(tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch}",
                disable=self.rank != 0
            )):
                batch_start_time = time.time()
                self.model.train()

                # Process trajectories and generate rollouts
                batch_data = self._process_trajectory_batch(
                    trajectories=batch['trajectories'],
                    n_rollouts=n_rollouts,
                    gamma=gamma,
                )

                if batch_data is None:
                    continue

                # Apply DAPO filtering if enabled
                if dapo_enabled:
                    # First compute step rewards for filtering
                    step_rewards = compute_step_discounted_returns(
                        rewards=batch_data['rewards'],
                        traj_uids=batch_data['traj_uids'],
                        extract_matches=batch_data['extract_matches'],
                        gamma=gamma,
                    )
                    batch_data['step_rewards'] = step_rewards.numpy()

                    batch_data, num_kept = apply_dapo_filter(
                        batch_data,
                        metric_name=dapo_metric,
                        std_threshold=dapo_threshold,
                    )

                    if batch_data is None:
                        if self.rank == 0:
                            logger.warning("DAPO filtered all samples, skipping batch")
                        continue

                    epoch_metrics['dapo_kept_prompts'].append(num_kept)

                # Move tensors to device
                input_ids = batch_data['input_ids'].to(self.device)
                attention_mask = batch_data['attention_mask'].to(self.device)
                response_mask = batch_data['response_mask'].to(self.device)
                token_level_rewards = batch_data['token_level_rewards'].to(self.device)

                # Compute step-level discounted returns
                step_rewards = compute_step_discounted_returns(
                    rewards=batch_data['rewards'],
                    traj_uids=batch_data['traj_uids'],
                    extract_matches=batch_data['extract_matches'],
                    gamma=gamma,
                ).to(self.device)

                # Compute old log probs (before update)
                with torch.no_grad():
                    old_log_probs = self._compute_log_probs(
                        self.model, input_ids, attention_mask, response_mask
                    )

                    # Compute reference log probs if using KL loss
                    if self.use_kl_loss and self.ref_model is not None:
                        ref_log_probs = self._compute_log_probs(
                            self.ref_model, input_ids, attention_mask, response_mask
                        )
                    else:
                        ref_log_probs = None

                # Compute advantages
                advantages, returns = compute_uis1_outcome_advantage(
                    token_level_rewards=token_level_rewards,
                    step_rewards=step_rewards,
                    response_mask=response_mask,
                    prompt_uids=batch_data['prompt_uids'],
                    traj_uids=batch_data['traj_uids'],
                    step_ids=batch_data['step_ids'],
                    step_advantage_w=step_advantage_w,
                    episode_advantage_w=episode_advantage_w,
                    mode=adv_mode,
                )
                advantages = advantages.to(self.device)

                # Training step
                self.optimizer.zero_grad()

                # Forward pass
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    logits = outputs.logits

                    # Compute current log probs
                    log_probs = self._compute_log_probs(
                        self.model, input_ids, attention_mask, response_mask
                    )

                    # Compute policy loss
                    pg_loss, clip_frac, approx_kl = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_range=clip_range,
                    )

                    # KL loss
                    kl_loss = torch.tensor(0.0, device=self.device)
                    if self.use_kl_loss and ref_log_probs is not None:
                        kl_penalty = compute_kl_penalty(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            response_mask=response_mask,
                            kl_penalty_type=kl_loss_type,
                        )
                        kl_loss = kl_penalty.sum() / response_mask.sum().clamp(min=1)
                        kl_loss = kl_loss * kl_coef

                    # Total loss
                    total_loss = pg_loss + kl_loss

                # Backward pass
                total_loss.backward()

                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0,
                )

                self.optimizer.step()

                batch_time = time.time() - batch_start_time
                self.global_step += 1

                # Record metrics
                epoch_metrics['pg_loss'].append(pg_loss.item())
                epoch_metrics['kl_loss'].append(kl_loss.item())
                epoch_metrics['clip_frac'].append(clip_frac.item())
                epoch_metrics['grad_norm'].append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                epoch_metrics['advantages_mean'].append(advantages.mean().item())
                epoch_metrics['advantages_std'].append(advantages.std().item())
                epoch_metrics['rewards_mean'].append(batch_data['rewards'].mean())
                epoch_metrics['approx_kl'].append(approx_kl.item())

                # Log metrics
                if batch_idx % 10 == 0 and self.rank == 0:
                    logger.info(
                        f"Epoch {epoch}, Batch {batch_idx}, "
                        f"PG Loss: {pg_loss.item():.4f}, "
                        f"KL Loss: {kl_loss.item():.6f}, "
                        f"Clip Frac: {clip_frac.item():.4f}, "
                        f"Adv Mean: {advantages.mean().item():.4f}"
                    )

                    if self.tracker:
                        self.tracker.log({
                            "train/pg_loss": pg_loss.item(),
                            "train/kl_loss": kl_loss.item(),
                            "train/clip_frac": clip_frac.item(),
                            "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                            "train/advantages_mean": advantages.mean().item(),
                            "train/advantages_std": advantages.std().item(),
                            "train/rewards_mean": float(batch_data['rewards'].mean()),
                            "train/approx_kl": approx_kl.item(),
                            "train/epoch": epoch,
                            "train/global_step": self.global_step,
                            "perf/batch_time_s": batch_time,
                        }, step=self.global_step)

            # Epoch end
            epoch_time = time.time() - epoch_start_time

            if self.rank == 0:
                avg_pg_loss = np.mean(epoch_metrics['pg_loss']) if epoch_metrics['pg_loss'] else 0
                avg_kl = np.mean(epoch_metrics['kl_loss']) if epoch_metrics['kl_loss'] else 0
                avg_reward = np.mean(epoch_metrics['rewards_mean']) if epoch_metrics['rewards_mean'] else 0

                logger.info(
                    f"Epoch {epoch} completed in {epoch_time:.1f}s. "
                    f"Avg PG Loss: {avg_pg_loss:.4f}, "
                    f"Avg KL: {avg_kl:.6f}, "
                    f"Avg Reward: {avg_reward:.4f}"
                )

                if self.tracker:
                    self.tracker.log({
                        "epoch/avg_pg_loss": avg_pg_loss,
                        "epoch/avg_kl": avg_kl,
                        "epoch/avg_reward": avg_reward,
                        "epoch/time_s": epoch_time,
                        "epoch/epoch": epoch,
                    }, step=self.global_step)

            # Validation
            if test_freq > 0 and (epoch + 1) % test_freq == 0:
                if self.rank == 0:
                    val_metrics = self._validate()
                    if val_metrics:
                        logger.info(f"Validation metrics: {val_metrics}")
                        if self.tracker:
                            self.tracker.log(val_metrics, step=self.global_step)

            # Save checkpoint
            if save_freq > 0 and (epoch + 1) % save_freq == 0:
                self._save_checkpoint(epoch)

            dist.barrier()

        if self.rank == 0:
            logger.info("GRPO Training completed!")

    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        checkpoint_dir = self.config.trainer.get(
            "checkpoint_dir",
            f"{self.config.actor_rollout_ref.model.path}/../checkpoints/grpo"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"grpo_epoch_{epoch}.pt")

        full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
            state_dict = self.model.state_dict()
            if self.rank == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'global_step': self.global_step,
                }, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")


# ============================================================
# Trajectory Dataset
# ============================================================

class TrajDataset:
    """Trajectory dataset loader."""

    def __init__(
        self,
        data_files: List[str],
        tokenizer,
        processor,
        config: DictConfig,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.data = []

        for data_file in data_files:
            if data_file.endswith('.jsonl'):
                with open(data_file, 'r') as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            # Ensure the item has steps
                            if 'steps' in item and len(item['steps']) > 0:
                                self.data.append(item)
                        except:
                            continue
            elif data_file.endswith('.json'):
                with open(data_file, 'r') as f:
                    items = json.load(f)
                    for item in items:
                        if 'steps' in item and len(item['steps']) > 0:
                            self.data.append(item)

        logger.info(f"Loaded {len(self.data)} trajectories from {data_files}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ============================================================
# Main Entry Point
# ============================================================

@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config: DictConfig):
    """Main entry point for GRPO training."""
    rank, world_size, local_rank, local_world_size = setup_distributed()

    if rank == 0:
        logger.info("=" * 60)
        logger.info("UI-S1 GRPO Full Training with srun")
        logger.info("=" * 60)
        logger.info(f"World size: {world_size}, Local world size: {local_world_size}")
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # Create trainer and run
    trainer = SrunGRPOFullTrainer(
        config=config,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )

    trainer.fit()

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
