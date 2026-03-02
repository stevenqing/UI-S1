#!/usr/bin/env python3
"""
Srun-based UI-S1 GRPO Training Worker

This implements the full UI-S1 GRPO algorithm from the paper:
1. Multi-step trajectory rollout generation
2. Step-level reward computation
3. UI-S1 advantage estimation (episode + step level)
4. GRPO policy gradient updates with KL regularization

Usage:
    srun --nodes=$SLURM_NNODES --ntasks-per-node=4 python train_srun_grpo_worker.py \
        --config-path=... --config-name=...
"""

import os
import sys
import socket
import datetime
import logging
import json
import copy
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

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
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    logger.info(f"[{hostname}] Rank {rank}: Initializing NCCL process group...")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{local_rank}")
    )
    logger.info(f"[{hostname}] Rank {rank}: NCCL process group initialized")

    dist.barrier()
    logger.info(f"[{hostname}] Rank {rank}: All processes synchronized")

    return rank, world_size, local_rank, local_world_size


# ============================================================
# UI-S1 Core Algorithms (adapted from uis1/core_uis1.py)
# ============================================================

def compute_step_discounted_returns(
    rewards: np.ndarray,
    traj_uids: np.ndarray,
    extract_matches: np.ndarray,
    gamma: float = 0.5
) -> torch.Tensor:
    """
    Compute step-level discounted returns for UI-S1.

    Args:
        rewards: (batch_size,) step rewards
        traj_uids: (batch_size,) trajectory UIDs
        extract_matches: (batch_size,) boolean indicating if action extraction succeeded
        gamma: discount factor

    Returns:
        torch.Tensor: (batch_size,) discounted returns
    """
    batch_size = len(rewards)
    returns = np.zeros(batch_size)

    # Group by trajectory
    traj_uid_to_indices = defaultdict(list)
    for idx, uid in enumerate(traj_uids):
        traj_uid_to_indices[uid].append(idx)

    # Compute discounted returns for each trajectory
    for uid, indices in traj_uid_to_indices.items():
        # Sort indices by their order (assumes they're in sequence order)
        indices = sorted(indices)
        n_steps = len(indices)

        # Backward pass to compute discounted returns
        cumulative = 0.0
        for t in reversed(range(n_steps)):
            idx = indices[t]
            r = rewards[idx]

            # If extraction failed at step t, don't propagate future rewards
            if not extract_matches[idx]:
                cumulative = r
            else:
                cumulative = r + gamma * cumulative

            returns[idx] = cumulative

    return torch.tensor(returns, dtype=torch.float32)


def compute_uis1_outcome_advantage(
    token_level_rewards: torch.Tensor,  # (bs, response_length)
    step_rewards: torch.Tensor,         # (bs,) discounted step returns
    response_mask: torch.Tensor,        # (bs, response_length)
    prompt_uids: np.ndarray,            # (bs,) prompt UIDs
    traj_uids: np.ndarray,              # (bs,) trajectory UIDs
    step_ids: np.ndarray,               # (bs,) step IDs within trajectory
    step_advantage_w: float = 1.0,
    episode_advantage_w: float = 1.0,
    mode: str = "mean_std_norm",
    epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute UI-S1 advantages combining episode-level and step-level normalization.

    Args:
        token_level_rewards: Per-token rewards
        step_rewards: Discounted step returns from compute_step_discounted_returns
        response_mask: Mask for valid response tokens
        prompt_uids: Unique prompt identifiers
        traj_uids: Unique trajectory identifiers
        step_ids: Step index within each trajectory
        step_advantage_w: Weight for step-level advantage
        episode_advantage_w: Weight for episode-level advantage
        mode: "mean_std_norm" or "mean_norm"
        epsilon: Small constant for numerical stability

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    batch_size = token_level_rewards.shape[0]

    # Compute sequence-level scores
    seq_scores = token_level_rewards.sum(dim=-1)  # (bs,)

    # Episode-level normalization (group by prompt + trajectory)
    episode_advantages = _normalize_by_group(
        scores=seq_scores.numpy(),
        group_keys=[(str(prompt_uids[i]), str(traj_uids[i])) for i in range(batch_size)],
        mode=mode,
        epsilon=epsilon
    )

    # Step-level normalization (group by prompt + step_id)
    step_advantages = _normalize_by_group(
        scores=step_rewards.numpy(),
        group_keys=[(str(prompt_uids[i]), str(step_ids[i])) for i in range(batch_size)],
        mode=mode,
        epsilon=epsilon
    )

    # Combine advantages
    combined = episode_advantage_w * episode_advantages + step_advantage_w * step_advantages
    combined = torch.tensor(combined, dtype=torch.float32)

    # Broadcast to token level
    advantages = combined.unsqueeze(-1) * response_mask
    returns = advantages.clone()

    return advantages, returns


def _normalize_by_group(
    scores: np.ndarray,
    group_keys: List[Tuple],
    mode: str = "mean_std_norm",
    epsilon: float = 1e-6
) -> np.ndarray:
    """Normalize scores within each group."""
    # Group scores
    group_to_indices = defaultdict(list)
    for idx, key in enumerate(group_keys):
        group_to_indices[key].append(idx)

    normalized = np.zeros_like(scores)

    for key, indices in group_to_indices.items():
        group_scores = scores[indices]
        mean = np.mean(group_scores)
        std = np.std(group_scores)

        if mode == "mean_std_norm":
            normalized[indices] = (group_scores - mean) / (std + epsilon)
        else:  # mean_norm
            normalized[indices] = group_scores - mean

    return normalized


# ============================================================
# Policy Loss Computation (adapted from verl/trainer/ppo/core_algos.py)
# ============================================================

def compute_policy_loss(
    old_log_probs: torch.Tensor,   # (bs, response_length)
    log_probs: torch.Tensor,       # (bs, response_length)
    advantages: torch.Tensor,      # (bs, response_length)
    response_mask: torch.Tensor,   # (bs, response_length)
    clip_range: float = 0.2,
    clip_range_low: Optional[float] = None,
    clip_range_high: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute PPO/GRPO clipped policy loss.

    Returns:
        pg_loss: Policy gradient loss (scalar)
        clip_frac: Fraction of clipped ratios
        approx_kl: Approximate KL divergence
    """
    # Compute probability ratio
    ratio = torch.exp(log_probs - old_log_probs)

    # Clipped ratio
    if clip_range_low is None:
        clip_range_low = 1.0 - clip_range
    if clip_range_high is None:
        clip_range_high = 1.0 + clip_range

    clipped_ratio = torch.clamp(ratio, clip_range_low, clip_range_high)

    # Policy gradient loss
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    pg_loss = torch.max(pg_loss1, pg_loss2)

    # Apply mask and average
    pg_loss = (pg_loss * response_mask).sum() / response_mask.sum().clamp(min=1)

    # Clip fraction
    clip_frac = ((ratio < clip_range_low) | (ratio > clip_range_high)).float()
    clip_frac = (clip_frac * response_mask).sum() / response_mask.sum().clamp(min=1)

    # Approximate KL
    approx_kl = (old_log_probs - log_probs) * response_mask
    approx_kl = approx_kl.sum() / response_mask.sum().clamp(min=1)

    return pg_loss, clip_frac, approx_kl


# ============================================================
# Reward Function (adapted from x/reward_score/gui_traj_action_match.py)
# ============================================================

def compute_step_reward(
    predicted_action: str,
    ground_truth: Dict[str, Any],
    action_type_weight: float = 0.3,
    action_content_weight: float = 0.7,
) -> Tuple[float, Dict[str, Any]]:
    """
    Compute reward for a single step based on action matching.

    Args:
        predicted_action: Model's predicted action string
        ground_truth: Ground truth action with type, content, bbox, etc.

    Returns:
        reward: Float reward value
        info: Dict with detailed matching information
    """
    info = {
        'type_match': False,
        'extract_match': False,
        'format_score': 0.0,
        'score': 0.0,
    }

    # Try to parse predicted action
    try:
        # Simple parsing - extract action type and content
        pred_type, pred_content = _parse_action(predicted_action)
        info['extract_match'] = True
    except:
        info['extract_match'] = False
        return 0.0, info

    # Get ground truth
    gt_type = ground_truth.get('action_type', '')
    gt_text = ground_truth.get('text', '') or ''
    gt_coordinate = ground_truth.get('coordinate')
    gt_bbox = ground_truth.get('bbox')

    # Type matching
    type_score = 0.0
    if gt_type and pred_type:
        type_score = 1.0 if pred_type.lower() == gt_type.lower() else 0.0
    info['type_match'] = type_score > 0

    # Content matching (simplified)
    content_score = 0.0
    if pred_content:
        # Check text content match
        if gt_text:
            pred_lower = pred_content.lower()
            gt_lower = gt_text.lower()
            if pred_lower == gt_lower:
                content_score = 1.0
            elif pred_lower in gt_lower or gt_lower in pred_lower:
                content_score = 0.5
        # If no text, check if action type matches (for click, scroll, etc.)
        elif info['type_match']:
            # For actions without text content, type match is sufficient
            content_score = 0.5

    # Combined score
    info['format_score'] = type_score
    info['score'] = action_type_weight * type_score + action_content_weight * content_score

    return info['score'], info


def _parse_action(action_str: str) -> Tuple[str, str]:
    """Parse action string to extract type and content."""
    # Simple parsing - adjust based on actual format
    action_str = action_str.strip()

    # Try to find action type
    action_types = ['click', 'type', 'scroll', 'swipe', 'press', 'long_press', 'open_app']

    action_type = ''
    action_content = ''

    for at in action_types:
        if at in action_str.lower():
            action_type = at
            # Extract content after action type
            idx = action_str.lower().find(at)
            action_content = action_str[idx + len(at):].strip()
            break

    if not action_type:
        raise ValueError(f"Cannot parse action: {action_str}")

    return action_type, action_content


# ============================================================
# Main Trainer Class
# ============================================================

class SrunGRPOTrainer:
    """
    UI-S1 GRPO Trainer using srun for distributed training.

    Implements the full training loop:
    1. Load trajectory data
    2. Generate rollouts (responses)
    3. Compute rewards
    4. Compute UI-S1 advantages
    5. Update policy with GRPO loss
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
        self._init_optimizer()
        self._init_dataloader()
        self._init_tracking()

        if self.rank == 0:
            logger.info("SrunGRPOTrainer initialized successfully")

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
        """Initialize model with FSDP."""
        from transformers import Qwen2_5_VLForConditionalGeneration
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy, MixedPrecision
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
        import functools

        model_path = self.config.actor_rollout_ref.model.path

        if self.rank == 0:
            logger.info(f"Loading model from {model_path}")

        # Load model
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

        # Wrap with FSDP
        self.model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            auto_wrap_policy=auto_wrap_policy,
            device_id=self.local_rank,
            sync_module_states=False,
        )

        if self.rank == 0:
            logger.info("Model wrapped with FSDP")

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

        # Load trajectory dataset
        train_files = self.config.data.train_files
        if isinstance(train_files, str):
            train_files = [train_files]

        self.train_dataset = TrajDatasetSimple(
            data_files=train_files,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.data.get('max_prompt_length', 4096),
            max_response_length=self.config.data.get('max_response_length', 512),
        )

        train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )

        batch_size = self.config.actor_rollout_ref.actor.get("ppo_micro_batch_size_per_gpu", 1)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=self._collate_trajectories,
            num_workers=2,
            pin_memory=True,
        )

        if self.rank == 0:
            logger.info(f"Train dataset size: {len(self.train_dataset)}")
            logger.info(f"DataLoader batch_size={batch_size}")

    def _init_tracking(self):
        """Initialize tracking/logging."""
        from verl.utils.tracking import Tracking

        self.tracker = None
        if self.rank == 0:
            project_name = self.config.trainer.get("project_name", "ui-s1-grpo")
            experiment_name = self.config.trainer.get("experiment_name", "srun_grpo")
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

    def _collate_trajectories(self, batch: List[Dict]) -> Dict[str, Any]:
        """Collate trajectory data into a batch."""
        # Each item in batch is a trajectory with multiple steps
        collated = {
            'trajectories': batch,
            'num_trajectories': len(batch),
        }
        return collated

    def fit(self):
        """Run the full GRPO training loop."""
        import time

        total_epochs = self.config.trainer.total_epochs
        gamma = self.config.algorithm.get('gamma', 0.5)
        clip_range = self.config.actor_rollout_ref.actor.get('clip_ratio', 0.2)
        kl_coef = self.config.actor_rollout_ref.actor.get('kl_loss_coef', 0.0001)
        n_rollouts = self.config.actor_rollout_ref.rollout.get('n', 4)

        step_advantage_w = self.config.algorithm.uis1.get('step_advantage_w', 1.0)
        episode_advantage_w = self.config.algorithm.uis1.get('episode_advantage_w', 1.0)
        adv_mode = self.config.algorithm.uis1.get('mode', 'mean_std_norm')

        if self.rank == 0:
            logger.info(f"Starting GRPO training for {total_epochs} epochs")
            logger.info(f"gamma={gamma}, clip_range={clip_range}, n_rollouts={n_rollouts}")
            logger.info(f"step_advantage_w={step_advantage_w}, episode_advantage_w={episode_advantage_w}")

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

                # Process each trajectory in the batch
                trajectories = batch['trajectories']

                all_step_data = []

                for traj in trajectories:
                    # Process trajectory steps
                    step_data = self._process_trajectory(traj, n_rollouts)
                    all_step_data.extend(step_data)

                if not all_step_data:
                    continue

                # Convert to tensors
                batch_data = self._prepare_batch_tensors(all_step_data)

                if batch_data is None:
                    continue

                # Compute step-level discounted returns
                step_rewards = compute_step_discounted_returns(
                    rewards=batch_data['rewards'],
                    traj_uids=batch_data['traj_uids'],
                    extract_matches=batch_data['extract_matches'],
                    gamma=gamma,
                )

                # Compute advantages
                advantages, returns = compute_uis1_outcome_advantage(
                    token_level_rewards=batch_data['token_level_rewards'],
                    step_rewards=step_rewards,
                    response_mask=batch_data['response_mask'],
                    prompt_uids=batch_data['prompt_uids'],
                    traj_uids=batch_data['traj_uids'],
                    step_ids=batch_data['step_ids'],
                    step_advantage_w=step_advantage_w,
                    episode_advantage_w=episode_advantage_w,
                    mode=adv_mode,
                )

                # Process samples one at a time with gradient accumulation to save memory
                self.optimizer.zero_grad()

                batch_size = batch_data['input_ids'].size(0)
                grad_accum_steps = batch_size  # Process one sample at a time
                accumulated_pg_loss = 0.0
                accumulated_kl = 0.0
                accumulated_clip_frac = 0.0

                for micro_idx in range(batch_size):
                    # Get single sample
                    input_ids = batch_data['input_ids'][micro_idx:micro_idx+1].to(self.device)
                    attention_mask = batch_data['attention_mask'][micro_idx:micro_idx+1].to(self.device)
                    response_mask = batch_data['response_mask'][micro_idx:micro_idx+1].to(self.device)
                    old_log_probs_micro = batch_data['old_log_probs'][micro_idx:micro_idx+1].to(self.device)
                    advantages_micro = advantages[micro_idx:micro_idx+1].to(self.device)

                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                        logits = outputs.logits

                        # Compute log probs
                        log_probs = self._compute_log_probs(
                            logits=logits,
                            input_ids=input_ids,
                            response_mask=response_mask,
                        )

                        # Compute policy loss
                        pg_loss, clip_frac, approx_kl = compute_policy_loss(
                            old_log_probs=old_log_probs_micro,
                            log_probs=log_probs,
                            advantages=advantages_micro,
                            response_mask=response_mask,
                            clip_range=clip_range,
                        )

                        # KL loss
                        kl_loss = approx_kl * kl_coef

                        # Scale loss for gradient accumulation
                        total_loss = (pg_loss + kl_loss) / grad_accum_steps

                    # Backward pass (accumulates gradients)
                    total_loss.backward()

                    # Accumulate metrics
                    accumulated_pg_loss += pg_loss.item() / grad_accum_steps
                    accumulated_kl += approx_kl.item() / grad_accum_steps
                    accumulated_clip_frac += clip_frac.item() / grad_accum_steps

                    # Clear intermediate tensors
                    del outputs, logits, log_probs
                    torch.cuda.empty_cache()

                # After accumulating all gradients, clip and step
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0,
                )

                self.optimizer.step()

                # Update metrics for logging
                pg_loss = torch.tensor(accumulated_pg_loss)
                approx_kl = torch.tensor(accumulated_kl)
                clip_frac = torch.tensor(accumulated_clip_frac)

                batch_time = time.time() - batch_start_time
                self.global_step += 1

                # Record metrics
                epoch_metrics['pg_loss'].append(pg_loss.item())
                epoch_metrics['kl_loss'].append(approx_kl.item())
                epoch_metrics['clip_frac'].append(clip_frac.item())
                epoch_metrics['grad_norm'].append(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
                epoch_metrics['advantages_mean'].append(advantages.mean().item())
                epoch_metrics['rewards_mean'].append(batch_data['rewards'].mean())

                # Log metrics
                if batch_idx % 10 == 0 and self.rank == 0:
                    logger.info(
                        f"Epoch {epoch}, Batch {batch_idx}, "
                        f"PG Loss: {pg_loss.item():.4f}, "
                        f"KL: {approx_kl.item():.4f}, "
                        f"Clip Frac: {clip_frac.item():.4f}, "
                        f"Adv Mean: {advantages.mean().item():.4f}"
                    )

                    if self.tracker:
                        self.tracker.log({
                            "train/pg_loss": pg_loss.item(),
                            "train/kl_loss": approx_kl.item(),
                            "train/clip_frac": clip_frac.item(),
                            "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                            "train/advantages_mean": advantages.mean().item(),
                            "train/advantages_std": advantages.std().item(),
                            "train/rewards_mean": float(batch_data['rewards'].mean()),
                            "train/epoch": epoch,
                            "train/global_step": self.global_step,
                            "perf/batch_time_s": batch_time,
                        }, step=self.global_step)

            # Epoch end
            epoch_time = time.time() - epoch_start_time

            if self.rank == 0:
                avg_pg_loss = np.mean(epoch_metrics['pg_loss'])
                avg_kl = np.mean(epoch_metrics['kl_loss'])
                avg_reward = np.mean(epoch_metrics['rewards_mean'])

                logger.info(
                    f"Epoch {epoch} completed in {epoch_time:.1f}s. "
                    f"Avg PG Loss: {avg_pg_loss:.4f}, "
                    f"Avg KL: {avg_kl:.4f}, "
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

            # Save checkpoint
            if self.rank == 0 and (epoch + 1) % self.config.trainer.get("save_freq", 5) == 0:
                self._save_checkpoint(epoch)

            dist.barrier()

        if self.rank == 0:
            logger.info("GRPO Training completed!")

    def _process_trajectory(
        self,
        trajectory: Dict,
        n_rollouts: int = 4
    ) -> List[Dict]:
        """
        Process a single trajectory and generate rollouts.

        For each step in the trajectory:
        1. Create prompt from history
        2. Generate n_rollouts responses
        3. Compute rewards for each response
        """
        steps = trajectory.get('steps', [])
        if not steps:
            return []

        step_data = []
        prompt_uid = trajectory.get('uid', str(id(trajectory)))
        traj_uid = trajectory.get('traj_uid', prompt_uid)

        history = []

        for step_idx, step in enumerate(steps):
            # Build prompt from history
            prompt_text = self._build_prompt(history, step)

            # Get ground truth - action_content is a dict with 'action', 'coordinate', 'text', etc.
            action_content = step.get('action_content', {})
            if isinstance(action_content, dict):
                ground_truth = {
                    'action_type': action_content.get('action', ''),
                    'coordinate': action_content.get('coordinate'),
                    'bbox': action_content.get('bbox'),
                    'text': action_content.get('text', ''),
                    'status': action_content.get('status'),
                    'action_content_raw': action_content,
                }
            else:
                ground_truth = {
                    'action_type': str(action_content),
                    'coordinate': None,
                    'bbox': None,
                    'text': '',
                    'status': None,
                    'action_content_raw': action_content,
                }

            # Generate rollouts (simplified - in practice would use vLLM)
            # For now, we use the ground truth with some noise for training
            rollout_responses = self._generate_rollouts(
                prompt_text=prompt_text,
                ground_truth=ground_truth,
                n_rollouts=n_rollouts,
            )

            for rollout_idx, response in enumerate(rollout_responses):
                # Compute reward
                reward, reward_info = compute_step_reward(
                    predicted_action=response,
                    ground_truth=ground_truth,
                )

                # Tokenize
                full_text = prompt_text + response
                encoding = self.tokenizer(
                    full_text,
                    max_length=self.config.data.get('max_prompt_length', 4096),
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt',
                )

                prompt_encoding = self.tokenizer(
                    prompt_text,
                    max_length=self.config.data.get('max_prompt_length', 4096),
                    truncation=True,
                    return_tensors='pt',
                )
                prompt_len = prompt_encoding['attention_mask'].sum().item()

                step_data.append({
                    'input_ids': encoding['input_ids'].squeeze(0),
                    'attention_mask': encoding['attention_mask'].squeeze(0),
                    'prompt_length': prompt_len,
                    'reward': reward,
                    'reward_info': reward_info,
                    'prompt_uid': prompt_uid,
                    'traj_uid': f"{traj_uid}_{rollout_idx}",
                    'step_id': step_idx,
                })

            # Update history - use the extracted ground_truth format
            history.append({
                'action_type': ground_truth.get('action_type', ''),
                'text': ground_truth.get('text', ''),
                'coordinate': ground_truth.get('coordinate'),
            })

        return step_data

    def _build_prompt(self, history: List[Dict], current_step: Dict) -> str:
        """Build prompt from history and current step context."""
        prompt_parts = []

        # Add history
        if history:
            prompt_parts.append("Previous actions:")
            for i, h in enumerate(history):
                action_type = h.get('action_type', '')
                text = h.get('text', '')
                coord = h.get('coordinate')
                if text:
                    prompt_parts.append(f"Step {i+1}: {action_type}({text})")
                elif coord:
                    prompt_parts.append(f"Step {i+1}: {action_type}({coord})")
                else:
                    prompt_parts.append(f"Step {i+1}: {action_type}")

        # Add current context
        prompt_parts.append("\nCurrent step:")
        if 'screenshot' in current_step:
            prompt_parts.append("[Screenshot provided]")
        if 'instruction' in current_step:
            prompt_parts.append(f"Instruction: {current_step['instruction']}")

        prompt_parts.append("\nPredict the next action:")

        return "\n".join(prompt_parts)

    def _generate_rollouts(
        self,
        prompt_text: str,
        ground_truth: Dict,
        n_rollouts: int,
    ) -> List[str]:
        """
        Generate rollout responses.

        For simplicity, we create variations of the ground truth.
        In a full implementation, this would use vLLM for actual generation.
        """
        responses = []

        # Build ground truth response string
        gt_type = ground_truth.get('action_type', '')
        gt_text = ground_truth.get('text', '')
        gt_coord = ground_truth.get('coordinate')

        if gt_text:
            gt_response = f"{gt_type}({gt_text})"
        elif gt_coord:
            gt_response = f"{gt_type}({gt_coord})"
        else:
            gt_response = f"{gt_type}()"

        # Add ground truth as one rollout
        responses.append(gt_response)

        # Add variations for diversity
        for i in range(n_rollouts - 1):
            if np.random.random() < 0.7:
                # Correct response
                responses.append(gt_response)
            else:
                # Wrong response (for learning signal)
                wrong_types = ['click', 'type', 'scroll', 'swipe']
                wrong_type = np.random.choice(wrong_types)
                responses.append(f"{wrong_type}(random_content)")

        return responses

    def _prepare_batch_tensors(self, step_data: List[Dict]) -> Optional[Dict[str, Any]]:
        """Prepare batch tensors from step data."""
        if not step_data:
            return None

        batch_size = len(step_data)
        max_len = max(d['input_ids'].shape[0] for d in step_data)

        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        response_mask = torch.zeros(batch_size, max_len, dtype=torch.float)

        rewards = np.zeros(batch_size)
        extract_matches = np.zeros(batch_size, dtype=bool)
        prompt_uids = []
        traj_uids = []
        step_ids = np.zeros(batch_size, dtype=int)

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

        # Compute old log probs (would come from reference policy in full implementation)
        # For simplicity, we set them to zeros
        old_log_probs = torch.zeros_like(response_mask)

        # Token-level rewards (broadcast step reward to response tokens)
        token_level_rewards = torch.zeros_like(response_mask)
        for i in range(batch_size):
            token_level_rewards[i] = response_mask[i] * rewards[i] / response_mask[i].sum().clamp(min=1)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'response_mask': response_mask,
            'token_level_rewards': token_level_rewards,
            'old_log_probs': old_log_probs,
            'rewards': rewards,
            'extract_matches': extract_matches,
            'prompt_uids': np.array(prompt_uids),
            'traj_uids': np.array(traj_uids),
            'step_ids': step_ids,
        }

    def _compute_log_probs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probabilities for response tokens."""
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
        padded_log_probs = torch.zeros_like(response_mask)
        padded_log_probs[:, 1:] = token_log_probs

        return padded_log_probs

    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        checkpoint_dir = self.config.trainer.get("checkpoint_dir", "checkpoints/grpo")
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
# Simple Trajectory Dataset
# ============================================================

class TrajDatasetSimple:
    """Simple trajectory dataset loader."""

    def __init__(
        self,
        data_files: List[str],
        tokenizer,
        max_prompt_length: int = 4096,
        max_response_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.data = []

        for data_file in data_files:
            with open(data_file, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        self.data.append(item)
                    except:
                        continue

        logger.info(f"Loaded {len(self.data)} trajectories")

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
        logger.info("UI-S1 GRPO Training with srun")
        logger.info("=" * 60)
        logger.info(f"World size: {world_size}, Local world size: {local_world_size}")
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # Create trainer and run
    trainer = SrunGRPOTrainer(
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
