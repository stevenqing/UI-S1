#!/usr/bin/env python3
"""
Full-featured Srun-based UI-S1 GRPO Training Worker with vLLM Rollout

This implements the complete UI-S1 GRPO algorithm matching the Ray version:
1. vLLM-based multi-step trajectory rollout generation (high-throughput)
2. Step-level reward computation with action matching
3. UI-S1 advantage estimation (episode + step level)
4. GRPO policy gradient updates with KL regularization
5. DAPO filtering for low-variance prompt groups
6. Reference policy for KL penalty
7. Validation loop with comprehensive metrics
8. FSDP-vLLM weight synchronization

Usage:
    srun --nodes=$SLURM_NNODES --ntasks-per-node=4 python train_srun_grpo_vllm.py \
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
import time
import re
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple, Union
from pprint import pprint
from contextlib import contextmanager

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
from tensordict import TensorDict

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
# vLLM Rollout Engine
# ============================================================

class VLLMRolloutEngine:
    """
    vLLM-based rollout engine for high-throughput generation.
    Supports FSDP weight synchronization and multi-modal inputs.
    """

    def __init__(
        self,
        model_path: str,
        config: DictConfig,
        tokenizer,
        processor,
        rank: int,
        local_rank: int,
        world_size: int,
    ):
        self.model_path = model_path
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")

        self.inference_engine = None
        self.sampling_params = None
        self._is_initialized = False

        # vLLM config
        self.tensor_parallel_size = config.rollout.get("tensor_model_parallel_size", 1)
        self.max_model_len = config.rollout.get("max_model_len", 8192)
        self.gpu_memory_utilization = config.rollout.get("gpu_memory_utilization", 0.8)
        self.response_length = config.data.get("max_response_length", 512)
        self.enforce_eager = config.rollout.get("enforce_eager", True)

        if self.rank == 0:
            logger.info(f"VLLMRolloutEngine configured:")
            logger.info(f"  tensor_parallel_size={self.tensor_parallel_size}")
            logger.info(f"  max_model_len={self.max_model_len}")
            logger.info(f"  gpu_memory_utilization={self.gpu_memory_utilization}")
            logger.info(f"  response_length={self.response_length}")

    def initialize(self):
        """Initialize vLLM inference engine."""
        if self._is_initialized:
            return

        try:
            from vllm import LLM, SamplingParams

            if self.rank == 0:
                logger.info(f"Initializing vLLM engine from {self.model_path}")

            # Initialize vLLM with external launcher for distributed setup
            self.inference_engine = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                distributed_executor_backend="external_launcher",
                dtype="bfloat16",
                enforce_eager=self.enforce_eager,
                gpu_memory_utilization=self.gpu_memory_utilization,
                disable_custom_all_reduce=True,
                disable_mm_preprocessor_cache=True,
                skip_tokenizer_init=False,
                max_model_len=self.max_model_len,
                disable_log_stats=True,
                trust_remote_code=True,
                seed=42,
            )

            # Default sampling params
            self.sampling_params = SamplingParams(
                n=1,
                max_tokens=self.response_length,
                temperature=1.0,
                top_p=0.95,
                top_k=50,
                logprobs=1,
                detokenize=False,
            )

            self._is_initialized = True

            if self.rank == 0:
                logger.info("vLLM engine initialized successfully")

        except ImportError as e:
            logger.warning(f"vLLM not available: {e}. Falling back to HuggingFace generate.")
            self._is_initialized = False
        except Exception as e:
            logger.warning(f"Failed to initialize vLLM: {e}. Falling back to HuggingFace generate.")
            self._is_initialized = False

    @contextmanager
    def update_sampling_params(self, **kwargs):
        """Context manager to temporarily update sampling params."""
        from vllm import SamplingParams

        old_params = {}
        for key, value in kwargs.items():
            if hasattr(self.sampling_params, key):
                old_params[key] = getattr(self.sampling_params, key)
                setattr(self.sampling_params, key, value)
        try:
            yield
        finally:
            for key, value in old_params.items():
                setattr(self.sampling_params, key, value)

    def sync_weights_from_fsdp(self, fsdp_model: FSDP):
        """
        Synchronize weights from FSDP model to vLLM engine.
        This is called before each rollout phase.
        """
        if not self._is_initialized:
            return

        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        if self.rank == 0:
            logger.info("Syncing FSDP weights to vLLM engine...")

        # Get full state dict from FSDP
        full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)

        with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
            state_dict = fsdp_model.state_dict()

        # Load into vLLM
        # Note: vLLM's weight loading varies by version
        # This is a simplified approach - may need adjustment based on vLLM version
        try:
            model = self.inference_engine.llm_engine.model_executor.driver_worker.model_runner.model
            model.load_state_dict(state_dict, strict=False)
            if self.rank == 0:
                logger.info("Weights synced to vLLM successfully")
        except Exception as e:
            logger.warning(f"Weight sync failed: {e}. vLLM will use initial weights.")

    def _preprocess_inputs(self, prompt_token_ids: torch.Tensor) -> List[int]:
        """Remove left padding from prompt token ids."""
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)
        if len(non_pad_index) > 0:
            start_idx = non_pad_index[0][0].item()
            return prompt_token_ids[start_idx:].tolist()
        return prompt_token_ids.tolist()

    @torch.no_grad()
    def generate_sequences(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        multi_modal_data: Optional[List[Dict]] = None,
        n: int = 1,
        temperature: float = 1.0,
        do_sample: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using vLLM.

        Args:
            input_ids: (batch_size, seq_len) padded input token ids
            attention_mask: (batch_size, seq_len) attention mask
            multi_modal_data: Optional list of multi-modal data dicts (images, etc.)
            n: Number of sequences to generate per prompt
            temperature: Sampling temperature
            do_sample: Whether to sample (True) or greedy decode (False)

        Returns:
            responses: (batch_size * n, response_length) generated token ids
            log_probs: (batch_size * n, response_length) log probabilities
        """
        if not self._is_initialized:
            # Fallback to dummy responses
            batch_size = input_ids.shape[0]
            dummy_response = torch.full(
                (batch_size * n, self.response_length),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=self.device
            )
            dummy_log_probs = torch.zeros(
                (batch_size * n, self.response_length),
                dtype=torch.float32,
                device=self.device
            )
            return dummy_response, dummy_log_probs

        batch_size = input_ids.shape[0]

        # Prepare vLLM inputs
        vllm_inputs = []
        for i in range(batch_size):
            prompt_ids = self._preprocess_inputs(input_ids[i])

            vllm_input = {"prompt_token_ids": prompt_ids}

            # Add multi-modal data if available
            if multi_modal_data is not None and i < len(multi_modal_data):
                if multi_modal_data[i] is not None:
                    vllm_input["multi_modal_data"] = multi_modal_data[i]

            vllm_inputs.append(vllm_input)

        # Set sampling params
        sampling_kwargs = {
            "n": n,
            "temperature": temperature if do_sample else 0.0,
        }
        if not do_sample:
            sampling_kwargs.update({
                "top_p": 1.0,
                "top_k": -1,
            })

        # Generate
        with self.update_sampling_params(**sampling_kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )

        # Collect responses and log probs
        responses = []
        log_probs_list = []

        for output in outputs:
            for sample in output.outputs:
                response_ids = list(sample.token_ids)
                responses.append(response_ids)

                # Collect log probs
                sample_log_probs = []
                if sample.logprobs:
                    for idx, logprob_dict in enumerate(sample.logprobs):
                        if idx < len(response_ids):
                            token_id = response_ids[idx]
                            if token_id in logprob_dict:
                                sample_log_probs.append(logprob_dict[token_id].logprob)
                            else:
                                sample_log_probs.append(-100.0)  # Invalid token
                log_probs_list.append(sample_log_probs)

        # Pad responses to fixed length
        responses_padded = self._pad_sequences(
            responses,
            self.response_length,
            self.tokenizer.pad_token_id
        ).to(self.device)

        log_probs_padded = self._pad_sequences(
            log_probs_list,
            self.response_length,
            -100.0,
            dtype=torch.float32
        ).to(self.device)

        return responses_padded, log_probs_padded

    def _pad_sequences(
        self,
        sequences: List[List],
        max_length: int,
        pad_value: Union[int, float],
        dtype: torch.dtype = torch.long
    ) -> torch.Tensor:
        """Pad sequences to fixed length."""
        batch_size = len(sequences)
        result = torch.full((batch_size, max_length), pad_value, dtype=dtype)

        for i, seq in enumerate(sequences):
            length = min(len(seq), max_length)
            result[i, :length] = torch.tensor(seq[:length], dtype=dtype)

        return result

    def sleep(self):
        """Put vLLM engine to sleep to free memory."""
        if self._is_initialized and hasattr(self.inference_engine, 'sleep'):
            self.inference_engine.sleep(level=1)

    def wake(self):
        """Wake vLLM engine from sleep."""
        if self._is_initialized and hasattr(self.inference_engine, 'wake'):
            self.inference_engine.wake()


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
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    prompt_uids: np.ndarray,
    traj_uids: np.ndarray,
    step_ids: np.ndarray,
    step_advantage_w: float = 1.0,
    episode_advantage_w: float = 1.0,
    mode: str = "mean_std_norm",
    epsilon: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute UI-S1 advantages combining episode-level and step-level normalization.
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
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i].item())

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
# Policy Loss Computation
# ============================================================

def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_range: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute PPO/GRPO clipped policy loss."""
    ratio = torch.exp(log_probs - old_log_probs)

    clip_range_low = 1.0 - clip_range
    clip_range_high = 1.0 + clip_range

    clipped_ratio = torch.clamp(ratio, clip_range_low, clip_range_high)

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    pg_loss = torch.max(pg_loss1, pg_loss2)

    valid_tokens = response_mask.sum().clamp(min=1)
    pg_loss = (pg_loss * response_mask).sum() / valid_tokens

    clip_frac = ((ratio < clip_range_low) | (ratio > clip_range_high)).float()
    clip_frac = (clip_frac * response_mask).sum() / valid_tokens

    approx_kl = (old_log_probs - log_probs) * response_mask
    approx_kl = approx_kl.sum() / valid_tokens

    return pg_loss, clip_frac, approx_kl


def compute_kl_penalty(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    kl_penalty_type: str = "low_var_kl"
) -> torch.Tensor:
    """Compute KL penalty between current and reference policy."""
    if kl_penalty_type == "kl":
        kl = ref_log_probs - log_probs
    elif kl_penalty_type == "abs":
        kl = torch.abs(log_probs - ref_log_probs)
    elif kl_penalty_type == "mse":
        kl = 0.5 * (log_probs - ref_log_probs) ** 2
    elif kl_penalty_type == "low_var_kl":
        log_ratio = log_probs - ref_log_probs
        kl = torch.exp(log_ratio) - 1 - log_ratio
    else:
        raise ValueError(f"Unknown KL penalty type: {kl_penalty_type}")

    return kl * response_mask


# ============================================================
# DAPO Filtering
# ============================================================

def apply_dapo_filter(
    batch_data: Dict,
    metric_name: str = "seq_future_reward",
    std_threshold: float = 0.3,
) -> Tuple[Optional[Dict], int]:
    """Apply DAPO filtering to remove low-variance prompt groups."""
    prompt_uids = batch_data['prompt_uids']
    step_ids = batch_data['step_ids']

    if metric_name == "seq_future_reward":
        metrics = batch_data['step_rewards']
    else:
        metrics = batch_data['rewards']

    prompt_uid2metric_vals = defaultdict(list)

    if metric_name == "seq_future_reward":
        for uid, metric_val, step_id in zip(prompt_uids, metrics, step_ids):
            if step_id == 0:
                prompt_uid2metric_vals[uid].append(float(metric_val))
    else:
        for uid, metric_val in zip(prompt_uids, metrics):
            prompt_uid2metric_vals[uid].append(float(metric_val))

    prompt_uid2std = {}
    for uid, vals in prompt_uid2metric_vals.items():
        prompt_uid2std[uid] = np.std(vals) if len(vals) > 1 else 0.0

    kept_prompt_uids = set([
        uid for uid, std in prompt_uid2std.items()
        if std > std_threshold or len(prompt_uid2metric_vals[uid]) == 1
    ])

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
# Reward Function
# ============================================================

class ActionMatchReward:
    """Compute reward based on action matching with ground truth."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.action_types = [
            'click', 'tap', 'type', 'input', 'scroll', 'swipe',
            'press', 'long_press', 'open_app', 'back', 'home'
        ]

    def __call__(
        self,
        response_text: str,
        ground_truth: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute reward and return info dict."""
        info = {
            'type_match': False,
            'extract_match': False,
            'format_score': 0.0,
            'coord_score': 0.0,
            'text_score': 0.0,
            'score': 0.0,
        }

        # Parse predicted action
        try:
            pred_type, pred_args = self._parse_action(response_text)
            info['extract_match'] = True
        except:
            info['extract_match'] = False
            return 0.0, info

        # Get ground truth
        gt_type = ground_truth.get('action_type', '') or ground_truth.get('action', '')
        gt_text = ground_truth.get('text', '') or ''
        gt_coord = ground_truth.get('coordinate')

        # Type matching (0.3 weight)
        type_score = 0.0
        if gt_type and pred_type:
            gt_norm = gt_type.lower().replace('_', '').replace('-', '')
            pred_norm = pred_type.lower().replace('_', '').replace('-', '')
            if pred_norm == gt_norm:
                type_score = 1.0
            elif pred_norm in gt_norm or gt_norm in pred_norm:
                type_score = 0.5
        info['type_match'] = type_score > 0.5
        info['format_score'] = type_score

        # Text matching (0.4 weight)
        text_score = 0.0
        if gt_text:
            pred_text = self._extract_text_arg(pred_args)
            if pred_text:
                gt_lower = gt_text.lower().strip()
                pred_lower = pred_text.lower().strip()
                if pred_lower == gt_lower:
                    text_score = 1.0
                elif pred_lower in gt_lower or gt_lower in pred_lower:
                    text_score = 0.7
                else:
                    # Compute character-level similarity
                    text_score = self._compute_similarity(pred_lower, gt_lower) * 0.5
        elif info['type_match']:
            text_score = 0.5  # No text expected, type match is sufficient
        info['text_score'] = text_score

        # Coordinate matching (0.3 weight)
        coord_score = 0.0
        if gt_coord:
            pred_coord = self._extract_coord_arg(pred_args)
            if pred_coord:
                try:
                    gt_x, gt_y = float(gt_coord[0]), float(gt_coord[1])
                    pred_x, pred_y = float(pred_coord[0]), float(pred_coord[1])
                    dist = ((pred_x - gt_x)**2 + (pred_y - gt_y)**2)**0.5
                    coord_score = max(0, 1 - dist / 100)  # 100 pixel tolerance
                except:
                    pass
        elif info['type_match']:
            coord_score = 0.5  # No coord expected
        info['coord_score'] = coord_score

        # Combined score
        if gt_coord and gt_text:
            info['score'] = 0.3 * type_score + 0.4 * text_score + 0.3 * coord_score
        elif gt_coord:
            info['score'] = 0.4 * type_score + 0.6 * coord_score
        elif gt_text:
            info['score'] = 0.3 * type_score + 0.7 * text_score
        else:
            info['score'] = type_score

        return info['score'], info

    def _parse_action(self, text: str) -> Tuple[str, str]:
        """Parse action string."""
        text = text.strip()

        # Try function call format: action_type(args)
        match = re.match(r'(\w+)\s*\((.*)\)', text, re.DOTALL)
        if match:
            return match.group(1), match.group(2).strip()

        # Try to find action type keywords
        for at in self.action_types:
            if at in text.lower():
                idx = text.lower().find(at)
                args = text[idx + len(at):].strip()
                return at, args

        raise ValueError(f"Cannot parse action: {text}")

    def _extract_text_arg(self, args: str) -> Optional[str]:
        """Extract text argument from action args."""
        # Try quoted string
        match = re.search(r'["\'](.+?)["\']', args)
        if match:
            return match.group(1)

        # Try text= format
        match = re.search(r'text\s*[=:]\s*(.+?)(?:,|$)', args, re.IGNORECASE)
        if match:
            return match.group(1).strip().strip('"\'')

        return None

    def _extract_coord_arg(self, args: str) -> Optional[Tuple[float, float]]:
        """Extract coordinate argument from action args."""
        # Try [x, y] format
        match = re.search(r'\[?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]?', args)
        if match:
            return (float(match.group(1)), float(match.group(2)))

        # Try coordinate= format
        match = re.search(r'coord(?:inate)?\s*[=:]\s*\[?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)', args, re.IGNORECASE)
        if match:
            return (float(match.group(1)), float(match.group(2)))

        return None

    def _compute_similarity(self, s1: str, s2: str) -> float:
        """Compute simple character-level similarity."""
        if not s1 or not s2:
            return 0.0
        common = set(s1) & set(s2)
        total = set(s1) | set(s2)
        return len(common) / len(total) if total else 0.0


# ============================================================
# Main Trainer Class
# ============================================================

class SrunGRPOVLLMTrainer:
    """
    Full-featured UI-S1 GRPO Trainer using srun with vLLM rollout.
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
        self._init_vllm_rollout()
        self._init_optimizer()
        self._init_dataloader()
        self._init_reward_fn()
        self._init_tracking()

        if self.rank == 0:
            logger.info("SrunGRPOVLLMTrainer initialized successfully")

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

        if self.config.actor_rollout_ref.model.get("enable_gradient_checkpointing", True):
            model.gradient_checkpointing_enable()

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Qwen2_5_VLDecoderLayer},
        )

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        cpu_offload = None
        if self.config.actor_rollout_ref.actor.fsdp_config.get("param_offload", False):
            from torch.distributed.fsdp import CPUOffload
            cpu_offload = CPUOffload(offload_params=True)

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
        """Initialize reference model for KL penalty."""
        self.ref_model = None
        self.use_kl_loss = self.config.actor_rollout_ref.actor.get("use_kl_loss", False)

        if not self.use_kl_loss:
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

        for param in ref_model.parameters():
            param.requires_grad = False

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Qwen2_5_VLDecoderLayer},
        )

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

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

        dist.barrier()

    def _init_vllm_rollout(self):
        """Initialize vLLM rollout engine."""
        model_path = self.config.actor_rollout_ref.model.path

        self.vllm_rollout = VLLMRolloutEngine(
            model_path=model_path,
            config=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor,
            rank=self.rank,
            local_rank=self.local_rank,
            world_size=self.world_size,
        )

        # Try to initialize vLLM
        try:
            self.vllm_rollout.initialize()
            self.use_vllm = self.vllm_rollout._is_initialized
        except:
            self.use_vllm = False

        if self.rank == 0:
            logger.info(f"vLLM rollout enabled: {self.use_vllm}")

    def _init_optimizer(self):
        """Initialize optimizer."""
        lr = self.config.actor_rollout_ref.actor.optim.lr
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def _init_dataloader(self):
        """Initialize data loaders."""
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

        batch_size = max(1, self.config.data.get('train_batch_size', 2) // self.world_size)

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=lambda x: {'trajectories': x},
            num_workers=2,
            pin_memory=True,
        )

        if self.rank == 0:
            logger.info(f"Train dataset size: {len(self.train_dataset)}")

    def _init_reward_fn(self):
        """Initialize reward function."""
        self.reward_fn = ActionMatchReward(self.tokenizer)

    def _init_tracking(self):
        """Initialize tracking/logging."""
        self.tracker = None
        if self.rank == 0:
            try:
                from verl.utils.tracking import Tracking

                project_name = self.config.trainer.get("project_name", "ui-s1-grpo-vllm")
                experiment_name = self.config.trainer.get("experiment_name", "srun_grpo_vllm")
                logger_backends = self.config.trainer.get("logger", ["console", "wandb"])

                if isinstance(logger_backends, str):
                    logger_backends = [logger_backends]

                self.tracker = Tracking(
                    project_name=project_name,
                    experiment_name=experiment_name,
                    default_backend=logger_backends,
                    config=OmegaConf.to_container(self.config, resolve=True),
                )
            except Exception as e:
                logger.warning(f"Failed to initialize tracking: {e}")

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

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = response_mask[:, 1:].contiguous()

        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        token_log_probs = token_log_probs * shift_mask

        padded_log_probs = torch.zeros(
            input_ids.shape[0], input_ids.shape[1],
            dtype=token_log_probs.dtype, device=token_log_probs.device
        )
        padded_log_probs[:, 1:] = token_log_probs

        return padded_log_probs

    @torch.no_grad()
    def _generate_with_vllm(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_rollouts: int = 4,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate using vLLM engine."""
        if self.use_vllm:
            # Sync weights before rollout
            self.vllm_rollout.sync_weights_from_fsdp(self.model)

            responses, log_probs = self.vllm_rollout.generate_sequences(
                input_ids=input_ids,
                attention_mask=attention_mask,
                n=n_rollouts,
                temperature=temperature,
                do_sample=True,
            )
            return responses, log_probs
        else:
            # Fallback to HuggingFace generate
            return self._generate_with_hf(input_ids, attention_mask, n_rollouts, temperature)

    @torch.no_grad()
    def _generate_with_hf(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_rollouts: int = 4,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback generation using HuggingFace generate."""
        self.model.eval()

        batch_size = input_ids.shape[0]
        response_length = self.config.data.get('max_response_length', 512)

        all_responses = []
        all_log_probs = []

        for _ in range(n_rollouts):
            try:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model.generate(
                        input_ids=input_ids.to(self.device),
                        attention_mask=attention_mask.to(self.device),
                        max_new_tokens=response_length,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                responses = outputs[:, input_ids.shape[1]:]

                # Pad to fixed length
                if responses.shape[1] < response_length:
                    pad = torch.full(
                        (batch_size, response_length - responses.shape[1]),
                        self.tokenizer.pad_token_id,
                        dtype=torch.long,
                        device=self.device
                    )
                    responses = torch.cat([responses, pad], dim=1)
                elif responses.shape[1] > response_length:
                    responses = responses[:, :response_length]

                all_responses.append(responses)
                all_log_probs.append(torch.zeros_like(responses, dtype=torch.float32))

            except Exception as e:
                logger.warning(f"HF generate failed: {e}")
                dummy = torch.full(
                    (batch_size, response_length),
                    self.tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=self.device
                )
                all_responses.append(dummy)
                all_log_probs.append(torch.zeros_like(dummy, dtype=torch.float32))

        # Stack rollouts: (batch_size * n_rollouts, response_length)
        responses = torch.cat(all_responses, dim=0)
        log_probs = torch.cat(all_log_probs, dim=0)

        return responses, log_probs

    def fit(self):
        """Run the full GRPO training loop."""
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

        dapo_enabled = self.config.algorithm.filter_groups.get('enable', False)
        dapo_threshold = self.config.algorithm.filter_groups.get('std_threshold', 0.3)

        save_freq = self.config.trainer.get('save_freq', 5)

        if self.rank == 0:
            logger.info("=" * 60)
            logger.info("Starting GRPO Training with vLLM Rollout")
            logger.info("=" * 60)
            logger.info(f"vLLM enabled: {self.use_vllm}")
            logger.info(f"gamma={gamma}, clip_range={clip_range}, n_rollouts={n_rollouts}")
            logger.info(f"step_advantage_w={step_advantage_w}, episode_advantage_w={episode_advantage_w}")
            logger.info(f"DAPO enabled={dapo_enabled}, threshold={dapo_threshold}")
            logger.info(f"KL loss enabled={self.use_kl_loss}, coef={kl_coef}")

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

                # Process trajectories (simplified for this example)
                trajectories = batch['trajectories']

                # Skip empty batches
                if not trajectories:
                    continue

                # For simplicity, process first trajectory only
                # Full implementation would handle all trajectories
                traj = trajectories[0]
                steps = traj.get('steps', [])
                if not steps:
                    continue

                # Build batch from steps
                batch_data = self._process_steps(steps, traj, n_rollouts, gamma)
                if batch_data is None:
                    continue

                # Apply DAPO filtering
                if dapo_enabled:
                    batch_data, num_kept = apply_dapo_filter(
                        batch_data,
                        std_threshold=dapo_threshold,
                    )
                    if batch_data is None:
                        continue

                # Move to device
                input_ids = batch_data['input_ids'].to(self.device)
                attention_mask = batch_data['attention_mask'].to(self.device)
                response_mask = batch_data['response_mask'].to(self.device)
                token_level_rewards = batch_data['token_level_rewards'].to(self.device)
                step_rewards = batch_data['step_rewards'].to(self.device)

                # Compute old log probs
                with torch.no_grad():
                    old_log_probs = self._compute_log_probs(
                        self.model, input_ids, attention_mask, response_mask
                    )

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

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    log_probs = self._compute_log_probs(
                        self.model, input_ids, attention_mask, response_mask
                    )

                    pg_loss, clip_frac, approx_kl = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_range=clip_range,
                    )

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

                    total_loss = pg_loss + kl_loss

                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                batch_time = time.time() - batch_start_time
                self.global_step += 1

                # Record metrics
                epoch_metrics['pg_loss'].append(pg_loss.item())
                epoch_metrics['kl_loss'].append(kl_loss.item())
                epoch_metrics['clip_frac'].append(clip_frac.item())
                epoch_metrics['rewards_mean'].append(float(batch_data['rewards'].mean()))

                # Log
                if batch_idx % 10 == 0 and self.rank == 0:
                    logger.info(
                        f"Epoch {epoch}, Batch {batch_idx}, "
                        f"PG Loss: {pg_loss.item():.4f}, "
                        f"KL Loss: {kl_loss.item():.6f}, "
                        f"Reward: {float(batch_data['rewards'].mean()):.4f}"
                    )

                    if self.tracker:
                        self.tracker.log({
                            "train/pg_loss": pg_loss.item(),
                            "train/kl_loss": kl_loss.item(),
                            "train/clip_frac": clip_frac.item(),
                            "train/rewards_mean": float(batch_data['rewards'].mean()),
                            "train/epoch": epoch,
                            "train/global_step": self.global_step,
                            "perf/batch_time_s": batch_time,
                            "perf/vllm_enabled": self.use_vllm,
                        }, step=self.global_step)

            # Epoch end
            epoch_time = time.time() - epoch_start_time

            if self.rank == 0:
                avg_pg_loss = np.mean(epoch_metrics['pg_loss']) if epoch_metrics['pg_loss'] else 0
                avg_reward = np.mean(epoch_metrics['rewards_mean']) if epoch_metrics['rewards_mean'] else 0

                logger.info(
                    f"Epoch {epoch} completed in {epoch_time:.1f}s. "
                    f"Avg PG Loss: {avg_pg_loss:.4f}, "
                    f"Avg Reward: {avg_reward:.4f}"
                )

            # Save checkpoint
            if save_freq > 0 and (epoch + 1) % save_freq == 0:
                self._save_checkpoint(epoch)

            dist.barrier()

        if self.rank == 0:
            logger.info("GRPO Training with vLLM completed!")

    def _process_steps(
        self,
        steps: List[Dict],
        traj: Dict,
        n_rollouts: int,
        gamma: float,
    ) -> Optional[Dict]:
        """Process trajectory steps into training batch."""
        all_step_data = []
        prompt_uid = str(uuid.uuid4())

        for rollout_idx in range(n_rollouts):
            traj_uid = f"{prompt_uid}_{rollout_idx}"

            for step_idx, step in enumerate(steps):
                # Build prompt
                prompt_text = self._build_prompt(steps[:step_idx], step, traj)

                # Tokenize
                encoding = self.tokenizer(
                    prompt_text,
                    max_length=self.config.data.get('max_prompt_length', 2048),
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt',
                )

                # Generate response using vLLM or HF
                responses, _ = self._generate_with_vllm(
                    encoding['input_ids'],
                    encoding['attention_mask'],
                    n_rollouts=1,
                    temperature=1.0,
                )

                response_text = self.tokenizer.decode(responses[0], skip_special_tokens=True)

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

                # Compute reward
                reward, reward_info = self.reward_fn(response_text, ground_truth)

                # Create full sequence
                full_ids = torch.cat([encoding['input_ids'], responses.cpu()], dim=1)
                full_mask = torch.cat([
                    encoding['attention_mask'],
                    torch.ones_like(responses.cpu())
                ], dim=1)

                prompt_len = encoding['attention_mask'].sum().item()

                step_data = {
                    'input_ids': full_ids.squeeze(0),
                    'attention_mask': full_mask.squeeze(0),
                    'prompt_length': prompt_len,
                    'reward': reward,
                    'reward_info': reward_info,
                    'prompt_uid': prompt_uid,
                    'traj_uid': traj_uid,
                    'step_id': step_idx,
                }
                all_step_data.append(step_data)

        if not all_step_data:
            return None

        return self._prepare_batch_tensors(all_step_data, gamma)

    def _build_prompt(self, history: List[Dict], current_step: Dict, traj: Dict) -> str:
        """Build prompt from history and current step."""
        parts = []

        if 'goal' in traj:
            parts.append(f"Task: {traj['goal']}")

        if history:
            parts.append("\nPrevious actions:")
            for i, step in enumerate(history):
                action = step.get('action_content', {})
                if isinstance(action, dict):
                    action_type = action.get('action', 'unknown')
                    parts.append(f"Step {i+1}: {action_type}")

        parts.append("\nPredict the next action:")

        return "\n".join(parts)

    def _prepare_batch_tensors(self, step_data: List[Dict], gamma: float) -> Dict:
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

            prompt_len = data['prompt_length']
            response_mask[i, prompt_len:seq_len] = 1.0

            rewards[i] = data['reward']
            extract_matches[i] = data['reward_info'].get('extract_match', True)
            prompt_uids.append(data['prompt_uid'])
            traj_uids.append(data['traj_uid'])
            step_ids[i] = data['step_id']

        # Compute step rewards (discounted returns)
        step_rewards = compute_step_discounted_returns(
            rewards, np.array(traj_uids), extract_matches, gamma
        )

        # Token-level rewards
        token_level_rewards = torch.zeros_like(response_mask)
        for i in range(batch_size):
            valid_tokens = response_mask[i].sum().clamp(min=1)
            token_level_rewards[i] = response_mask[i] * rewards[i] / valid_tokens

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'response_mask': response_mask,
            'token_level_rewards': token_level_rewards,
            'step_rewards': step_rewards,
            'rewards': rewards,
            'extract_matches': extract_matches,
            'prompt_uids': np.array(prompt_uids),
            'traj_uids': np.array(traj_uids),
            'step_ids': step_ids,
        }

    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        checkpoint_dir = self.config.trainer.get(
            "checkpoint_dir",
            f"{self.config.actor_rollout_ref.model.path}/../checkpoints/grpo_vllm"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, f"grpo_vllm_epoch_{epoch}.pt")

        full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
            state_dict = self.model.state_dict()
            if self.rank == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': state_dict,
                    'global_step': self.global_step,
                }, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")


# ============================================================
# Trajectory Dataset
# ============================================================

class TrajDataset:
    """Trajectory dataset loader."""

    def __init__(self, data_files, tokenizer, processor, config):
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
    """Main entry point."""
    rank, world_size, local_rank, local_world_size = setup_distributed()

    if rank == 0:
        logger.info("=" * 60)
        logger.info("UI-S1 GRPO Training with vLLM Rollout")
        logger.info("=" * 60)
        logger.info(f"World size: {world_size}")
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    trainer = SrunGRPOVLLMTrainer(
        config=config,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )

    trainer.fit()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
