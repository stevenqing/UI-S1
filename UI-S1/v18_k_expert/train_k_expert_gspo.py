#!/usr/bin/env python3
"""V18 GUI-360 On-Policy RL with K-Expert Cooperative LoRA.

Adapted from V15 train_trajectory_gspo.py with:
  - KExpertCooperativeVLMWrapper instead of IterativeCooperativeVLMWrapper
  - Softmax routing (K-way) instead of sigmoid (binary)
  - Diversity loss (pairwise cosine similarity between expert outputs)
  - New args: --num_experts, --comm_topology, --diversity_weight
  - Route LR default lowered (5e-4) since softmax is more sensitive
  - Logging: routing entropy, per-expert usage, diversity loss

Algorithm (same as V15):
  For each episode with T steps:
    1. Roll out K=8 full trajectories on-policy (stop on first error)
    2. For each step t:
       a. Collect active trajectories (those that reached step t)
       b. Compute group-relative advantage from step rewards
       c. Token-level PPO loss for each active trajectory
    3. Trajectory credit assignment is implicit

Usage:
  srun --ntasks-per-node=1 bash -c '
    torchrun --nproc_per_node=4 --nnodes=$SLURM_NNODES \\
      --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR \\
      v18_k_expert/train_k_expert_gspo.py --model_path ... --train_data ... --output_dir ...
  '
"""

import argparse
import datetime
import json
import os
import sys
import time
import traceback
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image

# Set NCCL environment variables BEFORE any NCCL operations
os.environ.setdefault("NCCL_SOCKET_IFNAME", "hsn0")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "hsn0")
os.environ.setdefault("NCCL_NET", "Socket")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_LEVEL", "LOC")
os.environ.setdefault("NCCL_CROSS_NIC", "1")

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("v18_k_expert_gspo")

# Project imports
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from v12_gui_360.reward import (
    compute_step_reward,
    parse_action_from_text,
    compute_trajectory_advantages,
    compute_sp_scores,
)


# -- Prompt templates ------------------------------------------------------

USER_PROMPT_TEMPLATE = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

First, explain your reasoning process—describe how you analyze the screenshot, understand the current state, and determine what action should be taken next based on the instruction and previous actions.

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished, you can output status as "FINISH" and take no action. Like:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time. If the instruction could apply to multiple elements, choose the most relevant one based on the context provided by the screenshot and previous actions.
"""

SUPPORTED_ACTIONS = """<action>
- click
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to click at.
    - button: str, One of 'left', 'right', 'middle' or 'x' (Default: 'left')
    - double: bool, Whether to perform a double click (Default: False)
    - pressed: str|None, Keyboard key to press while clicking (Default: None)
  - Example: click(coordinate=[100, 100], button='left', double=False, pressed=None)
- type
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to type at.
    - keys: str, The key to input.
    - clear_current_text: bool, Whether to clear the current text (Default: False)
    - control_focus: bool, Whether to focus on selected control before typing (Default: True)
  - Example: type(coordinate=[100, 100], keys='Hello')
- drag
  - Args:
    - start_coordinate: [x, y], where the drag starts.
    - end_coordinate: [x, y], where the drag ends.
    - button: str, 'left' or 'right' (Default: 'left')
    - duration: float, Duration in seconds (Default: 1.0)
  - Example: drag(start_coordinate=[100, 100], end_coordinate=[200, 200])
- wheel_mouse_input
  - Args:
    - coordinate: [x, y], position on the screen to scroll.
    - wheel_dist: int, Wheel notches. Positive=up, negative=down.
  - Example: wheel_mouse_input(coordinate=[100, 100], wheel_dist=-5)
</action>"""


# -- Dataset ---------------------------------------------------------------

class EpisodeDataset(Dataset):
    """Load episode-grouped JSONL data for trajectory RL."""

    def __init__(self, jsonl_path: str, max_episodes: int = 0):
        self.episodes = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                valid = True
                for step in ep.get("steps", []):
                    if not os.path.exists(step.get("screenshot", "")):
                        valid = False
                        break
                if valid and ep.get("steps"):
                    self.episodes.append(ep)

        if 0 < max_episodes < len(self.episodes):
            rng = np.random.RandomState(42)
            idx = rng.choice(len(self.episodes), max_episodes, replace=False)
            self.episodes = [self.episodes[i] for i in sorted(idx)]

        logger.info(f"Loaded {len(self.episodes)} episodes from {jsonl_path}")

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]


class TSortedDistributedSampler(Sampler):
    """DistributedSampler that groups episodes by step count (T).

    Standard DistributedSampler with random shuffle causes NCCL timeouts:
    different ranks get episodes with wildly different T (1 vs 56 steps),
    so fast ranks wait at all_reduce while slow ranks are still generating.

    This sampler sorts episodes by T, then assigns consecutive blocks to
    ranks.  Ranks processing similar-T episodes finish at similar times,
    reducing desync to a few minutes (vs hours with random assignment).

    Within each epoch, blocks of similar-T episodes are shuffled to add
    variety, but the T-grouping property is preserved.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, seed: int = 42):
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        # Sort indices by episode step count
        self.step_counts = [len(ep["steps"]) for ep in dataset.episodes]
        self.sorted_indices = sorted(range(len(dataset)), key=lambda i: self.step_counts[i])

        # Ensure total size is divisible by num_replicas
        self.total_size = (len(dataset) // num_replicas) * num_replicas
        self.num_samples = self.total_size // num_replicas

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        # Shuffle within T-groups: divide sorted indices into blocks,
        # shuffle blocks (not individual indices) to preserve T-grouping.
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = list(self.sorted_indices[:self.total_size])
        block_size = self.num_replicas  # one element per rank per block
        blocks = [indices[i:i + block_size] for i in range(0, len(indices), block_size)]
        perm = torch.randperm(len(blocks), generator=g).tolist()
        indices = []
        for bi in perm:
            indices.extend(blocks[bi])

        # Standard distributed shard: rank r gets indices[r::num_replicas]
        rank_indices = indices[self.rank::self.num_replicas]
        return iter(rank_indices)

    def __len__(self):
        return self.num_samples


# -- Prompt formatting -----------------------------------------------------

def format_pred_action_for_history(pred_action: Dict, step_id: int) -> str:
    """Convert model's predicted action dict to history format."""
    atype = pred_action.get("action", "click")
    coord = pred_action.get("coordinate")

    if atype == "click" and coord:
        return f"Step {step_id}: click(coordinate=[{int(coord[0])}, {int(coord[1])}])"
    elif atype == "type":
        text = pred_action.get("text", "")[:30]
        if coord:
            return f"Step {step_id}: type(coordinate=[{int(coord[0])}, {int(coord[1])}], keys='{text}')"
        return f"Step {step_id}: type(keys='{text}')"
    elif atype in ("swipe", "drag"):
        start = pred_action.get("coordinate")
        end = pred_action.get("endCoordinate")
        if start and end:
            return (f"Step {step_id}: drag(start_coordinate=[{int(start[0])}, {int(start[1])}], "
                    f"end_coordinate=[{int(end[0])}, {int(end[1])}])")
        return f"Step {step_id}: drag()"
    elif atype == "wheel_mouse_input":
        if coord:
            return f"Step {step_id}: wheel_mouse_input(coordinate=[{int(coord[0])}, {int(coord[1])}])"
        return f"Step {step_id}: wheel_mouse_input()"
    return f"Step {step_id}: {atype}()"


def build_step_messages(goal: str, action_history: List[str],
                        image_path: str) -> list:
    history_text = "\n".join(action_history) if action_history else "None"
    prompt_text = USER_PROMPT_TEMPLATE.format(
        instruction=goal,
        history=history_text,
        actions=SUPPORTED_ACTIONS,
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


# -- KL penalty -----------------------------------------------------------

def compute_kl_penalty(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Low-variance KL estimator: exp(r) - 1 - r."""
    log_ratio = log_probs - ref_log_probs
    kl = (torch.exp(log_ratio) - 1 - log_ratio) * response_mask
    valid_tokens = response_mask.sum().clamp(min=1)
    return kl.sum() / valid_tokens


# -- Trainer ---------------------------------------------------------------

class V18KExpertGSPOTrainer:
    def __init__(self, args):
        self.args = args
        self.global_step = 0
        self.resume_step = 0

        self._setup_distributed()
        self._setup_model()
        self._load_resume_checkpoint()
        self._setup_data()
        self._setup_optimizer()

    # -- distributed -------------------------------------------------------

    def _setup_distributed(self):
        rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
        world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))

        torch.cuda.set_device(local_rank)
        self.device = torch.device(f"cuda:{local_rank}")

        if not dist.is_initialized():
            master_addr = os.environ.get("MASTER_ADDR", "localhost")
            master_port = os.environ.get("MASTER_PORT", "29500")
            os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = master_port
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["RANK"] = str(rank)

            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=rank,
                timeout=datetime.timedelta(seconds=3600),
                device_id=torch.device(f"cuda:{local_rank}"),
            )

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = local_rank

        dist.barrier()
        if self.rank == 0:
            import socket
            logger.info(
                f"Distributed ready: rank={self.rank} world={self.world_size} "
                f"host={socket.gethostname()}"
            )

    # -- model (V18 KExpertCooperativeVLMWrapper) ---------------------------

    def _setup_model(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from v18_k_expert.k_expert_cooperative_wrapper import KExpertCooperativeVLMWrapper

        args = self.args

        self.processor = AutoProcessor.from_pretrained(args.model_path)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        if args.image_max_pixels > 0:
            self.processor.image_processor.max_pixels = args.image_max_pixels
        self.pad_id = self.processor.tokenizer.pad_token_id

        if self.rank == 0:
            logger.info(f"Loading base model: {args.model_path}")
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.model = KExpertCooperativeVLMWrapper(
            base_model=base_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=list(args.target_modules),
            balance_weight=args.balance_weight,
            diversity_weight=args.diversity_weight,
            num_comm_rounds=args.num_comm_rounds,
            num_experts=args.num_experts,
            comm_topology=args.comm_topology,
        )

        # Load SFT checkpoint if provided
        if args.sft_checkpoint:
            if self.rank == 0:
                logger.info(f"Loading SFT checkpoint: {args.sft_checkpoint}")
            self.model.load_cooperative(args.sft_checkpoint, device=self.device)

        # Break softmax symmetry: reinitialize route weights with random values
        if getattr(args, 'route_init_std', 0) > 0:
            if self.rank == 0:
                logger.info(f"Reinitializing route weights with N(0, {args.route_init_std}) "
                            f"to break softmax symmetry")
            for param in self.model.route_weights:
                param.data.normal_(0, args.route_init_std)

        # Snapshot initial cooperative weights as ref model for KL penalty
        if args.kl_coef > 0:
            self._ref_state = {
                name: param.data.clone()
                for name, param in self.model.named_parameters()
                if "lora_" in name or "route_weights" in name or "comm_" in name
            }
            if self.rank == 0:
                logger.info(f"Ref model snapshot: {len(self._ref_state)} tensors")
        else:
            self._ref_state = {}

        # Enable gradient checkpointing
        self.model.base_model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()

        for name, param in self.model.named_parameters():
            if "lora_" in name or "route_weights" in name or "comm_" in name:
                param.requires_grad = True

        # Enable expert output caching for diversity loss during training
        if args.diversity_weight > 0:
            self.model.enable_expert_output_caching()

        # Move to GPU (no DDP wrapper -- manual gradient all-reduce)
        self.model = self.model.to(self.device)

        dist.barrier()

        if self.rank == 0:
            counts = self.model.count_trainable_params()
            logger.info(f"Model ready (V18 K={args.num_experts} {args.comm_topology}): "
                        f"{counts['total']:,} trainable params "
                        f"(lora={counts['lora']:,}, route={counts['route_weights']:,}, "
                        f"comm={counts['comm']:,})")

    # -- resume from RL checkpoint -----------------------------------------

    def _load_resume_checkpoint(self):
        ckpt_dir = getattr(self.args, "resume_from", "")
        if not ckpt_dir:
            return

        if self.rank == 0:
            logger.info(f"Resuming from RL checkpoint: {ckpt_dir}")

        coop_dir = os.path.join(ckpt_dir, "cooperative")
        if os.path.isdir(coop_dir):
            self.model.load_cooperative(coop_dir, device=self.device)

        state_path = os.path.join(ckpt_dir, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            self.global_step = state.get("global_step", 0)
            self.resume_step = self.global_step
            if self.rank == 0:
                logger.info(f"  Resumed at global_step={self.global_step}")

        dist.barrier()

    # -- gradient checkpointing toggle -------------------------------------

    def _set_grad_checkpointing(self, enable: bool):
        if enable:
            self.model.gradient_checkpointing_enable()
        else:
            self.model.gradient_checkpointing_disable()
            for module in self.model.modules():
                if hasattr(module, "gradient_checkpointing"):
                    module.gradient_checkpointing = False

    # -- data --------------------------------------------------------------

    def _setup_data(self):
        args = self.args
        self.train_dataset = EpisodeDataset(
            args.train_data, max_episodes=args.max_episodes
        )
        # TSortedDistributedSampler: episodes sorted by step count T, then
        # block-shuffled.  Each rank gets different episodes (data parallel)
        # but with similar T → ranks finish at similar times → small desync
        # at all_reduce (mean 6 min, max 94 min vs 180 min timeout).
        # This gives 16x throughput vs SyncedShuffleSampler while avoiding
        # the NCCL timeout from random DistributedSampler (max 214 min desync).
        self.sampler = TSortedDistributedSampler(self.train_dataset, seed=42)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,
            sampler=self.sampler,
            collate_fn=lambda x: x[0],
            num_workers=2,
            pin_memory=True,
        )
        self.val_dataset = None
        if args.val_data and os.path.exists(args.val_data):
            self.val_dataset = EpisodeDataset(args.val_data)
            if self.rank == 0:
                logger.info(f"Val dataset: {len(self.val_dataset)} episodes")

    # -- optimizer (4 groups: lora_decay, lora_nodecay, route, comm) -------

    def _setup_optimizer(self):
        args = self.args

        route_params, comm_params, decay_params, no_decay_params = [], [], [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "route_weights" in name:
                route_params.append(param)
            elif "comm_" in name:
                comm_params.append(param)
            elif param.dim() == 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = []
        if decay_params:
            param_groups.append({
                "params": decay_params,
                "lr": args.lora_lr,
                "weight_decay": args.weight_decay,
            })
        if no_decay_params:
            param_groups.append({
                "params": no_decay_params,
                "lr": args.lora_lr,
                "weight_decay": 0.0,
            })
        if route_params:
            param_groups.append({
                "params": route_params,
                "lr": args.route_lr,
                "weight_decay": 0.0,
            })
        if comm_params:
            param_groups.append({
                "params": comm_params,
                "lr": args.route_lr,
                "weight_decay": 0.0,
            })

        self.optimizer = torch.optim.AdamW(param_groups)

        if self.rank == 0:
            logger.info(
                f"Optimizer: lora_lr={args.lora_lr} route_lr={args.route_lr} "
                f"decay={len(decay_params)} no_decay={len(no_decay_params)} "
                f"route={len(route_params)} comm={len(comm_params)}"
            )

    # -- tokenization helpers ----------------------------------------------

    def _tokenize_for_generation(
        self, messages: list, image: Image.Image,
    ) -> dict:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=False
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    # -- stop config -------------------------------------------------------

    def _get_stop_config(self) -> dict:
        """Get stop strings and token IDs for early stopping after action output."""
        if hasattr(self, "_stop_config_cache"):
            return self._stop_config_cache
        tokenizer = self.processor.tokenizer
        stop_ids = [tokenizer.eos_token_id]
        ids = tokenizer.encode("</tool_call>", add_special_tokens=False)
        if len(ids) == 1:
            stop_ids.append(ids[0])
        self._stop_config_cache = {
            "eos_token_id": stop_ids,
            "stop_strings": ["</action>"],
            "tokenizer": tokenizer,
        }
        if self.rank == 0:
            logger.info(f"Stop config: eos_ids={stop_ids}, stop_strings=['</action>']")
        return self._stop_config_cache

    # -- log prob computation ----------------------------------------------

    def _compute_token_log_probs(
        self,
        full_ids: torch.Tensor,
        prompt_len: int,
        inputs_for_fwd: dict,
        with_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Compute per-token log probs for the response portion."""
        ids = full_ids.unsqueeze(0) if full_ids.dim() == 1 else full_ids
        attn = torch.ones_like(ids)

        fwd_kwargs = {"input_ids": ids, "attention_mask": attn}
        for k in ("pixel_values", "image_grid_thw"):
            if k in inputs_for_fwd:
                fwd_kwargs[k] = inputs_for_fwd[k]

        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model.forward(**fwd_kwargs)

            logits = outputs.logits
            resp_logits = logits[:, prompt_len - 1:-1, :]
            resp_labels = ids[:, prompt_len:]
            log_p = F.log_softmax(resp_logits, dim=-1)
            tok_lp = torch.gather(
                log_p, -1, resp_labels.unsqueeze(-1)
            ).squeeze(-1)
            mask = (resp_labels != self.pad_id).float()

        n_tokens = int(mask.sum().item())
        return tok_lp.squeeze(0), mask.squeeze(0), n_tokens

    def _compute_ref_log_probs(
        self,
        full_ids: torch.Tensor,
        prompt_len: int,
        inputs_for_fwd: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reference log probs using snapshotted initial weights."""
        ids = full_ids.unsqueeze(0) if full_ids.dim() == 1 else full_ids
        attn = torch.ones_like(ids)

        fwd_kwargs = {"input_ids": ids, "attention_mask": attn}
        for k in ("pixel_values", "image_grid_thw"):
            if k in inputs_for_fwd:
                fwd_kwargs[k] = inputs_for_fwd[k]

        # Swap in ref weights
        current_state = {}
        for name, param in self.model.named_parameters():
            if name in self._ref_state:
                current_state[name] = param.data.clone()
                param.data.copy_(self._ref_state[name])

        self.model.eval()

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model.forward(**fwd_kwargs)

            logits = outputs.logits
            resp_logits = logits[:, prompt_len - 1:-1, :]
            resp_labels = ids[:, prompt_len:]
            log_p = F.log_softmax(resp_logits, dim=-1)
            tok_lp = torch.gather(
                log_p, -1, resp_labels.unsqueeze(-1)
            ).squeeze(-1)
            mask = (resp_labels != self.pad_id).float()

        # Restore current weights
        for name, param in self.model.named_parameters():
            if name in current_state:
                param.data.copy_(current_state[name])

        self.model.train()

        return tok_lp.squeeze(0), mask.squeeze(0)

    # -- batched K-sample generation (for step 0) ---------------------------

    @torch.no_grad()
    def _generate_k_samples(
        self, messages: list, image: Image.Image, K: int, max_new_tokens: int
    ) -> Tuple[torch.Tensor, int, dict]:
        """Generate K action samples in one batched call (shared prompt)."""
        inputs = self._tokenize_for_generation(messages, image)
        prompt_len = inputs["input_ids"].shape[1]

        # Repeat inputs K times for batched generation
        gen_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                rep = [K] + [1] * (v.dim() - 1)
                gen_inputs[k] = v.repeat(*rep)
            else:
                gen_inputs[k] = v

        stop_cfg = self._get_stop_config()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output_ids = self.model.generate(
                **gen_inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                do_sample=True,
                eos_token_id=stop_cfg["eos_token_id"],
                stop_strings=stop_cfg["stop_strings"],
                tokenizer=stop_cfg["tokenizer"],
            )

        return output_ids, prompt_len, inputs

    # -- batched generation for different prompts (steps 1+) ---------------

    @torch.no_grad()
    def _generate_batched_different_prompts(
        self,
        messages_list: List[list],
        image: Image.Image,
        max_new_tokens: int,
    ) -> Tuple[List[torch.Tensor], List[int], List[dict]]:
        """Batched generation for different prompts sharing the same image.

        Returns:
            output_ids_list: List of [1, prompt_len+resp_len] tensors (unpadded)
            prompt_lens: original prompt lengths
            inputs_list: original (unpadded) input dicts for log-prob computation
        """
        # 1. Tokenize each prompt individually
        inputs_list = []
        prompt_lens = []
        for messages in messages_list:
            inputs = self._tokenize_for_generation(messages, image)
            inputs_list.append(inputs)
            prompt_lens.append(inputs["input_ids"].shape[1])

        B = len(inputs_list)
        if B == 0:
            return [], [], []

        if B == 1:
            stop_cfg = self._get_stop_config()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output_ids = self.model.generate(
                    **inputs_list[0],
                    max_new_tokens=max_new_tokens,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    do_sample=True,
                    eos_token_id=stop_cfg["eos_token_id"],
                    stop_strings=stop_cfg["stop_strings"],
                    tokenizer=stop_cfg["tokenizer"],
                )
            return [output_ids], prompt_lens, inputs_list

        # 2. Left-pad input_ids and attention_mask to max length
        max_len = max(prompt_lens)
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0

        batched_ids = []
        batched_attn = []
        for inp, plen in zip(inputs_list, prompt_lens):
            ids = inp["input_ids"][0]  # [seq_len]
            pad_len = max_len - plen
            if pad_len > 0:
                ids_padded = torch.cat([
                    torch.full((pad_len,), pad_id, dtype=ids.dtype, device=ids.device),
                    ids,
                ])
                attn = torch.cat([
                    torch.zeros(pad_len, dtype=torch.long, device=ids.device),
                    torch.ones(plen, dtype=torch.long, device=ids.device),
                ])
            else:
                ids_padded = ids
                attn = torch.ones(plen, dtype=torch.long, device=ids.device)
            batched_ids.append(ids_padded)
            batched_attn.append(attn)

        batched_ids = torch.stack(batched_ids)    # [B, max_len]
        batched_attn = torch.stack(batched_attn)  # [B, max_len]

        # 3. Repeat pixel_values and image_grid_thw for batch
        gen_inputs = {"input_ids": batched_ids, "attention_mask": batched_attn}
        for key in ("pixel_values", "image_grid_thw"):
            if key in inputs_list[0]:
                val = inputs_list[0][key]
                rep = [B] + [1] * (val.dim() - 1)
                gen_inputs[key] = val.repeat(*rep)

        # 4. Generate
        stop_cfg = self._get_stop_config()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output_ids = self.model.generate(
                **gen_inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                do_sample=True,
                eos_token_id=stop_cfg["eos_token_id"],
                stop_strings=stop_cfg["stop_strings"],
                tokenizer=stop_cfg["tokenizer"],
            )
        # output_ids: [B, max_len + max_resp_len]

        # 5. Un-pad: remove left padding for each trajectory
        output_ids_list = []
        for i in range(B):
            pad_len = max_len - prompt_lens[i]
            unpadded = output_ids[i, pad_len:]  # [prompt_len_i + resp_len_i]
            output_ids_list.append(unpadded.unsqueeze(0))  # [1, seq_len]

        return output_ids_list, prompt_lens, inputs_list

    # -- batched log-prob computation --------------------------------------

    def _batch_compute_token_log_probs(
        self,
        full_ids_list: List[torch.Tensor],
        prompt_lens: List[int],
        inputs_list: List[dict],
        with_grad: bool = False,
    ) -> List[torch.Tensor]:
        """Batch compute token log probs for sequences sharing the same image."""
        B = len(full_ids_list)
        if B == 0:
            return []
        if B == 1:
            tok_lp, mask, _ = self._compute_token_log_probs(
                full_ids_list[0], prompt_lens[0], inputs_list[0], with_grad=with_grad
            )
            return [tok_lp]

        # Right-pad to max sequence length
        max_seq = max(ids.shape[0] for ids in full_ids_list)
        batched_ids = []
        batched_attn = []
        for ids in full_ids_list:
            pad_len = max_seq - ids.shape[0]
            if pad_len > 0:
                ids_p = torch.cat([ids, torch.full((pad_len,), self.pad_id, dtype=ids.dtype, device=ids.device)])
                attn = torch.cat([torch.ones(ids.shape[0], dtype=torch.long, device=ids.device),
                                  torch.zeros(pad_len, dtype=torch.long, device=ids.device)])
            else:
                ids_p = ids
                attn = torch.ones(ids.shape[0], dtype=torch.long, device=ids.device)
            batched_ids.append(ids_p)
            batched_attn.append(attn)

        batched_ids = torch.stack(batched_ids)
        batched_attn = torch.stack(batched_attn)

        fwd_kwargs = {"input_ids": batched_ids, "attention_mask": batched_attn}
        for key in ("pixel_values", "image_grid_thw"):
            if key in inputs_list[0]:
                val = inputs_list[0][key]
                rep = [B] + [1] * (val.dim() - 1)
                fwd_kwargs[key] = val.repeat(*rep)

        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model.forward(**fwd_kwargs)

            logits = outputs.logits  # [B, max_seq, vocab]
            results = []
            for i in range(B):
                plen = prompt_lens[i]
                seq_len = full_ids_list[i].shape[0]
                resp_logits = logits[i:i+1, plen - 1:seq_len - 1, :]
                resp_labels = batched_ids[i:i+1, plen:seq_len]
                log_p = F.log_softmax(resp_logits, dim=-1)
                tok_lp = torch.gather(log_p, -1, resp_labels.unsqueeze(-1)).squeeze(-1)
                results.append(tok_lp.squeeze(0))

        return results

    def _batch_compute_ref_log_probs(
        self,
        full_ids_list: List[torch.Tensor],
        prompt_lens: List[int],
        inputs_list: List[dict],
    ) -> List[torch.Tensor]:
        """Batch compute reference log probs (single weight-swap)."""
        # Swap in ref weights (once for entire batch)
        current_state = {}
        for name, param in self.model.named_parameters():
            if name in self._ref_state:
                current_state[name] = param.data.clone()
                param.data.copy_(self._ref_state[name])

        self.model.eval()
        results = self._batch_compute_token_log_probs(
            full_ids_list, prompt_lens, inputs_list, with_grad=False
        )

        # Restore current weights
        for name, param in self.model.named_parameters():
            if name in current_state:
                param.data.copy_(current_state[name])
        self.model.train()

        return results

    # -- on-policy trajectory rollout --------------------------------------

    @torch.no_grad()
    def generate_onpolicy_trajectory(self, episode: Dict, K: int = 8) -> Optional[Tuple[List[Dict], int]]:
        """Roll out K full trajectories on-policy (model's own predictions as history)."""
        args = self.args
        goal = episode["goal"]
        steps = episode["steps"]
        T = len(steps)

        if self.rank == 0:
            logger.info(f"[on-policy] goal={goal[:60]}... T={T} K={K}")

        self._set_grad_checkpointing(False)
        self.model.eval()

        # -- Step 0: batched generation --
        step0 = steps[0]
        screenshot0 = step0["screenshot"]
        gt_action0 = step0["action"]
        image_w0 = step0.get("image_w", 1040)
        image_h0 = step0.get("image_h", 736)

        try:
            image0 = Image.open(screenshot0).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to open step 0 screenshot: {e}")
            self.model.train()
            self._set_grad_checkpointing(True)
            return None

        messages0 = build_step_messages(goal, [], screenshot0)

        try:
            t_gen = time.time()
            output_ids_step0, prompt_len0, inputs0 = self._generate_k_samples(
                messages0, image0, K, args.max_new_tokens
            )
            if self.rank == 0:
                logger.info(f"  step 0 batched K={K} gen {time.time()-t_gen:.1f}s")
        except Exception as e:
            logger.warning(f"Step 0 batched generation failed: {e}")
            self.model.train()
            self._set_grad_checkpointing(True)
            return None

        trajectories = []
        for k in range(K):
            resp_ids = output_ids_step0[k, prompt_len0:]
            is_trunc = (resp_ids.shape[0] >= args.max_new_tokens)

            if is_trunc:
                text = ""
                reward = 0.0
            else:
                text = self.processor.tokenizer.decode(
                    resp_ids, skip_special_tokens=True
                )
                reward, _ = compute_step_reward(
                    text, gt_action0, image_w0, image_h0,
                    w_format=args.w_format, w_type=args.w_type, w_content=args.w_content,
                )

            correct = (reward >= args.match_threshold)
            step0_output = (output_ids_step0[k:k+1], prompt_len0, inputs0)

            traj = {
                "outputs": [step0_output],
                "step_rewards": [reward],
                "stopped": not correct,
                "history": [],
            }

            if correct:
                pred_action = parse_action_from_text(text) if text else None
                if pred_action:
                    traj["history"].append(
                        format_pred_action_for_history(pred_action, 1)
                    )
                else:
                    traj["history"].append("Step 1: action()")

            trajectories.append(traj)

        # -- Steps 1+: batched generation per step --
        for t in range(1, T):
            if all(traj["stopped"] for traj in trajectories):
                for traj in trajectories:
                    for _ in range(t, T):
                        traj["step_rewards"].append(0.0)
                        traj["outputs"].append(None)
                break

            step = steps[t]
            screenshot = step["screenshot"]
            gt_action = step["action"]
            image_w = step.get("image_w", 1040)
            image_h = step.get("image_h", 736)

            try:
                image = Image.open(screenshot).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to open {screenshot}: {e}")
                for traj in trajectories:
                    if not traj["stopped"]:
                        traj["stopped"] = True
                    traj["step_rewards"].append(0.0)
                    traj["outputs"].append(None)
                continue

            # Collect active trajectories
            active_indices = [k for k, tr in enumerate(trajectories) if not tr["stopped"]]

            # Append zeros for already-stopped trajectories
            for k, traj in enumerate(trajectories):
                if traj["stopped"]:
                    traj["step_rewards"].append(0.0)
                    traj["outputs"].append(None)

            # Build messages for each active trajectory
            messages_list = [
                build_step_messages(goal, trajectories[k]["history"], screenshot)
                for k in active_indices
            ]

            # Batched generation for all active trajectories at this step
            try:
                t_gen = time.time()
                gen_results, gen_prompt_lens, gen_inputs_list = \
                    self._generate_batched_different_prompts(
                        messages_list, image, args.max_new_tokens
                    )
                if self.rank == 0:
                    logger.info(
                        f"  step {t} batched B={len(active_indices)} "
                        f"gen {time.time()-t_gen:.1f}s"
                    )
            except Exception as e:
                logger.warning(f"Batched generation failed at step {t}: {e}")
                for k in active_indices:
                    trajectories[k]["stopped"] = True
                    trajectories[k]["step_rewards"].append(0.0)
                    trajectories[k]["outputs"].append(None)
                continue

            # Process each result
            for idx, k in enumerate(active_indices):
                traj = trajectories[k]
                output_ids = gen_results[idx]
                prompt_len = gen_prompt_lens[idx]
                inputs = gen_inputs_list[idx]

                resp_ids = output_ids[0, prompt_len:]
                is_trunc = (resp_ids.shape[0] >= args.max_new_tokens)

                if is_trunc:
                    text = ""
                    reward = 0.0
                else:
                    text = self.processor.tokenizer.decode(
                        resp_ids, skip_special_tokens=True
                    )
                    reward, _ = compute_step_reward(
                        text, gt_action, image_w, image_h,
                        w_format=args.w_format, w_type=args.w_type,
                        w_content=args.w_content,
                    )

                correct = (reward >= args.match_threshold)
                traj["step_rewards"].append(reward)
                traj["outputs"].append((output_ids, prompt_len, inputs))

                if not correct:
                    traj["stopped"] = True
                else:
                    pred_action = parse_action_from_text(text) if text else None
                    if pred_action:
                        traj["history"].append(
                            format_pred_action_for_history(pred_action, t + 1)
                        )
                    else:
                        traj["history"].append(f"Step {t + 1}: action()")

        # Restore training mode
        self.model.train()
        self._set_grad_checkpointing(True)

        # Compute trajectory rewards
        result_trajectories = []
        for k, traj in enumerate(trajectories):
            step_rewards = traj["step_rewards"]

            consecutive = 0
            for r in step_rewards:
                if r >= args.match_threshold:
                    consecutive += 1
                else:
                    break

            n_reward_steps = min(consecutive + 1, len(step_rewards))
            traj_reward = sum(step_rewards[:n_reward_steps]) / T

            result_trajectories.append({
                "outputs": traj["outputs"],
                "step_rewards": step_rewards,
                "traj_reward": traj_reward,
                "consecutive": consecutive,
            })

            if self.rank == 0:
                logger.info(
                    f"  traj {k}: consecutive={consecutive}/{T} "
                    f"reward={traj_reward:.3f} "
                    f"steps={[f'{r:.2f}' for r in step_rewards[:n_reward_steps]]}"
                )

        return result_trajectories, T

    # -- Per-step policy loss ------------------------------------------------

    def compute_per_step_loss(self, trajectories: List[Dict], T: int,
                              advantages_matrix: np.ndarray) -> Optional[Tuple[torch.Tensor, Dict]]:
        """Per-step PPO loss with pre-computed trajectory-level advantages.

        Args:
            trajectories: List of K trajectory dicts
            T: number of steps
            advantages_matrix: [K, T] pre-computed advantages (SPWA + step-level blend)
        """
        args = self.args
        clip_range = args.clip_range
        K = len(trajectories)

        total_loss = torch.tensor(0.0, device=self.device)
        total_kl = torch.tensor(0.0, device=self.device)
        n_step_updates = 0
        total_tokens = 0
        all_clip_fracs = []
        all_step_advs = []
        steps_with_signal = 0
        steps_without_signal = 0

        for t in range(T):
            for k in range(K):
                if (t >= len(trajectories[k]["outputs"])
                        or trajectories[k]["outputs"][t] is None
                        or trajectories[k]["old_log_probs"][t] is None):
                    continue

                A = float(advantages_matrix[k, t])
                if abs(A) < 1e-8:
                    steps_without_signal += 1
                    continue

                steps_with_signal += 1

                output_ids, prompt_len, inputs = trajectories[k]["outputs"][t]
                old_tok_lp = trajectories[k]["old_log_probs"][t]

                tok_lp, mask, n_tok = self._compute_token_log_probs(
                    output_ids[0], prompt_len, inputs, with_grad=True
                )

                ratio = torch.exp(tok_lp - old_tok_lp)

                adv_expanded = torch.full_like(mask, A)
                pg_loss1 = -adv_expanded * ratio
                pg_loss2 = -adv_expanded * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                pg_loss = torch.max(pg_loss1, pg_loss2)

                valid_tokens = mask.sum().clamp(min=1)
                step_loss = (pg_loss * mask).sum() / valid_tokens

                # KL penalty
                kl_loss = torch.tensor(0.0, device=self.device)
                if args.kl_coef > 0 and "ref_log_probs" in trajectories[k]:
                    ref_lp = trajectories[k]["ref_log_probs"][t]
                    if ref_lp is not None:
                        kl_loss = compute_kl_penalty(tok_lp, ref_lp, mask)

                total_loss = total_loss + step_loss + args.kl_coef * kl_loss
                total_kl = total_kl + kl_loss.detach()
                n_step_updates += 1
                total_tokens += n_tok

                clip_frac = ((ratio < 1.0 - clip_range) | (ratio > 1.0 + clip_range)).float()
                clip_frac = (clip_frac * mask).sum() / valid_tokens
                all_clip_fracs.append(clip_frac.item())
                all_step_advs.append(abs(A))

        if n_step_updates == 0:
            return None

        loss = total_loss / n_step_updates

        metrics = {
            "n_step_updates": n_step_updates,
            "total_tokens": total_tokens,
            "steps_with_signal": steps_with_signal,
            "steps_without_signal": steps_without_signal,
            "mean_clip_frac": float(np.mean(all_clip_fracs)) if all_clip_fracs else 0.0,
            "mean_step_adv": float(np.mean(all_step_advs)) if all_step_advs else 0.0,
            "mean_traj_reward": float(np.mean([t["traj_reward"] for t in trajectories])),
            "mean_kl": total_kl.item() / max(n_step_updates, 1),
        }

        return loss, metrics

    # -- training step for one episode -------------------------------------

    def train_episode(self, episode: Dict) -> Optional[Dict[str, float]]:
        """Full training step: on-policy rollout + per-step policy gradient."""
        args = self.args
        K = args.num_trajectories

        # 1. On-policy rollout
        result = self.generate_onpolicy_trajectory(episode, K=K)
        if result is None:
            return None

        trajectories, T = result

        # 2. Build [K, T] reward matrix and compute trajectory advantages
        reward_matrix = np.zeros((K, T), dtype=np.float32)
        for k, traj in enumerate(trajectories):
            for t in range(min(T, len(traj["step_rewards"]))):
                reward_matrix[k, t] = traj["step_rewards"][t]

        adv_result = compute_trajectory_advantages(
            reward_matrix,
            match_threshold=args.match_threshold,
            spwa_decay=args.spwa_decay,
            dapo_threshold=args.dapo_threshold,
            step_adv_weight=args.step_adv_weight,
        )
        if adv_result is None:
            if self.rank == 0:
                logger.info("  DAPO filtered (SP variance too low)")
            return None

        advantages_matrix, sp_scores, first_errors = adv_result

        if self.rank == 0:
            logger.info(
                f"  SP scores: {sp_scores.tolist()}"
                f"  first_errors: {first_errors.tolist()}"
            )

        # 3. Cache old log probs (batched per step for efficiency)
        for traj in trajectories:
            traj["old_log_probs"] = [None] * len(traj["outputs"])
            if args.kl_coef > 0:
                traj["ref_log_probs"] = [None] * len(traj["outputs"])

        for t in range(T):
            # Collect valid outputs at step t (same image → can batch)
            valid_items = []
            for k, traj in enumerate(trajectories):
                if t < len(traj["outputs"]) and traj["outputs"][t] is not None:
                    output_ids, prompt_len, inputs = traj["outputs"][t]
                    valid_items.append((k, output_ids[0], prompt_len, inputs))

            if not valid_items:
                continue

            ids_list = [item[1] for item in valid_items]
            plens = [item[2] for item in valid_items]
            inps = [item[3] for item in valid_items]

            tok_lps = self._batch_compute_token_log_probs(
                ids_list, plens, inps, with_grad=False
            )
            for idx, (k, _, _, _) in enumerate(valid_items):
                trajectories[k]["old_log_probs"][t] = tok_lps[idx].detach()

            if args.kl_coef > 0:
                ref_lps = self._batch_compute_ref_log_probs(ids_list, plens, inps)
                for idx, (k, _, _, _) in enumerate(valid_items):
                    trajectories[k]["ref_log_probs"][t] = ref_lps[idx].detach()

        # 4. Per-step policy loss with pre-computed advantages
        loss_result = self.compute_per_step_loss(trajectories, T, advantages_matrix)
        if loss_result is None:
            if self.rank == 0:
                logger.info("  No signal at any step (all active trajs same reward)")
            return None

        loss, step_metrics = loss_result

        # 4. Balance loss (K-way entropy regularization)
        bal_loss, bal_stats = self.model.compute_balance_loss()
        total_loss = loss + args.balance_weight * bal_loss

        # 5. Diversity loss (pairwise cosine similarity)
        div_loss = self.model.compute_diversity_loss()
        total_loss = total_loss + args.diversity_weight * div_loss

        # 6. Backward
        total_loss.backward()

        metrics = {
            "pg_loss": loss.item(),
            "balance_loss": bal_loss.item(),
            "diversity_loss": div_loss.item(),
            "routing_entropy": bal_stats.get("routing_entropy", 0.0),
            "max_expert_usage": bal_stats.get("max_expert_usage", 0.0),
            "min_expert_usage": bal_stats.get("min_expert_usage", 0.0),
            "clip_frac": step_metrics["mean_clip_frac"],
            "mean_step_adv": step_metrics["mean_step_adv"],
            "mean_traj_reward": step_metrics["mean_traj_reward"],
            "n_step_updates": step_metrics["n_step_updates"],
            "steps_with_signal": step_metrics["steps_with_signal"],
            "total_tokens": step_metrics["total_tokens"],
            "mean_kl": step_metrics["mean_kl"],
            "mean_consecutive": float(np.mean([t["consecutive"] for t in trajectories])),
            "mean_sp": float(np.mean(sp_scores)),
            "T": T,
        }

        return metrics

    # -- checkpoint --------------------------------------------------------

    def save_checkpoint(self, tag: str):
        if self.rank != 0:
            return

        ckpt_dir = os.path.join(self.args.output_dir, tag)
        coop_dir = os.path.join(ckpt_dir, "cooperative")
        self.model.save_cooperative(coop_dir)

        torch.save(
            {"global_step": self.global_step},
            os.path.join(ckpt_dir, "training_state.pt"),
        )
        logger.info(f"Saved checkpoint: {ckpt_dir}")

    # -- validation --------------------------------------------------------

    @torch.no_grad()
    def validate(self, tag: str) -> Dict[str, float]:
        """Validate with greedy on-policy rollout."""
        if self.val_dataset is None or self.rank != 0:
            return {}

        logger.info(f"Running validation ({tag})...")
        all_consecutive = []
        all_T = []
        per_ep_results = []

        for ep_idx in range(min(len(self.val_dataset), 20)):
            episode = self.val_dataset[ep_idx]
            goal = episode["goal"]
            steps = episode["steps"]
            T = len(steps)
            action_history: List[str] = []
            consecutive = 0

            for si, step in enumerate(steps):
                screenshot = step["screenshot"]
                gt_action = step["action"]
                image_w = step.get("image_w", 1040)
                image_h = step.get("image_h", 736)

                try:
                    image = Image.open(screenshot).convert("RGB")
                except Exception:
                    break

                messages = build_step_messages(goal, action_history, screenshot)
                inputs = self._tokenize_for_generation(messages, image)

                self._set_grad_checkpointing(False)
                self.model.eval()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.args.max_new_tokens,
                        do_sample=False,
                        eos_token_id=self._get_stop_config()["eos_token_id"],
                        stop_strings=self._get_stop_config()["stop_strings"],
                        tokenizer=self._get_stop_config()["tokenizer"],
                    )
                self.model.train()
                self._set_grad_checkpointing(True)

                prompt_len = inputs["input_ids"].shape[1]
                text = self.processor.tokenizer.decode(
                    output_ids[0, prompt_len:], skip_special_tokens=True
                )

                reward, _ = compute_step_reward(text, gt_action, image_w, image_h)
                correct = (reward >= self.args.match_threshold)

                if not correct:
                    break

                consecutive += 1

                pred_action = parse_action_from_text(text)
                if pred_action:
                    action_history.append(
                        format_pred_action_for_history(pred_action, si + 1)
                    )
                else:
                    action_history.append(f"Step {si + 1}: action()")

            traj_reward = consecutive / T
            all_consecutive.append(consecutive)
            all_T.append(T)

            per_ep_results.append({
                "episode_id": episode["episode_id"],
                "goal": goal[:100],
                "num_steps": T,
                "consecutive": consecutive,
                "traj_reward": traj_reward,
            })

        metrics = {
            "val/mean_traj_reward": float(np.mean([c / t for c, t in zip(all_consecutive, all_T)])) if all_T else 0.0,
            "val/mean_consecutive": float(np.mean(all_consecutive)) if all_consecutive else 0.0,
            "val/full_success_rate": float(np.mean([c == t for c, t in zip(all_consecutive, all_T)])) if all_T else 0.0,
            "val/n_episodes": len(per_ep_results),
        }

        val_dir = os.path.join(self.args.output_dir, "val_results")
        os.makedirs(val_dir, exist_ok=True)
        result_path = os.path.join(val_dir, f"{tag}.jsonl")
        with open(result_path, "w") as f:
            f.write(json.dumps({"_summary": metrics, "_step": self.global_step}) + "\n")
            for r in per_ep_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        logger.info(f"Validation {tag}:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        return metrics

    # -- main training loop ------------------------------------------------

    def train(self):
        args = self.args
        K_experts = args.num_experts

        if self.rank == 0:
            logger.info("=" * 60)
            logger.info("V18 K-Expert Cooperative LoRA RL (GSPO)")
            logger.info(f"  Architecture: K={K_experts} experts, topology={args.comm_topology}")
            logger.info(f"  K_traj={args.num_trajectories}  temp={args.temperature}  top_p={args.top_p}")
            logger.info(f"  clip_range={args.clip_range}  (token-level PPO)")
            logger.info(f"  num_comm_rounds={args.num_comm_rounds}")
            logger.info(f"  match_threshold={args.match_threshold}")
            logger.info(f"  grad_accum={args.gradient_accumulation_steps}")
            logger.info(f"  epochs={args.num_epochs}")
            logger.info(f"  lora_lr={args.lora_lr}  route_lr={args.route_lr}")
            logger.info(f"  balance_weight={args.balance_weight}  diversity_weight={args.diversity_weight}")
            logger.info(f"  routing_noise_scale={args.routing_noise_scale}")
            logger.info(f"  kl_coef={args.kl_coef}  spwa_decay={args.spwa_decay}")
            logger.info(f"  dapo_threshold={args.dapo_threshold}  step_adv_weight={args.step_adv_weight}")
            logger.info(f"  world_size={self.world_size}")
            logger.info(f"  episodes={len(self.train_dataset)}")
            if self.resume_step > 0:
                logger.info(f"  RESUMING from step {self.resume_step}")
            logger.info("=" * 60)

        # Set routing noise for exploration
        if args.routing_noise_scale > 0:
            self.model.set_routing_noise(args.routing_noise_scale)

        for epoch in range(args.num_epochs):
            self.sampler.set_epoch(epoch)

            epoch_metrics = defaultdict(list)
            accum_count = 0
            skipped = 0

            t_epoch = time.time()

            skip_episodes = self.resume_step * args.gradient_accumulation_steps
            if self.rank == 0 and skip_episodes > 0:
                logger.info(f"  Skipping first {skip_episodes} episodes (resume)")

            self.optimizer.zero_grad()

            for ep_idx, episode in enumerate(self.train_loader):
                if ep_idx < skip_episodes:
                    continue

                t_ep = time.time()

                try:
                    metrics = self.train_episode(episode)
                except Exception as e:
                    if self.rank == 0:
                        logger.warning(
                            f"Episode {ep_idx} failed: {e}\n"
                            f"{traceback.format_exc()}"
                        )
                    metrics = None

                if metrics is not None:
                    accum_count += 1
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            epoch_metrics[k].append(v)
                else:
                    skipped += 1
                    if self.rank == 0:
                        logger.info(f"  ep {ep_idx} skipped (no signal or failed)")

                # Per-episode barrier: prevent skipped episodes from causing
                # one rank to race far ahead and timeout at all_reduce.
                if self.world_size > 1:
                    dist.barrier()

                # Gradient accumulation boundary
                if (ep_idx + 1) % args.gradient_accumulation_steps == 0:
                    scale = 1.0 / args.gradient_accumulation_steps
                    for p in self.model.parameters():
                        if p.requires_grad and p.grad is not None:
                            p.grad.mul_(scale)

                    if self.world_size > 1:
                        # Bucket all gradients into a single flat tensor for
                        # ONE all_reduce instead of 1120 separate calls.
                        # The per-parameter approach caused NCCL timeouts due
                        # to massive small-message overhead.
                        trainable_with_grad = []
                        for p in self.model.parameters():
                            if p.requires_grad:
                                if p.grad is None:
                                    p.grad = torch.zeros_like(p.data)
                                trainable_with_grad.append(p)

                        # Flatten all grads into one contiguous buffer
                        flat_grads = torch.cat([p.grad.reshape(-1) for p in trainable_with_grad])
                        dist.all_reduce(flat_grads, op=dist.ReduceOp.AVG)

                        # Copy back
                        offset = 0
                        for p in trainable_with_grad:
                            numel = p.grad.numel()
                            p.grad.copy_(flat_grads[offset:offset + numel].reshape(p.grad.shape))
                            offset += numel

                    trainable = [p for p in self.model.parameters() if p.requires_grad]
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable, args.max_grad_norm
                    )

                    self.optimizer.step()
                    self.global_step += 1

                    # Logging
                    if self.rank == 0 and self.global_step % args.logging_steps == 0:
                        n_log = max(accum_count, 1)
                        recent_loss = epoch_metrics.get("pg_loss", [0])[-n_log:]
                        recent_reward = epoch_metrics.get("mean_traj_reward", [0])[-n_log:]
                        recent_consec = epoch_metrics.get("mean_consecutive", [0])[-n_log:]
                        recent_clip = epoch_metrics.get("clip_frac", [0])[-n_log:]
                        recent_entropy = epoch_metrics.get("routing_entropy", [0])[-n_log:]
                        recent_div = epoch_metrics.get("diversity_loss", [0])[-n_log:]
                        recent_max_usage = epoch_metrics.get("max_expert_usage", [0])[-n_log:]
                        recent_min_usage = epoch_metrics.get("min_expert_usage", [0])[-n_log:]
                        recent_kl = epoch_metrics.get("mean_kl", [0])[-n_log:]
                        recent_sp = epoch_metrics.get("mean_sp", [0])[-n_log:]

                        logger.info(
                            f"E{epoch} S{self.global_step} "
                            f"loss={np.mean(recent_loss):.4f} "
                            f"r={np.mean(recent_reward):.3f} "
                            f"consec={np.mean(recent_consec):.1f} "
                            f"sp={np.mean(recent_sp):.3f} "
                            f"kl={np.mean(recent_kl):.4f} "
                            f"clip={np.mean(recent_clip):.3f} "
                            f"ent={np.mean(recent_entropy):.3f} "
                            f"div={np.mean(recent_div):.4f} "
                            f"usage=[{np.mean(recent_min_usage):.2f},{np.mean(recent_max_usage):.2f}] "
                            f"gnorm={grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm:.2f} "
                            f"valid={accum_count}/{args.gradient_accumulation_steps} "
                            f"t={time.time()-t_ep:.1f}s"
                        )

                    # Save checkpoint
                    if (self.rank == 0 and args.save_steps > 0
                            and self.global_step % args.save_steps == 0):
                        self.save_checkpoint(
                            f"epoch-{epoch}_step-{self.global_step}"
                        )

                    # Mid-epoch validation
                    if (args.val_steps > 0
                            and self.global_step % args.val_steps == 0):
                        if self.rank == 0:
                            self.validate(tag=f"epoch-{epoch}_step-{self.global_step}")
                        if self.world_size > 1:
                            dist.barrier()

                    self.optimizer.zero_grad()
                    accum_count = 0
                    torch.cuda.empty_cache()

            # -- Epoch summary --
            if self.rank == 0:
                dur = time.time() - t_epoch
                logger.info(f"{'=' * 60}")
                logger.info(
                    f"Epoch {epoch} done in {dur / 60:.1f}min  "
                    f"skipped={skipped}"
                )
                for k in ["pg_loss", "mean_traj_reward", "mean_consecutive",
                           "mean_sp", "mean_kl", "clip_frac",
                           "routing_entropy", "diversity_loss",
                           "balance_loss", "max_expert_usage", "min_expert_usage",
                           "mean_step_adv"]:
                    vals = epoch_metrics.get(k, [0])
                    logger.info(f"  avg {k}: {np.mean(vals):.4f}")
                logger.info(f"{'=' * 60}")

                self.save_checkpoint(f"epoch-{epoch}")
                self.validate(tag=f"epoch-{epoch}_final")

            if self.world_size > 1:
                dist.barrier()

            self.resume_step = 0

        if self.rank == 0:
            logger.info("Training complete!")


# -- CLI -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="V18 K-Expert Cooperative LoRA RL (GSPO)")

    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sft_checkpoint", type=str, default="",
                        help="Path to SFT cooperative checkpoint dir")
    parser.add_argument("--resume_from", type=str, default="",
                        help="Path to RL checkpoint dir to resume from")

    # Data
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default="")
    parser.add_argument("--max_episodes", type=int, default=0)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"])

    # V18 K-expert cooperative LoRA
    parser.add_argument("--num_experts", type=int, default=4,
                        help="Number of experts K (default: 4)")
    parser.add_argument("--comm_topology", type=str, default="top2",
                        choices=["none", "top2", "shared", "full"],
                        help="Communication topology (default: top2)")
    parser.add_argument("--num_comm_rounds", type=int, default=2,
                        help="Number of iterative communication rounds (T)")
    parser.add_argument("--balance_weight", type=float, default=0.01)
    parser.add_argument("--diversity_weight", type=float, default=0.001,
                        help="Weight for diversity loss (pairwise cosine sim)")
    parser.add_argument("--routing_noise_scale", type=float, default=0.3,
                        help="Noise std added to routing logits for exploration")
    parser.add_argument("--route_init_std", type=float, default=0.0,
                        help="If >0, reinitialize route weights with N(0, std) after loading "
                             "checkpoint to break softmax symmetry (0=keep loaded values)")
    parser.add_argument("--image_max_pixels", type=int, default=602112)

    # RL (on-policy + per-step PPO)
    parser.add_argument("--num_trajectories", type=int, default=8,
                        help="K: number of on-policy trajectories per episode")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--clip_range", type=float, default=0.2,
                        help="PPO clip range for token-level importance ratio")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--match_threshold", type=float, default=0.5,
                        help="Step reward threshold for stop-on-error")
    parser.add_argument("--kl_coef", type=float, default=0.001,
                        help="KL penalty coefficient (0 = disabled)")
    parser.add_argument("--spwa_decay", type=float, default=0.5,
                        help="SPWA decay factor for steps after first error")
    parser.add_argument("--dapo_threshold", type=float, default=0.0,
                        help="DAPO: filter episodes with SP std below threshold")
    parser.add_argument("--step_adv_weight", type=float, default=0.5,
                        help="Mixing weight for step-level advantages (0=traj only)")

    # Reward weights
    parser.add_argument("--w_format", type=float, default=0.1)
    parser.add_argument("--w_type", type=float, default=0.2)
    parser.add_argument("--w_content", type=float, default=0.7)

    # Optimizer
    parser.add_argument("--lora_lr", type=float, default=1e-5)
    parser.add_argument("--route_lr", type=float, default=5e-4,
                        help="Route/comm LR (lower than V15 since softmax is more sensitive)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Training
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--val_steps", type=int, default=25)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    trainer = V18KExpertGSPOTrainer(args)
    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
