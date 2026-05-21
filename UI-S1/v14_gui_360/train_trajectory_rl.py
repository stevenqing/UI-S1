#!/usr/bin/env python3
"""V14 GUI-360 Cooperative LoRA Trajectory RL -- Separate B + Conditional Comm.

Adapted from v13_gui_360/train_trajectory_rl.py for V14 cooperative LoRA.

Changes from V13:
  - Uses V14 cooperative wrapper (separate B_1/B_2, comm always-on)
  - Import from v14_gui_360 instead of v13_gui_360
  - Logger name: v14_gui360_rl
  - Everything else (PPO, GiGPO, SPWA, data loading, generation, ref model) identical

Usage:
  srun --ntasks-per-node=1 bash -c '
    torchrun --nproc_per_node=4 --nnodes=$SLURM_NNODES \\
      --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR \\
      v14_gui_360/train_trajectory_rl.py --model_path ... --train_data ... --output_dir ...
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
from torch.utils.data import Dataset, DataLoader, DistributedSampler
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
logger = logging.getLogger("v14_gui360_rl")

# Project imports
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from v14_gui_360.reward import (
    compute_step_reward,
    compute_trajectory_advantages,
    compute_grpo_advantages,
    parse_action_from_text,
    apply_causal_type_masking,
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


# -- Prompt formatting -----------------------------------------------------

def format_gt_action_for_history(gt_action: Dict, step_id: int) -> str:
    """Convert GT action dict to GUI-360 history format."""
    atype = gt_action.get("action", "")
    coord = gt_action.get("coordinate")

    if atype == "click":
        if coord:
            return f"Step {step_id}: click(coordinate=[{int(coord[0])}, {int(coord[1])}])"
        return f"Step {step_id}: click()"

    elif atype == "type":
        text = gt_action.get("text", "")
        if coord and text:
            t = text[:30] + "..." if len(text) > 30 else text
            return f"Step {step_id}: type(coordinate=[{int(coord[0])}, {int(coord[1])}], keys='{t}')"
        elif text:
            t = text[:30] + "..." if len(text) > 30 else text
            return f"Step {step_id}: type(keys='{t}')"
        elif coord:
            return f"Step {step_id}: type(coordinate=[{int(coord[0])}, {int(coord[1])}])"
        return f"Step {step_id}: type()"

    elif atype in ("swipe", "drag"):
        start = gt_action.get("coordinate")
        end = gt_action.get("endCoordinate")
        if start and end:
            return (f"Step {step_id}: drag(start_coordinate=[{int(start[0])}, {int(start[1])}], "
                    f"end_coordinate=[{int(end[0])}, {int(end[1])}])")
        return f"Step {step_id}: drag()"

    else:
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


# -- GRPO loss helpers -----------------------------------------------------

def compute_policy_loss(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_range: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute PPO/GRPO clipped policy loss (token-level)."""
    ratio = torch.exp(log_probs - old_log_probs)
    clip_low = 1.0 - clip_range
    clip_high = 1.0 + clip_range
    clipped_ratio = torch.clamp(ratio, clip_low, clip_high)

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    pg_loss = torch.max(pg_loss1, pg_loss2)

    valid_tokens = response_mask.sum().clamp(min=1)
    pg_loss = (pg_loss * response_mask).sum() / valid_tokens

    clip_frac = ((ratio < clip_low) | (ratio > clip_high)).float()
    clip_frac = (clip_frac * response_mask).sum() / valid_tokens

    approx_kl = (old_log_probs - log_probs) * response_mask
    approx_kl = approx_kl.sum() / valid_tokens

    return pg_loss, clip_frac, approx_kl


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

class V14TrajectoryRLTrainer:
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

    # -- model (IterativeCooperativeVLMWrapper) ----------------------------

    def _setup_model(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from v14_gui_360.cooperative_wrapper import IterativeCooperativeVLMWrapper

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

        self.model = IterativeCooperativeVLMWrapper(
            base_model=base_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=list(args.target_modules),
            balance_weight=args.balance_weight,
            num_comm_rounds=args.num_comm_rounds,
        )

        # Load SFT checkpoint
        if args.sft_checkpoint:
            if self.rank == 0:
                logger.info(f"Loading SFT checkpoint: {args.sft_checkpoint}")
            self.model.load_cooperative(args.sft_checkpoint, device=self.device)

        # Enable gradient checkpointing
        self.model.base_model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()

        for name, param in self.model.named_parameters():
            if "lora_" in name or "route_weights" in name or "comm_" in name:
                param.requires_grad = True

        # Move to GPU (no DDP wrapper -- manual gradient all-reduce)
        self.model = self.model.to(self.device)

        dist.barrier()

        if self.rank == 0:
            counts = self.model.count_trainable_params()
            logger.info(f"Model ready: {counts['total']:,} trainable params "
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
        self.sampler = DistributedSampler(
            self.train_dataset, shuffle=True, drop_last=True
        )
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
                # Communication params use route_lr (routing infrastructure)
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

    # -- generation (K samples for one step) -------------------------------

    def _get_stop_config(self) -> dict:
        """Get stop strings and token IDs for early stopping after action output."""
        if hasattr(self, "_stop_config_cache"):
            return self._stop_config_cache
        tokenizer = self.processor.tokenizer
        # </tool_call> is a single token (151658) in Qwen tokenizer
        # </action> is multi-token, use stop_strings for it
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

    @torch.no_grad()
    def _generate_k_samples(
        self, messages: list, image: Image.Image, K: int, max_new_tokens: int
    ) -> Tuple[torch.Tensor, int, dict]:
        """Generate K action samples in one batched call."""
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

        noise_std = self.args.routing_noise_scale
        self.model.set_routing_noise(noise_std)

        self._set_grad_checkpointing(False)
        self.model.eval()

        # Early stopping after action output
        stop_cfg = self._get_stop_config()

        t_gen = time.time()
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
        if self.rank == 0:
            logger.info(f"    K={K} batch gen {time.time()-t_gen:.1f}s tokens={output_ids.shape[-1]-prompt_len}")

        self.model.set_routing_noise(0.0)
        self.model.train()
        self._set_grad_checkpointing(True)

        return output_ids, prompt_len, inputs

    # -- log prob computation ----------------------------------------------

    def _compute_token_log_probs(
        self,
        full_ids: torch.Tensor,
        prompt_len: int,
        inputs_for_fwd: dict,
        with_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
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

        return tok_lp.squeeze(0), mask.squeeze(0), mask.sum().item()

    def _compute_ref_log_probs(
        self,
        full_ids: torch.Tensor,
        prompt_len: int,
        inputs_for_fwd: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reference log probs with LoRA disabled."""
        ids = full_ids.unsqueeze(0) if full_ids.dim() == 1 else full_ids
        attn = torch.ones_like(ids)

        fwd_kwargs = {"input_ids": ids, "attention_mask": attn}
        for k in ("pixel_values", "image_grid_thw"):
            if k in inputs_for_fwd:
                fwd_kwargs[k] = inputs_for_fwd[k]

        # Disable LoRA for ref model
        self.model.disable_lora()
        self.model.eval()

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model.base_model(**fwd_kwargs)

            logits = outputs.logits
            resp_logits = logits[:, prompt_len - 1:-1, :]
            resp_labels = ids[:, prompt_len:]
            log_p = F.log_softmax(resp_logits, dim=-1)
            tok_lp = torch.gather(
                log_p, -1, resp_labels.unsqueeze(-1)
            ).squeeze(-1)
            mask = (resp_labels != self.pad_id).float()

        # Re-enable LoRA
        self.model.enable_lora()
        self.model.train()

        return tok_lp.squeeze(0), mask.squeeze(0)

    # -- episode rollout ---------------------------------------------------

    @torch.no_grad()
    def generate_episode_rollouts(self, episode: Dict) -> Optional[Dict]:
        """Generate K rollouts for each step, compute SP+GiGPO+SPWA advantages."""
        args = self.args
        K = args.num_samples
        goal = episode["goal"]
        steps = episode["steps"]
        num_steps = len(steps)

        if self.rank == 0:
            logger.info(f"[rollout] goal={goal[:60]}... steps={num_steps}")

        all_step_data = []
        all_rewards = np.zeros((K, num_steps), dtype=np.float32)
        action_history: List[str] = []

        for si, step in enumerate(steps):
            screenshot = step["screenshot"]
            gt_action = step["action"]
            image_w = step.get("image_w", 1040)
            image_h = step.get("image_h", 736)

            t_step = time.time()
            try:
                image = Image.open(screenshot).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to open {screenshot}: {e}")
                return None

            if self.rank == 0:
                logger.info(f"  [step {si}/{num_steps}] image loaded {time.time()-t_step:.1f}s")

            messages = build_step_messages(goal, action_history, screenshot)

            try:
                output_ids, prompt_len, inputs = self._generate_k_samples(
                    messages, image, K, args.max_new_tokens
                )
            except Exception as e:
                logger.warning(f"Generation failed at step {si}: {e}")
                return None

            if self.rank == 0:
                logger.info(f"  [step {si}/{num_steps}] K={K} gen done {time.time()-t_step:.1f}s prompt_len={prompt_len}")

            step_rewards = []
            step_texts = []
            truncated = []
            for k in range(K):
                resp_ids = output_ids[k, prompt_len:]
                text = self.processor.tokenizer.decode(
                    resp_ids, skip_special_tokens=True
                )
                step_texts.append(text)
                is_trunc = (resp_ids.shape[0] >= args.max_new_tokens)
                truncated.append(is_trunc)

                if is_trunc:
                    step_rewards.append(0.0)
                    all_rewards[k, si] = 0.0
                else:
                    reward, _ = compute_step_reward(
                        text, gt_action, image_w, image_h,
                        w_format=args.w_format,
                        w_type=args.w_type,
                        w_content=args.w_content,
                    )
                    step_rewards.append(reward)
                    all_rewards[k, si] = reward

            n_trunc = sum(truncated)
            if self.rank == 0 and n_trunc > 0:
                logger.info(f"    truncated: {n_trunc}/{K}")

            # Only compute log probs for non-truncated samples
            old_tok_lps = [None] * K
            masks = [None] * K
            ref_tok_lps = [None] * K
            for k in range(K):
                if truncated[k]:
                    continue
                tok_lp, mask, _ = self._compute_token_log_probs(
                    output_ids[k], prompt_len, inputs, with_grad=False
                )
                old_tok_lps[k] = tok_lp.detach()
                masks[k] = mask.detach()

                if args.kl_coef > 0:
                    ref_lp, _ = self._compute_ref_log_probs(
                        output_ids[k], prompt_len, inputs
                    )
                    ref_tok_lps[k] = ref_lp.detach()

            all_step_data.append({
                "step_idx": si,
                "output_ids": output_ids,
                "prompt_len": prompt_len,
                "inputs": inputs,
                "old_tok_lps": old_tok_lps,
                "masks": masks,
                "ref_tok_lps": ref_tok_lps,
                "truncated": truncated,
                "step_rewards": step_rewards,
                "step_texts": step_texts,
                "gt_action": gt_action,
                "image_w": image_w,
                "image_h": image_h,
            })

            action_history.append(format_gt_action_for_history(gt_action, si + 1))

        # -- Causal type masking (V14) --
        # Collect per-step texts and GT actions for masking
        # step_texts_by_step[t][k] = decoded text for rollout k at step t
        step_texts_by_step = [sd["step_texts"] for sd in all_step_data]
        gt_actions_list = [sd["gt_action"] for sd in all_step_data]

        all_rewards, type_error_steps = apply_causal_type_masking(
            all_rewards, step_texts_by_step, gt_actions_list,
        )

        n_type_errors = (type_error_steps < num_steps).sum()
        if self.rank == 0 and n_type_errors > 0:
            logger.info(f"  [causal mask] {n_type_errors}/{K} rollouts have type errors")

        # -- Compute advantages --
        if getattr(args, 'advantage_mode', 'sp') == 'grpo':
            result = compute_grpo_advantages(
                all_rewards,
                dapo_threshold=args.dapo_threshold,
                match_threshold=args.match_threshold,
            )
        else:
            result = compute_trajectory_advantages(
                all_rewards,
                match_threshold=args.match_threshold,
                spwa_decay=args.spwa_decay,
                dapo_threshold=args.dapo_threshold,
                step_adv_weight=args.step_adv_weight,
            )

        if result is None:
            return None  # DAPO filtered

        advantages, sp_scores, first_errors = result

        return {
            "episode_id": episode["episode_id"],
            "goal": goal,
            "num_steps": num_steps,
            "step_data": all_step_data,
            "rewards": all_rewards,
            "advantages": advantages,
            "sp_scores": sp_scores,
            "first_errors": first_errors,
            "type_error_steps": type_error_steps,
        }

    # -- policy gradient update --------------------------------------------

    def train_step(self, batch_rollouts: List[Dict]) -> Dict[str, float]:
        """One PPO update on accumulated episode rollouts."""
        args = self.args
        K = args.num_samples

        self.optimizer.zero_grad()

        total_pg_loss = 0.0
        total_kl = 0.0
        total_balance = 0.0
        total_clip_frac = 0.0
        n_seqs = 0
        n_zero_adv = 0
        n_total = 0
        n_balance = 0
        all_rewards = []
        all_sp = []
        advs_abs = []
        router_w_sum = 0.0

        total_seqs = sum(ep["num_steps"] * K for ep in batch_rollouts)
        total_seqs = max(total_seqs, 1)

        # Type error tracking (V14 causal masking)
        n_type_err_rollouts = 0
        n_total_rollouts = 0

        for ep in batch_rollouts:
            advantages = ep["advantages"]  # [K, num_steps]
            all_sp.extend(ep["sp_scores"].tolist())

            # Count rollouts with type errors
            if "type_error_steps" in ep:
                te = ep["type_error_steps"]
                n_type_err_rollouts += int((te < ep["num_steps"]).sum())
                n_total_rollouts += len(te)
            else:
                n_total_rollouts += K

            for si, step_data in enumerate(ep["step_data"]):
                for k in range(K):
                    n_total += 1
                    adv = advantages[k, si]
                    all_rewards.append(ep["rewards"][k, si])

                    # Skip truncated samples (no log probs computed)
                    if step_data.get("truncated", [False] * K)[k]:
                        n_zero_adv += 1
                        continue

                    if abs(adv) < 1e-8:
                        n_zero_adv += 1
                        continue

                    advs_abs.append(abs(adv))

                    # Recompute log probs with grad
                    tok_lp, mask, _ = self._compute_token_log_probs(
                        step_data["output_ids"][k],
                        step_data["prompt_len"],
                        step_data["inputs"],
                        with_grad=True,
                    )

                    old_tok_lp = step_data["old_tok_lps"][k]
                    adv_expanded = torch.full_like(mask, float(adv))

                    pg_loss, clip_frac, approx_kl = compute_policy_loss(
                        old_tok_lp, tok_lp, adv_expanded, mask, args.clip_range
                    )

                    kl_loss = torch.tensor(0.0, device=self.device)
                    if args.kl_coef > 0 and step_data["ref_tok_lps"]:
                        ref_lp = step_data["ref_tok_lps"][k]
                        kl_loss = compute_kl_penalty(tok_lp, ref_lp, mask)

                    # Balance loss
                    bal_loss, mean_w = self.model.compute_balance_loss()

                    loss = (pg_loss + args.kl_coef * kl_loss
                            + args.balance_weight * bal_loss) / total_seqs
                    loss.backward()

                    total_pg_loss += pg_loss.item()
                    total_kl += kl_loss.item()
                    if bal_loss.item() != 0:
                        total_balance += bal_loss.item()
                        router_w_sum += mean_w
                        n_balance += 1
                    total_clip_frac += clip_frac.item()
                    n_seqs += 1

                    del tok_lp, loss, pg_loss, kl_loss, bal_loss

        # -- All-reduce gradients across ranks --
        if self.world_size > 1:
            for p in self.model.parameters():
                if p.requires_grad:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        # -- Clip & step --
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)

        self.optimizer.step()
        self.global_step += 1

        return {
            "pg_loss": total_pg_loss / max(n_seqs, 1),
            "kl": total_kl / max(n_seqs, 1),
            "balance_loss": total_balance / max(n_balance, 1),
            "routing_w": router_w_sum / max(n_balance, 1),
            "clip_frac": total_clip_frac / max(n_seqs, 1),
            "grad_norm": grad_norm.item()
            if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
            "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "mean_sp": float(np.mean(all_sp)) if all_sp else 0.0,
            "nonzero_adv_frac": (n_total - n_zero_adv) / max(n_total, 1),
            "mean_abs_adv": float(np.mean(advs_abs)) if advs_abs else 0.0,
            "type_err": n_type_err_rollouts / max(n_total_rollouts, 1),
            "n_seqs": n_seqs,
        }

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
        if self.val_dataset is None or self.rank != 0:
            return {}

        logger.info(f"Running validation ({tag})...")
        all_rewards = []
        per_ep_results = []

        for ep_idx in range(min(len(self.val_dataset), 20)):
            episode = self.val_dataset[ep_idx]
            goal = episode["goal"]
            steps = episode["steps"]
            action_history: List[str] = []
            ep_rewards = []

            for si, step in enumerate(steps):
                screenshot = step["screenshot"]
                gt_action = step["action"]
                image_w = step.get("image_w", 1040)
                image_h = step.get("image_h", 736)

                try:
                    image = Image.open(screenshot).convert("RGB")
                except Exception:
                    continue

                messages = build_step_messages(goal, action_history, screenshot)
                inputs = self._tokenize_for_generation(messages, image)

                self._set_grad_checkpointing(False)
                self.model.eval()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    output_ids = self.model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        pixel_values=inputs.get("pixel_values"),
                        image_grid_thw=inputs.get("image_grid_thw"),
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
                ep_rewards.append(reward)
                all_rewards.append(reward)

                action_history.append(format_gt_action_for_history(gt_action, si + 1))

            per_ep_results.append({
                "episode_id": episode["episode_id"],
                "goal": goal[:100],
                "num_steps": len(steps),
                "mean_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
            })

        metrics = {
            "val/mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "val/n_episodes": len(per_ep_results),
            "val/n_steps": len(all_rewards),
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

    # -- main loop ---------------------------------------------------------

    def train(self):
        args = self.args

        if self.rank == 0:
            logger.info("=" * 60)
            logger.info("V14 GUI-360 Trajectory RL (Separate B + Conditional Comm + SP+GiGPO+SPWA)")
            logger.info(f"  K={args.num_samples}  temp={args.temperature}")
            logger.info(f"  num_comm_rounds={args.num_comm_rounds}")
            logger.info(f"  routing_noise_scale={args.routing_noise_scale}")
            logger.info(f"  spwa_decay={args.spwa_decay}  match_threshold={args.match_threshold}  step_adv_weight={args.step_adv_weight}")
            logger.info(f"  grad_accum={args.gradient_accumulation_steps}")
            logger.info(f"  epochs={args.num_epochs}  kl_coef={args.kl_coef}")
            logger.info(f"  lora_lr={args.lora_lr}  route_lr={args.route_lr}")
            logger.info(f"  balance_weight={args.balance_weight}")
            logger.info(f"  world_size={self.world_size}")
            logger.info(f"  episodes={len(self.train_dataset)}")
            logger.info(f"  dapo_threshold={args.dapo_threshold}")
            if self.resume_step > 0:
                logger.info(f"  RESUMING from step {self.resume_step}")
            logger.info("=" * 60)

        for epoch in range(args.num_epochs):
            self.sampler.set_epoch(epoch)

            epoch_metrics = defaultdict(list)
            batch_rollouts: List[Dict] = []
            skipped = 0

            t_epoch = time.time()

            skip_episodes = self.resume_step * args.gradient_accumulation_steps
            if self.rank == 0 and skip_episodes > 0:
                logger.info(f"  Skipping first {skip_episodes} episodes (resume)")

            for ep_idx, episode in enumerate(self.train_loader):
                if ep_idx < skip_episodes:
                    continue

                step_at_boundary = (
                    (ep_idx + 1) % args.gradient_accumulation_steps == 0
                )

                t_ep = time.time()

                try:
                    rollout = self.generate_episode_rollouts(episode)
                except Exception as e:
                    if self.rank == 0:
                        logger.warning(
                            f"Episode {ep_idx} failed: {e}\n"
                            f"{traceback.format_exc()}"
                        )
                    rollout = None

                if rollout is not None:
                    batch_rollouts.append(rollout)
                else:
                    skipped += 1
                    if self.rank == 0:
                        logger.info(f"  ep {ep_idx} skipped (DAPO filtered or failed)")

                if step_at_boundary:
                    if self.rank == 0:
                        logger.info(f"  train_step: {len(batch_rollouts)} rollouts, ep_idx={ep_idx}")
                    metrics = self.train_step(batch_rollouts)
                    if self.rank == 0:
                        logger.info(f"  train_step done: global_step={self.global_step}")

                    if batch_rollouts:
                        for k, v in metrics.items():
                            if isinstance(v, (int, float)):
                                epoch_metrics[k].append(v)

                        if self.rank == 0 and self.global_step % args.logging_steps == 0:
                            logger.info(
                                f"E{epoch} S{self.global_step} "
                                f"loss={metrics['pg_loss']:.4f} "
                                f"r={metrics['mean_reward']:.3f} "
                                f"sp={metrics['mean_sp']:.3f} "
                                f"kl={metrics['kl']:.4f} "
                                f"gnorm={metrics['grad_norm']:.2f} "
                                f"nz={metrics['nonzero_adv_frac']:.0%} "
                                f"adv={metrics['mean_abs_adv']:.3f} "
                                f"rw={metrics['routing_w']:.3f} "
                                f"te={metrics['type_err']:.0%} "
                                f"t={time.time()-t_ep:.1f}s"
                            )

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

                    batch_rollouts = []
                    torch.cuda.empty_cache()

            # -- Epoch summary --
            if self.rank == 0:
                dur = time.time() - t_epoch
                logger.info(f"{'='*60}")
                logger.info(
                    f"Epoch {epoch} done in {dur/60:.1f}min  "
                    f"skipped={skipped}"
                )
                for k in ["pg_loss", "mean_reward", "mean_sp", "kl",
                           "nonzero_adv_frac", "routing_w", "balance_loss",
                           "type_err"]:
                    vals = epoch_metrics.get(k, [0])
                    logger.info(f"  avg {k}: {np.mean(vals):.4f}")
                logger.info(f"{'='*60}")

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
        description="V14 GUI-360 Trajectory RL (Separate B + Conditional Comm)")

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

    # V14 cooperative LoRA
    parser.add_argument("--num_comm_rounds", type=int, default=2,
                        help="Number of iterative communication rounds (T)")
    parser.add_argument("--balance_weight", type=float, default=0.01)
    parser.add_argument("--routing_noise_scale", type=float, default=0.5,
                        help="Max routing noise std (linearly scaled per K)")
    parser.add_argument("--image_max_pixels", type=int, default=602112)

    # RL
    parser.add_argument("--num_samples", type=int, default=8, help="K rollouts per step")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--kl_coef", type=float, default=0.001)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    # Advantage mode
    parser.add_argument("--advantage_mode", type=str, default="sp",
                        choices=["sp", "grpo"],
                        help="Advantage mode: sp (SP+SPWA) or grpo (standard GRPO)")

    # SP+GiGPO+SPWA
    parser.add_argument("--match_threshold", type=float, default=0.5,
                        help="Step reward threshold for SP correct/incorrect")
    parser.add_argument("--spwa_decay", type=float, default=0.5,
                        help="SPWA decay factor for steps after first error")
    parser.add_argument("--dapo_threshold", type=float, default=0.1,
                        help="SP std threshold for DAPO filtering")
    parser.add_argument("--step_adv_weight", type=float, default=0.5,
                        help="Weight for step-level advantage (0=traj only, 1=step only)")

    # Reward weights
    parser.add_argument("--w_format", type=float, default=0.1)
    parser.add_argument("--w_type", type=float, default=0.2)
    parser.add_argument("--w_content", type=float, default=0.7)

    # Optimizer
    parser.add_argument("--lora_lr", type=float, default=1e-5)
    parser.add_argument("--route_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Training
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--val_steps", type=int, default=25)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    trainer = V14TrajectoryRLTrainer(args)
    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
