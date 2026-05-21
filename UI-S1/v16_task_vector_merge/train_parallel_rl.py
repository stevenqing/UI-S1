#!/usr/bin/env python3
"""V16 Parallel Dual-Prompt Cooperative RL — Hidden State Fusion.

Both prompts processed simultaneously:
  - Base model forward on grounder prompt → cache pooled hidden states per layer
  - Base model forward on actor prompt → Expert 1 uses cached grounder features,
    Expert 2 uses actor features, communication fuses them

Architecture:
  ┌─────────────────────────────────────────────────┐
  │  Base model forward (grounder prompt)           │
  │  → cache hidden states h_g[layer] at each layer │
  └─────────────────────────────────────────────────┘
                         ↓ cached (mean-pooled over seq)
  ┌─────────────────────────────────────────────────┐
  │  Base model forward (actor prompt)              │
  │  At each cooperative LoRA layer:                │
  │    h_1 = A_1 @ pool(h_g[layer])   ← grounder   │
  │    h_2 = A_2 @ x_actor            ← actor       │
  │    [communication rounds]                        │
  │    h_blend = r * h_1 + (1-r) * h_2              │
  │    delta = B @ h_blend * scaling                 │
  │    output = base_out + delta                     │
  └─────────────────────────────────────────────────┘
           ↓
      Generate in actor format (<tool_call>)

Based on v12_gui_360/train_trajectory_rl.py with:
  - DualInputCooperativeLoRALinear (Expert 1 uses alternative input)
  - DualPromptCooperativeWrapper (hidden state caching via forward hooks)
  - Extra base model forward pass (grounder) before each actor generation
  - Loads from V16 joint SVD checkpoint (V13 format)

Usage:
  srun --ntasks-per-node=1 bash -c '
    torchrun --nproc_per_node=4 --nnodes=$SLURM_NNODES \\
      --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR \\
      v16_task_vector_merge/train_parallel_rl.py \\
      --model_path checkpoints/Qwen2.5-VL-7B-Instruct \\
      --sft_checkpoint checkpoints/v16_coop_r128_attn \\
      --train_data v12_gui_360/data/gui360_train_2000.jsonl \\
      --output_dir v16_task_vector_merge/output/parallel_rl
  '
"""

import argparse
import datetime
import json
import math
import os
import re
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
logger = logging.getLogger("v16_parallel_rl")

# Project imports
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from v12_gui_360.reward import (
    compute_step_reward,
    compute_trajectory_advantages,
    compute_grpo_advantages,
    parse_action_from_text,
)
from v13_gui_360.iterative_cooperative_lora import IterativeCooperativeLoRALinear
from v13_gui_360.iterative_cooperative_wrapper import IterativeCooperativeVLMWrapper


# ═══════════════════════════════════════════════════════════════════════
# DualInputCooperativeLoRALinear
# ═══════════════════════════════════════════════════════════════════════

class DualInputCooperativeLoRALinear(IterativeCooperativeLoRALinear):
    """Cooperative LoRA that accepts a secondary input for Expert 1.

    When alt_input is set (from grounder hidden states), Expert 1 projects
    the alternative input instead of the actor input. Expert 2 always uses
    the actual forward pass input (actor).
    """

    def __init__(self, base_linear: nn.Linear, r: int = 128,
                 alpha: int = 256, dropout: float = 0.05):
        super().__init__(base_linear, r, alpha, dropout)
        # Alt input for Expert 1 (set by wrapper from grounder cache)
        object.__setattr__(self, "_alt_input", None)

    def set_alt_input(self, x_alt: Optional[torch.Tensor]):
        """Set alternative input for Expert 1 (from grounder forward pass).

        Args:
            x_alt: [1, D] mean-pooled grounder hidden state, or None to clear.
                   Will be broadcast to match actor sequence length.
        """
        object.__setattr__(self, "_alt_input", x_alt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dual input support.

        Args:
            x: [B, seq_len, D] actor input

        Returns:
            output: same shape as x, base + dual-input cooperative LoRA delta
        """
        base_out = self.base_linear(x)

        # If no route weight set, pass through base only (ref model mode)
        if self._route_weight is None:
            self._last_routing_weights = None
            return base_out

        x_drop = self.lora_dropout(x)
        dtype = x_drop.dtype

        # Compute routing weights: r = sigmoid(x @ w_route)
        w = self._route_weight.to(dtype)
        if w.shape[0] == x_drop.shape[-1]:
            logits = F.linear(x_drop, w.unsqueeze(0))  # [B, S, 1]

            if self._routing_noise_std > 0 and self.training:
                noise = torch.randn_like(logits) * self._routing_noise_std
                logits = logits + noise

            r = torch.sigmoid(logits)  # [B, S, 1]
        else:
            r = torch.full(
                (*x_drop.shape[:-1], 1), 0.5,
                device=x_drop.device, dtype=dtype,
            )
        self._last_routing_weights = r.detach()

        # Expert 1: use alternative input if available (grounder features)
        x_expert1 = self._alt_input if self._alt_input is not None else x_drop
        if x_expert1 is not self._alt_input:
            # Normal path: both experts use same input
            h_1 = F.linear(x_expert1, self.lora_A_1.to(dtype))
        else:
            # Dual-input path: Expert 1 uses grounder features
            # x_expert1 is [1, D] or [B, D] — broadcast to [B, S, r]
            alt = x_expert1.to(dtype)
            if alt.dim() == 2:
                # [B, D] → [B, 1, D] → broadcast with A_1
                alt = alt.unsqueeze(1)  # [B, 1, D]
            # Project and broadcast to match sequence length
            h_1 = F.linear(alt, self.lora_A_1.to(dtype))  # [B, 1, r]
            h_1 = h_1.expand(-1, x_drop.shape[1], -1)  # [B, S, r]

        # Expert 2: always uses actor input
        h_2 = F.linear(x_drop, self.lora_A_2.to(dtype))  # [B, S, r]

        # Iterative communication in r-space (same as parent)
        if self._comm_params is not None and not self._disable_comm:
            T = self._comm_params['T']
            gate_accum = 0
            for t in range(T):
                g_12 = torch.sigmoid(
                    F.linear(h_1, self._comm_params['gate_12'][t].to(dtype).unsqueeze(0))
                )
                h_1 = h_1 + g_12 * F.linear(h_2, self._comm_params['W_12'][t].to(dtype))

                g_21 = torch.sigmoid(
                    F.linear(h_2, self._comm_params['gate_21'][t].to(dtype).unsqueeze(0))
                )
                h_2 = h_2 + g_21 * F.linear(h_1, self._comm_params['W_21'][t].to(dtype))

                if self._record_gates:
                    gate_accum = gate_accum + g_12.detach() + g_21.detach()

            if self._record_gates:
                self._last_gate_mean = gate_accum / (2 * T)

        # Blend in r-space, single B matmul
        if self._inference_mode == "expert_1_only":
            h_blend = h_1
        elif self._inference_mode == "expert_2_only":
            h_blend = h_2
        else:
            h_blend = r * h_1 + (1 - r) * h_2
        delta = F.linear(h_blend, self.lora_B.to(dtype)) * self.scaling

        return base_out + delta


# ═══════════════════════════════════════════════════════════════════════
# DualPromptCooperativeWrapper
# ═══════════════════════════════════════════════════════════════════════

class DualPromptCooperativeWrapper(IterativeCooperativeVLMWrapper):
    """Cooperative VLM wrapper with dual-prompt hidden state caching.

    Extends IterativeCooperativeVLMWrapper to:
    1. Use DualInputCooperativeLoRALinear instead of IterativeCooperativeLoRALinear
    2. Cache grounder hidden states per layer via forward hooks
    3. Set cached grounder features as alt_input on Expert 1
    """

    def __init__(self, base_model: nn.Module, **kwargs):
        super().__init__(base_model, **kwargs)
        self._hooks: List = []

    def _replace_target_modules(self, r: int, alpha: int, dropout: float):
        """Replace nn.Linear with DualInputCooperativeLoRALinear (instead of parent's version)."""
        layers = self._get_transformer_layers()

        for layer_idx in range(len(layers)):
            layer = layers[layer_idx]
            for module_name in self.target_modules:
                if module_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    parent = layer.self_attn
                elif module_name in ("gate_proj", "up_proj", "down_proj"):
                    parent = layer.mlp
                else:
                    raise ValueError(f"Unknown target module: {module_name}")

                original = getattr(parent, module_name)
                coop_linear = DualInputCooperativeLoRALinear(
                    original, r, alpha, dropout,
                )
                coop_linear.set_route_weight(self.route_weights[layer_idx])
                setattr(parent, module_name, coop_linear)
                self.coop_modules.append(coop_linear)
                self._module_to_layer.append(layer_idx)

    def cache_grounder_hidden_states(self, grounder_inputs: dict):
        """Run base model on grounder prompt, cache mean-pooled hidden states per layer.

        After this call, each DualInputCooperativeLoRALinear module has its
        alt_input set to the mean-pooled grounder hidden state from its layer.

        Args:
            grounder_inputs: tokenized grounder prompt dict (input_ids, attention_mask,
                             pixel_values, image_grid_thw)
        """
        layers = self._get_transformer_layers()
        cached = {}
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # output is a tuple; first element is hidden states [B, S, D]
                if isinstance(output, tuple):
                    h = output[0]
                else:
                    h = output
                # Mean-pool over sequence dimension: [B, S, D] → [B, D]
                cached[layer_idx] = h.detach().mean(dim=1)  # [B, D]
            return hook_fn

        # Register forward hooks on each transformer layer
        for idx, layer in enumerate(layers):
            hook = layer.register_forward_hook(make_hook(idx))
            hooks.append(hook)

        # Run grounder forward pass (no grad, LoRA disabled for pure base model features)
        self.disable_lora()
        self.eval()

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                self.base_model(**grounder_inputs)

        self.enable_lora()
        self.train()

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Set cached grounder features on each dual-input module
        for module, layer_idx in zip(self.coop_modules, self._module_to_layer):
            if layer_idx in cached:
                module.set_alt_input(cached[layer_idx])

        if 0 in cached:
            logger.info(f"  Cached grounder hidden states: {len(cached)} layers, "
                        f"shape={cached[0].shape}")

    def clear_grounder_cache(self):
        """Clear cached grounder hidden states from all modules."""
        for module in self.coop_modules:
            module.set_alt_input(None)


# ═══════════════════════════════════════════════════════════════════════
# Prompt Templates
# ═══════════════════════════════════════════════════════════════════════

# Grounder prompt — for hidden state caching (not for text generation)
GROUNDER_PROMPT_TEMPLATE = """You are a helpful assistant. Given a screenshot of the current screen and user instruction, you need to output the position of the element you will operate.

The instruction is:
{instruction}

The history of actions are:
{history}

Output the coordinate of the element you will operate within <coordinate></coordinate> tag:
<coordinate> [x, y] </coordinate>"""

# Actor prompt — standard action generation
ACTOR_PROMPT_TEMPLATE = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

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


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# Prompt Formatting
# ═══════════════════════════════════════════════════════════════════════

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


def build_grounder_messages(goal: str, action_history: List[str],
                            image_path: str) -> list:
    """Build messages for the grounder pass (hidden state caching only)."""
    history_text = "\n".join(action_history) if action_history else "None"
    prompt_text = GROUNDER_PROMPT_TEMPLATE.format(
        instruction=goal,
        history=history_text,
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


def build_actor_messages(goal: str, action_history: List[str],
                         image_path: str) -> list:
    """Build messages for the actor pass (action generation)."""
    history_text = "\n".join(action_history) if action_history else "None"
    prompt_text = ACTOR_PROMPT_TEMPLATE.format(
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


# ═══════════════════════════════════════════════════════════════════════
# PPO Loss Helpers
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════

class V16ParallelRLTrainer:
    def __init__(self, args):
        self.args = args
        self.global_step = 0
        self.resume_step = 0

        self._setup_distributed()
        self._setup_model()
        self._load_resume_checkpoint()
        self._setup_data()
        self._setup_optimizer()

    # ── distributed ───────────────────────────────────────────────

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

    # ── model (DualPromptCooperativeWrapper) ─────────────────────

    def _setup_model(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

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

        self.model = DualPromptCooperativeWrapper(
            base_model=base_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=list(args.target_modules),
            balance_weight=args.balance_weight,
            num_comm_rounds=args.num_comm_rounds,
        )

        # Load SFT checkpoint (V16 joint SVD or V13 cooperative format)
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

        # Move to GPU
        self.model = self.model.to(self.device)

        dist.barrier()

        if self.rank == 0:
            counts = self.model.count_trainable_params()
            logger.info(f"Model ready: {counts['total']:,} trainable params "
                        f"(lora={counts['lora']:,}, route={counts['route_weights']:,}, "
                        f"comm={counts['comm']:,})")

    # ── resume from RL checkpoint ─────────────────────────────────

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

    # ── gradient checkpointing toggle ─────────────────────────────

    def _set_grad_checkpointing(self, enable: bool):
        if enable:
            self.model.gradient_checkpointing_enable()
        else:
            self.model.gradient_checkpointing_disable()
            for module in self.model.modules():
                if hasattr(module, "gradient_checkpointing"):
                    module.gradient_checkpointing = False

    # ── data ──────────────────────────────────────────────────────

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

    # ── optimizer (4 groups: lora, comm, route_weights, no_decay) ─

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
                "lr": args.comm_lr,
                "weight_decay": 0.0,
            })

        self.optimizer = torch.optim.AdamW(param_groups)

        if self.rank == 0:
            logger.info(
                f"Optimizer: lora_lr={args.lora_lr} route_lr={args.route_lr} "
                f"comm_lr={args.comm_lr} "
                f"decay={len(decay_params)} no_decay={len(no_decay_params)} "
                f"route={len(route_params)} comm={len(comm_params)}"
            )

    # ── tokenization helpers ──────────────────────────────────────

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

    # ── stop config ──────────────────────────────────────────────

    def _get_stop_config(self) -> dict:
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

    # ── grounder hidden state caching ────────────────────────────

    def _cache_grounder_features(self, messages: list, image: Image.Image):
        """Tokenize grounder prompt and cache hidden states in the model."""
        inputs = self._tokenize_for_generation(messages, image)
        self._set_grad_checkpointing(False)
        self.model.cache_grounder_hidden_states(inputs)
        self._set_grad_checkpointing(True)

    # ── actor generation (K samples with grounder features) ──────

    @torch.no_grad()
    def _generate_k_samples(
        self, messages: list, image: Image.Image, K: int, max_new_tokens: int
    ) -> Tuple[torch.Tensor, int, dict]:
        """Generate K action samples with grounder features cached."""
        inputs = self._tokenize_for_generation(messages, image)
        prompt_len = inputs["input_ids"].shape[1]

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

    # ── log prob computation ──────────────────────────────────────

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

        # Clear grounder cache + disable LoRA for ref model
        self.model.clear_grounder_cache()
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

        self.model.enable_lora()
        self.model.train()

        return tok_lp.squeeze(0), mask.squeeze(0)

    # ── episode rollout (parallel grounder+actor) ────────────────

    @torch.no_grad()
    def generate_episode_rollouts(self, episode: Dict) -> Optional[Dict]:
        """Generate K rollouts for each step with parallel grounder→actor."""
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

            # Step 1: Cache grounder hidden states
            grounder_msgs = build_grounder_messages(goal, action_history, screenshot)
            try:
                self._cache_grounder_features(grounder_msgs, image)
            except Exception as e:
                logger.warning(f"Grounder caching failed at step {si}: {e}")
                self.model.clear_grounder_cache()

            if self.rank == 0:
                logger.info(f"  [step {si}/{num_steps}] grounder cached ({time.time()-t_step:.1f}s)")

            # Step 2: Generate K actor samples (Expert 1 uses grounder features)
            actor_msgs = build_actor_messages(goal, action_history, screenshot)

            try:
                output_ids, prompt_len, inputs = self._generate_k_samples(
                    actor_msgs, image, K, args.max_new_tokens
                )
            except Exception as e:
                logger.warning(f"Actor generation failed at step {si}: {e}")
                self.model.clear_grounder_cache()
                return None

            # Step 3: Clear grounder cache
            self.model.clear_grounder_cache()

            if self.rank == 0:
                logger.info(f"  [step {si}/{num_steps}] K={K} actor gen done {time.time()-t_step:.1f}s prompt_len={prompt_len}")

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

            # Compute log probs — need to re-cache grounder features for policy eval
            grounder_msgs = build_grounder_messages(goal, action_history, screenshot)
            try:
                self._cache_grounder_features(grounder_msgs, image)
            except Exception:
                pass

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
                    # Re-cache grounder features after ref log prob computation
                    # (ref computation clears the cache)
                    try:
                        self._cache_grounder_features(grounder_msgs, image)
                    except Exception:
                        pass

            self.model.clear_grounder_cache()

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
                # Store grounder info for re-caching during train_step
                "grounder_msgs": grounder_msgs,
                "screenshot": screenshot,
            })

            action_history.append(format_gt_action_for_history(gt_action, si + 1))

        # ── Compute advantages ──
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
            return None

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
        }

    # ── policy gradient update ────────────────────────────────────

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

        for ep in batch_rollouts:
            advantages = ep["advantages"]
            all_sp.extend(ep["sp_scores"].tolist())

            for si, step_data in enumerate(ep["step_data"]):
                # Re-cache grounder features for this step's policy gradient
                grounder_msgs = step_data.get("grounder_msgs")
                screenshot = step_data.get("screenshot")
                if grounder_msgs and screenshot:
                    try:
                        image = Image.open(screenshot).convert("RGB")
                        self._cache_grounder_features(grounder_msgs, image)
                    except Exception:
                        pass

                for k in range(K):
                    n_total += 1
                    adv = advantages[k, si]
                    all_rewards.append(ep["rewards"][k, si])

                    if step_data.get("truncated", [False] * K)[k]:
                        n_zero_adv += 1
                        continue

                    if abs(adv) < 1e-8:
                        n_zero_adv += 1
                        continue

                    advs_abs.append(abs(adv))

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

                # Clear grounder cache after processing this step
                self.model.clear_grounder_cache()

        # ── All-reduce gradients across ranks ──
        if self.world_size > 1:
            for p in self.model.parameters():
                if p.requires_grad:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

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
            "n_seqs": n_seqs,
        }

    # ── checkpoint ────────────────────────────────────────────────

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

    # ── validation ────────────────────────────────────────────────

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

                # Cache grounder features
                grounder_msgs = build_grounder_messages(goal, action_history, screenshot)
                try:
                    self._cache_grounder_features(grounder_msgs, image)
                except Exception:
                    pass

                # Actor pass (greedy for val)
                actor_msgs = build_actor_messages(goal, action_history, screenshot)
                inputs = self._tokenize_for_generation(actor_msgs, image)

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
                self.model.clear_grounder_cache()

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

    # ── main loop ─────────────────────────────────────────────────

    def train(self):
        args = self.args

        if self.rank == 0:
            logger.info("=" * 60)
            logger.info("V16 Parallel Dual-Prompt RL (Hidden State Fusion)")
            logger.info(f"  K={args.num_samples}  temp={args.temperature}")
            logger.info(f"  routing_noise_scale={args.routing_noise_scale}")
            logger.info(f"  num_comm_rounds={args.num_comm_rounds}")
            logger.info(f"  spwa_decay={args.spwa_decay}  match_threshold={args.match_threshold}  step_adv_weight={args.step_adv_weight}")
            logger.info(f"  grad_accum={args.gradient_accumulation_steps}")
            logger.info(f"  epochs={args.num_epochs}  kl_coef={args.kl_coef}")
            logger.info(f"  lora_lr={args.lora_lr}  route_lr={args.route_lr}  comm_lr={args.comm_lr}")
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
                                f"t={time.time()-t_ep:.1f}s"
                            )

                        if (self.rank == 0 and args.save_steps > 0
                                and self.global_step % args.save_steps == 0):
                            self.save_checkpoint(
                                f"epoch-{epoch}_step-{self.global_step}"
                            )

                    if (args.val_steps > 0
                            and self.global_step % args.val_steps == 0):
                        if self.rank == 0:
                            self.validate(tag=f"epoch-{epoch}_step-{self.global_step}")
                        if self.world_size > 1:
                            dist.barrier()

                    batch_rollouts = []
                    torch.cuda.empty_cache()

            # ── Epoch summary ──
            if self.rank == 0:
                dur = time.time() - t_epoch
                logger.info(f"{'='*60}")
                logger.info(
                    f"Epoch {epoch} done in {dur/60:.1f}min  "
                    f"skipped={skipped}"
                )
                for k in ["pg_loss", "mean_reward", "mean_sp", "kl",
                           "nonzero_adv_frac", "routing_w", "balance_loss"]:
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


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="V16 Parallel Dual-Prompt RL (Hidden State Fusion)")

    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sft_checkpoint", type=str, default="",
                        help="Path to V16 joint SVD cooperative checkpoint dir")
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

    # V13/V16 cooperative
    parser.add_argument("--balance_weight", type=float, default=0.01)
    parser.add_argument("--num_comm_rounds", type=int, default=2)
    parser.add_argument("--routing_noise_scale", type=float, default=0.5)
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
                        choices=["sp", "grpo"])

    # SP+GiGPO+SPWA
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--spwa_decay", type=float, default=0.5)
    parser.add_argument("--dapo_threshold", type=float, default=0.1)
    parser.add_argument("--step_adv_weight", type=float, default=0.5)

    # Reward weights
    parser.add_argument("--w_format", type=float, default=0.1)
    parser.add_argument("--w_type", type=float, default=0.2)
    parser.add_argument("--w_content", type=float, default=0.7)

    # Optimizer
    parser.add_argument("--lora_lr", type=float, default=1e-5)
    parser.add_argument("--route_lr", type=float, default=1e-3)
    parser.add_argument("--comm_lr", type=float, default=1e-3,
                        help="Learning rate for communication params")
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

    trainer = V16ParallelRLTrainer(args)
    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
