#!/usr/bin/env python3
"""v11 Cooperative LoRA Trajectory GRPO Trainer — DDP + CooperativeVLMWrapper.

Architecture:
  Single forward pass using CooperativeVLMWrapper (v6.5):
    - Token routing: Image tokens → LoRA_V, text tokens → LoRA_A
    - Per-layer h-space communication: h_v += gate_av * W_av @ h_a (and vice versa)
    - Output: single model generates action directly (no separate grounder/actor)
  Fully differentiable communication channel.

Training (Trajectory GRPO):
  For each episode (multi-step task):
    1. Iterate through steps sequentially
    2. At each step: generate K action samples
    3. Compute dense multi-component reward per step
    4. Compute discounted returns (γ=0.5) backward through trajectory
    5. Dual-level advantage normalization (episode + step)
    6. DAPO filtering on collapsed advantages
    7. Clipped policy gradient + KL penalty

Infrastructure: DDP + manual gradient all-reduce (proven pattern from v10).

Usage:
  srun --ntasks-per-node=1 bash -c '
    torchrun --nproc_per_node=4 --nnodes=$SLURM_NNODES \\
      --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR \\
      v11/train_trajectory_grpo.py --model_path ... --train_data ... --output_dir ...
  '
"""

import argparse
import datetime
import json
import os
import re
import socket
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
logger = logging.getLogger("v11_traj_grpo")

# Project imports
_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from v11.reward import (
    compute_step_reward,
    compute_discounted_returns,
    normalize_advantages,
    should_filter_dapo,
    parse_action_from_text,
)

# ── Prompt templates ──────────────────────────────────────────────

# System prompt for the cooperative model (unified grounder+actor)
SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a screenshot and a task instruction. "
    "Perform the next action to complete the task.\n\n"
    "Action space:\n"
    '  click: {"action": "click", "coordinate": [x, y]}\n'
    '  long_press: {"action": "long_press", "coordinate": [x, y]}\n'
    '  type: {"action": "type", "text": "content"}\n'
    '  swipe: {"action": "swipe", "coordinate": [x1, y1], "endCoordinate": [x2, y2]}\n'
    '  open: {"action": "open", "text": "app name"}\n'
    '  system_button: {"action": "system_button", "button": "back|home|recent"}\n'
    '  wait: {"action": "wait"}\n'
    '  terminate: {"action": "terminate", "status": "success|failure"}\n\n'
    "Output format: <action>{JSON action}</action>"
)


# ── Dataset ───────────────────────────────────────────────────────

class EpisodeDataset(Dataset):
    """Load episode-grouped JSONL data for trajectory GRPO.

    Each sample is a full episode:
    {
        "episode_id": int,
        "goal": str,
        "num_steps": int,
        "steps": [{"step_idx": int, "action": {...}, "screenshot": str,
                    "image_w": int, "image_h": int}, ...]
    }
    """

    def __init__(self, jsonl_path: str, max_episodes: int = 0):
        self.episodes = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                # Validate: all screenshots must exist
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


# ── Prompt formatting ─────────────────────────────────────────────

def build_step_messages(goal: str, action_history: List[str],
                        image_path: str) -> list:
    """Build chat messages for one step in a trajectory.

    Args:
        goal: Task instruction
        action_history: List of previous action strings (from prior steps)
        image_path: Path to current screenshot
    """
    parts = [f"Task: {goal}"]
    if action_history:
        parts.append("\nPrevious actions:")
        for i, act in enumerate(action_history):
            parts.append(f"  Step {i+1}: {act}")
    parts.append("\nPerform the next action based on the current screenshot.")
    user_text = "\n".join(parts)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        },
    ]


# ── GRPO loss helpers ─────────────────────────────────────────────

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


# ── Trainer ───────────────────────────────────────────────────────

class V11TrajectoryGRPOTrainer:
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
            hostname = socket.gethostname()
            logger.info(
                f"Distributed ready: rank={self.rank} world={self.world_size} "
                f"host={hostname}"
            )

    # ── model (CooperativeVLMWrapper) ─────────────────────────────

    def _setup_model(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from verl.models.cooperative.cooperative_wrapper import CooperativeVLMWrapper

        args = self.args

        # Processor / tokenizer
        self.processor = AutoProcessor.from_pretrained(args.model_path)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.pad_id = self.processor.tokenizer.pad_token_id

        # 1. Load base model (bf16)
        if self.rank == 0:
            logger.info(f"Loading base model: {args.model_path}")
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        # 2. Wrap with CooperativeVLMWrapper (freezes base, adds LoRA)
        self.model = CooperativeVLMWrapper(
            base_model=base_model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=list(args.target_modules),
            num_agents=2,
            cooperative_comm=True,
            gate_type=args.gate_type,
            gate_init=args.gate_init,
            routing_mode=args.routing_mode,
            balance_weight=args.balance_weight,
            bind_weight=0.0,  # no binding loss in GRPO
        )

        # 3. Enable gradient checkpointing
        # enable_input_require_grads is required so that gradients flow
        # backward through frozen embeddings to LoRA layers when
        # gradient checkpointing recomputes forward passes.
        self.model.base_model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()
        # Ensure LoRA + communication + router params require grad
        for name, param in self.model.named_parameters():
            if ("lora_" in name or "gate_" in name
                    or name.endswith(("W_av", "W_va"))
                    or "routers." in name):
                param.requires_grad = True

        # 4. Move to GPU (no DDP wrapper — manual gradient all-reduce)
        self.model = self.model.to(self.device)

        # 5. Auxiliary coordinate prediction head (direct LoRA_V gradient signal)
        #    Pools hidden states at image token positions → predicts GT coordinate.
        #    Gradient flows: coord_loss → coord_head → hidden_state → LoRA_V
        self._last_hidden_state = None
        if args.aux_coord_weight > 0:
            hidden_size = self.model.base_model.config.hidden_size
            self.coord_head = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.GELU(),
                nn.Linear(256, 2),
                nn.Sigmoid(),  # output in [0, 1] for normalized coords
            ).to(device=self.device, dtype=torch.bfloat16)
            # Hook on final norm to capture last hidden state (outside
            # gradient-checkpointed region, so backprop works correctly)
            def _norm_hook(module, input, output):
                self._last_hidden_state = output
            self._norm_hook_handle = (
                self.model.base_model.model.language_model.norm.register_forward_hook(_norm_hook)
            )
        else:
            self.coord_head = None

        dist.barrier()

        if self.rank == 0:
            trainable = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            comm_n = sum(
                p.numel()
                for n, p in self.model.named_parameters()
                if p.requires_grad and any(
                    s in n for s in ("gate_av", "gate_va", "W_av", "W_va")
                )
            )
            router_n = sum(
                p.numel()
                for n, p in self.model.named_parameters()
                if p.requires_grad and "routers." in n
            )
            coord_n = sum(
                p.numel() for p in self.coord_head.parameters()
            ) if self.coord_head else 0
            logger.info(
                f"Model ready: {trainable:,} trainable params "
                f"(comm={comm_n:,}, router={router_n:,}, coord_head={coord_n:,})"
            )
            logger.info(f"  routing_mode={args.routing_mode}, balance_weight={args.balance_weight}")

    # ── resume from checkpoint ────────────────────────────────────

    def _load_resume_checkpoint(self):
        """Load cooperative LoRA weights from a checkpoint directory."""
        ckpt_dir = getattr(self.args, "resume_from", "")
        if not ckpt_dir:
            return

        if self.rank == 0:
            logger.info(f"Resuming from checkpoint: {ckpt_dir}")

        # Load cooperative LoRA state dict
        coop_dir = os.path.join(ckpt_dir, "cooperative")
        if os.path.isdir(coop_dir):
            for fname in ("lora_v.pt", "lora_a.pt", "lora_comm.pt", "lora_router.pt"):
                fpath = os.path.join(coop_dir, fname)
                if os.path.exists(fpath):
                    state = torch.load(fpath, map_location=self.device, weights_only=True)
                    missing, unexpected = self.model.load_state_dict(state, strict=False)
                    if self.rank == 0:
                        logger.info(f"  Loaded {fname}: {len(state)} tensors")

        # Load training state
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
            # Thorough disable: HF's disable may not reach all submodules
            # (e.g., Qwen2.5-VL's nested language_model). Force-clear.
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

    # ── optimizer (4-group: decay, no-decay, communication, router) ─

    def _setup_optimizer(self):
        args = self.args

        comm_suffixes = {"gate_av", "gate_va", "W_av", "W_va"}
        comm_params, router_params, decay_params, no_decay_params = [], [], [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "routers." in name:
                router_params.append(param)
            elif name.split(".")[-1] in comm_suffixes:
                comm_params.append(param)
            elif param.dim() == 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        # Build param groups (filter out empty)
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
        if comm_params:
            param_groups.append({
                "params": comm_params,
                "lr": args.comm_lr,
                "weight_decay": 0.0,
            })
        if router_params:
            param_groups.append({
                "params": router_params,
                "lr": args.comm_lr,  # same higher LR as comm
                "weight_decay": 0.0,
            })

        # Coord prediction head params (same higher LR as comm)
        if self.coord_head is not None:
            coord_head_params = list(self.coord_head.parameters())
            if coord_head_params:
                param_groups.append({
                    "params": coord_head_params,
                    "lr": args.comm_lr,
                    "weight_decay": 0.0,
                })

        self.optimizer = torch.optim.AdamW(param_groups)

        if self.rank == 0:
            logger.info(
                f"Optimizer: lora_lr={args.lora_lr} comm_lr={args.comm_lr} "
                f"decay={len(decay_params)} no_decay={len(no_decay_params)} "
                f"comm={len(comm_params)}"
            )

    # ── tokenization helpers ──────────────────────────────────────

    def _tokenize_for_generation(
        self, messages: list, image: Image.Image,
    ) -> dict:
        """Tokenize messages for generation."""
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=False
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    # ── generation (K samples for one step) ───────────────────────

    @torch.no_grad()
    def _generate_k_samples(
        self, messages: list, image: Image.Image, K: int, max_new_tokens: int
    ) -> Tuple[torch.Tensor, int, dict]:
        """Generate K action samples for one step.

        Returns:
            (output_ids [K, seq_len], prompt_len, inputs_for_fwd)
        """
        inputs = self._tokenize_for_generation(messages, image)
        prompt_len = inputs["input_ids"].shape[1]

        # Expand inputs for K samples
        gen_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                rep = [K] + [1] * (v.dim() - 1)
                gen_inputs[k] = v.repeat(*rep)
            else:
                gen_inputs[k] = v

        # Disable gradient checkpointing for generation (KV cache)
        self._set_grad_checkpointing(False)
        self.model.eval()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output_ids = self.model.generate(
                input_ids=gen_inputs["input_ids"],
                attention_mask=gen_inputs.get("attention_mask"),
                pixel_values=gen_inputs.get("pixel_values"),
                image_grid_thw=gen_inputs.get("image_grid_thw"),
                max_new_tokens=max_new_tokens,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                do_sample=True,
            )

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
        """Compute per-token log probs for the response portion.

        NOTE: Do NOT clear the token mask after forward. With gradient
        checkpointing, backward() re-runs forward and needs the mask.
        The mask is overwritten by the next call anyway.
        """
        ids = full_ids.unsqueeze(0) if full_ids.dim() == 1 else full_ids
        attn = torch.ones_like(ids)

        # With learned routing, no token mask needed — the per-layer router
        # decides routing from hidden states. For hard routing fallback,
        # build mask from image token IDs.
        if self.args.routing_mode == "learned":
            self.model._set_token_mask(None)
        else:
            from verl.models.cooperative.cooperative_wrapper import IMAGE_PAD_ID
            token_mask = (ids == IMAGE_PAD_ID)
            self.model._set_token_mask(token_mask)

        fwd_kwargs = {"input_ids": ids, "attention_mask": attn}
        for k in ("pixel_values", "image_grid_thw"):
            if k in inputs_for_fwd:
                fwd_kwargs[k] = inputs_for_fwd[k]

        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model.base_model(**fwd_kwargs)

            logits = outputs.logits
            resp_logits = logits[:, prompt_len - 1:-1, :]
            resp_labels = ids[:, prompt_len:]
            log_p = torch.nn.functional.log_softmax(resp_logits, dim=-1)
            tok_lp = torch.gather(
                log_p, -1, resp_labels.unsqueeze(-1)
            ).squeeze(-1)
            mask = (resp_labels != self.pad_id).float()

        # Do NOT clear token mask here — gradient checkpointing needs it
        # during backward() recomputation.

        return tok_lp.squeeze(0), mask.squeeze(0), mask.sum().item()

    def _compute_ref_log_probs(
        self,
        full_ids: torch.Tensor,
        prompt_len: int,
        inputs_for_fwd: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute reference (base model) log probs by disabling LoRA.

        With CooperativeVLMWrapper, the ref model = base model with all LoRA
        deltas zeroed. We switch to eval mode so that CooperativeLoRALinear
        returns base_out when token_mask is None (instead of raising RuntimeError
        which only fires in training mode).
        """
        ids = full_ids.unsqueeze(0) if full_ids.dim() == 1 else full_ids
        attn = torch.ones_like(ids)

        fwd_kwargs = {"input_ids": ids, "attention_mask": attn}
        for k in ("pixel_values", "image_grid_thw"):
            if k in inputs_for_fwd:
                fwd_kwargs[k] = inputs_for_fwd[k]

        # For ref log probs, we need base model output with NO LoRA delta.
        # With learned routing (routing_mode="learned"), mask=None still uses
        # the router. Temporarily switch to "hard" mode so mask=None + eval
        # → base_out only (cooperative_lora.py line 264-271).
        saved_modes = []
        if self.args.routing_mode == "learned":
            for m in self.model.coop_modules:
                saved_modes.append(m.routing_mode)
                m.routing_mode = "hard"

        self.model.eval()
        self.model._set_token_mask(None)
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model.base_model(**fwd_kwargs)

            logits = outputs.logits
            resp_logits = logits[:, prompt_len - 1:-1, :]
            resp_labels = ids[:, prompt_len:]
            log_p = torch.nn.functional.log_softmax(resp_logits, dim=-1)
            tok_lp = torch.gather(
                log_p, -1, resp_labels.unsqueeze(-1)
            ).squeeze(-1)
            mask = (resp_labels != self.pad_id).float()

        # Restore routing mode and training mode
        if saved_modes:
            for m, mode in zip(self.model.coop_modules, saved_modes):
                m.routing_mode = mode
        self.model.train()

        return tok_lp.squeeze(0), mask.squeeze(0)

    # ── balance loss (learned routing) ───────────────────────────

    def _compute_balance_loss(self) -> Tuple[torch.Tensor, float]:
        """Compute balance loss for learned routers.

        Pushes mean routing weight per layer toward 0.5 via binary entropy.
        Prevents router collapse (all tokens → one expert).

        Returns (loss, mean_router_w).
        """
        if self.args.routing_mode != "learned" or self.args.balance_weight <= 0:
            return torch.tensor(0.0, device=self.device), 0.5

        balance_terms = []
        w_sum = 0.0
        w_count = 0
        eps = 1e-6
        for m in self.model.coop_modules:
            if not hasattr(m, '_last_router_w') or m._last_router_w is None:
                continue
            w = m._last_router_w.mean()  # mean routing weight
            neg_entropy = w * torch.log(w + eps) + (1 - w) * torch.log(1 - w + eps)
            balance_terms.append(neg_entropy)
            w_sum += w.detach().item()
            w_count += 1

        if not balance_terms:
            return torch.tensor(0.0, device=self.device), 0.5

        loss = torch.stack(balance_terms).mean()
        mean_w = w_sum / w_count if w_count > 0 else 0.5
        return loss, mean_w

    # ── auxiliary coordinate loss (LoRA_V gradient signal) ────────

    def _compute_coord_aux_loss(
        self,
        full_ids: torch.Tensor,
        gt_action: Dict[str, Any],
        image_w: int,
        image_h: int,
    ) -> torch.Tensor:
        """Predict GT coordinate from pooled image token hidden states.

        Gives LoRA_V direct gradient signal. Only fires for click/long_press
        actions that have a GT coordinate.

        Uses self._last_hidden_state captured by the norm hook during the
        most recent forward pass.
        """
        if self.coord_head is None or self._last_hidden_state is None:
            return torch.tensor(0.0, device=self.device)

        gt_type = gt_action.get("action", "")
        if gt_type == "left_click":
            gt_type = "click"
        gt_coord = gt_action.get("coordinate")
        if gt_type not in ("click", "long_press") or gt_coord is None:
            return torch.tensor(0.0, device=self.device)

        from verl.models.cooperative.cooperative_wrapper import IMAGE_PAD_ID
        ids = full_ids.unsqueeze(0) if full_ids.dim() == 1 else full_ids
        img_mask = (ids == IMAGE_PAD_ID)  # [B, seq_len]

        if img_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        hidden = self._last_hidden_state  # [B, seq_len, hidden_size]
        img_hidden = hidden[img_mask]  # [n_img_tokens, hidden_size]
        pooled = img_hidden.mean(dim=0)  # [hidden_size]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred_coord = self.coord_head(pooled)  # [2], in [0, 1]

        gt_norm = torch.tensor(
            [gt_coord[0] / max(image_w, 1), gt_coord[1] / max(image_h, 1)],
            device=self.device, dtype=pred_coord.dtype,
        )

        return F.mse_loss(pred_coord, gt_norm)

    # ── episode rollout ───────────────────────────────────────────

    @torch.no_grad()
    def generate_episode_rollouts(self, episode: Dict) -> Optional[Dict]:
        """Generate K rollouts for each step in an episode.

        For each step:
            1. Build prompt with goal + action history
            2. Generate K action samples
            3. Compute per-step rewards
            4. Store log probs for policy gradient

        Returns dict with per-step, per-K data for the training update.
        """
        args = self.args
        K = args.num_samples
        goal = episode["goal"]
        steps = episode["steps"]
        num_steps = len(steps)

        # Per-step, per-K storage
        all_step_data = []  # List of dicts, one per step
        all_rewards = np.zeros((K, num_steps), dtype=np.float32)

        # Action history for trajectory context (use GT for training)
        action_history: List[str] = []

        for si, step in enumerate(steps):
            screenshot = step["screenshot"]
            gt_action = step["action"]
            image_w = step.get("image_w", 1080)
            image_h = step.get("image_h", 2400)

            try:
                image = Image.open(screenshot).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to open {screenshot}: {e}")
                return None

            # Build prompt
            messages = build_step_messages(goal, action_history, screenshot)

            # Generate K samples
            try:
                output_ids, prompt_len, inputs = self._generate_k_samples(
                    messages, image, K, args.max_new_tokens
                )
            except Exception as e:
                logger.warning(f"Generation failed at step {si}: {e}")
                return None

            # Decode and compute rewards
            step_rewards = []
            step_texts = []
            for k in range(K):
                resp_ids = output_ids[k, prompt_len:]
                text = self.processor.tokenizer.decode(
                    resp_ids, skip_special_tokens=True
                )
                step_texts.append(text)

                reward, _ = compute_step_reward(
                    text, gt_action, image_w, image_h,
                    w_format=args.w_format,
                    w_type=args.w_type,
                    w_content=args.w_content,
                )
                step_rewards.append(reward)
                all_rewards[k, si] = reward

            # Compute old log probs for each K sample
            old_tok_lps = []
            masks = []
            ref_tok_lps = []
            for k in range(K):
                tok_lp, mask, _ = self._compute_token_log_probs(
                    output_ids[k], prompt_len, inputs, with_grad=False
                )
                old_tok_lps.append(tok_lp.detach())
                masks.append(mask.detach())

                if args.kl_coef > 0:
                    ref_lp, _ = self._compute_ref_log_probs(
                        output_ids[k], prompt_len, inputs
                    )
                    ref_tok_lps.append(ref_lp.detach())

            all_step_data.append({
                "step_idx": si,
                "output_ids": output_ids,  # [K, seq_len]
                "prompt_len": prompt_len,
                "inputs": inputs,
                "old_tok_lps": old_tok_lps,  # list of K tensors
                "masks": masks,  # list of K tensors
                "ref_tok_lps": ref_tok_lps,  # list of K tensors (empty if kl_coef=0)
                "step_rewards": step_rewards,  # list of K floats
                "step_texts": step_texts,  # list of K strings
                "gt_action": gt_action,  # for aux coord loss
                "image_w": image_w,
                "image_h": image_h,
            })

            # Update action history with GT action for next step's context
            action_str = json.dumps(gt_action, ensure_ascii=False)
            action_history.append(f"<action>{action_str}</action>")

        # ── Compute discounted returns for each K rollout ──
        # all_rewards: [K, num_steps]
        returns = np.zeros_like(all_rewards)  # [K, num_steps]
        for k in range(K):
            returns[k] = compute_discounted_returns(
                all_rewards[k].tolist(), gamma=args.gamma
            )

        # ── DAPO filtering ──
        if should_filter_dapo(returns, std_threshold=args.dapo_threshold):
            return None  # Skip this episode (advantage collapsed)

        # ── Compute normalized advantages ──
        advantages = normalize_advantages(returns)  # [K, num_steps]

        return {
            "episode_id": episode["episode_id"],
            "goal": goal,
            "num_steps": num_steps,
            "step_data": all_step_data,
            "rewards": all_rewards,       # [K, num_steps]
            "returns": returns,           # [K, num_steps]
            "advantages": advantages,     # [K, num_steps]
        }

    # ── policy gradient update ────────────────────────────────────

    def train_step(self, batch_rollouts: List[Dict]) -> Dict[str, float]:
        """One GRPO update on accumulated episode rollouts.

        Iterates over all episodes, all steps, all K samples.
        Loss is normalized by total number of sequences.
        """
        args = self.args
        K = args.num_samples

        self.optimizer.zero_grad()

        total_pg_loss = 0.0
        total_kl = 0.0
        total_aux_coord = 0.0
        total_balance = 0.0
        total_clip_frac = 0.0
        n_seqs = 0
        n_zero_adv = 0
        n_total = 0
        n_coord = 0
        n_balance = 0
        all_rewards = []
        advs_abs = []
        router_w_sum = 0.0

        # Total normalization factor (batch_rollouts already contains all
        # accumulated episodes, so do NOT multiply by gradient_accumulation_steps)
        total_seqs = sum(
            ep["num_steps"] * K for ep in batch_rollouts
        )
        total_seqs = max(total_seqs, 1)

        for ep in batch_rollouts:
            advantages = ep["advantages"]  # [K, num_steps]

            for si, step_data in enumerate(ep["step_data"]):
                for k in range(K):
                    n_total += 1
                    adv = advantages[k, si]
                    all_rewards.append(ep["rewards"][k, si])

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

                    # Aux coord loss: direct LoRA_V gradient from coord prediction
                    aux_loss = torch.tensor(0.0, device=self.device)
                    if args.aux_coord_weight > 0 and self.coord_head is not None:
                        aux_loss = self._compute_coord_aux_loss(
                            step_data["output_ids"][k],
                            step_data["gt_action"],
                            step_data["image_w"],
                            step_data["image_h"],
                        )

                    # Balance loss: prevents learned router collapse
                    bal_loss, mean_w = self._compute_balance_loss()

                    loss = (pg_loss + args.kl_coef * kl_loss
                            + args.aux_coord_weight * aux_loss
                            + args.balance_weight * bal_loss) / total_seqs
                    loss.backward()

                    total_pg_loss += pg_loss.item()
                    total_kl += kl_loss.item()
                    if aux_loss.item() > 0:
                        total_aux_coord += aux_loss.item()
                        n_coord += 1
                    if bal_loss.item() != 0:
                        total_balance += bal_loss.item()
                        router_w_sum += mean_w
                        n_balance += 1
                    total_clip_frac += clip_frac.item()
                    n_seqs += 1

                    del tok_lp, loss, pg_loss, kl_loss, aux_loss, bal_loss

        # ── All-reduce gradients across ranks ──
        all_params = list(self.model.parameters())
        if self.coord_head is not None:
            all_params += list(self.coord_head.parameters())
        if self.world_size > 1:
            for p in all_params:
                if p.requires_grad:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        # ── Clip & step ──
        trainable = [p for p in all_params if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)

        self.optimizer.step()
        self.global_step += 1

        # ── Collect gate diagnostics ──
        gate_info = self._get_gate_info()

        router_info = {}
        if n_balance > 0:
            router_info["balance_loss"] = total_balance / n_balance
            router_info["router_w"] = router_w_sum / n_balance

        return {
            "pg_loss": total_pg_loss / max(n_seqs, 1),
            "kl": total_kl / max(n_seqs, 1),
            "aux_coord": total_aux_coord / max(n_coord, 1),
            "clip_frac": total_clip_frac / max(n_seqs, 1),
            "grad_norm": grad_norm.item()
            if isinstance(grad_norm, torch.Tensor)
            else float(grad_norm),
            "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "nonzero_adv_frac": (n_total - n_zero_adv) / max(n_total, 1),
            "mean_abs_adv": float(np.mean(advs_abs)) if advs_abs else 0.0,
            "n_seqs": n_seqs,
            **gate_info,
            **router_info,
        }

    def _get_gate_info(self) -> Dict[str, float]:
        """Collect gate magnitude info for monitoring communication learning."""
        gate_avs = []
        gate_vas = []
        for module in self.model.coop_modules:
            if hasattr(module, "gate_av"):
                if module.gate_type == "tanh":
                    gate_avs.append(torch.tanh(module.gate_av).item())
                    gate_vas.append(torch.tanh(module.gate_va).item())
                else:
                    gate_avs.append(torch.sigmoid(module.gate_av).item())
                    gate_vas.append(torch.sigmoid(module.gate_va).item())
        if not gate_avs:
            return {}
        return {
            "gate_av_mean": float(np.mean(gate_avs)),
            "gate_va_mean": float(np.mean(gate_vas)),
            "gate_av_max": float(np.max(np.abs(gate_avs))),
            "gate_va_max": float(np.max(np.abs(gate_vas))),
        }

    # ── checkpoint ────────────────────────────────────────────────

    def save_checkpoint(self, tag: str):
        """Save cooperative LoRA checkpoint. Only rank 0 saves."""
        if self.rank != 0:
            return

        ckpt_dir = os.path.join(self.args.output_dir, tag)
        coop_dir = os.path.join(ckpt_dir, "cooperative")
        os.makedirs(coop_dir, exist_ok=True)

        # Split into lora_v, lora_a, lora_comm, lora_router
        lora_v, lora_a, lora_comm, lora_router = {}, {}, {}, {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "routers." in name:
                lora_router[name] = param.data.cpu()
            elif any(s in name for s in ("gate_av", "gate_va", "W_av", "W_va")):
                lora_comm[name] = param.data.cpu()
            elif "lora_A_v" in name or "lora_B_v" in name:
                lora_v[name] = param.data.cpu()
            elif "lora_A_a" in name or "lora_B_a" in name:
                lora_a[name] = param.data.cpu()
            else:
                lora_a[name] = param.data.cpu()

        torch.save(lora_v, os.path.join(coop_dir, "lora_v.pt"))
        torch.save(lora_a, os.path.join(coop_dir, "lora_a.pt"))
        torch.save(lora_comm, os.path.join(coop_dir, "lora_comm.pt"))
        if lora_router:
            torch.save(lora_router, os.path.join(coop_dir, "lora_router.pt"))

        # Save coord head if exists
        if self.coord_head is not None:
            torch.save(
                self.coord_head.state_dict(),
                os.path.join(coop_dir, "coord_head.pt"),
            )

        # Save config
        config = {
            "lora_r": self.args.lora_r,
            "lora_alpha": self.args.lora_alpha,
            "target_modules": list(self.args.target_modules),
            "gate_type": self.args.gate_type,
            "gate_init": self.args.gate_init,
            "routing_mode": self.args.routing_mode,
            "balance_weight": self.args.balance_weight,
            "cooperative_comm": True,
            "num_agents": 2,
        }
        with open(os.path.join(coop_dir, "cooperative_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Save training state
        torch.save(
            {"global_step": self.global_step},
            os.path.join(ckpt_dir, "training_state.pt"),
        )
        logger.info(f"Saved checkpoint: {ckpt_dir}")

    # ── validation ────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, tag: str) -> Dict[str, float]:
        """Run greedy evaluation on val set (rank 0 only)."""
        if self.val_dataset is None or self.rank != 0:
            return {}

        logger.info(f"Running validation ({tag})...")
        all_rewards = []
        per_ep_results = []

        for ep_idx in range(min(len(self.val_dataset), 20)):  # Cap at 20 episodes
            episode = self.val_dataset[ep_idx]
            goal = episode["goal"]
            steps = episode["steps"]
            action_history: List[str] = []
            ep_rewards = []

            for si, step in enumerate(steps):
                screenshot = step["screenshot"]
                gt_action = step["action"]
                image_w = step.get("image_w", 1080)
                image_h = step.get("image_h", 2400)

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

                # Use GT action for history (matching training)
                action_str = json.dumps(gt_action, ensure_ascii=False)
                action_history.append(f"<action>{action_str}</action>")

            per_ep_results.append({
                "episode_id": episode["episode_id"],
                "goal": goal[:100],
                "num_steps": len(steps),
                "mean_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
                "rewards": ep_rewards,
            })

        metrics = {
            "val/mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "val/n_episodes": len(per_ep_results),
            "val/n_steps": len(all_rewards),
        }

        # Save results
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
        for r in per_ep_results[:3]:
            logger.info(
                f"  ep{r['episode_id']}: {r['goal'][:60]}... "
                f"reward={r['mean_reward']:.3f} steps={r['num_steps']}"
            )

        return metrics

    # ── main loop ─────────────────────────────────────────────────

    def train(self):
        args = self.args

        if self.rank == 0:
            logger.info("=" * 60)
            logger.info("v11 Trajectory GRPO Training (Cooperative LoRA)")
            logger.info(f"  K={args.num_samples}  gamma={args.gamma}  temp={args.temperature}")
            logger.info(f"  grad_accum={args.gradient_accumulation_steps}")
            logger.info(f"  epochs={args.num_epochs}  kl_coef={args.kl_coef}")
            logger.info(f"  lora_r={args.lora_r}  lora_lr={args.lora_lr}  comm_lr={args.comm_lr}")
            logger.info(f"  world_size={self.world_size}")
            logger.info(f"  episodes={len(self.train_dataset)}")
            logger.info(f"  reward weights: format={args.w_format} type={args.w_type} content={args.w_content}")
            logger.info(f"  aux_coord_weight={args.aux_coord_weight}  gate_init={args.gate_init}")
            logger.info(f"  dapo_threshold={args.dapo_threshold}")
            if self.resume_step > 0:
                logger.info(f"  RESUMING from step {self.resume_step}")
            logger.info("=" * 60)

        for epoch in range(args.num_epochs):
            self.sampler.set_epoch(epoch)

            epoch_metrics = defaultdict(list)
            batch_rollouts: List[Dict] = []
            skipped = 0
            dapo_filtered = 0

            t_epoch = time.time()

            # Number of episodes to skip for resume
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
                elif rollout is None:
                    # Could be DAPO filtered or error
                    skipped += 1

                # Fixed-interval train step
                if step_at_boundary:
                    metrics = self.train_step(batch_rollouts)

                    if batch_rollouts:
                        for k, v in metrics.items():
                            if isinstance(v, (int, float)):
                                epoch_metrics[k].append(v)

                        if self.rank == 0 and self.global_step % args.logging_steps == 0:
                            gate_str = ""
                            if "gate_av_mean" in metrics:
                                gate_str = (
                                    f"g_av={metrics['gate_av_mean']:.4f} "
                                    f"g_va={metrics['gate_va_mean']:.4f} "
                                )
                            coord_str = ""
                            if metrics.get("aux_coord", 0) > 0:
                                coord_str = f"aux={metrics['aux_coord']:.4f} "
                            router_str = ""
                            if "router_w" in metrics:
                                router_str = f"rw={metrics['router_w']:.3f} "
                            logger.info(
                                f"E{epoch} S{self.global_step} "
                                f"loss={metrics['pg_loss']:.4f} "
                                f"r={metrics['mean_reward']:.3f} "
                                f"kl={metrics['kl']:.4f} "
                                f"gnorm={metrics['grad_norm']:.2f} "
                                f"nz={metrics['nonzero_adv_frac']:.0%} "
                                f"adv={metrics['mean_abs_adv']:.3f} "
                                f"{gate_str}{coord_str}{router_str}"
                                f"t={time.time()-t_ep:.1f}s"
                            )

                        if (
                            self.rank == 0
                            and args.save_steps > 0
                            and self.global_step % args.save_steps == 0
                        ):
                            self.save_checkpoint(
                                f"epoch-{epoch}_step-{self.global_step}"
                            )

                    # Mid-epoch validation
                    if (
                        args.val_steps > 0
                        and self.global_step % args.val_steps == 0
                    ):
                        if self.rank == 0:
                            self.validate(
                                tag=f"epoch-{epoch}_step-{self.global_step}"
                            )
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
                for k in [
                    "pg_loss", "mean_reward", "kl", "aux_coord",
                    "nonzero_adv_frac", "gate_av_mean", "gate_va_mean",
                    "router_w", "balance_loss",
                ]:
                    vals = epoch_metrics.get(k, [0])
                    logger.info(f"  avg {k}: {np.mean(vals):.4f}")
                logger.info(f"{'='*60}")

                self.save_checkpoint(f"epoch-{epoch}")
                self.validate(tag=f"epoch-{epoch}_final")

            if self.world_size > 1:
                dist.barrier()

            # Reset resume after first epoch
            self.resume_step = 0

        if self.rank == 0:
            logger.info("Training complete!")


# ── CLI ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="v11 Trajectory GRPO Trainer (Cooperative LoRA)")

    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume_from", type=str, default="",
                        help="Path to checkpoint dir to resume from")

    # Data
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default="")
    parser.add_argument("--max_episodes", type=int, default=0)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=256)
    parser.add_argument("--lora_alpha", type=int, default=512)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # Cooperative communication and routing
    parser.add_argument("--gate_type", type=str, default="tanh")
    parser.add_argument("--gate_init", type=float, default=0.5)
    parser.add_argument("--routing_mode", type=str, default="learned",
                        choices=["hard", "merge", "learned"],
                        help="'hard': image→V/text→A; 'learned': per-layer router")
    parser.add_argument("--balance_weight", type=float, default=0.01,
                        help="Balance loss weight for learned router (prevents collapse)")

    # GRPO
    parser.add_argument("--num_samples", type=int, default=8,
                        help="K rollouts per step")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--kl_coef", type=float, default=0.001)
    parser.add_argument("--max_new_tokens", type=int, default=256)

    # Trajectory
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="Discount factor for step returns")
    parser.add_argument("--dapo_threshold", type=float, default=0.3,
                        help="Std threshold for DAPO filtering")

    # Reward weights
    parser.add_argument("--w_format", type=float, default=0.1)
    parser.add_argument("--w_type", type=float, default=0.2)
    parser.add_argument("--w_content", type=float, default=0.7)

    # Auxiliary loss (direct LoRA_V gradient signal)
    parser.add_argument("--aux_coord_weight", type=float, default=0.1,
                        help="Weight for auxiliary coord prediction loss on image tokens")

    # Optimizer
    parser.add_argument("--lora_lr", type=float, default=1e-5,
                        help="Learning rate for LoRA params")
    parser.add_argument("--comm_lr", type=float, default=1e-3,
                        help="Learning rate for communication params (gates, W)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Training
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--val_steps", type=int, default=25,
                        help="Run validation every N steps (0=only at epoch end)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    trainer = V11TrajectoryGRPOTrainer(args)
    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
