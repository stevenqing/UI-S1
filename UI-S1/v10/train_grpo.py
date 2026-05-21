#!/usr/bin/env python3
"""v10 Sequential Cooperative LoRA GRPO Trainer — DDP + PEFT.

Architecture:
  Pass 1: Grounder (base + LoRA_grounder) → grounding description
  Pass 2: Actor (base + LoRA_actor) → action (reads grounding as context)

Training (GRPO):
  For each sample, generate K grounder descriptions and K actor actions.
  Grounder reward: coord_correct (did actor produce correct coordinates?)
  Actor reward:    action_correct (did actor produce correct full action?)
  GRPO advantage normalization within the K-group.
  Clipped policy gradient update for both adapters.
  KL penalty against base model (disable_adapter_layers).

DDP synchronizes gradients across GPUs. Model fits per-GPU (~22GB with
r=128 on 95GB GH200 GPUs). No model sharding complexity.

Usage:
  srun --ntasks-per-node=1 bash -c '
    torchrun --nproc_per_node=4 --nnodes=$SLURM_NNODES \
      --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR \
      v10/train_grpo.py --model_path ... --train_data ... --output_dir ...
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
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
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
logger = logging.getLogger("v10_grpo")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

GROUNDER_SYSTEM = (
    "You are a GUI grounding agent. Given a screenshot and an instruction, "
    "determine the next action type and describe the target.\n\n"
    "Output format:\n"
    "<action_type>one of: click, type, open, swipe, long_press, wait, system_button, terminate</action_type>\n"
    "<target>description of the target (UI element location for click/long_press, "
    "app name for open, scroll direction for swipe, button name for system_button, "
    "reason for wait, or text to type)</target>"
)

ACTOR_SYSTEM = (
    "You are a GUI agent. Given a screenshot, instruction, and grounding "
    "analysis (action type + target description), perform the next action.\n"
    'Output format: <action>{"action": "...", ...}</action>'
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class V10Dataset(Dataset):
    """Parse cooperative-thought JSONL → (goal, history, gt_action, image_path)."""

    def __init__(self, jsonl_path: str, max_samples: int = 0):
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                parsed = self._parse(sample)
                if parsed is not None:
                    self.samples.append(parsed)

        if 0 < max_samples < len(self.samples):
            rng = np.random.RandomState(42)
            idx = rng.choice(len(self.samples), max_samples, replace=False)
            self.samples = [self.samples[i] for i in sorted(idx)]

        logger.info(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    _GOAL_RE = re.compile(
        r"## User Instruction\n(.+?)(?:\n\n## |\n## |\Z)", re.DOTALL
    )
    _HIST_RE = re.compile(
        r"## History of previous actions\n(.+?)(?:\n\n## |\n## |\Z)", re.DOTALL
    )
    _ACTION_TAG_RE = re.compile(
        r"<action>\s*(\{.*?\})\s*</action>", re.DOTALL
    )
    _ACTION_RAW_RE = re.compile(r'\{[^{}]*"action"[^{}]*\}')

    def _parse(self, sample):
        convs = sample.get("conversations", [])
        if len(convs) < 2:
            return None
        human_msg = convs[0]["value"]
        assistant_msg = convs[1]["value"]
        m = self._GOAL_RE.search(human_msg)
        goal = m.group(1).strip() if m else ""
        if not goal:
            return None
        m = self._HIST_RE.search(human_msg)
        history = m.group(1).strip() if m else ""
        gt_action = self._parse_action(assistant_msg)
        if gt_action is None:
            return None
        images = sample.get("images", [])
        image_path = images[0] if images else None
        if image_path is None or not os.path.exists(image_path):
            return None
        return {
            "goal": goal,
            "history": history,
            "gt_action": gt_action,
            "image_path": image_path,
        }

    def _parse_action(self, text: str) -> Optional[Dict]:
        m = self._ACTION_TAG_RE.search(text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        m = self._ACTION_RAW_RE.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def format_grounder_text(goal: str, history: str) -> str:
    parts = [f"Instruction: {goal}"]
    if history:
        parts.append(f"\nPrevious actions:\n{history}")
    parts.append("\nDetermine the action type and describe the target.")
    return "\n".join(parts)


def parse_grounder_output(text: str) -> Tuple[str, str]:
    """Parse structured grounder output into (action_type, target_description).

    Falls back to ("unknown", full_text) if parsing fails.
    """
    action_type = "unknown"
    target = text  # fallback: use full text as target

    m = re.search(r'<action_type>\s*(.*?)\s*</action_type>', text, re.DOTALL)
    if m:
        action_type = m.group(1).strip().lower()

    m = re.search(r'<target>\s*(.*?)\s*</target>', text, re.DOTALL)
    if m:
        target = m.group(1).strip()

    return action_type, target


def format_actor_text(goal: str, history: str, action_type: str, target: str) -> str:
    parts = [f"Instruction: {goal}"]
    if history:
        parts.append(f"\nPrevious actions:\n{history}")
    parts.append(f"\nGrounding action type: {action_type}")
    parts.append(f"Grounding target: {target}")
    parts.append("\nOutput the next action.")
    return "\n".join(parts)


def build_messages(system: str, image_path: str, user_text: str):
    user_text_clean = user_text.replace("<image>\n", "").replace("<image>", "")
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text_clean},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

from v10.reward import grounder_reward, actor_reward  # noqa: E402


# ---------------------------------------------------------------------------
# GRPO loss helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# DDP + PEFT GRPO Trainer
# ---------------------------------------------------------------------------


class V10FSDPGRPOTrainer:
    def __init__(self, args):
        self.args = args
        self.global_step = 0
        self.resume_step = 0

        self._setup_distributed()
        self._setup_model()
        self._load_resume_checkpoint()
        self._setup_data()
        self._setup_optimizer()

    # ── distributed ───────────────────────────────────────────────────

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
                timeout=datetime.timedelta(seconds=600),
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

    # ── model (PEFT + DDP) ───────────────────────────────────────────

    def _setup_model(self):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from peft import LoraConfig, get_peft_model

        args = self.args

        # Processor / tokenizer
        self.processor = AutoProcessor.from_pretrained(args.model_path)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.pad_id = self.processor.tokenizer.pad_token_id

        # 1. Load base model (bf16), freeze all params
        if self.rank == 0:
            logger.info(f"Loading base model: {args.model_path}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        for p in model.parameters():
            p.requires_grad = False

        # 2. LoRA config
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=list(args.target_modules),
            task_type="CAUSAL_LM",
        )

        # 3. Add two adapters
        model = get_peft_model(model, lora_cfg, adapter_name="grounder")
        model.add_adapter("actor", lora_cfg)

        # 4. Force ALL lora_ params requires_grad=True
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        # 5. Gradient checkpointing to save VRAM
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

        # Store PeftModel ref for adapter management
        self.peft_model = model

        # 6. Move to GPU (no DDP — manual gradient all-reduce avoids
        #    DDP incompatibilities with gradient checkpointing + PEFT
        #    + multiple backward passes per step)
        self.model = model.to(self.device)

        dist.barrier()

        if self.rank == 0:
            trainable = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            g_n = sum(
                p.numel()
                for n, p in model.named_parameters()
                if "grounder" in n and p.requires_grad
            )
            a_n = sum(
                p.numel()
                for n, p in model.named_parameters()
                if "actor" in n and p.requires_grad
            )
            logger.info(
                f"DDP model ready: {trainable:,} trainable "
                f"(grounder={g_n:,}, actor={a_n:,})"
            )

    # ── resume from checkpoint ─────────────────────────────────────

    def _load_resume_checkpoint(self):
        """Load adapter weights from a checkpoint directory if --resume_from is set."""
        ckpt_dir = getattr(self.args, "resume_from", "")
        if not ckpt_dir:
            return

        from peft import set_peft_model_state_dict
        import safetensors.torch

        if self.rank == 0:
            logger.info(f"Resuming from checkpoint: {ckpt_dir}")

        # Load grounder adapter weights
        g_path = os.path.join(ckpt_dir, "grounder", "grounder", "adapter_model.safetensors")
        if os.path.exists(g_path):
            g_weights = safetensors.torch.load_file(g_path)
            set_peft_model_state_dict(self.peft_model, g_weights, adapter_name="grounder")
            if self.rank == 0:
                logger.info(f"  Loaded grounder adapter: {len(g_weights)} tensors")

        # Load actor adapter weights
        a_path = os.path.join(ckpt_dir, "actor", "actor", "adapter_model.safetensors")
        if os.path.exists(a_path):
            a_weights = safetensors.torch.load_file(a_path)
            set_peft_model_state_dict(self.peft_model, a_weights, adapter_name="actor")
            if self.rank == 0:
                logger.info(f"  Loaded actor adapter: {len(a_weights)} tensors")

        # Load training state (global_step)
        state_path = os.path.join(ckpt_dir, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            self.global_step = state.get("global_step", 0)
            self.resume_step = self.global_step
            if self.rank == 0:
                logger.info(f"  Resumed at global_step={self.global_step}")

        # Re-enable requires_grad for all LoRA params (loading may reset them)
        for name, param in self.peft_model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        dist.barrier()

    # ── gradient checkpointing toggle ────────────────────────────────

    def _set_grad_checkpointing(self, enable: bool):
        """Toggle gradient checkpointing on the actual PreTrainedModel.

        PeftModel wraps: PeftModel -> LoraModel -> Qwen2_5_VLForConditionalGeneration.
        We call enable/disable on the actual PreTrainedModel (base_model.model)
        which has the proper _set_gradient_checkpointing implementation.
        """
        base = self.peft_model.base_model.model  # Qwen2_5_VLForConditionalGeneration
        if enable:
            base.gradient_checkpointing_enable()
        else:
            base.gradient_checkpointing_disable()

    # ── adapter switching ─────────────────────────────────────────────

    def _set_adapter_safe(self, name: str):
        """Set active adapter and restore requires_grad on all LoRA params."""
        self.peft_model.set_adapter(name)
        for n, p in self.peft_model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True

    # ── data ──────────────────────────────────────────────────────────

    def _setup_data(self):
        args = self.args
        self.train_dataset = V10Dataset(args.train_data, max_samples=args.max_samples)

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
            self.val_dataset = V10Dataset(args.val_data)
            if self.rank == 0:
                logger.info(f"Val dataset: {len(self.val_dataset)} samples")

    # ── optimizer ─────────────────────────────────────────────────────

    def _setup_optimizer(self):
        args = self.args
        # Separate param groups for grounder and actor
        grounder_params, actor_params = [], []
        for name, p in self.peft_model.named_parameters():
            if not p.requires_grad:
                continue
            if "grounder" in name:
                grounder_params.append(p)
            elif "actor" in name:
                actor_params.append(p)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": grounder_params, "lr": args.grounder_lr},
                {"params": actor_params, "lr": args.actor_lr},
            ],
            weight_decay=args.weight_decay,
        )
        if self.rank == 0:
            logger.info(
                f"Optimizer: grounder_lr={args.grounder_lr} actor_lr={args.actor_lr} "
                f"grounder_params={sum(p.numel() for p in grounder_params):,} "
                f"actor_params={sum(p.numel() for p in actor_params):,}"
            )

    # ── tokenization helpers ──────────────────────────────────────────

    def _tokenize_for_generation(
        self, system: str, image_path: str, user_text: str, image: Image.Image
    ) -> dict:
        msgs = build_messages(system, image_path, user_text)
        text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=False
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    # ── generation ────────────────────────────────────────────────────

    @torch.no_grad()
    def _generate_batch(self, inputs: dict, K: int, max_new_tokens: int):
        """Generate K samples. Uses PeftModel directly (no DDP wrapper needed)."""
        prompt_len = inputs["input_ids"].shape[1]

        gen_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                rep = [K] + [1] * (v.dim() - 1)
                gen_inputs[k] = v.repeat(*rep)
            else:
                gen_inputs[k] = v

        # Disable gradient checkpointing for generation so KV cache works
        self._set_grad_checkpointing(False)
        self.model.eval()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output_ids = self.model.generate(
                **gen_inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                do_sample=True,
            )
        self.model.train()
        self._set_grad_checkpointing(True)
        return output_ids, prompt_len

    # ── log prob computation ──────────────────────────────────────────

    def _compute_token_log_probs(
        self,
        full_ids: torch.Tensor,
        prompt_len: int,
        inputs_for_fwd: dict,
        with_grad: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Compute per-token log probs for the response portion."""
        ids = full_ids.unsqueeze(0)
        attn = torch.ones_like(ids)

        fwd_kwargs = {"input_ids": ids, "attention_mask": attn}
        for k in ("pixel_values", "image_grid_thw"):
            if k in inputs_for_fwd:
                fwd_kwargs[k] = inputs_for_fwd[k]

        ctx = torch.enable_grad() if with_grad else torch.no_grad()
        with ctx:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model(**fwd_kwargs)

            logits = outputs.logits
            resp_logits = logits[:, prompt_len - 1 : -1, :]
            resp_labels = ids[:, prompt_len:]
            log_p = torch.nn.functional.log_softmax(resp_logits, dim=-1)
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
        """Compute reference (base model) log probs by disabling adapters."""
        self.peft_model.disable_adapter_layers()
        try:
            tok_lp, mask, _ = self._compute_token_log_probs(
                full_ids, prompt_len, inputs_for_fwd, with_grad=False
            )
        finally:
            self.peft_model.enable_adapter_layers()
            for n, p in self.peft_model.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True
        return tok_lp, mask

    # ── generation + rewards for one sample ───────────────────────────

    @torch.no_grad()
    def generate_rollouts(self, sample: Dict) -> Optional[Dict]:
        """Generate K grounder→actor rollouts for one training sample."""
        t0 = time.time()
        args = self.args
        K = args.num_samples
        goal = sample["goal"]
        history = sample["history"]
        image_path = sample["image_path"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Rank {self.rank}: Cannot open image {image_path}: {e}")
            return None
        image_w, image_h = image.size

        # ── Grounder generation ──
        self._set_adapter_safe("grounder")
        g_user = format_grounder_text(goal, history)
        g_inputs = self._tokenize_for_generation(
            GROUNDER_SYSTEM, image_path, g_user, image
        )
        g_output_ids, g_prompt_len = self._generate_batch(
            g_inputs, K, args.max_grounder_tokens
        )
        if self.rank == 0:
            logger.info(f"  Grounder gen: {time.time()-t0:.1f}s prompt_len={g_prompt_len}")

        grounder_texts = []
        for k in range(K):
            resp_ids = g_output_ids[k, g_prompt_len:]
            text = self.processor.tokenizer.decode(
                resp_ids, skip_special_tokens=True
            )
            grounder_texts.append(text)

        g_fwd_inputs = {}
        for key in ("pixel_values", "image_grid_thw"):
            if key in g_inputs:
                g_fwd_inputs[key] = g_inputs[key]

        # ── Actor generation ──
        self._set_adapter_safe("actor")
        actor_texts = []
        actor_full_ids = []
        actor_prompt_lens = []
        actor_fwd_inputs_list = []

        grounder_parsed = [parse_grounder_output(t) for t in grounder_texts]

        for k in range(K):
            action_type, target = grounder_parsed[k]
            a_user = format_actor_text(goal, history, action_type, target)
            a_inputs = self._tokenize_for_generation(
                ACTOR_SYSTEM, image_path, a_user, image
            )
            a_output_ids, a_prompt_len = self._generate_batch(
                a_inputs, 1, args.max_actor_tokens
            )
            resp_ids = a_output_ids[0, a_prompt_len:]
            text = self.processor.tokenizer.decode(
                resp_ids, skip_special_tokens=True
            )
            actor_texts.append(text)
            actor_full_ids.append(a_output_ids[0])
            actor_prompt_lens.append(a_prompt_len)

            a_fwd = {}
            for key in ("pixel_values", "image_grid_thw"):
                if key in a_inputs:
                    a_fwd[key] = a_inputs[key]
            actor_fwd_inputs_list.append(a_fwd)
            del a_inputs
        torch.cuda.empty_cache()
        if self.rank == 0:
            logger.info(f"  Actor gen: {time.time()-t0:.1f}s total")

        # ── Rewards ──
        g_rewards, a_rewards = [], []
        for k in range(K):
            g_r = grounder_reward(
                actor_texts[k], sample["gt_action"], image_w, image_h,
                grounder_text=grounder_texts[k],
                w_format=args.grounder_format_weight,
                w_downstream=1.0 - args.grounder_format_weight,
            )
            a_r = actor_reward(
                actor_texts[k], sample["gt_action"], image_w, image_h
            )
            g_rewards.append(g_r)
            a_rewards.append(a_r)

        # ── GRPO advantages ──
        eps = 1e-6
        g_t = torch.tensor(g_rewards, dtype=torch.float32)
        a_t = torch.tensor(a_rewards, dtype=torch.float32)
        g_adv = (
            (g_t - g_t.mean()) / (g_t.std() + eps)
            if g_t.std() > eps
            else torch.zeros_like(g_t)
        )
        a_adv = (
            (a_t - a_t.mean()) / (a_t.std() + eps)
            if a_t.std() > eps
            else torch.zeros_like(a_t)
        )

        # ── Old log probs (per-token) ──
        self._set_adapter_safe("grounder")
        g_old_tok_lps = []
        g_masks = []
        for k in range(K):
            tok_lp, mask, _ = self._compute_token_log_probs(
                g_output_ids[k], g_prompt_len, g_fwd_inputs, with_grad=False
            )
            g_old_tok_lps.append(tok_lp.detach())
            g_masks.append(mask.detach())

        # Grounder ref log probs (for KL)
        g_ref_tok_lps = []
        if args.kl_coef > 0:
            for k in range(K):
                ref_lp, _ = self._compute_ref_log_probs(
                    g_output_ids[k], g_prompt_len, g_fwd_inputs
                )
                g_ref_tok_lps.append(ref_lp.detach())
            self._set_adapter_safe("grounder")

        # Actor
        self._set_adapter_safe("actor")
        a_old_tok_lps = []
        a_masks = []
        for k in range(K):
            tok_lp, mask, _ = self._compute_token_log_probs(
                actor_full_ids[k], actor_prompt_lens[k],
                actor_fwd_inputs_list[k], with_grad=False
            )
            a_old_tok_lps.append(tok_lp.detach())
            a_masks.append(mask.detach())

        a_ref_tok_lps = []
        if args.kl_coef > 0:
            for k in range(K):
                ref_lp, _ = self._compute_ref_log_probs(
                    actor_full_ids[k], actor_prompt_lens[k],
                    actor_fwd_inputs_list[k]
                )
                a_ref_tok_lps.append(ref_lp.detach())
            self._set_adapter_safe("actor")

        if self.rank == 0:
            logger.info(f"  Rollout total: {time.time()-t0:.1f}s  g_r={g_rewards} a_r={a_rewards}")

        return {
            "grounder_full_ids": g_output_ids,
            "grounder_prompt_len": g_prompt_len,
            "grounder_fwd_inputs": g_fwd_inputs,
            "grounder_advantages": g_adv,
            "grounder_old_tok_lps": g_old_tok_lps,
            "grounder_masks": g_masks,
            "grounder_ref_tok_lps": g_ref_tok_lps,
            "grounder_rewards": g_rewards,
            "actor_full_ids": actor_full_ids,
            "actor_prompt_lens": actor_prompt_lens,
            "actor_fwd_inputs": actor_fwd_inputs_list,
            "actor_advantages": a_adv,
            "actor_old_tok_lps": a_old_tok_lps,
            "actor_masks": a_masks,
            "actor_ref_tok_lps": a_ref_tok_lps,
            "actor_rewards": a_rewards,
        }

    # ── policy gradient update ────────────────────────────────────────

    def train_step(self, batch_rollouts: List[Dict]) -> Dict[str, float]:
        """One GRPO update on accumulated rollouts.

        Loss is normalized by (K * grad_accum_steps) so that gradient
        magnitude is independent of group size and accumulation count.
        """
        args = self.args
        K = args.num_samples
        # Normalization: each backward contributes 1/(K*accum) of the total
        loss_scale = K * args.gradient_accumulation_steps

        self.optimizer.zero_grad()
        total_g_loss = 0.0
        total_a_loss = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0
        n_g_seqs = 0
        n_a_seqs = 0
        # Track advantage stats for monitoring
        g_advs_abs = []
        a_advs_abs = []
        n_g_zero_adv = 0
        n_a_zero_adv = 0
        n_g_total = 0
        n_a_total = 0

        # ── Grounder backward ──
        self._set_adapter_safe("grounder")
        for data in batch_rollouts:
            for k in range(K):
                n_g_total += 1
                adv = data["grounder_advantages"][k].item()
                if abs(adv) < 1e-8:
                    n_g_zero_adv += 1
                    continue

                g_advs_abs.append(abs(adv))

                tok_lp, mask, n_tok = self._compute_token_log_probs(
                    data["grounder_full_ids"][k],
                    data["grounder_prompt_len"],
                    data["grounder_fwd_inputs"],
                    with_grad=True,
                )

                old_tok_lp = data["grounder_old_tok_lps"][k]
                adv_expanded = torch.full_like(mask, adv)

                pg_loss, clip_frac, approx_kl = compute_policy_loss(
                    old_tok_lp, tok_lp, adv_expanded, mask, args.clip_range
                )

                kl_loss = torch.tensor(0.0, device=self.device)
                if args.kl_coef > 0 and data["grounder_ref_tok_lps"]:
                    ref_lp = data["grounder_ref_tok_lps"][k]
                    kl_loss = compute_kl_penalty(tok_lp, ref_lp, mask)

                loss = (pg_loss + args.kl_coef * kl_loss) / loss_scale
                loss.backward()

                total_g_loss += pg_loss.item()
                total_kl += kl_loss.item()
                total_clip_frac += clip_frac.item()
                n_g_seqs += 1

                del tok_lp, loss, pg_loss, kl_loss

        # ── Actor backward ──
        self._set_adapter_safe("actor")
        for data in batch_rollouts:
            for k in range(K):
                n_a_total += 1
                adv = data["actor_advantages"][k].item()
                if abs(adv) < 1e-8:
                    n_a_zero_adv += 1
                    continue

                a_advs_abs.append(abs(adv))

                tok_lp, mask, n_tok = self._compute_token_log_probs(
                    data["actor_full_ids"][k],
                    data["actor_prompt_lens"][k],
                    data["actor_fwd_inputs"][k],
                    with_grad=True,
                )

                old_tok_lp = data["actor_old_tok_lps"][k]
                adv_expanded = torch.full_like(mask, adv)

                pg_loss, clip_frac, approx_kl = compute_policy_loss(
                    old_tok_lp, tok_lp, adv_expanded, mask, args.clip_range
                )

                kl_loss = torch.tensor(0.0, device=self.device)
                if args.kl_coef > 0 and data["actor_ref_tok_lps"]:
                    ref_lp = data["actor_ref_tok_lps"][k]
                    kl_loss = compute_kl_penalty(tok_lp, ref_lp, mask)

                loss = (pg_loss + args.kl_coef * kl_loss) / loss_scale
                loss.backward()

                total_a_loss += pg_loss.item()
                total_kl += kl_loss.item()
                total_clip_frac += clip_frac.item()
                n_a_seqs += 1

                del tok_lp, loss, pg_loss, kl_loss

        # ── All-reduce gradients across ranks ──
        # ALL trainable params must participate (even with None grad) so that
        # all ranks call all_reduce the same number of times.
        if self.world_size > 1:
            for p in self.peft_model.parameters():
                if p.requires_grad:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p.data)
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        # ── Clip & step ──
        trainable = [p for p in self.peft_model.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)

        self.optimizer.step()
        self.global_step += 1

        # ── Metrics ──
        all_g = [r for d in batch_rollouts for r in d["grounder_rewards"]]
        all_a = [r for d in batch_rollouts for r in d["actor_rewards"]]
        n_total = max(n_g_seqs + n_a_seqs, 1)

        return {
            "grounder_loss": total_g_loss / max(n_g_seqs, 1),
            "actor_loss": total_a_loss / max(n_a_seqs, 1),
            "grounder_reward": float(np.mean(all_g)) if all_g else 0.0,
            "actor_reward": float(np.mean(all_a)) if all_a else 0.0,
            "kl": total_kl / n_total,
            "clip_frac": total_clip_frac / n_total,
            "grad_norm": grad_norm.item()
            if isinstance(grad_norm, torch.Tensor)
            else float(grad_norm),
            "g_nonzero_frac": (n_g_total - n_g_zero_adv) / max(n_g_total, 1),
            "a_nonzero_frac": (n_a_total - n_a_zero_adv) / max(n_a_total, 1),
            "g_mean_abs_adv": float(np.mean(g_advs_abs)) if g_advs_abs else 0.0,
            "a_mean_abs_adv": float(np.mean(a_advs_abs)) if a_advs_abs else 0.0,
        }

    # ── checkpoint ────────────────────────────────────────────────────

    def save_checkpoint(self, tag: str):
        """Save LoRA checkpoints. Only rank 0 saves."""
        if self.rank != 0:
            return
        ckpt_dir = os.path.join(self.args.output_dir, tag)
        os.makedirs(ckpt_dir, exist_ok=True)

        for adapter_name in ("grounder", "actor"):
            adapter_dir = os.path.join(ckpt_dir, adapter_name)
            self.peft_model.save_pretrained(
                adapter_dir, selected_adapters=[adapter_name]
            )

        torch.save(
            {"global_step": self.global_step},
            os.path.join(ckpt_dir, "training_state.pt"),
        )
        logger.info(f"Saved checkpoint: {ckpt_dir}")

    # ── validation ─────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, tag: str) -> Dict[str, float]:
        """Run greedy grounder→actor on the val set (rank 0 only).

        Saves per-sample results to {output_dir}/val_results/{tag}.jsonl
        for offline analysis.
        """
        if self.val_dataset is None or self.rank != 0:
            return {}

        logger.info(f"Running validation ({tag})...")
        g_rewards, a_rewards = [], []
        per_sample_results = []

        for idx in range(len(self.val_dataset)):
            sample = self.val_dataset[idx]
            goal = sample["goal"]
            history = sample["history"]
            image_path = sample["image_path"]

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                continue
            image_w, image_h = image.size

            # Grounder — greedy
            self._set_adapter_safe("grounder")
            g_user = format_grounder_text(goal, history)
            g_inputs = self._tokenize_for_generation(
                GROUNDER_SYSTEM, image_path, g_user, image
            )
            self._set_grad_checkpointing(False)
            self.model.eval()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                g_out = self.model.generate(
                    **g_inputs,
                    max_new_tokens=self.args.max_grounder_tokens,
                    do_sample=False,
                )
            g_prompt_len = g_inputs["input_ids"].shape[1]
            g_text = self.processor.tokenizer.decode(
                g_out[0, g_prompt_len:], skip_special_tokens=True
            )

            # Actor — greedy
            self._set_adapter_safe("actor")
            action_type, target = parse_grounder_output(g_text)
            a_user = format_actor_text(goal, history, action_type, target)
            a_inputs = self._tokenize_for_generation(
                ACTOR_SYSTEM, image_path, a_user, image
            )
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                a_out = self.model.generate(
                    **a_inputs,
                    max_new_tokens=self.args.max_actor_tokens,
                    do_sample=False,
                )
            self.model.train()
            self._set_grad_checkpointing(True)
            a_prompt_len = a_inputs["input_ids"].shape[1]
            a_text = self.processor.tokenizer.decode(
                a_out[0, a_prompt_len:], skip_special_tokens=True
            )

            g_r = grounder_reward(a_text, sample["gt_action"], image_w, image_h)
            a_r = actor_reward(a_text, sample["gt_action"], image_w, image_h)
            g_rewards.append(g_r)
            a_rewards.append(a_r)

            per_sample_results.append({
                "idx": idx,
                "goal": goal,
                "history": history[:200],
                "image_path": image_path,
                "gt_action": sample["gt_action"],
                "grounder_text": g_text,
                "actor_text": a_text,
                "grounder_reward": g_r,
                "actor_reward": a_r,
            })

        metrics = {
            "val/grounder_reward": float(np.mean(g_rewards)) if g_rewards else 0.0,
            "val/actor_reward": float(np.mean(a_rewards)) if a_rewards else 0.0,
            "val/actor_exact": float(np.mean([r == 1.0 for r in a_rewards]))
            if a_rewards
            else 0.0,
            "val/n_samples": len(g_rewards),
        }

        # Save per-sample results for analysis
        val_dir = os.path.join(self.args.output_dir, "val_results")
        os.makedirs(val_dir, exist_ok=True)
        result_path = os.path.join(val_dir, f"{tag}.jsonl")
        with open(result_path, "w") as f:
            # Write summary as first line
            f.write(json.dumps({"_summary": metrics, "_tag": tag, "_step": self.global_step}) + "\n")
            for r in per_sample_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        logger.info(f"Validation {tag}:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        # Show a few examples
        for r in per_sample_results[:3]:
            logger.info(
                f"  [{r['idx']}] goal={r['goal'][:60]}...\n"
                f"       grounding={r['grounder_text'][:120]}\n"
                f"       actor={r['actor_text'][:120]}\n"
                f"       g_r={r['grounder_reward']:.3f} a_r={r['actor_reward']:.3f}"
            )
        logger.info(f"  Results saved to {result_path}")

        return metrics

    # ── main loop ─────────────────────────────────────────────────────

    def train(self):
        args = self.args

        if self.rank == 0:
            logger.info("=" * 60)
            logger.info("v10 DDP+PEFT GRPO Training")
            logger.info(f"  K={args.num_samples}  temp={args.temperature}")
            logger.info(f"  grad_accum={args.gradient_accumulation_steps}")
            logger.info(f"  epochs={args.num_epochs}  kl_coef={args.kl_coef}")
            logger.info(f"  lora_r={args.lora_r}  g_lr={args.grounder_lr}  a_lr={args.actor_lr}")
            logger.info(f"  world_size={self.world_size}")
            logger.info(f"  dataset size={len(self.train_dataset)}")
            if self.resume_step > 0:
                logger.info(f"  RESUMING from step {self.resume_step}")
            logger.info("=" * 60)

        for epoch in range(args.num_epochs):
            self.sampler.set_epoch(epoch)

            epoch_metrics = defaultdict(list)
            batch_rollouts: List[Dict] = []
            skipped = 0

            t_epoch = time.time()

            # Number of samples to skip for resume
            skip_samples = self.resume_step * args.gradient_accumulation_steps
            if self.rank == 0 and skip_samples > 0:
                logger.info(f"  Skipping first {skip_samples} samples (resume_step={self.resume_step})")

            for sample_idx, sample in enumerate(self.train_loader):
                # Skip samples already processed in the resumed checkpoint
                if sample_idx < skip_samples:
                    continue

                step_at_boundary = (sample_idx + 1) % args.gradient_accumulation_steps == 0

                t_sample = time.time()

                try:
                    rollout = self.generate_rollouts(sample)
                except Exception as e:
                    if self.rank == 0:
                        logger.warning(
                            f"Sample {sample_idx} failed: {e}\n"
                            f"{traceback.format_exc()}"
                        )
                    rollout = None

                if rollout is not None:
                    batch_rollouts.append(rollout)
                else:
                    skipped += 1

                # Fixed-interval train step: all ranks enter at the same
                # sample_idx (DistributedSampler gives equal counts).
                # Every rank MUST call train_step at every boundary so the
                # dist.all_reduce in train_step matches across all ranks.
                if step_at_boundary:
                    # train_step handles empty batch_rollouts gracefully
                    # (zero grad → all_reduce → optimizer step = no-op)
                    metrics = self.train_step(batch_rollouts)

                    if batch_rollouts:
                        for k, v in metrics.items():
                            epoch_metrics[k].append(v)

                        if self.rank == 0 and self.global_step % args.logging_steps == 0:
                            logger.info(
                                f"E{epoch} S{self.global_step} "
                                f"g_loss={metrics['grounder_loss']:.4f} "
                                f"a_loss={metrics['actor_loss']:.4f} "
                                f"g_r={metrics['grounder_reward']:.3f} "
                                f"a_r={metrics['actor_reward']:.3f} "
                                f"kl={metrics['kl']:.4f} "
                                f"gnorm={metrics['grad_norm']:.2f} "
                                f"g_nz={metrics['g_nonzero_frac']:.0%} "
                                f"a_nz={metrics['a_nonzero_frac']:.0%} "
                                f"g_adv={metrics['g_mean_abs_adv']:.3f} "
                                f"a_adv={metrics['a_mean_abs_adv']:.3f} "
                                f"t={time.time()-t_sample:.1f}s"
                            )

                        if (
                            self.rank == 0
                            and args.save_steps > 0
                            and self.global_step % args.save_steps == 0
                        ):
                            self.save_checkpoint(
                                f"epoch-{epoch}_step-{self.global_step}"
                            )

                    # Mid-epoch validation: all ranks must reach the barrier
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
                    f"Epoch {epoch} done in {dur/60:.1f}min  skipped={skipped}"
                )
                for k in [
                    "grounder_loss",
                    "actor_loss",
                    "grounder_reward",
                    "actor_reward",
                    "kl",
                ]:
                    vals = epoch_metrics.get(k, [0])
                    logger.info(f"  avg {k}: {np.mean(vals):.4f}")
                logger.info(f"{'='*60}")

                self.save_checkpoint(f"epoch-{epoch}")
                self.validate(tag=f"epoch-{epoch}_final")

            if self.world_size > 1:
                dist.barrier()

        if self.rank == 0:
            logger.info("Training complete!")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="v10 DDP+PEFT GRPO Trainer")

    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resume_from", type=str, default="",
                        help="Path to checkpoint dir to resume from")

    # Data
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default="")
    parser.add_argument("--max_samples", type=int, default=0)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # GRPO
    parser.add_argument(
        "--num_samples", type=int, default=4, help="K rollouts per prompt"
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--kl_coef", type=float, default=0.001)
    parser.add_argument("--max_grounder_tokens", type=int, default=256)
    parser.add_argument("--max_actor_tokens", type=int, default=256)
    parser.add_argument(
        "--grounder_format_weight", type=float, default=0.5,
        help="Weight for grounder format reward (vs downstream actor reward). "
             "0.5 = 50%% format + 50%% downstream."
    )

    # Optimizer
    parser.add_argument("--grounder_lr", type=float, default=1e-5)
    parser.add_argument("--actor_lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Training
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument(
        "--val_steps", type=int, default=50,
        help="Run validation every N steps (0=only at epoch end)",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    trainer = V10FSDPGRPOTrainer(args)
    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
