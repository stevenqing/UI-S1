#!/usr/bin/env python3
"""Offline GRPO trainer for GUI navigation (Qwen2.5-VL + LoRA).

Loads pre-generated candidate groups (from prepare_grpo_data.py),
computes group-relative advantages, and trains with policy gradient.

Algorithm:
  For each step group with N candidates:
    1. rewards = [r_1, ..., r_N]
    2. advantages = (rewards - mean) / (std + eps)    [GRPO normalization]
    3. For each candidate i:
       log_pi = sum(log P(token_j | context)) / n_tokens  [mean log-prob]
    4. loss = -sum(advantage_i * log_pi_i) / N
    5. KL penalty: beta * (log_pi - log_pi_ref)

Usage:
  torchrun --nproc_per_node=4 v19_step_aware/train_grpo.py \
    --model_path .../checkpoint-272 \
    --grpo_data v19_step_aware/data/grpo_candidates_n5.jsonl \
    --output_dir v19_step_aware/output/grpo_r1 \
    --lora_rank 64 --lr 1e-6 --epochs 2 --beta_kl 0.05
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image

os.environ.setdefault("NCCL_SOCKET_IFNAME", "hsn0")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "hsn0")
os.environ.setdefault("NCCL_NET", "Socket")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_LEVEL", "LOC")

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("v19_grpo")


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════

PREDICT_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

Before acting, determine:
1. What type of interaction is needed next? (click to select/navigate, type to enter text, drag to move, scroll to view more)
2. Which UI element should receive this interaction?
3. Output the action with precise coordinates.

After your reasoning, output your action within <tool_call></tool_call> tag:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time."""

SUPPORTED_ACTIONS = """click(coordinate=[x, y], button='left', double=False, pressed=None)
type(text='...', clear_current_text=False)
swipe(coordinate=[x1, y1], direction='up'|'down'|'left'|'right', dist='short'|'medium'|'long')"""


def _parse_stored(obj):
    """Parse a stored action that may be dict or string."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        import ast as _ast
        try:
            return _ast.literal_eval(obj)
        except Exception:
            pass
        try:
            return json.loads(obj.replace("'", '"'))
        except Exception:
            pass
    return {}


def _gaussian_coord_reward(pred_action, gt_action, sigma, img_w=1040, img_h=736):
    """Compute Gaussian distance-based reward for click actions."""
    gt_coord = gt_action.get("coordinate")
    pred_coord = pred_action.get("coordinate")
    if gt_coord is None or pred_coord is None:
        return 0.0
    try:
        dx = (float(pred_coord[0]) - float(gt_coord[0])) / img_w
        dy = (float(pred_coord[1]) - float(gt_coord[1])) / img_h
        dist = (dx ** 2 + dy ** 2) ** 0.5
    except (TypeError, IndexError, ValueError):
        return 0.0
    return float(np.exp(-(dist / sigma) ** 2))


class GRPODataset(Dataset):
    """Dataset of (prompt, response, advantage) tuples from pre-generated groups.

    Modes:
      - Standard GRPO: group-relative advantage normalization on rewards.
      - type_filtered: only same-type candidates per group.
      - gaussian_sigma: recompute click rewards with Gaussian distance decay.
      - rft_mode: Rejection Sampling FT — keep only correct candidates,
        set all advantages to 1.0. No negative signal, just SFT on good samples.
    """

    def __init__(self, data_path: str, min_group_std: float = 0.01,
                 type_filtered: bool = False, gaussian_sigma: float = 0.0,
                 rft_mode: bool = False, rft_threshold: float = 0.8):
        self.samples = []
        self.stats = {
            "total_groups": 0, "filtered_no_var": 0,
            "filtered_too_few": 0, "used_groups": 0, "total_samples": 0,
            "rft_accepted": 0, "rft_rejected": 0,
        }

        with open(data_path) as f:
            for line in f:
                group = json.loads(line)
                self.stats["total_groups"] += 1

                candidates = group["candidates"]
                gt_action = _parse_stored(group.get("gt_action", {}))
                gt_type = (gt_action.get("function")
                           or gt_action.get("action") or None)

                # Type-filtered: keep only candidates matching GT type
                if type_filtered and gt_type:
                    candidates = [c for c in candidates
                                  if c.get("pred_type") == gt_type]
                    if len(candidates) < 2 and not rft_mode:
                        self.stats["filtered_too_few"] += 1
                        continue

                # Compute rewards
                if gaussian_sigma > 0 and gt_type == "click":
                    # Recompute click rewards with Gaussian distance
                    rewards = []
                    for c in candidates:
                        pa = _parse_stored(c.get("pred_action", {}))
                        r = _gaussian_coord_reward(pa, gt_action, gaussian_sigma)
                        rewards.append(r)
                    rewards = np.array(rewards)
                else:
                    # Use stored content_reward (strips format+type constant)
                    rewards = np.array([float(c.get("content_reward", 0))
                                        for c in candidates])

                if rft_mode:
                    # RFT: keep only correct candidates, advantage = 1.0
                    group_accepted = 0
                    for i, cand in enumerate(candidates):
                        if rewards[i] >= rft_threshold:
                            self.stats["rft_accepted"] += 1
                            group_accepted += 1
                            self.samples.append({
                                "screenshot": group["screenshot"],
                                "goal": group["goal"],
                                "history": group["history"],
                                "response": cand["response"],
                                "advantage": 1.0,
                                "reward": float(rewards[i]),
                            })
                        else:
                            self.stats["rft_rejected"] += 1
                    if group_accepted > 0:
                        self.stats["used_groups"] += 1
                    continue

                # GRPO mode: need variance for learning signal
                if len(candidates) < 2:
                    self.stats["filtered_too_few"] += 1
                    continue

                if rewards.std() < min_group_std:
                    self.stats["filtered_no_var"] += 1
                    continue

                # GRPO: group-relative advantage normalization
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

                self.stats["used_groups"] += 1
                for i, cand in enumerate(candidates):
                    self.samples.append({
                        "screenshot": group["screenshot"],
                        "goal": group["goal"],
                        "history": group["history"],
                        "response": cand["response"],
                        "advantage": float(advantages[i]),
                        "reward": float(rewards[i]),
                    })

        self.stats["total_samples"] = len(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ═══════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════

class GRPOTrainer:
    """Offline GRPO trainer for Qwen2.5-VL with LoRA."""

    def __init__(self, args):
        self.args = args
        self._setup_distributed()
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()

    # -- Setup --------------------------------------------------------

    def _setup_distributed(self):
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        elif "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
            dist.init_process_group("nccl")
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.rank = 0
            self.world_size = 1
            self.device = torch.device("cuda:0")

        torch.cuda.set_device(self.device)
        if self.rank == 0:
            logger.info(f"Distributed: rank={self.rank}, world={self.world_size}")

    def _setup_model(self):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from peft import LoraConfig, get_peft_model

        if self.rank == 0:
            logger.info(f"Loading model from {self.args.model_path}")

        self.processor = AutoProcessor.from_pretrained(
            self.args.model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).to(self.device)

        # LoRA
        lora_config = LoraConfig(
            r=self.args.lora_rank,
            lora_alpha=self.args.lora_rank * 2,
            target_modules="all-linear",
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
            modules_to_save=None,
        )
        self.model = get_peft_model(self.model, lora_config)
        if self.rank == 0:
            self.model.print_trainable_parameters()

        # Freeze vision encoder
        for name, param in self.model.named_parameters():
            if "visual" in name or "vision" in name:
                param.requires_grad = False

        # Snapshot reference weights (LoRA only, for KL)
        self._ref_lora_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._ref_lora_state[name] = param.data.clone()

        self.model.train()

        # Gradient checkpointing
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})

        # DDP
        if self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.device],
                find_unused_parameters=True)
            self._raw_model = self.model.module
        else:
            self._raw_model = self.model

    def _setup_data(self):
        dataset = GRPODataset(
            self.args.grpo_data, self.args.min_group_std,
            type_filtered=self.args.type_filtered,
            gaussian_sigma=self.args.gaussian_sigma,
            rft_mode=self.args.rft_mode,
            rft_threshold=self.args.rft_threshold,
        )
        if self.rank == 0:
            logger.info(f"Dataset: {dataset.stats}")

        if self.world_size > 1:
            sampler = DistributedSampler(dataset, shuffle=True)
        else:
            sampler = None

        self.dataset = dataset
        self.sampler = sampler
        self.dataloader = DataLoader(
            dataset, batch_size=1, sampler=sampler,
            shuffle=(sampler is None), num_workers=0, collate_fn=lambda x: x[0])

    def _setup_optimizer(self):
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=self.args.lr, weight_decay=self.args.weight_decay)

        total_steps = len(self.dataloader) * self.args.epochs // self.args.grad_accum
        warmup_steps = int(total_steps * 0.05)

        from transformers import get_cosine_schedule_with_warmup
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps)

    # -- Core: log-prob computation -----------------------------------

    def _build_messages(self, sample: Dict) -> list:
        """Build chat messages from a sample."""
        prompt_text = PREDICT_PROMPT.format(
            instruction=sample["goal"],
            history=sample["history"],
            actions=SUPPORTED_ACTIONS,
        )
        return [
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{sample['screenshot']}"},
                {"type": "text", "text": prompt_text},
            ]},
            {"role": "assistant", "content": sample["response"]},
        ]

    def _tokenize_sample(self, sample: Dict) -> Dict:
        """Tokenize a sample into input_ids with prompt/response split."""
        messages = self._build_messages(sample)

        # Full text (prompt + response)
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)

        # Prompt only (to find prompt length)
        prompt_messages = [messages[0]]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True)

        # Load image
        image = Image.open(sample["screenshot"]).convert("RGB")

        # Tokenize full
        full_inputs = self.processor(
            text=[full_text], images=[image],
            return_tensors="pt", padding=False)

        # Tokenize prompt only (for length)
        prompt_inputs = self.processor(
            text=[prompt_text], images=[image],
            return_tensors="pt", padding=False)

        prompt_len = prompt_inputs["input_ids"].shape[1]

        # Action-loss-only: find where <tool_call> starts in response
        action_offset = 0
        if self.args.action_loss_only:
            response = sample["response"]
            tc_pos = response.find("<tool_call>")
            if tc_pos > 0:
                # Tokenize reasoning portion to get token count
                reasoning_part = response[:tc_pos]
                reasoning_ids = self.tokenizer.encode(
                    reasoning_part, add_special_tokens=False)
                action_offset = len(reasoning_ids)

        return {
            "input_ids": full_inputs["input_ids"].squeeze(0),
            "attention_mask": full_inputs["attention_mask"].squeeze(0),
            "pixel_values": full_inputs.get("pixel_values"),
            "image_grid_thw": full_inputs.get("image_grid_thw"),
            "prompt_len": prompt_len,
            "action_offset": action_offset,
            "advantage": sample["advantage"],
        }

    def _compute_response_logprob(
        self, tokenized: Dict, use_ref: bool = False
    ) -> torch.Tensor:
        """Compute mean log P(response | prompt) for one sample."""
        input_ids = tokenized["input_ids"].unsqueeze(0).to(self.device)
        attn_mask = tokenized["attention_mask"].unsqueeze(0).to(self.device)
        prompt_len = tokenized["prompt_len"]

        fwd_kwargs = {"input_ids": input_ids, "attention_mask": attn_mask}
        if tokenized.get("pixel_values") is not None:
            fwd_kwargs["pixel_values"] = tokenized["pixel_values"].to(self.device)
        if tokenized.get("image_grid_thw") is not None:
            fwd_kwargs["image_grid_thw"] = tokenized["image_grid_thw"].to(self.device)

        # Swap to ref weights if needed
        if use_ref:
            current_state = {}
            for name, param in self._raw_model.named_parameters():
                if name in self._ref_lora_state:
                    current_state[name] = param.data.clone()
                    param.data.copy_(self._ref_lora_state[name])

        ctx = torch.no_grad() if use_ref else torch.enable_grad()
        with ctx:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = self._raw_model.forward(**fwd_kwargs)

            logits = outputs.logits
            # Response tokens: from prompt_len to end
            resp_logits = logits[:, prompt_len - 1:-1, :]
            resp_labels = input_ids[:, prompt_len:]
            log_p = F.log_softmax(resp_logits, dim=-1)
            tok_lp = torch.gather(log_p, -1, resp_labels.unsqueeze(-1)).squeeze(-1)

            mask = (resp_labels != self.pad_id).float()

            # Action-loss-only: zero out reasoning tokens, keep only action
            if self.args.action_loss_only:
                action_offset = tokenized.get("action_offset", 0)
                if action_offset > 0 and action_offset < mask.shape[1]:
                    mask[:, :action_offset] = 0

            n_tokens = mask.sum()
            mean_lp = (tok_lp * mask).sum() / (n_tokens + 1e-8)

        # Restore weights
        if use_ref:
            for name, param in self._raw_model.named_parameters():
                if name in current_state:
                    param.data.copy_(current_state[name])

        return mean_lp

    # -- Training loop ------------------------------------------------

    def train_step(self, sample: Dict) -> Dict[str, float]:
        """One GRPO training step on a single sample."""
        tokenized = self._tokenize_sample(sample)
        advantage = tokenized["advantage"]

        # Skip if response is too short
        resp_len = tokenized["input_ids"].shape[0] - tokenized["prompt_len"]
        if resp_len < 2:
            return {"loss": 0.0, "skipped": 1.0}

        # Truncate if too long
        max_len = self.args.max_seq_len
        if tokenized["input_ids"].shape[0] > max_len:
            tokenized["input_ids"] = tokenized["input_ids"][:max_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:max_len]

        # Compute log prob under current policy
        log_prob = self._compute_response_logprob(tokenized, use_ref=False)

        # Policy gradient loss: -advantage * log_prob
        pg_loss = -advantage * log_prob

        # KL penalty
        kl_loss = torch.tensor(0.0, device=self.device)
        if self.args.beta_kl > 0:
            with torch.no_grad():
                ref_log_prob = self._compute_response_logprob(tokenized, use_ref=True)
            kl = log_prob - ref_log_prob
            kl_loss = self.args.beta_kl * kl

        loss = pg_loss + kl_loss

        # Scale for gradient accumulation
        loss = loss / self.args.grad_accum
        loss.backward()

        return {
            "loss": loss.item() * self.args.grad_accum,
            "pg_loss": pg_loss.item(),
            "kl_loss": kl_loss.item(),
            "advantage": advantage,
            "log_prob": log_prob.item(),
            "reward": sample["reward"],
        }

    def train(self):
        """Full training loop."""
        if self.rank == 0:
            logger.info(f"Starting GRPO training")
            logger.info(f"  Epochs: {self.args.epochs}")
            logger.info(f"  Samples: {len(self.dataset)}")
            logger.info(f"  Grad accum: {self.args.grad_accum}")
            logger.info(f"  LR: {self.args.lr}, beta_kl: {self.args.beta_kl}")
            logger.info(f"  LoRA rank: {self.args.lora_rank}")

        global_step = 0
        best_mean_reward = 0

        for epoch in range(self.args.epochs):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)

            epoch_stats = {
                "loss": [], "pg_loss": [], "kl_loss": [],
                "advantage": [], "log_prob": [], "reward": [],
            }
            accum_count = 0

            for i, sample in enumerate(self.dataloader):
                try:
                    stats = self.train_step(sample)
                except Exception as e:
                    if self.rank == 0:
                        logger.warning(f"Step error: {e}")
                    continue

                for k, v in stats.items():
                    if k in epoch_stats:
                        epoch_stats[k].append(v)

                accum_count += 1
                if accum_count >= self.args.grad_accum:
                    # Gradient step
                    if self.args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.args.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    accum_count = 0
                    global_step += 1

                    # Log
                    if self.rank == 0 and global_step % self.args.log_interval == 0:
                        n = min(self.args.log_interval * self.args.grad_accum,
                                len(epoch_stats["loss"]))
                        recent_loss = np.mean(epoch_stats["loss"][-n:])
                        recent_pg = np.mean(epoch_stats["pg_loss"][-n:])
                        recent_kl = np.mean(epoch_stats["kl_loss"][-n:])
                        recent_reward = np.mean(epoch_stats["reward"][-n:])
                        progress = (i + 1) / len(self.dataloader) * 100
                        logger.info(
                            f"[Epoch {epoch+1}] step={global_step} "
                            f"progress={progress:.1f}% "
                            f"loss={recent_loss:.4f} "
                            f"pg={recent_pg:.4f} kl={recent_kl:.4f} "
                            f"reward={recent_reward:.3f} "
                            f"lr={self.scheduler.get_last_lr()[0]:.2e}")

            # End of epoch
            if self.rank == 0:
                mean_loss = np.mean(epoch_stats["loss"])
                mean_reward = np.mean(epoch_stats["reward"])
                mean_kl = np.mean(epoch_stats["kl_loss"])
                pos_adv = np.mean([a for a in epoch_stats["advantage"] if a > 0])
                neg_adv = np.mean([a for a in epoch_stats["advantage"] if a < 0])
                logger.info(
                    f"\n{'='*60}\n"
                    f"Epoch {epoch+1}/{self.args.epochs} complete\n"
                    f"  Loss: {mean_loss:.4f}, KL: {mean_kl:.4f}\n"
                    f"  Mean reward: {mean_reward:.3f}\n"
                    f"  Mean advantage: pos={pos_adv:.3f}, neg={neg_adv:.3f}\n"
                    f"  Global steps: {global_step}\n"
                    f"{'='*60}")

                # Save checkpoint
                self._save_checkpoint(epoch + 1, global_step)

    def _save_checkpoint(self, epoch: int, step: int):
        """Save LoRA weights."""
        save_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        self._raw_model.save_pretrained(save_dir)
        self.processor.save_pretrained(save_dir)
        logger.info(f"Saved checkpoint to {save_dir}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Offline GRPO trainer")
    # Model
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    # Data
    parser.add_argument("--grpo_data", required=True)
    parser.add_argument("--min_group_std", type=float, default=0.01)
    parser.add_argument("--type_filtered", action="store_true", default=False,
                        help="Only keep same-type candidates per group to prevent mode collapse")
    parser.add_argument("--gaussian_sigma", type=float, default=0.0,
                        help="Gaussian sigma for continuous coordinate reward (0=disabled)")
    parser.add_argument("--rft_mode", action="store_true", default=False,
                        help="Rejection Sampling FT: keep only correct candidates, advantage=1.0")
    parser.add_argument("--rft_threshold", type=float, default=0.8,
                        help="Minimum reward for RFT acceptance")
    parser.add_argument("--action_loss_only", action="store_true", default=False,
                        help="Compute loss only on action tokens (after <tool_call>)")
    parser.add_argument("--max_seq_len", type=int, default=4096)
    # Training
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta_kl", type=float, default=0.05,
                        help="KL penalty coefficient")
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=20)
    # Output
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    trainer = GRPOTrainer(args)
    trainer.train()

    if trainer.rank == 0:
        logger.info("GRPO training complete!")


if __name__ == "__main__":
    main()
