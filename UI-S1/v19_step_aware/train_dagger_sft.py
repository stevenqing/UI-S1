#!/usr/bin/env python3
"""DAgger SFT trainer: fine-tune on (pred_history, GT_action) pairs.

Standard SFT trains on (GT_history, GT_action), creating a distribution mismatch
at test time when the model sees its own imperfect history. DAgger SFT trains on
the model's actual rollout states paired with expert (GT) actions.

Input: JSONL from prepare_dagger_data.py, each line has:
  - screenshot, pred_history, gt_response (expert action)

Training: Standard cross-entropy SFT loss on gt_response tokens,
conditioned on (screenshot + pred_history prompt).

Usage:
  torchrun --nproc_per_node=4 v19_step_aware/train_dagger_sft.py \
    --model_path .../checkpoint-272 \
    --dagger_data v19_step_aware/data/dagger_rollouts.jsonl \
    --output_dir v19_step_aware/output/dagger_sft \
    --lora_rank 64 --lr 2e-6 --epochs 2
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
logger = logging.getLogger("dagger_sft")


# Same prompt as used in rollout and eval
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


class DAggerDataset(Dataset):
    """Dataset of (pred_history_prompt, gt_response) pairs from DAgger rollouts.

    Modes:
      - all: Use all steps (both diverged and non-diverged)
      - diverged_only: Only steps where history has diverged from GT
      - mixed: 50/50 mix of pred_history and GT_history for regularization
    """

    def __init__(self, data_path: str, mode: str = "all",
                 include_gt_history: bool = False, mix_ratio: float = 0.5):
        self.samples = []
        self.stats = {
            "total_steps": 0, "diverged_steps": 0,
            "pred_history_samples": 0, "gt_history_samples": 0,
        }

        with open(data_path) as f:
            for line in f:
                step = json.loads(line)
                self.stats["total_steps"] += 1
                if step.get("history_diverged"):
                    self.stats["diverged_steps"] += 1

                gt_response = step["gt_response"]
                if not gt_response:
                    continue

                # Primary sample: pred_history + GT action
                if mode == "diverged_only" and not step.get("history_diverged"):
                    pass  # Skip non-diverged for this mode
                else:
                    self.samples.append({
                        "screenshot": step["screenshot"],
                        "goal": step["goal"],
                        "history": step["pred_history"],
                        "response": gt_response,
                        "history_type": "pred",
                    })
                    self.stats["pred_history_samples"] += 1

                # Optional: also include GT history samples for regularization
                if include_gt_history:
                    self.samples.append({
                        "screenshot": step["screenshot"],
                        "goal": step["goal"],
                        "history": step["gt_history"],
                        "response": gt_response,
                        "history_type": "gt",
                    })
                    self.stats["gt_history_samples"] += 1

        logger.info(f"DAgger dataset: {self.stats}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class DAggerTrainer:
    """SFT trainer for DAgger data (Qwen2.5-VL + LoRA)."""

    def __init__(self, args):
        self.args = args
        self._setup_distributed()
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()

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
        dataset = DAggerDataset(
            self.args.dagger_data,
            mode=self.args.data_mode,
            include_gt_history=self.args.include_gt_history,
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

        return {
            "input_ids": full_inputs["input_ids"].squeeze(0),
            "attention_mask": full_inputs["attention_mask"].squeeze(0),
            "pixel_values": full_inputs.get("pixel_values"),
            "image_grid_thw": full_inputs.get("image_grid_thw"),
            "prompt_len": prompt_len,
        }

    def train_step(self, sample: Dict) -> Dict[str, float]:
        """One SFT training step: cross-entropy loss on GT response."""
        tokenized = self._tokenize_sample(sample)

        # Skip if response is too short
        resp_len = tokenized["input_ids"].shape[0] - tokenized["prompt_len"]
        if resp_len < 2:
            return {"loss": 0.0, "skipped": 1.0}

        # Truncate if too long
        max_len = self.args.max_seq_len
        if tokenized["input_ids"].shape[0] > max_len:
            tokenized["input_ids"] = tokenized["input_ids"][:max_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:max_len]

        input_ids = tokenized["input_ids"].unsqueeze(0).to(self.device)
        attn_mask = tokenized["attention_mask"].unsqueeze(0).to(self.device)
        prompt_len = tokenized["prompt_len"]

        fwd_kwargs = {"input_ids": input_ids, "attention_mask": attn_mask}
        if tokenized.get("pixel_values") is not None:
            fwd_kwargs["pixel_values"] = tokenized["pixel_values"].to(self.device)
        if tokenized.get("image_grid_thw") is not None:
            fwd_kwargs["image_grid_thw"] = tokenized["image_grid_thw"].to(self.device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = self._raw_model.forward(**fwd_kwargs)

        logits = outputs.logits
        # Response tokens: from prompt_len to end
        resp_logits = logits[:, prompt_len - 1:-1, :]  # [1, resp_len, vocab]
        resp_labels = input_ids[:, prompt_len:]          # [1, resp_len]

        # Cross-entropy loss (standard SFT)
        loss = F.cross_entropy(
            resp_logits.reshape(-1, resp_logits.size(-1)),
            resp_labels.reshape(-1),
            ignore_index=self.pad_id,
        )

        # Scale for gradient accumulation
        scaled_loss = loss / self.args.grad_accum
        scaled_loss.backward()

        return {
            "loss": loss.item(),
            "resp_len": resp_len,
        }

    def train(self):
        """Full training loop."""
        if self.rank == 0:
            logger.info(f"Starting DAgger SFT training")
            logger.info(f"  Epochs: {self.args.epochs}")
            logger.info(f"  Samples: {len(self.dataset)}")
            logger.info(f"  Grad accum: {self.args.grad_accum}")
            logger.info(f"  LR: {self.args.lr}")
            logger.info(f"  LoRA rank: {self.args.lora_rank}")
            logger.info(f"  Data mode: {self.args.data_mode}")
            logger.info(f"  Include GT history: {self.args.include_gt_history}")

        global_step = 0

        for epoch in range(self.args.epochs):
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)

            epoch_losses = []
            accum_count = 0

            for i, sample in enumerate(self.dataloader):
                try:
                    stats = self.train_step(sample)
                except Exception as e:
                    if self.rank == 0:
                        logger.warning(f"Step error: {e}")
                    continue

                epoch_losses.append(stats["loss"])
                accum_count += 1

                if accum_count >= self.args.grad_accum:
                    if self.args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.args.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    accum_count = 0
                    global_step += 1

                    if self.rank == 0 and global_step % self.args.log_interval == 0:
                        n = min(self.args.log_interval * self.args.grad_accum,
                                len(epoch_losses))
                        recent_loss = np.mean(epoch_losses[-n:])
                        progress = (i + 1) / len(self.dataloader) * 100
                        logger.info(
                            f"[Epoch {epoch+1}] step={global_step} "
                            f"progress={progress:.1f}% "
                            f"loss={recent_loss:.4f} "
                            f"lr={self.scheduler.get_last_lr()[0]:.2e}")

            # End of epoch
            if self.rank == 0:
                mean_loss = np.mean(epoch_losses)
                logger.info(
                    f"\n{'='*60}\n"
                    f"Epoch {epoch+1}/{self.args.epochs} complete\n"
                    f"  Loss: {mean_loss:.4f}\n"
                    f"  Global steps: {global_step}\n"
                    f"{'='*60}")

                save_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch{epoch+1}")
                os.makedirs(save_dir, exist_ok=True)
                self._raw_model.save_pretrained(save_dir)
                self.processor.save_pretrained(save_dir)
                logger.info(f"Saved checkpoint to {save_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="DAgger SFT trainer")
    # Model
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    # Data
    parser.add_argument("--dagger_data", required=True)
    parser.add_argument("--data_mode", default="all",
                        choices=["all", "diverged_only"],
                        help="Which steps to train on")
    parser.add_argument("--include_gt_history", action="store_true", default=False,
                        help="Also include GT history samples for regularization")
    parser.add_argument("--max_seq_len", type=int, default=4096)
    # Training
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=20)
    # Output
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    trainer = DAggerTrainer(args)
    trainer.train()

    if trainer.rank == 0:
        logger.info("DAgger SFT training complete!")


if __name__ == "__main__":
    main()
