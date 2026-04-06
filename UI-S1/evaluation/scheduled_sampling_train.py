#!/usr/bin/env python3
"""
Exp 1: Scheduled Sampling SFT Training

Tests the hypothesis that teacher forcing causes representational fragmentation.
By mixing autoregressive steps into SFT training, we break the shortcuts that
teacher forcing permits, forcing the model to develop cross-subspace pathways.

For each training sample, with probability p_auto:
  - Context tokens (system + user): teacher forcing (standard)
  - Action tokens (assistant response): first `n_tf_tokens` use GT (teacher forcing),
    remaining tokens generated autoregressively using model's own predictions
  - CE loss computed against GT for ALL positions (both TF and AR)

Three conditions:
  p_auto = 0.0 → standard SFT (baseline, already have SFT v2)
  p_auto = 0.3 → light scheduled sampling
  p_auto = 0.7 → heavy scheduled sampling

Usage:
  torchrun --nproc_per_node=4 evaluation/scheduled_sampling_train.py \
      --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
      --train_data train_GUI_360/llamafactory/data/gui360_mixed_train.json \
      --val_data train_GUI_360/llamafactory/data/gui360_mixed_val.json \
      --output_dir train_GUI_360/llamafactory/output/scheduled_sampling_p0.3 \
      --p_auto 0.3 --num_epochs 2 \
      --per_device_batch_size 1 --gradient_accumulation_steps 16
"""

import argparse
import gc
import json
import os
import re
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from torch.utils.data import Dataset


# ── Constants ────────────────────────────────────────────────────────
IGNORE_INDEX = -100


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════

class GUI360SFTDataset(Dataset):
    """Loads ShareGPT-format GUI-360 data for SFT (supports both AP and Grounding)."""

    def __init__(self, json_path: str, processor, max_samples: int = 0,
                 max_length: int = 8192):
        with open(json_path) as f:
            self.data = json.load(f)
        if 0 < max_samples < len(self.data):
            np.random.seed(42)
            indices = np.random.choice(len(self.data), max_samples, replace=False)
            self.data = [self.data[i] for i in sorted(indices)]
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        convs = item["conversations"]
        images = item.get("images", [])

        # Build Qwen2.5-VL message format
        user_text = convs[0]["value"]
        assistant_text = convs[1]["value"]
        user_text_clean = user_text.replace("<image>\n", "").replace("<image>", "").strip()

        user_content = []
        if images:
            user_content.append({"type": "image", "image": images[0]})
        user_content.append({"type": "text", "text": user_text_clean})

        messages_full = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]
        messages_prompt = [
            {"role": "user", "content": user_content},
        ]

        try:
            full_text = self.processor.apply_chat_template(
                messages_full, tokenize=False, add_generation_prompt=False)
            prompt_text = self.processor.apply_chat_template(
                messages_prompt, tokenize=False, add_generation_prompt=True)

            # Load image
            image = None
            if images and os.path.exists(images[0]):
                image = Image.open(images[0]).convert("RGB")

            if image is not None:
                full_inputs = self.processor(
                    text=[full_text], images=[image],
                    return_tensors="pt", padding=False,
                    max_length=self.max_length, truncation=True)
                prompt_inputs = self.processor(
                    text=[prompt_text], images=[image],
                    return_tensors="pt", padding=False,
                    max_length=self.max_length, truncation=True)
            else:
                full_inputs = self.processor(
                    text=[full_text],
                    return_tensors="pt", padding=False,
                    max_length=self.max_length, truncation=True)
                prompt_inputs = self.processor(
                    text=[prompt_text],
                    return_tensors="pt", padding=False,
                    max_length=self.max_length, truncation=True)

            input_ids = full_inputs["input_ids"].squeeze(0)
            attention_mask = full_inputs["attention_mask"].squeeze(0)

            # Labels: mask prompt tokens with IGNORE_INDEX
            prompt_len = prompt_inputs["input_ids"].shape[1]
            labels = input_ids.clone()
            labels[:prompt_len] = IGNORE_INDEX

            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "prompt_len": prompt_len,  # boundary between context and action tokens
            }

            if "pixel_values" in full_inputs:
                result["pixel_values"] = full_inputs["pixel_values"].squeeze(0)
            if "image_grid_thw" in full_inputs:
                result["image_grid_thw"] = full_inputs["image_grid_thw"].squeeze(0)

            return result

        except Exception as e:
            return None


# ═══════════════════════════════════════════════════════════════════════
# Collate
# ═══════════════════════════════════════════════════════════════════════

def sft_collate_fn(batch):
    """Collate with padding, None filtering, and prompt_len passthrough."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids = []
    attention_mask = []
    labels = []
    pixel_values_list = []
    image_grid_thw_list = []
    prompt_lens = []

    for b in batch:
        seq_len = b["input_ids"].shape[0]
        pad_len = max_len - seq_len

        input_ids.append(F.pad(b["input_ids"], (0, pad_len), value=0))
        attention_mask.append(F.pad(b["attention_mask"], (0, pad_len), value=0))
        labels.append(F.pad(b["labels"], (0, pad_len), value=IGNORE_INDEX))

        if "pixel_values" in b:
            pixel_values_list.append(b["pixel_values"])
        if "image_grid_thw" in b:
            image_grid_thw_list.append(b["image_grid_thw"])

        prompt_lens.append(b["prompt_len"])

    result = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
        "prompt_lens": prompt_lens,
    }

    if pixel_values_list:
        result["pixel_values"] = torch.cat(pixel_values_list, dim=0)
    if image_grid_thw_list:
        result["image_grid_thw"] = torch.stack(image_grid_thw_list)

    return result


# ═══════════════════════════════════════════════════════════════════════
# Scheduled Sampling Trainer
# ═══════════════════════════════════════════════════════════════════════

class ScheduledSamplingTrainer(Trainer):
    """
    HF Trainer with scheduled sampling for action tokens.

    With probability p_auto, action tokens are generated autoregressively
    (using model's own predictions as input) instead of teacher forcing.
    CE loss is still computed against GT labels at all positions.

    Implementation:
    - For the autoregressive portion, we loop token-by-token
    - We build a new input_ids tensor where AR positions use predicted tokens
    - Then do a single forward pass with the modified input_ids to get the loss
    - This is more efficient than doing the forward pass token-by-token
      (one forward pass vs N forward passes)
    """

    def __init__(self, p_auto: float = 0.0, n_tf_tokens: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.p_auto = p_auto
        self.n_tf_tokens = n_tf_tokens  # number of action tokens to always teacher-force
        # Logging accumulators
        self._tf_count = 0
        self._ar_count = 0
        self._ar_token_count = 0
        self._total_token_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        prompt_lens = inputs.pop("prompt_lens", None)

        # Decide: teacher forcing or scheduled sampling for this batch
        # Apply p_auto per-sample in the batch
        batch_size = inputs["input_ids"].shape[0]
        use_ar = [False] * batch_size

        if self.p_auto > 0 and prompt_lens is not None:
            for i in range(batch_size):
                if torch.rand(1).item() < self.p_auto:
                    use_ar[i] = True

        any_ar = any(use_ar)

        if not any_ar:
            # Standard teacher forcing — fast path
            self._tf_count += batch_size
            outputs = model(**inputs, return_dict=True)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

        # ── Scheduled Sampling Path ──
        # For each sample marked as AR:
        # 1. Run a greedy forward pass to get the model's predictions for action tokens
        # 2. Replace input_ids at AR positions with predicted tokens
        # 3. Run a final forward pass with modified input_ids to compute loss

        input_ids = inputs["input_ids"].clone()
        labels = inputs["labels"]
        seq_len = input_ids.shape[1]

        # We need to generate predictions for action tokens
        # Strategy: single forward pass to get logits, then argmax for AR positions
        with torch.no_grad():
            # Forward pass to get logits at all positions
            logits_outputs = model(**inputs, return_dict=True)
            logits = logits_outputs.logits  # (B, seq_len, vocab)

        # Now modify input_ids for AR positions
        # For position t, the prediction comes from logits at position t-1
        modified_input_ids = input_ids.clone()

        for i in range(batch_size):
            if not use_ar[i]:
                self._tf_count += 1
                continue

            self._ar_count += 1
            prompt_len = prompt_lens[i]
            # Action tokens start at prompt_len
            # Keep first n_tf_tokens of action as teacher-forced
            ar_start = prompt_len + self.n_tf_tokens

            if ar_start >= seq_len:
                continue

            # Replace tokens from ar_start onward with model's greedy predictions
            # Prediction for position t comes from logits[t-1]
            n_replaced = 0
            for t in range(ar_start, seq_len):
                if labels[i, t] == IGNORE_INDEX:
                    break  # Past the end of response (padding)
                pred_token = logits[i, t - 1].argmax(dim=-1)
                modified_input_ids[i, t] = pred_token
                n_replaced += 1

            self._ar_token_count += n_replaced
            self._total_token_count += (seq_len - prompt_len)

        # Now do the real forward pass with modified input_ids
        inputs["input_ids"] = modified_input_ids
        outputs = model(**inputs, return_dict=True)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        """Inject scheduled sampling stats."""
        total = self._tf_count + self._ar_count
        if total > 0:
            logs["ss_ar_ratio"] = round(self._ar_count / total, 4)
        if self._total_token_count > 0:
            logs["ss_ar_token_ratio"] = round(
                self._ar_token_count / self._total_token_count, 4)

        self._tf_count = 0
        self._ar_count = 0
        self._ar_token_count = 0
        self._total_token_count = 0

        super().log(logs, *args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════
# Metrics Callback
# ═══════════════════════════════════════════════════════════════════════

class SSMetricsCallback(TrainerCallback):
    """Track scheduled sampling metrics over training."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.step_metrics = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry = {"step": state.global_step}
        for key in ("loss", "learning_rate", "ss_ar_ratio", "ss_ar_token_ratio"):
            if key in logs:
                entry[key] = logs[key]
        if len(entry) > 1:
            self.step_metrics.append(entry)

    def on_train_end(self, args, state, control, **kwargs):
        out_path = os.path.join(self.output_dir, "ss_metrics.json")
        with open(out_path, "w") as f:
            json.dump(self.step_metrics, f, indent=2)
        print(f"SS metrics saved to {out_path} ({len(self.step_metrics)} records)")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Exp 1: Scheduled Sampling SFT Training")
    parser.add_argument("--model_path",
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/checkpoints/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--train_data",
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/llamafactory/data/gui360_mixed_train.json")
    parser.add_argument("--val_data",
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/llamafactory/data/gui360_mixed_val.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--p_auto", type=float, default=0.3,
                        help="Probability of using autoregressive for action tokens")
    parser.add_argument("--n_tf_tokens", type=int, default=3,
                        help="Number of action tokens to always teacher-force before switching to AR")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--num_epochs", type=float, default=2.0)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--deepspeed_config", type=str, default=None,
                        help="Path to DeepSpeed config JSON (e.g., ds_z3_config.json)")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Exp 1: Scheduled Sampling SFT Training")
    print("=" * 80)
    print(f"Model:       {args.model_path}")
    print(f"Train data:  {args.train_data}")
    print(f"Val data:    {args.val_data}")
    print(f"Output:      {args.output_dir}")
    print(f"p_auto:      {args.p_auto}")
    print(f"n_tf_tokens: {args.n_tf_tokens}")
    print(f"Epochs:      {args.num_epochs}")
    print(f"Batch:       {args.per_device_batch_size} × grad_accum={args.gradient_accumulation_steps}")
    print(f"LR:          {args.learning_rate}")
    print()

    # ── Load model (full parameter, matching SFT v2/v3 training) ──
    print("Loading model...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Freeze vision tower (matching SFT v2/v3)
    for name, param in model.named_parameters():
        if "visual" in name:
            param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")

    # Enable gradient checkpointing
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    # ── Load processor ──
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True)

    # ── Load datasets ──
    print("Loading datasets...", flush=True)
    train_dataset = GUI360SFTDataset(
        args.train_data, processor,
        max_samples=args.max_train_samples,
        max_length=args.max_length)
    val_dataset = GUI360SFTDataset(
        args.val_data, processor,
        max_samples=200,
        max_length=args.max_length)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ── Save experiment config ──
    config_log = {
        "experiment": "scheduled_sampling",
        "p_auto": args.p_auto,
        "n_tf_tokens": args.n_tf_tokens,
        "model_path": args.model_path,
        "learning_rate": args.learning_rate,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_epochs": args.num_epochs,
        "max_length": args.max_length,
        "train_samples": len(train_dataset),
        "total_trainable_params": total_params,
        "frozen_vision": True,
    }
    with open(os.path.join(args.output_dir, "experiment_config.json"), "w") as f:
        json.dump(config_log, f, indent=2)

    # ── Training arguments (matching SFT v2/v3 config) ──
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.1,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=5,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="wandb",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
        ddp_timeout=10800,  # 3 hours — prevent NCCL timeout during ZeRO-3 allgather
        deepspeed=args.deepspeed_config,
    )

    # ── Callbacks ──
    ss_callback = SSMetricsCallback(output_dir=args.output_dir)

    # ── Trainer ──
    trainer = ScheduledSamplingTrainer(
        p_auto=args.p_auto,
        n_tf_tokens=args.n_tf_tokens,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=sft_collate_fn,
        callbacks=[ss_callback],
    )

    # ── Train ──
    print("Starting training...", flush=True)
    trainer.train()

    # ── Save final model ──
    trainer.save_model(os.path.join(args.output_dir, "final"))
    print(f"\nTraining complete. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
