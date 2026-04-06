#!/usr/bin/env python3
"""
Stage-wise LoRA Training for Composition Granularity Experiment

Conditions:
  A (uniform):         rank=16 all layers
  B (stagewise-equal): rank=16 all layers, tracked by group (same as A, control)
  C (stagewise-probe): rank=8 L0-9, rank=24 L10-18, rank=16 L19-27

Built-in: gradient conflict recording per layer group every N steps.

Usage:
  torchrun --nproc_per_node=4 evaluation/stagewise_lora_train.py \
      --condition A --output_dir output/stagewise_A
"""

import argparse
import gc
import json
import math
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
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset


# ── Constants ────────────────────────────────────────────────────────
LAYER_GROUPS = {
    "encode": list(range(0, 10)),    # L0-9: visual feature enrichment
    "bind":   list(range(10, 19)),   # L10-18: cross-modal binding
    "ground": list(range(19, 28)),   # L19-27: action composition/grounding
}

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Condition configs: {condition_name: {layer_id: rank}}
def get_rank_map(condition: str) -> Dict[int, int]:
    if condition == "A":
        return {i: 16 for i in range(28)}
    elif condition == "B":
        return {i: 16 for i in range(28)}  # same as A, control
    elif condition == "C":
        ranks = {}
        for i in LAYER_GROUPS["encode"]:
            ranks[i] = 8
        for i in LAYER_GROUPS["bind"]:
            ranks[i] = 24
        for i in LAYER_GROUPS["ground"]:
            ranks[i] = 16
        return ranks
    else:
        raise ValueError(f"Unknown condition: {condition}")


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════

class GUI360SFTDataset(Dataset):
    """Loads ShareGPT-format GUI-360 data for Qwen2.5-VL training."""

    def __init__(self, json_path: str, processor, max_samples: int = 0,
                 max_length: int = 4096):
        with open(json_path) as f:
            self.data = json.load(f)
        if 0 < max_samples < len(self.data):
            np.random.seed(42)
            indices = np.random.choice(len(self.data), max_samples, replace=False)
            self.data = [self.data[i] for i in indices]
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

        # Replace <image> tag with actual image reference
        user_text_clean = user_text.replace("<image>\n", "").replace("<image>", "").strip()

        # Build messages for full sequence (with assistant response)
        user_content = []
        if images:
            user_content.append({"type": "image", "image": images[0]})
        user_content.append({"type": "text", "text": user_text_clean})

        messages_full = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ]

        # Build messages for prompt only (to find label start position)
        messages_prompt = [
            {"role": "user", "content": user_content},
        ]

        # Tokenize
        try:
            full_text = self.processor.apply_chat_template(
                messages_full, tokenize=False, add_generation_prompt=False)
            prompt_text = self.processor.apply_chat_template(
                messages_prompt, tokenize=False, add_generation_prompt=True)

            # Load image
            if images and os.path.exists(images[0]):
                image = Image.open(images[0]).convert("RGB")
            else:
                image = None

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

            # Labels: mask prompt tokens with -100
            prompt_len = prompt_inputs["input_ids"].shape[1]
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

            # Add vision inputs if present
            if "pixel_values" in full_inputs:
                result["pixel_values"] = full_inputs["pixel_values"].squeeze(0)
            if "image_grid_thw" in full_inputs:
                result["image_grid_thw"] = full_inputs["image_grid_thw"].squeeze(0)

            return result

        except Exception as e:
            # Return a dummy sample on error (will be filtered by collator)
            return None


def collate_fn(batch):
    """Collate with padding and None filtering."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    # Find max length
    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids = []
    attention_mask = []
    labels = []
    pixel_values_list = []
    image_grid_thw_list = []

    for b in batch:
        seq_len = b["input_ids"].shape[0]
        pad_len = max_len - seq_len

        input_ids.append(F.pad(b["input_ids"], (0, pad_len), value=0))
        attention_mask.append(F.pad(b["attention_mask"], (0, pad_len), value=0))
        labels.append(F.pad(b["labels"], (0, pad_len), value=-100))

        if "pixel_values" in b:
            pixel_values_list.append(b["pixel_values"])
        if "image_grid_thw" in b:
            image_grid_thw_list.append(b["image_grid_thw"])

    result = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }

    if pixel_values_list:
        result["pixel_values"] = torch.cat(pixel_values_list, dim=0)
    if image_grid_thw_list:
        result["image_grid_thw"] = torch.stack(image_grid_thw_list)

    return result


# ═══════════════════════════════════════════════════════════════════════
# Gradient Conflict Callback (Experiment 2)
# ═══════════════════════════════════════════════════════════════════════

class GradientConflictCallback(TrainerCallback):
    """Record per-layer-group gradient statistics every N steps."""

    def __init__(self, record_every: int = 50, output_dir: str = "."):
        self.record_every = record_every
        self.output_dir = output_dir
        self.stats = []

    def _get_group_grads(self, model):
        """Collect LoRA gradients per layer group."""
        group_grads = {g: [] for g in LAYER_GROUPS}

        for name, param in model.named_parameters():
            if param.grad is None or "lora" not in name:
                continue

            # Extract layer index from name
            m = re.search(r'layers\.(\d+)\.', name)
            if m is None:
                continue
            layer_id = int(m.group(1))

            for group_name, layer_ids in LAYER_GROUPS.items():
                if layer_id in layer_ids:
                    group_grads[group_name].append(param.grad.detach().float().flatten())
                    break

        # Concatenate per group
        result = {}
        for g, grads in group_grads.items():
            if grads:
                result[g] = torch.cat(grads)
        return result

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.record_every != 0 or state.global_step == 0:
            return

        group_grads = self._get_group_grads(model)
        if len(group_grads) < 2:
            return

        # Compute pairwise cosine similarity
        groups = sorted(group_grads.keys())
        step_stats = {"step": state.global_step}

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1, g2 = groups[i], groups[j]
                v1, v2 = group_grads[g1], group_grads[g2]
                # Cosine similarity
                cos = float(F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)))
                step_stats[f"cos_{g1}_{g2}"] = round(cos, 6)

        # Gradient norms per group
        for g in groups:
            step_stats[f"norm_{g}"] = round(float(group_grads[g].norm()), 4)

        self.stats.append(step_stats)

        if state.global_step % (self.record_every * 5) == 0:
            print(f"[Grad Conflict Step {state.global_step}] " +
                  " | ".join(f"{k}={v}" for k, v in step_stats.items() if k != "step"))

    def on_train_end(self, args, state, control, **kwargs):
        out_path = os.path.join(self.output_dir, "gradient_conflict_stats.json")
        with open(out_path, "w") as f:
            json.dump(self.stats, f, indent=2)
        print(f"Gradient conflict stats saved to {out_path} ({len(self.stats)} records)")

        # Print summary
        if self.stats:
            print("\n── Gradient Conflict Summary ──")
            keys = [k for k in self.stats[0] if k.startswith("cos_")]
            for k in keys:
                vals = [s[k] for s in self.stats if k in s]
                print(f"  {k}: mean={np.mean(vals):+.4f}, std={np.std(vals):.4f}, "
                      f"min={np.min(vals):+.4f}, max={np.max(vals):+.4f}")


# ═══════════════════════════════════════════════════════════════════════
# Model Setup
# ═══════════════════════════════════════════════════════════════════════

def build_lora_config(condition: str) -> LoraConfig:
    """Build PEFT LoraConfig for the given condition."""
    rank_map = get_rank_map(condition)

    # Use the most common rank as default, override the rest with rank_pattern
    from collections import Counter
    rank_counts = Counter(rank_map.values())
    default_rank = rank_counts.most_common(1)[0][0]

    rank_pattern = {}
    alpha_pattern = {}
    for layer_id, rank in rank_map.items():
        if rank != default_rank:
            # Create regex pattern for this layer
            for module in LORA_TARGET_MODULES:
                key = f"model.layers.{layer_id}.*.{module}"
                rank_pattern[key] = rank
                alpha_pattern[key] = rank * 2  # alpha = 2 * rank

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=default_rank,
        lora_alpha=default_rank * 2,
        lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        bias="none",
    )
    return config


def count_lora_params(model):
    """Count trainable LoRA parameters per layer group."""
    group_params = {g: 0 for g in LAYER_GROUPS}
    total = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = param.numel()
        total += n

        m = re.search(r'layers\.(\d+)\.', name)
        if m:
            layer_id = int(m.group(1))
            for g, ids in LAYER_GROUPS.items():
                if layer_id in ids:
                    group_params[g] += n
                    break

    return total, group_params


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Stage-wise LoRA Training")
    parser.add_argument("--condition", required=True, choices=["A", "B", "C"])
    parser.add_argument("--model_path", default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/checkpoints/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--train_data", default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/llamafactory/data/gui360_train.json")
    parser.add_argument("--val_data", default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/llamafactory/data/gui360_val.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_train_samples", type=int, default=0,
                        help="0 = use all data")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--num_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1.5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--grad_record_every", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Log config
    rank_map = get_rank_map(args.condition)
    print(f"Condition: {args.condition}")
    print(f"Rank map: encode={rank_map[0]}, bind={rank_map[10]}, ground={rank_map[19]}")
    print(f"Total rank units: {sum(rank_map.values())}")
    print(f"Output: {args.output_dir}")
    print()

    # Load model
    print("Loading model...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Freeze vision tower
    for name, param in model.named_parameters():
        if "visual" in name:
            param.requires_grad = False

    # Apply LoRA
    print("Applying LoRA...", flush=True)
    lora_config = build_lora_config(args.condition)
    model = get_peft_model(model, lora_config)

    total_params, group_params = count_lora_params(model)
    print(f"Trainable params: {total_params:,}")
    for g, n in group_params.items():
        print(f"  {g}: {n:,} ({100*n/total_params:.1f}%)")
    print()

    # Enable gradient checkpointing
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Load processor
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # Load data
    print("Loading datasets...", flush=True)
    train_dataset = GUI360SFTDataset(
        args.train_data, processor,
        max_samples=args.max_train_samples,
        max_length=args.max_length)
    val_dataset = GUI360SFTDataset(
        args.val_data, processor,
        max_samples=100,
        max_length=args.max_length)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Save config
    lora_dict = lora_config.to_dict()
    # Convert sets to lists for JSON serialization
    for k, v in lora_dict.items():
        if isinstance(v, set):
            lora_dict[k] = sorted(list(v))
    config_log = {
        "condition": args.condition,
        "rank_map": {str(k): v for k, v in rank_map.items()},
        "total_trainable_params": total_params,
        "group_params": group_params,
        "lora_config": lora_dict,
    }
    with open(os.path.join(args.output_dir, "experiment_config.json"), "w") as f:
        json.dump(config_log, f, indent=2)

    # Training arguments
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
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
    )

    # Gradient conflict callback
    grad_callback = GradientConflictCallback(
        record_every=args.grad_record_every,
        output_dir=args.output_dir)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[grad_callback],
    )

    # Train
    print("Starting training...", flush=True)
    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(args.output_dir, "final"))
    print(f"\nTraining complete. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
