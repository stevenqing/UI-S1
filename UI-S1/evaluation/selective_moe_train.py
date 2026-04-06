#!/usr/bin/env python3
"""
Selective MoE Experiment: Training Script

Tests the routing channel bottleneck hypothesis by progressively applying MoE
(2 experts × r=16) to different module subsets while keeping total LoRA params
constant (~80.7M).

Conditions:
  C1b: No MoE modules (two-pass control, standard LoRA r=32 on all 7 modules)
  C2:  MoE on k_proj only
  C3:  MoE on q_proj + k_proj
  C4:  MoE on q,k,v,o_proj (full attention)
  C5:  MoE on all 7 modules

Uses MoEVLMWrapper in hybrid mode: MoE modules get 2 experts × r=16,
standard modules get 1 expert × r=32. Same total parameters per module.

Usage:
  torchrun --nproc_per_node=4 evaluation/selective_moe_train.py \
      --output_dir train_GUI_360/llamafactory/output/selective_moe_c2 \
      --moe_modules k_proj \
      --num_experts 2 --moe_r 16 --standard_r 32
"""

import argparse
import json
import math
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Ensure project root is on sys.path for imports
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# Ensure evaluation dir is on sys.path for sibling imports
_eval_dir = os.path.dirname(os.path.abspath(__file__))
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

# Reuse dataset and collate from bind_auxiliary_train (sibling module)
from bind_auxiliary_train import (
    GUI360BindDataset,
    bind_collate_fn,
    LORA_TARGET_MODULES,
)

from verl.models.moe.moe_wrapper import MoEVLMWrapper, MoEConfig


# ═══════════════════════════════════════════════════════════════════════
# SelectiveMoETrainer
# ═══════════════════════════════════════════════════════════════════════

class SelectiveMoETrainer(Trainer):
    """HF Trainer that delegates forward to MoEVLMWrapper (two-pass routing)."""

    def __init__(self, moe_wrapper: MoEVLMWrapper, **kwargs):
        # The Trainer expects `model`, but we pass the wrapper which wraps base_model
        super().__init__(**kwargs)
        self.moe_wrapper = moe_wrapper
        # Running averages for logging
        self._lm_loss_sum = 0.0
        self._balance_loss_sum = 0.0
        self._loss_count = 0
        self._routing_entropy_sum = 0.0
        self._routing_count = 0

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to handle MoEOutput (not subscriptable like CausalLMOutput)."""
        inputs.pop("gt_coords", None)
        inputs.pop("orig_sizes", None)
        with torch.no_grad():
            moe_output = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values"),
                labels=inputs["labels"],
                image_grid_thw=inputs.get("image_grid_thw"),
                return_routing_info=False,
            )
        loss = moe_output.loss
        return (loss, None, None)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Pop non-tensor fields that can't be sent to model
        inputs.pop("gt_coords", None)
        inputs.pop("orig_sizes", None)

        # `model` is the (possibly DDP-wrapped) MoEVLMWrapper
        # MoEVLMWrapper.forward() handles two-pass routing internally
        moe_output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            labels=inputs["labels"],
            image_grid_thw=inputs.get("image_grid_thw"),
            return_routing_info=True,
        )

        loss = moe_output.loss

        # Track metrics
        if loss is not None:
            self._loss_count += 1
            if moe_output.lm_loss is not None:
                self._lm_loss_sum += moe_output.lm_loss.detach().item()
            if moe_output.balance_loss is not None:
                self._balance_loss_sum += moe_output.balance_loss.detach().item()

        # Track routing entropy
        if moe_output.routing_weights is not None and moe_output.routing_weights.shape[1] > 1:
            from verl.models.moe.router import compute_routing_entropy
            entropy = compute_routing_entropy(moe_output.routing_weights)
            self._routing_entropy_sum += entropy.mean().detach().item()
            self._routing_count += 1

        if return_outputs:
            # Trainer expects outputs with .logits for prediction
            return loss, moe_output
        return loss

    def log(self, logs, *args, **kwargs):
        """Inject MoE-specific metrics into logs."""
        if self._loss_count > 0:
            logs["lm_loss"] = round(self._lm_loss_sum / self._loss_count, 6)
            logs["balance_loss"] = round(self._balance_loss_sum / self._loss_count, 6)
        if self._routing_count > 0:
            logs["routing_entropy"] = round(self._routing_entropy_sum / self._routing_count, 4)

        # Reset
        self._lm_loss_sum = 0.0
        self._balance_loss_sum = 0.0
        self._loss_count = 0
        self._routing_entropy_sum = 0.0
        self._routing_count = 0

        super().log(logs, *args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════
# MoE Checkpoint Callback
# ═══════════════════════════════════════════════════════════════════════

class MoECheckpointCallback(TrainerCallback):
    """Save MoE selective checkpoint alongside HF Trainer checkpoints."""

    def __init__(self, moe_wrapper: MoEVLMWrapper):
        self.moe_wrapper = moe_wrapper

    def on_save(self, args, state, control, **kwargs):
        # Save into the same checkpoint directory the Trainer uses
        checkpoint_dir = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}")
        moe_save_dir = os.path.join(checkpoint_dir, "moe_checkpoint")
        self.moe_wrapper.save_selective_checkpoint(moe_save_dir)

    def on_train_end(self, args, state, control, **kwargs):
        # Save final checkpoint
        final_dir = os.path.join(args.output_dir, "final", "moe_checkpoint")
        self.moe_wrapper.save_selective_checkpoint(final_dir)


# ═══════════════════════════════════════════════════════════════════════
# Metrics Callback
# ═══════════════════════════════════════════════════════════════════════

class SelectiveMoEMetricsCallback(TrainerCallback):
    """Track training metrics over time."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.step_metrics = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        entry = {"step": state.global_step}
        for key in ("loss", "lm_loss", "balance_loss", "routing_entropy",
                     "learning_rate"):
            if key in logs:
                entry[key] = logs[key]
        if len(entry) > 1:
            self.step_metrics.append(entry)

    def on_train_end(self, args, state, control, **kwargs):
        out_path = os.path.join(self.output_dir, "selective_moe_metrics.json")
        with open(out_path, "w") as f:
            json.dump(self.step_metrics, f, indent=2)
        print(f"Metrics saved to {out_path} ({len(self.step_metrics)} records)")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Selective MoE Experiment: Training")

    # Model / data
    parser.add_argument("--model_path",
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/checkpoints/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--train_data",
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/llamafactory/data/gui360_train.json")
    parser.add_argument("--val_data",
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/llamafactory/data/gui360_val.json")
    parser.add_argument("--output_dir", required=True)

    # MoE config
    parser.add_argument("--moe_modules", nargs="*", default=None,
                        help="Which target modules get MoE (None = none, C1b baseline)")
    parser.add_argument("--num_experts", type=int, default=2)
    parser.add_argument("--moe_r", type=int, default=16,
                        help="LoRA rank for MoE modules (2 experts × r=16)")
    parser.add_argument("--moe_alpha", type=int, default=32)
    parser.add_argument("--standard_r", type=int, default=32,
                        help="LoRA rank for standard (non-MoE) modules")
    parser.add_argument("--standard_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--balance_weight", type=float, default=0.1)
    parser.add_argument("--router_hidden", type=int, default=256)

    # Training
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--num_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1.5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Handle empty moe_modules (C1b baseline: two-pass overhead, no MoE)
    # If --moe_modules is not provided or empty list, treat as no-MoE baseline.
    # We implement this by setting moe_modules=[] and letting all modules be standard.
    has_moe = args.moe_modules is not None and len(args.moe_modules) > 0

    print("=" * 80)
    print("Selective MoE Experiment: Training")
    print("=" * 80)
    print(f"Model:       {args.model_path}")
    print(f"MoE modules: {args.moe_modules if has_moe else '(none — C1b baseline)'}")
    print(f"Experts:     {args.num_experts} × r={args.moe_r}" if has_moe else "N/A")
    print(f"Standard:    r={args.standard_r}")
    print(f"Output:      {args.output_dir}")
    print()

    # ── Load model ──
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

    # ── Load processor ──
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True)

    # ── Build MoEConfig ──
    # For C1b (no MoE): moe_modules=[] means all modules are standard LoRA.
    # We still need a dummy moe_modules list — use empty list and set
    # moe_modules to a non-existent name so nothing matches.
    if has_moe:
        moe_config = MoEConfig(
            num_experts=args.num_experts,
            top_k=args.num_experts,  # soft routing: all experts
            expert_lora_r=args.moe_r,
            expert_lora_alpha=args.moe_alpha,
            expert_lora_dropout=args.lora_dropout,
            target_modules=LORA_TARGET_MODULES,
            moe_modules=args.moe_modules,
            standard_lora_r=args.standard_r,
            standard_lora_alpha=args.standard_alpha,
            router_hidden=args.router_hidden,
            balance_weight=args.balance_weight,
        )
    else:
        # C1b baseline: all modules are standard LoRA r=32
        # Use moe_modules=[] → moe_module_set will be empty, all go to standard
        moe_config = MoEConfig(
            num_experts=args.num_experts,
            top_k=args.num_experts,
            expert_lora_r=args.moe_r,
            expert_lora_alpha=args.moe_alpha,
            expert_lora_dropout=args.lora_dropout,
            target_modules=LORA_TARGET_MODULES,
            moe_modules=[],  # empty = no MoE modules
            standard_lora_r=args.standard_r,
            standard_lora_alpha=args.standard_alpha,
            router_hidden=args.router_hidden,
            balance_weight=0.0,  # no balance loss for baseline
        )

    # ── Wrap model ──
    print("Building MoEVLMWrapper...", flush=True)
    moe_wrapper = MoEVLMWrapper(
        base_model=model,
        moe_config=moe_config,
        tokenizer=processor.tokenizer,
    )

    # Enable input require grads (needed for gradient checkpointing with LoRA)
    moe_wrapper.enable_input_require_grads()

    # Print parameter counts
    trainable = moe_wrapper.num_trainable_parameters()
    total = moe_wrapper.num_total_parameters()
    print(f"Trainable params: {trainable:,} ({trainable/1e6:.1f}M)")
    print(f"Total params:     {total:,}")
    print(f"MoE modules:      {len(moe_wrapper._moe_linear_modules)}")
    print(f"Standard modules:  {len(moe_wrapper._std_linear_modules)}")
    print()

    # ── Load datasets ──
    print("Loading datasets...", flush=True)
    train_dataset = GUI360BindDataset(
        args.train_data, processor,
        max_samples=args.max_train_samples,
        max_length=args.max_length)
    val_dataset = GUI360BindDataset(
        args.val_data, processor,
        max_samples=100,
        max_length=args.max_length)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ── Save experiment config ──
    config_log = {
        "moe_modules": args.moe_modules,
        "num_experts": args.num_experts,
        "moe_r": args.moe_r,
        "moe_alpha": args.moe_alpha,
        "standard_r": args.standard_r,
        "standard_alpha": args.standard_alpha,
        "balance_weight": moe_config.balance_weight,
        "trainable_params": trainable,
        "n_moe_modules": len(moe_wrapper._moe_linear_modules),
        "n_std_modules": len(moe_wrapper._std_linear_modules),
        "learning_rate": args.learning_rate,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_epochs": args.num_epochs,
        "train_samples": len(train_dataset),
    }
    with open(os.path.join(args.output_dir, "experiment_config.json"), "w") as f:
        json.dump(config_log, f, indent=2)

    # ── Training arguments ──
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
        report_to="wandb",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
    )

    # ── Callbacks ──
    moe_checkpoint_cb = MoECheckpointCallback(moe_wrapper)
    metrics_cb = SelectiveMoEMetricsCallback(output_dir=args.output_dir)

    # ── Trainer ──
    # Pass the wrapper as `model` so Trainer can handle DDP wrapping
    trainer = SelectiveMoETrainer(
        moe_wrapper=moe_wrapper,
        model=moe_wrapper,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=bind_collate_fn,
        callbacks=[moe_checkpoint_cb, metrics_cb],
    )

    # ── Train ──
    print("Starting training...", flush=True)
    trainer.train()

    # ── Save final ──
    moe_wrapper.save_selective_checkpoint(
        os.path.join(args.output_dir, "final", "moe_checkpoint"))
    print(f"\nTraining complete. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
