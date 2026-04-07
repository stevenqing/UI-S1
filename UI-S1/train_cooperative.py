#!/usr/bin/env python3
"""
Token-Level Cooperative LoRA Training with Thought-Augmented Data.

Image tokens route through LoRA_V (visual binding), text/action tokens through
LoRA_A (action). Attention naturally bridges: q(LoRA_A) @ k(LoRA_V).

Primary binding signal: thought CE loss — the model learns to generate
<thought>visual_desc</thought> before <tool_call>, forcing LoRA_V to encode
visual semantics that LoRA_A can attend to for thought generation.

Optional L_bind (contrastive) can be added as a boost via --bind_weight > 0.

Usage:
  torchrun --nproc_per_node=4 train_cooperative.py \
      --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
      --train_data datasets/cooperative_thought/gui360_train_thought.jsonl \
      --val_data datasets/cooperative_thought/gui360_val_thought.jsonl \
      --output_dir checkpoints/cooperative_thought_v1 \
      --lora_r 16 --bind_weight 0.0
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from verl.models.cooperative.cooperative_wrapper import CooperativeVLMWrapper


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════

class ThoughtAugmentedDataset(Dataset):
    """Loads thought-augmented JSONL data for cooperative LoRA training.

    Each line is:
    {
      "conversations": [{"from":"human","value":"..."}, {"from":"assistant","value":"<thought>...</thought>\n<tool_call>..."}],
      "images": ["path/to/img.png"],
      "has_thought": true/false,
      "gt_coords": [x, y],  # optional, for L_bind
    }
    """

    def __init__(self, jsonl_path: str, processor, max_samples: int = 0,
                 max_length: int = 4096):
        self.data = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))

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
        gt_coord = item.get("gt_coords")

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
            orig_size = None
            if images and os.path.exists(images[0]):
                image = Image.open(images[0]).convert("RGB")
                orig_size = image.size  # (width, height)

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
                "gt_coord": gt_coord,
                "orig_size": orig_size,
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

def collate_fn(batch):
    """Collate with padding, None filtering, and gt_coord/orig_size passthrough."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids = []
    attention_mask = []
    labels = []
    pixel_values_list = []
    image_grid_thw_list = []
    gt_coords = []
    orig_sizes = []

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

        gt_coords.append(b.get("gt_coord"))
        orig_sizes.append(b.get("orig_size"))

    result = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
        "gt_coords": gt_coords,
        "orig_sizes": orig_sizes,
    }

    if pixel_values_list:
        result["pixel_values"] = torch.cat(pixel_values_list, dim=0)
    if image_grid_thw_list:
        result["image_grid_thw"] = torch.stack(image_grid_thw_list)

    return result


# ═══════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════

class CooperativeTrainer(Trainer):
    """HF Trainer for cooperative LoRA.

    Loss = L_CE (thought + action CE) + λ·L_bind (optional contrastive).
    When bind_weight=0, this is pure thought-augmented SFT with cooperative routing.
    """

    def __init__(self, cooperative_model: CooperativeVLMWrapper, tokenizer=None,
                 **kwargs):
        super().__init__(model=cooperative_model, **kwargs)
        self.cooperative_model = cooperative_model
        self._tokenizer = tokenizer

        # Running averages for logging
        self._act_loss_sum = 0.0
        self._act_loss_count = 0
        self._bind_loss_sum = 0.0
        self._bind_loss_count = 0
        self._bind_sample_count = 0
        self._target_sim_sum = 0.0
        self._nontarget_sim_sum = 0.0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        gt_coords = inputs.pop("gt_coords", None)
        orig_sizes = inputs.pop("orig_sizes", None)

        loss, diagnostics = model(
            gt_coords=gt_coords,
            orig_sizes=orig_sizes,
            **inputs,
        )

        # Track metrics
        self._act_loss_sum += diagnostics["L_act"].item()
        self._act_loss_count += 1
        if diagnostics["bind_samples"] > 0:
            self._bind_loss_sum += diagnostics["L_bind"].item()
            self._bind_loss_count += 1
            self._target_sim_sum += diagnostics.get("target_sim", 0)
            self._nontarget_sim_sum += diagnostics.get("nontarget_sim", 0)
            self._bind_sample_count += diagnostics["bind_samples"]

        return (loss, diagnostics) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if self._act_loss_count > 0:
            logs["ce_loss"] = round(self._act_loss_sum / self._act_loss_count, 6)
        if self._bind_loss_count > 0:
            logs["bind_loss"] = round(self._bind_loss_sum / self._bind_loss_count, 6)
        if self._bind_sample_count > 0:
            logs["target_sim"] = round(
                self._target_sim_sum / self._bind_sample_count, 6)
            logs["nontarget_sim"] = round(
                self._nontarget_sim_sum / self._bind_sample_count, 6)

        # Log gate values for cooperative communication (v6)
        if self.cooperative_model.cooperative_comm:
            gates_av, gates_va = [], []
            for m in self.cooperative_model.coop_modules:
                if hasattr(m, 'gate_av'):
                    gates_av.append(torch.sigmoid(m.gate_av).item())
                    gates_va.append(torch.sigmoid(m.gate_va).item())
            if gates_av:
                logs["gate_av_mean"] = round(sum(gates_av) / len(gates_av), 6)
                logs["gate_va_mean"] = round(sum(gates_va) / len(gates_va), 6)
                logs["gate_av_max"] = round(max(gates_av), 6)
                logs["gate_va_max"] = round(max(gates_va), 6)

        self._act_loss_sum = 0.0
        self._act_loss_count = 0
        self._bind_loss_sum = 0.0
        self._bind_loss_count = 0
        self._target_sim_sum = 0.0
        self._nontarget_sim_sum = 0.0
        self._bind_sample_count = 0

        super().log(logs, *args, **kwargs)


class CooperativeSaveCallback(TrainerCallback):
    """Save cooperative checkpoint (lora_v.pt + lora_a.pt) at each save step.
    Also saves a persistent copy at each epoch boundary."""

    def _get_model(self, kwargs):
        model = kwargs.get("model")
        if model is None:
            return None
        if hasattr(model, "module"):
            model = model.module
        return model if isinstance(model, CooperativeVLMWrapper) else None

    def on_save(self, args, state, control, **kwargs):
        model = self._get_model(kwargs)
        if model is None:
            return
        ckpt_dir = os.path.join(
            args.output_dir,
            f"checkpoint-{state.global_step}",
            "cooperative",
        )
        model.save_cooperative_checkpoint(ckpt_dir)

    def on_epoch_end(self, args, state, control, **kwargs):
        """Save a persistent epoch checkpoint that won't be auto-deleted."""
        model = self._get_model(kwargs)
        if model is None:
            return
        epoch = int(round(state.epoch))
        epoch_dir = os.path.join(args.output_dir, f"epoch-{epoch}")
        if state.is_world_process_zero:
            print(f"Saving epoch {epoch} checkpoint to {epoch_dir}")
        model.save_cooperative_checkpoint(epoch_dir)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Token-Level Cooperative LoRA Training (Thought-Augmented)")
    parser.add_argument("--model_path",
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/checkpoints/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--train_data",
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/cooperative_thought/gui360_train_thought.jsonl")
    parser.add_argument("--val_data",
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/cooperative_thought/gui360_val_thought.jsonl")
    parser.add_argument("--output_dir", required=True)
    # LoRA config
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--num_agents", type=int, default=2, choices=[2, 3],
                        help="Number of cooperative agents: 2 (V,A) or 3 (V,T,A)")
    parser.add_argument("--soft_routing", action="store_true",
                        help="Use learned soft routing instead of hard torch.where")
    parser.add_argument("--init_sep", type=float, default=0.0,
                        help="Initial sep value (0=shared, 2=near-separated)")
    parser.add_argument("--cooperative_comm", action="store_true",
                        help="Enable per-layer cooperative communication (v6)")
    parser.add_argument("--gate_init", type=float, default=-3.0,
                        help="Initial gate logit (sigmoid(-3)~0.05)")
    # Binding loss (optional boost, default off — thought CE is primary signal)
    parser.add_argument("--bind_weight", type=float, default=0.0)
    parser.add_argument("--bind_layer", type=int, default=27)
    parser.add_argument("--bind_temperature", type=float, default=0.1)
    # Data
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=4096)
    # Training
    parser.add_argument("--num_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1.5e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--resume_coop_checkpoint", type=str, default=None,
                        help="Path to cooperative checkpoint dir to resume from (loads lora_v.pt + lora_a.pt)")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load base model ──
    print(f"Loading base model from {args.model_path}...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # ── Wrap with cooperative LoRA ──
    print(f"Wrapping with cooperative LoRA (r={args.lora_r}, "
          f"targets={args.target_modules}, num_agents={args.num_agents}, "
          f"soft_routing={args.soft_routing}, init_sep={args.init_sep}, "
          f"cooperative_comm={args.cooperative_comm}, gate_init={args.gate_init})...")
    model = CooperativeVLMWrapper(
        base_model=base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bind_weight=args.bind_weight,
        bind_layer=args.bind_layer,
        bind_temperature=args.bind_temperature,
        num_agents=args.num_agents,
        soft_routing=args.soft_routing,
        init_sep=args.init_sep,
        cooperative_comm=args.cooperative_comm,
        gate_init=args.gate_init,
    )

    # ── Resume from cooperative checkpoint (if provided) ──
    if args.resume_coop_checkpoint:
        print(f"Loading cooperative checkpoint from {args.resume_coop_checkpoint}...")
        model.load_cooperative_checkpoint(args.resume_coop_checkpoint)
        print("Cooperative weights loaded successfully.")

    trainable = model.get_trainable_param_count()
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    # ── Load processor ──
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True)

    # ── Load datasets ──
    print(f"Loading training data from {args.train_data}...")
    train_dataset = ThoughtAugmentedDataset(
        args.train_data, processor,
        max_samples=args.max_train_samples,
        max_length=args.max_length,
    )
    print(f"Loading validation data from {args.val_data}...")
    val_dataset = ThoughtAugmentedDataset(
        args.val_data, processor,
        max_samples=500,
        max_length=args.max_length,
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ── Save experiment config ──
    config_log = {
        "architecture": "cooperative_lora_thought",
        "num_agents": args.num_agents,
        "soft_routing": args.soft_routing,
        "init_sep": args.init_sep,
        "cooperative_comm": args.cooperative_comm,
        "gate_init": args.gate_init,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules,
        "bind_weight": args.bind_weight,
        "bind_layer": args.bind_layer,
        "bind_temperature": args.bind_temperature,
        "trainable_params": trainable,
        "learning_rate": args.learning_rate,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_epochs": args.num_epochs,
        "max_length": args.max_length,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
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
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
    )

    # ── Trainer ──
    trainer = CooperativeTrainer(
        cooperative_model=model,
        tokenizer=processor.tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[CooperativeSaveCallback()],
    )

    # ── Train ──
    print("Starting training...")
    print(f"  Bind weight: {args.bind_weight} "
          f"({'thought CE only' if args.bind_weight == 0 else 'thought CE + L_bind'})")

    # Resume from HF Trainer checkpoint (optimizer/scheduler state)
    resume_ckpt = None
    if args.resume_coop_checkpoint:
        # Check if the parent dir has a HF trainer checkpoint
        parent = os.path.dirname(args.resume_coop_checkpoint.rstrip("/"))
        if os.path.exists(os.path.join(parent, "trainer_state.json")):
            resume_ckpt = parent
            print(f"Resuming trainer state from {resume_ckpt}")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ── Save final ──
    model.save_cooperative_checkpoint(os.path.join(args.output_dir, "final"))
    print(f"Training complete. Saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
