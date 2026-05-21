#!/usr/bin/env python3
"""V12 Cooperative SFT: Soft-Routed Cooperative LoRA with HF Trainer.

Phase 1 of V12: SFT warm-start using thought-augmented data.
Uses SoftCooperativeVLMWrapper — no token masks, no side-channel comm.
Routing is purely input-dependent via learned per-layer routing vectors.

Usage:
  torchrun --nproc_per_node=4 v12_general/train_cooperative_sft.py \
      --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
      --train_data datasets/cooperative_thought_ac/ac_train_multiturn.jsonl \
      --val_data datasets/cooperative_thought_ac/ac_val_multiturn.jsonl \
      --output_dir checkpoints/v12_sft
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

from v12_general.soft_cooperative_wrapper import SoftCooperativeVLMWrapper


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════

class ThoughtAugmentedDataset(Dataset):
    """Loads thought-augmented JSONL for cooperative LoRA SFT.

    Supports both single-turn and multi-turn formats.
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
        for attempt in range(5):
            try:
                return self._process_item(idx)
            except Exception as e:
                if attempt == 0:
                    print(f"[Dataset] __getitem__ failed idx={idx}: {e}", flush=True)
                idx = (idx + 1) % len(self.data)
        return None

    # Token IDs for assistant span detection (Qwen2.5-VL)
    IM_START = 151644
    IM_END = 151645
    ASSISTANT_HEADER = [151644, 77091, 198]  # <|im_start|>assistant\n

    def _process_item(self, idx):
        item = self.data[idx]
        convs = item["conversations"]
        is_multiturn = convs[0].get("from") == "system"
        if is_multiturn:
            return self._process_multiturn(item)
        else:
            return self._process_singleturn(item)

    def _process_singleturn(self, item):
        convs = item["conversations"]
        images = item.get("images", [])

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

        full_text = self.processor.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False)
        prompt_text = self.processor.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True)

        image = None
        if images and os.path.exists(images[0]):
            image = Image.open(images[0]).convert("RGB")

        proc_kwargs = dict(text=[full_text], return_tensors="pt", padding=False,
                           max_length=self.max_length, truncation=True)
        prompt_kwargs = dict(text=[prompt_text], return_tensors="pt", padding=False,
                             max_length=self.max_length, truncation=True)
        if image is not None:
            proc_kwargs["images"] = [image]
            prompt_kwargs["images"] = [image]

        full_inputs = self.processor(**proc_kwargs)
        prompt_inputs = self.processor(**prompt_kwargs)

        input_ids = full_inputs["input_ids"].squeeze(0)
        attention_mask = full_inputs["attention_mask"].squeeze(0)
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if "pixel_values" in full_inputs:
            result["pixel_values"] = full_inputs["pixel_values"].squeeze(0)
        if "image_grid_thw" in full_inputs:
            result["image_grid_thw"] = self._ensure_2d_grid_thw(
                full_inputs["image_grid_thw"])
        return result

    def _process_multiturn(self, item):
        convs = item["conversations"]
        image_paths = item.get("images", [])

        messages = []
        img_idx = 0
        for conv in convs:
            role = conv["from"]
            text = conv["value"]
            if role == "system":
                messages.append({"role": "system",
                                 "content": [{"type": "text", "text": text}]})
            elif role in ("human", "user"):
                content = []
                if "<image>" in text:
                    text_clean = text.replace("<image>\n", "").replace("<image>", "").strip()
                    if img_idx < len(image_paths):
                        content.append({"type": "image", "image": image_paths[img_idx]})
                        img_idx += 1
                    content.append({"type": "text", "text": text_clean})
                else:
                    content.append({"type": "text", "text": text})
                messages.append({"role": "user", "content": content})
            elif role in ("assistant", "gpt"):
                messages.append({"role": "assistant",
                                 "content": [{"type": "text", "text": text}]})

        pil_images = []
        for p in image_paths[:img_idx]:
            if os.path.exists(p):
                pil_images.append(Image.open(p).convert("RGB"))

        num_turns = len([m for m in messages if m["role"] == "user"])
        for drop in range(num_turns):
            if drop > 0:
                trimmed = [messages[0]]
                trimmed.extend(messages[1 + drop * 2:])
                trimmed_images = pil_images[drop:]
            else:
                trimmed = messages
                trimmed_images = pil_images

            text = self.processor.apply_chat_template(
                trimmed, tokenize=False, add_generation_prompt=False)

            proc_kwargs = dict(text=[text], return_tensors="pt", padding=False)
            if trimmed_images:
                proc_kwargs["images"] = trimmed_images

            full_inputs = self.processor(**proc_kwargs)
            input_ids = full_inputs["input_ids"].squeeze(0)

            if input_ids.shape[0] <= self.max_length:
                break
        else:
            proc_kwargs["max_length"] = self.max_length
            proc_kwargs["truncation"] = True
            full_inputs = self.processor(**proc_kwargs)
            input_ids = full_inputs["input_ids"].squeeze(0)

        attention_mask = full_inputs["attention_mask"].squeeze(0)
        labels = self._mask_non_assistant(input_ids)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if "pixel_values" in full_inputs:
            result["pixel_values"] = full_inputs["pixel_values"].squeeze(0)
        if "image_grid_thw" in full_inputs:
            result["image_grid_thw"] = self._ensure_2d_grid_thw(
                full_inputs["image_grid_thw"])
        return result

    @staticmethod
    def _ensure_2d_grid_thw(thw):
        if thw.dim() == 3:
            thw = thw.squeeze(0)
        if thw.dim() == 1:
            thw = thw.unsqueeze(0)
        return thw

    def _mask_non_assistant(self, input_ids):
        ids = input_ids.tolist()
        labels = torch.full_like(input_ids, -100)
        header = self.ASSISTANT_HEADER
        i = 0
        while i < len(ids) - len(header) + 1:
            if ids[i:i + len(header)] == header:
                resp_start = i + len(header)
                resp_end = resp_start
                while resp_end < len(ids) and ids[resp_end] != self.IM_END:
                    resp_end += 1
                if resp_end < len(ids):
                    resp_end += 1
                labels[resp_start:resp_end] = input_ids[resp_start:resp_end]
                i = resp_end
            else:
                i += 1
        return labels


# ═══════════════════════════════════════════════════════════════════════
# Collate
# ═══════════════════════════════════════════════════════════════════════

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        dummy_ids = torch.zeros(1, 1, dtype=torch.long)
        return {
            "input_ids": dummy_ids,
            "attention_mask": torch.zeros(1, 1, dtype=torch.long),
            "labels": torch.full((1, 1), -100, dtype=torch.long),
        }

    max_len = max(b["input_ids"].shape[0] for b in batch)

    input_ids, attention_mask, labels = [], [], []
    pixel_values_list, image_grid_thw_list = [], []

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
        result["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)
    return result


# ═══════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════

class V12SFTTrainer(Trainer):
    """HF Trainer for V12 soft cooperative LoRA SFT.

    Loss = CE on assistant tokens + balance_weight * balance_loss.
    """

    def __init__(self, cooperative_model: SoftCooperativeVLMWrapper,
                 route_lr_multiplier: float = 100.0, tokenizer=None, **kwargs):
        super().__init__(model=cooperative_model, **kwargs)
        self.cooperative_model = cooperative_model
        self._tokenizer = tokenizer
        self.route_lr_multiplier = route_lr_multiplier
        self._balance_loss_sum = 0.0
        self._balance_loss_count = 0
        self._routing_w_sum = 0.0
        self._routing_w_count = 0

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        route_params, decay_params, no_decay_params = [], [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "route_weights" in name:
                route_params.append(param)
            elif param.dim() == 1 or name.endswith(".bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        base_lr = self.args.learning_rate
        base_wd = self.args.weight_decay

        param_groups = []
        if decay_params:
            param_groups.append({
                "params": decay_params, "lr": base_lr,
                "weight_decay": base_wd,
            })
        if no_decay_params:
            param_groups.append({
                "params": no_decay_params, "lr": base_lr,
                "weight_decay": 0.0,
            })
        if route_params:
            param_groups.append({
                "params": route_params,
                "lr": base_lr * self.route_lr_multiplier,
                "weight_decay": 0.0,
            })

        self.optimizer = torch.optim.AdamW(param_groups)

        rank = int(os.environ.get("LOCAL_RANK", 0))
        if rank == 0:
            print(f"[V12SFTTrainer] Optimizer: base_lr={base_lr} "
                  f"route_lr={base_lr * self.route_lr_multiplier} "
                  f"decay={len(decay_params)} no_decay={len(no_decay_params)} "
                  f"route={len(route_params)}")

        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)

        outputs = model(**inputs)
        logits = outputs.logits

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        else:
            ce_loss = torch.tensor(0.0, device=logits.device)

        # Balance loss
        bal_loss, mean_w = self.cooperative_model.compute_balance_loss()
        loss = ce_loss + self.cooperative_model.balance_weight * bal_loss

        self._balance_loss_sum += bal_loss.item()
        self._balance_loss_count += 1
        self._routing_w_sum += mean_w
        self._routing_w_count += 1

        return (loss, outputs) if return_outputs else loss


class LoggingCallback(TrainerCallback):
    """Log routing info periodically."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        model = kwargs.get("model")
        if model is None or not hasattr(model, "cooperative_model"):
            return

        trainer = kwargs.get("trainer")
        if trainer is None:
            return

        if trainer._balance_loss_count > 0:
            avg_bal = trainer._balance_loss_sum / trainer._balance_loss_count
            avg_w = trainer._routing_w_sum / trainer._routing_w_count
            if logs:
                logs["balance_loss"] = avg_bal
                logs["routing_w"] = avg_w
            trainer._balance_loss_sum = 0.0
            trainer._balance_loss_count = 0
            trainer._routing_w_sum = 0.0
            trainer._routing_w_count = 0


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="V12 Cooperative SFT")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default="")
    parser.add_argument("--resume_from", type=str, default="",
                        help="Path to cooperative checkpoint to resume from")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"])

    # V12 specific
    parser.add_argument("--balance_weight", type=float, default=0.01)
    parser.add_argument("--route_lr_multiplier", type=float, default=100.0)

    # Training
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=float, default=4.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=16384)
    parser.add_argument("--image_max_pixels", type=int, default=602112)
    parser.add_argument("--max_samples", type=int, default=0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    rank = int(os.environ.get("LOCAL_RANK", 0))

    # ── Load model ──
    if rank == 0:
        print(f"Loading base model: {args.model_path}")

    processor = AutoProcessor.from_pretrained(args.model_path)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if args.image_max_pixels > 0:
        processor.image_processor.max_pixels = args.image_max_pixels

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # ── Wrap with SoftCooperativeVLMWrapper ──
    model = SoftCooperativeVLMWrapper(
        base_model=base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(args.target_modules),
        balance_weight=args.balance_weight,
    )

    # Load from checkpoint if resuming
    if args.resume_from:
        if rank == 0:
            print(f"Loading cooperative weights from: {args.resume_from}")
        model.load_cooperative(args.resume_from)

    # Enable gradient checkpointing
    model.base_model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Ensure LoRA + route params require grad
    for name, param in model.named_parameters():
        if "lora_" in name or "route_weights" in name:
            param.requires_grad = True

    if rank == 0:
        counts = model.count_trainable_params()
        print(f"Trainable params: {counts['total']:,} "
              f"(lora={counts['lora']:,}, route={counts['route_weights']:,})")

    # ── Datasets ──
    train_dataset = ThoughtAugmentedDataset(
        args.train_data, processor,
        max_samples=args.max_samples, max_length=args.max_length,
    )
    val_dataset = None
    if args.val_data and os.path.exists(args.val_data):
        val_dataset = ThoughtAugmentedDataset(
            args.val_data, processor, max_length=args.max_length,
        )

    if rank == 0:
        print(f"Train: {len(train_dataset)} samples")
        if val_dataset:
            print(f"Val: {len(val_dataset)} samples")

    # ── Training args ──
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=args.eval_steps if val_dataset else None,
        bf16=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to="none",
        save_total_limit=3,
    )

    # ── Trainer ──
    trainer = V12SFTTrainer(
        cooperative_model=model,
        route_lr_multiplier=args.route_lr_multiplier,
        tokenizer=processor.tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=[LoggingCallback()],
    )

    # ── Train ──
    trainer.train()

    # ── Save final cooperative weights ──
    if rank == 0:
        final_dir = os.path.join(args.output_dir, "cooperative_final")
        model.save_cooperative(final_dir)
        print(f"Saved cooperative weights to: {final_dir}")


if __name__ == "__main__":
    main()
