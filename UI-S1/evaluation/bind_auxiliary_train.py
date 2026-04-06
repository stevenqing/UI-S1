#!/usr/bin/env python3
"""
Step 2α: Auxiliary Binding Loss Training (LoRA r=32)

Trains Qwen2.5-VL-7B-Instruct with L_total = L_act + λ·L_bind (λ=0.1).
The binding loss forces target image tokens (near GT click coordinate) to align
with task text tokens in hidden-state space, addressing orthogonal neglect.

Architecture:
  GUI360BindDataset → bind_collate_fn → BindAuxTrainer (custom compute_loss)

Usage:
  torchrun --nproc_per_node=4 evaluation/bind_auxiliary_train.py \
      --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
      --train_data train_GUI_360/llamafactory/data/gui360_train.json \
      --val_data train_GUI_360/llamafactory/data/gui360_val.json \
      --output_dir train_GUI_360/llamafactory/output/bind_aux_r32 \
      --lora_rank 32 --bind_weight 0.1 \
      --per_device_batch_size 2 --gradient_accumulation_steps 8
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
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset


# ── Constants ────────────────────────────────────────────────────────
VISION_START_ID = 151652
VISION_END_ID = 151653
IMAGE_PAD_ID = 151655

PATCH_SIZE = 14
SPATIAL_MERGE_SIZE = 2
TOKEN_PIXEL_SIZE = SPATIAL_MERGE_SIZE * PATCH_SIZE  # = 28

TARGET_BBOX_RADIUS = 56  # ±2 tokens around GT coordinate
BIND_LAYER = 27          # compute L_bind at final representation
TEMPERATURE = 0.1        # contrastive loss temperature

LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ═══════════════════════════════════════════════════════════════════════
# Utilities (from gradient_conflict_analysis.py)
# ═══════════════════════════════════════════════════════════════════════

def parse_tool_call(text):
    """Parse tool_call JSON from assistant response."""
    if not text:
        return None
    m = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return None


def get_image_token_positions(image_grid_thw):
    """
    Map image token indices to pixel bboxes in resized image space.

    image_grid_thw: (t, h, w) -- PRE-MERGE patch dimensions
    Returns: list of (x1, y1, x2, y2) in resized pixel coords, one per token
    """
    t, h, w = image_grid_thw
    token_h = h // SPATIAL_MERGE_SIZE
    token_w = w // SPATIAL_MERGE_SIZE

    positions = []
    for row in range(token_h):
        for col in range(token_w):
            y1 = row * TOKEN_PIXEL_SIZE
            x1 = col * TOKEN_PIXEL_SIZE
            positions.append((x1, y1, x1 + TOKEN_PIXEL_SIZE, y1 + TOKEN_PIXEL_SIZE))

    return positions, token_h, token_w


def find_overlapping_tokens(positions, gt_bbox, orig_size, resized_size):
    """
    Find image token indices overlapping with GT bounding box.

    gt_bbox: dict with left, top, right, bottom (in original image pixel coords)
    orig_size: (width, height) of original image
    resized_size: (width, height) of resized image
    Returns: list of token indices
    """
    orig_w, orig_h = orig_size
    resized_w, resized_h = resized_size

    scale_w = resized_w / orig_w
    scale_h = resized_h / orig_h

    bl = gt_bbox["left"] * scale_w
    bt = gt_bbox["top"] * scale_h
    br = gt_bbox["right"] * scale_w
    bb = gt_bbox["bottom"] * scale_h

    overlapping = []
    for i, (x1, y1, x2, y2) in enumerate(positions):
        if x2 > bl and x1 < br and y2 > bt and y1 < bb:
            overlapping.append(i)

    return overlapping


def identify_text_regions(input_ids, tokenizer):
    """Identify task text token indices using text-based matching."""
    ids = input_ids.squeeze().tolist()

    cum_len = 0
    token_char_starts = []
    for tid in ids:
        token_char_starts.append(cum_len)
        cum_len += len(tokenizer.decode([tid]))

    full_text = tokenizer.decode(ids)

    def find_token_pos(marker):
        pos = full_text.rfind(marker)
        if pos == -1:
            pos = full_text.lower().rfind(marker.lower())
        if pos == -1:
            return None
        for i in range(len(token_char_starts) - 1, -1, -1):
            if token_char_starts[i] <= pos:
                return i
        return None

    instr_pos = find_token_pos("instruction is:\n")
    hist_pos = find_token_pos("history of actions are:\n")
    act_pos = find_token_pos("actions supported are:\n")

    task_indices = []
    if instr_pos is not None:
        task_end = hist_pos if hist_pos is not None else (
            act_pos if act_pos is not None else len(ids))
        if hist_pos is not None:
            task_end = max(0, hist_pos - 2)
        task_start = max(0, instr_pos - 2)
        task_indices = list(range(task_start, task_end))

    return task_indices


def coord_to_bbox(coord, radius=TARGET_BBOX_RADIUS):
    """Synthesize a bounding box from a coordinate with given radius."""
    return {
        "left": coord[0] - radius,
        "top": coord[1] - radius,
        "right": coord[0] + radius,
        "bottom": coord[1] + radius,
    }


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════

class GUI360BindDataset(Dataset):
    """Loads ShareGPT-format GUI-360 data with GT coordinate extraction."""

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

        # Parse GT coordinate from assistant response
        assistant_text = convs[1]["value"]
        gt_action = parse_tool_call(assistant_text)
        gt_coord = None
        if gt_action is not None:
            gt_func = gt_action.get("function", "")
            args = gt_action.get("args", {})
            coord = args.get("coordinate", [])
            if gt_func == "click" and len(coord) == 2 and coord[0] is not None and coord[1] is not None:
                gt_coord = coord

        # Build Qwen2.5-VL message format
        user_text = convs[0]["value"]
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
                "gt_coord": gt_coord,       # None for non-click samples
                "orig_size": orig_size,      # None if no image
            }

            if "pixel_values" in full_inputs:
                result["pixel_values"] = full_inputs["pixel_values"].squeeze(0)
            if "image_grid_thw" in full_inputs:
                result["image_grid_thw"] = full_inputs["image_grid_thw"].squeeze(0)

            return result

        except Exception as e:
            return None


# ═══════════════════════════════════════════════════════════════════════
# Collate Function
# ═══════════════════════════════════════════════════════════════════════

def bind_collate_fn(batch):
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
        "gt_coords": gt_coords,      # list of coord or None per sample
        "orig_sizes": orig_sizes,     # list of (w,h) or None per sample
    }

    if pixel_values_list:
        result["pixel_values"] = torch.cat(pixel_values_list, dim=0)
    if image_grid_thw_list:
        result["image_grid_thw"] = torch.stack(image_grid_thw_list)

    return result


# ═══════════════════════════════════════════════════════════════════════
# BindAuxTrainer
# ═══════════════════════════════════════════════════════════════════════

class BindAuxTrainer(Trainer):
    """HF Trainer with auxiliary binding loss: L_total = L_act + λ·L_bind."""

    def __init__(self, bind_weight: float = 0.1, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.bind_weight = bind_weight
        self._tokenizer = tokenizer
        # Running averages for logging (accumulated across all forward passes
        # between logging events, then injected via log() override)
        self._bind_loss_sum = 0.0
        self._bind_loss_count = 0
        self._act_loss_sum = 0.0
        self._act_loss_count = 0
        self._target_sim_sum = 0.0
        self._nontarget_sim_sum = 0.0
        self._bind_sample_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract non-tensor fields before forwarding
        gt_coords = inputs.pop("gt_coords", None)
        orig_sizes = inputs.pop("orig_sizes", None)

        # Skip hidden states when bind_weight=0 (pure SFT baseline)
        need_hidden = self.bind_weight > 0 and gt_coords is not None

        outputs = model(
            **inputs,
            output_hidden_states=need_hidden,
            return_dict=True,
        )

        # L_act: standard cross-entropy loss on masked labels
        L_act = outputs.loss

        # Track L_act for separate logging
        self._act_loss_sum += L_act.detach().item()
        self._act_loss_count += 1

        # Compute L_bind per sample
        L_bind_samples = []
        batch_size = inputs["input_ids"].shape[0]

        if gt_coords is not None and outputs.hidden_states is not None:
            # hidden_states[0] = embedding, hidden_states[l+1] = after layer l
            # BIND_LAYER=27 → index 28
            hidden_states = outputs.hidden_states[BIND_LAYER + 1]  # (B, seq_len, D)

            image_grid_thw = inputs.get("image_grid_thw")

            for i in range(batch_size):
                gt_coord = gt_coords[i]
                orig_size = orig_sizes[i]

                # Skip non-click samples
                if gt_coord is None or orig_size is None:
                    continue

                # Skip samples without image grid info
                if image_grid_thw is None:
                    continue

                try:
                    L_bind_i, t_sim, nt_sim = self._compute_bind_loss_single(
                        hidden_states[i],
                        inputs["input_ids"][i],
                        image_grid_thw[i].tolist(),
                        gt_coord,
                        orig_size,
                    )
                    if L_bind_i is not None:
                        L_bind_samples.append(L_bind_i)
                        self._target_sim_sum += t_sim
                        self._nontarget_sim_sum += nt_sim
                        self._bind_sample_count += 1
                except Exception:
                    pass

        # Aggregate L_bind
        if L_bind_samples:
            L_bind = torch.stack(L_bind_samples).mean()
            L_total = L_act + self.bind_weight * L_bind

            self._bind_loss_sum += L_bind.detach().item()
            self._bind_loss_count += 1
        else:
            L_total = L_act

        return (L_total, outputs) if return_outputs else L_total

    def _compute_bind_loss_single(self, hs, input_ids, grid_thw, gt_coord, orig_size):
        """
        Compute L_bind for a single sample.

        hs: (seq_len, D) hidden states at BIND_LAYER
        input_ids: (seq_len,) token IDs
        grid_thw: [t, h, w] image grid dimensions
        gt_coord: [x, y] ground truth click coordinate
        orig_size: (width, height) original image size

        Returns: (L_bind, target_sim, nontarget_sim) or (None, None, None)
        """
        # Get image token spatial positions
        positions, token_h, token_w = get_image_token_positions(grid_thw)
        n_image_tokens = token_h * token_w

        resized_h = grid_thw[1] * PATCH_SIZE
        resized_w = grid_thw[2] * PATCH_SIZE
        resized_size = (resized_w, resized_h)

        # Find image token range in input_ids
        ids_list = input_ids.tolist()
        img_start = img_end = None
        for j, t in enumerate(ids_list):
            if t == VISION_START_ID and img_start is None:
                img_start = j
            if t == VISION_END_ID:
                img_end = j

        if img_start is None or img_end is None:
            return None, None, None

        img_token_start = img_start + 1
        img_token_end = img_end  # exclusive

        actual_n = img_token_end - img_token_start
        if actual_n != n_image_tokens:
            return None, None, None

        # Synthesize bbox from gt_coord
        gt_bbox = coord_to_bbox(gt_coord)

        # Find target tokens (overlapping with bbox)
        target_indices = find_overlapping_tokens(
            positions, gt_bbox, orig_size, resized_size)
        if len(target_indices) == 0:
            return None, None, None

        # Map to sequence positions
        target_seq = [img_token_start + idx for idx in target_indices]
        all_img_seq = list(range(img_token_start, img_token_end))
        target_set = set(target_seq)
        nontarget_seq = [p for p in all_img_seq if p not in target_set]

        if len(nontarget_seq) == 0:
            return None, None, None

        # Identify task text tokens
        task_indices = identify_text_regions(input_ids.unsqueeze(0), self._tokenizer)
        if len(task_indices) == 0:
            return None, None, None

        # Compute contrastive binding loss
        target_mean = hs[target_seq].mean(dim=0)
        nontarget_mean = hs[nontarget_seq].mean(dim=0)
        task_mean = hs[task_indices].mean(dim=0)

        target_sim = F.cosine_similarity(
            target_mean.unsqueeze(0), task_mean.unsqueeze(0))
        nontarget_sim = F.cosine_similarity(
            nontarget_mean.unsqueeze(0), task_mean.unsqueeze(0))

        # Contrastive loss with temperature
        logit_target = target_sim / TEMPERATURE
        logit_nontarget = nontarget_sim / TEMPERATURE
        L_bind = -torch.log(
            torch.exp(logit_target) /
            (torch.exp(logit_target) + torch.exp(logit_nontarget))
        )

        return L_bind, target_sim.detach().item(), nontarget_sim.detach().item()

    def log(self, logs, *args, **kwargs):
        """Override to inject bind metrics alongside standard Trainer metrics.

        This fires when the Trainer's own logging triggers (every logging_steps),
        ensuring bind metrics are averaged over the correct window of forward passes.
        """
        if self._act_loss_count > 0:
            logs["act_loss"] = round(self._act_loss_sum / self._act_loss_count, 6)
        if self._bind_loss_count > 0:
            logs["bind_loss"] = round(self._bind_loss_sum / self._bind_loss_count, 6)
        if self._bind_sample_count > 0:
            logs["target_sim"] = round(self._target_sim_sum / self._bind_sample_count, 6)
            logs["nontarget_sim"] = round(self._nontarget_sim_sum / self._bind_sample_count, 6)

        # Reset running averages
        self._bind_loss_sum = 0.0
        self._bind_loss_count = 0
        self._act_loss_sum = 0.0
        self._act_loss_count = 0
        self._target_sim_sum = 0.0
        self._nontarget_sim_sum = 0.0
        self._bind_sample_count = 0

        super().log(logs, *args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════
# Metrics Callback
# ═══════════════════════════════════════════════════════════════════════

class BindMetricsCallback(TrainerCallback):
    """Track and save bind metrics over training."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.step_metrics = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # Capture bind-related logs
        entry = {"step": state.global_step}
        for key in ("loss", "act_loss", "bind_loss", "target_sim",
                     "nontarget_sim", "learning_rate"):
            if key in logs:
                entry[key] = logs[key]
        if len(entry) > 1:
            self.step_metrics.append(entry)

    def on_train_end(self, args, state, control, **kwargs):
        out_path = os.path.join(self.output_dir, "bind_metrics.json")
        with open(out_path, "w") as f:
            json.dump(self.step_metrics, f, indent=2)
        print(f"Bind metrics saved to {out_path} ({len(self.step_metrics)} records)")

        # Print summary
        if self.step_metrics:
            act_losses = [m["act_loss"] for m in self.step_metrics if "act_loss" in m]
            bind_losses = [m["bind_loss"] for m in self.step_metrics if "bind_loss" in m]
            target_sims = [m["target_sim"] for m in self.step_metrics if "target_sim" in m]
            nontarget_sims = [m["nontarget_sim"] for m in self.step_metrics if "nontarget_sim" in m]

            print("\n── Bind Metrics Summary ──")
            if act_losses:
                print(f"  L_act: first={act_losses[0]:.4f}, last={act_losses[-1]:.4f}, "
                      f"mean={np.mean(act_losses):.4f}")
            if bind_losses:
                print(f"  L_bind: first={bind_losses[0]:.4f}, last={bind_losses[-1]:.4f}, "
                      f"mean={np.mean(bind_losses):.4f}")
            if target_sims:
                print(f"  target_sim: first={target_sims[0]:.4f}, last={target_sims[-1]:.4f}, "
                      f"mean={np.mean(target_sims):.4f}")
            if nontarget_sims:
                print(f"  nontarget_sim: first={nontarget_sims[0]:.4f}, last={nontarget_sims[-1]:.4f}, "
                      f"mean={np.mean(nontarget_sims):.4f}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Step 2α: Auxiliary Binding Loss Training")
    parser.add_argument("--model_path",
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/checkpoints/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--train_data",
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/llamafactory/data/gui360_train.json")
    parser.add_argument("--val_data",
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/llamafactory/data/gui360_val.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--bind_weight", type=float, default=0.1,
                        help="λ weight for L_bind auxiliary loss")
    parser.add_argument("--max_train_samples", type=int, default=0,
                        help="0 = use all data")
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

    print("=" * 80)
    print("Step 2α: Auxiliary Binding Loss Training")
    print("=" * 80)
    print(f"Model:       {args.model_path}")
    print(f"Train data:  {args.train_data}")
    print(f"Val data:    {args.val_data}")
    print(f"Output:      {args.output_dir}")
    print(f"LoRA:        r={args.lora_rank}, α={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Bind weight: λ={args.bind_weight}")
    print(f"Bind layer:  {BIND_LAYER}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"BBox radius: {TARGET_BBOX_RADIUS}")
    print(f"Batch:       {args.per_device_batch_size} × grad_accum={args.gradient_accumulation_steps}")
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

    # Apply LoRA
    print("Applying LoRA...", flush=True)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # Count trainable params
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")

    # Enable gradient checkpointing (use_reentrant=False is CRITICAL for
    # output_hidden_states gradients to flow through checkpointed layers)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    # ── Load processor ──
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True)

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

    # ── Save config ──
    lora_dict = lora_config.to_dict()
    for k, v in lora_dict.items():
        if isinstance(v, set):
            lora_dict[k] = sorted(list(v))
    config_log = {
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bind_weight": args.bind_weight,
        "bind_layer": BIND_LAYER,
        "temperature": TEMPERATURE,
        "bbox_radius": TARGET_BBOX_RADIUS,
        "total_trainable_params": total_params,
        "lora_config": lora_dict,
        "learning_rate": args.learning_rate,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_epochs": args.num_epochs,
        "max_length": args.max_length,
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
    bind_callback = BindMetricsCallback(output_dir=args.output_dir)

    # ── Trainer ──
    trainer = BindAuxTrainer(
        bind_weight=args.bind_weight,
        tokenizer=processor.tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=bind_collate_fn,
        callbacks=[bind_callback],
    )

    # ── Train ──
    print("Starting training...", flush=True)
    trainer.train()

    # ── Save final model ──
    trainer.save_model(os.path.join(args.output_dir, "final"))
    print(f"\nTraining complete. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
