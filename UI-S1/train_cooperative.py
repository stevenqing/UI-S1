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
from peft import LoraConfig, get_peft_model


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
        # Retry with different indices on failure (e.g. truncation breaks image tokens)
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
        images = item.get("images", [])
        gt_coord = item.get("gt_coords")

        # Detect format: multi-turn (first msg is system) vs legacy single-turn
        is_multiturn = convs[0].get("from") == "system"

        if is_multiturn:
            return self._process_multiturn(item)
        else:
            return self._process_singleturn(item)

    def _process_singleturn(self, item):
        """Legacy single-turn format: human + assistant, 1 image."""
        convs = item["conversations"]
        images = item.get("images", [])
        gt_coord = item.get("gt_coords")

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
        orig_size = None
        if images and os.path.exists(images[0]):
            image = Image.open(images[0]).convert("RGB")
            orig_size = image.size

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
            "gt_coord": gt_coord,
            "orig_size": orig_size,
        }
        if "pixel_values" in full_inputs:
            result["pixel_values"] = full_inputs["pixel_values"].squeeze(0)
        if "image_grid_thw" in full_inputs:
            result["image_grid_thw"] = self._ensure_2d_grid_thw(
                full_inputs["image_grid_thw"])
        return result

    def _process_multiturn(self, item):
        """Multi-turn trajectory format: system + N×(user+assistant), N images."""
        convs = item["conversations"]
        image_paths = item.get("images", [])

        # Build Qwen2.5-VL message format
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

        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)

        # Load all images
        pil_images = []
        for p in image_paths[:img_idx]:
            if os.path.exists(p):
                pil_images.append(Image.open(p).convert("RGB"))

        # Process without truncation first — truncation breaks image token
        # consistency in Qwen2.5-VL processor. If too long, retry with fewer
        # turns (drop oldest turns to keep recent actions).
        num_turns = len([m for m in messages if m["role"] == "user"])
        for drop in range(num_turns):
            if drop > 0:
                # Drop the first user+assistant pair (keep system message)
                trimmed = [messages[0]]  # system
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
            # Even single turn exceeds max_length — use truncation as last resort
            proc_kwargs["max_length"] = self.max_length
            proc_kwargs["truncation"] = True
            full_inputs = self.processor(**proc_kwargs)
            input_ids = full_inputs["input_ids"].squeeze(0)

        attention_mask = full_inputs["attention_mask"].squeeze(0)

        # Labels: only train on assistant response content
        labels = self._mask_non_assistant(input_ids)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "gt_coord": None,
            "orig_size": None,
        }
        if "pixel_values" in full_inputs:
            result["pixel_values"] = full_inputs["pixel_values"].squeeze(0)
        if "image_grid_thw" in full_inputs:
            result["image_grid_thw"] = self._ensure_2d_grid_thw(
                full_inputs["image_grid_thw"])
        return result

    @staticmethod
    def _ensure_2d_grid_thw(thw):
        """Ensure image_grid_thw is always 2D [num_images, 3].

        The processor may return [1, N, 3] (with batch dim) or [N, 3] or [3]
        (single image squeezed). The model's rot_pos_emb iterates rows as
        (t, h, w), so it must be [N, 3].
        """
        if thw.dim() == 3:
            thw = thw.squeeze(0)  # [1, N, 3] -> [N, 3]
        if thw.dim() == 1:
            thw = thw.unsqueeze(0)  # [3] -> [1, 3]
        return thw

    def _mask_non_assistant(self, input_ids):
        """Create labels that only keep assistant response tokens.

        Finds all <|im_start|>assistant\\n spans and unmasks content up to <|im_end|>.
        """
        ids = input_ids.tolist()
        labels = torch.full_like(input_ids, -100)
        header = self.ASSISTANT_HEADER
        i = 0
        while i < len(ids) - len(header) + 1:
            if ids[i:i + len(header)] == header:
                # Found assistant header — content starts after it
                resp_start = i + len(header)
                resp_end = resp_start
                while resp_end < len(ids) and ids[resp_end] != self.IM_END:
                    resp_end += 1
                if resp_end < len(ids):
                    resp_end += 1  # include <|im_end|> token
                labels[resp_start:resp_end] = input_ids[resp_start:resp_end]
                i = resp_end
            else:
                i += 1
        return labels


# ═══════════════════════════════════════════════════════════════════════
# Collate
# ═══════════════════════════════════════════════════════════════════════

def collate_fn(batch):
    """Collate with padding, None filtering, and gt_coord/orig_size passthrough."""
    batch = [b for b in batch if b is not None]
    if not batch:
        # Return a dummy batch with labels all -100 so loss is 0
        # This prevents Trainer from crashing on None batches
        dummy_ids = torch.zeros(1, 1, dtype=torch.long)
        return {
            "input_ids": dummy_ids,
            "attention_mask": torch.zeros(1, 1, dtype=torch.long),
            "labels": torch.full((1, 1), -100, dtype=torch.long),
            "gt_coords": [None],
            "orig_sizes": [None],
        }

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
        result["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0)

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
                 gate_lr_multiplier: float = 1.0, gate_weight_decay: float = 0.0,
                 diversity_loss_weight: float = 0.0,
                 **kwargs):
        super().__init__(model=cooperative_model, **kwargs)
        self.cooperative_model = cooperative_model
        self._tokenizer = tokenizer
        self.gate_lr_multiplier = gate_lr_multiplier
        self.gate_weight_decay = gate_weight_decay
        self.diversity_loss_weight = diversity_loss_weight

        # Running averages for logging
        self._act_loss_sum = 0.0
        self._act_loss_count = 0
        self._bind_loss_sum = 0.0
        self._bind_loss_count = 0
        self._bind_sample_count = 0
        self._target_sim_sum = 0.0
        self._nontarget_sim_sum = 0.0
        self._div_loss_sum = 0.0
        self._div_loss_count = 0
        self._div_cos_sum = 0.0
        # v8: learned router tracking
        self._balance_loss_sum = 0.0
        self._balance_loss_count = 0
        self._router_w_sum = 0.0
        self._router_w_count = 0

    def create_optimizer(self):
        """Custom optimizer with separate param group for cooperative
        communication params (gate_av, gate_va, W_av, W_va).

        Three groups:
          1. Comm:    lr × gate_lr_multiplier, wd = gate_weight_decay
          2. Decay:   lr × 1,                  wd = args.weight_decay
          3. NoDecay: lr × 1,                  wd = 0  (biases, 1-d params)

        Motivation: in v6 thought training (job 3713229) we observed that
        gates barely moved (avg logit delta +0.0019 over 0.89 epoch, vs the
        ~+1.5 needed to reach sigmoid=0.5). Root cause is weak gradient flow
        in the sigmoid cold-start regime + default weight_decay=0.1 dragging
        gate logits toward 0. Putting comm params in a dedicated group with
        higher LR and no WD lets the cooperative mechanism actually learn
        end-to-end without hand-crafted layer priors.
        """
        if self.optimizer is not None:
            return self.optimizer

        comm_suffixes = {"gate_av", "gate_va", "W_av", "W_va"}
        comm_params, decay_params, no_decay_params = [], [], []
        comm_names, decay_names, no_decay_names = [], [], []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            suffix = name.split(".")[-1]
            if suffix in comm_suffixes:
                comm_params.append(param)
                comm_names.append(name)
            elif param.dim() == 1 or name.endswith(".bias"):
                no_decay_params.append(param)
                no_decay_names.append(name)
            else:
                decay_params.append(param)
                decay_names.append(name)

        base_lr = self.args.learning_rate
        base_wd = self.args.weight_decay

        param_groups = [
            {"params": decay_params, "lr": base_lr, "weight_decay": base_wd,
             "group_name": "lora_decay"},
            {"params": no_decay_params, "lr": base_lr, "weight_decay": 0.0,
             "group_name": "no_decay"},
            {"params": comm_params,
             "lr": base_lr * self.gate_lr_multiplier,
             "weight_decay": self.gate_weight_decay,
             "group_name": "comm"},
        ]
        # Filter out empty groups (e.g., no_decay_params may be empty)
        param_groups = [g for g in param_groups if len(g["params"]) > 0]

        if self.is_world_process_zero():
            print(f"[CooperativeTrainer] optimizer param groups:")
            print(f"  lora_decay:  {len(decay_params)} params, "
                  f"lr={base_lr}, wd={base_wd}")
            print(f"  no_decay:    {len(no_decay_params)} params, "
                  f"lr={base_lr}, wd=0")
            print(f"  comm:        {len(comm_params)} params, "
                  f"lr={base_lr * self.gate_lr_multiplier} "
                  f"(x{self.gate_lr_multiplier}), wd={self.gate_weight_decay}")

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args)
        # Drop 'lr' and 'weight_decay' from global kwargs since each group
        # already specifies them.
        optimizer_kwargs.pop("lr", None)
        optimizer_kwargs.pop("weight_decay", None)
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer

    def save_model(self, output_dir=None, _internal_call=False):
        """Skip full wrapper state_dict dump.

        CooperativeVLMWrapper is a plain nn.Module containing the frozen 7B
        base model + our tiny LoRA delta. HF Trainer's default _save writes the
        ENTIRE state_dict (~17 GB model.safetensors per checkpoint), which is:
          - Wasted disk (frozen base is identical to the source checkpoint)
          - A NCCL-timeout risk: rank 0 blocks all other ranks at a barrier
            while writing 17 GB to networked FS, which killed the v6 thought
            run at step 50 (job 3701676) — see train_v6_thought_3701676.err.

        Only cooperative LoRA params need to be persisted, and that is handled
        by CooperativeSaveCallback.on_save (writes lora_v.pt / lora_a.pt /
        lora_comm.pt / cooperative_config.json into <ckpt>/cooperative/).
        Optimizer, scheduler, RNG, and trainer_state.json are still saved
        normally by HF Trainer's _save_checkpoint, so resume keeps working.
        """
        if output_dir is None:
            output_dir = self.args.output_dir
        if self.args.should_save:
            os.makedirs(output_dir, exist_ok=True)
            # Persist args so HF resume detection finds a valid checkpoint dir.
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Skip HF Trainer's model-weight reload on resume.

        Because save_model() above writes no model weights, the checkpoint dir
        contains only optimizer.pt / scheduler.pt / rng_state_*.pth /
        trainer_state.json / training_args.bin and our cooperative/ subfolder.
        HF Trainer's default _load_from_checkpoint raises ValueError when no
        model.safetensors / pytorch_model.bin is found, blocking resume.

        The cooperative LoRA params are already loaded from the
        `cooperative/` subfolder by `model.load_cooperative_checkpoint()`
        in main() (called before trainer.train()), so this hook just needs
        to be a no-op for the model side. HF Trainer will still call
        `_load_optimizer_and_scheduler` and `_load_rng_state` separately
        from the same checkpoint dir, restoring all the resume state we
        actually need.
        """
        if self.is_world_process_zero():
            print(f"[CooperativeTrainer] Skipping model weight reload from "
                  f"{resume_from_checkpoint} (cooperative/ already loaded "
                  f"separately).")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        gt_coords = inputs.pop("gt_coords", None)
        orig_sizes = inputs.pop("orig_sizes", None)

        loss, diagnostics = model(
            gt_coords=gt_coords,
            orig_sizes=orig_sizes,
            **inputs,
        )

        # Track metrics (skip NaN from dummy batches with all -100 labels)
        act_val = diagnostics["L_act"].item()
        if not (act_val != act_val):  # NaN check
            self._act_loss_sum += act_val
            self._act_loss_count += 1
        if diagnostics["bind_samples"] > 0:
            self._bind_loss_sum += diagnostics["L_bind"].item()
            self._bind_loss_count += 1
            self._target_sim_sum += diagnostics.get("target_sim", 0)
            self._nontarget_sim_sum += diagnostics.get("nontarget_sim", 0)
            self._bind_sample_count += diagnostics["bind_samples"]

        # v8: learned router balance loss tracking
        if "L_balance" in diagnostics:
            self._balance_loss_sum += diagnostics["L_balance"].item()
            self._balance_loss_count += 1
            self._router_w_sum += float(diagnostics.get("mean_router_w", 0.0))
            self._router_w_count += 1

        # ── v7 diversity loss ────────────────────────────────────────
        # L_div = λ * mean_m cos(flat(lora_B_v_m), flat(lora_B_a_m))
        #
        # We penalize similarity on lora_B matrices (not lora_A). Rationale:
        #   - lora_A is Kaiming-initialized and barely moves during training
        #     (empirically: v6.4 ep4 cos(A_v, A_a) ≈ 0 ± 0.003, norms still
        #     at Kaiming init). Penalizing A in parameter space would push
        #     against random init noise.
        #   - lora_B is zero-initialized and carries the actual learned
        #     signal. delta(x) = B @ A @ x, so B's column span defines the
        #     adapter's output subspace. Two agents with identical B would
        #     mode-collapse even if A were different.
        # Cosine is bounded [-1, 1]; gradient has scale 1/‖B‖, so λ directly
        # scales the effective pressure.
        if self.diversity_loss_weight > 0:
            cos_list = []
            eps = 1e-8
            for m in self.cooperative_model.coop_modules:
                # In shared-B mode, only A matrices differ between agents —
                # penalize cos(A_v, A_a) instead. lora_A is Kaiming-initialized
                # but will move under training pressure.
                if getattr(m, "shared_B", False):
                    bv = m.lora_A_v.flatten()
                    ba = m.lora_A_a.flatten()
                else:
                    bv = m.lora_B_v.flatten()
                    ba = m.lora_B_a.flatten()
                # At step 0 both B are exactly 0 -> cos is undefined.
                # Guard with norm check; once B has any signal, cos is valid.
                if bv.norm() > eps and ba.norm() > eps:
                    cos_list.append(F.cosine_similarity(
                        bv.unsqueeze(0), ba.unsqueeze(0)).squeeze())
            if cos_list:
                mean_cos = torch.stack(cos_list).mean()
                L_div = mean_cos  # pressure: push toward -∞ (anti-correlation)
                loss = loss + self.diversity_loss_weight * L_div
                self._div_loss_sum += L_div.item()
                self._div_cos_sum += mean_cos.item()
                self._div_loss_count += 1

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
        if self._div_loss_count > 0:
            logs["div_loss"] = round(self._div_loss_sum / self._div_loss_count, 6)
            logs["div_cos"] = round(self._div_cos_sum / self._div_loss_count, 6)
        # v8: learned router stats
        if self._balance_loss_count > 0:
            logs["balance_loss"] = round(self._balance_loss_sum / self._balance_loss_count, 6)
        if self._router_w_count > 0:
            logs["router_w_mean"] = round(self._router_w_sum / self._router_w_count, 4)

        # Log gate values for cooperative communication (v6)
        if self.cooperative_model.cooperative_comm:
            gates_av, gates_va = [], []
            w_av_norms, w_va_norms = [], []
            gate_act = torch.tanh if self.cooperative_model.gate_type == "tanh" else torch.sigmoid
            for m in self.cooperative_model.coop_modules:
                if hasattr(m, 'gate_av'):
                    gates_av.append(gate_act(m.gate_av).item())
                    gates_va.append(gate_act(m.gate_va).item())
                    w_av_norms.append(m.W_av.detach().float().norm().item())
                    w_va_norms.append(m.W_va.detach().float().norm().item())
            if gates_av:
                import statistics
                # ── Distribution stats (compact, console-friendly) ──
                logs["gate_av_mean"] = round(statistics.mean(gates_av), 6)
                logs["gate_av_std"]  = round(statistics.pstdev(gates_av), 6)
                logs["gate_av_min"]  = round(min(gates_av), 6)
                logs["gate_av_max"]  = round(max(gates_av), 6)
                logs["gate_va_mean"] = round(statistics.mean(gates_va), 6)
                logs["gate_va_std"]  = round(statistics.pstdev(gates_va), 6)
                logs["gate_va_min"]  = round(min(gates_va), 6)
                logs["gate_va_max"]  = round(max(gates_va), 6)
                logs["W_av_norm_mean"] = round(statistics.mean(w_av_norms), 4)
                logs["W_va_norm_mean"] = round(statistics.mean(w_va_norms), 4)

                # ── Per-layer means (saved to JSONL on rank 0 only) ──
                # coop_modules is flat: layer 0 first N modules, layer 1 next N, etc.
                # where N = len(target_modules) per layer.
                if self.is_world_process_zero():
                    n_per_layer = len(self.cooperative_model.target_modules)
                    n_layers = len(gates_av) // n_per_layer
                    layer_av = [
                        statistics.mean(gates_av[i*n_per_layer:(i+1)*n_per_layer])
                        for i in range(n_layers)
                    ]
                    layer_va = [
                        statistics.mean(gates_va[i*n_per_layer:(i+1)*n_per_layer])
                        for i in range(n_layers)
                    ]
                    layer_w_av = [
                        statistics.mean(w_av_norms[i*n_per_layer:(i+1)*n_per_layer])
                        for i in range(n_layers)
                    ]
                    layer_w_va = [
                        statistics.mean(w_va_norms[i*n_per_layer:(i+1)*n_per_layer])
                        for i in range(n_layers)
                    ]
                    record = {
                        "step": self.state.global_step,
                        "epoch": round(self.state.epoch, 4) if self.state.epoch else 0,
                        "loss": logs.get("loss"),
                        "gate_av_per_layer": [round(v, 6) for v in layer_av],
                        "gate_va_per_layer": [round(v, 6) for v in layer_va],
                        "W_av_norm_per_layer": [round(v, 4) for v in layer_w_av],
                        "W_va_norm_per_layer": [round(v, 4) for v in layer_w_va],
                    }
                    history_path = os.path.join(self.args.output_dir,
                                                "gate_history.jsonl")
                    try:
                        with open(history_path, "a") as fp:
                            fp.write(json.dumps(record) + "\n")
                    except Exception:
                        pass

        self._act_loss_sum = 0.0
        self._act_loss_count = 0
        self._bind_loss_sum = 0.0
        self._bind_loss_count = 0
        self._target_sim_sum = 0.0
        self._nontarget_sim_sum = 0.0
        self._bind_sample_count = 0
        self._div_loss_sum = 0.0
        self._div_loss_count = 0
        self._div_cos_sum = 0.0
        # v8: reset router accumulators
        self._balance_loss_sum = 0.0
        self._balance_loss_count = 0
        self._router_w_sum = 0.0
        self._router_w_count = 0

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


class VanillaEpochSaveCallback(TrainerCallback):
    """Save PEFT adapter at each epoch boundary (mirrors CooperativeSaveCallback)."""

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        epoch = int(round(state.epoch))
        epoch_dir = os.path.join(args.output_dir, f"epoch-{epoch}")
        if state.is_world_process_zero:
            print(f"Saving vanilla LoRA epoch {epoch} to {epoch_dir}")
        # Unwrap DDP if needed
        m = model.module if hasattr(model, "module") else model
        m.save_pretrained(epoch_dir)


class VanillaSFTTrainer(Trainer):
    """Standard SFT Trainer using the same data pipeline as CooperativeTrainer.

    Strips cooperative-specific fields (gt_coords, orig_sizes) from inputs
    and uses the standard HF model forward (returns CausalLMOutput with .loss).
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs.pop("gt_coords", None)
        inputs.pop("orig_sizes", None)
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


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
    parser.add_argument("--vanilla_lora", action="store_true",
                        help="Use standard PEFT LoRA (single-agent) instead of cooperative. "
                             "Keeps the same data pipeline for fair comparison.")
    parser.add_argument("--num_agents", type=int, default=2, choices=[2, 3],
                        help="Number of cooperative agents: 2 (V,A) or 3 (V,T,A)")
    parser.add_argument("--soft_routing", action="store_true",
                        help="Use learned soft routing instead of hard torch.where")
    parser.add_argument("--init_sep", type=float, default=0.0,
                        help="Initial sep value (0=shared, 2=near-separated)")
    parser.add_argument("--cooperative_comm", action="store_true",
                        help="Enable per-layer cooperative communication (v6)")
    parser.add_argument("--gate_init", type=float, default=-3.0,
                        help="Initial gate logit (sigmoid(-3)~0.05; tanh(0)=0)")
    parser.add_argument("--gate_type", type=str, default="sigmoid",
                        choices=["sigmoid", "tanh"],
                        help="Activation for comm gates. v6/v6.1: sigmoid. "
                             "v6.2: tanh (bounded [-1,1], init-0 gradient=1.0 vs "
                             "sigmoid's 0.25; allows learning negative coupling).")
    parser.add_argument("--gate_lr_multiplier", type=float, default=1.0,
                        help="LR multiplier for communication params "
                             "(gate_av/gate_va/W_av/W_va). v6.1 uses ~10.")
    parser.add_argument("--gate_weight_decay", type=float, default=0.0,
                        help="Weight decay for communication params "
                             "(v6.1: 0 to avoid dragging gates toward 0).")
    # v7: routing mode and emergent differentiation
    parser.add_argument("--routing_mode", type=str, default="hard",
                        choices=["hard", "merge", "learned"],
                        help="v6 'hard' routes image->V / text->A; v7 'merge' "
                             "removes routing (delta=0.5*(delta_v+delta_a)); "
                             "v8 'learned' uses a per-layer nn.Linear(D,1) "
                             "router to compute per-token soft routing weights "
                             "from hidden states.")
    parser.add_argument("--diversity_loss_weight", type=float, default=0.0,
                        help="v7: weight on cosine-similarity diversity loss "
                             "between LoRA_V and LoRA_A. 0 disables.")
    parser.add_argument("--coord_routing", action="store_true",
                        help="Route coordinate/bbox digit tokens through LoRA_V "
                             "instead of LoRA_A. Gives LoRA_V direct CE loss "
                             "gradient on spatial tokens.")
    parser.add_argument("--coord_only_routing", action="store_true",
                        help="v10: ONLY route coordinate/bbox digit tokens to LoRA_V. "
                             "All other tokens (including image) go through LoRA_A. "
                             "Implies coord_routing behavior but disables image→V.")
    parser.add_argument("--coop_reasoning_alpha", type=float, default=0.0,
                        help="Cooperative reasoning α: assistant tokens get "
                             "α·V + (1-α)·A mixing. 0 disables (hard routing). "
                             "Recommended: 0.3")
    # v8: learned router options
    parser.add_argument("--balance_weight", type=float, default=0.0,
                        help="v8: load-balance loss weight for learned router. "
                             "Pushes mean routing weight per layer toward 0.5 "
                             "via binary entropy. Recommended: 0.01.")
    parser.add_argument("--router_warmstart_samples", type=int, default=0,
                        help="v8: number of training samples to use for warm-"
                             "starting routers from token-type hidden states. "
                             "0 disables (zero-init router, uniform 0.5 blend).")
    parser.add_argument("--shared_b", action="store_true",
                        help="v8+: use a single shared B matrix per module "
                             "with per-agent A matrices. Blend happens in "
                             "r-dim space (tiny) before the single B matmul, "
                             "saving ~14 GiB per forward on 7B @ seq=12288. "
                             "Params −42% on up/gate/down_proj.")
    # Binding loss (optional boost, default off — thought CE is primary signal)
    parser.add_argument("--bind_weight", type=float, default=0.0)
    parser.add_argument("--bind_layer", type=int, default=27)
    parser.add_argument("--bind_temperature", type=float, default=0.1)
    # Data
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--image_max_pixels", type=int, default=0,
                        help="Override processor max_pixels for images. "
                             "0 = use model default (~1003520). "
                             "Lower values reduce visual tokens per image, "
                             "e.g. 401408 (512*28*28) ≈ 506 tokens/image.")
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

    # ── Build model ──
    is_vanilla = args.vanilla_lora

    if is_vanilla:
        print(f"Building vanilla PEFT LoRA (r={args.lora_r}, "
              f"alpha={args.lora_alpha}, targets={args.target_modules})...")
        # Freeze vision tower and projector (same as cooperative)
        for name, param in base_model.named_parameters():
            if "visual" in name or "multi_modal_projector" in name:
                param.requires_grad = False
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
    else:
        print(f"Wrapping with cooperative LoRA (r={args.lora_r}, "
              f"targets={args.target_modules}, num_agents={args.num_agents}, "
              f"soft_routing={args.soft_routing}, init_sep={args.init_sep}, "
              f"cooperative_comm={args.cooperative_comm}, "
              f"gate_type={args.gate_type}, gate_init={args.gate_init}, "
              f"routing_mode={args.routing_mode}, "
              f"coord_routing={args.coord_routing}, "
              f"coord_only_routing={args.coord_only_routing}, "
              f"coop_reasoning_alpha={args.coop_reasoning_alpha}, "
              f"balance_weight={args.balance_weight}, "
              f"shared_B={args.shared_b})...")
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
            gate_type=args.gate_type,
            routing_mode=args.routing_mode,
            coord_routing=args.coord_routing,
            coord_only_routing=args.coord_only_routing,
            coop_reasoning_alpha=args.coop_reasoning_alpha,
            balance_weight=args.balance_weight,
            shared_B=args.shared_b,
        )

        # ── Resume from cooperative checkpoint (if provided) ──
        if args.resume_coop_checkpoint:
            print(f"Loading cooperative checkpoint from {args.resume_coop_checkpoint}...")
            model.load_cooperative_checkpoint(args.resume_coop_checkpoint)
            print("Cooperative weights loaded successfully.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    # ── Load processor ──
    proc_kwargs = dict(trust_remote_code=True)
    if args.image_max_pixels > 0:
        proc_kwargs["max_pixels"] = args.image_max_pixels
        proc_kwargs["min_pixels"] = min(args.image_max_pixels // 4, 200704)
        print(f"Overriding image resolution: max_pixels={args.image_max_pixels}, "
              f"min_pixels={proc_kwargs['min_pixels']}")
    processor = AutoProcessor.from_pretrained(args.model_path, **proc_kwargs)

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
        "architecture": "vanilla_peft_lora" if is_vanilla else "cooperative_lora_thought",
        "vanilla_lora": is_vanilla,
        "num_agents": 1 if is_vanilla else args.num_agents,
        "soft_routing": args.soft_routing,
        "init_sep": args.init_sep,
        "cooperative_comm": args.cooperative_comm,
        "gate_init": args.gate_init,
        "gate_type": args.gate_type,
        "gate_lr_multiplier": args.gate_lr_multiplier,
        "gate_weight_decay": args.gate_weight_decay,
        "routing_mode": args.routing_mode,
        "diversity_loss_weight": args.diversity_loss_weight,
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
        eval_strategy="no",
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False,
    )

    # ── Trainer ──
    if is_vanilla:
        trainer = VanillaSFTTrainer(
            model=model,
            tokenizer=processor.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collate_fn,
            callbacks=[VanillaEpochSaveCallback()],
        )
    else:
        trainer = CooperativeTrainer(
            cooperative_model=model,
            tokenizer=processor.tokenizer,
            gate_lr_multiplier=args.gate_lr_multiplier,
            gate_weight_decay=args.gate_weight_decay,
            diversity_loss_weight=args.diversity_loss_weight,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collate_fn,
            callbacks=[CooperativeSaveCallback()],
        )

    # ── Train ──
    print("Starting training...")
    if is_vanilla:
        print("  Mode: vanilla PEFT LoRA (fair baseline)")
    else:
        print(f"  Bind weight: {args.bind_weight} "
              f"({'thought CE only' if args.bind_weight == 0 else 'thought CE + L_bind'})")

    # Resume from HF Trainer checkpoint (optimizer/scheduler state)
    resume_ckpt = None
    if not is_vanilla and args.resume_coop_checkpoint:
        # Check if the parent dir has a HF trainer checkpoint
        parent = os.path.dirname(args.resume_coop_checkpoint.rstrip("/"))
        if os.path.exists(os.path.join(parent, "trainer_state.json")):
            resume_ckpt = parent
            print(f"Resuming trainer state from {resume_ckpt}")

    # ── v8: router warm-start (before DDP wrap happens inside trainer.train) ──
    if (not is_vanilla and args.routing_mode == "learned"
            and args.router_warmstart_samples > 0 and resume_ckpt is None):
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            print(f"[warmstart] Collecting {args.router_warmstart_samples} samples "
                  "from training data for router warm-start...")
            # Move model to GPU for the pass
            target_device = f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
            model.to(target_device)
            N = min(args.router_warmstart_samples, len(train_dataset))
            input_ids_list = []
            attention_mask_list = []
            pixel_values_list = []
            image_grid_thw_list = []
            labels_list = []
            collected = 0
            for i in range(len(train_dataset)):
                if collected >= N:
                    break
                sample = train_dataset[i]
                if sample is None:
                    continue
                input_ids_list.append(sample["input_ids"])
                attention_mask_list.append(sample["attention_mask"])
                labels_list.append(sample["labels"])
                pixel_values_list.append(sample.get("pixel_values"))
                image_grid_thw_list.append(sample.get("image_grid_thw"))
                collected += 1
            model.warmstart_routers_from_token_type(
                input_ids_list, attention_mask_list,
                pixel_values_list, image_grid_thw_list, labels_list,
            )
        else:
            print(f"[warmstart] rank {rank}: skipping (rank 0 handles warm-start; "
                  "DDP will broadcast from rank 0)")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ── Save final ──
    if is_vanilla:
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"Training complete. PEFT adapter saved to {args.output_dir}")
    else:
        model.save_cooperative_checkpoint(os.path.join(args.output_dir, "final"))
        print(f"Training complete. Saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
