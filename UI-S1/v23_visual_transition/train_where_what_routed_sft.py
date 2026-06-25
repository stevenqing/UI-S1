#!/usr/bin/env python3
"""Token-routed WHAT/WHERE SFT for GUI-360 cooperative LoRA.

This trainer keeps the target as the full <tool_call> while supervising the
cooperative router with token roles:

- WHAT tokens route to expert 1 (target route value 1)
- WHERE tokens route to expert 2 (target route value 0)

WHERE tokens are the location-key spans: coordinate, start_coordinate,
end_coordinate. Everything else in the assistant target is treated as WHAT.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from v15_gui_360.train_trajectory_gspo import V15TrajectoryGSPOTrainer, logger  # noqa: E402
from v23_visual_transition.prepare_offline_data import tool_call_text  # noqa: E402
from v23_visual_transition.train_offline_grpo import build_eval_style_messages  # noqa: E402


WHERE_KEYS = ("coordinate", "start_coordinate", "end_coordinate")


def read_jsonl(path: str, max_rows: int = 0) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_rows and len(rows) >= max_rows:
                break
    return rows


def find_matching_bracket(text: str, start: int) -> int:
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def where_char_mask(target_text: str) -> List[int]:
    """Return 1 for WHAT chars and 0 for WHERE chars."""
    mask = [1] * len(target_text)
    for key in WHERE_KEYS:
        needle = f'"{key}"'
        pos = 0
        while True:
            key_start = target_text.find(needle, pos)
            if key_start < 0:
                break
            colon = target_text.find(":", key_start + len(needle))
            value_start = target_text.find("[", colon)
            if colon < 0 or value_start < 0:
                pos = key_start + len(needle)
                continue
            value_end = find_matching_bracket(target_text, value_start)
            if value_end < 0:
                pos = key_start + len(needle)
                continue
            for idx in range(key_start, value_end + 1):
                mask[idx] = 0
            pos = value_end + 1
    return mask


def token_role_labels(tokenizer, target_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        target_text,
        add_special_tokens=False,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()
    input_ids = encoded["input_ids"].squeeze(0)
    char_mask = where_char_mask(target_text)

    labels: List[int] = []
    for start, end in offsets:
        if end <= start:
            labels.append(-100)
            continue
        span = char_mask[start:end]
        labels.append(0 if any(value == 0 for value in span) else 1)
    return input_ids, torch.tensor(labels, dtype=torch.long)


class WhereWhatRoutedDataset(Dataset):
    def __init__(self, jsonl_path: str, max_rows: int = 0):
        rows = read_jsonl(jsonl_path, max_rows)
        self.rows = [row for row in rows if row.get("image") and os.path.exists(row.get("image"))]
        logger.info(f"Loaded {len(self.rows)} routed where/what rows from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


class WhereWhatRoutedSFTTrainer(V15TrajectoryGSPOTrainer):
    def __init__(self, args):
        super().__init__(args)
        if args.disable_gradient_checkpointing:
            self._set_grad_checkpointing(False)

    def _setup_optimizer(self):
        self._apply_trainable_mode()
        super()._setup_optimizer()

    def _apply_trainable_mode(self) -> None:
        mode = self.args.trainable_mode
        counts = defaultdict(int)
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            trainable = False
            if mode == "all":
                trainable = True
            elif mode == "route_only":
                trainable = "route_weights" in name
            elif mode == "route_a_only":
                trainable = (
                    "route_weights" in name
                    or "lora_A_1" in name
                    or "lora_A_2" in name
                )
            elif mode == "where_heavy":
                trainable = "route_weights" in name or "lora_A_2" in name
            else:
                raise ValueError(f"Unknown trainable_mode: {mode}")
            param.requires_grad = trainable
            if trainable:
                if "route_weights" in name:
                    counts["route"] += param.numel()
                elif "lora_A_1" in name:
                    counts["what_A"] += param.numel()
                elif "lora_A_2" in name:
                    counts["where_A"] += param.numel()
                elif "lora_B" in name:
                    counts["shared_B"] += param.numel()
                elif "comm_" in name:
                    counts["comm"] += param.numel()
                else:
                    counts["other"] += param.numel()
        if self.rank == 0:
            logger.info(f"Trainable mode: {mode} counts={dict(counts)}")

    def _setup_data(self) -> None:
        args = self.args
        self.train_dataset = WhereWhatRoutedDataset(args.train_data, args.max_rows)
        if len(self.train_dataset) == 0:
            raise ValueError(f"No usable rows in {args.train_data}")
        drop_last = self.world_size > 1 and len(self.train_dataset) >= self.world_size
        self.sampler = DistributedSampler(self.train_dataset, shuffle=True, drop_last=drop_last)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,
            sampler=self.sampler,
            collate_fn=lambda items: items[0],
            num_workers=args.num_workers,
            pin_memory=True,
        )
        self.val_dataset = None

    def build_target_text(self, row: Dict[str, Any]) -> str:
        full_tool_call = row.get("full_tool_call") or {}
        if full_tool_call:
            payload = json.dumps(full_tool_call, ensure_ascii=False, indent=2)
            return f"<tool_call>\n{payload}\n</tool_call>"
        return tool_call_text(row.get("gt_action", {}) or {}, False)

    def prepare_example(self, row: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, int]:
        image = Image.open(row["image"]).convert("RGB")
        messages = build_eval_style_messages(row["goal"], row.get("history", []), row["image"])
        prompt_inputs = self._tokenize_for_generation(messages, image)
        prompt_len = int(prompt_inputs["input_ids"].shape[1])

        target_text = self.build_target_text(row)
        response_ids, role_labels = token_role_labels(self.processor.tokenizer, target_text)
        response_ids = response_ids.to(self.device)
        role_labels = role_labels.to(self.device)

        full_ids = torch.cat([prompt_inputs["input_ids"][0], response_ids], dim=0)
        return prompt_inputs, full_ids, role_labels, prompt_len

    def train_one_row(self, row: Dict[str, Any]) -> Optional[Dict[str, float]]:
        args = self.args
        prompt_inputs, full_ids, role_labels, prompt_len = self.prepare_example(row)
        ids = full_ids.unsqueeze(0)
        attention_mask = torch.ones_like(ids)

        fwd_kwargs = {"input_ids": ids, "attention_mask": attention_mask}
        for key in ("pixel_values", "image_grid_thw"):
            if key in prompt_inputs:
                fwd_kwargs[key] = prompt_inputs[key]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model.forward(**fwd_kwargs)
            logits = outputs.logits

            resp_logits = logits[:, prompt_len - 1:-1, :]
            resp_labels = ids[:, prompt_len:]
            token_loss = F.cross_entropy(
                resp_logits.reshape(-1, resp_logits.shape[-1]),
                resp_labels.reshape(-1),
                reduction="none",
            ).view_as(resp_labels)
            lm_mask = (resp_labels != self.pad_id).float()
            lm_loss = (token_loss * lm_mask).sum() / lm_mask.sum().clamp(min=1.0)

            route_values = self.model.get_route_values_for_loss()
            route_loss = torch.tensor(0.0, device=self.device)
            route_what_mean = 0.5
            route_where_mean = 0.5
            n_where = int((role_labels == 0).sum().item())
            n_what = int((role_labels == 1).sum().item())
            if route_values is not None:
                route_resp = route_values[:, prompt_len - 1:prompt_len - 1 + role_labels.shape[0]].squeeze(0)
                route_mask = role_labels >= 0
                if route_mask.any():
                    targets = role_labels[route_mask].float()
                    route_selected = route_resp[route_mask].float().clamp(1e-4, 1.0 - 1e-4)
                    with torch.amp.autocast("cuda", enabled=False):
                        route_loss = F.binary_cross_entropy(route_selected, targets)
                if (role_labels == 1).any():
                    route_what_mean = float(route_resp[role_labels == 1].detach().mean().item())
                if (role_labels == 0).any():
                    route_where_mean = float(route_resp[role_labels == 0].detach().mean().item())

            row_weight = min(float(row.get("weight") or 1.0), args.weight_clip)
            total_loss = row_weight * (lm_loss + args.route_loss_weight * route_loss)

        total_loss.backward()

        return {
            "lm_loss": float(lm_loss.detach().item()),
            "route_loss": float(route_loss.detach().item()),
            "total_loss": float(total_loss.detach().item()),
            "route_what_mean": route_what_mean,
            "route_where_mean": route_where_mean,
            "n_what_tokens": float(n_what),
            "n_where_tokens": float(n_where),
            "row_weight": float(row_weight),
        }

    def train(self) -> None:
        args = self.args
        if self.rank == 0:
            logger.info("=" * 60)
            logger.info("V23 WHAT/WHERE Routed SFT")
            logger.info(f"  train_data={args.train_data}")
            logger.info(f"  rows={len(self.train_dataset)} world_size={self.world_size}")
            logger.info(f"  route_loss_weight={args.route_loss_weight}")
            logger.info(f"  trainable_mode={args.trainable_mode}")
            logger.info(f"  grad_accum={args.gradient_accumulation_steps}")
            logger.info("=" * 60)

        for epoch in range(args.num_epochs):
            self.sampler.set_epoch(epoch)
            metrics_accum: Dict[str, List[float]] = defaultdict(list)
            accum_count = 0
            skipped = 0
            t_epoch = time.time()
            self.optimizer.zero_grad(set_to_none=True)

            for row_idx, row in enumerate(self.train_loader):
                if args.max_steps > 0 and self.global_step >= args.max_steps:
                    break
                t_row = time.time()
                try:
                    metrics = self.train_one_row(row)
                except Exception as exc:
                    if self.rank == 0:
                        logger.warning(f"Row {row_idx} failed: {exc}\n{traceback.format_exc()}")
                    self.optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    skipped += 1
                    if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower():
                        torch.cuda.empty_cache()
                    continue

                if metrics is None:
                    skipped += 1
                else:
                    accum_count += 1
                    for key, value in metrics.items():
                        metrics_accum[key].append(float(value))

                if (row_idx + 1) % args.gradient_accumulation_steps != 0:
                    continue

                scale = 1.0 / args.gradient_accumulation_steps
                for param in self.model.parameters():
                    if param.requires_grad and param.grad is not None:
                        param.grad.mul_(scale)

                if self.world_size > 1:
                    for param in self.model.parameters():
                        if param.requires_grad:
                            if param.grad is None:
                                param.grad = torch.zeros_like(param.data)
                            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

                trainable = [param for param in self.model.parameters() if param.requires_grad]
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                self.optimizer.step()
                self.global_step += 1

                if self.rank == 0 and self.global_step % args.logging_steps == 0:
                    n_log = max(accum_count, 1)
                    recent = {key: values[-n_log:] for key, values in metrics_accum.items()}
                    logger.info(
                        f"E{epoch} S{self.global_step} "
                        f"lm={np.mean(recent.get('lm_loss', [0.0])):.4f} "
                        f"route={np.mean(recent.get('route_loss', [0.0])):.4f} "
                        f"total={np.mean(recent.get('total_loss', [0.0])):.4f} "
                        f"r_what={np.mean(recent.get('route_what_mean', [0.5])):.3f} "
                        f"r_where={np.mean(recent.get('route_where_mean', [0.5])):.3f} "
                        f"gnorm={grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm:.2f} "
                        f"valid={accum_count}/{args.gradient_accumulation_steps} "
                        f"skipped={skipped} t={time.time() - t_row:.1f}s"
                    )

                if self.rank == 0 and args.save_steps > 0 and self.global_step % args.save_steps == 0:
                    self.save_checkpoint(f"epoch-{epoch}_step-{self.global_step}")

                self.optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                torch.cuda.empty_cache()

            if self.rank == 0:
                logger.info(
                    f"Epoch {epoch} done in {(time.time() - t_epoch) / 60:.1f} min; "
                    f"steps={self.global_step} skipped={skipped}"
                )
                self.save_checkpoint(f"epoch-{epoch}_final")
                logger.info("WHAT/WHERE routed SFT complete!")
            if self.world_size > 1:
                dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser(description="V23 WHAT/WHERE token-routed SFT")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sft_checkpoint", default="")
    parser.add_argument("--resume_from", default="")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--num_comm_rounds", type=int, default=2)
    parser.add_argument("--balance_weight", type=float, default=0.0)
    parser.add_argument("--image_max_pixels", type=int, default=602112)
    parser.add_argument("--route_loss_weight", type=float, default=0.2)
    parser.add_argument(
        "--trainable_mode",
        choices=["all", "route_only", "route_a_only", "where_heavy"],
        default="all",
    )
    parser.add_argument("--weight_clip", type=float, default=2.0)
    parser.add_argument("--disable_gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--lora_lr", type=float, default=1e-5)
    parser.add_argument("--route_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--max_steps", type=int, default=0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    trainer = WhereWhatRoutedSFTTrainer(args)
    trainer.train()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()