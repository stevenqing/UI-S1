#!/usr/bin/env python3
"""Plain or regularized SFT injection on heterogeneous correct GUI actions."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v15_gui_360.train_trajectory_gspo import V15TrajectoryGSPOTrainer, logger  # noqa: E402
from v23_visual_transition.train_offline_grpo import build_eval_style_messages  # noqa: E402


class HeteroInjectDataset(Dataset):
    def __init__(self, path: str, max_rows: int = 0):
        self.rows: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                image = row.get("image") or row.get("screenshot")
                target_text = row.get("target_text")
                if image and target_text and os.path.exists(image):
                    row["image"] = image
                    self.rows.append(row)
                if max_rows and len(self.rows) >= max_rows:
                    break
        logger.info(f"Loaded {len(self.rows)} hetero injection rows from {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


class HeteroInjectSFTTrainer(V15TrajectoryGSPOTrainer):
    def __init__(self, args: argparse.Namespace):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        super().__init__(args)
        if args.disable_gradient_checkpointing:
            self._set_grad_checkpointing(False)
        if self.rank == 0:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            self.metrics_path = Path(args.output_dir) / "metrics.jsonl"
            if not args.resume_from:
                self.metrics_path.write_text("", encoding="utf-8")

    def _setup_data(self) -> None:
        args = self.args
        self.train_dataset = HeteroInjectDataset(args.train_data, args.max_rows)
        if len(self.train_dataset) == 0:
            raise ValueError(f"No usable rows in {args.train_data}")
        drop_last = self.world_size > 1 and len(self.train_dataset) >= self.world_size
        self.sampler = DistributedSampler(
            self.train_dataset,
            shuffle=True,
            drop_last=drop_last,
            seed=args.seed,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,
            sampler=self.sampler,
            collate_fn=lambda items: items[0],
            num_workers=args.num_workers,
            pin_memory=True,
        )
        self.val_dataset = None

    def _setup_optimizer(self) -> None:
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
            elif mode == "lora_only":
                trainable = "lora_" in name
            elif mode == "lora_route":
                trainable = "lora_" in name or "route_weights" in name
            elif mode == "route_only":
                trainable = "route_weights" in name
            elif mode == "comm_only":
                trainable = "comm_" in name
            else:
                raise ValueError(f"Unknown trainable_mode: {mode}")
            param.requires_grad = trainable
            if trainable:
                if "lora_" in name:
                    counts["lora"] += param.numel()
                elif "route_weights" in name:
                    counts["route"] += param.numel()
                elif "comm_" in name:
                    counts["comm"] += param.numel()
                else:
                    counts["other"] += param.numel()
        if self.rank == 0:
            logger.info(f"Trainable mode: {mode} counts={dict(counts)}")

    def prepare_example(self, row: dict[str, Any]) -> Tuple[dict[str, torch.Tensor], torch.Tensor, int]:
        image = Image.open(row["image"]).convert("RGB")
        messages = build_eval_style_messages(row["goal"], row.get("history") or [], row["image"])
        prompt_inputs = self._tokenize_for_generation(messages, image)
        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        response_ids = self.processor.tokenizer(
            row["target_text"],
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device).squeeze(0)
        if response_ids.numel() == 0:
            raise ValueError("empty target_text")
        if self.args.append_eos:
            eos = torch.tensor([self.processor.tokenizer.eos_token_id], device=self.device)
            response_ids = torch.cat([response_ids, eos], dim=0)
        full_ids = torch.cat([prompt_inputs["input_ids"][0], response_ids], dim=0)
        return prompt_inputs, full_ids, prompt_len

    def reference_logits(self, fwd_kwargs: dict[str, torch.Tensor], prompt_len: int) -> Optional[torch.Tensor]:
        if self.args.kl_weight <= 0:
            return None
        self.model.disable_lora()
        try:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                ref_outputs = self.model.forward(**fwd_kwargs)
            return ref_outputs.logits[:, prompt_len - 1:-1, :].detach()
        finally:
            self.model.enable_lora()

    def regularization_losses(
        self,
        resp_logits: torch.Tensor,
        ref_resp_logits: Optional[torch.Tensor],
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero = torch.tensor(0.0, device=self.device)
        kl_loss = zero
        # Entropy is always measured, even when its objective weight is zero.
        # The previous logger conflated "entropy bonus disabled" with
        # "model entropy equals zero", producing misleading all-zero curves.
        current_logp = F.log_softmax(resp_logits.float(), dim=-1)
        current_p = current_logp.exp()
        token_entropy = -(current_p * current_logp).sum(dim=-1)
        entropy = (token_entropy * mask).sum() / mask.sum().clamp(min=1.0)
        if ref_resp_logits is not None:
            ref_logp = F.log_softmax(ref_resp_logits.float(), dim=-1)
            token_kl = (current_p * (current_logp - ref_logp)).sum(dim=-1)
            kl_loss = (token_kl * mask).sum() / mask.sum().clamp(min=1.0)
        return kl_loss, entropy

    def train_one_row(self, row: dict[str, Any]) -> Optional[dict[str, float]]:
        prompt_inputs, full_ids, prompt_len = self.prepare_example(row)
        ids = full_ids.unsqueeze(0)
        attention_mask = torch.ones_like(ids)
        fwd_kwargs = {"input_ids": ids, "attention_mask": attention_mask}
        for key in ("pixel_values", "image_grid_thw"):
            if key in prompt_inputs:
                fwd_kwargs[key] = prompt_inputs[key]
        resp_labels = ids[:, prompt_len:]
        mask = (resp_labels != self.pad_id).float()
        ref_resp_logits = self.reference_logits(fwd_kwargs, prompt_len)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model.forward(**fwd_kwargs)
            logits = outputs.logits
            resp_logits = logits[:, prompt_len - 1:-1, :]
            token_loss = F.cross_entropy(
                resp_logits.reshape(-1, resp_logits.shape[-1]),
                resp_labels.reshape(-1),
                reduction="none",
                label_smoothing=self.args.label_smoothing,
            ).view_as(resp_labels)
            lm_loss = (token_loss * mask).sum() / mask.sum().clamp(min=1.0)

        kl_loss, entropy = self.regularization_losses(resp_logits, ref_resp_logits, mask)
        balance_loss, route_mean = self.model.compute_balance_loss()
        total_loss = (
            self.args.lm_loss_weight * lm_loss
            + self.args.kl_weight * kl_loss
            + self.args.balance_weight * balance_loss
            - self.args.entropy_bonus * entropy
        )
        total_loss.backward()
        return {
            "lm_loss": float(lm_loss.detach().item()),
            "kl_loss": float(kl_loss.detach().item()),
            "entropy": float(entropy.detach().item()),
            "balance_loss": float(balance_loss.detach().item()),
            "route_mean": float(route_mean),
            "total_loss": float(total_loss.detach().item()),
            "tokens": float(mask.sum().detach().item()),
            "entropy_bonus_weight": float(self.args.entropy_bonus),
            "entropy_bonus_contribution": float((-self.args.entropy_bonus * entropy).detach().item()),
        }

    def log_metrics(self, epoch: int, metrics_accum: dict[str, list[float]], grad_norm: Any, accum_count: int, skipped: int, row_time: float) -> None:
        if self.rank != 0:
            return
        n_log = max(accum_count, 1)
        recent = {key: values[-n_log:] for key, values in metrics_accum.items()}
        payload = {
            "epoch": epoch,
            "global_step": self.global_step,
            "skipped": skipped,
            "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
            "row_time_sec": row_time,
        }
        for key, values in recent.items():
            payload[key] = float(np.mean(values)) if values else 0.0
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        logger.info(
            f"E{epoch} S{self.global_step} "
            f"lm={payload.get('lm_loss', 0.0):.4f} kl={payload.get('kl_loss', 0.0):.4f} "
            f"ent={payload.get('entropy', 0.0):.3f} total={payload.get('total_loss', 0.0):.4f} "
            f"gnorm={payload['grad_norm']:.2f} valid={accum_count}/{self.args.gradient_accumulation_steps} "
            f"skipped={skipped} t={row_time:.1f}s"
        )

    def train(self) -> None:
        args = self.args
        if self.rank == 0:
            logger.info("=" * 60)
            logger.info("Heterogeneous Correct-Answer Injection SFT")
            logger.info(f"  train_data={args.train_data}")
            logger.info(f"  rows={len(self.train_dataset)} world_size={self.world_size}")
            logger.info(f"  label_smoothing={args.label_smoothing} kl_weight={args.kl_weight} entropy_bonus={args.entropy_bonus}")
            logger.info(f"  trainable_mode={args.trainable_mode} max_steps={args.max_steps}")
            logger.info("=" * 60)

        for epoch in range(args.num_epochs):
            self.sampler.set_epoch(epoch)
            metrics_accum: dict[str, list[float]] = defaultdict(list)
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
                except Exception as exc:  # noqa: BLE001
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

                if accum_count == 0 or accum_count % args.gradient_accumulation_steps != 0:
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

                if self.global_step % args.logging_steps == 0:
                    self.log_metrics(epoch, metrics_accum, grad_norm, accum_count, skipped, time.time() - t_row)
                if self.rank == 0 and args.save_steps > 0 and self.global_step % args.save_steps == 0:
                    self.save_checkpoint(f"epoch-{epoch}_step-{self.global_step}")

                self.optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                torch.cuda.empty_cache()

            if self.rank == 0:
                logger.info(f"Epoch {epoch} done in {(time.time() - t_epoch) / 60:.1f} min; steps={self.global_step} skipped={skipped}")
                self.save_checkpoint(f"epoch-{epoch}_final")
            if self.world_size > 1:
                dist.barrier()
            if args.max_steps > 0 and self.global_step >= args.max_steps:
                break
        if self.rank == 0:
            self.save_checkpoint("final")
            logger.info("Heterogeneous injection training complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sft_checkpoint", default="")
    parser.add_argument("--resume_from", default="")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    parser.add_argument("--num_comm_rounds", type=int, default=2)
    parser.add_argument("--balance_weight", type=float, default=0.0)
    parser.add_argument("--image_max_pixels", type=int, default=602112)
    parser.add_argument("--trainable_mode", choices=["all", "lora_only", "lora_route", "route_only", "comm_only"], default="lora_only")
    parser.add_argument("--disable_gradient_checkpointing", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--lm_loss_weight", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--kl_weight", type=float, default=0.0)
    parser.add_argument("--entropy_bonus", type=float, default=0.0)
    parser.add_argument("--append_eos", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--lora_lr", type=float, default=2e-5)
    parser.add_argument("--route_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer = HeteroInjectSFTTrainer(args)
    trainer.train()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()