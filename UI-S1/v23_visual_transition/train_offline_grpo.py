#!/usr/bin/env python3
"""Offline GRPO-style training from matcher-scored GUI-360 candidates.

This trainer is intentionally offline: it never calls model.generate during
training. Each row contains one GT screen state plus K candidate tool calls that
were sampled beforehand and scored by the GUI-360 matcher. Training optimizes a
group-relative reward objective over those fixed candidates, with an optional
GT-action SFT anchor for stability.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from v13_gui_360.eval_gui360_template import (  # noqa: E402
    SUPPORTED_ACTIONS as EVAL_SUPPORTED_ACTIONS,
    USER_PROMPT_TEMPLATE as EVAL_USER_PROMPT_TEMPLATE,
)
from v15_gui_360.train_trajectory_gspo import (  # noqa: E402
    V15TrajectoryGSPOTrainer,
    logger,
)
from v23_visual_transition.prepare_offline_data import (  # noqa: E402
    format_action_for_history,
    tool_call_text,
)


def read_episode_jsonl(path: str) -> Dict[str, Dict[str, Any]]:
    episodes: Dict[str, Dict[str, Any]] = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                episode = json.loads(line)
                episodes[str(episode.get("episode_id"))] = episode
    return episodes


def build_eval_style_messages(goal: str, history: List[str], image_path: str) -> List[Dict[str, Any]]:
    """Use the V13 evaluator prompt text with HF-local image payloads."""
    history_text = "\n".join(history) if history else "None"
    prompt_text = EVAL_USER_PROMPT_TEMPLATE.format(
        instruction=goal,
        history=history_text,
        actions=EVAL_SUPPORTED_ACTIONS,
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


class OfflineCandidateDataset(Dataset):
    """Matcher-scored action candidates with GT-history reconstruction."""

    def __init__(self, candidate_path: str, episode_path: str, max_rows: int = 0):
        self.episodes = read_episode_jsonl(episode_path)
        self.rows: List[Dict[str, Any]] = []
        skipped = 0

        with open(candidate_path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                normalized = self._normalize_row(row)
                if normalized is None:
                    skipped += 1
                    continue
                self.rows.append(normalized)
                if max_rows and len(self.rows) >= max_rows:
                    break

        logger.info(
            f"Loaded {len(self.rows)} offline candidate rows from {candidate_path} "
            f"(skipped={skipped})"
        )

    def _normalize_row(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        episode_id = str(row.get("episode_id"))
        episode = self.episodes.get(episode_id)
        if not episode:
            return None

        steps = episode.get("steps") or []
        step_idx = int(row.get("step_idx", 0))
        if step_idx < 0 or step_idx >= len(steps):
            return None

        step = steps[step_idx]
        screenshot = row.get("screenshot") or step.get("screenshot")
        if not screenshot or not os.path.exists(screenshot):
            return None

        candidates = [
            candidate for candidate in row.get("candidates", [])
            if isinstance(candidate, dict) and candidate.get("text")
        ]
        if not candidates and not (row.get("gt_action") or step.get("action")):
            return None

        history = [
            format_action_for_history(steps[idx].get("action", {}) or {}, idx + 1)
            for idx in range(step_idx)
        ]

        hard_state = row.get("hard_state") or {}
        return {
            **row,
            "episode_id": episode_id,
            "step_idx": step_idx,
            "num_steps": len(steps),
            "goal": row.get("goal") or episode.get("goal", ""),
            "screenshot": screenshot,
            "gt_action": row.get("gt_action") or step.get("action") or {},
            "is_last_step": step_idx == len(steps) - 1,
            "history": history,
            "source_weight": float(
                hard_state.get("weight")
                or row.get("weight")
                or row.get("source_weight")
                or 1.0
            ),
            "candidates": candidates,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


class V23OfflineCandidateGRPOTrainer(V15TrajectoryGSPOTrainer):
    """Reuse V15 model/LoRA infrastructure, replace rollout with offline rows."""

    def _setup_data(self) -> None:
        args = self.args
        self.train_dataset = OfflineCandidateDataset(
            args.candidate_data,
            args.episode_data,
            max_rows=args.max_rows,
        )
        if len(self.train_dataset) == 0:
            raise ValueError(f"No usable offline candidate rows in {args.candidate_data}")

        drop_last = self.world_size > 1 and len(self.train_dataset) >= self.world_size
        self.sampler = DistributedSampler(
            self.train_dataset,
            shuffle=True,
            drop_last=drop_last,
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=1,
            sampler=self.sampler,
            collate_fn=lambda x: x[0],
            num_workers=args.num_workers,
            pin_memory=True,
        )
        self.val_dataset = None

    def _candidate_group(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        args = self.args
        candidates: List[Dict[str, Any]] = []
        seen: Dict[str, int] = {}

        raw_candidates = row.get("candidates", [])
        if args.max_candidates > 0:
            raw_candidates = raw_candidates[:args.max_candidates]

        for candidate in raw_candidates:
            text = str(candidate.get("text") or "")
            if not text.strip():
                continue
            reward = float(candidate.get("reward") or 0.0)
            if text in seen:
                idx = seen[text]
                candidates[idx]["reward"] = max(candidates[idx]["reward"], reward)
                continue
            seen[text] = len(candidates)
            candidates.append({
                "text": text,
                "reward": reward,
                "source": "sample",
            })

        if args.include_gt_candidate:
            gt_text = tool_call_text(row.get("gt_action", {}) or {}, bool(row.get("is_last_step")))
            if gt_text and gt_text not in seen:
                candidates.append({
                    "text": gt_text,
                    "reward": args.gt_reward,
                    "source": "gt",
                })

        return candidates

    def _encode_candidate(
        self,
        inputs: Dict[str, torch.Tensor],
        text: str,
    ) -> Optional[Tuple[torch.Tensor, int]]:
        tokenized = self.processor.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        response_ids = tokenized["input_ids"].to(self.device).squeeze(0)
        if response_ids.numel() == 0:
            return None
        if self.args.append_eos:
            eos = torch.tensor([self.processor.tokenizer.eos_token_id], device=self.device)
            response_ids = torch.cat([response_ids, eos], dim=0)

        prompt_ids = inputs["input_ids"][0]
        full_ids = torch.cat([prompt_ids, response_ids], dim=0)
        return full_ids, int(prompt_ids.shape[0])

    def _sequence_log_prob(
        self,
        full_ids: torch.Tensor,
        prompt_len: int,
        inputs: Dict[str, torch.Tensor],
        with_grad: bool,
    ) -> Tuple[torch.Tensor, int]:
        tok_lp, mask, n_tokens = self._compute_token_log_probs(
            full_ids,
            prompt_len,
            inputs,
            with_grad=with_grad,
        )
        denom = mask.sum().clamp(min=1.0) if self.args.mean_token_logprob else 1.0
        return (tok_lp * mask).sum() / denom, n_tokens

    def train_candidate_group(self, row: Dict[str, Any]) -> Optional[Dict[str, float]]:
        args = self.args
        if args.objective == "conservative_distill":
            return self.train_conservative_distill_group(row)

        candidates = self._candidate_group(row)
        if len(candidates) < 2:
            return None

        rewards = torch.tensor(
            [candidate["reward"] for candidate in candidates],
            dtype=torch.float32,
            device=self.device,
        )
        reward_std = rewards.std(unbiased=False)
        if reward_std.item() < args.min_reward_std:
            return None

        advantages = (rewards - rewards.mean()) / (reward_std + 1e-8)
        if args.advantage_clip > 0:
            advantages = advantages.clamp(-args.advantage_clip, args.advantage_clip)

        image = Image.open(row["screenshot"]).convert("RGB")
        messages = build_eval_style_messages(row["goal"], row.get("history", []), row["screenshot"])
        inputs = self._tokenize_for_generation(messages, image)

        encoded_items = []
        for idx, candidate in enumerate(candidates):
            encoded = self._encode_candidate(inputs, candidate["text"])
            if encoded is not None:
                encoded_items.append((idx, candidate, encoded))

        n_used = len(encoded_items)
        if n_used < 2:
            return None

        total_tokens = 0
        sample_logprobs: List[float] = []
        gt_logprobs: List[float] = []
        group_loss_value = 0.0
        row_weight = min(float(row.get("source_weight") or 1.0), args.weight_clip)
        did_backward = False

        for idx, candidate, encoded in encoded_items:
            full_ids, prompt_len = encoded
            seq_logprob, n_tokens = self._sequence_log_prob(
                full_ids,
                prompt_len,
                inputs,
                with_grad=True,
            )
            seq_logprob_value = float(seq_logprob.detach().item())

            if candidate["source"] == "gt":
                gt_logprobs.append(seq_logprob_value)
            else:
                sample_logprobs.append(seq_logprob_value)

            loss_term = -advantages[idx].detach() * seq_logprob / n_used
            if candidate["source"] == "gt" and args.sft_anchor_weight > 0:
                loss_term = loss_term - args.sft_anchor_weight * seq_logprob
            loss_term = loss_term * row_weight
            if loss_term.requires_grad:
                loss_term.backward()
                did_backward = True

            group_loss_value += float(loss_term.detach().item())
            total_tokens += n_tokens

        bal_loss, mean_w = self.model.compute_balance_loss()
        balance_term = args.balance_weight * bal_loss
        if balance_term.requires_grad:
            balance_term.backward()
            did_backward = True

        if not did_backward:
            return None

        return {
            "offline_loss": group_loss_value,
            "balance_loss": float(bal_loss.detach().item()),
            "routing_w": float(mean_w),
            "mean_reward": float(rewards.mean().item()),
            "best_reward": float(rewards.max().item()),
            "reward_std": float(reward_std.item()),
            "mean_abs_adv": float(advantages.abs().mean().item()),
            "n_candidates": float(n_used),
            "total_tokens": float(total_tokens),
            "row_weight": float(row_weight),
            "gt_logprob": float(np.mean(gt_logprobs)) if gt_logprobs else 0.0,
            "sample_logprob": float(np.mean(sample_logprobs)) if sample_logprobs else 0.0,
            "any_success": float(any(candidate["reward"] >= args.match_threshold for candidate in candidates)),
        }

    def train_conservative_distill_group(self, row: Dict[str, Any]) -> Optional[Dict[str, float]]:
        args = self.args
        raw_candidates = row.get("candidates", [])
        if args.max_candidates > 0:
            raw_candidates = raw_candidates[:args.max_candidates]

        sample_candidates = [
            {
                "text": str(candidate.get("text") or ""),
                "reward": float(candidate.get("reward") or 0.0),
            }
            for candidate in raw_candidates
            if isinstance(candidate, dict) and str(candidate.get("text") or "").strip()
        ]
        best_sample = max(sample_candidates, key=lambda item: item["reward"], default=None)
        use_best = best_sample is not None and best_sample["reward"] >= args.best_reward_threshold

        image = Image.open(row["screenshot"]).convert("RGB")
        messages = build_eval_style_messages(row["goal"], row.get("history", []), row["screenshot"])
        inputs = self._tokenize_for_generation(messages, image)

        gt_text = tool_call_text(row.get("gt_action", {}) or {}, bool(row.get("is_last_step")))
        row_weight = min(float(row.get("source_weight") or 1.0), args.weight_clip)
        total_tokens = 0
        did_backward = False
        loss_value = 0.0
        gt_logprobs: List[float] = []
        best_logprobs: List[float] = []

        encoded_gt = self._encode_candidate(inputs, gt_text)
        if encoded_gt is None:
            return None
        full_ids, prompt_len = encoded_gt
        gt_logprob, n_tokens = self._sequence_log_prob(full_ids, prompt_len, inputs, with_grad=True)
        gt_loss = -args.gt_loss_weight * gt_logprob * row_weight
        if gt_loss.requires_grad:
            gt_loss.backward()
            did_backward = True
        loss_value += float(gt_loss.detach().item())
        gt_logprobs.append(float(gt_logprob.detach().item()))
        total_tokens += n_tokens

        if use_best and best_sample["text"] != gt_text:
            encoded_best = self._encode_candidate(inputs, best_sample["text"])
            if encoded_best is not None:
                full_ids, prompt_len = encoded_best
                best_logprob, n_tokens = self._sequence_log_prob(full_ids, prompt_len, inputs, with_grad=True)
                best_loss = -args.best_candidate_weight * best_sample["reward"] * best_logprob * row_weight
                if best_loss.requires_grad:
                    best_loss.backward()
                    did_backward = True
                loss_value += float(best_loss.detach().item())
                best_logprobs.append(float(best_logprob.detach().item()))
                total_tokens += n_tokens

        bal_loss, mean_w = self.model.compute_balance_loss()
        balance_term = args.balance_weight * bal_loss
        if balance_term.requires_grad:
            balance_term.backward()
            did_backward = True

        if not did_backward:
            return None

        rewards = [candidate["reward"] for candidate in sample_candidates]
        return {
            "offline_loss": loss_value,
            "balance_loss": float(bal_loss.detach().item()),
            "routing_w": float(mean_w),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "best_reward": float(best_sample["reward"]) if best_sample else 0.0,
            "reward_std": float(np.std(rewards)) if rewards else 0.0,
            "mean_abs_adv": 0.0,
            "n_candidates": float(1 + int(bool(best_logprobs))),
            "total_tokens": float(total_tokens),
            "row_weight": float(row_weight),
            "gt_logprob": float(np.mean(gt_logprobs)) if gt_logprobs else 0.0,
            "sample_logprob": float(np.mean(best_logprobs)) if best_logprobs else 0.0,
            "any_success": float(use_best),
        }

    def train(self) -> None:
        args = self.args
        if self.rank == 0:
            logger.info("=" * 60)
            logger.info("V23 GUI-360 Offline Candidate GRPO")
            logger.info(f"  model={args.model_path}")
            logger.info(f"  candidates={args.candidate_data}")
            logger.info(f"  episode_data={args.episode_data}")
            logger.info(f"  rows={len(self.train_dataset)} world_size={self.world_size}")
            logger.info(f"  objective={args.objective}")
            logger.info(f"  include_gt_candidate={args.include_gt_candidate}")
            logger.info(f"  sft_anchor_weight={args.sft_anchor_weight}")
            logger.info(f"  grad_accum={args.gradient_accumulation_steps}")
            logger.info("=" * 60)

        for epoch in range(args.num_epochs):
            self.sampler.set_epoch(epoch)
            epoch_metrics: Dict[str, List[float]] = defaultdict(list)
            accum_count = 0
            skipped = 0
            t_epoch = time.time()
            skip_rows = self.resume_step * args.gradient_accumulation_steps

            self.optimizer.zero_grad()

            for row_idx, row in enumerate(self.train_loader):
                if row_idx < skip_rows:
                    continue
                if args.max_steps > 0 and self.global_step >= args.max_steps:
                    break

                t_row = time.time()
                try:
                    metrics = self.train_candidate_group(row)
                except Exception as exc:
                    if self.rank == 0:
                        logger.warning(
                            f"Offline row {row_idx} failed: {exc}\n{traceback.format_exc()}"
                        )
                    self.optimizer.zero_grad(set_to_none=True)
                    accum_count = 0
                    if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower():
                        torch.cuda.empty_cache()
                    metrics = None

                if metrics is None:
                    skipped += 1
                    if self.rank == 0:
                        logger.info(f"  row {row_idx} skipped (no reward contrast)")
                else:
                    accum_count += 1
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            epoch_metrics[key].append(float(value))

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
                    recent_loss = epoch_metrics.get("offline_loss", [0.0])[-n_log:]
                    recent_reward = epoch_metrics.get("mean_reward", [0.0])[-n_log:]
                    recent_best = epoch_metrics.get("best_reward", [0.0])[-n_log:]
                    recent_std = epoch_metrics.get("reward_std", [0.0])[-n_log:]
                    recent_gt_lp = epoch_metrics.get("gt_logprob", [0.0])[-n_log:]
                    recent_sample_lp = epoch_metrics.get("sample_logprob", [0.0])[-n_log:]
                    logger.info(
                        f"E{epoch} S{self.global_step} "
                        f"loss={np.mean(recent_loss):.4f} "
                        f"r={np.mean(recent_reward):.3f} "
                        f"best={np.mean(recent_best):.3f} "
                        f"std={np.mean(recent_std):.3f} "
                        f"gt_lp={np.mean(recent_gt_lp):.3f} "
                        f"sample_lp={np.mean(recent_sample_lp):.3f} "
                        f"gnorm={grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm:.2f} "
                        f"valid={accum_count}/{args.gradient_accumulation_steps} "
                        f"skipped={skipped} t={time.time() - t_row:.1f}s"
                    )

                if self.rank == 0 and args.save_steps > 0 and self.global_step % args.save_steps == 0:
                    self.save_checkpoint(f"epoch-{epoch}_step-{self.global_step}")

                self.optimizer.zero_grad()
                accum_count = 0
                torch.cuda.empty_cache()

            if self.rank == 0:
                logger.info(
                    f"Epoch {epoch} done in {(time.time() - t_epoch) / 60:.1f} min; "
                    f"steps={self.global_step} skipped={skipped}"
                )
                self.save_checkpoint(f"epoch-{epoch}_final")

            if self.world_size > 1:
                dist.barrier()
            self.resume_step = 0

        if self.rank == 0:
            logger.info("Offline GRPO training complete!")


def main() -> None:
    parser = argparse.ArgumentParser(description="V23 offline candidate GRPO for GUI-360")

    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sft_checkpoint", default="")
    parser.add_argument("--resume_from", default="")

    parser.add_argument("--candidate_data", required=True)
    parser.add_argument("--episode_data", required=True)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--num_comm_rounds", type=int, default=2)
    parser.add_argument("--balance_weight", type=float, default=0.01)
    parser.add_argument("--image_max_pixels", type=int, default=602112)

    parser.add_argument("--max_candidates", type=int, default=0)
    parser.add_argument("--objective", choices=["grpo", "conservative_distill"], default="grpo")
    parser.add_argument("--include_gt_candidate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gt_reward", type=float, default=1.0)
    parser.add_argument("--gt_loss_weight", type=float, default=1.0)
    parser.add_argument("--best_candidate_weight", type=float, default=0.25)
    parser.add_argument("--best_reward_threshold", type=float, default=0.5)
    parser.add_argument("--sft_anchor_weight", type=float, default=0.25)
    parser.add_argument("--min_reward_std", type=float, default=1e-6)
    parser.add_argument("--advantage_clip", type=float, default=5.0)
    parser.add_argument("--weight_clip", type=float, default=5.0)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--append_eos", action="store_true")
    parser.add_argument("--mean_token_logprob", action=argparse.BooleanOptionalAction, default=True)

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

    trainer = V23OfflineCandidateGRPOTrainer(args)
    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()