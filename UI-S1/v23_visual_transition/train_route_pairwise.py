#!/usr/bin/env python3
"""Route-only pairwise preference training for V23 GUI-360.

This is the conservative follow-up to WHAT/WHERE routed SFT: keep the actor
experts frozen and update only the router so matcher-success actions score
above valid hard negatives on the same GT screen.
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

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from v15_gui_360.train_trajectory_gspo import logger  # noqa: E402
from v23_visual_transition.prepare_offline_data import tool_call_text  # noqa: E402
from v23_visual_transition.train_offline_grpo import (  # noqa: E402
    V23OfflineCandidateGRPOTrainer,
    build_eval_style_messages,
)


class V23RoutePairwiseTrainer(V23OfflineCandidateGRPOTrainer):
    def __init__(self, args):
        super().__init__(args)
        if args.disable_gradient_checkpointing:
            self._set_grad_checkpointing(False)
        self._initial_route_params = {
            name: param.detach().float().clone()
            for name, param in self.model.named_parameters()
            if "route_weights" in name
        }

    def _setup_optimizer(self):
        self._apply_trainable_mode()
        super()._setup_optimizer()

    def _apply_trainable_mode(self) -> None:
        mode = self.args.trainable_mode
        counts = defaultdict(int)
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if mode == "route_only":
                trainable = "route_weights" in name
            elif mode == "route_a_only":
                trainable = "route_weights" in name or "lora_A_1" in name or "lora_A_2" in name
            elif mode == "all":
                trainable = True
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

    def _route_l2_loss(self) -> torch.Tensor:
        if self.args.route_l2_weight <= 0:
            return torch.tensor(0.0, device=self.device)
        terms = []
        for name, param in self.model.named_parameters():
            if name in self._initial_route_params:
                ref = self._initial_route_params[name].to(device=param.device)
                terms.append((param.float() - ref).pow(2).mean())
        if not terms:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(terms).mean()

    def _candidate_text_key(self, text: str) -> str:
        return "\n".join(line.rstrip() for line in text.strip().splitlines())

    def _select_pairs(self, row: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        args = self.args
        hard_state = row.get("hard_state") or {}
        if args.focus_families and hard_state.get("family") not in set(args.focus_families):
            return []

        positives: List[Dict[str, Any]] = []
        negatives: List[Dict[str, Any]] = []
        seen_pos = set()
        seen_neg = set()

        if args.include_gt_positive:
            gt_text = tool_call_text(row.get("gt_action", {}) or {}, bool(row.get("is_last_step")))
            if gt_text.strip():
                positives.append({
                    "text": gt_text,
                    "reward": args.gt_reward,
                    "source": "gt",
                    "pair_type": f"{(row.get('gt_action') or {}).get('action')}->gt",
                })
                seen_pos.add(self._candidate_text_key(gt_text))

        raw_candidates = row.get("candidates", [])
        if args.max_candidates > 0:
            raw_candidates = raw_candidates[:args.max_candidates]

        allowed_pair_types = set(args.negative_pair_types or [])
        for candidate in raw_candidates:
            if not isinstance(candidate, dict):
                continue
            text = str(candidate.get("text") or "")
            if not text.strip():
                continue
            reward = float(candidate.get("reward") or 0.0)
            key = self._candidate_text_key(text)
            pair_type = f"{candidate.get('gt_type')}->{candidate.get('pred_type')}"

            if reward >= args.positive_reward_threshold and args.include_success_positives:
                if key not in seen_pos:
                    positives.append({
                        "text": text,
                        "reward": reward,
                        "source": "success_sample",
                        "pair_type": pair_type,
                    })
                    seen_pos.add(key)
                continue

            if reward > args.negative_reward_threshold:
                continue
            if args.require_negative_format and float(candidate.get("format_reward") or 0.0) < args.min_negative_format_reward:
                continue
            if allowed_pair_types and pair_type not in allowed_pair_types:
                continue
            if key in seen_neg or key in seen_pos:
                continue
            negatives.append({
                "text": text,
                "reward": reward,
                "source": "hard_negative",
                "pair_type": pair_type,
            })
            seen_neg.add(key)

        positives.sort(key=lambda item: (item["source"] != "gt", -item["reward"]))
        negatives.sort(key=lambda item: item["reward"], reverse=True)
        if args.max_positives_per_row > 0:
            positives = positives[:args.max_positives_per_row]
        if args.max_negatives_per_row > 0:
            negatives = negatives[:args.max_negatives_per_row]

        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for positive in positives:
            for negative in negatives:
                if positive["reward"] <= negative["reward"]:
                    continue
                pairs.append((positive, negative))
                if args.max_pairs_per_row > 0 and len(pairs) >= args.max_pairs_per_row:
                    return pairs
        return pairs

    def _encode_pair_item(
        self,
        inputs: Dict[str, torch.Tensor],
        text: str,
    ) -> Optional[Tuple[torch.Tensor, int]]:
        return self._encode_candidate(inputs, text)

    def train_candidate_group(self, row: Dict[str, Any]) -> Optional[Dict[str, float]]:
        args = self.args
        pairs = self._select_pairs(row)
        if not pairs:
            return None

        image = Image.open(row["screenshot"]).convert("RGB")
        messages = build_eval_style_messages(row["goal"], row.get("history", []), row["screenshot"])
        inputs = self._tokenize_for_generation(messages, image)

        row_weight = min(float(row.get("source_weight") or 1.0), args.weight_clip)
        loss_value = 0.0
        total_tokens = 0
        n_used = 0
        margins: List[float] = []
        pair_accs: List[float] = []
        pos_lps: List[float] = []
        neg_lps: List[float] = []
        neg_rewards: List[float] = []
        pair_types: List[str] = []

        for positive, negative in pairs:
            pos_encoded = self._encode_pair_item(inputs, positive["text"])
            neg_encoded = self._encode_pair_item(inputs, negative["text"])
            if pos_encoded is None or neg_encoded is None:
                continue

            pos_ids, pos_prompt_len = pos_encoded
            neg_ids, neg_prompt_len = neg_encoded
            pos_lp, pos_tokens = self._sequence_log_prob(pos_ids, pos_prompt_len, inputs, with_grad=True)
            neg_lp, neg_tokens = self._sequence_log_prob(neg_ids, neg_prompt_len, inputs, with_grad=True)
            preference = pos_lp - neg_lp
            reward_gap = float(positive["reward"] - negative["reward"])
            margin = args.logprob_margin + args.reward_margin_scale * reward_gap
            pair_loss = F.softplus(-args.beta * (preference - margin)) / max(args.beta, 1e-8)
            pair_loss = pair_loss * row_weight
            pair_loss.backward()

            loss_value += float(pair_loss.detach().item())
            total_tokens += pos_tokens + neg_tokens
            n_used += 1
            pref_value = float(preference.detach().item())
            margins.append(pref_value)
            pair_accs.append(float(pref_value > margin))
            pos_lps.append(float(pos_lp.detach().item()))
            neg_lps.append(float(neg_lp.detach().item()))
            neg_rewards.append(float(negative["reward"]))
            pair_types.append(str(negative.get("pair_type") or "unknown"))

        if n_used == 0:
            return None

        route_l2 = self._route_l2_loss()
        route_l2_term = args.route_l2_weight * route_l2
        if route_l2_term.requires_grad:
            route_l2_term.backward()
            loss_value += float(route_l2_term.detach().item())

        bal_loss, mean_w = self.model.compute_balance_loss()
        balance_term = args.balance_weight * bal_loss
        if balance_term.requires_grad:
            balance_term.backward()
            loss_value += float(balance_term.detach().item())

        top_pair_type = max(set(pair_types), key=pair_types.count) if pair_types else "none"
        return {
            "offline_loss": loss_value,
            "balance_loss": float(bal_loss.detach().item()),
            "route_l2": float(route_l2.detach().item()),
            "routing_w": float(mean_w),
            "mean_reward": float(np.mean([p[0]["reward"] for p in pairs])),
            "best_reward": float(np.mean([p[0]["reward"] - p[1]["reward"] for p in pairs])),
            "reward_std": float(np.std(neg_rewards)) if neg_rewards else 0.0,
            "mean_abs_adv": float(np.mean(np.abs(margins))) if margins else 0.0,
            "n_candidates": float(n_used),
            "total_tokens": float(total_tokens),
            "row_weight": float(row_weight),
            "gt_logprob": float(np.mean(pos_lps)) if pos_lps else 0.0,
            "sample_logprob": float(np.mean(neg_lps)) if neg_lps else 0.0,
            "any_success": float(np.mean(pair_accs)) if pair_accs else 0.0,
            "pref_margin": float(np.mean(margins)) if margins else 0.0,
            "top_pair_type_hash": float(abs(hash(top_pair_type)) % 1000),
        }

    def train(self) -> None:
        args = self.args
        if self.rank == 0:
            logger.info("=" * 60)
            logger.info("V23 Route Pairwise Preference Training")
            logger.info(f"  model={args.model_path}")
            logger.info(f"  candidates={args.candidate_data}")
            logger.info(f"  rows={len(self.train_dataset)} world_size={self.world_size}")
            logger.info(f"  trainable_mode={args.trainable_mode}")
            logger.info(f"  beta={args.beta} route_l2_weight={args.route_l2_weight}")
            logger.info(f"  negative_pair_types={args.negative_pair_types}")
            logger.info(f"  focus_families={args.focus_families}")
            logger.info("=" * 60)

        for epoch in range(args.num_epochs):
            self.sampler.set_epoch(epoch)
            epoch_metrics: Dict[str, List[float]] = defaultdict(list)
            accum_count = 0
            skipped = 0
            t_epoch = time.time()
            self.optimizer.zero_grad(set_to_none=True)

            for row_idx, row in enumerate(self.train_loader):
                if args.max_steps > 0 and self.global_step >= args.max_steps:
                    break
                t_row = time.time()
                try:
                    metrics = self.train_candidate_group(row)
                except Exception as exc:
                    if self.rank == 0:
                        logger.warning(f"Pairwise row {row_idx} failed: {exc}\n{traceback.format_exc()}")
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
                        if isinstance(value, (int, float)):
                            epoch_metrics[key].append(float(value))

                if (row_idx + 1) % args.gradient_accumulation_steps != 0:
                    continue

                if accum_count == 0:
                    self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
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
                    recent = {key: values[-n_log:] for key, values in epoch_metrics.items()}
                    logger.info(
                        f"E{epoch} S{self.global_step} "
                        f"loss={np.mean(recent.get('offline_loss', [0.0])):.4f} "
                        f"pref={np.mean(recent.get('pref_margin', [0.0])):.3f} "
                        f"acc={np.mean(recent.get('any_success', [0.0])):.3f} "
                        f"pos_lp={np.mean(recent.get('gt_logprob', [0.0])):.3f} "
                        f"neg_lp={np.mean(recent.get('sample_logprob', [0.0])):.3f} "
                        f"route_l2={np.mean(recent.get('route_l2', [0.0])):.6f} "
                        f"pairs={np.mean(recent.get('n_candidates', [0.0])):.1f} "
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
                logger.info("Route pairwise training complete!")
            if self.world_size > 1:
                dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser(description="V23 route-only pairwise preference trainer")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sft_checkpoint", default="")
    parser.add_argument("--resume_from", default="")

    parser.add_argument("--candidate_data", required=True)
    parser.add_argument("--episode_data", required=True)
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
    parser.add_argument("--disable_gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--trainable_mode", choices=["route_only", "route_a_only", "all"], default="route_only")
    parser.add_argument("--max_candidates", type=int, default=0)
    parser.add_argument("--include_gt_positive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_success_positives", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gt_reward", type=float, default=1.0)
    parser.add_argument("--positive_reward_threshold", type=float, default=0.5)
    parser.add_argument("--negative_reward_threshold", type=float, default=0.49)
    parser.add_argument("--require_negative_format", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_negative_format_reward", type=float, default=1.0)
    parser.add_argument("--max_positives_per_row", type=int, default=1)
    parser.add_argument("--max_negatives_per_row", type=int, default=2)
    parser.add_argument("--max_pairs_per_row", type=int, default=2)
    parser.add_argument("--negative_pair_types", nargs="*", default=[])
    parser.add_argument("--focus_families", nargs="*", default=[])
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--logprob_margin", type=float, default=0.0)
    parser.add_argument("--reward_margin_scale", type=float, default=0.0)
    parser.add_argument("--route_l2_weight", type=float, default=0.0)
    parser.add_argument("--weight_clip", type=float, default=2.0)
    parser.add_argument("--append_eos", action="store_true")
    parser.add_argument("--mean_token_logprob", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--lora_lr", type=float, default=1e-5)
    parser.add_argument("--route_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--max_steps", type=int, default=0)

    # Kept for parent-trainer compatibility/logging.
    parser.add_argument("--objective", default="route_pairwise")
    parser.add_argument("--include_gt_candidate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sft_anchor_weight", type=float, default=0.0)
    parser.add_argument("--match_threshold", type=float, default=0.5)

    args = parser.parse_args()
    malformed_pair_types = [value for value in args.negative_pair_types if "->" not in value]
    if malformed_pair_types:
        raise ValueError(
            "Malformed --negative_pair_types values: "
            f"{malformed_pair_types}. Quote values like 'click->click' so the shell "
            "does not treat '>' as output redirection."
        )
    os.makedirs(args.output_dir, exist_ok=True)
    trainer = V23RoutePairwiseTrainer(args)
    trainer.train()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()