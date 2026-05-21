#!/usr/bin/env python3
"""v9 step-1: Reweighted-SFT for cooperative LoRA with (coord, action) credit.

Mirrors train_cooperative.py's CLI exactly so v9 slurm scripts are 1:1 with
the v6.5 ones, plus two additions:

  --credit_file FILE [FILE ...]
      One or more credit_cache*.jsonl files produced by v9/decode_credits.py.
      Each line maps a sample_idx to (w_v, w_a) — the mistake-driven SFT
      weights. Multiple files are merged (later overrides earlier).

  --credit_fallback_w FLOAT
      Per-token weight applied to samples NOT covered by any credit file
      (default 1.0 = plain SFT).

Hard requirements:
  - --coord_routing must be set (credit V-tokens are coord digits, which only
    exist as a separate routing class when coord_routing is on).
  - --bind_weight 0 (the v9 trainer skips the second forward needed for L_bind).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

# Repo root import path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from transformers import (  # noqa: E402
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
)

from train_cooperative import (  # noqa: E402
    CooperativeTrainer,
    CooperativeSaveCallback,
    ThoughtAugmentedDataset,
    collate_fn as _base_collate_fn,
)
from verl.models.cooperative.cooperative_wrapper import (  # noqa: E402
    CooperativeVLMWrapper,
    IMAGE_PAD_ID,
)


# ══════════════════════════════════════════════════════════════════════
# Credit cache loading
# ══════════════════════════════════════════════════════════════════════

def load_credit_cache(paths: List[str]) -> Dict[int, Dict[str, float]]:
    """Load one or more credit_cache*.jsonl files into {sample_idx: credit}.

    Later entries override earlier ones (useful for refreshed caches).
    """
    cache: Dict[int, Dict[str, float]] = {}
    for path in paths:
        if not os.path.exists(path):
            print(f"[v9] WARNING: credit file not found: {path}")
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                idx = rec["sample_idx"]
                cache[idx] = {
                    "w_v": float(rec.get("w_v", 1.0)),
                    "w_a": float(rec.get("w_a", 1.0)),
                    "has_coord": bool(rec.get("has_coord", False)),
                    "coord_correct": rec.get("coord_correct"),
                    "action_correct": bool(rec.get("action_correct", False)),
                }
    return cache


# ══════════════════════════════════════════════════════════════════════
# Credit-aware dataset & collate
# ══════════════════════════════════════════════════════════════════════

class CreditThoughtDataset(ThoughtAugmentedDataset):
    """Adds per-sample (w_v, w_a) to each item."""

    def __init__(
        self,
        jsonl_path: str,
        processor,
        credit_cache: Optional[Dict[int, Dict[str, float]]] = None,
        fallback_w: float = 1.0,
        **kwargs,
    ):
        super().__init__(jsonl_path, processor, **kwargs)
        self.credit_cache = credit_cache or {}
        self.fallback_w = fallback_w

        n_covered = sum(1 for i in range(len(self.data)) if i in self.credit_cache)
        print(f"[CreditThoughtDataset] {n_covered}/{len(self.data)} samples "
              f"have credit entries (fallback w={fallback_w} for the rest)")

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if item is None:
            return None
        credit = self.credit_cache.get(idx)
        if credit is None:
            item["w_v"] = self.fallback_w
            item["w_a"] = self.fallback_w
        else:
            item["w_v"] = credit["w_v"]
            item["w_a"] = credit["w_a"]
        item["sample_idx"] = idx
        return item


def credit_collate_fn(batch):
    """Wrap base collate_fn and stack per-sample credit weights."""
    batch_valid = [b for b in batch if b is not None]
    result = _base_collate_fn(batch)

    if not batch_valid:
        result["w_v"] = torch.ones(1, dtype=torch.float32)
        result["w_a"] = torch.ones(1, dtype=torch.float32)
        return result

    result["w_v"] = torch.tensor(
        [b.get("w_v", 1.0) for b in batch_valid], dtype=torch.float32)
    result["w_a"] = torch.tensor(
        [b.get("w_a", 1.0) for b in batch_valid], dtype=torch.float32)
    return result


# ══════════════════════════════════════════════════════════════════════
# Credit-aware Trainer
# ══════════════════════════════════════════════════════════════════════

def _build_routing_mask(wrapper: CooperativeVLMWrapper, input_ids, labels):
    """Mirror wrapper.forward step 1 to produce the routing mask."""
    if wrapper.routing_mode in ("merge", "learned"):
        return None
    if wrapper.num_agents >= 3:
        return wrapper._build_3way_mask(input_ids)
    if wrapper.coop_reasoning_alpha > 0:
        token_mask = torch.zeros(
            input_ids.shape, dtype=torch.float32, device=input_ids.device)
        token_mask[input_ids == IMAGE_PAD_ID] = 1.0
        if wrapper.coord_routing:
            wrapper._mark_coord_tokens(token_mask, input_ids)
        if labels is not None:
            is_assistant = (labels != -100)
            is_v = (token_mask > 0.5)
            is_reasoning = is_assistant & ~is_v
            token_mask[is_reasoning] = wrapper.coop_reasoning_alpha
        return token_mask
    token_mask = (input_ids == IMAGE_PAD_ID)
    if wrapper.coord_routing:
        wrapper._mark_coord_tokens(token_mask, input_ids)
    return token_mask


class CreditCooperativeTrainer(CooperativeTrainer):
    """Subclass of CooperativeTrainer that applies per-sample (w_v, w_a)
    weights to the coord-token vs non-coord-token components of CE.

    Hard-routing split:
      V-tokens: assistant tokens whose input_id is a coordinate/bbox digit
                (detected by wrapper._mark_coord_tokens).
      A-tokens: assistant tokens that are NOT V-tokens.

    Loss form (per-token weighted, mean over assistant tokens):
      weight_per_tok[b, t] = w_v[b]  if t is a V-token of sample b
                             w_a[b]  if t is an A-token of sample b
                             0       else (non-assistant)

      L_act = sum_(b,t) (CE(b,t) * weight_per_tok[b,t]) / count_assistant_tokens

    When w_v=w_a=1.0 we recover vanilla mean-over-assistant CE (so v9 with
    fallback weight=1.0 and an empty credit cache is identical to baseline SFT).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Extra credit-specific running averages.
        self._v_loss_sum = 0.0
        self._a_loss_sum = 0.0
        self._v_loss_count = 0
        self._a_loss_count = 0
        self._w_v_sum = 0.0
        self._w_a_sum = 0.0
        self._credit_step_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Unwrap DDP if present.
        wrapper: CooperativeVLMWrapper = (
            model.module if hasattr(model, "module") else model
        )
        device = wrapper.base_model.device

        # Pop credit fields and bookkeeping fields.
        w_v = inputs.pop("w_v").to(device)
        w_a = inputs.pop("w_a").to(device)
        gt_coords = inputs.pop("gt_coords", None)
        orig_sizes = inputs.pop("orig_sizes", None)
        labels = inputs.pop("labels")
        input_ids = inputs["input_ids"]

        # Dummy batch shortcut.
        if labels.numel() == 0 or (labels == -100).all():
            dummy = torch.zeros((), device=device, dtype=torch.float32,
                                requires_grad=True)
            diagnostics = {"L_act": dummy.detach(), "L_bind": dummy.detach(),
                           "loss": dummy.detach(), "bind_samples": 0}
            return (dummy, diagnostics) if return_outputs else dummy

        # Step 1: set token mask (mirrors wrapper.forward step 1).
        token_mask = _build_routing_mask(wrapper, input_ids, labels)
        wrapper._set_token_mask(token_mask)

        # Step 2: forward base model WITHOUT labels so we get raw logits.
        outputs = wrapper.base_model(
            **inputs,
            labels=None,
            output_hidden_states=False,
            return_dict=True,
        )
        logits = outputs.logits  # [B, S, V]

        # Step 3: causal shift.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        V = shift_logits.size(-1)
        ce_per_tok = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)  # [B, S-1]

        # Step 4: coord-token mask aligned to shift_labels.
        # coord_mask[t] is True iff input_ids[t] is a coord digit. Since
        # shift_labels[t] = input_ids[t+1], we use coord_mask_full[..., 1:].
        coord_mask_full = torch.zeros_like(input_ids, dtype=torch.bool)
        wrapper._mark_coord_tokens(coord_mask_full, input_ids)
        coord_mask_shift = coord_mask_full[..., 1:].contiguous()

        is_assistant = (shift_labels != -100)
        is_v_token = is_assistant & coord_mask_shift
        is_a_token = is_assistant & ~coord_mask_shift

        # Step 5: per-token credit weighting.
        wv_b = w_v.unsqueeze(-1)  # [B, 1]
        wa_b = w_a.unsqueeze(-1)
        weight_per_tok = (wv_b * is_v_token.float() +
                          wa_b * is_a_token.float())  # [B, S-1]

        weighted_ce_sum = (ce_per_tok * weight_per_tok).sum()
        n_assistant = is_assistant.float().sum().clamp(min=1.0)
        L_act = weighted_ce_sum / n_assistant

        # Step 6: balance loss (learned routing only). We skip bind for v9.
        L_bind = torch.zeros((), device=L_act.device)
        L_balance = torch.zeros((), device=L_act.device)
        if wrapper.routing_mode == "learned" and wrapper.balance_weight > 0:
            balance_terms = []
            for m in wrapper.coop_modules:
                if m._last_router_w is None:
                    continue
                w = m._last_router_w
                mean_w = w.mean()
                eps = 1e-6
                neg_entropy = (mean_w * torch.log(mean_w + eps) +
                               (1 - mean_w) * torch.log(1 - mean_w + eps))
                balance_terms.append(neg_entropy)
            if balance_terms:
                L_balance = torch.stack(balance_terms).mean()

        loss = L_act + wrapper.balance_weight * L_balance

        # Per-component diagnostics (unweighted, per-sample).
        v_count = is_v_token.float().sum(dim=-1)
        a_count = is_a_token.float().sum(dim=-1)
        v_loss_sample = (ce_per_tok * is_v_token.float()).sum(dim=-1) / v_count.clamp(min=1.0)
        a_loss_sample = (ce_per_tok * is_a_token.float()).sum(dim=-1) / a_count.clamp(min=1.0)
        v_has = (v_count > 0)
        a_has = (a_count > 0)

        if v_has.any():
            self._v_loss_sum += v_loss_sample[v_has].mean().item()
            self._v_loss_count += 1
        if a_has.any():
            self._a_loss_sum += a_loss_sample[a_has].mean().item()
            self._a_loss_count += 1
        self._w_v_sum += w_v.mean().item()
        self._w_a_sum += w_a.mean().item()
        self._credit_step_count += 1

        # Update parent's tracker so its log() emits ce_loss.
        act_val = L_act.item()
        if act_val == act_val:
            self._act_loss_sum += act_val
            self._act_loss_count += 1

        diagnostics = {
            "L_act": L_act.detach(),
            "L_bind": L_bind.detach(),
            "loss": loss.detach(),
            "bind_samples": 0,
        }
        if wrapper.routing_mode == "learned":
            diagnostics["L_balance"] = L_balance.detach()
            diagnostics["mean_router_w"] = 0.0

        return (loss, diagnostics) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if self._credit_step_count > 0:
            if self._v_loss_count > 0:
                logs["v_loss"] = round(self._v_loss_sum / self._v_loss_count, 6)
            if self._a_loss_count > 0:
                logs["a_loss"] = round(self._a_loss_sum / self._a_loss_count, 6)
            logs["w_v_mean"] = round(self._w_v_sum / self._credit_step_count, 4)
            logs["w_a_mean"] = round(self._w_a_sum / self._credit_step_count, 4)
        super().log(logs, *args, **kwargs)
        self._v_loss_sum = 0.0
        self._a_loss_sum = 0.0
        self._v_loss_count = 0
        self._a_loss_count = 0
        self._w_v_sum = 0.0
        self._w_a_sum = 0.0
        self._credit_step_count = 0


# ══════════════════════════════════════════════════════════════════════
# main()
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="v9 step-1 Reweighted-SFT for cooperative LoRA.")
    # ── v9 additions ──
    parser.add_argument("--credit_file", nargs="+", required=True,
                        help="Credit cache file(s) from v9/decode_credits.py")
    parser.add_argument("--credit_fallback_w", type=float, default=1.0,
                        help="Weight for samples not in the credit cache")

    # ── Mirror train_cooperative.py CLI ──
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--val_data", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--lora_r", type=int, default=256)
    parser.add_argument("--lora_alpha", type=int, default=512)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"])
    parser.add_argument("--num_agents", type=int, default=2, choices=[2, 3])
    parser.add_argument("--soft_routing", action="store_true")
    parser.add_argument("--init_sep", type=float, default=0.0)
    parser.add_argument("--cooperative_comm", action="store_true",
                        help="Enable cross-agent communication gates (v6+)")
    parser.add_argument("--gate_init", type=float, default=0.0)
    parser.add_argument("--gate_type", type=str, default="tanh",
                        choices=["sigmoid", "tanh"])
    parser.add_argument("--gate_lr_multiplier", type=float, default=100.0)
    parser.add_argument("--gate_weight_decay", type=float, default=0.0)
    parser.add_argument("--routing_mode", type=str, default="hard",
                        choices=["hard", "merge", "learned"])
    parser.add_argument("--coord_routing", action="store_true",
                        help="REQUIRED for v9: route coord digits to LoRA_V")
    parser.add_argument("--coop_reasoning_alpha", type=float, default=0.0)
    parser.add_argument("--balance_weight", type=float, default=0.0)
    parser.add_argument("--shared_b", action="store_true")

    parser.add_argument("--bind_weight", type=float, default=0.0,
                        help="MUST be 0 for v9")
    parser.add_argument("--bind_layer", type=int, default=27)
    parser.add_argument("--bind_temperature", type=float, default=0.1)

    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--image_max_pixels", type=int, default=0)

    parser.add_argument("--num_epochs", type=float, default=4.0)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--resume_coop_checkpoint", type=str, default=None,
                        help="Cooperative LoRA checkpoint to start from "
                             "(typically the v6.5 ep4 ckpt)")
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()

    # ── Hard checks ──
    if args.bind_weight > 0:
        raise ValueError("v9 trainer requires --bind_weight 0 "
                         "(skips the second forward needed for L_bind).")
    if not args.coord_routing:
        raise ValueError("v9 hard-routing credit requires --coord_routing. "
                         "Coord-digit tokens must be routed to LoRA_V for "
                         "w_v to have meaning.")

    # ── Load credit cache ──
    print(f"[v9] Loading credit cache from {args.credit_file}")
    credit_cache = load_credit_cache(args.credit_file)
    print(f"[v9] Credit cache: {len(credit_cache)} entries")

    # ── Load base model ──
    print(f"[v9] Loading base model from {args.model_path} ...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    print(f"[v9] Wrapping cooperative LoRA "
          f"(r={args.lora_r}, alpha={args.lora_alpha}, "
          f"comm={args.cooperative_comm}, gate={args.gate_type}, "
          f"routing={args.routing_mode}, coord_routing={args.coord_routing}) ...")
    model = CooperativeVLMWrapper(
        base_model=base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bind_weight=0.0,
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
        coop_reasoning_alpha=args.coop_reasoning_alpha,
        balance_weight=args.balance_weight,
        shared_B=args.shared_b,
    )

    if args.resume_coop_checkpoint:
        print(f"[v9] Loading cooperative checkpoint from "
              f"{args.resume_coop_checkpoint} ...")
        model.load_cooperative_checkpoint(args.resume_coop_checkpoint)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[v9] Trainable: {trainable:,} / {total:,} "
          f"({100*trainable/total:.3f}%)")

    # ── Processor + datasets ──
    proc_kwargs = dict(trust_remote_code=True)
    if args.image_max_pixels > 0:
        proc_kwargs["max_pixels"] = args.image_max_pixels
        proc_kwargs["min_pixels"] = min(args.image_max_pixels // 4, 200704)
    processor = AutoProcessor.from_pretrained(args.model_path, **proc_kwargs)

    train_dataset = CreditThoughtDataset(
        args.train_data, processor,
        credit_cache=credit_cache,
        fallback_w=args.credit_fallback_w,
        max_samples=args.max_train_samples,
        max_length=args.max_length,
    )
    val_dataset = CreditThoughtDataset(
        args.val_data, processor,
        credit_cache=credit_cache,
        fallback_w=args.credit_fallback_w,
        max_samples=500,
        max_length=args.max_length,
    )
    print(f"[v9] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # ── Config dump ──
    os.makedirs(args.output_dir, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        with open(os.path.join(args.output_dir, "v9_config.json"), "w") as f:
            json.dump({
                "version": "v9-step1-reweighted-sft",
                "credit_files": args.credit_file,
                "credit_cache_size": len(credit_cache),
                "credit_fallback_w": args.credit_fallback_w,
                "cooperative_comm": args.cooperative_comm,
                "coord_routing": args.coord_routing,
                "routing_mode": args.routing_mode,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "num_epochs": args.num_epochs,
                "learning_rate": args.learning_rate,
                "resume_coop_checkpoint": args.resume_coop_checkpoint,
            }, f, indent=2)

    # ── TrainingArguments (mirrors train_cooperative.py) ──
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
    trainer = CreditCooperativeTrainer(
        cooperative_model=model,
        tokenizer=processor.tokenizer,
        gate_lr_multiplier=args.gate_lr_multiplier,
        gate_weight_decay=args.gate_weight_decay,
        diversity_loss_weight=0.0,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=credit_collate_fn,
        callbacks=[CooperativeSaveCallback()],
    )

    # Resume from HF Trainer state if a parent dir holds trainer_state.json.
    resume_ckpt = None
    if args.resume_coop_checkpoint:
        parent = os.path.dirname(args.resume_coop_checkpoint.rstrip("/"))
        if os.path.exists(os.path.join(parent, "trainer_state.json")):
            resume_ckpt = parent

    trainer.train(resume_from_checkpoint=resume_ckpt)

    model.save_cooperative_checkpoint(os.path.join(args.output_dir, "final"))
    print(f"[v9] Training complete. Saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
