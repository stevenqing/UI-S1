#!/usr/bin/env python3
"""Phase 0 gate for representation-conflict-gated GUI-360 multi-agent work.

This script does not build agents. It measures whether WHAT and WHERE losses
produce meaningfully opposing gradients on the GUI-360 full-SFT model, and it
reconstructs a far/near grounding separability check from an existing eval file.

Gate policy:
- CONFLICT CONFIRMED: stable negative cosine on shared parameters/projections.
- NO CONFLICT: near-zero cosine.
- ALIGNED: positive cosine.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from v13_gui_360.reward import parse_action_from_text  # noqa: E402
from v15_gui_360.train_trajectory_gspo import V15TrajectoryGSPOTrainer  # noqa: E402
from v23_visual_transition.prepare_offline_data import format_action_for_history  # noqa: E402
from v23_visual_transition.train_offline_grpo import build_eval_style_messages  # noqa: E402
from v23_visual_transition.train_where_what_routed_sft import token_role_labels  # noqa: E402


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class BucketTotals:
    dot: float = 0.0
    what_norm_sq: float = 0.0
    where_norm_sq: float = 0.0
    n: int = 0

    def add(self, what_grad: torch.Tensor, where_grad: torch.Tensor) -> None:
        self.dot += float(torch.dot(what_grad, where_grad))
        self.what_norm_sq += float(torch.dot(what_grad, what_grad))
        self.where_norm_sq += float(torch.dot(where_grad, where_grad))
        self.n += 1

    def cosine(self) -> float:
        denom = math.sqrt(self.what_norm_sq) * math.sqrt(self.where_norm_sq)
        return self.dot / (denom + 1e-12)

    def ratio(self) -> float:
        return math.sqrt(self.what_norm_sq) / (math.sqrt(self.where_norm_sq) + 1e-12)

    def as_dict(self) -> Dict[str, float]:
        return {
            "cosine": self.cosine(),
            "what_norm": math.sqrt(self.what_norm_sq),
            "where_norm": math.sqrt(self.where_norm_sq),
            "what_where_norm_ratio": self.ratio(),
            "n_param_tensors": self.n,
        }


@dataclass
class RunningScalar:
    values: List[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        if math.isfinite(value):
            self.values.append(float(value))

    def as_dict(self) -> Dict[str, float]:
        if not self.values:
            return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        mean = sum(self.values) / len(self.values)
        var = sum((value - mean) ** 2 for value in self.values) / len(self.values)
        return {
            "n": len(self.values),
            "mean": mean,
            "std": math.sqrt(max(var, 0.0)),
            "min": min(self.values),
            "max": max(self.values),
        }


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


def read_episode_jsonl(path: str) -> Dict[str, Dict[str, Any]]:
    episodes: Dict[str, Dict[str, Any]] = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            episodes[str(episode.get("episode_id"))] = episode
    return episodes


def build_target_text(row: Dict[str, Any]) -> str:
    full_tool_call = row.get("full_tool_call") or {}
    if full_tool_call:
        payload = json.dumps(full_tool_call, ensure_ascii=False, indent=2)
        return f"<tool_call>\n{payload}\n</tool_call>"
    return ""


def layer_and_projection(param_name: str) -> Optional[Tuple[int, str]]:
    match = re.search(r"model\.layers\.(\d+)\.", param_name)
    if match is None:
        return None
    layer_idx = int(match.group(1))
    for projection in PROJECTIONS:
        if f".{projection}." in param_name:
            return layer_idx, projection
    return None


def store_target_grads(model) -> Dict[str, Tuple[int, str, torch.Tensor]]:
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        parsed = layer_and_projection(name)
        if parsed is None:
            continue
        layer_idx, projection = parsed
        grads[name] = (layer_idx, projection, param.grad.detach().float().cpu().flatten())
    return grads


def accumulate_grad_cosines(
    what_grads: Dict[str, Tuple[int, str, torch.Tensor]],
    where_grads: Dict[str, Tuple[int, str, torch.Tensor]],
    layer_totals: Dict[int, BucketTotals],
    projection_totals: Dict[str, BucketTotals],
    layer_projection_totals: Dict[Tuple[int, str], BucketTotals],
    global_total: BucketTotals,
) -> Dict[str, float]:
    batch_global = BucketTotals()
    for name, (layer_idx, projection, what_grad) in what_grads.items():
        if name not in where_grads:
            continue
        _, _, where_grad = where_grads[name]
        global_total.add(what_grad, where_grad)
        layer_totals[layer_idx].add(what_grad, where_grad)
        projection_totals[projection].add(what_grad, where_grad)
        layer_projection_totals[(layer_idx, projection)].add(what_grad, where_grad)
        batch_global.add(what_grad, where_grad)
    return batch_global.as_dict()


def response_token_loss(
    model,
    fwd_kwargs: Dict[str, torch.Tensor],
    ids: torch.Tensor,
    prompt_len: int,
    role_labels: torch.Tensor,
    role_value: int,
    pad_id: int,
) -> Optional[torch.Tensor]:
    outputs = model(**fwd_kwargs)
    logits = outputs.logits
    resp_logits = logits[:, prompt_len - 1:-1, :]
    resp_labels = ids[:, prompt_len:]
    if resp_logits.shape[1] != role_labels.shape[0]:
        usable = min(resp_logits.shape[1], role_labels.shape[0])
        resp_logits = resp_logits[:, :usable, :]
        resp_labels = resp_labels[:, :usable]
        role_labels = role_labels[:usable]
    token_loss = F.cross_entropy(
        resp_logits.reshape(-1, resp_logits.shape[-1]),
        resp_labels.reshape(-1),
        reduction="none",
    ).view_as(resp_labels)
    mask = (role_labels.unsqueeze(0) == role_value) & (resp_labels != pad_id)
    if not mask.any():
        return None
    return token_loss[mask].mean()


def prepare_row(processor, row: Dict[str, Any], device: str) -> Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor, int, torch.Tensor]]:
    image_path = row.get("image")
    if not image_path or not os.path.exists(image_path):
        return None
    target_text = build_target_text(row)
    if not target_text:
        return None

    image = Image.open(image_path).convert("RGB")
    messages = build_eval_style_messages(row["goal"], row.get("history", []), image_path)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_inputs = processor(text=[prompt_text], images=[image], return_tensors="pt", padding=False)
    prompt_inputs = {key: value.to(device) for key, value in prompt_inputs.items()}
    prompt_len = int(prompt_inputs["input_ids"].shape[1])

    response_ids, role_labels = token_role_labels(processor.tokenizer, target_text)
    if (role_labels == 0).sum().item() == 0 or (role_labels == 1).sum().item() == 0:
        return None
    response_ids = response_ids.to(device)
    role_labels = role_labels.to(device)
    ids = torch.cat([prompt_inputs["input_ids"][0], response_ids], dim=0).unsqueeze(0)
    attention_mask = torch.ones_like(ids)
    fwd_kwargs = {"input_ids": ids, "attention_mask": attention_mask}
    for key in ("pixel_values", "image_grid_thw"):
        if key in prompt_inputs:
            fwd_kwargs[key] = prompt_inputs[key]
    return fwd_kwargs, ids, prompt_len, role_labels


def load_model(args):
    processor = AutoProcessor.from_pretrained(args.model_path)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if args.image_max_pixels > 0:
        processor.image_processor.max_pixels = args.image_max_pixels
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    V15TrajectoryGSPOTrainer._patch_legacy_mrope_config(config)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(args.device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.train()
    return processor, model


def get_transformer_layers(model):
    vlm = model.model
    if hasattr(vlm, "language_model"):
        return vlm.language_model.layers
    return vlm.layers


def register_projection_hooks(model, captures: Dict[str, torch.Tensor], projections: Iterable[str]):
    hooks = []
    projection_set = set(projections)
    for layer_idx, layer in enumerate(get_transformer_layers(model)):
        modules = {}
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            if hasattr(layer.self_attn, name):
                modules[name] = getattr(layer.self_attn, name)
        for name in ("gate_proj", "up_proj", "down_proj"):
            if hasattr(layer.mlp, name):
                modules[name] = getattr(layer.mlp, name)
        for projection, module in modules.items():
            if projection not in projection_set:
                continue
            key = f"L{layer_idx:02d}.{projection}"

            def make_hook(capture_key: str):
                def hook(_module, _inputs, output):
                    captures[capture_key] = output
                return hook

            hooks.append(module.register_forward_hook(make_hook(key)))
    return hooks


def split_locus(key: str) -> Tuple[int, str]:
    layer_text, projection = key.split(".", 1)
    return int(layer_text[1:]), projection


def run_projection_gradient_conflict(args) -> Dict[str, Any]:
    rows = read_jsonl(args.paired_data, args.max_rows)
    processor, model = load_model(args)
    pad_id = processor.tokenizer.pad_token_id
    captures: Dict[str, torch.Tensor] = {}
    hooks = register_projection_hooks(model, captures, args.projections)

    global_total = BucketTotals()
    layer_totals: Dict[int, BucketTotals] = defaultdict(BucketTotals)
    projection_totals: Dict[str, BucketTotals] = defaultdict(BucketTotals)
    layer_projection_totals: Dict[str, BucketTotals] = defaultdict(BucketTotals)
    batch_cosines = RunningScalar()
    batch_records = []
    skipped = 0
    used = 0

    for row_idx, row in enumerate(rows):
        if used >= args.num_batches:
            break
        prepared = prepare_row(processor, row, args.device)
        if prepared is None:
            skipped += 1
            continue
        fwd_kwargs, ids, prompt_len, role_labels = prepared

        captures.clear()
        what_loss = response_token_loss(model, fwd_kwargs, ids, prompt_len, role_labels, 1, pad_id)
        if what_loss is None:
            skipped += 1
            continue
        what_keys = list(captures.keys())
        what_targets = [captures[key] for key in what_keys]
        what_grads = torch.autograd.grad(what_loss, what_targets, retain_graph=False, allow_unused=True)
        what_by_key = {
            key: grad.detach().to(device="cpu", dtype=torch.float16).flatten()
            for key, grad in zip(what_keys, what_grads)
            if grad is not None
        }
        del what_targets, what_grads, what_loss
        torch.cuda.empty_cache()

        captures.clear()
        where_loss = response_token_loss(model, fwd_kwargs, ids, prompt_len, role_labels, 0, pad_id)
        if where_loss is None:
            skipped += 1
            continue
        where_keys = list(captures.keys())
        where_targets = [captures[key] for key in where_keys]
        where_grads = torch.autograd.grad(where_loss, where_targets, retain_graph=False, allow_unused=True)

        batch_total = BucketTotals()
        for key, where_grad in zip(where_keys, where_grads):
            what_grad = what_by_key.get(key)
            if what_grad is None or where_grad is None:
                continue
            where_cpu = where_grad.detach().to(device="cpu", dtype=torch.float16).flatten()
            what_float = what_grad.float()
            where_float = where_cpu.float()
            layer, projection = split_locus(key)
            global_total.add(what_float, where_float)
            batch_total.add(what_float, where_float)
            layer_totals[layer].add(what_float, where_float)
            projection_totals[projection].add(what_float, where_float)
            layer_projection_totals[key].add(what_float, where_float)

        batch_cosines.add(batch_total.cosine())
        batch_records.append({
            "row_idx": row_idx,
            "episode_id": row.get("episode_id"),
            "step_idx": row.get("step_idx"),
            "what_loss": None,
            "where_loss": float(where_loss.detach().item()),
            "global_cosine": batch_total.cosine(),
        })
        used += 1
        if used % args.log_every == 0:
            print(f"processed {used}/{args.num_batches} batches; last projection cosine={batch_total.cosine():.6f}")
        del where_targets, where_grads, where_loss, what_by_key, prepared
        torch.cuda.empty_cache()

    for hook in hooks:
        hook.remove()
    del model
    torch.cuda.empty_cache()

    return {
        "mode": "projection",
        "num_batches_requested": args.num_batches,
        "num_batches_used": used,
        "num_rows_scanned": min(len(rows), args.max_rows or len(rows)),
        "num_rows_skipped": skipped,
        "global": global_total.as_dict(),
        "batch_cosine": batch_cosines.as_dict(),
        "layers": {str(layer): total.as_dict() for layer, total in sorted(layer_totals.items())},
        "projections": {proj: total.as_dict() for proj, total in sorted(projection_totals.items())},
        "layer_projections": {key: total.as_dict() for key, total in sorted(layer_projection_totals.items())},
        "batches": batch_records,
    }


def hidden_role_losses(model, fwd_kwargs, ids, prompt_len, role_labels, pad_id):
    outputs = model(**fwd_kwargs, output_hidden_states=True)
    logits = outputs.logits
    resp_logits = logits[:, prompt_len - 1:-1, :]
    resp_labels = ids[:, prompt_len:]
    if resp_logits.shape[1] != role_labels.shape[0]:
        usable = min(resp_logits.shape[1], role_labels.shape[0])
        resp_logits = resp_logits[:, :usable, :]
        resp_labels = resp_labels[:, :usable]
        role_labels = role_labels[:usable]
    token_loss = F.cross_entropy(
        resp_logits.reshape(-1, resp_logits.shape[-1]),
        resp_labels.reshape(-1),
        reduction="none",
    ).view_as(resp_labels).squeeze(0)
    what_mask = (role_labels == 1) & (resp_labels.squeeze(0) != pad_id)
    where_mask = (role_labels == 0) & (resp_labels.squeeze(0) != pad_id)
    if not what_mask.any() or not where_mask.any():
        return None
    return token_loss[what_mask].mean(), token_loss[where_mask].mean(), outputs.hidden_states


def run_hidden_gradient_conflict(args) -> Dict[str, Any]:
    rows = read_jsonl(args.paired_data, args.max_rows)
    processor, model = load_model(args)
    pad_id = processor.tokenizer.pad_token_id

    global_total = BucketTotals()
    layer_totals: Dict[int, BucketTotals] = defaultdict(BucketTotals)
    batch_cosines = RunningScalar()
    batch_records = []
    skipped = 0
    used = 0

    for row_idx, row in enumerate(rows):
        if used >= args.num_batches:
            break
        prepared = prepare_row(processor, row, args.device)
        if prepared is None:
            skipped += 1
            continue
        fwd_kwargs, ids, prompt_len, role_labels = prepared
        model.zero_grad(set_to_none=True)
        result = hidden_role_losses(model, fwd_kwargs, ids, prompt_len, role_labels, pad_id)
        if result is None:
            skipped += 1
            continue
        what_loss, where_loss, hidden_states = result
        retain_hidden = [state for state in hidden_states[1:]]
        for state in retain_hidden:
            state.retain_grad()
        what_grads = torch.autograd.grad(what_loss, retain_hidden, retain_graph=True, allow_unused=True)
        where_grads = torch.autograd.grad(where_loss, retain_hidden, retain_graph=False, allow_unused=True)

        batch_total = BucketTotals()
        for layer_idx, (what_grad, where_grad) in enumerate(zip(what_grads, where_grads)):
            if what_grad is None or where_grad is None:
                continue
            wg = what_grad.detach().float().reshape(-1).cpu()
            hg = where_grad.detach().float().reshape(-1).cpu()
            layer_totals[layer_idx].add(wg, hg)
            global_total.add(wg, hg)
            batch_total.add(wg, hg)
        batch_cosines.add(batch_total.cosine())
        batch_records.append({
            "row_idx": row_idx,
            "episode_id": row.get("episode_id"),
            "step_idx": row.get("step_idx"),
            "what_loss": float(what_loss.detach().item()),
            "where_loss": float(where_loss.detach().item()),
            "global_cosine": batch_total.cosine(),
        })
        used += 1
        if used % args.log_every == 0:
            print(f"processed {used}/{args.num_batches} batches; last hidden cosine={batch_total.cosine():.6f}")
        del result, what_loss, where_loss, hidden_states, retain_hidden, what_grads, where_grads
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    layer_summary = {str(layer): total.as_dict() for layer, total in sorted(layer_totals.items())}
    return {
        "mode": "hidden",
        "num_batches_requested": args.num_batches,
        "num_batches_used": used,
        "num_rows_scanned": min(len(rows), args.max_rows or len(rows)),
        "num_rows_skipped": skipped,
        "global": global_total.as_dict(),
        "batch_cosine": batch_cosines.as_dict(),
        "layers": layer_summary,
        "projections": {},
        "layer_projections": {},
        "batches": batch_records,
    }


def run_gradient_conflict(args) -> Dict[str, Any]:
    rows = read_jsonl(args.paired_data, args.max_rows)
    processor, model = load_model(args)
    pad_id = processor.tokenizer.pad_token_id

    layer_totals: Dict[int, BucketTotals] = defaultdict(BucketTotals)
    projection_totals: Dict[str, BucketTotals] = defaultdict(BucketTotals)
    layer_projection_totals: Dict[Tuple[int, str], BucketTotals] = defaultdict(BucketTotals)
    global_total = BucketTotals()
    batch_cosines = RunningScalar()
    batch_records = []
    skipped = 0
    used = 0

    for row_idx, row in enumerate(rows):
        if used >= args.num_batches:
            break
        prepared = prepare_row(processor, row, args.device)
        if prepared is None:
            skipped += 1
            continue
        fwd_kwargs, ids, prompt_len, role_labels = prepared

        model.zero_grad(set_to_none=True)
        what_loss = response_token_loss(model, fwd_kwargs, ids, prompt_len, role_labels, 1, pad_id)
        if what_loss is None:
            skipped += 1
            continue
        what_loss.backward()
        what_grads = store_target_grads(model)

        model.zero_grad(set_to_none=True)
        where_loss = response_token_loss(model, fwd_kwargs, ids, prompt_len, role_labels, 0, pad_id)
        if where_loss is None:
            skipped += 1
            continue
        where_loss.backward()
        where_grads = store_target_grads(model)

        batch = accumulate_grad_cosines(
            what_grads,
            where_grads,
            layer_totals,
            projection_totals,
            layer_projection_totals,
            global_total,
        )
        batch_cosines.add(batch["cosine"])
        batch_records.append({
            "row_idx": row_idx,
            "episode_id": row.get("episode_id"),
            "step_idx": row.get("step_idx"),
            "what_loss": float(what_loss.detach().item()),
            "where_loss": float(where_loss.detach().item()),
            "global_cosine": batch["cosine"],
        })
        used += 1
        if used % args.log_every == 0:
            print(f"processed {used}/{args.num_batches} batches; last cosine={batch['cosine']:.6f}")

    del model
    torch.cuda.empty_cache()

    layer_summary = {str(layer): total.as_dict() for layer, total in sorted(layer_totals.items())}
    projection_summary = {projection: total.as_dict() for projection, total in sorted(projection_totals.items())}
    layer_projection_summary = {
        f"L{layer:02d}.{projection}": total.as_dict()
        for (layer, projection), total in sorted(layer_projection_totals.items())
    }

    return {
        "num_batches_requested": args.num_batches,
        "num_batches_used": used,
        "num_rows_scanned": min(len(rows), args.max_rows or len(rows)),
        "num_rows_skipped": skipped,
        "global": global_total.as_dict(),
        "batch_cosine": batch_cosines.as_dict(),
        "layers": layer_summary,
        "projections": projection_summary,
        "layer_projections": layer_projection_summary,
        "batches": batch_records,
    }


def normalize_action_type(action_type: Any) -> str:
    aliases = {
        "left_click": "click",
        "tap": "click",
        "double_click": "click",
        "drag": "swipe",
        "scroll": "swipe",
        "wheel_mouse_input": "swipe",
        "input": "type",
    }
    value = str(action_type or "").strip().lower()
    return aliases.get(value, value)


def coord_distance_px(pred_action: Optional[Dict[str, Any]], gt_action: Dict[str, Any]) -> Optional[float]:
    if not pred_action:
        return None
    pred = pred_action.get("coordinate")
    gt = gt_action.get("coordinate")
    if pred is None or gt is None:
        return None
    try:
        return math.sqrt((float(pred[0]) - float(gt[0])) ** 2 + (float(pred[1]) - float(gt[1])) ** 2)
    except (TypeError, ValueError, IndexError):
        return None


def first_bad_step(eval_ep: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for step in eval_ep.get("steps", []):
        if not step.get("success"):
            return step
    return None


def run_failure_separability(args) -> Dict[str, Any]:
    with open(args.eval_results) as handle:
        eval_results = json.load(handle)
    episodes = read_episode_jsonl(args.episode_data)

    counts = Counter()
    distance_bins = Counter()
    examples = []
    total_failed = 0
    for episode_id, eval_ep in eval_results.items():
        if eval_ep.get("task_success"):
            continue
        bad = first_bad_step(eval_ep)
        if bad is None:
            continue
        total_failed += 1
        gt_type = normalize_action_type(bad.get("gt_type"))
        pred_type = normalize_action_type(bad.get("pred_type"))
        if gt_type != pred_type:
            counts[f"type_mismatch:{gt_type}->{pred_type}"] += 1
            continue
        if gt_type != "click":
            counts[f"same_type_non_click:{gt_type}"] += 1
            continue

        episode = episodes.get(str(episode_id))
        step_idx = int(bad.get("step_idx", 0))
        gt_action = {}
        if episode and step_idx < len(episode.get("steps", [])):
            gt_action = episode["steps"][step_idx].get("action", {}) or {}
        pred_action = bad.get("pred_action") or parse_action_from_text(bad.get("pred_text", "") or "")
        dist = coord_distance_px(pred_action, gt_action)
        if dist is None:
            counts["grounding_missing_coord"] += 1
            continue
        counts["grounding"] += 1
        if dist <= args.near_px:
            counts["near_miss"] += 1
            distance_bins["near"] += 1
        elif dist >= args.far_px:
            counts["far_miss"] += 1
            distance_bins["far"] += 1
        else:
            counts["mid_miss"] += 1
            distance_bins["mid"] += 1
        if len(examples) < 20:
            examples.append({
                "episode_id": episode_id,
                "step_idx": step_idx,
                "distance_px": dist,
                "bin": "near" if dist <= args.near_px else ("far" if dist >= args.far_px else "mid"),
                "goal": eval_ep.get("goal", ""),
                "pred_action": pred_action,
                "gt_action": gt_action,
            })

    grounding = counts["grounding"]
    return {
        "eval_results": args.eval_results,
        "episode_data": args.episode_data,
        "near_px": args.near_px,
        "far_px": args.far_px,
        "total_failed_episodes": total_failed,
        "counts": dict(counts),
        "grounding_share_of_failures": grounding / max(total_failed, 1),
        "near_share_of_grounding": counts["near_miss"] / max(grounding, 1),
        "far_share_of_grounding": counts["far_miss"] / max(grounding, 1),
        "mid_share_of_grounding": counts["mid_miss"] / max(grounding, 1),
        "support_ok": {
            "grounding_n_ge_30": grounding >= 30,
            "near_n_ge_30": counts["near_miss"] >= 30,
            "far_n_ge_30": counts["far_miss"] >= 30,
        },
        "examples": examples,
    }


def gate_verdict(gradient: Dict[str, Any], stable_negative_threshold: float, near_zero_threshold: float) -> Dict[str, Any]:
    cosine = float(gradient["global"]["cosine"])
    batch_mean = float(gradient["batch_cosine"]["mean"])
    batch_std = float(gradient["batch_cosine"]["std"])
    layers = gradient.get("layers", {})
    layer_values = [float(value["cosine"]) for value in layers.values()]
    min_layer = min(layer_values) if layer_values else cosine
    layer_proj = gradient.get("layer_projections", {})
    min_locus_key = None
    min_locus_val = None
    for key, value in layer_proj.items():
        val = float(value["cosine"])
        if min_locus_val is None or val < min_locus_val:
            min_locus_key = key
            min_locus_val = val

    stable_sign = batch_mean < 0 and abs(batch_mean) > batch_std
    if cosine <= stable_negative_threshold and stable_sign:
        verdict = "CONFLICT CONFIRMED"
        rationale = "global cosine is meaningfully negative and batch sign is stable"
    elif abs(cosine) <= near_zero_threshold:
        verdict = "NO CONFLICT"
        rationale = "global cosine is near zero; objectives appear orthogonal rather than opposing"
    elif cosine > near_zero_threshold:
        verdict = "ALIGNED"
        rationale = "global cosine is positive"
    else:
        verdict = "NO CONFLICT"
        rationale = "negative signal is too small or unstable for the hard conflict gate"

    return {
        "verdict": verdict,
        "rationale": rationale,
        "global_cosine": cosine,
        "batch_mean": batch_mean,
        "batch_std": batch_std,
        "min_layer_cosine": min_layer,
        "most_negative_locus": min_locus_key,
        "most_negative_locus_cosine": min_locus_val,
    }


def write_report(path: str, args, gradient: Dict[str, Any], failure: Dict[str, Any], verdict: Dict[str, Any]) -> None:
    layer_rows = sorted(
        ((int(layer), value["cosine"], value["what_where_norm_ratio"]) for layer, value in gradient["layers"].items()),
        key=lambda item: item[0],
    )
    projection_rows = sorted(
        ((projection, value["cosine"], value["what_where_norm_ratio"]) for projection, value in gradient["projections"].items()),
        key=lambda item: item[1],
    )
    locus_rows = sorted(
        ((key, value["cosine"], value["what_where_norm_ratio"]) for key, value in gradient["layer_projections"].items()),
        key=lambda item: item[1],
    )[:20]
    counts = Counter(failure.get("counts", {}))

    lines = [
        "# Phase 0 GUI-360 WHAT/WHERE Conflict Gate",
        "",
        "## Gate Verdict",
        "",
        f"**{verdict['verdict']}**",
        "",
        verdict["rationale"],
        "",
        "No multi-agent agents should be built unless this verdict is `CONFLICT CONFIRMED`.",
        "",
        "## Gradient Conflict Summary",
        "",
        f"- model: `{args.model_path}`",
        f"- paired data: `{args.paired_data}`",
        f"- conflict mode: `{gradient.get('mode', 'weights')}`",
        f"- batches used: `{gradient['num_batches_used']}` / requested `{gradient['num_batches_requested']}`",
        f"- skipped rows: `{gradient['num_rows_skipped']}`",
        f"- global cosine: `{gradient['global']['cosine']:.6f}`",
        f"- batch cosine mean/std: `{gradient['batch_cosine']['mean']:.6f}` / `{gradient['batch_cosine']['std']:.6f}`",
        f"- most negative layer/projection locus: `{verdict['most_negative_locus']}` = `{(verdict['most_negative_locus_cosine'] or 0.0):.6f}`",
        "",
        "## Per-Layer Cosine",
        "",
        "| layer | cosine | norm ratio WHAT/WHERE |",
        "|---:|---:|---:|",
    ]
    for layer, cosine, ratio in layer_rows:
        lines.append(f"| {layer} | {cosine:.6f} | {ratio:.3f} |")
    if projection_rows:
        lines.extend([
            "",
            "## Per-Projection Cosine",
            "",
            "| projection | cosine | norm ratio WHAT/WHERE |",
            "|---|---:|---:|",
        ])
        for projection, cosine, ratio in projection_rows:
            lines.append(f"| {projection} | {cosine:.6f} | {ratio:.3f} |")
    if locus_rows:
        lines.extend([
            "",
            "## Most Negative Layer/Projection Loci",
            "",
            "| locus | cosine | norm ratio WHAT/WHERE |",
            "|---|---:|---:|",
        ])
        for key, cosine, ratio in locus_rows:
            lines.append(f"| {key} | {cosine:.6f} | {ratio:.3f} |")
    lines.extend([
        "",
        "## Failure-Mode Separability",
        "",
        f"- eval results: `{failure['eval_results']}`",
        f"- total failed episodes: `{failure['total_failed_episodes']}`",
        f"- grounding count: `{counts['grounding']}` ({failure['grounding_share_of_failures']:.2%} of failures)",
        f"- far miss count: `{counts['far_miss']}` ({failure['far_share_of_grounding']:.2%} of grounding)",
        f"- near miss count: `{counts['near_miss']}` ({failure['near_share_of_grounding']:.2%} of grounding)",
        f"- mid miss count: `{counts['mid_miss']}` ({failure['mid_share_of_grounding']:.2%} of grounding)",
        f"- support flags: `{json.dumps(failure['support_ok'])}`",
        "",
        "Top failure counts:",
        "",
        "| bucket | count |",
        "|---|---:|",
    ])
    for key, value in counts.most_common(20):
        lines.append(f"| {key} | {value} |")
    lines.extend([
        "",
        "## Phase 0 Decision",
        "",
    ])
    if verdict["verdict"] == "CONFLICT CONFIRMED":
        if failure["support_ok"].get("near_n_ge_30") and failure["support_ok"].get("far_n_ge_30"):
            lines.append("Conflict is confirmed and far/near grounding support is adequate: a 3-agent design is allowed by the gate.")
        else:
            lines.append("Conflict is confirmed, but far/near support is not adequate for a 3-agent design; reduce the agent count according to support.")
    else:
        lines.append("Conflict is not confirmed. Do not build factored multi-agent grounding agents from this gate result.")
    lines.append("")
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 WHAT/WHERE conflict gate for GUI-360")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--paired_data", required=True)
    parser.add_argument("--eval_results", required=True)
    parser.add_argument("--episode_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--max_rows", type=int, default=512)
    parser.add_argument("--mode", choices=["hidden", "projection", "weights"], default="hidden")
    parser.add_argument("--projections", nargs="+", default=list(PROJECTIONS))
    parser.add_argument("--image_max_pixels", type=int, default=602112)
    parser.add_argument("--near_px", type=float, default=50.0)
    parser.add_argument("--far_px", type=float, default=150.0)
    parser.add_argument("--stable_negative_threshold", type=float, default=-0.05)
    parser.add_argument("--near_zero_threshold", type=float, default=0.02)
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.mode == "hidden":
        gradient = run_hidden_gradient_conflict(args)
    elif args.mode == "projection":
        gradient = run_projection_gradient_conflict(args)
    else:
        gradient = run_gradient_conflict(args)
    failure = run_failure_separability(args)
    verdict = gate_verdict(gradient, args.stable_negative_threshold, args.near_zero_threshold)
    out_json = os.path.join(args.output_dir, "phase0_conflict_summary.json")
    out_md = os.path.join(args.output_dir, "cosine_report.md")
    with open(out_json, "w") as handle:
        json.dump({"gradient": gradient, "failure": failure, "gate": verdict}, handle, indent=2)
    write_report(out_md, args, gradient, failure, verdict)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"GATE: {verdict['verdict']} ({verdict['rationale']})")


if __name__ == "__main__":
    main()