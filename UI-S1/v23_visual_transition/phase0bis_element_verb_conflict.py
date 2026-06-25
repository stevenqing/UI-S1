#!/usr/bin/env python3
"""Phase 0-bis: ELEMENT-selection vs ACTION-TYPE conflict gate.

This is the final gradient-conflict gate for factored-specialization
multi-agent grounding on GUI-360. It reuses the Phase 0 measurement style but
changes the token masks:

- ELEMENT loss: coordinate/start/end coordinate value spans only.
- VERB loss: the JSON string value of `function` only.

Text-content args are intentionally excluded from both losses.
The script builds no agents.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
from v23_visual_transition.train_offline_grpo import build_eval_style_messages  # noqa: E402
from v23_visual_transition.train_where_what_routed_sft import find_matching_bracket  # noqa: E402


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
COORD_KEYS = ("coordinate", "start_coordinate", "end_coordinate")


@dataclass
class BucketTotals:
    dot: float = 0.0
    verb_norm_sq: float = 0.0
    elem_norm_sq: float = 0.0
    n: int = 0

    def add(self, verb_grad: torch.Tensor, elem_grad: torch.Tensor) -> None:
        self.dot += float(torch.dot(verb_grad, elem_grad))
        self.verb_norm_sq += float(torch.dot(verb_grad, verb_grad))
        self.elem_norm_sq += float(torch.dot(elem_grad, elem_grad))
        self.n += 1

    def cosine(self) -> float:
        denom = math.sqrt(self.verb_norm_sq) * math.sqrt(self.elem_norm_sq)
        return self.dot / (denom + 1e-12)

    def ratio(self) -> float:
        return math.sqrt(self.verb_norm_sq) / (math.sqrt(self.elem_norm_sq) + 1e-12)

    def as_dict(self) -> Dict[str, float]:
        return {
            "cosine": self.cosine(),
            "verb_norm": math.sqrt(self.verb_norm_sq),
            "element_norm": math.sqrt(self.elem_norm_sq),
            "verb_element_norm_ratio": self.ratio(),
            "n_tensors": self.n,
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
        var = sum((value - mean) ** 2 for value in self.values) / max(len(self.values) - 1, 1)
        return {"n": len(self.values), "mean": mean, "std": math.sqrt(max(var, 0.0)), "min": min(self.values), "max": max(self.values)}


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
    if not full_tool_call:
        return ""
    payload = json.dumps(full_tool_call, ensure_ascii=False, indent=2)
    return f"<tool_call>\n{payload}\n</tool_call>"


def find_json_string_value_span(text: str, key: str) -> Optional[Tuple[int, int]]:
    needle = f'"{key}"'
    key_start = text.find(needle)
    if key_start < 0:
        return None
    colon = text.find(":", key_start + len(needle))
    if colon < 0:
        return None
    quote_start = text.find('"', colon + 1)
    if quote_start < 0:
        return None
    idx = quote_start + 1
    escape = False
    while idx < len(text):
        char = text[idx]
        if escape:
            escape = False
        elif char == "\\":
            escape = True
        elif char == '"':
            return quote_start + 1, idx
        idx += 1
    return None


def element_verb_char_masks(target_text: str) -> Tuple[List[int], List[int]]:
    """Return char masks: verb=1 on function value, element=1 on coordinate arrays."""
    verb_mask = [0] * len(target_text)
    elem_mask = [0] * len(target_text)

    verb_span = find_json_string_value_span(target_text, "function")
    if verb_span is not None:
        start, end = verb_span
        for idx in range(start, end):
            verb_mask[idx] = 1

    for key in COORD_KEYS:
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
            for idx in range(value_start, value_end + 1):
                elem_mask[idx] = 1
            pos = value_end + 1
    return verb_mask, elem_mask


def token_element_verb_labels(tokenizer, target_text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Label response tokens: 1=verb, 0=element coordinate, -100=neither."""
    encoded = tokenizer(target_text, add_special_tokens=False, return_tensors="pt", return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping")[0].tolist()
    input_ids = encoded["input_ids"].squeeze(0)
    verb_mask, elem_mask = element_verb_char_masks(target_text)
    labels: List[int] = []
    for start, end in offsets:
        if end <= start:
            labels.append(-100)
            continue
        is_verb = any(verb_mask[start:end])
        is_elem = any(elem_mask[start:end])
        if is_verb and is_elem:
            labels.append(-100)
        elif is_verb:
            labels.append(1)
        elif is_elem:
            labels.append(0)
        else:
            labels.append(-100)
    return input_ids, torch.tensor(labels, dtype=torch.long)


def decode_labeled_tokens(tokenizer, input_ids: torch.Tensor, labels: torch.Tensor, value: int) -> List[str]:
    tokens = []
    for token_id, label in zip(input_ids.tolist(), labels.tolist()):
        if label == value:
            tokens.append(tokenizer.decode([token_id]))
    return tokens


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


def prepare_row(processor, row: Dict[str, Any], device: str):
    image_path = row.get("image")
    if not image_path or not os.path.exists(image_path):
        return None
    target_text = build_target_text(row)
    if not target_text:
        return None
    response_ids, labels = token_element_verb_labels(processor.tokenizer, target_text)
    if (labels == 0).sum().item() == 0 or (labels == 1).sum().item() == 0:
        return None
    image = Image.open(image_path).convert("RGB")
    messages = build_eval_style_messages(row["goal"], row.get("history", []), image_path)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=False)
    prompt_inputs = {key: value.to(device) for key, value in prompt_inputs.items()}
    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    response_ids = response_ids.to(device)
    labels = labels.to(device)
    ids = torch.cat([prompt_inputs["input_ids"][0], response_ids], dim=0).unsqueeze(0)
    fwd_kwargs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    for key in ("pixel_values", "image_grid_thw"):
        if key in prompt_inputs:
            fwd_kwargs[key] = prompt_inputs[key]
    return fwd_kwargs, ids, prompt_len, labels, target_text, response_ids.cpu(), labels.cpu()


def response_token_loss(model, fwd_kwargs, ids, prompt_len: int, labels: torch.Tensor, label_value: int, pad_id: int):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(**fwd_kwargs)
        logits = outputs.logits
        resp_logits = logits[:, prompt_len - 1:-1, :]
        resp_labels = ids[:, prompt_len:]
        usable = min(resp_logits.shape[1], labels.shape[0])
        resp_logits = resp_logits[:, :usable, :]
        resp_labels = resp_labels[:, :usable]
        labels = labels[:usable]
        token_loss = F.cross_entropy(
            resp_logits.reshape(-1, resp_logits.shape[-1]),
            resp_labels.reshape(-1),
            reduction="none",
        ).view_as(resp_labels)
        mask = (labels.unsqueeze(0) == label_value) & (resp_labels != pad_id)
        if not mask.any():
            return None
        return token_loss[mask].mean()


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
    model.train()
    return processor, model


def add_to_buckets(key: str, verb_grad: torch.Tensor, elem_grad: torch.Tensor, global_total, layer_totals, projection_totals, locus_totals, batch_total):
    layer_text, projection = key.split(".", 1)
    layer = int(layer_text[1:])
    vg = verb_grad.detach().float().reshape(-1).cpu()
    eg = elem_grad.detach().float().reshape(-1).cpu()
    global_total.add(vg, eg)
    batch_total.add(vg, eg)
    layer_totals[layer].add(vg, eg)
    projection_totals[projection].add(vg, eg)
    locus_totals[key].add(vg, eg)


def hidden_role_losses(model, fwd_kwargs, ids, prompt_len, labels, pad_id):
    outputs = model(**fwd_kwargs, output_hidden_states=True)
    logits = outputs.logits
    resp_logits = logits[:, prompt_len - 1:-1, :]
    resp_labels = ids[:, prompt_len:]
    usable = min(resp_logits.shape[1], labels.shape[0])
    resp_logits = resp_logits[:, :usable, :]
    resp_labels = resp_labels[:, :usable]
    labels = labels[:usable]
    token_loss = F.cross_entropy(resp_logits.reshape(-1, resp_logits.shape[-1]), resp_labels.reshape(-1), reduction="none").view_as(resp_labels).squeeze(0)
    verb_mask = (labels == 1) & (resp_labels.squeeze(0) != pad_id)
    elem_mask = (labels == 0) & (resp_labels.squeeze(0) != pad_id)
    if not verb_mask.any() or not elem_mask.any():
        return None
    return token_loss[verb_mask].mean(), token_loss[elem_mask].mean(), outputs.hidden_states


def run_hidden_conflict(args):
    rows = read_jsonl(args.paired_data, args.max_rows)
    processor, model = load_model(args)
    pad_id = processor.tokenizer.pad_token_id
    global_total = BucketTotals()
    layer_totals: Dict[int, BucketTotals] = defaultdict(BucketTotals)
    batch_values = RunningScalar()
    mask_examples = []
    batch_records = []
    used = skipped = 0
    for row_idx, row in enumerate(rows):
        if used >= args.num_batches:
            break
        prepared = prepare_row(processor, row, args.device)
        if prepared is None:
            skipped += 1
            continue
        fwd_kwargs, ids, prompt_len, labels, target_text, response_ids_cpu, labels_cpu = prepared
        if len(mask_examples) < args.mask_examples:
            mask_examples.append({
                "episode_id": row.get("episode_id"),
                "step_idx": row.get("step_idx"),
                "target_text": target_text,
                "verb_tokens": decode_labeled_tokens(processor.tokenizer, response_ids_cpu, labels_cpu, 1),
                "element_tokens": decode_labeled_tokens(processor.tokenizer, response_ids_cpu, labels_cpu, 0),
                "overlap_count": 0,
            })
        result = hidden_role_losses(model, fwd_kwargs, ids, prompt_len, labels, pad_id)
        if result is None:
            skipped += 1
            continue
        verb_loss, elem_loss, hidden_states = result
        states = [state for state in hidden_states[1:]]
        verb_grads = torch.autograd.grad(verb_loss, states, retain_graph=True, allow_unused=True)
        elem_grads = torch.autograd.grad(elem_loss, states, retain_graph=False, allow_unused=True)
        batch_total = BucketTotals()
        for layer, (vg, eg) in enumerate(zip(verb_grads, elem_grads)):
            if vg is None or eg is None:
                continue
            vgf = vg.detach().float().reshape(-1).cpu()
            egf = eg.detach().float().reshape(-1).cpu()
            global_total.add(vgf, egf)
            layer_totals[layer].add(vgf, egf)
            batch_total.add(vgf, egf)
        batch_values.add(batch_total.cosine())
        batch_records.append({"row_idx": row_idx, "episode_id": row.get("episode_id"), "step_idx": row.get("step_idx"), "global_cosine": batch_total.cosine(), "verb_loss": float(verb_loss.detach().item()), "element_loss": float(elem_loss.detach().item())})
        used += 1
        if used % args.log_every == 0:
            print(f"processed {used}/{args.num_batches}; last hidden cosine={batch_total.cosine():.6f}", flush=True)
        del result, verb_loss, elem_loss, hidden_states, states, verb_grads, elem_grads
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    return {
        "mode": "hidden",
        "num_batches_requested": args.num_batches,
        "num_batches_used": used,
        "num_rows_skipped": skipped,
        "global": global_total.as_dict(),
        "batch_cosine": batch_values.as_dict(),
        "layers": {str(layer): total.as_dict() for layer, total in sorted(layer_totals.items())},
        "projections": {},
        "layer_projections": {},
        "mask_examples": mask_examples,
        "batches": batch_records,
    }


def run_projection_conflict(args):
    rows = read_jsonl(args.paired_data, args.max_rows)
    processor, model = load_model(args)
    pad_id = processor.tokenizer.pad_token_id
    captures: Dict[str, torch.Tensor] = {}
    hooks = register_projection_hooks(model, captures, args.projections)
    global_total = BucketTotals()
    layer_totals: Dict[int, BucketTotals] = defaultdict(BucketTotals)
    projection_totals: Dict[str, BucketTotals] = defaultdict(BucketTotals)
    locus_totals: Dict[str, BucketTotals] = defaultdict(BucketTotals)
    batch_values = RunningScalar()
    mask_examples = []
    batch_records = []
    used = skipped = 0
    for row_idx, row in enumerate(rows):
        if used >= args.num_batches:
            break
        prepared = prepare_row(processor, row, args.device)
        if prepared is None:
            skipped += 1
            continue
        fwd_kwargs, ids, prompt_len, labels, target_text, response_ids_cpu, labels_cpu = prepared
        if len(mask_examples) < args.mask_examples:
            mask_examples.append({
                "episode_id": row.get("episode_id"),
                "step_idx": row.get("step_idx"),
                "target_text": target_text,
                "verb_tokens": decode_labeled_tokens(processor.tokenizer, response_ids_cpu, labels_cpu, 1),
                "element_tokens": decode_labeled_tokens(processor.tokenizer, response_ids_cpu, labels_cpu, 0),
                "overlap_count": 0,
            })
        captures.clear()
        verb_loss = response_token_loss(model, fwd_kwargs, ids, prompt_len, labels, 1, pad_id)
        if verb_loss is None:
            skipped += 1
            continue
        verb_keys = list(captures.keys())
        verb_targets = [captures[key] for key in verb_keys]
        verb_grads = torch.autograd.grad(verb_loss, verb_targets, retain_graph=False, allow_unused=True)
        verb_by_key = {key: grad.detach().to(device="cpu", dtype=torch.float16).flatten() for key, grad in zip(verb_keys, verb_grads) if grad is not None}
        del verb_targets, verb_grads, verb_loss
        torch.cuda.empty_cache()
        captures.clear()
        elem_loss = response_token_loss(model, fwd_kwargs, ids, prompt_len, labels, 0, pad_id)
        if elem_loss is None:
            skipped += 1
            continue
        elem_keys = list(captures.keys())
        elem_targets = [captures[key] for key in elem_keys]
        elem_grads = torch.autograd.grad(elem_loss, elem_targets, retain_graph=False, allow_unused=True)
        batch_total = BucketTotals()
        for key, eg in zip(elem_keys, elem_grads):
            vg = verb_by_key.get(key)
            if vg is None or eg is None:
                continue
            add_to_buckets(key, vg.float(), eg.detach().to(device="cpu", dtype=torch.float16).flatten().float(), global_total, layer_totals, projection_totals, locus_totals, batch_total)
        batch_values.add(batch_total.cosine())
        batch_records.append({"row_idx": row_idx, "episode_id": row.get("episode_id"), "step_idx": row.get("step_idx"), "global_cosine": batch_total.cosine(), "element_loss": float(elem_loss.detach().item())})
        used += 1
        if used % args.log_every == 0:
            print(f"processed {used}/{args.num_batches}; last projection cosine={batch_total.cosine():.6f}", flush=True)
        del elem_targets, elem_grads, elem_loss, verb_by_key
        torch.cuda.empty_cache()
    for hook in hooks:
        hook.remove()
    del model
    torch.cuda.empty_cache()
    return {
        "mode": "projection",
        "num_batches_requested": args.num_batches,
        "num_batches_used": used,
        "num_rows_skipped": skipped,
        "global": global_total.as_dict(),
        "batch_cosine": batch_values.as_dict(),
        "layers": {str(layer): total.as_dict() for layer, total in sorted(layer_totals.items())},
        "projections": {projection: total.as_dict() for projection, total in sorted(projection_totals.items())},
        "layer_projections": {key: total.as_dict() for key, total in sorted(locus_totals.items())},
        "mask_examples": mask_examples,
        "batches": batch_records,
    }


def normalize_action_type(value: Any) -> str:
    aliases = {"left_click": "click", "tap": "click", "double_click": "click", "drag": "swipe", "scroll": "swipe", "wheel_mouse_input": "swipe", "input": "type"}
    key = str(value or "").strip().lower()
    return aliases.get(key, key)


def action_coord(action: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    if not action:
        return None
    value = action.get("coordinate") or action.get("startCoordinate") or action.get("start_coordinate")
    if isinstance(value, list) and len(value) >= 2 and value[0] is not None and value[1] is not None:
        return float(value[0]), float(value[1])
    return None


def coord_distance_px(pred_action: Optional[Dict[str, Any]], gt_action: Dict[str, Any]) -> Optional[float]:
    pred = action_coord(pred_action)
    gt = action_coord(gt_action)
    if pred is None or gt is None:
        return None
    return math.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2)


def read_episode_jsonl(path: str) -> Dict[str, Dict[str, Any]]:
    episodes: Dict[str, Dict[str, Any]] = {}
    with open(path) as handle:
        for line in handle:
            if line.strip():
                ep = json.loads(line)
                episodes[str(ep.get("episode_id"))] = ep
    return episodes


def run_failure_separability(args) -> Dict[str, Any]:
    with open(args.eval_results) as handle:
        eval_results = json.load(handle)
    episodes = read_episode_jsonl(args.episode_data)
    counts = Counter()
    total_failed = 0
    for episode_id, eval_ep in eval_results.items():
        if eval_ep.get("task_success"):
            continue
        bad = next((step for step in eval_ep.get("steps", []) if not step.get("success")), None)
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
        ep = episodes.get(str(episode_id))
        step_idx = int(bad.get("step_idx", 0))
        gt_action = {}
        if ep and step_idx < len(ep.get("steps", [])):
            gt_action = ep["steps"][step_idx].get("action", {}) or {}
        pred_action = bad.get("pred_action") or parse_action_from_text(bad.get("pred_text", "") or "")
        dist = coord_distance_px(pred_action, gt_action)
        if dist is None:
            counts["grounding_missing_coord"] += 1
            continue
        counts["grounding"] += 1
        if dist <= args.near_px:
            counts["near_miss"] += 1
        elif dist >= args.far_px:
            counts["far_miss"] += 1
        else:
            counts["mid_miss"] += 1
    grounding = counts["grounding"]
    type_mismatch = sum(value for key, value in counts.items() if key.startswith("type_mismatch"))
    return {
        "eval_results": args.eval_results,
        "episode_data": args.episode_data,
        "total_failed_episodes": total_failed,
        "counts": dict(counts),
        "grounding_total": grounding,
        "type_mismatch_total": type_mismatch,
        "grounding_share_of_failures": grounding / max(total_failed, 1),
        "type_mismatch_share_of_failures": type_mismatch / max(total_failed, 1),
        "near_share_of_grounding": counts["near_miss"] / max(grounding, 1),
        "far_share_of_grounding": counts["far_miss"] / max(grounding, 1),
        "support_ok": {"grounding_n_ge_30": grounding >= 30, "near_n_ge_30": counts["near_miss"] >= 30, "far_n_ge_30": counts["far_miss"] >= 30, "type_mismatch_n_ge_30": type_mismatch >= 30},
    }


def gate_verdict(gradient: Dict[str, Any], stable_negative_threshold: float, near_zero_threshold: float) -> Dict[str, Any]:
    cosine = float(gradient["global"]["cosine"])
    batch_mean = float(gradient["batch_cosine"]["mean"])
    batch_std = float(gradient["batch_cosine"]["std"])
    stable_negative = batch_mean < 0 and abs(batch_mean) > batch_std
    if cosine <= stable_negative_threshold and stable_negative:
        verdict = "CONFLICT CONFIRMED"
        rationale = "global cosine is meaningfully negative and batch sign is stable"
    elif abs(cosine) <= near_zero_threshold:
        verdict = "NO CONFLICT"
        rationale = "global cosine is near zero"
    elif cosine > near_zero_threshold:
        verdict = "ALIGNED"
        rationale = "global cosine is positive"
    else:
        verdict = "INCONCLUSIVE"
        rationale = "negative/near-zero signal is not stable enough for a gate decision"
    min_locus = None
    min_value = None
    for key, stats in gradient.get("layer_projections", {}).items():
        value = stats["cosine"]
        if min_value is None or value < min_value:
            min_locus = key
            min_value = value
    return {"verdict": verdict, "rationale": rationale, "global_cosine": cosine, "batch_mean": batch_mean, "batch_std": batch_std, "most_negative_locus": min_locus, "most_negative_locus_cosine": min_value}


def write_report(path: str, args, gradient, failure, verdict) -> None:
    lines = [
        "# Phase 0-bis GUI-360 ELEMENT vs VERB Conflict Gate",
        "",
        "## Gate Verdict",
        "",
        f"**{verdict['verdict']}**",
        "",
        verdict["rationale"],
        "",
        "No element/action-type agents should be built unless this verdict is `CONFLICT CONFIRMED`.",
        "",
        "## Exact Token-Mask Sanity Check",
        "",
        "VERB tokens are only the JSON `function` value. ELEMENT tokens are only coordinate array spans. Text-content args are excluded.",
        "",
    ]
    for idx, example in enumerate(gradient.get("mask_examples", [])[: args.mask_examples]):
        lines.extend([
            f"### Example {idx + 1}",
            "",
            f"- episode: `{example.get('episode_id')}` step `{example.get('step_idx')}`",
            f"- verb tokens: `{example.get('verb_tokens')}`",
            f"- element tokens: `{example.get('element_tokens')}`",
            f"- overlap count: `{example.get('overlap_count')}`",
            "",
        ])
    lines.extend([
        "## Gradient Conflict Summary",
        "",
        f"- model: `{args.model_path}`",
        f"- paired data: `{args.paired_data}`",
        f"- mode: `{gradient['mode']}`",
        f"- batches used: `{gradient['num_batches_used']}` / `{gradient['num_batches_requested']}`",
        f"- skipped rows: `{gradient['num_rows_skipped']}`",
        f"- global cosine: `{gradient['global']['cosine']:.6f}`",
        f"- batch mean/std: `{gradient['batch_cosine']['mean']:.6f}` / `{gradient['batch_cosine']['std']:.6f}`",
        f"- most negative locus: `{verdict.get('most_negative_locus')}` = `{(verdict.get('most_negative_locus_cosine') or 0.0):.6f}`",
        "",
        "## Per-Layer Cosine",
        "",
        "| layer | cosine | VERB/ELEMENT norm ratio |",
        "|---:|---:|---:|",
    ])
    for layer, stats in sorted(((int(k), v) for k, v in gradient.get("layers", {}).items()), key=lambda x: x[0]):
        lines.append(f"| {layer} | {stats['cosine']:.6f} | {stats['verb_element_norm_ratio']:.3f} |")
    if gradient.get("projections"):
        lines.extend(["", "## Per-Projection Cosine", "", "| projection | cosine | VERB/ELEMENT norm ratio |", "|---|---:|---:|"])
        for proj, stats in sorted(gradient["projections"].items(), key=lambda item: item[1]["cosine"]):
            lines.append(f"| {proj} | {stats['cosine']:.6f} | {stats['verb_element_norm_ratio']:.3f} |")
    if gradient.get("layer_projections"):
        lines.extend(["", "## Most Negative Layer/Projection Loci", "", "| locus | cosine | VERB/ELEMENT norm ratio |", "|---|---:|---:|"])
        for locus, stats in sorted(gradient["layer_projections"].items(), key=lambda item: item[1]["cosine"])[:20]:
            lines.append(f"| {locus} | {stats['cosine']:.6f} | {stats['verb_element_norm_ratio']:.3f} |")
    counts = Counter(failure["counts"])
    lines.extend([
        "",
        "## Failure Bucket Support",
        "",
        f"- failed episodes: `{failure['total_failed_episodes']}`",
        f"- grounding total: `{failure['grounding_total']}` ({failure['grounding_share_of_failures']:.2%})",
        f"- far miss: `{counts['far_miss']}` ({failure['far_share_of_grounding']:.2%} of grounding)",
        f"- near miss: `{counts['near_miss']}` ({failure['near_share_of_grounding']:.2%} of grounding)",
        f"- type mismatch total: `{failure['type_mismatch_total']}` ({failure['type_mismatch_share_of_failures']:.2%})",
        f"- support flags: `{json.dumps(failure['support_ok'])}`",
        "",
        "| bucket | count |",
        "|---|---:|",
    ])
    for key, value in counts.most_common(20):
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Phase 0-bis Decision", ""])
    if verdict["verdict"] == "CONFLICT CONFIRMED":
        lines.append("Element-selection and verb-decision conflict is confirmed. A 2-agent element/action-type system may be designed for review.")
    else:
        lines.append("Element-selection and verb-decision conflict is not confirmed. Do not build a factored-specialization multi-agent system from this gate. The next possible multi-agent basis is candidate-source error orthogonality, not gradient conflict.")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0-bis ELEMENT vs VERB conflict gate")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--paired_data", required=True)
    parser.add_argument("--eval_results", required=True)
    parser.add_argument("--episode_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--max_rows", type=int, default=512)
    parser.add_argument("--mode", choices=["hidden", "projection"], default="hidden")
    parser.add_argument("--projections", nargs="+", default=list(PROJECTIONS))
    parser.add_argument("--image_max_pixels", type=int, default=602112)
    parser.add_argument("--near_px", type=float, default=50.0)
    parser.add_argument("--far_px", type=float, default=150.0)
    parser.add_argument("--stable_negative_threshold", type=float, default=-0.05)
    parser.add_argument("--near_zero_threshold", type=float, default=0.02)
    parser.add_argument("--mask_examples", type=int, default=3)
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    gradient = run_hidden_conflict(args) if args.mode == "hidden" else run_projection_conflict(args)
    failure = run_failure_separability(args)
    verdict = gate_verdict(gradient, args.stable_negative_threshold, args.near_zero_threshold)
    summary_path = os.path.join(args.output_dir, "phase0bis_summary.json")
    report_path = os.path.join(args.output_dir, "cosine_report.md")
    with open(summary_path, "w") as handle:
        json.dump({"gradient": gradient, "failure": failure, "gate": verdict}, handle, indent=2)
    write_report(report_path, args, gradient, failure, verdict)
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    print(f"GATE: {verdict['verdict']} ({verdict['rationale']})")


if __name__ == "__main__":
    main()