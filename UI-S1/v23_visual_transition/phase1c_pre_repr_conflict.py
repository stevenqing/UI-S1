#!/usr/bin/env python3
"""Phase 1c-pre representation conflict gate for GUI-360.

Diagnostic only: no LoRA/training. Measures gradient cosine between proxy
representation losses for element-disambiguation and affordance-encoding at the
Qwen2.5-VL visual merger, plus an output-layer contrast control.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

ACTION_ALIASES = {
    "drag": "swipe",
    "scroll": "swipe",
    "wheel_mouse_input": "swipe",
    "input": "type",
    "left_click": "click",
    "tap": "click",
    "double_click": "click",
}
AFFORDANCE_TO_ID = {"click": 0, "type": 1, "other": 2}
ID_TO_AFFORDANCE = {idx: name for name, idx in AFFORDANCE_TO_ID.items()}


@dataclass
class Record:
    episode_id: str
    step_idx: int
    action_type: str
    affordance: str
    bbox: List[float]
    image_w: int
    image_h: int
    image_bytes: bytes
    goal: str


@dataclass
class BucketTotals:
    dot: float = 0.0
    dis_norm_sq: float = 0.0
    afford_norm_sq: float = 0.0
    n_tensors: int = 0

    def add(self, grad_dis: Optional[torch.Tensor], grad_afford: Optional[torch.Tensor]) -> None:
        if grad_dis is None and grad_afford is None:
            return
        if grad_dis is None:
            ga = grad_afford.detach().float().flatten()
            self.afford_norm_sq += float(torch.dot(ga, ga).cpu())
            self.n_tensors += 1
            return
        if grad_afford is None:
            gd = grad_dis.detach().float().flatten()
            self.dis_norm_sq += float(torch.dot(gd, gd).cpu())
            self.n_tensors += 1
            return
        gd = grad_dis.detach().float().flatten()
        ga = grad_afford.detach().float().flatten()
        self.dot += float(torch.dot(gd, ga).cpu())
        self.dis_norm_sq += float(torch.dot(gd, gd).cpu())
        self.afford_norm_sq += float(torch.dot(ga, ga).cpu())
        self.n_tensors += 1

    def cosine(self) -> float:
        denom = math.sqrt(self.dis_norm_sq) * math.sqrt(self.afford_norm_sq)
        return self.dot / (denom + 1e-12)

    def ratio(self) -> float:
        return math.sqrt(self.dis_norm_sq) / (math.sqrt(self.afford_norm_sq) + 1e-12)

    def as_dict(self) -> Dict[str, float]:
        return {
            "cosine": self.cosine(),
            "disambig_norm": math.sqrt(self.dis_norm_sq),
            "afford_norm": math.sqrt(self.afford_norm_sq),
            "disambig_afford_norm_ratio": self.ratio(),
            "n_tensors": self.n_tensors,
        }


@dataclass
class RunningStats:
    values: List[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        if math.isfinite(value):
            self.values.append(float(value))

    def as_dict(self) -> Dict[str, float]:
        if not self.values:
            return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "negative_fraction": 0.0}
        arr = np.asarray(self.values, dtype=np.float64)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "negative_fraction": float((arr < 0).mean()),
        }


def normalize_action_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    return ACTION_ALIASES.get(text, text)


def patch_legacy_mrope_config(config: Any) -> None:
    """Make legacy Qwen2.5-VL mrope configs load on newer transformers."""
    candidates = [config, getattr(config, "text_config", None), getattr(config, "vision_config", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        rope_scaling = getattr(candidate, "rope_scaling", None)
        if isinstance(rope_scaling, dict) and rope_scaling.get("rope_type") == "mrope":
            rope_scaling["rope_type"] = "default"
        if getattr(candidate, "rope_type", None) == "mrope":
            setattr(candidate, "rope_type", "default")


def affordance_label(action_type: str) -> str:
    if action_type == "click":
        return "click"
    if action_type == "type":
        return "type"
    return "other"


def read_records(dataset_dir: str, max_records: int = 0) -> List[Record]:
    root = Path(dataset_dir) / "data"
    records: List[Record] = []
    for parquet_path in sorted(root.glob("*.parquet")):
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(batch_size=8, columns=["episode_id", "goal", "steps", "screenshots"]):
            for row in batch.to_pylist():
                steps = json.loads(row["steps"])
                screenshots = row.get("screenshots") or []
                for step in steps:
                    bbox = step.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue
                    step_idx = int(step.get("step_idx", 0))
                    if step_idx >= len(screenshots):
                        continue
                    image_info = screenshots[step_idx] or {}
                    image_bytes = image_info.get("bytes")
                    if not image_bytes:
                        continue
                    action_type = normalize_action_type((step.get("action") or {}).get("action"))
                    records.append(Record(
                        episode_id=str(row["episode_id"]),
                        step_idx=step_idx,
                        action_type=action_type,
                        affordance=affordance_label(action_type),
                        bbox=[float(x) for x in bbox],
                        image_w=int(step.get("image_w") or 1040),
                        image_h=int(step.get("image_h") or 736),
                        image_bytes=image_bytes,
                        goal=str(row.get("goal") or ""),
                    ))
                    if max_records and len(records) >= max_records:
                        return records
    return records


def select_records(records: Sequence[Record], args: argparse.Namespace) -> Tuple[List[Record], List[Record], List[Record], Dict[str, Any]]:
    by_affordance: Dict[str, List[Record]] = defaultdict(list)
    click_records: List[Record] = []
    for record in records:
        by_affordance[record.affordance].append(record)
        if record.action_type == "click":
            click_records.append(record)
    rng = random.Random(args.seed)
    for bucket in by_affordance.values():
        rng.shuffle(bucket)
    rng.shuffle(click_records)

    prototype_records: List[Record] = []
    for label in ("click", "type", "other"):
        prototype_records.extend(by_affordance[label][: args.prototype_per_class])

    disambig_records = click_records[args.prototype_per_class : args.prototype_per_class + args.num_batches]
    cursors = {label: args.prototype_per_class for label in ("click", "type", "other")}
    affordance_records: List[Record] = []
    labels = ("click", "type", "other")
    while len(affordance_records) < args.num_batches:
        label = labels[len(affordance_records) % len(labels)]
        idx = cursors[label]
        if idx >= len(by_affordance[label]):
            break
        affordance_records.append(by_affordance[label][idx])
        cursors[label] += 1

    support = {
        "total_records": len(records),
        "click_records": len(by_affordance["click"]),
        "type_records": len(by_affordance["type"]),
        "other_records": len(by_affordance["other"]),
        "prototype_records": len(prototype_records),
        "disambig_records": len(disambig_records),
        "affordance_records": len(affordance_records),
    }
    return disambig_records, affordance_records, prototype_records, support


def image_from_record(record: Record) -> Image.Image:
    return Image.open(io.BytesIO(record.image_bytes)).convert("RGB")


def prepare_visual_inputs(processor: AutoProcessor, record: Record, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    image = image_from_record(record)
    encoded = processor(text="image", images=image, return_tensors="pt")
    return {
        "pixel_values": encoded["pixel_values"].to(device=device, dtype=dtype),
        "image_grid_thw": encoded["image_grid_thw"].to(device=device),
    }


def prepare_full_inputs(processor: AutoProcessor, record: Record, device: torch.device) -> Dict[str, torch.Tensor]:
    image = image_from_record(record)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Inspect the UI screenshot."},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[text], images=[image], return_tensors="pt")
    return {key: value.to(device) for key, value in encoded.items()}


def visual_forward(model: Qwen2_5_VLForConditionalGeneration, processor: AutoProcessor, record: Record, device: torch.device) -> Tuple[torch.Tensor, Tuple[int, int]]:
    dtype = next(model.visual.parameters()).dtype
    inputs = prepare_visual_inputs(processor, record, device, dtype)
    hidden = model.visual(inputs["pixel_values"], inputs["image_grid_thw"])
    grid = inputs["image_grid_thw"][0]
    out_h = int(grid[1].item()) // int(model.visual.spatial_merge_size)
    out_w = int(grid[2].item()) // int(model.visual.spatial_merge_size)
    return hidden, (out_h, out_w)


def output_forward(model: Qwen2_5_VLForConditionalGeneration, processor: AutoProcessor, record: Record, device: torch.device, image_token_id: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    inputs = prepare_full_inputs(processor, record, device)
    outputs = model(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)
    hidden = outputs.hidden_states[-1][0]
    image_positions = (inputs["input_ids"][0] == image_token_id).nonzero(as_tuple=False).flatten()
    image_hidden = hidden.index_select(0, image_positions)
    grid = inputs["image_grid_thw"][0]
    out_h = int(grid[1].item()) // int(model.visual.spatial_merge_size)
    out_w = int(grid[2].item()) // int(model.visual.spatial_merge_size)
    expected = out_h * out_w
    if image_hidden.shape[0] != expected:
        image_hidden = image_hidden[:expected]
    return image_hidden, (out_h, out_w)


def bbox_indices(bbox: Sequence[float], image_w: int, image_h: int, out_hw: Tuple[int, int], device: torch.device) -> torch.Tensor:
    out_h, out_w = out_hw
    x1, y1, x2, y2 = bbox
    xa = max(0, min(out_w - 1, int(math.floor(x1 / max(image_w, 1) * out_w))))
    xb = max(0, min(out_w - 1, int(math.ceil(x2 / max(image_w, 1) * out_w) - 1)))
    ya = max(0, min(out_h - 1, int(math.floor(y1 / max(image_h, 1) * out_h))))
    yb = max(0, min(out_h - 1, int(math.ceil(y2 / max(image_h, 1) * out_h) - 1)))
    if xb < xa:
        xb = xa
    if yb < ya:
        yb = ya
    indices = [y * out_w + x for y in range(ya, yb + 1) for x in range(xa, xb + 1)]
    return torch.tensor(indices, device=device, dtype=torch.long)


def pool_bbox(hidden: torch.Tensor, record: Record, bbox: Sequence[float], out_hw: Tuple[int, int]) -> torch.Tensor:
    indices = bbox_indices(bbox, record.image_w, record.image_h, out_hw, hidden.device)
    return hidden.index_select(0, indices).mean(dim=0)


def shifted_bboxes(record: Record, n: int = 6) -> List[List[float]]:
    x1, y1, x2, y2 = record.bbox
    width = max(8.0, x2 - x1)
    height = max(8.0, y2 - y1)
    shifts = [
        (width * 1.5, 0.0), (-width * 1.5, 0.0), (0.0, height * 1.5), (0.0, -height * 1.5),
        (width * 1.5, height * 1.5), (-width * 1.5, height * 1.5), (width * 2.5, 0.0), (0.0, height * 2.5),
    ]
    boxes: List[List[float]] = []
    for dx, dy in shifts:
        nx1 = min(max(0.0, x1 + dx), max(0.0, record.image_w - width))
        ny1 = min(max(0.0, y1 + dy), max(0.0, record.image_h - height))
        box = [nx1, ny1, min(record.image_w, nx1 + width), min(record.image_h, ny1 + height)]
        if (abs(box[0] - x1) > 1 or abs(box[1] - y1) > 1) and box not in boxes:
            boxes.append(box)
        if len(boxes) >= n:
            break
    return boxes


def disambig_loss(hidden: torch.Tensor, record: Record, out_hw: Tuple[int, int], temperature: float) -> torch.Tensor:
    anchor = F.normalize(pool_bbox(hidden, record, record.bbox, out_hw).float(), dim=0)
    positive = anchor.detach()
    negatives = [F.normalize(pool_bbox(hidden, record, box, out_hw).float(), dim=0) for box in shifted_bboxes(record)]
    if not negatives:
        return anchor.sum() * 0.0
    logits = torch.stack([torch.dot(anchor, positive)] + [torch.dot(anchor, neg) for neg in negatives], dim=0) / temperature
    return F.cross_entropy(logits.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=hidden.device))


def affordance_loss(hidden: torch.Tensor, record: Record, out_hw: Tuple[int, int], prototypes: torch.Tensor, temperature: float) -> torch.Tensor:
    rep = F.normalize(pool_bbox(hidden, record, record.bbox, out_hw).float(), dim=0)
    logits = torch.matmul(prototypes.to(hidden.device), rep) / temperature
    label = torch.tensor([AFFORDANCE_TO_ID[record.affordance]], dtype=torch.long, device=hidden.device)
    return F.cross_entropy(logits.unsqueeze(0), label)


def compute_prototypes(model, processor, records: Sequence[Record], device: torch.device, image_token_id: int, space: str) -> torch.Tensor:
    sums: Dict[str, torch.Tensor] = {}
    counts = Counter()
    with torch.no_grad():
        for record in records:
            if space == "visual":
                hidden, out_hw = visual_forward(model, processor, record, device)
            else:
                hidden, out_hw = output_forward(model, processor, record, device, image_token_id)
            rep = F.normalize(pool_bbox(hidden, record, record.bbox, out_hw).float(), dim=0).detach().cpu()
            sums.setdefault(record.affordance, torch.zeros_like(rep))
            sums[record.affordance] += rep
            counts[record.affordance] += 1
            del hidden
    prototypes = []
    for label in ("click", "type", "other"):
        if counts[label] == 0:
            raise RuntimeError(f"Missing prototype records for label {label}")
        prototypes.append(F.normalize(sums[label] / counts[label], dim=0))
    return torch.stack(prototypes, dim=0)


def locus_for_param(name: str) -> str:
    if name.startswith("model.visual.merger.ln_q") or name.startswith("visual.merger.ln_q"):
        return "visual.merger.ln_q"
    if name.startswith("model.visual.merger.mlp.0") or name.startswith("visual.merger.mlp.0"):
        return "visual.merger.mlp.0"
    if name.startswith("model.visual.merger.mlp.2") or name.startswith("visual.merger.mlp.2"):
        return "visual.merger.mlp.2"
    match = re.search(r"(?:model\.)?visual\.blocks\.(\d+)\.", name)
    if match:
        return f"visual.blocks.{match.group(1)}"
    match = re.search(r"model\.layers\.(\d+)\..*\.(q_proj|k_proj|v_proj|o_proj|down_proj|up_proj|gate_proj)\.", name)
    if match:
        return f"text.layers.{match.group(1)}.{match.group(2)}"
    if name.startswith("model.norm"):
        return "text.norm"
    return name.rsplit(".", 1)[0]


def visual_param_names(model, late_blocks: int) -> set[str]:
    block_ids = []
    for name, _param in model.named_parameters():
        match = re.search(r"(?:model\.)?visual\.blocks\.(\d+)\.", name)
        if match:
            block_ids.append(int(match.group(1)))
    last_blocks = set(sorted(set(block_ids))[-late_blocks:]) if late_blocks > 0 else set()
    names = set()
    for name, _param in model.named_parameters():
        if name.startswith("model.visual.merger.") or name.startswith("visual.merger."):
            names.add(name)
        else:
            match = re.search(r"(?:model\.)?visual\.blocks\.(\d+)\.", name)
            if match and int(match.group(1)) in last_blocks:
                names.add(name)
    return names


def output_param_names(model, last_layers: int, projections: Sequence[str]) -> set[str]:
    layer_ids = []
    for name, _param in model.named_parameters():
        match = re.search(r"model\.layers\.(\d+)\.", name)
        if match:
            layer_ids.append(int(match.group(1)))
    last = set(sorted(set(layer_ids))[-last_layers:])
    names = set()
    for name, _param in model.named_parameters():
        match = re.search(r"model\.layers\.(\d+)\.", name)
        if match and int(match.group(1)) in last and any(f".{proj}." in name for proj in projections):
            names.add(name)
        if name == "model.norm.weight":
            names.add(name)
    return names


def set_trainable(model, selected_names: set[str]) -> List[Tuple[str, torch.nn.Parameter, str]]:
    selected = []
    for name, param in model.named_parameters():
        flag = name in selected_names
        param.requires_grad_(flag)
        param.grad = None
        if flag:
            selected.append((name, param, locus_for_param(name)))
    return selected


def clear_grads(selected: Sequence[Tuple[str, torch.nn.Parameter, str]]) -> None:
    for _name, param, _locus in selected:
        param.grad = None


def collect_grads(selected: Sequence[Tuple[str, torch.nn.Parameter, str]]) -> Dict[str, Optional[torch.Tensor]]:
    return {name: (param.grad.detach().clone() if param.grad is not None else None) for name, param, _locus in selected}


def measure_space(model, processor, records_dis, records_aff, prototypes, selected, device, image_token_id, args, space: str) -> Dict[str, Any]:
    global_totals = BucketTotals()
    locus_totals: Dict[str, BucketTotals] = defaultdict(BucketTotals)
    batch_stats = RunningStats()
    per_batch = []
    n = min(args.num_batches, len(records_dis), len(records_aff))
    for idx in range(n):
        dis_record = records_dis[idx]
        aff_record = records_aff[idx]
        clear_grads(selected)
        if space == "visual":
            hidden, out_hw = visual_forward(model, processor, dis_record, device)
        else:
            hidden, out_hw = output_forward(model, processor, dis_record, device, image_token_id)
        loss = disambig_loss(hidden, dis_record, out_hw, args.contrast_temperature)
        loss.backward()
        grads_dis = collect_grads(selected)
        del hidden, loss
        torch.cuda.empty_cache()

        clear_grads(selected)
        if space == "visual":
            hidden, out_hw = visual_forward(model, processor, aff_record, device)
        else:
            hidden, out_hw = output_forward(model, processor, aff_record, device, image_token_id)
        loss = affordance_loss(hidden, aff_record, out_hw, prototypes, args.afford_temperature)
        loss.backward()

        batch_totals = BucketTotals()
        for name, param, locus in selected:
            grad_dis = grads_dis.get(name)
            grad_aff = param.grad
            global_totals.add(grad_dis, grad_aff)
            locus_totals[locus].add(grad_dis, grad_aff)
            batch_totals.add(grad_dis, grad_aff)
        cosine = batch_totals.cosine()
        batch_stats.add(cosine)
        per_batch.append({
            "batch": idx,
            "cosine": cosine,
            "disambig_record": f"{dis_record.episode_id}:{dis_record.step_idx}",
            "afford_record": f"{aff_record.episode_id}:{aff_record.step_idx}",
        })
        clear_grads(selected)
        del hidden, loss
        torch.cuda.empty_cache()
        if args.log_every and (idx + 1) % args.log_every == 0:
            print(f"{space}: processed {idx + 1}/{n}; last cosine={cosine:.6f}", flush=True)
    per_locus = {locus: totals.as_dict() for locus, totals in sorted(locus_totals.items())}
    most_negative = min(per_locus.items(), key=lambda item: item[1]["cosine"]) if per_locus else (None, {"cosine": 0.0})
    return {
        "space": space,
        "num_batches_requested": args.num_batches,
        "num_batches_used": n,
        "num_params": len(selected),
        "global": global_totals.as_dict(),
        "batch_cosine": batch_stats.as_dict(),
        "per_locus": per_locus,
        "most_negative_locus": most_negative[0],
        "most_negative_locus_cosine": most_negative[1]["cosine"],
        "per_batch": per_batch,
    }


def gate_verdict(visual: Dict[str, Any], output: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    visual_cos = visual["global"]["cosine"]
    batch = visual["batch_cosine"]
    output_cos = output["global"]["cosine"]
    separable = (
        visual_cos <= args.negative_threshold
        and batch["mean"] <= args.batch_negative_threshold
        and batch["negative_fraction"] >= args.negative_fraction_threshold
        and visual_cos <= output_cos - args.merger_output_margin
    )
    aligned = visual_cos >= args.aligned_threshold and batch["mean"] >= 0.0 and batch["negative_fraction"] <= 0.40
    if separable:
        verdict = "REPRESENTATION-SEPARABLE"
        consequent = "proxy objectives show visual-merger gradient conflict; review before training visual-merger LoRAs"
    elif aligned:
        verdict = "NOT SEPARABLE / ALIGNED"
        consequent = "proxy visual-merger gradients are aligned/non-conflicting; Route A has no proxy representation-conflict basis"
    else:
        verdict = "INCONCLUSIVE"
        consequent = "proxy objective signs are near-zero or unstable; sharpen element/affordance extraction before training"
    return {
        "verdict": verdict,
        "consequent": consequent,
        "visual_global_cosine": visual_cos,
        "visual_batch_mean": batch["mean"],
        "visual_negative_fraction": batch["negative_fraction"],
        "output_global_cosine": output_cos,
        "proxy_objectives": True,
    }


def write_outputs(output_dir: Path, visual: Dict[str, Any], output: Dict[str, Any], gate: Dict[str, Any], support: Dict[str, Any], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_locus.jsonl").open("w") as handle:
        for space, result in [("visual_merger", visual), ("output_control", output)]:
            for locus, stats in result["per_locus"].items():
                handle.write(json.dumps({"space": space, "locus": locus, **stats}, ensure_ascii=False) + "\n")
    payload = {
        "gate": gate,
        "support": support,
        "objective_extraction": {
            "status": "PROXY_OBJECTIVES",
            "element_regions": "GT bbox plus deterministic shifted spatial distractor bboxes; no explicit control_infos in downloaded a11y parquet",
            "disambig_loss": "InfoNCE: GT bbox representation vs shifted distractor-region representations",
            "afford_loss": "prototype cross-entropy over visual/output element representations with labels click/type/other from GT action type",
            "caveat": "Indicative proxy-level representation conflict measurement; reconfirm if real UIA control_infos become available.",
        },
        "visual_merger": {key: value for key, value in visual.items() if key != "per_batch"},
        "output_control": {key: value for key, value in output.items() if key != "per_batch"},
        "args": vars(args),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "summary.md").write_text(render_summary(payload))


def render_locus_table(result: Dict[str, Any], limit: int = 14) -> List[str]:
    rows = ["| locus | cosine | disambig/afford norm ratio | n tensors |", "|---|---:|---:|---:|"]
    for locus, stats in sorted(result["per_locus"].items(), key=lambda item: item[1]["cosine"])[:limit]:
        rows.append(f"| `{locus}` | {stats['cosine']:.6f} | {stats['disambig_afford_norm_ratio']:.3f} | {stats['n_tensors']} |")
    return rows


def render_summary(payload: Dict[str, Any]) -> str:
    gate = payload["gate"]
    visual = payload["visual_merger"]
    output = payload["output_control"]
    support = payload["support"]
    obj = payload["objective_extraction"]
    lines = [
        "# Phase 1c-pre Representation Conflict Gate",
        "",
        "## Gate Verdict",
        "",
        f"**{gate['verdict']}**",
        "",
        gate["consequent"],
        "",
        "## Objective Construction",
        "",
        f"- extraction status: `{obj['status']}`",
        f"- element regions: {obj['element_regions']}",
        f"- disambiguation loss: {obj['disambig_loss']}",
        f"- affordance loss: {obj['afford_loss']}",
        f"- caveat: {obj['caveat']}",
        "",
        "## Support",
        "",
    ]
    for key, value in support.items():
        lines.append(f"- {key}: `{value}`")
    lines += [
        "",
        "## Visual Merger / Late Visual Blocks",
        "",
        f"- global cosine: `{visual['global']['cosine']:.6f}`",
        f"- batch mean/std: `{visual['batch_cosine']['mean']:.6f}` / `{visual['batch_cosine']['std']:.6f}`",
        f"- negative fraction: `{visual['batch_cosine']['negative_fraction']:.6f}`",
        f"- most negative locus: `{visual['most_negative_locus']}` = `{visual['most_negative_locus_cosine']:.6f}`",
        "",
    ]
    lines += render_locus_table(visual)
    lines += [
        "",
        "## Output-Layer Contrast Control",
        "",
        f"- global cosine: `{output['global']['cosine']:.6f}`",
        f"- batch mean/std: `{output['batch_cosine']['mean']:.6f}` / `{output['batch_cosine']['std']:.6f}`",
        f"- negative fraction: `{output['batch_cosine']['negative_fraction']:.6f}`",
        f"- most negative locus: `{output['most_negative_locus']}` = `{output['most_negative_locus_cosine']:.6f}`",
        "",
    ]
    lines += render_locus_table(output)
    lines += [
        "",
        "## Decision Rule",
        "",
        "REPRESENTATION-SEPARABLE requires meaningfully negative, sign-stable visual-merger cosine that is more negative than output control. NOT SEPARABLE / ALIGNED is a stable non-negative visual-merger result. PROXY_OBJECTIVES means the verdict is indicative and should be rechecked if real control_infos become available.",
        "",
        "No LoRA or source training is performed in this phase.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="checkpoints/gui360-fullparam-sft-step250")
    parser.add_argument("--dataset_dir", default="datasets/gui360-balanced-a11y")
    parser.add_argument("--output_dir", default="outputs/candidate_orthogonality/phase1c_pre_repr_conflict")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_batches", type=int, default=50)
    parser.add_argument("--prototype_per_class", type=int, default=12)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--late_visual_blocks", type=int, default=4)
    parser.add_argument("--output_last_layers", type=int, default=4)
    parser.add_argument("--output_projections", nargs="+", default=["q_proj", "k_proj"])
    parser.add_argument("--contrast_temperature", type=float, default=0.1)
    parser.add_argument("--afford_temperature", type=float, default=0.1)
    parser.add_argument("--negative_threshold", type=float, default=-0.05)
    parser.add_argument("--batch_negative_threshold", type=float, default=-0.02)
    parser.add_argument("--negative_fraction_threshold", type=float, default=0.60)
    parser.add_argument("--aligned_threshold", type=float, default=0.02)
    parser.add_argument("--merger_output_margin", type=float, default=0.03)
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print("Loading records...", flush=True)
    records = read_records(args.dataset_dir, args.max_records)
    disambig_records, affordance_records, prototype_records, support = select_records(records, args)
    if len(disambig_records) < args.num_batches or len(affordance_records) < args.num_batches:
        raise SystemExit(f"Insufficient records for {args.num_batches} batches: {support}")

    print("Loading model...", flush=True)
    config = AutoConfig.from_pretrained(args.model_path)
    patch_legacy_mrope_config(config)
    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False
    image_token_id = int(getattr(config, "image_token_id", getattr(model.config, "image_token_id")))

    for param in model.parameters():
        param.requires_grad_(False)

    print("Computing prototypes...", flush=True)
    visual_prototypes = compute_prototypes(model, processor, prototype_records, device, image_token_id, "visual")
    output_prototypes = compute_prototypes(model, processor, prototype_records, device, image_token_id, "output")

    print("Measuring visual merger/late visual blocks...", flush=True)
    visual_selected = set_trainable(model, visual_param_names(model, args.late_visual_blocks))
    visual_result = measure_space(model, processor, disambig_records, affordance_records, visual_prototypes, visual_selected, device, image_token_id, args, "visual")

    print("Measuring output contrast control...", flush=True)
    output_selected = set_trainable(model, output_param_names(model, args.output_last_layers, args.output_projections))
    output_result = measure_space(model, processor, disambig_records, affordance_records, output_prototypes, output_selected, device, image_token_id, args, "output")

    gate = gate_verdict(visual_result, output_result, args)
    write_outputs(Path(args.output_dir), visual_result, output_result, gate, support, args)
    print(f"Wrote {Path(args.output_dir) / 'summary.md'}")
    print(f"Wrote {Path(args.output_dir) / 'per_locus.jsonl'}")
    print(f"GATE: {gate['verdict']} - {gate['consequent']}")


if __name__ == "__main__":
    main()
