#!/usr/bin/env python3
"""Analyze V13 communication-gate information on WHAT/WHERE target tokens.

The diagnostic runs teacher-forced GT tool calls on GT screens and asks whether
communication gates carry role information:

- WHERE coordinate tokens should need WHAT -> WHERE communication (`g_21`).
- WHAT/action tokens may need less cross-expert communication.

It does not train; it only records gate statistics for a cooperative checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from v13_gui_360.iterative_cooperative_wrapper import IterativeCooperativeVLMWrapper  # noqa: E402
from v15_gui_360.train_trajectory_gspo import V15TrajectoryGSPOTrainer  # noqa: E402
from v23_visual_transition.train_offline_grpo import build_eval_style_messages  # noqa: E402
from v23_visual_transition.train_where_what_routed_sft import token_role_labels  # noqa: E402


@dataclass
class RunningStats:
    n: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    min_value: float = math.inf
    max_value: float = -math.inf

    def add(self, value: float, count: int = 1) -> None:
        if count <= 0 or not math.isfinite(value):
            return
        self.n += count
        self.total += value * count
        self.total_sq += value * value * count
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

    def as_dict(self) -> Dict[str, float]:
        if self.n == 0:
            return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        mean = self.total / self.n
        var = max(self.total_sq / self.n - mean * mean, 0.0)
        return {
            "n": self.n,
            "mean": mean,
            "std": math.sqrt(var),
            "min": self.min_value,
            "max": self.max_value,
        }


@dataclass
class GateAccumulator:
    stats: Dict[str, RunningStats] = field(default_factory=lambda: defaultdict(RunningStats))

    def add(self, key: str, tensor: torch.Tensor) -> None:
        if tensor.numel() == 0:
            return
        values = tensor.detach().float().reshape(-1)
        self.stats[key].add(float(values.mean().item()), int(values.numel()))

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {key: value.as_dict() for key, value in sorted(self.stats.items())}


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


def build_target_text(row: Dict[str, Any]) -> str:
    full_tool_call = row.get("full_tool_call") or {}
    payload = json.dumps(full_tool_call, ensure_ascii=False, indent=2)
    return f"<tool_call>\n{payload}\n</tool_call>"


def safe_mean(tensor: torch.Tensor, mask: torch.Tensor) -> Optional[torch.Tensor]:
    if tensor.numel() == 0 or not mask.any():
        return None
    return tensor[mask].float().mean()


def role_name(role_value: int) -> str:
    return "where" if role_value == 0 else "what"


def load_model(args) -> tuple[AutoProcessor, IterativeCooperativeVLMWrapper]:
    processor = AutoProcessor.from_pretrained(args.model_path)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if args.image_max_pixels > 0:
        processor.image_processor.max_pixels = args.image_max_pixels

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    V15TrajectoryGSPOTrainer._patch_legacy_mrope_config(config)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(args.device)

    config_path = os.path.join(args.coop_checkpoint, "cooperative_config.json")
    with open(config_path) as handle:
        coop_config = json.load(handle)

    model = IterativeCooperativeVLMWrapper(
        base_model=base_model,
        lora_r=coop_config.get("lora_r", args.lora_r),
        lora_alpha=coop_config.get("lora_alpha", args.lora_alpha),
        lora_dropout=args.lora_dropout,
        target_modules=coop_config.get("target_modules", args.target_modules),
        balance_weight=0.0,
        num_comm_rounds=coop_config.get("num_comm_rounds", args.num_comm_rounds),
    ).to(args.device)
    model.load_cooperative(args.coop_checkpoint, device=args.device)
    model.eval()
    model.enable_gate_recording()
    return processor, model


def analyze_row(
    row: Dict[str, Any],
    processor: AutoProcessor,
    model: IterativeCooperativeVLMWrapper,
    device: str,
    accum: GateAccumulator,
    row_records: List[Dict[str, Any]],
) -> bool:
    image_path = row.get("image")
    if not image_path or not os.path.exists(image_path):
        return False

    image = Image.open(image_path).convert("RGB")
    messages = build_eval_style_messages(row["goal"], row.get("history", []), image_path)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_inputs = processor(text=[text], images=[image], return_tensors="pt", padding=False)
    prompt_inputs = {key: value.to(device) for key, value in prompt_inputs.items()}
    prompt_len = int(prompt_inputs["input_ids"].shape[1])

    target_text = build_target_text(row)
    response_ids, role_labels = token_role_labels(processor.tokenizer, target_text)
    response_ids = response_ids.to(device)
    role_labels = role_labels.to(device)
    if response_ids.numel() == 0 or (role_labels >= 0).sum().item() == 0:
        return False

    full_ids = torch.cat([prompt_inputs["input_ids"][0], response_ids], dim=0).unsqueeze(0)
    attention_mask = torch.ones_like(full_ids)
    fwd_kwargs = {"input_ids": full_ids, "attention_mask": attention_mask}
    for key in ("pixel_values", "image_grid_thw"):
        if key in prompt_inputs:
            fwd_kwargs[key] = prompt_inputs[key]

    model.enable_gate_recording()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _ = model(**fwd_kwargs)

    resp_start = max(prompt_len - 1, 0)
    resp_end = resp_start + int(role_labels.shape[0])
    where_mask = role_labels == 0
    what_mask = role_labels == 1
    metadata = row.get("metadata") or {}
    family = metadata.get("family") or "unknown"
    action_type = metadata.get("action_type") or "unknown"

    row_g21_where: List[float] = []
    row_g21_what: List[float] = []
    row_g12_where: List[float] = []
    row_g12_what: List[float] = []

    for module_idx, module in enumerate(model.coop_modules):
        records = getattr(module, "_last_comm_gate_records", None) or []
        if not records:
            continue
        layer_idx = model._module_to_layer[module_idx]
        for record in records:
            round_idx = int(record["round"])
            for direction in ("g_12", "g_21"):
                values = record[direction].squeeze(0).squeeze(-1)[resp_start:resp_end]
                if values.shape[0] != role_labels.shape[0]:
                    continue
                for mask, role in ((where_mask, "where"), (what_mask, "what")):
                    selected = values[mask]
                    if selected.numel() == 0:
                        continue
                    accum.add(f"global/{direction}/{role}", selected)
                    accum.add(f"family/{family}/{direction}/{role}", selected)
                    accum.add(f"action/{action_type}/{direction}/{role}", selected)
                    accum.add(f"layer/{layer_idx:02d}/{direction}/{role}", selected)
                    accum.add(f"round/{round_idx}/{direction}/{role}", selected)
                    mean_value = float(selected.float().mean().item())
                    if direction == "g_21" and role == "where":
                        row_g21_where.append(mean_value)
                    elif direction == "g_21" and role == "what":
                        row_g21_what.append(mean_value)
                    elif direction == "g_12" and role == "where":
                        row_g12_where.append(mean_value)
                    elif direction == "g_12" and role == "what":
                        row_g12_what.append(mean_value)

    def avg(values: Iterable[float]) -> Optional[float]:
        values = list(values)
        return float(sum(values) / len(values)) if values else None

    row_records.append({
        "episode_id": row.get("episode_id"),
        "step_idx": row.get("step_idx"),
        "family": family,
        "action_type": action_type,
        "n_where": int(where_mask.sum().item()),
        "n_what": int(what_mask.sum().item()),
        "g21_where": avg(row_g21_where),
        "g21_what": avg(row_g21_what),
        "g12_where": avg(row_g12_where),
        "g12_what": avg(row_g12_what),
    })
    return True


def add_delta(summary: Dict[str, Any], left: str, right: str, out_key: str) -> None:
    stats = summary.get("stats", {})
    if left in stats and right in stats:
        summary[out_key] = stats[left]["mean"] - stats[right]["mean"]


def write_markdown(path: str, summary: Dict[str, Any]) -> None:
    stats = summary["stats"]
    lines = [
        "# V23 Communication Gate Information Diagnostic",
        "",
        f"- checkpoint: `{summary['coop_checkpoint']}`",
        f"- rows attempted: `{summary['rows_attempted']}`",
        f"- rows analyzed: `{summary['rows_analyzed']}`",
        "",
        "## Global Role Means",
        "",
        "| direction | role | n | mean | std | |mean-0.5| |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for direction in ("g_12", "g_21"):
        for role in ("what", "where"):
            key = f"global/{direction}/{role}"
            value = stats.get(key, {"n": 0, "mean": 0.0, "std": 0.0})
            mean = value["mean"]
            lines.append(
                f"| {direction} | {role} | {value['n']} | {mean:.6f} | "
                f"{value['std']:.6f} | {abs(mean - 0.5):.6f} |"
            )
    lines.extend([
        "",
        "## Deltas",
        "",
        f"- `g21_where - g21_what`: `{summary.get('delta_g21_where_minus_what', 0.0):.6f}`",
        f"- `g12_where - g12_what`: `{summary.get('delta_g12_where_minus_what', 0.0):.6f}`",
        f"- `(g21-g12)_where`: `{summary.get('delta_where_g21_minus_g12', 0.0):.6f}`",
        f"- `(g21-g12)_what`: `{summary.get('delta_what_g21_minus_g12', 0.0):.6f}`",
        "",
        "## Top Layer Role Deltas",
        "",
        "| layer | g21 where-what | g12 where-what |",
        "|---:|---:|---:|",
    ])
    layer_rows = []
    for layer in range(28):
        g21_w = stats.get(f"layer/{layer:02d}/g_21/where")
        g21_t = stats.get(f"layer/{layer:02d}/g_21/what")
        g12_w = stats.get(f"layer/{layer:02d}/g_12/where")
        g12_t = stats.get(f"layer/{layer:02d}/g_12/what")
        if not all((g21_w, g21_t, g12_w, g12_t)):
            continue
        d21 = g21_w["mean"] - g21_t["mean"]
        d12 = g12_w["mean"] - g12_t["mean"]
        layer_rows.append((abs(d21) + abs(d12), layer, d21, d12))
    for _, layer, d21, d12 in sorted(layer_rows, reverse=True)[:12]:
        lines.append(f"| {layer} | {d21:.6f} | {d12:.6f} |")
    lines.append("")
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze V13 comm-gate role information")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--coop_checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_rows", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image_max_pixels", type=int, default=602112)
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--num_comm_rounds", type=int, default=2)
    parser.add_argument(
        "--target_modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = read_jsonl(args.data, args.max_rows)
    processor, model = load_model(args)
    accum = GateAccumulator()
    row_records: List[Dict[str, Any]] = []
    analyzed = 0

    for idx, row in enumerate(rows):
        try:
            if analyze_row(row, processor, model, args.device, accum, row_records):
                analyzed += 1
        except Exception as exc:
            print(f"row {idx} failed: {exc}")

    summary: Dict[str, Any] = {
        "model_path": args.model_path,
        "coop_checkpoint": args.coop_checkpoint,
        "data": args.data,
        "rows_attempted": len(rows),
        "rows_analyzed": analyzed,
        "stats": accum.to_dict(),
        "row_records": row_records,
    }
    add_delta(summary, "global/g_21/where", "global/g_21/what", "delta_g21_where_minus_what")
    add_delta(summary, "global/g_12/where", "global/g_12/what", "delta_g12_where_minus_what")
    add_delta(summary, "global/g_21/where", "global/g_12/where", "delta_where_g21_minus_g12")
    add_delta(summary, "global/g_21/what", "global/g_12/what", "delta_what_g21_minus_g12")

    json_path = os.path.join(args.output_dir, "comm_gate_information_summary.json")
    md_path = os.path.join(args.output_dir, "comm_gate_information_report.md")
    with open(json_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    write_markdown(md_path, summary)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()