#!/usr/bin/env python3
"""E0.2-R temperature-ladder audit for REG-INJECT and the locked SFT base."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.hetero_inject_phase0 import (  # noqa: E402
    file_sha256,
    generate_batch,
    native_sampling_score,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from scripts.hetero_inject_eval import parse_prediction  # noqa: E402
from scripts.minimal_validation import build_messages, load_model_for_eval  # noqa: E402
from scripts.rl_feasibility_sampling import action_key, sanitize_jsonable  # noqa: E402


TEMPERATURES = (0.0, 0.7, 1.0, 1.3, 1.5)
ARMS = ("reg_inject", "base_sft")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize_jsonable(dict(row)), ensure_ascii=False) + "\n")


def target_sort_key(target_id: str) -> tuple[Any, int]:
    episode_id, step_text = str(target_id).rsplit(":", 1)
    episode_key: Any = int(episode_id) if episode_id.isdigit() else episode_id
    return episode_key, int(step_text)


def load_positive_pairs(pairs_path: Path, prior_eval_path: Path) -> list[dict[str, Any]]:
    pairs = {str(row["target_id"]): row for row in read_jsonl(pairs_path)}
    positive_ids = []
    for episode in read_jsonl(prior_eval_path):
        for step in episode.get("steps", []) or []:
            if step.get("is_blind_injected") and step.get("greedy_is_chosen"):
                positive_ids.append(str(step["target_id"]))
    positive_ids = sorted(set(positive_ids), key=target_sort_key)
    missing = [target_id for target_id in positive_ids if target_id not in pairs]
    if missing:
        raise ValueError(f"positive target IDs missing from pair file: {missing[:5]}")
    if len(positive_ids) != 48:
        raise ValueError(f"expected 48 cached positive positions, found {len(positive_ids)}")
    return [pairs[target_id] for target_id in positive_ids]


def freeze_run_manifest(args: argparse.Namespace) -> None:
    pairs_path = Path(args.pairs)
    prior_path = Path(args.prior_eval)
    positive_pairs = load_positive_pairs(pairs_path, prior_path)
    positive_payload = [
        {
            "target_id": str(row["target_id"]),
            "chosen_action_key": str(row["chosen_action_key"]),
        }
        for row in positive_pairs
    ]
    positive_sha256 = hashlib.sha256(
        json.dumps(positive_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    phase0_identity = read_json(Path(args.phase0_identity)) or {}
    worker_commands = []
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            command = (process_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError):
            continue
        if "hetero_inject_e02r.py run-ladder-shard" in command:
            worker_commands.append(command)
    protocol = {
        "temperatures": list(TEMPERATURES),
        "arms": list(ARMS),
        "k": 16,
        "sample_batch_size": 16,
        "top_p": 0.95,
        "max_new_tokens": 384,
        "seed": 42,
        "coord_bucket": 25,
        "match_threshold": 0.5,
        "parser": "strict_phase0_native_sampling_score",
        "constrained_decoding": False,
    }
    payload = {
        "version": "E0.2-R-v1.1",
        "created_unix": time.time(),
        "fresh_output_directory_attestation": True,
        "pairs_path": str(pairs_path),
        "pairs_sha256": file_sha256(pairs_path),
        "prior_eval_path": str(prior_path),
        "prior_eval_sha256": file_sha256(prior_path),
        "positive_positions": positive_payload,
        "positive_positions_sha256": positive_sha256,
        "positive_count": len(positive_payload),
        "model_path": args.model_path,
        "adapter_dir": args.adapter_dir,
        "complete_model_adapter_manifest_hash": phase0_identity.get("complete_manifest_hash"),
        "checkpoint_canonical_hash": phase0_identity.get("checkpoint_canonical_hash"),
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "worker_count": len(worker_commands),
        "worker_commands": sorted(worker_commands),
    }
    write_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2), flush=True)


def sample_k_at_temperature(
    model: Any,
    processor: Any,
    device: torch.device,
    messages: list[dict[str, Any]],
    image_path: str,
    *,
    k: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[str]:
    outputs: list[str] = []
    do_sample = temperature > 0.0
    while len(outputs) < k:
        size = min(batch_size, k - len(outputs))
        outputs.extend(generate_batch(
            model,
            processor,
            device,
            messages,
            image_path,
            n=size,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        ))
    return outputs


def evaluate_cell_with_model(
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    device: torch.device,
    selected: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    temperature: float,
    output_path: Path,
    overwrite: bool,
) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if overwrite:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        done: set[str] = set()
    else:
        done = {str(row["target_id"]) for row in read_jsonl(output_path)}

    for pair in tqdm(selected, desc=f"{arm}-T{temperature:g}"):
        target_id = str(pair["target_id"])
        if target_id in done:
            continue
        target_seed = args.seed + int(hashlib.sha256(
            f"{arm}|{temperature:g}|{target_id}".encode("utf-8")
        ).hexdigest()[:8], 16)
        torch.manual_seed(target_seed)
        torch.cuda.manual_seed_all(target_seed)

        messages, image_w, image_h = build_messages(
            pair["goal"],
            pair.get("history") or [],
            pair["image"],
            args.image_max_pixels,
        )
        texts = sample_k_at_temperature(
            model,
            processor,
            device,
            messages,
            pair["image"],
            k=args.k,
            batch_size=args.sample_batch_size,
            temperature=temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        chosen_key = str(pair["chosen_action_key"])
        samples = []
        for sample_index, text in enumerate(texts):
            scored = native_sampling_score(
                text,
                pair["gt_action"],
                image_w,
                image_h,
                args.match_threshold,
                args.coord_bucket,
            )
            token_count = len(processor.tokenizer.encode(text, add_special_tokens=False))
            samples.append({
                **scored,
                "sample_index": sample_index,
                "raw_output": text,
                "generated_token_count": token_count,
                "hit_max_new_tokens": token_count >= args.max_new_tokens - 1,
                "has_tool_call_open": "<tool_call>" in text,
                "has_tool_call_close": "</tool_call>" in text,
                "chosen": scored["action_key"] == chosen_key,
            })
        chosen_count = sum(1 for sample in samples if sample["chosen"])
        try:
            chosen_type = str(json.loads(chosen_key).get("type") or "unknown")
        except (TypeError, ValueError, json.JSONDecodeError):
            chosen_type = "unknown"
        append_jsonl(output_path, {
            "arm": arm,
            "temperature": temperature,
            "target_id": target_id,
            "episode_id": str(pair["episode_id"]),
            "step_idx": int(pair["step_idx"]),
            "chosen_action_key": chosen_key,
            "chosen_action_type": chosen_type,
            "k": args.k,
            "parse_count": sum(1 for sample in samples if sample["parse_ok"]),
            "chosen_count": chosen_count,
            "chosen_sampled_any": chosen_count > 0,
            "correct_count": sum(1 for sample in samples if sample["success"]),
            "max_length_count": sum(1 for sample in samples if sample["hit_max_new_tokens"]),
            "seed": target_seed,
            "base_seed": args.seed,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "sample_batch_size": args.sample_batch_size,
            "samples": samples,
        })
        done.add(target_id)
        torch.cuda.empty_cache()

    print(json.dumps({
        "arm": arm,
        "temperature": temperature,
        "output": str(output_path),
        "rows": len(read_jsonl(output_path)),
    }, indent=2), flush=True)


def run_cell(args: argparse.Namespace) -> None:
    if not str(args.arm).strip():
        raise ValueError("arm name is required")
    if args.temperature not in TEMPERATURES:
        raise ValueError(f"temperature {args.temperature} is outside the frozen ladder {TEMPERATURES}")
    if args.k != 16 and not (args.diagnostic_k32 and args.k == 32):
        raise ValueError("E0.2-R requires K=16; only an explicit diagnostic K=32 arm is allowed")

    positive_pairs = load_positive_pairs(Path(args.pairs), Path(args.prior_eval))
    selected = positive_pairs[args.start:args.end]
    if args.arm == "base_sft":
        args.adapter_dir = ""
    elif not args.adapter_dir:
        raise ValueError("REG-INJECT arm requires --adapter-dir")
    model, processor, device = load_model_for_eval(args)
    evaluate_cell_with_model(
        args,
        model,
        processor,
        device,
        selected,
        arm=args.arm,
        temperature=args.temperature,
        output_path=Path(args.output),
        overwrite=args.overwrite,
    )


def run_mini_ladder(args: argparse.Namespace) -> None:
    if not str(args.arm).strip() or args.arm == "base_sft":
        raise ValueError("Phase-1 mini-ladder requires a named adapter arm")
    if not args.adapter_dir:
        raise ValueError("Phase-1 mini-ladder requires --adapter-dir")
    if args.k != 16 or args.sample_batch_size != 16:
        raise ValueError("Phase-1 mini-ladder requires K=16 and sample_batch_size=16")
    positive_pairs = load_positive_pairs(Path(args.pairs), Path(args.prior_eval))
    model, processor, device = load_model_for_eval(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for temperature in (1.0, 1.3, 1.5):
        tag = f"{args.arm}_T{str(temperature).replace('.', 'p')}"
        evaluate_cell_with_model(
            args,
            model,
            processor,
            device,
            positive_pairs,
            arm=args.arm,
            temperature=temperature,
            output_path=output_dir / f"{tag}.jsonl",
            overwrite=args.overwrite,
        )


def run_ladder_shard(args: argparse.Namespace) -> None:
    if args.k != 16 or args.sample_batch_size != 16:
        raise ValueError("frozen E0.2-R ladder requires K=16 and sample_batch_size=16")
    positive_pairs = load_positive_pairs(Path(args.pairs), Path(args.prior_eval))
    selected = positive_pairs[args.start:args.end]
    if len(selected) != args.end - args.start:
        raise ValueError("ladder shard slice is incomplete")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reg_adapter_dir = args.adapter_dir

    for arm in ARMS:
        args.adapter_dir = reg_adapter_dir if arm == "reg_inject" else ""
        model, processor, device = load_model_for_eval(args)
        try:
            for temperature in TEMPERATURES:
                tag = f"{'reg' if arm == 'reg_inject' else 'base'}_T{str(temperature).replace('.', 'p')}_shard_{args.shard_index}"
                evaluate_cell_with_model(
                    args,
                    model,
                    processor,
                    device,
                    selected,
                    arm=arm,
                    temperature=temperature,
                    output_path=output_dir / f"{tag}.jsonl",
                    overwrite=args.overwrite,
                )
        finally:
            del model
            del processor
            torch.cuda.empty_cache()


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positions = len(rows)
    samples = [sample for row in rows for sample in row.get("samples", [])]
    by_type: dict[str, dict[str, Any]] = {}
    for action_type in sorted({str(row.get("chosen_action_type") or "unknown") for row in rows}):
        type_rows = [row for row in rows if str(row.get("chosen_action_type") or "unknown") == action_type]
        type_samples = [sample for row in type_rows for sample in row.get("samples", [])]
        by_type[action_type] = {
            "positions": len(type_rows),
            "samples": len(type_samples),
            "parse_rate": sum(1 for sample in type_samples if sample.get("parse_ok")) / max(1, len(type_samples)),
            "chosen_frequency": sum(1 for sample in type_samples if sample.get("chosen")) / max(1, len(type_samples)),
            "chosen_sampled_any": sum(1 for row in type_rows if row.get("chosen_sampled_any")) / max(1, len(type_rows)),
        }
    return {
        "positions": positions,
        "samples": len(samples),
        "parse_rate": sum(1 for sample in samples if sample.get("parse_ok")) / max(1, len(samples)),
        "chosen_frequency": sum(1 for sample in samples if sample.get("chosen")) / max(1, len(samples)),
        "chosen_sampled_any": sum(1 for row in rows if row.get("chosen_sampled_any")) / max(1, positions),
        "max_length_rate": sum(1 for sample in samples if sample.get("hit_max_new_tokens")) / max(1, len(samples)),
        "by_action_type": by_type,
    }


def merge_cell(args: argparse.Namespace) -> None:
    rows = []
    for path_text in args.shards:
        rows.extend(read_jsonl(Path(path_text)))
    ids = [str(row["target_id"]) for row in rows]
    if len(rows) != len(set(ids)):
        raise ValueError("duplicate target IDs across cell shards")
    rows = sorted(rows, key=lambda row: target_sort_key(str(row["target_id"])))
    expected_ids = {
        str(row["target_id"])
        for row in load_positive_pairs(Path(args.pairs), Path(args.prior_eval))
    }
    if len(rows) != 48:
        raise ValueError(f"expected 48 cell rows, found {len(rows)}")
    if set(ids) != expected_ids:
        raise ValueError("cell target IDs do not exactly match the frozen 48-position subset")
    if any(int(row.get("k") or 0) != 16 for row in rows):
        raise ValueError("cell contains non-K16 rows")
    protocol_values = {
        "arm": sorted({str(row.get("arm")) for row in rows}),
        "temperature": sorted({float(row.get("temperature")) for row in rows}),
        "top_p": sorted({float(row.get("top_p")) for row in rows}),
        "max_new_tokens": sorted({int(row.get("max_new_tokens")) for row in rows}),
        "sample_batch_size": sorted({int(row.get("sample_batch_size")) for row in rows}),
        "base_seed": sorted({int(row.get("base_seed")) for row in rows}),
    }
    if any(len(values) != 1 for values in protocol_values.values()):
        raise ValueError(f"cell protocol is inconsistent: {protocol_values}")
    write_jsonl(Path(args.output), rows)
    summary = {
        "arm": rows[0]["arm"],
        "temperature": rows[0]["temperature"],
        **summarize_rows(rows),
        "source_shards": args.shards,
        "protocol": {key: values[0] for key, values in protocol_values.items()},
        "exact_target_set": True,
    }
    write_json(Path(args.output).with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2), flush=True)


def extract_taxonomy_samples(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.cell))
    if len(rows) != 48:
        raise ValueError(f"taxonomy source must contain 48 positions, found {len(rows)}")
    if any(str(row.get("arm")) != "reg_inject" or float(row.get("temperature")) != 1.5 for row in rows):
        raise ValueError("taxonomy source must be the REG-INJECT T=1.5 cell")
    candidates = []
    for row in sorted(rows, key=lambda value: target_sort_key(str(value["target_id"]))):
        for sample in sorted(row.get("samples", []), key=lambda value: int(value.get("sample_index") or 0)):
            if sample.get("parse_ok"):
                continue
            candidates.append({
                "target_id": str(row["target_id"]),
                "chosen_action_type": row.get("chosen_action_type"),
                "sample_index": int(sample.get("sample_index") or 0),
                "generated_token_count": int(sample.get("generated_token_count") or 0),
                "hit_max_new_tokens": bool(sample.get("hit_max_new_tokens")),
                "has_tool_call_open": bool(sample.get("has_tool_call_open")),
                "has_tool_call_close": bool(sample.get("has_tool_call_close")),
                "raw_output": sample.get("raw_output", ""),
                "manual_label": None,
                "manual_rationale": None,
            })
    selected = candidates[: args.count]
    if len(selected) != args.count:
        raise ValueError(f"not enough unparsable samples: requested {args.count}, found {len(selected)}")
    write_jsonl(Path(args.output), selected)
    print(json.dumps({
        "source": args.cell,
        "eligible_unparsable": len(candidates),
        "selected": len(selected),
        "output": args.output,
    }, indent=2), flush=True)


def score_taxonomy(args: argparse.Namespace) -> None:
    samples = read_jsonl(Path(args.samples))
    labels_payload = read_json(Path(args.labels)) or {}
    protocol = read_json(Path(args.protocol)) or {}
    labels = list(labels_payload.get("labels") or [])
    if len(samples) != 20 or len(labels) != 20:
        raise ValueError(f"taxonomy requires exactly 20 samples/labels, found {len(samples)}/{len(labels)}")
    allowed = set((protocol.get("exclusive_primary_labels") or {}).keys())
    sample_ids = [(str(row["target_id"]), int(row["sample_index"])) for row in samples]
    label_ids = [(str(row["target_id"]), int(row["sample_index"])) for row in labels]
    if sample_ids != label_ids or len(set(label_ids)) != 20:
        raise ValueError("manual taxonomy identities/order do not exactly match the frozen extractor output")
    if any(str(row.get("label")) not in allowed for row in labels):
        raise ValueError("manual taxonomy contains an unknown label")
    if any(not str(row.get("rationale") or "").strip() for row in labels):
        raise ValueError("every manual taxonomy label requires a rationale")

    labeled_rows = []
    for sample, label in zip(samples, labels):
        labeled_rows.append({
            **sample,
            "manual_label": str(label["label"]),
            "manual_rationale": str(label["rationale"]),
        })
    counts = Counter(row["manual_label"] for row in labeled_rows)
    for label in allowed:
        counts.setdefault(label, 0)
    max_count = max(counts.values())
    tied = {label for label, count in counts.items() if count == max_count}
    tie_break = list((protocol.get("dominant_class") or {}).get("tie_break_order") or [])
    dominant = next((label for label in tie_break if label in tied), None)
    if dominant is None:
        raise ValueError("cannot resolve dominant taxonomy class using frozen tie-break rule")
    route = (protocol.get("repair_routes") or {}).get(dominant)
    if not route:
        raise ValueError(f"missing predeclared repair route for {dominant}")

    write_jsonl(Path(args.output_labeled), labeled_rows)
    summary = {
        "protocol_version": protocol.get("version"),
        "source_cell": {"arm": "REG-INJECT", "temperature": 1.5},
        "source_samples": args.samples,
        "manual_labels": args.labels,
        "labeled_count": len(labeled_rows),
        "counts": dict(sorted(counts.items())),
        "dominant_class": dominant,
        "selected_repair_route": route,
        "constrained_decoding_used": False,
    }
    write_json(Path(args.output_summary), summary)
    print(json.dumps(summary, indent=2), flush=True)


ALLOWED_LENIENT_ACTIONS = ("click", "type", "drag", "wheel_mouse_input")
FUNCTION_ALIASES = ("function", ".function", "-function", "funktionname", "功能", "action")
COORDINATE_ALIASES = ("coordinate", "coordinates", "coord", "coords", "coordintate", "position", "坐标")


def lenient_parse_action(text: str, max_chars: int = 500) -> tuple[dict[str, Any] | None, str]:
    strict = parse_prediction(text)
    if strict is not None:
        return strict, "strict"
    if not text or len(text) > max_chars:
        return None, "rejected_empty_or_long"
    payload = text.split("<tool_call>", 1)[-1].split("</tool_call>", 1)[0]
    lowered = payload.lower()

    action_hits: list[tuple[int, str]] = []
    for alias in FUNCTION_ALIASES:
        start = 0
        alias_lower = alias.lower()
        while True:
            index = lowered.find(alias_lower, start)
            if index < 0:
                break
            window = lowered[index:index + 90]
            match = re.search(r"\b(click|type|drag|wheel_mouse_input)\b", window)
            if match:
                action_hits.append((index + match.start(), match.group(1)))
            start = index + len(alias_lower)
    leading = re.search(r"[\"']\s*(click|type|drag|wheel_mouse_input)\s*[\"']\s*[:(]", lowered)
    if leading:
        action_hits.append((leading.start(1), leading.group(1)))
    pseudo = re.search(r"\b(click|type|drag|wheel_mouse_input)\s*\(", lowered)
    if pseudo:
        action_hits.append((pseudo.start(1), pseudo.group(1)))
    action_hits.sort()
    distinct_actions = {action for _index, action in action_hits}
    if len(distinct_actions) != 1:
        return None, "rejected_missing_or_conflicting_action"
    action_type = action_hits[0][1]

    coordinate_numbers: list[float] = []
    alias_positions = []
    for alias in COORDINATE_ALIASES:
        position = lowered.find(alias.lower())
        if position >= 0:
            alias_positions.append(position)
    if alias_positions:
        coordinate_window = payload[min(alias_positions):min(alias_positions) + 150]
        for number_text in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", coordinate_window):
            try:
                value = float(number_text)
            except ValueError:
                continue
            if math.isfinite(value):
                coordinate_numbers.append(value)

    if action_type == "click":
        if len(coordinate_numbers) < 2:
            return None, "rejected_click_without_xy"
        return {"action": "click", "coordinate": coordinate_numbers[:2]}, "lenient"
    if action_type == "type":
        text_value = ""
        text_match = re.search(r"(?:text|keys)\s*[\"']?\s*[:=]\s*[\"']([^\"']*)[\"']", payload, re.IGNORECASE)
        if text_match:
            text_value = text_match.group(1)
        action: dict[str, Any] = {"action": "type", "text": text_value}
        if len(coordinate_numbers) >= 2:
            action["coordinate"] = coordinate_numbers[:2]
        return action, "lenient"
    if action_type == "drag":
        if len(coordinate_numbers) < 4:
            return None, "rejected_drag_without_four_coords"
        return {
            "action": "swipe",
            "coordinate": coordinate_numbers[:2],
            "endCoordinate": coordinate_numbers[2:4],
        }, "lenient"
    if action_type == "wheel_mouse_input":
        if len(coordinate_numbers) < 2:
            return None, "rejected_wheel_without_xy"
        return {"action": "swipe", "coordinate": coordinate_numbers[:2]}, "lenient"
    return None, "rejected_unknown"


def rescore_lenient(args: argparse.Namespace) -> None:
    protocol = read_json(Path(args.protocol)) or {}
    if protocol.get("version") != "lenient-tool-call-v1" or not protocol.get("predeclared_before_rescoring"):
        raise ValueError("lenient parser protocol is missing or was not predeclared")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for path_text in args.cell_files:
        path = Path(path_text)
        rows = read_jsonl(path)
        rescored_rows = []
        for row in rows:
            chosen_key = str(row["chosen_action_key"])
            rescored_samples = []
            for sample in row.get("samples", []):
                action, parser_mode = lenient_parse_action(str(sample.get("raw_output") or ""))
                key = action_key(action, 25)
                rescored_samples.append({
                    **sample,
                    "strict_parse_ok": bool(sample.get("parse_ok")),
                    "strict_action_key": sample.get("action_key"),
                    "parser_mode": parser_mode,
                    "parse_ok": action is not None,
                    "pred_action": action,
                    "action_key": key,
                    "chosen": key == chosen_key,
                })
            chosen_count = sum(1 for sample in rescored_samples if sample["chosen"])
            rescored_rows.append({
                **row,
                "parser": "lenient-tool-call-v1",
                "parse_count": sum(1 for sample in rescored_samples if sample["parse_ok"]),
                "chosen_count": chosen_count,
                "chosen_sampled_any": chosen_count > 0,
                "samples": rescored_samples,
            })
        output = out_dir / path.name
        write_jsonl(output, rescored_rows)
        summary = {
            "arm": rescored_rows[0]["arm"],
            "temperature": rescored_rows[0]["temperature"],
            "parser": "lenient-tool-call-v1",
            **summarize_rows(rescored_rows),
            "source_strict_cell": str(path),
            "protocol": str(args.protocol),
        }
        write_json(output.with_suffix(".summary.json"), summary)
        outputs.append(str(output))

    parity_rows = read_jsonl(Path(args.phase0_per_step))
    parity_details = []
    for row in parity_rows:
        action, parser_mode = lenient_parse_action(str(row.get("greedy_raw_output") or ""))
        key = action_key(action, 25)
        chosen = key == str(row.get("chosen_action_key"))
        parity_details.append({
            "target_id": row["target_id"],
            "parser_mode": parser_mode,
            "expected_action_key": row.get("new_greedy_action_key"),
            "lenient_action_key": key,
            "action_key_agree": key == row.get("new_greedy_action_key"),
            "chosen_verdict_agree": chosen == bool(row.get("new_greedy_is_chosen")),
        })
    parity = {
        "positions": len(parity_details),
        "action_key_agreement": sum(1 for row in parity_details if row["action_key_agree"]) / max(1, len(parity_details)),
        "chosen_verdict_agreement": sum(1 for row in parity_details if row["chosen_verdict_agree"]) / max(1, len(parity_details)),
        "disagreements": sum(1 for row in parity_details if not (row["action_key_agree"] and row["chosen_verdict_agree"])),
        "protocol": str(args.protocol),
        "cell_outputs": outputs,
    }
    write_json(Path(args.output_parity), parity)
    write_jsonl(Path(args.output_parity).with_suffix(".per_step.jsonl"), parity_details)
    print(json.dumps(parity, indent=2), flush=True)


def historical_context(summary_path: Path, baseline_rows_path: Path, t15_path: Path) -> dict[str, Any]:
    summary = read_json(summary_path) or {}
    baseline_rows = read_jsonl(baseline_rows_path)
    def rank_key(row: Mapping[str, Any]) -> tuple[Any, Any, int]:
        target_id = str(row["target_id"])
        episode_id, step_text = target_id.rsplit(":", 1)
        episode_key: Any = int(episode_id) if episode_id.isdigit() else episode_id
        return -float(row.get("greedy_decode_share") or 0.0), episode_key, int(step_text)
    strong_ids = {str(row["target_id"]) for row in sorted(baseline_rows, key=rank_key)[:962]}
    t15_rows = read_jsonl(t15_path)
    def parse_payload(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        samples = [sample for row in rows for sample in (row.get("samples") or [])[:32]]
        parsed = sum(
            sample.get("pred_action") is not None and sample.get("action_key") != "__unparsed__"
            for sample in samples
        )
        return {"steps": len(rows), "samples": len(samples), "parse_rate": parsed / max(1, len(samples))}
    strong_t15_rows = [row for row in t15_rows if str(row["target_id"]) in strong_ids]
    strong_summary = summary["summaries"]["sft"]
    t07_g32 = next(row for row in strong_summary["0.7"]["strong_reflex"] if row["G"] == 32)
    t15_g32 = next(row for row in strong_summary["1.5"]["strong_reflex"] if row["G"] == 32)
    return {
        "source_summary": str(summary_path),
        "source_t15_samples": str(t15_path),
        "strong_reflex_steps": 962,
        "g32_all_wrong_t0_7": t07_g32["all_wrong_fraction"],
        "g32_all_wrong_t1_5": t15_g32["all_wrong_fraction"],
        "t1_5_all_parse": parse_payload(t15_rows),
        "t1_5_strong_parse": parse_payload(strong_t15_rows),
    }


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.2f}%"


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def report(args: argparse.Namespace) -> None:
    manifest = read_json(Path(args.run_manifest)) or {}
    expected_position_rows = list(manifest.get("positive_positions") or [])
    expected_ids = {str(row["target_id"]) for row in expected_position_rows}
    expected_keys = {str(row["target_id"]): str(row["chosen_action_key"]) for row in expected_position_rows}
    cell_records = []
    cell_integrity = {}
    strict_rows_by_cell: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for path_text in args.cell_files:
        path = Path(path_text)
        rows = read_jsonl(path)
        row_ids = [str(row.get("target_id")) for row in rows]
        arm_values = {str(row.get("arm")) for row in rows}
        temperature_values = {float(row.get("temperature")) for row in rows}
        checks = {
            "rows_48": len(rows) == 48,
            "unique_ids_48": len(set(row_ids)) == len(row_ids) == 48,
            "exact_frozen_ids": set(row_ids) == expected_ids,
            "chosen_keys_match_manifest": all(expected_keys.get(str(row.get("target_id"))) == str(row.get("chosen_action_key")) for row in rows),
            "single_arm": len(arm_values) == 1,
            "single_temperature": len(temperature_values) == 1,
            "k16_every_position": all(int(row.get("k") or 0) == 16 for row in rows),
            "samples16_every_position": all(len(row.get("samples") or []) == 16 for row in rows),
            "top_p_frozen": all(float(row.get("top_p") or 0.0) == 0.95 for row in rows),
            "max_tokens_frozen": all(int(row.get("max_new_tokens") or 0) == 384 for row in rows),
            "batch16_frozen": all(int(row.get("sample_batch_size") or 0) == 16 for row in rows),
            "seed42_frozen": all(int(row.get("base_seed") or -1) == 42 for row in rows),
        }
        cell_integrity[str(path)] = checks
        if not rows or len(arm_values) != 1 or len(temperature_values) != 1:
            raise ValueError(f"cannot identify cell {path}: {checks}")
        summary = {
            "arm": next(iter(arm_values)),
            "temperature": next(iter(temperature_values)),
            **summarize_rows(rows),
            "source_cell": str(path),
            "integrity": checks,
        }
        cell_records.append(summary)
        strict_rows_by_cell[(summary["arm"], summary["temperature"])] = rows
    cells = {(str(summary["arm"]), float(summary["temperature"])): summary for summary in cell_records}
    expected = {(arm, temperature) for arm in ARMS for temperature in TEMPERATURES}
    if set(cells) != expected:
        raise ValueError(f"cell grid mismatch missing={sorted(expected-set(cells))} extra={sorted(set(cells)-expected)}")
    lenient_records = []
    lenient_integrity = {}
    for path_text in args.lenient_cell_files:
        path = Path(path_text)
        rows = read_jsonl(path)
        arm_values = {str(row.get("arm")) for row in rows}
        temperature_values = {float(row.get("temperature")) for row in rows}
        if not rows or len(arm_values) != 1 or len(temperature_values) != 1:
            raise ValueError(f"cannot identify lenient cell {path}")
        key = (next(iter(arm_values)), next(iter(temperature_values)))
        strict_rows = strict_rows_by_cell.get(key) or []
        strict_by_id = {str(row["target_id"]): row for row in strict_rows}
        repaired_by_id = {str(row["target_id"]): row for row in rows}
        raw_equal = True
        strict_flags_preserved = True
        for target_id in expected_ids:
            strict_row = strict_by_id.get(target_id) or {}
            repaired_row = repaired_by_id.get(target_id) or {}
            strict_samples = list(strict_row.get("samples") or [])
            repaired_samples = list(repaired_row.get("samples") or [])
            if len(strict_samples) != len(repaired_samples):
                raw_equal = False
                strict_flags_preserved = False
                continue
            for strict_sample, repaired_sample in zip(strict_samples, repaired_samples):
                raw_equal = raw_equal and strict_sample.get("raw_output") == repaired_sample.get("raw_output")
                strict_flags_preserved = strict_flags_preserved and bool(strict_sample.get("parse_ok")) == bool(repaired_sample.get("strict_parse_ok"))
        checks = {
            "rows_48": len(rows) == 48,
            "unique_exact_ids": len(repaired_by_id) == 48 and set(repaired_by_id) == expected_ids,
            "parser_version": all(row.get("parser") == "lenient-tool-call-v1" for row in rows),
            "samples16_every_position": all(len(row.get("samples") or []) == 16 for row in rows),
            "raw_outputs_identical_to_strict": raw_equal,
            "strict_parse_flags_preserved": strict_flags_preserved,
        }
        lenient_integrity[str(path)] = checks
        lenient_records.append({
            "arm": key[0],
            "temperature": key[1],
            "parser": "lenient-tool-call-v1",
            **summarize_rows(rows),
            "source_cell": str(path),
            "integrity": checks,
        })
    lenient_cells = {(str(summary["arm"]), float(summary["temperature"])): summary for summary in lenient_records}
    if set(lenient_cells) != expected:
        raise ValueError(f"lenient grid mismatch missing={sorted(expected-set(lenient_cells))} extra={sorted(set(lenient_cells)-expected)}")
    manifest_protocol = manifest.get("protocol") or {}
    lenient_protocol = read_json(Path(args.lenient_protocol)) or {}
    lenient_parity = read_json(Path(args.lenient_parity)) or {}
    manifest_protocol_sha = hashlib.sha256(
        json.dumps(manifest_protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    positive_payload_sha = hashlib.sha256(
        json.dumps(expected_position_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    phase0_identity = read_json(Path(args.phase0_identity)) or {}
    worker_commands = list(manifest.get("worker_commands") or [])
    common_fragments = [
        "hetero_inject_e02r.py run-ladder-shard",
        f"--pairs {manifest.get('pairs_path')}",
        f"--prior-eval {manifest.get('prior_eval_path')}",
        f"--model-path {manifest.get('model_path')}",
        f"--adapter-dir {manifest.get('adapter_dir')}",
        "--k 16",
        "--sample-batch-size 16",
        "--top-p 0.95",
        "--max-new-tokens 384",
        "--seed 42",
        "--coord-bucket 25",
        "--match-threshold 0.5",
    ]
    worker_commands_match = len(worker_commands) == 8 and all(
        all(fragment in command for fragment in common_fragments)
        and f"--shard-index {index}" in command
        and f"--start {index * 6}" in command
        and f"--end {(index + 1) * 6}" in command
        for index, command in enumerate(worker_commands)
    )
    manifest_checks = {
        "manifest_version": manifest.get("version") == "E0.2-R-v1.1",
        "positive_count_48": manifest.get("positive_count") == 48 and len(expected_ids) == 48,
        "positive_manifest_hash_match": manifest.get("positive_positions_sha256") == positive_payload_sha,
        "pairs_hash_match": manifest.get("pairs_sha256") == file_sha256(Path(manifest.get("pairs_path", ""))),
        "prior_eval_hash_match": manifest.get("prior_eval_sha256") == file_sha256(Path(manifest.get("prior_eval_path", ""))),
        "model_manifest_bound": manifest.get("complete_model_adapter_manifest_hash") == phase0_identity.get("complete_manifest_hash"),
        "checkpoint_hash_bound": manifest.get("checkpoint_canonical_hash") == phase0_identity.get("checkpoint_canonical_hash"),
        "worker_count_8": manifest.get("worker_count") == 8,
        "worker_commands_match": worker_commands_match,
        "protocol_hash_match": manifest.get("protocol_sha256") == manifest_protocol_sha,
        "temperatures_frozen": manifest_protocol.get("temperatures") == list(TEMPERATURES),
        "arms_frozen": manifest_protocol.get("arms") == list(ARMS),
        "sampling_protocol_frozen": all([
            manifest_protocol.get("k") == 16,
            manifest_protocol.get("sample_batch_size") == 16,
            manifest_protocol.get("top_p") == 0.95,
            manifest_protocol.get("max_new_tokens") == 384,
            manifest_protocol.get("seed") == 42,
            manifest_protocol.get("coord_bucket") == 25,
            manifest_protocol.get("match_threshold") == 0.5,
            manifest_protocol.get("constrained_decoding") is False,
        ]),
    }
    parser_repair_checks = {
        "protocol_version": lenient_protocol.get("version") == "lenient-tool-call-v1",
        "predeclared_before_rescoring": lenient_protocol.get("predeclared_before_rescoring") is True,
        "no_generation_change": lenient_protocol.get("generation_changed") is False,
        "no_constrained_decoding": lenient_protocol.get("constrained_decoding") is False,
        "parity_positions_212": lenient_parity.get("positions") == 212,
        "parity_action_key_100pct": lenient_parity.get("action_key_agreement") == 1.0,
        "parity_chosen_verdict_100pct": lenient_parity.get("chosen_verdict_agreement") == 1.0,
        "parity_zero_disagreements": lenient_parity.get("disagreements") == 0,
    }
    integrity = all(manifest_checks.values()) and all(
        all(checks.values()) for checks in cell_integrity.values()
    ) and all(all(checks.values()) for checks in lenient_integrity.values()) and all(parser_repair_checks.values())

    taxonomy = read_json(Path(args.taxonomy)) or {}
    taxonomy_protocol = read_json(Path(args.taxonomy_protocol)) or {}
    taxonomy_counts = taxonomy.get("counts") or {}
    allowed_labels = set((taxonomy_protocol.get("exclusive_primary_labels") or {}).keys())
    tie_break = list((taxonomy_protocol.get("dominant_class") or {}).get("tie_break_order") or [])
    max_count = max((int(value) for value in taxonomy_counts.values()), default=-1)
    tied = {label for label, value in taxonomy_counts.items() if int(value) == max_count}
    computed_dominant = next((label for label in tie_break if label in tied), None)
    expected_route = (taxonomy_protocol.get("repair_routes") or {}).get(computed_dominant)
    taxonomy_complete = bool(
        taxonomy.get("labeled_count") == 20
        and sum(int(value) for value in taxonomy_counts.values()) == 20
        and set(taxonomy_counts) <= allowed_labels
        and taxonomy.get("dominant_class") == computed_dominant
        and taxonomy.get("selected_repair_route") == expected_route
        and taxonomy.get("protocol_version") == taxonomy_protocol.get("version")
        and taxonomy.get("source_cell") == {"arm": "REG-INJECT", "temperature": 1.5}
    )
    if not taxonomy_complete:
        raise ValueError("taxonomy must contain 20 exclusive labels, dominant_class, and selected_repair_route before scoring")
    use_lenient_for_gate = computed_dominant == "format_drift" and all(parser_repair_checks.values())
    gate_cells = lenient_cells if use_lenient_for_gate else cells
    gate_parser = "lenient-tool-call-v1" if use_lenient_for_gate else "strict"
    reg_cells = {temperature: gate_cells[("reg_inject", temperature)] for temperature in TEMPERATURES}
    sentinel = reg_cells[0.0]["chosen_sampled_any"] == 1.0
    healthy = [temperature for temperature in TEMPERATURES if reg_cells[temperature]["parse_rate"] >= args.healthy_parse_threshold]
    t_star = max(healthy) if healthy else None
    qualifying_temperatures = [
        temperature for temperature in healthy
        if reg_cells[temperature]["chosen_sampled_any"] >= args.chosen_any_threshold
    ]
    g0_r_pass_temperature = max(qualifying_temperatures) if qualifying_temperatures else None
    if not integrity:
        gate = "G0-R PIPELINE ISSUE"
        reason = "One or more ladder, manifest, taxonomy-selected parser, or parity integrity checks failed."
    elif not sentinel:
        gate = "G0-R FAIL - SENTINEL"
        reason = "REG-INJECT T=0 chosen-sampled-any is below 100%; harness audit is reopened."
    elif g0_r_pass_temperature is not None:
        gate = "G0-R PASS"
        reason = f"Using {gate_parser}, T={g0_r_pass_temperature:g} has parse rate at least {100*args.healthy_parse_threshold:.0f}%, chosen-sampled-any at least {100*args.chosen_any_threshold:.0f}%, and the T=0 sentinel holds."
    elif not healthy:
        gate = "G0-R FAIL - NO HEALTHY TEMPERATURE"
        reason = f"No REG-INJECT ladder temperature reaches the {100*args.healthy_parse_threshold:.0f}% parse threshold."
    else:
        gate = "G0-R FAIL AT HEALTHY T*"
        reason = f"Using {gate_parser}, parsing is healthy at {healthy}, but no healthy cell reaches {100*args.chosen_any_threshold:.0f}% chosen-sampled-any."
    historical = historical_context(
        Path(args.historical_summary),
        Path(args.historical_baseline_rows),
        Path(args.historical_t15_samples),
    )
    historical_parse_support_threshold = 0.90
    base_t15_healthy = cells[("base_sft", 1.5)]["parse_rate"] >= historical_parse_support_threshold
    history_t15_healthy = historical["t1_5_strong_parse"]["parse_rate"] >= historical_parse_support_threshold
    if base_t15_healthy and history_t15_healthy:
        cross_chain = "Base SFT and the historical G=32 run are both healthy at T=1.5; degeneration is attributable to REG-INJECT, and the upstream recovery claim remains parse-context-supported."
        root_trace_action = "UPSTREAM_T1P5_RECOVERY_PARSE_SUPPORTED"
    elif not base_t15_healthy:
        cross_chain = "Base SFT also degenerates at T=1.5; flag the historical temperature-recovery claim for re-verification before reuse."
        root_trace_action = "REVERIFY_UPSTREAM_T1P5_RECOVERY"
    else:
        cross_chain = "Historical T=1.5 parse context is not healthy; flag the temperature-recovery claim for re-verification."
        root_trace_action = "REVERIFY_UPSTREAM_T1P5_RECOVERY"

    summary = {
        "gate_g0_r": gate,
        "reason": reason,
        "integrity_pass": integrity,
        "manifest_checks": manifest_checks,
        "cell_integrity": cell_integrity,
        "lenient_cell_integrity": lenient_integrity,
        "parser_repair_checks": parser_repair_checks,
        "run_manifest": args.run_manifest,
        "taxonomy_complete": taxonomy_complete,
        "gate_parser": gate_parser,
        "sentinel_pass": sentinel,
        "healthy_parse_threshold": args.healthy_parse_threshold,
        "chosen_any_threshold": args.chosen_any_threshold,
        "healthy_temperatures_reg": healthy,
        "qualifying_temperatures_reg": qualifying_temperatures,
        "g0_r_pass_temperature": g0_r_pass_temperature,
        "t_star": t_star,
        "strict_cells": {f"{arm}|{temperature:g}": cells[(arm, temperature)] for arm in ARMS for temperature in TEMPERATURES},
        "repaired_cells": {f"{arm}|{temperature:g}": lenient_cells[(arm, temperature)] for arm in ARMS for temperature in TEMPERATURES},
        "taxonomy": taxonomy,
        "historical_context": historical,
        "historical_parse_support_threshold": historical_parse_support_threshold,
        "cross_chain_verdict": cross_chain,
        "root_trace_action": root_trace_action,
        "phase_1_started": False,
        "phase_2_started": False,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "e02r_summary.json", summary)

    ladder_rows = []
    for parser_name, grid in (("strict", cells), ("lenient-tool-call-v1", lenient_cells)):
        for arm in ARMS:
            for temperature in TEMPERATURES:
                cell = grid[(arm, temperature)]
                ladder_rows.append([
                    parser_name,
                    arm,
                    f"{temperature:g}",
                    pct(cell["parse_rate"]),
                    pct(cell["chosen_frequency"]),
                    pct(cell["chosen_sampled_any"]),
                    pct(cell["max_length_rate"]),
                ])
    type_rows = []
    for parser_name, grid in (("strict", cells), ("lenient-tool-call-v1", lenient_cells)):
        for arm in ARMS:
            for temperature in TEMPERATURES:
                for action_type, payload in sorted(grid[(arm, temperature)]["by_action_type"].items()):
                    type_rows.append([
                        parser_name,
                        arm,
                        f"{temperature:g}",
                        action_type,
                        payload["positions"],
                        pct(payload["parse_rate"]),
                        pct(payload["chosen_frequency"]),
                        pct(payload["chosen_sampled_any"]),
                    ])
    if gate == "G0-R PASS":
        adjudication_note = "Harness chain is fully validated. The v1.1 T=1.3 failure is reclassified as expected temperature flattening, not harness or model-capacity evidence."
    else:
        adjudication_note = "A failure with healthy parsing is admissible as model-behavior evidence, but it does not by itself establish a capacity lock."
    lines = [
        "# REG-INJECT E0.2-R Temperature Ladder",
        "",
        "Same cached 48 REG-INJECT greedy-equals-chosen positions, K=16 per arm/temperature cell. Raw generations are reported under the strict parser and under the separately predeclared lenient-parser repair arm. No constrained decoding or generation change was used.",
        "",
        "## Main Ladder",
        "",
        table(["parser", "arm", "T", "parse rate", "chosen frequency", "chosen sampled-any", "max-length rate"], ladder_rows),
        "",
        "## Per Action Type",
        "",
        table(["parser", "arm", "T", "chosen type", "positions", "parse rate", "chosen frequency", "chosen sampled-any"], type_rows),
        "",
        "## T=1.5 Failure Taxonomy (20 predeclared samples)",
        "",
        table(["label", "count"], [[label, count] for label, count in sorted((taxonomy.get("counts") or {}).items())]),
        "",
        f"Dominant class: **{taxonomy.get('dominant_class', 'PENDING')}**. Selected repair route: {taxonomy.get('selected_repair_route', 'PENDING')}.",
        "",
        "## Lenient Parser Repair Parity",
        "",
        table(["metric", "value", "requirement"], [
            ["greedy positions", lenient_parity.get("positions"), "212"],
            ["action-key agreement", pct(lenient_parity.get("action_key_agreement")), "100%"],
            ["chosen-verdict agreement", pct(lenient_parity.get("chosen_verdict_agreement")), "100%"],
            ["disagreements", lenient_parity.get("disagreements"), "0"],
        ]),
        "",
        f"Parser selected for G0-R: **{gate_parser}**.",
        "",
        "## Historical G=32 Parse Context",
        "",
        table(["metric", "value"], [
            ["strong-reflex G32 all-wrong T=0.7", pct(historical["g32_all_wrong_t0_7"])],
            ["strong-reflex G32 all-wrong T=1.5", pct(historical["g32_all_wrong_t1_5"])],
            ["historical T=1.5 overall parse", pct(historical["t1_5_all_parse"]["parse_rate"])],
            ["historical T=1.5 strong-reflex parse", pct(historical["t1_5_strong_parse"]["parse_rate"])],
        ]),
        "",
        cross_chain,
        "",
        "## Gate G0-R",
        "",
        gate,
        "",
        reason,
        "",
        f"Highest parse-healthy temperature T*: **{t_star if t_star is not None else 'none'}** under **{gate_parser}**. Highest G0-R qualifying temperature: **{g0_r_pass_temperature if g0_r_pass_temperature is not None else 'none'}**. T=0 sentinel: **{'PASS' if sentinel else 'FAIL'}**.",
        "",
        adjudication_note,
        "",
        "STOP for review. Phase 1 and Phase 2 were not started.",
        "",
    ]
    (out_dir / "e02r.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"gate_g0_r": gate, "t_star": t_star, "report": str(out_dir / "e02r.md")}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("freeze-manifest")
    manifest.add_argument("--pairs", default="outputs/hetero_inject/data/hetero_inject_train.jsonl")
    manifest.add_argument("--prior-eval", default="outputs/hetero_inject/eval/reg_inject.jsonl")
    manifest.add_argument("--model-path", default="checkpoints/gui360-fullparam-sft-step250")
    manifest.add_argument("--adapter-dir", default="outputs/hetero_inject/checkpoints/reg_inject/final/cooperative")
    manifest.add_argument("--phase0-identity", default="outputs/hetero_inject_followup/phase0/hash_sampling.json")
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(func=freeze_run_manifest)

    cell = sub.add_parser("run-cell")
    cell.add_argument("--arm", required=True)
    cell.add_argument("--temperature", type=float, choices=TEMPERATURES, required=True)
    cell.add_argument("--pairs", default="outputs/hetero_inject/data/hetero_inject_train.jsonl")
    cell.add_argument("--prior-eval", default="outputs/hetero_inject/eval/reg_inject.jsonl")
    cell.add_argument("--output", required=True)
    cell.add_argument("--start", type=int, default=0)
    cell.add_argument("--end", type=int, default=None)
    cell.add_argument("--overwrite", action="store_true")
    cell.add_argument("--k", type=int, default=16)
    cell.add_argument("--diagnostic-k32", action="store_true")
    cell.add_argument("--sample-batch-size", type=int, default=4)
    cell.add_argument("--top-p", type=float, default=0.95)
    cell.add_argument("--max-new-tokens", type=int, default=384)
    cell.add_argument("--seed", type=int, default=42)
    cell.add_argument("--coord-bucket", type=int, default=25)
    cell.add_argument("--match-threshold", type=float, default=0.5)
    cell.add_argument("--model-path", default="checkpoints/gui360-fullparam-sft-step250")
    cell.add_argument("--adapter-dir", default="outputs/hetero_inject/checkpoints/reg_inject/final/cooperative")
    cell.add_argument("--device", default="cuda:0")
    cell.add_argument("--image-max-pixels", type=int, default=602112)
    cell.add_argument("--lora-r", type=int, default=64)
    cell.add_argument("--lora-alpha", type=int, default=128)
    cell.add_argument("--num-comm-rounds", type=int, default=2)
    cell.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    cell.set_defaults(func=run_cell)

    mini = sub.add_parser("run-mini-ladder")
    mini.add_argument("--arm", required=True)
    mini.add_argument("--pairs", default="outputs/hetero_inject/data/hetero_inject_train.jsonl")
    mini.add_argument("--prior-eval", default="outputs/hetero_inject/eval/reg_inject.jsonl")
    mini.add_argument("--output-dir", required=True)
    mini.add_argument("--overwrite", action="store_true")
    mini.add_argument("--k", type=int, default=16)
    mini.add_argument("--sample-batch-size", type=int, default=16)
    mini.add_argument("--top-p", type=float, default=0.95)
    mini.add_argument("--max-new-tokens", type=int, default=384)
    mini.add_argument("--seed", type=int, default=42)
    mini.add_argument("--coord-bucket", type=int, default=25)
    mini.add_argument("--match-threshold", type=float, default=0.5)
    mini.add_argument("--model-path", default="checkpoints/gui360-fullparam-sft-step250")
    mini.add_argument("--adapter-dir", required=True)
    mini.add_argument("--device", default="cuda:0")
    mini.add_argument("--image-max-pixels", type=int, default=602112)
    mini.add_argument("--lora-r", type=int, default=64)
    mini.add_argument("--lora-alpha", type=int, default=128)
    mini.add_argument("--num-comm-rounds", type=int, default=2)
    mini.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    mini.set_defaults(func=run_mini_ladder)

    ladder_shard = sub.add_parser("run-ladder-shard")
    ladder_shard.add_argument("--pairs", default="outputs/hetero_inject/data/hetero_inject_train.jsonl")
    ladder_shard.add_argument("--prior-eval", default="outputs/hetero_inject/eval/reg_inject.jsonl")
    ladder_shard.add_argument("--output-dir", required=True)
    ladder_shard.add_argument("--shard-index", type=int, required=True)
    ladder_shard.add_argument("--start", type=int, required=True)
    ladder_shard.add_argument("--end", type=int, required=True)
    ladder_shard.add_argument("--overwrite", action="store_true")
    ladder_shard.add_argument("--k", type=int, default=16)
    ladder_shard.add_argument("--sample-batch-size", type=int, default=16)
    ladder_shard.add_argument("--top-p", type=float, default=0.95)
    ladder_shard.add_argument("--max-new-tokens", type=int, default=384)
    ladder_shard.add_argument("--seed", type=int, default=42)
    ladder_shard.add_argument("--coord-bucket", type=int, default=25)
    ladder_shard.add_argument("--match-threshold", type=float, default=0.5)
    ladder_shard.add_argument("--model-path", default="checkpoints/gui360-fullparam-sft-step250")
    ladder_shard.add_argument("--adapter-dir", default="outputs/hetero_inject/checkpoints/reg_inject/final/cooperative")
    ladder_shard.add_argument("--device", default="cuda:0")
    ladder_shard.add_argument("--image-max-pixels", type=int, default=602112)
    ladder_shard.add_argument("--lora-r", type=int, default=64)
    ladder_shard.add_argument("--lora-alpha", type=int, default=128)
    ladder_shard.add_argument("--num-comm-rounds", type=int, default=2)
    ladder_shard.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    ladder_shard.set_defaults(func=run_ladder_shard)

    merge = sub.add_parser("merge-cell")
    merge.add_argument("--shards", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--pairs", default="outputs/hetero_inject/data/hetero_inject_train.jsonl")
    merge.add_argument("--prior-eval", default="outputs/hetero_inject/eval/reg_inject.jsonl")
    merge.set_defaults(func=merge_cell)

    taxonomy_extract = sub.add_parser("extract-taxonomy")
    taxonomy_extract.add_argument("--cell", required=True)
    taxonomy_extract.add_argument("--output", required=True)
    taxonomy_extract.add_argument("--count", type=int, default=20)
    taxonomy_extract.set_defaults(func=extract_taxonomy_samples)

    taxonomy_score = sub.add_parser("score-taxonomy")
    taxonomy_score.add_argument("--samples", required=True)
    taxonomy_score.add_argument("--labels", required=True)
    taxonomy_score.add_argument("--protocol", required=True)
    taxonomy_score.add_argument("--output-labeled", required=True)
    taxonomy_score.add_argument("--output-summary", required=True)
    taxonomy_score.set_defaults(func=score_taxonomy)

    lenient_rescore = sub.add_parser("rescore-lenient")
    lenient_rescore.add_argument("--cell-files", nargs="+", required=True)
    lenient_rescore.add_argument("--protocol", required=True)
    lenient_rescore.add_argument("--phase0-per-step", required=True)
    lenient_rescore.add_argument("--output-dir", required=True)
    lenient_rescore.add_argument("--output-parity", required=True)
    lenient_rescore.set_defaults(func=rescore_lenient)

    report_parser = sub.add_parser("report")
    report_parser.add_argument("--cell-files", nargs="+", required=True)
    report_parser.add_argument("--lenient-cell-files", nargs="+", required=True)
    report_parser.add_argument("--lenient-parity", required=True)
    report_parser.add_argument("--lenient-protocol", required=True)
    report_parser.add_argument("--run-manifest", required=True)
    report_parser.add_argument("--phase0-identity", default="outputs/hetero_inject_followup/phase0/hash_sampling.json")
    report_parser.add_argument("--taxonomy", required=True)
    report_parser.add_argument("--taxonomy-protocol", default="outputs/hetero_inject_followup/e02r/taxonomy_protocol.json")
    report_parser.add_argument("--output-dir", default="outputs/hetero_inject_followup/e02r")
    report_parser.add_argument("--healthy-parse-threshold", type=float, default=0.95)
    report_parser.add_argument("--chosen-any-threshold", type=float, default=0.90)
    report_parser.add_argument("--historical-summary", default="outputs/temp_restores_signal/summary.json")
    report_parser.add_argument("--historical-baseline-rows", default="outputs/rl_feasibility/per_step.jsonl")
    report_parser.add_argument("--historical-t15-samples", default="outputs/temp_restores_signal/sft_T1p5_critical.jsonl")
    report_parser.set_defaults(func=report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
