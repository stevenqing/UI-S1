#!/usr/bin/env python3
"""Phase-0 harness audit for heterogeneous REG-INJECT.

This script does not train. It verifies cooperative-checkpoint identity,
re-runs greedy parser parity on all injection positions, and runs the K=16
positive-control sampler on the cached REG-INJECT greedy-equals-chosen subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.hetero_inject_eval import parse_prediction, score_text  # noqa: E402
from scripts.minimal_validation import build_messages, load_model_for_eval  # noqa: E402
from scripts.rl_feasibility_sampling import action_key, sanitize_jsonable  # noqa: E402
from v13_gui_360.eval_gui360_template import parse_tool_call  # noqa: E402
from v13_gui_360.reward import compute_step_reward  # noqa: E402


ADAPTER_FILES = ("lora_weights.pt", "route_weights.pt", "comm_weights.pt")
BASE_MANIFEST_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "video_preprocessor_config.json",
    "vocab.json",
}


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_jsonable(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(sanitize_jsonable(dict(row)), ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def update_tensor_hash(digest: Any, name: str, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.view(torch.uint8).numpy().tobytes())
    digest.update(b"\0")


def canonical_state_hash(items: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(items):
        update_tensor_hash(digest, name, items[name])
    return digest.hexdigest()


def load_saved_adapter_state(adapter_dir: Path) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for filename in ADAPTER_FILES:
        path = adapter_dir / filename
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        overlap = set(state) & set(loaded)
        if overlap:
            raise ValueError(f"duplicate adapter keys across files: {sorted(overlap)[:3]}")
        state.update(loaded)
    return state


def base_manifest_hash(model_path: Path) -> tuple[str, list[str]]:
    candidates = sorted(
        path for path in model_path.iterdir()
        if path.is_file() and (
            path.name.endswith(".safetensors")
            or path.name in BASE_MANIFEST_FILES
        )
    )
    digest = hashlib.sha256()
    for path in candidates:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest(), [path.name for path in candidates]


def hash_harness(args: argparse.Namespace) -> None:
    adapter_dir = Path(args.adapter_dir)
    saved_state = load_saved_adapter_state(adapter_dir)
    checkpoint_hash = canonical_state_hash(saved_state)
    model, _processor, _device = load_model_for_eval(args)
    named = dict(model.named_parameters())
    expected_adapter_names = {
        name for name in named
        if "lora_" in name or "route_weights" in name or "comm_" in name
    }
    missing = sorted(set(saved_state) - set(named))
    omitted_expected = sorted(expected_adapter_names - set(saved_state))
    unexpected_saved = sorted(set(saved_state) - expected_adapter_names)
    unexpected_shape = sorted(
        name for name, value in saved_state.items()
        if name in named and tuple(value.shape) != tuple(named[name].shape)
    )
    actual_state = {name: named[name] for name in saved_state if name in named}
    loaded_hash = canonical_state_hash(actual_state)
    base_hash, base_files = base_manifest_hash(Path(args.model_path))
    cooperative_config_hash = file_sha256(adapter_dir / "cooperative_config.json")
    complete_digest = hashlib.sha256()
    for component in (base_hash, checkpoint_hash, cooperative_config_hash):
        complete_digest.update(component.encode("ascii"))
        complete_digest.update(b"\0")
    complete_manifest_hash = complete_digest.hexdigest()
    result = {
        "harness": args.harness,
        "adapter_dir": str(adapter_dir),
        "checkpoint_canonical_hash": checkpoint_hash,
        "loaded_canonical_hash": loaded_hash,
        "hash_match": (
            checkpoint_hash == loaded_hash
            and not missing
            and not unexpected_shape
            and not omitted_expected
            and not unexpected_saved
        ),
        "tensor_count_checkpoint": len(saved_state),
        "tensor_count_loaded": len(actual_state),
        "missing_tensor_names": missing,
        "omitted_expected_tensor_names": omitted_expected,
        "unexpected_saved_tensor_names": unexpected_saved,
        "shape_mismatches": unexpected_shape,
        "checkpoint_file_sha256": {
            filename: file_sha256(adapter_dir / filename)
            for filename in (*ADAPTER_FILES, "cooperative_config.json")
        },
        "base_model_path": args.model_path,
        "base_manifest_hash": base_hash,
        "base_manifest_files": base_files,
        "cooperative_config_hash": cooperative_config_hash,
        "complete_manifest_hash": complete_manifest_hash,
    }
    write_json(Path(args.output), result)
    print(json.dumps(result, indent=2), flush=True)


def run_manifest(args: argparse.Namespace) -> None:
    pairs_path = Path(args.pairs)
    prior_path = Path(args.prior_eval)
    worker_commands = []
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            command = (process_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8").strip()
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError):
            continue
        if (
            "hetero_inject_phase0.py audit-shard" in command
            and str(pairs_path) in command
        ):
            worker_commands.append(command)
    protocol = {
        "k": args.k,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "sample_batch_size": args.sample_batch_size,
        "seed": args.seed,
        "coord_bucket": args.coord_bucket,
        "match_threshold": args.match_threshold,
    }
    payload = {
        "created_unix": time.time(),
        "pairs_path": str(pairs_path),
        "pairs_sha256": file_sha256(pairs_path),
        "prior_eval_path": str(prior_path),
        "prior_eval_sha256": file_sha256(prior_path),
        "model_path": args.model_path,
        "adapter_dir": args.adapter_dir,
        "protocol": protocol,
        "protocol_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "worker_count": len(worker_commands),
        "worker_commands": sorted(worker_commands),
    }
    write_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2), flush=True)


def prepare_inputs(processor: Any, device: torch.device, messages: list[dict[str, Any]], image_path: str) -> tuple[dict[str, torch.Tensor], int]:
    image = Image.open(image_path).convert("RGB")
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    return inputs, int(inputs["input_ids"].shape[1])


def generation_stop(processor: Any) -> dict[str, Any]:
    tokenizer = processor.tokenizer
    stop_ids = [tokenizer.eos_token_id]
    tool_end = tokenizer.encode("</tool_call>", add_special_tokens=False)
    if len(tool_end) == 1:
        stop_ids.append(tool_end[0])
    return {
        "eos_token_id": stop_ids,
        "stop_strings": ["</tool_call>"],
        "tokenizer": tokenizer,
    }


def generate_batch(
    model: Any,
    processor: Any,
    device: torch.device,
    messages: list[dict[str, Any]],
    image_path: str,
    *,
    n: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[str]:
    inputs, prompt_len = prepare_inputs(processor, device, messages, image_path)
    generated_inputs: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and n > 1:
            repeats = [n] + [1] * (value.dim() - 1)
            generated_inputs[key] = value.repeat(*repeats)
        else:
            generated_inputs[key] = value
    kwargs: dict[str, Any] = {
        **generated_inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        **generation_stop(processor),
    }
    if do_sample:
        kwargs.update({"temperature": temperature, "top_p": top_p})
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output_ids = model.generate(**kwargs)
    return [
        processor.tokenizer.decode(output_ids[index, prompt_len:], skip_special_tokens=True)
        for index in range(int(output_ids.shape[0]))
    ]


def sample_k(
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
    while len(outputs) < k:
        size = min(batch_size, k - len(outputs))
        outputs.extend(generate_batch(
            model,
            processor,
            device,
            messages,
            image_path,
            n=size,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        ))
    return outputs


def native_sampling_score(
    text: str,
    gt_action: Mapping[str, Any],
    image_w: int,
    image_h: int,
    match_threshold: float,
    coord_bucket: int,
) -> dict[str, Any]:
    try:
        parsed = parse_tool_call(text)
    except Exception:
        parsed = None
    fake_text = f"<action>{json.dumps(parsed, ensure_ascii=False)}</action>" if parsed else text
    reward, info = compute_step_reward(fake_text, dict(gt_action), image_w=image_w, image_h=image_h)
    pred_action = info.get("pred_action")
    return {
        "parse_ok": pred_action is not None,
        "action_key": action_key(pred_action, coord_bucket),
        "success": bool(reward >= match_threshold),
        "reward": float(reward),
        "pred_action": pred_action,
    }


def prior_step_map(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for episode in read_jsonl(path):
        for step in episode.get("steps", []) or []:
            if step.get("is_blind_injected"):
                result[str(step["target_id"])] = step
    return result


def audit_shard(args: argparse.Namespace) -> None:
    all_pairs = read_jsonl(Path(args.pairs))
    pairs = all_pairs[args.start:args.end]
    prior = prior_step_map(Path(args.prior_eval))
    model, processor, device = load_model_for_eval(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows = []
    for pair in tqdm(pairs, desc=f"e0-{args.start}-{args.end}"):
        tid = str(pair["target_id"])
        target_seed = args.seed + int(hashlib.sha256(tid.encode("utf-8")).hexdigest()[:8], 16)
        torch.manual_seed(target_seed)
        torch.cuda.manual_seed_all(target_seed)
        messages, image_w, image_h = build_messages(
            pair["goal"],
            pair.get("history") or [],
            pair["image"],
            args.image_max_pixels,
        )
        greedy_text = generate_batch(
            model,
            processor,
            device,
            messages,
            pair["image"],
            n=1,
            do_sample=False,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )[0]
        greedy_score = score_text(
            greedy_text,
            pair["gt_action"],
            image_w,
            image_h,
            args.match_threshold,
            args.coord_bucket,
        )
        sampling_score = native_sampling_score(
            greedy_text,
            pair["gt_action"],
            image_w,
            image_h,
            args.match_threshold,
            args.coord_bucket,
        )
        chosen_key = str(pair["chosen_action_key"])
        greedy_chosen = greedy_score["action_key"] == chosen_key
        sampling_chosen = sampling_score["action_key"] == chosen_key
        parity = {
            "parse_agree": bool(greedy_score["parse_ok"]) == bool(sampling_score["parse_ok"]),
            "action_key_agree": greedy_score["action_key"] == sampling_score["action_key"],
            "matcher_agree": bool(greedy_score["success"]) == bool(sampling_score["success"]),
            "chosen_verdict_agree": greedy_chosen == sampling_chosen,
        }
        parity["all_agree"] = all(parity.values())

        prior_row = prior.get(tid) or {}
        positive_control = bool(prior_row.get("greedy_is_chosen"))
        sample_payload: dict[str, Any] = {
            "k": 0,
            "parse_count": 0,
            "correct_count": 0,
            "chosen_count": 0,
            "chosen_frequency": None,
            "chosen_sampled_any": None,
            "samples": [],
        }
        if positive_control:
            sample_texts = sample_k(
                model,
                processor,
                device,
                messages,
                pair["image"],
                k=args.k,
                batch_size=args.sample_batch_size,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )
            scored = [
                native_sampling_score(
                    text,
                    pair["gt_action"],
                    image_w,
                    image_h,
                    args.match_threshold,
                    args.coord_bucket,
                ) | {"raw_output": text[: args.store_chars]}
                for text in sample_texts
            ]
            chosen_count = sum(1 for item in scored if item["action_key"] == chosen_key)
            sample_payload = {
                "k": len(scored),
                "parse_count": sum(1 for item in scored if item["parse_ok"]),
                "correct_count": sum(1 for item in scored if item["success"]),
                "chosen_count": chosen_count,
                "chosen_frequency": chosen_count / max(1, len(scored)),
                "chosen_sampled_any": chosen_count > 0,
                "samples": scored,
            }

        rows.append({
            "target_id": tid,
            "episode_id": str(pair["episode_id"]),
            "step_idx": int(pair["step_idx"]),
            "chosen_action_key": chosen_key,
            "prior_greedy_is_chosen": positive_control,
            "prior_greedy_action_key": prior_row.get("action_key"),
            "new_greedy_action_key": greedy_score["action_key"],
            "new_greedy_is_chosen": greedy_chosen,
            "greedy_parse_ok": greedy_score["parse_ok"],
            "greedy_success": greedy_score["success"],
            "greedy_raw_output": greedy_text[: args.store_chars],
            "sampling_parser_on_greedy": sampling_score,
            "parser_parity": parity,
            "positive_control": sample_payload,
        })
        torch.cuda.empty_cache()

    write_jsonl(Path(args.output), rows)
    print(json.dumps({
        "output": args.output,
        "rows": len(rows),
        "positive_control_rows": sum(1 for row in rows if row["prior_greedy_is_chosen"]),
        "parity_disagreements": sum(1 for row in rows if not row["parser_parity"]["all_agree"]),
    }, indent=2), flush=True)


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.2f}%"


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def report(args: argparse.Namespace) -> None:
    hash_greedy = read_json(Path(args.hash_greedy)) or {}
    hash_sampling = read_json(Path(args.hash_sampling)) or {}
    manifest = read_json(Path(args.run_manifest)) or {}
    entropy_rows = read_jsonl(Path(args.entropy_metrics))
    entropy_evidence = entropy_rows[-1] if entropy_rows else {}
    entropy_value = entropy_evidence.get("entropy")
    entropy_weight = entropy_evidence.get("entropy_bonus_weight")
    entropy_contribution = entropy_evidence.get("entropy_bonus_contribution")
    entropy_logger_repaired = bool(
        isinstance(entropy_value, (int, float))
        and math.isfinite(float(entropy_value))
        and float(entropy_value) > 0.0
        and float(entropy_weight or 0.0) == 0.0
        and float(entropy_contribution or 0.0) == 0.0
    )
    manifest_pairs_path = Path(manifest.get("pairs_path") or args.pairs)
    manifest_prior_path = Path(manifest.get("prior_eval_path") or args.prior_eval)
    expected_pairs = read_jsonl(manifest_pairs_path)
    expected_ids = [str(row["target_id"]) for row in expected_pairs]
    raw_rows = []
    for path_text in args.audit_shards:
        raw_rows.extend(read_jsonl(Path(path_text)))
    raw_ids = [str(row["target_id"]) for row in raw_rows]
    by_target = {str(row["target_id"]): row for row in raw_rows}
    rows = list(by_target.values())
    positive = [row for row in rows if row.get("prior_greedy_is_chosen")]
    historical_steps = prior_step_map(manifest_prior_path)
    historical_positive = [step for step in historical_steps.values() if step.get("greedy_is_chosen")]
    expected_positive_ids = {
        str(target_id) for target_id, step in historical_steps.items()
        if step.get("greedy_is_chosen")
    }
    actual_positive_ids = {str(row["target_id"]) for row in positive}
    historical_sample_count = sum(int(step.get("sample_count") or 0) for step in historical_positive)
    historical_parse_count = sum(int(step.get("sample_parse_count") or 0) for step in historical_positive)
    historical_chosen_count = sum(int(step.get("sample_chosen_count") or 0) for step in historical_positive)
    historical_chosen_any = sum(int(bool(step.get("sample_chosen_any"))) for step in historical_positive) / max(1, len(historical_positive))

    checkpoint_hash = hash_greedy.get("checkpoint_canonical_hash")
    e01 = bool(
        hash_greedy.get("hash_match")
        and hash_sampling.get("hash_match")
        and checkpoint_hash
        and hash_greedy.get("harness") == "greedy"
        and hash_sampling.get("harness") == "sampling"
        and hash_greedy.get("adapter_dir") == hash_sampling.get("adapter_dir") == manifest.get("adapter_dir")
        and hash_greedy.get("base_model_path") == hash_sampling.get("base_model_path") == manifest.get("model_path")
        and hash_greedy.get("loaded_canonical_hash") == checkpoint_hash
        and hash_sampling.get("loaded_canonical_hash") == checkpoint_hash
        and hash_greedy.get("base_manifest_hash") == hash_sampling.get("base_manifest_hash")
        and hash_greedy.get("cooperative_config_hash") == hash_sampling.get("cooperative_config_hash")
        and hash_greedy.get("complete_manifest_hash") == hash_sampling.get("complete_manifest_hash")
    )
    positive_any = sum(1 for row in positive if row["positive_control"].get("chosen_sampled_any")) / max(1, len(positive))
    positive_parse = sum(int(row["positive_control"].get("parse_count") or 0) for row in positive) / max(
        1,
        sum(int(row["positive_control"].get("k") or 0) for row in positive),
    )
    positive_frequency = sum(int(row["positive_control"].get("chosen_count") or 0) for row in positive) / max(
        1,
        sum(int(row["positive_control"].get("k") or 0) for row in positive),
    )
    positive_by_action_type: dict[str, dict[str, Any]] = {}
    for row in positive:
        try:
            action_type = str(json.loads(row["chosen_action_key"]).get("type") or "unknown")
        except (TypeError, ValueError, json.JSONDecodeError):
            action_type = "unknown"
        bucket = positive_by_action_type.setdefault(action_type, {
            "positions": 0,
            "sample_count": 0,
            "parse_count": 0,
            "chosen_count": 0,
            "chosen_any_count": 0,
        })
        payload = row["positive_control"]
        bucket["positions"] += 1
        bucket["sample_count"] += int(payload.get("k") or 0)
        bucket["parse_count"] += int(payload.get("parse_count") or 0)
        bucket["chosen_count"] += int(payload.get("chosen_count") or 0)
        bucket["chosen_any_count"] += int(bool(payload.get("chosen_sampled_any")))
    for bucket in positive_by_action_type.values():
        bucket["parse_rate"] = bucket["parse_count"] / max(1, bucket["sample_count"])
        bucket["chosen_frequency"] = bucket["chosen_count"] / max(1, bucket["sample_count"])
        bucket["chosen_sampled_any"] = bucket["chosen_any_count"] / max(1, bucket["positions"])
    parity_rate = sum(1 for row in rows if row["parser_parity"].get("all_agree")) / max(1, len(rows))
    prior_key_rows = [row for row in rows if row.get("prior_greedy_action_key") is not None]
    prior_key_stability = sum(
        1 for row in prior_key_rows
        if row.get("prior_greedy_action_key") == row.get("new_greedy_action_key")
    ) / max(1, len(prior_key_rows))
    prior_chosen_stability = sum(
        1 for row in rows
        if bool(row.get("prior_greedy_is_chosen")) == bool(row.get("new_greedy_is_chosen"))
    ) / max(1, len(rows))
    e02 = len(positive) > 0 and positive_any >= args.positive_any_threshold
    e03 = len(rows) > 0 and parity_rate == 1.0
    expected_protocol = {
        "k": args.k,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "sample_batch_size": args.sample_batch_size,
        "seed": args.seed,
        "coord_bucket": args.coord_bucket,
        "match_threshold": args.match_threshold,
    }
    protocol_hash = hashlib.sha256(
        json.dumps(expected_protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    worker_commands = list(manifest.get("worker_commands") or [])
    required_command_fragments = [
        f"--pairs {manifest_pairs_path}",
        f"--prior-eval {manifest_prior_path}",
        f"--model-path {manifest.get('model_path')}",
        f"--adapter-dir {manifest.get('adapter_dir')}",
        f"--k {args.k}",
        f"--temperature {args.temperature}",
        f"--top-p {args.top_p}",
        f"--max-new-tokens {args.max_new_tokens}",
        f"--seed {args.seed}",
    ]
    worker_commands_match = (
        len(worker_commands) == 8
        and all(all(fragment in command for fragment in required_command_fragments) for command in worker_commands)
    )
    expected_id_set = set(expected_ids)
    raw_id_set = set(raw_ids)
    actual_k_values = [int(row["positive_control"].get("k") or 0) for row in positive]
    integrity_checks = {
        "manifest_present": bool(manifest),
        "pairs_hash_match": bool(manifest_pairs_path.exists() and manifest.get("pairs_sha256") == file_sha256(manifest_pairs_path)),
        "prior_eval_hash_match": bool(manifest_prior_path.exists() and manifest.get("prior_eval_sha256") == file_sha256(manifest_prior_path)),
        "protocol_match": manifest.get("protocol") == expected_protocol and manifest.get("protocol_sha256") == protocol_hash,
        "worker_commands_match": worker_commands_match,
        "expected_pair_count_212": len(expected_ids) == 212,
        "expected_pair_ids_unique": len(expected_ids) == len(expected_id_set),
        "audit_row_count_212": len(raw_rows) == 212,
        "audit_row_ids_unique": len(raw_ids) == len(raw_id_set),
        "no_missing_ids": expected_id_set == raw_id_set,
        "positive_subset_exact_48": len(expected_positive_ids) == 48 and actual_positive_ids == expected_positive_ids,
        "positive_actual_k_16": bool(actual_k_values) and all(value == args.k == 16 for value in actual_k_values),
        "audit_shard_count_8": len(args.audit_shards) == 8,
    }
    integrity_pass = all(integrity_checks.values())
    gate = "G0 PASS" if integrity_pass and e01 and e02 and e03 else "G0 FAIL"

    if not integrity_pass:
        diagnosis = "Audit integrity failed closed; incomplete, stale, or protocol-mismatched evidence cannot adjudicate G0."
    elif not e01:
        diagnosis = "Checkpoint identity failed; greedy/sampling harness checkpoint loading must be fixed before any sweep."
    elif not e03:
        diagnosis = "Parser/judge parity failed; normalization mismatch must be fixed before any sweep."
    elif not e02 and positive_parse < 0.5:
        diagnosis = "Checkpoint and parser parity pass, but high-temperature samples are mostly unparsable; the old zero is driven by sampled-output degeneration, and G0 positive control fails."
    elif not e02:
        diagnosis = "Checkpoint and parser parity pass, but chosen sampled-any fails the positive control at the frozen temperature."
    else:
        diagnosis = "Harness identity, positive control, and parser parity all pass. Previous Metric 1 remains void until corrected arm-wide re-sampling is published."

    summary = {
        "gate_g0": gate,
        "diagnosis": diagnosis,
        "old_metric_1_void": True,
        "phase_1_started": False,
        "audit_integrity": {
            "pass": integrity_pass,
            "checks": integrity_checks,
            "expected_ids": len(expected_id_set),
            "raw_rows": len(raw_rows),
            "unique_rows": len(raw_id_set),
            "expected_positive_ids": len(expected_positive_ids),
            "actual_positive_ids": len(actual_positive_ids),
            "actual_k_values": sorted(set(actual_k_values)),
            "run_manifest": args.run_manifest,
        },
        "e0_1": {
            "pass": e01,
            "checkpoint_canonical_hash": checkpoint_hash,
            "greedy_loaded_hash": hash_greedy.get("loaded_canonical_hash"),
            "sampling_loaded_hash": hash_sampling.get("loaded_canonical_hash"),
            "greedy_hash_match": hash_greedy.get("hash_match"),
            "sampling_hash_match": hash_sampling.get("hash_match"),
            "complete_manifest_hash": hash_greedy.get("complete_manifest_hash"),
        },
        "e0_2": {
            "pass": e02,
            "positions": len(positive),
            "k_per_position": args.k,
            "chosen_sampled_any": positive_any,
            "chosen_frequency": positive_frequency,
            "parse_rate": positive_parse,
            "threshold": args.positive_any_threshold,
            "by_action_type": positive_by_action_type,
            "historical_k4_void_diagnostic": {
                "positions": len(historical_positive),
                "sample_count": historical_sample_count,
                "parse_rate": historical_parse_count / max(1, historical_sample_count),
                "chosen_frequency": historical_chosen_count / max(1, historical_sample_count),
                "chosen_sampled_any": historical_chosen_any,
            },
        },
        "e0_3": {
            "pass": e03,
            "positions": len(rows),
            "all_verdict_agreement": parity_rate,
            "disagreements": len(rows) - sum(1 for row in rows if row["parser_parity"].get("all_agree")),
            "prior_vs_new_greedy_action_key_stability": prior_key_stability,
            "prior_vs_new_chosen_verdict_stability": prior_chosen_stability,
        },
        "e0_4": {
            "entropy_bonus_in_previous_run": False,
            "previous_entropy_bonus_weight": 0.0,
            "logger_repaired": entropy_logger_repaired,
            "repair": "Always compute and log response-token entropy; separately log entropy bonus weight and contribution.",
            "evidence_path": args.entropy_metrics,
            "evidence": entropy_evidence,
        },
        "independent_side_finding": {
            "name": "sft_inject_retention_collapse",
            "injection_rows": 212,
            "reported_training_updates": 75,
            "final_checkpoint_global_step": 78,
            "floor_nonblind_retention": 1.0,
            "sft_inject_nonblind_retention": 0.5673,
            "retention_delta": -0.4327,
            "interpretation": "Small offline action injection strongly disturbs already-correct behavior, independently measuring reflex embedding depth.",
        },
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "e0_summary.json", summary)
    write_jsonl(out_dir / "e0_per_step.jsonl", rows)

    lines = [
        "# REG-INJECT Follow-up Phase 0 - Harness Audit",
        "",
        "The previous `INJECTION FAILS - LOCKED TOO DEEP` verdict was voided for this audit and remains void because G0 failed. No Phase 1 training or Phase 2 rollout was run.",
        "",
        "## E0.1 Checkpoint Identity",
        "",
        table(["state", "canonical SHA-256", "matches final checkpoint"], [
            ["final REG-INJECT checkpoint", checkpoint_hash, "reference"],
            ["greedy harness loaded state", hash_greedy.get("loaded_canonical_hash"), str(bool(hash_greedy.get("hash_match"))).lower()],
            ["sampling harness loaded state", hash_sampling.get("loaded_canonical_hash"), str(bool(hash_sampling.get("hash_match"))).lower()],
        ]),
        "",
        f"Complete base+processor+tokenizer+adapter manifest: `{hash_greedy.get('complete_manifest_hash')}`.",
        "",
        "## Audit Integrity (fail-closed)",
        "",
        table(["check", "pass"], [[name, str(bool(value)).lower()] for name, value in integrity_checks.items()]),
        "",
        f"E0.1: **{'PASS' if e01 else 'FAIL'}**.",
        "",
        "## E0.2 Positive Control (prior greedy-equals-chosen subset)",
        "",
        table(["metric", "value", "requirement"], [
            ["positions", len(positive), "cached 22.64% subset"],
            ["K per position", args.k, "16"],
            ["chosen sampled-any", pct(positive_any), f">= {pct(args.positive_any_threshold)}"],
            ["aggregate exact chosen frequency", pct(positive_frequency), "diagnostic"],
            ["sample parse rate", pct(positive_parse), "diagnostic"],
        ]),
        "",
        table(["run", "K", "positions", "parse rate", "chosen frequency", "chosen sampled-any"], [
            [
                "historical (void)",
                4,
                len(historical_positive),
                pct(historical_parse_count / max(1, historical_sample_count)),
                pct(historical_chosen_count / max(1, historical_sample_count)),
                pct(historical_chosen_any),
            ],
            ["Phase-0 audit", args.k, len(positive), pct(positive_parse), pct(positive_frequency), pct(positive_any)],
        ]),
        "",
        "Positive-control breakdown by chosen action type:",
        "",
        table(["action type", "positions", "parse rate", "chosen frequency", "chosen sampled-any"], [
            [
                action_type,
                payload["positions"],
                pct(payload["parse_rate"]),
                pct(payload["chosen_frequency"]),
                pct(payload["chosen_sampled_any"]),
            ]
            for action_type, payload in sorted(positive_by_action_type.items())
        ]),
        "",
        f"E0.2: **{'PASS' if e02 else 'FAIL'}**.",
        "",
        "## E0.3 Parser / Exact-Match Judge Parity",
        "",
        table(["metric", "positions", "value", "requirement"], [
            ["same greedy output through both parser/judges", len(rows), pct(parity_rate), "100%"],
            ["prior vs new greedy action-key stability", len(prior_key_rows), pct(prior_key_stability), "diagnostic"],
            ["prior vs new greedy-chosen verdict stability", len(rows), pct(prior_chosen_stability), "diagnostic"],
        ]),
        "",
        f"E0.3: **{'PASS' if e03 else 'FAIL'}**.",
        "",
        "## E0.4 Entropy Logging Repair",
        "",
        "The prior regularizer was label smoothing plus KL (`entropy_bonus=0`), so entropy was not intended to contribute to the objective. The logger was nevertheless wrong: it emitted `0.0` instead of measuring entropy. It now always records response-token entropy and separately records the entropy-bonus weight/contribution.",
        "",
        table(["evidence", "value"], [
            ["metrics artifact", args.entropy_metrics],
            ["measured token entropy", entropy_value],
            ["entropy bonus weight", entropy_weight],
            ["entropy bonus contribution", entropy_contribution],
        ]),
        "",
        f"Logger repaired and empirically verified: **{str(entropy_logger_repaired).lower()}**.",
        "",
        "## Independent Side Finding - Reflex Embedding Depth",
        "",
        "Plain SFT-INJECT used 212 heterogeneous action rows; its last logged curve point was step 75 and the saved final checkpoint was global step 78. Non-blind originally-correct retention fell from 100.00% to 56.73% (-43.27pp). This remains valid regardless of G0 and is recorded as an independent measurement of reflex embedding depth.",
        "",
        "## Gate G0",
        "",
        gate,
        "",
        diagnosis,
        "",
        "Required next action before Phase 1: repair the high-temperature sampling channel (the frozen T=1.5 run produced only 2.99% parseable samples), predeclare the repair, and repeat E0.2/G0. Do not reinterpret this G0 failure as model capacity evidence.",
        "",
        "Old Metric 1 numbers and the old capacity-lock verdict remain void. STOP for review; Phase 1 was not started.",
        "",
    ]
    (out_dir / "harness_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"gate_g0": gate, "report": str(out_dir / "harness_audit.md")}, indent=2), flush=True)


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", default="checkpoints/gui360-fullparam-sft-step250")
    parser.add_argument("--adapter-dir", default="outputs/hetero_inject/checkpoints/reg_inject/final/cooperative")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-max-pixels", type=int, default=602112)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--num-comm-rounds", type=int, default=2)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    hash_parser = subparsers.add_parser("hash")
    add_model_args(hash_parser)
    hash_parser.add_argument("--harness", choices=["greedy", "sampling"], required=True)
    hash_parser.add_argument("--output", required=True)
    hash_parser.set_defaults(func=hash_harness)

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--pairs", required=True)
    manifest_parser.add_argument("--prior-eval", default="outputs/hetero_inject/eval/reg_inject.jsonl")
    manifest_parser.add_argument("--model-path", default="checkpoints/gui360-fullparam-sft-step250")
    manifest_parser.add_argument("--adapter-dir", default="outputs/hetero_inject/checkpoints/reg_inject/final/cooperative")
    manifest_parser.add_argument("--output", required=True)
    manifest_parser.add_argument("--k", type=int, default=16)
    manifest_parser.add_argument("--sample-batch-size", type=int, default=4)
    manifest_parser.add_argument("--temperature", type=float, default=1.5)
    manifest_parser.add_argument("--top-p", type=float, default=0.95)
    manifest_parser.add_argument("--max-new-tokens", type=int, default=384)
    manifest_parser.add_argument("--seed", type=int, default=42)
    manifest_parser.add_argument("--coord-bucket", type=int, default=25)
    manifest_parser.add_argument("--match-threshold", type=float, default=0.5)
    manifest_parser.set_defaults(func=run_manifest)

    audit_parser = subparsers.add_parser("audit-shard")
    add_model_args(audit_parser)
    audit_parser.add_argument("--pairs", default="outputs/hetero_inject/data/hetero_inject_train.jsonl")
    audit_parser.add_argument("--prior-eval", default="outputs/hetero_inject/eval/reg_inject.jsonl")
    audit_parser.add_argument("--output", required=True)
    audit_parser.add_argument("--start", type=int, default=0)
    audit_parser.add_argument("--end", type=int, default=None)
    audit_parser.add_argument("--k", type=int, default=16)
    audit_parser.add_argument("--sample-batch-size", type=int, default=4)
    audit_parser.add_argument("--temperature", type=float, default=1.5)
    audit_parser.add_argument("--top-p", type=float, default=0.95)
    audit_parser.add_argument("--max-new-tokens", type=int, default=384)
    audit_parser.add_argument("--match-threshold", type=float, default=0.5)
    audit_parser.add_argument("--coord-bucket", type=int, default=25)
    audit_parser.add_argument("--store-chars", type=int, default=2000)
    audit_parser.add_argument("--seed", type=int, default=42)
    audit_parser.set_defaults(func=audit_shard)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--hash-greedy", required=True)
    report_parser.add_argument("--hash-sampling", required=True)
    report_parser.add_argument("--audit-shards", nargs="+", required=True)
    report_parser.add_argument("--run-manifest", default="outputs/hetero_inject_followup/phase0/run_manifest.json")
    report_parser.add_argument("--pairs", default="outputs/hetero_inject_followup/phase0/audit_pairs_balanced.jsonl")
    report_parser.add_argument("--prior-eval", default="outputs/hetero_inject/eval/reg_inject.jsonl")
    report_parser.add_argument("--output-dir", default="outputs/hetero_inject_followup/phase0")
    report_parser.add_argument("--k", type=int, default=16)
    report_parser.add_argument("--sample-batch-size", type=int, default=4)
    report_parser.add_argument("--temperature", type=float, default=1.5)
    report_parser.add_argument("--top-p", type=float, default=0.95)
    report_parser.add_argument("--max-new-tokens", type=int, default=384)
    report_parser.add_argument("--seed", type=int, default=42)
    report_parser.add_argument("--coord-bucket", type=int, default=25)
    report_parser.add_argument("--match-threshold", type=float, default=0.5)
    report_parser.add_argument("--positive-any-threshold", type=float, default=0.90)
    report_parser.add_argument("--entropy-metrics", default="outputs/hetero_inject_followup/phase0/entropy_smoke/metrics.jsonl")
    report_parser.add_argument("--entropy-logger-repaired", action=argparse.BooleanOptionalAction, default=True)
    report_parser.set_defaults(func=report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()