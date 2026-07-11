#!/usr/bin/env python3
"""Prepare equal-budget LoRA screening subsets from revision causal arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


COMMON_ARMS = (
    "a1_gt_target_gt_history",
    "a2_random_target_gt_history",
    "a4_revision_target_revision_history",
    "a5_revision_target_gt_history",
    "a6_gt_target_revision_history",
)
SPECIAL_ARMS = (
    "a7_revision_clean_prefix",
    "a9_revision_internvl3_only",
    "a10_revision_qwen3_vl_only",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def paired_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["correction_id"]), int(row["step_idx"])


def deterministic_sample(rows: Sequence[Mapping[str, Any]], count: int, seed: str) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"insufficient rows: {len(rows)} < {count}")
    copied = [dict(row) for row in rows]
    random.Random(seed).shuffle(copied)
    return copied[:count]


def balanced_common_keys(rows: Sequence[Mapping[str, Any]], per_stratum: int, seed: int) -> list[tuple[str, int]]:
    keys: list[tuple[str, int]] = []
    for actor in sorted({str(row["actor"]) for row in rows}):
        for clean in (True, False):
            group = [row for row in rows if str(row["actor"]) == actor and bool(row["prefix_clean"]) is clean]
            chosen = deterministic_sample(group, per_stratum, f"{seed}:common:{actor}:{int(clean)}")
            keys.extend(paired_key(row) for row in chosen)
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate common screening key")
    return keys


def annotate(rows: Sequence[Mapping[str, Any]], screen_arm: str, sampling: str) -> list[dict[str, Any]]:
    output = []
    for index, source in enumerate(rows):
        row = dict(source)
        row["screen_arm"] = screen_arm
        row["screen_sampling"] = sampling
        row["screen_index"] = index
        row["original_sample_id"] = row["sample_id"]
        row["sample_id"] = f"screen:{screen_arm}:{row['correction_id']}:{row['step_idx']}"
        output.append(row)
    return output


def summarize(rows: Sequence[Mapping[str, Any]], path: Path, optimizer_steps: int) -> dict[str, Any]:
    return {
        "arm": str(rows[0]["screen_arm"]),
        "rows": len(rows),
        "optimizer_steps": optimizer_steps,
        "gradient_accumulation_steps": len(rows) // optimizer_steps,
        "diagnostic_label_accuracy": sum(bool(row["diagnostic_matcher_correct"]) for row in rows) / len(rows),
        "prefix_clean_fraction": sum(bool(row["prefix_clean"]) for row in rows) / len(rows),
        "actor_counts": dict(Counter(str(row["actor"]) for row in rows)),
        "target_source_counts": dict(Counter(str(row["target_source"]) for row in rows)),
        "history_source_counts": dict(Counter(str(row["history_source"]) for row in rows)),
        "action_type_counts": dict(Counter(str((row.get("chosen_action") or {}).get("action") or "unknown") for row in rows)),
        "output": str(path),
        "output_sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-arm-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rows-per-arm", type=int, default=800)
    parser.add_argument("--optimizer-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.rows_per_arm <= 0 or args.optimizer_steps <= 0:
        raise ValueError("rows/steps must be positive")
    if args.rows_per_arm % args.optimizer_steps:
        raise ValueError("rows-per-arm must be divisible by optimizer-steps")
    if args.rows_per_arm % 4:
        raise ValueError("rows-per-arm must be divisible by four actor/prefix strata")

    causal_dir = Path(args.causal_arm_dir)
    out_dir = Path(args.output_dir)
    arms = {
        arm: read_jsonl(causal_dir / f"{arm}.jsonl")
        for arm in (*COMMON_ARMS, *SPECIAL_ARMS)
    }
    common_keys = balanced_common_keys(
        arms["a4_revision_target_revision_history"], args.rows_per_arm // 4, args.seed
    )

    selected: dict[str, list[dict[str, Any]]] = {}
    for arm in COMMON_ARMS:
        by_key = {paired_key(row): row for row in arms[arm]}
        missing = [key for key in common_keys if key not in by_key]
        if missing:
            raise ValueError(f"{arm} missing common keys: {missing[:10]}")
        selected[arm] = annotate(
            [by_key[key] for key in common_keys], arm, "matched_actor_x_prefix_balanced"
        )

    a7 = arms["a7_revision_clean_prefix"]
    a7_rows = []
    for actor in sorted({str(row["actor"]) for row in a7}):
        group = [row for row in a7 if str(row["actor"]) == actor]
        a7_rows.extend(deterministic_sample(group, args.rows_per_arm // 2, f"{args.seed}:a7:{actor}"))
    selected["a7_revision_clean_prefix"] = annotate(a7_rows, "a7_revision_clean_prefix", "actor_balanced_clean_prefix")

    for arm in ("a9_revision_internvl3_only", "a10_revision_qwen3_vl_only"):
        source = arms[arm]
        rows = []
        for clean in (True, False):
            group = [row for row in source if bool(row["prefix_clean"]) is clean]
            rows.extend(deterministic_sample(group, args.rows_per_arm // 2, f"{args.seed}:{arm}:{int(clean)}"))
        selected[arm] = annotate(rows, arm, "prefix_balanced_source_specific")

    summaries = {}
    for arm, rows in selected.items():
        if len(rows) != args.rows_per_arm:
            raise ValueError(f"screen arm row mismatch: {arm}={len(rows)}")
        path = out_dir / f"{arm}.jsonl"
        write_jsonl(path, rows)
        summaries[arm] = summarize(rows, path, args.optimizer_steps)
        write_json(path.with_suffix(".summary.json"), summaries[arm])

    common_key_sets = [
        {paired_key(row) for row in selected[arm]} for arm in COMMON_ARMS
    ]
    manifest = {
        "version": "revision-lora-screen-v1",
        "seed": args.seed,
        "rows_per_arm": args.rows_per_arm,
        "optimizer_steps": args.optimizer_steps,
        "gradient_accumulation_steps": args.rows_per_arm // args.optimizer_steps,
        "common_arms": list(COMMON_ARMS),
        "common_grid_exact_match": all(keys == common_key_sets[0] for keys in common_key_sets[1:]),
        "special_arms": list(SPECIAL_ARMS),
        "arms": summaries,
        "training_policy": {
            "method": "single-GPU LoRA-only SFT",
            "max_steps": args.optimizer_steps,
            "num_epochs": 1,
            "same_update_budget": True,
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps({
        "common_grid_exact_match": manifest["common_grid_exact_match"],
        "rows_per_arm": args.rows_per_arm,
        "optimizer_steps": args.optimizer_steps,
        "arms": {arm: summary["diagnostic_label_accuracy"] for arm, summary in summaries.items()},
        "output_dir": str(out_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
