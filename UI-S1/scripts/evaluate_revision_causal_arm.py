#!/usr/bin/env python3
"""Greedy evaluation of a model on a revision causal-arm row grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.hetero_inject_phase1_eval import generate_batch, score_output  # noqa: E402
from scripts.minimal_validation import load_model_for_eval  # noqa: E402
from scripts.rl_feasibility_sampling import sanitize_jsonable  # noqa: E402
from v23_visual_transition.train_offline_grpo import build_eval_style_messages  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(sanitize_jsonable(dict(row)), ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_actor: dict[str, dict[str, Any]] = {}
    for actor in sorted({str(row["actor"]) for row in rows}):
        group = [row for row in rows if str(row["actor"]) == actor]
        by_actor[actor] = {
            "rows": len(group),
            "accuracy": sum(bool(row["student_correct"]) for row in group) / len(group),
            "parse_rate": sum(bool(row["parse_ok"]) for row in group) / len(group),
        }
    by_prefix: dict[str, dict[str, Any]] = {}
    for name, clean in (("clean", True), ("dirty", False)):
        group = [row for row in rows if bool(row["prefix_clean"]) is clean]
        by_prefix[name] = {
            "rows": len(group),
            "accuracy": sum(bool(row["student_correct"]) for row in group) / max(1, len(group)),
            "parse_rate": sum(bool(row["parse_ok"]) for row in group) / max(1, len(group)),
        }
    by_trajectory: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_trajectory.setdefault(str(row["correction_id"]), []).append(row)
    complete_trajectories = sum(
        bool(group) and all(bool(row["student_correct"]) for row in group)
        for group in by_trajectory.values()
    )
    return {
        "rows": len(rows),
        "student_correct": sum(bool(row["student_correct"]) for row in rows),
        "student_accuracy": sum(bool(row["student_correct"]) for row in rows) / max(1, len(rows)),
        "parse_rate": sum(bool(row["parse_ok"]) for row in rows) / max(1, len(rows)),
        "mean_reward": sum(float(row["reward"]) for row in rows) / max(1, len(rows)),
        "trajectories": len(by_trajectory),
        "complete_trajectories": complete_trajectories,
        "trajectory_success_rate": complete_trajectories / max(1, len(by_trajectory)),
        "by_actor": by_actor,
        "by_prefix": by_prefix,
        "predicted_action_types": dict(Counter(str((row.get("student_action") or {}).get("action") or "unparsed") for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", default="checkpoints/gui360-fullparam-sft-step250")
    parser.add_argument("--adapter-dir", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--image-max-pixels", type=int, default=602112)
    parser.add_argument("--match-threshold", type=float, default=0.5)
    parser.add_argument("--coord-bucket", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--num-comm-rounds", type=int, default=2)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    args = parser.parse_args()

    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard index/count")
    input_path = Path(args.input)
    all_rows = read_jsonl(input_path)
    if args.max_rows > 0:
        all_rows = all_rows[: args.max_rows]
    selected = [row for index, row in enumerate(all_rows) if index % args.shard_count == args.shard_index]
    if not selected:
        raise ValueError("empty evaluation shard")
    jobs = [
        {
            "sample_id": str(row["sample_id"]),
            "messages": build_eval_style_messages(str(row["goal"]), list(row.get("history") or []), str(row["image"])),
            "image": str(row["image"]),
            "image_w": int(row.get("image_w") or 1040),
            "image_h": int(row.get("image_h") or 736),
            "gt_action": row["gt_action"],
            "source": row,
        }
        for row in selected
    ]

    model, processor, device = load_model_for_eval(args)
    processor.tokenizer.padding_side = "left"
    outputs: list[dict[str, Any]] = []
    for start in tqdm(range(0, len(jobs), args.batch_size), desc=f"{args.arm}:shard{args.shard_index}"):
        batch = jobs[start : start + args.batch_size]
        texts = generate_batch(
            model,
            processor,
            device,
            batch,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed + start,
        )
        for job, text in zip(batch, texts):
            scored = score_output(text, job, args.coord_bucket, args.match_threshold)
            source = job["source"]
            outputs.append(
                {
                    "arm": args.arm,
                    "sample_id": source["sample_id"],
                    "correction_id": source["correction_id"],
                    "trajectory_id": source["trajectory_id"],
                    "target_id": source["target_id"],
                    "episode_id": source["episode_id"],
                    "step_idx": source["step_idx"],
                    "actor": source["actor"],
                    "corrector": source["corrector"],
                    "target_source": source["target_source"],
                    "history_source": source["history_source"],
                    "prefix_clean": source["prefix_clean"],
                    "prefix_wrong_count": source["prefix_wrong_count"],
                    "actor_correct": source["actor_correct"],
                    "revision_correct": source["revision_correct"],
                    "student_correct": scored["success"],
                    "reward": scored["reward"],
                    "parse_ok": scored["parse_ok"],
                    "parser_mode": scored["parser_mode"],
                    "student_action_key": scored["action_key"],
                    "student_action": scored["pred_action"],
                    "raw_output": scored["raw_output"],
                }
            )
        torch.cuda.empty_cache()

    output_path = Path(args.output)
    write_jsonl(output_path, outputs)
    summary = {
        "arm": args.arm,
        **summarize(outputs),
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "model_path": args.model_path,
        "adapter_dir": args.adapter_dir,
        "seed": args.seed,
        "greedy": True,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "max_rows": args.max_rows,
        "output": str(output_path),
        "output_sha256": sha256(output_path),
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
