#!/usr/bin/env python3
"""Summarize GUI-Odyssey trajectory_results.jsonl and flatten errors."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]


def iter_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def length_bucket(num_steps: int) -> str:
    if num_steps <= 3:
        return "1-3"
    if num_steps <= 6:
        return "4-6"
    if num_steps <= 10:
        return "7-10"
    if num_steps <= 15:
        return "11-15"
    return "16+"


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def summarize(results: list[JsonDict]) -> tuple[JsonDict, list[JsonDict], list[JsonDict]]:
    total_episodes = len(results)
    success_episodes = 0
    total_steps = 0
    evaluated_steps = 0
    correct_steps = 0
    parse_ok_steps = 0
    type_match_steps = 0
    trajectory_rows: list[JsonDict] = []
    error_rows: list[JsonDict] = []
    by_action: dict[str, Counter[str]] = defaultdict(Counter)
    by_length: dict[str, Counter[str]] = defaultdict(Counter)
    by_category: dict[str, Counter[str]] = defaultdict(Counter)

    for result in results:
        episode_id = result.get("episode_id")
        num_steps = int(result.get("num_steps", 0))
        steps = result.get("step_results", []) or []
        episode_correct_steps = sum(bool(step.get("extract_match")) for step in steps)
        episode_success = bool(result.get("task_success")) and episode_correct_steps == num_steps and len(steps) == num_steps
        bucket = result.get("length_bucket") or length_bucket(num_steps)
        category = str(result.get("category", "unknown"))
        first_error_step = None

        total_steps += num_steps
        evaluated_steps += len(steps)
        correct_steps += episode_correct_steps
        success_episodes += int(episode_success)
        by_length[bucket]["episodes"] += 1
        by_length[bucket]["success"] += int(episode_success)
        by_length[bucket]["steps"] += num_steps
        by_length[bucket]["correct_steps"] += episode_correct_steps
        by_category[category]["episodes"] += 1
        by_category[category]["success"] += int(episode_success)
        by_category[category]["steps"] += num_steps
        by_category[category]["correct_steps"] += episode_correct_steps

        for step in steps:
            action_type = str(step.get("gt_action_type", "unknown"))
            step_ok = bool(step.get("extract_match"))
            type_ok = bool(step.get("type_match"))
            parse_ok = bool(step.get("parse_ok", True))
            parse_ok_steps += int(parse_ok)
            type_match_steps += int(type_ok)
            by_action[action_type]["steps"] += 1
            by_action[action_type]["type_match"] += int(type_ok)
            by_action[action_type]["semantic_match"] += int(step_ok)
            if step_ok:
                continue
            if first_error_step is None:
                first_error_step = int(step.get("step_num", -1))
            error_rows.append(
                {
                    "episode_id": episode_id,
                    "goal": result.get("goal"),
                    "category": category,
                    "device_name": result.get("device_name"),
                    "num_steps": num_steps,
                    "length_bucket": bucket,
                    "step_num": step.get("step_num"),
                    "gt_action_type": action_type,
                    "gt_action": step.get("gt_action"),
                    "pred_action": step.get("pred_action"),
                    "parse_ok": parse_ok,
                    "parse_error": step.get("parse_error", ""),
                    "type_match": type_ok,
                    "extract_match": step_ok,
                    "raw_response": step.get("raw_response", ""),
                    "pred_coord_1k": step.get("pred_coord_1k"),
                    "gt_coord_1k": step.get("gt_coord_1k"),
                    "resized_width": step.get("resized_width"),
                    "resized_height": step.get("resized_height"),
                }
            )

        if len(steps) < num_steps:
            for missing_step in range(len(steps), num_steps):
                error_rows.append(
                    {
                        "episode_id": episode_id,
                        "goal": result.get("goal"),
                        "category": category,
                        "device_name": result.get("device_name"),
                        "num_steps": num_steps,
                        "length_bucket": bucket,
                        "step_num": missing_step,
                        "gt_action_type": "not_evaluated",
                        "gt_action": None,
                        "pred_action": None,
                        "parse_ok": False,
                        "parse_error": "step_not_evaluated",
                        "type_match": False,
                        "extract_match": False,
                        "raw_response": "",
                    }
                )

        trajectory_rows.append(
            {
                "episode_id": episode_id,
                "goal": result.get("goal"),
                "category": category,
                "device_name": result.get("device_name"),
                "num_steps": num_steps,
                "steps_evaluated": len(steps),
                "correct_steps": episode_correct_steps,
                "episode_success": episode_success,
                "step_success_rate": ratio(episode_correct_steps, num_steps),
                "first_error_step": first_error_step,
                "length_bucket": bucket,
            }
        )

    summary = {
        "episodes": total_episodes,
        "episode_success": success_episodes,
        "episode_success_rate": ratio(success_episodes, total_episodes),
        "total_steps": total_steps,
        "evaluated_steps": evaluated_steps,
        "correct_steps": correct_steps,
        "step_success_rate": ratio(correct_steps, total_steps),
        "evaluated_step_success_rate": ratio(correct_steps, evaluated_steps),
        "parse_ok_steps": parse_ok_steps,
        "parse_ok_rate": ratio(parse_ok_steps, evaluated_steps),
        "type_match_steps": type_match_steps,
        "type_match_rate": ratio(type_match_steps, evaluated_steps),
        "error_steps": len(error_rows),
        "action_type_stats": {
            action_type: {
                "steps": counts["steps"],
                "type_match": counts["type_match"],
                "semantic_match": counts["semantic_match"],
                "type_match_rate": ratio(counts["type_match"], counts["steps"]),
                "semantic_match_rate": ratio(counts["semantic_match"], counts["steps"]),
            }
            for action_type, counts in sorted(by_action.items())
        },
        "length_bucket_stats": {
            bucket: {
                "episodes": counts["episodes"],
                "episode_success": counts["success"],
                "episode_success_rate": ratio(counts["success"], counts["episodes"]),
                "steps": counts["steps"],
                "correct_steps": counts["correct_steps"],
                "step_success_rate": ratio(counts["correct_steps"], counts["steps"]),
            }
            for bucket, counts in sorted(by_length.items())
        },
        "category_stats": {
            category: {
                "episodes": counts["episodes"],
                "episode_success": counts["success"],
                "episode_success_rate": ratio(counts["success"], counts["episodes"]),
                "steps": counts["steps"],
                "correct_steps": counts["correct_steps"],
                "step_success_rate": ratio(counts["correct_steps"], counts["steps"]),
            }
            for category, counts in sorted(by_category.items())
        },
    }
    return summary, trajectory_rows, error_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GUI-Odyssey trajectory evaluation results")
    parser.add_argument("--trajectory-results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    results = list(iter_jsonl(args.trajectory_results))
    summary, trajectory_rows, error_rows = summarize(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary_enriched.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(args.output_dir / "trajectory_metrics.jsonl", trajectory_rows)
    write_jsonl(args.output_dir / "error_samples.jsonl", error_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
