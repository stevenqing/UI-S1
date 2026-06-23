#!/usr/bin/env python3
"""Recompute GUI-Odyssey trajectory metrics from saved raw model outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "gui_odyssey_eval"))
sys.path.insert(0, str(PROJECT_ROOT))

from odyssey_action_matching import evaluate_odyssey_action, pred_coord_to_1k  # noqa: E402
from scripts.summarize_gui_odyssey_trajectory_results import summarize, write_jsonl  # noqa: E402


JsonDict = dict[str, Any]


def iter_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_gt(jsonl_path: Path) -> dict[str, JsonDict]:
    return {str(row["episode_id"]): row for row in iter_jsonl(jsonl_path)}


def recompute_row(result: JsonDict, gt_episode: JsonDict) -> JsonDict:
    out = dict(result)
    new_steps = []
    for step in result.get("step_results", []) or []:
        new_step = dict(step)
        step_num = int(step.get("step_num", 0))
        gt_step = gt_episode["steps"][step_num]
        check = gt_step["check_options"]
        pred_action = step.get("pred_action")
        if pred_action is None:
            type_match = False
            extract_match = False
            pred_coord_1k = None
        else:
            type_match, extract_match = evaluate_odyssey_action(
                pred_action,
                check,
                step.get("resized_width") or 1000,
                step.get("resized_height") or 1000,
            )
            pred_coord_1k = None
            coord = pred_action.get("coordinate") if isinstance(pred_action, dict) else None
            if coord and isinstance(coord, (list, tuple)) and len(coord) >= 2:
                pred_coord_1k = pred_coord_to_1k(
                    [float(coord[0]), float(coord[1])],
                    step.get("resized_width") or 1000,
                    step.get("resized_height") or 1000,
                )
        new_step["type_match"] = bool(type_match)
        new_step["extract_match"] = bool(extract_match)
        new_step["pred_coord_1k"] = pred_coord_1k
        new_step["gt_coord_1k"] = check.get("coordinate")
        new_steps.append(new_step)
    correct_steps = sum(bool(step.get("extract_match")) for step in new_steps)
    num_steps = int(result.get("num_steps", len(gt_episode.get("steps", []))))
    out["step_results"] = new_steps
    out["final_step_id"] = correct_steps
    out["task_success"] = bool(correct_steps == num_steps and len(new_steps) == num_steps)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute GUI-Odyssey metrics from trajectory_results.jsonl")
    parser.add_argument("--trajectory-results", type=Path, required=True)
    parser.add_argument("--jsonl-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    gt = load_gt(args.jsonl_file)
    recomputed = []
    for result in iter_jsonl(args.trajectory_results):
        episode_id = str(result.get("episode_id"))
        recomputed.append(recompute_row(result, gt[episode_id]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "trajectory_results.jsonl", recomputed)
    summary, trajectory_rows, error_rows = summarize(recomputed)
    (args.output_dir / "summary_enriched.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(args.output_dir / "trajectory_metrics.jsonl", trajectory_rows)
    write_jsonl(args.output_dir / "error_samples.jsonl", error_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()