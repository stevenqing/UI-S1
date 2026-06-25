#!/usr/bin/env python3
"""Build Condition C Stage2 dose-response pairs."""

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


def load_pilot_ids(path: Path) -> set[str]:
    return {str(row["episode_id"]) for row in iter_jsonl(path)}


def load_rollout_by_episode(path: Path) -> dict[str, JsonDict]:
    output = {}
    index = {}
    duplicates = []
    for row in iter_jsonl(path):
        episode_id = str(row["episode_id"])
        output[episode_id] = row
        for step in row.get("step_results", []) or []:
            key = (episode_id, int(step["step_num"]))
            if key in index:
                duplicates.append(key)
            index[key] = step
    if duplicates:
        raise ValueError(f"duplicate rollout keys: {duplicates[:5]}")
    return output


def load_dataset_by_id(path: Path) -> dict[str, JsonDict]:
    return {str(row["episode_id"]): row for row in iter_jsonl(path)}


def injectable(step: JsonDict) -> bool:
    return (not bool(step.get("extract_match"))) and bool(step.get("parse_ok")) and isinstance(step.get("pred_action"), dict)


def family_for_action(action: JsonDict) -> str:
    action_type = str((action or {}).get("action", "unknown"))
    if action_type in {"system_button", "open", "terminate"}:
        return "navigation_phase"
    if action_type == "click":
        return "click_grounding"
    if action_type in {"type", "answer", "key"}:
        return "text_value_carry"
    if action_type == "swipe":
        return "swipe_scroll"
    return action_type


def max_contiguous_dose(steps: list[JsonDict], target_step: int) -> int:
    max_dose = 0
    for dose in (1, 2, 3):
        if target_step - dose < 0:
            break
        if all(injectable(steps[target_step - offset]) for offset in range(1, dose + 1)):
            max_dose = dose
        else:
            break
    return max_dose


def injection_map(episode: JsonDict, steps: list[JsonDict], target_step: int, dose: int, mode: str) -> dict[str, JsonDict]:
    output = {}
    for offset in range(1, dose + 1):
        source_step = target_step - offset
        if mode == "gt":
            action = episode["steps"][source_step]["action_content"]
        elif mode == "wrong":
            action = steps[source_step]["pred_action"]
        else:
            raise KeyError(mode)
        output[str(source_step)] = action
    return output


def build_rows(args: argparse.Namespace) -> tuple[list[JsonDict], JsonDict]:
    rollout_by_episode = load_rollout_by_episode(args.rollout_results)
    dataset_by_id = load_dataset_by_id(args.jsonl_file)
    pilot_ids = load_pilot_ids(args.pilot_episode_ids)
    rows = []
    skipped = Counter()
    by_dose = Counter()
    by_family = defaultdict(Counter)
    for episode_id in sorted(pilot_ids):
        rollout = rollout_by_episode.get(episode_id)
        episode = dataset_by_id.get(episode_id)
        if not rollout or not episode:
            skipped["missing_episode"] += 1
            continue
        steps = rollout.get("step_results", []) or []
        for target in steps:
            target_step = int(target["step_num"])
            if not target.get("extract_match"):
                skipped["target_not_clean"] += 1
                continue
            max_dose = max_contiguous_dose(steps, target_step)
            if max_dose == 0:
                skipped["no_injectable_recent_error"] += 1
                continue
            target_gt_action = episode["steps"][target_step]["action_content"]
            family = family_for_action(target_gt_action)
            if args.stage == "zeropoint":
                if max_dose < 3:
                    skipped["not_dose3_eligible"] += 1
                    continue
                doses = [3]
                mode = "gt"
                set_type = "dose3_zero_point"
            else:
                doses = list(range(1, max_dose + 1))
                mode = "wrong"
                set_type = "per_dose"
            for dose in doses:
                row = {
                    "pair_id": f"{episode_id}:{target_step}:dose{dose}:{args.stage}",
                    "stage": "stage2",
                    "stage2_mode": args.stage,
                    "set_type": set_type,
                    "episode_id": episode_id,
                    "target_step": target_step,
                    "dose": dose,
                    "headline_dose3_eligible": max_dose >= 3,
                    "max_eligible_dose": max_dose,
                    "source_steps": [target_step - offset for offset in range(1, dose + 1)],
                    "inject_mode": mode,
                    "inject_actions_by_step": injection_map(episode, steps, target_step, dose, mode),
                    "source_wrong_actions_by_step": injection_map(episode, steps, target_step, dose, "wrong"),
                    "target_gt_action": target_gt_action,
                    "target_gt_action_type": target_gt_action.get("action"),
                    "target_family": family,
                    "target_screenshot": episode["steps"][target_step].get("screenshot"),
                }
                rows.append(row)
                by_dose[dose] += 1
                by_family[family][dose] += 1
    manifest = {
        "stage": args.stage,
        "rows": len(rows),
        "by_dose": dict(sorted(by_dose.items())),
        "by_family": {family: dict(counts) for family, counts in by_family.items()},
        "skipped": dict(skipped),
        "jsonl_file": str(args.jsonl_file),
        "rollout_results": str(args.rollout_results),
        "pilot_episode_ids": str(args.pilot_episode_ids),
    }
    return rows, manifest


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Condition C Stage2 pairs")
    parser.add_argument("--jsonl-file", type=Path, required=True)
    parser.add_argument("--rollout-results", type=Path, required=True)
    parser.add_argument("--pilot-episode-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["zeropoint", "main"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, manifest = build_rows(args)
    count = write_jsonl(args.output, rows)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "rows": count, "manifest": str(manifest_path), "by_dose": manifest["by_dose"], "skipped": manifest["skipped"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()