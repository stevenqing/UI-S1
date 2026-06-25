#!/usr/bin/env python3
"""Build valid Condition C single-error injection pairs."""

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


def load_pilot_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    ids = set()
    for row in iter_jsonl(path):
        ids.add(str(row["episode_id"]))
    return ids


def load_rollout_index(path: Path) -> tuple[dict[tuple[str, int], JsonDict], dict[str, JsonDict]]:
    index: dict[tuple[str, int], JsonDict] = {}
    by_episode: dict[str, JsonDict] = {}
    duplicates = []
    for row in iter_jsonl(path):
        episode_id = str(row["episode_id"])
        by_episode[episode_id] = row
        for step in row.get("step_results", []) or []:
            key = (episode_id, int(step["step_num"]))
            if key in index:
                duplicates.append(key)
            index[key] = step
    if duplicates:
        raise ValueError(f"duplicate rollout keys: {duplicates[:5]}")
    return index, by_episode


def load_dataset_by_id(path: Path) -> dict[str, JsonDict]:
    return {str(row["episode_id"]): row for row in iter_jsonl(path)}


def distance_bin(distance: int) -> str:
    if distance <= 3:
        return str(distance)
    if distance <= 5:
        return "4-5"
    if distance <= 10:
        return "6-10"
    if distance <= 20:
        return "11-20"
    return "21+"


def action_family(action: JsonDict) -> str:
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


def injectable_source(step: JsonDict) -> bool:
    return (not bool(step.get("extract_match"))) and bool(step.get("parse_ok")) and isinstance(step.get("pred_action"), dict)


def target_ok(step: JsonDict) -> bool:
    return bool(step.get("extract_match"))


def parse_distances(raw: str) -> list[int]:
    output = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            output.extend(range(int(start), int(end) + 1))
        else:
            output.append(int(chunk))
    return sorted(set(output))


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_pairs(args: argparse.Namespace) -> tuple[list[JsonDict], JsonDict]:
    rollout_index, rollout_by_episode = load_rollout_index(args.rollout_results)
    dataset_by_id = load_dataset_by_id(args.jsonl_file)
    pilot_ids = load_pilot_ids(args.pilot_episode_ids)
    distances = parse_distances(args.distances)
    pairs = []
    skipped = Counter()
    by_distance = Counter()
    by_bin = Counter()
    by_family = defaultdict(Counter)
    for episode_id in sorted(pilot_ids or rollout_by_episode.keys()):
        rollout = rollout_by_episode.get(episode_id)
        episode = dataset_by_id.get(episode_id)
        if not rollout or not episode:
            skipped["missing_episode"] += 1
            continue
        steps = rollout.get("step_results", []) or []
        for target_step in steps:
            k = int(target_step["step_num"])
            if not target_ok(target_step):
                skipped["target_not_clean"] += 1
                continue
            target_gt_action = episode["steps"][k]["action_content"]
            for distance in distances:
                j = k - distance
                if j < 0:
                    skipped["source_before_episode"] += 1
                    continue
                source_step = rollout_index.get((episode_id, j))
                if source_step is None:
                    skipped["missing_source"] += 1
                    continue
                if not injectable_source(source_step):
                    skipped["source_not_injectable"] += 1
                    continue
                source_gt_action = episode["steps"][j]["action_content"]
                wrong_action = source_step["pred_action"]
                inject_action = source_gt_action if args.inject_mode == "gt" else wrong_action
                row = {
                    "pair_id": f"{episode_id}:{k}:{distance}:{args.inject_mode}",
                    "stage": args.stage,
                    "inject_mode": args.inject_mode,
                    "episode_id": episode_id,
                    "target_step": k,
                    "source_step": j,
                    "distance": distance,
                    "distance_bin": distance_bin(distance),
                    "target_gt_action": target_gt_action,
                    "source_gt_action": source_gt_action,
                    "source_wrong_action": wrong_action,
                    "inject_action": inject_action,
                    "target_gt_action_type": target_gt_action.get("action"),
                    "target_family": action_family(target_gt_action),
                    "source_wrong_action_type": wrong_action.get("action"),
                    "target_screenshot": episode["steps"][k].get("screenshot"),
                }
                pairs.append(row)
                by_distance[distance] += 1
                by_bin[row["distance_bin"]] += 1
                by_family[row["target_family"]][row["distance_bin"]] += 1
    manifest = {
        "stage": args.stage,
        "inject_mode": args.inject_mode,
        "jsonl_file": str(args.jsonl_file),
        "rollout_results": str(args.rollout_results),
        "pilot_episode_ids": str(args.pilot_episode_ids) if args.pilot_episode_ids else None,
        "distances": distances,
        "pairs": len(pairs),
        "skipped": dict(skipped),
        "by_distance": dict(sorted(by_distance.items())),
        "by_distance_bin": dict(by_bin),
        "by_target_family": {family: dict(counts) for family, counts in by_family.items()},
    }
    return pairs, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Condition C pairs")
    parser.add_argument("--jsonl-file", type=Path, required=True)
    parser.add_argument("--rollout-results", type=Path, required=True)
    parser.add_argument("--pilot-episode-ids", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["stage0", "stage1"], required=True)
    parser.add_argument("--inject-mode", choices=["gt", "wrong"], required=True)
    parser.add_argument("--distances", required=True, help="Comma/range list, e.g. 1,3,5 or 1-5,6-10")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs, manifest = build_pairs(args)
    count = write_jsonl(args.output, pairs)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pairs": count, "manifest": str(manifest_path), "by_distance": manifest["by_distance"], "by_distance_bin": manifest["by_distance_bin"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()