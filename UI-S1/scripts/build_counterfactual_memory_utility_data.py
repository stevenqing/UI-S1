#!/usr/bin/env python3
"""Build minimal counterfactual memory utility data from behavior results."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]
CONDITIONS = ["no_history", "segment_summary", "full_history", "wrong_summary"]


def iter_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_episodes(paths: list[Path]) -> dict[str, JsonDict]:
    episodes = {}
    for path in paths:
        for episode in iter_jsonl(path):
            episodes[str(episode.get("episode_id"))] = episode
    return episodes


def segment_for_step(episode: JsonDict | None, step_index: int) -> JsonDict | None:
    if not episode:
        return None
    for segment in episode.get("segments", []):
        if int(segment.get("start_step", 0)) <= step_index <= int(segment.get("end_step", -1)):
            return segment
    return None


def previous_segments(episode: JsonDict | None, step_index: int) -> list[JsonDict]:
    if not episode:
        return []
    return [segment for segment in episode.get("segments", []) if int(segment.get("end_step", -1)) < step_index]


def segment_memory_text(segments: list[JsonDict], max_segments: int = 4) -> str:
    if not segments:
        return "No previous segment memory."
    lines = []
    for segment in segments[-max_segments:]:
        memory = segment.get("memory_need", {}) or {}
        carried = segment.get("carried_values", []) or []
        lines.append(
            f"Segment {segment.get('segment_id')} steps {segment.get('start_step')}-{segment.get('end_step')}: "
            f"{segment.get('summary', '')}; capability={segment.get('dominant_capability', 'unknown')}; "
            f"memory={memory.get('strength', 'unknown')}; carried_values={carried}"
        )
    return "\n".join(lines)


def all_segment_memories(episodes: dict[str, JsonDict]) -> list[tuple[str, str]]:
    pool = []
    for episode_id, episode in episodes.items():
        for segment in episode.get("segments", []):
            text = segment_memory_text([segment], max_segments=1)
            if text:
                pool.append((episode_id, text))
    return pool


def choose_wrong_memory(pool: list[tuple[str, str]], episode_id: str, case_id: int, thinking_mode: str) -> str:
    if not pool:
        return "No unrelated memory available."
    digest = hashlib.md5(f"{episode_id}|{case_id}|{thinking_mode}".encode("utf-8")).hexdigest()
    start = int(digest[:8], 16) % len(pool)
    for offset in range(len(pool)):
        candidate_episode, text = pool[(start + offset) % len(pool)]
        if candidate_episode != episode_id:
            return text
    return pool[start][1]


def step_text(episode: JsonDict | None, step_index: int) -> JsonDict:
    if not episode or step_index >= len(episode.get("steps", [])):
        return {"instruction": "", "thought": "", "observation": ""}
    step = episode["steps"][step_index]
    fields = step.get("text_fields", {}) or {}
    return {
        "instruction": fields.get("instruction", "") or step.get("instruction", ""),
        "thought": fields.get("thought", "") or step.get("thought", ""),
        "observation": fields.get("observation", ""),
    }


def current_state_parts(episode: JsonDict | None, step_index: int) -> JsonDict:
    fields = step_text(episode, step_index)
    current_segment = segment_for_step(episode, step_index)
    parts = {
        "goal": episode.get("task_goal", "") if episode else "",
        "instruction": fields["instruction"],
        "observation": fields["observation"],
        "local_hint": fields["thought"][:300] if fields["thought"] else "",
        "current_segment": "",
        "current_segment_capability": "unknown",
    }
    if current_segment:
        parts["current_segment"] = str(current_segment.get("summary", ""))
        parts["current_segment_capability"] = str(current_segment.get("dominant_capability", "unknown"))
    return parts


def current_state_text(episode: JsonDict | None, step_index: int) -> str:
    state_parts = current_state_parts(episode, step_index)
    parts = []
    if episode:
        parts.append(f"Goal: {state_parts['goal']}")
    if state_parts["instruction"]:
        parts.append(f"Current instruction: {state_parts['instruction']}")
    if state_parts["observation"]:
        parts.append(f"Current screen observation: {state_parts['observation']}")
    if state_parts["local_hint"]:
        parts.append(f"Local hint: {state_parts['local_hint']}")
    if state_parts["current_segment"]:
        parts.append(
            f"Current segment hypothesis: {state_parts['current_segment']}; "
            f"capability={state_parts['current_segment_capability']}"
        )
    return "\n".join(parts)


def long_features(episode: JsonDict | None, step_index: int) -> JsonDict:
    prev = previous_segments(episode, step_index)
    current_segment = segment_for_step(episode, step_index)
    carried = current_segment.get("carried_values", []) if current_segment else []
    memory = current_segment.get("memory_need", {}) if current_segment else {}
    total_steps = len(episode.get("steps", [])) if episode else None
    return {
        "step_index": step_index,
        "total_steps": total_steps,
        "prev_segments": len(prev),
        "segment_len_so_far": step_index - int(current_segment.get("start_step", step_index)) + 1 if current_segment else None,
        "carried_values": carried or [],
        "carried_value_count": len(carried or []),
        "memory_strength": memory.get("strength", "unknown") if memory else "unknown",
        "dominant_capability": current_segment.get("dominant_capability", "unknown") if current_segment else "unknown",
    }


def group_results(paths: list[Path]) -> dict[tuple[str, str, str, int], dict[str, JsonDict]]:
    grouped: dict[tuple[str, str, str, int], dict[str, JsonDict]] = defaultdict(dict)
    for path in paths:
        for row in iter_jsonl(path):
            key = (row.get("model_key", "unknown"), row.get("thinking_mode", "unknown"), row.get("case_kind", "unknown"), int(row.get("case_id", -1)))
            grouped[key][row.get("condition")] = row
    return {key: value for key, value in grouped.items() if all(condition in value for condition in CONDITIONS)}


def ok(row: JsonDict) -> bool:
    return bool(row.get("value_match")) and not row.get("error")


def utility_label(rows_by_condition: dict[str, JsonDict]) -> str:
    no_ok = ok(rows_by_condition["no_history"])
    seg_ok = ok(rows_by_condition["segment_summary"])
    full_ok = ok(rows_by_condition["full_history"])
    wrong_ok = ok(rows_by_condition["wrong_summary"])
    if (not no_ok) and seg_ok and (not wrong_ok):
        return "positive"
    if no_ok and (not seg_ok):
        return "negative"
    if (not no_ok) and (not seg_ok) and full_ok:
        return "summary_insufficient"
    if not (no_ok or seg_ok or full_ok or wrong_ok):
        return "unresolved"
    if no_ok:
        return "neutral"
    if seg_ok:
        return "nonspecific_positive"
    return "unresolved"


def build_rows(results: list[Path], episodes: dict[str, JsonDict], wrong_pool: list[tuple[str, str]]) -> list[JsonDict]:
    rows = []
    grouped = group_results(results)
    for (model_key, thinking_mode, case_kind, case_id), by_condition in grouped.items():
        base = by_condition["no_history"]
        episode_id = str(base.get("episode_id"))
        step_index = int(base.get("step_index", 0))
        episode = episodes.get(episode_id)
        prev = previous_segments(episode, step_index)
        true_memory = segment_memory_text(prev)
        wrong_memory = choose_wrong_memory(wrong_pool, episode_id, case_id, thinking_mode)
        label = utility_label(by_condition)
        condition_value_match = {condition: ok(by_condition[condition]) for condition in CONDITIONS}
        current_parts = current_state_parts(episode, step_index)
        current = current_state_text(episode, step_index)
        features = long_features(episode, step_index)
        features.update({
            "episode_id": episode_id,
            "case_id": case_id,
            "case_kind": case_kind,
            "model_key": model_key,
            "thinking_mode": thinking_mode,
            "gt_action": base.get("gt_action"),
            "gt_action_type": (base.get("gt_action", {}) or {}).get("action", "unknown"),
            "screenshot": base.get("screenshot", ""),
        })
        pairs = []
        if label == "positive":
            pairs.append(["true_memory", "wrong_memory"])
            pairs.append(["true_memory", "no_memory"])
        elif label == "negative":
            pairs.append(["no_memory", "true_memory"])
        elif label == "summary_insufficient":
            pairs.append(["full_history", "true_memory"])
        rows.append({
            "current_state_text": current,
            "current_state_parts": current_parts,
            "true_memory_text": true_memory,
            "wrong_memory_text": wrong_memory,
            "utility_label": label,
            "preference_pairs": pairs,
            "condition_value_match": condition_value_match,
            "condition_type_match": {condition: bool(by_condition[condition].get("type_match")) for condition in CONDITIONS},
            "pred_actions": {condition: by_condition[condition].get("pred_action") for condition in CONDITIONS},
            "metadata": features,
        })
    return rows


def split_rows(rows: list[JsonDict], seed: int) -> dict[str, list[JsonDict]]:
    episode_ids = sorted({row["metadata"]["episode_id"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(episode_ids)
    train_end = int(0.8 * len(episode_ids))
    dev_end = int(0.9 * len(episode_ids))
    split = {}
    for episode_id in episode_ids[:train_end]:
        split[episode_id] = "train"
    for episode_id in episode_ids[train_end:dev_end]:
        split[episode_id] = "dev"
    for episode_id in episode_ids[dev_end:]:
        split[episode_id] = "test"
    out = {"train": [], "dev": [], "test": []}
    for row in rows:
        out[split[row["metadata"]["episode_id"]]].append(row)
    return out


def write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build counterfactual memory utility triplets")
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--episodes", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes([Path(path) for path in args.episodes])
    wrong_pool = all_segment_memories(episodes)
    rows = build_rows([Path(path) for path in args.results], episodes, wrong_pool)
    splits = split_rows(rows, args.seed)
    for split, split_rows_ in splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", split_rows_)
    write_jsonl(output_dir / "all.jsonl", rows)
    stats = {
        "rows": len(rows),
        "splits": {name: len(value) for name, value in splits.items()},
        "labels": {},
    }
    for row in rows:
        stats["labels"][row["utility_label"]] = stats["labels"].get(row["utility_label"], 0) + 1
    (output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()