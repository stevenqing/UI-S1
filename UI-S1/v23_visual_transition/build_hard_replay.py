#!/usr/bin/env python3
"""Build weighted ShareGPT replay examples from mined hard states."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from v23_visual_transition.prepare_offline_data import (  # noqa: E402
    build_action_prompt,
    format_action_for_history,
    tool_call_text,
)


def load_episode_jsonl(path: str) -> Dict[str, Dict[str, Any]]:
    episodes: Dict[str, Dict[str, Any]] = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            episodes[str(episode.get("episode_id"))] = episode
    return episodes


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_history(steps: List[Dict[str, Any]], step_idx: int) -> List[str]:
    history: List[str] = []
    for idx in range(step_idx):
        history.append(format_action_for_history(steps[idx].get("action", {}) or {}, idx + 1))
    return history


def build_replay_rows(hard_rows: Iterable[Dict[str, Any]], episodes: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    replay_rows: List[Dict[str, Any]] = []
    missing = 0

    for hard in hard_rows:
        episode_id = str(hard.get("episode_id"))
        episode = episodes.get(episode_id)
        if not episode:
            missing += 1
            continue

        steps = episode.get("steps", [])
        step_idx = int(hard.get("step_idx", 0))
        if step_idx < 0 or step_idx >= len(steps):
            missing += 1
            continue

        step = steps[step_idx]
        action = step.get("action", {}) or {}
        is_last_step = step_idx == len(steps) - 1
        history = build_history(steps, step_idx)

        metadata = {
            "episode_id": episode_id,
            "step_idx": step_idx,
            "num_steps": len(steps),
            "is_first_error": bool(hard.get("is_first_error")),
            "failure_kind": hard.get("failure_kind"),
            "family": hard.get("family"),
            "depth_bin": hard.get("depth_bin"),
            "source_weight": hard.get("weight", 1.0),
            "pred_action": hard.get("pred_action"),
            "reward": hard.get("reward"),
        }

        replay_rows.append({
            "conversations": [
                {"from": "human", "value": build_action_prompt(episode.get("goal", ""), history)},
                {"from": "gpt", "value": tool_call_text(action, is_last_step)},
            ],
            "images": [step.get("screenshot")],
            "weight": float(hard.get("weight") or 1.0),
            "metadata": metadata,
        })

    if missing:
        print(f"[warn] skipped {missing} hard rows with missing episode/step", file=sys.stderr)
    return replay_rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weighted hard-state ShareGPT replay")
    parser.add_argument("--hard_states", required=True)
    parser.add_argument("--episode_data", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    hard_rows = load_jsonl(args.hard_states)
    episodes = load_episode_jsonl(args.episode_data)
    replay_rows = build_replay_rows(hard_rows, episodes)

    output_path = os.path.join(args.output_dir, "hard_action_replay.jsonl")
    n = write_jsonl(output_path, replay_rows)
    summary = {
        "num_hard_rows": len(hard_rows),
        "num_replay_rows": n,
        "episode_data": args.episode_data,
        "hard_states": args.hard_states,
        "output": output_path,
    }
    with open(os.path.join(args.output_dir, "hard_replay_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()