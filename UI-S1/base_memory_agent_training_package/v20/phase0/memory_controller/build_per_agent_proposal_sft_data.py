#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_proposal_sft_data import action_to_tool_call, split_episode  # noqa: E402
from v13_gui_360.eval_gui360_template import SUPPORTED_ACTIONS, _format_action_for_history  # noqa: E402


def load_episodes(path: Path) -> dict[str, dict[str, Any]]:
    episodes: dict[str, dict[str, Any]] = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                episode = json.loads(line)
                episodes[str(episode["episode_id"])] = episode
    return episodes


def load_run(root: Path) -> dict[str, dict[str, Any]]:
    files = sorted(root.glob("shard_*/eval_results_*.json"))
    if not files:
        raise FileNotFoundError(f"No eval_results files under {root}")
    episodes: dict[str, dict[str, Any]] = {}
    for file in files:
        with file.open() as handle:
            episodes.update(json.load(handle))
    return episodes


def build_history(episode: dict[str, Any], step_idx: int) -> str:
    history: list[str] = []
    for prior_idx, step in enumerate(episode.get("steps", [])[:step_idx], start=1):
        history.append(_format_action_for_history(step.get("action"), prior_idx))
    return "\n".join(history) if history else "None"


def prompt_for_agent(agent_name: str, episode: dict[str, Any], step_idx: int, step: dict[str, Any]) -> str:
    gt_type = step.get("gt_type") or (episode.get("steps", [{}])[step_idx].get("action") or {}).get("action", "unknown")
    mode_text = {
        "type_recovery": "focus on text-entry, field focus, keyboard input, and type-action alternatives",
        "click_recovery": "focus on alternative grounded click targets and nearby-target recovery",
        "swipe_navigation": "focus on scroll, drag, navigation, and actions that reveal hidden targets",
        "minimal_next_step": "focus on the simplest valid next action that advances the task",
        "escape_finish": "focus on modal escape, finish guards, and avoiding premature FINISH",
        "spreadsheet_formula": "focus on spreadsheet cells, formulas, table editing, and office-document structure",
        "non_alpha": "focus on diverse non-base alternatives that can improve a candidate pool",
    }.get(agent_name, f"focus on specialized proposal behavior for {agent_name}")
    return "\n".join([
        "You are a trainable memory-specialized proposal agent for GUI control.",
        "Your goal is to generate one candidate action for a multi-agent candidate pool.",
        "Do not imitate a generic base policy when a useful specialized alternative exists.",
        f"Agent specialization: {agent_name} ({mode_text}).",
        f"Task: {episode.get('goal', '')}",
        f"Step: {step_idx + 1} / {episode.get('num_steps', len(episode.get('steps', [])))}",
        f"Training diagnostic GT action type: {gt_type}",
        "History:",
        build_history(episode, step_idx),
        "Supported actions:",
        SUPPORTED_ACTIONS,
        "Return exactly one action in <tool_call></tool_call> format.",
    ])


def make_record(agent_name: str, episode_id: str, episode: dict[str, Any], step_idx: int, step: dict[str, Any], split: str, fmt: str) -> dict[str, Any] | None:
    action = step.get("pred_action")
    if not isinstance(action, dict):
        return None
    screenshot = None
    if step_idx < len(episode.get("steps", [])):
        screenshot = episode["steps"][step_idx].get("screenshot")
    user_text = prompt_for_agent(agent_name, episode, step_idx, step)
    assistant_text = action_to_tool_call(action)
    base = {
        "id": f"{episode_id}:{step_idx}:{agent_name}",
        "split": split,
        "source": agent_name,
        "reward": float(step.get("reward", 0.0) or 0.0),
        "pred_type": step.get("pred_type") or "unknown",
        "gt_type": step.get("gt_type") or "unknown",
    }
    if fmt == "sharegpt":
        base.update({
            "conversations": [
                {"from": "human", "value": user_text},
                {"from": "gpt", "value": assistant_text},
            ],
            "images": [screenshot] if screenshot else [],
        })
    elif fmt == "chat":
        base.update({
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
        })
    else:
        base.update({"prompt": user_text, "completion": assistant_text})
    return base


def balanced_sample(records: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(records) <= limit:
        return records
    rng = random.Random(seed)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_type[str(record.get("pred_type") or "unknown")].append(record)
    for values in by_type.values():
        values.sort(key=lambda item: float(item.get("reward", 0.0) or 0.0), reverse=True)
    selected: list[dict[str, Any]] = []
    type_order = sorted(by_type, key=lambda key: len(by_type[key]))
    cursor = {key: 0 for key in by_type}
    while len(selected) < limit and type_order:
        progressed = False
        for action_type in list(type_order):
            idx = cursor[action_type]
            values = by_type[action_type]
            if idx >= len(values):
                type_order.remove(action_type)
                continue
            selected.append(values[idx])
            cursor[action_type] += 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    rng.shuffle(selected)
    return selected[:limit]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--run", action="append", required=True, help="agent_name=eval_result_root")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_reward", type=float, default=0.5)
    parser.add_argument("--max_train_per_agent", type=int, default=2000)
    parser.add_argument("--max_dev_per_agent", type=int, default=300)
    parser.add_argument("--format", choices=["sharegpt", "chat", "completion"], default="sharegpt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    episodes = load_episodes(Path(args.test_data))
    output_dir = Path(args.output_dir)
    summary: dict[str, Any] = {"agents": {}, "min_reward": args.min_reward, "format": args.format}
    for run_spec in args.run:
        if "=" not in run_spec:
            raise ValueError(f"run must be agent_name=path, got {run_spec}")
        agent_name, root_text = run_spec.split("=", 1)
        run = load_run(Path(root_text))
        split_records: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
        for episode_id, run_episode in run.items():
            episode = episodes.get(str(episode_id))
            if episode is None:
                continue
            split = split_episode(str(episode_id))
            for step in run_episode.get("steps", []):
                reward = float(step.get("reward", 0.0) or 0.0)
                if reward < args.min_reward:
                    continue
                step_idx = int(step.get("step_idx", 0))
                record = make_record(agent_name, str(episode_id), episode, step_idx, step, split, args.format)
                if record is not None:
                    split_records[split].append(record)
        train_records = balanced_sample(split_records["train"], args.max_train_per_agent, args.seed)
        dev_records = balanced_sample(split_records["dev"], args.max_dev_per_agent, args.seed + 1)
        write_jsonl(output_dir / f"{agent_name}_train_sharegpt.jsonl", train_records)
        write_jsonl(output_dir / f"{agent_name}_dev_sharegpt.jsonl", dev_records)
        summary["agents"][agent_name] = {
            "available": {split: len(records) for split, records in split_records.items()},
            "written": {"train": len(train_records), "dev": len(dev_records)},
            "train_types": dict(Counter(record.get("pred_type", "unknown") for record in train_records)),
            "dev_types": dict(Counter(record.get("pred_type", "unknown") for record in dev_records)),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
