#!/usr/bin/env python3
"""Run text-space error-horizon probes on GUI-Odyssey.

This runner keeps the current screen teacher-forced to the GT step screenshot and
varies only the textual action history. It is intentionally offline: no emulator
or environment stepping is used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
sys.path.insert(0, str(PROJECT_ROOT / "gui_odyssey_eval"))

import evaluation.qwenvl_utils as qwen_utils  # noqa: E402
from gui_odyssey_eval.eval_ar_trajectory import safe_parse_response  # noqa: E402
from gui_odyssey_eval.odyssey_action_matching import evaluate_odyssey_action  # noqa: E402
from x.data.agent.json import JsonFormat  # noqa: E402
from x.data.agent.space.std_space import RAW_SPACE  # noqa: E402
from x.qwen.data_format import slim_messages  # noqa: E402


JsonDict = dict[str, Any]


def iter_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_episodes(path: Path) -> list[JsonDict]:
    return list(iter_jsonl(path))


def load_episode_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                if line.startswith("{"):
                    ids.add(str(json.loads(line)["episode_id"]))
                elif line.startswith('"'):
                    ids.add(str(json.loads(line)))
                else:
                    ids.add(line)
    return ids


def action_response(action: JsonDict | None) -> str:
    if not isinstance(action, dict):
        action = {"action": "invalid_or_unparseable"}
    return "<action>\n" + json.dumps(action, ensure_ascii=False) + "\n</action>"


def fallback_action(raw_output: str) -> JsonDict:
    return {
        "action": "invalid_or_unparseable",
        "raw_response_prefix": str(raw_output or "")[:240],
    }


def action_family(gt_action: JsonDict) -> str:
    action = str((gt_action or {}).get("action", "unknown"))
    if action in {"system_button", "open", "terminate"}:
        return "navigation_phase"
    if action == "click":
        return "click_grounding"
    if action in {"type", "answer", "key"}:
        return "text_value_carry"
    if action == "swipe":
        return "swipe_scroll"
    return action


def call_and_eval(
    fm: JsonFormat,
    episode: JsonDict,
    state: JsonDict,
    step_index: int,
    model_name: str,
    n_history_image_limit: int,
) -> JsonDict:
    messages = slim_messages(messages=state["messages"], num_image_limit=n_history_image_limit)
    _, _width, _height, resized_width, resized_height = qwen_utils.find_last_image_ele(messages)
    raw_output = qwen_utils.call_mobile_agent_vllm(messages=messages, model_name=model_name)
    parse_ok = True
    parse_error = ""
    pred_action = None
    type_match = False
    value_match = False
    try:
        pred = safe_parse_response(fm, raw_output)
        pred_action = pred["action_content"]
        type_match, value_match = evaluate_odyssey_action(
            pred_action,
            episode["steps"][step_index]["check_options"],
            resized_width,
            resized_height,
        )
    except Exception as exc:
        parse_ok = False
        parse_error = repr(exc)
    return {
        "raw_response": raw_output,
        "pred_action": pred_action,
        "parse_ok": bool(parse_ok),
        "parse_error": parse_error,
        "type_match": bool(type_match),
        "value_match": bool(value_match),
        "resized_width": resized_width,
        "resized_height": resized_height,
    }


def process_episode(episode: JsonDict, args: argparse.Namespace) -> list[JsonDict]:
    fm = JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True)
    state_a = None
    state_b = None
    prev_a_response = None
    prev_b_response = None
    prefix_error_count = 0
    last_error_step: int | None = None
    rows = []
    num_steps = len(episode.get("steps", []) or [])
    for step_index in range(num_steps):
        state_a = fm.gen_next_round(episode, state_a, previous_model_response=prev_a_response)
        state_b = fm.gen_next_round(episode, state_b, previous_model_response=prev_b_response)
        if state_a is None or state_b is None:
            break

        nearest_error_distance = None if last_error_step is None else step_index - last_error_step
        gt_action = episode["steps"][step_index]["action_content"]
        result_a = call_and_eval(fm, episode, state_a, step_index, args.model_name, args.n_history_image_limit)
        result_b = call_and_eval(fm, episode, state_b, step_index, args.model_name, args.n_history_image_limit)

        rows.append(
            {
                "episode_id": str(episode.get("episode_id")),
                "category": episode.get("category", ""),
                "device_name": episode.get("device_name", ""),
                "step_index": step_index,
                "num_steps": num_steps,
                "absolute_depth": step_index,
                "normalized_depth": step_index / num_steps if num_steps else 0.0,
                "prefix_error_count": prefix_error_count,
                "nearest_error_distance": nearest_error_distance,
                "gt_action": gt_action,
                "gt_action_type": gt_action.get("action"),
                "action_family": action_family(gt_action),
                "screenshot": episode["steps"][step_index].get("screenshot"),
                "condition_a": result_a,
                "condition_b": result_b,
                "gap_value_match": int(bool(result_b["value_match"])) - int(bool(result_a["value_match"])),
                "gap_type_match": int(bool(result_b["type_match"])) - int(bool(result_a["type_match"])),
                "history_policy": "action_only_oracle_corrected",
            }
        )

        a_history_action = result_a["pred_action"] if result_a["parse_ok"] else fallback_action(result_a["raw_response"])
        a_wrong = not bool(result_a["value_match"])
        b_history_action = gt_action if a_wrong else a_history_action
        prev_a_response = action_response(a_history_action)
        prev_b_response = action_response(b_history_action)
        if a_wrong:
            prefix_error_count += 1
            last_error_step = step_index
    return rows


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run text-space error-horizon probe")
    parser.add_argument("--jsonl-file", type=Path, required=True)
    parser.add_argument("--episode-ids", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="qwen3.5-9b")
    parser.add_argument("--endpoint", default=os.environ.get("QWENVL_ENDPOINT", "http://localhost:8000/v1"))
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--n-history-image-limit", type=int, default=2)
    parser.add_argument("--limit-episodes", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Keep existing output rows and skip completed episode IDs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["QWENVL_ENDPOINT"] = args.endpoint
    qwen_utils.END_POINT = args.endpoint
    episodes = load_episodes(args.jsonl_file)
    episode_ids = load_episode_ids(args.episode_ids)
    if episode_ids is not None:
        episodes = [episode for episode in episodes if str(episode.get("episode_id")) in episode_ids]
    if args.limit_episodes > 0:
        episodes = episodes[: args.limit_episodes]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed_ids = set()
    if args.resume and args.output.exists():
        for row in iter_jsonl(args.output):
            completed_ids.add(str(row.get("episode_id")))
        before = len(episodes)
        episodes = [episode for episode in episodes if str(episode.get("episode_id")) not in completed_ids]
        print(f"resume: skipped_completed_episodes={before - len(episodes)} remaining={len(episodes)}")
    elif args.output.exists():
        args.output.unlink()

    completed = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_episode, episode, args): episode for episode in episodes}
        for future in as_completed(futures):
            episode = futures[future]
            rows = future.result()
            write_jsonl(args.output, rows)
            completed += 1
            if completed % 10 == 0:
                print(f"completed_episodes={completed}/{len(episodes)} rows_written_for_last={len(rows)}")
            if not rows:
                print(f"warning: no rows for episode {episode.get('episode_id')}", file=sys.stderr)

    manifest = {
        "jsonl_file": str(args.jsonl_file),
        "episode_ids": str(args.episode_ids) if args.episode_ids else None,
        "episodes": len(episodes),
        "output": str(args.output),
        "model_name": args.model_name,
        "endpoint": args.endpoint,
        "resume": args.resume,
        "skipped_completed_episodes": len(completed_ids),
        "history_policy": "action_only_oracle_corrected",
        "condition_a": "self predicted action-only history generated in this run",
        "condition_b": "oracle-corrected action-only history: keep A action when A is matcher-correct, replace with GT action when A is wrong",
        "screen_policy": "GT screenshot at each step",
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()