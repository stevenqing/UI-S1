#!/usr/bin/env python3
"""Run row-level GUI-Odyssey baseline-template evaluation through vLLM.

The prompt is the existing GUI-Odyssey baseline format used by
gui_odyssey_eval/eval_ar_trajectory.py: JsonFormat(RAW_SPACE,
add_thought=True, force_add_thought=True). Unlike the AR evaluator, this script
can evaluate an arbitrary list of row keys and uses ground-truth previous
actions as the history context for single-step comparability.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))
sys.path.insert(0, str(PROJECT_ROOT / "gui_odyssey_eval"))

from evaluation.qwenvl_utils import find_last_image_ele, image_to_data_url, message_translate  # noqa: E402
from gui_odyssey_eval.odyssey_action_matching import evaluate_odyssey_action, pred_coord_to_1k  # noqa: E402
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


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_episodes(path: Path) -> list[JsonDict]:
    return list(iter_jsonl(path))


def load_keys(path: Path | None) -> set[tuple[str, int]] | None:
    if path is None:
        return None
    keys: set[tuple[str, int]] = set()
    for row in iter_jsonl(path):
        episode_id = row.get("episode_id")
        step_index = row.get("step_index")
        if episode_id is None or step_index is None:
            metadata = row.get("metadata") or {}
            episode_id = metadata.get("episode_id", episode_id)
            step_index = metadata.get("step_index", step_index)
        if episode_id is None or step_index is None:
            continue
        keys.add((str(episode_id), int(step_index)))
    return keys


def selected_rows(episodes: list[JsonDict], keys: set[tuple[str, int]] | None, all_steps: bool, limit: int) -> list[tuple[JsonDict, int]]:
    rows = []
    for episode in episodes:
        episode_id = str(episode.get("episode_id"))
        for step_index in range(len(episode.get("steps", []))):
            if keys is not None and (episode_id, step_index) not in keys:
                continue
            if keys is None and not all_steps:
                continue
            rows.append((episode, step_index))
            if limit > 0 and len(rows) >= limit:
                return rows
    return rows


def safe_parse_response(formatter: JsonFormat, model_response: str) -> JsonDict | None:
    try:
        return formatter.parse_response(model_response)
    except Exception:
        match = re.search(r"<action>\s*(\{.*?\})\s*</action>", model_response, re.DOTALL)
        if not match:
            match = re.search(r"(\{[^{}]*\})", model_response, re.DOTALL)
        if not match:
            return None
        action_text = match.group(1)
        try:
            action_content = json.loads(action_text)
        except json.JSONDecodeError:
            return None
        think_match = re.search(r"<think>(.*?)</think>", model_response, re.DOTALL)
        return {
            "think": think_match.group(1).strip() if think_match else "",
            "action": action_text,
            "action_content": action_content,
        }


def build_gt_history_state(formatter: JsonFormat, episode: JsonDict, target_step: int, image_limit: int) -> JsonDict:
    state = None
    previous_response = None
    for step_index in range(target_step + 1):
        state = formatter.gen_next_round(episode, state, previous_model_response=previous_response)
        if state is None:
            raise ValueError(f"failed to build state for step {target_step}")
        if step_index < target_step:
            image_ele = state.get("screenshot_ele")
            previous_response = formatter.format_response(
                episode["steps"][step_index],
                image_ele,
                add_thought=bool(state.get("line_can_thought", True)),
            )
    state = copy.deepcopy(state)
    state["messages"] = slim_messages(state["messages"], num_image_limit=image_limit)
    return state


def call_openai(client: OpenAI, model_name: str, messages: list[JsonDict], max_tokens: int, temperature: float) -> str:
    openai_messages, screenshot_list = message_translate(messages, to_format="openai")
    screenshot_ptr = 0
    for message in openai_messages:
        for content in message["content"]:
            if "image_url" in content:
                with Image.open(screenshot_list[screenshot_ptr]) as image:
                    content["image_url"]["url"] = image_to_data_url(image)
                screenshot_ptr += 1
    if screenshot_ptr != len(screenshot_list):
        raise RuntimeError(f"image translation mismatch: {screenshot_ptr} != {len(screenshot_list)}")
    response = client.chat.completions.create(
        model=model_name,
        messages=openai_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={"top_k": 1},
    )
    return response.choices[0].message.content or ""


def evaluate_one(
    item: tuple[JsonDict, int],
    formatter: JsonFormat,
    client: OpenAI,
    model_name: str,
    image_limit: int,
    max_tokens: int,
    temperature: float,
) -> JsonDict:
    episode, step_index = item
    step = episode["steps"][step_index]
    episode_id = str(episode.get("episode_id"))
    state = build_gt_history_state(formatter, episode, step_index, image_limit)
    _, width, height, resized_width, resized_height = find_last_image_ele(state["messages"])
    error = ""
    raw_output = ""
    pred_action = None
    parse_ok = False
    type_match = False
    value_match = False
    pred_coord_1k = None
    try:
        raw_output = call_openai(client, model_name, state["messages"], max_tokens, temperature)
        parsed = safe_parse_response(formatter, raw_output)
        if parsed and parsed.get("action_content") is not None:
            parse_ok = True
            pred_action = parsed["action_content"]
            type_match, value_match = evaluate_odyssey_action(
                pred_action,
                step["check_options"],
                resized_width,
                resized_height,
            )
            coord = pred_action.get("coordinate") if isinstance(pred_action, dict) else None
            if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                pred_coord_1k = pred_coord_to_1k([float(coord[0]), float(coord[1])], resized_width, resized_height)
    except Exception as exc:
        error = repr(exc)
    return {
        "episode_id": episode_id,
        "step_index": step_index,
        "category": episode.get("category", ""),
        "device_name": episode.get("device_name", ""),
        "goal": episode.get("goal", ""),
        "screenshot": step.get("screenshot"),
        "prompt_template": "JsonFormat_RAW_SPACE_thought_action_gt_history",
        "model_name": model_name,
        "gt_action": step.get("action_content"),
        "pred_action": pred_action,
        "raw_output": raw_output,
        "error": error,
        "parse_ok": parse_ok,
        "type_match": bool(type_match),
        "value_match": bool(value_match),
        "image_size": [width, height],
        "resized_size": [resized_width, resized_height],
        "pred_coord_1k": pred_coord_1k,
        "gt_coord_1k": step.get("check_options", {}).get("coordinate"),
    }


def summarize(rows: list[JsonDict]) -> JsonDict:
    total = len(rows)
    parse_ok = sum(bool(row.get("parse_ok")) for row in rows)
    type_ok = sum(bool(row.get("type_match")) for row in rows)
    value_ok = sum(bool(row.get("value_match")) for row in rows)
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    by_episode: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        gt_type = str((row.get("gt_action") or {}).get("action", "unknown"))
        by_type[gt_type]["rows"] += 1
        by_type[gt_type]["type_match"] += int(bool(row.get("type_match")))
        by_type[gt_type]["value_match"] += int(bool(row.get("value_match")))
        by_episode[str(row.get("episode_id"))].append(bool(row.get("value_match")))
    type_stats = {}
    for action_type, counts in sorted(by_type.items()):
        rows_count = counts["rows"]
        type_stats[action_type] = {
            "rows": rows_count,
            "type_match": counts["type_match"],
            "value_match": counts["value_match"],
            "type_match_rate": counts["type_match"] / rows_count if rows_count else 0.0,
            "value_match_rate": counts["value_match"] / rows_count if rows_count else 0.0,
        }
    episode_success = sum(all(values) for values in by_episode.values())
    return {
        "rows": total,
        "parse_ok": parse_ok,
        "parse_rate": parse_ok / total if total else 0.0,
        "type_match": type_ok,
        "type_match_rate": type_ok / total if total else 0.0,
        "semantic_match": value_ok,
        "semantic_match_rate": value_ok / total if total else 0.0,
        "episodes": len(by_episode),
        "strict_episode_success": episode_success,
        "strict_episode_success_rate": episode_success / len(by_episode) if by_episode else 0.0,
        "action_type_stats": type_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GUI-Odyssey rows with the baseline RAW_SPACE prompt template")
    parser.add_argument("--jsonl-file", required=True, type=Path)
    parser.add_argument("--sample-keys", type=Path, default=None, help="JSONL with episode_id and step_index fields")
    parser.add_argument("--all-steps", action="store_true", help="Evaluate every step if --sample-keys is not provided")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--api-url", default="http://localhost:8000/v1")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--n-history-image-limit", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes = load_episodes(args.jsonl_file)
    keys = load_keys(args.sample_keys)
    work = selected_rows(episodes, keys, args.all_steps, args.limit)
    if not work:
        raise SystemExit("no rows selected; pass --sample-keys or --all-steps")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    formatter = JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True)
    client = OpenAI(base_url=args.api_url, api_key="EMPTY", timeout=600)
    rows: list[JsonDict] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.threads)) as executor:
        futures = [
            executor.submit(
                evaluate_one,
                item,
                formatter,
                client,
                args.model_name,
                args.n_history_image_limit,
                args.max_tokens,
                args.temperature,
            )
            for item in work
        ]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            if index % 50 == 0 or index == len(futures):
                summary = summarize(rows)
                print(
                    f"progress {index}/{len(futures)} "
                    f"semantic={summary['semantic_match_rate']:.4f} "
                    f"type={summary['type_match_rate']:.4f}"
                )
    rows.sort(key=lambda row: (str(row.get("episode_id")), int(row.get("step_index", -1))))
    result_path = args.output_dir / "baseline_template_predictions.jsonl"
    write_jsonl(result_path, rows)
    summary = summarize(rows)
    summary.update(
        {
            "jsonl_file": str(args.jsonl_file),
            "sample_keys": str(args.sample_keys) if args.sample_keys else None,
            "model_name": args.model_name,
            "api_url": args.api_url,
            "prompt_template": "JsonFormat_RAW_SPACE_thought_action_gt_history",
            "elapsed_seconds": time.time() - started,
            "predictions": str(result_path),
        }
    )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()