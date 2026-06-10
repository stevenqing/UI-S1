#!/usr/bin/env python3
"""Evaluate HAR-GUI on GUI-Odyssey with HAR native prompts.

This script is intended for checking the public HAR-GUI-3B-GUI-Odyssey
checkpoint against the GUI-Odyssey paper setting. It uses HAR's native
<think>/<answer> prompt plus Act2Sum history, then scores each predicted
action with the local GUI-Odyssey matcher.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "related_work" / "har"))
sys.path.insert(0, str(WORKSPACE_ROOT / "related_work" / "har" / "Prompts"))
sys.path.insert(0, str(WORKSPACE_ROOT / "gui_odyssey_eval"))

from Prompts.Act2Sum import ODYSSEY_ACT2SUM_PROMPT  # noqa: E402
from Prompts.Inference import ODYSSEY_EXECUTION_PROMPT  # noqa: E402
from gui_odyssey_eval.odyssey_action_matching import evaluate_odyssey_action  # noqa: E402
from related_work.har.action_parser import extract_summary, parse_har_output  # noqa: E402


PAPER_GUI_ODYSSEY_OVERALL = 62.31
CATEGORY_TO_PAPER_COLUMN = {
    "General_Tool": "Tool",
    "Information_Management": "Info.",
    "Web_Shopping": "Shop.",
    "Media_Entertainment": "Media",
    "Social_Sharing": "Social",
    "Multi_Apps": "M.Apps",
}


@dataclass
class EncodedImage:
    b64: str
    width: int
    height: int


@dataclass
class ChatResult:
    text: str
    finish_reason: str

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


def encode_image(path: str, max_pixels: Optional[int]) -> EncodedImage:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    if max_pixels and width * height > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
        width = max(1, int(width * scale))
        height = max(1, int(height * scale))
        image = image.resize((width, height), Image.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return EncodedImage(base64.b64encode(buffer.getvalue()).decode("utf-8"), width, height)


def call_chat(
    client: OpenAI,
    model_name: str,
    image: EncodedImage,
    prompt: str,
    max_tokens: int,
) -> ChatResult:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image.b64}"}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    choice = response.choices[0]
    return ChatResult(choice.message.content or "", choice.finish_reason or "")


def action_to_text(action: Optional[Dict[str, Any]], fallback: str = "") -> str:
    if fallback:
        return fallback
    if not action:
        return "IMPOSSIBLE"
    kind = action.get("action", "")
    if kind in {"click", "long_press"}:
        coord = action.get("coordinate", [0, 0])
        name = "CLICK" if kind == "click" else "LONG_PRESS"
        return f"{name}:({int(coord[0])},{int(coord[1])})"
    if kind == "type":
        return f'TYPE:"{action.get("text", "")}"'
    if kind == "swipe":
        c1 = action.get("coordinate", [0, 0])
        c2 = action.get("coordinate2", [0, 0])
        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]
        if abs(dx) > abs(dy):
            direction = "RIGHT" if dx > 0 else "LEFT"
        else:
            direction = "DOWN" if dy > 0 else "UP"
        return f"SCROLL:{direction}"
    if kind == "system_button":
        return action.get("button", "").upper()
    if kind == "terminate":
        return "COMPLETE"
    return json.dumps(action, ensure_ascii=False)


def scroll_to_action(direction: str, width: int, height: int) -> Dict[str, Any]:
    x_mid = width / 2
    y_mid = height / 2
    x_left = width * 0.3
    x_right = width * 0.7
    y_top = height * 0.3
    y_bottom = height * 0.7
    direction = direction.upper()
    if direction == "UP":
        start, end = [x_mid, y_bottom], [x_mid, y_top]
    elif direction == "DOWN":
        start, end = [x_mid, y_top], [x_mid, y_bottom]
    elif direction == "LEFT":
        start, end = [x_right, y_mid], [x_left, y_mid]
    else:
        start, end = [x_left, y_mid], [x_right, y_mid]
    return {
        "action": "swipe",
        "coordinate": [int(start[0]), int(start[1])],
        "coordinate2": [int(end[0]), int(end[1])],
    }


def normalize_har_action(answer: str, parsed: Optional[Dict[str, Any]], width: int, height: int) -> Optional[Dict[str, Any]]:
    text = answer.strip()
    upper = text.upper()

    if parsed:
        action = dict(parsed)
        if action.get("action") == "swipe" and "endCoordinate" in action:
            action["coordinate2"] = action.pop("endCoordinate")
        return action

    if upper in {"COMPLETE"}:
        return {"action": "terminate", "status": "success"}
    if upper in {"IMPOSSIBLE"}:
        return {"action": "terminate", "status": "impossible"}
    if upper in {"BACK", "HOME", "PRESS_RECENT"}:
        button = {"BACK": "Back", "HOME": "Home", "PRESS_RECENT": "Recent"}[upper]
        return {"action": "system_button", "button": button}

    match = re.match(r"SCROLL:\s*(UP|DOWN|LEFT|RIGHT)", text, re.IGNORECASE)
    if match:
        return scroll_to_action(match.group(1), width, height)

    return None


def load_episodes(path: Path, start: int, end: Optional[int], max_episodes: Optional[int]) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    with path.open() as handle:
        for idx, line in enumerate(handle):
            if idx < start:
                continue
            if end is not None and idx >= end:
                break
            episodes.append(json.loads(line))
            if max_episodes is not None and len(episodes) >= max_episodes:
                break
    return episodes


def maybe_limit_steps(episode: Dict[str, Any], max_steps: Optional[int]) -> Dict[str, Any]:
    if max_steps is None:
        return episode
    limited = dict(episode)
    limited["steps"] = episode["steps"][:max_steps]
    return limited


def format_history(summaries: List[str], k_history: int) -> str:
    if not summaries:
        return "This is the task's initial state."
    return "".join(f"Step{i + 1}: {summary}.\n" for i, summary in enumerate(summaries[-k_history:]))


def evaluate_episode(episode: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    client = OpenAI(
        base_url=args.api_url,
        api_key=args.api_key,
        timeout=args.request_timeout,
        max_retries=args.max_retries,
        default_headers={"Connection": "close"},
    )
    episode = maybe_limit_steps(episode, args.max_steps_per_episode)
    history: List[str] = []
    step_results: List[Dict[str, Any]] = []
    total_calls = 0
    first_error_step: Optional[int] = None

    for step_idx, step in enumerate(episode["steps"]):
        image = encode_image(step["screenshot"], args.image_max_pixels)
        prompt = ODYSSEY_EXECUTION_PROMPT.replace("(goal)", episode["goal"]).replace(
            "(history)", format_history(history, args.k_history)
        )

        action_response = call_chat(client, args.model_name, image, prompt, args.max_tokens)
        total_calls += 1
        raw_text = action_response.text
        think, answer, parsed = parse_har_output(raw_text)
        pred_action = normalize_har_action(answer, parsed, image.width, image.height)

        type_match = False
        extract_match = False
        error = ""
        try:
            if pred_action is not None:
                type_match, extract_match = evaluate_odyssey_action(
                    pred_action,
                    step["check_options"],
                    image.width,
                    image.height,
                )
        except Exception as exc:
            error = repr(exc)

        summary_finish_reason = ""
        summary_truncated = False
        action_text = action_to_text(pred_action, answer)
        if args.use_act2sum:
            sum_prompt = ODYSSEY_ACT2SUM_PROMPT.replace("(goal)", episode["goal"]).replace("(action)", action_text)
            sum_response = call_chat(client, args.model_name, image, sum_prompt, args.summary_max_tokens)
            total_calls += 1
            summary_finish_reason = sum_response.finish_reason
            summary_truncated = sum_response.truncated
            history.append(extract_summary(sum_response.text) or action_text[:120])
        else:
            history.append(action_text)

        if not extract_match and first_error_step is None:
            first_error_step = step_idx + 1

        step_results.append(
            {
                "step_idx": step_idx,
                "type_match": bool(type_match),
                "extract_match": bool(extract_match),
                "gt_action": step["check_options"],
                "pred_action": pred_action,
                "answer": answer,
                "think": think,
                "raw_text": raw_text,
                "screenshot": step.get("screenshot", ""),
                "image_width": image.width,
                "image_height": image.height,
                "error": error,
                "finish_reason": action_response.finish_reason,
                "truncated": action_response.truncated or summary_truncated,
                "action_truncated": action_response.truncated,
                "summary_finish_reason": summary_finish_reason,
                "summary_truncated": summary_truncated,
            }
        )

        if not extract_match and not args.no_stop:
            break

    correct_steps = sum(1 for step in step_results if step["extract_match"])
    client.close()
    return {
        "episode_id": episode.get("episode_id"),
        "goal": episode.get("goal"),
        "category": episode.get("category", ""),
        "num_steps": len(episode["steps"]),
        "steps_evaluated": len(step_results),
        "correct_steps": correct_steps,
        "task_success": correct_steps == len(episode["steps"]) and len(step_results) == len(episode["steps"]),
        "first_error_step": first_error_step,
        "total_calls": total_calls,
        "steps": step_results,
    }


def summarize(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(results)
    total_steps = sum(row["num_steps"] for row in rows)
    total_evaluated = sum(row["steps_evaluated"] for row in rows)
    correct_steps = sum(row["correct_steps"] for row in rows)
    success_count = sum(1 for row in rows if row["task_success"])
    total_generations = 0
    truncated_generations = 0
    action_generations = 0
    truncated_actions = 0
    for row in rows:
        for step in row.get("steps", []):
            action_generations += 1
            total_generations += 1
            if step.get("action_truncated") or step.get("finish_reason") == "length":
                truncated_actions += 1
                truncated_generations += 1
            if step.get("summary_finish_reason"):
                total_generations += 1
                if step.get("summary_truncated") or step.get("summary_finish_reason") == "length":
                    truncated_generations += 1
    by_category: Dict[str, Dict[str, Any]] = {}
    category_counts = Counter(row["category"] for row in rows)

    for category in sorted(category_counts):
        subset = [row for row in rows if row["category"] == category]
        cat_total_steps = sum(row["num_steps"] for row in subset)
        cat_correct_steps = sum(row["correct_steps"] for row in subset)
        cat_success = sum(1 for row in subset if row["task_success"])
        by_category[category] = {
            "paper_column": CATEGORY_TO_PAPER_COLUMN.get(category, category),
            "episodes": len(subset),
            "steps": cat_total_steps,
            "step_sr_percent": 100 * cat_correct_steps / cat_total_steps if cat_total_steps else 0,
            "tsr_percent": 100 * cat_success / len(subset) if subset else 0,
        }

    progress_step_sr = 100 * correct_steps / total_steps if total_steps else 0
    evaluated_step_sr = 100 * correct_steps / total_evaluated if total_evaluated else 0
    return {
        "episodes": len(rows),
        "total_steps": total_steps,
        "steps_evaluated": total_evaluated,
        "correct_steps": correct_steps,
        "step_sr_percent": progress_step_sr,
        "progress_step_sr_percent": progress_step_sr,
        "evaluated_step_sr_percent": evaluated_step_sr,
        "tsr_percent": 100 * success_count / len(rows) if rows else 0,
        "success_count": success_count,
        "by_category": by_category,
        "truncation": {
            "generations": total_generations,
            "truncated_generations": truncated_generations,
            "truncated_generation_percent": 100 * truncated_generations / total_generations if total_generations else 0,
            "action_generations": action_generations,
            "truncated_action_generations": truncated_actions,
            "truncated_action_percent": 100 * truncated_actions / action_generations if action_generations else 0,
        },
        "paper_gui_odyssey_overall_percent": PAPER_GUI_ODYSSEY_OVERALL,
        "delta_vs_paper_overall_points": progress_step_sr - PAPER_GUI_ODYSSEY_OVERALL,
    }


def load_partial_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_checkpoint_summary(path: Path, results: List[Dict[str, Any]], args: argparse.Namespace, test_data: Path, timestamp: str) -> None:
    summary = summarize(results)
    summary.update(build_run_metadata(args, test_data, timestamp))
    with path.open("w") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def build_run_metadata(args: argparse.Namespace, test_data: Path, timestamp: str) -> Dict[str, Any]:
    return {
        "model_name": args.model_name,
        "test_data": str(test_data),
        "start": args.start,
        "end": args.end,
        "max_episodes": args.max_episodes,
        "threads": args.threads,
        "use_act2sum": args.use_act2sum,
        "no_stop": args.no_stop,
        "request_timeout": args.request_timeout,
        "max_retries": args.max_retries,
        "max_steps_per_episode": args.max_steps_per_episode,
        "image_max_pixels": args.image_max_pixels,
        "created_at": timestamp,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", default="datasets/GUI-Odyssey/gui_odyssey_random_split_test.jsonl")
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model_name", default="har-gui-3b-gui-odyssey")
    parser.add_argument("--output_dir", default="related_work/har/outputs/gui_odyssey_paper")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--k_history", type=int, default=4)
    parser.add_argument("--image_max_pixels", type=int, default=602112)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--summary_max_tokens", type=int, default=256)
    parser.add_argument("--request_timeout", type=float, default=300.0)
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--max_steps_per_episode", type=int, default=None)
    parser.add_argument("--partial_results_path", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--no_act2sum", dest="use_act2sum", action="store_false")
    parser.add_argument("--no_stop", action="store_true")
    parser.set_defaults(use_act2sum=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_data = Path(args.test_data)
    if not test_data.is_absolute():
        test_data = WORKSPACE_ROOT / test_data
    episodes = load_episodes(test_data, args.start, args.end, args.max_episodes)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = WORKSPACE_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    partial_path = Path(args.partial_results_path) if args.partial_results_path else output_dir / f"results_{timestamp}.jsonl"
    if not partial_path.is_absolute():
        partial_path = WORKSPACE_ROOT / partial_path
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_summary_path = output_dir / f"summary_{timestamp}.checkpoint.json"

    results: List[Dict[str, Any]] = load_partial_results(partial_path) if args.resume else []
    completed_episode_ids = {row.get("episode_id") for row in results if "episode_id" in row}
    if completed_episode_ids:
        episodes = [episode for episode in episodes if episode.get("episode_id") not in completed_episode_ids]
        print(f"Resuming from {partial_path}: {len(completed_episode_ids)} completed episodes, {len(episodes)} remaining")
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(evaluate_episode, episode, args): episode.get("episode_id") for episode in episodes}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating GUI-Odyssey"):
            try:
                row = future.result()
            except Exception as exc:
                row = {"episode_id": futures[future], "error": repr(exc), "num_steps": 0, "steps_evaluated": 0, "correct_steps": 0, "task_success": False, "category": ""}
            results.append(row)
            append_jsonl(partial_path, row)
            if args.checkpoint_every and len(results) % args.checkpoint_every == 0:
                write_checkpoint_summary(checkpoint_summary_path, results, args, test_data, timestamp)

    summary = summarize(results)
    summary.update(build_run_metadata(args, test_data, timestamp))
    summary["partial_results_path"] = str(partial_path)

    results_path = output_dir / f"results_{timestamp}.json"
    summary_path = output_dir / f"summary_{timestamp}.json"
    with results_path.open("w") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    with summary_path.open("w") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved results: {results_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()