from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.config import resolve_path


def load_episode_rows(path: str | Path) -> List[Dict[str, Any]]:
    resolved = resolve_path(path) if not Path(path).is_absolute() else Path(path)
    assert resolved is not None
    if not resolved.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL row in {resolved} at line {line_number}: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    resolved = resolve_path(path) if not Path(path).is_absolute() else Path(path)
    assert resolved is not None
    resolved.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with resolved.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def flatten_episode_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    step_rows: List[Dict[str, Any]] = []
    for episode_index, episode in enumerate(rows):
        first_error_step = episode.get("first_error_step")
        for step in episode.get("steps", []):
            step_idx = int(step.get("step_idx", 0))
            gt_action = step.get("gt_action") or {}
            pred_action = step.get("pred_action") or {}
            step_rows.append(
                {
                    "benchmark": "gui_odyssey",
                    "result_source": "har_gui_odyssey_episode_jsonl",
                    "episode_index": episode_index,
                    "episode_id": episode.get("episode_id"),
                    "category": episode.get("category", ""),
                    "goal": episode.get("goal", ""),
                    "num_steps": episode.get("num_steps", 0),
                    "steps_evaluated": episode.get("steps_evaluated", 0),
                    "correct_steps": episode.get("correct_steps", 0),
                    "task_success": bool(episode.get("task_success", False)),
                    "first_error_step": first_error_step,
                    "step_idx": step_idx,
                    "step_number": step_idx + 1,
                    "extract_match": bool(step.get("extract_match", False)),
                    "type_match": bool(step.get("type_match", False)),
                    "baseline_error": not bool(step.get("extract_match", False)),
                    "gt_action": gt_action,
                    "pred_action": pred_action,
                    "gt_action_type": gt_action.get("action"),
                    "pred_action_type": pred_action.get("action"),
                    "answer": step.get("answer", ""),
                    "think": step.get("think", ""),
                    "raw_text": step.get("raw_text", ""),
                    "screenshot": step.get("screenshot", ""),
                    "current_screenshot": step.get("screenshot", ""),
                    "finish_reason": step.get("finish_reason"),
                    "truncated": bool(step.get("truncated", False)),
                    "action_truncated": bool(step.get("action_truncated", False)),
                    "summary_finish_reason": step.get("summary_finish_reason"),
                    "summary_truncated": bool(step.get("summary_truncated", False)),
                    "image_width": step.get("image_width"),
                    "image_height": step.get("image_height"),
                    "error": step.get("error"),
                    "is_first_error_step": first_error_step == step_idx + 1,
                    "after_first_error": first_error_step is not None and step_idx + 1 >= int(first_error_step),
                    "sota_proxy": True,
                }
            )
    return step_rows


def summarize_step_rows(step_rows: Iterable[Dict[str, Any]], episodes_completed: Optional[int] = None) -> Dict[str, Any]:
    rows = list(step_rows)
    total = len(rows)
    correct = sum(1 for row in rows if row.get("extract_match"))
    action_generations = total
    summary_generations = sum(1 for row in rows if row.get("summary_finish_reason"))
    truncated_actions = sum(1 for row in rows if row.get("action_truncated") or row.get("finish_reason") == "length")
    truncated_summaries = sum(
        1 for row in rows if row.get("summary_truncated") or row.get("summary_finish_reason") == "length"
    )
    total_generations = action_generations + summary_generations
    by_category: Dict[str, Dict[str, Any]] = {}
    category_counts = Counter(str(row.get("category", "")) for row in rows)
    for category in sorted(category_counts):
        subset = [row for row in rows if str(row.get("category", "")) == category]
        cat_total = len(subset)
        cat_correct = sum(1 for row in subset if row.get("extract_match"))
        by_category[category] = {
            "steps": cat_total,
            "correct_steps": cat_correct,
            "step_sr_percent": 100 * cat_correct / cat_total if cat_total else 0.0,
            "baseline_error_steps": cat_total - cat_correct,
        }
    return {
        "episodes_completed": episodes_completed,
        "steps": total,
        "correct_steps": correct,
        "baseline_error_steps": total - correct,
        "step_sr_percent": 100 * correct / total if total else 0.0,
        "baseline_error_percent": 100 * (total - correct) / total if total else 0.0,
        "by_category": by_category,
        "truncation": {
            "generations": total_generations,
            "truncated_generations": truncated_actions + truncated_summaries,
            "truncated_generation_percent": 100 * (truncated_actions + truncated_summaries) / total_generations
            if total_generations
            else 0.0,
            "action_generations": action_generations,
            "truncated_action_generations": truncated_actions,
            "truncated_action_percent": 100 * truncated_actions / action_generations if action_generations else 0.0,
            "summary_generations": summary_generations,
            "truncated_summary_generations": truncated_summaries,
            "truncated_summary_percent": 100 * truncated_summaries / summary_generations if summary_generations else 0.0,
        },
    }


def compact_step_for_queue(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "benchmark": row.get("benchmark"),
        "episode_id": row.get("episode_id"),
        "category": row.get("category"),
        "goal": row.get("goal"),
        "step_idx": row.get("step_idx"),
        "step_number": row.get("step_number"),
        "gt_action": row.get("gt_action"),
        "pred_action": row.get("pred_action"),
        "extract_match": row.get("extract_match"),
        "type_match": row.get("type_match"),
        "baseline_error": row.get("baseline_error"),
        "answer": row.get("answer"),
        "raw_text": row.get("raw_text"),
        "screenshot": row.get("screenshot"),
        "finish_reason": row.get("finish_reason"),
        "truncated": row.get("truncated"),
        "sota_proxy": row.get("sota_proxy", True),
    }


def category_error_table(step_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, Dict[str, int]] = defaultdict(lambda: {"steps": 0, "errors": 0, "type_errors": 0})
    for row in step_rows:
        category = str(row.get("category", ""))
        buckets[category]["steps"] += 1
        if row.get("baseline_error"):
            buckets[category]["errors"] += 1
        if not row.get("type_match"):
            buckets[category]["type_errors"] += 1
    table = []
    for category, values in sorted(buckets.items()):
        steps = values["steps"]
        errors = values["errors"]
        table.append(
            {
                "category": category,
                "steps": steps,
                "baseline_errors": errors,
                "baseline_error_percent": 100 * errors / steps if steps else 0.0,
                "type_errors": values["type_errors"],
            }
        )
    return table