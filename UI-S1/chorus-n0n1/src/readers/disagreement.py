from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from src.readers.input import ReaderInput


N1_READERS = [
    "action_format_reader",
    "screen_goal_reader",
    "coordinate_consistency_reader",
]


def build_disagreement_summary(step_rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n1_status": "READER_INPUTS_READY",
        "definition": "N1 proper requires independent reader calls from GT-isolated ReaderInput records.",
        "steps": len(list(step_rows)),
        "reader_model_calls": "not_started",
        "model_call_requirement": "All reader calls must use src/infer/wrapper.py.",
        "truncation_gate": "Do not run N1 readers if Phase A truncated_generation_percent exceeds 1%.",
        "gt_isolation": "reader inputs contain no ground-truth actions, labels, or target_action fields",
        "readers": N1_READERS,
    }


def build_reader_queue(
    step_rows: Iterable[Dict[str, Any]],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    queue: List[Dict[str, Any]] = []
    for row in step_rows:
        item = reader_input_from_step_row(row).to_json()
        item.update({
            "n1_reader_status": "queued",
            "n1_reader_family": "disagreement",
            "required_readers": N1_READERS,
            "expected_outputs": {
                "reader_votes": "independent reader judgments produced without GT fields",
                "disagreement_score": "normalized pairwise disagreement across readers",
            },
            "model_calls_must_use": "src/infer/wrapper.py",
        })
        queue.append(item)
        if limit is not None and len(queue) >= limit:
            break
    return queue


def build_disagreement_manifest(
    step_rows: Iterable[Dict[str, Any]],
    queue_path: str,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    rows = list(step_rows)
    queue = build_reader_queue(rows, limit=limit)
    summary = build_disagreement_summary(rows)
    summary.update(
        {
            "queue_path": queue_path,
            "queue_items": len(queue),
            "queue_limit": limit,
            "sota_proxy": True,
        }
    )
    return summary


def reader_input_from_step_row(row: Dict[str, Any]) -> ReaderInput:
    schema_payload = {
        "subject": "HAR",
        "benchmark": row.get("benchmark", "gui_odyssey"),
        "action_space": ["click", "long_press", "swipe", "type", "system_button", "terminate", "wait", "open"],
        "image_width": row.get("image_width"),
        "image_height": row.get("image_height"),
    }
    return ReaderInput(
        goal=str(row.get("goal", "")),
        current_screenshot=str(row.get("current_screenshot", row.get("screenshot", "")) or ""),
        schema_payload=schema_payload,
        episode_id=str(row.get("episode_id", "")),
        step_idx=int(row.get("step_idx", 0)),
        episode_len=int(row.get("num_steps", row.get("episode_len", 0)) or 0),
    )