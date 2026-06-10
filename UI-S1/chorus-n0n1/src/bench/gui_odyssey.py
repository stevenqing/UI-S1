from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.bench.common import StepRecord, exact_action_match, normalize_repo_path
from src.config import REPO_ROOT


def load_steps(jsonl_path: Path, split: str = "test") -> List[StepRecord]:
    records: List[StepRecord] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            episode = json.loads(line)
            episode_id = str(episode.get("episode_id", line_idx))
            steps = episode.get("steps", [])
            for step_idx, step in enumerate(steps):
                gt_action = copy.deepcopy(step.get("action_content") or step.get("check_options") or {})
                check_options = copy.deepcopy(step.get("check_options") or gt_action)
                records.append(StepRecord(
                    benchmark="gui_odyssey",
                    split=split,
                    episode_id=episode_id,
                    step_idx=step_idx,
                    episode_len=len(steps),
                    goal=episode.get("goal") or episode.get("question", ""),
                    screenshot=normalize_repo_path(step.get("screenshot") or step.get("image", "")),
                    gt_action=gt_action,
                    check_options=check_options,
                    raw_step=step,
                ))
    return records


def official_match(
    pred_action: Optional[Dict[str, Any]],
    record: StepRecord,
    resized_width: int = 1080,
    resized_height: int = 2400,
) -> Dict[str, Any]:
    if pred_action is None:
        return {"exact_match": False, "type_match": False, "semantic_match": False, "match_error": None}

    eval_dir = REPO_ROOT / "gui_odyssey_eval"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    from odyssey_action_matching import evaluate_odyssey_action  # type: ignore

    try:
        type_match, extract_match = evaluate_odyssey_action(
            copy.deepcopy(pred_action),
            copy.deepcopy(record.check_options),
            resized_width,
            resized_height,
        )
    except Exception as exc:
        return {
            "exact_match": exact_action_match(pred_action, record.gt_action),
            "type_match": False,
            "semantic_match": False,
            "match_error": repr(exc),
        }
    return {
        "exact_match": exact_action_match(pred_action, record.gt_action),
        "type_match": bool(type_match),
        "semantic_match": bool(extract_match),
        "match_error": None,
    }
