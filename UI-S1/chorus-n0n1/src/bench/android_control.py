from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from src.bench.common import StepRecord, exact_action_match, normalize_repo_path
from src.config import REPO_ROOT


def load_steps(jsonl_path: Path, split: str = "high_level") -> List[StepRecord]:
    records: List[StepRecord] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            episode = json.loads(line)
            episode_id = str(episode.get("episode_id", line_idx))
            steps = episode.get("steps", [])
            for step_idx, step in enumerate(steps):
                gt_action = copy.deepcopy(step.get("action_content") or step.get("check_options") or {})
                check_options = copy.deepcopy(step.get("check_options") or gt_action)
                if "candidate_bbox" not in check_options:
                    check_options["candidate_bbox"] = step.get("bbox", [])
                records.append(StepRecord(
                    benchmark="android_control",
                    split=split,
                    episode_id=episode_id,
                    step_idx=step_idx,
                    episode_len=len(steps),
                    goal=episode.get("goal", ""),
                    screenshot=normalize_repo_path(step.get("screenshot", "")),
                    gt_action=gt_action,
                    check_options=check_options,
                    raw_step=step,
                ))
    return records


def official_match(
    pred_action: Optional[Dict[str, Any]],
    record: StepRecord,
    resized_width: Optional[int] = None,
    resized_height: Optional[int] = None,
) -> Dict[str, Any]:
    if pred_action is None:
        return {"exact_match": False, "type_match": False, "semantic_match": False, "match_error": None}

    eval_dir = REPO_ROOT / "evaluation"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))
    from qwenvl_utils import evaluate_android_control_action  # type: ignore

    width, height = _image_size(record.screenshot)
    resized_width = resized_width or width
    resized_height = resized_height or height
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            type_match, extract_match = evaluate_android_control_action(
                copy.deepcopy(pred_action),
                copy.deepcopy(record.check_options),
                width,
                height,
                resized_width,
                resized_height,
                ignore_actions=[],
            )
    except Exception as exc:  # official adapter should surface failures in rows
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


def _image_size(path: str) -> Tuple[int, int]:
    if path and Path(path).exists():
        with Image.open(path) as img:
            return img.size
    return 1080, 2400
