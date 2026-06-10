from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.config import REPO_ROOT


@dataclass(frozen=True)
class StepRecord:
    benchmark: str
    split: str
    episode_id: str
    step_idx: int
    episode_len: int
    goal: str
    screenshot: str
    gt_action: Dict[str, Any]
    check_options: Dict[str, Any]
    raw_step: Dict[str, Any]

    def to_json(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "split": self.split,
            "episode_id": self.episode_id,
            "step_idx": self.step_idx,
            "episode_len": self.episode_len,
            "goal": self.goal,
            "screenshot": self.screenshot,
            "gt_action": self.gt_action,
            "check_options": self.check_options,
        }


def normalize_repo_path(path: str) -> str:
    if not path:
        return path
    if str(path).startswith(str(REPO_ROOT)):
        return path
    if path.startswith("/datasets/"):
        return str(REPO_ROOT / path.lstrip("/"))
    marker = "/UI-S1/"
    if marker in path:
        suffix = path.split(marker, 1)[1]
        return str(REPO_ROOT / suffix)
    return path


def summarize_assets(records: Iterable[StepRecord], max_missing: int = 5) -> Dict[str, Any]:
    total = 0
    missing: List[str] = []
    for record in records:
        total += 1
        screenshot = normalize_repo_path(record.screenshot)
        if screenshot and not Path(screenshot).exists() and len(missing) < max_missing:
            missing.append(screenshot)
    return {
        "total_steps_checked": total,
        "missing_screenshot_examples": missing,
        "screenshots_available": len(missing) == 0,
    }


def exact_action_match(pred_action: Optional[Dict[str, Any]], gt_action: Dict[str, Any]) -> bool:
    if pred_action is None:
        return False
    return _normalize_action_dict(pred_action) == _normalize_action_dict(gt_action)


def _normalize_action_dict(action: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {}
    for key, value in action.items():
        if isinstance(value, str):
            normalized[key] = " ".join(value.lower().strip().split())
        elif isinstance(value, list):
            normalized[key] = [round(float(v), 4) if isinstance(v, (int, float)) else v for v in value]
        else:
            normalized[key] = value
    return normalized
