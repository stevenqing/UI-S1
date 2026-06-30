"""Runtime stage runners for the GUI-360 long-horizon orchestrator."""

from __future__ import annotations

from typing import Any, Dict, List

from .data.loader import load_image, load_trajectories
from .run_all import StageResult


def _first_shard(config: Dict[str, Any]) -> Dict[str, Any]:
    shards = config.get("shards") or []
    if not shards:
        raise ValueError("config.shards must contain at least one shard")
    shard = dict(shards[0])
    shard.setdefault("app", "excel")
    shard.setdefault("tag", "in_app")
    return shard


def _load_one(repo: str, split: str, app: str, tag: str) -> Dict[str, Any]:
    trajectories = load_trajectories(repo, split, app, tag, limit=20 if split in {"train", "test"} else 1)
    if not trajectories:
        return {"split": split, "passed": False, "reason": "no trajectories loaded"}
    exec_id, steps = next(iter(trajectories.items()))
    if split in {"train", "test"}:
        for candidate_exec_id, candidate_steps in trajectories.items():
            if candidate_steps and all(step.gt_xy is not None and step.gt_rect is not None for step in candidate_steps):
                exec_id, steps = candidate_exec_id, candidate_steps
                break
    if len(steps) < 1:
        return {"split": split, "passed": False, "exec_id": exec_id, "reason": "empty trajectory"}
    image_checks: List[Dict[str, Any]] = []
    for step in steps[: min(2, len(steps))]:
        try:
            image = load_image(repo, step.image_rel_path)
            image_checks.append({"step_id": step.step_id, "image_rel_path": step.image_rel_path, "ok": True, "size": list(image.size), "mode": image.mode})
        except Exception as exc:
            image_checks.append({"step_id": step.step_id, "image_rel_path": step.image_rel_path, "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:240]}"})
    gt_ok = all(step.gt_xy is not None and step.gt_rect is not None for step in steps) if split in {"train", "test"} else all(step.gt_xy is None and step.gt_rect is None for step in steps)
    passed = bool(len(steps) >= 1 and steps[0].contiguous and gt_ok and all(check["ok"] for check in image_checks))
    return {
        "split": split,
        "passed": passed,
        "exec_id": exec_id,
        "n_steps": len(steps),
        "contiguous": bool(steps[0].contiguous),
        "gt_ok": bool(gt_ok),
        "image_checks": image_checks,
    }


def loader_smoke(config: Dict[str, Any]) -> StageResult:
    repo = str(config.get("repo") or "vyokky/GUI-360")
    shard = _first_shard(config)
    app, tag = str(shard["app"]), str(shard["tag"])
    splits = [split for split in shard.get("splits", ["test", "fail"]) if split in {"train", "test", "fail"}]
    details = {"repo": repo, "app": app, "tag": tag, "splits": []}
    for split in splits:
        details["splits"].append(_load_one(repo, split, app, tag))
    passed = bool(details["splits"] and all(item.get("passed") for item in details["splits"]))
    return StageResult(name="loader_smoke", passed=passed, details=details)


def default_runners() -> Dict[str, Any]:
    return {"loader_smoke": loader_smoke}
