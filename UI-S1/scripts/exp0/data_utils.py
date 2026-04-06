#!/usr/bin/env python3
"""
Shared data utilities for Experiment 0.x scripts.

Handles:
- Loading paired success/fail trajectories from GUI-360 dataset
- Parquet evaluation data loading
- Action format normalization across different data sources
- Coordinate matching utilities
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "datasets" / "GUI-360"

# Data directories
SUCCESS_TRAIN_DIR = DATASET_ROOT / "train" / "data"
SUCCESS_TEST_DIR = DATASET_ROOT / "test" / "data"
FAIL_DIR = DATASET_ROOT / "fail" / "data"

# Image directories
SUCCESS_TRAIN_IMAGE_DIR = DATASET_ROOT / "train" / "image"
SUCCESS_TEST_IMAGE_DIR = DATASET_ROOT / "test" / "image"
FAIL_IMAGE_DIR = DATASET_ROOT / "fail"  # fail images use relative paths from fail/

# Parquet eval data
PARQUET_EVAL_PATH = PROJECT_ROOT / "train_GUI_360" / "data" / "gui360_test_sft_eval_format_with_bbox.parquet"

DOMAINS = ["excel", "word", "ppt"]
SUBCATEGORIES = ["in_app", "online", "search"]


def load_trajectory(jsonl_path: Path) -> list[dict]:
    """Load a trajectory from a per-step JSONL file. Returns list of step dicts."""
    steps = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            steps.append(json.loads(line))
    return steps


def normalize_action(step: dict) -> dict:
    """
    Normalize action format from different data sources into a common format.

    Returns dict with:
        action_type: str (click, type, drag, etc.)
        coordinate: [x, y] or None (absolute pixels)
        coordinate2: [x2, y2] or None (for drag/select_text)
        text: str or None
        bbox: dict or None
        direction: str or None
        button: str or None
        status: str or None
    """
    # Format from fail data: step.action.function, step.action.args.x/y
    step_data = step.get("step", {})
    action_raw = step_data.get("action", {})

    if isinstance(action_raw, dict) and "function" in action_raw:
        # Fail data format
        args = action_raw.get("args", {})
        coord = None
        if action_raw.get("coordinate_x") is not None and action_raw.get("coordinate_y") is not None:
            coord = [action_raw["coordinate_x"], action_raw["coordinate_y"]]
        elif args.get("x") is not None and args.get("y") is not None:
            coord = [args["x"], args["y"]]

        rect = action_raw.get("rectangle", action_raw.get("desktop_rectangle"))
        bbox = None
        if rect:
            bbox = {
                "left": rect.get("left"),
                "top": rect.get("top"),
                "right": rect.get("right"),
                "bottom": rect.get("bottom"),
            }

        return {
            "action_type": action_raw.get("function", ""),
            "coordinate": coord,
            "coordinate2": None,  # fail data doesn't typically have coord2
            "text": args.get("text"),
            "bbox": bbox,
            "direction": args.get("direction"),
            "button": args.get("button", "left"),
            "status": step_data.get("status", "CONTINUE"),
        }

    # Format from RL data (gui360_test.jsonl): action_content.action, action_content.coordinate
    action_content = step.get("action_content", {})
    if isinstance(action_content, dict) and "action" in action_content:
        return {
            "action_type": action_content.get("action", ""),
            "coordinate": action_content.get("coordinate"),
            "coordinate2": action_content.get("coordinate2"),
            "text": action_content.get("text"),
            "bbox": None,
            "direction": None,
            "button": action_content.get("button"),
            "status": action_content.get("status"),
        }

    return {
        "action_type": "",
        "coordinate": None,
        "coordinate2": None,
        "text": None,
        "bbox": None,
        "direction": None,
        "button": None,
        "status": None,
    }


def get_screenshot_path(step: dict, source: str = "test") -> Optional[str]:
    """
    Get absolute screenshot path from a step dict.

    Args:
        step: Step dict from trajectory
        source: 'test', 'train', or 'fail'
    """
    step_data = step.get("step", {})
    rel_path = step_data.get("screenshot_clean", "")
    if not rel_path:
        # RL data format
        rel_path = step.get("screenshot", "")

    if not rel_path:
        return None

    if os.path.isabs(rel_path):
        return rel_path

    # Resolve relative paths
    if source == "fail":
        # Fail screenshots: relative to fail/ dir
        base = DATASET_ROOT / "fail" / "image"
        # Try different possible locations
        for domain in DOMAINS:
            for subcat in SUBCATEGORIES:
                candidate = base / domain / subcat / rel_path
                if candidate.exists():
                    return str(candidate)
        # Direct from fail dir
        candidate = DATASET_ROOT / "fail" / rel_path
        if candidate.exists():
            return str(candidate)
    elif source == "test":
        base = DATASET_ROOT / "test" / "image"
        for domain in DOMAINS:
            for subcat in SUBCATEGORIES:
                candidate = base / domain / subcat / rel_path
                if candidate.exists():
                    return str(candidate)
    elif source == "train":
        base = DATASET_ROOT / "train" / "image"
        for domain in DOMAINS:
            for subcat in SUBCATEGORIES:
                candidate = base / domain / subcat / rel_path
                if candidate.exists():
                    return str(candidate)

    # Fallback: try from project root
    candidate = PROJECT_ROOT / rel_path
    if candidate.exists():
        return str(candidate)

    return None


def scan_trajectory_files(data_dir: Path) -> dict[str, Path]:
    """Scan a data directory for trajectory JSONL files. Returns {execution_id: path}."""
    result = {}
    for f in data_dir.rglob("*.jsonl"):
        result[f.stem] = f
    return result


INDEX_CACHE_PATH = PROJECT_ROOT / "outputs" / "pair_index_cache.json"


def _build_request_index(rebuild: bool = False) -> dict:
    """
    Build a cached index mapping request text → {success_files, fail_files}.

    Scanning 62K+ fail files is slow (~5 min), so we cache the result.
    Returns: {request_text: {'success': [(eid, path_str)], 'fail': [(eid, path_str)]}}
    """
    if not rebuild and INDEX_CACHE_PATH.exists():
        print(f"Loading cached pair index from {INDEX_CACHE_PATH}")
        with open(INDEX_CACHE_PATH) as f:
            return json.load(f)

    print("Building request → file index (this takes a few minutes for 62K+ files)...")
    index = defaultdict(lambda: {"success": [], "fail": []})

    # Index success files (fast: ~17K files)
    for base, label in [(SUCCESS_TRAIN_DIR, "success"), (SUCCESS_TEST_DIR, "success")]:
        if not base.exists():
            continue
        for fpath in base.rglob("*.jsonl"):
            try:
                with open(fpath) as fh:
                    line = fh.readline().strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    req = d.get("request", "")
                    eid = d.get("execution_id", fpath.stem)
                    if req:
                        index[req]["success"].append((eid, str(fpath)))
            except Exception:
                continue

    # Index fail files (slow: ~62K files)
    if FAIL_DIR.exists():
        count = 0
        for fpath in FAIL_DIR.rglob("*.jsonl"):
            try:
                with open(fpath) as fh:
                    line = fh.readline().strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    req = d.get("request", "")
                    eid = d.get("execution_id", fpath.stem)
                    if req:
                        index[req]["fail"].append((eid, str(fpath)))
                count += 1
                if count % 10000 == 0:
                    print(f"  Indexed {count} fail files...")
            except Exception:
                continue

    # Cache to disk
    INDEX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_CACHE_PATH, "w") as f:
        json.dump(dict(index), f)

    paired = sum(1 for v in index.values() if v["success"] and v["fail"])
    print(f"Index built: {len(index)} unique requests, {paired} paired")
    return dict(index)


def load_paired_trajectories(
    max_pairs: int = 0,
    use_request_matching: bool = True,
    rebuild_index: bool = False,
) -> list[dict]:
    """
    Load paired success/fail trajectories.

    Returns list of dicts:
        {
            'request': str,
            'execution_id': str,  # success execution_id
            'domain': str,
            'success_steps': [step_dicts],
            'fail_steps': [step_dicts],
            'success_path': Path,
            'fail_path': Path,
        }

    Uses a cached index for fast request-based matching (~6,700 pairs available).
    First run builds the cache (~5 min); subsequent runs load from cache (<1s).
    """
    # Build or load the request index
    index = _build_request_index(rebuild=rebuild_index)

    pairs = []
    used_requests = set()

    # Iterate over requests that have both success and fail
    for req, files in index.items():
        if max_pairs and len(pairs) >= max_pairs:
            break
        if not files.get("success") or not files.get("fail"):
            continue
        if req in used_requests:
            continue

        s_eid, s_path = files["success"][0]
        f_eid, f_path = files["fail"][0]

        try:
            s_steps = load_trajectory(Path(s_path))
            f_steps = load_trajectory(Path(f_path))
            if not s_steps or not f_steps:
                continue

            domain = s_eid.split("_")[0] if "_" in s_eid else "unknown"
            pairs.append({
                "request": req,
                "execution_id": s_eid,
                "domain": domain,
                "success_steps": s_steps,
                "fail_steps": f_steps,
                "success_path": Path(s_path),
                "fail_path": Path(f_path),
            })
            used_requests.add(req)
        except Exception:
            continue

    print(f"Loaded {len(pairs)} paired trajectories")
    return pairs[:max_pairs] if max_pairs else pairs


def find_divergence_step(success_steps: list[dict], fail_steps: list[dict], threshold: float = 50.0) -> int:
    """
    Find the step index where success and fail trajectories diverge.

    Uses coordinate distance as the primary signal: if the coordinates
    at a given step differ by more than threshold pixels, that's a divergence.

    Returns the step index (0-based) of the first divergence, or min(len_s, len_f) - 1.
    """
    min_len = min(len(success_steps), len(fail_steps))

    for i in range(min_len):
        s_action = normalize_action(success_steps[i])
        f_action = normalize_action(fail_steps[i])

        # Different action types = divergence
        if s_action["action_type"] != f_action["action_type"]:
            return i

        # Both have coordinates: check distance
        s_coord = s_action["coordinate"]
        f_coord = f_action["coordinate"]
        if s_coord is not None and f_coord is not None:
            dist = np.sqrt((s_coord[0] - f_coord[0]) ** 2 + (s_coord[1] - f_coord[1]) ** 2)
            if dist > threshold:
                return i

        # Text mismatch for type actions
        if s_action["action_type"] == "type":
            s_text = (s_action.get("text") or "").lower().strip()
            f_text = (f_action.get("text") or "").lower().strip()
            if s_text != f_text and s_text and f_text:
                return i

    return max(0, min_len - 1)


def coordinate_distance(coord1, coord2) -> float:
    """Euclidean distance between two [x, y] coordinates. Returns inf if either is None."""
    if coord1 is None or coord2 is None:
        return float("inf")
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


def is_coord_in_bbox(coord, bbox, margin: float = 0) -> bool:
    """Check if coordinate falls within bounding box."""
    if coord is None or bbox is None:
        return False
    x, y = coord
    return (
        bbox.get("left", 0) - margin <= x <= bbox.get("right", 0) + margin
        and bbox.get("top", 0) - margin <= y <= bbox.get("bottom", 0) + margin
    )
