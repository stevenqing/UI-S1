#!/usr/bin/env python3
"""Build UIA-enriched GUI-360 JSONL directly from balanced split parquets.

Unlike the older test-only joiner, this script works for train/test split
parquets and writes local screenshot PNGs from the parquet `screenshots` bytes.
It joins raw `control_infos` by the original screenshot `action_step` id.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


DEFAULT_OUTPUT_ROOT = "outputs/gui360_history_ab/original_eval"
DEFAULT_IMAGE_ROOT = "train_GUI_360/llamafactory/data/gui360_history_arm_images/original_template"


def parse_steps(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, list):
        return value
    raise TypeError(f"unsupported steps value: {type(value)!r}")


def parse_original_screenshot(path_value: str) -> Optional[Tuple[str, str, str, str, int]]:
    parts = Path(path_value).parts
    try:
        image_idx = parts.index("image")
    except ValueError:
        return None
    if image_idx + 5 >= len(parts):
        return None
    app = parts[image_idx + 1]
    category = parts[image_idx + 2]
    status = parts[image_idx + 3]
    execution_id = parts[image_idx + 4]
    filename = parts[image_idx + 5]
    if not filename.startswith("action_step") or not filename.endswith(".png"):
        return None
    try:
        step_id = int(filename.removeprefix("action_step").removesuffix(".png"))
    except ValueError:
        return None
    return app, category, status, execution_id, step_id


def action_step_id(path_value: str) -> Optional[int]:
    filename = Path(path_value).name
    if not filename.startswith("action_step") or not filename.endswith(".png"):
        return None
    try:
        return int(filename.removeprefix("action_step").removesuffix(".png").removesuffix("_annotated"))
    except ValueError:
        return None


def load_raw_episode(raw_root: Path, app: str, category: str, status: str, execution_id: str) -> Dict[int, Dict[str, Any]]:
    raw_path = raw_root / app / category / status / f"{execution_id}.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"missing raw trajectory {raw_path}")
    out: Dict[int, Dict[str, Any]] = {}
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            raw_step = row.get("step") or {}
            key = action_step_id(str(raw_step.get("screenshot_clean") or ""))
            if key is None:
                key = int(row.get("step_id") or 0)
            out[key] = row
    return out


def screenshot_bytes(item: Any) -> Optional[bytes]:
    if isinstance(item, dict):
        value = item.get("bytes")
    else:
        value = getattr(item, "bytes", None)
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    return None


def controls_count(control_infos: Dict[str, Any]) -> int:
    controls = control_infos.get("uia_controls_info") or control_infos.get("merged_controls_info") or []
    return len(controls) if isinstance(controls, list) else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--balanced-parquets", default="", help="Glob for split parquet files. Defaults to datasets/gui360-balanced/data/{split}-*.parquet")
    parser.add_argument("--raw-root", default="", help="Raw GUI-360 split data root. Defaults to datasets/GUI-360-raw-jsonl/{split}/data")
    parser.add_argument("--output", default="", help="Output JSONL path")
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--write-images", action="store_true")
    parser.add_argument("--skip-existing-images", action="store_true")
    args = parser.parse_args()

    parquet_glob = args.balanced_parquets or f"datasets/gui360-balanced/data/{args.split}-*.parquet"
    raw_root = Path(args.raw_root or f"datasets/GUI-360-raw-jsonl/{args.split}/data")
    output = Path(args.output or f"{DEFAULT_OUTPUT_ROOT}/gui360_{args.split}_balanced_uia.jsonl")
    image_root = Path(args.image_root) / args.split

    parquet_paths = sorted(Path().glob(parquet_glob))
    if not parquet_paths:
        raise FileNotFoundError(f"no parquet files matched {parquet_glob}")

    output.parent.mkdir(parents=True, exist_ok=True)
    raw_cache: Dict[Tuple[str, str, str, str], Dict[int, Dict[str, Any]]] = {}
    episodes = []
    total_steps = 0
    joined_steps = 0
    missing_controls = 0
    images_written = 0
    images_existing = 0

    for parquet_path in parquet_paths:
        frame = pd.read_parquet(parquet_path)
        for _, row in frame.iterrows():
            episode_id = str(row["episode_id"])
            steps = parse_steps(row["steps"])
            screenshots_value = row.get("screenshots")
            screenshots = [] if screenshots_value is None else list(screenshots_value)
            new_steps = []
            for step in steps:
                total_steps += 1
                step_idx = int(step.get("step_idx") or len(new_steps))
                parsed = parse_original_screenshot(str(step.get("screenshot") or ""))
                if parsed is None:
                    raise ValueError(f"could not parse original screenshot for episode {episode_id} step {step_idx}: {step.get('screenshot')}")
                app, category, status, execution_id, step_id = parsed
                cache_key = (app, category, status, execution_id)
                if cache_key not in raw_cache:
                    raw_cache[cache_key] = load_raw_episode(raw_root, app, category, status, execution_id)
                raw_row = raw_cache[cache_key].get(step_id)
                if raw_row is None:
                    raise ValueError(f"missing raw step {step_id} for {cache_key}")
                raw_step = raw_row.get("step") or {}
                control_infos = raw_step.get("control_infos") or {}
                if controls_count(control_infos):
                    joined_steps += 1
                else:
                    missing_controls += 1

                local_image = image_root / f"episode_{episode_id}" / f"step_{step_idx:04d}.png"
                if args.write_images:
                    image_data = screenshot_bytes(screenshots[step_idx]) if step_idx < len(screenshots) else None
                    if image_data is None:
                        raise ValueError(f"missing screenshot bytes for episode {episode_id} step {step_idx}")
                    if local_image.exists() and args.skip_existing_images:
                        images_existing += 1
                    else:
                        local_image.parent.mkdir(parents=True, exist_ok=True)
                        local_image.write_bytes(image_data)
                        images_written += 1

                new_step = {key: value for key, value in step.items() if key not in {"conversation_human", "conversation_gpt"}}
                new_step["screenshot"] = str(local_image.resolve()) if args.write_images else str(step.get("screenshot") or "")
                new_step["control_infos"] = control_infos
                new_step["raw_execution_id"] = execution_id
                new_step["raw_step_id"] = step_id
                new_step["raw_app"] = app
                new_step["raw_category"] = category
                new_step["raw_status"] = status
                new_step["raw_screenshot_clean"] = raw_step.get("screenshot_clean")
                new_step["raw_screenshot_annotated"] = raw_step.get("screenshot_annotated")
                new_step["raw_action"] = raw_step.get("action")
                new_steps.append(new_step)
            episodes.append({
                "episode_id": episode_id,
                "goal": row.get("goal", ""),
                "num_steps": int(row.get("num_steps") or len(new_steps)),
                "steps": new_steps,
            })

    with output.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")

    summary = {
        "split": args.split,
        "output": str(output),
        "episodes": len(episodes),
        "total_steps": total_steps,
        "joined_steps_with_controls": joined_steps,
        "missing_controls": missing_controls,
        "raw_trajectories_loaded": len(raw_cache),
        "images_written": images_written,
        "images_existing": images_existing,
        "write_images": args.write_images,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()