#!/usr/bin/env python3
"""Convert GUI-360 HF parquet shards to episode JSONL plus local PNG files."""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, Iterable, List, Optional


def import_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required for parquet conversion. Run this script with "
            ".venv-qwen3-vllm/bin/python, where pyarrow is already installed."
        ) from exc
    return pq


def expand_inputs(patterns: Iterable[str]) -> List[str]:
    paths: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(matches)
        elif os.path.exists(pattern):
            paths.append(pattern)
    deduped = sorted(dict.fromkeys(paths))
    if not deduped:
        raise SystemExit(f"No parquet files matched: {list(patterns)}")
    return deduped


def ensure_bytes(value: Any) -> Optional[bytes]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    return None


def write_png(path: str, data: bytes, overwrite: bool = False) -> None:
    if os.path.exists(path) and not overwrite:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)


def convert_row(
    row: Dict[str, Any],
    image_root: str,
    split_name: str,
    overwrite_images: bool = False,
) -> tuple[Optional[Dict[str, Any]], Dict[str, int]]:
    stats = {
        "missing_steps": 0,
        "missing_screenshot_bytes": 0,
        "written_images": 0,
        "step_image_mismatch": 0,
    }

    episode_id = str(row.get("episode_id"))
    goal = row.get("goal", "")
    try:
        steps = json.loads(row.get("steps") or "[]")
    except json.JSONDecodeError:
        stats["missing_steps"] += 1
        return None, stats

    if not steps:
        stats["missing_steps"] += 1
        return None, stats

    screenshots = row.get("screenshots") or []
    if len(screenshots) != len(steps):
        stats["step_image_mismatch"] += 1

    episode_dir = os.path.join(image_root, split_name, f"episode_{int(episode_id):05d}")

    for step_idx, step in enumerate(steps):
        image_entry = screenshots[step_idx] if step_idx < len(screenshots) else None
        image_bytes = None
        if isinstance(image_entry, dict):
            image_bytes = ensure_bytes(image_entry.get("bytes"))
        if image_bytes is None:
            stats["missing_screenshot_bytes"] += 1
            continue

        image_path = os.path.join(episode_dir, f"step_{step_idx:03d}.png")
        existed = os.path.exists(image_path)
        write_png(image_path, image_bytes, overwrite=overwrite_images)
        if overwrite_images or not existed:
            stats["written_images"] += 1
        step["screenshot"] = image_path
        step.setdefault("step_idx", step_idx)
        step.setdefault("image_w", 1040)
        step.setdefault("image_h", 736)

    episode = {
        "episode_id": episode_id,
        "goal": goal,
        "num_steps": int(row.get("num_steps") or len(steps)),
        "steps": steps,
    }
    return episode, stats


def add_stats(total: Dict[str, int], part: Dict[str, int]) -> None:
    for key, value in part.items():
        total[key] = total.get(key, 0) + int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GUI-360 parquet shards to episode JSONL")
    parser.add_argument("--parquet", nargs="+", required=True, help="Parquet paths or glob patterns")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--split_name", default="train")
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--overwrite_images", action="store_true")
    args = parser.parse_args()

    pq = import_pyarrow_parquet()
    parquet_paths = expand_inputs(args.parquet)
    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)
    os.makedirs(args.image_root, exist_ok=True)

    stats: Dict[str, int] = {
        "input_files": len(parquet_paths),
        "rows_seen": 0,
        "episodes_written": 0,
        "steps_written": 0,
        "written_images": 0,
        "missing_steps": 0,
        "missing_screenshot_bytes": 0,
        "step_image_mismatch": 0,
    }

    with open(args.output_jsonl, "w") as output:
        for parquet_path in parquet_paths:
            parquet_file = pq.ParquetFile(parquet_path)
            for row_group_idx in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(row_group_idx)
                for row in table.to_pylist():
                    if args.max_episodes and stats["episodes_written"] >= args.max_episodes:
                        break
                    stats["rows_seen"] += 1
                    episode, row_stats = convert_row(
                        row,
                        args.image_root,
                        args.split_name,
                        overwrite_images=args.overwrite_images,
                    )
                    add_stats(stats, row_stats)
                    if episode is None:
                        continue
                    output.write(json.dumps(episode, ensure_ascii=False) + "\n")
                    stats["episodes_written"] += 1
                    stats["steps_written"] += len(episode["steps"])
                if args.max_episodes and stats["episodes_written"] >= args.max_episodes:
                    break
            if args.max_episodes and stats["episodes_written"] >= args.max_episodes:
                break

    summary_path = os.path.splitext(args.output_jsonl)[0] + ".summary.json"
    summary = {
        **stats,
        "parquet_paths": parquet_paths,
        "output_jsonl": args.output_jsonl,
        "image_root": args.image_root,
        "split_name": args.split_name,
    }
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {args.output_jsonl}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()