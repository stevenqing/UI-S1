#!/usr/bin/env python3
"""Build GUI-360 balanced test JSONL for the original template evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reproduce_gui360_fullparam_sft import (  # noqa: E402
    DEFAULT_BALANCED_DIR,
    image_bytes_from_cell,
    load_rows,
    safe_rel_image_path,
    write_image,
)


def build_jsonl(*, balanced_data_dir: Path, output_jsonl: Path, image_root: Path, max_episodes: int, require_images: bool) -> Dict[str, Any]:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    episodes = 0
    steps_written = 0
    skipped = 0
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in load_rows(balanced_data_dir, "test", max_episodes=max_episodes):
            episode_id = row.get("episode_id", episodes)
            goal = str(row.get("goal") or "")
            try:
                steps = json.loads(row.get("steps") or "[]")
            except json.JSONDecodeError:
                skipped += 1
                continue
            screenshots_value = row.get("screenshots")
            screenshots = list(screenshots_value) if screenshots_value is not None else []
            out_steps: List[Dict[str, Any]] = []
            for step_pos, step in enumerate(steps):
                action = step.get("action") if isinstance(step.get("action"), dict) else {}
                if not action:
                    skipped += 1
                    continue
                step_idx = step.get("step_idx", step_pos)
                rel_image = safe_rel_image_path("test", episode_id, step_idx)
                img_cell = screenshots[step_pos] if step_pos < len(screenshots) else None
                image_path = write_image(
                    image_root,
                    rel_image,
                    image_bytes_from_cell(img_cell),
                    str(step.get("screenshot") or ""),
                    require_image=require_images,
                )
                if not image_path:
                    skipped += 1
                    continue
                out_step = {
                    "step_idx": int(step_idx),
                    "action": action,
                    "screenshot": image_path,
                    "image_w": int(step.get("image_w") or 1040),
                    "image_h": int(step.get("image_h") or 736),
                    "bbox": step.get("bbox"),
                    "status": step.get("status", "CONTINUE"),
                }
                out_steps.append(out_step)
                steps_written += 1
            if out_steps:
                payload = {"episode_id": episode_id, "goal": goal, "num_steps": len(out_steps), "steps": out_steps}
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            episodes += 1
    return {
        "output_jsonl": str(output_jsonl),
        "image_root": str(image_root),
        "episodes_read": episodes,
        "steps_written": steps_written,
        "skipped_steps_or_episodes": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSONL for v13_gui_360/eval_gui360_template.py")
    parser.add_argument("--balanced-data-dir", default=DEFAULT_BALANCED_DIR)
    parser.add_argument("--output-jsonl", default="outputs/gui360_history_ab/original_eval/gui360_test_1000_balanced_reconstructed.jsonl")
    parser.add_argument("--image-dir", default="train_GUI_360/llamafactory/data/gui360_history_arm_images/original_template")
    parser.add_argument("--max-episodes", type=int, default=-1)
    parser.add_argument("--require-images", action="store_true")
    args = parser.parse_args()
    summary = build_jsonl(
        balanced_data_dir=Path(args.balanced_data_dir),
        output_jsonl=Path(args.output_jsonl),
        image_root=Path(args.image_dir),
        max_episodes=args.max_episodes,
        require_images=args.require_images,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()