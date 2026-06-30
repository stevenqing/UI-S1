#!/usr/bin/env python3
"""Build a compact GT-history ShareGPT JSON for GUI-360 balanced test.

The balanced test split has action labels and screenshots, but no stored
conversation_human/conversation_gpt text. This script reconstructs the same
compact G-arm prompt format used for history SFT evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reproduce_gui360_fullparam_sft import (
    DEFAULT_BALANCED_DIR,
    image_bytes_from_cell,
    load_rows,
    safe_rel_image_path,
    write_image,
)


def compact_human(goal: str) -> str:
    instruction = " ".join(str(goal or "").split())
    return (
        "<image>\n"
        "You are a GUI action agent. Use the current screenshot, the user instruction, and previous conversation turns to choose the next action.\n\n"
        f"Instruction: {instruction}\n\n"
        "History: previous turns in this conversation. If there are no previous turns, history is None.\n\n"
        "Supported actions: click(coordinate=[x,y]), type(text=...), swipe(start_coordinate=[x,y], end_coordinate=[x,y]), press(key=...), wait().\n"
        "Output exactly one JSON action inside <tool_call></tool_call>."
    )


def _coord(value: Any) -> List[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None


def action_to_tool_call(action: Dict[str, Any], status: str = "CONTINUE") -> str:
    function = str(action.get("function") or action.get("action") or "").strip().lower()
    args: Dict[str, Any] = {}
    if function == "click":
        coord = _coord(action.get("coordinate") or action.get("xy"))
        if coord is not None:
            args["coordinate"] = coord
    elif function in {"type", "input", "paste"}:
        function = "type"
        text = action.get("text") or action.get("keys") or action.get("value")
        if text is not None:
            args["text"] = str(text)
        coord = _coord(action.get("coordinate") or action.get("xy"))
        if coord is not None:
            args["coordinate"] = coord
    elif function in {"swipe", "drag", "scroll", "wheel_mouse_input"}:
        function = "swipe"
        start = _coord(action.get("start_coordinate") or action.get("startCoordinate") or action.get("coordinate"))
        end = _coord(action.get("end_coordinate") or action.get("endCoordinate"))
        if start is not None:
            args["start_coordinate"] = start
        if end is not None:
            args["end_coordinate"] = end
    payload = {"function": function, "args": args, "status": status or "CONTINUE"}
    return "<tool_call>\n" + json.dumps(payload, indent=4, ensure_ascii=False) + "\n</tool_call>"


def build_test_dataset(*, balanced_data_dir: Path, output_json: Path, image_root: Path, max_episodes: int, require_images: bool) -> Dict[str, Any]:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    examples: List[Dict[str, Any]] = []
    episodes = 0
    steps_seen = 0
    skipped = 0
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
        conversations: List[Dict[str, str]] = []
        images: List[str] = []
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
            conversations.append({"from": "human", "value": compact_human(goal)})
            conversations.append({"from": "gpt", "value": action_to_tool_call(action, str(step.get("status") or "CONTINUE"))})
            images.append(image_path)
            steps_seen += 1
        if conversations:
            examples.append({"conversations": conversations, "images": images})
        episodes += 1
    output_json.write_text(json.dumps(examples, ensure_ascii=False), encoding="utf-8")
    return {
        "output_json": str(output_json),
        "image_root": str(image_root),
        "episodes_read": episodes,
        "examples_written": len(examples),
        "steps_seen": steps_seen,
        "skipped_steps_or_episodes": skipped,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact GUI-360 GT-history test ShareGPT JSON")
    parser.add_argument("--balanced-data-dir", default=DEFAULT_BALANCED_DIR)
    parser.add_argument("--output-json", default="train_GUI_360/llamafactory/data/gui360_gt_history_test.json")
    parser.add_argument("--image-dir", default="train_GUI_360/llamafactory/data/gui360_history_arm_images/gt_history")
    parser.add_argument("--max-episodes", type=int, default=-1)
    parser.add_argument("--require-images", action="store_true")
    args = parser.parse_args()
    summary = build_test_dataset(
        balanced_data_dir=Path(args.balanced_data_dir),
        output_json=Path(args.output_json),
        image_root=Path(args.image_dir),
        max_episodes=args.max_episodes,
        require_images=args.require_images,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()