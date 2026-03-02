#!/usr/bin/env python3
"""
Convert GUI-360 raw test trajectories to parquet format matching training data format.

This script converts test data to the SAME format as the training data
(processed_data/action_prediction_train_resize/training_data.json), ensuring
consistency for proper evaluation.

Usage:
    python train_GUI_360/data_preparation/prepare_gui360_test_parquet_matched.py \
        --test-dir datasets/GUI-360/test/data \
        --image-dir datasets/GUI-360/test/image \
        --output train_GUI_360/data/gui360_test_sft_matched.parquet
"""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd


# Action format template (extracted from training data)
ACTION_TEMPLATE = """<action>
- click
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to click at.
    - button: str, The mouse button to click. One of ''left'', ''right'', ''middle'' or ''x'' (Default: ''left'')
    - double: bool, Whether to perform a double click or not (Default: False)'
    - pressed: str|None, The keyboard key to press while clicking. Common keys include: CONTROL (Ctrl), SHIFT (Shift), MENU (Alt), etc. Use the key names without VK_ prefix or braces. For example, 'CONTROL' for the Control key (Default: None)
  - Example: click(coordinate=[100, 100], button='left', double=False, pressed=None), click(coordinate=[100, 100], button='x')
- type
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to type at.
    - keys: str, The key to input. It can be any key on the keyboard, with special keys represented by their virtual key codes. For example, "{VK_CONTROL}c" represents the Ctrl+C shortcut key.
    - clear_current_text: bool, Whether to clear the current text in the Edit before setting the new text. If True, the current text will be completely replaced by the new text. (Default: False)
    - control_focus: bool, Whether to focus on your selected control item before typing the keys. If False, the hotkeys will operate on the application window. (Default: True)
  - Example: type(coordinate=[100, 100], keys='Hello'), type(coordinate=[100, 100], keys='{VK_CONTROL}c'), type(coordinate=[100, 100], keys="{TAB 2}")
- drag
  - Args:
    - start_coordinate: [x, y], the absolute position on the screen where the drag starts.
    - end_coordinate: [x, y], the absolute position on the screen where the drag ends.
    - pressed: str|None, The keyboard key to press while dragging. (Default: None)
  - Example: drag(start_coordinate=[100, 100], end_coordinate=[200, 200], pressed=None)
- wheel_mouse_input
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to scroll at.
    - wheel_dist: int, The distance of the wheel. A positive value indicates scrolling up, and a negative value indicates scrolling down.
  - Example: wheel_mouse_input(coordinate=[100, 100], wheel_dist=5)
- select_text
  - Args:
    - start_coordinate: [x, y], the absolute position on the screen where the text selection starts.
    - end_coordinate: [x, y], the absolute position on the screen where the text selection ends.
  - Example: select_text(start_coordinate=[100, 100], end_coordinate=[200, 200])
- summary
  - Args:
    - text: str, the complete summary of the task.
  - Example: summary(text='The task is to open the file and save it.')
</action>"""


def build_human_prompt(request: str, observation: str = "", history: str = "") -> str:
    """Build human prompt matching training data format."""
    prompt = f"""<image>
You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{request}

The history of actions are:
{history}

The actions supported are:
{ACTION_TEMPLATE}

Please output your action in the following format:
<tool_call>
{{
    "function": "function_name",
    "args": {{
        "arg1": value1,
        "arg2": value2
    }},
    "status": "CONTINUE or FINISH"
}}
</tool_call>"""
    return prompt


def build_gpt_response(action: Dict[str, Any], status: str) -> str:
    """Build GPT response matching training data format."""
    function_name = action.get("function", "click")

    # Build args based on function type
    args = {}

    # Get coordinates
    coord_x = action.get("coordinate_x", action.get("desktop_coordinate_x"))
    coord_y = action.get("coordinate_y", action.get("desktop_coordinate_y"))

    if function_name == "click":
        if coord_x is not None and coord_y is not None:
            args["coordinate"] = [int(coord_x), int(coord_y)]
        action_args = action.get("args", {})
        if isinstance(action_args, dict):
            args["button"] = action_args.get("button", "left")
            args["double"] = action_args.get("double", False)

    elif function_name == "type":
        if coord_x is not None and coord_y is not None:
            args["coordinate"] = [int(coord_x), int(coord_y)]
        action_args = action.get("args", {})
        if isinstance(action_args, dict):
            if "keys" in action_args:
                args["text"] = action_args["keys"]
            elif "text" in action_args:
                args["text"] = action_args["text"]
        # Also check control_text
        if "text" not in args and action.get("control_text"):
            args["text"] = action["control_text"]

    elif function_name == "drag":
        action_args = action.get("args", {})
        if isinstance(action_args, dict):
            start_x = action_args.get("start_x", coord_x)
            start_y = action_args.get("start_y", coord_y)
            end_x = action_args.get("end_x")
            end_y = action_args.get("end_y")
            if start_x is not None and start_y is not None:
                args["start_coordinate"] = [int(start_x), int(start_y)]
            if end_x is not None and end_y is not None:
                args["end_coordinate"] = [int(end_x), int(end_y)]

    elif function_name == "wheel_mouse_input":
        if coord_x is not None and coord_y is not None:
            args["coordinate"] = [int(coord_x), int(coord_y)]
        action_args = action.get("args", {})
        if isinstance(action_args, dict):
            args["wheel_dist"] = action_args.get("wheel_dist", 3)

    elif function_name == "select_text":
        action_args = action.get("args", {})
        if isinstance(action_args, dict):
            if "start_x" in action_args:
                args["start_coordinate"] = [int(action_args["start_x"]), int(action_args["start_y"])]
            if "end_x" in action_args:
                args["end_coordinate"] = [int(action_args["end_x"]), int(action_args["end_y"])]

    elif function_name == "summary":
        action_args = action.get("args", {})
        if isinstance(action_args, dict):
            args["text"] = action_args.get("text", "")

    else:
        # For other functions, try to extract coordinate
        if coord_x is not None and coord_y is not None:
            args["coordinate"] = [int(coord_x), int(coord_y)]
        # Copy other args
        action_args = action.get("args", {})
        if isinstance(action_args, dict):
            for k, v in action_args.items():
                if k not in ["x", "y"]:
                    args[k] = v

    # Map status
    status_map = {
        "CONTINUE": "CONTINUE",
        "FINISH": "FINISH",
        "OVERALL_FINISH": "FINISH"
    }
    mapped_status = status_map.get(status, "CONTINUE")

    response_dict = {
        "function": function_name,
        "args": args,
        "status": mapped_status
    }

    response = f"<tool_call>\n{json.dumps(response_dict, indent=4, ensure_ascii=False)}\n</tool_call>"
    return response


def get_bbox_from_action(action: Dict[str, Any]) -> Optional[List[float]]:
    """Extract bounding box from action rectangle."""
    rect = action.get("rectangle", action.get("desktop_rectangle"))
    if rect and isinstance(rect, dict):
        return [
            float(rect.get("left", 0)),
            float(rect.get("top", 0)),
            float(rect.get("right", 0)),
            float(rect.get("bottom", 0))
        ]
    return None


def process_test_trajectory(
    trajectory: Dict[str, Any],
    image_base_dir: str,
    validate_images: bool = False
) -> Optional[Dict]:
    """
    Process a raw GUI-360 test trajectory to match training data format.
    """
    execution_id = trajectory.get("execution_id", "")
    request = trajectory.get("request", "")
    step_id = trajectory.get("step_id", 1)
    step = trajectory.get("step", {})

    if not request or not step:
        return None

    # Get screenshot path
    screenshot = step.get("screenshot_clean", step.get("screenshot_desktop", ""))
    if not screenshot:
        return None

    # Build full image path
    # screenshot format: "success/ppt_1_42/action_step1.png"
    full_img_path = os.path.join(image_base_dir, screenshot)

    if validate_images and not os.path.exists(full_img_path):
        return None

    # Get action and status
    action = step.get("action", {})
    status = step.get("status", "CONTINUE")

    if not action:
        return None

    # Build ID
    sample_id = f"{execution_id}_{step_id}"

    # Build conversation
    observation = step.get("observation", "")
    human_prompt = build_human_prompt(request, observation)
    gpt_response = build_gpt_response(action, status)

    conversation = [
        {"from": "human", "value": human_prompt},
        {"from": "gpt", "value": gpt_response}
    ]

    # Get bbox
    bbox = get_bbox_from_action(action)

    # Build result matching training format
    result = {
        "id": sample_id,
        "images": [full_img_path],
        "conversation": conversation,
        "reward": 1,  # Test data is successful trajectories
        "bbox": bbox
    }

    return result


def convert_to_parquet_format(sample: Dict) -> Dict:
    """Convert to parquet format (messages as JSON string)."""
    messages = []

    for turn in sample["conversation"]:
        role = "user" if turn["from"] == "human" else "assistant"
        value = turn["value"]

        if role == "user":
            # Process user turn - extract text and add image
            text = value.replace("<image>", "").strip()
            if text.startswith("\n"):
                text = text[1:]

            user_content = [
                {"type": "text", "text": text},
                {"type": "image", "image": sample["images"][0]}
            ]
            messages.append({"role": "user", "content": user_content})
        else:
            # Process assistant turn
            messages.append({"role": "assistant", "content": value})

    return {"messages": json.dumps(messages, ensure_ascii=False)}


def convert_test_to_parquet(
    test_dir: str,
    image_dir: str,
    output_file: str,
    max_samples: int = -1,
    validate_images: bool = False
) -> None:
    """
    Convert GUI-360 test trajectories to parquet format matching training data.
    """
    print(f"Scanning: {test_dir}", file=sys.stderr)

    # Find all JSONL files
    jsonl_files = glob.glob(os.path.join(test_dir, "**/*.jsonl"), recursive=True)
    print(f"Found {len(jsonl_files)} trajectory files", file=sys.stderr)

    samples = []
    skipped = 0

    for jsonl_file in jsonl_files:
        if max_samples > 0 and len(samples) >= max_samples:
            break

        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue

                    if max_samples > 0 and len(samples) >= max_samples:
                        break

                    trajectory = json.loads(line)
                    result = process_test_trajectory(
                        trajectory,
                        image_dir,
                        validate_images
                    )

                    if result:
                        parquet_sample = convert_to_parquet_format(result)
                        samples.append(parquet_sample)
                    else:
                        skipped += 1

        except Exception as e:
            print(f"Warning: Error processing {jsonl_file}: {e}", file=sys.stderr)
            continue

        if len(samples) % 5000 == 0 and len(samples) > 0:
            print(f"Processed {len(samples)} samples...", file=sys.stderr)

    print(f"Valid samples: {len(samples)}, Skipped: {skipped}", file=sys.stderr)

    if not samples:
        print("ERROR: No valid samples!", file=sys.stderr)
        return

    # Create output directory
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save to parquet
    df = pd.DataFrame(samples)
    df["messages"] = df["messages"].astype("object")
    df.to_parquet(output_file, index=False, engine="pyarrow")

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Conversion Complete", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Output: {output_file}", file=sys.stderr)
    print(f"Samples: {len(samples)}", file=sys.stderr)
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Convert GUI-360 test data to parquet format matching training data",
    )

    parser.add_argument(
        "--test-dir", "-t",
        type=str,
        required=True,
        help="Test data directory (e.g., datasets/GUI-360/test/data)"
    )
    parser.add_argument(
        "--image-dir", "-i",
        type=str,
        required=True,
        help="Test image directory (e.g., datasets/GUI-360/test/image)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output parquet file path"
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=-1,
        help="Maximum samples to process (-1 for all)"
    )
    parser.add_argument(
        "--validate-images",
        action="store_true",
        help="Validate that image files exist"
    )

    args = parser.parse_args()

    convert_test_to_parquet(
        test_dir=args.test_dir,
        image_dir=args.image_dir,
        output_file=args.output,
        max_samples=args.max_samples,
        validate_images=args.validate_images
    )


if __name__ == "__main__":
    main()
