#!/usr/bin/env python3
"""
Prepare GUI-360 evaluation data in A11y format.

This script creates evaluation data that:
1. Uses annotated screenshots with Set-of-Mark (SoM) labels
2. Includes element list in the prompt
3. Uses element_id in ground truth for evaluation

Usage:
    python prepare_gui360_eval_a11y.py \
        --input-dir datasets/GUI-360/test/data \
        --output train_GUI_360/data/gui360_test_a11y_eval.parquet \
        --image-base-dir datasets/GUI-360/test/image
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional
from pathlib import Path

import pandas as pd


# System prompt with a11y element list support (same as training)
SYSTEM_PROMPT_A11Y = """You are a helpful assistant. Given a screenshot of the current screen with labeled elements, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The elements on the screen are labeled with numbers. Here is the list of interactive elements:
{element_list}

The actions supported are:
<action>
- click
  - Args:
    - element_id: int, the label number of the element to click (from the element list above)
    - button: str, The mouse button to click. One of 'left', 'right', 'middle' or 'x' (Default: 'left')
    - double: bool, Whether to perform a double click or not (Default: False)
  - Example: click(element_id=1, button='left', double=False), click(element_id=5, button='x')
- type
  - Args:
    - element_id: int, the label number of the element to type into (from the element list above)
    - text: str, The text to input.
    - clear_current_text: bool, Whether to clear the current text before typing. (Default: False)
  - Example: type(element_id=3, text='Hello'), type(element_id=10, text='World', clear_current_text=True)
- drag
  - Args:
    - start_element_id: int, the label number of the element to start dragging from
    - end_element_id: int, the label number of the element to end dragging at
    - button: str, The mouse button to drag. One of 'left', 'right'. (Default: 'left')
    - duration: float, The duration of the drag action in seconds. (Default: 1.0)
  - Example: drag(start_element_id=1, end_element_id=5, button='left', duration=1.0)
- wheel_mouse_input
  - Args:
    - element_id: int, the label number of the element to scroll on
    - wheel_dist: int, The number of wheel notches to scroll. Positive for up, negative for down.
  - Example: wheel_mouse_input(element_id=2, wheel_dist=-5)
- hotkey
  - Args:
    - keys: str, The hotkey combination. Use format like "Ctrl+C", "Alt+F4", "Ctrl+Shift+S"
  - Example: hotkey(keys="Ctrl+C"), hotkey(keys="Ctrl+Shift+S")
</action>

Important: Use element_id from the labeled elements in the screenshot, NOT pixel coordinates.

First, explain your reasoning process—describe how you analyze the screenshot, understand the current state, and determine what action should be taken next based on the instruction and previous actions.

Then output your action within <execute> tag like:
<execute>
{{
  "function": "<function name>",
  "args": {{"element_id": <id>, ...}},
  "status": "CONTINUE"
}}
</execute>

If the think the task is finished, output:
<execute>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</execute>

Only **ONE** action should be taken at a time."""


def format_element_list(controls: List[Dict]) -> str:
    """Format control list for prompt."""
    lines = []
    for ctrl in controls:
        label = ctrl.get('label', '?')
        text = ctrl.get('control_text', '') or ctrl.get('text', '')
        ctrl_type = ctrl.get('control_type', 'Unknown')
        rect = ctrl.get('control_rect', ctrl.get('rectangle', []))

        # Format: [1] Button "Save" (position: 100,200)
        if rect and len(rect) >= 4:
            center_x = (rect[0] + rect[2]) // 2
            center_y = (rect[1] + rect[3]) // 2
            lines.append(f"[{label}] {ctrl_type} \"{text}\" (position: {center_x},{center_y})")
        else:
            lines.append(f"[{label}] {ctrl_type} \"{text}\"")

    return "\n".join(lines)


def find_element_id_for_coordinate(controls: List[Dict], coordinate: List[float], tolerance: int = 20) -> Optional[int]:
    """Find the element ID that contains or is closest to the given coordinate."""
    if not coordinate or len(coordinate) < 2:
        return None

    target_x, target_y = coordinate[0], coordinate[1]

    # Skip if coordinates are None
    if target_x is None or target_y is None:
        return None

    best_match = None
    best_distance = float('inf')

    for ctrl in controls:
        label = ctrl.get('label')
        rect = ctrl.get('control_rect', ctrl.get('rectangle', []))

        if not label or not rect or len(rect) < 4:
            continue

        left, top, right, bottom = rect

        # Skip if any rect value is None
        if any(v is None for v in [left, top, right, bottom]):
            continue

        center_x = (left + right) / 2
        center_y = (top + bottom) / 2

        # Check if coordinate is inside the element
        if left - tolerance <= target_x <= right + tolerance and top - tolerance <= target_y <= bottom + tolerance:
            distance = abs(center_x - target_x) + abs(center_y - target_y)
            if distance < best_distance:
                best_distance = distance
                best_match = label

    return best_match


def extract_coordinate_from_action(action: Dict) -> Optional[List[float]]:
    """Extract coordinate from action in various formats."""
    args = action.get('args', {})

    if 'coordinate' in args and args['coordinate']:
        return args['coordinate']

    if 'coordinate_x' in action and 'coordinate_y' in action:
        return [action['coordinate_x'], action['coordinate_y']]

    if 'x' in action and 'y' in action:
        return [action['x'], action['y']]

    if 'start_coordinate' in args:
        return args['start_coordinate']

    return None


def get_element_bbox(controls: List[Dict], element_id: int) -> Optional[List[int]]:
    """Get bounding box for an element by its ID."""
    for ctrl in controls:
        if ctrl.get('label') == element_id:
            rect = ctrl.get('control_rect', ctrl.get('rectangle', []))
            if rect and len(rect) >= 4:
                return rect
    return None


def convert_action_to_a11y_format(action: Dict, controls: List[Dict]) -> Dict:
    """Convert action from coordinate-based to element_id-based format."""
    function = action.get('function', '')
    args = action.get('args', {})
    status = action.get('status', 'CONTINUE')

    new_args = {}
    element_bbox = None

    if function == 'click':
        coord = extract_coordinate_from_action(action)
        element_id = find_element_id_for_coordinate(controls, coord)
        if element_id:
            new_args['element_id'] = element_id
            element_bbox = get_element_bbox(controls, element_id)
        elif coord:
            new_args['coordinate'] = coord
        if args.get('button') and args['button'] != 'left':
            new_args['button'] = args['button']
        if args.get('double'):
            new_args['double'] = True

    elif function == 'type':
        coord = extract_coordinate_from_action(action)
        element_id = find_element_id_for_coordinate(controls, coord)
        if element_id:
            new_args['element_id'] = element_id
            element_bbox = get_element_bbox(controls, element_id)
        elif coord:
            new_args['coordinate'] = coord
        new_args['text'] = args.get('text', args.get('keys', ''))
        if args.get('clear_current_text'):
            new_args['clear_current_text'] = True

    elif function == 'drag':
        start_coord = args.get('start_coordinate')
        end_coord = args.get('end_coordinate')
        start_id = find_element_id_for_coordinate(controls, start_coord)
        end_id = find_element_id_for_coordinate(controls, end_coord)

        if start_id:
            new_args['start_element_id'] = start_id
        elif start_coord:
            new_args['start_coordinate'] = start_coord
        if end_id:
            new_args['end_element_id'] = end_id
        elif end_coord:
            new_args['end_coordinate'] = end_coord
        if args.get('button') and args['button'] != 'left':
            new_args['button'] = args['button']
        if args.get('duration') and args['duration'] != 1.0:
            new_args['duration'] = args['duration']

    elif function == 'wheel_mouse_input':
        coord = extract_coordinate_from_action(action)
        element_id = find_element_id_for_coordinate(controls, coord)
        if element_id:
            new_args['element_id'] = element_id
            element_bbox = get_element_bbox(controls, element_id)
        elif coord:
            new_args['coordinate'] = coord
        new_args['wheel_dist'] = args.get('wheel_dist', 0)

    else:
        new_args = args

    result = {
        "function": function,
        "args": new_args,
        "status": status
    }

    # Include bbox for evaluation
    if element_bbox:
        result['bbox'] = element_bbox

    return result


def process_trajectory_file(
    jsonl_path: str,
    image_base_dir: str,
    use_annotated_screenshot: bool = True
) -> List[Dict]:
    """Process a single trajectory JSONL file and return evaluation samples."""
    samples = []

    # Extract app and category from jsonl_path
    # Path format: .../test/data/{app}/{category}/.../{filename}.jsonl
    path_parts = Path(jsonl_path).parts
    app = None
    category = None
    for i, part in enumerate(path_parts):
        if part in ['word', 'excel', 'ppt'] and i + 1 < len(path_parts):
            app = part
            category = path_parts[i + 1]
            break

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue

        step = data.get('step', {})
        controls = step.get('control_infos', {}).get('merged_controls_info', [])

        if not controls:
            continue

        # Get screenshot path
        if use_annotated_screenshot:
            screenshot_path = step.get('screenshot_annotated', step.get('screenshot_clean', ''))
        else:
            screenshot_path = step.get('screenshot_clean', '')

        if not screenshot_path:
            continue

        # Build full image path: image_base_dir/{app}/{category}/{screenshot_path}
        if app and category:
            full_image_path = os.path.join(image_base_dir, app, category, screenshot_path)
        else:
            full_image_path = os.path.join(image_base_dir, screenshot_path)

        # Format element list
        element_list = format_element_list(controls)

        # Get instruction and history
        instruction = data.get('request', '')
        history = ""  # Could include previous actions if available

        # Create user message
        user_message = SYSTEM_PROMPT_A11Y.format(
            instruction=instruction,
            history=history if history else "(none)",
            element_list=element_list
        )

        # Convert action to a11y format (ground truth)
        action = step.get('action', {})
        a11y_action = convert_action_to_a11y_format(action, controls)

        # Create message format (only user message for evaluation)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {"type": "image", "image": full_image_path}
                ]
            }
        ]

        samples.append({
            "messages": json.dumps(messages, ensure_ascii=False),
            "ground_truth": json.dumps(a11y_action, ensure_ascii=False),
            "trajectory_id": data.get('execution_id', ''),
            "step_id": data.get('step_id', 0),
            "instruction": instruction,
            "element_count": len(controls)
        })

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Prepare GUI-360 evaluation data in A11y format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        required=True,
        help="Input directory containing trajectory JSONL files"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output parquet file path"
    )
    parser.add_argument(
        "--image-base-dir",
        type=str,
        required=True,
        help="Base directory for image paths"
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=-1,
        help="Maximum samples to process (-1 for all)"
    )
    parser.add_argument(
        "--use-clean-screenshot",
        action="store_true",
        help="Use clean screenshots instead of annotated ones"
    )
    parser.add_argument(
        "--apps",
        type=str,
        nargs="+",
        default=["word", "excel", "ppt"],
        help="Applications to include (word, excel, ppt)"
    )

    args = parser.parse_args()

    all_samples = []
    input_dir = Path(args.input_dir)

    # Find all JSONL files
    jsonl_files = []
    for app in args.apps:
        app_dir = input_dir / app
        if app_dir.exists():
            for jsonl_file in app_dir.rglob("*.jsonl"):
                jsonl_files.append(str(jsonl_file))

    print(f"Found {len(jsonl_files)} trajectory files", file=sys.stderr)

    # Process files
    processed = 0
    for i, jsonl_path in enumerate(jsonl_files):
        if args.max_samples > 0 and len(all_samples) >= args.max_samples:
            break

        samples = process_trajectory_file(
            jsonl_path,
            args.image_base_dir,
            use_annotated_screenshot=not args.use_clean_screenshot
        )
        all_samples.extend(samples)

        processed += 1
        if (processed) % 100 == 0:
            print(f"Processed {processed}/{len(jsonl_files)} files, {len(all_samples)} samples", file=sys.stderr)

    print(f"Total samples: {len(all_samples)}", file=sys.stderr)

    if not all_samples:
        print("ERROR: No samples generated!", file=sys.stderr)
        return

    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save to parquet
    df = pd.DataFrame(all_samples)
    df["messages"] = df["messages"].astype("object")
    df["ground_truth"] = df["ground_truth"].astype("object")
    df.to_parquet(args.output, index=False, engine="pyarrow")

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Conversion Complete", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)
    print(f"Samples: {len(all_samples)}", file=sys.stderr)
    print(f"File size: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB", file=sys.stderr)


if __name__ == "__main__":
    main()
