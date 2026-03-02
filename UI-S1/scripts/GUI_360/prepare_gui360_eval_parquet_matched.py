#!/usr/bin/env python3
"""
Convert GUI-360 raw test trajectories to parquet format for evaluation.

IMPORTANT: This script produces output in the SAME FORMAT as the training data
(from action_prediction_train_resize) to ensure train/val format consistency.

Key differences from prepare_gui360_eval_parquet.py:
1. Uses the same prompt template with full action schema
2. Uses <tool_call> wrapper for responses
3. Uses "function" key instead of "action"
4. Includes "status" field

Usage:
    python scripts/GUI_360/prepare_gui360_eval_parquet_matched.py \
        --input-dir datasets/GUI-360/test/data \
        --output train_GUI_360/data/gui360_eval_sft.parquet
"""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, Optional

import pandas as pd


# Action schema matching training data format exactly
ACTION_SCHEMA = '''The actions supported are:
<action>
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
    - button: str, The mouse button to drag. One of 'left', 'right'. (Default: 'left')
    - duration: float, The duration of the drag action in seconds. (Default: 1.0)
    - key_hold: str|None, The keyboard key to hold while dragging. Common keys include: shift (Shift), control (Ctrl), alt (Alt), etc. Use lowercase key names. For example, 'shift' for the shift key (Default: None)
  - Example: drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='left', duration=1.0, key_hold=None), drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='right', duration=1.0, key_hold='shift')
- wheel_mouse_input
  - Args:
    - coordinate: [x, y], the absolute position on the screen to scroll.
    - wheel_dist: int, The number of wheel notches to scroll. Positive values indicate upward scrolling, negative values indicate downward scrolling.
  - Example: wheel_mouse_input(coordinate=[100, 100], wheel_dist=-5), wheel_mouse_input(coordinate=[100, 100], wheel_dist=3)
- table2markdown
  - Args:
    - sheet_name: str|int, The name or index of the sheet to get the table content. The index starts from 1.
  - Example: table2markdown(sheet_name=1)
- insert_excel_table
  - Args:
    - table: list[list], The table content to insert. The table is a list of list of strings or numbers.
    - sheet_name: str, The name of the sheet to insert the table.
    - start_row: int, The start row to insert the table, starting from 1.
    - start_col: int, The start column to insert the table, starting from 1.
  - Example: insert_excel_table(table=[["Name", "Age", "Gender"], ["Alice", 30, "Female"], ["Bob", 25, "Male"], ["Charlie", 35, "Male"]], sheet_name="Sheet1", start_row=1, start_col=1)
- select_table_range
  - Args:
    - sheet_name: str, The name of the sheet.
    - start_row: int, The start row, starting from 1.
    - start_col: int, The start column, starting from 1.
    - end_row: int, The end row. If ==-1, select to the end of the document with content.
    - end_col: int, The end column. If ==-1, select to the end of the document with content.
  - Example: select_table_range(sheet_name="Sheet1", start_row=1, start_col=1, end_row=3, end_col=3)
- set_cell_value
  - Args:
    - sheet_name: str, The name of the sheet.
    - row: int, The row number (1-based).
    - col: int, The column number (1-based).
    - value: str|int|float|None, The value to set in the cell. If None, just select the cell.
    - is_formula: bool, If True, treat the value as a formula, otherwise treat it as a normal value. (Default: False)
  - Example: set_cell_value(sheet_name="Sheet1", row=1, col=1, value="Hello", is_formula=False), set_cell_value(sheet_name="Sheet1", row=2, col=2, value="=SUM(A1:A10)", is_formula=True)
- auto_fill
  - Args:
    - sheet_name: str, The name of the sheet.
    - start_row: int, The starting row number (1-based).
    - start_col: int, The starting column number (1-based).
    - end_row: int, The ending row number (1-based).
    - end_col: int, The ending column number (1-based).
  - Example: auto_fill(sheet_name="Sheet1", start_row=1, start_col=1, end_row=10, end_col=3)
- reorder_columns
  - Args:
    - sheet_name: str, The name of the sheet.
    - desired_order: list[str], The list of column names in the new order.
  - Example: reorder_columns(sheet_name="Sheet1", desired_order=["Income", "Date", "Expense"])
</action>
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

First, explain your reasoning process—describe how you analyze the screenshot, understand the current state, and determine what action should be taken next based on the instruction and previous actions.

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{
  "function": "<function name>",
  "args": {},
  "status": "CONTINUE"
}
</tool_call>

If you think the task is finished, you can output status as "FINISH" and take no action. Like:
<tool_call>
{
  "function": "",
  "args": {},
  "status": "FINISH"
}
</tool_call>

Only **ONE** action should be taken at a time. If the instruction could apply to multiple elements, choose the most relevant one based on the context provided by the screenshot and previous actions.
'''


def build_prompt(request: str, history: str = "") -> str:
    """Build prompt matching training data format."""
    prompt = f"""You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{request}

The history of actions are:
{history}

{ACTION_SCHEMA}"""
    return prompt


def format_action_response(action: Dict[str, Any], status: str = "CONTINUE") -> str:
    """Format action to tool_call response string matching training format."""
    action_type = action.get("function", "click")

    # Build args dict based on action type
    args = {}

    # Get coordinates
    coord_x = action.get("coordinate_x") or action.get("desktop_coordinate_x")
    coord_y = action.get("coordinate_y") or action.get("desktop_coordinate_y")

    if coord_x is not None and coord_y is not None:
        args["coordinate"] = [int(coord_x), int(coord_y)]

    # Handle type action
    if action_type == "type":
        action_args = action.get("args", {})
        if isinstance(action_args, dict):
            if "keys" in action_args:
                args["keys"] = action_args["keys"]
            if "text" in action_args:
                args["text"] = action_args["text"]
            if "clear_current_text" in action_args:
                args["clear_current_text"] = action_args["clear_current_text"]
        if "control_text" in action and "keys" not in args and "text" not in args:
            args["text"] = action["control_text"]

    # Handle click action
    elif action_type == "click":
        action_args = action.get("args", {})
        if isinstance(action_args, dict):
            if "button" in action_args:
                args["button"] = action_args["button"]
            if "double" in action_args:
                args["double"] = action_args["double"]

    # Handle drag action
    elif action_type == "drag":
        action_args = action.get("args", {})
        if isinstance(action_args, dict):
            if "start_coordinate" in action_args:
                args["start_coordinate"] = action_args["start_coordinate"]
            if "end_coordinate" in action_args:
                args["end_coordinate"] = action_args["end_coordinate"]

    # Handle scroll action
    elif action_type == "wheel_mouse_input":
        action_args = action.get("args", {})
        if isinstance(action_args, dict) and "wheel_dist" in action_args:
            args["wheel_dist"] = action_args["wheel_dist"]

    # Build response matching training format
    response = {
        "function": action_type,
        "args": args,
        "status": status
    }

    # Format with tool_call wrapper and indentation matching training data
    response_json = json.dumps(response, indent=4, ensure_ascii=False)
    return f"<tool_call>\n{response_json}\n</tool_call>"


def process_trajectory(
    trajectory: Dict[str, Any],
    base_path: str,
    validate_images: bool = False
) -> Optional[Dict]:
    """Process a raw GUI-360 trajectory to message format matching training data."""
    request = trajectory.get("request", "")
    step = trajectory.get("step", {})

    if not request or not step:
        return None

    # Get screenshot path
    screenshot = step.get("screenshot_clean") or step.get("screenshot_desktop", "")
    if not screenshot:
        return None

    full_img_path = os.path.join(base_path, screenshot)

    if validate_images and not os.path.exists(full_img_path):
        return None

    # Build prompt matching training format
    prompt = build_prompt(request, history="")

    # Build user message with image
    user_content = [
        {"type": "text", "text": prompt},
        {"type": "image", "image": full_img_path}
    ]

    messages = [{"role": "user", "content": user_content}]

    # Build assistant response in tool_call format
    action = step.get("action", {})
    if not action:
        return None

    # Determine status
    status_val = step.get("status", "CONTINUE")
    if status_val not in ["CONTINUE", "FINISH", "OVERALL_FINISH"]:
        status_val = "CONTINUE"
    if status_val == "OVERALL_FINISH":
        status_val = "FINISH"

    response = format_action_response(action, status_val)
    messages.append({"role": "assistant", "content": response})

    return {"messages": json.dumps(messages, ensure_ascii=False)}


def convert_to_parquet(
    input_dir: str,
    output_file: str,
    max_samples: int = -1,
    validate_images: bool = False
) -> None:
    """Convert GUI-360 raw test trajectories to parquet format."""
    print(f"Scanning: {input_dir}", file=sys.stderr)

    # Find all JSONL files
    jsonl_files = glob.glob(os.path.join(input_dir, "**/*.jsonl"), recursive=True)
    print(f"Found {len(jsonl_files)} trajectory files", file=sys.stderr)

    samples = []
    skipped = 0

    for jsonl_file in jsonl_files:
        if max_samples > 0 and len(samples) >= max_samples:
            break

        base_path = os.path.dirname(jsonl_file)

        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue

                    if max_samples > 0 and len(samples) >= max_samples:
                        break

                    trajectory = json.loads(line)
                    result = process_trajectory(trajectory, base_path, validate_images)

                    if result:
                        samples.append(result)
                    else:
                        skipped += 1

        except Exception as e:
            print(f"Warning: Error processing {jsonl_file}: {e}", file=sys.stderr)
            continue

        if len(samples) % 1000 == 0 and len(samples) > 0:
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
    print(f"Conversion Complete (Format Matched to Training Data)", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Output: {output_file}", file=sys.stderr)
    print(f"Samples: {len(samples)}", file=sys.stderr)
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Convert GUI-360 raw test data to parquet format (matching training format)",
    )

    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        required=True,
        help="Input directory containing JSONL trajectory files"
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

    convert_to_parquet(
        input_dir=args.input_dir,
        output_file=args.output,
        max_samples=args.max_samples,
        validate_images=args.validate_images
    )


if __name__ == "__main__":
    main()
