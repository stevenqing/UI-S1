#!/usr/bin/env python3
"""
Convert GUI-360 test trajectories to parquet format matching EVAL data format.

This script converts test data to the SAME format as the eval data,
ensuring consistency for proper evaluation comparison.

Key differences from previous test format:
1. Uses full action template (same as train/eval)
2. Includes reasoning instruction ("First, explain your reasoning process...")
3. Properly formats action history (even if empty)
4. Uses same output format instructions

Usage:
    python train_GUI_360/data_preparation/prepare_gui360_test_parquet_eval_format.py \
        --test-dir datasets/GUI-360/test/data \
        --image-dir datasets/GUI-360/test/image \
        --output train_GUI_360/data/gui360_test_sft_eval_format.parquet
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# Full action template matching training/eval data
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
    - button: str, The mouse button to drag. One of 'left', 'right'. (Default: 'left')
    - duration: float, The duration of the drag action in seconds. (Default: 1.0)
    - key_hold: str|None, The keyboard key to hold while dragging. Common keys include: shift (Shift), control (Ctrl), alt (Alt), etc. Use lowercase key names. For example, 'shift' for the shift key (Default: None)
  - Example: drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='left', duration=1.0, key_hold=None), drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='right', duration=1.0, key_hold='shift')
- wheel_mouse_input
  - Args:
    - coordinate: [x, y], the absolute position on the screen to scroll.
    - wheel_dist: int, The number of wheel notches to scroll. Positive values indicate upward scrolling, negative values indicate downward scrolling.
  - Example: wheel_mouse_input(coordinate=[100, 100], wheel_dist=-5), wheel_mouse_input(coordinate=[100, 100], wheel_dist=3)
- insert_table
  - Args:
    - rows: int, The number of rows in the table, starting from 1.
    - columns: int, The number of columns in the table, starting from 1.
  - Example: insert_table(rows=3, columns=3)
- select_text
  - Args:
    - text: str, The exact text to be selected.
  - Example: select_text(text="Hello")
- select_table
  - Args:
    - number: int, The index number of the table to be selected.
  - Example: select_table(number=1)
- select_paragraph
  - Args:
    - start_index: int, The start index of the paragraph to be selected.
    - end_index: int, The end index of the paragraph, if ==-1, select to the end of the document.
    - non_empty: bool, If True, select the non-empty paragraphs only. (Default: True)
  - Example: select_paragraph(start_index=1, end_index=3, non_empty=True)
- save_as
  - Args:
    - file_dir: str, The directory to save the file. If not specified, the current directory will be used. (Default: "")
    - file_name: str, The name of the file without extension. If not specified, the name of the current document will be used. (Default: "")
    - file_ext: str, The extension of the file. If not specified, the default extension is ".pdf". (Default: ".pdf")
  - Example: save_as(file_dir="", file_name="", file_ext=".pdf")
- set_font
  - Args:
    - font_name: str|None, The name of the font (e.g., "Arial", "Times New Roman", "宋体"). If None, the font name will not be changed. (Default: None)
    - font_size: int|None, The font size (e.g., 12). If None, the font size will not be changed. (Default: None)
  - Example: set_font(font_name="Times New Roman")
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
- summary
  - Args:
    - text: str, the complete summary of the task.
  - Example: summary(text='The task is completed.')
- set_background_color
  - Args:
    - color: str, The color to set as the background.
  - Example: set_background_color(color="red")
- set_focus
  - Args:
    - control_name: str, The name of the control to focus on.
  - Example: set_focus(control_name="Edit1")
- run_shell
  - Args:
    - command: str, The shell command to run.
  - Example: run_shell(command="ls -la")
</action>"""


def build_human_prompt(request: str, history: str = "") -> str:
    """Build human prompt matching eval data format."""
    prompt = f"""You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{request}

The history of actions are:
{history}

The actions supported are:
{ACTION_TEMPLATE}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

First, explain your reasoning process—describe how you analyze the screenshot, understand the current state, and determine what action should be taken next based on the instruction and previous actions.

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished, you can output status as "FINISH" and take no action. Like:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time. If the instruction could apply to multiple elements, choose the most relevant one based on the context provided by the screenshot and previous actions."""
    return prompt


def build_gpt_response(action: Dict[str, Any], status: str, bbox: Optional[Dict[str, int]] = None) -> str:
    """Build GPT response matching eval data format.

    Args:
        action: Action dictionary from raw data
        status: Status string (CONTINUE/FINISH/OVERALL_FINISH)
        bbox: Optional bounding box dict with left, top, right, bottom keys
    """
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
            start_x = action_args.get("start_x", action_args.get("startCoordinate", [coord_x])[0] if isinstance(action_args.get("startCoordinate"), list) else coord_x)
            start_y = action_args.get("start_y", action_args.get("startCoordinate", [0, coord_y])[1] if isinstance(action_args.get("startCoordinate"), list) else coord_y)
            end_x = action_args.get("end_x", action_args.get("endCoordinate", [None])[0] if isinstance(action_args.get("endCoordinate"), list) else None)
            end_y = action_args.get("end_y", action_args.get("endCoordinate", [0, None])[1] if isinstance(action_args.get("endCoordinate"), list) else None)

            if "startCoordinate" in action_args:
                args["startCoordinate"] = action_args["startCoordinate"]
            elif start_x is not None and start_y is not None:
                args["startCoordinate"] = [int(start_x), int(start_y)]

            if "endCoordinate" in action_args:
                args["endCoordinate"] = action_args["endCoordinate"]
            elif end_x is not None and end_y is not None:
                args["endCoordinate"] = [int(end_x), int(end_y)]

    elif function_name == "wheel_mouse_input":
        if coord_x is not None and coord_y is not None:
            args["coordinate"] = [int(coord_x), int(coord_y)]
        action_args = action.get("args", {})
        if isinstance(action_args, dict):
            if "wheel_dist" in action_args:
                args["wheel_dist"] = action_args["wheel_dist"]
            elif "direction" in action_args:
                # Map direction to wheel_dist
                direction = action_args["direction"]
                args["direction"] = direction

    elif function_name == "select_text":
        action_args = action.get("args", {})
        if isinstance(action_args, dict):
            if "text" in action_args:
                args["text"] = action_args["text"]
            elif "start_x" in action_args or "startCoordinate" in action_args:
                # Position-based select_text - keep the coordinates
                if "startCoordinate" in action_args:
                    args["startCoordinate"] = action_args["startCoordinate"]
                if "endCoordinate" in action_args:
                    args["endCoordinate"] = action_args["endCoordinate"]

    elif function_name == "summary":
        action_args = action.get("args", {})
        if isinstance(action_args, dict):
            args["text"] = action_args.get("text", "")

    elif function_name in ["insert_table", "select_table", "select_paragraph", "save_as", "set_font",
                           "table2markdown", "insert_excel_table", "select_table_range", "set_cell_value",
                           "set_background_color", "set_focus", "run_shell"]:
        # Copy all args for these function types
        action_args = action.get("args", {})
        if isinstance(action_args, dict):
            args.update(action_args)

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

    # Add bbox info for coordinate-based actions (for evaluation purposes)
    if bbox and function_name in ["click", "type", "drag", "wheel_mouse_input"]:
        response_dict["bbox"] = bbox

    response = f"<tool_call>\n{json.dumps(response_dict, indent=4, ensure_ascii=False)}\n</tool_call>"
    return response


def process_test_trajectory(
    trajectory: Dict[str, Any],
    image_base_dir: str,
    jsonl_file_path: str = "",
    validate_images: bool = False
) -> Optional[Dict]:
    """
    Process a raw GUI-360 test trajectory to match eval data format.

    Args:
        trajectory: The trajectory data
        image_base_dir: Base directory for images (e.g., datasets/GUI-360/test/image)
        jsonl_file_path: Path to the JSONL file (used to extract app_type and category)
        validate_images: Whether to validate image existence
    """
    execution_id = trajectory.get("execution_id", "")
    app_domain = trajectory.get("app_domain", "")
    request = trajectory.get("request", "")
    step_id = trajectory.get("step_id", 1)
    step = trajectory.get("step", {})

    if not request or not step:
        return None

    # Get screenshot path
    screenshot = step.get("screenshot_clean", step.get("screenshot_desktop", ""))
    if not screenshot:
        return None

    # Extract app_type and category from jsonl_file_path
    # jsonl_file_path format: test/data/ppt/in_app/success/ppt_1_42.jsonl
    # image path format: test/image/ppt/in_app/success/ppt_1_42/action_step1.png
    path_parts = Path(jsonl_file_path).parts
    app_type = None
    category = None

    # Find app_type (excel/word/ppt) and category (in_app/search/online/wikihow)
    for i, part in enumerate(path_parts):
        if part in ['excel', 'word', 'ppt'] and i + 1 < len(path_parts):
            app_type = part
            if path_parts[i + 1] in ['in_app', 'search', 'online', 'wikihow']:
                category = path_parts[i + 1]
                break

    # Build full image path with app_type and category
    # screenshot format: "success/ppt_1_42/action_step1.png"
    if app_type and category:
        # Remove leading "success/" from screenshot if present
        if screenshot.startswith("success/"):
            screenshot = screenshot[8:]  # Remove "success/"
        full_img_path = os.path.join(image_base_dir, app_type, category, "success", screenshot)
    else:
        # Fallback to old method
        full_img_path = os.path.join(image_base_dir, screenshot)

    if validate_images and not os.path.exists(full_img_path):
        return None

    # Get action and status
    action = step.get("action", {})
    status = step.get("status", "CONTINUE")

    if not action:
        return None

    # Build history from thought if available
    thought = step.get("thought", "")
    history = ""
    if thought:
        # Use thought as history context
        history = f"Step {step_id}: {thought}"

    # Build human prompt (matching eval format)
    human_prompt = build_human_prompt(request, history)

    # Extract bbox from action for evaluation
    bbox = None
    rectangle = action.get("rectangle", action.get("desktop_rectangle", {}))
    if rectangle and isinstance(rectangle, dict):
        bbox = {
            "left": rectangle.get("left"),
            "top": rectangle.get("top"),
            "right": rectangle.get("right"),
            "bottom": rectangle.get("bottom")
        }
        # Filter out None values
        if any(v is None for v in bbox.values()):
            bbox = None

    # Build GPT response (with bbox for evaluation)
    gpt_response = build_gpt_response(action, status, bbox)

    # Build messages in parquet format
    user_content = [
        {"text": human_prompt},
        {"image": full_img_path}
    ]

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": gpt_response}
    ]

    return {"messages": json.dumps(messages, ensure_ascii=False)}


def convert_test_to_parquet(
    test_dir: str,
    image_dir: str,
    output_file: str,
    max_samples: int = -1,
    validate_images: bool = False
) -> None:
    """
    Convert GUI-360 test trajectories to parquet format matching eval data.
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
                        jsonl_file,
                        validate_images
                    )

                    if result:
                        samples.append(result)
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
        description="Convert GUI-360 test data to parquet format matching eval data",
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
