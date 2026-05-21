#!/usr/bin/env python3
"""Convert balanced episode JSONL to ShareGPT conversation format.

Input: Episode-grouped JSONL (v12_gui_360/data/gui360_train_2000_balanced.jsonl)
Output: ShareGPT JSONL for LLaMA-Factory and train_cooperative.py

Each episode step becomes one ShareGPT conversation:
  - human: <image> + system prompt + instruction + history + action docs
  - gpt: <tool_call>action JSON</tool_call>
  - images: [screenshot_path]

History is built from GT actions of previous steps in the same episode.

Usage:
  python v15_gui_360/prepare_balanced_sharegpt.py \
      --train_input v12_gui_360/data/gui360_train_2000_balanced.jsonl \
      --val_input v12_gui_360/data/gui360_val_50.jsonl \
      --output_dir v15_gui_360/data
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional


# -- Prompt template (matches gui360_train.json format from LLaMA-Factory) ----

SYSTEM_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take."""

SUPPORTED_ACTIONS = """<action>
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
</action>"""

USER_PROMPT_TEMPLATE = """<image>
{system_prompt}

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

First, explain your reasoning process\u2014describe how you analyze the screenshot, understand the current state, and determine what action should be taken next based on the instruction and previous actions.

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

Only **ONE** action should be taken at a time. If the instruction could apply to multiple elements, choose the most relevant one based on the context provided by the screenshot and previous actions.
"""


def _valid_coord(c):
    return c is not None and len(c) >= 2 and c[0] is not None and c[1] is not None


def format_gt_action_for_history(gt_action: Dict, step_id: int) -> str:
    """Convert GT action dict to GUI-360 history format."""
    atype = gt_action.get("action", "")
    coord = gt_action.get("coordinate")

    if atype == "click":
        if _valid_coord(coord):
            return f"Step {step_id}: click(coordinate=[{int(coord[0])}, {int(coord[1])}])"
        return f"Step {step_id}: click()"

    elif atype == "type":
        text = gt_action.get("text", "")
        if _valid_coord(coord) and text:
            t = text[:30] + "..." if len(text) > 30 else text
            return f"Step {step_id}: type(coordinate=[{int(coord[0])}, {int(coord[1])}], keys='{t}')"
        elif text:
            t = text[:30] + "..." if len(text) > 30 else text
            return f"Step {step_id}: type(keys='{t}')"
        elif _valid_coord(coord):
            return f"Step {step_id}: type(coordinate=[{int(coord[0])}, {int(coord[1])}])"
        return f"Step {step_id}: type()"

    elif atype in ("swipe", "drag"):
        start = gt_action.get("coordinate")
        end = gt_action.get("endCoordinate")
        if _valid_coord(start) and _valid_coord(end):
            return (f"Step {step_id}: drag(start_coordinate=[{int(start[0])}, {int(start[1])}], "
                    f"end_coordinate=[{int(end[0])}, {int(end[1])}])")
        return f"Step {step_id}: drag()"

    else:
        return f"Step {step_id}: {atype}()"


def build_action_json(gt_action: Dict, is_last_step: bool) -> str:
    """Convert GT action dict to the tool_call JSON format."""
    atype = gt_action.get("action", "")

    # Map episode action types to model output format
    if atype == "swipe":
        func = "drag"
    else:
        func = atype

    args = {}
    coord = gt_action.get("coordinate")

    if atype == "click":
        if _valid_coord(coord):
            args["coordinate"] = [int(coord[0]), int(coord[1])]

    elif atype == "type":
        if _valid_coord(coord):
            args["coordinate"] = [int(coord[0]), int(coord[1])]
        text = gt_action.get("text", "")
        if text:
            args["text"] = text
            args["clear_current_text"] = gt_action.get("clear_current_text", True)

    elif atype in ("swipe", "drag"):
        start = gt_action.get("coordinate")
        end = gt_action.get("endCoordinate")
        if _valid_coord(start):
            args["start_coordinate"] = [int(start[0]), int(start[1])]
        if _valid_coord(end):
            args["end_coordinate"] = [int(end[0]), int(end[1])]

    status = "FINISH" if is_last_step else "CONTINUE"

    action_json = {
        "function": func,
        "args": args,
        "status": status,
    }
    return json.dumps(action_json, indent=4)


def convert_episode_to_sharegpt(episode: Dict) -> List[Dict]:
    """Convert one episode to a list of ShareGPT conversation dicts (one per step)."""
    goal = episode["goal"]
    steps = episode["steps"]
    num_steps = len(steps)

    conversations = []
    action_history: List[str] = []

    for si, step in enumerate(steps):
        screenshot = step["screenshot"]
        gt_action = step["action"]
        is_last_step = (si == num_steps - 1)

        # Build history text
        history_text = "\n".join(action_history) if action_history else ""

        # Build user prompt
        user_text = USER_PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            instruction=goal,
            history=history_text,
            actions=SUPPORTED_ACTIONS,
        )

        # Build assistant response
        action_json = build_action_json(gt_action, is_last_step)
        assistant_text = f"<tool_call>\n{action_json}\n</tool_call>"

        conv = {
            "conversations": [
                {"from": "human", "value": user_text},
                {"from": "gpt", "value": assistant_text},
            ],
            "images": [screenshot],
        }
        conversations.append(conv)

        # Update history for next step
        action_history.append(format_gt_action_for_history(gt_action, si + 1))

    return conversations


def process_file(input_path: str, output_path: str) -> int:
    """Process one JSONL file. Returns number of conversations written."""
    n_eps = 0
    n_convs = 0
    n_skipped = 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            episode = json.loads(line)
            n_eps += 1

            # Validate screenshots exist
            valid = True
            for step in episode.get("steps", []):
                if not os.path.exists(step.get("screenshot", "")):
                    valid = False
                    break

            if not valid or not episode.get("steps"):
                n_skipped += 1
                continue

            convs = convert_episode_to_sharegpt(episode)
            for conv in convs:
                fout.write(json.dumps(conv, ensure_ascii=False) + "\n")
                n_convs += 1

    return n_eps, n_convs, n_skipped


def main():
    parser = argparse.ArgumentParser(
        description="Convert balanced episode JSONL to ShareGPT format")
    parser.add_argument("--train_input", type=str, required=True,
                        help="Path to balanced training JSONL")
    parser.add_argument("--val_input", type=str, default="",
                        help="Path to validation JSONL")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for ShareGPT JSONL files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Process training data
    train_output = os.path.join(args.output_dir, "gui360_balanced_train.jsonl")
    print(f"Converting training data: {args.train_input}")
    n_eps, n_convs, n_skipped = process_file(args.train_input, train_output)
    print(f"  Episodes: {n_eps} ({n_skipped} skipped)")
    print(f"  Conversations written: {n_convs}")
    print(f"  Output: {train_output}")

    # Process validation data
    if args.val_input and os.path.exists(args.val_input):
        val_output = os.path.join(args.output_dir, "gui360_balanced_val.jsonl")
        print(f"\nConverting validation data: {args.val_input}")
        n_eps, n_convs, n_skipped = process_file(args.val_input, val_output)
        print(f"  Episodes: {n_eps} ({n_skipped} skipped)")
        print(f"  Conversations written: {n_convs}")
        print(f"  Output: {val_output}")

    # Verify output format
    print("\n--- Verification ---")
    with open(train_output) as f:
        first = json.loads(f.readline())
    print(f"Sample keys: {list(first.keys())}")
    print(f"Num conversations: {len(first['conversations'])}")
    print(f"Human message length: {len(first['conversations'][0]['value'])}")
    print(f"GPT message: {first['conversations'][1]['value'][:200]}")
    print(f"Images: {first['images']}")


if __name__ == "__main__":
    main()
