#!/usr/bin/env python3
"""
Convert GUI-360 JSONL trajectory data to SFT eval parquet format with multi-step data.

Creates a parquet with samples at different step positions (step 0, 1, 2, ...)
for D2 attention diagnostic (step position analysis).

Input: gui360_test.jsonl (trajectory format)
Output: gui360_test_multistep_eval.parquet (SFT messages format)
"""

import json
import os
import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

JSONL_PATH = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/GUI-360/rl_data/gui360_test.jsonl"
OUTPUT_PATH = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/data/gui360_test_multistep_eval.parquet"

# Action space template (same as in existing parquet)
ACTION_SPACE = """<action>
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

Only **ONE** action should be taken at a time. If the instruction could apply to multiple elements, choose the most relevant one based on the context provided by the screenshot and previous actions."""


def action_content_to_tool_call(action_content):
    """Convert JSONL action_content dict to tool_call format string."""
    action = action_content.get("action", "")
    if not action:
        return None

    args = {}
    coord = action_content.get("coordinate")
    if coord is not None and all(c is not None for c in coord):
        args["coordinate"] = [int(c) if c == int(c) else c for c in coord]

    text = action_content.get("text")
    if text is not None:
        if action == "type":
            args["keys"] = text
        else:
            args["text"] = text

    button = action_content.get("button")
    if button is not None:
        args["button"] = button

    coord2 = action_content.get("coordinate2")
    if coord2 is not None and all(c is not None for c in coord2):
        args["end_coordinate"] = [int(c) if c == int(c) else c for c in coord2]
        if "coordinate" in args:
            args["start_coordinate"] = args.pop("coordinate")

    status = action_content.get("status")
    if status is None:
        status = "CONTINUE"

    tool_call = {
        "function": action,
        "args": args,
        "status": status,
    }

    return "<tool_call>\n" + json.dumps(tool_call, indent=4) + "\n</tool_call>"


def build_history_text(steps, up_to_step):
    """Build history text for steps 0..up_to_step-1."""
    if up_to_step == 0:
        return ""

    lines = []
    for i in range(up_to_step):
        thought = steps[i].get("thought", "")
        lines.append(f"Step {i + 1}: {thought}")

    return "\n".join(lines)


def build_message(goal, history_text, screenshot_path):
    """Build the SFT message format."""
    system_text = "You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take."

    if history_text:
        text = f"""{system_text}

The instruction is:
{goal}

The history of actions are:
{history_text}

The actions supported are:
{ACTION_SPACE}"""
    else:
        text = f"""{system_text}

The instruction is:
{goal}

The history of actions are:
(No previous actions)

The actions supported are:
{ACTION_SPACE}"""

    user_msg = {
        "role": "user",
        "content": [
            {"text": text},
            {"image": screenshot_path},
        ],
    }
    return user_msg


def main():
    print(f"Loading trajectories from {JSONL_PATH}")
    trajectories = []
    with open(JSONL_PATH) as f:
        for line in f:
            trajectories.append(json.loads(line))
    print(f"Loaded {len(trajectories)} trajectories")

    # Build multi-step samples
    records = []
    n_skipped = 0
    for traj in trajectories:
        goal = traj["goal"]
        steps = traj["steps"]

        for step_idx in range(len(steps)):
            step = steps[step_idx]
            screenshot = step["screenshot"]

            # Check screenshot exists
            if not os.path.exists(screenshot):
                n_skipped += 1
                continue

            # Build GT action
            gt_tool_call = action_content_to_tool_call(step["action_content"])
            if gt_tool_call is None:
                n_skipped += 1
                continue

            # Build history
            history_text = build_history_text(steps, step_idx)

            # Build message
            user_msg = build_message(goal, history_text, screenshot)
            assistant_msg = {"role": "assistant", "content": gt_tool_call}

            messages = [user_msg, assistant_msg]
            records.append({"messages": json.dumps(messages)})

    print(f"Created {len(records)} samples ({n_skipped} skipped)")

    # Analyze step distribution
    from collections import Counter
    step_counts = Counter()
    for rec in records:
        msgs = json.loads(rec["messages"])
        text = ""
        for item in msgs[0]["content"]:
            if isinstance(item, dict) and "text" in item:
                text = item["text"]
        n_steps = len(re.findall(r"Step \d+:", text))
        step_counts[n_steps] += 1

    print("\nStep distribution:")
    for k in sorted(step_counts.keys()):
        print(f"  {k} history steps: {step_counts[k]} samples")

    # Save
    df = pd.DataFrame(records)
    df.to_parquet(OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Total samples: {len(df)}")


if __name__ == "__main__":
    main()
