"""Verifier-guided AR inference with probe-based temperature selection.

Standalone script that runs AR inference with HF Transformers (not vLLM) to access
intermediate hidden states. Uses an L26 correctness probe to guide temperature:
  - "verifier": temp=0 if P(correct)>0.5, else temp=1.0
  - "always_greedy": temp=0 always (baseline control)
  - "always_temp1": temp=1.0 always (stochastic control)

Reuses data loading, action parsing, and comparison logic from the eval framework
but does NOT import from it (to avoid vLLM dependencies).
"""

import argparse
import json
import os
import re
import time
import traceback
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Supported actions per domain (copied from prompts/prompt_action_prediction.py)
# ---------------------------------------------------------------------------
SUPPORTED_ACTIONS_WORD = """<action>
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
    - key_hold: str|None, The keyboard key to hold while dragging. (Default: None)
  - Example: drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='left', duration=1.0, key_hold=None)
- wheel_mouse_input
  - Args:
    - coordinate: [x, y], the absolute position on the screen to scroll.
    - wheel_dist: int, The number of wheel notches to scroll.
  - Example: wheel_mouse_input(coordinate=[100, 100], wheel_dist=-5)
- insert_table
  - Args: rows: int, columns: int
- select_text
  - Args: text: str
- select_table
  - Args: number: int
- select_paragraph
  - Args: start_index: int, end_index: int, non_empty: bool (Default: True)
- save_as
  - Args: file_dir: str, file_name: str, file_ext: str (Default: ".pdf")
- set_font
  - Args: font_name: str|None, font_size: int|None
</action>"""

SUPPORTED_ACTIONS_EXCEL = """<action>
- click
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to click at.
    - button: str, The mouse button to click. One of ''left'', ''right'', ''middle'' or ''x'' (Default: ''left'')
    - double: bool, Whether to perform a double click or not (Default: False)'
    - pressed: str|None, The keyboard key to press while clicking. (Default: None)
  - Example: click(coordinate=[100, 100], button='left', double=False, pressed=None)
- type
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to type at.
    - keys: str
    - clear_current_text: bool (Default: False)
    - control_focus: bool (Default: True)
  - Example: type(coordinate=[100, 100], keys='Hello')
- drag
  - Args:
    - start_coordinate: [x, y], end_coordinate: [x, y]
    - button: str (Default: 'left'), duration: float (Default: 1.0), key_hold: str|None (Default: None)
- wheel_mouse_input
  - Args: coordinate: [x, y], wheel_dist: int
- table2markdown
  - Args: sheet_name: str|int
- insert_excel_table
  - Args: table: list[list], sheet_name: str, start_row: int, start_col: int
- select_table_range
  - Args: sheet_name: str, start_row: int, start_col: int, end_row: int, end_col: int
- set_cell_value
  - Args: sheet_name: str, row: int, col: int, value: str|int|float|None, is_formula: bool (Default: False)
- auto_fill
  - Args: sheet_name: str, start_row: int, start_col: int, end_row: int, end_col: int
- reorder_columns
  - Args: sheet_name: str, desired_order: list[str]
</action>"""

SUPPORTED_ACTIONS_PPT = """<action>
- click
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to click at.
    - button: str, The mouse button to click. One of ''left'', ''right'', ''middle'' or ''x'' (Default: ''left'')
    - double: bool, Whether to perform a double click or not (Default: False)'
    - pressed: str|None, The keyboard key to press while clicking. (Default: None)
  - Example: click(coordinate=[100, 100], button='left', double=False, pressed=None)
- type
  - Args:
    - coordinate: [x, y], keys: str
    - clear_current_text: bool (Default: False), control_focus: bool (Default: True)
- drag
  - Args:
    - start_coordinate: [x, y], end_coordinate: [x, y]
    - button: str (Default: 'left'), duration: float (Default: 1.0), key_hold: str|None (Default: None)
- wheel_mouse_input
  - Args: coordinate: [x, y], wheel_dist: int
- set_background_color
  - Args: color: str, slide_index: list[int]|None
- save_as
  - Args: file_dir: str, file_name: str, file_ext: str (Default: ".pptx"), current_slide_only: bool (Default: False)
</action>"""


# ---------------------------------------------------------------------------
# Subtask isolated prompt template (from action_prediction_ar_context.py)
# ---------------------------------------------------------------------------
SUBTASK_ISOLATED_USER_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

Overall Task:
{instruction}

Current Subtask:
{subtask_description}

The history of actions are:
{history}

The actions supported are:
{actions}
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

Only **ONE** action should be taken at a time. If the instruction could apply to multiple elements, choose the most relevant one based on the context provided by the screenshot and previous actions.
"""


# ---------------------------------------------------------------------------
# Tool defaults for argument normalization (from evaluator/tool_definitions.py)
# ---------------------------------------------------------------------------
TOOL_DEFAULTS = {
    "click": {"coordinate": [0, 0], "button": "left", "double": False, "pressed": None},
    "type": {"coordinate": [0, 0], "keys": "", "clear_current_text": False, "control_focus": True},
    "drag": {"start_coordinate": [0, 0], "end_coordinate": [0, 0], "button": "left", "duration": 1.0, "key_hold": None},
    "wheel_mouse_input": {"coordinate": [0, 0], "wheel_dist": 0},
    "insert_table": {"rows": 1, "columns": 1},
    "select_text": {"text": ""},
    "select_table": {"number": 1},
    "select_paragraph": {"start_index": 1, "end_index": -1, "non_empty": True},
    "save_as": {"file_dir": "", "file_name": "", "file_ext": ".pdf"},
    "set_font": {"font_name": None, "font_size": None},
    "table2markdown": {"sheet_name": 1},
    "insert_excel_table": {"table": [], "sheet_name": "Sheet1", "start_row": 1, "start_col": 1},
    "select_table_range": {"sheet_name": "Sheet1", "start_row": 1, "start_col": 1, "end_row": -1, "end_col": -1},
    "set_cell_value": {"sheet_name": "Sheet1", "row": 1, "col": 1, "value": None, "is_formula": False},
    "auto_fill": {"sheet_name": "Sheet1", "start_row": 1, "start_col": 1, "end_row": 1, "end_col": 1},
    "reorder_columns": {"sheet_name": "Sheet1", "desired_order": []},
    "set_background_color": {"color": "FFFFFF", "slide_index": None},
}


def normalize_tool_args(function_name, args):
    if function_name not in TOOL_DEFAULTS:
        return args.copy() if args else {}
    normalized = TOOL_DEFAULTS[function_name].copy()
    if args:
        normalized.update(args)
    return normalized


# ---------------------------------------------------------------------------
# Action parsing (from models/qwen2.5_vl_7b.py, simplified — no coordinate transform needed
# because HF Transformers processor handles resolution internally via smart_resize)
# ---------------------------------------------------------------------------
def smart_resize(height, width, factor=28, min_pixels=56 * 56, max_pixels=14 * 14 * 4 * 1280):
    new_height = int(np.round(height / factor) * factor)
    new_width = int(np.round(width / factor) * factor)
    new_pixels = new_height * new_width
    if new_pixels < min_pixels:
        scale = np.sqrt(min_pixels / new_pixels)
        new_height = int(np.round(new_height * scale / factor) * factor)
        new_width = int(np.round(new_width * scale / factor) * factor)
    elif new_pixels > max_pixels:
        scale = np.sqrt(max_pixels / new_pixels)
        new_height = int(np.round(new_height * scale / factor) * factor)
        new_width = int(np.round(new_width * scale / factor) * factor)
    return new_height, new_width


def transform_coordinates(coords, original_width, original_height, factor=28):
    """Transform coordinates from resized image space back to original."""
    resized_h, resized_w = smart_resize(original_height, original_width, factor=factor)
    scale_x = original_width / resized_w
    scale_y = original_height / resized_h
    transformed = []
    for i, c in enumerate(coords):
        if i % 2 == 0:
            transformed.append(c * scale_x)
        else:
            transformed.append(c * scale_y)
    return transformed


def parse_action(response, original_width, original_height):
    """Parse action from model response. Returns (function, args, status)."""
    try:
        # Primary: <tool_call> JSON </tool_call>
        pattern = r'<tool_call>\s*\{\s*"function":\s*"([^"]*)",\s*"args":\s*(\{.*?\}),\s*"status":\s*"([^"]+)"\s*\}\s*</tool_call>'
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)

        if not match:
            # Fallback: without tags
            pattern2 = r'\{\s*"function":\s*"([^"]*)",\s*"args":\s*(\{.*?\}),\s*"status":\s*"([^"]+)"\s*\}'
            match = re.search(pattern2, response, re.DOTALL)

        if not match:
            # Try JSON blocks
            json_blocks = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            for block in json_blocks:
                try:
                    data = json.loads(block)
                    if isinstance(data, dict) and "function" in data:
                        fn = data.get("function")
                        args = data.get("args", {})
                        st = data.get("status", "CONTINUE")
                        args = _transform_action_coords(args, fn, original_width, original_height)
                        return fn, args, st
                except json.JSONDecodeError:
                    continue

            # Last resort: loose regex
            func_m = re.search(r'"function":\s*"([^"]+)"', response)
            args_m = re.search(r'"args":\s*(\{.*?\})', response, re.DOTALL)
            status_m = re.search(r'"status":\s*"([^"]+)"', response)
            if func_m:
                fn = func_m.group(1)
                args = {}
                if args_m:
                    try:
                        args = json.loads(args_m.group(1))
                    except json.JSONDecodeError:
                        pass
                st = status_m.group(1) if status_m else "CONTINUE"
                args = _transform_action_coords(args, fn, original_width, original_height)
                return fn, args, st

            return None, None, None

        function_name = match.group(1)
        args_str = match.group(2)
        status = match.group(3)

        try:
            args_dict = json.loads(args_str)
        except json.JSONDecodeError:
            args_dict = {}

        args_dict = _transform_action_coords(args_dict, function_name, original_width, original_height)
        return function_name, args_dict, status

    except Exception as e:
        print(f"Error parsing action: {e}")
        return None, None, None


def _transform_action_coords(args, function_name, orig_w, orig_h):
    """Transform predicted coordinates from resized to original image space."""
    if not args or not function_name:
        return args or {}
    if function_name == "drag":
        if "start_coordinate" in args and isinstance(args["start_coordinate"], list) and len(args["start_coordinate"]) >= 2:
            args["start_coordinate"] = transform_coordinates(args["start_coordinate"], orig_w, orig_h)
        if "end_coordinate" in args and isinstance(args["end_coordinate"], list) and len(args["end_coordinate"]) >= 2:
            args["end_coordinate"] = transform_coordinates(args["end_coordinate"], orig_w, orig_h)
    if "coordinate" in args and isinstance(args["coordinate"], list) and len(args["coordinate"]) >= 2:
        args["coordinate"] = transform_coordinates(args["coordinate"], orig_w, orig_h)
    return args


# ---------------------------------------------------------------------------
# Action comparison (from evaluator/action_prediction.py)
# ---------------------------------------------------------------------------
def compare_actions(pred_fn, pred_args, pred_status, gt_fn, gt_args, gt_status,
                    gt_rect=None, gt_rect_end=None):
    """Returns (function_match, args_match, status_match)."""
    function_match = pred_fn == gt_fn if pred_fn is not None else False
    status_match = pred_status == gt_status if pred_status else False

    args_match = False
    if pred_args is not None and gt_args is not None:
        if pred_fn == "drag" and gt_fn == "drag":
            args_match = _compare_drag_args(pred_args, gt_args, gt_rect, gt_rect_end)
        else:
            args_match = _compare_regular_args(pred_args, gt_args, gt_rect, pred_fn, gt_fn)

    return function_match, args_match, status_match


def _compare_drag_args(pred_args, gt_args, gt_rect=None, gt_rect_end=None):
    try:
        pn = normalize_tool_args("drag", pred_args)
        gn = normalize_tool_args("drag", gt_args)
        for key in ["start_coordinate", "end_coordinate"]:
            if key not in pn or key not in gn:
                return False
            if not (isinstance(pn[key], (list, tuple)) and len(pn[key]) == 2):
                return False
            if not (isinstance(gn[key], (list, tuple)) and len(gn[key]) == 2):
                return False

        def _coord_in_rect(coord, rect):
            if rect:
                return rect["left"] <= float(coord[0]) <= rect["right"] and rect["top"] <= float(coord[1]) <= rect["bottom"]
            return False

        def _coord_close(pred_c, gt_c, tol=25.0):
            return abs(float(pred_c[0]) - float(gt_c[0])) <= tol and abs(float(pred_c[1]) - float(gt_c[1])) <= tol

        start_ok = _coord_in_rect(pn["start_coordinate"], gt_rect) if gt_rect else _coord_close(pn["start_coordinate"], gn["start_coordinate"])
        end_ok = _coord_in_rect(pn["end_coordinate"], gt_rect_end) if gt_rect_end else _coord_close(pn["end_coordinate"], gn["end_coordinate"])

        other_ok = True
        for key in ["button", "duration", "key_hold"]:
            if str(pn.get(key)).lower() != str(gn.get(key)).lower():
                other_ok = False
                break
        return start_ok and end_ok and other_ok
    except Exception:
        return False


def _compare_regular_args(pred_args, gt_args, gt_rect=None, pred_fn=None, gt_fn=None):
    try:
        pn = normalize_tool_args(pred_fn or "click", pred_args)
        gn = normalize_tool_args(gt_fn or "click", gt_args)

        if "coordinate" in pn and "coordinate" in gn:
            pc = pn["coordinate"]
            gc = gn["coordinate"]
            if isinstance(pc, (list, tuple)) and len(pc) == 2 and isinstance(gc, (list, tuple)) and len(gc) == 2:
                if gt_rect:
                    coord_ok = gt_rect["left"] <= float(pc[0]) <= gt_rect["right"] and gt_rect["top"] <= float(pc[1]) <= gt_rect["bottom"]
                else:
                    coord_ok = abs(float(pc[0]) - float(gc[0])) <= 25.0 and abs(float(pc[1]) - float(gc[1])) <= 25.0

                other_ok = True
                for key in pn:
                    if key != "coordinate":
                        pv = str(pn[key]).lower() if pn[key] is not None else "none"
                        gv = str(gn.get(key)).lower() if gn.get(key) is not None else "none"
                        if pv != gv:
                            other_ok = False
                            break
                return coord_ok and other_ok

        # Non-coordinate comparison
        ps = {k: str(v).lower() if isinstance(v, (str, bool)) else ("none" if v is None else v) for k, v in pn.items()}
        gs = {k: str(v).lower() if isinstance(v, (str, bool)) else ("none" if v is None else v) for k, v in gn.items()}
        return ps == gs
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Brief action formatting (from action_prediction_ar_context.py)
# ---------------------------------------------------------------------------
def format_action_brief(pred_function, pred_args):
    if pred_function is None:
        return "unknown action"
    if pred_function == "click":
        coord = pred_args.get("coordinate", [0, 0]) if pred_args else [0, 0]
        button = pred_args.get("button", "left") if pred_args else "left"
        return f"click({coord}, button='{button}')"
    elif pred_function == "type":
        keys = pred_args.get("keys", "") if pred_args else ""
        if len(keys) > 30:
            keys = keys[:30] + "..."
        return f"type(keys='{keys}')"
    elif pred_function == "drag":
        start = pred_args.get("start_coordinate", [0, 0]) if pred_args else [0, 0]
        end = pred_args.get("end_coordinate", [0, 0]) if pred_args else [0, 0]
        return f"drag({start} -> {end})"
    elif pred_function == "scroll" or pred_function == "wheel_mouse_input":
        coord = pred_args.get("coordinate", [0, 0]) if pred_args else [0, 0]
        dist = pred_args.get("wheel_dist", 0) if pred_args else 0
        return f"scroll({coord}, dist={dist})"
    elif pred_function == "hotkey":
        keys = pred_args.get("keys", "") if pred_args else ""
        return f"hotkey(keys='{keys}')"
    else:
        return f"{pred_function}({pred_args})"


# ---------------------------------------------------------------------------
# Trajectory loading (from evaluator/action_prediction_autoregressive.py)
# ---------------------------------------------------------------------------
def load_trajectories(data_root, trajectory_ids=None):
    """Load trajectories from GUI-Odyssey dataset."""
    data_path = os.path.join(data_root, "data")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    id_set = set(trajectory_ids) if trajectory_ids else None
    trajectories = []

    for domain in sorted(os.listdir(data_path)):
        domain_path = os.path.join(data_path, domain)
        if not os.path.isdir(domain_path):
            continue
        for category in sorted(os.listdir(domain_path)):
            cat_path = os.path.join(domain_path, category, "success")
            if not os.path.isdir(cat_path):
                continue
            for jsonl_file in sorted(os.listdir(cat_path)):
                if not jsonl_file.endswith(".jsonl"):
                    continue
                file_stem = os.path.splitext(jsonl_file)[0]
                traj_id = f"{domain}_{category}_{file_stem}"

                if id_set and traj_id not in id_set:
                    continue

                file_path = os.path.join(cat_path, jsonl_file)
                try:
                    steps = []
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line_num, line in enumerate(f, 1):
                            if not line.strip():
                                continue
                            try:
                                data = json.loads(line.strip())
                            except json.JSONDecodeError:
                                continue

                            if "action_prediction" not in data["step"].get("tags", []):
                                continue

                            clean_img = os.path.join(
                                data_root, "image", domain, category,
                                data["step"]["screenshot_clean"],
                            )
                            if not os.path.exists(clean_img):
                                continue

                            action = data["step"]["action"]
                            if action.get("function", "") == "drag" or not action.get("rectangle", {}):
                                continue

                            status = data["step"]["status"]
                            if status == "OVERALL_FINISH":
                                status = "FINISH"
                            elif status == "FINISH":
                                status = "CONTINUE"

                            steps.append({
                                "sample_id": f"{traj_id}_{line_num}",
                                "line_num": line_num,
                                "request": data["request"],
                                "screenshot_clean": clean_img,
                                "thought": data["step"]["thought"],
                                "subtask": data["step"].get("subtask", ""),
                                "action": action,
                                "status": status,
                                "domain": domain,
                                "category": category,
                                "step_index": len(steps) + 1,
                            })

                    if steps:
                        trajectories.append({
                            "trajectory_id": traj_id,
                            "request": steps[0]["request"],
                            "domain": domain,
                            "category": category,
                            "steps": steps,
                        })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue

    print(f"Loaded {len(trajectories)} trajectories, "
          f"{sum(len(t['steps']) for t in trajectories)} total steps")
    return trajectories


# ---------------------------------------------------------------------------
# Subtask segmentation (from action_prediction_ar_context.py)
# ---------------------------------------------------------------------------
def segment_by_subtask(steps):
    """Group consecutive steps by subtask field."""
    if not steps:
        return []
    segments = []
    current_subtask = steps[0].get("subtask", "")
    current_indices = [0]
    for i in range(1, len(steps)):
        st = steps[i].get("subtask", "")
        if st != current_subtask and st:
            segments.append((current_subtask, current_indices))
            current_subtask = st
            current_indices = [i]
        else:
            current_indices.append(i)
    segments.append((current_subtask, current_indices))
    return segments


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------
def run_inference(args):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    # Load model
    print(f"Loading model from {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    print("Model loaded")

    # Load probe
    probe = None
    if args.mode == "verifier":
        print(f"Loading probe from {args.probe_path}")
        probe = joblib.load(args.probe_path)
        print(f"  Layer: {probe.get('layer', 26)}, "
              f"train acc: {probe.get('train_accuracy', 'N/A')}")

    # Load trajectories
    traj_ids = None
    if args.trajectory_ids:
        with open(args.trajectory_ids, "r") as f:
            traj_ids = json.load(f)
        print(f"Filtering to {len(traj_ids)} trajectory IDs")

    trajectories = load_trajectories(args.data_root, traj_ids)
    if args.max_trajectories:
        trajectories = trajectories[:args.max_trajectories]
        print(f"Limited to {len(trajectories)} trajectories")

    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Process trajectories
    all_trajectory_results = []
    total_steps = 0
    total_correct = 0

    for traj_idx, trajectory in enumerate(tqdm(trajectories, desc=f"[{args.mode}] Trajectories")):
        traj_result = evaluate_trajectory(
            model, processor, probe, trajectory, args.mode,
        )
        all_trajectory_results.append(traj_result)

        n_steps = traj_result["num_steps"]
        n_correct = sum(1 for s in traj_result["step_results"] if s.get("success", False))
        total_steps += n_steps
        total_correct += n_correct

        print(f"  [{traj_idx+1}/{len(trajectories)}] {traj_result['trajectory_id']}: "
              f"TSR={'Y' if traj_result['trajectory_success'] else 'N'}, "
              f"progress={traj_result['progress_rate']:.2f}, "
              f"scattered={traj_result['scattered_progress_rate']:.2f}, "
              f"steps={n_correct}/{n_steps}")

    # Compute statistics
    n_traj = len(all_trajectory_results)
    stats = {
        "num_trajectories": n_traj,
        "num_steps": total_steps,
        "trajectory_success_rate": sum(1 for t in all_trajectory_results if t["trajectory_success"]) / n_traj if n_traj else 0,
        "avg_progress_rate": np.mean([t["progress_rate"] for t in all_trajectory_results]) if n_traj else 0,
        "avg_scattered_progress_rate": np.mean([t["scattered_progress_rate"] for t in all_trajectory_results]) if n_traj else 0,
        "step_success_rate": total_correct / total_steps if total_steps else 0,
    }

    # Probe stats for verifier mode
    if args.mode == "verifier":
        all_probe_probs = []
        all_probe_decisions = []
        all_step_success = []
        for t in all_trajectory_results:
            for s in t["step_results"]:
                if "probe_prob_correct" in s:
                    all_probe_probs.append(s["probe_prob_correct"])
                    all_probe_decisions.append(s["probe_decision"])
                    all_step_success.append(s["success"])
        if all_probe_probs:
            probs = np.array(all_probe_probs)
            successes = np.array(all_step_success)
            decisions = all_probe_decisions
            stats["probe_stats"] = {
                "n_greedy": sum(1 for d in decisions if d == "greedy"),
                "n_resample": sum(1 for d in decisions if d == "resample"),
                "avg_prob_correct": float(probs.mean()),
                "avg_prob_when_actually_correct": float(probs[successes].mean()) if successes.any() else None,
                "avg_prob_when_actually_wrong": float(probs[~successes].mean()) if (~successes).any() else None,
            }

    # Save results
    output = {
        "config": {
            "mode": args.mode,
            "model_path": args.model_path,
            "data_root": args.data_root,
            "probe_path": args.probe_path if args.mode == "verifier" else None,
            "num_trajectories": n_traj,
            "timestamp": datetime.now().isoformat(),
        },
        "statistics": stats,
        "trajectory_results": [
            {
                "trajectory_id": t["trajectory_id"],
                "num_steps": t["num_steps"],
                "trajectory_success": t["trajectory_success"],
                "progress_rate": t["progress_rate"],
                "scattered_progress_rate": t["scattered_progress_rate"],
                "first_error_step": t["first_error_step"],
                "domain": t["domain"],
                "category": t["category"],
            }
            for t in all_trajectory_results
        ],
        "detailed_results": all_trajectory_results,
    }

    output_file = os.path.join(args.output_dir, f"verifier_{args.mode}_results.json")
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")
    print(f"\n=== Summary ({args.mode}) ===")
    print(f"TSR: {stats['trajectory_success_rate']:.4f}")
    print(f"Avg Progress: {stats['avg_progress_rate']:.4f}")
    print(f"Avg Scattered Progress: {stats['avg_scattered_progress_rate']:.4f}")
    print(f"Step Accuracy: {stats['step_success_rate']:.4f}")


def evaluate_trajectory(model, processor, probe, trajectory, mode):
    """Evaluate a single trajectory with probe-guided temperature."""
    from qwen_vl_utils import process_vision_info

    traj_id = trajectory["trajectory_id"]
    steps = trajectory["steps"]
    domain = trajectory["domain"]
    category = trajectory["category"]

    segments = segment_by_subtask(steps)

    all_step_results = []
    first_error_step = None

    for seg_idx, (subtask_desc, step_indices) in enumerate(segments):
        compressed_history = []

        if not subtask_desc:
            subtask_desc = steps[step_indices[0]]["request"]

        for local_i, global_i in enumerate(step_indices):
            step = steps[global_i]
            start_time = time.time()
            step_num = global_i + 1

            try:
                clean_img_path = step["screenshot_clean"]
                action = step["action"].copy()
                action_args = action.get("args", {}).copy()
                action["args"] = action_args

                img = Image.open(clean_img_path)
                original_width, original_height = img.size

                # Prepare GT
                gt_rect = action.get("rectangle", {})
                if action["function"] == "drag":
                    sx, sy = action_args["start_x"], action_args["start_y"]
                    ex, ey = action_args["end_x"], action_args["end_y"]
                    action_args["start_coordinate"] = [sx, sy]
                    action_args["end_coordinate"] = [ex, ey]
                    for k in ["start_x", "start_y", "end_x", "end_y"]:
                        action_args.pop(k, None)
                    gt_rect = {"left": max(0, sx) - 25, "top": max(0, sy) - 25,
                               "right": min(sx + 25, original_width), "bottom": min(sy + 25, original_height)}
                    gt_rect_end = {"left": max(0, ex) - 25, "top": max(0, ey) - 25,
                                   "right": min(ex + 25, original_width), "bottom": min(ey + 25, original_height)}
                else:
                    action_args.pop("x", None)
                    action_args.pop("y", None)
                    if "coordinate_x" in action and action["coordinate_x"]:
                        action_args["coordinate"] = [action["coordinate_x"], action["coordinate_y"]]
                    gt_rect_end = None

                gt_function = action.get("function", "")
                gt_args = action_args
                gt_status = step.get("status", "")

                # Get supported actions
                if domain.lower() == "word":
                    actions_text = SUPPORTED_ACTIONS_WORD
                elif domain.lower() == "excel":
                    actions_text = SUPPORTED_ACTIONS_EXCEL
                elif domain.lower() == "ppt":
                    actions_text = SUPPORTED_ACTIONS_PPT
                else:
                    actions_text = SUPPORTED_ACTIONS_WORD

                # Build prompt
                history_text = "\n".join(compressed_history) if compressed_history else "None"
                user_prompt = SUBTASK_ISOLATED_USER_PROMPT.format(
                    instruction=step["request"],
                    subtask_description=subtask_desc,
                    history=history_text,
                    actions=actions_text,
                )

                # Build messages for processor
                messages = [{"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image", "image": clean_img_path},
                ]}]

                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text], images=image_inputs, videos=video_inputs,
                    padding=True, return_tensors="pt",
                ).to(model.device)

                # ---- Probe + Generate ----
                probe_prob = None
                probe_decision = None
                temperature = 0.0  # default greedy
                input_len = inputs["input_ids"].shape[1]

                if mode == "verifier":
                    # Fused: greedy generate WITH hidden states (single prefill)
                    # → extract probe signal from prefill hidden states
                    # → if probe says "wrong", re-generate with temperature
                    with torch.no_grad():
                        gen_out = model.generate(
                            **inputs, max_new_tokens=512, do_sample=False,
                            output_hidden_states=True, return_dict_in_generate=True,
                        )
                    # Prefill hidden states: gen_out.hidden_states[0] = first step
                    # [0][layer+1] where layer+1=27 for L26
                    prefill_hs = gen_out.hidden_states[0][27][0, -1, :].cpu().float().numpy()
                    x = probe["scaler"].transform(prefill_hs.reshape(1, -1))
                    x = probe["pca"].transform(x)
                    probe_prob = float(probe["clf"].predict_proba(x)[0, 1])

                    if probe_prob > 0.5:
                        # Probe says correct → keep greedy output (no extra work)
                        temperature = 0.0
                        probe_decision = "greedy"
                        generated_ids = gen_out.sequences[0, input_len:]
                        response = processor.decode(generated_ids, skip_special_tokens=True)
                    else:
                        # Probe says wrong → discard greedy, re-generate with temp=1.0
                        temperature = 1.0
                        probe_decision = "resample"
                        del gen_out
                        torch.cuda.empty_cache()
                        with torch.no_grad():
                            output_ids = model.generate(
                                **inputs, max_new_tokens=512,
                                do_sample=True, temperature=1.0,
                            )
                        generated_ids = output_ids[0, input_len:]
                        response = processor.decode(generated_ids, skip_special_tokens=True)

                    if "gen_out" in locals():
                        del gen_out
                    torch.cuda.empty_cache()

                else:
                    # always_greedy or always_temp1: single generate, no probe
                    temperature = 0.0 if mode == "always_greedy" else 1.0
                    gen_kwargs = dict(max_new_tokens=512)
                    if temperature > 0:
                        gen_kwargs["do_sample"] = True
                        gen_kwargs["temperature"] = temperature
                    else:
                        gen_kwargs["do_sample"] = False
                    with torch.no_grad():
                        output_ids = model.generate(**inputs, **gen_kwargs)
                    generated_ids = output_ids[0, input_len:]
                    response = processor.decode(generated_ids, skip_special_tokens=True)

                # Parse action
                pred_function, pred_args, pred_status = parse_action(
                    response, original_width, original_height,
                )

                # Compare
                if gt_function == "drag":
                    fn_match, args_match, status_match = compare_actions(
                        pred_function, pred_args, pred_status,
                        gt_function, gt_args, gt_status, gt_rect, gt_rect_end,
                    )
                else:
                    fn_match, args_match, status_match = compare_actions(
                        pred_function, pred_args, pred_status,
                        gt_function, gt_args, gt_status, gt_rect,
                    )

                success = fn_match and args_match and status_match
                exec_time = time.time() - start_time

                # Build step result
                step_result = {
                    "sample_id": step["sample_id"],
                    "step_num": step_num,
                    "subtask_idx": seg_idx,
                    "subtask_description": subtask_desc,
                    "local_step_num": local_i + 1,
                    "success": success,
                    "function_match": fn_match,
                    "args_match": args_match,
                    "status_match": status_match,
                    "predicted_function": pred_function,
                    "predicted_args": pred_args,
                    "predicted_status": pred_status,
                    "ground_truth_function": gt_function,
                    "ground_truth_args": gt_args,
                    "ground_truth_status": gt_status,
                    "ground_truth_rect": gt_rect,
                    "raw_model_output": response,
                    "execution_time": exec_time,
                    "temperature_used": temperature,
                    "error_message": None,
                }

                if mode == "verifier":
                    step_result["probe_prob_correct"] = probe_prob
                    step_result["probe_decision"] = probe_decision

                if not success and first_error_step is None:
                    first_error_step = step_num

                all_step_results.append(step_result)

                # Update history
                brief = format_action_brief(pred_function, pred_args)
                compressed_history.append(f"Step {local_i + 1}: {brief}")

                status_char = "\u2713" if success else "\u2717"
                probe_info = f" P(c)={probe_prob:.2f}->{probe_decision}" if probe_prob is not None else ""
                print(f"  [{traj_id}] seg{seg_idx+1} step {local_i+1}/{len(step_indices)} "
                      f"(g{step_num}): {pred_function} vs {gt_function} ({status_char})"
                      f" T={temperature}{probe_info}")

            except Exception as e:
                exec_time = time.time() - start_time
                print(f"  ERROR at step {step_num}: {traceback.format_exc()}")
                all_step_results.append({
                    "sample_id": step["sample_id"],
                    "step_num": step_num,
                    "subtask_idx": seg_idx,
                    "success": False,
                    "function_match": False,
                    "args_match": False,
                    "status_match": False,
                    "error_message": str(e),
                    "execution_time": exec_time,
                    "temperature_used": temperature if 'temperature' in dir() else 0.0,
                })
                if first_error_step is None:
                    first_error_step = step_num
                compressed_history.append(f"Step {local_i + 1}: Error occurred")

    # Trajectory-level metrics
    num_correct = sum(1 for s in all_step_results if s.get("success", False))
    n_total = len(steps)
    traj_success = num_correct == n_total and len(all_step_results) == n_total

    if first_error_step is not None:
        seq_progress = (first_error_step - 1) / n_total
    else:
        seq_progress = 1.0

    scattered_progress = num_correct / n_total if n_total > 0 else 0.0

    return {
        "trajectory_id": traj_id,
        "num_steps": n_total,
        "trajectory_success": traj_success,
        "progress_rate": seq_progress,
        "scattered_progress_rate": scattered_progress,
        "first_error_step": first_error_step,
        "domain": domain,
        "category": category,
        "step_results": all_step_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Verifier-guided AR inference")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to SFT v2 HF model")
    parser.add_argument("--data_root", type=str, required=True,
                        help="GUI-Odyssey dataset root")
    parser.add_argument("--probe_path", type=str, default=None,
                        help="Path to correctness_probe_L26.pkl (required for verifier mode)")
    parser.add_argument("--trajectory_ids", type=str, default=None,
                        help="Path to JSON file with trajectory ID list")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["verifier", "always_greedy", "always_temp1"],
                        help="Temperature selection mode")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--max_trajectories", type=int, default=None,
                        help="Max trajectories to process (for debugging)")
    args = parser.parse_args()

    if args.mode == "verifier" and not args.probe_path:
        parser.error("--probe_path is required for verifier mode")

    run_inference(args)


if __name__ == "__main__":
    main()
