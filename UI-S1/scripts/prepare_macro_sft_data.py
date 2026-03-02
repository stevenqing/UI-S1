#!/usr/bin/env python3
"""
Task 5: Prepare macro-augmented SFT training data.

This script identifies bottleneck-crossing trajectories from GUI-360, classifies
them into macro types (navigate_to_dialog, switch_ui_mode, navigate_and_return),
and generates augmented SFT training data that teaches the model to call macro
tools at the right time.

Pipeline:
  1. Load f-values and transitions to identify crossing trajectories
  2. Load raw trajectory data for each crossing
  3. Classify crossing type → select macro tool
  4. Construct macro-augmented SFT samples
  5. Mix with original SFT data and output parquet

Usage:
    # Full run (all apps)
    python scripts/prepare_macro_sft_data.py

    # Single app
    python scripts/prepare_macro_sft_data.py --app excel

    # Custom thresholds
    python scripts/prepare_macro_sft_data.py --crossing-threshold 1.5 --upsample-factor 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Default paths
# ──────────────────────────────────────────────────────────────────────
DEFAULT_FNET_DIR = PROJECT_ROOT / "outputs" / "fnet" / "gui360"
DEFAULT_TRANSITIONS_DIR = PROJECT_ROOT / "outputs" / "transitions" / "gui360_full"
DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / "datasets" / "GUI-360" / "train" / "data"
DEFAULT_IMAGE_BASE_DIR = PROJECT_ROOT / "datasets" / "GUI-360" / "train" / "image"
DEFAULT_BASELINE_PARQUET = PROJECT_ROOT / "train_GUI_360" / "data" / "gui360_train_sft_a11y_thinking.parquet"
DEFAULT_EVAL_PARQUET = PROJECT_ROOT / "train_GUI_360" / "data" / "gui360_eval_sft_a11y_thinking.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "macro_sft"

APPS = ["excel", "word", "ppt"]

# ──────────────────────────────────────────────────────────────────────
# Macro tool definitions
# ──────────────────────────────────────────────────────────────────────
MACRO_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "navigate_to_dialog",
            "description": (
                "Navigate from the current view to open a specific dialog or options panel. "
                "Use this when you need to access a settings dialog, options menu, or configuration "
                "panel that requires multiple navigation steps (e.g., File > Options, right-click > Format Cells)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_dialog": {
                        "type": "string",
                        "description": (
                            "The name of the target dialog to open (e.g., 'Word Options', "
                            "'Format Cells', 'Excel Options', 'PowerPoint Options', 'Insert Picture')"
                        ),
                    }
                },
                "required": ["target_dialog"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_ui_mode",
            "description": (
                "Switch between different UI editing modes. Use this when you need to exit the "
                "current editing context (e.g., exit table editing mode in Word, exit embedded "
                "Excel chart in PPT, switch from Data-only view to full ribbon in Excel)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "from_mode": {
                        "type": "string",
                        "description": (
                            "The current UI mode to exit (e.g., 'table_edit', 'embedded_chart', "
                            "'data_view', 'developer_view')"
                        ),
                    },
                    "to_mode": {
                        "type": "string",
                        "description": (
                            "The target UI mode (e.g., 'normal_edit', 'slide_edit', 'full_ribbon')"
                        ),
                    },
                },
                "required": ["from_mode", "to_mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "navigate_and_return",
            "description": (
                "Perform a multi-step round-trip: navigate away from the current context, complete "
                "an operation, and return. Use this for tasks that require temporarily leaving the "
                "current editing mode to access a different part of the application (e.g., exit "
                "table mode > open Word Options > change a setting > return to document)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": (
                            "Brief description of the operation to perform during the excursion "
                            "(e.g., 'disable AutoRecover in Save settings', 'encrypt workbook with password')"
                        ),
                    }
                },
                "required": ["operation"],
            },
        },
    },
]

# Macro tool descriptions to append to the system prompt's <action> block
MACRO_ACTION_TEXT = """
- navigate_to_dialog
  - Args:
    - target_dialog: str, The name of the target dialog to open
  - Description: Navigate from current view to open a specific dialog or options panel.
    Use this when accessing settings or dialogs that require multiple navigation steps.
  - Example: navigate_to_dialog(target_dialog="Word Options"), navigate_to_dialog(target_dialog="Format Cells")

- switch_ui_mode
  - Args:
    - from_mode: str, The current UI mode to exit
    - to_mode: str, The target UI mode
  - Description: Switch between different UI editing modes.
    Use when exiting table edit mode, embedded chart mode, or other specialized views.
  - Example: switch_ui_mode(from_mode="table_edit", to_mode="normal_edit")

- navigate_and_return
  - Args:
    - operation: str, Brief description of the operation to perform
  - Description: Perform a multi-step round-trip operation that temporarily leaves
    the current context, completes a task, and returns.
  - Example: navigate_and_return(operation="disable AutoRecover in Save settings")
"""


# ──────────────────────────────────────────────────────────────────────
# Step 1: Identify crossing trajectories
# ──────────────────────────────────────────────────────────────────────

def load_f_values(fnet_dir: Path, app: str) -> Dict[str, float]:
    """Load f-value mapping {state_hash: f_value} for an app."""
    npz_path = fnet_dir / app / "f_values.npz"
    if not npz_path.exists():
        logger.warning(f"f_values.npz not found for {app} at {npz_path}")
        return {}
    data = np.load(npz_path)
    hashes = data["hashes"]
    f_values = data["f_values"]
    return {h: float(v) for h, v in zip(hashes, f_values)}


def load_transitions(transitions_dir: Path) -> List[Dict]:
    """Load all transition records."""
    path = transitions_dir / "transitions.jsonl"
    records = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_state_registry(transitions_dir: Path) -> Dict[str, Dict]:
    """Load state registry."""
    path = transitions_dir / "state_registry.json"
    with open(path, "r") as f:
        return json.load(f)


def identify_crossings(
    transitions: List[Dict],
    f_value_map: Dict[str, float],
    app: str,
    threshold: float = 1.0,
) -> List[Dict]:
    """
    Identify trajectories that cross bottleneck boundaries.

    A crossing trajectory has f-value range > threshold, indicating it
    traverses from one side of the eigenfunction landscape to the other.

    Returns list of crossing info dicts with:
      - execution_id, app, f_value_range, crossing_point, steps
    """
    # Group transitions by execution_id
    traj_transitions: Dict[str, List[Dict]] = defaultdict(list)
    for t in transitions:
        eid = t["execution_id"]
        if eid.startswith(app + "_"):
            traj_transitions[eid].append(t)

    crossings = []
    for eid, steps in traj_transitions.items():
        # Sort by step_id
        steps = sorted(steps, key=lambda x: x["step_id"])

        # Build f-value sequence for each step
        f_sequence = []
        hash_sequence = []
        for s in steps:
            sh = s["state_hash"]
            fv = f_value_map.get(sh)
            if fv is not None:
                f_sequence.append(fv)
                hash_sequence.append(sh)
            else:
                f_sequence.append(None)
                hash_sequence.append(sh)

        # Filter out None values for range computation
        valid_fvals = [v for v in f_sequence if v is not None]
        if len(valid_fvals) < 2:
            continue

        f_range = max(valid_fvals) - min(valid_fvals)
        if f_range < threshold:
            continue

        # Find the crossing point: position with max |f[i+1] - f[i]|
        max_jump = 0.0
        crossing_idx = 0
        for i in range(len(f_sequence) - 1):
            if f_sequence[i] is not None and f_sequence[i + 1] is not None:
                jump = abs(f_sequence[i + 1] - f_sequence[i])
                if jump > max_jump:
                    max_jump = jump
                    crossing_idx = i

        # Determine crossing span: find contiguous high-gradient region
        # Start from crossing_idx, expand while gradient is significant
        span_start = crossing_idx
        span_end = crossing_idx + 1

        # Expand backwards
        for i in range(crossing_idx - 1, -1, -1):
            if f_sequence[i] is not None and f_sequence[i + 1] is not None:
                if abs(f_sequence[i + 1] - f_sequence[i]) > 0.1:
                    span_start = i
                else:
                    break
            else:
                break

        # Expand forwards
        for i in range(crossing_idx + 1, len(f_sequence) - 1):
            if f_sequence[i] is not None and f_sequence[i + 1] is not None:
                if abs(f_sequence[i + 1] - f_sequence[i]) > 0.1:
                    span_end = i + 1
                else:
                    break
            else:
                break

        crossings.append({
            "execution_id": eid,
            "app": app,
            "f_value_range": f_range,
            "max_jump": max_jump,
            "crossing_idx": crossing_idx,
            "span_start": span_start,
            "span_end": span_end,
            "span_length": span_end - span_start + 1,
            "total_steps": len(steps),
            "f_sequence": f_sequence,
            "hash_sequence": hash_sequence,
            "step_ids": [s["step_id"] for s in steps],
        })

    # Sort by f_value_range descending
    crossings.sort(key=lambda x: -x["f_value_range"])
    return crossings


# ──────────────────────────────────────────────────────────────────────
# Step 2: Load raw trajectory data
# ──────────────────────────────────────────────────────────────────────

def find_trajectory_file(raw_data_dir: Path, execution_id: str, app: str) -> Optional[Path]:
    """Find the JSONL file for a given execution_id."""
    # Search in success directories
    for subdir in ["in_app/success", "online/success", "search/success",
                    "in_app", "online", "search"]:
        path = raw_data_dir / app / subdir / f"{execution_id}.jsonl"
        if path.exists():
            return path
    # Fallback: recursive search
    for p in (raw_data_dir / app).rglob(f"{execution_id}.jsonl"):
        return p
    return None


def resolve_image_path(
    screenshot_relative: str,
    traj_file: Path,
    raw_data_dir: Path,
    image_root: Path,
) -> str:
    """
    Resolve the actual image path from a screenshot relative path.

    The raw trajectory JSONL stores paths like 'success/excel_1_87/action_step1_annotated.png'.
    The actual images live at 'image/{app}/{category}/success/{exec_id}/...'

    We derive the correct image path by mirroring the trajectory file's location
    within the data directory to the image directory.

    Returns the resolved absolute path string.
    """
    # Get the trajectory file's path relative to raw_data_dir
    # e.g., traj_file = .../data/excel/in_app/success/excel_1_105.jsonl
    #        raw_data_dir = .../data
    #        rel = excel/in_app/success/excel_1_105.jsonl
    try:
        traj_rel = traj_file.relative_to(raw_data_dir)
    except ValueError:
        # Fallback: just join with image_root
        return str(image_root / screenshot_relative)

    # The image base mirrors the data hierarchy:
    # data/excel/in_app/success/excel_1_105.jsonl →
    # image/excel/in_app/<screenshot_relative>
    # where screenshot_relative = success/excel_1_105/action_step1_annotated.png
    traj_parent = traj_rel.parent  # excel/in_app/success
    # Go up one level from 'success' to get the category dir
    category_dir = traj_parent.parent  # excel/in_app
    resolved = image_root / category_dir / screenshot_relative

    if resolved.exists():
        return str(resolved)

    # Fallback: try the direct join (for flat image structures)
    direct = image_root / screenshot_relative
    if direct.exists():
        return str(direct)

    # Fallback: use the mirrored path even if it doesn't exist
    # (the dataset/training may handle missing images differently)
    return str(resolved)


def load_raw_trajectory(filepath: Path) -> List[Dict]:
    """Load all steps from a trajectory JSONL file. Each line is one step."""
    steps = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    steps.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    # Sort by step_id
    steps.sort(key=lambda x: x.get("step_id", 0))
    return steps


# ──────────────────────────────────────────────────────────────────────
# Step 3: Classify crossing type → select macro tool
# ──────────────────────────────────────────────────────────────────────

def classify_crossing(
    crossing: Dict,
    state_registry: Dict[str, Dict],
) -> Tuple[str, Dict[str, Any]]:
    """
    Classify a crossing into a macro type based on state features.

    Returns (macro_type, macro_args) where macro_type is one of:
      - "navigate_to_dialog"
      - "switch_ui_mode"
      - "navigate_and_return"
    """
    span_start = crossing["span_start"]
    span_end = crossing["span_end"]
    hash_seq = crossing["hash_sequence"]

    # Get state info before and after crossing
    start_hash = hash_seq[span_start] if span_start < len(hash_seq) else None
    end_hash = hash_seq[min(span_end, len(hash_seq) - 1)]

    start_state = state_registry.get(start_hash, {}) if start_hash else {}
    end_state = state_registry.get(end_hash, {}) if end_hash else {}

    start_dialog = start_state.get("dialog_state", "none")
    end_dialog = end_state.get("dialog_state", "none")
    start_tabs = start_state.get("active_tab_signature", "")
    end_tabs = end_state.get("active_tab_signature", "")

    # Check for round-trip: f-values go up and come back down (or vice versa)
    f_seq = crossing["f_sequence"]
    valid_f = [(i, v) for i, v in enumerate(f_seq) if v is not None]
    is_round_trip = False
    if len(valid_f) >= 3:
        first_f = valid_f[0][1]
        last_f = valid_f[-1][1]
        mid_f = valid_f[len(valid_f) // 2][1]
        # Round trip if start and end are on same side, but middle is different
        if abs(first_f - last_f) < 0.5 and abs(mid_f - first_f) > 0.5:
            is_round_trip = True

    # Classification rules:

    # Rule 1: If post-crossing state has a dialog → navigate_to_dialog
    if end_dialog != "none":
        return "navigate_to_dialog", {"target_dialog": end_dialog}

    # Check intermediate states for dialogs
    for i in range(span_start, min(span_end + 1, len(hash_seq))):
        h = hash_seq[i]
        st = state_registry.get(h, {})
        dialog = st.get("dialog_state", "none")
        if dialog != "none":
            if is_round_trip:
                return "navigate_and_return", {"operation": f"access {dialog} dialog"}
            else:
                return "navigate_to_dialog", {"target_dialog": dialog}

    # Rule 2: Tab signature change → switch_ui_mode
    if start_tabs and end_tabs and start_tabs != end_tabs:
        # Determine mode names from tab signatures
        from_mode = _infer_ui_mode(start_tabs, crossing["app"])
        to_mode = _infer_ui_mode(end_tabs, crossing["app"])
        if from_mode != to_mode:
            if is_round_trip:
                return "navigate_and_return", {"operation": f"switch from {from_mode} to {to_mode} and back"}
            else:
                return "switch_ui_mode", {"from_mode": from_mode, "to_mode": to_mode}

    # Rule 3: Round trip with no clear dialog/tab change → navigate_and_return
    if is_round_trip:
        return "navigate_and_return", {"operation": "perform multi-step navigation operation"}

    # Default: navigate_to_dialog (most common crossing pattern)
    return "navigate_to_dialog", {"target_dialog": "settings"}


def _infer_ui_mode(tab_signature: str, app: str) -> str:
    """Infer UI mode name from tab signature."""
    tabs = set(tab_signature.split(","))

    if app == "excel":
        if "Table Design" in tabs or "Table" in tabs:
            return "table_edit"
        if "Chart Design" in tabs or "Chart" in tabs:
            return "chart_edit"
        if "Drawing" in tabs:
            return "drawing_mode"
        if "Developer" in tabs and len(tabs) <= 3:
            return "developer_view"
        return "normal_edit"
    elif app == "word":
        if "Table Design" in tabs or "Layout" in tabs and "Table" in str(tabs):
            return "table_edit"
        if "Header & Footer" in tabs:
            return "header_footer_edit"
        if "Drawing" in tabs:
            return "drawing_mode"
        return "normal_edit"
    elif app == "ppt":
        if "Chart Design" in tabs:
            return "embedded_chart"
        if "Table Design" in tabs:
            return "table_edit"
        if "Drawing" in tabs:
            return "drawing_mode"
        if "Recording" in tabs:
            return "recording_mode"
        return "slide_edit"

    return "normal_edit"


# ──────────────────────────────────────────────────────────────────────
# Step 4: Construct macro-augmented SFT samples
# ──────────────────────────────────────────────────────────────────────

# Import helpers from the existing SFT preparation script
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "GUI_360"))
try:
    from prepare_gui360_sft_a11y_with_thinking import (
        SYSTEM_PROMPT_A11Y,
        format_element_list,
        convert_action_to_a11y_format,
    )
except ImportError:
    logger.warning("Could not import from prepare_gui360_sft_a11y_with_thinking, using local copies")
    # Minimal fallback - these should be available from the import
    SYSTEM_PROMPT_A11Y = ""
    def format_element_list(controls):
        return ""
    def convert_action_to_a11y_format(action, controls):
        return action, False


def _build_system_prompt_with_macros(instruction: str, history: str, element_list: str) -> str:
    """Build the system prompt with macro tool definitions appended to the action block."""
    base_prompt = SYSTEM_PROMPT_A11Y.format(
        instruction=instruction,
        history=history if history else "(none)",
        element_list=element_list,
    )
    # Insert macro actions before </action> tag
    if "</action>" in base_prompt:
        base_prompt = base_prompt.replace("</action>", MACRO_ACTION_TEXT + "</action>")
    else:
        # Append at end if no </action> tag
        base_prompt += "\n\nAdditional macro actions available:\n" + MACRO_ACTION_TEXT
    return base_prompt


def _build_macro_reasoning(macro_type: str, macro_args: Dict, step_data: Dict) -> str:
    """Build the reasoning text for a macro tool call."""
    thought = step_data.get("step", {}).get("thought", "")

    if macro_type == "navigate_to_dialog":
        target = macro_args.get("target_dialog", "the dialog")
        reasoning = (
            f"I need to open {target}, which requires navigating through multiple menus. "
            f"I'll use the navigate_to_dialog tool to handle this multi-step navigation."
        )
    elif macro_type == "switch_ui_mode":
        from_mode = macro_args.get("from_mode", "current mode")
        to_mode = macro_args.get("to_mode", "target mode")
        reasoning = (
            f"I need to switch from {from_mode} to {to_mode}. "
            f"I'll use the switch_ui_mode tool to handle this UI mode transition."
        )
    elif macro_type == "navigate_and_return":
        operation = macro_args.get("operation", "the required operation")
        reasoning = (
            f"I need to temporarily leave the current context to {operation}, "
            f"then return. I'll use the navigate_and_return tool for this round-trip."
        )
    else:
        reasoning = "I'll use a macro tool to handle this multi-step operation."

    # Prepend the original thought if available for richer context
    if thought:
        reasoning = f"{thought}\n\n{reasoning}"

    return reasoning


def _resolve_step_image(
    step_info: Dict,
    traj_file: Optional[Path],
    raw_data_dir: Optional[Path],
    image_base_dir: Path,
) -> Optional[str]:
    """Resolve the image path for a trajectory step, returning None if not found."""
    screenshot_path = step_info.get("screenshot_annotated", step_info.get("screenshot_clean", ""))
    if not screenshot_path:
        return None
    if traj_file and raw_data_dir:
        full_path = resolve_image_path(screenshot_path, traj_file, raw_data_dir, image_base_dir)
    else:
        full_path = str(image_base_dir / screenshot_path)
    if os.path.exists(full_path):
        return full_path
    return None


def build_macro_augmented_sample(
    crossing: Dict,
    raw_steps: List[Dict],
    macro_type: str,
    macro_args: Dict,
    image_base_dir: Path,
    state_registry: Dict[str, Dict],
    traj_file: Path = None,
    raw_data_dir: Path = None,
) -> Optional[Dict]:
    """
    Build a macro-augmented SFT sample from a crossing trajectory.

    The macro replaces the crossing span with a single tool call.
    """
    span_start = crossing["span_start"]
    span_end = crossing["span_end"]
    step_ids = crossing["step_ids"]
    execution_id = crossing["execution_id"]
    app = crossing["app"]

    # Map step_ids to raw_steps indices
    step_id_to_raw = {}
    for raw_step in raw_steps:
        sid = raw_step.get("step_id")
        if sid is not None:
            step_id_to_raw[sid] = raw_step

    # We need at minimum: the step at span_start (for the macro call)
    # and the step after span_end (for continuing)
    if span_start >= len(step_ids):
        return None

    macro_step_id = step_ids[span_start]
    macro_step_data = step_id_to_raw.get(macro_step_id)
    if macro_step_data is None:
        return None

    # Get the step after the crossing span for the "result" screenshot
    result_step_id = step_ids[span_end] if span_end < len(step_ids) else None
    result_step_data = step_id_to_raw.get(result_step_id) if result_step_id else None

    # Build messages
    messages = []
    instruction = macro_step_data.get("request", "")

    # Collect action history from steps before the crossing
    history_parts = []
    for i, sid in enumerate(step_ids):
        if i >= span_start:
            break
        step_data = step_id_to_raw.get(sid)
        if step_data:
            action = step_data.get("step", {}).get("action", {})
            func = action.get("function", "")
            if func:
                history_parts.append(f"Step {i+1}: {func}({json.dumps(action.get('args', {}))})")

    history = "\n".join(history_parts) if history_parts else "(none)"

    # ── Pre-crossing steps (if any non-trivial ones exist before span_start) ──
    for i in range(min(span_start, len(step_ids))):
        sid = step_ids[i]
        step_data = step_id_to_raw.get(sid)
        if step_data is None:
            continue

        step_info = step_data.get("step", {})
        controls = step_info.get("control_infos", {}).get("merged_controls_info", [])

        full_image_path = _resolve_step_image(step_info, traj_file, raw_data_dir, image_base_dir)
        if not full_image_path:
            continue

        element_list = format_element_list(controls) if controls else "(no elements detected)"

        # Build history up to this step
        step_history_parts = []
        for j in range(i):
            prev_sid = step_ids[j]
            prev_data = step_id_to_raw.get(prev_sid)
            if prev_data:
                prev_action = prev_data.get("step", {}).get("action", {})
                prev_func = prev_action.get("function", "")
                if prev_func:
                    step_history_parts.append(f"Step {j+1}: {prev_func}({json.dumps(prev_action.get('args', {}))})")

        step_history = "\n".join(step_history_parts) if step_history_parts else "(none)"

        user_text = _build_system_prompt_with_macros(instruction, step_history, element_list)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image", "image": full_image_path},
            ],
        })

        # Assistant response: normal action
        action = step_info.get("action", {})
        a11y_action, _ = convert_action_to_a11y_format(action, controls)
        thought = step_info.get("thought", "")
        assistant_msg = ""
        if thought:
            assistant_msg = f"Reasoning: {thought}\n\n"
        assistant_msg += f"<tool_call>\n{json.dumps(a11y_action, indent=2)}\n</tool_call>"
        messages.append({"role": "assistant", "content": assistant_msg})

    # ── Macro call step (at span_start) ──
    step_info = macro_step_data.get("step", {})
    controls = step_info.get("control_infos", {}).get("merged_controls_info", [])

    full_image_path = _resolve_step_image(step_info, traj_file, raw_data_dir, image_base_dir)
    if not full_image_path:
        return None

    element_list = format_element_list(controls) if controls else "(no elements detected)"
    user_text = _build_system_prompt_with_macros(instruction, history, element_list)

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_text},
            {"type": "image", "image": full_image_path},
        ],
    })

    # Assistant: macro tool call
    reasoning = _build_macro_reasoning(macro_type, macro_args, macro_step_data)
    macro_call = {
        "function": macro_type,
        "args": macro_args,
        "status": "CONTINUE",
    }
    assistant_msg = f"Reasoning: {reasoning}\n\n<tool_call>\n{json.dumps(macro_call, indent=2)}\n</tool_call>"
    messages.append({"role": "assistant", "content": assistant_msg})

    # ── Macro result (post-crossing screenshot) ──
    if result_step_data:
        result_step_info = result_step_data.get("step", {})
        result_image_path = _resolve_step_image(result_step_info, traj_file, raw_data_dir, image_base_dir)
        if result_image_path:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": "Macro executed successfully. Here is the current screen:"},
                    {"type": "image", "image": result_image_path},
                ],
            })

            # Continue with normal actions after the crossing
            result_controls = result_step_info.get("control_infos", {}).get("merged_controls_info", [])
            result_action = result_step_info.get("action", {})
            if result_action.get("function"):
                a11y_action, _ = convert_action_to_a11y_format(result_action, result_controls)
                thought = result_step_info.get("thought", "")
                assistant_msg = ""
                if thought:
                    assistant_msg = f"Reasoning: {thought}\n\n"
                assistant_msg += f"<tool_call>\n{json.dumps(a11y_action, indent=2)}\n</tool_call>"
                messages.append({"role": "assistant", "content": assistant_msg})

    # ── Post-crossing steps ──
    for i in range(span_end + 1, len(step_ids)):
        sid = step_ids[i]
        step_data = step_id_to_raw.get(sid)
        if step_data is None:
            continue

        step_info = step_data.get("step", {})
        controls = step_info.get("control_infos", {}).get("merged_controls_info", [])

        full_image_path = _resolve_step_image(step_info, traj_file, raw_data_dir, image_base_dir)
        if not full_image_path:
            continue

        element_list = format_element_list(controls) if controls else "(no elements detected)"

        # Build history including the macro call
        post_history_parts = list(history_parts)
        post_history_parts.append(f"Step {span_start+1}: {macro_type}({json.dumps(macro_args)})")
        for j in range(span_end, i):
            prev_sid = step_ids[j]
            prev_data = step_id_to_raw.get(prev_sid)
            if prev_data:
                prev_action = prev_data.get("step", {}).get("action", {})
                prev_func = prev_action.get("function", "")
                if prev_func:
                    post_history_parts.append(
                        f"Step {j+1}: {prev_func}({json.dumps(prev_action.get('args', {}))})"
                    )

        post_history = "\n".join(post_history_parts) if post_history_parts else "(none)"
        user_text = _build_system_prompt_with_macros(instruction, post_history, element_list)

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image", "image": full_image_path},
            ],
        })

        action = step_info.get("action", {})
        a11y_action, _ = convert_action_to_a11y_format(action, controls)
        thought = step_info.get("thought", "")
        assistant_msg = ""
        if thought:
            assistant_msg = f"Reasoning: {thought}\n\n"
        assistant_msg += f"<tool_call>\n{json.dumps(a11y_action, indent=2)}\n</tool_call>"
        messages.append({"role": "assistant", "content": assistant_msg})

    if len(messages) < 2:
        return None

    return {
        "messages": json.dumps(messages, ensure_ascii=False),
        "enable_thinking": True,
        "trajectory_id": execution_id,
        "step_id": macro_step_id,
        "app_domain": app,
        "has_macro": True,
    }


# ──────────────────────────────────────────────────────────────────────
# Step 5: Build macro playbook
# ──────────────────────────────────────────────────────────────────────

def build_playbook_entry(
    crossing: Dict,
    raw_steps: List[Dict],
    macro_type: str,
    macro_args: Dict,
) -> Dict:
    """Build a playbook entry recording the typical action sequence for this macro."""
    span_start = crossing["span_start"]
    span_end = crossing["span_end"]
    step_ids = crossing["step_ids"]

    step_id_to_raw = {}
    for s in raw_steps:
        sid = s.get("step_id")
        if sid is not None:
            step_id_to_raw[sid] = s

    action_sequence = []
    for i in range(span_start, min(span_end + 1, len(step_ids))):
        sid = step_ids[i]
        step_data = step_id_to_raw.get(sid)
        if step_data:
            action = step_data.get("step", {}).get("action", {})
            action_sequence.append({
                "step": i - span_start + 1,
                "function": action.get("function", ""),
                "args": action.get("args", {}),
                "control_text": action.get("control_test", ""),
            })

    return {
        "macro_type": macro_type,
        "macro_args": macro_args,
        "app": crossing["app"],
        "execution_id": crossing["execution_id"],
        "action_sequence": action_sequence,
        "num_steps": len(action_sequence),
    }


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare macro-augmented SFT training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--app", type=str, default=None, choices=APPS,
        help="Process a single app (default: all apps)",
    )
    parser.add_argument(
        "--crossing-threshold", type=float, default=1.0,
        help="Minimum f-value range for a trajectory to be considered crossing (default: 1.0)",
    )
    parser.add_argument(
        "--upsample-factor", type=int, default=10,
        help="Upsample factor for macro-augmented data in the mixed dataset (default: 10)",
    )
    parser.add_argument(
        "--fnet-dir", type=Path, default=DEFAULT_FNET_DIR,
        help=f"Directory containing per-app f_values.npz (default: {DEFAULT_FNET_DIR})",
    )
    parser.add_argument(
        "--transitions-dir", type=Path, default=DEFAULT_TRANSITIONS_DIR,
        help=f"Directory with transitions.jsonl and state_registry.json (default: {DEFAULT_TRANSITIONS_DIR})",
    )
    parser.add_argument(
        "--raw-data-dir", type=Path, default=DEFAULT_RAW_DATA_DIR,
        help=f"GUI-360 raw trajectory directory (default: {DEFAULT_RAW_DATA_DIR})",
    )
    parser.add_argument(
        "--image-base-dir", type=Path, default=DEFAULT_IMAGE_BASE_DIR,
        help=f"Base directory for screenshot images (default: {DEFAULT_IMAGE_BASE_DIR})",
    )
    parser.add_argument(
        "--baseline-parquet", type=Path, default=DEFAULT_BASELINE_PARQUET,
        help=f"Baseline SFT parquet for mixing (default: {DEFAULT_BASELINE_PARQUET})",
    )
    parser.add_argument(
        "--eval-parquet", type=Path, default=DEFAULT_EVAL_PARQUET,
        help=f"Evaluation parquet for augmenting eval set (default: {DEFAULT_EVAL_PARQUET})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-crossings-per-app", type=int, default=None,
        help="Limit crossings per app (for testing)",
    )

    args = parser.parse_args()

    apps = [args.app] if args.app else APPS
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "crossing_trajectories").mkdir(exist_ok=True)

    # ── Load shared data ──
    logger.info("Loading transitions and state registry...")
    transitions = load_transitions(args.transitions_dir)
    state_registry = load_state_registry(args.transitions_dir)
    logger.info(f"Loaded {len(transitions)} transitions, {len(state_registry)} states")

    # ── Process each app ──
    all_macro_samples = []
    all_crossings = {}
    playbook_entries = []
    stats = {
        "crossing_threshold": args.crossing_threshold,
        "upsample_factor": args.upsample_factor,
        "per_app": {},
    }

    for app in apps:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing app: {app}")
        logger.info(f"{'='*60}")

        # Step 1: Load f-values and identify crossings
        f_values = load_f_values(args.fnet_dir, app)
        if not f_values:
            logger.warning(f"No f-values for {app}, skipping")
            continue
        logger.info(f"Loaded {len(f_values)} f-values for {app}")

        crossings = identify_crossings(
            transitions, f_values, app, threshold=args.crossing_threshold,
        )
        if args.max_crossings_per_app:
            crossings = crossings[: args.max_crossings_per_app]

        logger.info(f"Found {len(crossings)} crossing trajectories for {app}")
        all_crossings[app] = crossings

        # Save crossing list
        crossing_path = output_dir / "crossing_trajectories" / f"{app}_crossings.json"
        # Save without f_sequence/hash_sequence for readability
        crossings_export = []
        for c in crossings:
            c_export = {k: v for k, v in c.items() if k not in ("f_sequence", "hash_sequence")}
            crossings_export.append(c_export)
        with open(crossing_path, "w") as f:
            json.dump(crossings_export, f, indent=2)

        # Step 2-4: Process each crossing trajectory
        app_samples = []
        macro_type_counts = defaultdict(int)
        skipped_no_file = 0
        skipped_build_fail = 0

        for ci, crossing in enumerate(crossings):
            eid = crossing["execution_id"]

            # Find and load raw trajectory
            traj_file = find_trajectory_file(args.raw_data_dir, eid, app)
            if traj_file is None:
                skipped_no_file += 1
                continue

            raw_steps = load_raw_trajectory(traj_file)
            if not raw_steps:
                skipped_no_file += 1
                continue

            # Classify crossing
            macro_type, macro_args = classify_crossing(crossing, state_registry)
            macro_type_counts[macro_type] += 1

            # Build playbook entry
            playbook_entries.append(
                build_playbook_entry(crossing, raw_steps, macro_type, macro_args)
            )

            # Build macro-augmented sample
            sample = build_macro_augmented_sample(
                crossing, raw_steps, macro_type, macro_args,
                args.image_base_dir, state_registry,
                traj_file=traj_file, raw_data_dir=args.raw_data_dir,
            )
            if sample is None:
                skipped_build_fail += 1
                continue

            app_samples.append(sample)

            if (ci + 1) % 50 == 0:
                logger.info(f"  Processed {ci+1}/{len(crossings)} crossings, {len(app_samples)} samples")

        all_macro_samples.extend(app_samples)

        logger.info(f"  {app}: {len(app_samples)} macro samples generated")
        logger.info(f"  Macro type distribution: {dict(macro_type_counts)}")
        if skipped_no_file:
            logger.info(f"  Skipped (no file): {skipped_no_file}")
        if skipped_build_fail:
            logger.info(f"  Skipped (build fail): {skipped_build_fail}")

        stats["per_app"][app] = {
            "num_crossings": len(crossings),
            "num_samples": len(app_samples),
            "macro_type_distribution": dict(macro_type_counts),
            "skipped_no_file": skipped_no_file,
            "skipped_build_fail": skipped_build_fail,
        }

    # ── Save macro playbook ──
    playbook_path = output_dir / "macro_playbook.json"
    with open(playbook_path, "w") as f:
        json.dump(playbook_entries, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(playbook_entries)} playbook entries to {playbook_path}")

    # ── Save tool definitions ──
    tool_defs_path = output_dir / "macro_tool_definitions.json"
    with open(tool_defs_path, "w") as f:
        json.dump(MACRO_TOOL_DEFINITIONS, f, indent=2)
    logger.info(f"Saved tool definitions to {tool_defs_path}")

    # ── Create macro-only parquet ──
    if not all_macro_samples:
        logger.error("No macro samples generated! Check input data and thresholds.")
        stats["total_macro_samples"] = 0
        with open(output_dir / "statistics.json", "w") as f:
            json.dump(stats, f, indent=2)
        return

    df_macro = pd.DataFrame(all_macro_samples)
    df_macro["messages"] = df_macro["messages"].astype("object")
    macro_only_path = output_dir / "macro_augmented_train.parquet"
    df_macro.to_parquet(macro_only_path, index=False, engine="pyarrow")
    logger.info(f"Saved {len(df_macro)} macro samples to {macro_only_path}")

    # ── Create mixed parquet (baseline + upsampled macro) ──
    if args.baseline_parquet.exists():
        logger.info(f"Loading baseline parquet: {args.baseline_parquet}")
        df_baseline = pd.read_parquet(args.baseline_parquet)

        # Add has_macro column to baseline
        df_baseline["has_macro"] = False

        # Upsample macro data
        df_macro_upsampled = pd.concat(
            [df_macro] * args.upsample_factor, ignore_index=True,
        )
        logger.info(
            f"Upsampled macro data: {len(df_macro)} x {args.upsample_factor} = {len(df_macro_upsampled)}"
        )

        # Mix
        df_mixed = pd.concat([df_baseline, df_macro_upsampled], ignore_index=True)
        df_mixed = df_mixed.sample(frac=1, random_state=42).reset_index(drop=True)
        df_mixed["messages"] = df_mixed["messages"].astype("object")

        mixed_path = output_dir / "macro_mixed_train.parquet"
        df_mixed.to_parquet(mixed_path, index=False, engine="pyarrow")
        logger.info(f"Saved mixed dataset: {len(df_mixed)} samples to {mixed_path}")

        stats["baseline_samples"] = len(df_baseline)
        stats["macro_upsampled"] = len(df_macro_upsampled)
        stats["mixed_total"] = len(df_mixed)
        stats["macro_percentage"] = round(len(df_macro_upsampled) / len(df_mixed) * 100, 1)
    else:
        logger.warning(f"Baseline parquet not found: {args.baseline_parquet}, skipping mixed dataset")

    # ── Create eval parquet with some macro samples ──
    if args.eval_parquet.exists():
        logger.info(f"Loading eval parquet: {args.eval_parquet}")
        df_eval = pd.read_parquet(args.eval_parquet)
        df_eval["has_macro"] = False

        # Take 10% of macro samples for eval (no upsampling)
        n_eval_macro = max(1, len(df_macro) // 10)
        df_eval_macro = df_macro.sample(n=min(n_eval_macro, len(df_macro)), random_state=42)

        df_eval_mixed = pd.concat([df_eval, df_eval_macro], ignore_index=True)
        df_eval_mixed["messages"] = df_eval_mixed["messages"].astype("object")

        eval_path = output_dir / "macro_mixed_eval.parquet"
        df_eval_mixed.to_parquet(eval_path, index=False, engine="pyarrow")
        logger.info(f"Saved eval dataset: {len(df_eval_mixed)} samples ({n_eval_macro} macro)")

        stats["eval_samples"] = len(df_eval_mixed)
        stats["eval_macro_samples"] = n_eval_macro
    else:
        logger.warning(f"Eval parquet not found: {args.eval_parquet}, skipping eval dataset")

    # ── Save statistics ──
    stats["total_macro_samples"] = len(all_macro_samples)
    stats["total_playbook_entries"] = len(playbook_entries)

    with open(output_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics to {output_dir / 'statistics.json'}")

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"  Macro SFT Data Preparation Summary")
    print(f"{'='*60}")
    print(f"  Total macro samples:    {len(all_macro_samples)}")
    for app in apps:
        app_stats = stats["per_app"].get(app, {})
        print(f"  {app:>6s}: {app_stats.get('num_crossings', 0):>4d} crossings → {app_stats.get('num_samples', 0):>4d} samples")
        dist = app_stats.get("macro_type_distribution", {})
        for mt, cnt in dist.items():
            print(f"          {mt}: {cnt}")
    if "mixed_total" in stats:
        print(f"\n  Mixed dataset:          {stats['mixed_total']} samples")
        print(f"  Macro percentage:       {stats['macro_percentage']}%")
    print(f"\n  Output directory:       {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
