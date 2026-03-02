#!/usr/bin/env python3
"""
Convert GUI-360 raw JSONL to parquet format with thinking/reasoning included.

This script reads from raw JSONL trajectory files (not processed training_data.json)
and includes the 'thought' field as reasoning in the assistant response.

Key differences from prepare_gui360_sft_parquet.py:
1. Reads from raw JSONL files (train/data/**/*.jsonl)
2. Includes 'thought' field as reasoning prefix
3. Adds 'enable_thinking' column for Qwen tokenizer
4. Builds action history from trajectory steps

Usage:
    # Full training data with thinking
    python train_GUI_360/data_preparation/prepare_gui360_sft_with_thinking.py \
        --input-dir datasets/GUI-360/train/data \
        --output train_GUI_360/data/gui360_train_sft_with_thinking.parquet \
        --image-base-dir datasets/GUI-360/train

    # Test data
    python train_GUI_360/data_preparation/prepare_gui360_sft_with_thinking.py \
        --input-dir datasets/GUI-360/test/data \
        --output train_GUI_360/data/gui360_test_sft_with_thinking.parquet \
        --image-base-dir datasets/GUI-360/test

    # With sample limit for testing
    python train_GUI_360/data_preparation/prepare_gui360_sft_with_thinking.py \
        --input-dir datasets/GUI-360/train/data \
        --output train_GUI_360/data/gui360_train_sft_with_thinking_sample.parquet \
        --image-base-dir datasets/GUI-360/train \
        --max-samples 1000
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd


# System prompt template (matches current training format)
SYSTEM_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
<action>
- click
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to click at.
    - button: str, The mouse button to click. One of ''left'', ''right'', ''middle'' or ''x'' (Default: ''left'')
    - double: bool, Whether to perform a double click or not (Default: False)
  - Example: click(coordinate=[100, 200], button=''left'', double=False), click(coordinate=[200, 300], button=''x'')
- type
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to type at.
    - text: str, The text to input.
    - clear_current_text: bool, Whether to clear the current text before typing. (Default: False)
    - keys: str, The special keys to press. Such as ''{{ENTER}}'', ''{{TAB}}'', ''{{BACKSPACE}}'', ''{{ESC}}'', ''{{DOWN}}'', etc.
    - control_focus: bool, Whether to focus on the coordinate position first before pressing the keys. (Default: False)
  - Example: type(coordinate=[100, 200], text=''Hello''), type(coordinate=[200, 300], keys=''{{ENTER}}'', control_focus=True)
- drag
  - Args:
    - start_coordinate: [x, y], the absolute position on the screen you want to drag from.
    - end_coordinate: [x, y], the absolute position on the screen you want to drag to.
    - button: str, The mouse button to drag. One of ''left'', ''right''. (Default: ''left'')
    - duration: float, The duration of the drag action in seconds. (Default: 1.0)
  - Example: drag(start_coordinate=[100, 200], end_coordinate=[200, 300], button=''left'', duration=1.0)
- wheel_mouse_input
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to scroll at.
    - wheel_dist: int, The number of wheel notches to scroll. Positive for up, negative for down.
  - Example: wheel_mouse_input(coordinate=[100, 200], wheel_dist=-5)
- hotkey
  - Args:
    - keys: str, The hotkey keys to press. Use format like "Ctrl+C", "Alt+F4", "Ctrl+Shift+S"
  - Example: hotkey(keys="Ctrl+C"), hotkey(keys="Ctrl+Shift+S")
</action>

First, explain your reasoning process—describe how you analyze the screenshot, understand the current state, and determine what action should be taken next based on the instruction and previous actions.

Then output your action within <tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{"coordinate": [x, y], ...}},
  "status": "CONTINUE"
}}
</tool_call>

If you think the task is finished, output:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


def format_action_for_history(action: Dict, step_id: int) -> str:
    """Format an action for the history string."""
    function = action.get('function', '')
    if not function:
        return f"Step {step_id}: Task completed (FINISH)"

    args = action.get('args', {})
    coord_x = action.get('coordinate_x')
    coord_y = action.get('coordinate_y')

    if function == 'click':
        button = args.get('button', 'left')
        double = args.get('double', False)
        return f"Step {step_id}: click(coordinate=[{coord_x}, {coord_y}], button='{button}', double={double})"

    elif function == 'type':
        text = args.get('text', '')
        keys = args.get('keys', '')
        if text:
            return f"Step {step_id}: type(coordinate=[{coord_x}, {coord_y}], text='{text[:30]}...')" if len(text) > 30 else f"Step {step_id}: type(coordinate=[{coord_x}, {coord_y}], text='{text}')"
        elif keys:
            return f"Step {step_id}: type(coordinate=[{coord_x}, {coord_y}], keys='{keys}')"
        else:
            return f"Step {step_id}: type(coordinate=[{coord_x}, {coord_y}])"

    elif function == 'drag':
        start = args.get('start_coordinate', [coord_x, coord_y])
        end = args.get('end_coordinate', [0, 0])
        return f"Step {step_id}: drag(start_coordinate={start}, end_coordinate={end})"

    elif function == 'wheel_mouse_input':
        wheel_dist = args.get('wheel_dist', 0)
        return f"Step {step_id}: wheel_mouse_input(coordinate=[{coord_x}, {coord_y}], wheel_dist={wheel_dist})"

    elif function == 'hotkey':
        keys = args.get('keys', '')
        return f"Step {step_id}: hotkey(keys='{keys}')"

    else:
        # Generic format for other functions
        return f"Step {step_id}: {function}({args})"


def build_action_json(action: Dict, step: Dict) -> Dict:
    """Build the action JSON for the assistant response."""
    function = action.get('function', '')
    status = step.get('status', 'CONTINUE')

    # Handle FINISH status
    if status in ['FINISH', 'OVERALL_FINISH'] or not function:
        return {
            "function": "",
            "args": {},
            "status": "FINISH"
        }

    # Get coordinates
    coord_x = action.get('coordinate_x')
    coord_y = action.get('coordinate_y')
    args = action.get('args', {})

    # Build args based on function type
    result_args = {}

    if function == 'click':
        result_args['coordinate'] = [coord_x, coord_y]
        if args.get('button') and args['button'] != 'left':
            result_args['button'] = args['button']
        if args.get('double'):
            result_args['double'] = True

    elif function == 'type':
        result_args['coordinate'] = [coord_x, coord_y]
        if args.get('text'):
            result_args['text'] = args['text']
        if args.get('keys'):
            result_args['keys'] = args['keys']
        if args.get('clear_current_text'):
            result_args['clear_current_text'] = True
        if args.get('control_focus'):
            result_args['control_focus'] = True

    elif function == 'drag':
        result_args['start_coordinate'] = args.get('start_coordinate', [coord_x, coord_y])
        result_args['end_coordinate'] = args.get('end_coordinate', [0, 0])
        if args.get('button') and args['button'] != 'left':
            result_args['button'] = args['button']
        if args.get('duration') and args['duration'] != 1.0:
            result_args['duration'] = args['duration']

    elif function == 'wheel_mouse_input':
        result_args['coordinate'] = [coord_x, coord_y]
        result_args['wheel_dist'] = args.get('wheel_dist', 0)

    elif function == 'hotkey':
        result_args['keys'] = args.get('keys', '')

    else:
        # For other functions, include coordinate and all args
        result_args['coordinate'] = [coord_x, coord_y]
        result_args.update(args)

    return {
        "function": function,
        "args": result_args,
        "status": "CONTINUE"
    }


def create_assistant_response(step: Dict, include_thinking: bool = True) -> str:
    """Create the assistant response with optional thinking/reasoning."""
    thought = step.get('thought', '')
    action = step.get('action', {})
    action_json = build_action_json(action, step)

    response = ""

    # Add reasoning if available and requested
    if include_thinking and thought:
        response += f"Reasoning: {thought}\n\n"

    # Add tool call
    response += f"<tool_call>\n{json.dumps(action_json, indent=4)}\n</tool_call>"

    return response


def process_trajectory_file(
    jsonl_path: str,
    image_base_dir: str,
    include_thinking: bool = True,
    include_history: bool = True,
    validate_images: bool = False
) -> List[Dict]:
    """
    Process a single trajectory JSONL file and return training samples.

    Each line in the JSONL file represents one step in the trajectory.
    We create one training sample per step.

    Args:
        jsonl_path: Path to the trajectory JSONL file
        image_base_dir: Base directory for image paths
        include_thinking: Whether to include thought in assistant response
        include_history: Whether to include action history in prompt
        validate_images: Whether to check if images exist

    Returns:
        List of training samples
    """
    samples = []

    # Read all steps from the trajectory
    steps_data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                steps_data.append(data)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line in {jsonl_path}: {e}", file=sys.stderr)
                continue

    if not steps_data:
        return []

    # Sort by step_id to ensure correct order
    steps_data.sort(key=lambda x: x.get('step_id', 0))

    # Build history and create samples
    history_actions = []

    for data in steps_data:
        step = data.get('step', {})
        step_id = data.get('step_id', 0)
        instruction = data.get('request', '')

        # Get screenshot path
        screenshot_path = step.get('screenshot_clean', '')
        if not screenshot_path:
            continue

        # Build full image path
        # The screenshot_path is relative like "success/ppt_1_42/action_step1.png"
        # We need to construct the full path based on image_base_dir
        app_domain = data.get('app_domain', '')
        execution_id = data.get('execution_id', '')

        # Try different path constructions
        possible_paths = [
            os.path.join(image_base_dir, 'data', app_domain, 'in_app', screenshot_path),
            os.path.join(image_base_dir, 'data', app_domain, 'search', screenshot_path),
            os.path.join(image_base_dir, 'data', app_domain, 'online', screenshot_path),
            os.path.join(image_base_dir, screenshot_path),
        ]

        full_image_path = None
        for path in possible_paths:
            if os.path.exists(path) or not validate_images:
                full_image_path = path
                break

        if not full_image_path:
            # Try to construct from the jsonl path
            jsonl_dir = os.path.dirname(jsonl_path)
            full_image_path = os.path.join(jsonl_dir, screenshot_path)

        if validate_images and not os.path.exists(full_image_path):
            print(f"Warning: Image not found: {full_image_path}", file=sys.stderr)
            continue

        # Build history string
        if include_history and history_actions:
            history_str = "\n".join(history_actions)
        else:
            history_str = "(none)"

        # Create user message
        user_text = SYSTEM_PROMPT.format(
            instruction=instruction,
            history=history_str
        )

        user_content = [
            {"type": "text", "text": user_text},
            {"type": "image", "image": full_image_path}
        ]

        # Create assistant response with thinking
        assistant_response = create_assistant_response(step, include_thinking=include_thinking)

        # Build messages
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_response}
        ]

        # Create sample
        sample = {
            "messages": json.dumps(messages, ensure_ascii=False),
            "enable_thinking": include_thinking,
            "trajectory_id": data.get('execution_id', ''),
            "step_id": step_id,
            "app_domain": app_domain
        }

        samples.append(sample)

        # Update history for next step
        action = step.get('action', {})
        history_str_new = format_action_for_history(action, step_id)
        history_actions.append(history_str_new)

    return samples


def find_trajectory_files(input_dir: str, apps: List[str] = None) -> List[str]:
    """Find all trajectory JSONL files in the input directory."""
    input_path = Path(input_dir)

    if apps is None:
        apps = ['excel', 'word', 'ppt']

    jsonl_files = []
    for app in apps:
        app_dir = input_path / app
        if app_dir.exists():
            for jsonl_file in app_dir.rglob("*.jsonl"):
                jsonl_files.append(str(jsonl_file))

    return sorted(jsonl_files)


def main():
    parser = argparse.ArgumentParser(
        description="Convert GUI-360 raw JSONL to parquet with thinking/reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        required=True,
        help="Input directory containing trajectory JSONL files (e.g., datasets/GUI-360/train/data)"
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
        help="Base directory for image paths (e.g., datasets/GUI-360/train)"
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=-1,
        help="Maximum samples to process (-1 for all)"
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=-1,
        help="Maximum trajectories to process (-1 for all)"
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable thinking/reasoning in assistant response"
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Disable action history in prompts"
    )
    parser.add_argument(
        "--validate-images",
        action="store_true",
        help="Validate that image files exist"
    )
    parser.add_argument(
        "--apps",
        type=str,
        nargs="+",
        default=["excel", "word", "ppt"],
        help="Applications to include (default: excel word ppt)"
    )

    args = parser.parse_args()

    include_thinking = not args.no_thinking
    include_history = not args.no_history

    print(f"Settings:", file=sys.stderr)
    print(f"  Include thinking: {include_thinking}", file=sys.stderr)
    print(f"  Include history: {include_history}", file=sys.stderr)
    print(f"  Applications: {args.apps}", file=sys.stderr)
    print(f"  Validate images: {args.validate_images}", file=sys.stderr)
    print(file=sys.stderr)

    # Find trajectory files
    jsonl_files = find_trajectory_files(args.input_dir, args.apps)
    print(f"Found {len(jsonl_files)} trajectory files", file=sys.stderr)

    if args.max_trajectories > 0:
        jsonl_files = jsonl_files[:args.max_trajectories]
        print(f"Limited to {args.max_trajectories} trajectories", file=sys.stderr)

    # Process trajectories
    all_samples = []
    processed = 0
    skipped = 0

    for i, jsonl_path in enumerate(jsonl_files):
        if args.max_samples > 0 and len(all_samples) >= args.max_samples:
            break

        try:
            samples = process_trajectory_file(
                jsonl_path,
                args.image_base_dir,
                include_thinking=include_thinking,
                include_history=include_history,
                validate_images=args.validate_images
            )
            all_samples.extend(samples)
            processed += 1
        except Exception as e:
            print(f"Error processing {jsonl_path}: {e}", file=sys.stderr)
            skipped += 1
            continue

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(jsonl_files)} files, {len(all_samples)} samples", file=sys.stderr)

    print(f"\nProcessed {processed} trajectories, skipped {skipped}", file=sys.stderr)
    print(f"Total samples: {len(all_samples)}", file=sys.stderr)

    if not all_samples:
        print("ERROR: No samples generated!", file=sys.stderr)
        return 1

    # Apply max_samples limit
    if args.max_samples > 0 and len(all_samples) > args.max_samples:
        all_samples = all_samples[:args.max_samples]
        print(f"Limited to {args.max_samples} samples", file=sys.stderr)

    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Create DataFrame and save
    df = pd.DataFrame(all_samples)

    # Ensure correct types
    df["messages"] = df["messages"].astype("object")
    df["enable_thinking"] = df["enable_thinking"].astype("bool")

    # Save to parquet
    df.to_parquet(args.output, index=False, engine="pyarrow")

    # Print summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Conversion Complete", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)
    print(f"Samples: {len(all_samples)}", file=sys.stderr)
    print(f"File size: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB", file=sys.stderr)
    print(f"Columns: {df.columns.tolist()}", file=sys.stderr)
    print(f"Enable thinking: {df['enable_thinking'].all()}", file=sys.stderr)

    # Show sample
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Sample Assistant Response:", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    sample_messages = json.loads(all_samples[0]['messages'])
    for msg in sample_messages:
        if msg['role'] == 'assistant':
            print(msg['content'][:500], file=sys.stderr)
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
