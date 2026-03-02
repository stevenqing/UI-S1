#!/usr/bin/env python3
"""
Convert UI-S1 JSONL trajectory data to parquet format for SFT training.

This creates a multiturn conversation format where each step is:
- User message: task goal + image
- Assistant message: action response

Usage:
    # Convert training data
    python scripts/prepare_gui_sft_parquet.py \
        --input datasets/ui_s1_train.jsonl \
        --output datasets/ui_s1_train_sft.parquet

    # Convert evaluation data
    python scripts/prepare_gui_sft_parquet.py \
        --input evaluation/dataset/android_control_evaluation_std.jsonl \
        --output datasets/ui_s1_eval_sft.parquet

    # Convert with limit
    python scripts/prepare_gui_sft_parquet.py \
        --input datasets/ui_s1_train.jsonl \
        --output datasets/ui_s1_train_sft.parquet \
        --max-trajectories 100
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def format_action_to_response(action_content: Dict[str, Any]) -> str:
    """
    Format action content to JSON response string.

    Args:
        action_content: Dictionary containing action type and parameters

    Returns:
        JSON string representation of the action
    """
    action = action_content.get("action", "")
    response_parts = [f'{{"action": "{action}"']

    # Add action-specific parameters
    if action == "click":
        coord = action_content.get("coordinate")
        if coord is not None:
            response_parts.append(f', "coordinate": {list(coord)}')

    elif action == "type":
        text = action_content.get("text", "")
        # Escape quotes and newlines
        text = text.replace('"', '\\"').replace('\n', '\\n')
        response_parts.append(f', "text": "{text}"')

    elif action == "swipe":
        coord = action_content.get("coordinate")
        coord2 = action_content.get("coordinate2")
        if coord is not None and coord2 is not None:
            response_parts.append(f', "coordinate": {list(coord)}, "coordinate2": {list(coord2)}')

    elif action == "open":
        text = action_content.get("text", "")
        text = text.replace('"', '\\"')
        response_parts.append(f', "text": "{text}"')

    elif action == "long_press":
        coord = action_content.get("coordinate")
        if coord is not None:
            response_parts.append(f', "coordinate": {list(coord)}')

    elif action == "wait":
        time_val = action_content.get("time", 1)
        response_parts.append(f', "time": {time_val}')

    elif action == "system_button":
        button = action_content.get("button", "")
        response_parts.append(f', "button": "{button}"')

    elif action == "terminate":
        status = action_content.get("status", "success")
        response_parts.append(f', "status": "{status}"')

    elif action == "key":
        key = action_content.get("key", "")
        response_parts.append(f', "key": "{key}"')

    elif action == "answer":
        text = action_content.get("text", "")
        text = text.replace('"', '\\"').replace('\n', '\\n')
        response_parts.append(f', "text": "{text}"')

    elif action == "navigate":
        text = action_content.get("text", "")
        text = text.replace('"', '\\"')
        response_parts.append(f', "text": "{text}"')

    response_parts.append("}")
    return "".join(response_parts)


def trajectory_to_messages(trajectory: Dict[str, Any], include_terminate: bool = True) -> List[Dict[str, Any]]:
    """
    Convert a trajectory to multiturn messages format.

    Each step becomes a user-assistant message pair.

    Args:
        trajectory: Dictionary containing goal, steps, and metadata
        include_terminate: Whether to include terminate actions

    Returns:
        List of message dictionaries with role and content
    """
    messages = []
    goal = trajectory.get("goal", "")
    steps = trajectory.get("steps", [])

    for step_idx, step in enumerate(steps):
        action_content = step.get("action_content", {})
        action = action_content.get("action", "")

        # Skip terminate actions if not requested
        if action == "terminate" and not include_terminate:
            continue

        screenshot_path = step.get("screenshot", "")

        # Check if screenshot exists
        if not os.path.exists(screenshot_path):
            print(f"Warning: Screenshot not found: {screenshot_path}", file=sys.stderr)
            continue

        # User message with image and task
        user_content = [
            {
                "type": "text",
                "text": f"Task: {goal}\n\nPlease perform the appropriate action to complete this task."
            },
            {"type": "image", "image": screenshot_path}
        ]
        messages.append({"role": "user", "content": user_content})

        # Assistant message with action
        response = format_action_to_response(action_content)
        messages.append({"role": "assistant", "content": response})

    return messages


def convert_jsonl_to_parquet(
    input_jsonl: str,
    output_parquet: str,
    only_successful: bool = True,
    include_terminate: bool = True,
    max_trajectories: int = -1,
    validate_images: bool = True,
) -> None:
    """
    Convert UI-S1 JSONL trajectory data to parquet format for SFT training.

    Args:
        input_jsonl: Path to input JSONL file
        output_parquet: Path to output parquet file
        only_successful: Only include successful trajectories
        include_terminate: Include terminate actions in training
        max_trajectories: Maximum number of trajectories to process (-1 for all)
        validate_images: Validate that image files exist
    """
    data_samples = []
    action_counts = {}
    total_steps = 0
    skipped_no_image = 0
    skipped_unsuccessful = 0

    print(f"Reading from: {input_jsonl}", file=sys.stderr)

    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue

            if max_trajectories > 0 and len(data_samples) >= max_trajectories:
                break

            try:
                trajectory = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}", file=sys.stderr)
                continue

            # Filter by success status
            is_successful = trajectory.get("is_successful", True)
            if only_successful and not is_successful:
                skipped_unsuccessful += 1
                continue

            # Convert trajectory to messages
            messages = trajectory_to_messages(trajectory, include_terminate=include_terminate)

            if not messages:
                continue

            # Count actions for statistics
            for msg in messages:
                if msg["role"] == "assistant":
                    try:
                        action_data = json.loads(msg["content"])
                        action = action_data.get("action", "unknown")
                        action_counts[action] = action_counts.get(action, 0) + 1
                        total_steps += 1
                    except json.JSONDecodeError:
                        pass

            # Store messages as JSON string for parquet compatibility
            data_samples.append({
                "messages": json.dumps(messages, ensure_ascii=False),
            })

            # Progress update
            if len(data_samples) % 100 == 0:
                print(f"Processed {len(data_samples)} trajectories...", file=sys.stderr)

    # Create output directory if needed
    output_dir = os.path.dirname(output_parquet)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Create DataFrame and save to parquet
    # Convert messages to object type for proper parquet serialization
    df = pd.DataFrame(data_samples)
    df["messages"] = df["messages"].astype("object")

    # Use pyarrow engine for better compatibility
    df.to_parquet(output_parquet, index=False, engine="pyarrow")

    # Print summary statistics
    print(f"\n" + "="*60, file=sys.stderr)
    print(f"Conversion Summary", file=sys.stderr)
    print(f"="*60, file=sys.stderr)
    print(f"Input file:  {input_jsonl}", file=sys.stderr)
    print(f"Output file: {output_parquet}", file=sys.stderr)
    print(f"\nTrajectories converted: {len(data_samples)}", file=sys.stderr)
    print(f"Total training samples (user-assistant pairs): {total_steps}", file=sys.stderr)
    if skipped_unsuccessful > 0:
        print(f"Skipped (unsuccessful): {skipped_unsuccessful}", file=sys.stderr)
    if skipped_no_image > 0:
        print(f"Skipped (missing images): {skipped_no_image}", file=sys.stderr)

    print(f"\nAction distribution:", file=sys.stderr)
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {action:20s}: {count:6d}", file=sys.stderr)
    print(f"="*60, file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Convert UI-S1 JSONL to parquet format for SFT training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert training data
  python scripts/prepare_gui_sft_parquet.py \\
      --input datasets/ui_s1_train.jsonl \\
      --output datasets/ui_s1_train_sft.parquet

  # Convert with no terminate actions
  python scripts/prepare_gui_sft_parquet.py \\
      --input datasets/ui_s1_train.jsonl \\
      --output datasets/ui_s1_train_sft.parquet \\
      --no-terminate

  # Convert with limit
  python scripts/prepare_gui_sft_parquet.py \\
      --input datasets/ui_s1_train.jsonl \\
      --output datasets/ui_s1_train_sft.parquet \\
      --max-trajectories 100
        """
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output parquet file path"
    )
    parser.add_argument(
        "--only-successful",
        action="store_true",
        default=True,
        help="Only include successful trajectories (default: True)"
    )
    parser.add_argument(
        "--include-unsuccessful",
        action="store_true",
        help="Include unsuccessful trajectories in training"
    )
    parser.add_argument(
        "--no-terminate",
        action="store_true",
        help="Exclude terminate actions from training"
    )
    parser.add_argument(
        "--max-trajectories", "-n",
        type=int,
        default=-1,
        help="Maximum number of trajectories to process (-1 for all, default: -1)"
    )
    parser.add_argument(
        "--skip-image-validation",
        action="store_true",
        help="Skip validation that image files exist"
    )

    args = parser.parse_args()

    # Handle conflicting options
    only_successful = args.only_successful
    if args.include_unsuccessful:
        only_successful = False

    include_terminate = not args.no_terminate
    validate_images = not args.skip_image_validation

    # Convert the data
    convert_jsonl_to_parquet(
        input_jsonl=args.input,
        output_parquet=args.output,
        only_successful=only_successful,
        include_terminate=include_terminate,
        max_trajectories=args.max_trajectories,
        validate_images=validate_images,
    )


if __name__ == "__main__":
    main()
