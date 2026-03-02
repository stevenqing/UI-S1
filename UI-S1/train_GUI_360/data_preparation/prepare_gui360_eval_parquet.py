#!/usr/bin/env python3
"""
Convert GUI-360 raw test trajectories to parquet format for evaluation.

GUI-360 raw test data format (JSONL files):
{
    "execution_id": "excel_1_101",
    "request": "Remove the header...",
    "step": {
        "screenshot_clean": "success/excel_1_101/action_step1.png",
        "action": {...},
        "thought": "...",
        "observation": "..."
    }
}

Usage:
    python scripts/GUI_360/prepare_gui360_eval_parquet.py \
        --input-dir datasets/GUI-360/test/data \
        --output train_GUI_360/data/gui360_eval_sft.parquet \
        --max-samples 1000
"""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd


def format_action_response(action: Dict[str, Any]) -> str:
    """Format action to JSON response string."""
    action_type = action.get("function", "click")

    response = {"action": action_type}

    # Add coordinates
    if "coordinate_x" in action and "coordinate_y" in action:
        response["coordinate"] = [action["coordinate_x"], action["coordinate_y"]]
    elif "desktop_coordinate_x" in action and "desktop_coordinate_y" in action:
        response["coordinate"] = [action["desktop_coordinate_x"], action["desktop_coordinate_y"]]

    # Add text for type actions
    if action_type == "type":
        args = action.get("args", {})
        if isinstance(args, dict) and "keys" in args:
            response["text"] = args["keys"]
        elif "control_text" in action:
            response["text"] = action["control_text"]

    return json.dumps(response, ensure_ascii=False)


def process_raw_trajectory(
    trajectory: Dict[str, Any],
    base_path: str,
    validate_images: bool = False
) -> Optional[Dict]:
    """
    Process a raw GUI-360 trajectory to message format.
    """
    request = trajectory.get("request", "")
    step = trajectory.get("step", {})

    if not request or not step:
        return None

    # Get screenshot path
    screenshot = step.get("screenshot_clean", step.get("screenshot_desktop", ""))
    if not screenshot:
        return None

    full_img_path = os.path.join(base_path, screenshot)

    if validate_images and not os.path.exists(full_img_path):
        return None

    # Build prompt
    observation = step.get("observation", "")
    thought = step.get("thought", "")

    prompt_parts = [f"You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.\n\nThe instruction is:\n{request}"]

    if observation:
        prompt_parts.append(f"\nObservation: {observation}")

    prompt_parts.append("\n\nPlease decide the next action.")
    prompt = "".join(prompt_parts)

    # Build user message
    user_content = [
        {"type": "text", "text": prompt},
        {"type": "image", "image": full_img_path}
    ]

    messages = [{"role": "user", "content": user_content}]

    # Build assistant response
    action = step.get("action", {})
    if action:
        response = format_action_response(action)
        messages.append({"role": "assistant", "content": response})
    else:
        return None

    return {"messages": json.dumps(messages, ensure_ascii=False)}


def convert_raw_to_parquet(
    input_dir: str,
    output_file: str,
    max_samples: int = -1,
    validate_images: bool = False
) -> None:
    """
    Convert GUI-360 raw test trajectories to parquet format.
    """
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
                    result = process_raw_trajectory(trajectory, base_path, validate_images)

                    if result:
                        samples.append(result)
                    else:
                        skipped += 1

        except Exception as e:
            print(f"Warning: Error processing {jsonl_file}: {e}", file=sys.stderr)
            continue

        if len(samples) % 500 == 0 and len(samples) > 0:
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
        description="Convert GUI-360 raw test data to parquet format",
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

    convert_raw_to_parquet(
        input_dir=args.input_dir,
        output_file=args.output,
        max_samples=args.max_samples,
        validate_images=args.validate_images
    )


if __name__ == "__main__":
    main()
