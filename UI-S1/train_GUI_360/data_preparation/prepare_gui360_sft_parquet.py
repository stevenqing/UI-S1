#!/usr/bin/env python3
"""
Convert GUI-360 dataset to parquet format for SFT training.

GUI-360 pre-processed data format (training_data.json):
[
    {
        "id": "sample_id",
        "images": ["images/path/to/screenshot.png"],
        "conversation": [
            {"from": "human", "value": "<image>\nInstruction..."},
            {"from": "gpt", "value": "Action response..."}
        ],
        "reward": 1,
        "bbox": [x1, y1, x2, y2]
    }
]

Usage:
    # From pre-processed JSON (recommended)
    python scripts/GUI_360/prepare_gui360_sft_parquet.py \
        --input datasets/GUI-360/processed_data/action_prediction_train_resize/training_data.json \
        --output train_GUI_360/data/gui360_train_sft.parquet \
        --image-base-dir datasets/GUI-360/processed_data/action_prediction_train_resize

    # With sample limit for testing
    python scripts/GUI_360/prepare_gui360_sft_parquet.py \
        --input datasets/GUI-360/processed_data/action_prediction_train_resize/training_data.json \
        --output train_GUI_360/data/gui360_train_sft.parquet \
        --image-base-dir datasets/GUI-360/processed_data/action_prediction_train_resize \
        --max-samples 1000
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd


def process_gui360_sample(
    sample: Dict[str, Any],
    image_base_dir: str,
    validate_images: bool = False
) -> Optional[Dict]:
    """
    Process a single GUI-360 sample to multi-turn message format.

    Args:
        sample: Dictionary containing id, images, conversation, etc.
        image_base_dir: Base directory for resolving image paths
        validate_images: Whether to check if images exist

    Returns:
        Dictionary with 'messages' key containing JSON string, or None if invalid
    """
    conversation = sample.get("conversation", [])
    images = sample.get("images", [])

    if not conversation:
        return None

    messages = []
    image_idx = 0

    for turn in conversation:
        role_from = turn.get("from", "")
        value = turn.get("value", "")

        if role_from == "human":
            # Process user turn
            # Remove <image> placeholder from text
            text = value.replace("<image>", "").strip()
            if text.startswith("\n"):
                text = text[1:]

            user_content = [{"type": "text", "text": text}]

            # Add image if available
            if image_idx < len(images):
                img_path = images[image_idx]
                full_img_path = os.path.join(image_base_dir, img_path)

                if validate_images and not os.path.exists(full_img_path):
                    print(f"Warning: Image not found: {full_img_path}", file=sys.stderr)
                    return None

                user_content.append({"type": "image", "image": full_img_path})
                image_idx += 1

            messages.append({"role": "user", "content": user_content})

        elif role_from == "gpt":
            # Process assistant turn
            messages.append({"role": "assistant", "content": value})

    if not messages:
        return None

    return {"messages": json.dumps(messages, ensure_ascii=False)}


def convert_gui360_to_parquet(
    input_file: str,
    output_file: str,
    image_base_dir: str,
    max_samples: int = -1,
    validate_images: bool = False,
    train_split: float = 1.0,
    eval_output: Optional[str] = None
) -> None:
    """
    Convert GUI-360 training_data.json to parquet format.

    Args:
        input_file: Path to training_data.json
        output_file: Path to output parquet file
        image_base_dir: Base directory for image paths
        max_samples: Maximum samples to process (-1 for all)
        validate_images: Whether to validate image existence
        train_split: Fraction for training (rest goes to eval)
        eval_output: Path for eval parquet (if train_split < 1.0)
    """
    print(f"Loading: {input_file}", file=sys.stderr)

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total samples in file: {len(data)}", file=sys.stderr)

    if max_samples > 0:
        data = data[:max_samples]
        print(f"Limited to: {max_samples} samples", file=sys.stderr)

    samples = []
    skipped = 0

    for i, sample in enumerate(data):
        result = process_gui360_sample(sample, image_base_dir, validate_images)
        if result:
            samples.append(result)
        else:
            skipped += 1

        if (i + 1) % 10000 == 0:
            print(f"Processed {i + 1}/{len(data)} samples...", file=sys.stderr)

    print(f"Valid samples: {len(samples)}, Skipped: {skipped}", file=sys.stderr)

    if not samples:
        print("ERROR: No valid samples!", file=sys.stderr)
        return

    # Create output directory
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Split if needed
    if train_split < 1.0 and eval_output:
        split_idx = int(len(samples) * train_split)
        train_samples = samples[:split_idx]
        eval_samples = samples[split_idx:]

        # Save training set
        df_train = pd.DataFrame(train_samples)
        df_train["messages"] = df_train["messages"].astype("object")
        df_train.to_parquet(output_file, index=False, engine="pyarrow")
        print(f"Train set: {len(train_samples)} samples -> {output_file}", file=sys.stderr)

        # Save eval set
        eval_dir = os.path.dirname(eval_output)
        if eval_dir:
            os.makedirs(eval_dir, exist_ok=True)
        df_eval = pd.DataFrame(eval_samples)
        df_eval["messages"] = df_eval["messages"].astype("object")
        df_eval.to_parquet(eval_output, index=False, engine="pyarrow")
        print(f"Eval set: {len(eval_samples)} samples -> {eval_output}", file=sys.stderr)
    else:
        # Save all to single file
        df = pd.DataFrame(samples)
        df["messages"] = df["messages"].astype("object")
        df.to_parquet(output_file, index=False, engine="pyarrow")

    # Print summary
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Conversion Complete", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"Output: {output_file}", file=sys.stderr)
    print(f"Samples: {len(samples)}", file=sys.stderr)
    print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Convert GUI-360 dataset to parquet format for SFT training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input training_data.json file"
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
        "--validate-images",
        action="store_true",
        help="Validate that image files exist"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=1.0,
        help="Fraction for training set (default: 1.0, no split)"
    )
    parser.add_argument(
        "--eval-output",
        type=str,
        help="Output path for eval set (required if train-split < 1.0)"
    )

    args = parser.parse_args()

    if args.train_split < 1.0 and not args.eval_output:
        parser.error("--eval-output required when --train-split < 1.0")

    convert_gui360_to_parquet(
        input_file=args.input,
        output_file=args.output,
        image_base_dir=args.image_base_dir,
        max_samples=args.max_samples,
        validate_images=args.validate_images,
        train_split=args.train_split,
        eval_output=args.eval_output
    )


if __name__ == "__main__":
    main()
