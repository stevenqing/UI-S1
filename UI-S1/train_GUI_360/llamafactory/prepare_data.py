#!/usr/bin/env python3
"""
Convert GUI-360 training_data.json to LlamaFactory-compatible ShareGPT format.

The source data is already in ShareGPT format (from/value with human/gpt tags).
This script resolves image paths to absolute paths so LlamaFactory can load them
regardless of the working directory.

Usage:
    python train_GUI_360/llamafactory/prepare_data.py

    # Custom paths
    python train_GUI_360/llamafactory/prepare_data.py \
        --input datasets/GUI-360/processed_data/action_prediction_train_resize/training_data.json \
        --output train_GUI_360/llamafactory/data/gui360_train.json \
        --image-base-dir datasets/GUI-360/processed_data/action_prediction_train_resize \
        --val-size 100
"""

import argparse
import json
import os
import random
import sys


def convert_to_llamafactory_format(
    input_file: str,
    output_file: str,
    image_base_dir: str,
    val_output: str = None,
    val_size: int = 0,
    max_samples: int = -1,
    validate_images: bool = False,
    seed: int = 42,
):
    """Convert GUI-360 training_data.json to LlamaFactory ShareGPT format."""
    print(f"Loading: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total samples: {len(data)}")

    image_base_dir = os.path.abspath(image_base_dir)

    if max_samples > 0:
        data = data[:max_samples]
        print(f"Limited to {max_samples} samples")

    converted = []
    skipped = 0
    missing_images = 0

    for i, sample in enumerate(data):
        conversation = sample.get("conversation", [])
        images = sample.get("images", [])

        if not conversation:
            skipped += 1
            continue

        # Resolve image paths to absolute
        abs_images = []
        has_missing = False
        for img_path in images:
            abs_path = os.path.join(image_base_dir, img_path)
            if validate_images and not os.path.exists(abs_path):
                has_missing = True
                missing_images += 1
                break
            abs_images.append(abs_path)

        if has_missing:
            skipped += 1
            continue

        # Build LlamaFactory ShareGPT format entry
        entry = {
            "conversations": conversation,  # already has from/value with human/gpt
            "images": abs_images,
        }
        converted.append(entry)

        if (i + 1) % 20000 == 0:
            print(f"  Processed {i + 1}/{len(data)} samples...")

    print(f"Converted: {len(converted)}, Skipped: {skipped}")
    if missing_images > 0:
        print(f"Missing images: {missing_images}")

    # Split train/val if requested
    if val_size > 0 and val_output:
        random.seed(seed)
        indices = list(range(len(converted)))
        random.shuffle(indices)

        val_indices = set(indices[:val_size])
        train_data = [converted[i] for i in range(len(converted)) if i not in val_indices]
        val_data = [converted[i] for i in val_indices]

        # Save train
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False)
        print(f"Train set: {len(train_data)} samples -> {output_file}")
        print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")

        # Save val
        os.makedirs(os.path.dirname(val_output), exist_ok=True)
        with open(val_output, "w", encoding="utf-8") as f:
            json.dump(val_data, f, ensure_ascii=False)
        print(f"Val set: {len(val_data)} samples -> {val_output}")
        print(f"  File size: {os.path.getsize(val_output) / 1024 / 1024:.1f} MB")
    else:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(converted, f, ensure_ascii=False)
        print(f"Output: {len(converted)} samples -> {output_file}")
        print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert GUI-360 data to LlamaFactory ShareGPT format"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="datasets/GUI-360/processed_data/action_prediction_train_resize/training_data.json",
        help="Input training_data.json file",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="train_GUI_360/llamafactory/data/gui360_train.json",
        help="Output JSON file for LlamaFactory",
    )
    parser.add_argument(
        "--image-base-dir",
        type=str,
        default="datasets/GUI-360/processed_data/action_prediction_train_resize",
        help="Base directory for resolving image paths",
    )
    parser.add_argument(
        "--val-output",
        type=str,
        default="train_GUI_360/llamafactory/data/gui360_val.json",
        help="Output JSON file for validation set",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=100,
        help="Number of validation samples (0 to skip)",
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=-1,
        help="Maximum samples to process (-1 for all)",
    )
    parser.add_argument(
        "--validate-images",
        action="store_true",
        help="Validate that image files exist",
    )

    args = parser.parse_args()

    convert_to_llamafactory_format(
        input_file=args.input,
        output_file=args.output,
        image_base_dir=args.image_base_dir,
        val_output=args.val_output,
        val_size=args.val_size,
        max_samples=args.max_samples,
        validate_images=args.validate_images,
    )


if __name__ == "__main__":
    main()
