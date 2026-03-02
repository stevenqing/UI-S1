#!/usr/bin/env python3
"""
Evaluate models on GUI-360 parquet dataset.

This script evaluates a model on the GUI-360 SFT evaluation dataset,
comparing model predictions with ground truth actions.
"""

import argparse
import json
import os
import re
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from collections import Counter

import pandas as pd
from PIL import Image
from openai import OpenAI

# Global variables
result_lock = Lock()
END_POINT = "http://localhost:8000/v1"


def image_to_data_url(image_path: str) -> str:
    """Convert image file to base64 data URL."""
    image = Image.open(image_path)
    buf = BytesIO()
    image.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64_str}"


def parse_tool_call(text: str) -> dict:
    """Parse tool_call from model response."""
    try:
        # Extract JSON from <tool_call> tags
        match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        # Try parsing as raw JSON
        return json.loads(text)
    except Exception:
        return None


def convert_message_to_openai_format(messages: list, base_dir: str) -> list:
    """Convert messages to OpenAI API format with base64 images."""
    openai_messages = []

    for msg in messages:
        role = msg['role']
        content = msg['content']

        if isinstance(content, str):
            openai_messages.append({
                "role": role,
                "content": [{"type": "text", "text": content}]
            })
        elif isinstance(content, list):
            new_content = []
            for item in content:
                if isinstance(item, str):
                    new_content.append({"type": "text", "text": item})
                elif isinstance(item, dict):
                    if "text" in item:
                        new_content.append({"type": "text", "text": item["text"]})
                    elif "image" in item:
                        image_path = item["image"]
                        if not os.path.isabs(image_path):
                            image_path = os.path.join(base_dir, image_path)
                        data_url = image_to_data_url(image_path)
                        new_content.append({
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        })
            openai_messages.append({"role": role, "content": new_content})

    return openai_messages


def evaluate_action(pred_action: dict, gt_action: dict, threshold: float = 0.05, use_bbox: bool = True) -> dict:
    """
    Compare predicted action with ground truth.

    Args:
        pred_action: Predicted action dict
        gt_action: Ground truth action dict (may contain 'bbox' field)
        threshold: Relative threshold for coordinate matching
        use_bbox: If True, use bbox containment for click/type evaluation (paper method)
                  If False, use 50px Euclidean distance

    Returns dict with:
        - function_match: whether function type matches
        - args_match: whether arguments match (with tolerance for coordinates)
        - full_match: both function and args match
        - bbox_match: whether coordinate falls within bbox (only for bbox-based actions)
    """
    result = {
        "function_match": False,
        "args_match": False,
        "full_match": False,
        "bbox_match": None  # None if bbox not applicable
    }

    if pred_action is None or gt_action is None:
        return result

    pred_func = pred_action.get("function", "")
    gt_func = gt_action.get("function", "")

    # Check function match
    result["function_match"] = (pred_func == gt_func)

    if not result["function_match"]:
        return result

    # Check arguments based on function type
    pred_args = pred_action.get("args", {})
    gt_args = gt_action.get("args", {})
    gt_bbox = gt_action.get("bbox", None)

    if pred_func == "click":
        pred_coord = pred_args.get("coordinate", [])
        gt_coord = gt_args.get("coordinate", [])

        if len(pred_coord) == 2:
            pred_x, pred_y = pred_coord

            # Paper method: bbox containment (preferred)
            if use_bbox and gt_bbox:
                left = gt_bbox.get("left")
                top = gt_bbox.get("top")
                right = gt_bbox.get("right")
                bottom = gt_bbox.get("bottom")

                if all(v is not None for v in [left, top, right, bottom]):
                    result["bbox_match"] = (left <= pred_x <= right) and (top <= pred_y <= bottom)
                    result["args_match"] = result["bbox_match"]
                else:
                    result["bbox_match"] = False

            # Fallback: distance-based matching
            if not result["args_match"] and len(gt_coord) == 2:
                dist = ((pred_coord[0] - gt_coord[0])**2 + (pred_coord[1] - gt_coord[1])**2)**0.5
                result["args_match"] = dist < 50  # 50 pixel tolerance

    elif pred_func == "type":
        pred_text = pred_args.get("text", "").lower().strip()
        gt_text = gt_args.get("text", "").lower().strip()

        # Paper method: exact match for type
        if use_bbox:
            result["args_match"] = (pred_text == gt_text)
        else:
            # Fallback: inclusion match
            result["args_match"] = (pred_text == gt_text) or (pred_text in gt_text) or (gt_text in pred_text)

        # Also check bbox for type actions if available
        if use_bbox and gt_bbox and "coordinate" in pred_args:
            pred_coord = pred_args.get("coordinate", [])
            if len(pred_coord) == 2:
                pred_x, pred_y = pred_coord
                left = gt_bbox.get("left")
                top = gt_bbox.get("top")
                right = gt_bbox.get("right")
                bottom = gt_bbox.get("bottom")
                if all(v is not None for v in [left, top, right, bottom]):
                    result["bbox_match"] = (left <= pred_x <= right) and (top <= pred_y <= bottom)

    elif pred_func == "drag":
        pred_start = pred_args.get("startCoordinate", [])
        pred_end = pred_args.get("endCoordinate", [])
        gt_start = gt_args.get("startCoordinate", [])
        gt_end = gt_args.get("endCoordinate", [])

        # Paper method: bbox containment for both start and end
        if use_bbox and gt_bbox:
            # For drag, check if both points are within reasonable range
            if len(pred_start) == 2 and len(pred_end) == 2:
                start_dist = ((pred_start[0] - gt_start[0])**2 + (pred_start[1] - gt_start[1])**2)**0.5 if len(gt_start) == 2 else float('inf')
                end_dist = ((pred_end[0] - gt_end[0])**2 + (pred_end[1] - gt_end[1])**2)**0.5 if len(gt_end) == 2 else float('inf')
                result["args_match"] = start_dist < 50 and end_dist < 50
        elif len(pred_start) == 2 and len(gt_start) == 2:
            start_dist = ((pred_start[0] - gt_start[0])**2 + (pred_start[1] - gt_start[1])**2)**0.5
            end_dist = ((pred_end[0] - gt_end[0])**2 + (pred_end[1] - gt_end[1])**2)**0.5 if len(gt_end) == 2 and len(pred_end) == 2 else float('inf')
            result["args_match"] = start_dist < 50 and end_dist < 50

    elif pred_func in ["wheel_mouse_input", "scroll"]:
        # Direction match is sufficient
        pred_dir = pred_args.get("direction", "")
        gt_dir = gt_args.get("direction", "")
        result["args_match"] = pred_dir == gt_dir

    elif pred_func == "summary":
        # Summary actions always match if function matches
        result["args_match"] = True

    else:
        # For other functions, do a simple comparison
        result["args_match"] = (pred_args == gt_args)

    result["full_match"] = result["function_match"] and result["args_match"]
    return result


def call_model(messages: list, model_name: str, max_retries: int = 3) -> str:
    """Call vLLM server with the given messages."""
    client = OpenAI(
        api_key="EMPTY",
        base_url=END_POINT,
        timeout=300
    )

    for i in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                extra_body={"top_k": 1}
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call failed (attempt {i+1}/{max_retries}): {e}")
            if i < max_retries - 1:
                import time
                time.sleep(2)

    return ""


def process_sample(idx: int, row: pd.Series, args, base_dir: str) -> dict:
    """Process a single sample."""
    try:
        messages = row['messages']
        if isinstance(messages, str):
            messages = json.loads(messages)

        # Extract user message (input) and ground truth (assistant response)
        user_msg = messages[0]
        gt_response = messages[1]['content']

        # Parse ground truth action
        gt_action = parse_tool_call(gt_response)

        # Convert to OpenAI format (only user message)
        openai_messages = convert_message_to_openai_format([user_msg], base_dir)

        # Call model
        model_response = call_model(openai_messages, args.model_name)

        # Parse predicted action
        pred_action = parse_tool_call(model_response)

        # Evaluate
        use_bbox = getattr(args, 'use_bbox', True)
        eval_result = evaluate_action(pred_action, gt_action, use_bbox=use_bbox)

        result = {
            "idx": idx,
            "gt_function": gt_action.get("function", "") if gt_action else "",
            "pred_function": pred_action.get("function", "") if pred_action else "",
            "function_match": eval_result["function_match"],
            "args_match": eval_result["args_match"],
            "full_match": eval_result["full_match"],
            "bbox_match": eval_result.get("bbox_match"),
            "gt_response": gt_response[:200],
            "pred_response": model_response[:200] if model_response else ""
        }

        # Thread-safe write to output file
        with result_lock:
            result_path = os.path.join(args.output_dir, f"{args.model_name}_gui360.jsonl")
            with open(result_path, 'a') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        return result

    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "idx": idx,
            "function_match": False,
            "args_match": False,
            "full_match": False,
            "bbox_match": None,
            "error": str(e)
        }


def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Clear previous results
    result_path = os.path.join(args.output_dir, f"{args.model_name}_gui360.jsonl")
    if os.path.exists(result_path):
        os.remove(result_path)

    # Load dataset
    print(f"Loading dataset: {args.parquet_file}")
    df = pd.read_parquet(args.parquet_file)

    # Optionally limit samples
    if args.max_samples > 0:
        df = df.head(args.max_samples)

    print(f"Evaluating {len(df)} samples...")

    # Get base directory for image paths
    base_dir = os.path.dirname(os.path.dirname(args.parquet_file))
    # Adjust base_dir if needed - images are relative to project root
    if "train_GUI_360" in args.parquet_file:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(args.parquet_file)))

    print(f"Image base directory: {base_dir}")

    # Process samples in parallel
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_sample, idx, row, args, base_dir): idx
            for idx, row in df.iterrows()
        }

        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            if completed % 100 == 0:
                print(f"Progress: {completed}/{len(df)}")

    # Calculate metrics
    total = len(results)
    function_matches = sum(1 for r in results if r.get("function_match", False))
    args_matches = sum(1 for r in results if r.get("args_match", False))
    full_matches = sum(1 for r in results if r.get("full_match", False))
    bbox_matches = sum(1 for r in results if r.get("bbox_match") == True)
    bbox_applicable = sum(1 for r in results if r.get("bbox_match") is not None)

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Total samples: {total}")
    print(f"Function accuracy: {function_matches}/{total} ({function_matches/total*100:.2f}%)")
    print(f"Args accuracy (given correct function): {args_matches}/{function_matches} ({args_matches/function_matches*100:.2f}%)" if function_matches > 0 else "Args accuracy: N/A")
    print(f"Full accuracy: {full_matches}/{total} ({full_matches/total*100:.2f}%)")
    if bbox_applicable > 0:
        print(f"BBox accuracy (paper method): {bbox_matches}/{bbox_applicable} ({bbox_matches/bbox_applicable*100:.2f}%)")

    # Per-function breakdown
    print("\nPer-function breakdown:")
    gt_funcs = [r.get("gt_function", "") for r in results]
    for func, count in Counter(gt_funcs).most_common():
        func_results = [r for r in results if r.get("gt_function") == func]
        func_matches = sum(1 for r in func_results if r.get("full_match", False))
        func_bbox_matches = sum(1 for r in func_results if r.get("bbox_match") == True)
        func_bbox_applicable = sum(1 for r in func_results if r.get("bbox_match") is not None)
        print(f"  {func}: {func_matches}/{count} ({func_matches/count*100:.2f}%)")
        if func_bbox_applicable > 0:
            print(f"    BBox: {func_bbox_matches}/{func_bbox_applicable} ({func_bbox_matches/func_bbox_applicable*100:.2f}%)")

    # Save summary
    summary = {
        "model_name": args.model_name,
        "total_samples": total,
        "function_accuracy": function_matches / total if total > 0 else 0,
        "args_accuracy": args_matches / function_matches if function_matches > 0 else 0,
        "full_accuracy": full_matches / total if total > 0 else 0,
        "bbox_accuracy": bbox_matches / bbox_applicable if bbox_applicable > 0 else None,
        "bbox_applicable_samples": bbox_applicable,
    }
    summary_path = os.path.join(args.output_dir, f"{args.model_name}_gui360_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {result_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models on GUI-360 parquet dataset")

    parser.add_argument(
        "--parquet_file",
        type=str,
        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/data/gui360_test_sft_eval_format_with_bbox.parquet",
        help="Path to evaluation parquet file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/evaluation/results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name served by vLLM"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples to evaluate (0 for all)"
    )
    parser.add_argument(
        "--use_bbox",
        action="store_true",
        default=True,
        help="Use bbox containment for click/type evaluation (paper method)"
    )
    parser.add_argument(
        "--no_bbox",
        action="store_true",
        help="Disable bbox evaluation, use 50px distance threshold instead"
    )

    args = parser.parse_args()
    if args.no_bbox:
        args.use_bbox = False
    main(args)
