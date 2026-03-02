#!/usr/bin/env python3
"""
Evaluate models on GUI-360 A11y format dataset.

This script evaluates models trained with A11y format (element_id based)
on the GUI-360 evaluation dataset.
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
        # Extract JSON from <execute> tags (A11y format)
        match = re.search(r'<execute>\s*(.*?)\s*</execute>', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        # Also try <tool_call > tags (legacy format with various closings)
        match = re.search(r'<tool_call >\s*(.*?)\s*(?:<\|tool_call \|>|</tool_call >)', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        # Try finding JSON directly in the text
        match = re.search(r'\{[^{}]*"function"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^{}]*\}[^{}]*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
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


def evaluate_action_a11y(pred_action: dict, gt_action: dict) -> dict:
    """
    Compare predicted action with ground truth for A11y format.

    For A11y format, we compare:
    - function_match: function type matches
    - element_id_match: element_id matches (for click/type)
    - full_match: both function and element_id match
    - bbox_match: if pred has coordinate, check bbox containment
    """
    result = {
        "function_match": False,
        "element_id_match": None,
        "full_match": False,
        "bbox_match": None,
        "text_match": None
    }

    if pred_action is None or gt_action is None:
        return result

    pred_func = pred_action.get("function", "")
    gt_func = gt_action.get("function", "")

    # Check function match
    result["function_match"] = (pred_func == gt_func)

    if not result["function_match"]:
        return result

    pred_args = pred_action.get("args", {})
    gt_args = gt_action.get("args", {})
    gt_bbox = gt_action.get("bbox", None)

    if pred_func in ["click", "type"]:
        # Check element_id match
        pred_element_id = pred_args.get("element_id")
        gt_element_id = gt_args.get("element_id")

        if pred_element_id is not None and gt_element_id is not None:
            result["element_id_match"] = (pred_element_id == gt_element_id)
            result["full_match"] = result["element_id_match"]

        # Also check bbox if prediction has coordinate
        if gt_bbox and "coordinate" in pred_args:
            pred_coord = pred_args.get("coordinate", [])
            if len(pred_coord) == 2:
                pred_x, pred_y = pred_coord
                if isinstance(gt_bbox, list) and len(gt_bbox) >= 4:
                    left, top, right, bottom = gt_bbox
                    result["bbox_match"] = (left <= pred_x <= right) and (top <= pred_y <= bottom)
                elif isinstance(gt_bbox, dict):
                    left = gt_bbox.get("left")
                    top = gt_bbox.get("top")
                    right = gt_bbox.get("right")
                    bottom = gt_bbox.get("bottom")
                    if all(v is not None for v in [left, top, right, bottom]):
                        result["bbox_match"] = (left <= pred_x <= right) and (top <= pred_y <= bottom)

        # For type, also check text match
        if pred_func == "type":
            pred_text = pred_args.get("text", "").lower().strip()
            gt_text = gt_args.get("text", "").lower().strip()
            result["text_match"] = (pred_text == gt_text) or (pred_text in gt_text) or (gt_text in pred_text)
            if result["element_id_match"] and result["text_match"]:
                result["full_match"] = True

    elif pred_func == "drag":
        pred_start_id = pred_args.get("start_element_id")
        pred_end_id = pred_args.get("end_element_id")
        gt_start_id = gt_args.get("start_element_id")
        gt_end_id = gt_args.get("end_element_id")

        if all(v is not None for v in [pred_start_id, pred_end_id, gt_start_id, gt_end_id]):
            result["element_id_match"] = (pred_start_id == gt_start_id) and (pred_end_id == gt_end_id)
            result["full_match"] = result["element_id_match"]

    elif pred_func in ["wheel_mouse_input", "scroll"]:
        pred_id = pred_args.get("element_id")
        gt_id = gt_args.get("element_id")
        if pred_id is not None and gt_id is not None:
            result["element_id_match"] = (pred_id == gt_id)
            result["full_match"] = result["element_id_match"]

    elif pred_func == "summary":
        result["full_match"] = True
        result["element_id_match"] = True

    return result


def call_model(messages: list, model_name: str) -> str:
    """Call the vLLM model."""
    try:
        client = OpenAI(
            api_key="EMPTY",
            base_url=END_POINT,
            timeout=120
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            extra_body={"top_k": 1}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling model: {e}")
        return ""


def process_sample(row: dict, model_name: str, base_dir: str) -> dict:
    """Process a single sample."""
    try:
        messages = json.loads(row["messages"])
        gt_action = json.loads(row["ground_truth"])

        # Convert to OpenAI format
        openai_messages = convert_message_to_openai_format(messages, base_dir)

        # Call model
        response = call_model(openai_messages, model_name)

        # Parse prediction
        pred_action = parse_tool_call(response)

        # Evaluate
        eval_result = evaluate_action_a11y(pred_action, gt_action)

        return {
            "trajectory_id": row.get("trajectory_id", ""),
            "step_id": row.get("step_id", 0),
            "instruction": row.get("instruction", ""),
            "pred_action": pred_action,
            "gt_action": gt_action,
            "response": response[:500],
            **eval_result
        }
    except Exception as e:
        return {
            "trajectory_id": row.get("trajectory_id", ""),
            "step_id": row.get("step_id", 0),
            "instruction": row.get("instruction", ""),
            "pred_action": None,
            "gt_action": None,
            "response": str(e)[:500],
            "function_match": False,
            "element_id_match": None,
            "full_match": False,
            "bbox_match": None,
            "text_match": None
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on GUI-360 A11y format")
    parser.add_argument("--parquet_file", type=str, required=True, help="Path to evaluation parquet file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--model_name", type=str, required=True, help="Model name in vLLM")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--base_dir", type=str, default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1",
                        help="Base directory for relative paths")

    args = parser.parse_args()

    # Load data
    df = pd.read_parquet(args.parquet_file)
    print(f"Loaded {len(df)} samples")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process samples in parallel
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_sample, row.to_dict(), args.model_name, args.base_dir): idx
            for idx, row in df.iterrows()
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing sample: {e}")

    # Calculate metrics
    total = len(results)
    function_matches = sum(1 for r in results if r.get("function_match", False))
    full_matches = sum(1 for r in results if r.get("full_match", False))

    # Element ID matches (only where applicable)
    element_id_applicable = [r for r in results if r.get("element_id_match") is not None]
    element_id_matches = sum(1 for r in element_id_applicable if r.get("element_id_match", False))

    # BBox matches (only where applicable)
    bbox_applicable = [r for r in results if r.get("bbox_match") is not None]
    bbox_matches = sum(1 for r in bbox_applicable if r.get("bbox_match", False))

    # Text matches (only where applicable)
    text_applicable = [r for r in results if r.get("text_match") is not None]
    text_matches = sum(1 for r in text_applicable if r.get("text_match", False))

    metrics = {
        "model_name": args.model_name,
        "total_samples": total,
        "function_accuracy": function_matches / total if total > 0 else 0,
        "element_id_accuracy": element_id_matches / len(element_id_applicable) if element_id_applicable else 0,
        "full_accuracy": full_matches / total if total > 0 else 0,
        "bbox_accuracy": bbox_matches / len(bbox_applicable) if bbox_applicable else 0,
        "text_accuracy": text_matches / len(text_applicable) if text_applicable else 0,
        "element_id_applicable_samples": len(element_id_applicable),
        "bbox_applicable_samples": len(bbox_applicable),
        "text_applicable_samples": len(text_applicable)
    }

    # Save results
    output_file = os.path.join(args.output_dir, f"{args.model_name}_a11y_eval.jsonl")
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    summary_file = os.path.join(args.output_dir, f"{args.model_name}_a11y_eval_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Complete")
    print(f"{'='*60}")
    print(f"Total samples: {total}")
    print(f"Function accuracy: {metrics['function_accuracy']*100:.2f}%")
    print(f"Element ID accuracy: {metrics['element_id_accuracy']*100:.2f}% ({element_id_matches}/{len(element_id_applicable)})")
    print(f"Full accuracy: {metrics['full_accuracy']*100:.2f}%")
    print(f"BBox accuracy: {metrics['bbox_accuracy']*100:.2f}% ({bbox_matches}/{len(bbox_applicable)})")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
