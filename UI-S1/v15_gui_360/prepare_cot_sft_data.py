#!/usr/bin/env python3
"""
Generate CoT (Chain-of-Thought) SFT training data.

For each action_prediction sample that has matching screen_parsing data,
construct a CoT response that includes explicit reasoning:
  1. CURRENT STATE: What the screen shows
  2. SUB-GOAL: What element to interact with (from GT screen parsing)
  3. LOCATE: Where that element is
  4. ACTION: The actual tool_call

This teaches the model to reason before acting, rather than directly
predicting coordinates.

Output format: LLaMA-Factory sharegpt format
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


def parse_gt_tool_call(gt_response: str) -> Optional[Dict]:
    """Parse the GT tool_call from response."""
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', gt_response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


def extract_tool_call_text(gt_response: str) -> str:
    """Extract the <tool_call>...</tool_call> block from response."""
    m = re.search(r'(<tool_call>.*?</tool_call>)', gt_response, re.DOTALL)
    return m.group(1) if m else gt_response


def find_gt_control(controls: List[Dict], gt_coord: List[float]) -> Optional[Dict]:
    """Find the control whose bbox contains GT coordinate."""
    if gt_coord is None or any(x is None for x in gt_coord):
        return None
    best, best_dist = None, float("inf")
    # First: controls containing GT
    for c in controls:
        rect = c.get("control_rect", [0, 0, 0, 0])
        if any(x is None for x in rect):
            continue
        if rect[0] <= gt_coord[0] <= rect[2] and rect[1] <= gt_coord[1] <= rect[3]:
            cx = (rect[0] + rect[2]) / 2
            cy = (rect[1] + rect[3]) / 2
            dist = ((cx - gt_coord[0])**2 + (cy - gt_coord[1])**2)**0.5
            if dist < best_dist:
                best, best_dist = c, dist
    # Fallback: nearest center (within 30px)
    if best is None:
        for c in controls:
            rect = c.get("control_rect", [0, 0, 0, 0])
            if any(x is None for x in rect):
                continue
            cx = (rect[0] + rect[2]) / 2
            cy = (rect[1] + rect[3]) / 2
            dist = ((cx - gt_coord[0])**2 + (cy - gt_coord[1])**2)**0.5
            if dist < best_dist and dist < 30:
                best, best_dist = c, dist
    return best


def extract_app(image_path: str) -> str:
    p = image_path.lower()
    if "excel" in p:
        return "Excel"
    if "word" in p:
        return "Word"
    if "ppt" in p:
        return "PowerPoint"
    return "desktop application"


def describe_region(cx: int, cy: int, w: int = 1040, h: int = 736) -> str:
    vpos = "top" if cy < h / 3 else ("middle" if cy < 2 * h / 3 else "bottom")
    hpos = "left" if cx < w / 3 else ("center" if cx < 2 * w / 3 else "right")
    return f"{vpos}-{hpos} area"


def build_cot_response(
    gt_response: str,
    func: str,
    gt_ctrl: Dict,
    app: str,
) -> str:
    """Build CoT reasoning + action response."""
    ctrl_text = gt_ctrl.get("control_text", "").strip()
    ctrl_type = gt_ctrl.get("control_type", "Control")
    ctrl_rect = gt_ctrl.get("control_rect", [0, 0, 0, 0])
    cx = (ctrl_rect[0] + ctrl_rect[2]) // 2
    cy = (ctrl_rect[1] + ctrl_rect[3]) // 2
    region = describe_region(cx, cy)

    # Build sub-goal description
    if func == "click":
        verb = "click on"
    elif func == "type":
        verb = "type in"
    else:
        verb = f"perform {func} on"

    if ctrl_text:
        subgoal = f'I need to {verb} "{ctrl_text}" ({ctrl_type}).'
        locate = f'"{ctrl_text}" is located in the {region} of the screen, at approximately [{cx}, {cy}].'
    else:
        subgoal = f'I need to {verb} the {ctrl_type} element.'
        locate = f'The target {ctrl_type} is in the {region} of the screen, at approximately [{cx}, {cy}].'

    tool_call_text = extract_tool_call_text(gt_response)

    return (
        f"1. CURRENT STATE: I see the {app} interface.\n"
        f"2. SUB-GOAL: {subgoal}\n"
        f"3. LOCATE: {locate}\n"
        f"4. ACTION:\n"
        f"{tool_call_text}"
    )


COT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Given a screenshot of the current screen, "
    "user instruction and history of actions, you need to decide the next action to take.\n\n"
    "Before acting, always reason step by step:\n"
    "1. CURRENT STATE: Briefly describe what you see on the screen.\n"
    "2. SUB-GOAL: Identify which specific UI element you need to interact with.\n"
    "3. LOCATE: Describe where that element is on the screen.\n"
    "4. ACTION: Output the action within <tool_call></tool_call> tags."
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ap_data", required=True, help="action_prediction training_data.json")
    parser.add_argument("--sp_data", required=True, help="screen_parsing training_data.json")
    parser.add_argument("--output", required=True, help="Output jsonl path")
    parser.add_argument("--also_output_standard", default=None,
                        help="Also output standard (non-CoT) data for comparison training")
    args = parser.parse_args()

    print("Loading action prediction data...")
    with open(args.ap_data) as f:
        ap_data = json.load(f)
    print(f"  {len(ap_data)} samples")

    print("Loading screen parsing data...")
    with open(args.sp_data) as f:
        sp_data = json.load(f)
    print(f"  {len(sp_data)} samples")

    sp_index = {d["images"][0]: d for d in sp_data}

    # Process
    cot_samples = []
    std_samples = []
    stats = {"total": 0, "matched": 0, "no_sp": 0, "no_tc": 0,
             "no_coord": 0, "no_ctrl": 0, "success": 0}

    for d in ap_data:
        stats["total"] += 1
        img = d["images"][0]

        if img not in sp_index:
            stats["no_sp"] += 1
            continue
        stats["matched"] += 1

        gt_response = d["conversation"][1]["value"]
        gt_tc = parse_gt_tool_call(gt_response)
        if gt_tc is None:
            stats["no_tc"] += 1
            continue

        func = gt_tc.get("function", "")
        gt_args = gt_tc.get("args", {})
        gt_coord = gt_args.get("coordinate")

        if gt_coord is None or any(x is None for x in gt_coord):
            stats["no_coord"] += 1
            # For non-coordinate actions (e.g. keyboard), keep original format
            if args.also_output_standard:
                std_samples.append(d)
            continue

        try:
            controls = json.loads(sp_index[img]["conversation"][1]["value"])
        except (json.JSONDecodeError, KeyError):
            stats["no_ctrl"] += 1
            continue

        gt_ctrl = find_gt_control(controls, gt_coord)
        if gt_ctrl is None:
            stats["no_ctrl"] += 1
            # No matching control, keep standard format
            if args.also_output_standard:
                std_samples.append(d)
            continue

        stats["success"] += 1

        # Build CoT version
        app = extract_app(img)
        cot_response = build_cot_response(gt_response, func, gt_ctrl, app)

        cot_sample = {
            "id": d["id"] + "_cot",
            "images": d["images"],
            "conversation": [
                d["conversation"][0],  # Same user message
                {"from": "gpt", "value": cot_response},
            ],
        }
        # Copy optional fields
        if "reward" in d:
            cot_sample["reward"] = d["reward"]
        if "bbox" in d:
            cot_sample["bbox"] = d["bbox"]

        cot_samples.append(cot_sample)

        if args.also_output_standard:
            std_samples.append(d)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(cot_samples, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(cot_samples)} CoT samples to {args.output}")

    if args.also_output_standard and std_samples:
        with open(args.also_output_standard, "w") as f:
            json.dump(std_samples, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(std_samples)} standard samples to {args.also_output_standard}")

    print(f"\nStats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
