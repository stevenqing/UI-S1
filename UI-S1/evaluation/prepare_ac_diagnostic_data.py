#!/usr/bin/env python3
"""
Convert AndroidControl JSONL → Parquet for diagnostic scripts.

Extracts click steps with candidate_bbox from AC evaluation data and converts
them into the GUI-360 parquet format expected by probing_diagnostic.py and
binding_analysis.py.

Output format (per row):
  messages = [
    {"role": "user", "content": [{"text": "..."}, {"image": "relative/path"}]},
    {"role": "assistant", "content": "<tool_call>{...}</tool_call>"}
  ]

Usage:
  python prepare_ac_diagnostic_data.py \
      --input evaluation/dataset/android_control_evaluation_std.jsonl \
      --output evaluation/dataset/ac_diagnostic_click_steps.parquet \
      --image_base /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets
"""

import argparse
import json
import os
import sys

import pandas as pd

# ── System prompt (matches JsonFormat.gen_next_round / to_multiround) ──

SYSTEM_TEXT = """\
You are a helpful assistant. Given a screenshot of the current screen, \
user instruction and history of actions, you need to decide the next action to take."""

USER_TEMPLATE = """\
The instruction is:
{goal}

The step instruction is:
{step_instruction}

The history of actions are:
{history}

The actions supported are:
- click(coordinate=[x, y])
- type(coordinate=[x, y], text="...")
- swipe(coordinate=[x1, y1], coordinate2=[x2, y2])
- long_press(coordinate=[x, y])
- open(text="app_name")
- system_button(button="Back|Home|Overview")
- wait(time=seconds)
"""


def find_best_bbox(candidate_bboxes, gt_coord):
    """Find the candidate bbox that contains gt_coord, preferring smallest area."""
    cx, cy = gt_coord
    matches = []
    for bbox in candidate_bboxes:
        x1, y1, x2, y2 = bbox
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            area = (x2 - x1) * (y2 - y1)
            matches.append((area, bbox))
    if matches:
        matches.sort(key=lambda x: x[0])
        return matches[0][1]
    # Fallback: nearest bbox by center distance
    if candidate_bboxes:
        best = None
        best_dist = float("inf")
        for bbox in candidate_bboxes:
            x1, y1, x2, y2 = bbox
            bcx = (x1 + x2) / 2
            bcy = (y1 + y2) / 2
            dist = ((bcx - cx) ** 2 + (bcy - cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = bbox
        return best
    return None


def build_history_text(episode_steps, current_step_idx):
    """Build action history string from previous steps."""
    if current_step_idx == 0:
        return "(none)"
    parts = []
    for i in range(current_step_idx):
        s = episode_steps[i]
        ac = s["action_content"]
        action = ac["action"]
        if action == "click":
            parts.append(f"Step {i+1}: click({ac.get('coordinate', [])})")
        elif action == "type":
            parts.append(f"Step {i+1}: type(text=\"{ac.get('text', '')}\")")
        elif action == "swipe":
            parts.append(f"Step {i+1}: swipe({ac.get('coordinate', [])} → {ac.get('coordinate2', [])})")
        elif action == "long_press":
            parts.append(f"Step {i+1}: long_press({ac.get('coordinate', [])})")
        elif action == "open":
            parts.append(f"Step {i+1}: open(\"{ac.get('text', '')}\")")
        elif action == "system_button":
            parts.append(f"Step {i+1}: system_button(\"{ac.get('button', '')}\")")
        elif action == "wait":
            parts.append(f"Step {i+1}: wait({ac.get('time', 2)})")
        else:
            parts.append(f"Step {i+1}: {action}")
    return "\n".join(parts)


def convert(args):
    image_base = args.image_base
    rows = []
    stats = {"total_steps": 0, "click_steps": 0, "click_with_bbox": 0, "skipped_no_bbox": 0,
             "skipped_no_image": 0, "kept": 0}

    with open(args.input) as f:
        for line_no, line in enumerate(f):
            episode = json.loads(line.strip())
            goal = episode["goal"]
            steps = episode["steps"]

            for si, step in enumerate(steps):
                stats["total_steps"] += 1
                ac = step["action_content"]
                if ac["action"] != "click":
                    continue
                stats["click_steps"] += 1

                gt_coord = ac.get("coordinate")
                if not gt_coord or len(gt_coord) != 2:
                    continue

                # Get candidate_bbox from check_options
                check = step.get("check_options", {})
                candidate_bboxes = check.get("candidate_bbox", [])
                if not candidate_bboxes:
                    stats["skipped_no_bbox"] += 1
                    continue
                stats["click_with_bbox"] += 1

                # Find best bbox
                bbox = find_best_bbox(candidate_bboxes, gt_coord)
                if bbox is None:
                    stats["skipped_no_bbox"] += 1
                    continue

                # Image path: make relative to image_base
                screenshot = step["screenshot"]
                # Handle absolute paths
                if screenshot.startswith("/datasets/"):
                    rel_path = screenshot[len("/"):]  # -> "datasets/..."
                elif screenshot.startswith(image_base):
                    rel_path = os.path.relpath(screenshot, image_base)
                else:
                    rel_path = screenshot

                # Verify image exists
                full_path = os.path.join(image_base, rel_path)
                if not os.path.exists(full_path):
                    # Try common prefix variations
                    alt = screenshot.replace("/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/",
                                             "")
                    alt_path = os.path.join(image_base, alt)
                    if os.path.exists(alt_path):
                        rel_path = alt
                        full_path = alt_path
                    else:
                        stats["skipped_no_image"] += 1
                        continue

                # Build history
                history = build_history_text(steps, si)

                # Build user text (matches probing_diagnostic.py identify_text_regions markers)
                user_text = SYSTEM_TEXT + "\n\n"
                user_text += USER_TEMPLATE.format(
                    goal=goal,
                    step_instruction=step.get("step_instruction", ""),
                    history=history,
                )

                # Build assistant tool_call
                x1, y1, x2, y2 = bbox
                tool_call = {
                    "function": "click",
                    "args": {"coordinate": [gt_coord[0], gt_coord[1]]},
                    "bbox": {"left": x1, "top": y1, "right": x2, "bottom": y2},
                }
                assistant_content = f"<tool_call>\n{json.dumps(tool_call)}\n</tool_call>"

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"text": user_text},
                            {"image": rel_path},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": assistant_content,
                    },
                ]

                rows.append({
                    "messages": json.dumps(messages),
                    "episode_id": episode.get("episode_id", line_no),
                    "step_id": si,
                    "goal": goal,
                    "step_instruction": step.get("step_instruction", ""),
                    "action_type": "click",
                })
                stats["kept"] += 1

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_parquet(args.output, index=False)

    print(f"Conversion complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"Output: {args.output} ({len(df)} rows)")
    return df


def main():
    parser = argparse.ArgumentParser(description="Convert AC JSONL to diagnostic parquet")
    parser.add_argument("--input",
                        default="evaluation/dataset/android_control_evaluation_std.jsonl",
                        help="Input AC JSONL file")
    parser.add_argument("--output",
                        default="evaluation/dataset/ac_diagnostic_click_steps.parquet",
                        help="Output parquet file")
    parser.add_argument("--image_base",
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets",
                        help="Root directory for images")
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
