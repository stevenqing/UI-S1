#!/usr/bin/env python3
"""
Oracle Element Experiment: How much does perfect element enumeration help?

Uses GT screen parsing data (merged_controls_info) as oracle element list,
then asks the actor model to select the correct element and generate action.

Compares:
  1. one_pass: standard direct action prediction (baseline)
  2. oracle_elements: actor gets GT element list, selects and acts

Uses training data (action_prediction + screen_parsing share same images).
Samples 200 click steps for evaluation.
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v13_gui_360.reward import compute_step_reward


# ═══════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════

# One-pass baseline (standard prompt from training data)
ONE_PASS_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen, e.g., click(coordinate=[100, 200], button='left', double=False, pressed=None)

First, explain your reasoning process—describe how you analyze the screenshot, understand the current state, and determine what action should be taken next based on the instruction and previous actions.

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

Only **ONE** action should be taken at a time."""

# Oracle elements prompt
ORACLE_ELEMENTS_PROMPT = """You are a GUI automation agent. Given the instruction, the list of interactive UI elements on the current screen, and action history, select the correct element and generate the action.

The instruction is:
{instruction}

Interactive UI elements on the current screen:
{elements}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: Use the coordinates from the UI elements listed above. Select the element that best matches what you need to interact with.

First, briefly explain which element you're selecting and why.
Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

Only **ONE** action should be taken at a time."""

SUPPORTED_ACTIONS = """<action>
- click
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to click at.
    - button: str, One of 'left', 'right', 'middle' or 'x' (Default: 'left')
    - double: bool, Whether to perform a double click (Default: False)
    - pressed: str|None, Keyboard key to press while clicking (Default: None)
  - Example: click(coordinate=[100, 100], button='left', double=False, pressed=None)
- type
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to type at.
    - keys: str, The key to input.
    - clear_current_text: bool, Whether to clear the current text (Default: False)
    - control_focus: bool, Whether to focus on selected control before typing (Default: True)
  - Example: type(coordinate=[100, 100], keys='Hello')
- swipe
  - Args:
    - coordinate: [x, y], the absolute position to start
    - direction: str, The direction of the swipe (up/down/left/right)
    - distance: int, The distance of the swipe
  - Example: swipe(coordinate=[100, 100], direction='up', distance=200)
</action>"""


# ═══════════════════════════════════════════════════════════════════
# Model loading and inference
# ═══════════════════════════════════════════════════════════════════

def load_model(model_path, image_max_pixels=602112):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_path,
        max_pixels=image_max_pixels,
        min_pixels=256 * 28 * 28,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    return model, processor


def generate_response(model, processor, messages, image, max_new_tokens=512):
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", padding=False
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                top_p=0.95,
                do_sample=True,
            )

    prompt_len = inputs["input_ids"].shape[1]
    resp_ids = output_ids[0, prompt_len:]
    return processor.tokenizer.decode(resp_ids, skip_special_tokens=True)


# ═══════════════════════════════════════════════════════════════════
# Action parsing & reward
# ═══════════════════════════════════════════════════════════════════

def parse_action(text: str) -> Optional[Dict]:
    """Parse action from model output."""
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1))
            func = data.get("function", "")
            args = data.get("args", {})
            if func:
                return {"type": func, **args}
        except json.JSONDecodeError:
            pass

    # Fallback: function call pattern
    m = re.search(r'(click|type|swipe)\s*\(', text)
    if m:
        func = m.group(1)
        coord_m = re.search(r'coordinate\s*=\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', text[m.start():])
        if coord_m:
            result = {"type": func, "coordinate": [int(coord_m.group(1)), int(coord_m.group(2))]}
            if func == "type":
                keys_m = re.search(r"keys\s*=\s*['\"](.+?)['\"]", text[m.start():])
                if keys_m:
                    result["keys"] = keys_m.group(1)
            return result

    return None


def format_action_as_tool_call(action: Dict) -> str:
    """Format parsed action back to <tool_call> text for reward computation."""
    if action is None:
        return ""
    tc = {"function": action["type"], "args": {}, "status": "CONTINUE"}
    for k in ["coordinate", "keys", "direction", "distance", "button", "double", "pressed",
              "clear_current_text", "control_focus", "text"]:
        if k in action:
            tc["args"][k] = action[k]
    return f"<tool_call>\n{json.dumps(tc)}\n</tool_call>"


def format_elements(controls: List[Dict]) -> str:
    """Format control_infos into a numbered element list."""
    lines = []
    for i, c in enumerate(controls, 1):
        text = c.get("control_text", "")
        rect = c.get("control_rect", [0, 0, 0, 0])
        ctype = c.get("control_type", "")
        # Compute center coordinate
        cx = (rect[0] + rect[2]) // 2
        cy = (rect[1] + rect[3]) // 2
        if text:
            lines.append(f'[{i}] {ctype}: "{text}" at [{cx}, {cy}] (rect: {rect})')
        else:
            lines.append(f'[{i}] {ctype}: (no text) at [{cx}, {cy}] (rect: {rect})')
    return "\n".join(lines)


def extract_gt_from_conversation(conv: List[Dict]) -> Tuple[str, str, str]:
    """Extract instruction, history, and GT action text from conversation."""
    human_msg = conv[0]["value"]
    gt_response = conv[1]["value"]

    # Extract instruction
    m = re.search(r'The instruction is:\s*\n(.+?)(?:\n\nThe history|$)', human_msg, re.DOTALL)
    instruction = m.group(1).strip() if m else ""

    # Extract history
    m = re.search(r'The history of actions are:\s*\n(.+?)(?:\n\nThe actions supported|$)', human_msg, re.DOTALL)
    history = m.group(1).strip() if m else "None"

    return instruction, history, gt_response


def parse_gt_action(gt_response: str) -> Optional[Dict]:
    """Parse GT action from training data response."""
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', gt_response, re.DOTALL)
    if m:
        try:
            tc = json.loads(m.group(1))
            func = tc.get("function", "")
            args = tc.get("args", {})
            result = {"action": func}
            if "coordinate" in args:
                result["coordinate"] = args["coordinate"]
            if "text" in args:
                result["text"] = args["text"]
            if "keys" in args:
                result["text"] = args["keys"]
            return result
        except json.JSONDecodeError:
            pass
    return None


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--ap_data", required=True, help="action_prediction training_data.json")
    parser.add_argument("--sp_data", required=True, help="screen_parsing training_data.json")
    parser.add_argument("--image_dir", required=True, help="Directory containing images/ folder")
    parser.add_argument("--output_dir", default="v15_gui_360/outputs/preliminary_tests/oracle_elements")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--image_max_pixels", type=int, default=602112)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading action prediction data...")
    with open(args.ap_data) as f:
        ap_data = json.load(f)
    print(f"  {len(ap_data)} samples")

    print("Loading screen parsing data...")
    with open(args.sp_data) as f:
        sp_data = json.load(f)
    print(f"  {len(sp_data)} samples")

    # Build SP index by image path
    sp_index = {}
    for d in sp_data:
        sp_index[d["images"][0]] = d

    # Filter: click actions with matching screen parsing
    candidates = []
    for d in ap_data:
        img = d["images"][0]
        if img not in sp_index:
            continue
        gt_response = d["conversation"][1]["value"]
        if '"click"' not in gt_response:
            continue
        gt_action = parse_gt_action(gt_response)
        if gt_action is None or gt_action.get("action") != "click":
            continue
        # Parse SP controls
        sp_resp = sp_data_response = sp_index[img]["conversation"][1]["value"]
        try:
            controls = json.loads(sp_resp)
        except json.JSONDecodeError:
            continue
        if not controls or len(controls) < 3:
            continue

        instruction, history, _ = extract_gt_from_conversation(d["conversation"])
        if not instruction:
            continue

        candidates.append({
            "id": d["id"],
            "image_path": os.path.join(args.image_dir, img),
            "instruction": instruction,
            "history": history,
            "gt_action": gt_action,
            "gt_response": gt_response,
            "controls": controls,
            "bbox": d.get("bbox"),
        })

    print(f"Click steps with screen parsing: {len(candidates)}")

    # Subsample
    rng = np.random.default_rng(42)
    indices = rng.choice(len(candidates), size=min(args.max_steps, len(candidates)), replace=False)
    eval_steps = [candidates[i] for i in sorted(indices)]
    print(f"Evaluating {len(eval_steps)} steps")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model, processor = load_model(args.model_path, args.image_max_pixels)
    print("Model loaded.", flush=True)

    # Evaluate
    results = {"one_pass": [], "oracle_elements": []}

    for i, item in enumerate(eval_steps):
        try:
            image = Image.open(item["image_path"]).convert("RGB")
        except Exception as e:
            print(f"  [{i}] Error loading image: {e}", flush=True)
            for mode in results:
                results[mode].append({"error": str(e), "reward": 0, "is_correct": False, "id": item["id"]})
            continue

        image_w, image_h = image.size
        gt_action = item["gt_action"]

        for mode in ["one_pass", "oracle_elements"]:
            try:
                if mode == "one_pass":
                    prompt = ONE_PASS_PROMPT.format(
                        instruction=item["instruction"],
                        history=item["history"],
                        actions=SUPPORTED_ACTIONS,
                    )
                else:
                    elements_text = format_elements(item["controls"])
                    prompt = ORACLE_ELEMENTS_PROMPT.format(
                        instruction=item["instruction"],
                        elements=elements_text,
                        history=item["history"],
                        actions=SUPPORTED_ACTIONS,
                    )

                messages = [{"role": "user", "content": [
                    {"type": "image", "image": item["image_path"]},
                    {"type": "text", "text": prompt},
                ]}]

                t0 = time.time()
                response = generate_response(model, processor, messages, image)
                gen_time = time.time() - t0

                action = parse_action(response)
                pred_text = format_action_as_tool_call(action) if action else ""

                reward, info = compute_step_reward(
                    pred_text, gt_action, image_w, image_h,
                    w_format=0.1, w_type=0.2, w_content=0.7,
                )
                is_correct = reward >= 0.5

                results[mode].append({
                    "id": item["id"],
                    "mode": mode,
                    "action": action,
                    "reward": reward,
                    "is_correct": is_correct,
                    "gen_time": gen_time,
                    "response": response[:300],
                    "n_controls": len(item["controls"]) if mode == "oracle_elements" else None,
                    "content_reward": info.get("content_reward", 0),
                })
            except Exception as e:
                results[mode].append({
                    "id": item["id"], "mode": mode, "error": str(e),
                    "reward": 0, "is_correct": False,
                })

        # Progress
        if (i + 1) % 10 == 0:
            op_acc = np.mean([r["is_correct"] for r in results["one_pass"]])
            oe_acc = np.mean([r["is_correct"] for r in results["oracle_elements"]])
            op_r = np.mean([r["reward"] for r in results["one_pass"]])
            oe_r = np.mean([r["reward"] for r in results["oracle_elements"]])
            print(f"  [{i+1}/{len(eval_steps)}] "
                  f"one_pass: acc={op_acc:.1%} r={op_r:.3f} | "
                  f"oracle_elements: acc={oe_acc:.1%} r={oe_r:.3f}", flush=True)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    for mode in ["one_pass", "oracle_elements"]:
        rs = results[mode]
        acc = np.mean([r["is_correct"] for r in rs])
        mean_r = np.mean([r["reward"] for r in rs])
        mean_cr = np.mean([r.get("content_reward", 0) for r in rs])
        mean_time = np.mean([r.get("gen_time", 0) for r in rs])
        print(f"\n  {mode}:")
        print(f"    Accuracy (r>=0.5): {sum(r['is_correct'] for r in rs)}/{len(rs)} = {acc:.1%}")
        print(f"    Mean reward: {mean_r:.3f}")
        print(f"    Mean content_reward: {mean_cr:.3f}")
        print(f"    Mean gen time: {mean_time:.1f}s")

    # Per-step comparison
    print(f"\n  Per-step comparison:")
    both_correct = sum(a["is_correct"] and b["is_correct"]
                       for a, b in zip(results["one_pass"], results["oracle_elements"]))
    op_only = sum(a["is_correct"] and not b["is_correct"]
                  for a, b in zip(results["one_pass"], results["oracle_elements"]))
    oe_only = sum(not a["is_correct"] and b["is_correct"]
                  for a, b in zip(results["one_pass"], results["oracle_elements"]))
    both_wrong = sum(not a["is_correct"] and not b["is_correct"]
                     for a, b in zip(results["one_pass"], results["oracle_elements"]))
    n = len(results["one_pass"])
    print(f"    Both correct:        {both_correct:3d} ({both_correct/n:.1%})")
    print(f"    One-pass only:       {op_only:3d} ({op_only/n:.1%})")
    print(f"    Oracle-elem only:    {oe_only:3d} ({oe_only/n:.1%})")
    print(f"    Both wrong:          {both_wrong:3d} ({both_wrong/n:.1%})")

    # Analyze: does giving elements help find the GT control?
    print(f"\n  Oracle element analysis:")
    gt_in_list = 0
    closest_dists = []
    for i, item in enumerate(eval_steps):
        gt_coord = item["gt_action"].get("coordinate", [0, 0])
        controls = item["controls"]
        min_dist = float("inf")
        for c in controls:
            rect = c.get("control_rect", [0, 0, 0, 0])
            cx = (rect[0] + rect[2]) / 2
            cy = (rect[1] + rect[3]) / 2
            dist = ((cx - gt_coord[0])**2 + (cy - gt_coord[1])**2)**0.5
            min_dist = min(min_dist, dist)
            # Check if GT is inside bounding box
            if rect[0] <= gt_coord[0] <= rect[2] and rect[1] <= gt_coord[1] <= rect[3]:
                gt_in_list += 1
                break
        closest_dists.append(min_dist)

    print(f"    GT coord inside a control bbox: {gt_in_list}/{len(eval_steps)} ({gt_in_list/len(eval_steps):.1%})")
    print(f"    Mean closest control dist: {np.mean(closest_dists):.1f}px")
    print(f"    Median closest control dist: {np.median(closest_dists):.1f}px")

    # Save
    output_file = os.path.join(args.output_dir, f"oracle_elements_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
