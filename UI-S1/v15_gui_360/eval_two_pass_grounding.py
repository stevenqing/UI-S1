#!/usr/bin/env python3
"""
Two-Pass Grounding Experiment: Element Enumeration + Action Selection

Tests whether decomposing the task into:
  Pass 1: "What UI elements are on screen?" → element list
  Pass 2: "Given these elements + instruction, which one and what action?" → action

improves accuracy compared to one-pass direct prediction.

Also tests oracle element provision (GT element in the list) as an upper bound.
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
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v13_gui_360.reward import compute_step_reward

# ═══════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════

# Pass 1: Element Detection
ELEMENT_DETECTION_PROMPT = """Look at this screenshot and list ALL interactive UI elements you can see.
For each element, provide a brief description and its approximate pixel coordinates [x, y].

Format each element as a numbered list:
[1] "Description of element" at [x, y]
[2] "Description of element" at [x, y]
...

Include: buttons, menu items, tabs, text fields, icons, checkboxes, links, toolbars, ribbons, and any other clickable or interactive elements. Be thorough — list at least 15 elements if visible."""

# Pass 2: Action Selection (with detected elements)
ACTION_SELECTION_PROMPT = """You are a GUI automation agent. Given the instruction, detected UI elements, and action history, select the correct element and generate the action.

The instruction is:
{instruction}

Detected UI elements on the current screen:
{elements}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: Use the coordinates from the detected elements above. Select the element that best matches the instruction.

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

# One-pass baseline (standard prompt)
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
                temperature=0.1,  # near-greedy for eval
                top_p=0.95,
                do_sample=True,
            )

    prompt_len = inputs["input_ids"].shape[1]
    resp_ids = output_ids[0, prompt_len:]
    return processor.tokenizer.decode(resp_ids, skip_special_tokens=True)


# ═══════════════════════════════════════════════════════════════════
# Action parsing
# ═══════════════════════════════════════════════════════════════════

def parse_action(text: str) -> Optional[Dict]:
    """Parse action from model output."""
    # Try tool_call format
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

    # Fallback: try function call pattern
    m = re.search(r'(click|type|swipe)\s*\(', text)
    if m:
        func = m.group(1)
        # Extract coordinate
        coord_m = re.search(r'coordinate\s*=\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', text[m.start():])
        if coord_m:
            result = {"type": func, "coordinate": [int(coord_m.group(1)), int(coord_m.group(2))]}
            if func == "type":
                keys_m = re.search(r"keys\s*=\s*['\"](.+?)['\"]", text[m.start():])
                if keys_m:
                    result["keys"] = keys_m.group(1)
            return result

    return None


def parse_elements(text: str) -> List[Dict]:
    """Parse element list from Pass 1 output."""
    elements = []
    # Match [N] "description" at [x, y]
    pattern = r'\[(\d+)\]\s*["\']?(.+?)["\']?\s+at\s+\[\s*(\d+)\s*,\s*(\d+)\s*\]'
    for m in re.finditer(pattern, text):
        elements.append({
            "id": int(m.group(1)),
            "desc": m.group(2).strip().strip('"\''),
            "x": int(m.group(3)),
            "y": int(m.group(4)),
        })

    # Fallback: try other patterns
    if not elements:
        pattern2 = r'(\d+)[.)]\s*(.+?)\s*[\-–:]\s*\[?\s*(\d+)\s*,\s*(\d+)\s*\]?'
        for m in re.finditer(pattern2, text):
            elements.append({
                "id": int(m.group(1)),
                "desc": m.group(2).strip(),
                "x": int(m.group(3)),
                "y": int(m.group(4)),
            })

    return elements


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate_step(model, processor, step, goal, history_text, mode="one_pass"):
    """Evaluate a single step with different modes."""
    screenshot = step["screenshot"]
    gt_action = step["action"]
    image_w = step.get("image_w", 1040)
    image_h = step.get("image_h", 736)

    try:
        image = Image.open(screenshot).convert("RGB")
    except Exception as e:
        return {"error": str(e), "reward": 0, "is_correct": False}

    if mode == "one_pass":
        # Standard single-pass prediction
        prompt = ONE_PASS_PROMPT.format(
            instruction=goal,
            history=history_text,
            actions=SUPPORTED_ACTIONS,
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": screenshot},
            {"type": "text", "text": prompt},
        ]}]

        t0 = time.time()
        response = generate_response(model, processor, messages, image)
        gen_time = time.time() - t0
        action = parse_action(response)

        return {
            "mode": "one_pass",
            "response": response[:200],
            "action": action,
            "gen_time": gen_time,
            "elements": None,
        }

    elif mode == "two_pass":
        # Pass 1: Detect elements
        messages_p1 = [{"role": "user", "content": [
            {"type": "image", "image": screenshot},
            {"type": "text", "text": ELEMENT_DETECTION_PROMPT},
        ]}]

        t0 = time.time()
        p1_response = generate_response(model, processor, messages_p1, image, max_new_tokens=1024)
        p1_time = time.time() - t0

        elements = parse_elements(p1_response)

        if not elements:
            elements_text = p1_response  # Use raw text if parsing fails
        else:
            elements_text = "\n".join(
                f'[{e["id"]}] "{e["desc"]}" at [{e["x"]}, {e["y"]}]'
                for e in elements
            )

        # Pass 2: Select element and generate action
        prompt_p2 = ACTION_SELECTION_PROMPT.format(
            instruction=goal,
            elements=elements_text,
            history=history_text,
            actions=SUPPORTED_ACTIONS,
        )
        messages_p2 = [{"role": "user", "content": [
            {"type": "image", "image": screenshot},
            {"type": "text", "text": prompt_p2},
        ]}]

        t1 = time.time()
        p2_response = generate_response(model, processor, messages_p2, image)
        p2_time = time.time() - t1

        action = parse_action(p2_response)

        return {
            "mode": "two_pass",
            "p1_response": p1_response[:300],
            "p2_response": p2_response[:200],
            "n_elements": len(elements),
            "action": action,
            "gen_time": p1_time + p2_time,
            "p1_time": p1_time,
            "p2_time": p2_time,
            "elements": elements,
        }


def compute_reward(result, gt_action, image_w, image_h):
    """Compute reward for a prediction."""
    action = result.get("action")
    if action is None:
        return 0.0, False

    # Format as <tool_call> JSON so compute_step_reward can parse it
    tc = {"function": action["type"], "args": {}, "status": "CONTINUE"}
    if "coordinate" in action:
        tc["args"]["coordinate"] = action["coordinate"]
    if "keys" in action:
        tc["args"]["keys"] = action["keys"]
    if "direction" in action:
        tc["args"]["direction"] = action["direction"]
    if "distance" in action:
        tc["args"]["distance"] = action["distance"]
    text = f"<tool_call>\n{json.dumps(tc)}\n</tool_call>"

    reward, _ = compute_step_reward(
        text, gt_action, image_w, image_h,
        w_format=0.1, w_type=0.2, w_content=0.7,
    )
    is_correct = reward >= 0.5
    return reward, is_correct


def _format_action(act):
    """Format action dict/str for history display."""
    if isinstance(act, dict):
        atype = act.get("action", "unknown")
        coord = act.get("coordinate", [])
        if atype == "click" and coord:
            return f"click(coordinate=[{coord[0]}, {coord[1]}])"
        elif atype == "type" and coord:
            keys = act.get("keys", "")
            return f"type(coordinate=[{coord[0]}, {coord[1]}], keys='{keys}')"
        elif atype == "swipe" and coord:
            d = act.get("direction", "up")
            dist = act.get("distance", 200)
            return f"swipe(coordinate=[{coord[0]}, {coord[1]}], direction='{d}', distance={dist})"
        return str(act)
    return str(act)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output_dir", default="v15_gui_360/outputs/preliminary_tests/two_pass_grounding")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--image_max_pixels", type=int, default=602112)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading test data...")
    episodes = []
    with open(args.test_data) as f:
        for line in f:
            episodes.append(json.loads(line))
    print(f"Loaded {len(episodes)} episodes")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model, processor = load_model(args.model_path, args.image_max_pixels)
    print("Model loaded.")

    # Collect steps (only click steps for this experiment, since that's the bottleneck)
    all_steps = []
    for ep in episodes:
        for s_idx, step in enumerate(ep["steps"]):
            act = step["action"]
            if isinstance(act, dict):
                gt_type = act.get("action", "")
            else:
                gt_type = act.split("(")[0].strip() if "(" in act else ""
            all_steps.append({
                "episode_id": ep["episode_id"],
                "goal": ep["goal"],
                "step_idx": s_idx,
                "step": step,
                "gt_type": gt_type,
                "history": [
                    f"Step {i+1}: {_format_action(ep['steps'][i]['action'])}"
                    for i in range(s_idx)
                ],
            })

    # Filter to click steps (primary bottleneck)
    click_steps = [s for s in all_steps if s["gt_type"] == "click"]
    print(f"Total click steps: {len(click_steps)}, evaluating {min(args.max_steps, len(click_steps))}")

    # Subsample
    rng = np.random.default_rng(42)
    indices = rng.choice(len(click_steps), size=min(args.max_steps, len(click_steps)), replace=False)
    eval_steps = [click_steps[i] for i in sorted(indices)]

    # Evaluate both modes
    results = {"one_pass": [], "two_pass": []}

    for i, item in enumerate(eval_steps):
        step = item["step"]
        goal = item["goal"]
        history_text = "\n".join(item["history"]) if item["history"] else "None"
        gt_action = step["action"]
        image_w = step.get("image_w", 1040)
        image_h = step.get("image_h", 736)

        for mode in ["one_pass", "two_pass"]:
            try:
                result = evaluate_step(model, processor, step, goal, history_text, mode=mode)
                reward, is_correct = compute_reward(result, gt_action, image_w, image_h)
                result["reward"] = reward
                result["is_correct"] = is_correct
                result["episode_id"] = item["episode_id"]
                result["step_idx"] = item["step_idx"]
            except Exception as e:
                result = {
                    "mode": mode, "error": str(e),
                    "reward": 0, "is_correct": False,
                    "episode_id": item["episode_id"],
                    "step_idx": item["step_idx"],
                }

            results[mode].append(result)

        # Progress
        if (i + 1) % 10 == 0:
            op_acc = np.mean([r["is_correct"] for r in results["one_pass"]])
            tp_acc = np.mean([r["is_correct"] for r in results["two_pass"]])
            op_r = np.mean([r["reward"] for r in results["one_pass"]])
            tp_r = np.mean([r["reward"] for r in results["two_pass"]])
            n_elem = np.mean([r.get("n_elements", 0) for r in results["two_pass"] if "n_elements" in r])
            print(f"  [{i+1}/{len(eval_steps)}] "
                  f"one_pass: acc={op_acc:.1%} r={op_r:.3f} | "
                  f"two_pass: acc={tp_acc:.1%} r={tp_r:.3f} | "
                  f"avg_elements={n_elem:.1f}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    for mode in ["one_pass", "two_pass"]:
        rs = results[mode]
        acc = np.mean([r["is_correct"] for r in rs])
        mean_r = np.mean([r["reward"] for r in rs])
        mean_time = np.mean([r.get("gen_time", 0) for r in rs])
        print(f"\n  {mode}:")
        print(f"    Accuracy: {sum(r['is_correct'] for r in rs)}/{len(rs)} = {acc:.1%}")
        print(f"    Mean reward: {mean_r:.3f}")
        print(f"    Mean gen time: {mean_time:.1f}s")

        if mode == "two_pass":
            n_elems = [r.get("n_elements", 0) for r in rs if "n_elements" in r]
            if n_elems:
                print(f"    Mean elements detected: {np.mean(n_elems):.1f}")
                print(f"    Element detection rate: {sum(1 for n in n_elems if n > 0)}/{len(n_elems)}")

    # Per-step comparison
    print(f"\n  Per-step comparison:")
    both_correct = 0
    op_only = 0
    tp_only = 0
    both_wrong = 0
    for op, tp in zip(results["one_pass"], results["two_pass"]):
        if op["is_correct"] and tp["is_correct"]:
            both_correct += 1
        elif op["is_correct"]:
            op_only += 1
        elif tp["is_correct"]:
            tp_only += 1
        else:
            both_wrong += 1

    n = len(results["one_pass"])
    if n > 0:
        print(f"    Both correct:     {both_correct:3d} ({both_correct/n:.1%})")
        print(f"    One-pass only:    {op_only:3d} ({op_only/n:.1%})")
        print(f"    Two-pass only:    {tp_only:3d} ({tp_only/n:.1%})")
        print(f"    Both wrong:       {both_wrong:3d} ({both_wrong/n:.1%})")
    else:
        print("    No steps evaluated.")

    # Save detailed results
    output_file = os.path.join(args.output_dir, f"two_pass_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
    # Clean results for JSON serialization
    for mode in results:
        for r in results[mode]:
            if "elements" in r and r["elements"]:
                pass  # Already serializable
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
