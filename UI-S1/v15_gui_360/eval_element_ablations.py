#!/usr/bin/env python3
"""
Element Ablation Experiments: What is the real bottleneck?

Decomposes the action prediction problem to identify where the model fails:
  1. one_pass:         Standard direct prediction (baseline)
  2. oracle_top5:      Only 5 nearest elements to GT (tests list length confusion)
  3. oracle_subgoal:   Tell model WHAT to click (GT element text), not WHERE (tests grounding vs reasoning)
  4. subgoal_cot:      Force model to first state sub-goal, then act (tests if explicit reasoning helps)

Key question: Is the bottleneck...
  - "too many elements" → oracle_top5 >> oracle_all
  - "doesn't know WHAT to click" → oracle_subgoal >> one_pass
  - "doesn't reason step-by-step" → subgoal_cot >> one_pass
  - "can't visually locate" → oracle_subgoal ≈ one_pass (even knowing WHAT doesn't help)
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

# Mode 1: one_pass (standard)
PROMPT_ONE_PASS = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

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

# Mode 2: oracle_top5 (short element list)
PROMPT_ORACLE_TOP5 = """You are a GUI automation agent. Given the instruction, a short list of nearby interactive UI elements, and action history, select the correct element and generate the action.

The instruction is:
{instruction}

Nearby interactive UI elements:
{elements}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: Use the coordinates from the UI elements listed above.

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

# Mode 3: oracle_subgoal (tell model WHAT to click)
PROMPT_ORACLE_SUBGOAL = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

Hint: In this step, you should interact with the UI element described as: "{gt_element_text}" ({gt_element_type}).
Find this element on the screenshot and click on it at the correct pixel coordinates.

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen.

Output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

Only **ONE** action should be taken at a time."""

# Mode 4: subgoal_cot (force sub-goal reasoning)
PROMPT_SUBGOAL_COT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

The instruction is:
{instruction}

The history of actions are:
{history}

The actions supported are:
{actions}
Important: All coordinate parameters for a predicted action must be absolute pixel positions on the screen.

Follow these steps:
1. CURRENT STATE: What do you see on the screen right now?
2. SUB-GOAL: Based on the instruction and history, what specific UI element do you need to interact with in this step? Name it precisely.
3. LOCATE: Where exactly is that element on the screenshot? Describe its position.
4. ACTION: Generate the action with the correct coordinates.

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{{
  "function": "<function name>",
  "args": {{}},
  "status": "CONTINUE"
}}
</tool_call>

Only **ONE** action should be taken at a time."""


# ═══════════════════════════════════════════════════════════════════
# Model loading and inference
# ═══════════════════════════════════════════════════════════════════

def load_model(model_path, image_max_pixels=602112):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(
        model_path, max_pixels=image_max_pixels, min_pixels=256 * 28 * 28,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
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
                **inputs, max_new_tokens=max_new_tokens,
                temperature=0.1, top_p=0.95, do_sample=True,
            )
    prompt_len = inputs["input_ids"].shape[1]
    resp_ids = output_ids[0, prompt_len:]
    return processor.tokenizer.decode(resp_ids, skip_special_tokens=True)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def parse_action(text: str) -> Optional[Dict]:
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
    m = re.search(r'(click|type|swipe)\s*\(', text)
    if m:
        func = m.group(1)
        coord_m = re.search(r'coordinate\s*=\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', text[m.start():])
        if coord_m:
            return {"type": func, "coordinate": [int(coord_m.group(1)), int(coord_m.group(2))]}
    return None


def format_action_as_tool_call(action: Dict) -> str:
    if action is None:
        return ""
    tc = {"function": action["type"], "args": {}, "status": "CONTINUE"}
    for k in ["coordinate", "keys", "direction", "distance", "button", "double",
              "pressed", "clear_current_text", "control_focus", "text"]:
        if k in action:
            tc["args"][k] = action[k]
    return f"<tool_call>\n{json.dumps(tc)}\n</tool_call>"


def format_elements(controls: List[Dict]) -> str:
    lines = []
    for i, c in enumerate(controls, 1):
        text = c.get("control_text", "")
        rect = c.get("control_rect", [0, 0, 0, 0])
        ctype = c.get("control_type", "")
        cx = (rect[0] + rect[2]) // 2
        cy = (rect[1] + rect[3]) // 2
        if text:
            lines.append(f'[{i}] {ctype}: "{text}" at [{cx}, {cy}]')
        else:
            lines.append(f'[{i}] {ctype}: (no text) at [{cx}, {cy}]')
    return "\n".join(lines)


def find_gt_control(controls, gt_coord):
    """Find the control whose bbox contains GT coordinate, or nearest."""
    if gt_coord is None or any(x is None for x in gt_coord):
        return None, float("inf")
    best_control = None
    best_dist = float("inf")
    # First try: find control containing GT
    for c in controls:
        rect = c.get("control_rect", [0, 0, 0, 0])
        if any(x is None for x in rect):
            continue
        if rect[0] <= gt_coord[0] <= rect[2] and rect[1] <= gt_coord[1] <= rect[3]:
            cx = (rect[0] + rect[2]) / 2
            cy = (rect[1] + rect[3]) / 2
            dist = ((cx - gt_coord[0])**2 + (cy - gt_coord[1])**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best_control = c
    # Fallback: nearest center
    if best_control is None:
        for c in controls:
            rect = c.get("control_rect", [0, 0, 0, 0])
            cx = (rect[0] + rect[2]) / 2
            cy = (rect[1] + rect[3]) / 2
            dist = ((cx - gt_coord[0])**2 + (cy - gt_coord[1])**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best_control = c
    return best_control, best_dist


def find_top_k_controls(controls, gt_coord, k=5):
    """Find k controls nearest to GT coordinate."""
    dists = []
    for c in controls:
        rect = c.get("control_rect", [0, 0, 0, 0])
        if any(x is None for x in rect):
            continue
        cx = (rect[0] + rect[2]) / 2
        cy = (rect[1] + rect[3]) / 2
        dist = ((cx - gt_coord[0])**2 + (cy - gt_coord[1])**2)**0.5
        dists.append((dist, c))
    dists.sort(key=lambda x: x[0])
    return [c for _, c in dists[:k]]


def extract_from_conversation(conv):
    human_msg = conv[0]["value"]
    m = re.search(r'The instruction is:\s*\n(.+?)(?:\n\nThe history|$)', human_msg, re.DOTALL)
    instruction = m.group(1).strip() if m else ""
    m = re.search(r'The history of actions are:\s*\n(.+?)(?:\n\nThe actions supported|$)', human_msg, re.DOTALL)
    history = m.group(1).strip() if m else "None"
    return instruction, history


def parse_gt_action(gt_response: str) -> Optional[Dict]:
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

MODES = ["one_pass", "oracle_top5", "oracle_subgoal", "subgoal_cot"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--ap_data", required=True)
    parser.add_argument("--sp_data", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", default="v15_gui_360/outputs/preliminary_tests/element_ablations")
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--image_max_pixels", type=int, default=602112)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data...", flush=True)
    with open(args.ap_data) as f:
        ap_data = json.load(f)
    with open(args.sp_data) as f:
        sp_data = json.load(f)
    sp_index = {d["images"][0]: d for d in sp_data}

    # Build candidates: click actions with screen parsing + GT control found
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
        gt_coord = gt_action.get("coordinate", [0, 0])
        if gt_coord is None or any(x is None for x in gt_coord):
            continue

        try:
            controls = json.loads(sp_index[img]["conversation"][1]["value"])
        except json.JSONDecodeError:
            continue
        if not controls or len(controls) < 3:
            continue

        # Find GT control
        gt_control, gt_dist = find_gt_control(controls, gt_coord)
        if gt_control is None:
            continue

        instruction, history = extract_from_conversation(d["conversation"])
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
            "gt_control": gt_control,
            "gt_control_dist": gt_dist,
        })

    print(f"Candidates: {len(candidates)}", flush=True)

    # Subsample
    rng = np.random.default_rng(42)
    indices = rng.choice(len(candidates), size=min(args.max_steps, len(candidates)), replace=False)
    eval_steps = [candidates[i] for i in sorted(indices)]
    print(f"Evaluating {len(eval_steps)} steps × {len(MODES)} modes = {len(eval_steps)*len(MODES)} inferences", flush=True)

    # Load model
    print(f"Loading model...", flush=True)
    model, processor = load_model(args.model_path, args.image_max_pixels)
    print("Model loaded.", flush=True)

    # Evaluate
    results = {mode: [] for mode in MODES}

    for i, item in enumerate(eval_steps):
        try:
            image = Image.open(item["image_path"]).convert("RGB")
        except Exception as e:
            for mode in MODES:
                results[mode].append({"id": item["id"], "error": str(e), "reward": 0, "is_correct": False})
            continue

        image_w, image_h = image.size
        gt_action = item["gt_action"]
        gt_coord = gt_action.get("coordinate", [0, 0])

        for mode in MODES:
            try:
                if mode == "one_pass":
                    prompt = PROMPT_ONE_PASS.format(
                        instruction=item["instruction"],
                        history=item["history"],
                        actions=SUPPORTED_ACTIONS,
                    )
                elif mode == "oracle_top5":
                    top5 = find_top_k_controls(item["controls"], gt_coord, k=5)
                    # Shuffle to avoid position bias (GT might always be first)
                    rng2 = np.random.default_rng(hash(item["id"]) % 2**31)
                    order = rng2.permutation(len(top5))
                    top5_shuffled = [top5[j] for j in order]
                    prompt = PROMPT_ORACLE_TOP5.format(
                        instruction=item["instruction"],
                        elements=format_elements(top5_shuffled),
                        history=item["history"],
                        actions=SUPPORTED_ACTIONS,
                    )
                elif mode == "oracle_subgoal":
                    gt_ctrl = item["gt_control"]
                    gt_text = gt_ctrl.get("control_text", "(unnamed element)")
                    gt_type = gt_ctrl.get("control_type", "Control")
                    prompt = PROMPT_ORACLE_SUBGOAL.format(
                        instruction=item["instruction"],
                        history=item["history"],
                        gt_element_text=gt_text,
                        gt_element_type=gt_type,
                        actions=SUPPORTED_ACTIONS,
                    )
                elif mode == "subgoal_cot":
                    prompt = PROMPT_SUBGOAL_COT.format(
                        instruction=item["instruction"],
                        history=item["history"],
                        actions=SUPPORTED_ACTIONS,
                    )

                messages = [{"role": "user", "content": [
                    {"type": "image", "image": item["image_path"]},
                    {"type": "text", "text": prompt},
                ]}]

                t0 = time.time()
                max_tokens = 768 if mode == "subgoal_cot" else 512
                response = generate_response(model, processor, messages, image, max_new_tokens=max_tokens)
                gen_time = time.time() - t0

                action = parse_action(response)
                pred_text = format_action_as_tool_call(action) if action else ""
                reward, info = compute_step_reward(
                    pred_text, gt_action, image_w, image_h,
                    w_format=0.1, w_type=0.2, w_content=0.7,
                )

                results[mode].append({
                    "id": item["id"],
                    "mode": mode,
                    "action": action,
                    "reward": reward,
                    "is_correct": reward >= 0.5,
                    "content_reward": info.get("content_reward", 0),
                    "gen_time": gen_time,
                    "response": response[:400],
                })
            except Exception as e:
                results[mode].append({
                    "id": item["id"], "mode": mode, "error": str(e),
                    "reward": 0, "is_correct": False,
                })

        # Progress every 10 steps
        if (i + 1) % 10 == 0:
            line = f"  [{i+1}/{len(eval_steps)}]"
            for mode in MODES:
                acc = np.mean([r["is_correct"] for r in results[mode]])
                line += f"  {mode}={acc:.1%}"
            print(line, flush=True)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 80, flush=True)
    print("ELEMENT ABLATION RESULTS", flush=True)
    print("=" * 80, flush=True)

    # Per-mode stats
    for mode in MODES:
        rs = results[mode]
        acc = np.mean([r["is_correct"] for r in rs])
        mean_r = np.mean([r["reward"] for r in rs])
        mean_cr = np.mean([r.get("content_reward", 0) for r in rs])
        mean_t = np.mean([r.get("gen_time", 0) for r in rs])
        n_correct = sum(r["is_correct"] for r in rs)
        print(f"\n  {mode}:")
        print(f"    Accuracy: {n_correct}/{len(rs)} = {acc:.1%}")
        print(f"    Mean reward: {mean_r:.3f}  content_reward: {mean_cr:.3f}")
        print(f"    Mean gen time: {mean_t:.1f}s")

    # Pairwise comparisons vs baseline
    print(f"\n  Pairwise vs one_pass:")
    for mode in MODES[1:]:
        both_c = sum(a["is_correct"] and b["is_correct"]
                     for a, b in zip(results["one_pass"], results[mode]))
        op_only = sum(a["is_correct"] and not b["is_correct"]
                      for a, b in zip(results["one_pass"], results[mode]))
        m_only = sum(not a["is_correct"] and b["is_correct"]
                     for a, b in zip(results["one_pass"], results[mode]))
        both_w = sum(not a["is_correct"] and not b["is_correct"]
                     for a, b in zip(results["one_pass"], results[mode]))
        n = len(results["one_pass"])
        net = m_only - op_only
        print(f"\n    {mode} vs one_pass:")
        print(f"      Both correct: {both_c} ({both_c/n:.1%})  |  one_pass only: {op_only} ({op_only/n:.1%})")
        print(f"      {mode} only: {m_only} ({m_only/n:.1%})  |  Both wrong: {both_w} ({both_w/n:.1%})")
        print(f"      Net gain: {'+' if net > 0 else ''}{net} steps")

    # Diagnostic: what happens when oracle_subgoal flips?
    print(f"\n  Oracle subgoal diagnostic:")
    subgoal_gains = []  # cases where subgoal helped
    subgoal_losses = []
    for i, (op, sg) in enumerate(zip(results["one_pass"], results["oracle_subgoal"])):
        if not op["is_correct"] and sg["is_correct"]:
            item = eval_steps[i]
            subgoal_gains.append({
                "instruction": item["instruction"][:80],
                "gt_element": item["gt_control"].get("control_text", "")[:60],
            })
        elif op["is_correct"] and not sg["is_correct"]:
            item = eval_steps[i]
            subgoal_losses.append({
                "instruction": item["instruction"][:80],
                "gt_element": item["gt_control"].get("control_text", "")[:60],
            })

    print(f"    Gains ({len(subgoal_gains)} steps where oracle_subgoal fixed one_pass):")
    for g in subgoal_gains[:5]:
        print(f"      instr: {g['instruction']}")
        print(f"      elem:  {g['gt_element']}")
    print(f"    Losses ({len(subgoal_losses)} steps where oracle_subgoal broke one_pass):")
    for l in subgoal_losses[:5]:
        print(f"      instr: {l['instruction']}")
        print(f"      elem:  {l['gt_element']}")

    # GT control text analysis
    print(f"\n  GT control text analysis:")
    empty_text = sum(1 for item in eval_steps if not item["gt_control"].get("control_text", "").strip())
    print(f"    GT controls with empty text: {empty_text}/{len(eval_steps)} ({empty_text/len(eval_steps):.1%})")
    text_lens = [len(item["gt_control"].get("control_text", "")) for item in eval_steps]
    print(f"    GT control text length: mean={np.mean(text_lens):.0f}, median={np.median(text_lens):.0f}")

    # Save
    output_file = os.path.join(args.output_dir, f"ablation_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {output_file}", flush=True)


if __name__ == "__main__":
    main()
