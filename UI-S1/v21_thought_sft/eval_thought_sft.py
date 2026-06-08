#!/usr/bin/env python3
"""Evaluate V21 Thought-Augmented SFT model.

The V21 model was trained with response = thought + <tool_call>action</tool_call>.
At eval time, it should generate reasoning text before the action.

This script uses the EXACT same prompt format as the training data (including
the full 10-action-type block from GUI-360) to maximize format matching.

Supports two action description modes:
  --actions_mode training   Use the full training actions block (10 types, default)
  --actions_mode eval       Use the simplified eval actions block (4 types)

Usage:
    python v21_thought_sft/eval_thought_sft.py \
        --test_data v12_gui_360/data/gui360_test_1000_balanced.jsonl \
        --api_url http://localhost:8000/v1 \
        --model_name thought_sft \
        --output_dir v21_thought_sft/outputs/eval_pred \
        --threads 128
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v13_gui_360.eval_gui360_template import (
    parse_tool_call, _format_action_for_history, _encode_screenshot,
    SUPPORTED_ACTIONS as EVAL_ACTIONS,
)
from v13_gui_360.reward import compute_step_reward


# ═══════════════════════════════════════════════════════════════════════
# Training-matched actions block (full GUI-360 action set, 10 types)
# Extracted from gui360_thought_train.jsonl — identical across all 17,264 samples
# ═══════════════════════════════════════════════════════════════════════

TRAINING_ACTIONS = """<action>
- click
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to click at.
    - button: str, The mouse button to click. One of ''left'', ''right'', ''middle'' or ''x'' (Default: ''left'')
    - double: bool, Whether to perform a double click or not (Default: False)'
    - pressed: str|None, The keyboard key to press while clicking. Common keys include: CONTROL (Ctrl), SHIFT (Shift), MENU (Alt), etc. Use the key names without VK_ prefix or braces. For example, 'CONTROL' for the Control key (Default: None)
  - Example: click(coordinate=[100, 100], button='left', double=False, pressed=None), click(coordinate=[100, 100], button='x')
- type
  - Args:
    - coordinate: [x, y], the absolute position on the screen you want to type at.
    - keys: str, The key to input. It can be any key on the keyboard, with special keys represented by their virtual key codes. For example, "{VK_CONTROL}c" represents the Ctrl+C shortcut key.
    - clear_current_text: bool, Whether to clear the current text in the Edit before setting the new text. If True, the current text will be completely replaced by the new text. (Default: False)
    - control_focus: bool, Whether to focus on your selected control item before typing the keys. If False, the hotkeys will operate on the application window. (Default: True)
  - Example: type(coordinate=[100, 100], keys='Hello'), type(coordinate=[100, 100], keys='{VK_CONTROL}c'), type(coordinate=[100, 100], keys="{TAB 2}")
- drag
  - Args:
    - start_coordinate: [x, y], the absolute position on the screen where the drag starts.
    - end_coordinate: [x, y], the absolute position on the screen where the drag ends.
    - button: str, The mouse button to drag. One of 'left', 'right'. (Default: 'left')
    - duration: float, The duration of the drag action in seconds. (Default: 1.0)
    - key_hold: str|None, The keyboard key to hold while dragging. Common keys include: shift (Shift), control (Ctrl), alt (Alt), etc. Use lowercase key names. For example, 'shift' for the shift key (Default: None)
  - Example: drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='left', duration=1.0, key_hold=None), drag(start_coordinate=[100, 100], end_coordinate=[200, 200], button='right', duration=1.0, key_hold='shift')
- wheel_mouse_input
  - Args:
    - coordinate: [x, y], the absolute position on the screen to scroll.
    - wheel_dist: int, The number of wheel notches to scroll. Positive values indicate upward scrolling, negative values indicate downward scrolling.
  - Example: wheel_mouse_input(coordinate=[100, 100], wheel_dist=-5), wheel_mouse_input(coordinate=[100, 100], wheel_dist=3)
- table2markdown
  - Args:
    - sheet_name: str|int, The name or index of the sheet to get the table content. The index starts from 1.
  - Example: table2markdown(sheet_name=1)
- insert_excel_table
  - Args:
    - table: list[list], The table content to insert. The table is a list of list of strings or numbers.
    - sheet_name: str, The name of the sheet to insert the table.
    - start_row: int, The start row to insert the table, starting from 1.
    - start_col: int, The start column to insert the table, starting from 1.
  - Example: insert_excel_table(table=[["Name", "Age", "Gender"], ["Alice", 30, "Female"], ["Bob", 25, "Male"], ["Charlie", 35, "Male"]], sheet_name="Sheet1", start_row=1, start_col=1)
- select_table_range
  - Args:
    - sheet_name: str, The name of the sheet.
    - start_row: int, The start row, starting from 1.
    - start_col: int, The start column, starting from 1.
    - end_row: int, The end row. If ==-1, select to the end of the document with content.
    - end_col: int, The end column. If ==-1, select to the end of the document with content.
  - Example: select_table_range(sheet_name="Sheet1", start_row=1, start_col=1, end_row=3, end_col=3)
- set_cell_value
  - Args:
    - sheet_name: str, The name of the sheet.
    - row: int, The row number (1-based).
    - col: int, The column number (1-based).
    - value: str|int|float|None, The value to set in the cell. If None, just select the cell.
    - is_formula: bool, If True, treat the value as a formula, otherwise treat it as a normal value. (Default: False)
  - Example: set_cell_value(sheet_name="Sheet1", row=1, col=1, value="Hello", is_formula=False), set_cell_value(sheet_name="Sheet1", row=2, col=2, value="=SUM(A1:A10)", is_formula=True)
- auto_fill
  - Args:
    - sheet_name: str, The name of the sheet.
    - start_row: int, The starting row number (1-based).
    - start_col: int, The starting column number (1-based).
    - end_row: int, The ending row number (1-based).
    - end_col: int, The ending column number (1-based).
  - Example: auto_fill(sheet_name="Sheet1", start_row=1, start_col=1, end_row=10, end_col=3)
- reorder_columns
  - Args:
    - sheet_name: str, The name of the sheet.
    - desired_order: list[str], The list of column names in the new order.
  - Example: reorder_columns(sheet_name="Sheet1", desired_order=["Income", "Date", "Expense"])
</action>"""

# Training-matched prompt template (identical to training data user message)
THOUGHT_PROMPT = """You are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.

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

If you think the task is finished, you can output status as "FINISH" and take no action. Like:
<tool_call>
{{
  "function": "",
  "args": {{}},
  "status": "FINISH"
}}
</tool_call>

Only **ONE** action should be taken at a time. If the instruction could apply to multiple elements, choose the most relevant one based on the context provided by the screenshot and previous actions.
"""


# ═══════════════════════════════════════════════════════════════════════
# Episode Evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_episode(
    client: OpenAI,
    model_name: str,
    episode: Dict,
    gt_history: bool = True,
    match_threshold: float = 0.5,
    image_max_pixels: Optional[int] = None,
    actions_text: str = TRAINING_ACTIONS,
) -> Dict[str, Any]:
    """Evaluate one episode with thought-augmented model."""
    episode_id = episode["episode_id"]
    goal = episode["goal"]
    steps = episode["steps"]
    num_steps = len(steps)

    history = []
    step_results = []
    first_error_step = None
    correct_steps = 0
    total_reasoning_chars = 0
    steps_with_reasoning = 0

    for i, step in enumerate(steps):
        gt_action = step["action"]
        screenshot = step["screenshot"]
        image_w = step.get("image_w", 1040)
        image_h = step.get("image_h", 736)

        history_text = "\n".join(history) if history else ""

        b64 = _encode_screenshot(screenshot, image_max_pixels)

        prompt_text = THOUGHT_PROMPT.format(
            instruction=goal,
            history=history_text,
            actions=actions_text,
        )

        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt_text},
        ]}]

        pred_text = ""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.0,
            )
            pred_text = response.choices[0].message.content or ""
        except Exception as e:
            print(f"  [ep {episode_id}] step {i+1} error: {e}")

        # Parse action from <tool_call>
        pred_action = parse_tool_call(pred_text)
        if pred_action is None:
            m = re.search(r'<action>\s*(\{.*?\})\s*</action>', pred_text, re.DOTALL)
            if m:
                try:
                    pred_action = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        if pred_action:
            fake_text = f"<action>{json.dumps(pred_action)}</action>"
        else:
            fake_text = pred_text

        reward, info = compute_step_reward(fake_text, gt_action, image_w, image_h)
        success = reward >= match_threshold

        # Extract reasoning (text before <tool_call>)
        reasoning = ""
        tc_match = re.search(r'<tool_call>', pred_text)
        if tc_match:
            reasoning = pred_text[:tc_match.start()].strip()
        if reasoning:
            total_reasoning_chars += len(reasoning)
            steps_with_reasoning += 1

        step_results.append({
            "step_idx": i,
            "success": success,
            "reward": reward,
            "reasoning": reasoning[:500],
            "reasoning_len": len(reasoning),
            "pred_text": pred_text[:800],
            "pred_action": info.get("pred_action"),
            "pred_type": info.get("pred_type"),
            "gt_type": info.get("gt_type"),
            "type_reward": info.get("type_reward", 0),
            "content_reward": info.get("content_reward", 0),
        })

        # Update history (same format as training: "Step N: action(args)")
        if gt_history:
            history.append(_format_action_for_history(gt_action, i + 1))
        else:
            history.append(_format_action_for_history(pred_action, i + 1))

        if success:
            correct_steps += 1
        else:
            if first_error_step is None:
                first_error_step = i + 1

    progress = (first_error_step - 1) / num_steps if first_error_step else 1.0
    task_success = first_error_step is None and len(step_results) == num_steps

    return {
        "episode_id": episode_id,
        "goal": goal,
        "num_steps": num_steps,
        "steps_evaluated": len(step_results),
        "correct_steps": correct_steps,
        "task_success": task_success,
        "progress": progress,
        "first_error_step": first_error_step,
        "steps_with_reasoning": steps_with_reasoning,
        "avg_reasoning_len": total_reasoning_chars / max(steps_with_reasoning, 1),
        "steps": step_results,
    }


# ═══════════════════════════════════════════════════════════════════════
# Breakdown Analysis
# ═══════════════════════════════════════════════════════════════════════

def compute_breakdown(results: Dict[str, Dict]) -> Dict[str, Any]:
    """Per-step-position and per-task-length metrics."""

    step_pos_correct = defaultdict(int)
    step_pos_total = defaultdict(int)
    step_pos_type_correct = defaultdict(int)
    step_pos_reasoning_lens = defaultdict(list)

    length_buckets = {"1": (1, 1), "2-3": (2, 3), "4-5": (4, 5), "6+": (6, 999)}
    length_success = defaultdict(int)
    length_total = defaultdict(int)
    length_progress = defaultdict(float)
    length_steps_correct = defaultdict(int)
    length_steps_total = defaultdict(int)

    total_reasoning = 0
    total_steps_with_reasoning = 0

    for eid, result in results.items():
        num_steps = result["num_steps"]
        total_reasoning += result.get("steps_with_reasoning", 0)

        bucket = "6+"
        for bname, (lo, hi) in length_buckets.items():
            if lo <= num_steps <= hi:
                bucket = bname
                break

        length_total[bucket] += 1
        if result["task_success"]:
            length_success[bucket] += 1
        length_progress[bucket] += result["progress"]

        for step in result["steps"]:
            idx = step["step_idx"]
            step_pos_total[idx] += 1
            if step["success"]:
                step_pos_correct[idx] += 1
            if step.get("type_reward", 0) >= 1.0:
                step_pos_type_correct[idx] += 1
            step_pos_reasoning_lens[idx].append(step.get("reasoning_len", 0))

            length_steps_total[bucket] += 1
            if step["success"]:
                length_steps_correct[bucket] += 1

    total_all_steps = sum(step_pos_total.values())

    # Per-step-position
    max_step = max(step_pos_total.keys()) if step_pos_total else 0
    step_position_acc = {}
    for idx in range(min(max_step + 1, 15)):
        total = step_pos_total.get(idx, 0)
        correct = step_pos_correct.get(idx, 0)
        type_correct = step_pos_type_correct.get(idx, 0)
        r_lens = step_pos_reasoning_lens.get(idx, [])
        if total > 0:
            step_position_acc[f"step_{idx}"] = {
                "accuracy": correct / total,
                "type_accuracy": type_correct / total,
                "avg_reasoning_len": sum(r_lens) / len(r_lens) if r_lens else 0,
                "total": total,
                "correct": correct,
            }

    # Per-task-length
    task_length_metrics = {}
    for bname in length_buckets:
        total = length_total.get(bname, 0)
        if total > 0:
            s_total = length_steps_total.get(bname, 0)
            task_length_metrics[bname] = {
                "tsr": length_success[bname] / total,
                "avg_progress": length_progress[bname] / total,
                "step_sr": length_steps_correct[bname] / s_total if s_total > 0 else 0,
                "num_episodes": total,
                "num_success": length_success[bname],
            }

    return {
        "step_position_accuracy": step_position_acc,
        "task_length_metrics": task_length_metrics,
        "reasoning_stats": {
            "steps_with_reasoning": total_reasoning,
            "total_steps": total_all_steps,
            "reasoning_rate": total_reasoning / max(total_all_steps, 1),
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="V21 Thought-Augmented SFT Evaluation")
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--gt_history", action="store_true", default=False)
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument("--actions_mode", choices=["training", "eval"], default="training",
                        help="Which actions block to use: 'training' = full 10-type (matches training data), "
                             "'eval' = simplified 4-type (matches prior evals)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    actions_text = TRAINING_ACTIONS if args.actions_mode == "training" else EVAL_ACTIONS

    episodes = []
    with open(args.test_data) as f:
        for line in f:
            episodes.append(json.loads(line))
    total_loaded = len(episodes)
    episodes = episodes[args.start:args.end]
    print(f"Loaded {len(episodes)} episodes (shard [{args.start}:{args.end}] of {total_loaded})")
    print(f"Actions mode: {args.actions_mode}")
    print(f"GT history: {args.gt_history}")

    client = OpenAI(base_url=args.api_url, api_key="dummy")
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    total_success = 0
    total_progress = 0.0
    total_steps = 0
    total_correct = 0

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                evaluate_episode, client, args.model_name, ep,
                args.gt_history, args.match_threshold,
                args.image_max_pixels, actions_text,
            ): ep["episode_id"]
            for ep in episodes
        }

        pbar = tqdm(total=len(episodes), desc="Eval thought_sft")
        for future in as_completed(futures):
            result = future.result()
            eid = result["episode_id"]
            results[eid] = result

            if result["task_success"]:
                total_success += 1
            total_progress += result["progress"]
            total_steps += result["steps_evaluated"]
            total_correct += result["correct_steps"]

            n = len(results)
            pbar.update(1)
            pbar.set_postfix({
                "TSR": f"{total_success/n:.3f}",
                "StepSR": f"{total_correct/total_steps:.3f}" if total_steps > 0 else "0",
            })
        pbar.close()

    # Compute breakdowns
    breakdown = compute_breakdown(results)

    n = len(results)
    summary = {
        "num_episodes": n,
        "tsr": total_success / n if n > 0 else 0,
        "avg_progress": total_progress / n if n > 0 else 0,
        "step_sr": total_correct / total_steps if total_steps > 0 else 0,
        "total_steps_evaluated": total_steps,
        "total_steps_correct": total_correct,
        "gt_history": args.gt_history,
        "actions_mode": args.actions_mode,
        "match_threshold": args.match_threshold,
        **breakdown,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output_dir, f"eval_summary_{ts}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, f"eval_results_{ts}.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print results
    print(f"\n{'='*60}")
    print(f"V21 Thought SFT Eval Results")
    print(f"{'='*60}")
    print(f"  TSR:      {summary['tsr']*100:.1f}%")
    print(f"  Step SR:  {summary['step_sr']*100:.1f}%")
    print(f"  Progress: {summary['avg_progress']*100:.1f}%")
    hist_mode = "GT history" if args.gt_history else "Pred history"
    print(f"  Mode:     {hist_mode}, actions={args.actions_mode}")

    rs = breakdown.get("reasoning_stats", {})
    print(f"\n  --- Reasoning Stats ---")
    print(f"  Steps with reasoning: {rs.get('steps_with_reasoning', 0)}/{rs.get('total_steps', 0)} "
          f"({rs.get('reasoning_rate', 0)*100:.1f}%)")

    print(f"\n  --- By Task Length ---")
    print(f"  {'Length':>7s}  {'TSR':>7s}  {'StepSR':>7s}  {'Progress':>9s}  {'n':>4s}")
    print(f"  {'-'*42}")
    for bname in ["1", "2-3", "4-5", "6+"]:
        metrics = summary.get("task_length_metrics", {}).get(bname, {})
        if metrics:
            print(f"  {bname:>7s}  {metrics['tsr']*100:6.1f}%  {metrics['step_sr']*100:6.1f}%  "
                  f"{metrics['avg_progress']*100:8.1f}%  {metrics['num_episodes']:4d}")

    print(f"\n  --- By Step Position ---")
    for sname, metrics in sorted(summary.get("step_position_accuracy", {}).items(),
                                  key=lambda x: int(x[0].split("_")[1])):
        print(f"  {sname:>8s}: acc={metrics['accuracy']*100:5.1f}%  "
              f"type_acc={metrics['type_accuracy']*100:5.1f}%  "
              f"avg_reason_len={metrics['avg_reasoning_len']:.0f}  "
              f"(n={metrics['total']})")

    print(f"\n  Baselines: standard=21.9%, CoT=23.1%, type_focused=23.6%")
    print(f"  Oracle (GT thought): 35.2%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
