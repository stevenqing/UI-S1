#!/usr/bin/env python3
"""
Experiment 0.5: Retry with Error Context

Test if giving the model "your action failed" context improves its next
prediction. At divergence steps from paired trajectories:
1. Get the original fail action (what the model predicted wrong)
2. Construct a recovery prompt with error context
3. Generate a new action with the model
4. Compare the retry action to the ground truth (success action)

Success criteria: Retry-with-context accuracy > original fail accuracy

Usage:
    python scripts/exp0/exp0_5_retry_with_context.py \
        --model_name gui360_lora_v4_ckpt354 \
        --num_examples 100

    # Analysis only:
    python scripts/exp0/exp0_5_retry_with_context.py --analyze_only
"""

import argparse
import base64
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.exp0.data_utils import (
    coordinate_distance,
    find_divergence_step,
    get_screenshot_path,
    is_coord_in_bbox,
    load_paired_trajectories,
    normalize_action,
)

# System prompt with <tool_call> format instructions (from SFT training data)
SYSTEM_PROMPT = """You are a helpful assistant.

# Tools

You can use the following tools:

## click

Click on an element on the screen.

```json
{"function": "click", "args": {"coordinate": [x, y]}}
```

## type

Type text at a specific position.

```json
{"function": "type", "args": {"coordinate": [x, y], "text": "content"}}
```

## drag

Drag from one position to another.

```json
{"function": "drag", "args": {"startCoordinate": [x1, y1], "endCoordinate": [x2, y2]}}
```

## select_text

Select text between two coordinates.

```json
{"function": "select_text", "args": {"startCoordinate": [x1, y1], "endCoordinate": [x2, y2]}}
```

## scroll

Scroll the screen.

```json
{"function": "scroll", "args": {"coordinate": [x, y], "direction": "up/down/left/right"}}
```

## hotkey

Press a hotkey combination.

```json
{"function": "hotkey", "args": {"key": "ctrl+c"}}
```

First, explain your reasoning process—describe how you analyze the screenshot, understand the current state, and determine what action should be taken next based on the instruction and previous actions.

Then output your action within <tool_call></tool_call> tag like:
<tool_call>
{
  "function": "<function name>",
  "args": {},
  "status": "CONTINUE"
}
</tool_call>

If you think the task is finished, you can output status as "FINISH" and take no action. Like:
<tool_call>
{
  "function": "",
  "args": {},
  "status": "FINISH"
}
</tool_call>

Only **ONE** action should be taken at a time."""

# Prompt templates (user messages)
ORIGINAL_PROMPT_TEMPLATE = """Given the current screenshot, user instruction and action history, decide the next action.

The instruction is:
{task}

{history_text}

Based on the current screenshot, decide the next action."""

RETRY_PROMPT_TEMPLATE = """Given the current screenshot, user instruction and action history, decide the next action.

The instruction is:
{task}

{history_text}

IMPORTANT: The previous action attempted was: {failed_action_description}
However, this action did NOT achieve the intended goal. The action was incorrect.
Please carefully re-examine the screenshot and choose a DIFFERENT, correct action."""

RETRY_WITH_REASONING_TEMPLATE = """Given the current screenshot, user instruction and action history, decide the next action.

The instruction is:
{task}

{history_text}

IMPORTANT FEEDBACK: The previous action attempted was: {failed_action_description}
This action FAILED to achieve the goal. Common reasons for failure include:
- Clicking on the wrong UI element
- Clicking on a visually similar but incorrect element
- The target element is in a different location than expected

Please:
1. Re-examine the screenshot carefully
2. Identify the correct UI element for the current step
3. Choose a different action from the failed one"""


def image_to_data_url(image_path: str) -> str:
    """Convert image file to base64 data URL."""
    from PIL import Image

    image = Image.open(image_path)
    buf = BytesIO()
    image.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64_str}"


def parse_tool_call(text: str) -> dict | None:
    """Parse tool_call from model response."""
    try:
        match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(text)
    except Exception:
        return None


def describe_action(action: dict) -> str:
    """Create human-readable description of a normalized action."""
    action_type = action.get("action_type", "unknown")
    coord = action.get("coordinate")
    text = action.get("text")

    if action_type == "click" and coord:
        return f"Click at ({coord[0]:.0f}, {coord[1]:.0f})"
    elif action_type == "type" and text:
        coord_str = f" at ({coord[0]:.0f}, {coord[1]:.0f})" if coord else ""
        return f'Type "{text}"{coord_str}'
    elif action_type == "drag" and coord:
        coord2 = action.get("coordinate2")
        if coord2:
            return f"Drag from ({coord[0]:.0f}, {coord[1]:.0f}) to ({coord2[0]:.0f}, {coord2[1]:.0f})"
        return f"Drag starting at ({coord[0]:.0f}, {coord[1]:.0f})"
    elif action_type in ("scroll", "wheel_mouse_input"):
        direction = action.get("direction", "unknown")
        return f"Scroll {direction}"
    else:
        return f"{action_type}" + (f" at ({coord[0]:.0f}, {coord[1]:.0f})" if coord else "")


def build_history_text(steps: list[dict], up_to_step: int) -> str:
    """Build action history text from steps before the current step."""
    if up_to_step <= 0:
        return "No previous actions."

    history_lines = ["Previous actions:"]
    for i in range(min(up_to_step, len(steps))):
        action = normalize_action(steps[i])
        desc = describe_action(action)
        history_lines.append(f"  Step {i + 1}: {desc}")

    return "\n".join(history_lines)


def evaluate_prediction(pred_text: str, gt_action: dict, threshold: float = 50.0) -> dict:
    """Evaluate a model prediction against the ground truth success action."""
    pred_action = parse_tool_call(pred_text)

    result = {
        "function_match": False,
        "coord_match": False,
        "coord_distance": float("inf"),
        "full_match": False,
    }

    if pred_action is None:
        return result

    pred_func = pred_action.get("function", "")
    gt_type = gt_action.get("action_type", "")

    result["function_match"] = pred_func == gt_type

    if not result["function_match"]:
        return result

    # Coordinate comparison
    pred_args = pred_action.get("args", {})
    pred_coord = pred_args.get("coordinate")
    gt_coord = gt_action.get("coordinate")

    if pred_coord is not None and gt_coord is not None:
        dist = coordinate_distance(pred_coord, gt_coord)
        result["coord_distance"] = dist
        result["coord_match"] = dist < threshold

        # Also check bbox if available
        gt_bbox = gt_action.get("bbox")
        if gt_bbox:
            result["coord_match"] = result["coord_match"] or is_coord_in_bbox(pred_coord, gt_bbox)

    # Text comparison for type actions
    if gt_type == "type":
        pred_text_val = (pred_args.get("text") or "").lower().strip()
        gt_text_val = (gt_action.get("text") or "").lower().strip()
        result["text_match"] = pred_text_val == gt_text_val or pred_text_val in gt_text_val or gt_text_val in pred_text_val
        result["coord_match"] = result.get("text_match", False)

    result["full_match"] = result["function_match"] and result["coord_match"]
    return result


def build_retry_examples(max_examples: int = 100) -> list[dict]:
    """
    Build retry examples from paired trajectories at divergence points.

    Each example has:
    - The current screenshot (at divergence step)
    - The fail action (what went wrong)
    - The success action (ground truth)
    - Task description and history
    """
    print("Loading paired trajectories...")
    pairs = load_paired_trajectories(max_pairs=max(500, max_examples * 5))
    print(f"Found {len(pairs)} pairs")

    examples = []
    for pair in pairs:
        if len(examples) >= max_examples:
            break

        s_steps = pair["success_steps"]
        f_steps = pair["fail_steps"]
        div_step = find_divergence_step(s_steps, f_steps)

        if div_step >= len(s_steps) or div_step >= len(f_steps):
            continue

        # Get screenshot at divergence step (from fail trajectory since that's the "current state")
        screenshot_path = get_screenshot_path(f_steps[div_step], source="fail")
        if not screenshot_path:
            # Try success screenshot (before divergence, state should be similar)
            screenshot_path = get_screenshot_path(s_steps[div_step], source="test")
            if not screenshot_path:
                screenshot_path = get_screenshot_path(s_steps[div_step], source="train")
        if not screenshot_path or not os.path.exists(screenshot_path):
            continue

        fail_action = normalize_action(f_steps[div_step])
        success_action = normalize_action(s_steps[div_step])

        # Skip empty/terminate GT actions (not evaluable)
        if not success_action["action_type"] or success_action["action_type"] in ("", "FINISH", "terminate"):
            continue
        # Skip if GT has no coordinate (can't evaluate coordinate accuracy)
        if success_action["coordinate"] is None and success_action["action_type"] in ("click", "type", "drag"):
            continue

        # Skip if both actions are identical (not a real divergence)
        if fail_action["action_type"] == success_action["action_type"]:
            dist = coordinate_distance(fail_action["coordinate"], success_action["coordinate"])
            if dist < 20:
                continue

        examples.append({
            "request": pair["request"],
            "execution_id": pair["execution_id"],
            "domain": pair["domain"],
            "div_step": div_step,
            "total_success_steps": len(s_steps),
            "screenshot_path": screenshot_path,
            "fail_action": fail_action,
            "success_action": success_action,
            "history_steps": f_steps[:div_step],  # Steps before divergence
        })

    print(f"Built {len(examples)} retry examples")
    return examples


def run_retry_experiment(args):
    """Run the retry-with-context experiment."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = build_retry_examples(args.num_examples)
    if not examples:
        print("ERROR: No retry examples could be built!")
        return

    from openai import OpenAI

    client = OpenAI(api_key="EMPTY", base_url=args.endpoint, timeout=300)

    results_path = output_dir / "results.jsonl"
    if results_path.exists():
        results_path.unlink()

    for i, ex in enumerate(examples):
        try:
            img_url = image_to_data_url(ex["screenshot_path"])
        except Exception as e:
            print(f"Error loading image for example {i}: {e}")
            continue

        history_text = build_history_text(ex["history_steps"], ex["div_step"])
        fail_desc = describe_action(ex["fail_action"])

        # === Condition 1: Original prompt (no error context) ===
        original_prompt = ORIGINAL_PROMPT_TEMPLATE.format(
            task=ex["request"],
            history_text=history_text,
        )
        original_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": original_prompt},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ],
            }
        ]

        try:
            resp = client.chat.completions.create(
                model=args.model_name,
                messages=original_messages,
                temperature=0.0,
            )
            original_response = resp.choices[0].message.content
        except Exception as e:
            print(f"API error (original) for example {i}: {e}")
            original_response = ""

        # === Condition 2: Retry with simple error context ===
        retry_prompt = RETRY_PROMPT_TEMPLATE.format(
            task=ex["request"],
            history_text=history_text,
            failed_action_description=fail_desc,
        )
        retry_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": retry_prompt},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ],
            }
        ]

        try:
            resp = client.chat.completions.create(
                model=args.model_name,
                messages=retry_messages,
                temperature=0.0,
            )
            retry_response = resp.choices[0].message.content
        except Exception as e:
            print(f"API error (retry) for example {i}: {e}")
            retry_response = ""

        # === Condition 3: Retry with detailed reasoning context ===
        reasoning_prompt = RETRY_WITH_REASONING_TEMPLATE.format(
            task=ex["request"],
            history_text=history_text,
            failed_action_description=fail_desc,
        )
        reasoning_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": reasoning_prompt},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ],
            }
        ]

        try:
            resp = client.chat.completions.create(
                model=args.model_name,
                messages=reasoning_messages,
                temperature=0.0,
            )
            reasoning_response = resp.choices[0].message.content
        except Exception as e:
            print(f"API error (reasoning) for example {i}: {e}")
            reasoning_response = ""

        # Evaluate all conditions against ground truth
        gt = ex["success_action"]
        original_eval = evaluate_prediction(original_response, gt)
        retry_eval = evaluate_prediction(retry_response, gt)
        reasoning_eval = evaluate_prediction(reasoning_response, gt)

        result = {
            "idx": i,
            "execution_id": ex["execution_id"],
            "domain": ex["domain"],
            "div_step": ex["div_step"],
            "total_success_steps": ex["total_success_steps"],
            "fail_action_type": ex["fail_action"]["action_type"],
            "gt_action_type": gt["action_type"],
            "gt_coordinate": gt["coordinate"],
            # Original prediction (no context)
            "original_func_match": original_eval["function_match"],
            "original_coord_match": original_eval["coord_match"],
            "original_full_match": original_eval["full_match"],
            "original_distance": original_eval["coord_distance"],
            "original_response": original_response[:300],
            # Simple retry
            "retry_func_match": retry_eval["function_match"],
            "retry_coord_match": retry_eval["coord_match"],
            "retry_full_match": retry_eval["full_match"],
            "retry_distance": retry_eval["coord_distance"],
            "retry_response": retry_response[:300],
            # Reasoning retry
            "reasoning_func_match": reasoning_eval["function_match"],
            "reasoning_coord_match": reasoning_eval["coord_match"],
            "reasoning_full_match": reasoning_eval["full_match"],
            "reasoning_distance": reasoning_eval["coord_distance"],
            "reasoning_response": reasoning_response[:300],
        }

        with open(results_path, "a") as f:
            f.write(json.dumps(result, default=str) + "\n")

        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(examples)}")

    print(f"Results saved to: {results_path}")
    return results_path


def analyze_results(results_path: str):
    """Analyze retry experiment results."""
    results = []
    with open(results_path) as f:
        for line in f:
            results.append(json.loads(line))

    if not results:
        print("No results to analyze!")
        return

    n = len(results)
    print("\n" + "=" * 60)
    print(f"  Experiment 0.5: Retry with Error Context (N={n})")
    print("=" * 60)

    # 1. Overall accuracy comparison
    conditions = [
        ("Original (no context)", "original"),
        ("Retry (simple)", "retry"),
        ("Retry (reasoning)", "reasoning"),
    ]

    print(f"\n  Full Match Accuracy:")
    for label, prefix in conditions:
        correct = sum(1 for r in results if r.get(f"{prefix}_full_match", False))
        print(f"    {label:30s}: {correct}/{n} ({correct / n:.1%})")

    print(f"\n  Function Match Accuracy:")
    for label, prefix in conditions:
        correct = sum(1 for r in results if r.get(f"{prefix}_func_match", False))
        print(f"    {label:30s}: {correct}/{n} ({correct / n:.1%})")

    print(f"\n  Coordinate Match Accuracy (given correct function):")
    for label, prefix in conditions:
        func_correct = [r for r in results if r.get(f"{prefix}_func_match", False)]
        if func_correct:
            coord_correct = sum(1 for r in func_correct if r.get(f"{prefix}_coord_match", False))
            print(f"    {label:30s}: {coord_correct}/{len(func_correct)} ({coord_correct / len(func_correct):.1%})")

    # 2. Distance improvement
    print(f"\n  Mean Coordinate Distance (pixels, lower is better):")
    for label, prefix in conditions:
        dists = [r[f"{prefix}_distance"] for r in results
                 if r.get(f"{prefix}_distance", float("inf")) < float("inf")]
        if dists:
            print(f"    {label:30s}: mean={np.mean(dists):.1f}  median={np.median(dists):.1f}")

    # 3. Pairwise comparison: did retry improve over original?
    print(f"\n  Pairwise Comparison (retry vs original):")
    for label, prefix in [("Simple retry", "retry"), ("Reasoning retry", "reasoning")]:
        improved = sum(1 for r in results
                       if r.get(f"{prefix}_full_match") and not r.get("original_full_match"))
        regressed = sum(1 for r in results
                        if not r.get(f"{prefix}_full_match") and r.get("original_full_match"))
        same_correct = sum(1 for r in results
                          if r.get(f"{prefix}_full_match") and r.get("original_full_match"))
        same_wrong = sum(1 for r in results
                        if not r.get(f"{prefix}_full_match") and not r.get("original_full_match"))

        print(f"\n    {label}:")
        print(f"      Improved (wrong→right): {improved}/{n} ({improved / n:.1%})")
        print(f"      Regressed (right→wrong): {regressed}/{n} ({regressed / n:.1%})")
        print(f"      Same correct:           {same_correct}/{n}")
        print(f"      Same wrong:             {same_wrong}/{n}")
        net = improved - regressed
        print(f"      Net improvement:        {'+' if net >= 0 else ''}{net} samples")

    # 4. Distance improvement for retry
    print(f"\n  Coordinate Distance Improvement:")
    for label, prefix in [("Simple retry", "retry"), ("Reasoning retry", "reasoning")]:
        closer = 0
        farther = 0
        dist_improvements = []
        for r in results:
            orig_d = r.get("original_distance", float("inf"))
            retry_d = r.get(f"{prefix}_distance", float("inf"))
            if orig_d < float("inf") and retry_d < float("inf"):
                if retry_d < orig_d:
                    closer += 1
                    dist_improvements.append(orig_d - retry_d)
                elif retry_d > orig_d:
                    farther += 1

        print(f"\n    {label}:")
        print(f"      Got closer: {closer}/{n}")
        print(f"      Got farther: {farther}/{n}")
        if dist_improvements:
            print(f"      Mean improvement (when closer): {np.mean(dist_improvements):.1f}px")

    # 5. Per-domain breakdown
    print(f"\n  Per-Domain Full Match (retry simple):")
    domains = set(r["domain"] for r in results)
    for domain in sorted(domains):
        domain_results = [r for r in results if r["domain"] == domain]
        if domain_results:
            orig_correct = sum(1 for r in domain_results if r.get("original_full_match"))
            retry_correct = sum(1 for r in domain_results if r.get("retry_full_match"))
            nd = len(domain_results)
            print(f"    {domain:10s}: original={orig_correct}/{nd} ({orig_correct / nd:.1%})  "
                  f"retry={retry_correct}/{nd} ({retry_correct / nd:.1%})")

    # 6. By action type
    print(f"\n  Per-Action-Type (retry simple):")
    action_types = set(r.get("gt_action_type", "") for r in results)
    for atype in sorted(action_types):
        type_results = [r for r in results if r.get("gt_action_type") == atype]
        if type_results:
            orig = sum(1 for r in type_results if r.get("original_full_match"))
            retry = sum(1 for r in type_results if r.get("retry_full_match"))
            nt = len(type_results)
            print(f"    {atype:20s}: original={orig}/{nt} ({orig / nt:.1%})  "
                  f"retry={retry}/{nt} ({retry / nt:.1%})")

    # Verdict
    original_acc = sum(1 for r in results if r.get("original_full_match")) / n
    retry_acc = sum(1 for r in results if r.get("retry_full_match")) / n
    reasoning_acc = sum(1 for r in results if r.get("reasoning_full_match")) / n
    best_retry = max(retry_acc, reasoning_acc)

    print(f"\n  VERDICT: {'PASS' if best_retry > original_acc else 'NO IMPROVEMENT'}")
    if best_retry > original_acc:
        improvement = best_retry - original_acc
        print(f"    Retry improves accuracy by {improvement:.1%} ({original_acc:.1%} → {best_retry:.1%})")
        print(f"    Error context helps the model correct its predictions!")
    else:
        print(f"    Original: {original_acc:.1%}, Retry: {retry_acc:.1%}, Reasoning: {reasoning_acc:.1%}")
        print(f"    Error context doesn't help. May need verification-specific training.")

    print("=" * 60)

    # Save summary
    summary = {
        "n_examples": n,
        "original_accuracy": original_acc,
        "retry_simple_accuracy": retry_acc,
        "retry_reasoning_accuracy": reasoning_acc,
        "improvement_simple": retry_acc - original_acc,
        "improvement_reasoning": reasoning_acc - original_acc,
    }
    summary_path = Path(results_path).parent / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 0.5: Retry with Error Context")
    parser.add_argument("--model_name", type=str, default="gui360_lora_v4_ckpt354")
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "exp0_5"))
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--results_path", type=str, default="")

    args = parser.parse_args()

    if args.analyze_only:
        path = args.results_path or str(Path(args.output_dir) / "results.jsonl")
        analyze_results(path)
    else:
        results_path = run_retry_experiment(args)
        if results_path:
            analyze_results(str(results_path))


if __name__ == "__main__":
    main()
