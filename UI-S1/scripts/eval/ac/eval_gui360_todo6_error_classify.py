"""GUI-360 TODO 6: LLM-Based Error Classification for Desktop Tasks.

Uses VLM to classify each failure step from GUI-360 multisample results into
fine-grained error categories.

Error categories (adapted for desktop):
  1. PLANNING_WRONG_FUNCTION: Chose entirely wrong function type
  2. PLANNING_WRONG_TARGET: Right function but wrong UI element
  3. GROUNDING_NEAR_MISS: Right function, nearby but outside target rectangle
  4. TIMING_ERROR: Wrong temporal action (premature finish, missed wait)
  5. GOAL_MISUNDERSTANDING: Misunderstood the task goal
  6. STATE_CONFUSION: Confused about current screen/app state
  7. APP_NAVIGATION_ERROR: Failed to navigate to correct menu/panel
  8. OTHER: Doesn't fit above
"""

import argparse
import json
import os
import sys
import time
import traceback
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from io import BytesIO

PROJECT_DIR = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
GUI360_EVAL_DIR = os.path.join(PROJECT_DIR, "train_GUI_360", "GUI-360-eval")
sys.path.insert(0, GUI360_EVAL_DIR)

from openai import OpenAI
from PIL import Image
import re

result_lock = Lock()

ERROR_CATEGORIES = [
    "PLANNING_WRONG_FUNCTION", "PLANNING_WRONG_TARGET", "GROUNDING_NEAR_MISS",
    "TIMING_ERROR", "GOAL_MISUNDERSTANDING", "STATE_CONFUSION",
    "APP_NAVIGATION_ERROR", "OTHER",
]

CLASSIFY_PROMPT = """You are an expert evaluator for a desktop GUI automation agent working on {domain} tasks. The agent made an error at this step.

Task Goal: {goal}
Step: {step_num} of {total_steps}

Ground Truth Action: function={gt_function}, args={gt_args}
Predicted Action: function={pred_function}, args={pred_args}

Classify this error into ONE category:
1. PLANNING_WRONG_FUNCTION: Wrong function type (e.g., type instead of click)
2. PLANNING_WRONG_TARGET: Right function but wrong UI element/area
3. GROUNDING_NEAR_MISS: Right function, right general area, coordinates slightly off
4. TIMING_ERROR: Wrong temporal action (premature finish, etc.)
5. GOAL_MISUNDERSTANDING: Misunderstood what the task requires
6. STATE_CONFUSION: Confused about current screen state
7. APP_NAVIGATION_ERROR: Failed to navigate to correct menu/panel/tab
8. OTHER: Doesn't fit above

Output ONLY JSON: {{"category": "<category>", "reasoning": "<brief explanation>"}}"""


def classify_error(screenshot_path, domain, goal, step_num, total_steps,
                   gt_function, gt_args, pred_function, pred_args, api_url, model_name):
    img = Image.open(screenshot_path)
    buf = BytesIO()
    img.save(buf, format="PNG")
    data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    prompt = CLASSIFY_PROMPT.format(
        domain=domain, goal=goal[:300], step_num=step_num + 1, total_steps=total_steps,
        gt_function=gt_function, gt_args=json.dumps(gt_args)[:200],
        pred_function=pred_function, pred_args=json.dumps(pred_args)[:200],
    )

    messages = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_url}},
        {"type": "text", "text": prompt},
    ]}]

    for attempt in range(3):
        try:
            bot = OpenAI(api_key="EMPTY", base_url=api_url, timeout=120)
            resp = bot.chat.completions.create(model=model_name, messages=messages,
                                                extra_body={"top_k": 1}, max_tokens=512)
            return _parse_classification(resp.choices[0].message.content)
        except Exception:
            if attempt < 2:
                time.sleep(2)
    return "OTHER", "classification_failed"


def _parse_classification(text):
    try:
        parsed = json.loads(text)
        cat = parsed.get("category", "OTHER").upper().strip()
        if cat in ERROR_CATEGORIES:
            return cat, parsed.get("reasoning", "")
    except (json.JSONDecodeError, AttributeError):
        pass
    match = re.search(r'"category"\s*:\s*"([^"]+)"', text)
    if match:
        cat = match.group(1).upper().strip()
        if cat in ERROR_CATEGORIES:
            reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
            return cat, reason_match.group(1) if reason_match else ""
    for cat in ERROR_CATEGORIES:
        if cat in text.upper():
            return cat, text[:200]
    return "OTHER", text[:200]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multisample_results", type=str,
                        default=os.path.join(PROJECT_DIR, "outputs/eval_gui360_multisample/multisample_results.jsonl"))
    parser.add_argument("--output_dir", type=str, default=os.path.join(PROJECT_DIR, "outputs/eval_gui360_todo6"))
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--api_url", type=str, default="http://localhost:19815/v1")
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "error_classifications.jsonl")
    if os.path.exists(out_path):
        os.remove(out_path)

    # Load multisample results, filter to greedy failures
    failures = []
    with open(args.multisample_results) as f:
        for line in f:
            r = json.loads(line.strip())
            if not r.get("greedy_function_match", False):
                failures.append(r)

    if args.max_samples:
        failures = failures[:args.max_samples]
    print(f"Loaded {len(failures)} greedy failure steps for classification")

    # Find screenshot paths
    root_dir = os.path.join(PROJECT_DIR, "datasets/GUI-360/test")

    classified = []
    completed = 0

    def process_one(r):
        nonlocal completed
        sample_id = r["sample_id"]
        # Reconstruct screenshot path from sample_id
        # sample_id format: domain_category_trajectory_linenum
        parts = sample_id.split("_")
        domain = parts[0]
        # Try to find the screenshot from the data
        # We need the actual screenshot path - reconstruct from sample data
        # The greedy_pred_function is available
        gt_function = r.get("gt_function", "")
        pred_function = r.get("greedy_pred_function", "")

        # We need to find the screenshot. The sample_id encodes it.
        # Reconstruct the path from trajectory_id + step_index
        traj_id = r.get("trajectory_id", "")
        step_idx = r.get("step_index", 0)

        # The screenshot path would be in the dataset
        # Since we don't have it directly, we'll skip classification for samples without screenshots
        # Instead, do text-only classification
        category, reasoning = "OTHER", "screenshot_unavailable"

        # Try to find screenshot in dataset
        # trajectory_id format: domain_category_filename
        traj_parts = traj_id.split("_", 2)
        if len(traj_parts) >= 3:
            t_domain = traj_parts[0]
            t_category = traj_parts[1]
            t_filename = traj_parts[2]

            # Load the trajectory file to get screenshot path
            data_path = os.path.join(root_dir, "data", t_domain, t_category, "success", f"{t_filename}.jsonl")
            if os.path.exists(data_path):
                with open(data_path) as tf:
                    steps = []
                    for tl in tf:
                        if tl.strip():
                            try:
                                steps.append(json.loads(tl.strip()))
                            except json.JSONDecodeError:
                                continue

                    # Filter to action_prediction tagged steps
                    ap_steps = [s for s in steps if "action_prediction" in s.get("step", {}).get("tags", [])
                                and s.get("step", {}).get("action", {}).get("function") != "drag"
                                and s.get("step", {}).get("action", {}).get("rectangle")]

                    if step_idx < len(ap_steps):
                        step_data = ap_steps[step_idx]
                        screenshot_rel = step_data["step"]["screenshot_clean"]
                        screenshot_path = os.path.join(root_dir, "image", t_domain, t_category, screenshot_rel)

                        if os.path.exists(screenshot_path):
                            gt_args = dict(step_data["step"]["action"].get("args", {}))
                            pred_args = {}  # We don't have full pred_args in multisample results
                            # Get K_predictions for the greedy sample
                            k_preds = r.get("K_predictions", [])
                            if k_preds:
                                # First prediction is greedy
                                pred_args = {}  # Not stored in detail

                            category, reasoning = classify_error(
                                screenshot_path, t_domain,
                                r.get("sample_id", ""),  # use sample_id as proxy for goal
                                step_idx, r.get("total_steps", 0),
                                gt_function, gt_args,
                                pred_function, pred_args,
                                args.api_url, args.model_name,
                            )

        result = {
            "sample_id": sample_id,
            "trajectory_id": traj_id,
            "domain": domain,
            "step_index": step_idx,
            "gt_function": gt_function,
            "pred_function": pred_function,
            "category": category,
            "reasoning": reasoning[:300],
        }

        with result_lock:
            with open(out_path, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        completed += 1
        if completed % 200 == 0:
            print(f"Progress: {completed}/{len(failures)}")

        return result

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_one, r): r for r in failures}
        for future in as_completed(futures):
            try:
                result = future.result()
                classified.append(result)
            except Exception:
                traceback.print_exc()

    # Analysis
    print(f"\n{'='*70}")
    print("GUI-360 TODO 6: Error Classification Results")
    print(f"{'='*70}")
    print(f"Total classified: {len(classified)}")

    cat_counts = Counter(c["category"] for c in classified)
    for cat, count in cat_counts.most_common():
        print(f"  {cat:30s}: {count:5d} ({count/len(classified)*100:.1f}%)")

    # By domain
    print("\n--- By Domain ---")
    by_domain = defaultdict(lambda: Counter())
    for c in classified:
        by_domain[c["domain"]][c["category"]] += 1
    for domain in sorted(by_domain):
        total = sum(by_domain[domain].values())
        print(f"\n  {domain} (N={total}):")
        for cat, count in by_domain[domain].most_common(5):
            print(f"    {cat}: {count} ({count/total*100:.1f}%)")

    # By GT function
    print("\n--- By GT Function ---")
    by_func = defaultdict(lambda: Counter())
    for c in classified:
        by_func[c["gt_function"]][c["category"]] += 1
    for func in sorted(by_func, key=lambda x: -sum(by_func[x].values())):
        total = sum(by_func[func].values())
        print(f"\n  {func} (N={total}):")
        for cat, count in by_func[func].most_common(3):
            print(f"    {cat}: {count} ({count/total*100:.1f}%)")

    summary = {
        "total": len(classified),
        "categories": dict(cat_counts),
    }
    with open(os.path.join(args.output_dir, "error_classification_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
