#!/usr/bin/env python3
"""
Eval D9: Critic Zero-Shot Performance

Tests V2's ability to judge whether a trajectory is on track or has deviated.

For each step in test trajectories:
  Input: screenshot + task goal + step number
  Output: PASS (on track) / FAIL (deviated)
  Ground truth: all steps up to and including this one are correct

This tests the foundation for a Critic agent. If precision ~100% (like Exp 0.2 verifier),
the model has good global task understanding → Critic SFT has a strong starting point.

Usage:
    python scripts/eval/eval_d9_critic_zeroshot.py \
        --v2_endpoint http://localhost:19816/v1
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "train_GUI_360" / "GUI-360-eval"))

from scripts.exp1.grounding_utils import preprocess_image


CRITIC_PROMPT = """You are a Critic agent evaluating whether a GUI automation task is progressing correctly.

Task: {instruction}
Current step: {step_num} of approximately {total_steps} steps

Look at the current screenshot and assess:
1. Does the current screen state look consistent with making progress on the task?
2. Are we in the right application/dialog/menu for this stage of the task?
3. Are there any signs of error (wrong dialog, unexpected state, obvious mistakes)?

Respond with exactly one word:
- PASS if the task appears to be progressing correctly
- FAIL if something appears wrong or the task has deviated

Your response (PASS or FAIL):"""

CRITIC_WITH_CONTEXT_PROMPT = """You are a Critic agent evaluating whether a GUI automation task is progressing correctly.

Task: {instruction}
Current step: {step_num} of approximately {total_steps} steps

The last action taken was: {last_action}

Look at the current screenshot and assess whether the last action achieved its intended effect and the task is still on track.

Respond with exactly one word:
- PASS if the task appears to be progressing correctly
- FAIL if something appears wrong or the task has deviated

Your response (PASS or FAIL):"""


def load_trajectories(dataset_root, max_trajectories=0):
    data_path = os.path.join(dataset_root, "data")
    count = 0
    for domain in sorted(os.listdir(data_path)):
        domain_path = os.path.join(data_path, domain)
        if not os.path.isdir(domain_path):
            continue
        for category in sorted(os.listdir(domain_path)):
            success_path = os.path.join(domain_path, category, "success")
            if not os.path.isdir(success_path):
                continue
            for jsonl_file in sorted(os.listdir(success_path)):
                if not jsonl_file.endswith(".jsonl"):
                    continue
                file_path = os.path.join(success_path, jsonl_file)
                file_stem = os.path.splitext(jsonl_file)[0]
                trajectory_id = f"{domain}_{category}_{file_stem}"
                try:
                    steps = []
                    with open(file_path, "r") as f:
                        for line_num, line in enumerate(f, 1):
                            if not line.strip():
                                continue
                            data = json.loads(line.strip())
                            if "action_prediction" not in data["step"].get("tags", []):
                                continue
                            clean_img = os.path.join(
                                dataset_root, "image", domain, category,
                                data["step"]["screenshot_clean"],
                            )
                            if not os.path.exists(clean_img):
                                continue
                            status = data["step"]["status"]
                            if status == "OVERALL_FINISH":
                                status = "FINISH"
                            elif status == "FINISH":
                                status = "CONTINUE"
                            action = data["step"]["action"]
                            if action.get("function", "") == "drag" or not action.get("rectangle", {}):
                                continue
                            steps.append({
                                "line_num": line_num,
                                "request": data["request"],
                                "screenshot_clean": clean_img,
                                "thought": data["step"]["thought"],
                                "action": action, "status": status,
                                "domain": domain, "category": category,
                            })
                    if steps:
                        count += 1
                        yield {
                            "trajectory_id": trajectory_id,
                            "request": steps[0]["request"],
                            "domain": domain, "category": category,
                            "steps": steps,
                        }
                        if max_trajectories > 0 and count >= max_trajectories:
                            return
                except Exception:
                    continue


def _call_v2(client, model, messages, max_tokens=32):
    for retry in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.0, max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception:
            if retry == 2:
                return ""
            time.sleep(2)
    return ""


def parse_critic_response(response):
    """Parse PASS/FAIL from Critic response."""
    text = response.strip().upper()
    if "FAIL" in text:
        return "FAIL"
    elif "PASS" in text:
        return "PASS"
    return "UNKNOWN"


def evaluate_trajectory_critic(trajectory, v2_client, v2_model, eval_a_result):
    """Run Critic on each step. Ground truth from Eval A step results."""
    traj_id = trajectory["trajectory_id"]
    steps = trajectory["steps"]
    total_steps = len(steps)

    # Get ground truth from Eval A
    eval_a_steps = eval_a_result.get("step_results", [])

    step_critic_results = []

    for step_idx, step in enumerate(steps):
        step_num = step_idx + 1
        clean_img = step["screenshot_clean"]

        # Ground truth: is the trajectory still on track at this step?
        # On track = all steps up to this one are correct (condition B from Eval A)
        gt_on_track = True
        if step_idx < len(eval_a_steps):
            for s in eval_a_steps[:step_idx + 1]:
                if not s.get("b_success", False):
                    gt_on_track = False
                    break

        # Prepare Critic input
        data_url, _, _ = preprocess_image(clean_img)

        if step_idx == 0:
            prompt = CRITIC_PROMPT.format(
                instruction=step["request"],
                step_num=step_num,
                total_steps=total_steps,
            )
        else:
            # Include last action for context
            prev_action = steps[step_idx - 1]["action"]
            last_action_desc = prev_action.get("function", "unknown")
            prompt = CRITIC_WITH_CONTEXT_PROMPT.format(
                instruction=step["request"],
                step_num=step_num,
                total_steps=total_steps,
                last_action=last_action_desc,
            )

        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt},
        ]}]

        response = _call_v2(v2_client, v2_model, messages, max_tokens=32)
        prediction = parse_critic_response(response)

        step_critic_results.append({
            "step_num": step_num,
            "gt_on_track": gt_on_track,
            "critic_prediction": prediction,
            "raw_response": response[:100],
        })

    return {
        "trajectory_id": traj_id,
        "domain": trajectory["domain"],
        "category": trajectory["category"],
        "num_steps": total_steps,
        "step_results": step_critic_results,
    }


def run_evaluation(args):
    from openai import OpenAI

    output_dir = PROJECT_ROOT / "outputs" / "eval_d9"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "trajectory_results.jsonl"

    # Load Eval A results for ground truth
    eval_a = {}
    with open(PROJECT_ROOT / "outputs" / "eval_a" / "trajectory_results.jsonl") as f:
        for line in f:
            d = json.loads(line)
            eval_a[d["trajectory_id"]] = d

    completed = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                d = json.loads(line)
                completed.add(d["trajectory_id"])
        print(f"Resuming: {len(completed)} already complete")

    dataset_root = str(PROJECT_ROOT / "datasets" / "GUI-360" / "test")
    trajectories = list(load_trajectories(dataset_root, max_trajectories=args.max_trajectories))
    # Only evaluate trajectories we have Eval A results for
    trajectories = [t for t in trajectories if t["trajectory_id"] in eval_a]
    remaining = [t for t in trajectories if t["trajectory_id"] not in completed]
    print(f"Loaded {len(trajectories)} total (with Eval A), {len(remaining)} remaining")

    v2_client = OpenAI(base_url=args.v2_endpoint, api_key="none")
    v2_model = v2_client.models.list().data[0].id
    print(f"V2: {v2_model}")

    n_done = len(completed)
    n_total = len(trajectories)
    t_start = time.time()

    def _eval_one(traj):
        return evaluate_trajectory_critic(traj, v2_client, v2_model, eval_a[traj["trajectory_id"]])

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(_eval_one, t): t for t in remaining}
        for future in as_completed(futures):
            traj = futures[future]
            n_done += 1
            try:
                result = future.result()
                with open(results_path, "a") as f:
                    f.write(json.dumps(result) + "\n")
                elapsed = time.time() - t_start
                rate = (n_done - len(completed)) / elapsed * 3600
                # Quick summary
                steps = result["step_results"]
                n_pass = sum(1 for s in steps if s["critic_prediction"] == "PASS")
                print(f"[{n_done}/{n_total}] {traj['trajectory_id']} "
                      f"PASS={n_pass}/{len(steps)} [{rate:.0f}/hr]")
            except Exception as e:
                print(f"[{n_done}/{n_total}] {traj['trajectory_id']} ERROR: {e}")

    print(f"\nDone. Results at {results_path}")


def analyze_results():
    results_path = PROJECT_ROOT / "outputs" / "eval_d9" / "trajectory_results.jsonl"
    results = []
    with open(results_path) as f:
        for line in f:
            results.append(json.loads(line))

    n = len(results)
    print(f"\n{'='*70}")
    print(f"  Eval D9: Critic Zero-Shot Performance ({n} trajectories)")
    print(f"{'='*70}")

    # Aggregate all step-level predictions
    all_steps = []
    for r in results:
        for s in r["step_results"]:
            all_steps.append({
                "domain": r["domain"],
                "num_steps": r["num_steps"],
                **s,
            })

    total = len(all_steps)
    valid = [s for s in all_steps if s["critic_prediction"] != "UNKNOWN"]
    unknown = total - len(valid)

    print(f"\n  Total step evaluations: {total}")
    print(f"  Valid predictions: {len(valid)} ({len(valid)/total*100:.1f}%)")
    print(f"  Unknown/unparseable: {unknown} ({unknown/total*100:.1f}%)")

    if not valid:
        print("  No valid predictions to analyze.")
        return

    # Confusion matrix
    tp = sum(1 for s in valid if s["gt_on_track"] and s["critic_prediction"] == "PASS")
    fp = sum(1 for s in valid if not s["gt_on_track"] and s["critic_prediction"] == "PASS")
    tn = sum(1 for s in valid if not s["gt_on_track"] and s["critic_prediction"] == "FAIL")
    fn = sum(1 for s in valid if s["gt_on_track"] and s["critic_prediction"] == "FAIL")

    print(f"\n  Confusion Matrix (Critic → PASS/FAIL vs Ground Truth on-track/deviated):")
    print(f"                        GT: On-track    GT: Deviated")
    print(f"  Critic: PASS          {tp:>8d} (TP)   {fp:>8d} (FP)")
    print(f"  Critic: FAIL          {fn:>8d} (FN)   {tn:>8d} (TN)")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(valid) if valid else 0

    # For FAIL detection (the critical metric)
    fail_precision = tn / (tn + fn) if (tn + fn) > 0 else 0  # When Critic says FAIL, how often correct?
    fail_recall = tn / (tn + fp) if (tn + fp) > 0 else 0  # Of actual deviations, how many caught?

    print(f"\n  PASS prediction metrics:")
    print(f"    Precision (PASS=on-track):   {precision:.1%}")
    print(f"    Recall (PASS=on-track):      {recall:.1%}")
    print(f"    F1:                          {f1:.1%}")

    print(f"\n  FAIL detection metrics (critical for Critic role):")
    print(f"    Precision (FAIL=deviated):   {fail_precision:.1%}")
    print(f"    Recall (FAIL=deviated):      {fail_recall:.1%}")
    print(f"    Accuracy:                    {accuracy:.1%}")

    # Bias analysis
    total_pass = sum(1 for s in valid if s["critic_prediction"] == "PASS")
    total_fail = sum(1 for s in valid if s["critic_prediction"] == "FAIL")
    gt_on_track = sum(1 for s in valid if s["gt_on_track"])
    gt_deviated = sum(1 for s in valid if not s["gt_on_track"])
    print(f"\n  Bias analysis:")
    print(f"    Critic says PASS: {total_pass}/{len(valid)} = {total_pass/len(valid)*100:.1f}%")
    print(f"    Critic says FAIL: {total_fail}/{len(valid)} = {total_fail/len(valid)*100:.1f}%")
    print(f"    GT on-track:      {gt_on_track}/{len(valid)} = {gt_on_track/len(valid)*100:.1f}%")
    print(f"    GT deviated:      {gt_deviated}/{len(valid)} = {gt_deviated/len(valid)*100:.1f}%")

    # Per-domain
    print(f"\n  Per-domain FAIL detection:")
    by_domain = defaultdict(list)
    for s in valid:
        by_domain[s["domain"]].append(s)

    for domain in sorted(by_domain.keys()):
        subset = by_domain[domain]
        d_tn = sum(1 for s in subset if not s["gt_on_track"] and s["critic_prediction"] == "FAIL")
        d_fp = sum(1 for s in subset if not s["gt_on_track"] and s["critic_prediction"] == "PASS")
        d_fn = sum(1 for s in subset if s["gt_on_track"] and s["critic_prediction"] == "FAIL")
        d_tp = sum(1 for s in subset if s["gt_on_track"] and s["critic_prediction"] == "PASS")
        d_prec = d_tn / (d_tn + d_fn) if (d_tn + d_fn) > 0 else 0
        d_rec = d_tn / (d_tn + d_fp) if (d_tn + d_fp) > 0 else 0
        print(f"    {domain}: precision={d_prec:.1%}, recall={d_rec:.1%} "
              f"(TP={d_tp}, FP={d_fp}, TN={d_tn}, FN={d_fn})")

    # By step position
    print(f"\n  FAIL detection recall by step:")
    by_step = defaultdict(list)
    for s in valid:
        if not s["gt_on_track"]:  # Only deviated steps
            by_step[min(s["step_num"], 10)].append(s)
    for step in sorted(by_step.keys()):
        subset = by_step[step]
        caught = sum(1 for s in subset if s["critic_prediction"] == "FAIL")
        label = f"Step {step}" if step < 10 else "Step 10+"
        print(f"    {label}: {caught}/{len(subset)} = {caught/len(subset)*100:.0f}%")

    # Save
    output_dir = PROJECT_ROOT / "outputs" / "eval_d9"
    summary = {
        "n_trajectories": n,
        "n_steps": total,
        "pass_precision": float(precision),
        "pass_recall": float(recall),
        "fail_precision": float(fail_precision),
        "fail_recall": float(fail_recall),
        "accuracy": float(accuracy),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {output_dir / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2_endpoint", default="http://localhost:19816/v1")
    parser.add_argument("--max_trajectories", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--analyze_only", action="store_true")
    args = parser.parse_args()

    if args.analyze_only:
        analyze_results()
    else:
        run_evaluation(args)
        analyze_results()


if __name__ == "__main__":
    main()
