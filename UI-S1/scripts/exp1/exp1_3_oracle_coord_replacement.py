#!/usr/bin/env python3
"""
Experiment 1.3: Oracle Coordinate Replacement ★ (Go/No-Go Gate)

Core question: If coordinates were perfectly correct, how much would action
prediction improve?

Method:
  Run SFT v2 action prediction, then:
    A. SFT v2 original prediction (baseline)
    B. SFT v2 action type + SFT v3 predicted coordinate
    C. SFT v2 action type + ground truth coordinate (oracle)

  Evaluate: function_match, args_match, step_success

Success criteria:
  - Oracle (C) args_match > baseline (A) by ≥10pp → coord IS the bottleneck
  - SFT v3 coord (B) > baseline (A) → SFT v3 grounding helps in practice

This is the go/no-go gate for the entire multi-agent framework.

Usage:
    # Needs two vLLM servers running:
    #   SFT v2 on port 19816, SFT v3 on port 19815
    python scripts/exp1/exp1_3_oracle_coord_replacement.py \
        --sft_v2_endpoint http://localhost:19816/v1 \
        --sft_v3_endpoint http://localhost:19815/v1

    # Analysis only:
    python scripts/exp1/exp1_3_oracle_coord_replacement.py \
        --analyze_only --results_dir outputs/exp1_3
"""

import argparse
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.exp0.exp0_1_uncertainty_analysis import (
    _json_default,
    extract_coordinate,
    parse_tool_call,
)
from scripts.exp0.data_utils import (
    DATASET_ROOT,
    is_coord_in_bbox,
    load_trajectory,
)
from scripts.exp1.grounding_utils import (
    preprocess_image,
    parse_coordinate_response,
    transform_coord_to_original,
    GROUNDING_PROMPT,
)


def load_action_prediction_samples(dataset_root: str, max_samples: int = 0):
    """Load action prediction samples from GUI-360 test set.

    Walks the test data directory structure and loads samples with their
    screenshots, instructions, previous actions, and ground truth.
    """
    test_root = Path(dataset_root) / "test"
    data_path = test_root / "data"
    image_path = test_root / "image"

    samples = []
    for domain in sorted(data_path.iterdir()):
        if not domain.is_dir():
            continue
        for category in sorted(domain.iterdir()):
            if not category.is_dir():
                continue
            success_dir = category / "success"
            if not success_dir.exists():
                continue

            for jsonl_file in sorted(success_dir.glob("*.jsonl")):
                steps = load_trajectory(jsonl_file)
                for i, step_data in enumerate(steps):
                    step = step_data.get("step", {})
                    action = step.get("action", {})
                    tags = step.get("tags", [])

                    # Only visual action prediction samples
                    if "action_prediction" not in tags:
                        continue

                    # Skip drag for simplicity (different coordinate format)
                    if action.get("function") == "drag":
                        continue

                    # Need rectangle for proper evaluation
                    if not action.get("rectangle"):
                        continue

                    screenshot = image_path / domain.name / category.name / step.get("screenshot_clean", "")
                    if not screenshot.exists():
                        continue

                    # Build previous actions context
                    prev_actions = []
                    for j in range(i):
                        prev_step = steps[j].get("step", {})
                        prev_actions.append(f"Step {j+1}: {prev_step.get('thought', '')}")

                    # Normalize args: convert x/y to coordinate list
                    args = dict(action.get("args", {}))
                    if "x" in args and "y" in args and "coordinate" not in args:
                        args["coordinate"] = [args["x"], args["y"]]
                    if "start_x" in args:
                        args["start_coordinate"] = [args.pop("start_x"), args.pop("start_y")]
                        args["end_coordinate"] = [args.pop("end_x"), args.pop("end_y")]

                    sample = {
                        "sample_id": f"{domain.name}_{category.name}_{jsonl_file.stem}_{i+1}",
                        "request": step_data.get("request", ""),
                        "screenshot": str(screenshot),
                        "thought": step.get("thought", ""),
                        "domain": domain.name,
                        "category": category.name,
                        "step_index": i + 1,
                        "previous_actions": prev_actions,
                        "gt_function": action.get("function", ""),
                        "gt_args": args,
                        "gt_status": step.get("status", "CONTINUE"),
                        "gt_rectangle": action.get("rectangle", {}),
                    }
                    samples.append(sample)

                    if max_samples > 0 and len(samples) >= max_samples:
                        return samples

    return samples


def call_action_prediction(
    client, model_name: str, screenshot_path: str, request: str,
    previous_actions: list, domain: str, temperature: float = 0.0
) -> tuple[str, tuple, tuple]:
    """Call model for action prediction (tool_call format).
    Returns (response_text, orig_wh, resized_wh)."""
    data_url, orig_wh, resized_wh = preprocess_image(screenshot_path)

    prev_action_text = "\n".join(previous_actions) if previous_actions else "None"
    user_text = (
        f"You are a helpful assistant that helps users interact with their computer.\n\n"
        f"Task: {request}\n\n"
        f"Previous actions:\n{prev_action_text}\n\n"
        f"Screenshot resolution: {resized_wh}\n\n"
        f"Based on the screenshot and task, determine the next action to take. "
        f"Output your response as a tool_call in the format:\n"
        f"<tool_call>\n"
        f'{{"function": "action_name", "args": {{"key": "value"}}, "status": "CONTINUE|FINISH"}}\n'
        f"</tool_call>"
    )

    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": user_text},
        ]}
    ]

    try:
        response = client.chat.completions.create(
            model=model_name, messages=messages,
            temperature=temperature, max_tokens=2048,
        )
        return response.choices[0].message.content or "", orig_wh, resized_wh
    except Exception as e:
        print(f"Action prediction call failed: {e}")
        return "", orig_wh, resized_wh


def call_grounding(
    client, model_name: str, screenshot_path: str, thought: str,
    temperature: float = 0.0
) -> tuple[str, tuple, tuple]:
    """Call grounding model to get coordinate prediction.
    Returns (response_text, orig_wh, resized_wh)."""
    data_url, orig_wh, resized_wh = preprocess_image(screenshot_path)
    user_text = GROUNDING_PROMPT.format(instruction=thought)

    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": user_text},
        ]}
    ]

    try:
        response = client.chat.completions.create(
            model=model_name, messages=messages,
            temperature=temperature, max_tokens=512,
        )
        return response.choices[0].message.content or "", orig_wh, resized_wh
    except Exception as e:
        print(f"Grounding call failed: {e}")
        return "", orig_wh, resized_wh


def evaluate_action(
    pred_function: str | None, pred_args: dict | None, pred_status: str | None,
    gt_function: str, gt_args: dict, gt_status: str, gt_rect: dict
) -> dict:
    """Evaluate predicted action against ground truth."""
    function_match = pred_function == gt_function if pred_function else False

    status_str = (pred_status or "").upper()
    gt_status_str = gt_status.upper()
    if gt_status_str == "OVERALL_FINISH":
        gt_status_str = "FINISH"
    elif gt_status_str == "FINISH":
        gt_status_str = "CONTINUE"
    status_match = status_str == gt_status_str

    args_match = False
    coord_match = False

    if pred_args and gt_args and function_match:
        pred_coord = pred_args.get("coordinate")
        gt_coord = gt_args.get("coordinate")

        if pred_coord and gt_rect and isinstance(pred_coord, list) and len(pred_coord) >= 2 and pred_coord[0] is not None and pred_coord[1] is not None:
            try:
                x, y = float(pred_coord[0]), float(pred_coord[1])
                coord_match = (
                    gt_rect.get("left", 0) <= x <= gt_rect.get("right", 0)
                    and gt_rect.get("top", 0) <= y <= gt_rect.get("bottom", 0)
                )
            except (TypeError, ValueError):
                coord_match = False

        # Check non-coordinate args
        other_match = True
        for key in pred_args:
            if key == "coordinate":
                continue
            pred_val = str(pred_args.get(key, "")).lower()
            gt_val = str(gt_args.get(key, "")).lower()
            if pred_val != gt_val and gt_val:
                other_match = False
                break

        args_match = coord_match and other_match

    return {
        "function_match": function_match,
        "args_match": args_match,
        "coord_match": coord_match,
        "status_match": status_match,
        "step_success": function_match and args_match and status_match,
    }


def replace_coordinate(action: dict | None, new_coord: list) -> dict | None:
    """Replace coordinate in a parsed action dict."""
    if action is None:
        return None
    new_action = dict(action)
    args = dict(new_action.get("args", {}))
    args["coordinate"] = new_coord
    new_action["args"] = args
    return new_action


def run_experiment(args):
    """Run the oracle coordinate replacement experiment."""
    from openai import OpenAI

    print("Loading action prediction samples...")
    samples = load_action_prediction_samples(str(DATASET_ROOT), max_samples=args.num_samples)
    print(f"Loaded {len(samples)} samples")

    # Filter to coordinate-based actions (click, right_click, double_click)
    coord_actions = {"click", "right_click", "double_click"}
    samples = [s for s in samples if s["gt_function"] in coord_actions]
    print(f"Filtered to {len(samples)} coordinate-based action samples")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"

    # Resume support
    completed_ids = set()
    if results_path.exists() and not args.overwrite:
        with open(results_path) as f:
            for line in f:
                r = json.loads(line)
                completed_ids.add(r["sample_id"])
        print(f"Resuming: {len(completed_ids)} already completed")

    # Initialize clients
    v2_client = OpenAI(api_key="EMPTY", base_url=args.sft_v2_endpoint, timeout=300)
    v3_client = OpenAI(api_key="EMPTY", base_url=args.sft_v3_endpoint, timeout=300)

    total = len(samples)
    pending_samples = [s for s in samples if s["sample_id"] not in completed_ids]
    t0 = time.time()
    processed = 0
    write_lock = threading.Lock()

    def process_one(sample):
        """Process a single sample with parallel V2/V3 calls."""
        from concurrent.futures import ThreadPoolExecutor as InnerPool

        # Run V2 action and V3 grounding in parallel (different servers)
        with InnerPool(max_workers=2) as inner:
            v2_future = inner.submit(
                call_action_prediction, v2_client, args.sft_v2_model,
                sample["screenshot"], sample["request"],
                sample["previous_actions"], sample["domain"]
            )
            v3_future = inner.submit(
                call_grounding, v3_client, args.sft_v3_model,
                sample["screenshot"], sample["thought"]
            )
            v2_response, v2_orig_wh, v2_resized_wh = v2_future.result()
            v3_response, v3_orig_wh, v3_resized_wh = v3_future.result()

        v2_action = parse_tool_call(v2_response)
        v2_coord = extract_coordinate(v2_action)
        if v2_coord is not None:
            v2_coord = transform_coord_to_original(v2_coord, v2_orig_wh, v2_resized_wh)
            if v2_action and v2_action.get("args", {}).get("coordinate"):
                v2_action["args"]["coordinate"] = v2_coord

        v3_coord = parse_coordinate_response(v3_response)
        if v3_coord is not None:
            v3_coord = transform_coord_to_original(v3_coord, v3_orig_wh, v3_resized_wh)

        gt_coord = sample["gt_args"].get("coordinate")

        eval_a = evaluate_action(
            v2_action.get("function") if v2_action else None,
            v2_action.get("args") if v2_action else None,
            v2_action.get("status") if v2_action else None,
            sample["gt_function"], sample["gt_args"], sample["gt_status"],
            sample["gt_rectangle"]
        )

        if v2_action and v3_coord:
            replaced_b = replace_coordinate(v2_action, v3_coord)
            eval_b = evaluate_action(
                replaced_b.get("function"), replaced_b.get("args"),
                replaced_b.get("status"),
                sample["gt_function"], sample["gt_args"], sample["gt_status"],
                sample["gt_rectangle"]
            )
        else:
            eval_b = {"function_match": False, "args_match": False,
                      "coord_match": False, "status_match": False, "step_success": False}

        if v2_action and gt_coord:
            replaced_c = replace_coordinate(v2_action, gt_coord)
            eval_c = evaluate_action(
                replaced_c.get("function"), replaced_c.get("args"),
                replaced_c.get("status"),
                sample["gt_function"], sample["gt_args"], sample["gt_status"],
                sample["gt_rectangle"]
            )
        else:
            eval_c = {"function_match": False, "args_match": False,
                      "coord_match": True, "status_match": False, "step_success": False}

        return {
            "sample_id": sample["sample_id"],
            "domain": sample["domain"],
            "gt_function": sample["gt_function"],
            "gt_coord": gt_coord,
            "v2_function": v2_action.get("function") if v2_action else None,
            "v2_coord": v2_coord,
            "v3_coord": v3_coord,
            "a_function_match": eval_a["function_match"],
            "a_args_match": eval_a["args_match"],
            "a_coord_match": eval_a["coord_match"],
            "a_status_match": eval_a["status_match"],
            "b_function_match": eval_b["function_match"],
            "b_args_match": eval_b["args_match"],
            "b_coord_match": eval_b["coord_match"],
            "b_status_match": eval_b["status_match"],
            "c_function_match": eval_c["function_match"],
            "c_args_match": eval_c["args_match"],
            "c_coord_match": eval_c["coord_match"],
            "c_status_match": eval_c["status_match"],
        }

    num_workers = getattr(args, "num_workers", 8)
    print(f"Processing {len(pending_samples)} samples with {num_workers} workers...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_one, s): s for s in pending_samples}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                sid = futures[future]["sample_id"]
                print(f"  Sample {sid} failed: {e}")
                continue

            with write_lock:
                with open(results_path, "a") as f:
                    f.write(json.dumps(result, default=_json_default) + "\n")
                processed += 1
                if processed % 100 == 0:
                    elapsed = time.time() - t0
                    rate = processed / elapsed
                    remaining = len(pending_samples) - processed
                    eta = remaining / rate if rate > 0 else 0
                    print(f"Progress: {processed}/{len(pending_samples)}  "
                          f"rate={rate:.1f}/s  ETA={eta:.0f}s")

    print(f"Results saved to: {results_path}")
    return str(results_path)


def analyze_results(results_dir: str):
    """Analyze oracle coordinate replacement results."""
    results_path = Path(results_dir) / "results.jsonl"
    if not results_path.exists():
        print(f"No results file found at {results_path}")
        return

    results = [json.loads(line) for line in open(results_path)]
    n = len(results)

    print("\n" + "=" * 70)
    print("  Experiment 1.3: Oracle Coordinate Replacement (Go/No-Go Gate)")
    print("=" * 70)

    # Overall metrics
    for label, prefix in [
        ("A. SFT v2 original", "a_"),
        ("B. V2 action + V3 coord", "b_"),
        ("C. V2 action + GT coord (oracle)", "c_"),
    ]:
        func_m = sum(1 for r in results if r[f"{prefix}function_match"]) / n
        args_m = sum(1 for r in results if r[f"{prefix}args_match"]) / n
        coord_m = sum(1 for r in results if r[f"{prefix}coord_match"]) / n
        print(f"\n  {label} (N={n}):")
        print(f"    function_match: {func_m:.1%}")
        print(f"    coord_match:    {coord_m:.1%}")
        print(f"    args_match:     {args_m:.1%}")

    # Delta analysis
    a_args = sum(1 for r in results if r["a_args_match"]) / n
    b_args = sum(1 for r in results if r["b_args_match"]) / n
    c_args = sum(1 for r in results if r["c_args_match"]) / n

    print(f"\n  {'='*50}")
    print(f"  KEY DELTAS:")
    print(f"    Oracle - Baseline (C-A):  {c_args - a_args:+.1%}  (coord as bottleneck)")
    print(f"    V3 coord - Baseline (B-A): {b_args - a_args:+.1%}  (V3 practical gain)")
    print(f"    Oracle - V3 coord (C-B):   {c_args - b_args:+.1%}  (room for improvement)")

    # Go/No-Go
    print(f"\n  GO/NO-GO CHECK:")
    oracle_delta = c_args - a_args
    v3_delta = b_args - a_args

    if oracle_delta >= 0.10:
        print(f"    ✓ Oracle Δ = {oracle_delta:+.1%} ≥ 10pp → Coordinate IS the bottleneck")
    elif oracle_delta >= 0.05:
        print(f"    ~ Oracle Δ = {oracle_delta:+.1%} → Moderate bottleneck (5-10pp)")
    else:
        print(f"    ✗ Oracle Δ = {oracle_delta:+.1%} < 5pp → Coord is NOT the main bottleneck")

    if v3_delta > 0:
        print(f"    ✓ V3 coord Δ = {v3_delta:+.1%} > 0 → SFT v3 grounding helps")
    else:
        print(f"    ✗ V3 coord Δ = {v3_delta:+.1%} ≤ 0 → SFT v3 grounding doesn't help")

    # Per-domain breakdown
    domains = sorted(set(r["domain"] for r in results))
    if len(domains) > 1:
        print(f"\n  Per-Domain Breakdown:")
        for domain in domains:
            dr = [r for r in results if r["domain"] == domain]
            nd = len(dr)
            a_d = sum(1 for r in dr if r["a_args_match"]) / nd
            b_d = sum(1 for r in dr if r["b_args_match"]) / nd
            c_d = sum(1 for r in dr if r["c_args_match"]) / nd
            print(f"    {domain:>6s} (N={nd:>4d}):  A={a_d:.1%}  B={b_d:.1%}  C={c_d:.1%}  Δ(C-A)={c_d-a_d:+.1%}")

    # Function type breakdown
    func_types = sorted(set(r["gt_function"] for r in results))
    if len(func_types) > 1:
        print(f"\n  Per-Function Breakdown:")
        for ft in func_types:
            fr = [r for r in results if r["gt_function"] == ft]
            nf = len(fr)
            a_f = sum(1 for r in fr if r["a_args_match"]) / nf
            c_f = sum(1 for r in fr if r["c_args_match"]) / nf
            print(f"    {ft:>15s} (N={nf:>4d}):  A={a_f:.1%}  C={c_f:.1%}  Δ={c_f-a_f:+.1%}")

    # Save summary
    summary = {
        "n_samples": n,
        "a_args_match": a_args,
        "b_args_match": b_args,
        "c_args_match": c_args,
        "oracle_delta": oracle_delta,
        "v3_delta": v3_delta,
        "go_nogo": "GO" if oracle_delta >= 0.10 and v3_delta > 0 else "NO-GO" if oracle_delta < 0.05 else "MARGINAL",
    }
    summary_path = Path(results_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Exp 1.3: Oracle Coordinate Replacement")
    parser.add_argument("--sft_v2_model", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/llamafactory/output/gui360_full_sft_v2")
    parser.add_argument("--sft_v3_model", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/llamafactory/output/gui360_full_sft_v3_grounding")
    parser.add_argument("--sft_v2_endpoint", type=str, default="http://localhost:19816/v1")
    parser.add_argument("--sft_v3_endpoint", type=str, default="http://localhost:19815/v1")
    parser.add_argument("--num_samples", type=int, default=0, help="0=all")
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "exp1_3"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of concurrent workers for parallel processing")
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--results_dir", type=str, default="")

    args = parser.parse_args()

    if args.analyze_only:
        analyze_results(args.results_dir or args.output_dir)
    else:
        run_experiment(args)
        analyze_results(args.output_dir)


if __name__ == "__main__":
    main()
