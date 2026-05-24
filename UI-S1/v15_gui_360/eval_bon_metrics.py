#!/usr/bin/env python3
"""Best-of-N metric analysis: find which metric best predicts correctness.

Generates N samples per step with logprobs enabled, computes multiple
metrics per sample, and analyzes correlation with correctness.

Metrics computed:
  - mean_logprob: average per-token log probability
  - sum_logprob: total sequence log probability
  - min_logprob: lowest confidence token
  - response_length: number of tokens in response
  - perplexity: exp(-mean_logprob)
  - type_agreement: fraction of N samples sharing this sample's action type
  - coord_in_bounds: whether predicted coordinate is within image bounds

Goal: find a metric that separates correct from incorrect samples,
enabling a "free" verifier for Best-of-N selection.

Usage:
    python v15_gui_360/eval_bon_metrics.py \
        --test_data v12_gui_360/data/gui360_test_1000_balanced.jsonl \
        --api_url http://localhost:8000/v1 \
        --model_name balanced_sft_step272 \
        --output_dir v15_gui_360/outputs/preliminary_tests/bon_metrics \
        --threads 64 --gt_history --num_samples 5 --max_episodes 200
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v13_gui_360.eval_gui360_template import (
    parse_tool_call, _format_action_for_history, SUPPORTED_ACTIONS,
    _encode_screenshot, USER_PROMPT_TEMPLATE,
)
from v13_gui_360.reward import compute_step_reward, _normalize_action_type


def compute_sample_metrics(choice, all_parsed, image_w, image_h):
    """Compute metrics for a single sample from its API response choice."""
    metrics = {}

    # ── Logprob metrics ──
    if choice.logprobs and choice.logprobs.content:
        token_logprobs = [t.logprob for t in choice.logprobs.content]
        metrics["mean_logprob"] = float(np.mean(token_logprobs))
        metrics["sum_logprob"] = float(np.sum(token_logprobs))
        metrics["min_logprob"] = float(np.min(token_logprobs))
        metrics["max_logprob"] = float(np.max(token_logprobs))
        metrics["std_logprob"] = float(np.std(token_logprobs))
        metrics["response_length"] = len(token_logprobs)
        metrics["perplexity"] = float(math.exp(-metrics["mean_logprob"]))

        # Logprob of bottom-10% tokens (tail uncertainty)
        sorted_lp = sorted(token_logprobs)
        n_tail = max(1, len(sorted_lp) // 10)
        metrics["tail_mean_logprob"] = float(np.mean(sorted_lp[:n_tail]))

        # Length-normalized log probability
        metrics["length_norm_logprob"] = metrics["sum_logprob"] / max(metrics["response_length"], 1)
    else:
        metrics["mean_logprob"] = None
        metrics["sum_logprob"] = None
        metrics["min_logprob"] = None
        metrics["max_logprob"] = None
        metrics["std_logprob"] = None
        metrics["response_length"] = len(choice.message.content or "")
        metrics["perplexity"] = None
        metrics["tail_mean_logprob"] = None
        metrics["length_norm_logprob"] = None

    # ── Text-based metrics ──
    text = choice.message.content or ""
    metrics["text_length"] = len(text)

    # Parse action
    parsed = parse_tool_call(text)
    if parsed is None:
        m = re.search(r'<action>\s*(\{.*?\})\s*</action>', text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

    metrics["has_valid_action"] = parsed is not None

    if parsed:
        action_type = _normalize_action_type(parsed.get("action", ""))
        coord = parsed.get("coordinate")

        # Coordinate in bounds?
        if coord and len(coord) >= 2 and coord[0] is not None and coord[1] is not None:
            metrics["coord_in_bounds"] = (
                0 <= coord[0] <= image_w and 0 <= coord[1] <= image_h
            )
            metrics["coord_x"] = float(coord[0])
            metrics["coord_y"] = float(coord[1])
        else:
            metrics["coord_in_bounds"] = None
            metrics["coord_x"] = None
            metrics["coord_y"] = None

        # Type agreement with other samples
        other_types = [_normalize_action_type((p or {}).get("action", "")) for p in all_parsed]
        type_count = sum(1 for t in other_types if t == action_type)
        metrics["type_agreement"] = type_count / max(len(other_types), 1)

        # Coordinate agreement: how many others have coords within 50px?
        if coord and coord[0] is not None:
            nearby = 0
            total_with_coord = 0
            for p in all_parsed:
                if p and p.get("coordinate") and p["coordinate"][0] is not None:
                    total_with_coord += 1
                    d = ((coord[0] - p["coordinate"][0])**2 +
                         (coord[1] - p["coordinate"][1])**2) ** 0.5
                    if d < 50:
                        nearby += 1
            metrics["coord_agreement"] = nearby / max(total_with_coord, 1)
        else:
            metrics["coord_agreement"] = None
    else:
        metrics["coord_in_bounds"] = None
        metrics["coord_x"] = None
        metrics["coord_y"] = None
        metrics["type_agreement"] = 0
        metrics["coord_agreement"] = None

    return metrics, parsed


def evaluate_episode_bon(
    client: OpenAI,
    model_name: str,
    episode: Dict,
    num_samples: int = 5,
    temperature: float = 0.7,
    gt_history: bool = True,
    match_threshold: float = 0.5,
    image_max_pixels: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate one episode collecting per-sample metrics with logprobs."""
    episode_id = episode["episode_id"]
    goal = episode["goal"]
    steps = episode["steps"]

    history = []
    step_results = []

    for i, step in enumerate(steps):
        gt_action = step["action"]
        screenshot = step["screenshot"]
        image_w = step.get("image_w", 1040)
        image_h = step.get("image_h", 736)

        history_text = "\n".join(history) if history else "None"
        b64 = _encode_screenshot(screenshot, image_max_pixels)

        prompt_text = USER_PROMPT_TEMPLATE.format(
            instruction=goal,
            history=history_text,
            actions=SUPPORTED_ACTIONS,
        )
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt_text},
        ]}]

        samples = []
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1024,
                temperature=temperature,
                n=num_samples,
                logprobs=True,
                top_logprobs=3,
            )

            # First pass: parse all actions
            all_parsed = []
            for choice in response.choices:
                text = choice.message.content or ""
                parsed = parse_tool_call(text)
                if parsed is None:
                    m = re.search(r'<action>\s*(\{.*?\})\s*</action>', text, re.DOTALL)
                    if m:
                        try:
                            parsed = json.loads(m.group(1))
                        except json.JSONDecodeError:
                            pass
                all_parsed.append(parsed)

            # Second pass: compute metrics
            for j, choice in enumerate(response.choices):
                metrics, parsed = compute_sample_metrics(
                    choice, all_parsed, image_w, image_h
                )

                # Compute reward
                if parsed:
                    fake = f"<action>{json.dumps(parsed)}</action>"
                else:
                    fake = choice.message.content or ""
                reward, info = compute_step_reward(fake, gt_action, image_w, image_h)

                metrics["reward"] = reward
                metrics["is_correct"] = reward >= match_threshold
                metrics["gt_type"] = info.get("gt_type", "")
                metrics["pred_type"] = info.get("pred_type", "")
                metrics["type_reward"] = info.get("type_reward", 0)
                metrics["content_reward"] = info.get("content_reward", 0)
                samples.append(metrics)

        except Exception as e:
            print(f"  [ep {episode_id}] step {i+1} error: {e}")

        step_results.append({
            "step_idx": i,
            "gt_type": gt_action.get("action", ""),
            "samples": samples,
        })

        if gt_history:
            history.append(_format_action_for_history(gt_action, i + 1))

    return {
        "episode_id": episode_id,
        "steps": step_results,
    }


def analyze_correlations(results):
    """Analyze which metric best predicts correctness."""
    # Collect all samples across all steps
    metric_names = [
        "mean_logprob", "sum_logprob", "min_logprob", "max_logprob",
        "std_logprob", "perplexity", "response_length", "text_length",
        "tail_mean_logprob", "length_norm_logprob",
        "type_agreement", "coord_agreement",
    ]

    correct_vals = {m: [] for m in metric_names}
    wrong_vals = {m: [] for m in metric_names}

    # For Best-of-N selection accuracy
    bon_correct_by_metric = {m: 0 for m in metric_names}
    bon_total = 0

    for eid, ep in results.items():
        for step in ep["steps"]:
            samples = step["samples"]
            if len(samples) < 2:
                continue

            # Check if any sample is correct and any is wrong (mixed step)
            has_correct = any(s["is_correct"] for s in samples)
            has_wrong = any(not s["is_correct"] for s in samples)

            for s in samples:
                for m in metric_names:
                    val = s.get(m)
                    if val is not None:
                        if s["is_correct"]:
                            correct_vals[m].append(val)
                        else:
                            wrong_vals[m].append(val)

            # Best-of-N selection: for each metric, select the sample
            # with the highest/best value and check if it's correct
            if has_correct:  # only count steps where correct answer exists
                bon_total += 1
                for m in metric_names:
                    valid_samples = [(j, s) for j, s in enumerate(samples) if s.get(m) is not None]
                    if not valid_samples:
                        continue

                    # Higher is better for most metrics; lower for perplexity
                    if m in ("perplexity", "std_logprob"):
                        best_idx = min(valid_samples, key=lambda x: x[1][m])[0]
                    else:
                        best_idx = max(valid_samples, key=lambda x: x[1][m])[0]

                    if samples[best_idx]["is_correct"]:
                        bon_correct_by_metric[m] += 1

    return correct_vals, wrong_vals, bon_correct_by_metric, bon_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--gt_history", action="store_true", default=False)
    parser.add_argument("--max_episodes", type=int, default=200)
    parser.add_argument("--image_max_pixels", type=int, default=None)
    args = parser.parse_args()

    episodes = []
    with open(args.test_data) as f:
        for line in f:
            episodes.append(json.loads(line))

    if args.max_episodes and args.max_episodes < len(episodes):
        episodes = episodes[:args.max_episodes]
    print(f"Loaded {len(episodes)} episodes (max_episodes={args.max_episodes})")

    client = OpenAI(base_url=args.api_url, api_key="dummy")
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                evaluate_episode_bon, client, args.model_name, ep,
                args.num_samples, args.temperature, args.gt_history,
                args.match_threshold, args.image_max_pixels,
            ): ep["episode_id"]
            for ep in episodes
        }

        pbar = tqdm(total=len(episodes), desc="BoN metrics")
        for future in as_completed(futures):
            result = future.result()
            results[result["episode_id"]] = result
            pbar.update(1)
        pbar.close()

    # Save raw results
    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output_dir, f"bon_raw_{ts}.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ── Correlation analysis ──
    correct_vals, wrong_vals, bon_acc, bon_total = analyze_correlations(results)

    print(f"\n{'='*75}")
    print(f"Best-of-N Metric Analysis (N={args.num_samples}, {len(episodes)} episodes)")
    print(f"{'='*75}")

    # 1. Mean values: correct vs wrong
    print(f"\n{'Metric':<25} {'Correct (mean)':<18} {'Wrong (mean)':<18} {'Gap':<12} {'Direction':<10}")
    print("-" * 85)

    metric_gaps = {}
    for m in sorted(correct_vals.keys()):
        c = correct_vals[m]
        w = wrong_vals[m]
        if len(c) > 10 and len(w) > 10:
            cm = np.mean(c)
            wm = np.mean(w)
            gap = cm - wm
            direction = "higher=better" if gap > 0 else "lower=better"
            if m == "perplexity":
                direction = "lower=better" if gap < 0 else "higher=better"
            print(f"  {m:<23} {cm:<18.4f} {wm:<18.4f} {gap:<12.4f} {direction}")
            metric_gaps[m] = abs(gap)

    # 2. Best-of-N selection accuracy by metric
    print(f"\n{'='*75}")
    print(f"Best-of-N Selection Accuracy (select sample with best metric value)")
    print(f"Total steps with ≥1 correct sample: {bon_total}")
    print(f"{'='*75}")
    print(f"\n{'Metric':<25} {'Correct selections':<20} {'Accuracy':<10}")
    print("-" * 55)

    ranked = sorted(bon_acc.items(), key=lambda x: x[1], reverse=True)
    for m, count in ranked:
        if bon_total > 0:
            acc = count / bon_total * 100
            print(f"  {m:<23} {count:<20} {acc:.1f}%")

    # 3. Random baseline
    if bon_total > 0:
        # Compute average number of correct samples per step
        total_correct_samples = 0
        total_samples = 0
        for eid, ep in results.items():
            for step in ep["steps"]:
                samples = step["samples"]
                if any(s["is_correct"] for s in samples):
                    total_correct_samples += sum(1 for s in samples if s["is_correct"])
                    total_samples += len(samples)
        if total_samples > 0:
            random_acc = total_correct_samples / total_samples * 100
            print(f"\n  {'Random baseline':<23} {'—':<20} {random_acc:.1f}%")
            print(f"  {'Oracle (any correct)':<23} {'—':<20} 100.0%")

    # 4. Separability analysis (AUC-like)
    print(f"\n{'='*75}")
    print(f"Separability (Cohen's d: effect size)")
    print(f"{'='*75}")
    print(f"\n{'Metric':<25} {'Cohen d':<10} {'Interpretation'}")
    print("-" * 55)

    for m in sorted(correct_vals.keys()):
        c = correct_vals[m]
        w = wrong_vals[m]
        if len(c) > 10 and len(w) > 10:
            cm, cs = np.mean(c), np.std(c)
            wm, ws = np.mean(w), np.std(w)
            pooled_std = ((cs**2 + ws**2) / 2) ** 0.5
            if pooled_std > 0:
                d = abs(cm - wm) / pooled_std
                interp = "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"
                print(f"  {m:<23} {d:<10.3f} {interp}")

    # Save summary
    summary = {
        "num_episodes": len(episodes),
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "bon_total_steps": bon_total,
        "bon_accuracy_by_metric": {m: c / bon_total if bon_total > 0 else 0 for m, c in bon_acc.items()},
        "metric_means_correct": {m: float(np.mean(v)) for m, v in correct_vals.items() if len(v) > 0},
        "metric_means_wrong": {m: float(np.mean(v)) for m, v in wrong_vals.items() if len(v) > 0},
    }
    with open(os.path.join(args.output_dir, f"bon_summary_{ts}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
