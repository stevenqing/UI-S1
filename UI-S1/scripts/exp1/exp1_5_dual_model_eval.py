#!/usr/bin/env python3
"""
Experiment 1.5: Dual-Model Combination vs Single Model

Question: SFT v2 action type + SFT v3 coordinate > any single model?

Conditions:
  A. Baseline:     SFT v2 action + SFT v2 coordinate
  B. SFT v3 only:  SFT v3 action + SFT v3 coordinate
  C. Dual (K=1):   SFT v2 action + SFT v3 greedy coordinate
  D. Dual (K=5):   SFT v2 action + SFT v3 K=5 clustered coordinate

Success criteria: D args_match > A by ≥5pp

Usage:
    # Needs two vLLM servers:
    python scripts/exp1/exp1_5_dual_model_eval.py \
        --sft_v2_endpoint http://localhost:19816/v1 \
        --sft_v3_endpoint http://localhost:19815/v1

    python scripts/exp1/exp1_5_dual_model_eval.py \
        --analyze_only --results_dir outputs/exp1_5
"""

import argparse
import json
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.exp0.exp0_1_uncertainty_analysis import (
    _json_default,
    extract_coordinate,
    parse_tool_call,
)
from scripts.exp0.data_utils import DATASET_ROOT
from scripts.exp1.exp1_3_oracle_coord_replacement import (
    call_action_prediction,
    call_grounding,
    evaluate_action,
    load_action_prediction_samples,
    replace_coordinate,
)
from scripts.exp1.grounding_utils import (
    preprocess_image,
    parse_coordinate_response,
    transform_coord_to_original,
    cluster_coordinates,
    GROUNDING_PROMPT,
)


def call_grounding_k_times_with_transform(client, model_name, screenshot, thought, K, temperature):
    """Call grounding model K times with smart_resize and cluster the coordinates.
    Uses n=K for batched inference."""
    data_url, orig_wh, resized_wh = preprocess_image(screenshot)
    user_text = GROUNDING_PROMPT.format(instruction=thought)

    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": user_text},
        ]}
    ]

    coords = []
    try:
        resp = client.chat.completions.create(
            model=model_name, messages=messages,
            temperature=temperature, max_tokens=512,
            n=K,
        )
        for choice in resp.choices:
            text = choice.message.content or ""
            coord = parse_coordinate_response(text)
            if coord is not None:
                coord = transform_coord_to_original(coord, orig_wh, resized_wh)
                coords.append(coord)
    except Exception as e:
        # Fallback to sequential
        for _ in range(K):
            try:
                resp = client.chat.completions.create(
                    model=model_name, messages=messages,
                    temperature=temperature, max_tokens=512,
                )
                text = resp.choices[0].message.content or ""
            except Exception as e2:
                text = ""
            coord = parse_coordinate_response(text)
            if coord is not None:
                coord = transform_coord_to_original(coord, orig_wh, resized_wh)
                coords.append(coord)

    cluster_result = cluster_coordinates(coords, eps=30.0)
    return coords, cluster_result


def run_experiment(args):
    """Run dual-model combination experiment."""
    from openai import OpenAI

    print("Loading action prediction samples...")
    samples = load_action_prediction_samples(str(DATASET_ROOT), max_samples=args.num_samples)

    # Filter to coordinate-based actions
    coord_actions = {"click", "right_click", "double_click"}
    samples = [s for s in samples if s["gt_function"] in coord_actions]
    print(f"Loaded {len(samples)} coordinate-based action samples")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"

    completed_ids = set()
    if results_path.exists() and not args.overwrite:
        with open(results_path) as f:
            for line in f:
                completed_ids.add(json.loads(line)["sample_id"])
        print(f"Resuming: {len(completed_ids)} already completed")

    v2_client = OpenAI(api_key="EMPTY", base_url=args.sft_v2_endpoint, timeout=300)
    v3_client = OpenAI(api_key="EMPTY", base_url=args.sft_v3_endpoint, timeout=300)

    pending_samples = [s for s in samples if s["sample_id"] not in completed_ids]
    t0 = time.time()
    processed = 0
    write_lock = threading.Lock()

    def process_one(sample):
        """Process a single sample with parallel V2/V3 calls."""
        from concurrent.futures import ThreadPoolExecutor as InnerPool

        # Run V2 action, V3 action, V3 grounding, V3 K-sample all in parallel
        with InnerPool(max_workers=4) as inner:
            f_v2 = inner.submit(
                call_action_prediction, v2_client, args.sft_v2_model,
                sample["screenshot"], sample["request"],
                sample["previous_actions"], sample["domain"]
            )
            f_v3a = inner.submit(
                call_action_prediction, v3_client, args.sft_v3_model,
                sample["screenshot"], sample["request"],
                sample["previous_actions"], sample["domain"]
            )
            f_v3g = inner.submit(
                call_grounding, v3_client, args.sft_v3_model,
                sample["screenshot"], sample["thought"]
            )
            f_v3k = inner.submit(
                call_grounding_k_times_with_transform,
                v3_client, args.sft_v3_model, sample["screenshot"],
                sample["thought"], args.K, args.temperature
            )

            v2_response, v2_orig_wh, v2_resized_wh = f_v2.result()
            v3_action_response, v3a_orig_wh, v3a_resized_wh = f_v3a.result()
            v3_grounding_resp, v3g_orig_wh, v3g_resized_wh = f_v3g.result()
            all_coords, cluster_result = f_v3k.result()

        # Process V2 action
        v2_action = parse_tool_call(v2_response)
        v2_coord = extract_coordinate(v2_action)
        if v2_coord is not None:
            v2_coord = transform_coord_to_original(v2_coord, v2_orig_wh, v2_resized_wh)
            if v2_action and v2_action.get("args", {}).get("coordinate"):
                v2_action["args"]["coordinate"] = v2_coord

        eval_a = evaluate_action(
            v2_action.get("function") if v2_action else None,
            v2_action.get("args") if v2_action else None,
            v2_action.get("status") if v2_action else None,
            sample["gt_function"], sample["gt_args"], sample["gt_status"],
            sample["gt_rectangle"]
        )

        # Process V3 action
        v3_action = parse_tool_call(v3_action_response)
        v3a_coord = extract_coordinate(v3_action)
        if v3a_coord is not None:
            v3a_coord = transform_coord_to_original(v3a_coord, v3a_orig_wh, v3a_resized_wh)
            if v3_action and v3_action.get("args", {}).get("coordinate"):
                v3_action["args"]["coordinate"] = v3a_coord

        eval_b = evaluate_action(
            v3_action.get("function") if v3_action else None,
            v3_action.get("args") if v3_action else None,
            v3_action.get("status") if v3_action else None,
            sample["gt_function"], sample["gt_args"], sample["gt_status"],
            sample["gt_rectangle"]
        )

        # Process V3 grounding (greedy)
        v3_greedy_coord = parse_coordinate_response(v3_grounding_resp)
        if v3_greedy_coord is not None:
            v3_greedy_coord = transform_coord_to_original(v3_greedy_coord, v3g_orig_wh, v3g_resized_wh)

        if v2_action and v3_greedy_coord:
            replaced_c = replace_coordinate(v2_action, v3_greedy_coord)
            eval_c = evaluate_action(
                replaced_c.get("function"), replaced_c.get("args"),
                replaced_c.get("status"),
                sample["gt_function"], sample["gt_args"], sample["gt_status"],
                sample["gt_rectangle"]
            )
        else:
            eval_c = {"function_match": False, "args_match": False,
                      "coord_match": False, "status_match": False, "step_success": False}

        # Process V3 K-sample cluster
        v3_cluster_coord = cluster_result["cluster_center"]
        if v2_action and v3_cluster_coord:
            replaced_d = replace_coordinate(v2_action, v3_cluster_coord)
            eval_d = evaluate_action(
                replaced_d.get("function"), replaced_d.get("args"),
                replaced_d.get("status"),
                sample["gt_function"], sample["gt_args"], sample["gt_status"],
                sample["gt_rectangle"]
            )
        else:
            eval_d = {"function_match": False, "args_match": False,
                      "coord_match": False, "status_match": False, "step_success": False}

        return {
            "sample_id": sample["sample_id"],
            "domain": sample["domain"],
            "gt_function": sample["gt_function"],
            "v2_function": v2_action.get("function") if v2_action else None,
            "v3_function": v3_action.get("function") if v3_action else None,
            "v3_greedy_coord": v3_greedy_coord,
            "v3_cluster_coord": v3_cluster_coord,
            "v3_agreement_rate": cluster_result["agreement_rate"],
            "K": args.K,
            "a_function_match": eval_a["function_match"],
            "a_args_match": eval_a["args_match"],
            "a_coord_match": eval_a["coord_match"],
            "b_function_match": eval_b["function_match"],
            "b_args_match": eval_b["args_match"],
            "b_coord_match": eval_b["coord_match"],
            "c_function_match": eval_c["function_match"],
            "c_args_match": eval_c["args_match"],
            "c_coord_match": eval_c["coord_match"],
            "d_function_match": eval_d["function_match"],
            "d_args_match": eval_d["args_match"],
            "d_coord_match": eval_d["coord_match"],
        }

    num_workers = getattr(args, "num_workers", 4)
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
                if processed % 50 == 0:
                    elapsed = time.time() - t0
                    rate = processed / elapsed
                    remaining = len(pending_samples) - processed
                    eta = remaining / rate if rate > 0 else 0
                    print(f"Progress: {processed}/{len(pending_samples)}  "
                          f"rate={rate:.1f}/s  ETA={eta:.0f}s")

    print(f"Results saved to: {results_path}")


def analyze_results(results_dir: str):
    """Analyze dual-model combination results."""
    results_path = Path(results_dir) / "results.jsonl"
    if not results_path.exists():
        print(f"No results at {results_path}")
        return

    results = [json.loads(line) for line in open(results_path)]
    n = len(results)

    print("\n" + "=" * 70)
    print("  Experiment 1.5: Dual-Model Combination vs Single Model")
    print("=" * 70)

    conditions = [
        ("A. SFT v2 only", "a_"),
        ("B. SFT v3 only (action prompt)", "b_"),
        ("C. Dual (K=1): v2 act + v3 coord", "c_"),
        ("D. Dual (K=5): v2 act + v3 cluster", "d_"),
    ]

    for label, prefix in conditions:
        func_m = sum(1 for r in results if r[f"{prefix}function_match"]) / n
        args_m = sum(1 for r in results if r[f"{prefix}args_match"]) / n
        coord_m = sum(1 for r in results if r[f"{prefix}coord_match"]) / n
        print(f"\n  {label} (N={n}):")
        print(f"    function_match: {func_m:.1%}")
        print(f"    coord_match:    {coord_m:.1%}")
        print(f"    args_match:     {args_m:.1%}")

    # Key comparison
    a_args = sum(1 for r in results if r["a_args_match"]) / n
    d_args = sum(1 for r in results if r["d_args_match"]) / n
    delta = d_args - a_args

    print(f"\n  {'='*50}")
    print(f"  KEY RESULT: Dual(K=5) - SFTv2 = {delta:+.1%}")

    if delta >= 0.05:
        print(f"  ✓ PASS: ≥5pp improvement → dual-model combination is effective")
    elif delta > 0:
        print(f"  ~ MARGINAL: some improvement but <5pp")
    else:
        print(f"  ✗ FAIL: no improvement → dual-model grounding not beneficial")

    # Adaptive K analysis based on agreement rate
    print(f"\n  Agreement Rate → Coord Accuracy (Condition D):")
    for lo, hi, label in [(0.9, 1.01, "≥0.9"), (0.5, 0.9, "0.5-0.9"), (0.0, 0.5, "<0.5")]:
        subset = [r for r in results if lo <= r.get("v3_agreement_rate", 0) < hi]
        if subset:
            coord_acc = sum(1 for r in subset if r["d_coord_match"]) / len(subset)
            print(f"    {label:>8s}: coord_match={coord_acc:.1%} (n={len(subset)})")

    summary = {
        "n_samples": n,
        "a_args_match": a_args,
        "d_args_match": d_args,
        "delta_d_minus_a": delta,
        "verdict": "effective" if delta >= 0.05 else "marginal" if delta > 0 else "ineffective",
    }
    summary_path = Path(results_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Exp 1.5: Dual-Model Eval")
    parser.add_argument("--sft_v2_model", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/llamafactory/output/gui360_full_sft_v2")
    parser.add_argument("--sft_v3_model", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/llamafactory/output/gui360_full_sft_v3_grounding")
    parser.add_argument("--sft_v2_endpoint", type=str, default="http://localhost:19816/v1")
    parser.add_argument("--sft_v3_endpoint", type=str, default="http://localhost:19815/v1")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "exp1_5"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4,
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
