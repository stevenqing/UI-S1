"""GUI-360 Multi-Sample Collection for Temperature Degradation & Coord Spread Analysis.

Collects K samples per step with temperature=1.0 for GUI-360 action prediction,
parallel to AC's eval_c4c7_multisample.py.

Outputs: per-step results with K predictions, function_match, args_match, coordinates.
"""

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add GUI-360-eval to path for model + evaluator imports
PROJECT_DIR = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
GUI360_EVAL_DIR = os.path.join(PROJECT_DIR, "train_GUI_360", "GUI-360-eval")
sys.path.insert(0, GUI360_EVAL_DIR)

from openai import OpenAI

result_lock = Lock()


def load_gui360_samples(root_dir, max_samples=None):
    """Load GUI-360 action prediction samples (same logic as ActionPredictionEvaluator.load_data)."""
    from prompts.prompt_action_prediction import (
        SUPPORTED_ACTIONS_EXCEL,
        SUPPORTED_ACTIONS_PPT,
        SUPPORTED_ACTIONS_WORD,
    )

    data_path = os.path.join(root_dir, "data")
    samples = []

    for domain in sorted(os.listdir(data_path)):
        domain_path = os.path.join(data_path, domain)
        if not os.path.isdir(domain_path):
            continue

        # Get supported actions for this domain
        if domain.lower() == "word":
            actions = SUPPORTED_ACTIONS_WORD
        elif domain.lower() == "excel":
            actions = SUPPORTED_ACTIONS_EXCEL
        elif domain.lower() == "ppt":
            actions = SUPPORTED_ACTIONS_PPT
        else:
            continue

        for category in sorted(os.listdir(domain_path)):
            category_path = os.path.join(domain_path, category, "success")
            if not os.path.exists(category_path):
                continue

            for jsonl_file in sorted(os.listdir(category_path)):
                if not jsonl_file.endswith(".jsonl"):
                    continue

                file_path = os.path.join(category_path, jsonl_file)
                all_steps = []
                with open(file_path, "r") as f:
                    for line_num, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line.strip())
                            all_steps.append({"line_num": line_num, "data": data})
                        except json.JSONDecodeError:
                            continue

                for i, step_info in enumerate(all_steps):
                    data = step_info["data"]
                    line_num = step_info["line_num"]

                    sample_id = f"{domain}_{category}_{os.path.splitext(jsonl_file)[0]}_{line_num}"

                    clean_img_path = os.path.join(
                        root_dir, "image", domain, category,
                        data["step"]["screenshot_clean"],
                    )

                    if not os.path.exists(clean_img_path):
                        continue

                    # Check for action_prediction tag
                    if "action_prediction" not in data["step"].get("tags", []):
                        continue

                    # Build previous actions
                    previous_actions = []
                    for j in range(i):
                        prev_thought = all_steps[j]["data"]["step"]["thought"]
                        previous_actions.append(f"Step {j+1}: {prev_thought}")

                    status = data["step"]["status"]
                    if status == "OVERALL_FINISH":
                        status = "FINISH"
                    elif status == "FINISH":
                        status = "CONTINUE"

                    action = data["step"]["action"]
                    # Skip drag actions (same as evaluator)
                    if action.get("function", "") == "drag":
                        continue
                    if not action.get("rectangle", {}):
                        continue

                    # Build ground truth rect
                    gt_rect = action.get("rectangle", {})

                    # Build GT function/args
                    gt_function = action.get("function", "")
                    gt_args = dict(action.get("args", {}))

                    # Handle coordinates
                    if "coordinate_x" in action and action["coordinate_x"]:
                        gt_args["coordinate"] = [action["coordinate_x"], action["coordinate_y"]]

                    # Remove raw x/y from args
                    gt_args.pop("x", None)
                    gt_args.pop("y", None)

                    samples.append({
                        "sample_id": sample_id,
                        "request": data["request"],
                        "screenshot_clean": clean_img_path,
                        "thought": data["step"]["thought"],
                        "domain": domain,
                        "category": category,
                        "actions_str": actions,
                        "previous_actions": previous_actions,
                        "gt_function": gt_function,
                        "gt_args": gt_args,
                        "gt_status": status,
                        "gt_rect": gt_rect,
                        "step_index": i,
                        "total_steps": len(all_steps),
                        "trajectory_id": f"{domain}_{category}_{os.path.splitext(jsonl_file)[0]}",
                    })

                    if max_samples and len(samples) >= max_samples:
                        print(f"Reached max_samples={max_samples}")
                        return samples

    return samples


def init_model(api_url, model_name):
    """Initialize the Qwen2.5-VL model."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qwen2_5_vl_7b",
        os.path.join(GUI360_EVAL_DIR, "models", "qwen2.5_vl_7b.py"),
    )
    qwen_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qwen_module)

    model = qwen_module.Qwen25VL7B(
        api_url=api_url, model_name=model_name,
        coordinate_system="absolute", resize_factor=28,
    )
    return model


# Global model reference
_model = None


def call_model(sample, temperature=0.0):
    """Call model for a single prediction."""
    global _model

    system_prompt, user_prompt = _model.construct_action_prompt(
        instruction=sample["request"],
        history="\n".join(sample["previous_actions"]) if sample["previous_actions"] else "",
        actions=sample["actions_str"],
        resolution=None,  # Will be determined from image
    )

    raw_response = _model.predict(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_path=sample["screenshot_clean"],
        temperature=temperature,
        max_tokens=4096,
    )

    # Parse response
    pred_function, pred_args, pred_status = _model.parse_action(raw_response)

    return {
        "raw_response": raw_response,
        "pred_function": pred_function,
        "pred_args": pred_args,
        "pred_status": pred_status,
    }


def evaluate_prediction(pred, sample):
    """Evaluate a single prediction against ground truth."""
    from evaluator.tool_definitions import normalize_tool_args

    pred_function = pred.get("pred_function")
    pred_args = pred.get("pred_args") or {}
    gt_function = sample["gt_function"]
    gt_args = sample["gt_args"]
    gt_rect = sample["gt_rect"]

    function_match = pred_function == gt_function if pred_function else False

    # Args match
    args_match = False
    if pred_args and gt_args:
        try:
            pred_normalized = normalize_tool_args(pred_function or "unknown", pred_args)
            gt_normalized = normalize_tool_args(gt_function, gt_args)

            if "coordinate" in pred_normalized and "coordinate" in gt_normalized:
                pred_coord = pred_normalized["coordinate"]
                gt_coord = gt_normalized["coordinate"]

                if (isinstance(pred_coord, (list, tuple)) and len(pred_coord) == 2
                        and isinstance(gt_coord, (list, tuple)) and len(gt_coord) == 2):
                    if gt_rect:
                        pred_x, pred_y = float(pred_coord[0]), float(pred_coord[1])
                        coordinate_match = (
                            gt_rect["left"] <= pred_x <= gt_rect["right"]
                            and gt_rect["top"] <= pred_y <= gt_rect["bottom"]
                        )
                    else:
                        tolerance = 25.0
                        coordinate_match = (
                            abs(float(pred_coord[0]) - float(gt_coord[0])) <= tolerance
                            and abs(float(pred_coord[1]) - float(gt_coord[1])) <= tolerance
                        )

                    other_match = True
                    for key in pred_normalized:
                        if key != "coordinate":
                            if str(pred_normalized.get(key, "")).lower() != str(gt_normalized.get(key, "")).lower():
                                other_match = False
                                break

                    args_match = coordinate_match and other_match
            else:
                # Non-coordinate comparison
                args_match = all(
                    str(pred_normalized.get(k, "")).lower() == str(gt_normalized.get(k, "")).lower()
                    for k in set(list(pred_normalized.keys()) + list(gt_normalized.keys()))
                )
        except Exception:
            args_match = False

    # Extract coordinates for coord_spread analysis
    coordinates = None
    if pred_args and "coordinate" in pred_args:
        try:
            coord = pred_args["coordinate"]
            if isinstance(coord, (list, tuple)) and len(coord) == 2:
                coordinates = [float(coord[0]), float(coord[1])]
        except (ValueError, TypeError):
            pass

    return {
        "function_match": function_match,
        "args_match": args_match,
        "pred_function": pred_function,
        "coordinates": coordinates,
    }


def process_sample(sample, args):
    """Process a single sample: collect K predictions."""
    results = []

    for k in range(args.K):
        try:
            temp = 0.0 if k == 0 else args.temperature
            pred = call_model(sample, temperature=temp)
            eval_result = evaluate_prediction(pred, sample)
            results.append({
                "k": k,
                "temperature": temp,
                "pred_function": eval_result["pred_function"],
                "function_match": eval_result["function_match"],
                "args_match": eval_result["args_match"],
                "coordinates": eval_result["coordinates"],
                "raw_response": pred["raw_response"][:500],  # truncate for storage
            })
        except Exception as e:
            results.append({
                "k": k,
                "temperature": temp if k > 0 else 0.0,
                "pred_function": None,
                "function_match": False,
                "args_match": False,
                "coordinates": None,
                "error": str(e),
            })

    # Compute per-step statistics
    greedy = results[0] if results else {}
    temp_samples = results[1:] if len(results) > 1 else []

    # Agreement: most common predicted function across all K samples
    all_functions = [r["pred_function"] for r in results if r.get("pred_function")]
    if all_functions:
        func_counts = Counter(all_functions)
        most_common_func, most_common_count = func_counts.most_common(1)[0]
        agreement_rate = most_common_count / len(results)
    else:
        agreement_rate = 0.0
        most_common_func = None

    # Action entropy
    import math
    if all_functions:
        total = len(all_functions)
        entropy = -sum((c/total) * math.log2(c/total) for c in func_counts.values())
    else:
        entropy = 0.0

    # Coordinate spread (for coordinate-based actions)
    all_coords = [r["coordinates"] for r in results if r.get("coordinates")]
    coord_stats = {}
    if len(all_coords) >= 2:
        import numpy as np
        coords_array = np.array(all_coords)
        coord_stats = {
            "x_std": float(np.std(coords_array[:, 0])),
            "y_std": float(np.std(coords_array[:, 1])),
            "mean_std": float((np.std(coords_array[:, 0]) + np.std(coords_array[:, 1])) / 2),
            "n_coord_samples": len(all_coords),
        }

    # Oracle: any sample correct?
    oracle_function_match = any(r.get("function_match", False) for r in results)
    oracle_args_match = any(r.get("args_match", False) for r in results)

    step_result = {
        "sample_id": sample["sample_id"],
        "trajectory_id": sample["trajectory_id"],
        "domain": sample["domain"],
        "category": sample["category"],
        "step_index": sample["step_index"],
        "total_steps": sample["total_steps"],
        "gt_function": sample["gt_function"],
        # Greedy results
        "greedy_function_match": greedy.get("function_match", False),
        "greedy_args_match": greedy.get("args_match", False),
        "greedy_pred_function": greedy.get("pred_function"),
        # Multi-sample stats
        "agreement_rate": agreement_rate,
        "action_entropy": entropy,
        "most_common_function": most_common_func,
        "coord_stats": coord_stats,
        "oracle_function_match": oracle_function_match,
        "oracle_args_match": oracle_args_match,
        # Temperature samples accuracy
        "temp_function_match_rate": sum(1 for r in temp_samples if r.get("function_match", False)) / max(len(temp_samples), 1),
        "temp_args_match_rate": sum(1 for r in temp_samples if r.get("args_match", False)) / max(len(temp_samples), 1),
        # All K samples detail (for downstream analysis)
        "K_predictions": [{
            "pred_function": r.get("pred_function"),
            "function_match": r.get("function_match", False),
            "args_match": r.get("args_match", False),
            "coordinates": r.get("coordinates"),
        } for r in results],
    }

    return step_result


def main():
    global _model

    parser = argparse.ArgumentParser(description="GUI-360 Multi-Sample Collection")
    parser.add_argument("--root_dir", type=str,
                        default=os.path.join(PROJECT_DIR, "datasets/GUI-360/test"))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(PROJECT_DIR, "outputs/eval_gui360_multisample"))
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--api_url", type=str, default="http://localhost:19815/v1")
    parser.add_argument("--K", type=int, default=10, help="Number of samples per step")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_workers", type=int, default=48)
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "multisample_results.jsonl")

    # Load data
    print(f"Loading GUI-360 data from {args.root_dir}...")
    samples = load_gui360_samples(args.root_dir, max_samples=args.max_samples)
    print(f"Loaded {len(samples)} samples")

    # Resume support
    completed_ids = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed_ids.add(r.get("sample_id"))
                except json.JSONDecodeError:
                    pass
        print(f"Resuming: {len(completed_ids)} samples already completed")
        samples = [s for s in samples if s["sample_id"] not in completed_ids]
        print(f"Remaining: {len(samples)} samples")
    elif os.path.exists(out_path):
        os.remove(out_path)

    if not samples:
        print("No samples to process")
        return

    # Initialize model
    print(f"Initializing model from {args.model_name}...")
    _model = init_model(args.api_url, args.model_name)
    print("Model initialized")

    # Process samples
    print(f"Collecting K={args.K} samples per step, temperature={args.temperature}")
    print(f"Max workers: {args.max_workers}")

    completed = len(completed_ids)
    total = completed + len(samples)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_sample, s, args): s for s in samples}
        for future in as_completed(futures):
            try:
                result = future.result()
                completed += 1

                with result_lock:
                    with open(out_path, "a") as f:
                        f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")

                if completed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (completed - len(completed_ids)) / elapsed if elapsed > 0 else 0
                    eta = (total - completed) / rate if rate > 0 else 0
                    print(f"Progress: {completed}/{total} ({rate:.1f} samples/s, ETA: {eta/60:.0f}m)")
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()

    # Summary
    print(f"\nDone! {completed}/{total} samples processed")

    # Load all results and compute summary
    all_results = []
    with open(out_path) as f:
        for line in f:
            try:
                all_results.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    n = len(all_results)
    greedy_func = sum(1 for r in all_results if r["greedy_function_match"]) / n
    greedy_args = sum(1 for r in all_results if r["greedy_args_match"]) / n
    oracle_func = sum(1 for r in all_results if r["oracle_function_match"]) / n
    oracle_args = sum(1 for r in all_results if r["oracle_args_match"]) / n
    temp_func = sum(r["temp_function_match_rate"] for r in all_results) / n
    temp_args = sum(r["temp_args_match_rate"] for r in all_results) / n
    mean_agree = sum(r["agreement_rate"] for r in all_results) / n
    mean_entropy = sum(r["action_entropy"] for r in all_results) / n

    summary = {
        "total_samples": n,
        "K": args.K,
        "temperature": args.temperature,
        "greedy_function_match": greedy_func,
        "greedy_args_match": greedy_args,
        "temp_function_match_rate": temp_func,
        "temp_args_match_rate": temp_args,
        "oracle_function_match": oracle_func,
        "oracle_args_match": oracle_args,
        "delta_temp_function": greedy_func - temp_func,
        "delta_temp_args": greedy_args - temp_args,
        "mean_agreement_rate": mean_agree,
        "mean_action_entropy": mean_entropy,
        "domain_breakdown": {},
    }

    # Per-domain breakdown
    by_domain = defaultdict(list)
    for r in all_results:
        by_domain[r["domain"]].append(r)

    for domain, domain_results in by_domain.items():
        dn = len(domain_results)
        summary["domain_breakdown"][domain] = {
            "n": dn,
            "greedy_function_match": sum(1 for r in domain_results if r["greedy_function_match"]) / dn,
            "greedy_args_match": sum(1 for r in domain_results if r["greedy_args_match"]) / dn,
            "temp_function_match_rate": sum(r["temp_function_match_rate"] for r in domain_results) / dn,
            "temp_args_match_rate": sum(r["temp_args_match_rate"] for r in domain_results) / dn,
            "oracle_function_match": sum(1 for r in domain_results if r["oracle_function_match"]) / dn,
            "oracle_args_match": sum(1 for r in domain_results if r["oracle_args_match"]) / dn,
        }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {n}")
    print(f"Greedy function_match: {greedy_func:.4f}")
    print(f"Greedy args_match:     {greedy_args:.4f}")
    print(f"Temp function_match:   {temp_func:.4f}")
    print(f"Temp args_match:       {temp_args:.4f}")
    print(f"Oracle function_match: {oracle_func:.4f}")
    print(f"Oracle args_match:     {oracle_args:.4f}")
    print(f"")
    print(f"Δ_temp (function):     {summary['delta_temp_function']:+.4f}")
    print(f"Δ_temp (args):         {summary['delta_temp_args']:+.4f}")
    print(f"Mean agreement:        {mean_agree:.4f}")
    print(f"Mean entropy:          {mean_entropy:.4f}")
    print(f"{'='*60}")
    print(f"Results: {out_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
