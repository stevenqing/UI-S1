#!/usr/bin/env python3
"""
Experiment 0.1: Model Uncertainty Analysis (K=10 sampling)

Run the eval model K times on test steps with temperature>0 to study
coordinate prediction uncertainty.

For each step compute:
- Coordinate variance σ(x), σ(y)
- Whether any sample's coordinate is correct (best-of-K)
- Self-consistency accuracy (DBSCAN cluster center correct?)
- Distribution shape (unimodal vs. multimodal)

Success criteria: Best-of-K args_match >> Best-of-1 → multi-sampling works

Usage:
    # Start vLLM server first, then:
    python scripts/exp0/exp0_1_uncertainty_analysis.py \
        --model_name gui360_lora_v4_ckpt354 \
        --num_samples 200 \
        --K 10 \
        --temperature 0.7

    # Analysis only (from saved results):
    python scripts/exp0/exp0_1_uncertainty_analysis.py --analyze_only --results_path outputs/exp0_1/results.jsonl
"""

import argparse
import base64
import json
import os
import re
import sys
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.exp0.data_utils import PARQUET_EVAL_PATH, is_coord_in_bbox


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def parse_tool_call(text: str) -> dict | None:
    """Parse tool_call from model response."""
    try:
        match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return json.loads(text)
    except Exception:
        return None


def extract_coordinate(action: dict | None) -> list | None:
    """Extract [x, y] coordinate from parsed action. Returns None if invalid."""
    if action is None:
        return None
    args = action.get("args", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            return None
    if not isinstance(args, dict):
        return None
    coord = args.get("coordinate", [])
    if isinstance(coord, list) and len(coord) == 2:
        # Ensure both values are numeric (not None)
        try:
            x, y = float(coord[0]), float(coord[1])
            return [x, y]
        except (TypeError, ValueError):
            return None
    return None


def image_to_data_url(image_path: str) -> str:
    """Convert image file to base64 data URL."""
    from PIL import Image

    image = Image.open(image_path)
    buf = BytesIO()
    image.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64_str}"


def convert_message_to_openai_format(messages: list, base_dir: str) -> list:
    """Convert messages to OpenAI API format with base64 images."""
    openai_messages = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            openai_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
        elif isinstance(content, list):
            new_content = []
            for item in content:
                if isinstance(item, str):
                    new_content.append({"type": "text", "text": item})
                elif isinstance(item, dict):
                    if "text" in item:
                        new_content.append({"type": "text", "text": item["text"]})
                    elif "image" in item:
                        image_path = item["image"]
                        if not os.path.isabs(image_path):
                            image_path = os.path.join(base_dir, image_path)
                        if os.path.exists(image_path):
                            data_url = image_to_data_url(image_path)
                            new_content.append({"type": "image_url", "image_url": {"url": data_url}})
            openai_messages.append({"role": role, "content": new_content})
    return openai_messages


def call_model_k_times(
    messages: list, model_name: str, K: int, temperature: float, endpoint: str
) -> list[str]:
    """Call model K times with temperature>0 to get diverse samples."""
    from openai import OpenAI

    client = OpenAI(api_key="EMPTY", base_url=endpoint, timeout=300)
    responses = []

    for _ in range(K):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                extra_body={"top_k": 50},  # Allow diverse sampling
            )
            responses.append(response.choices[0].message.content)
        except Exception as e:
            print(f"API call failed: {e}")
            responses.append("")

    return responses


def cluster_coordinates(coords: list[list], eps: float = 30.0, min_samples: int = 1) -> dict:
    """
    Cluster K coordinate predictions using DBSCAN.

    Returns:
        {
            'cluster_center': [x, y] of largest cluster center,
            'num_clusters': int,
            'largest_cluster_size': int,
            'agreement_rate': float (largest_cluster_size / total),
            'all_clusters': [{center, size, coords}],
            'coord_std': [std_x, std_y],
            'is_multimodal': bool,
        }
    """
    if not coords:
        return {
            "cluster_center": None,
            "num_clusters": 0,
            "largest_cluster_size": 0,
            "agreement_rate": 0.0,
            "all_clusters": [],
            "coord_std": [float("inf"), float("inf")],
            "is_multimodal": False,
        }

    coords_arr = np.array(coords)
    coord_std = coords_arr.std(axis=0).tolist()

    if len(coords) == 1:
        return {
            "cluster_center": coords[0],
            "num_clusters": 1,
            "largest_cluster_size": 1,
            "agreement_rate": 1.0,
            "all_clusters": [{"center": coords[0], "size": 1, "coords": coords}],
            "coord_std": coord_std,
            "is_multimodal": False,
        }

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_arr)
    labels = clustering.labels_
    unique_labels = set(labels) - {-1}

    clusters = []
    for label in unique_labels:
        mask = labels == label
        cluster_coords = coords_arr[mask]
        center = cluster_coords.mean(axis=0).tolist()
        clusters.append({
            "center": center,
            "size": int(mask.sum()),
            "coords": cluster_coords.tolist(),
        })

    # Handle noise points
    noise_mask = labels == -1
    noise_count = int(noise_mask.sum())

    clusters.sort(key=lambda c: c["size"], reverse=True)

    largest = clusters[0] if clusters else None
    return {
        "cluster_center": largest["center"] if largest else None,
        "num_clusters": len(clusters),
        "largest_cluster_size": largest["size"] if largest else 0,
        "agreement_rate": largest["size"] / len(coords) if largest else 0.0,
        "noise_count": noise_count,
        "all_clusters": clusters,
        "coord_std": coord_std,
        "is_multimodal": len(clusters) >= 2 and clusters[1]["size"] >= 2,
    }


def evaluate_coord(pred_coord, gt_action: dict, threshold: float = 50.0) -> dict:
    """Evaluate a predicted coordinate against ground truth."""
    if pred_coord is None:
        return {"correct": False, "distance": float("inf"), "bbox_hit": False}

    gt_args = gt_action.get("args", {})
    gt_coord = gt_args.get("coordinate", [])
    gt_bbox = gt_action.get("bbox")

    distance = float("inf")
    if isinstance(gt_coord, list) and len(gt_coord) == 2:
        distance = np.sqrt((pred_coord[0] - gt_coord[0]) ** 2 + (pred_coord[1] - gt_coord[1]) ** 2)

    bbox_hit = is_coord_in_bbox(pred_coord, gt_bbox) if gt_bbox else False
    correct = bbox_hit or (distance < threshold)

    return {"correct": correct, "distance": distance, "bbox_hit": bbox_hit}


def run_sampling(args):
    """Run K-sampling on test data."""
    print(f"Loading parquet: {args.parquet_file}")
    df = pd.read_parquet(args.parquet_file)

    # Filter to click actions only (coordinate-based)
    samples = []
    for idx, row in df.iterrows():
        msgs = json.loads(row["messages"])
        gt_response = msgs[1]["content"]
        gt_action = parse_tool_call(gt_response)
        if gt_action and gt_action.get("function") == "click":
            samples.append((idx, msgs, gt_action))
        if args.num_samples > 0 and len(samples) >= args.num_samples:
            break

    print(f"Selected {len(samples)} click-action samples for K={args.K} sampling")

    base_dir = str(PROJECT_ROOT)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"

    # Clear previous results
    if results_path.exists():
        results_path.unlink()

    for i, (idx, msgs, gt_action) in enumerate(samples):
        user_msg = msgs[0]
        openai_msgs = convert_message_to_openai_format([user_msg], base_dir)

        # Call model K times
        responses = call_model_k_times(
            openai_msgs, args.model_name, args.K, args.temperature, args.endpoint
        )

        # Parse coordinates from all responses
        all_coords = []
        all_actions = []
        for resp in responses:
            action = parse_tool_call(resp)
            coord = extract_coordinate(action)
            all_actions.append(action)
            if coord is not None:
                all_coords.append(coord)

        # Cluster analysis
        cluster_result = cluster_coordinates(all_coords, eps=args.cluster_eps)

        # Evaluate each sample
        sample_evals = []
        for coord in all_coords:
            ev = evaluate_coord(coord, gt_action)
            sample_evals.append(ev)

        # Evaluate cluster center
        center_eval = evaluate_coord(cluster_result["cluster_center"], gt_action)

        # Best-of-K: any sample correct?
        best_of_k = any(ev["correct"] for ev in sample_evals)
        # Best distance
        best_distance = min((ev["distance"] for ev in sample_evals), default=float("inf"))

        result = {
            "sample_idx": int(idx),
            "gt_function": gt_action.get("function", ""),
            "gt_coordinate": gt_action.get("args", {}).get("coordinate"),
            "gt_bbox": gt_action.get("bbox"),
            "K": args.K,
            "num_valid_coords": len(all_coords),
            "all_coords": all_coords,
            # Per-sample evaluation
            "greedy_correct": sample_evals[0]["correct"] if sample_evals else False,
            "greedy_distance": sample_evals[0]["distance"] if sample_evals else float("inf"),
            "best_of_k_correct": best_of_k,
            "best_of_k_distance": best_distance,
            # Cluster-based evaluation
            "cluster_center_correct": center_eval["correct"],
            "cluster_center_distance": center_eval["distance"],
            "num_clusters": cluster_result["num_clusters"],
            "agreement_rate": cluster_result["agreement_rate"],
            "is_multimodal": cluster_result["is_multimodal"],
            "coord_std_x": cluster_result["coord_std"][0],
            "coord_std_y": cluster_result["coord_std"][1],
        }

        with open(results_path, "a") as f:
            f.write(json.dumps(result, default=_json_default) + "\n")

        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(samples)}")

    print(f"Results saved to: {results_path}")
    return results_path


def analyze_results(results_path: str):
    """Analyze K-sampling results and print summary."""
    results = []
    with open(results_path) as f:
        for line in f:
            results.append(json.loads(line))

    if not results:
        print("No results to analyze!")
        return

    n = len(results)
    print("\n" + "=" * 60)
    print(f"  Experiment 0.1: Uncertainty Analysis Results (N={n})")
    print("=" * 60)

    # 1. Accuracy comparison
    greedy_acc = sum(1 for r in results if r["greedy_correct"]) / n
    best_of_k_acc = sum(1 for r in results if r["best_of_k_correct"]) / n
    cluster_acc = sum(1 for r in results if r["cluster_center_correct"]) / n
    K = results[0]["K"]

    print(f"\n  Accuracy Comparison (K={K}):")
    print(f"    Greedy (first sample):      {greedy_acc:.1%}  ({sum(1 for r in results if r['greedy_correct'])}/{n})")
    print(f"    Best-of-K (oracle):         {best_of_k_acc:.1%}  ({sum(1 for r in results if r['best_of_k_correct'])}/{n})")
    print(f"    Self-consistency (cluster):  {cluster_acc:.1%}  ({sum(1 for r in results if r['cluster_center_correct'])}/{n})")
    print(f"    Improvement (best-of-K):    +{(best_of_k_acc - greedy_acc):.1%}")
    print(f"    Improvement (cluster):      +{(cluster_acc - greedy_acc):.1%}")

    # 2. Distance statistics
    greedy_dists = [r["greedy_distance"] for r in results if r["greedy_distance"] < float("inf")]
    best_dists = [r["best_of_k_distance"] for r in results if r["best_of_k_distance"] < float("inf")]
    cluster_dists = [r["cluster_center_distance"] for r in results if r["cluster_center_distance"] < float("inf")]

    print(f"\n  Coordinate Distance (pixels):")
    if greedy_dists:
        print(f"    Greedy mean:     {np.mean(greedy_dists):.1f}  median: {np.median(greedy_dists):.1f}")
    if best_dists:
        print(f"    Best-of-K mean:  {np.mean(best_dists):.1f}  median: {np.median(best_dists):.1f}")
    if cluster_dists:
        print(f"    Cluster mean:    {np.mean(cluster_dists):.1f}  median: {np.median(cluster_dists):.1f}")

    # 3. Uncertainty statistics
    std_x = [r["coord_std_x"] for r in results if r["coord_std_x"] < float("inf")]
    std_y = [r["coord_std_y"] for r in results if r["coord_std_y"] < float("inf")]
    agreement = [r["agreement_rate"] for r in results]
    multimodal = sum(1 for r in results if r["is_multimodal"])

    print(f"\n  Uncertainty Statistics:")
    if std_x:
        print(f"    Mean σ(x): {np.mean(std_x):.1f}px  σ(y): {np.mean(std_y):.1f}px")
        print(f"    Median σ(x): {np.median(std_x):.1f}px  σ(y): {np.median(std_y):.1f}px")
    print(f"    Mean agreement rate: {np.mean(agreement):.2f}")
    print(f"    Multimodal predictions: {multimodal}/{n} ({multimodal / n:.1%})")

    # 4. Uncertainty vs. accuracy correlation
    # Split by low/high variance
    median_std = np.median([r["coord_std_x"] + r["coord_std_y"] for r in results
                           if r["coord_std_x"] < float("inf")])
    low_var = [r for r in results if r["coord_std_x"] + r["coord_std_y"] <= median_std
               and r["coord_std_x"] < float("inf")]
    high_var = [r for r in results if r["coord_std_x"] + r["coord_std_y"] > median_std
                and r["coord_std_x"] < float("inf")]

    if low_var and high_var:
        low_acc = sum(1 for r in low_var if r["cluster_center_correct"]) / len(low_var)
        high_acc = sum(1 for r in high_var if r["cluster_center_correct"]) / len(high_var)
        print(f"\n  Uncertainty vs. Accuracy:")
        print(f"    Low variance (≤{median_std:.0f}px):  cluster_acc={low_acc:.1%} (n={len(low_var)})")
        print(f"    High variance (>{median_std:.0f}px): cluster_acc={high_acc:.1%} (n={len(high_var)})")

    # 5. Agreement rate vs. accuracy
    high_agree = [r for r in results if r["agreement_rate"] >= 0.6]
    low_agree = [r for r in results if r["agreement_rate"] < 0.6]
    if high_agree and low_agree:
        ha_acc = sum(1 for r in high_agree if r["cluster_center_correct"]) / len(high_agree)
        la_acc = sum(1 for r in low_agree if r["cluster_center_correct"]) / len(low_agree)
        print(f"\n  Agreement Rate vs. Accuracy:")
        print(f"    High agreement (≥0.6):  acc={ha_acc:.1%} (n={len(high_agree)})")
        print(f"    Low agreement (<0.6):   acc={la_acc:.1%} (n={len(low_agree)})")

    # 6. Adaptive K suggestion
    print(f"\n  Adaptive K Suggestion:")
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        high_agree_n = sum(1 for r in results if r["agreement_rate"] >= threshold)
        if high_agree_n > 0:
            high_agree_acc = sum(1 for r in results
                                if r["agreement_rate"] >= threshold and r["cluster_center_correct"]) / high_agree_n
        else:
            high_agree_acc = 0
        print(f"    agreement≥{threshold:.1f}: {high_agree_n}/{n} samples ({high_agree_n/n:.0%}), "
              f"cluster_acc={high_agree_acc:.1%}")

    print("\n" + "=" * 60)

    # Save summary
    summary = {
        "n_samples": n,
        "K": K,
        "greedy_accuracy": greedy_acc,
        "best_of_k_accuracy": best_of_k_acc,
        "cluster_accuracy": cluster_acc,
        "mean_std_x": float(np.mean(std_x)) if std_x else None,
        "mean_std_y": float(np.mean(std_y)) if std_y else None,
        "mean_agreement_rate": float(np.mean(agreement)),
        "multimodal_fraction": multimodal / n,
    }
    summary_path = Path(results_path).parent / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 0.1: Model Uncertainty Analysis")
    parser.add_argument("--model_name", type=str, default="gui360_lora_v4_ckpt354")
    parser.add_argument("--parquet_file", type=str, default=str(PARQUET_EVAL_PATH))
    parser.add_argument("--num_samples", type=int, default=200, help="Number of test steps to sample")
    parser.add_argument("--K", type=int, default=10, help="Number of samples per step")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cluster_eps", type=float, default=30.0, help="DBSCAN epsilon in pixels")
    parser.add_argument("--endpoint", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "exp0_1"))
    parser.add_argument("--analyze_only", action="store_true", help="Only analyze existing results")
    parser.add_argument("--results_path", type=str, default="", help="Path to results for analysis")

    args = parser.parse_args()

    if args.analyze_only:
        path = args.results_path or str(Path(args.output_dir) / "results.jsonl")
        analyze_results(path)
    else:
        results_path = run_sampling(args)
        analyze_results(str(results_path))


if __name__ == "__main__":
    main()
