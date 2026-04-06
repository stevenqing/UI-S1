#!/usr/bin/env python3
"""
Experiment 1.1: SFT v3 Multi-Sample Grounding Upper Bound

Test: 80% grounder + multi-sampling → how high can we go?

Method:
  1. Start vLLM with Grounding SFT v3 final checkpoint
  2. On GUI-360 test grounding eval:
     - Greedy (K=1, temp=0.0)
     - K=5, temp=0.7: DBSCAN clustering + best-of-5
     - K=10, temp=0.7: DBSCAN clustering + best-of-10
  3. Record: greedy_acc, cluster_acc, best_of_k_acc, agreement_rate segmented analysis

Success criteria: K=5 cluster accuracy > 83% (≥3.5pp over greedy 79.48%)

Usage:
    python scripts/exp1/exp1_1_sft_v3_multisample.py \
        --endpoint http://localhost:19815/v1 --K 1

    python scripts/exp1/exp1_1_sft_v3_multisample.py \
        --endpoint http://localhost:19815/v1 --K 5 --temperature 0.7

    python scripts/exp1/exp1_1_sft_v3_multisample.py \
        --analyze_only --results_dir outputs/exp1_1
"""

import argparse
import json
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.exp1.grounding_utils import (
    call_grounding_k_times,
    call_grounding_once,
    cluster_coordinates,
    evaluate_grounding,
    load_grounding_samples,
)


def _json_default(o):
    if isinstance(o, (np.floating, float)):
        if np.isnan(o) or np.isinf(o):
            return str(o)
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    return str(o)


def run_grounding_multisample(args):
    """Run K-sampling grounding eval on GUI-360 test set."""
    from openai import OpenAI

    print("Loading grounding samples from GUI-360 test set...")
    samples = load_grounding_samples(max_samples=args.num_samples)
    print(f"Loaded {len(samples)} grounding samples")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"results_K{args.K}.jsonl"

    # Resume support
    completed_ids = set()
    if results_path.exists() and not args.overwrite:
        with open(results_path) as f:
            for line in f:
                completed_ids.add(json.loads(line)["sample_id"])
        print(f"Resuming: {len(completed_ids)} samples already completed")

    client = OpenAI(api_key="EMPTY", base_url=args.endpoint, timeout=300)
    total = len(samples)
    pending_samples = [s for s in samples if s["sample_id"] not in completed_ids]
    t0 = time.time()
    processed = 0
    write_lock = threading.Lock()

    def process_one(sample):
        """Process a single sample (greedy + K-sample)."""
        greedy_coord, greedy_text = call_grounding_once(
            client, args.model_name, sample, temperature=0.0
        )
        greedy_eval = evaluate_grounding(greedy_coord, sample["gt_rectangle"])

        if args.K > 1:
            k_results = call_grounding_k_times(
                client, args.model_name, sample,
                K=args.K, temperature=args.temperature
            )
        else:
            k_results = [(greedy_coord, greedy_text)]

        all_coords = [c for c, _ in k_results if c is not None]
        cluster_result = cluster_coordinates(all_coords, eps=args.cluster_eps)
        sample_evals = [evaluate_grounding(c, sample["gt_rectangle"]) for c in all_coords]
        center_eval = evaluate_grounding(cluster_result["cluster_center"], sample["gt_rectangle"])

        best_of_k = any(ev["correct"] for ev in sample_evals)
        best_distance = min((ev["distance"] for ev in sample_evals), default=float("inf"))

        return {
            "sample_id": sample["sample_id"],
            "domain": sample["domain"],
            "gt_function": sample["gt_function"],
            "gt_rectangle": sample["gt_rectangle"],
            "K": args.K,
            "num_valid_coords": len(all_coords),
            "all_coords": all_coords,
            "greedy_coord": greedy_coord,
            "greedy_correct": greedy_eval["correct"],
            "greedy_distance": greedy_eval["distance"],
            "best_of_k_correct": best_of_k,
            "best_of_k_distance": best_distance,
            "cluster_center": cluster_result["cluster_center"],
            "cluster_center_correct": center_eval["correct"],
            "cluster_center_distance": center_eval["distance"],
            "num_clusters": cluster_result["num_clusters"],
            "agreement_rate": cluster_result["agreement_rate"],
            "is_multimodal": cluster_result["is_multimodal"],
            "coord_std_x": cluster_result["coord_std"][0],
            "coord_std_y": cluster_result["coord_std"][1],
            "greedy_response": greedy_text,
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


def analyze_results(results_dir: str, K_values: list[int] | None = None):
    """Analyze multi-sample grounding results with detailed breakdowns."""
    results_dir = Path(results_dir)

    if K_values is None:
        K_values = sorted([
            int(f.stem.split("_K")[1])
            for f in results_dir.glob("results_K*.jsonl")
        ])

    if not K_values:
        print("No results found!")
        return

    print("\n" + "=" * 70)
    print("  Experiment 1.1: SFT v3 Multi-Sample Grounding Analysis")
    print("=" * 70)

    all_summaries = {}

    for K in K_values:
        results_path = results_dir / f"results_K{K}.jsonl"
        if not results_path.exists():
            print(f"\n  K={K}: no results file found")
            continue

        results = [json.loads(line) for line in open(results_path)]
        n = len(results)

        greedy_acc = sum(1 for r in results if r["greedy_correct"]) / n
        best_of_k_acc = sum(1 for r in results if r["best_of_k_correct"]) / n
        cluster_acc = sum(1 for r in results if r["cluster_center_correct"]) / n

        # Valid coord rate
        valid_rate = np.mean([r["num_valid_coords"] / max(r["K"], 1) for r in results])

        print(f"\n  K={K} (N={n}):")
        print(f"    Greedy accuracy:       {greedy_acc:.1%}")
        print(f"    Best-of-K (oracle):    {best_of_k_acc:.1%}  (+{best_of_k_acc - greedy_acc:.1%})")
        print(f"    Cluster (DBSCAN):      {cluster_acc:.1%}  (+{cluster_acc - greedy_acc:.1%})")
        print(f"    Valid coord rate:       {valid_rate:.1%}")

        # Agreement rate segmented analysis (key for Exp 1.7)
        if K > 1:
            print(f"\n    Agreement Rate Calibration:")
            agreement_bins = [
                (0.9, 1.01, ">=0.9"),
                (0.7, 0.9, "0.7-0.9"),
                (0.5, 0.7, "0.5-0.7"),
                (0.3, 0.5, "0.3-0.5"),
                (0.0, 0.3, "<0.3"),
            ]
            for lo, hi, label in agreement_bins:
                subset = [r for r in results if lo <= r["agreement_rate"] < hi]
                if subset:
                    acc = sum(1 for r in subset if r["cluster_center_correct"]) / len(subset)
                    print(f"      {label:>8s}: acc={acc:.1%}  (n={len(subset)}, {len(subset)/n:.0%} of total)")

        # Distance stats
        cluster_dists = [r["cluster_center_distance"] for r in results
                         if r["cluster_center_distance"] < float("inf")]
        if cluster_dists:
            print(f"\n    Distance (cluster->GT center):")
            print(f"      mean={np.mean(cluster_dists):.1f}px  median={np.median(cluster_dists):.1f}px")

        # Per-domain breakdown
        domains = sorted(set(r["domain"] for r in results))
        if len(domains) > 1:
            print(f"\n    Per-Domain:")
            for domain in domains:
                dr = [r for r in results if r["domain"] == domain]
                nd = len(dr)
                g = sum(1 for r in dr if r["greedy_correct"]) / nd
                c = sum(1 for r in dr if r["cluster_center_correct"]) / nd
                print(f"      {domain:>12s} (N={nd:>4d}): greedy={g:.1%}  cluster={c:.1%}  delta={c-g:+.1%}")

        # Multimodal detection
        multimodal_count = sum(1 for r in results if r["is_multimodal"])
        print(f"    Multimodal: {multimodal_count}/{n} ({multimodal_count/n:.0%})")

        all_summaries[str(K)] = {
            "n_samples": n,
            "greedy_accuracy": greedy_acc,
            "best_of_k_accuracy": best_of_k_acc,
            "cluster_accuracy": cluster_acc,
            "valid_coord_rate": float(valid_rate),
            "mean_agreement": float(np.mean([r["agreement_rate"] for r in results])),
            "multimodal_fraction": multimodal_count / n,
        }

    # Go/No-Go check
    if "5" in all_summaries:
        s = all_summaries["5"]
        print(f"\n  {'='*50}")
        print(f"  GO/NO-GO CHECK (Exp 1.1):")
        print(f"    K=5 cluster accuracy = {s['cluster_accuracy']:.1%}")
        if s["cluster_accuracy"] > 0.83:
            print(f"    PASS: > 83% threshold -> multi-sampling is effective")
        elif s["cluster_accuracy"] > 0.80:
            print(f"    MARGINAL: 80-83% -> some benefit but limited")
        else:
            print(f"    FAIL: < 80% -> multi-sampling insufficient")

    # Save summary
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n  Summary saved to: {summary_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Exp 1.1: SFT v3 Multi-Sample Grounding")
    parser.add_argument("--model_name", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/llamafactory/output/gui360_full_sft_v3_grounding")
    parser.add_argument("--num_samples", type=int, default=0, help="0=all samples")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cluster_eps", type=float, default=30.0)
    parser.add_argument("--endpoint", type=str, default="http://localhost:19815/v1")
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "exp1_1"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of concurrent workers for parallel processing")
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--results_dir", type=str, default="")

    args = parser.parse_args()

    if args.analyze_only:
        analyze_results(args.results_dir or args.output_dir)
    else:
        run_grounding_multisample(args)
        analyze_results(args.output_dir, [args.K])


if __name__ == "__main__":
    main()
