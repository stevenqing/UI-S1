#!/usr/bin/env python3
"""
Experiment 1.2: SFT v2 Multi-Sample Grounding Upper Bound (Comparison)

Same methodology as Exp 1.1 but using SFT v2 (all-round model).
Key question: does a stronger base model (80% vs 70%) benefit more from multi-sampling?

Key comparisons:
  - SFT v3 K=5 cluster vs SFT v2 K=5 cluster
  - If SFT v3 K=5 > SFT v2 K=10 → base accuracy > sampling count

Usage:
    python scripts/exp1/exp1_2_sft_v2_multisample.py \
        --endpoint http://localhost:19816/v1 --K 1

    python scripts/exp1/exp1_2_sft_v2_multisample.py \
        --endpoint http://localhost:19816/v1 --K 5 --temperature 0.7

    python scripts/exp1/exp1_2_sft_v2_multisample.py \
        --analyze_only --results_dir outputs/exp1_2
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
    """Run K-sampling grounding eval on GUI-360 test set with SFT v2."""
    from openai import OpenAI

    print("Loading grounding samples from GUI-360 test set...")
    samples = load_grounding_samples(max_samples=args.num_samples)
    print(f"Loaded {len(samples)} grounding samples")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"results_K{args.K}.jsonl"

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


def analyze_and_compare(exp1_1_dir: str, exp1_2_dir: str):
    """Compare SFT v3 vs SFT v2 multi-sampling results."""
    exp1_1_dir = Path(exp1_1_dir)
    exp1_2_dir = Path(exp1_2_dir)

    print("\n" + "=" * 70)
    print("  Experiment 1.2: SFT v2 vs SFT v3 Multi-Sample Comparison")
    print("=" * 70)

    for model_label, rdir in [("SFT v3", exp1_1_dir), ("SFT v2", exp1_2_dir)]:
        print(f"\n  --- {model_label} ---")
        for fpath in sorted(rdir.glob("results_K*.jsonl")):
            K = int(fpath.stem.split("_K")[1])
            results = [json.loads(line) for line in open(fpath)]
            n = len(results)
            if n == 0:
                continue
            greedy = sum(1 for r in results if r["greedy_correct"]) / n
            best_k = sum(1 for r in results if r["best_of_k_correct"]) / n
            cluster = sum(1 for r in results if r["cluster_center_correct"]) / n
            print(f"    K={K:>2d} (N={n}):  greedy={greedy:.1%}  best-of-K={best_k:.1%}  cluster={cluster:.1%}")

    # Cross-model comparison
    print(f"\n  --- Head-to-Head ---")
    for fpath_v3 in sorted(exp1_1_dir.glob("results_K*.jsonl")):
        K = int(fpath_v3.stem.split("_K")[1])
        fpath_v2 = exp1_2_dir / f"results_K{K}.jsonl"
        if not fpath_v2.exists():
            continue

        v3 = {json.loads(l)["sample_id"]: json.loads(l) for l in open(fpath_v3)}
        v2 = {json.loads(l)["sample_id"]: json.loads(l) for l in open(fpath_v2)}
        common = set(v3.keys()) & set(v2.keys())

        if not common:
            continue

        v3_cluster = sum(1 for sid in common if v3[sid]["cluster_center_correct"]) / len(common)
        v2_cluster = sum(1 for sid in common if v2[sid]["cluster_center_correct"]) / len(common)
        delta = v3_cluster - v2_cluster

        print(f"    K={K} (N={len(common)} common):  V3={v3_cluster:.1%}  V2={v2_cluster:.1%}  delta={delta:+.1%}")
        if delta > 0.05:
            print(f"    -> SFT v3 significantly better: base accuracy matters more than K")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Exp 1.2: SFT v2 Multi-Sample Grounding")
    parser.add_argument("--model_name", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/llamafactory/output/gui360_full_sft_v2")
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cluster_eps", type=float, default=30.0)
    parser.add_argument("--endpoint", type=str, default="http://localhost:19816/v1")
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "exp1_2"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of concurrent workers for parallel processing")
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--results_dir", type=str, default="")
    parser.add_argument("--exp1_1_dir", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "exp1_1"))

    args = parser.parse_args()

    if args.analyze_only:
        analyze_and_compare(args.exp1_1_dir, args.results_dir or args.output_dir)
    else:
        run_grounding_multisample(args)
        analyze_and_compare(args.exp1_1_dir, args.output_dir)


if __name__ == "__main__":
    main()
