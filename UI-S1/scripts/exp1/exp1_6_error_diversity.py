#!/usr/bin/env python3
"""
Experiment 1.6: Error Diversity Analysis (SFT v2 vs SFT v3)

Question: Do the two models make complementary errors, or do they fail on the
same samples?

Method:
  Analyze per-sample correctness from existing grounding eval results.
  This is a CPU-only analysis—no GPU/vLLM needed.

Key metric: Oracle ensemble accuracy = at least one model correct

Expected:
  Both correct:  ~65%
  V3 only:       ~15%
  V2 only:       ~5%
  Both wrong:    ~15%
  → Oracle ensemble ≈ 85%+

Success criteria: Oracle ensemble > max(V2, V3) by ≥5pp

Usage:
    # From existing GUI-360 eval result JSONs:
    python scripts/exp1/exp1_6_error_diversity.py \
        --v3_results_dir train_GUI_360/GUI-360-eval/results/grounding_sft_v3_final \
        --v2_results_dir train_GUI_360/GUI-360-eval/results/full_sft_v2

    # Or from Exp 1.1/1.2 multi-sample results:
    python scripts/exp1/exp1_6_error_diversity.py \
        --v3_exp1_results outputs/exp1_1/results_K1.jsonl \
        --v2_exp1_results outputs/exp1_2/results_K1.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_gui360_eval_results(results_dir: str) -> dict[str, bool]:
    """Load per-sample results from GUI-360 eval output.

    Returns: {sample_id: success_bool}
    """
    results_dir = Path(results_dir)
    sample_results = {}

    # Find the evaluation results JSON files
    for json_file in results_dir.rglob("evaluation_results_*.json"):
        with open(json_file) as f:
            data = json.load(f)

        for result in data.get("detailed_results", []):
            sample_id = result.get("sample_id", "")
            success = result.get("success", False)
            sample_results[sample_id] = success

    if not sample_results:
        # Try evaluation_summary files
        for json_file in results_dir.rglob("evaluation_summary_*.json"):
            print(f"Found summary but no detailed results: {json_file}")

    return sample_results


def load_exp1_results(results_path: str) -> dict[int, bool]:
    """Load per-sample results from Exp 1.1/1.2 JSONL.

    Returns: {sample_idx: greedy_correct}
    """
    sample_results = {}
    with open(results_path) as f:
        for line in f:
            r = json.loads(line)
            sample_results[r["sample_id"]] = r["greedy_correct"]
    return sample_results


def analyze_diversity(v3_results: dict, v2_results: dict, v3_label="SFT v3", v2_label="SFT v2"):
    """Analyze error diversity between two models."""
    # Find common samples
    common_keys = set(v3_results.keys()) & set(v2_results.keys())
    if not common_keys:
        print("ERROR: No common samples found between the two result sets!")
        print(f"  V3 keys sample: {list(v3_results.keys())[:5]}")
        print(f"  V2 keys sample: {list(v2_results.keys())[:5]}")
        return None

    n = len(common_keys)

    # Count categories
    both_correct = sum(1 for k in common_keys if v3_results[k] and v2_results[k])
    v3_only = sum(1 for k in common_keys if v3_results[k] and not v2_results[k])
    v2_only = sum(1 for k in common_keys if not v3_results[k] and v2_results[k])
    both_wrong = sum(1 for k in common_keys if not v3_results[k] and not v2_results[k])

    v3_acc = sum(1 for k in common_keys if v3_results[k]) / n
    v2_acc = sum(1 for k in common_keys if v2_results[k]) / n
    oracle_acc = (both_correct + v3_only + v2_only) / n
    best_single = max(v3_acc, v2_acc)

    print("\n" + "=" * 70)
    print("  Experiment 1.6: Error Diversity Analysis")
    print("=" * 70)

    print(f"\n  Common samples: {n}")
    print(f"\n  Individual Accuracy:")
    print(f"    {v3_label}: {v3_acc:.1%} ({sum(1 for k in common_keys if v3_results[k])}/{n})")
    print(f"    {v2_label}: {v2_acc:.1%} ({sum(1 for k in common_keys if v2_results[k])}/{n})")

    print(f"\n  Error Diversity Matrix:")
    print(f"    {'':>20s}  {v2_label} ✓   {v2_label} ✗")
    print(f"    {v3_label} ✓    {both_correct:>6d} ({both_correct/n:.1%})  {v3_only:>6d} ({v3_only/n:.1%})")
    print(f"    {v3_label} ✗    {v2_only:>6d} ({v2_only/n:.1%})  {both_wrong:>6d} ({both_wrong/n:.1%})")

    print(f"\n  Oracle Ensemble (at least one correct): {oracle_acc:.1%}")
    print(f"  Best single model:                       {best_single:.1%}")
    print(f"  Ensemble gain:                           {oracle_acc - best_single:+.1%}")

    # Complementarity score
    # How many errors of the worse model does the better model cover?
    if v3_acc >= v2_acc:
        stronger, weaker = v3_label, v2_label
        weaker_errors = v2_only + both_wrong
        covered = v3_only  # Errors in v2 that v3 catches
        # Actually: v3 catches (v3_only + both_correct) where v2 would fail at (v2_only + both_wrong)
        # Complementarity = v3_only / (v2_only + both_wrong) — fraction of v2 errors that v3 alone fixes
        # Wait, let me think again. V2 errors = v3_only + both_wrong (samples where v2 is wrong)
        # Of those, v3 covers v3_only.
        v2_error_count = v3_only + both_wrong
        v3_covers_v2_errors = v3_only
    else:
        stronger, weaker = v2_label, v3_label
        v3_error_count = v2_only + both_wrong
        v2_covers_v3_errors = v2_only

    if v3_acc >= v2_acc:
        complementarity = v3_covers_v2_errors / v2_error_count if v2_error_count > 0 else 0
        print(f"\n  Complementarity: {stronger} covers {v3_covers_v2_errors}/{v2_error_count} "
              f"({complementarity:.0%}) of {weaker}'s errors")
    else:
        complementarity = v2_covers_v3_errors / v3_error_count if v3_error_count > 0 else 0
        print(f"\n  Complementarity: {stronger} covers {v2_covers_v3_errors}/{v3_error_count} "
              f"({complementarity:.0%}) of {weaker}'s errors")

    # Cohen's kappa for agreement
    po = (both_correct + both_wrong) / n  # observed agreement
    pe = (v3_acc * v2_acc + (1 - v3_acc) * (1 - v2_acc))  # expected by chance
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    print(f"\n  Cohen's Kappa (error correlation): {kappa:.3f}")
    if kappa < 0.4:
        print(f"    → Low agreement: models make different errors (good for ensemble)")
    elif kappa < 0.6:
        print(f"    → Moderate agreement: some overlap in errors")
    else:
        print(f"    → High agreement: errors are correlated (limited ensemble benefit)")

    # Go/No-Go
    ensemble_gain = oracle_acc - best_single
    print(f"\n  GO/NO-GO CHECK:")
    if ensemble_gain >= 0.05:
        print(f"    ✓ PASS: Oracle ensemble gain = {ensemble_gain:+.1%} ≥ 5pp")
    elif ensemble_gain > 0:
        print(f"    ~ MARGINAL: ensemble gain = {ensemble_gain:+.1%} (0-5pp)")
    else:
        print(f"    ✗ FAIL: no ensemble gain → errors too correlated")

    summary = {
        "n_common_samples": n,
        "v3_accuracy": v3_acc,
        "v2_accuracy": v2_acc,
        "oracle_ensemble_accuracy": oracle_acc,
        "ensemble_gain": ensemble_gain,
        "both_correct": both_correct,
        "v3_only_correct": v3_only,
        "v2_only_correct": v2_only,
        "both_wrong": both_wrong,
        "cohens_kappa": kappa,
    }

    print("=" * 70)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Exp 1.6: Error Diversity Analysis")
    parser.add_argument("--v3_results_dir", type=str, default="",
                        help="GUI-360 eval results dir for SFT v3")
    parser.add_argument("--v2_results_dir", type=str, default="",
                        help="GUI-360 eval results dir for SFT v2")
    parser.add_argument("--v3_exp1_results", type=str, default="",
                        help="Exp 1.1 JSONL results for SFT v3 (alternative)")
    parser.add_argument("--v2_exp1_results", type=str, default="",
                        help="Exp 1.2 JSONL results for SFT v2 (alternative)")
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "exp1_6"))

    args = parser.parse_args()

    # Load results from either GUI-360 eval or Exp 1.x
    if args.v3_exp1_results and args.v2_exp1_results:
        v3_results = load_exp1_results(args.v3_exp1_results)
        v2_results = load_exp1_results(args.v2_exp1_results)
    elif args.v3_results_dir and args.v2_results_dir:
        v3_results = load_gui360_eval_results(args.v3_results_dir)
        v2_results = load_gui360_eval_results(args.v2_results_dir)
    else:
        # Try default paths
        v3_default = PROJECT_ROOT / "train_GUI_360" / "GUI-360-eval" / "results" / "grounding_sft_v3_final"
        v2_default = PROJECT_ROOT / "train_GUI_360" / "GUI-360-eval" / "results" / "full_sft_v2"

        if v3_default.exists() and v2_default.exists():
            print(f"Using default result dirs:")
            print(f"  V3: {v3_default}")
            print(f"  V2: {v2_default}")
            v3_results = load_gui360_eval_results(str(v3_default))
            v2_results = load_gui360_eval_results(str(v2_default))
        else:
            # Try exp1 results
            v3_exp1 = PROJECT_ROOT / "outputs" / "exp1_1" / "results_K5.jsonl"
            v2_exp1 = PROJECT_ROOT / "outputs" / "exp1_2" / "results_K5.jsonl"
            if v3_exp1.exists() and v2_exp1.exists():
                v3_results = load_exp1_results(str(v3_exp1))
                v2_results = load_exp1_results(str(v2_exp1))
            else:
                print("ERROR: No result files found. Provide paths via --v3_results_dir/--v2_results_dir "
                      "or --v3_exp1_results/--v2_exp1_results")
                sys.exit(1)

    print(f"V3 results: {len(v3_results)} samples")
    print(f"V2 results: {len(v2_results)} samples")

    summary = analyze_diversity(v3_results, v2_results)

    if summary:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
