import argparse
import contextlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR / "repo/eval/android_control"))
from eval import Evaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    if args.require_complete and len(rows) != 7708:
        raise ValueError(f"complete score requires 7708 rows, found {len(rows)}")
    evaluator = Evaluator()
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        for row in rows:
            evaluator.evaluate_prediction(row)

    totals = defaultdict(lambda: defaultdict(int))
    categories = {}
    for category, metrics in evaluator.scores.items():
        categories[category] = {}
        for metric, counts in metrics.items():
            correct = counts.get("correct", 0)
            total = counts.get("total", 0)
            categories[category][metric] = {"correct": correct, "total": total}
            totals[metric]["correct"] += correct
            totals[metric]["total"] += total
    result = {
        "coverage": "COMPLETE" if len(rows) == 7708 else "PARTIAL",
        "rows": len(rows),
        "metrics": {},
        "categories": categories,
    }
    for metric in ("action", "grounding", "text", "step_success"):
        correct = totals[metric]["correct"]
        total = totals[metric]["total"]
        result["metrics"][metric] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
        }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()