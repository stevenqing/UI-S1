#!/usr/bin/env python3
"""Choose a rescue-ranker threshold on dev utility and evaluate it once on test."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def utility(row: Mapping[str, Any]) -> int:
    return 1 if row["utility_outcome"] == "rescue" else -1 if row["utility_outcome"] == "regress" else 0


def stats(rows: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    selected = [row for row in rows if float(row["score"]) >= threshold]
    rescue = sum(row["utility_outcome"] == "rescue" for row in selected)
    regress = sum(row["utility_outcome"] == "regress" for row in selected)
    baseline_correct = sum(row["utility_outcome"] in {"regress", "both_correct"} for row in rows)
    return {
        "threshold": threshold,
        "rows": len(rows),
        "accepted": len(selected),
        "coverage": len(selected) / len(rows),
        "rescue": rescue,
        "regress": regress,
        "neutral": len(selected) - rescue - regress,
        "rescue_precision": rescue / max(1, len(selected)),
        "regress_rate": regress / max(1, len(selected)),
        "accepted_net_utility": (rescue - regress) / max(1, len(selected)),
        "population_net_utility": (rescue - regress) / len(rows),
        "student_baseline_accuracy": baseline_correct / len(rows),
        "fallback_student_accuracy": (baseline_correct + rescue - regress) / len(rows),
    }


def bootstrap(rows: Sequence[Mapping[str, Any]], threshold: float, draws: int, seed: int) -> dict[str, float]:
    by_episode: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_episode.setdefault(str(row["episode_id"]), []).append(row)
    episodes = list(by_episode)
    rng = random.Random(seed); values = []
    for _ in range(draws):
        sampled = [by_episode[episodes[rng.randrange(len(episodes))]] for _ in episodes]
        flat = [row for group in sampled for row in group]
        values.append(sum(utility(row) for row in flat if float(row["score"]) >= threshold) / len(flat))
    values.sort()
    return {"mean": sum(values)/len(values), "lo": values[int(0.025*draws)], "hi": values[min(draws-1,int(0.975*draws))], "draws": draws, "sampling_unit": "episode"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-scores", required=True)
    parser.add_argument("--test-scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-dev-accepted", type=int, default=10)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dev = read_jsonl(Path(args.dev_scores)); test = read_jsonl(Path(args.test_scores))
    candidates = sorted({float(row["score"]) for row in dev}, reverse=True)
    dev_results = [stats(dev, threshold) for threshold in candidates]
    viable = [row for row in dev_results if row["accepted"] >= args.min_dev_accepted and row["population_net_utility"] > 0]
    viable.sort(key=lambda row: (row["population_net_utility"], row["accepted_net_utility"], row["accepted"]), reverse=True)
    if viable:
        selected_dev = viable[0]; threshold = float(selected_dev["threshold"])
        selected_dev["bootstrap"] = bootstrap(dev, threshold, args.bootstrap_draws, args.seed)
        locked_test = stats(test, threshold)
        locked_test["selected_sample_ids"] = [str(row["sample_id"]) for row in test if float(row["score"]) >= threshold]
        locked_test["bootstrap"] = bootstrap(test, threshold, args.bootstrap_draws, args.seed)
        gate = "POSITIVE_TEST_UTILITY" if locked_test["population_net_utility"] > 0 else "NO_TEST_UTILITY"
    else:
        threshold = 1.0 + 1e-9; selected_dev = stats(dev, threshold); locked_test = stats(test, threshold); gate = "NO_POSITIVE_DEV_THRESHOLD"
    summary = {
        "selection_uses_test_labels": False,
        "threshold_selected_on": "episode-disjoint dev population utility",
        "min_dev_accepted": args.min_dev_accepted,
        "selected_threshold": threshold,
        "dev": selected_dev,
        "test": locked_test,
        "gate": gate,
        "top_dev_thresholds": viable[:20],
    }
    out_dir = Path(args.output_dir); write_json(out_dir / "summary.json", summary)
    lines = [
        "# Calibrated Revision Rescue Ranker",
        "",
        "Threshold is selected on dev rescue-minus-regression population utility and evaluated once on test.",
        "",
        f"Selected threshold: **{threshold:.6f}**. Gate: **{gate}**.",
        "",
        "| split | accepted | coverage | rescue | regress | rescue precision | population utility | fallback accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in (("dev", selected_dev), ("test", locked_test)):
        lines.append(
            f"| {name} | {row['accepted']} | {100*row['coverage']:.2f}% | {row['rescue']} | {row['regress']} | "
            f"{100*row['rescue_precision']:.2f}% | {100*row['population_net_utility']:+.2f}pp | {100*row['fallback_student_accuracy']:.2f}% |"
        )
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"threshold":threshold,"gate":gate,"dev":{k:v for k,v in selected_dev.items() if k!='selected_sample_ids'},"test":{k:v for k,v in locked_test.items() if k!='selected_sample_ids'},"report":str(out_dir/'report.md')},indent=2))


if __name__ == "__main__":
    main()
