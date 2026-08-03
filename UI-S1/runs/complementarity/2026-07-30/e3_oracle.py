import argparse
import json
from pathlib import Path

from common import episode_macro, load_rows, micro, model_success_sets, parse_failure_rate, pivot_rows


POOLS = (("androidcontrol", "low"), ("androidcontrol", "high"), ("mind2web", "visual"))


def greedy_curve(identities, models, success_sets):
    selected = []
    covered = set()
    remaining = set(models)
    curve = []
    while remaining:
        model = max(remaining, key=lambda candidate: (len(covered | success_sets[candidate]), candidate))
        previous = len(covered)
        covered |= success_sets[model]
        selected.append(model)
        remaining.remove(model)
        curve.append({
            "step": len(selected), "added_model": model, "marginal_successes": len(covered) - previous,
            "oracle_successes": len(covered), "oracle_micro": len(covered) / len(identities),
        })
    return curve


def summarize_pool(bench, setting):
    rows = load_rows(bench, setting)
    identities, models, pivot = pivot_rows(rows)
    success_sets = model_success_sets(identities, models, pivot)
    per_model = {}
    for model in models:
        successes = {row_id: pivot[row_id][model]["success"] for row_id in identities}
        per_model[model] = {
            "parse_failure_rate": parse_failure_rate(identities, model, pivot),
            "step_successes": sum(successes.values()),
            "step_micro": micro(successes.values()),
            "episode_macro": episode_macro(successes, pivot),
        }
    deployable = [
        model for model in models
        if per_model[model]["parse_failure_rate"] < 0.05 and per_model[model]["step_micro"] > 0.30
    ]
    full_union = set().union(*(success_sets[model] for model in models))
    deployable_union = set().union(*(success_sets[model] for model in deployable)) if deployable else set()
    full_success = {row_id: row_id in full_union for row_id in identities}
    deployable_success = {row_id: row_id in deployable_union for row_id in identities}
    curve = greedy_curve(identities, models, success_sets)
    target = 0.95 * len(full_union)
    minimum_95 = next(item["step"] for item in curve if item["oracle_successes"] >= target)
    saturation = next(
        (item["step"] for item in curve if item["marginal_successes"] == 0),
        len(curve),
    )
    return {
        "rows": len(identities),
        "episodes": len({next(iter(pivot[row_id].values()))["episode_id"] for row_id in identities}),
        "aggregation_contract": {
            "step_micro": "mean over row identities",
            "episode_macro": "mean of within-episode Step SR over episodes",
        },
        "per_model": per_model,
        "full_models": models,
        "full_oracle": {
            "successes": len(full_union), "step_micro": len(full_union) / len(identities),
            "episode_macro": episode_macro(full_success, pivot),
        },
        "deployable_definition": "parse_failure_rate < 0.05 and step_micro > 0.30",
        "deployable_models": deployable,
        "deployable_oracle": {
            "successes": len(deployable_union), "step_micro": len(deployable_union) / len(identities),
            "episode_macro": episode_macro(deployable_success, pivot) if deployable else None,
        },
        "greedy_forward_curve": curve,
        "minimum_models_for_95_percent_full_oracle": minimum_95,
        "saturation_step_first_zero_marginal_or_full_length": saturation,
    }


def render_table(result):
    lines = [
        "# E3 Oracle and aggregation table", "",
        "All oracle values are descriptive upper bounds. Deployable subsets require parse failure rate < 5% and step-micro > 30%.", "",
    ]
    for pool, summary in result["pools"].items():
        lines.extend([
            f"## {pool}", "",
            "| Model | Parse failure | Step micro | Episode macro |",
            "|---|---:|---:|---:|",
        ])
        for model, metrics in sorted(summary["per_model"].items(), key=lambda item: -item[1]["step_micro"]):
            lines.append(
                f"| {model} | {metrics['parse_failure_rate']:.2%} | {metrics['step_micro']:.2%} | {metrics['episode_macro']:.2%} |"
            )
        full = summary["full_oracle"]
        deployable = summary["deployable_oracle"]
        lines.extend([
            "", "| Oracle scope | Models | Step micro | Episode macro |",
            "|---|---:|---:|---:|",
            f"| Full | {len(summary['full_models'])} | {full['step_micro']:.2%} | {full['episode_macro']:.2%} |",
            f"| Deployable | {len(summary['deployable_models'])} | {deployable['step_micro']:.2%} | {deployable['episode_macro']:.2%} |",
            "", f"Deployable models: {', '.join(summary['deployable_models'])}",
            f"Minimum greedy subset reaching 95% of full-oracle successes: {summary['minimum_models_for_95_percent_full_oracle']}", "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    result = {"status": "PASS", "pools": {}}
    for bench, setting in POOLS:
        result["pools"][f"{bench}/{setting}"] = summarize_pool(bench, setting)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.table.write_text(render_table(result))
    print(json.dumps({pool: {
        "full_micro": value["full_oracle"]["step_micro"],
        "deployable_micro": value["deployable_oracle"]["step_micro"],
        "deployable_models": value["deployable_models"],
        "minimum_95": value["minimum_models_for_95_percent_full_oracle"],
    } for pool, value in result["pools"].items()}, indent=2))


if __name__ == "__main__":
    main()