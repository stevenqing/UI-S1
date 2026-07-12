#!/usr/bin/env python3
"""Analyze the target-label × history-source LoRA factorial intervention."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence


ARMS = {
    "a1": "a1_gt_target_gt_history",
    "a4": "a4_revision_target_revision_history",
    "a5": "a5_revision_target_gt_history",
    "a6": "a6_gt_target_revision_history",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def episode_vectors(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for episode in rows:
        episode_id = str(episode["episode_id"])
        steps = [int(bool(step["success"])) for step in episode.get("steps", [])]
        if episode_id in output:
            raise ValueError(f"duplicate episode: {episode_id}")
        output[episode_id] = {"task": int(bool(episode["task_success"])), "steps": steps}
    return output


def aggregate(data: Mapping[str, Mapping[str, Any]], episode_ids: Sequence[str]) -> dict[str, float]:
    tasks = [int(data[episode_id]["task"]) for episode_id in episode_ids]
    steps = [value for episode_id in episode_ids for value in data[episode_id]["steps"]]
    return {"tsr": sum(tasks) / len(tasks), "step_accuracy": sum(steps) / len(steps)}


def effects(metrics: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    result = {}
    for metric in ("tsr", "step_accuracy"):
        gt_history_effect_gt_target = metrics["a1"][metric] - metrics["a6"][metric]
        gt_history_effect_revision_target = metrics["a5"][metric] - metrics["a4"][metric]
        revision_label_effect_gt_history = metrics["a5"][metric] - metrics["a1"][metric]
        revision_label_effect_revision_history = metrics["a4"][metric] - metrics["a6"][metric]
        result[metric] = {
            "gt_history_effect_given_gt_target": gt_history_effect_gt_target,
            "gt_history_effect_given_revision_target": gt_history_effect_revision_target,
            "revision_label_effect_given_gt_history": revision_label_effect_gt_history,
            "revision_label_effect_given_revision_history": revision_label_effect_revision_history,
            "label_history_interaction": gt_history_effect_revision_target - gt_history_effect_gt_target,
        }
    return result


def percentile(values: Sequence[float], q: float) -> float:
    idx = min(len(values) - 1, max(0, int(q * len(values))))
    return float(values[idx])


def bootstrap(
    data: Mapping[str, Mapping[str, Mapping[str, Any]]],
    episode_ids: Sequence[str],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    collected = {
        metric: {effect: [] for effect in effects({arm: {"tsr": 0.0, "step_accuracy": 0.0} for arm in ARMS})[metric]}
        for metric in ("tsr", "step_accuracy")
    }
    for _ in range(draws):
        sampled = [episode_ids[rng.randrange(len(episode_ids))] for _ in episode_ids]
        metrics = {arm: aggregate(data[arm], sampled) for arm in ARMS}
        draw_effects = effects(metrics)
        for metric, items in draw_effects.items():
            for name, value in items.items():
                collected[metric][name].append(value)
    output = {}
    for metric, items in collected.items():
        output[metric] = {}
        for name, values in items.items():
            values.sort()
            output[metric][name] = {
                "lo": percentile(values, 0.025),
                "hi": percentile(values, 0.975),
                "draws": draws,
                "sampling_unit": "held_out_episode",
            }
    return output


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    return f"{100.0 * value:+.2f}pp"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    data = {short: episode_vectors(read_jsonl(eval_dir / f"{arm}.jsonl")) for short, arm in ARMS.items()}
    episode_sets = [set(rows) for rows in data.values()]
    if not all(items == episode_sets[0] for items in episode_sets[1:]):
        raise ValueError("factorial episode grids do not match")
    episode_ids = sorted(episode_sets[0], key=lambda value: int(value) if value.isdigit() else value)
    for episode_id in episode_ids:
        step_counts = {len(data[arm][episode_id]["steps"]) for arm in ARMS}
        if len(step_counts) != 1:
            raise ValueError(f"factorial step grid mismatch: {episode_id}")

    metrics = {arm: aggregate(data[arm], episode_ids) for arm in ARMS}
    point_effects = effects(metrics)
    intervals = bootstrap(data, episode_ids, args.bootstrap_draws, args.seed)
    summary = {
        "episodes": len(episode_ids),
        "steps": sum(len(data["a1"][episode_id]["steps"]) for episode_id in episode_ids),
        "arms": {ARMS[arm]: values for arm, values in metrics.items()},
        "effects": point_effects,
        "bootstrap": intervals,
        "interpretation": {
            "history_mismatch": "A1-A6: same GT labels, GT history versus revised history",
            "clean_context_noise": "A5-A4: same revision labels, GT history versus revised history",
            "label_history_interaction": "(A5-A4) - (A1-A6)",
        },
    }
    out_dir = Path(args.output_dir)
    write_json(out_dir / "summary.json", summary)

    arm_rows = []
    for short, name in ARMS.items():
        arm_rows.append(f"| {name} | {pct(metrics[short]['tsr'])} | {pct(metrics[short]['step_accuracy'])} |")
    effect_rows = []
    for name, value in point_effects["step_accuracy"].items():
        interval = intervals["step_accuracy"][name]
        effect_rows.append(f"| {name} | {pp(value)} | [{pp(interval['lo'])}, {pp(interval['hi'])}] |")
    lines = [
        "# Revision Target × History Factorial Analysis",
        "",
        f"Paired held-out grid: {len(episode_ids)} episodes / {summary['steps']} steps.",
        "",
        "## Arms",
        "",
        "| arm | TSR | step accuracy |",
        "|---|---:|---:|",
        *arm_rows,
        "",
        "## Step-Accuracy Effects",
        "",
        "| effect | estimate | episode-bootstrap 95% interval |",
        "|---|---:|---:|",
        *effect_rows,
        "",
        "A1−A6 measures the effect of replacing revised history with GT history when labels are correct. A5−A4 measures the same history replacement when labels are revisions. Their difference is the label×history interaction.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"effects": point_effects, "report": str(out_dir / "report.md")}, indent=2))


if __name__ == "__main__":
    main()
