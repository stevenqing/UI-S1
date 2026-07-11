#!/usr/bin/env python3
"""Analyze actor-relative utility of globally revised GUI trajectories.

The frozen matcher is used only for post-hoc diagnostics. For every aligned step,
the revision is classified as one of:

* rescue: actor wrong, revision correct
* regress: actor correct, revision wrong
* preserve_correct: actor correct, revision correct
* unresolved: actor wrong, revision wrong

This script does not select data or train a model.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires non-empty values")
    idx = min(len(sorted_values) - 1, max(0, int(q * len(sorted_values))))
    return float(sorted_values[idx])


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for idx in order[start:end]:
            ranks[idx] = rank
        start = end
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) != len(y) or not x:
        raise ValueError("pearson requires equal non-empty vectors")
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    dx = [value - mx for value in x]
    dy = [value - my for value in y]
    denom = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    if denom == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denom


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    return pearson(average_ranks(x), average_ranks(y))


def binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = average_ranks(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def threshold_average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    previous_recall = 0.0
    area = 0.0
    for threshold in sorted(set(scores), reverse=True):
        selected = [score >= threshold for score in scores]
        true_positives = sum(label and keep for label, keep in zip(labels, selected))
        false_positives = sum((not label) and keep for label, keep in zip(labels, selected))
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def outcome(actor_correct: bool, revised_correct: bool) -> str:
    if actor_correct and revised_correct:
        return "preserve_correct"
    if actor_correct and not revised_correct:
        return "regress"
    if not actor_correct and revised_correct:
        return "rescue"
    return "unresolved"


def safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def bootstrap_net_utility(
    trajectory_counts: Sequence[Mapping[str, int]], draws: int, seed: int
) -> dict[str, float] | None:
    if not trajectory_counts or draws <= 0:
        return None
    rng = random.Random(seed)
    n = len(trajectory_counts)
    values: list[float] = []
    for _ in range(draws):
        rescue = 0
        regress = 0
        steps = 0
        for _ in range(n):
            row = trajectory_counts[rng.randrange(n)]
            rescue += row["rescue"]
            regress += row["regress"]
            steps += row["steps"]
        values.append((rescue - regress) / steps)
    values.sort()
    return {
        "mean": sum(values) / len(values),
        "lo": percentile(values, 0.025),
        "hi": percentile(values, 0.975),
        "draws": draws,
        "sampling_unit": "trajectory",
    }


def analyze_group(
    corrections: Sequence[Mapping[str, Any]],
    actors: Mapping[str, Mapping[str, Any]],
    bootstrap_draws: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    trajectory_counts: list[dict[str, Any]] = []

    for correction in corrections:
        trajectory_id = str(correction["trajectory_id"])
        if trajectory_id not in actors:
            raise ValueError(f"missing actor trajectory: {trajectory_id}")
        actor = actors[trajectory_id]
        if bool(actor.get("task_success")):
            raise ValueError(f"expected an erroneous actor trajectory: {trajectory_id}")
        actor_steps = {int(step["step_idx"]): step for step in actor["steps"]}
        revised_steps = list(correction.get("revised_steps", []))
        revised_indices = [int(step["step_idx"]) for step in revised_steps]
        if len(revised_indices) != len(set(revised_indices)):
            raise ValueError(f"duplicate revised step index: {trajectory_id}")
        if set(revised_indices) != set(actor_steps):
            raise ValueError(f"actor/revision step-grid mismatch: {trajectory_id}")

        local: Counter[str] = Counter()
        for revised_step in revised_steps:
            step_idx = int(revised_step["step_idx"])
            actor_correct = bool(actor_steps[step_idx]["actor_correct"])
            revised_correct = bool(revised_step["diagnostic_correct"])
            changed = bool(revised_step["changed_from_actor"])
            category = outcome(actor_correct, revised_correct)
            counts[category] += 1
            counts["steps"] += 1
            counts["actor_correct"] += int(actor_correct)
            counts["revised_correct"] += int(revised_correct)
            counts["changed"] += int(changed)
            counts[f"changed_{category}"] += int(changed)
            counts[f"unchanged_{category}"] += int(not changed)
            local[category] += 1
            local["steps"] += 1

            if category in {"rescue", "regress"} and not changed:
                raise ValueError(f"utility-changing step marked unchanged: {trajectory_id}:{step_idx}")

        counts["trajectories"] += 1
        task_rescue = bool(correction.get("diagnostic_task_success"))
        counts["trajectory_rescues"] += int(task_rescue)
        trajectory_counts.append(
            {
                "trajectory_id": trajectory_id,
                "actor": str(correction["actor"]),
                "confidence": float(correction.get("confidence") or 0.0),
                "task_rescue": int(task_rescue),
                "steps": local["steps"],
                "rescue": local["rescue"],
                "regress": local["regress"],
                "preserve_correct": local["preserve_correct"],
                "unresolved": local["unresolved"],
                "actor_accuracy": safe_rate(local["preserve_correct"] + local["regress"], local["steps"]),
                "revised_accuracy": safe_rate(local["preserve_correct"] + local["rescue"], local["steps"]),
                "net_revision_utility": safe_rate(local["rescue"] - local["regress"], local["steps"]),
                "changed_rate": safe_rate(int(correction.get("changed_steps", 0)), local["steps"]),
            }
        )

    steps = counts["steps"]
    actor_wrong = steps - counts["actor_correct"]
    changed = counts["changed"]
    result: dict[str, Any] = {
        "trajectories": counts["trajectories"],
        "steps": steps,
        "outcomes": {
            name: {
                "count": counts[name],
                "fraction": safe_rate(counts[name], steps),
            }
            for name in ("rescue", "regress", "preserve_correct", "unresolved")
        },
        "actor_correct_steps": counts["actor_correct"],
        "revised_correct_steps": counts["revised_correct"],
        "actor_accuracy_on_usable_subset": safe_rate(counts["actor_correct"], steps),
        "revised_accuracy": safe_rate(counts["revised_correct"], steps),
        "net_revision_utility": safe_rate(counts["rescue"] - counts["regress"], steps),
        "rescue_rate_given_actor_wrong": safe_rate(counts["rescue"], actor_wrong),
        "regression_rate_given_actor_correct": safe_rate(counts["regress"], counts["actor_correct"]),
        "changed_steps": changed,
        "changed_rate": safe_rate(changed, steps),
        "changed_outcomes": {
            name: {
                "count": counts[f"changed_{name}"],
                "fraction_of_changed": safe_rate(counts[f"changed_{name}"], changed),
            }
            for name in ("rescue", "regress", "preserve_correct", "unresolved")
        },
        "net_revision_utility_on_changed_steps": safe_rate(
            counts["changed_rescue"] - counts["changed_regress"], changed
        ),
        "trajectory_rescues": counts["trajectory_rescues"],
        "trajectory_rescue_rate": safe_rate(counts["trajectory_rescues"], counts["trajectories"]),
        "net_revision_utility_cluster_bootstrap": bootstrap_net_utility(
            trajectory_counts, bootstrap_draws, seed
        ),
    }
    return result, trajectory_counts


def confidence_diagnostics(trajectory_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    confidence = [float(row["confidence"]) for row in trajectory_rows]
    task_rescue = [int(row["task_rescue"]) for row in trajectory_rows]
    revised_accuracy = [float(row["revised_accuracy"]) for row in trajectory_rows]
    utility = [float(row["net_revision_utility"]) for row in trajectory_rows]
    changed_rate = [float(row["changed_rate"]) for row in trajectory_rows]
    confidence_counts = Counter(confidence)
    dominant_confidence, dominant_count = confidence_counts.most_common(1)[0]
    return {
        "unit": "trajectory",
        "confidence_distribution": {
            f"{value:.2f}": count for value, count in sorted(confidence_counts.items())
        },
        "dominant_confidence": dominant_confidence,
        "dominant_confidence_fraction": dominant_count / len(confidence),
        "task_rescue_base_rate": sum(task_rescue) / len(task_rescue),
        "confidence_auc_for_task_rescue": binary_auc(task_rescue, confidence),
        "confidence_threshold_ap_for_task_rescue": threshold_average_precision(task_rescue, confidence),
        "confidence_spearman_revised_accuracy": spearman(confidence, revised_accuracy),
        "confidence_spearman_net_revision_utility": spearman(confidence, utility),
        "confidence_spearman_changed_rate": spearman(confidence, changed_rate),
        "training_visibility": False,
        "training_visibility_note": (
            "correction_confidence is retained as raw metadata but is absent from the "
            "LLaMA-Factory conversations used for SFT"
        ),
    }


def prefix_consistency_diagnostics(
    corrections: Sequence[Mapping[str, Any]],
    sft_rows: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    by_prior_wrong: dict[str, Counter[str]] = {}
    prefix_strata: dict[tuple[str, int], dict[bool, Counter[str]]] = {}
    expected_steps: set[tuple[str, int]] = set()

    for correction in corrections:
        correction_id = str(correction["correction_id"])
        actor = str(correction["actor"])
        prior_wrong = 0
        for step in sorted(correction["revised_steps"], key=lambda row: int(row["step_idx"])):
            step_idx = int(step["step_idx"])
            expected_steps.add((correction_id, step_idx))
            revised_correct = bool(step["diagnostic_correct"])
            bucket = str(min(prior_wrong, 4)) + ("+" if prior_wrong >= 4 else "")
            bucket_counts = by_prior_wrong.setdefault(bucket, Counter())
            bucket_counts["steps"] += 1
            bucket_counts["correct"] += int(revised_correct)
            counts["steps"] += 1
            counts["correct"] += int(revised_correct)
            counts["clean_prefix_steps"] += int(prior_wrong == 0)
            counts["dirty_prefix_steps"] += int(prior_wrong > 0)
            counts["correct_clean_prefix"] += int(prior_wrong == 0 and revised_correct)
            counts["correct_dirty_prefix"] += int(prior_wrong > 0 and revised_correct)
            stratum = prefix_strata.setdefault(
                (actor, step_idx), {True: Counter(), False: Counter()}
            )[prior_wrong == 0]
            stratum["steps"] += 1
            stratum["correct"] += int(revised_correct)
            prior_wrong += int(not revised_correct)

    overlap_weight = 0
    overlap_difference_sum = 0.0
    overlap_strata = 0
    for groups in prefix_strata.values():
        clean = groups[True]
        dirty = groups[False]
        if not clean["steps"] or not dirty["steps"]:
            continue
        weight = min(clean["steps"], dirty["steps"])
        clean_accuracy = clean["correct"] / clean["steps"]
        dirty_accuracy = dirty["correct"] / dirty["steps"]
        overlap_difference_sum += weight * (clean_accuracy - dirty_accuracy)
        overlap_weight += weight
        overlap_strata += 1

    result: dict[str, Any] = {
        "definition": "clean prefix iff every earlier revised action is matcher-correct",
        "status": "diagnostic proxy; not an executed transition-equivalence test",
        "steps": counts["steps"],
        "clean_prefix_steps": counts["clean_prefix_steps"],
        "clean_prefix_fraction": safe_rate(counts["clean_prefix_steps"], counts["steps"]),
        "dirty_prefix_steps": counts["dirty_prefix_steps"],
        "dirty_prefix_fraction": safe_rate(counts["dirty_prefix_steps"], counts["steps"]),
        "current_accuracy_given_clean_prefix": safe_rate(
            counts["correct_clean_prefix"], counts["clean_prefix_steps"]
        ),
        "current_accuracy_given_dirty_prefix": safe_rate(
            counts["correct_dirty_prefix"], counts["dirty_prefix_steps"]
        ),
        "actor_step_index_overlap_adjustment": {
            "strata": overlap_strata,
            "overlap_weighted_steps": overlap_weight,
            "clean_minus_dirty_accuracy": safe_rate(overlap_difference_sum, overlap_weight),
            "note": "descriptive overlap weighting within actor and absolute step index; not a causal estimator",
        },
        "by_prior_wrong_actions": {
            bucket: {
                "steps": bucket_counts["steps"],
                "current_accuracy": safe_rate(bucket_counts["correct"], bucket_counts["steps"]),
            }
            for bucket, bucket_counts in sorted(
                by_prior_wrong.items(), key=lambda item: int(item[0].rstrip("+"))
            )
        },
    }

    if sft_rows is not None:
        observed_steps: set[tuple[str, int]] = set()
        history_length_mismatches = 0
        nonempty_histories = 0
        for row in sft_rows:
            key = (str(row["correction_id"]), int(row["step_idx"]))
            if key in observed_steps:
                raise ValueError(f"duplicate SFT correction step: {key}")
            observed_steps.add(key)
            history = list(row.get("history", []))
            nonempty_histories += int(bool(history))
            history_length_mismatches += int(len(history) != int(row["step_idx"]))
        if observed_steps != expected_steps:
            raise ValueError("SFT/revision correction-step grid mismatch")
        result["sft_validation"] = {
            "rows": len(sft_rows),
            "nonempty_history_rows": nonempty_histories,
            "history_length_mismatches": history_length_mismatches,
            "uses_revised_prefix_history": history_length_mismatches == 0,
        }
    return result


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def pp(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:+.2f}pp"


def render_markdown(summary: Mapping[str, Any]) -> str:
    overall = summary["overall"]
    by_actor = summary["by_actor"]
    confidence = summary["confidence_diagnostics"]
    prefix = summary["prefix_consistency_diagnostics"]
    evaluation = summary.get("downstream_evaluation")
    bootstrap = overall["net_revision_utility_cluster_bootstrap"]

    lines = [
        "# Counterfactual Revision Utility Analysis",
        "",
        "## Definition",
        "",
        "For a frozen matcher reward $M$ and step $t$:",
        "",
        r"$$u_t = M(a_t^{revision}) - M(a_t^{actor}) \in \{-1, 0, +1\}.$$",
        "",
        "- `rescue`: actor wrong, revision correct ($u_t=+1$)",
        "- `regress`: actor correct, revision wrong ($u_t=-1$)",
        "- `preserve_correct`: both correct ($u_t=0$)",
        "- `unresolved`: both wrong ($u_t=0$)",
        "",
        "The matcher is diagnostic only; these labels were not used to select the formal noisy-SFT rows.",
        "",
        "## Overall Step Utility",
        "",
        "| outcome | count | fraction |",
        "|---|---:|---:|",
    ]
    for name in ("rescue", "regress", "preserve_correct", "unresolved"):
        row = overall["outcomes"][name]
        lines.append(f"| {name} | {row['count']} | {pct(row['fraction'])} |")
    lines.extend(
        [
            "",
            f"Actor accuracy on the structurally usable subset: **{pct(overall['actor_accuracy_on_usable_subset'])}**.",
            f"Revised accuracy: **{pct(overall['revised_accuracy'])}**.",
            f"Net actor-relative revision utility: **{pp(overall['net_revision_utility'])}**.",
            f"Rescue rate conditional on actor-wrong: **{pct(overall['rescue_rate_given_actor_wrong'])}**.",
            f"Regression rate conditional on actor-correct: **{pct(overall['regression_rate_given_actor_correct'])}**.",
            f"Complete-trajectory rescues: **{overall['trajectory_rescues']} / {overall['trajectories']} ({pct(overall['trajectory_rescue_rate'])})**.",
            "",
            f"Trajectory-clustered {bootstrap['draws']:,}-draw bootstrap for net revision utility: "
            f"**[{pp(bootstrap['lo'])}, {pp(bootstrap['hi'])}]**.",
            "",
            "## Source-Conditioned Utility",
            "",
            "| actor source | trajectories | steps | actor acc | revised acc | net utility | rescue / regress | trajectory rescue |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for actor, row in sorted(by_actor.items()):
        lines.append(
            f"| {actor} | {row['trajectories']} | {row['steps']} | "
            f"{pct(row['actor_accuracy_on_usable_subset'])} | {pct(row['revised_accuracy'])} | "
            f"{pp(row['net_revision_utility'])} | "
            f"{row['outcomes']['rescue']['count']} / {row['outcomes']['regress']['count']} | "
            f"{pct(row['trajectory_rescue_rate'])} |"
        )
    lines.extend(
        [
            "",
            "The same corrector is beneficial relative to the weaker InternVL3 source but harmful relative to the stronger Qwen3-VL source. Revision utility is therefore source-conditioned, not an intrinsic property of the correction alone.",
            "",
            "## Changed-Step Selectivity",
            "",
            f"The corrector changed **{overall['changed_steps']} / {overall['steps']} ({pct(overall['changed_rate'])})** usable steps.",
            "",
            "| changed-step outcome | count | fraction of changed steps |",
            "|---|---:|---:|",
        ]
    )
    for name in ("rescue", "regress", "preserve_correct", "unresolved"):
        row = overall["changed_outcomes"][name]
        lines.append(f"| {name} | {row['count']} | {pct(row['fraction_of_changed'])} |")
    lines.extend(
        [
            "",
            f"Net utility among changed steps is **{pp(overall['net_revision_utility_on_changed_steps'])}**, "
            "but most changed steps remain unresolved.",
            "",
            "## Confidence as a Selector",
            "",
            f"- {pct(confidence['dominant_confidence_fraction'])} of usable trajectories have the same self-reported confidence {confidence['dominant_confidence']:.2f}.",
            f"- Confidence AUC for complete-trajectory rescue: **{confidence['confidence_auc_for_task_rescue']:.3f}**.",
            f"- Confidence threshold AP: **{confidence['confidence_threshold_ap_for_task_rescue']:.3f}**, versus a rescue base rate of **{pct(confidence['task_rescue_base_rate'])}**.",
            f"- Spearman confidence vs revised step accuracy: **{confidence['confidence_spearman_revised_accuracy']:.3f}**.",
            f"- Spearman confidence vs net revision utility: **{confidence['confidence_spearman_net_revision_utility']:.3f}**.",
            "- Confidence was not visible in the actual ShareGPT conversations used for SFT, so it is a diagnostic of poor selection/calibration and cannot be claimed as the causal feature learned by the student.",
            "",
            "## Prefix Consistency Under Teacher-Forced Screens",
            "",
            "A diagnostic clean prefix means every earlier revised action matches the frozen reference. Because screenshots remain on the GT trajectory while SFT histories contain revised actions, a dirty prefix is a proxy for history–screen inconsistency; it is not an executed transition-equivalence test.",
            "",
            f"- Clean-prefix rows: **{prefix['clean_prefix_steps']} / {prefix['steps']} ({pct(prefix['clean_prefix_fraction'])})**; current-label accuracy **{pct(prefix['current_accuracy_given_clean_prefix'])}**.",
            f"- Dirty-prefix rows: **{prefix['dirty_prefix_steps']} / {prefix['steps']} ({pct(prefix['dirty_prefix_fraction'])})**; current-label accuracy **{pct(prefix['current_accuracy_given_dirty_prefix'])}**.",
            f"- Within actor × absolute-step overlap strata, the weighted clean-minus-dirty accuracy difference remains **{pp(prefix['actor_step_index_overlap_adjustment']['clean_minus_dirty_accuracy'])}** over {prefix['actor_step_index_overlap_adjustment']['overlap_weighted_steps']} overlap-weighted rows. This is descriptive, not causal.",
            "",
            "| prior matcher-wrong revised actions | rows | current-label accuracy |",
            "|---:|---:|---:|",
        ]
    )
    for bucket, row in prefix["by_prior_wrong_actions"].items():
        lines.append(f"| {bucket} | {row['steps']} | {pct(row['current_accuracy'])} |")
    lines.extend(
        [
            "",
            "This monotonic degradation is consistent with a sequential contamination mechanism: globally rewriting actions over fixed future screenshots does not guarantee that the revised action prefix causally reaches those screenshots.",
            "",
            "## Local-to-Downstream Gap",
            "",
        ]
    )
    if evaluation:
        lines.extend(
            [
                f"- Local actor-relative step utility: **{pp(overall['net_revision_utility'])}**.",
                f"- Held-out post-training step-accuracy delta: **{pp(evaluation['step_accuracy_delta'])}**.",
                f"- Held-out post-training TSR delta: **{pp(evaluation['tsr_delta'])}**.",
                "",
                "These quantities use different reference policies and splits, so their difference is not a causal estimator. Together they show that being better than weak source actors is insufficient for a revision set to supervise the stronger starting checkpoint.",
            ]
        )
    lines.extend(
        [
            "",
            "## Research Interpretation",
            "",
            "1. **Diversity–utility decoupling:** heterogeneous error diversity does not imply useful supervision.",
            "2. **Source-conditioned correctability:** one corrector can rescue a weak actor while degrading a stronger actor.",
            "3. **Step–trajectory composition gap:** positive average step utility coexists with only a small complete-trajectory rescue rate.",
            "4. **Revision–student gap:** actor-relative improvement is not enough; labels must also be trustworthy relative to the student being trained.",
            "5. **Prefix-consistency gap:** most training rows condition on a revised prefix already diagnosed as wrong while observing a teacher-forced GT screenshot.",
            "6. **Self-reported confidence is insufficient:** it has little rank correlation with step quality or net utility and was not an SFT input.",
            "",
            "These are diagnostic findings, not yet a positive quality-gating method. A publishable method claim requires a learned or rule-based revision utility gate plus clean, random-label, actor-label, and source-specific training controls.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-trajectories", required=True)
    parser.add_argument("--corrections", required=True)
    parser.add_argument("--sft-data")
    parser.add_argument("--training-eval-summary")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    actor_rows = read_jsonl(Path(args.actor_trajectories))
    actors = {str(row["trajectory_id"]): row for row in actor_rows}
    if len(actors) != len(actor_rows):
        raise ValueError("duplicate actor trajectory_id")
    correction_rows = [
        row for row in read_jsonl(Path(args.corrections)) if bool(row.get("parse_ok"))
    ]
    if not correction_rows:
        raise ValueError("no parseable corrections")

    overall, trajectory_rows = analyze_group(
        correction_rows, actors, args.bootstrap_draws, args.seed
    )
    by_actor: dict[str, Any] = {}
    for actor in sorted({str(row["actor"]) for row in correction_rows}):
        group = [row for row in correction_rows if str(row["actor"]) == actor]
        by_actor[actor], _ = analyze_group(
            group, actors, args.bootstrap_draws, args.seed
        )

    summary: dict[str, Any] = {
        "definition": "matcher(revision) - matcher(actor)",
        "matcher_role": "diagnostic_only",
        "actor_trajectories": args.actor_trajectories,
        "corrections": args.corrections,
        "overall": overall,
        "by_actor": by_actor,
        "confidence_diagnostics": confidence_diagnostics(trajectory_rows),
        "prefix_consistency_diagnostics": prefix_consistency_diagnostics(
            correction_rows,
            read_jsonl(Path(args.sft_data)) if args.sft_data else None,
        ),
    }
    if args.training_eval_summary:
        evaluation = read_json(Path(args.training_eval_summary))
        summary["downstream_evaluation"] = {
            "held_out_episodes": evaluation["held_out_episodes"],
            "held_out_steps": evaluation["held_out_steps"],
            "baseline_tsr": evaluation["baseline_tsr"],
            "post_tsr": evaluation["post_tsr"],
            "tsr_delta": evaluation["tsr_delta"],
            "baseline_step_accuracy": evaluation["baseline_step_accuracy"],
            "post_step_accuracy": evaluation["post_step_accuracy"],
            "step_accuracy_delta": evaluation["step_accuracy_delta"],
            "gate": evaluation["gate"],
        }

    out_dir = Path(args.output_dir)
    write_json(out_dir / "revision_utility_summary.json", summary)
    (out_dir / "revision_utility_report.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "trajectories": overall["trajectories"],
                "steps": overall["steps"],
                "net_revision_utility": overall["net_revision_utility"],
                "trajectory_rescue_rate": overall["trajectory_rescue_rate"],
                "output_dir": str(out_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
