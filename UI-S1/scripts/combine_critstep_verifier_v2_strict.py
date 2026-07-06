#!/usr/bin/env python3
"""Strict train/test verifier aggregation: fit on TRAIN scores, evaluate on TEST scores."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.combine_critstep_verifier_v2 import (  # noqa: E402
    aggregate_pick,
    apply_stump,
    attach_scores,
    candidate_by_verifier,
    fit_best_stump,
    fit_best_weight,
    fit_pipeline,
    fraction,
    pipeline_pick,
    reject_greedy_for_pick,
    summarize_by,
)
from scripts.critstep_verifier_full_action import ORACLE_CRITICAL_TSR_CEILING_PP, projected_tsr_lift_pp  # noqa: E402
from scripts.score_critstep_verifier_v2_cot_voting import read_jsonl, write_json, write_jsonl  # noqa: E402


DEFAULT_OUTPUT_DIR = "outputs/critstep_verifier_v2/strict/combine"


def build_rows(stage1_steps: Sequence[Mapping[str, Any]], stage2_steps: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    stage2_by_target = {str(row["target_id"]): row for row in stage2_steps}
    rows = []
    for stage1_step in stage1_steps:
        target_id = str(stage1_step["target_id"])
        stage2_step = stage2_by_target[target_id]
        candidates = attach_scores(stage1_step, stage2_step)
        stage1_pick = candidate_by_verifier({**stage1_step, **stage2_step}, candidates, "stage1")
        stage2_pick = candidate_by_verifier({**stage1_step, **stage2_step}, candidates, "stage2")
        if stage1_pick.get("is_correct") and not stage2_pick.get("is_correct"):
            route_label = "stage1"
        elif stage2_pick.get("is_correct") and not stage1_pick.get("is_correct"):
            route_label = "stage2"
        elif stage1_pick.get("is_correct") and stage2_pick.get("is_correct"):
            route_label = "both"
        else:
            route_label = "neither"
        rows.append({
            "target_id": target_id,
            "episode_id": str(stage1_step.get("episode_id")),
            "split": str(stage1_step.get("split") or ""),
            "episode_key": str(stage1_step.get("episode_key") or stage1_step.get("episode_id")),
            "step_idx": stage1_step.get("step_idx"),
            "subset": stage1_step.get("subset"),
            "depth_bin": stage1_step.get("depth_bin"),
            "candidates": candidates,
            "features": candidate_set_features(candidates, stage1_pick, stage2_pick),
            "stage1_pick": stage1_pick,
            "stage2_pick": stage2_pick,
            "stage1_pick_key": stage1_pick["distinct_key"],
            "stage2_pick_key": stage2_pick["distinct_key"],
            "stage1_correct": bool(stage1_pick.get("is_correct")),
            "stage2_correct": bool(stage2_pick.get("is_correct")),
            "oracle_route_label": route_label,
            "oracle_routing_correct": bool(stage1_pick.get("is_correct") or stage2_pick.get("is_correct")),
            "greedy_correct": stage1_step.get("greedy_correct"),
            "first_sample_correct": stage1_step.get("first_sample_correct"),
        })
    return rows


def safe_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = safe_mean(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


def candidate_set_features(candidates: Sequence[Mapping[str, Any]], stage1_pick: Mapping[str, Any], stage2_pick: Mapping[str, Any]) -> Dict[str, float]:
    n = len(candidates)
    pred_types = [str(candidate.get("pred_type") or "") for candidate in candidates]
    control_keys = [candidate.get("control_key") for candidate in candidates if candidate.get("control_key")]
    control_texts = [str(candidate.get("control_text") or "") for candidate in candidates if candidate.get("control_text")]
    control_types = [str(candidate.get("control_type") or "") for candidate in candidates if candidate.get("control_type")]
    s1 = [float(candidate.get("stage1_score") or 0.0) for candidate in candidates]
    s1n = sorted([float(candidate.get("stage1_norm") or 0.0) for candidate in candidates], reverse=True)
    s2n = sorted([float(candidate.get("stage2_norm") or 0.0) for candidate in candidates], reverse=True)
    greedy = next((candidate for candidate in candidates if "greedy" in set(candidate.get("sources") or [])), None)
    return {
        "n_distinct": float(n),
        "n_pred_types": float(len(set(pred_types))),
        "frac_click": sum(1 for item in pred_types if item == "click") / n if n else 0.0,
        "frac_type": sum(1 for item in pred_types if item == "type") / n if n else 0.0,
        "frac_swipe": sum(1 for item in pred_types if item == "swipe") / n if n else 0.0,
        "frac_no_control": sum(1 for candidate in candidates if not candidate.get("control_key")) / n if n else 0.0,
        "n_control_keys": float(len(set(control_keys))),
        "n_control_texts": float(len(set(control_texts))),
        "n_control_types": float(len(set(control_types))),
        "control_key_spread": len(set(control_keys)) / n if n else 0.0,
        "control_text_spread": len(set(control_texts)) / n if n else 0.0,
        "stage1_score_mean": safe_mean(s1),
        "stage1_score_std": safe_std(s1),
        "stage1_norm_margin": (s1n[0] - s1n[1]) if len(s1n) > 1 else 0.0,
        "stage2_norm_margin": (s2n[0] - s2n[1]) if len(s2n) > 1 else 0.0,
        "stage1_stage2_agree": 1.0 if stage1_pick["distinct_key"] == stage2_pick["distinct_key"] else 0.0,
        "stage1_pick_is_greedy": 1.0 if greedy and greedy["distinct_key"] == stage1_pick["distinct_key"] else 0.0,
        "stage2_pick_is_greedy": 1.0 if greedy and greedy["distinct_key"] == stage2_pick["distinct_key"] else 0.0,
        "greedy_is_click": 1.0 if greedy and greedy.get("pred_type") == "click" else 0.0,
        "greedy_is_type": 1.0 if greedy and greedy.get("pred_type") == "type" else 0.0,
        "greedy_is_swipe": 1.0 if greedy and greedy.get("pred_type") == "swipe" else 0.0,
    }


def quantile_thresholds(rows: Sequence[Mapping[str, Any]], feature: str) -> list[float]:
    values = sorted({float(row["features"].get(feature, 0.0)) for row in rows})
    if not values:
        return []
    picks = {values[0], values[-1]}
    for quantile in (0.25, 0.5, 0.75):
        picks.add(values[min(len(values) - 1, max(0, int(round(quantile * (len(values) - 1)))) )])
    return sorted(picks)


def fit_conditional_quantile(train_rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str], grid: Sequence[float]) -> Dict[str, Any]:
    best = {"feature": None, "threshold": None, "polarity": 1, "w_true": fit_best_weight(train_rows, grid)["weight_stage1"], "w_false": fit_best_weight(train_rows, grid)["weight_stage1"], "train_acc": -1.0}
    for feature in feature_names:
        for threshold in quantile_thresholds(train_rows, feature):
            for polarity in (1, -1):
                for w_true in grid:
                    for w_false in grid:
                        correct = 0
                        for row in train_rows:
                            condition = (float(row["features"].get(feature, 0.0)) >= threshold) == (polarity == 1)
                            pick = aggregate_pick(row["candidates"], w_true if condition else w_false)
                            correct += int(bool(pick.get("is_correct")))
                        acc = correct / len(train_rows) if train_rows else 0.0
                        if acc > best["train_acc"]:
                            best = {"feature": feature, "threshold": threshold, "polarity": polarity, "w_true": w_true, "w_false": w_false, "train_acc": acc}
    return best


def apply_conditional(model: Mapping[str, Any], row: Mapping[str, Any]) -> Tuple[Dict[str, Any], float]:
    feature = model.get("feature")
    if feature is None:
        weight = float(model["w_true"])
    else:
        condition = (float(row["features"].get(str(feature), 0.0)) >= float(model["threshold"])) == (int(model["polarity"]) == 1)
        weight = float(model["w_true"] if condition else model["w_false"])
    return aggregate_pick(row["candidates"], weight), weight


def apply_models(train_rows: list[Dict[str, Any]], test_rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    feature_names = sorted(train_rows[0]["features"].keys())
    conditional_features = [
        "stage1_norm_margin", "stage2_norm_margin", "stage1_stage2_agree", "stage1_pick_is_greedy", "stage2_pick_is_greedy",
        "n_pred_types", "control_key_spread", "control_text_spread", "frac_click", "frac_type", "frac_swipe", "frac_no_control", "n_distinct",
    ]
    grid = [round(value / 20.0, 2) for value in range(21)]
    fast_grid = [round(value / 10.0, 2) for value in range(11)]
    router_model = fit_best_stump(train_rows, feature_names, default_route="stage1")
    scalar_weight = fit_best_weight(train_rows, grid)
    conditional = fit_conditional_quantile(train_rows, conditional_features, fast_grid)
    pipeline = fit_pipeline(train_rows, thresholds=grid)
    for row in test_rows:
        route = apply_stump(router_model, row["features"])
        router_pick = row["stage2_pick"] if route == "stage2" else row["stage1_pick"]
        row.update({"router_route": route, "router_candidate_id": router_pick["representative_candidate_id"], "router_distinct_key": router_pick["distinct_key"], "router_correct": bool(router_pick.get("is_correct"))})
        aggregate = aggregate_pick(row["candidates"], scalar_weight["weight_stage1"])
        row.update({"aggregate_weight_stage1": scalar_weight["weight_stage1"], "aggregate_candidate_id": aggregate["representative_candidate_id"], "aggregate_distinct_key": aggregate["distinct_key"], "aggregate_correct": bool(aggregate.get("is_correct"))})
        cond_pick, cond_weight = apply_conditional(conditional, row)
        row.update({"conditional_aggregate_weight_stage1": cond_weight, "conditional_aggregate_candidate_id": cond_pick["representative_candidate_id"], "conditional_aggregate_distinct_key": cond_pick["distinct_key"], "conditional_aggregate_correct": bool(cond_pick.get("is_correct"))})
        pipe = pipeline_pick(row["candidates"], float(pipeline["threshold"]), bool(pipeline["remove_stage2_losers"]))
        row.update({"pipeline_threshold": pipeline["threshold"], "pipeline_remove_stage2_losers": pipeline["remove_stage2_losers"], "pipeline_candidate_id": pipe["representative_candidate_id"], "pipeline_distinct_key": pipe["distinct_key"], "pipeline_correct": bool(pipe.get("is_correct")), "pipeline_survivors": pipe["n_pipeline_survivors"], "pipeline_removed": pipe["pipeline_removed"]})
    return {"router": router_model, "scalar_weight": scalar_weight, "conditional_weight": conditional, "pipeline": pipeline}


def oracle_by_subset(rows: Sequence[Mapping[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    groups: Dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("subset"))].append(row)
    correct = 0
    policy = {}
    for subset, group in groups.items():
        s1 = sum(1 for row in group if row.get("stage1_correct")) / len(group)
        s2 = sum(1 for row in group if row.get("stage2_correct")) / len(group)
        chosen = "stage1" if s1 >= s2 else "stage2"
        correct += sum(1 for row in group if row.get("stage1_correct" if chosen == "stage1" else "stage2_correct"))
        policy[subset] = {"n": len(group), "chosen": chosen, "stage1_accuracy": s1, "stage2_accuracy": s2, "small_n": len(group) < 10}
    return correct / len(rows) if rows else 0.0, policy


def metric_lift(acc: float) -> Tuple[float, float]:
    lift = projected_tsr_lift_pp(acc, 488 / 862) or 0.0
    return lift, lift / ORACLE_CRITICAL_TSR_CEILING_PP


def summarize(train_rows: Sequence[Mapping[str, Any]], test_rows: list[Dict[str, Any]], models: Mapping[str, Any], output_dir: Path, train_episode_intersection: Sequence[str]) -> Dict[str, Any]:
    fields = {
        "stage1_cot_vote_k8": "stage1_correct",
        "stage2_tournament": "stage2_correct",
        "router": "router_correct",
        "weighted_aggregation": "aggregate_correct",
        "conditional_weighted_aggregation": "conditional_aggregate_correct",
        "reject_then_select": "pipeline_correct",
    }
    oracle_subset, oracle_policy = oracle_by_subset(test_rows)
    selection = {
        "oracle_in_pool": 1.0,
        "oracle_routing_by_subset": oracle_subset,
        "oracle_routing_stage1_or_stage2": fraction(test_rows, "oracle_routing_correct"),
        "greedy": fraction(test_rows, "greedy_correct"),
        "sample_order_first": fraction(test_rows, "first_sample_correct"),
    }
    for name, field in fields.items():
        selection[name] = fraction(test_rows, field)
    depth = {}
    for key, item in summarize_by(test_rows, "depth_bin", list(fields.values())).items():
        depth[key] = {"n": item["n"]}
        for name, field in fields.items():
            depth[key][name] = item[field]
    subset = {}
    for key, item in summarize_by(test_rows, "subset", list(fields.values())).items():
        subset[key] = {"n": item["n"], "small_n": item["n"] < 10}
        for name, field in fields.items():
            subset[key][name] = item[field]
    reject = {
        "stage1_cot_vote_k8": sum(reject_greedy_for_pick(row, "stage1_pick_key") for row in test_rows) / len(test_rows),
        "stage2_tournament": sum(reject_greedy_for_pick(row, "stage2_pick_key") for row in test_rows) / len(test_rows),
        "router": sum(reject_greedy_for_pick(row, "router_distinct_key") for row in test_rows) / len(test_rows),
        "weighted_aggregation": sum(reject_greedy_for_pick(row, "aggregate_distinct_key") for row in test_rows) / len(test_rows),
        "conditional_weighted_aggregation": sum(reject_greedy_for_pick(row, "conditional_aggregate_distinct_key") for row in test_rows) / len(test_rows),
        "reject_then_select": sum(reject_greedy_for_pick(row, "pipeline_distinct_key") for row in test_rows) / len(test_rows),
    }
    methods = ["weighted_aggregation", "conditional_weighted_aggregation", "router", "reject_then_select"]
    best = max(methods, key=lambda name: selection[name] or 0.0)
    best_lift, best_frac = metric_lift(selection[best] or 0.0)
    stage1_acc = selection["stage1_cot_vote_k8"] or 0.0
    best_deep = depth.get("deep_21_50", {}).get(best) or 0.0
    stage1_deep = depth.get("deep_21_50", {}).get("stage1_cot_vote_k8") or 0.0
    scalar_acc = selection["weighted_aggregation"] or 0.0
    conditional_acc = selection["conditional_weighted_aggregation"] or 0.0
    if scalar_acc >= stage1_acc + 0.02 and best_deep >= stage1_deep:
        gate = "STRICT COMBINATION HOLDS"
        paper_method = "weighted_aggregation"
    elif conditional_acc >= stage1_acc + 0.02 and scalar_acc < stage1_acc + 0.02:
        gate = "CONDITIONAL-OVERFIT CONFIRMED"
        paper_method = "stage1_cot_vote_k8" if stage1_acc >= scalar_acc else "weighted_aggregation"
    else:
        gate = "STRICT COMBINATION SHRINKS"
        paper_method = "stage1_cot_vote_k8"
    summary = {
        "n_train_steps": len(train_rows),
        "n_test_steps": len(test_rows),
        "train_episodes": len({row["episode_key"] for row in train_rows}),
        "test_episodes": len({row["episode_key"] for row in test_rows}),
        "episode_intersection_count": len(train_episode_intersection),
        "episode_intersection_examples": list(train_episode_intersection)[:20],
        "selection_accuracy": selection,
        "depth_stratified": depth,
        "subset_stratified": subset,
        "reject_greedy": reject,
        "oracle_routing_by_subset_policy": oracle_policy,
        "models": models,
        "slice_diagnostic_reference": {"conditional_weighted_aggregation": 0.45, "stage1_cot_vote_k8": 0.375, "stage2_tournament": 0.365},
        "optimism_gap_vs_slice": {"conditional_weighted_aggregation": 0.45 - (selection["conditional_weighted_aggregation"] or 0.0)},
        "best_method": best,
        "paper_method": paper_method,
        "projected_tsr_lift_pp_best": best_lift,
        "projected_ceiling_fraction_best": best_frac,
        "gate": gate,
        "audit_note": "Router/weights fitted on TRAIN-side verifier scores and evaluated once on upstream TEST episodes. Subset/depth tags are excluded from features and used only for reporting. Small-n subsets are flagged with small_n=true.",
    }
    write_json(output_dir / "strict_summary.json", summary)
    return summary


def report(summary: Mapping[str, Any], output_dir: Path) -> None:
    lines = ["# Strict Train/Test Verifier Aggregation", ""]
    lines.append("## Selection Accuracy on Held-Out TEST")
    lines.append("")
    lines.append("| method | accuracy | projected TSR lift | fraction of +23.77pp ceiling |")
    lines.append("|---|---:|---:|---:|")
    for name, acc in summary["selection_accuracy"].items():
        lift, frac = metric_lift(acc or 0.0)
        lines.append(f"| {name} | {(acc or 0.0)*100:.2f}% | {lift:.2f}pp | {frac*100:.2f}% |")
    lines.append("")
    lines.append(f"Train/test episodes disjoint: `{summary['episode_intersection_count'] == 0}` (`intersection_count={summary['episode_intersection_count']}`).")
    lines.append(f"Best method on TEST: `{summary['best_method']}` at `{summary['selection_accuracy'][summary['best_method']]*100:.2f}%`.")
    lines.append(f"Paper-preferred method: `{summary['paper_method']}`.")
    lines.append("")
    lines.append("## Depth-Stratified")
    lines.append("")
    fields = ["stage1_cot_vote_k8", "stage2_tournament", "weighted_aggregation", "conditional_weighted_aggregation", "router", "reject_then_select"]
    lines.append("| depth | n | " + " | ".join(fields) + " |")
    lines.append("|---|---:|" + "---:|" * len(fields))
    for depth, item in summary["depth_stratified"].items():
        lines.append(f"| {depth} | {item['n']} | " + " | ".join(f"{(item.get(field) or 0.0)*100:.2f}%" for field in fields) + " |")
    lines.append("")
    lines.append("## Per-Subset")
    lines.append("")
    lines.append("| subset | n | small-n | " + " | ".join(fields) + " |")
    lines.append("|---|---:|---:|" + "---:|" * len(fields))
    for subset, item in summary["subset_stratified"].items():
        lines.append(f"| {subset} | {item['n']} | {item['small_n']} | " + " | ".join(f"{(item.get(field) or 0.0)*100:.2f}%" for field in fields) + " |")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(f"**{summary['gate']}**")
    lines.append("")
    lines.append("## Audit")
    lines.append("")
    lines.append(summary["audit_note"])
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{output_dir / 'strict_eval.md'}`")
    lines.append(f"- `{output_dir / 'strict_summary.json'}`")
    lines.append(f"- `{output_dir / 'per_step.jsonl'}`")
    lines.append("")
    lines.append("STOP for review.")
    (output_dir / "strict_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: row.get(key) for key in [
        "target_id", "episode_id", "episode_key", "split", "step_idx", "subset", "depth_bin", "features", "stage1_correct", "stage2_correct", "router_route", "router_correct", "aggregate_correct", "conditional_aggregate_correct", "pipeline_correct", "greedy_correct", "first_sample_correct", "oracle_routing_correct",
    ]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-stage1-per-step", required=True)
    parser.add_argument("--train-stage2-per-step", required=True)
    parser.add_argument("--test-stage1-per-step", required=True)
    parser.add_argument("--test-stage2-per-step", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = build_rows(read_jsonl(Path(args.train_stage1_per_step)), read_jsonl(Path(args.train_stage2_per_step)))
    test_rows = build_rows(read_jsonl(Path(args.test_stage1_per_step)), read_jsonl(Path(args.test_stage2_per_step)))
    models = apply_models(train_rows, test_rows)
    intersection = sorted({row["episode_key"] for row in train_rows} & {row["episode_key"] for row in test_rows})
    summary = summarize(train_rows, test_rows, models, output_dir, intersection)
    write_jsonl(output_dir / "per_step.jsonl", [compact_row(row) for row in test_rows])
    report(summary, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "gate": summary["gate"], "paper_method": summary["paper_method"], "paper_accuracy": summary["selection_accuracy"][summary["paper_method"]], "best_method": summary["best_method"], "best_accuracy": summary["selection_accuracy"][summary["best_method"]]}, indent=2), flush=True)


if __name__ == "__main__":
    main()