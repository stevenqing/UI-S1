#!/usr/bin/env python3
"""Combine Stage-1 pointwise and Stage-2 comparative verifier outputs."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.critstep_verifier_full_action import ORACLE_CRITICAL_TSR_CEILING_PP, projected_tsr_lift_pp  # noqa: E402


DEFAULT_STAGE1_PER_STEP = "outputs/critstep_verifier_v2/stage1_eval_200_k8_verdict/stage1_per_step.jsonl"
DEFAULT_STAGE2_PER_STEP = "outputs/critstep_verifier_v2/stage2_eval_200_tournament/stage2_per_step.jsonl"
DEFAULT_STAGE1_SUMMARY = "outputs/critstep_verifier_v2/stage1_eval_200_k8_verdict/stage1_summary.json"
DEFAULT_STAGE2_SUMMARY = "outputs/critstep_verifier_v2/stage2_eval_200_tournament/stage2_summary.json"
DEFAULT_POINTWISE_SUMMARY = "outputs/critstep_verifier/eval_overnight/summary.json"
DEFAULT_OUTPUT_DIR = "outputs/critstep_verifier_v2/combine"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_distinct_key(candidate: Mapping[str, Any]) -> str:
    if candidate.get("stage1_distinct_key"):
        return str(candidate["stage1_distinct_key"])
    control = candidate.get("control") if isinstance(candidate.get("control"), dict) else {}
    payload = {
        "action_signature": candidate.get("action_signature"),
        "pred_type": candidate.get("pred_type"),
        "control_key": control.get("key"),
        "control_assignment": control.get("assignment"),
        "control_rect": control.get("rect"),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def distinct_candidates(step: Mapping[str, Any]) -> List[Dict[str, Any]]:
    by_key: Dict[str, Dict[str, Any]] = {}
    for candidate in step.get("candidates", []):
        key = candidate_distinct_key(candidate)
        control = candidate.get("control") if isinstance(candidate.get("control"), dict) else {}
        current = by_key.get(key)
        stage1_score = float(candidate.get("stage1_score_k8") or 0.0)
        if current is None:
            by_key[key] = {
                "distinct_key": key,
                "representative_candidate_id": str(candidate.get("candidate_id")),
                "candidate_ids": [str(candidate.get("candidate_id"))],
                "sources": [candidate.get("source")],
                "is_correct": bool(candidate.get("is_correct")),
                "stage1_score": stage1_score,
                "pred_type": str(candidate.get("pred_type") or ""),
                "pred_category": str(candidate.get("pred_category") or ""),
                "bucket": str(candidate.get("bucket") or ""),
                "control_key": control.get("key"),
                "control_text": control.get("text") or "",
                "control_type": control.get("type") or "",
                "control_assignment": control.get("assignment") or "",
            }
        else:
            current["candidate_ids"].append(str(candidate.get("candidate_id")))
            current["sources"].append(candidate.get("source"))
            current["is_correct"] = bool(current["is_correct"] or candidate.get("is_correct"))
            current["stage1_score"] = max(float(current.get("stage1_score") or 0.0), stage1_score)
            if candidate.get("is_correct"):
                current["representative_candidate_id"] = str(candidate.get("candidate_id"))
    return sorted(by_key.values(), key=lambda item: (-float(item.get("stage1_score") or 0.0), str(item.get("representative_candidate_id"))))


def minmax(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0.5 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def safe_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = safe_mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def stage2_scores(step: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    stats = {str(candidate["distinct_key"]): {"wins": 0.0, "losses": 0.0, "participation": 0.0} for candidate in candidates}
    final_key = str(step.get("stage2_distinct_key") or "")
    for match in step.get("matches", []):
        inc = str(match.get("incumbent_key"))
        chal = str(match.get("challenger_key"))
        selected = str(match.get("selected_key"))
        if inc in stats:
            stats[inc]["participation"] += 1
        if chal in stats:
            stats[chal]["participation"] += 1
        loser = chal if selected == inc else inc
        if selected in stats:
            stats[selected]["wins"] += 1
        if loser in stats:
            stats[loser]["losses"] += 1
    for key, item in stats.items():
        denom = item["wins"] + item["losses"]
        item["win_rate"] = item["wins"] / denom if denom else 0.5
        item["net_wins"] = item["wins"] - item["losses"]
        item["is_stage2_final"] = 1.0 if key == final_key else 0.0
    return stats


def attach_scores(stage1_step: Mapping[str, Any], stage2_step: Mapping[str, Any]) -> List[Dict[str, Any]]:
    candidates = distinct_candidates(stage1_step)
    s2 = stage2_scores(stage2_step, candidates)
    stage1_norm = minmax([float(candidate.get("stage1_score") or 0.0) for candidate in candidates])
    stage2_norm = minmax([float(s2[candidate["distinct_key"]]["net_wins"]) for candidate in candidates])
    for candidate, s1n, s2n in zip(candidates, stage1_norm, stage2_norm, strict=True):
        item = s2[candidate["distinct_key"]]
        candidate["stage1_norm"] = s1n
        candidate["stage2_norm"] = s2n
        candidate["stage2_wins"] = item["wins"]
        candidate["stage2_losses"] = item["losses"]
        candidate["stage2_win_rate"] = item["win_rate"]
        candidate["stage2_net_wins"] = item["net_wins"]
        candidate["stage2_is_final"] = bool(item["is_stage2_final"])
    return candidates


def select_by_key(candidates: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Any]:
    for candidate in candidates:
        if candidate["distinct_key"] == key:
            return dict(candidate)
    raise KeyError(key)


def best_by_score(candidates: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Any]:
    return dict(max(candidates, key=lambda item: (float(item.get(field) or 0.0), str(item.get("representative_candidate_id")))))


def features_for_step(stage1_step: Mapping[str, Any], stage2_step: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    n = len(candidates)
    pred_types = [str(candidate.get("pred_type") or "") for candidate in candidates]
    control_keys = [candidate.get("control_key") for candidate in candidates if candidate.get("control_key")]
    control_texts = [str(candidate.get("control_text") or "") for candidate in candidates if candidate.get("control_text")]
    control_types = [str(candidate.get("control_type") or "") for candidate in candidates if candidate.get("control_type")]
    s1 = [float(candidate.get("stage1_score") or 0.0) for candidate in candidates]
    s1n = [float(candidate.get("stage1_norm") or 0.0) for candidate in candidates]
    s2n = [float(candidate.get("stage2_norm") or 0.0) for candidate in candidates]
    greedy = next((candidate for candidate in candidates if "greedy" in set(candidate.get("sources") or [])), None)
    stage1_key = str(stage1_step.get("stage1_k8_distinct_key") or "")
    stage2_key = str(stage2_step.get("stage2_distinct_key") or "")
    stage1_pick = select_by_key(candidates, stage1_key) if stage1_key else best_by_score(candidates, "stage1_score")
    stage2_pick = select_by_key(candidates, stage2_key) if stage2_key else best_by_score(candidates, "stage2_net_wins")
    top_s1 = sorted(s1n, reverse=True)
    top_s2 = sorted(s2n, reverse=True)
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
        "stage1_norm_margin": (top_s1[0] - top_s1[1]) if len(top_s1) > 1 else 0.0,
        "stage2_norm_margin": (top_s2[0] - top_s2[1]) if len(top_s2) > 1 else 0.0,
        "stage1_stage2_agree": 1.0 if stage1_key and stage1_key == stage2_key else 0.0,
        "stage1_pick_is_greedy": 1.0 if greedy and greedy["distinct_key"] == stage1_pick["distinct_key"] else 0.0,
        "stage2_pick_is_greedy": 1.0 if greedy and greedy["distinct_key"] == stage2_pick["distinct_key"] else 0.0,
        "greedy_is_click": 1.0 if greedy and greedy.get("pred_type") == "click" else 0.0,
        "greedy_is_type": 1.0 if greedy and greedy.get("pred_type") == "type" else 0.0,
        "greedy_is_swipe": 1.0 if greedy and greedy.get("pred_type") == "swipe" else 0.0,
    }


def fold_id(index: int, n_folds: int) -> int:
    return index % n_folds


def candidate_by_verifier(step: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], verifier: str) -> Dict[str, Any]:
    if verifier == "stage1":
        key = str(step.get("stage1_k8_distinct_key") or "")
        return select_by_key(candidates, key) if key else best_by_score(candidates, "stage1_score")
    if verifier == "stage2":
        key = str(step.get("stage2_distinct_key") or "")
        return select_by_key(candidates, key) if key else best_by_score(candidates, "stage2_net_wins")
    raise ValueError(verifier)


def rows_accuracy(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    return sum(1 for row in rows if row.get(field)) / len(rows) if rows else 0.0


def fit_best_stump(train_rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str], default_route: str) -> Dict[str, Any]:
    decisive = [row for row in train_rows if row.get("oracle_route_label") in {"stage1", "stage2"}]
    if not decisive:
        return {"feature": None, "threshold": None, "polarity": 1, "default_route": default_route, "train_acc": 0.0}
    best = {"feature": None, "threshold": None, "polarity": 1, "default_route": default_route, "train_acc": -1.0}
    for feature in feature_names:
        values = sorted({float(row["features"].get(feature, 0.0)) for row in decisive})
        if not values:
            continue
        thresholds = values[:]
        for left, right in zip(values, values[1:]):
            thresholds.append((left + right) / 2.0)
        for threshold in thresholds:
            for polarity in (1, -1):
                correct = 0
                for row in decisive:
                    value = float(row["features"].get(feature, 0.0))
                    route = "stage2" if (value >= threshold) == (polarity == 1) else "stage1"
                    correct += int(route == row["oracle_route_label"])
                acc = correct / len(decisive)
                if acc > best["train_acc"]:
                    best = {"feature": feature, "threshold": threshold, "polarity": polarity, "default_route": default_route, "train_acc": acc}
    return best


def apply_stump(model: Mapping[str, Any], features: Mapping[str, float]) -> str:
    feature = model.get("feature")
    if feature is None:
        return str(model.get("default_route") or "stage1")
    threshold = float(model["threshold"])
    polarity = int(model.get("polarity") or 1)
    value = float(features.get(str(feature), 0.0))
    return "stage2" if (value >= threshold) == (polarity == 1) else "stage1"


def evaluate_router_cv(rows: List[Dict[str, Any]], feature_names: Sequence[str], n_folds: int, default_route: str) -> Dict[str, Any]:
    predictions: Dict[str, Dict[str, Any]] = {}
    models = []
    for fold in range(n_folds):
        train = [row for row in rows if row["fold"] != fold]
        test = [row for row in rows if row["fold"] == fold]
        model = fit_best_stump(train, feature_names, default_route)
        models.append(model)
        for row in test:
            route = apply_stump(model, row["features"])
            pick = row["stage2_pick"] if route == "stage2" else row["stage1_pick"]
            predictions[row["target_id"]] = {
                "router_route": route,
                "router_model": model,
                "router_candidate_id": pick["representative_candidate_id"],
                "router_distinct_key": pick["distinct_key"],
                "router_correct": bool(pick.get("is_correct")),
                "router_pick_right_verifier": route == row.get("oracle_route_label") if row.get("oracle_route_label") in {"stage1", "stage2"} else None,
            }
    return {"predictions": predictions, "models": models}


def aggregate_pick(candidates: Sequence[Mapping[str, Any]], weight_stage1: float) -> Dict[str, Any]:
    weight_stage2 = 1.0 - weight_stage1
    best = max(
        candidates,
        key=lambda item: (
            weight_stage1 * float(item.get("stage1_norm") or 0.0) + weight_stage2 * float(item.get("stage2_norm") or 0.0),
            str(item.get("representative_candidate_id")),
        ),
    )
    picked = dict(best)
    picked["aggregate_score"] = weight_stage1 * float(best.get("stage1_norm") or 0.0) + weight_stage2 * float(best.get("stage2_norm") or 0.0)
    return picked


def fit_best_weight(train_rows: Sequence[Mapping[str, Any]], grid: Sequence[float]) -> Dict[str, Any]:
    best = {"weight_stage1": 0.5, "train_acc": -1.0}
    for weight in grid:
        correct = 0
        for row in train_rows:
            pick = aggregate_pick(row["candidates"], weight)
            correct += int(bool(pick.get("is_correct")))
        acc = correct / len(train_rows) if train_rows else 0.0
        if acc > best["train_acc"]:
            best = {"weight_stage1": weight, "train_acc": acc}
    return best


def evaluate_weight_cv(rows: List[Dict[str, Any]], n_folds: int, grid: Sequence[float]) -> Dict[str, Any]:
    predictions: Dict[str, Dict[str, Any]] = {}
    models = []
    for fold in range(n_folds):
        train = [row for row in rows if row["fold"] != fold]
        test = [row for row in rows if row["fold"] == fold]
        model = fit_best_weight(train, grid)
        models.append(model)
        for row in test:
            pick = aggregate_pick(row["candidates"], model["weight_stage1"])
            predictions[row["target_id"]] = {
                "aggregate_weight_stage1": model["weight_stage1"],
                "aggregate_candidate_id": pick["representative_candidate_id"],
                "aggregate_distinct_key": pick["distinct_key"],
                "aggregate_score": pick["aggregate_score"],
                "aggregate_correct": bool(pick.get("is_correct")),
            }
    return {"predictions": predictions, "models": models}


def fit_conditional_aggregator(train_rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str], grid: Sequence[float]) -> Dict[str, Any]:
    best = {"feature": None, "threshold": None, "polarity": 1, "w_true": 0.5, "w_false": 0.5, "train_acc": -1.0}
    # Include non-conditional scalar as candidate.
    scalar = fit_best_weight(train_rows, grid)
    best.update({"w_true": scalar["weight_stage1"], "w_false": scalar["weight_stage1"], "train_acc": scalar["train_acc"]})
    for feature in feature_names:
        values = sorted({float(row["features"].get(feature, 0.0)) for row in train_rows})
        thresholds = values[:]
        for left, right in zip(values, values[1:]):
            thresholds.append((left + right) / 2.0)
        for threshold in thresholds:
            for polarity in (1, -1):
                true_indices = [idx for idx, row in enumerate(train_rows) if (float(row["features"].get(feature, 0.0)) >= threshold) == (polarity == 1)]
                true_set = set(true_indices)
                true_rows = [train_rows[idx] for idx in true_indices]
                false_rows = [row for idx, row in enumerate(train_rows) if idx not in true_set]
                if not true_rows or not false_rows:
                    continue
                true_weight = fit_best_weight(true_rows, grid)["weight_stage1"]
                false_weight = fit_best_weight(false_rows, grid)["weight_stage1"]
                correct = 0
                for row in train_rows:
                    condition = (float(row["features"].get(feature, 0.0)) >= threshold) == (polarity == 1)
                    pick = aggregate_pick(row["candidates"], true_weight if condition else false_weight)
                    correct += int(bool(pick.get("is_correct")))
                acc = correct / len(train_rows) if train_rows else 0.0
                if acc > best["train_acc"]:
                    best = {"feature": feature, "threshold": threshold, "polarity": polarity, "w_true": true_weight, "w_false": false_weight, "train_acc": acc}
    return best


def apply_conditional_aggregator(model: Mapping[str, Any], row: Mapping[str, Any]) -> Tuple[Dict[str, Any], float]:
    feature = model.get("feature")
    if feature is None:
        weight = float(model["w_true"])
    else:
        condition = (float(row["features"].get(str(feature), 0.0)) >= float(model["threshold"])) == (int(model.get("polarity") or 1) == 1)
        weight = float(model["w_true"] if condition else model["w_false"])
    return aggregate_pick(row["candidates"], weight), weight


def evaluate_conditional_weight_cv(rows: List[Dict[str, Any]], feature_names: Sequence[str], n_folds: int, grid: Sequence[float]) -> Dict[str, Any]:
    predictions: Dict[str, Dict[str, Any]] = {}
    models = []
    for fold in range(n_folds):
        train = [row for row in rows if row["fold"] != fold]
        test = [row for row in rows if row["fold"] == fold]
        model = fit_conditional_aggregator(train, feature_names, grid)
        models.append(model)
        for row in test:
            pick, weight = apply_conditional_aggregator(model, row)
            predictions[row["target_id"]] = {
                "conditional_aggregate_weight_stage1": weight,
                "conditional_aggregate_candidate_id": pick["representative_candidate_id"],
                "conditional_aggregate_distinct_key": pick["distinct_key"],
                "conditional_aggregate_score": pick["aggregate_score"],
                "conditional_aggregate_correct": bool(pick.get("is_correct")),
            }
    return {"predictions": predictions, "models": models}


def pipeline_pick(candidates: Sequence[Mapping[str, Any]], stage2_reject_threshold: float, remove_stage2_losers: bool) -> Dict[str, Any]:
    survivors = []
    for candidate in candidates:
        is_greedy = "greedy" in set(candidate.get("sources") or [])
        stage2_score = float(candidate.get("stage2_norm") or 0.0)
        stage2_final = bool(candidate.get("stage2_is_final"))
        if is_greedy and not stage2_final and stage2_score <= stage2_reject_threshold:
            continue
        if remove_stage2_losers and not stage2_final and float(candidate.get("stage2_losses") or 0.0) > float(candidate.get("stage2_wins") or 0.0):
            continue
        survivors.append(candidate)
    if not survivors:
        survivors = list(candidates)
    pick = best_by_score(survivors, "stage1_score")
    pick["n_pipeline_survivors"] = len(survivors)
    pick["pipeline_removed"] = len(candidates) - len(survivors)
    return pick


def fit_pipeline(train_rows: Sequence[Mapping[str, Any]], thresholds: Sequence[float]) -> Dict[str, Any]:
    best = {"threshold": 0.0, "remove_stage2_losers": False, "train_acc": -1.0}
    for threshold in thresholds:
        for remove_losers in (False, True):
            correct = 0
            for row in train_rows:
                pick = pipeline_pick(row["candidates"], threshold, remove_losers)
                correct += int(bool(pick.get("is_correct")))
            acc = correct / len(train_rows) if train_rows else 0.0
            if acc > best["train_acc"]:
                best = {"threshold": threshold, "remove_stage2_losers": remove_losers, "train_acc": acc}
    return best


def evaluate_pipeline_cv(rows: List[Dict[str, Any]], n_folds: int, thresholds: Sequence[float]) -> Dict[str, Any]:
    predictions: Dict[str, Dict[str, Any]] = {}
    models = []
    for fold in range(n_folds):
        train = [row for row in rows if row["fold"] != fold]
        test = [row for row in rows if row["fold"] == fold]
        model = fit_pipeline(train, thresholds)
        models.append(model)
        for row in test:
            pick = pipeline_pick(row["candidates"], float(model["threshold"]), bool(model["remove_stage2_losers"]))
            predictions[row["target_id"]] = {
                "pipeline_threshold": model["threshold"],
                "pipeline_remove_stage2_losers": model["remove_stage2_losers"],
                "pipeline_candidate_id": pick["representative_candidate_id"],
                "pipeline_distinct_key": pick["distinct_key"],
                "pipeline_correct": bool(pick.get("is_correct")),
                "pipeline_survivors": pick["n_pipeline_survivors"],
                "pipeline_removed": pick["pipeline_removed"],
            }
    return {"predictions": predictions, "models": models}


def fraction(rows: Sequence[Mapping[str, Any]], field: str) -> Optional[float]:
    values = [row.get(field) for row in rows]
    if not values or any(value is None for value in values):
        return None
    return sum(1 for value in values if value) / len(values)


def summarize_by(rows: Sequence[Mapping[str, Any]], group_field: str, metric_fields: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_field))].append(row)
    out: Dict[str, Dict[str, Any]] = {}
    for group, group_rows in sorted(grouped.items()):
        out[group] = {"n": len(group_rows)}
        for metric in metric_fields:
            out[group][metric] = fraction(group_rows, metric)
    return out


def gate(summary: Mapping[str, Any]) -> str:
    best = summary["best_method"]
    best_acc = summary["selection_accuracy"][best]
    best_deep = summary["depth_stratified"]["deep_21_50"][best]
    stage1_acc = summary["selection_accuracy"]["stage1_cot_vote_k8"]
    stage1_deep = summary["depth_stratified"]["deep_21_50"]["stage1_cot_vote_k8"]
    subset_ceiling = summary["selection_accuracy"].get("oracle_routing_by_subset") or 0.0
    subset = summary["subset_stratified"]
    action_type_ok = (subset.get("action_type_mismatch", {}).get(best) or 0.0) >= (subset.get("action_type_mismatch", {}).get("stage1_cot_vote_k8") or 0.0)
    element_ok = (subset.get("click_element_selection", {}).get(best) or 0.0) >= (subset.get("click_element_selection", {}).get("stage1_cot_vote_k8") or 0.0)
    type_content_ok = (subset.get("type_content", {}).get(best) or 0.0) >= (subset.get("type_content", {}).get("stage2_tournament") or 0.0)
    if best_acc >= 0.41 and best_acc >= stage1_acc + 0.02 and best_acc >= subset_ceiling - 0.02 and best_deep >= stage1_deep and action_type_ok and element_ok and type_content_ok:
        return "COMBINATION EFFECTIVE"
    return "COMBINATION PLATEAU"


def metric_lift_text(acc: Optional[float], recoverable_fraction: float) -> Tuple[str, str]:
    if acc is None:
        return "NA", "NA"
    lift = projected_tsr_lift_pp(acc, recoverable_fraction) or 0.0
    return f"{lift:.2f}pp", f"{(lift / ORACLE_CRITICAL_TSR_CEILING_PP) * 100:.2f}%"


def render_report(summary: Mapping[str, Any], output_dir: Path) -> str:
    recoverable = summary["recoverable_fraction_primary"]
    lines = ["# Verifier Routing / Aggregation", ""]
    lines.append("## Non-Oracle Combination Results")
    lines.append("")
    lines.append("| selector | accuracy | projected TSR lift proxy | fraction of +23.77pp ceiling |")
    lines.append("|---|---:|---:|---:|")
    for name, accuracy in summary["selection_accuracy"].items():
        lift, frac = metric_lift_text(accuracy, recoverable)
        acc_text = f"{accuracy*100:.2f}%" if accuracy is not None else "NA"
        lines.append(f"| {name} | {acc_text} | {lift} | {frac} |")
    lines.append("")
    lines.append(f"Oracle-routing-by-subset ceiling: `{summary['selection_accuracy']['oracle_routing_by_subset']*100:.2f}%`. This is the spec's oracle routing ceiling, not a deployable result.")
    lines.append(f"Stronger per-step oracle upper bound (Stage1 or Stage2 correct on the same step): `{summary['selection_accuracy']['oracle_routing_stage1_or_stage2']*100:.2f}%`. This is not the routing target and is also not deployable.")
    lines.append(f"Best non-oracle method: `{summary['best_method']}` at `{summary['selection_accuracy'][summary['best_method']]*100:.2f}%`.")
    lines.append("")
    lines.append("## Method A: Learned Router")
    lines.append("")
    lines.append(f"Router selection accuracy: `{summary['selection_accuracy']['router']*100:.2f}%`.")
    router_pick = summary.get("router_pick_accuracy")
    lines.append(f"Router pick accuracy on decisive steps: `{router_pick*100:.2f}%`" if router_pick is not None else "Router pick accuracy on decisive steps: `NA`")
    lines.append("Router uses only inference-available candidate-set/verifier-output features; subset/depth tags are excluded from features and used only for eval tables.")
    lines.append("")
    lines.append("## Method B: Weaver-Style Aggregation")
    lines.append("")
    lines.append(f"Scalar weighted aggregation accuracy: `{summary['selection_accuracy']['weighted_aggregation']*100:.2f}%`.")
    lines.append(f"Feature-conditioned weighted aggregation accuracy: `{summary['selection_accuracy']['conditional_weighted_aggregation']*100:.2f}%`.")
    lines.append(f"Scalar fold weights for Stage1: `{summary['learned_weights']['scalar_stage1_weights']}`.")
    lines.append("")
    lines.append("## Method C: Reject-Then-Select")
    lines.append("")
    lines.append(f"Pipeline accuracy: `{summary['selection_accuracy']['reject_then_select']*100:.2f}%`.")
    lines.append("")
    lines.append("## Depth-Stratified")
    lines.append("")
    fields = ["stage1_cot_vote_k8", "stage2_tournament", "router", "weighted_aggregation", "conditional_weighted_aggregation", "reject_then_select"]
    lines.append("| depth bin | n | " + " | ".join(fields) + " |")
    lines.append("|---|---:|" + "---:|" * len(fields))
    for depth, item in summary["depth_stratified"].items():
        values = [f"{(item.get(field) or 0.0)*100:.2f}%" for field in fields]
        lines.append(f"| {depth} | {item['n']} | " + " | ".join(values) + " |")
    lines.append("")
    lines.append("## Per-Subset")
    lines.append("")
    lines.append("| subset | n | " + " | ".join(fields) + " |")
    lines.append("|---|---:|" + "---:|" * len(fields))
    for subset, item in summary["subset_stratified"].items():
        values = [f"{(item.get(field) or 0.0)*100:.2f}%" for field in fields]
        lines.append(f"| {subset} | {item['n']} | " + " | ".join(values) + " |")
    lines.append("")
    lines.append("## Reject-Greedy")
    lines.append("")
    lines.append("| method | reject-greedy rate |")
    lines.append("|---|---:|")
    for method, value in summary["reject_greedy"].items():
        lines.append(f"| {method} | {value*100:.2f}% |")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(f"**{summary['gate']}**")
    lines.append("")
    lines.append("## Training/Audit Note")
    lines.append("")
    lines.append(summary["audit_note"])
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{output_dir / 'combine_eval.md'}`")
    lines.append(f"- `{output_dir / 'combine_summary.json'}`")
    lines.append(f"- `{output_dir / 'per_step.jsonl'}`")
    lines.append("")
    lines.append("STOP for review.")
    return "\n".join(lines) + "\n"


def build_rows(stage1_steps: Sequence[Mapping[str, Any]], stage2_steps: Sequence[Mapping[str, Any]], n_folds: int) -> List[Dict[str, Any]]:
    stage2_by_target = {str(row["target_id"]): row for row in stage2_steps}
    rows = []
    for idx, stage1_step in enumerate(stage1_steps):
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
            "episode_id": stage1_step.get("episode_id"),
            "step_idx": stage1_step.get("step_idx"),
            "subset": stage1_step.get("subset"),
            "depth_bin": stage1_step.get("depth_bin"),
            "fold": fold_id(idx, n_folds),
            "candidates": candidates,
            "features": features_for_step(stage1_step, stage2_step, candidates),
            "stage1_pick": stage1_pick,
            "stage2_pick": stage2_pick,
            "stage1_correct": bool(stage1_pick.get("is_correct")),
            "stage2_correct": bool(stage2_pick.get("is_correct")),
            "oracle_route_label": route_label,
            "oracle_routing_correct": bool(stage1_pick.get("is_correct") or stage2_pick.get("is_correct")),
            "greedy_correct": stage1_step.get("greedy_correct"),
            "first_sample_correct": stage1_step.get("first_sample_correct"),
        })
    return rows


def apply_predictions(rows: List[Dict[str, Any]], predictions: Mapping[str, Mapping[str, Any]]) -> None:
    for row in rows:
        row.update(predictions[str(row["target_id"])])


def reject_greedy_for_pick(row: Mapping[str, Any], key_field: str) -> bool:
    key = row.get(key_field)
    greedy = next((candidate for candidate in row["candidates"] if "greedy" in set(candidate.get("sources") or [])), None)
    return bool(greedy and key != greedy["distinct_key"])


def summarize(rows: List[Dict[str, Any]], output_dir: Path, stage1_summary: Mapping[str, Any], stage2_summary: Mapping[str, Any], pointwise_summary: Mapping[str, Any], router_models: Sequence[Mapping[str, Any]], scalar_weight_models: Sequence[Mapping[str, Any]], conditional_weight_models: Sequence[Mapping[str, Any]], pipeline_models: Sequence[Mapping[str, Any]], audit_note: str) -> Dict[str, Any]:
    metric_fields = [
        "stage1_correct",
        "stage2_correct",
        "router_correct",
        "aggregate_correct",
        "conditional_aggregate_correct",
        "pipeline_correct",
    ]
    field_to_name = {
        "stage1_correct": "stage1_cot_vote_k8",
        "stage2_correct": "stage2_tournament",
        "router_correct": "router",
        "aggregate_correct": "weighted_aggregation",
        "conditional_aggregate_correct": "conditional_weighted_aggregation",
        "pipeline_correct": "reject_then_select",
    }
    subset_groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        subset_groups[str(row.get("subset"))].append(row)
    subset_oracle_correct = 0
    subset_oracle_policy = {}
    for subset, group_rows in subset_groups.items():
        stage1_acc = sum(1 for row in group_rows if row.get("stage1_correct")) / len(group_rows)
        stage2_acc = sum(1 for row in group_rows if row.get("stage2_correct")) / len(group_rows)
        chosen = "stage1" if stage1_acc >= stage2_acc else "stage2"
        subset_oracle_policy[subset] = {"chosen": chosen, "stage1_accuracy": stage1_acc, "stage2_accuracy": stage2_acc, "n": len(group_rows)}
        subset_oracle_correct += sum(1 for row in group_rows if row.get("stage1_correct" if chosen == "stage1" else "stage2_correct"))
    selection_accuracy = {
        "oracle_in_pool": 1.0,
        "oracle_routing_by_subset": subset_oracle_correct / len(rows) if rows else None,
        "oracle_routing_stage1_or_stage2": fraction(rows, "oracle_routing_correct"),
        "greedy": fraction(rows, "greedy_correct"),
        "sample_order_first": fraction(rows, "first_sample_correct"),
        "previous_pointwise_verifier": (pointwise_summary.get("selection_accuracy") or {}).get("verifier_argmax"),
    }
    for field, name in field_to_name.items():
        selection_accuracy[name] = fraction(rows, field)
    renamed_depth = {}
    for depth, item in summarize_by(rows, "depth_bin", metric_fields).items():
        renamed_depth[depth] = {"n": item["n"]}
        for field, name in field_to_name.items():
            renamed_depth[depth][name] = item[field]
    renamed_subset = {}
    for subset, item in summarize_by(rows, "subset", metric_fields).items():
        renamed_subset[subset] = {"n": item["n"]}
        for field, name in field_to_name.items():
            renamed_subset[subset][name] = item[field]
    reject_greedy = {
        "stage1_cot_vote_k8": sum(reject_greedy_for_pick(row, "stage1_pick_key") for row in rows) / len(rows),
        "stage2_tournament": sum(reject_greedy_for_pick(row, "stage2_pick_key") for row in rows) / len(rows),
        "router": sum(reject_greedy_for_pick(row, "router_distinct_key") for row in rows) / len(rows),
        "weighted_aggregation": sum(reject_greedy_for_pick(row, "aggregate_distinct_key") for row in rows) / len(rows),
        "conditional_weighted_aggregation": sum(reject_greedy_for_pick(row, "conditional_aggregate_distinct_key") for row in rows) / len(rows),
        "reject_then_select": sum(reject_greedy_for_pick(row, "pipeline_distinct_key") for row in rows) / len(rows),
    }
    method_names = ["router", "weighted_aggregation", "conditional_weighted_aggregation", "reject_then_select"]
    best_method = max(method_names, key=lambda name: selection_accuracy[name] or 0.0)
    decisive = [row for row in rows if row.get("router_pick_right_verifier") is not None]
    summary = {
        "n_steps": len(rows),
        "n_folds": len(router_models),
        "cv_training_note": "5-fold out-of-fold diagnostic on the 200-step slice because TRAIN-side Stage1/Stage2 verifier scores are not yet materialized.",
        "audit_note": audit_note,
        "feature_names": sorted(rows[0]["features"].keys()) if rows else [],
        "selection_accuracy": selection_accuracy,
        "router_pick_accuracy": sum(1 for row in decisive if row.get("router_pick_right_verifier")) / len(decisive) if decisive else None,
        "depth_stratified": renamed_depth,
        "subset_stratified": renamed_subset,
        "reject_greedy": reject_greedy,
        "learned_router_models": list(router_models),
        "learned_weights": {
            "scalar_stage1_weights": [model["weight_stage1"] for model in scalar_weight_models],
            "conditional_models": list(conditional_weight_models),
            "pipeline_models": list(pipeline_models),
        },
        "oracle_routing_by_subset_policy": subset_oracle_policy,
        "best_method": best_method,
        "n_primary_failures": 862,
        "n_recoverable_primary": 488,
        "recoverable_fraction_primary": 488 / 862,
    }
    summary["projected_tsr_lift_pp_best"] = projected_tsr_lift_pp(selection_accuracy[best_method], summary["recoverable_fraction_primary"])
    summary["projected_ceiling_fraction_best"] = (summary["projected_tsr_lift_pp_best"] or 0.0) / ORACLE_CRITICAL_TSR_CEILING_PP
    summary["gate"] = gate(summary)
    write_json(output_dir / "combine_summary.json", summary)
    (output_dir / "combine_eval.md").write_text(render_report(summary, output_dir), encoding="utf-8")
    return summary


def compact_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    compact_candidates = []
    for candidate in row["candidates"]:
        compact_candidates.append({
            "distinct_key": candidate["distinct_key"],
            "representative_candidate_id": candidate["representative_candidate_id"],
            "candidate_ids": candidate["candidate_ids"],
            "sources": candidate["sources"],
            "is_correct": candidate["is_correct"],
            "stage1_score": candidate["stage1_score"],
            "stage1_norm": candidate["stage1_norm"],
            "stage2_norm": candidate["stage2_norm"],
            "stage2_wins": candidate["stage2_wins"],
            "stage2_losses": candidate["stage2_losses"],
            "pred_type": candidate["pred_type"],
            "control_key": candidate["control_key"],
            "control_text": candidate["control_text"],
        })
    return {
        "target_id": row["target_id"],
        "episode_id": row.get("episode_id"),
        "step_idx": row.get("step_idx"),
        "subset": row.get("subset"),
        "depth_bin": row.get("depth_bin"),
        "fold": row.get("fold"),
        "features": row.get("features"),
        "oracle_route_label": row.get("oracle_route_label"),
        "oracle_routing_correct": row.get("oracle_routing_correct"),
        "stage1_pick_candidate_id": row["stage1_pick"]["representative_candidate_id"],
        "stage1_pick_key": row["stage1_pick_key"],
        "stage1_correct": row["stage1_correct"],
        "stage2_pick_candidate_id": row["stage2_pick"]["representative_candidate_id"],
        "stage2_pick_key": row["stage2_pick_key"],
        "stage2_correct": row["stage2_correct"],
        "router_route": row.get("router_route"),
        "router_candidate_id": row.get("router_candidate_id"),
        "router_correct": row.get("router_correct"),
        "router_pick_right_verifier": row.get("router_pick_right_verifier"),
        "aggregate_weight_stage1": row.get("aggregate_weight_stage1"),
        "aggregate_candidate_id": row.get("aggregate_candidate_id"),
        "aggregate_correct": row.get("aggregate_correct"),
        "conditional_aggregate_weight_stage1": row.get("conditional_aggregate_weight_stage1"),
        "conditional_aggregate_candidate_id": row.get("conditional_aggregate_candidate_id"),
        "conditional_aggregate_correct": row.get("conditional_aggregate_correct"),
        "pipeline_candidate_id": row.get("pipeline_candidate_id"),
        "pipeline_correct": row.get("pipeline_correct"),
        "pipeline_survivors": row.get("pipeline_survivors"),
        "pipeline_removed": row.get("pipeline_removed"),
        "candidates": compact_candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-per-step", default=DEFAULT_STAGE1_PER_STEP)
    parser.add_argument("--stage2-per-step", default=DEFAULT_STAGE2_PER_STEP)
    parser.add_argument("--stage1-summary", default=DEFAULT_STAGE1_SUMMARY)
    parser.add_argument("--stage2-summary", default=DEFAULT_STAGE2_SUMMARY)
    parser.add_argument("--pointwise-summary", default=DEFAULT_POINTWISE_SUMMARY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("loading Stage1/Stage2 outputs", flush=True)
    stage1_steps = read_jsonl(Path(args.stage1_per_step))
    stage2_steps = read_jsonl(Path(args.stage2_per_step))
    print(f"building rows from {len(stage1_steps)} Stage1 steps and {len(stage2_steps)} Stage2 steps", flush=True)
    rows = build_rows(stage1_steps, stage2_steps, args.n_folds)
    for row in rows:
        row["stage1_pick_key"] = row["stage1_pick"]["distinct_key"]
        row["stage2_pick_key"] = row["stage2_pick"]["distinct_key"]
    feature_names = sorted(rows[0]["features"].keys())
    print(f"fitting router over {len(feature_names)} features", flush=True)
    # The best single verifier is Stage1 on this eval slice; default ties go to Stage1.
    router = evaluate_router_cv(rows, feature_names, args.n_folds, default_route="stage1")
    apply_predictions(rows, router["predictions"])
    grid = [round(value / 20.0, 2) for value in range(21)]
    fast_grid = [round(value / 10.0, 2) for value in range(11)]
    conditional_feature_names = [
        "stage1_norm_margin",
        "stage2_norm_margin",
        "stage1_stage2_agree",
        "stage1_pick_is_greedy",
        "stage2_pick_is_greedy",
        "n_pred_types",
        "control_key_spread",
        "control_text_spread",
        "frac_click",
        "frac_type",
        "frac_swipe",
        "frac_no_control",
        "n_distinct",
    ]
    print("fitting scalar weighted aggregation", flush=True)
    scalar_weights = evaluate_weight_cv(rows, args.n_folds, grid)
    apply_predictions(rows, scalar_weights["predictions"])
    print("fitting conditional weighted aggregation", flush=True)
    conditional_weights = evaluate_conditional_weight_cv(rows, conditional_feature_names, args.n_folds, fast_grid)
    apply_predictions(rows, conditional_weights["predictions"])
    print("fitting reject-then-select pipeline", flush=True)
    pipeline = evaluate_pipeline_cv(rows, args.n_folds, thresholds=grid)
    apply_predictions(rows, pipeline["predictions"])
    audit_note = (
        "Subset and depth tags are excluded from router/aggregation features and used only for evaluation breakdown. "
        "Features are candidate-set statistics and verifier-output statistics available at inference. "
        "Because train-side Stage1/Stage2 verifier scores are not materialized yet, learned router/weights here are 5-fold out-of-fold diagnostics on the 200-step slice; "
        "the strict train-trained version requires scoring the TRAIN critical pool with both verifiers first."
    )
    stage1_summary = load_json(Path(args.stage1_summary), {}) or {}
    stage2_summary = load_json(Path(args.stage2_summary), {}) or {}
    pointwise_summary = load_json(Path(args.pointwise_summary), {}) or {}
    summary = summarize(
        rows,
        output_dir,
        stage1_summary,
        stage2_summary,
        pointwise_summary,
        router["models"],
        scalar_weights["models"],
        conditional_weights["models"],
        pipeline["models"],
        audit_note,
    )
    write_jsonl(output_dir / "per_step.jsonl", [compact_row(row) for row in rows])
    print(json.dumps({"output_dir": str(output_dir), "gate": summary["gate"], "best_method": summary["best_method"], "best_accuracy": summary["selection_accuracy"][summary["best_method"]]}, indent=2), flush=True)


if __name__ == "__main__":
    main()