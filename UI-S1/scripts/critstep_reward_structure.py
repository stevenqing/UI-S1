#!/usr/bin/env python3
"""Spatial structure diagnostic for recoverable GUI-360 critical-step samples.

This is an offline diagnostic over the already-sampled C.1 pool. It does not
train or call a model. The goal is to decide whether the recoverable critical
answers look like coordinate refinements suitable for Gaussian-distance RLVR, or
like discrete element-selection errors that require verifier selection.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SAMPLES = "outputs/critstep_elicit/per_step.jsonl"
DEFAULT_TEST_DATA = "outputs/gui360_history_ab/original_eval/gui360_test_1000_balanced_reconstructed.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/critstep_reward_structure"
DEFAULT_SIGMAS = (25.0, 50.0, 100.0, 150.0, 250.0)

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_episodes(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row["episode_id"]): row for row in read_jsonl(path)}


def as_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        value = float(value)
        if not math.isfinite(value):
            return None
        return value
    except (TypeError, ValueError):
        return None


def valid_point(value: Any) -> Optional[Point]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x = as_float(value[0])
    y = as_float(value[1])
    if x is None or y is None:
        return None
    return (x, y)


def valid_bbox(value: Any) -> Optional[BBox]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    vals = [as_float(item) for item in value[:4]]
    if any(item is None for item in vals):
        return None
    x1, y1, x2, y2 = vals  # type: ignore[misc]
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def bbox_center(bbox: BBox) -> Point:
    left, top, right, bottom = bbox
    return ((left + right) / 2.0, (top + bottom) / 2.0)


def point_in_bbox(point: Optional[Point], bbox: Optional[BBox], margin: float = 0.0) -> bool:
    if point is None or bbox is None:
        return False
    x, y = point
    left, top, right, bottom = bbox
    return left - margin <= x <= right + margin and top - margin <= y <= bottom + margin


def action_type(action: Optional[Dict[str, Any]], fallback: str = "unknown") -> str:
    if not isinstance(action, dict):
        return fallback
    return str(action.get("action") or fallback or "unknown").strip().lower() or fallback


def action_point(action: Optional[Dict[str, Any]]) -> Optional[Point]:
    if not isinstance(action, dict):
        return None
    atype = action_type(action)
    if atype in {"click", "type", "wheel_mouse_input"}:
        return valid_point(action.get("coordinate"))
    if atype in {"swipe", "drag"}:
        start = valid_point(action.get("coordinate") or action.get("start_coordinate"))
        end = valid_point(action.get("endCoordinate") or action.get("end_coordinate"))
        if start and end:
            return ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
        return start or end
    return valid_point(action.get("coordinate"))


def gt_point(gt_action: Optional[Dict[str, Any]], bbox: Optional[BBox]) -> Optional[Point]:
    point = action_point(gt_action)
    if point is not None:
        return point
    if bbox is not None:
        return bbox_center(bbox)
    return None


def distance(p1: Optional[Point], p2: Optional[Point]) -> Optional[float]:
    if p1 is None or p2 is None:
        return None
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def screen_diag(width: int, height: int) -> float:
    return math.hypot(float(width), float(height))


def normalized_distance_px(value_px: Optional[float], width: int, height: int) -> Optional[float]:
    if value_px is None:
        return None
    diag = screen_diag(width, height)
    return value_px / diag if diag else None


def distance_bucket(value_px: Optional[float], near_px: float, far_px: float) -> str:
    if value_px is None:
        return "missing_point"
    if value_px <= near_px:
        return "near"
    if value_px <= far_px:
        return "mid"
    return "far"


def gaussian_reward(distance_px: Optional[float], sigma_px: float) -> Optional[float]:
    if distance_px is None:
        return None
    return math.exp(-(distance_px * distance_px) / (2.0 * sigma_px * sigma_px))


def quantize_point(point: Optional[Point], cell_px: float) -> str:
    if point is None:
        return "no_point"
    return f"q{int(point[0] // cell_px)}_{int(point[1] // cell_px)}"


def text_signature(action: Optional[Dict[str, Any]]) -> str:
    if not isinstance(action, dict):
        return ""
    text = str(action.get("text") or action.get("keys") or "")
    text = " ".join(text.split())
    return text[:40]


def action_signature(action: Optional[Dict[str, Any]], bbox: Optional[BBox], cell_px: float) -> str:
    atype = action_type(action)
    point = action_point(action)
    element = "gt_bbox" if point_in_bbox(point, bbox) else quantize_point(point, cell_px)
    text = text_signature(action)
    if text:
        return f"{atype}:{element}:text={text}"
    return f"{atype}:{element}"


def safe_ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def format_pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def format_optional(value: Optional[float], ndigits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{value:.{ndigits}f}"


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def summarize_values(values: Sequence[float]) -> Dict[str, Optional[float]]:
    return {
        "n": float(len(values)),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
    }


def pairwise_distances(points: Sequence[Point]) -> List[float]:
    return [distance(a, b) or 0.0 for a, b in combinations(points, 2)]


def majority_fraction(flags: Sequence[bool]) -> float:
    return safe_ratio(sum(1 for flag in flags if flag), len(flags))


def choose_temperature(value: str) -> float:
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default=DEFAULT_SAMPLES)
    parser.add_argument("--test-data", default=DEFAULT_TEST_DATA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temperature", type=choose_temperature, default=0.7)
    parser.add_argument("--population", default="critical")
    parser.add_argument("--near-px", type=float, default=50.0)
    parser.add_argument("--far-px", type=float, default=150.0)
    parser.add_argument("--modal-cell-px", type=float, default=50.0)
    parser.add_argument("--sigmas", default=",".join(str(v) for v in DEFAULT_SIGMAS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sigmas = [float(item) for item in str(args.sigmas).split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = read_episodes(Path(args.test_data))
    rows = []
    for row in read_jsonl(Path(args.samples)):
        if str(row.get("population")) != args.population:
            continue
        if float(row.get("temperature")) != float(args.temperature):
            continue
        if not row.get("recoverable"):
            continue
        rows.append(row)

    per_step: List[Dict[str, Any]] = []
    aggregate = Counter()
    gaussian_acc: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    binary_variances: List[float] = []
    distances_min: List[float] = []
    distances_median: List[float] = []
    spread_max_values: List[float] = []
    modal_wrong_fracs: List[float] = []

    for row in rows:
        episode_id = str(row["episode_id"])
        step_idx = int(row["step_idx"])
        episode = episodes.get(episode_id)
        if not episode:
            raise ValueError(f"missing episode {episode_id}")
        steps = episode.get("steps") or []
        if step_idx >= len(steps):
            raise ValueError(f"missing step {step_idx} in episode {episode_id}")
        step = steps[step_idx]
        image_w = int(step.get("image_w") or 1040)
        image_h = int(step.get("image_h") or 736)
        bbox = valid_bbox(step.get("bbox"))
        gt_action = step.get("action") if isinstance(step.get("action"), dict) else {}
        target_point = gt_point(gt_action, bbox)
        greedy = row.get("greedy") or {}
        greedy_action = greedy.get("pred_action") if isinstance(greedy, dict) else None
        greedy_point = action_point(greedy_action)
        greedy_distance_to_gt_px = distance(greedy_point, target_point)
        greedy_in_gt_bbox = point_in_bbox(greedy_point, bbox)
        samples = list(row.get("samples") or [])
        correct_samples = [sample for sample in samples if sample.get("success")]
        wrong_samples = [sample for sample in samples if not sample.get("success")]
        correct_points = [point for point in (action_point(sample.get("pred_action")) for sample in correct_samples) if point is not None]
        wrong_points = [point for point in (action_point(sample.get("pred_action")) for sample in wrong_samples) if point is not None]
        correct_to_greedy = [distance(point, greedy_point) for point in correct_points if greedy_point is not None]
        correct_to_greedy = [value for value in correct_to_greedy if value is not None]
        correct_to_gt = [distance(point, target_point) for point in correct_points if target_point is not None]
        correct_to_gt = [value for value in correct_to_gt if value is not None]
        wrong_to_gt = [distance(point, target_point) for point in wrong_points if target_point is not None]
        wrong_to_gt = [value for value in wrong_to_gt if value is not None]
        correct_in_gt_flags = [point_in_bbox(point, bbox) for point in correct_points]
        wrong_in_gt_flags = [point_in_bbox(point, bbox) for point in wrong_points]
        correct_in_gt_frac = majority_fraction(correct_in_gt_flags)
        wrong_in_gt_frac = majority_fraction(wrong_in_gt_flags)
        correct_missing_point = len(correct_samples) - len(correct_points)
        wrong_missing_point = len(wrong_samples) - len(wrong_points)
        if correct_to_greedy:
            min_dist = min(correct_to_greedy)
            med_dist = median(correct_to_greedy)
            distances_min.append(min_dist)
            distances_median.append(med_dist)
        else:
            min_dist = None
            med_dist = None
        pairwise = pairwise_distances(correct_points)
        spread_mean = mean(pairwise) if pairwise else 0.0 if len(correct_points) == 1 else None
        spread_max = max(pairwise) if pairwise else 0.0 if len(correct_points) == 1 else None
        if spread_max is not None and len(correct_points) >= 2:
            spread_max_values.append(spread_max)

        same_gt_element_any = bool(greedy_in_gt_bbox and any(correct_in_gt_flags))
        different_gt_element = bool((not greedy_in_gt_bbox) and correct_in_gt_frac >= 0.5 and correct_points)
        spatial_applicable = greedy_point is not None and bool(correct_points)
        min_bucket = distance_bucket(min_dist, args.near_px, args.far_px)
        med_bucket = distance_bucket(med_dist, args.near_px, args.far_px)
        if not spatial_applicable:
            structure_label = "missing_spatial_point"
        elif different_gt_element:
            structure_label = "element_selection"
        elif same_gt_element_any:
            structure_label = "same_gt_element_coordinate_or_content"
        elif min_dist is not None and min_dist > args.far_px:
            structure_label = "far_unknown_element"
        else:
            structure_label = "mixed_or_ambiguous"

        wrong_signatures = [action_signature(sample.get("pred_action"), bbox, args.modal_cell_px) for sample in wrong_samples]
        wrong_counter = Counter(wrong_signatures)
        modal_wrong_sig, modal_wrong_count = (None, 0)
        if wrong_counter:
            modal_wrong_sig, modal_wrong_count = wrong_counter.most_common(1)[0]
        modal_wrong_frac = safe_ratio(modal_wrong_count, len(wrong_samples))
        if wrong_samples:
            modal_wrong_fracs.append(modal_wrong_frac)
        greedy_signature = action_signature(greedy_action, bbox, args.modal_cell_px)
        modal_matches_greedy = bool(modal_wrong_sig == greedy_signature)

        p_success = safe_ratio(len(correct_samples), len(samples))
        binary_var = p_success * (1.0 - p_success)
        binary_variances.append(binary_var)
        sigma_rows = {}
        for sigma in sigmas:
            correct_rewards = [gaussian_reward(value, sigma) for value in correct_to_gt]
            wrong_rewards = [gaussian_reward(value, sigma) for value in wrong_to_gt]
            correct_rewards = [value for value in correct_rewards if value is not None]
            wrong_rewards = [value for value in wrong_rewards if value is not None]
            correct_mean = mean(correct_rewards) if correct_rewards else None
            wrong_mean = mean(wrong_rewards) if wrong_rewards else None
            gap = correct_mean - wrong_mean if correct_mean is not None and wrong_mean is not None else None
            sigma_rows[str(sigma)] = {
                "correct_mean": correct_mean,
                "wrong_mean": wrong_mean,
                "gap": gap,
                "n_correct_with_point": len(correct_rewards),
                "n_wrong_with_point": len(wrong_rewards),
            }
            if gap is not None:
                gaussian_acc[sigma].append({"gap": gap, "correct_mean": correct_mean, "wrong_mean": wrong_mean})

        rec = {
            "target_id": row.get("target_id"),
            "episode_id": episode_id,
            "step_idx": step_idx,
            "temperature": float(row.get("temperature")),
            "action_type": action_type(gt_action, str(row.get("action_type") or "unknown")),
            "task_k": int(row.get("task_k") or len(steps)),
            "image_w": image_w,
            "image_h": image_h,
            "gt_bbox": list(bbox) if bbox else None,
            "gt_point": list(target_point) if target_point else None,
            "greedy_point": list(greedy_point) if greedy_point else None,
            "greedy_action": greedy_action,
            "greedy_bucket": row.get("greedy_bucket"),
            "greedy_distance_to_gt_px": greedy_distance_to_gt_px,
            "greedy_in_gt_bbox": greedy_in_gt_bbox,
            "n_samples": len(samples),
            "n_correct": len(correct_samples),
            "n_wrong": len(wrong_samples),
            "n_correct_with_point": len(correct_points),
            "n_wrong_with_point": len(wrong_points),
            "correct_missing_point": correct_missing_point,
            "wrong_missing_point": wrong_missing_point,
            "correct_in_gt_bbox_fraction": correct_in_gt_frac,
            "wrong_in_gt_bbox_fraction": wrong_in_gt_frac,
            "correct_to_greedy_min_px": min_dist,
            "correct_to_greedy_median_px": med_dist,
            "correct_to_greedy_min_norm": normalized_distance_px(min_dist, image_w, image_h),
            "correct_to_greedy_median_norm": normalized_distance_px(med_dist, image_w, image_h),
            "correct_to_greedy_min_bucket": min_bucket,
            "correct_to_greedy_median_bucket": med_bucket,
            "same_gt_element_any": same_gt_element_any,
            "different_gt_element": different_gt_element,
            "spatial_applicable": spatial_applicable,
            "structure_label": structure_label,
            "correct_pairwise_spread_mean_px": spread_mean,
            "correct_pairwise_spread_max_px": spread_max,
            "correct_pairwise_spread_max_bucket": distance_bucket(spread_max, args.near_px, args.far_px),
            "modal_wrong_signature": modal_wrong_sig,
            "modal_wrong_count": modal_wrong_count,
            "modal_wrong_fraction": modal_wrong_frac,
            "greedy_signature": greedy_signature,
            "modal_wrong_matches_greedy": modal_matches_greedy,
            "binary_success_fraction": p_success,
            "binary_reward_variance": binary_var,
            "gaussian_by_sigma": sigma_rows,
        }
        per_step.append(rec)
        aggregate[structure_label] += 1
        aggregate[f"min_bucket:{min_bucket}"] += 1
        aggregate[f"med_bucket:{med_bucket}"] += 1
        aggregate[f"action:{rec['action_type']}"] += 1
        if correct_missing_point:
            aggregate["has_correct_missing_point"] += 1
        if spatial_applicable:
            aggregate["spatial_applicable"] += 1
        if modal_wrong_frac >= 0.50:
            aggregate["modal_wrong_peaked_ge_50"] += 1
        if modal_matches_greedy:
            aggregate["modal_wrong_matches_greedy"] += 1

    with (output_dir / "per_step.jsonl").open("w", encoding="utf-8") as handle:
        for rec in per_step:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_steps = len(per_step)
    spatial_n = aggregate["spatial_applicable"]
    element_selection_n = aggregate["element_selection"] + aggregate["far_unknown_element"]
    same_element_n = aggregate["same_gt_element_coordinate_or_content"]
    missing_spatial_n = aggregate["missing_spatial_point"]
    near_min_n = aggregate[f"min_bucket:near"]
    far_min_n = aggregate[f"min_bucket:far"]
    correct_missing_steps = aggregate["has_correct_missing_point"]

    gaussian_summary: Dict[str, Dict[str, Any]] = {}
    for sigma in sigmas:
        values = gaussian_acc.get(sigma, [])
        gaps = [item["gap"] for item in values]
        gaussian_summary[str(sigma)] = {
            "n_steps_with_correct_and_wrong_points": len(values),
            "gap_mean": mean(gaps) if gaps else None,
            "gap_median": median(gaps) if gaps else None,
            "gap_p10": percentile(gaps, 0.10) if gaps else None,
            "gap_p90": percentile(gaps, 0.90) if gaps else None,
            "fraction_gap_positive": safe_ratio(sum(1 for gap in gaps if gap > 0), len(gaps)),
            "fraction_gap_ge_0_10": safe_ratio(sum(1 for gap in gaps if gap >= 0.10), len(gaps)),
            "fraction_gap_ge_0_25": safe_ratio(sum(1 for gap in gaps if gap >= 0.25), len(gaps)),
        }

    binary_var_summary = summarize_values(binary_variances)
    distance_min_summary = summarize_values(distances_min)
    distance_median_summary = summarize_values(distances_median)
    spread_summary = summarize_values(spread_max_values)
    modal_summary = summarize_values(modal_wrong_fracs)

    primary_sigma = 100.0 if 100.0 in sigmas else sigmas[len(sigmas) // 2]
    primary_gap = gaussian_summary[str(primary_sigma)]["gap_mean"]
    primary_gap_fraction = gaussian_summary[str(primary_sigma)]["fraction_gap_ge_0_10"]
    element_selection_frac = safe_ratio(element_selection_n, spatial_n)
    same_element_frac = safe_ratio(same_element_n, spatial_n)
    near_frac = safe_ratio(near_min_n, spatial_n)
    far_frac = safe_ratio(far_min_n, spatial_n)

    if element_selection_frac >= 0.50 or far_frac >= 0.50:
        gate = "PATH A REQUIRED"
        gate_reason = "Recoverable correct samples are mostly spatially separated from the greedy error / outside the greedy target region; this is an element-selection problem, not coordinate refinement."
    elif same_element_frac >= 0.50 and primary_gap is not None and primary_gap >= 0.10 and primary_gap_fraction >= 0.60:
        gate = "PATH B VIABLE"
        gate_reason = "Most spatially-applicable recoverable steps are same-element/near and Gaussian reward separates correct from wrong samples."
    else:
        gate = "MIXED"
        gate_reason = "The population is not cleanly same-element coordinate error nor cleanly element-selection under the available spatial proxy."

    summary = {
        "input_samples": args.samples,
        "test_data": args.test_data,
        "temperature": float(args.temperature),
        "population": args.population,
        "n_recoverable_steps": n_steps,
        "element_identity_source": "target_bbox_only_no_uia_controls_info_found",
        "counts": dict(aggregate),
        "metric1": {
            "spatial_applicable_steps": spatial_n,
            "missing_spatial_point_steps": missing_spatial_n,
            "same_gt_element_coordinate_or_content_fraction": same_element_frac,
            "element_selection_or_far_unknown_fraction": element_selection_frac,
            "near_min_distance_fraction": near_frac,
            "far_min_distance_fraction": far_frac,
            "correct_to_greedy_min_px": distance_min_summary,
            "correct_to_greedy_median_px": distance_median_summary,
            "correct_missing_point_steps": correct_missing_steps,
        },
        "metric2": {
            "steps_with_multiple_correct_points": len(spread_max_values),
            "correct_pairwise_spread_max_px": spread_summary,
            "tight_correct_cluster_le_50px_fraction": safe_ratio(sum(1 for value in spread_max_values if value <= args.near_px), len(spread_max_values)),
            "diffuse_correct_cluster_gt_150px_fraction": safe_ratio(sum(1 for value in spread_max_values if value > args.far_px), len(spread_max_values)),
            "correct_on_gt_bbox_mean_fraction": mean([rec["correct_in_gt_bbox_fraction"] for rec in per_step if rec["n_correct_with_point"]]) if any(rec["n_correct_with_point"] for rec in per_step) else None,
        },
        "metric3": {
            "modal_wrong_fraction": modal_summary,
            "modal_wrong_peaked_ge_50_fraction": safe_ratio(aggregate["modal_wrong_peaked_ge_50"], n_steps),
            "modal_wrong_matches_greedy_fraction": safe_ratio(aggregate["modal_wrong_matches_greedy"], n_steps),
        },
        "metric4": {
            "sigmas_px": sigmas,
            "gaussian_summary": gaussian_summary,
            "binary_reward_variance": binary_var_summary,
            "primary_sigma_px": primary_sigma,
        },
        "gate": gate,
        "gate_reason": gate_reason,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Spatial Structure of Recoverable Critical-Step Answers")
    lines.append("")
    lines.append("Diagnostic only: existing C.1 samples + frozen matcher. No training was performed.")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- samples: `{args.samples}`")
    lines.append(f"- test data: `{args.test_data}`")
    lines.append(f"- population: `{args.population}`")
    lines.append(f"- temperature: `{args.temperature}`")
    lines.append(f"- recoverable critical steps analyzed: `{n_steps}`")
    lines.append("- element identity source: `target bbox only`; no `uia_controls_info` was present in the reconstructed test JSONL. Same/different element is therefore measured as target-bbox membership, not full a11y-control identity.")
    lines.append("")
    lines.append("## Metric 1: Correct-vs-Greedy Spatial Relationship")
    lines.append("")
    lines.append("| measure | value |")
    lines.append("|---|---:|")
    lines.append(f"| spatial-applicable steps | {spatial_n} / {n_steps} ({format_pct(safe_ratio(spatial_n, n_steps))}) |")
    lines.append(f"| missing correct/greedy point | {missing_spatial_n} / {n_steps} ({format_pct(safe_ratio(missing_spatial_n, n_steps))}) |")
    lines.append(f"| same GT bbox / coordinate-or-content-like | {same_element_n} / {spatial_n} ({format_pct(same_element_frac)}) |")
    lines.append(f"| different/far element-selection-like | {element_selection_n} / {spatial_n} ({format_pct(element_selection_frac)}) |")
    lines.append(f"| nearest correct to greedy: near <= {args.near_px:.0f}px | {near_min_n} / {spatial_n} ({format_pct(near_frac)}) |")
    lines.append(f"| nearest correct to greedy: far > {args.far_px:.0f}px | {far_min_n} / {spatial_n} ({format_pct(far_frac)}) |")
    lines.append("")
    lines.append("Correct-to-greedy nearest distance in pixels:")
    lines.append("")
    lines.append(f"- mean: `{format_optional(distance_min_summary['mean'], 2)}`")
    lines.append(f"- median: `{format_optional(distance_min_summary['median'], 2)}`")
    lines.append(f"- p10/p90: `{format_optional(distance_min_summary['p10'], 2)}` / `{format_optional(distance_min_summary['p90'], 2)}`")
    lines.append("")
    lines.append("## Metric 2: Correct-Answer Concentration")
    lines.append("")
    lines.append(f"Steps with at least two correct samples carrying spatial points: `{len(spread_max_values)}`.")
    lines.append("")
    lines.append("| spread measure | value |")
    lines.append("|---|---:|")
    lines.append(f"| max pairwise spread mean px | {format_optional(spread_summary['mean'], 2)} |")
    lines.append(f"| max pairwise spread median px | {format_optional(spread_summary['median'], 2)} |")
    lines.append(f"| tight cluster <= {args.near_px:.0f}px | {format_pct(summary['metric2']['tight_correct_cluster_le_50px_fraction'])} |")
    lines.append(f"| diffuse cluster > {args.far_px:.0f}px | {format_pct(summary['metric2']['diffuse_correct_cluster_gt_150px_fraction'])} |")
    lines.append(f"| mean fraction of correct points inside GT bbox | {format_pct(summary['metric2']['correct_on_gt_bbox_mean_fraction'] or 0.0)} |")
    lines.append("")
    lines.append("## Metric 3: Modal Wrong Distractor Peakedness")
    lines.append("")
    lines.append(f"- mean modal-wrong frequency: `{format_optional(modal_summary['mean'], 3)}`")
    lines.append(f"- median modal-wrong frequency: `{format_optional(modal_summary['median'], 3)}`")
    lines.append(f"- fraction with modal wrong >= 50% of wrong samples: `{format_pct(summary['metric3']['modal_wrong_peaked_ge_50_fraction'])}`")
    lines.append(f"- fraction where modal wrong signature matches greedy signature: `{format_pct(summary['metric3']['modal_wrong_matches_greedy_fraction'])}`")
    lines.append("")
    lines.append("## Metric 4: Gaussian Distance-Reward Gradient Simulation")
    lines.append("")
    lines.append("Gaussian reward is `exp(-d^2 / 2 sigma^2)`, where `d` is distance to the GT target point. Rows require both correct and wrong samples with spatial points.")
    lines.append("")
    lines.append("| sigma px | steps | mean gap correct-wrong | median gap | gap > 0 | gap >= 0.10 | gap >= 0.25 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for sigma in sigmas:
        item = gaussian_summary[str(sigma)]
        lines.append(
            f"| {sigma:.0f} | {item['n_steps_with_correct_and_wrong_points']} | "
            f"{format_optional(item['gap_mean'], 4)} | {format_optional(item['gap_median'], 4)} | "
            f"{format_pct(item['fraction_gap_positive'])} | {format_pct(item['fraction_gap_ge_0_10'])} | {format_pct(item['fraction_gap_ge_0_25'])} |"
        )
    lines.append("")
    lines.append("Binary success-reward variance inside the N=50 pools:")
    lines.append("")
    lines.append(f"- mean: `{format_optional(binary_var_summary['mean'], 4)}`")
    lines.append(f"- median: `{format_optional(binary_var_summary['median'], 4)}`")
    lines.append(f"- p10/p90: `{format_optional(binary_var_summary['p10'], 4)}` / `{format_optional(binary_var_summary['p90'], 4)}`")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(f"**{gate}**")
    lines.append("")
    lines.append(gate_reason)
    lines.append("")
    if gate == "PATH A REQUIRED":
        lines.append("Training direction: use verifier / generative verifier selection over discrete candidates. A Gaussian distance reward to a wrong greedy element is not the right objective for most recoverable critical steps.")
    elif gate == "PATH B VIABLE":
        lines.append("Training direction: Gaussian-reward RLVR is plausible for this slice because correct answers are near/same-element and receive a usable distance gradient.")
    else:
        lines.append("Training direction: split the population; use Gaussian reward only for same-element coordinate-like steps and verifier selection for element-selection steps.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{output_dir / 'spatial.md'}`")
    lines.append(f"- `{output_dir / 'summary.json'}`")
    lines.append(f"- `{output_dir / 'per_step.jsonl'}`")
    lines.append("")
    lines.append("STOP for review.")
    (output_dir / "spatial.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output_dir / "spatial.md"), "per_step": str(output_dir / "per_step.jsonl"), "summary": str(output_dir / "summary.json"), "gate": gate, "n": n_steps}, indent=2), flush=True)


if __name__ == "__main__":
    main()