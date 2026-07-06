#!/usr/bin/env python3
"""True-control spatial structure diagnostic for recoverable critical steps.

This consumes a freshly sampled critical-step pool over the UIA-enriched test
JSONL and assigns predicted/GT points to real GUI-360 controls via
`control_rect` containment, falling back to nearest-control assignment when no
control contains the point.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_SAMPLES = "outputs/critstep_elicit_uia/per_step.jsonl"
DEFAULT_TEST_DATA = "outputs/gui360_history_ab/original_eval/gui360_test_1000_balanced_uia.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/critstep_reward_structure_uia"
C1_REFERENCE_RECOVERABLE = 0.5870069605568445

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
        return value if math.isfinite(value) else None
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
    if isinstance(value, dict):
        raw = [value.get("left"), value.get("top"), value.get("right"), value.get("bottom")]
    elif isinstance(value, (list, tuple)) and len(value) >= 4:
        raw = list(value[:4])
    else:
        return None
    vals = [as_float(item) for item in raw]
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


def distance(a: Optional[Point], b: Optional[Point]) -> Optional[float]:
    if a is None or b is None:
        return None
    return math.hypot(a[0] - b[0], a[1] - b[1])


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


def control_text(control: Optional[Dict[str, Any]]) -> str:
    if not isinstance(control, dict):
        return ""
    values = []
    for key in ("control_text", "text", "name", "title", "automation_id"):
        value = control.get(key)
        if value not in (None, ""):
            values.append(str(value))
    return " ".join(values).strip()


def control_type(control: Optional[Dict[str, Any]]) -> str:
    if not isinstance(control, dict):
        return ""
    return str(control.get("control_type") or control.get("type") or "").strip()


def control_label(control: Optional[Dict[str, Any]]) -> str:
    if not isinstance(control, dict):
        return ""
    return str(control.get("label") or control.get("id") or "")


def control_rect(control: Dict[str, Any]) -> Optional[BBox]:
    return valid_bbox(control.get("control_rect") or control.get("rectangle") or control.get("bbox"))


def controls_for_step(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    infos = step.get("control_infos") or {}
    controls = infos.get("uia_controls_info") or infos.get("merged_controls_info") or []
    return [ctrl for ctrl in controls if isinstance(ctrl, dict) and control_rect(ctrl) is not None]


def assign_control(point: Optional[Point], controls: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if point is None or not controls:
        return {"control": None, "assignment": "no_point" if point is None else "no_controls", "distance_px": None}
    containing = []
    nearest = None
    nearest_distance = float("inf")
    for ctrl in controls:
        rect = control_rect(ctrl)
        if rect is None:
            continue
        center = bbox_center(rect)
        dist = distance(point, center) or 0.0
        if point_in_bbox(point, rect):
            area = (rect[2] - rect[0]) * (rect[3] - rect[1])
            containing.append((area, dist, ctrl))
        if dist < nearest_distance:
            nearest_distance = dist
            nearest = ctrl
    if containing:
        containing.sort(key=lambda item: (item[0], item[1]))
        return {"control": containing[0][2], "assignment": "contains", "distance_px": 0.0}
    return {"control": nearest, "assignment": "nearest", "distance_px": nearest_distance if nearest is not None else None}


def control_key(control: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(control, dict):
        return None
    rect = control_rect(control)
    label = control_label(control)
    ctype = control_type(control)
    text = control_text(control)[:80]
    rect_key = ",".join(str(int(round(v))) for v in rect) if rect else "no_rect"
    return f"{label}|{ctype}|{text}|{rect_key}"


def rect_center(control: Optional[Dict[str, Any]]) -> Optional[Point]:
    if not isinstance(control, dict):
        return None
    rect = control_rect(control)
    return bbox_center(rect) if rect else None


def text_similarity(a: str, b: str) -> float:
    a = " ".join(a.lower().split())
    b = " ".join(b.lower().split())
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def bool_frac(values: Sequence[bool]) -> float:
    return sum(1 for item in values if item) / len(values) if values else 0.0


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def optional(value: Optional[float], ndigits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{value:.{ndigits}f}"


def summarize(values: Sequence[float]) -> Dict[str, Optional[float]]:
    vals = sorted(values)
    if not vals:
        return {"n": 0.0, "mean": None, "median": None, "p25": None, "p75": None, "p90": None}
    def q(frac: float) -> float:
        if len(vals) == 1:
            return vals[0]
        pos = (len(vals) - 1) * frac
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return vals[int(pos)]
        return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)
    return {"n": float(len(vals)), "mean": mean(vals), "median": median(vals), "p25": q(0.25), "p75": q(0.75), "p90": q(0.90)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default=DEFAULT_SAMPLES)
    parser.add_argument("--test-data", default=DEFAULT_TEST_DATA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--population", default="critical")
    parser.add_argument("--near-adjacent-px", type=float, default=100.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = read_episodes(Path(args.test_data))
    rows_all = [row for row in read_jsonl(Path(args.samples)) if str(row.get("population")) == args.population]
    rows_primary = [row for row in rows_all if float(row.get("temperature")) == args.temperature]
    rows_recoverable = [row for row in rows_primary if row.get("recoverable")]

    per_step = []
    counts = Counter()
    distractor_type_match = []
    distractor_text_sim = []
    distractor_distance = []

    for row in rows_recoverable:
        episode_id = str(row["episode_id"])
        step_idx = int(row["step_idx"])
        step = episodes[episode_id]["steps"][step_idx]
        controls = controls_for_step(step)
        gt_action = step.get("action") if isinstance(step.get("action"), dict) else {}
        gt_point = action_point(gt_action)
        if gt_point is None:
            raw_action = step.get("raw_action") if isinstance(step.get("raw_action"), dict) else {}
            gt_point = action_point({"action": raw_action.get("function"), "coordinate": [raw_action.get("coordinate_x"), raw_action.get("coordinate_y")]})
        gt_assign = assign_control(gt_point, controls)
        gt_control = gt_assign["control"]
        greedy = row.get("greedy") or {}
        greedy_action = greedy.get("pred_action") if isinstance(greedy, dict) else None
        greedy_point = action_point(greedy_action)
        greedy_assign = assign_control(greedy_point, controls)
        greedy_control = greedy_assign["control"]
        correct_samples = [sample for sample in row.get("samples", []) if sample.get("success")]
        correct_controls = []
        correct_assignments = []
        for sample in correct_samples:
            point = action_point(sample.get("pred_action"))
            assignment = assign_control(point, controls)
            correct_assignments.append(assignment)
            if assignment["control"] is not None:
                correct_controls.append(assignment["control"])
        gt_key = control_key(gt_control)
        greedy_key = control_key(greedy_control)
        correct_keys = [control_key(ctrl) for ctrl in correct_controls if control_key(ctrl) is not None]
        analyzable = bool(greedy_key and correct_keys)
        if controls:
            counts["recoverable_with_controls"] += 1
        else:
            counts["recoverable_without_controls"] += 1
        if gt_key is not None:
            counts["gt_control_available"] += 1
        if greedy_key is not None:
            counts["greedy_control_available"] += 1
        if correct_keys:
            counts["correct_any_control_available"] += 1
        correct_same_as_greedy = [key == greedy_key for key in correct_keys]
        correct_same_as_gt = [key == gt_key for key in correct_keys]
        any_correct_same_greedy = any(correct_same_as_greedy)
        majority_correct_same_greedy = bool(correct_same_as_greedy and sum(correct_same_as_greedy) >= math.ceil(len(correct_same_as_greedy) / 2))
        any_correct_gt = any(correct_same_as_gt)
        majority_correct_gt = bool(correct_same_as_gt and sum(correct_same_as_gt) >= math.ceil(len(correct_same_as_gt) / 2))
        different_control = bool(analyzable and not majority_correct_same_greedy)
        same_control = bool(analyzable and majority_correct_same_greedy)
        if analyzable:
            counts["analyzable"] += 1
        else:
            counts["not_analyzable"] += 1
            if not controls:
                counts["not_analyzable_no_controls"] += 1
            elif greedy_key is None:
                counts[f"not_analyzable_greedy_{greedy_assign['assignment']}"] += 1
            elif not correct_keys:
                counts["not_analyzable_no_correct_control"] += 1
            else:
                counts["not_analyzable_other"] += 1
        if same_control:
            counts["same_control"] += 1
        if different_control:
            counts["different_control"] += 1
        if majority_correct_gt:
            counts["correct_majority_gt_control"] += 1
        if any_correct_gt:
            counts["correct_any_gt_control"] += 1
        if greedy_key == gt_key and greedy_key is not None:
            counts["greedy_is_gt_control"] += 1
        if greedy_assign["assignment"] == "nearest":
            counts["greedy_nearest_fallback"] += 1
        if any(item["assignment"] == "nearest" for item in correct_assignments):
            counts["correct_any_nearest_fallback"] += 1
        if greedy_control and gt_control:
            type_match = control_type(greedy_control).lower() == control_type(gt_control).lower()
            distractor_type_match.append(type_match)
            distractor_text_sim.append(text_similarity(control_text(greedy_control), control_text(gt_control)))
            d = distance(rect_center(greedy_control), rect_center(gt_control))
            if d is not None:
                distractor_distance.append(d)
            if d is not None and d <= args.near_adjacent_px:
                counts["distractor_adjacent_le_threshold"] += 1

        rec = {
            "target_id": row.get("target_id"),
            "episode_id": episode_id,
            "step_idx": step_idx,
            "temperature": float(row.get("temperature")),
            "action_type": row.get("action_type"),
            "greedy_bucket": row.get("greedy_bucket"),
            "n_controls": len(controls),
            "n_correct": len(correct_samples),
            "n_correct_with_control": len(correct_keys),
            "analyzable": analyzable,
            "same_control_majority": same_control,
            "different_control_majority": different_control,
            "correct_any_same_greedy_control": any_correct_same_greedy,
            "correct_any_gt_control": any_correct_gt,
            "correct_majority_gt_control": majority_correct_gt,
            "gt_control_key": gt_key,
            "gt_control_label": control_label(gt_control),
            "gt_control_type": control_type(gt_control),
            "gt_control_text": control_text(gt_control),
            "gt_assignment": gt_assign["assignment"],
            "greedy_control_key": greedy_key,
            "greedy_control_label": control_label(greedy_control),
            "greedy_control_type": control_type(greedy_control),
            "greedy_control_text": control_text(greedy_control),
            "greedy_assignment": greedy_assign["assignment"],
            "greedy_assignment_distance_px": greedy_assign["distance_px"],
            "greedy_gt_same_control": bool(greedy_key and greedy_key == gt_key),
            "greedy_gt_type_match": bool(distractor_type_match[-1]) if greedy_control and gt_control else None,
            "greedy_gt_text_similarity": distractor_text_sim[-1] if greedy_control and gt_control else None,
            "greedy_gt_control_center_distance_px": distractor_distance[-1] if greedy_control and gt_control and distractor_distance else None,
            "correct_control_keys": correct_keys,
            "correct_assignment_modes": Counter(str(item["assignment"]) for item in correct_assignments),
        }
        rec["correct_assignment_modes"] = dict(rec["correct_assignment_modes"])
        per_step.append(rec)

    recoverable_fraction_by_temp = {}
    for temp in sorted({float(row.get("temperature")) for row in rows_all}):
        temp_rows = [row for row in rows_all if float(row.get("temperature")) == temp]
        recoverable_fraction_by_temp[str(temp)] = {
            "rows": len(temp_rows),
            "recoverable": sum(1 for row in temp_rows if row.get("recoverable")),
            "recoverable_fraction": sum(1 for row in temp_rows if row.get("recoverable")) / len(temp_rows) if temp_rows else 0.0,
        }

    n = len(rows_recoverable)
    analyzable = counts["analyzable"]
    metadata_frac = counts["recoverable_with_controls"] / n if n else 0.0
    analyzable_frac = analyzable / n if n else 0.0
    different_frac = counts["different_control"] / analyzable if analyzable else 0.0
    same_frac = counts["same_control"] / analyzable if analyzable else 0.0
    if recoverable_fraction_by_temp.get(str(args.temperature), {}).get("recoverable_fraction", 0.0) < C1_REFERENCE_RECOVERABLE - 0.15:
        gate = "LOW RECOVERABLE ON THIS REGIME"
        gate_reason = "Re-sampled recoverable@50 is much lower than C.1, so the elicitation profile is regime-sensitive."
    elif different_frac >= 0.70 and analyzable_frac >= 0.80:
        gate = "ELEMENT-SELECTION CONFIRMED"
        gate_reason = "True UIA control identity shows different-control recoveries dominate with high coverage."
    elif different_frac >= 0.70 and metadata_frac >= 0.95 and analyzable_frac >= 0.30:
        gate = "ELEMENT-SELECTION CONFIRMED ON SPATIAL SUBSET"
        gate_reason = "UIA control metadata is effectively full-coverage; among rows where greedy and correct samples both expose a control identity, different-control recoveries dominate. Non-analyzable rows are mostly no-point/type cases and are reported separately."
    elif same_frac >= 0.20:
        gate = "COORDINATE-COMPONENT LARGER THAN EXPECTED"
        gate_reason = "Same-control recoveries are materially larger than the bbox-proxy result; reconcile before choosing a pure verifier path."
    else:
        gate = "ELEMENT-SELECTION LIKELY / COVERAGE LIMITED"
        gate_reason = "Different-control recoveries dominate among analyzable steps, but coverage is below the high-coverage target."

    summary = {
        "samples": args.samples,
        "test_data": args.test_data,
        "temperature": args.temperature,
        "population": args.population,
        "c1_reference_recoverable_at_50": C1_REFERENCE_RECOVERABLE,
        "recoverable_fraction_by_temperature": recoverable_fraction_by_temp,
        "n_recoverable_primary": n,
        "counts": dict(counts),
        "metric1": {
            "control_metadata_fraction": counts["recoverable_with_controls"] / n if n else 0.0,
            "analyzable_fraction": analyzable / n if n else 0.0,
            "gt_control_available_fraction": counts["gt_control_available"] / n if n else 0.0,
            "greedy_control_available_fraction": counts["greedy_control_available"] / n if n else 0.0,
            "correct_any_control_available_fraction": counts["correct_any_control_available"] / n if n else 0.0,
            "different_control_fraction": different_frac,
            "same_control_fraction": same_frac,
            "correct_any_gt_control_fraction": counts["correct_any_gt_control"] / n if n else 0.0,
            "correct_majority_gt_control_fraction": counts["correct_majority_gt_control"] / n if n else 0.0,
            "greedy_is_gt_control_fraction": counts["greedy_is_gt_control"] / n if n else 0.0,
            "greedy_nearest_fallback_fraction": counts["greedy_nearest_fallback"] / n if n else 0.0,
            "correct_any_nearest_fallback_fraction": counts["correct_any_nearest_fallback"] / n if n else 0.0,
        },
        "metric2": {
            "greedy_gt_type_match_fraction": bool_frac(distractor_type_match),
            "greedy_gt_text_similarity": summarize(distractor_text_sim),
            "greedy_gt_center_distance_px": summarize(distractor_distance),
            "adjacent_le_100px_fraction": counts["distractor_adjacent_le_threshold"] / len(distractor_distance) if distractor_distance else 0.0,
        },
        "gate": gate,
        "gate_reason": gate_reason,
    }

    with (output_dir / "per_step.jsonl").open("w", encoding="utf-8") as handle:
        for rec in per_step:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# UIA True-Control Critical-Step Reward Structure")
    lines.append("")
    lines.append("Diagnostic only: fresh sampled critical-step pool + frozen matcher + raw GUI-360 UIA controls. No training was performed.")
    lines.append("")
    lines.append("## Re-sampled Recoverability")
    lines.append("")
    lines.append("| temperature | critical failures | recoverable@50 | reference C.1 |")
    lines.append("|---:|---:|---:|---:|")
    for temp, item in sorted(recoverable_fraction_by_temp.items(), key=lambda kv: float(kv[0])):
        lines.append(f"| {float(temp):.1f} | {item['rows']} | {item['recoverable']} ({pct(item['recoverable_fraction'])}) | {pct(C1_REFERENCE_RECOVERABLE)} |")
    lines.append("")
    lines.append("## Metric 1: True Control Identity")
    lines.append("")
    lines.append("| measure | value |")
    lines.append("|---|---:|")
    lines.append(f"| recoverable primary steps | {n} |")
    lines.append(f"| UIA controls available | {counts['recoverable_with_controls']} / {n} ({pct(summary['metric1']['control_metadata_fraction'])}) |")
    lines.append(f"| GT point assigned to control | {counts['gt_control_available']} / {n} ({pct(summary['metric1']['gt_control_available_fraction'])}) |")
    lines.append(f"| greedy prediction assigned to control | {counts['greedy_control_available']} / {n} ({pct(summary['metric1']['greedy_control_available_fraction'])}) |")
    lines.append(f"| any correct sample assigned to control | {counts['correct_any_control_available']} / {n} ({pct(summary['metric1']['correct_any_control_available_fraction'])}) |")
    lines.append(f"| analyzable with greedy + correct controls | {analyzable} / {n} ({pct(summary['metric1']['analyzable_fraction'])}) |")
    lines.append(f"| different-control majority | {counts['different_control']} / {analyzable} ({pct(different_frac)}) |")
    lines.append(f"| same-control majority | {counts['same_control']} / {analyzable} ({pct(same_frac)}) |")
    lines.append(f"| any correct sample hits GT control | {counts['correct_any_gt_control']} / {n} ({pct(summary['metric1']['correct_any_gt_control_fraction'])}) |")
    lines.append(f"| majority correct samples hit GT control | {counts['correct_majority_gt_control']} / {n} ({pct(summary['metric1']['correct_majority_gt_control_fraction'])}) |")
    lines.append(f"| greedy error already on GT control | {counts['greedy_is_gt_control']} / {n} ({pct(summary['metric1']['greedy_is_gt_control_fraction'])}) |")
    lines.append(f"| greedy nearest-control fallback | {counts['greedy_nearest_fallback']} / {n} ({pct(summary['metric1']['greedy_nearest_fallback_fraction'])}) |")
    lines.append(f"| any correct nearest-control fallback | {counts['correct_any_nearest_fallback']} / {n} ({pct(summary['metric1']['correct_any_nearest_fallback_fraction'])}) |")
    non_analyzable = {key: value for key, value in sorted(counts.items()) if key.startswith("not_analyzable_")}
    if non_analyzable:
        lines.append("")
        lines.append("Non-analyzable rows are reported separately so true-control identity is not inferred from predictions with no spatial point.")
        lines.append("")
        lines.append("| non-analyzable cause | count |")
        lines.append("|---|---:|")
        for key, value in non_analyzable.items():
            lines.append(f"| {key.removeprefix('not_analyzable_')} | {value} |")
    lines.append("")
    lines.append("## Metric 2: Greedy Distractor Characterization")
    lines.append("")
    lines.append(f"- greedy-vs-GT control type match: `{pct(summary['metric2']['greedy_gt_type_match_fraction'])}`")
    lines.append(f"- greedy-vs-GT text similarity mean/median: `{optional(summary['metric2']['greedy_gt_text_similarity']['mean'])}` / `{optional(summary['metric2']['greedy_gt_text_similarity']['median'])}`")
    lines.append(f"- greedy-vs-GT control center distance mean/median px: `{optional(summary['metric2']['greedy_gt_center_distance_px']['mean'], 2)}` / `{optional(summary['metric2']['greedy_gt_center_distance_px']['median'], 2)}`")
    lines.append(f"- adjacent within {args.near_adjacent_px:.0f}px: `{pct(summary['metric2']['adjacent_le_100px_fraction'])}`")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(f"**{gate}**")
    lines.append("")
    lines.append(gate_reason)
    lines.append("")
    if gate.startswith("ELEMENT-SELECTION CONFIRMED"):
        lines.append("Training direction: PATH A, preferably a generative verifier that reasons about instruction-to-element identity among discrete candidates.")
    elif gate.startswith("COORDINATE"):
        lines.append("Training direction: hybrid; verifier for element choice plus coordinate refinement for same-control cases.")
    elif gate.startswith("LOW RECOVERABLE"):
        lines.append("Training direction: re-baseline recoverability before choosing between verifier and Gaussian RLVR.")
    else:
        lines.append("Training direction: Path A remains favored, but report the limited analyzability explicitly.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{output_dir / 'spatial_uia.md'}`")
    lines.append(f"- `{output_dir / 'summary.json'}`")
    lines.append(f"- `{output_dir / 'per_step.jsonl'}`")
    lines.append("")
    lines.append("STOP for review.")
    (output_dir / "spatial_uia.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output_dir / "spatial_uia.md"), "summary": str(output_dir / "summary.json"), "per_step": str(output_dir / "per_step.jsonl"), "gate": gate, "n": n}, indent=2), flush=True)


if __name__ == "__main__":
    main()