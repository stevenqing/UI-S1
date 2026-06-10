#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
for path in [ROOT, WORKSPACE_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.error_sets import assert_error_set_version  # noqa: E402


RUN_SCOPE = "half_run"
SAMPLE_NAME = "har_gui_odyssey_latest"
EXPECTED_ERROR_SET = "E_v1"
BUCKETS = ("B1", "B2", "B3", "B4a", "B4b")


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "manifest.json"
    manifest = load_json(manifest_path)
    validate_manifest(manifest, args.error_set_version)

    error_set_path = resolve(manifest["error_set"]["canonical_error_set"])
    error_rows = load_jsonl(error_set_path)
    assert_error_set_version(
        error_rows,
        expected_error_set_version=args.error_set_version,
        expected_run_scope=RUN_SCOPE,
        expected_sample_name=SAMPLE_NAME,
    )

    step_rows_path = run_dir / "har_gui_odyssey_steps.jsonl"
    step_rows = load_jsonl(step_rows_path) if step_rows_path.exists() else []
    if step_rows:
        assert_error_set_version(
            step_rows,
            expected_error_set_version=args.error_set_version,
            expected_run_scope=RUN_SCOPE,
            expected_sample_name=SAMPLE_NAME,
        )

    benchmark_rows = load_benchmark_step_rows(resolve(args.benchmark_jsonl)) if args.benchmark_jsonl else {}

    assignments = build_schema_assignments(error_rows)
    sampled_error_rows, sampling_summary = sample_error_rows(error_rows, args.sample_cap, args.seed)
    teacher_inputs = build_teacher_probe_inputs(sampled_error_rows, step_rows, benchmark_rows, sampling_summary["weights_by_key"])
    teacher_result_summary = summarize_teacher_results(resolve_optional(args.teacher_results), sampled_error_rows)

    summary = build_summary(
        manifest=manifest,
        error_rows=error_rows,
        step_rows=step_rows,
        assignments=assignments,
        teacher_inputs=teacher_inputs,
        sampling_summary=sampling_summary,
        teacher_result_summary=teacher_result_summary,
        output_dir=output_dir,
        error_set_path=error_set_path,
        error_set_version=args.error_set_version,
    )

    assignment_path = output_dir / "phase_s_schema_assignment.jsonl"
    teacher_input_path = output_dir / "phase_p_teacher_probe_inputs.jsonl"
    summary_path = output_dir / "phase_sp_summary.json"
    write_jsonl(assignment_path, assignments)
    write_jsonl(teacher_input_path, teacher_inputs)
    summary["files"].update(
        {
            "phase_s_schema_assignment": workspace_relative(assignment_path),
            "phase_p_teacher_probe_inputs": workspace_relative(teacher_input_path),
            "phase_sp_summary": workspace_relative(summary_path),
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    update_manifest(manifest_path, summary)
    update_report(output_dir / "REPORT_V2.md", summary)

    print(
        json.dumps(
            {
                "phase_s_status": summary["phase_s_status"],
                "gate_p_status": summary["gate_p_status"],
                "errors": summary["error_steps"],
                "schema_other_percent": summary["schema_coverage"]["other_percent"],
                "teacher_probe_inputs": len(teacher_inputs),
                "teacher_results_status": teacher_result_summary["status"],
                "report": summary["files"]["report"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline Phase S/P scaffold for CHORUS N0/N1")
    parser.add_argument("--run_dir", default="chorus-n0n1/runs/n0n1_inputs/har_gui_odyssey_latest")
    parser.add_argument("--output_dir", default="chorus-n0n1/runs/n0n1_inputs/har_gui_odyssey_latest/offline_analysis")
    parser.add_argument("--error_set_version", default=EXPECTED_ERROR_SET)
    parser.add_argument("--benchmark_jsonl", default="datasets/GUI-Odyssey/gui_odyssey_random_split_test.jsonl")
    parser.add_argument("--teacher_results", default="")
    parser.add_argument("--sample_cap", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260610)
    return parser.parse_args()


def resolve(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def resolve_optional(path_text: str) -> Optional[Path]:
    return resolve(path_text) if path_text else None


def workspace_relative(path: Path) -> str:
    try:
        return str(path.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL row in {path} at line {line_number}: {exc}") from exc
    return rows


def load_benchmark_step_rows(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    rows: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                episode = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid benchmark JSONL row in {path} at line {line_number}: {exc}") from exc
            episode_id = str(episode.get("episode_id", ""))
            for step_idx, step in enumerate(episode.get("steps", [])):
                screenshot = step.get("screenshot") or step.get("image") or ""
                rows[(episode_id, step_idx)] = {
                    "episode_id": episode_id,
                    "step_idx": step_idx,
                    "screenshot": screenshot,
                    "device_name": episode.get("device_name", ""),
                    "source_width": episode.get("width"),
                    "source_height": episode.get("height"),
                }
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def validate_manifest(manifest: Dict[str, Any], error_set_version: str) -> None:
    if manifest.get("gate_g_status") != "APPROVED":
        raise ValueError(f"GATE G is not approved in manifest: {manifest.get('gate_g_status')!r}")
    if manifest.get("gate_h_status") != "APPROVED":
        raise ValueError(f"GATE H is not approved in manifest: {manifest.get('gate_h_status')!r}")
    error_set = manifest.get("error_set") or {}
    if error_set.get("version") != error_set_version:
        raise ValueError(f"Manifest error set mismatch: expected {error_set_version!r}, got {error_set.get('version')!r}")
    if error_set.get("run_scope") != RUN_SCOPE or error_set.get("sample_name") != SAMPLE_NAME:
        raise ValueError("Manifest error set run_scope/sample_name does not match Phase S/P defaults")


def action_type(row: Dict[str, Any], key: str) -> str:
    value = row.get(key) or {}
    return str(value.get("action", ""))


def build_schema_assignments(error_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    assignments: List[Dict[str, Any]] = []
    for row in error_rows:
        schema_id, schema_name = classify_error_schema(row)
        assignments.append(
            {
                "benchmark": row.get("benchmark"),
                "episode_id": row.get("episode_id"),
                "step_idx": row.get("step_idx"),
                "category": row.get("category"),
                "schema_id": schema_id,
                "schema_name": schema_name,
                "gt_action_type": row.get("gt_action_type") or action_type(row, "gt_action"),
                "pred_action_type": row.get("pred_action_type") or action_type(row, "pred_action"),
                "type_match": row.get("type_match"),
                "baseline_error": bool(row.get("baseline_error")),
                "error_set_version": row.get("error_set_version"),
                "run_scope": row.get("run_scope"),
                "sample_name": row.get("sample_name"),
            }
        )
    return assignments


def classify_error_schema(row: Dict[str, Any]) -> Tuple[str, str]:
    gt_type = row.get("gt_action_type") or action_type(row, "gt_action")
    pred_type = row.get("pred_action_type") or action_type(row, "pred_action")
    if gt_type == "long_press":
        return "P6", "long_press_behavior_gap"
    if gt_type == "terminate":
        if pred_type == "terminate":
            return "P4", "terminate_wrong_status"
        return "P4", "terminate_missed_stop"
    if pred_type == "terminate" and gt_type != "terminate":
        return "P5", "false_stop_outside_terminate_gt"
    if gt_type == "swipe" or pred_type == "swipe":
        return "P7", "swipe_gesture_or_navigation"
    if row.get("type_match") is False:
        return "P1", "action_type_mismatch"
    if gt_type in {"click", "long_press"} or pred_type in {"click", "long_press"}:
        return "P2", "spatial_target_mismatch"
    if gt_type == "type" or pred_type == "type":
        return "P3", "text_entry_or_semantic_mismatch"
    if gt_type == "system_button" or pred_type == "system_button":
        return "P8", "system_navigation_or_sequence_residual"
    return "OTHER", "other"


def build_teacher_probe_inputs(
    error_rows: List[Dict[str, Any]],
    step_rows: List[Dict[str, Any]],
    benchmark_rows: Dict[Tuple[str, int], Dict[str, Any]],
    weights_by_key: Dict[str, float],
) -> List[Dict[str, Any]]:
    rows_by_episode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in step_rows:
        rows_by_episode[str(row.get("episode_id", ""))].append(row)
    for rows in rows_by_episode.values():
        rows.sort(key=lambda item: int(item.get("step_idx", 0)))

    items: List[Dict[str, Any]] = []
    for row in error_rows:
        episode_rows = rows_by_episode.get(str(row.get("episode_id", "")), [])
        previous_rows = [item for item in episode_rows if int(item.get("step_idx", 0)) < int(row.get("step_idx", 0))]
        action_history = [compact_action_history_item(item) for item in previous_rows]
        current_screenshot = current_screenshot_value(row, benchmark_rows)
        observation_history = [
            {"step_idx": item.get("step_idx"), "screenshot": current_screenshot_value(item, benchmark_rows)}
            for item in previous_rows
        ]
        has_screen = bool(current_screenshot)
        has_action_history = int(row.get("step_idx", 0)) == 0 or all(bool(item.get("answer")) for item in previous_rows)
        has_full_history = has_screen and all(bool(item.get("screenshot")) for item in observation_history)
        missing = []
        if not has_screen:
            missing.append("current_screenshot")
        if not has_action_history:
            missing.append("action_history")
        if not has_full_history:
            missing.append("full_observation_history")
        items.append(
            {
                "benchmark": row.get("benchmark"),
                "episode_id": row.get("episode_id"),
                "step_idx": row.get("step_idx"),
                "category": row.get("category"),
                "teacher_probe_status": "queued" if not missing else "blocked_missing_payload",
                "missing_payload": missing,
                "required_probes": ["T_screen", "T_act", "T_full"],
                "teacher_inputs": {
                    "T_screen": {"goal": row.get("goal", ""), "current_screenshot": current_screenshot},
                    "T_act": {
                        "goal": row.get("goal", ""),
                        "current_screenshot": current_screenshot,
                        "action_history": action_history,
                    },
                    "T_full": {
                        "goal": row.get("goal", ""),
                        "current_screenshot": current_screenshot,
                        "action_history": action_history,
                        "observation_history": observation_history,
                        "observation_history_available": has_full_history,
                    },
                },
                "scoring_join_key": {
                    "episode_id": row.get("episode_id"),
                    "step_idx": row.get("step_idx"),
                    "error_set_version": row.get("error_set_version"),
                },
                "sampling_weight": weights_by_key.get(sample_key(row), 1.0),
                "model_calls_must_use": "src/infer/wrapper.py",
                "error_set_version": row.get("error_set_version"),
                "run_scope": row.get("run_scope"),
                "sample_name": row.get("sample_name"),
            }
        )
    return items


def sample_error_rows(error_rows: List[Dict[str, Any]], cap: Optional[int], seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if cap is None or len(error_rows) <= cap:
        return error_rows, {
            "policy": "all_error_rows",
            "cap": cap,
            "population_error_rows": len(error_rows),
            "sampled_error_rows": len(error_rows),
            "weights_by_key": {sample_key(row): 1.0 for row in error_rows},
            "strata": [],
        }

    strata = build_strata(error_rows)
    allocations = allocate_stratified_counts(strata, cap)
    rng = random.Random(seed)
    sampled: List[Dict[str, Any]] = []
    weights: Dict[str, float] = {}
    stratum_rows = []
    for stratum, rows in sorted(strata.items()):
        take = allocations.get(stratum, 0)
        rows_sorted = sorted(rows, key=lambda row: (str(row.get("episode_id", "")), int(row.get("step_idx", 0))))
        chosen = rng.sample(rows_sorted, take) if take < len(rows_sorted) else rows_sorted
        sampled.extend(chosen)
        weight = len(rows_sorted) / take if take else 0.0
        for row in chosen:
            weights[sample_key(row)] = weight
        stratum_rows.append({"stratum": stratum, "population": len(rows_sorted), "sampled": take, "sampling_weight": weight})
    sampled.sort(key=lambda row: (str(row.get("episode_id", "")), int(row.get("step_idx", 0))))
    return sampled, {
        "policy": "stratified_by_episode_length_quartile_and_step_position_third",
        "cap": cap,
        "seed": seed,
        "population_error_rows": len(error_rows),
        "sampled_error_rows": len(sampled),
        "weights_by_key": weights,
        "strata": stratum_rows,
    }


def build_strata(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    lengths = sorted(int(row.get("num_steps", row.get("episode_len", 0)) or 0) for row in rows)
    q1 = lengths[len(lengths) // 4]
    q2 = lengths[len(lengths) // 2]
    q3 = lengths[(3 * len(lengths)) // 4]
    strata: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        length = int(row.get("num_steps", row.get("episode_len", 0)) or 0)
        step_idx = int(row.get("step_idx", 0))
        quartile = "q1" if length <= q1 else "q2" if length <= q2 else "q3" if length <= q3 else "q4"
        ratio = step_idx / max(length, 1)
        position = "early" if ratio < 1 / 3 else "mid" if ratio < 2 / 3 else "late"
        strata[f"len_{quartile}__pos_{position}"].append(row)
    return strata


def allocate_stratified_counts(strata: Dict[str, List[Dict[str, Any]]], cap: int) -> Dict[str, int]:
    total = sum(len(rows) for rows in strata.values())
    raw = {key: cap * len(rows) / total for key, rows in strata.items()}
    allocations = {key: min(len(strata[key]), max(1, int(math.floor(value)))) for key, value in raw.items()}
    while sum(allocations.values()) < cap:
        candidates = sorted(
            (raw[key] - allocations[key], key)
            for key in strata
            if allocations[key] < len(strata[key])
        )
        if not candidates:
            break
        _, key = candidates[-1]
        allocations[key] += 1
    while sum(allocations.values()) > cap:
        candidates = sorted(
            (allocations[key] - raw[key], key)
            for key in strata
            if allocations[key] > 1
        )
        if not candidates:
            break
        _, key = candidates[-1]
        allocations[key] -= 1
    return allocations


def sample_key(row: Dict[str, Any]) -> str:
    return f"{row.get('episode_id')}::{int(row.get('step_idx', 0))}"


def compact_action_history_item(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "step_idx": row.get("step_idx"),
        "answer": row.get("answer", ""),
        "pred_action_type": row.get("pred_action_type") or action_type(row, "pred_action"),
    }


def current_screenshot_value(row: Dict[str, Any], benchmark_rows: Dict[Tuple[str, int], Dict[str, Any]]) -> str:
    value = row.get("current_screenshot") or row.get("screenshot") or ""
    if not value:
        key = (str(row.get("episode_id", "")), int(row.get("step_idx", 0)))
        value = (benchmark_rows.get(key) or {}).get("screenshot", "")
    return "" if value is None else str(value)


def summarize_teacher_results(path: Optional[Path], error_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {
            "status": "NOT_STARTED",
            "path": workspace_relative(path) if path is not None else "",
            "matched_rows": 0,
            "expected_error_rows": len(error_rows),
            "bucket_counts": {bucket: 0 for bucket in BUCKETS},
            "weighted_b3_share": None,
            "weighted_b3_wilson_ci95": None,
        }
    rows = load_jsonl(path)
    error_keys = {(str(row.get("episode_id")), int(row.get("step_idx", 0))) for row in error_rows}
    counts: Counter[str] = Counter()
    weights: Counter[str] = Counter()
    matched = 0
    for row in rows:
        key = (str(row.get("episode_id")), int(row.get("step_idx", 0)))
        if key not in error_keys:
            continue
        bucket = str(row.get("bucket") or row.get("n0_bucket") or "")
        if bucket not in BUCKETS:
            continue
        weight = float(row.get("sampling_weight", 1.0) or 1.0)
        counts[bucket] += 1
        weights[bucket] += weight
        matched += 1
    denominator = sum(weights[bucket] for bucket in ("B1", "B2", "B3", "B4b"))
    b3_weight = weights["B3"]
    share = b3_weight / denominator if denominator else None
    ci = wilson_ci(b3_weight, denominator) if denominator else None
    status = "COMPLETE" if matched == len(error_rows) else "INCOMPLETE"
    return {
        "status": status,
        "path": workspace_relative(path),
        "matched_rows": matched,
        "expected_error_rows": len(error_rows),
        "bucket_counts": {bucket: counts[bucket] for bucket in BUCKETS},
        "bucket_weight_sums": {bucket: weights[bucket] for bucket in BUCKETS},
        "weighted_b3_share": share,
        "weighted_b3_wilson_ci95": ci,
    }


def wilson_ci(successes: float, total: float, z: float = 1.959963984540054) -> Dict[str, float]:
    if total <= 0:
        return {"low": 0.0, "high": 0.0}
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    half_width = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return {"low": max(0.0, center - half_width), "high": min(1.0, center + half_width)}


def build_summary(
    *,
    manifest: Dict[str, Any],
    error_rows: List[Dict[str, Any]],
    step_rows: List[Dict[str, Any]],
    assignments: List[Dict[str, Any]],
    teacher_inputs: List[Dict[str, Any]],
    sampling_summary: Dict[str, Any],
    teacher_result_summary: Dict[str, Any],
    output_dir: Path,
    error_set_path: Path,
    error_set_version: str,
) -> Dict[str, Any]:
    schema_coverage = summarize_schema_coverage(assignments)
    payload_coverage = summarize_payload_coverage(teacher_inputs)
    terminate_summary = build_terminate_summary(error_rows, step_rows)
    blockers = []
    if schema_coverage["other_percent"] > 25.0:
        blockers.append("schema_other_above_25_percent")
    if teacher_result_summary["status"] != "COMPLETE":
        blockers.append("teacher_probe_results_missing_or_incomplete")
    if payload_coverage["queued_percent"] < 100.0:
        blockers.append("teacher_probe_payload_incomplete")

    gate_p_status = "READY_FOR_C1_HUMAN_REVIEW" if not blockers else "BLOCKED"
    may_start_r = gate_p_status == "READY_FOR_C1_HUMAN_REVIEW"
    return {
        "phase_s_status": "COMPLETE",
        "phase_p_status": "COMPLETE_OFFLINE_SCAFFOLD",
        "gate_p_status": gate_p_status,
        "gate_p_blockers": blockers,
        "may_start_phase_r": may_start_r,
        "error_set_version": error_set_version,
        "run_scope": RUN_SCOPE,
        "sample_name": SAMPLE_NAME,
        "error_steps": len(error_rows),
        "total_steps": len(step_rows) or manifest.get("summary", {}).get("steps"),
        "schema_coverage": schema_coverage,
        "sampling": {key: value for key, value in sampling_summary.items() if key != "weights_by_key"},
        "teacher_payload_coverage": payload_coverage,
        "teacher_result_summary": teacher_result_summary,
        "terminate_deep_dive": terminate_summary,
        "files": {
            "canonical_error_set": workspace_relative(error_set_path),
            "report": workspace_relative(output_dir / "REPORT_V2.md"),
        },
    }


def summarize_schema_coverage(assignments: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter(str(row.get("schema_name", "other")) for row in assignments)
    total = len(assignments)
    rows = []
    for schema_name, count in sorted(counts.items()):
        rows.append({"schema_name": schema_name, "count": count, "percent": percent(count, total)})
    other_count = counts.get("other", 0)
    return {"total": total, "rows": rows, "other_count": other_count, "other_percent": percent(other_count, total)}


def summarize_payload_coverage(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(items)
    queued = sum(1 for item in items if item.get("teacher_probe_status") == "queued")
    missing_counts: Counter[str] = Counter()
    for item in items:
        missing_counts.update(item.get("missing_payload", []))
    return {
        "total": total,
        "queued": queued,
        "queued_percent": percent(queued, total),
        "blocked": total - queued,
        "blocked_percent": percent(total - queued, total),
        "missing_payload_counts": dict(sorted(missing_counts.items())),
    }


def build_terminate_summary(error_rows: List[Dict[str, Any]], step_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    terminate_gt_errors = 0
    missed_stop = 0
    wrong_status = 0
    false_stop = 0
    for row in error_rows:
        gt_type = row.get("gt_action_type") or action_type(row, "gt_action")
        pred_type = row.get("pred_action_type") or action_type(row, "pred_action")
        if gt_type == "terminate":
            terminate_gt_errors += 1
            if pred_type == "terminate":
                wrong_status += 1
            else:
                missed_stop += 1
    source_rows = step_rows or error_rows
    for row in source_rows:
        gt_type = row.get("gt_action_type") or action_type(row, "gt_action")
        pred_type = row.get("pred_action_type") or action_type(row, "pred_action")
        if gt_type != "terminate" and pred_type == "terminate" and row.get("baseline_error"):
            false_stop += 1
    return {
        "terminate_gt_errors": terminate_gt_errors,
        "missed_stop": missed_stop,
        "wrong_status": wrong_status,
        "false_stop_outside_terminate_gt": false_stop,
        "p4_probe_priority": "missed_stop_first",
    }


def percent(count: float, total: float) -> float:
    return 100.0 * count / total if total else 0.0


def update_manifest(path: Path, summary: Dict[str, Any]) -> None:
    manifest = load_json(path)
    manifest["phase_s_status"] = summary["phase_s_status"]
    manifest["gate_p_status"] = summary["gate_p_status"]
    manifest["may_start_phase_r"] = summary["may_start_phase_r"]
    manifest.setdefault("files", {}).update(
        {
            "phase_s_schema_assignment": summary["files"]["phase_s_schema_assignment"],
            "phase_p_teacher_probe_inputs": summary["files"]["phase_p_teacher_probe_inputs"],
            "phase_sp_summary": summary["files"]["phase_sp_summary"],
        }
    )
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def update_report(path: Path, summary: Dict[str, Any]) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else "# REPORT V2 - HAR GUI-Odyssey N0/N1 Gates\n"
    section = build_report_section(summary).rstrip() + "\n"
    heading = "## Phase S/P - C1 Offline Scaffold"
    if heading in text:
        start = text.index(heading)
        next_start = text.find("\n## ", start + 1)
        text = text[:start].rstrip() + "\n\n" + section if next_start == -1 else text[:start].rstrip() + "\n\n" + section + text[next_start:]
    else:
        text = text.rstrip() + "\n\n" + section
    path.write_text(text, encoding="utf-8")


def build_report_section(summary: Dict[str, Any]) -> str:
    schema = summary["schema_coverage"]
    payload = summary["teacher_payload_coverage"]
    teacher = summary["teacher_result_summary"]
    terminate = summary["terminate_deep_dive"]
    may_start_phase_r = "true" if summary["may_start_phase_r"] else "false"
    lines = [
        "## Phase S/P - C1 Offline Scaffold",
        "",
        f"All rows consume error-set `{summary['error_set_version']}`, run scope `{RUN_SCOPE}`, sample `{SAMPLE_NAME}`.",
        "Phase S/P in this pass is offline-only: zero model calls were made.",
        "",
        "### S1 Schema Coverage",
        "",
        "| Schema | Count | Share |",
        "| --- | ---: | ---: |",
    ]
    for row in schema["rows"]:
        lines.append(f"| `{row['schema_name']}` | {row['count']} | {row['percent']:.2f}% |")
    lines.extend(
        [
            "",
            f"Other share: `{schema['other_percent']:.2f}%` (C1 schema rule: `other > 25%` stops before R).",
            "",
            "### P1 Teacher Probe Payload Readiness",
            "",
            f"Sampling policy: `{summary['sampling']['policy']}`; population errors `{summary['sampling']['population_error_rows']}`; sampled errors `{summary['sampling']['sampled_error_rows']}`.",
            "",
            "| Item | Value |",
            "| --- | ---: |",
            f"| Probe input rows | {payload['total']} |",
            f"| Queued rows | {payload['queued']} |",
            f"| Queued share | {payload['queued_percent']:.2f}% |",
            f"| Blocked rows | {payload['blocked']} |",
        ]
    )
    for key, value in payload["missing_payload_counts"].items():
        lines.append(f"| Missing `{key}` | {value} |")
    lines.extend(
        [
            "",
            "### P2 Weighted B3 Share",
            "",
        ]
    )
    if teacher["weighted_b3_share"] is None:
        lines.append("Weighted B3 share is `not_computable`: `T_screen`, `T_act`, and `T_full` teacher probe results are not complete.")
    else:
        ci = teacher["weighted_b3_wilson_ci95"]
        lines.append(
            f"Weighted B3 share: `{100 * teacher['weighted_b3_share']:.2f}%` "
            f"(Wilson 95% CI `{100 * ci['low']:.2f}%` to `{100 * ci['high']:.2f}%`)."
        )
    lines.extend(
        [
            "",
            "Bucket counts from teacher results:",
            "",
            "| Bucket | Count |",
            "| --- | ---: |",
        ]
    )
    for bucket in BUCKETS:
        lines.append(f"| `{bucket}` | {teacher['bucket_counts'].get(bucket, 0)} |")
    lines.extend(
        [
            "",
            "### P4 Terminate Deep-Dive",
            "",
            "| Bucket | Count |",
            "| --- | ---: |",
            f"| `terminate_gt_errors` | {terminate['terminate_gt_errors']} |",
            f"| `missed_stop` | {terminate['missed_stop']} |",
            f"| `wrong_status` | {terminate['wrong_status']} |",
            f"| `false_stop_outside_terminate_gt` | {terminate['false_stop_outside_terminate_gt']} |",
            "",
            "P4 probe priority: `missed_stop_first`.",
            "",
            "### GATE P / C1",
            "",
            f"Status: `{summary['gate_p_status']}`.",
            f"May start Phase R: `{may_start_phase_r}`.",
        ]
    )
    if summary["gate_p_blockers"]:
        lines.extend(["", "Blockers:"])
        for blocker in summary["gate_p_blockers"]:
            lines.append(f"- `{blocker}`")
    lines.extend(
        [
            "",
            f"Artifacts: `{summary['files']['phase_s_schema_assignment']}`, `{summary['files']['phase_p_teacher_probe_inputs']}`, `{summary['files']['phase_sp_summary']}`.",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())