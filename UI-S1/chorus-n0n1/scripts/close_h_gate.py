#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent
for path in [ROOT, WORKSPACE_ROOT, WORKSPACE_ROOT / "related_work" / "har", WORKSPACE_ROOT / "gui_odyssey_eval"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gui_odyssey_eval.odyssey_action_matching import evaluate_odyssey_action, get_scroll_direction, pred_coord_to_1k  # noqa: E402
from related_work.har.action_parser import parse_har_output  # noqa: E402
from src.error_sets import assert_error_set_version, tag_records  # noqa: E402
from src.probes.headroom import build_headroom_probe_queue  # noqa: E402
from src.readers.disagreement import build_reader_queue  # noqa: E402


DIRECTIONS = ["up", "down", "left", "right"]
ANTI_PAIRS = {("up", "down"), ("down", "up"), ("left", "right"), ("right", "left")}
RUN_SCOPE = "half_run"
SAMPLE_NAME = "har_gui_odyssey_latest"


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    output_dir = resolve(args.output_dir)
    audit_dir = output_dir / "audit"
    error_set_dir = output_dir / "error_sets"
    audit_dir.mkdir(parents=True, exist_ok=True)
    error_set_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(resolve(args.step_rows))
    hc1 = build_swipe_confusion(rows)
    hc2 = build_long_press_summary(rows)
    decision = decide_error_set(hc1)
    final_rows = rescore_rows(rows, flip_swipes=decision["flip_swipes"])
    final_summary = summarize_rows(final_rows)
    hc3 = build_terminate_summary(final_rows)

    version = decision["error_set_version"]
    tagged_rows = tag_records(final_rows, error_set_version=version, run_scope=RUN_SCOPE, sample_name=SAMPLE_NAME)
    canonical_errors = [row for row in tagged_rows if row.get("baseline_error")]
    assert_error_set_version(
        canonical_errors,
        expected_error_set_version=version,
        expected_run_scope=RUN_SCOPE,
        expected_sample_name=SAMPLE_NAME,
    )

    error_set_path = error_set_dir / f"error_set_{version}.jsonl"
    write_jsonl(error_set_path, canonical_errors)
    if decision["status"] == "AMBIGUOUS_HUMAN_REVIEW_REQUIRED":
        write_swipe_cases(audit_dir / "swipe_ambiguous_extra_cases.md", sample_swipe_errors(final_rows, args.seed, 30))

    rewrite_versioned_artifacts(run_dir, tagged_rows, version)
    summary = build_summary(decision, version, hc1, hc2, hc3, final_summary, output_dir, error_set_path)
    (output_dir / "h_closure_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_manifest(run_dir / "manifest.json", summary)
    update_report(output_dir / "REPORT_V2.md", summary)

    print(json.dumps({
        "status": summary["status"],
        "gate_h_final": summary["gate_h_final"],
        "error_set_version": version,
        "diag": hc1["diag"],
        "anti": hc1["anti"],
        "canonical_errors": len(canonical_errors),
        "report": str(output_dir / "REPORT_V2.md"),
    }, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Close HAR GUI-Odyssey Gate H with offline H-closure checks")
    parser.add_argument("--run_dir", default="chorus-n0n1/runs/n0n1_inputs/har_gui_odyssey_latest")
    parser.add_argument("--step_rows", default="chorus-n0n1/runs/n0n1_inputs/har_gui_odyssey_latest/har_gui_odyssey_steps.jsonl")
    parser.add_argument("--output_dir", default="chorus-n0n1/runs/n0n1_inputs/har_gui_odyssey_latest/offline_analysis")
    parser.add_argument("--seed", type=int, default=20260610)
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def workspace_relative(path: Path) -> str:
    try:
        return str(path.relative_to(WORKSPACE_ROOT))
    except ValueError:
        return str(path)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def action_type(action: Optional[Dict[str, Any]]) -> str:
    return str((action or {}).get("action", ""))


def axis(direction: str) -> str:
    if direction in {"up", "down"}:
        return "vertical"
    if direction in {"left", "right"}:
        return "horizontal"
    return "unknown"


def swipe_direction(action: Dict[str, Any], *, normalize_pred: bool, row: Dict[str, Any]) -> str:
    if action_type(action) != "swipe":
        return "n/a"
    start = action.get("coordinate", [0, 0])
    end = action.get("coordinate2") or action.get("endCoordinate") or [0, 0]
    if normalize_pred:
        width = int(row.get("image_width") or 1)
        height = int(row.get("image_height") or 1)
        start = pred_coord_to_1k(start, width, height)
        end = pred_coord_to_1k(end, width, height)
    return get_scroll_direction(start, end)


def build_swipe_confusion(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    matrix: Dict[str, Dict[str, int]] = {gt: {pred: 0 for pred in DIRECTIONS} for gt in DIRECTIONS}
    error_counts = Counter()
    gt_swipes = [row for row in rows if action_type(row.get("gt_action")) == "swipe"]
    pred_swipe_on_gt = 0
    for row in gt_swipes:
        pred_type = action_type(row.get("pred_action"))
        if pred_type == "swipe":
            gt_dir = swipe_direction(row.get("gt_action") or {}, normalize_pred=False, row=row)
            pred_dir = swipe_direction(row.get("pred_action") or {}, normalize_pred=True, row=row)
            if gt_dir in DIRECTIONS and pred_dir in DIRECTIONS:
                matrix[gt_dir][pred_dir] += 1
                pred_swipe_on_gt += 1
        if not row.get("baseline_error"):
            continue
        if pred_type != "swipe":
            error_counts["type_mismatch"] += 1
        else:
            gt_dir = swipe_direction(row.get("gt_action") or {}, normalize_pred=False, row=row)
            pred_dir = swipe_direction(row.get("pred_action") or {}, normalize_pred=True, row=row)
            error_counts["axis_mismatch" if axis(gt_dir) != axis(pred_dir) else "same_axis_direction_mismatch"] += 1
    diag = sum(matrix[direction][direction] for direction in DIRECTIONS)
    anti = sum(matrix[gt][pred] for gt, pred in ANTI_PAIRS)
    return {
        "gt_swipe_steps": len(gt_swipes),
        "pred_swipe_on_gt_swipe_steps": pred_swipe_on_gt,
        "swipe_error_steps": sum(1 for row in gt_swipes if row.get("baseline_error")),
        "matrix_gt_by_pred": matrix,
        "diag": diag,
        "anti": anti,
        "swipe_error_decomposition": {
            "type_mismatch": error_counts.get("type_mismatch", 0),
            "axis_mismatch": error_counts.get("axis_mismatch", 0),
            "same_axis_direction_mismatch": error_counts.get("same_axis_direction_mismatch", 0),
        },
        "cross_evidence": build_cross_evidence(),
    }


def build_long_press_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    emissions = []
    gt_long_press_emissions = []
    for row in rows:
        _, answer, parsed = parse_har_output(row.get("raw_text") or row.get("answer", ""))
        if action_type(parsed) != "long_press" and action_type(row.get("pred_action")) != "long_press":
            continue
        emission = {
            "episode_id": row.get("episode_id"),
            "step_idx": row.get("step_idx"),
            "answer": answer or row.get("answer", ""),
            "gt_action_type": action_type(row.get("gt_action")),
            "type_match": bool(row.get("type_match")),
            "extract_match": bool(row.get("extract_match")),
        }
        emissions.append(emission)
        if action_type(row.get("gt_action")) == "long_press":
            gt_long_press_emissions.append(emission)
    return {
        "model_long_press_emissions_all_steps": len(emissions),
        "gt_long_press_steps_with_long_press_emission": len(gt_long_press_emissions),
        "gt_long_press_emission_failures": classify_long_press_emissions(gt_long_press_emissions),
        "verdict": "behavioral_gap_model_never_long_presses" if not emissions else "long_press_emissions_present_errors_genuine",
    }


def classify_long_press_emissions(emissions: List[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter()
    for emission in emissions:
        if not emission["type_match"]:
            counts["type_failure"] += 1
        elif not emission["extract_match"]:
            counts["coordinate_failure"] += 1
        else:
            counts["match"] += 1
    return dict(sorted(counts.items()))


def decide_error_set(hc1: Dict[str, Any]) -> Dict[str, Any]:
    diag = int(hc1["diag"])
    anti = int(hc1["anti"])
    if anti > 2 * diag:
        return {"status": "CONVENTION_INVERSION_CONFIRMED", "gate_h_final": "APPROVED_AFTER_GLOBAL_SWIPE_FLIP", "error_set_version": "E_v2", "flip_swipes": True}
    if diag >= anti:
        return {"status": "CONVENTION_CORRECT_ERRORS_GENUINE", "gate_h_final": "APPROVED", "error_set_version": "E_v1", "flip_swipes": False}
    return {"status": "AMBIGUOUS_HUMAN_REVIEW_REQUIRED", "gate_h_final": "PENDING_HUMAN_REVIEW", "error_set_version": "E_unresolved", "flip_swipes": False}


def rescore_rows(rows: List[Dict[str, Any]], *, flip_swipes: bool) -> List[Dict[str, Any]]:
    final_rows = []
    for row in rows:
        updated = copy.deepcopy(row)
        pred_action = copy.deepcopy(updated.get("pred_action") or {})
        if flip_swipes and action_type(pred_action) == "swipe":
            pred_action = invert_swipe(pred_action)
            updated["pred_action"] = pred_action
            updated["pred_action_type"] = "swipe"
            updated["h_closure_swipe_mapping"] = "global_inverted"
        elif action_type(pred_action) == "swipe":
            updated["h_closure_swipe_mapping"] = "current"
        if pred_action and updated.get("gt_action") and updated.get("image_width") and updated.get("image_height"):
            try:
                type_match, extract_match = evaluate_odyssey_action(copy.deepcopy(pred_action), copy.deepcopy(updated["gt_action"]), int(updated["image_width"]), int(updated["image_height"]))
                updated["type_match"] = bool(type_match)
                updated["extract_match"] = bool(extract_match)
                updated["baseline_error"] = not bool(extract_match)
            except Exception as exc:
                updated["h_closure_rescore_error"] = repr(exc)
        final_rows.append(updated)
    return final_rows


def invert_swipe(action: Dict[str, Any]) -> Dict[str, Any]:
    inverted = dict(action)
    start = inverted.get("coordinate")
    end = inverted.get("coordinate2") or inverted.get("endCoordinate")
    if start is not None and end is not None:
        inverted["coordinate"] = list(end)
        inverted["coordinate2"] = list(start)
        inverted.pop("endCoordinate", None)
    return inverted


def build_terminate_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    terminate_gt_errors = [row for row in rows if action_type(row.get("gt_action")) == "terminate" and row.get("baseline_error")]
    missed_stop = [row for row in terminate_gt_errors if action_type(row.get("pred_action")) != "terminate"]
    wrong_status = [row for row in terminate_gt_errors if action_type(row.get("pred_action")) == "terminate"]
    false_stop = [row for row in rows if action_type(row.get("gt_action")) != "terminate" and action_type(row.get("pred_action")) == "terminate"]
    return {"terminate_gt_errors": len(terminate_gt_errors), "missed_stop": len(missed_stop), "wrong_status": len(wrong_status), "false_stop_outside_terminate_gt": len(false_stop), "p4_probe_priority": "missed_stop_first"}


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row.get("extract_match"))
    error_kinds = Counter()
    action_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"steps": 0, "errors": 0})
    for row in rows:
        gt_type = action_type(row.get("gt_action")) or "unknown"
        action_counts[gt_type]["steps"] += 1
        if row.get("baseline_error"):
            action_counts[gt_type]["errors"] += 1
            error_kinds[error_kind(row)] += 1
    return {"steps": total, "correct_steps": correct, "baseline_error_steps": total - correct, "step_sr_percent": round(100 * correct / total, 4) if total else 0.0, "error_kind_counts": dict(sorted(error_kinds.items())), "by_gt_action": action_table(action_counts)}


def error_kind(row: Dict[str, Any]) -> str:
    if row.get("type_match") is False:
        return "action_type"
    if action_type(row.get("gt_action")) in {"click", "swipe", "long_press"} or action_type(row.get("pred_action")) in {"click", "swipe", "long_press"}:
        return "coordinate_or_target"
    if row.get("truncated"):
        return "truncation"
    if row.get("error"):
        return "parse_or_runtime"
    return "semantic_or_sequence"


def action_table(action_counts: Dict[str, Dict[str, int]]) -> List[Dict[str, Any]]:
    table = []
    for action, counts in sorted(action_counts.items(), key=lambda item: (-item[1]["errors"], item[0])):
        steps = counts["steps"]
        errors = counts["errors"]
        table.append({"gt_action": action, "steps": steps, "errors": errors, "error_rate_percent": round(100 * errors / steps, 2) if steps else 0.0})
    return table


def build_cross_evidence() -> Dict[str, Any]:
    prompt_text = read_text(WORKSPACE_ROOT / "related_work" / "har" / "Prompts" / "Inference.py")
    original_eval_text = read_text(WORKSPACE_ROOT / "related_work" / "har" / "Inference" / "demo_episode_inference.py")
    parser_text = read_text(WORKSPACE_ROOT / "related_work" / "har" / "action_parser.py")
    return {
        "har_prompt_scroll_semantics_line": extract_scroll_semantics_line(prompt_text),
        "original_eval_scroll_mapping": "not_found" if "scroll_to_action" not in original_eval_text and "_SCROLL_MAP" not in original_eval_text else "found",
        "original_eval_note": "HAR original demo keeps SCROLL as an action string in inference_data; no coordinate mapping implementation was found in demo_episode_inference.py.",
        "local_parser_mapping_lines": extract_mapping_lines(parser_text),
    }


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def extract_scroll_semantics_line(text: str) -> str:
    for line in text.splitlines():
        if "SCROLL:UP/DOWN/LEFT/RIGHT" in line:
            return line.strip()
    return "SCROLL semantics line not found in the HAR prompt."


def extract_mapping_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if any(f'"{direction.upper()}"' in line for direction in DIRECTIONS) and ("coordinate" in line or "endCoordinate" in line)]


def sample_swipe_errors(rows: List[Dict[str, Any]], seed: int, count: int) -> List[Dict[str, Any]]:
    swipe_errors = [row for row in rows if action_type(row.get("gt_action")) == "swipe" and row.get("baseline_error")]
    return random.Random(seed).sample(swipe_errors, min(count, len(swipe_errors)))


def write_swipe_cases(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = ["# Ambiguous Swipe Cases - H-Closure", "", "Generated only if HC1 is ambiguous.", ""]
    for index, row in enumerate(rows, start=1):
        lines.extend([f"## Case {index}: episode {row.get('episode_id')} step {row.get('step_idx')}", "", f"- Screenshot: `{row.get('screenshot', '')}`", f"- GT action: `{json.dumps(row.get('gt_action'), ensure_ascii=False)}`", f"- Pred action: `{json.dumps(row.get('pred_action'), ensure_ascii=False)}`", f"- GT direction: `{swipe_direction(row.get('gt_action') or {}, normalize_pred=False, row=row)}`", f"- Pred direction: `{swipe_direction(row.get('pred_action') or {}, normalize_pred=True, row=row)}`", "", "Raw model output:", "", "```text", row.get("raw_text") or row.get("answer", ""), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def rewrite_versioned_artifacts(run_dir: Path, tagged_rows: List[Dict[str, Any]], version: str) -> None:
    write_jsonl(run_dir / "har_gui_odyssey_steps.jsonl", tagged_rows)
    write_jsonl(run_dir / "n0_headroom_queue.jsonl", tag_records(build_headroom_probe_queue(tagged_rows), error_set_version=version, run_scope=RUN_SCOPE, sample_name=SAMPLE_NAME))
    write_jsonl(run_dir / "n1_reader_inputs_queue.jsonl", tag_records(build_reader_queue(tagged_rows), error_set_version=version, run_scope=RUN_SCOPE, sample_name=SAMPLE_NAME))


def build_summary(decision: Dict[str, Any], version: str, hc1: Dict[str, Any], hc2: Dict[str, Any], hc3: Dict[str, Any], final_summary: Dict[str, Any], output_dir: Path, error_set_path: Path) -> Dict[str, Any]:
    return {"status": decision["status"], "gate_h_final": decision["gate_h_final"], "error_set_version": version, "run_scope": RUN_SCOPE, "sample_name": SAMPLE_NAME, "hc1": hc1, "hc2": hc2, "hc3": hc3, "final_summary": final_summary, "files": {"canonical_error_set": workspace_relative(error_set_path), "h_closure_summary": workspace_relative(output_dir / "h_closure_summary.json"), "report": workspace_relative(output_dir / "REPORT_V2.md")}}


def update_manifest(path: Path, summary: Dict[str, Any]) -> None:
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["gate_g_status"] = "APPROVED"
    manifest["gate_h_status"] = summary["gate_h_final"]
    manifest["error_set"] = {"version": summary["error_set_version"], "run_scope": RUN_SCOPE, "sample_name": SAMPLE_NAME, "canonical_error_set": summary["files"]["canonical_error_set"], "h_closure_status": summary["status"]}
    manifest.setdefault("files", {})["canonical_error_set"] = summary["files"]["canonical_error_set"]
    manifest.setdefault("files", {})["h_closure_summary"] = summary["files"]["h_closure_summary"]
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def update_report(path: Path, summary: Dict[str, Any]) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else "# REPORT V2 - HAR GUI-Odyssey N0/N1 Gates\n"
    h_status = "APPROVED_BY_H_CLOSURE" if summary["gate_h_final"] != "PENDING_HUMAN_REVIEW" else "PENDING_HUMAN_REVIEW"
    text = replace_gate_status(replace_gate_status(text, "### GATE H", h_status), "### GATE G", "APPROVED")
    section = build_report_section(summary).rstrip() + "\n"
    heading = "## Phase H-Closure - v3 Offline Resolution"
    if heading in text:
        start = text.index(heading)
        next_start = text.find("\n## ", start + 1)
        text = text[:start].rstrip() + "\n\n" + section if next_start == -1 else text[:start].rstrip() + "\n\n" + section + text[next_start:]
    else:
        text = text.rstrip() + "\n\n" + section
    path.write_text(text, encoding="utf-8")


def replace_gate_status(text: str, heading: str, status: str) -> str:
    start = text.find(heading)
    if start == -1:
        return text
    end = text.find("\n## ", start + 1)
    if end == -1:
        end = len(text)
    block = text[start:end]
    if "Status: `" not in block:
        return text[:start] + block.rstrip() + f"\n\nStatus: `{status}`.\n" + text[end:]
    before, rest = block.split("Status: `", 1)
    _, after = rest.split("`.", 1)
    return text[:start] + before + f"Status: `{status}`." + after + text[end:]


def build_report_section(summary: Dict[str, Any]) -> str:
    hc1 = summary["hc1"]
    hc2 = summary["hc2"]
    hc3 = summary["hc3"]
    final_summary = summary["final_summary"]
    evidence = hc1["cross_evidence"]
    version = summary["error_set_version"]
    lines = ["## Phase H-Closure - v3 Offline Resolution", "", f"All numbers in this section are tagged with error-set `{version}`, run scope `{RUN_SCOPE}`, sample `{SAMPLE_NAME}`.", "", "### HC1 Swipe Direction Confusion Matrix", "", f"GT swipe steps: `{hc1['gt_swipe_steps']}`; pred-swipe-on-GT-swipe steps: `{hc1['pred_swipe_on_gt_swipe_steps']}`; swipe error steps: `{hc1['swipe_error_steps']}`.", "", "| GT direction | Pred up | Pred down | Pred left | Pred right |", "| --- | ---: | ---: | ---: | ---: |"]
    for gt_direction in DIRECTIONS:
        row = hc1["matrix_gt_by_pred"][gt_direction]
        lines.append(f"| `{gt_direction}` | {row['up']} | {row['down']} | {row['left']} | {row['right']} |")
    lines.extend(["", f"Decision masses: `diag={hc1['diag']}`, `anti={hc1['anti']}`.", "", "Swipe error decomposition:", "", "| Bucket | Count |", "| --- | ---: |"])
    for key, value in hc1["swipe_error_decomposition"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "Cross-evidence:", "", f"- HAR prompt line: `{evidence['har_prompt_scroll_semantics_line']}`", f"- HAR original eval mapping diff: `{evidence['original_eval_scroll_mapping']}`; {evidence['original_eval_note']}", f"- Local parser mapping: `{'; '.join(evidence['local_parser_mapping_lines'])}`", "", "### HC2 long_press Emission Count", "", f"Model outputs parsed/emitted as `long_press` across all 7374 steps: `{hc2['model_long_press_emissions_all_steps']}`.", f"GT-long_press steps with long_press emission: `{hc2['gt_long_press_steps_with_long_press_emission']}`.", f"Verdict: `{hc2['verdict']}`.", "", "### HC3 terminate Decomposition", "", "| Bucket | Count |", "| --- | ---: |", f"| `terminate_gt_errors` | {hc3['terminate_gt_errors']} |", f"| `missed_stop` | {hc3['missed_stop']} |", f"| `wrong_status` | {hc3['wrong_status']} |", f"| `false_stop_outside_terminate_gt` | {hc3['false_stop_outside_terminate_gt']} |", "", "P4 probe priority: `missed_stop_first`.", "", "### H-Closure Re-baseline", "", "| Metric | Value |", "| --- | ---: |", f"| Steps | {final_summary['steps']} |", f"| Correct steps | {final_summary['correct_steps']} |", f"| Error steps | {final_summary['baseline_error_steps']} |", f"| Step SR | {final_summary['step_sr_percent']:.2f}% |", "", "Per-GT-action table:", "", "| GT action | Steps | Errors | Error rate |", "| --- | ---: | ---: | ---: |"])
    for row in final_summary["by_gt_action"]:
        lines.append(f"| `{row['gt_action']}` | {row['steps']} | {row['errors']} | {row['error_rate_percent']:.2f}% |")
    lines.extend(["", "### GATE H FINAL", "", f"Verdict: `{summary['gate_h_final']}`.", f"Error set: `{version}`.", f"Canonical error set: `{summary['files']['canonical_error_set']}`.", "GATE G: `APPROVED` on record from Spec v3.", ""])
    if summary["gate_h_final"] == "PENDING_HUMAN_REVIEW":
        lines.append("Do not proceed to S/P/R/X/Y until the ambiguous swipe cases receive a human call.")
    else:
        lines.append("H is closed for downstream phases; S/P/R/X/Y must consume rows tagged with this error-set version.")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
