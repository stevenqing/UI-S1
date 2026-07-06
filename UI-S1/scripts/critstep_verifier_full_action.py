#!/usr/bin/env python3
"""Build and evaluate a critical-step full-action verifier slice.

This script prepares no-leakage candidate-level verifier examples from an
existing UIA sampled pool and writes a 200-step held-out evaluation slice. It
does not train a verifier unless a separate TRAIN-side sampled pool is supplied;
by default it treats the current TEST-side pool as evaluation-only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for import_path in (REPO_ROOT, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from critstep_reward_structure_uia import (  # noqa: E402
    action_point,
    assign_control,
    control_key,
    control_label,
    control_rect,
    control_text,
    control_type,
    controls_for_step,
)
from v13_gui_360.eval_gui360_template import _format_action_for_history  # noqa: E402


DEFAULT_SAMPLES = "outputs/critstep_elicit_uia/per_step.jsonl"
DEFAULT_TEST_DATA = "outputs/gui360_history_ab/original_eval/gui360_test_1000_balanced_uia.jsonl"
DEFAULT_UIA_PER_STEP = "outputs/critstep_reward_structure_uia/per_step.jsonl"
DEFAULT_SCOPE_PER_STEP = "outputs/critstep_scope/per_step.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/critstep_verifier"
ORACLE_CRITICAL_TSR_CEILING_PP = 23.767937761890312

ACTION_ALIASES = {
    "tap": "click",
    "left_click": "click",
    "double_click": "click",
    "double_click_input": "click",
    "double_click_on_coordinates": "click",
    "input": "type",
    "drag": "swipe",
    "scroll": "swipe",
    "wheel_mouse_input": "swipe",
    "press": "key",
    "hotkey": "key",
    "shortcut": "key",
    "back": "system_button",
    "home": "system_button",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_episodes(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row["episode_id"]): row for row in read_jsonl(path)}


def key_for(row: Mapping[str, Any]) -> Tuple[str, int, str]:
    return (str(row.get("episode_id")), int(row.get("step_idx")), str(row.get("target_id")))


def normalize_action_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    return ACTION_ALIASES.get(text, text)


def action_category(value: Any) -> str:
    action_type_value = normalize_action_type(value)
    if action_type_value in {"click", "long_press"}:
        return "click"
    if action_type_value == "type":
        return "type"
    if action_type_value == "swipe":
        return "swipe"
    if action_type_value in {"key", "system_button"}:
        return "special-key"
    return "other"


def pct(numerator: float, denominator: float) -> str:
    if not denominator:
        return "0.00%"
    return f"{100.0 * numerator / denominator:.2f}%"


def action_signature(action: Any) -> str:
    if not isinstance(action, dict):
        return "null"
    return json.dumps(action, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def get_gt_type(sample_row: Mapping[str, Any]) -> str:
    greedy = sample_row.get("greedy") if isinstance(sample_row.get("greedy"), dict) else {}
    return normalize_action_type(greedy.get("gt_type") or sample_row.get("action_type"))


def get_greedy_type(sample_row: Mapping[str, Any]) -> str:
    greedy = sample_row.get("greedy") if isinstance(sample_row.get("greedy"), dict) else {}
    return normalize_action_type(greedy.get("pred_type") or "")


def first_correct_rank(sample_row: Mapping[str, Any]) -> Optional[int]:
    value = sample_row.get("first_correct_rank")
    if value is None:
        return None
    try:
        rank = int(value)
    except (TypeError, ValueError):
        return None
    return rank if rank > 0 else None


def depth_bin(rank: Optional[int]) -> str:
    if rank is None:
        return "missing@50"
    if rank <= 5:
        return "shallow_1_5"
    if rank <= 20:
        return "mid_6_20"
    return "deep_21_50"


def history_for_step(steps: Sequence[Mapping[str, Any]], step_idx: int) -> List[str]:
    history: List[str] = []
    for history_idx, step in enumerate(steps[:step_idx]):
        action = step.get("action") if isinstance(step.get("action"), dict) else {}
        history.append(_format_action_for_history(action, history_idx + 1))
    return history


def load_index(path: Path) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    if not path.exists():
        return {}
    return {key_for(row): row for row in read_jsonl(path)}


def load_scope_index(path: Path) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    if not path.exists():
        return {}
    return {key_for(row): row for row in read_jsonl(path)}


def control_payload(control: Optional[Dict[str, Any]], assignment: Mapping[str, Any]) -> Dict[str, Any]:
    rect = control_rect(control) if isinstance(control, dict) else None
    return {
        "key": control_key(control),
        "label": control_label(control),
        "type": control_type(control),
        "text": control_text(control),
        "rect": list(rect) if rect else None,
        "assignment": assignment.get("assignment"),
        "distance_px": assignment.get("distance_px"),
    }


def render_history(history: Sequence[str]) -> str:
    if not history:
        return "None"
    return "\n".join(history)


def render_candidate_control(control_info: Mapping[str, Any]) -> str:
    if not control_info.get("key"):
        return "No UIA control assigned to this candidate action. For type/key/wait actions, judge the action type and content directly."
    return "\n".join([
        f"assignment: {control_info.get('assignment')}",
        f"label: {control_info.get('label')}",
        f"control_type: {control_info.get('type')}",
        f"control_text: {control_info.get('text')}",
        f"control_rect: {control_info.get('rect')}",
        f"distance_px: {control_info.get('distance_px')}",
    ])


def verifier_prompt(goal: str, history: Sequence[str], candidate_action: Mapping[str, Any], control_info: Mapping[str, Any]) -> str:
    action_text = json.dumps(candidate_action, ensure_ascii=False, sort_keys=True)
    return (
        "<image>\n"
        "You are a generative verifier for GUI actions. Given the current screenshot, user instruction, action history, "
        "and exactly one candidate action, decide whether the candidate is the correct next FULL ACTION.\n\n"
        "Judge the whole action: action type, target element/control if any, coordinates if relevant, and typed/key content if relevant.\n"
        "Use only the screenshot, instruction, history, candidate action, and candidate UIA metadata below. Do not assume candidate frequency or rank.\n\n"
        f"Instruction:\n{goal}\n\n"
        f"Action history:\n{render_history(history)}\n\n"
        f"Candidate action JSON:\n{action_text}\n\n"
        f"Candidate UIA control metadata:\n{render_candidate_control(control_info)}\n\n"
        "Return a brief reason, then a final line exactly one of:\n"
        "VERDICT: correct\n"
        "VERDICT: incorrect"
    )


def assistant_target(is_correct: bool) -> str:
    verdict = "correct" if is_correct else "incorrect"
    if is_correct:
        reason = "The candidate matches the required next full action under the instruction and current GUI state."
    else:
        reason = "The candidate does not match the required next full action under the instruction and current GUI state."
    return f"Reason: {reason}\nVERDICT: {verdict}"


def build_sharegpt_example(prompt: str, image_path: str, is_correct: bool, metadata: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": assistant_target(is_correct)},
        ],
        "images": [image_path],
        "metadata": dict(metadata),
    }


def candidate_from_payload(
    *,
    candidate_id: str,
    source: str,
    payload: Mapping[str, Any],
    step: Mapping[str, Any],
    controls: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    candidate_action = payload.get("pred_action") if isinstance(payload.get("pred_action"), dict) else {}
    candidate_point = action_point(candidate_action)
    assignment = assign_control(candidate_point, controls)
    control_info = control_payload(assignment.get("control"), assignment)
    return {
        "candidate_id": candidate_id,
        "source": source,
        "action": candidate_action,
        "action_signature": action_signature(candidate_action),
        "pred_type": normalize_action_type(payload.get("pred_type") or candidate_action.get("action")),
        "pred_category": action_category(payload.get("pred_type") or candidate_action.get("action")),
        "bucket": payload.get("bucket"),
        "reward": payload.get("reward"),
        "is_correct": bool(payload.get("success")),
        "pred_text": payload.get("pred_text"),
        "control": control_info,
        "image_w": int(step.get("image_w") or 1040),
        "image_h": int(step.get("image_h") or 736),
        "verifier_score": None,
    }


def full_action_subset(sample_row: Mapping[str, Any], uia_row: Optional[Mapping[str, Any]], scope_row: Optional[Mapping[str, Any]]) -> str:
    greedy_bucket = str(sample_row.get("greedy_bucket") or "")
    if greedy_bucket == "type_mismatch":
        return "action_type_mismatch"
    if greedy_bucket.startswith("same_type_non_click:type"):
        return "type_content"
    if greedy_bucket == "format_error":
        return "format_parse"
    if uia_row and uia_row.get("analyzable") and uia_row.get("different_control_majority"):
        return "click_element_selection"
    if uia_row and uia_row.get("analyzable") and uia_row.get("same_control_majority"):
        return "click_coordinate"
    if scope_row:
        if scope_row.get("scope_flag") == "click_parse_or_action_format_gap":
            return "action_type_mismatch"
        if scope_row.get("failure_kind") == "TYPE-CONTENT error":
            return "type_content"
        if scope_row.get("failure_kind") == "ACTION-TYPE mismatch":
            return "action_type_mismatch"
    if greedy_bucket in {"far_miss", "mid_miss"} and action_category(get_gt_type(sample_row)) == "click":
        return "click_candidate_grounding"
    return "other"


def build_step_record(
    sample_row: Mapping[str, Any],
    episode: Mapping[str, Any],
    uia_row: Optional[Mapping[str, Any]],
    scope_row: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    step_idx = int(sample_row.get("step_idx"))
    steps = episode.get("steps") if isinstance(episode.get("steps"), list) else []
    step = steps[step_idx]
    controls = controls_for_step(step)
    candidates: List[Dict[str, Any]] = []
    greedy_payload = sample_row.get("greedy") if isinstance(sample_row.get("greedy"), dict) else {}
    candidates.append(candidate_from_payload(candidate_id="greedy", source="greedy", payload=greedy_payload, step=step, controls=controls))
    samples = sample_row.get("samples") if isinstance(sample_row.get("samples"), list) else []
    for sample_idx, sample_payload in enumerate(samples):
        if isinstance(sample_payload, dict):
            candidates.append(candidate_from_payload(candidate_id=f"sample_{sample_idx:02d}", source="sample", payload=sample_payload, step=step, controls=controls))

    correct_candidates = [candidate for candidate in candidates if candidate["is_correct"]]
    first_sample = next((candidate for candidate in candidates if candidate["source"] == "sample"), None)
    first_correct = first_correct_rank(sample_row)
    sample_correct_ids = [candidate["candidate_id"] for candidate in candidates if candidate["source"] == "sample" and candidate["is_correct"]]
    oracle_pick = sample_correct_ids[0] if sample_correct_ids else None
    return {
        "target_id": sample_row.get("target_id"),
        "episode_id": str(sample_row.get("episode_id")),
        "step_idx": step_idx,
        "instruction": episode.get("goal", ""),
        "screenshot": step.get("screenshot"),
        "history": history_for_step(steps, step_idx),
        "gt_action_type": get_gt_type(sample_row),
        "gt_action_category": action_category(get_gt_type(sample_row)),
        "greedy_action_type": get_greedy_type(sample_row),
        "greedy_bucket": sample_row.get("greedy_bucket"),
        "subset": full_action_subset(sample_row, uia_row, scope_row),
        "first_correct_rank": first_correct,
        "depth_bin": depth_bin(first_correct),
        "success_count": int(sample_row.get("success_count") or len(correct_candidates)),
        "n_candidates": len(candidates),
        "n_correct_candidates": len(correct_candidates),
        "oracle_candidate_id": oracle_pick,
        "first_sample_candidate_id": first_sample["candidate_id"] if first_sample else None,
        "first_sample_correct": bool(first_sample and first_sample["is_correct"]),
        "greedy_correct": bool(candidates[0]["is_correct"]),
        "logprob_rank_candidate_id": None,
        "logprob_rank_correct": None,
        "logprob_rank_available": False,
        "verifier_candidate_id": None,
        "verifier_correct": None,
        "greedy_rejected_by_verifier": None,
        "candidates": candidates,
    }


def stratified_slice(rows: Sequence[Dict[str, Any]], limit: int, seed: int) -> List[Dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["depth_bin"])].append(row)
    rng = random.Random(seed)
    for group_rows in grouped.values():
        rng.shuffle(group_rows)
    total = len(rows)
    quotas = {key: max(1, math.floor(limit * len(group_rows) / total)) for key, group_rows in grouped.items() if group_rows}
    while sum(quotas.values()) > limit:
        key = max(quotas, key=lambda item: quotas[item])
        quotas[key] -= 1
    while sum(quotas.values()) < limit:
        key = max(grouped, key=lambda item: len(grouped[item]) - quotas.get(item, 0))
        quotas[key] = quotas.get(key, 0) + 1
    selected: List[Dict[str, Any]] = []
    for key, count in quotas.items():
        selected.extend(grouped[key][:count])
    return sorted(selected, key=lambda row: (str(row["depth_bin"]), str(row["episode_id"]), int(row["step_idx"])))


def selector_accuracy(rows: Sequence[Mapping[str, Any]], field: str) -> Optional[float]:
    values = [row.get(field) for row in rows]
    if any(value is None for value in values):
        return None
    return sum(1 for value in values if value) / len(rows) if rows else 0.0


def summarize_by(rows: Sequence[Mapping[str, Any]], key_name: str, correct_field: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key_name))].append(row)
    summary: Dict[str, Dict[str, Any]] = {}
    for key, group_rows in sorted(grouped.items()):
        accuracy = selector_accuracy(group_rows, correct_field)
        summary[key] = {"n": len(group_rows), "accuracy": accuracy}
    return summary


def row_for_sft(step_record: Mapping[str, Any], candidate: Mapping[str, Any]) -> Dict[str, Any]:
    prompt = verifier_prompt(
        str(step_record.get("instruction") or ""),
        step_record.get("history") if isinstance(step_record.get("history"), list) else [],
        candidate.get("action") if isinstance(candidate.get("action"), dict) else {},
        candidate.get("control") if isinstance(candidate.get("control"), dict) else {},
    )
    metadata = {
        "target_id": step_record.get("target_id"),
        "episode_id": step_record.get("episode_id"),
        "step_idx": step_record.get("step_idx"),
        "candidate_id": candidate.get("candidate_id"),
        "candidate_source": candidate.get("source"),
        "subset": step_record.get("subset"),
        "depth_bin": step_record.get("depth_bin"),
        "label": "correct" if candidate.get("is_correct") else "incorrect",
        "leakage_note": "source/label/rank metadata is excluded from prompt text; this metadata is for audit/eval only.",
    }
    return build_sharegpt_example(prompt, str(step_record.get("screenshot") or ""), bool(candidate.get("is_correct")), metadata)


def dataset_entry(file_name: str) -> Dict[str, Any]:
    return {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "images": "images"},
        "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"},
    }


def write_dataset_info(output_dir: Path, entries: Mapping[str, str]) -> None:
    info = {dataset_name: dataset_entry(file_name) for dataset_name, file_name in entries.items()}
    text = json.dumps(info, indent=2, ensure_ascii=False) + "\n"
    (output_dir / "dataset_info.json").write_text(text, encoding="utf-8")
    (output_dir / "dataset_info.snippet.json").write_text(text, encoding="utf-8")


def build_training_sft(
    *,
    train_samples_path: Path,
    train_data_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    train_episodes = read_episodes(train_data_path)
    train_rows_all = [row for row in read_jsonl(train_samples_path) if str(row.get("population")) == args.population]
    train_primary = [row for row in train_rows_all if float(row.get("temperature")) == args.temperature]
    train_recoverable = [row for row in train_primary if row.get("recoverable")]
    examples_by_target: Dict[str, List[Dict[str, Any]]] = {}
    random_source = random.Random(args.seed)
    for sample_row in train_recoverable:
        episode = train_episodes.get(str(sample_row.get("episode_id")))
        if episode is None:
            raise SystemExit(f"missing TRAIN episode {sample_row.get('episode_id')} for {sample_row.get('target_id')}")
        step_record = build_step_record(sample_row, episode, None, None)
        selected_candidates: List[Dict[str, Any]] = []
        greedy_candidates = [candidate for candidate in step_record["candidates"] if candidate["source"] == "greedy"]
        positive_candidates = [candidate for candidate in step_record["candidates"] if candidate["is_correct"]]
        other_negatives = [candidate for candidate in step_record["candidates"] if candidate["source"] == "sample" and not candidate["is_correct"]]
        random_source.shuffle(other_negatives)
        for greedy_candidate in greedy_candidates:
            if not greedy_candidate["is_correct"]:
                selected_candidates.extend(greedy_candidate for _ in range(max(1, args.hard_negative_copies)))
        selected_candidates.extend(positive_candidates)
        selected_candidates.extend(other_negatives[: max(0, args.other_negatives_per_step)])
        examples_by_target[str(step_record["target_id"])] = [row_for_sft(step_record, candidate) for candidate in selected_candidates]

    target_ids = sorted(examples_by_target)
    random_source.shuffle(target_ids)
    val_count = max(1, int(round(len(target_ids) * args.val_fraction))) if target_ids else 0
    val_targets = set(target_ids[:val_count])
    train_examples: List[Dict[str, Any]] = []
    val_examples: List[Dict[str, Any]] = []
    for target_id in target_ids:
        if target_id in val_targets:
            val_examples.extend(examples_by_target[target_id])
        else:
            train_examples.extend(examples_by_target[target_id])
    (output_dir / "train_sft.json").write_text(json.dumps(train_examples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "val_sft.json").write_text(json.dumps(val_examples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "train_samples": str(train_samples_path),
        "train_data": str(train_data_path),
        "train_primary_rows": len(train_primary),
        "train_recoverable_rows": len(train_recoverable),
        "train_targets": len(target_ids) - val_count,
        "val_targets": val_count,
        "train_sft_rows": len(train_examples),
        "val_sft_rows": len(val_examples),
        "hard_negative_copies": args.hard_negative_copies,
        "other_negatives_per_step": args.other_negatives_per_step,
        "val_fraction": args.val_fraction,
    }


def metrics_table(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    first_sample_correct = sum(1 for row in rows if row.get("first_sample_correct"))
    greedy_correct = sum(1 for row in rows if row.get("greedy_correct"))
    oracle_correct = sum(1 for row in rows if row.get("oracle_candidate_id"))
    return {
        "n": total,
        "oracle_accuracy": oracle_correct / total if total else 0.0,
        "greedy_accuracy": greedy_correct / total if total else 0.0,
        "first_sample_accuracy": first_sample_correct / total if total else 0.0,
        "logprob_rank_accuracy": selector_accuracy(rows, "logprob_rank_correct"),
        "verifier_accuracy": selector_accuracy(rows, "verifier_correct"),
    }


def projected_tsr_lift_pp(selection_accuracy: Optional[float], recoverable_fraction: float) -> Optional[float]:
    if selection_accuracy is None:
        return None
    return ORACLE_CRITICAL_TSR_CEILING_PP * recoverable_fraction * selection_accuracy


def optional_pct(value: Optional[float]) -> str:
    if value is None:
        return "not available"
    return f"{100.0 * value:.2f}%"


def optional_pp(value: Optional[float]) -> str:
    if value is None:
        return "not available"
    return f"{value:.2f}pp"


def write_report(
    output_dir: Path,
    slice_rows: Sequence[Mapping[str, Any]],
    all_recoverable_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    training_data = summary.get("training_status", {}).get("train_sft_data") if isinstance(summary.get("training_status"), dict) else None
    if training_data:
        status_line = "**TRAIN SFT DATA READY; VERIFIER TRAINING/EVAL NOT RUN YET.**"
        gate_line = "**NOT EVALUATED: VERIFIER CHECKPOINT MISSING**"
        gate_body = "This is not a negative verifier result. The no-leakage train/val SFT data and 200-step held-out eval slice are ready, but none of `VERIFIER EFFECTIVE`, `VERIFIER SHALLOW-ONLY`, or `VERIFIER INEFFECTIVE` can be assigned before training a verifier and scoring this held-out slice."
    else:
        status_line = "**DATA + 200-STEP EVAL SLICE READY; TRAINING BLOCKED BY MISSING TRAIN-SIDE SAMPLED POOL.**"
        gate_line = "**NOT EVALUATED: TRAIN-SIDE SAMPLED POOL MISSING**"
        gate_body = "This is not a negative verifier result. The no-leakage candidate data and 200-step slice are ready, but none of `VERIFIER EFFECTIVE`, `VERIFIER SHALLOW-ONLY`, or `VERIFIER INEFFECTIVE` can be assigned before training a verifier on TRAIN-side sampled candidates and scoring this held-out slice."
    slice_metrics = metrics_table(slice_rows)
    all_metrics = metrics_table(all_recoverable_rows)
    recoverable_fraction = float(summary["recoverable_fraction_primary"])
    selector_rows = [
        ("oracle_in_pool", slice_metrics["oracle_accuracy"], "Correct sample exists by construction on recoverable eval rows."),
        ("greedy", slice_metrics["greedy_accuracy"], "Base greedy is wrong on this failure pool."),
        ("sample_order_first", slice_metrics["first_sample_accuracy"], "Training-free first sampled candidate baseline; not logprob-rank."),
        ("logprob_rank", slice_metrics["logprob_rank_accuracy"], "Unavailable: sampled pool does not store token logprobs."),
        ("trained_verifier", slice_metrics["verifier_accuracy"], "Not run: no TRAIN-side sampled pool/checkpoint yet."),
    ]
    lines: List[str] = []
    lines.append("# Critical-Step Full-Action Verifier Slice")
    lines.append("")
    lines.append("Diagnostic/data-construction stage only: existing TEST-side UIA sampled pool, frozen matcher labels, no base-model change, and no trained verifier checkpoint yet.")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append(status_line)
    lines.append("")
    if training_data:
        lines.append("The current workspace now has a TRAIN-side sampled candidate pool and train/val verifier SFT data. The TEST-side `outputs/critstep_elicit_uia/per_step.jsonl` pool remains evaluation-only under the leakage ban.")
    else:
        lines.append("The current workspace has the TEST-side `outputs/critstep_elicit_uia/per_step.jsonl` sampled pool and full TRAIN critical-state diagnostics, but not a TRAIN-side sampled candidate pool with N candidates per critical step. To keep the leakage ban intact, this script does not train on the TEST pool.")
    lines.append("")
    lines.append("## Data Construction")
    lines.append("")
    lines.append(f"- recoverable critical eval steps: `{summary['n_recoverable_primary']}`")
    lines.append(f"- eval slice steps: `{slice_metrics['n']}`")
    lines.append(f"- candidates per step: greedy hard negative + 50 sampled candidates")
    lines.append(f"- eval slice candidate SFT rows: `{summary['slice_sft_rows']}`")
    if training_data:
        lines.append(f"- train verifier targets: `{training_data['train_targets']}`")
        lines.append(f"- validation verifier targets: `{training_data['val_targets']}`")
        lines.append(f"- train candidate SFT rows: `{training_data['train_sft_rows']}`")
        lines.append(f"- validation candidate SFT rows: `{training_data['val_sft_rows']}`")
    lines.append(f"- feature policy: instruction + screenshot + history + candidate action + candidate UIA metadata only")
    lines.append(f"- excluded from prompt: matcher verdict, GT action, candidate rank, sample frequency, success count, candidate source")
    lines.append("")
    lines.append("## Scope Map over All Recoverable Critical Steps")
    lines.append("")
    lines.append("| subset | count | share |")
    lines.append("|---|---:|---:|")
    subset_counts = Counter(str(row.get("subset")) for row in all_recoverable_rows)
    for subset, count in subset_counts.most_common():
        lines.append(f"| {subset} | {count} | {pct(count, len(all_recoverable_rows))} |")
    full_action_addressable = subset_counts["click_element_selection"] + subset_counts["action_type_mismatch"] + subset_counts["type_content"]
    lines.append("")
    lines.append(f"Full-action verifier-addressable by candidate selection: `{full_action_addressable} / {len(all_recoverable_rows)}` ({pct(full_action_addressable, len(all_recoverable_rows))}) using current labels: click element-selection + action-type mismatch + type-content.")
    lines.append("")
    lines.append("## 3.1 Selection Accuracy on 200-Step Slice")
    lines.append("")
    lines.append("| selector | accuracy | projected TSR lift proxy | fraction of +23.77pp oracle-critical ceiling | note |")
    lines.append("|---|---:|---:|---:|---|")
    for selector_name, accuracy, note in selector_rows:
        lift = projected_tsr_lift_pp(accuracy, recoverable_fraction)
        fraction = lift / ORACLE_CRITICAL_TSR_CEILING_PP if lift is not None else None
        lines.append(f"| {selector_name} | {optional_pct(accuracy)} | {optional_pp(lift)} | {optional_pct(fraction)} | {note} |")
    lines.append("")
    lines.append("Projection is a linear proxy from selected recoverable critical fixes into the bottom-2 oracle-critical TSR ceiling. The final P5 result should re-plug verifier-measured per-step deltas into the compound product once a trained verifier exists.")
    lines.append("")
    lines.append("## 3.1 Per-Subset First-Sample Baseline")
    lines.append("")
    lines.append("| subset | n | first-sample accuracy | oracle | verifier |")
    lines.append("|---|---:|---:|---:|---:|")
    by_subset = summarize_by(slice_rows, "subset", "first_sample_correct")
    for subset, item in sorted(by_subset.items()):
        subset_rows = [row for row in slice_rows if str(row.get("subset")) == subset]
        oracle_accuracy = selector_accuracy([{**row, "oracle_correct": bool(row.get("oracle_candidate_id"))} for row in subset_rows], "oracle_correct")
        lines.append(f"| {subset} | {item['n']} | {optional_pct(item['accuracy'])} | {optional_pct(oracle_accuracy)} | not available |")
    lines.append("")
    lines.append("## 3.2 Depth-Stratified Recovery")
    lines.append("")
    lines.append("| depth bin | n | first-sample accuracy | oracle | trained verifier |")
    lines.append("|---|---:|---:|---:|---:|")
    by_depth = summarize_by(slice_rows, "depth_bin", "first_sample_correct")
    for depth_name, item in sorted(by_depth.items()):
        depth_rows = [row for row in slice_rows if str(row.get("depth_bin")) == depth_name]
        oracle_accuracy = selector_accuracy([{**row, "oracle_correct": bool(row.get("oracle_candidate_id"))} for row in depth_rows], "oracle_correct")
        lines.append(f"| {depth_name} | {item['n']} | {optional_pct(item['accuracy'])} | {optional_pct(oracle_accuracy)} | not available |")
    lines.append("")
    lines.append("The decisive C.1 question is still pending: a trained verifier must be evaluated here, especially on `deep_21_50`.")
    lines.append("")
    lines.append("## 3.3 Reject-the-Distractor Rate")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| greedy correct on slice | {optional_pct(slice_metrics['greedy_accuracy'])} |")
    lines.append("| oracle rejects greedy when a correct tail candidate exists | 100.00% |")
    lines.append("| trained verifier rejects greedy | not available |")
    lines.append("")
    lines.append("## 3.4 Compound Projection")
    lines.append("")
    lines.append(f"- bottom-2 oracle-critical compound ceiling: `+{ORACLE_CRITICAL_TSR_CEILING_PP:.2f}pp`")
    lines.append(f"- recoverable@50 primary pool: `{summary['n_recoverable_primary']} / {summary['n_primary_failures']}` ({pct(summary['n_recoverable_primary'], summary['n_primary_failures'])})")
    lines.append(f"- oracle selector over recoverable candidates realizes proxy `+{projected_tsr_lift_pp(1.0, recoverable_fraction):.2f}pp`, or `{pct(projected_tsr_lift_pp(1.0, recoverable_fraction) or 0.0, ORACLE_CRITICAL_TSR_CEILING_PP)}` of the +23.77pp ceiling before verifier errors.")
    first_lift = projected_tsr_lift_pp(slice_metrics["first_sample_accuracy"], recoverable_fraction)
    lines.append(f"- first-sample baseline on the 200-step slice realizes proxy `{optional_pp(first_lift)}`, or `{optional_pct((first_lift / ORACLE_CRITICAL_TSR_CEILING_PP) if first_lift is not None else None)}` of the ceiling.")
    lines.append("- trained verifier projection: not available until a no-leakage TRAIN-side verifier is trained and scored.")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(gate_line)
    lines.append("")
    lines.append(gate_body)
    lines.append("")
    lines.append("## Required Next Run")
    lines.append("")
    if training_data:
        lines.append("1. Train the generative verifier LoRA from `train_sft.json` / `val_sft.json`.")
        lines.append("2. Score `per_step.jsonl` candidates with verifier verdict-token scores.")
        lines.append("3. Re-run the eval report with verifier scores populated to assign the final gate.")
    else:
        lines.append("1. Build TRAIN-side UIA candidate pool from TRAIN critical states with the same sampler and frozen matcher.")
        lines.append("2. Run this script with `--train-samples <train_pool.jsonl> --train-data <train_uia.jsonl>` to emit `train_sft.json`/`val_sft.json` for LLaMA-Factory.")
        lines.append("3. Train the generative verifier LoRA and score `per_step.jsonl` candidates with verdict-token logprob.")
        lines.append("4. Re-run the eval report with verifier scores populated to assign the final gate.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{output_dir / 'verifier_eval.md'}`")
    lines.append(f"- `{output_dir / 'per_step.jsonl'}`")
    lines.append(f"- `{output_dir / 'eval_slice_sft.json'}`")
    lines.append(f"- `{output_dir / 'eval_slice_examples.jsonl'}`")
    lines.append(f"- `{output_dir / 'manifest.json'}`")
    lines.append(f"- `{output_dir / 'dataset_info.json'}`")
    lines.append(f"- `{output_dir / 'dataset_info.snippet.json'}`")
    lines.append("")
    lines.append("STOP for review.")
    (output_dir / "verifier_eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default=DEFAULT_SAMPLES)
    parser.add_argument("--test-data", default=DEFAULT_TEST_DATA)
    parser.add_argument("--uia-per-step", default=DEFAULT_UIA_PER_STEP)
    parser.add_argument("--scope-per-step", default=DEFAULT_SCOPE_PER_STEP)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--population", default="critical")
    parser.add_argument("--eval-limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-samples", default="", help="Optional TRAIN-side sampled candidate pool. If supplied, train/val SFT data is emitted.")
    parser.add_argument("--train-data", default="", help="TRAIN-side UIA-enriched episode JSONL matching --train-samples.")
    parser.add_argument("--hard-negative-copies", type=int, default=3)
    parser.add_argument("--other-negatives-per-step", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = read_episodes(Path(args.test_data))
    uia_by_key = load_index(Path(args.uia_per_step))
    scope_by_key = load_scope_index(Path(args.scope_per_step))
    sample_rows_all = [row for row in read_jsonl(Path(args.samples)) if str(row.get("population")) == args.population]
    primary_rows = [row for row in sample_rows_all if float(row.get("temperature")) == args.temperature]
    recoverable_rows = [row for row in primary_rows if row.get("recoverable")]

    step_records: List[Dict[str, Any]] = []
    for sample_row in recoverable_rows:
        row_key = key_for(sample_row)
        episode = episodes.get(str(sample_row.get("episode_id")))
        uia_row = uia_by_key.get(row_key)
        if episode is None or uia_row is None:
            raise SystemExit(f"missing episode/UIA row for {row_key}")
        step_records.append(build_step_record(sample_row, episode, uia_row, scope_by_key.get(row_key)))

    slice_records = stratified_slice(step_records, args.eval_limit, args.seed)
    eval_examples = []
    eval_jsonl_examples = []
    for step_record in slice_records:
        for candidate in step_record["candidates"]:
            example = row_for_sft(step_record, candidate)
            eval_examples.append(example)
            eval_jsonl_examples.append({
                "target_id": step_record["target_id"],
                "candidate_id": candidate["candidate_id"],
                "label": "correct" if candidate["is_correct"] else "incorrect",
                "subset": step_record["subset"],
                "depth_bin": step_record["depth_bin"],
                "messages": example["conversations"],
                "images": example["images"],
                "metadata": example["metadata"],
            })

    write_jsonl(output_dir / "per_step.jsonl", slice_records)
    write_jsonl(output_dir / "eval_slice_examples.jsonl", eval_jsonl_examples)
    (output_dir / "eval_slice_sft.json").write_text(json.dumps(eval_examples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    training_data_status: Optional[Dict[str, Any]] = None
    if args.train_samples:
        if not args.train_data:
            raise SystemExit("--train-data is required when --train-samples is supplied")
        training_data_status = build_training_sft(
            train_samples_path=Path(args.train_samples),
            train_data_path=Path(args.train_data),
            output_dir=output_dir,
            args=args,
        )
    dataset_entries = {"critstep_full_action_verifier_eval_slice": "eval_slice_sft.json"}
    if training_data_status is not None:
        dataset_entries["critstep_full_action_verifier_train"] = "train_sft.json"
        dataset_entries["critstep_full_action_verifier_val"] = "val_sft.json"
    write_dataset_info(output_dir, dataset_entries)

    subset_counts = Counter(str(row.get("subset")) for row in step_records)
    depth_counts = Counter(str(row.get("depth_bin")) for row in step_records)
    slice_subset_counts = Counter(str(row.get("subset")) for row in slice_records)
    slice_depth_counts = Counter(str(row.get("depth_bin")) for row in slice_records)
    summary = {
        "samples": args.samples,
        "test_data": args.test_data,
        "uia_per_step": args.uia_per_step,
        "scope_per_step": args.scope_per_step,
        "output_dir": str(output_dir),
        "temperature": args.temperature,
        "population": args.population,
        "n_primary_failures": len(primary_rows),
        "n_recoverable_primary": len(recoverable_rows),
        "recoverable_fraction_primary": len(recoverable_rows) / len(primary_rows) if primary_rows else 0.0,
        "eval_limit": args.eval_limit,
        "eval_slice_steps": len(slice_records),
        "slice_sft_rows": len(eval_examples),
        "subset_counts_all_recoverable": dict(subset_counts),
        "depth_counts_all_recoverable": dict(depth_counts),
        "subset_counts_eval_slice": dict(slice_subset_counts),
        "depth_counts_eval_slice": dict(slice_depth_counts),
        "selector_metrics_eval_slice": metrics_table(slice_records),
        "selector_metrics_all_recoverable": metrics_table(step_records),
        "leakage_audit": {
            "prompt_features": ["instruction", "screenshot", "history", "candidate_action", "candidate_uia_control_metadata"],
            "excluded_from_prompt": ["matcher_verdict", "gt_action", "candidate_rank", "sample_frequency", "success_count", "candidate_source"],
            "train_on_test_pool": False,
            "status": "pass_for_eval_slice_data_construction",
        },
        "training_status": {
            "trained_verifier_checkpoint": None,
            "training_log": None,
            "train_sft_data": training_data_status,
            "reason": "TRAIN-side sampled candidate pool was not found/provided; TEST-side pool is evaluation-only under leakage ban." if training_data_status is None else "TRAIN-side SFT data emitted; model training has not been launched by this script.",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(output_dir, slice_records, step_records, summary)
    output_gate = "NOT EVALUATED: VERIFIER CHECKPOINT MISSING" if training_data_status is not None else "NOT EVALUATED: TRAIN-SIDE SAMPLED POOL MISSING"
    print(json.dumps({
        "report": str(output_dir / "verifier_eval.md"),
        "per_step": str(output_dir / "per_step.jsonl"),
        "eval_slice_sft": str(output_dir / "eval_slice_sft.json"),
        "eval_slice_examples": str(output_dir / "eval_slice_examples.jsonl"),
        "manifest": str(output_dir / "manifest.json"),
        "eval_slice_steps": len(slice_records),
        "eval_slice_sft_rows": len(eval_examples),
        "train_sft_data": training_data_status,
        "gate": output_gate,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()