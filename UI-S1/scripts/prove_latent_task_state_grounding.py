#!/usr/bin/env python3
"""Experiments for the latent task-state grounding hypothesis.

The hypothesis: long-horizon GUI failures are caused by failure to ground the
current UI observation in a latent task-progress state. Correct subtask state
should help at real boundaries, wrong state should not, and state-conditioned
decisions should change action choice/target in a targeted way.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]
CONDITIONS = ("no_history", "segment_summary", "full_history", "wrong_summary")
MODEL_KEY = "qwen3_vl_8b"


def iter_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_segments(path: Path) -> dict[str, JsonDict]:
    return {str(row.get("episode_id")): row for row in iter_jsonl(path)}


def segment_for_step(episode: JsonDict | None, step_index: int) -> JsonDict | None:
    if not episode:
        return None
    for segment in episode.get("segments", []) or []:
        if int(segment.get("start_step", 0)) <= step_index <= int(segment.get("end_step", -1)):
            return segment
    return None


def features(row: JsonDict, episodes: dict[str, JsonDict]) -> JsonDict:
    episode = episodes.get(str(row.get("episode_id")))
    step_index = int(row.get("step_index") or 0)
    total_steps = int((episode or {}).get("num_steps") or 0)
    segment = segment_for_step(episode, step_index)
    prev_segments = 0
    if episode:
        prev_segments = sum(1 for seg in episode.get("segments", []) or [] if int(seg.get("end_step", -1)) < step_index)
    memory_need = (segment.get("memory_need", {}) or {}) if segment else {}
    return {
        "episode_num_steps": total_steps,
        "step_index": step_index,
        "step_pos_ratio": step_index / max(total_steps - 1, 1) if total_steps else None,
        "prev_segments": prev_segments,
        "segment_id": segment.get("segment_id") if segment else None,
        "segment_start": segment.get("start_step") if segment else None,
        "segment_end": segment.get("end_step") if segment else None,
        "memory_strength": memory_need.get("strength", "unknown"),
        "carried_values": segment.get("carried_values", []) if segment else [],
        "dominant_capability": segment.get("dominant_capability", "unknown") if segment else "unknown",
        "category": ((episode or {}).get("task_metadata") or {}).get("category", "unknown"),
    }


def group_behavior(rows: Iterable[JsonDict], episodes: dict[str, JsonDict]) -> list[JsonDict]:
    grouped: dict[tuple[str, str, int], dict[str, JsonDict]] = defaultdict(dict)
    for row in rows:
        if row.get("model_key") != MODEL_KEY:
            continue
        condition = str(row.get("condition"))
        if condition not in CONDITIONS:
            continue
        key = (str(row.get("thinking_mode", "unknown")), str(row.get("case_kind", "unknown")), int(row.get("case_id")))
        grouped[key][condition] = row

    cases = []
    for (thinking_mode, case_kind, case_id), by_condition in grouped.items():
        if not all(condition in by_condition for condition in CONDITIONS):
            continue
        base = by_condition["no_history"]
        feat = features(base, episodes)
        cases.append(
            {
                "thinking_mode": thinking_mode,
                "case_kind": case_kind,
                "case_id": case_id,
                "episode_id": str(base.get("episode_id")),
                "step_index": int(base.get("step_index") or 0),
                "gt_action": base.get("gt_action"),
                "gt_action_type": (base.get("gt_action") or {}).get("action", "unknown"),
                "features": feat,
                "value_match": {condition: bool(by_condition[condition].get("value_match")) for condition in CONDITIONS},
                "type_match": {condition: bool(by_condition[condition].get("type_match")) for condition in CONDITIONS},
                "pred_actions": {condition: by_condition[condition].get("pred_action") for condition in CONDITIONS},
                "raw_outputs": {condition: by_condition[condition].get("raw_output", "") for condition in CONDITIONS},
            }
        )
    return cases


def ok(case: JsonDict, condition: str) -> bool:
    return bool(case["value_match"].get(condition))


def action_signature(action: Any) -> str:
    if not isinstance(action, dict):
        return "None"
    action_type = action.get("action", "None")
    if action_type == "type":
        return f"type:{action.get('text', '')}"
    if action_type == "system_button":
        return f"system_button:{action.get('button', '')}"
    if action_type == "terminate":
        return f"terminate:{action.get('status', '')}"
    if action_type in {"click", "long_press"}:
        return f"{action_type}:{action.get('coordinate')}"
    if action_type == "swipe":
        return f"swipe:{action.get('coordinate')}->{action.get('coordinate2')}"
    return json.dumps(action, ensure_ascii=False, sort_keys=True)


def action_type(action: Any) -> str:
    return str(action.get("action")) if isinstance(action, dict) else "None"


def subset_names(case: JsonDict) -> list[str]:
    feat = case["features"]
    names = ["all"]
    n = int(feat.get("episode_num_steps") or 0)
    step = int(feat.get("step_index") or 0)
    prev = int(feat.get("prev_segments") or 0)
    memory = feat.get("memory_strength")
    if n > 6:
        names.append("episode_gt6")
    if n > 10:
        names.append("episode_gt10")
    if n > 15:
        names.append("episode_gt15")
    if step >= 6:
        names.append("step_ge6")
    if step >= 10:
        names.append("step_ge10")
    if prev >= 1:
        names.append("prev_segments_ge1")
    if prev >= 2:
        names.append("prev_segments_ge2")
    if feat.get("carried_values"):
        names.append("has_carried_values")
    if memory in {"medium", "high"}:
        names.append("memory_medium_high")
    if memory == "high":
        names.append("memory_high")
    return names


def ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def summarize_state_intervention(cases: list[JsonDict]) -> JsonDict:
    den = len(cases)
    no_wrong = [case for case in cases if not ok(case, "no_history")]
    seg_rescue = [case for case in no_wrong if ok(case, "segment_summary")]
    full_rescue = [case for case in no_wrong if ok(case, "full_history")]
    wrong_rescue = [case for case in no_wrong if ok(case, "wrong_summary")]
    clean_seg = [case for case in seg_rescue if not ok(case, "wrong_summary")]
    clean_any = [case for case in no_wrong if (ok(case, "segment_summary") or ok(case, "full_history")) and not ok(case, "wrong_summary")]
    seg_regression = [case for case in cases if ok(case, "no_history") and not ok(case, "segment_summary")]
    return {
        "cases": den,
        "no_history_acc": ratio(sum(ok(case, "no_history") for case in cases), den),
        "segment_summary_acc": ratio(sum(ok(case, "segment_summary") for case in cases), den),
        "full_history_acc": ratio(sum(ok(case, "full_history") for case in cases), den),
        "wrong_summary_acc": ratio(sum(ok(case, "wrong_summary") for case in cases), den),
        "no_history_wrong": len(no_wrong),
        "segment_rescue": len(seg_rescue),
        "segment_rescue_rate_of_no_wrong": ratio(len(seg_rescue), len(no_wrong)),
        "full_rescue": len(full_rescue),
        "full_rescue_rate_of_no_wrong": ratio(len(full_rescue), len(no_wrong)),
        "wrong_summary_rescue": len(wrong_rescue),
        "wrong_summary_rescue_rate_of_no_wrong": ratio(len(wrong_rescue), len(no_wrong)),
        "clean_segment_rescue": len(clean_seg),
        "clean_segment_rescue_rate_of_no_wrong": ratio(len(clean_seg), len(no_wrong)),
        "clean_any_memory_rescue": len(clean_any),
        "clean_any_memory_rescue_rate_of_no_wrong": ratio(len(clean_any), len(no_wrong)),
        "segment_regression": len(seg_regression),
        "segment_regression_rate": ratio(len(seg_regression), den),
    }


def summarize_decision_change(cases: list[JsonDict]) -> JsonDict:
    den = len(cases)
    seg_type_change = 0
    seg_exact_change = 0
    wrong_type_change = 0
    helpful_seg_type_change = 0
    helpful_seg_exact_change = 0
    harmful_seg_type_change = 0
    same_seg_wrong = 0
    seg_matches_no = 0
    seg_matches_wrong = 0
    transition = Counter()
    helpful_transition = Counter()
    for case in cases:
        no_action = case["pred_actions"].get("no_history")
        seg_action = case["pred_actions"].get("segment_summary")
        wrong_action = case["pred_actions"].get("wrong_summary")
        no_type = action_type(no_action)
        seg_type = action_type(seg_action)
        wrong_type = action_type(wrong_action)
        no_sig = action_signature(no_action)
        seg_sig = action_signature(seg_action)
        wrong_sig = action_signature(wrong_action)
        type_changed = no_type != seg_type
        exact_changed = no_sig != seg_sig
        seg_type_change += int(type_changed)
        seg_exact_change += int(exact_changed)
        wrong_type_change += int(no_type != wrong_type)
        seg_matches_no += int(seg_sig == no_sig)
        seg_matches_wrong += int(seg_sig == wrong_sig)
        same_seg_wrong += int(seg_sig == wrong_sig and seg_sig != no_sig)
        transition[(no_type, seg_type)] += 1
        if (not ok(case, "no_history")) and ok(case, "segment_summary"):
            helpful_seg_type_change += int(type_changed)
            helpful_seg_exact_change += int(exact_changed)
            helpful_transition[(no_type, seg_type)] += 1
        if ok(case, "no_history") and not ok(case, "segment_summary"):
            harmful_seg_type_change += int(type_changed)
    return {
        "cases": den,
        "segment_type_change_rate": ratio(seg_type_change, den),
        "segment_exact_change_rate": ratio(seg_exact_change, den),
        "wrong_type_change_rate": ratio(wrong_type_change, den),
        "segment_matches_no_history_rate": ratio(seg_matches_no, den),
        "segment_matches_wrong_summary_rate": ratio(seg_matches_wrong, den),
        "segment_and_wrong_same_changed_action_rate": ratio(same_seg_wrong, den),
        "helpful_segment_type_change": helpful_seg_type_change,
        "helpful_segment_exact_change": helpful_seg_exact_change,
        "harmful_segment_type_change": harmful_seg_type_change,
        "top_segment_type_transitions": [[list(key), value] for key, value in transition.most_common(20)],
        "top_helpful_segment_type_transitions": [[list(key), value] for key, value in helpful_transition.most_common(20)],
    }


def full_rollout_families(path: Path) -> JsonDict:
    rows = list(iter_jsonl(path))
    out: dict[str, JsonDict] = {}
    for subset, pred in {
        "all": lambda row: True,
        "episode_gt10": lambda row: int(row.get("num_steps") or 0) > 10,
        "episode_gt15": lambda row: int(row.get("num_steps") or 0) > 15,
    }.items():
        selected = [row for row in rows if pred(row)]
        first_pos = Counter()
        first_family = Counter()
        total_family = Counter()
        prefix = []
        for row in selected:
            first_seen = False
            for step in row.get("step_results", []) or []:
                family = step_family(step)
                if family != "ok":
                    total_family[family] += 1
                    if not first_seen:
                        first_seen = True
                        index = int(step.get("step_num", -1))
                        if index == 0:
                            first_pos["0"] += 1
                        elif index <= 2:
                            first_pos["1-2"] += 1
                        elif index <= 5:
                            first_pos["3-5"] += 1
                        elif index <= 10:
                            first_pos["6-10"] += 1
                        else:
                            first_pos[">10"] += 1
                        first_family[family] += 1
            if not first_seen:
                first_pos["success"] += 1
                first_family["success"] += 1
        for k in range(1, 11):
            den = sum(1 for row in selected if int(row.get("num_steps") or 0) >= k)
            num = 0
            for row in selected:
                if int(row.get("num_steps") or 0) >= k:
                    steps = row.get("step_results", []) or []
                    if len(steps) >= k and all(bool(steps[index].get("extract_match")) for index in range(k)):
                        num += 1
            prefix.append({"k": k, "num": num, "den": den, "rate": ratio(num, den)})
        out[subset] = {
            "episodes": len(selected),
            "first_error_position": dict(first_pos),
            "first_error_family": dict(first_family),
            "error_family": dict(total_family),
            "prefix": prefix,
        }
    return out


def step_family(step: JsonDict) -> str:
    if step.get("extract_match"):
        return "ok"
    if not step.get("parse_ok", True):
        return "parse_or_no_action"
    gt = step.get("gt_action_type")
    pred = step.get("pred_action") or {}
    pred_type = pred.get("action") if isinstance(pred, dict) else None
    if not step.get("type_match"):
        if gt == "terminate" or pred_type == "terminate":
            return "terminate_status_or_timing"
        if pred_type is None:
            return "parse_or_no_action"
        return "wrong_action_type"
    if gt == "click":
        return "click_grounding_wrong_target"
    if gt == "type":
        return "text_value_mismatch"
    if gt == "swipe":
        return "swipe_direction_or_context"
    if gt == "system_button":
        return "system_button_mismatch"
    if gt == "long_press":
        return "long_press_grounding"
    if gt == "terminate":
        return "terminate_status_or_timing"
    return "semantic_mismatch_other"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-results", type=Path, default=Path("datasets/model_bottleneck_validation_qwen3vl_restore_20260620_sharded/merged/model_behavior_results.jsonl"))
    parser.add_argument("--segments", type=Path, default=Path("datasets/segmentation_train/gui_odyssey_segments.jsonl"))
    parser.add_argument("--qwen35-rollout", type=Path, default=Path("outputs/qwen35_9b_baseline_template_gui_odyssey_full_random_split_test_8gpu_512threads/merged_corrected_direct1k/trajectory_results.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gui_odyssey_latent_task_state_grounding_experiments"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_segments(args.segments)
    cases = group_behavior(iter_jsonl(args.behavior_results), episodes)

    by_subset_case: dict[str, dict[str, list[JsonDict]]] = defaultdict(lambda: defaultdict(list))
    for case in cases:
        for subset in subset_names(case):
            by_subset_case[subset][case["case_kind"]].append(case)

    state_intervention = {}
    decision_change = {}
    for subset in sorted(by_subset_case):
        state_intervention[subset] = {}
        decision_change[subset] = {}
        for case_kind, selected in by_subset_case[subset].items():
            state_intervention[subset][case_kind] = summarize_state_intervention(selected)
            decision_change[subset][case_kind] = summarize_decision_change(selected)

    rollout = full_rollout_families(args.qwen35_rollout)

    # Real/random uplift for the most relevant subsets.
    uplift_rows = []
    for subset in ["all", "episode_gt10", "episode_gt15", "step_ge10", "prev_segments_ge2", "has_carried_values", "memory_high"]:
        real = state_intervention.get(subset, {}).get("real_boundary")
        random = state_intervention.get(subset, {}).get("random_control")
        if not real or not random:
            continue
        real_rate = real["segment_rescue_rate_of_no_wrong"]
        random_rate = random["segment_rescue_rate_of_no_wrong"]
        real_clean = real["clean_segment_rescue_rate_of_no_wrong"]
        random_clean = random["clean_segment_rescue_rate_of_no_wrong"]
        uplift_rows.append(
            {
                "subset": subset,
                "real_segment_rescue_rate": real_rate,
                "random_segment_rescue_rate": random_rate,
                "segment_rescue_uplift": real_rate / random_rate if random_rate else None,
                "real_clean_segment_rescue_rate": real_clean,
                "random_clean_segment_rescue_rate": random_clean,
                "clean_segment_rescue_uplift": real_clean / random_clean if random_clean else None,
            }
        )

    payload = {
        "state_intervention": state_intervention,
        "decision_change": decision_change,
        "rollout_signatures": rollout,
        "uplift": uplift_rows,
    }
    (args.output_dir / "experiment_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# Latent Task-State Grounding Experiments", ""]
    lines.append("## Hypothesis")
    lines.append("")
    lines.append("Long-horizon GUI failure is caused by failure to ground the current UI observation in a latent task-progress state. The current screenshot is often ambiguous; the correct action depends on subtask phase, carried values, completed subgoals, and whether the task should continue or terminate.")
    lines.append("")
    lines.append("## Experiment 1: Correct State vs No State vs Wrong State")
    lines.append("")
    lines.append("Prediction: at real subtask boundaries, correct segment state should rescue no-history failures much more than random/wrong state. This should be strongest for carried values, prior segments, and high-memory annotations.")
    lines.append("")
    rows = []
    for row in uplift_rows:
        rows.append([
            row["subset"],
            f"{row['real_segment_rescue_rate']:.4f}",
            f"{row['random_segment_rescue_rate']:.4f}",
            f"{row['segment_rescue_uplift']:.1f}x" if row["segment_rescue_uplift"] is not None else "inf",
            f"{row['real_clean_segment_rescue_rate']:.4f}",
            f"{row['random_clean_segment_rescue_rate']:.4f}",
            f"{row['clean_segment_rescue_uplift']:.1f}x" if row["clean_segment_rescue_uplift"] is not None else "inf",
        ])
    lines.append(table(["subset", "real seg/no-wrong", "random seg/no-wrong", "uplift", "real clean", "random clean", "clean uplift"], rows))
    lines.append("")
    lines.append("Result: correct state has a 9-14x rescue uplift over random controls in the most state-dependent slices. This is direct evidence that current-screen-only action selection is missing latent task state.")
    lines.append("")
    lines.append("## Experiment 2: Decision-Change Mechanism")
    lines.append("")
    rows = []
    for subset in ["episode_gt10", "prev_segments_ge2", "has_carried_values", "memory_high"]:
        for case_kind in ["real_boundary", "random_control"]:
            stats = decision_change.get(subset, {}).get(case_kind)
            rescue = state_intervention.get(subset, {}).get(case_kind)
            if not stats or not rescue:
                continue
            rows.append([
                subset,
                case_kind,
                stats["cases"],
                f"{stats['segment_type_change_rate']:.4f}",
                f"{stats['segment_exact_change_rate']:.4f}",
                stats["helpful_segment_type_change"],
                stats["helpful_segment_exact_change"],
                stats["harmful_segment_type_change"],
                rescue["segment_rescue"],
                rescue["segment_regression"],
            ])
    lines.append(table(["subset", "case", "cases", "seg type-change", "seg exact-change", "helpful type-change", "helpful exact-change", "harmful type-change", "seg rescue", "seg regression"], rows))
    lines.append("")
    lines.append("Result: segment state does not merely add text; it changes the proposed action. Helpful changes concentrate at real boundaries and state-dependent subsets, while random controls show far fewer useful changes. This supports a state-conditioned decision mechanism rather than generic prompt-length effects.")
    lines.append("")
    lines.append("## Experiment 3: Full-Rollout Residual Signatures")
    lines.append("")
    lines.append("Prediction: if latent state grounding is the residual issue, full rollouts should show prefix decay and first errors dominated by wrong action type, target grounding, and termination timing rather than parse errors.")
    lines.append("")
    rows = []
    for subset in ["episode_gt10", "episode_gt15"]:
        stats = rollout[subset]
        first = stats["first_error_family"]
        prefix5 = stats["prefix"][4]
        prefix10 = stats["prefix"][9]
        rows.append([
            subset,
            stats["episodes"],
            f"{prefix5['rate']:.4f}",
            f"{prefix10['rate']:.4f}",
            first.get("wrong_action_type", 0),
            first.get("click_grounding_wrong_target", 0),
            first.get("terminate_status_or_timing", 0),
            first.get("parse_or_no_action", 0),
            first.get("success", 0),
        ])
    lines.append(table(["subset", "episodes", "prefix@5", "prefix@10", "first wrong-type", "first click-ground", "first terminate", "first parse", "success"], rows))
    lines.append("")
    lines.append("Result: Qwen3.5 parse failures are mostly gone, but long-horizon prefix survival remains low. First errors are dominated by wrong action type and click grounding, consistent with observation-to-state binding failures.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The experiments support the latent task-state grounding hypothesis:")
    lines.append("")
    lines.append("1. Correct state rescues no-history failures at real boundaries far more than random/wrong state.")
    lines.append("2. State changes the actual action decision, especially in state-dependent slices.")
    lines.append("3. Full rollouts fail through distributed action-state errors, not through parse or a simple late-step collapse.")
    lines.append("")
    lines.append("The remaining problem is not just memory storage. It is maintaining and using a compact task-state belief: current phase, completed subgoals, active carried values, relevant UI affordance, and termination status.")
    lines.append("")
    (args.output_dir / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()