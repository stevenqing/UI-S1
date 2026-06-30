#!/usr/bin/env python3
"""Prove or falsify the compound independent-step structure on GUI-360.

This is a pure diagnostic over frozen per-step eval outputs. It joins per-step
correctness with the reconstructed balanced test JSONL, estimates non-uniform
per-step accuracies p_i, and tests whether task success behaves like the
independent product of those p_i values.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_TEST_DATA = "outputs/gui360_history_ab/original_eval/gui360_test_1000_balanced_reconstructed.jsonl"
DEFAULT_RESULTS = [
    "outputs/gui360_history_ab/original_template_ckpt39_gpuutil_part0_20260630/eval_results_20260630_131128.json",
    "outputs/gui360_history_ab/original_template_ckpt39_gpuutil_part1_20260630/eval_results_20260630_130752.json",
]
DEFAULT_STEP_ROWS = None
DEFAULT_OUTPUT_DIR = "outputs/compound_proof"

ALPHA = 0.5
EPS = 1e-12


@dataclass
class StepRecord:
    episode_id: str
    k: int
    step_idx: int
    success: bool
    reward: float
    action_type: str
    bbox_area_frac: Optional[float]
    bbox_area_bin: str
    bbox_aspect_bin: str
    position_bin: str
    step_phase: str
    k_bin: str
    label_detail: str
    has_bbox: bool
    features: Dict[str, Any]
    p_binned: float = 0.0
    p_binned_source: str = ""
    p_heldout: float = 0.0
    p_heldout_source: str = ""

    def key(self, level: str) -> Tuple[Any, ...]:
        if level == "fine":
            return (
                self.action_type,
                self.bbox_area_bin,
                self.bbox_aspect_bin,
                self.position_bin,
                self.step_phase,
                self.k_bin,
                self.label_detail,
            )
        if level == "mid":
            return (
                self.action_type,
                self.bbox_area_bin,
                self.position_bin,
                self.step_phase,
                self.label_detail,
            )
        if level == "coarse":
            return (self.action_type, self.bbox_area_bin, self.step_phase, self.label_detail)
        if level == "action_bbox":
            return (self.action_type, self.bbox_area_bin, self.label_detail)
        if level == "action":
            return (self.action_type, self.label_detail)
        return ("global",)


@dataclass
class TaskRecord:
    episode_id: str
    k: int
    goal: str
    steps: List[StepRecord] = field(default_factory=list)

    @property
    def actual_success(self) -> bool:
        return bool(self.steps) and all(step.success for step in self.steps)

    @property
    def actual_progress(self) -> float:
        for index, step in enumerate(self.steps):
            if not step.success:
                return index / max(1, self.k)
        return 1.0


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _action_type(action: Dict[str, Any], step_result: Dict[str, Any]) -> str:
    value = action.get("action") or step_result.get("gt_type") or "unknown"
    return str(value).strip().lower() or "unknown"


def _bbox_features(step: Dict[str, Any]) -> Tuple[Optional[float], str, str, str, bool]:
    width = float(step.get("image_w") or 1040)
    height = float(step.get("image_h") or 736)
    bbox = step.get("bbox")
    action = step.get("action") if isinstance(step.get("action"), dict) else {}
    center_x: Optional[float] = None
    center_y: Optional[float] = None
    area_frac: Optional[float] = None
    aspect_bin = "missing"
    has_bbox = False

    if isinstance(bbox, list) and len(bbox) >= 4:
        x1, y1, x2, y2 = [_safe_float(v) for v in bbox[:4]]
        if None not in (x1, y1, x2, y2):
            bw = max(0.0, float(x2) - float(x1))
            bh = max(0.0, float(y2) - float(y1))
            area_frac = (bw * bh) / max(1.0, width * height)
            center_x = (float(x1) + float(x2)) / 2.0 / width
            center_y = (float(y1) + float(y2)) / 2.0 / height
            has_bbox = True
            if bw <= 0 or bh <= 0:
                aspect_bin = "degenerate"
            else:
                ratio = bw / bh
                if ratio >= 3.0:
                    aspect_bin = "wide"
                elif ratio <= 1.0 / 3.0:
                    aspect_bin = "tall"
                else:
                    aspect_bin = "normal"

    if center_x is None or center_y is None:
        coord = action.get("coordinate") or action.get("xy")
        if isinstance(coord, list) and len(coord) >= 2:
            x = _safe_float(coord[0])
            y = _safe_float(coord[1])
            if x is not None and y is not None:
                center_x = x / width
                center_y = y / height

    if area_frac is None:
        area_bin = "missing"
    elif area_frac < 0.001:
        area_bin = "tiny"
    elif area_frac < 0.005:
        area_bin = "small"
    elif area_frac < 0.02:
        area_bin = "medium"
    else:
        area_bin = "large"

    if center_x is None or center_y is None:
        pos_bin = "missing"
    else:
        col = min(2, max(0, int(center_x * 3)))
        row = min(2, max(0, int(center_y * 3)))
        pos_bin = f"r{row}c{col}"

    return area_frac, area_bin, aspect_bin, pos_bin, has_bbox


def _step_phase(step_idx: int, k: int) -> str:
    if k <= 1:
        return "single"
    if step_idx == 0:
        return "first"
    if step_idx == k - 1:
        return "last"
    frac = (step_idx + 1) / k
    if frac <= 0.33:
        return "early"
    if frac <= 0.67:
        return "middle"
    return "late"


def _k_bin(k: int) -> str:
    return str(k) if k <= 7 else "8+"


def _label_detail(action_type: str, action: Dict[str, Any]) -> str:
    coord = action.get("coordinate") or action.get("xy")
    has_coord = isinstance(coord, list) and len(coord) >= 2 and coord[0] is not None and coord[1] is not None
    if action_type == "type":
        text = action.get("text") or action.get("keys") or action.get("value")
        return "type_text_present" if text else "type_text_missing"
    if action_type in {"swipe", "drag"}:
        end = action.get("endCoordinate") or action.get("end_coordinate")
        has_end = isinstance(end, list) and len(end) >= 2 and end[0] is not None and end[1] is not None
        if has_coord and has_end:
            return "drag_full_coords"
        return "drag_partial_or_missing"
    if action_type == "click":
        return "click_coord_present" if has_coord else "click_coord_missing"
    return f"{action_type}_other"


def read_test_episodes(path: Path) -> Dict[str, Dict[str, Any]]:
    episodes: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            episode = json.loads(line)
            episodes[str(episode["episode_id"])] = episode
    return episodes


def read_eval_results(paths: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            episode_id = str(value.get("episode_id", key))
            if episode_id in results:
                raise ValueError(f"duplicate episode_id {episode_id} across eval result files")
            results[episode_id] = value
    return results


def _parse_function_from_tool_call(text: str) -> Optional[str]:
    marker = '"function"'
    if marker not in text:
        return None
    try:
        import re

        match = re.search(r'"function"\s*:\s*"([^"]*)"', text)
        if match:
            return match.group(1).strip().lower() or None
    except Exception:
        return None
    return None


def read_step_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            episode_id = str(row.get("global_example_id", row.get("example_id")))
            grouped[episode_id].append(row)

    results: Dict[str, Dict[str, Any]] = {}
    for episode_id, rows in grouped.items():
        rows.sort(key=lambda item: int(item.get("turn_index", item.get("step_idx", 0))))
        steps = []
        correct_steps = 0
        first_error_step: Optional[int] = None
        for index, row in enumerate(rows):
            success = bool(row.get("step_correct", row.get("correct")))
            correct_steps += 1 if success else 0
            if not success and first_error_step is None:
                first_error_step = index + 1
            steps.append(
                {
                    "step_idx": index,
                    "success": success,
                    "reward": 1.0 if success else 0.0,
                    "gt_type": _parse_function_from_tool_call(str(row.get("target_text") or "")),
                    "pred_type": _parse_function_from_tool_call(str(row.get("pred_text") or "")),
                }
            )
        results[episode_id] = {
            "episode_id": int(episode_id) if episode_id.isdigit() else episode_id,
            "steps": steps,
            "num_steps": len(steps),
            "correct_steps": correct_steps,
            "task_success": bool(steps) and correct_steps == len(steps),
            "progress": (first_error_step - 1) / len(steps) if first_error_step and steps else 1.0,
            "first_error_step": first_error_step,
        }
    return results


def build_records(test_episodes: Dict[str, Dict[str, Any]], eval_results: Dict[str, Dict[str, Any]]) -> List[TaskRecord]:
    tasks: List[TaskRecord] = []
    missing_eval = sorted(set(test_episodes) - set(eval_results), key=lambda x: int(x) if x.isdigit() else x)
    if missing_eval:
        raise ValueError(f"missing eval results for {len(missing_eval)} episodes; first={missing_eval[:5]}")

    for episode_id, episode in sorted(test_episodes.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0]):
        result = eval_results[episode_id]
        test_steps = episode.get("steps") or []
        eval_steps = result.get("steps") or []
        if len(test_steps) != len(eval_steps):
            raise ValueError(f"episode {episode_id} step mismatch: test={len(test_steps)} eval={len(eval_steps)}")
        k = len(test_steps)
        task = TaskRecord(episode_id=episode_id, k=k, goal=str(episode.get("goal") or ""))
        for index, (step, step_result) in enumerate(zip(test_steps, eval_steps)):
            action = step.get("action") if isinstance(step.get("action"), dict) else {}
            action_type = _action_type(action, step_result)
            area_frac, area_bin, aspect_bin, pos_bin, has_bbox = _bbox_features(step)
            label_detail = _label_detail(action_type, action)
            record = StepRecord(
                episode_id=episode_id,
                k=k,
                step_idx=index,
                success=bool(step_result.get("success")),
                reward=float(step_result.get("reward") or 0.0),
                action_type=action_type,
                bbox_area_frac=area_frac,
                bbox_area_bin=area_bin,
                bbox_aspect_bin=aspect_bin,
                position_bin=pos_bin,
                step_phase=_step_phase(index, k),
                k_bin=_k_bin(k),
                label_detail=label_detail,
                has_bbox=has_bbox,
                features={
                    "action_type": action_type,
                    "bbox_area_frac": area_frac,
                    "bbox_area_bin": area_bin,
                    "bbox_aspect_bin": aspect_bin,
                    "position_bin": pos_bin,
                    "step_phase": _step_phase(index, k),
                    "k_bin": _k_bin(k),
                    "label_detail": label_detail,
                    "has_bbox": has_bbox,
                    "similar_element_density": None,
                },
            )
            task.steps.append(record)
        tasks.append(task)
    return tasks


def flatten_steps(tasks: Sequence[TaskRecord]) -> List[StepRecord]:
    return [step for task in tasks for step in task.steps]


def _build_stats(records: Iterable[StepRecord], levels: Sequence[str]) -> Dict[str, Dict[Tuple[Any, ...], List[int]]]:
    stats: Dict[str, Dict[Tuple[Any, ...], List[int]]] = {level: defaultdict(lambda: [0, 0]) for level in levels}
    for record in records:
        y = 1 if record.success else 0
        for level in levels:
            bucket = stats[level][record.key(level)]
            bucket[0] += 1
            bucket[1] += y
    return stats


def _smooth(correct: int, count: int, alpha: float) -> float:
    return (correct + alpha) / (count + 2.0 * alpha)


def estimate_leave_one_out(records: Sequence[StepRecord], *, min_bucket: int, alpha: float) -> Counter:
    levels = ("fine", "mid", "coarse", "action_bbox", "action", "global")
    stats = _build_stats(records, levels)
    source_counts: Counter = Counter()
    for record in records:
        y = 1 if record.success else 0
        for level in levels:
            count, correct = stats[level][record.key(level)]
            count -= 1
            correct -= y
            if count >= min_bucket or level == "global":
                record.p_binned = _smooth(correct, max(count, 0), alpha)
                record.p_binned_source = level
                source_counts[level] += 1
                break
    return source_counts


def estimate_heldout_cv(records: Sequence[StepRecord], *, folds: int, min_bucket: int, alpha: float) -> Counter:
    levels = ("fine", "mid", "coarse", "action_bbox", "action", "global")
    fold_records: Dict[int, List[StepRecord]] = defaultdict(list)
    for record in records:
        fold = int(record.episode_id) % folds if record.episode_id.isdigit() else hash(record.episode_id) % folds
        fold_records[fold].append(record)

    source_counts: Counter = Counter()
    for fold in range(folds):
        train = [record for other_fold, values in fold_records.items() if other_fold != fold for record in values]
        stats = _build_stats(train, levels)
        for record in fold_records.get(fold, []):
            for level in levels:
                count, correct = stats[level].get(record.key(level), [0, 0])
                if count >= min_bucket or level == "global":
                    record.p_heldout = _smooth(correct, count, alpha)
                    record.p_heldout_source = level
                    source_counts[level] += 1
                    break
    return source_counts


def product(values: Iterable[float]) -> float:
    out = 1.0
    for value in values:
        out *= max(EPS, min(1.0 - EPS, value))
    return out


def task_prob(task: TaskRecord, attr: str) -> float:
    return product(getattr(step, attr) for step in task.steps)


def neg_log_shares(values: Sequence[float]) -> Tuple[float, float]:
    weights = [-math.log(max(EPS, min(1.0 - EPS, value))) for value in values]
    total = sum(weights)
    if total <= 0:
        return 0.0, 0.0
    ordered = sorted(weights, reverse=True)
    bottom1 = ordered[0] / total if ordered else 0.0
    bottom2 = sum(ordered[:2]) / total if ordered else 0.0
    return bottom1, bottom2


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def ols_fit(points: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    if len(points) < 2:
        return {"slope": 0.0, "intercept": 0.0, "r2": 0.0, "n": len(points)}
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    x_bar = mean(xs)
    y_bar = mean(ys)
    ss_x = sum((x - x_bar) ** 2 for x in xs)
    if ss_x <= 0:
        return {"slope": 0.0, "intercept": y_bar, "r2": 0.0, "n": len(points)}
    slope = sum((x - x_bar) * (y - y_bar) for x, y in points) / ss_x
    intercept = y_bar - slope * x_bar
    ss_tot = sum((y - y_bar) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in points)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"slope": slope, "intercept": intercept, "r2": r2, "n": len(points)}


def layer1(tasks: Sequence[TaskRecord], step_sr: float) -> Dict[str, Any]:
    actual = mean([1.0 if task.actual_success else 0.0 for task in tasks])
    predicted_binned = mean([task_prob(task, "p_binned") for task in tasks])
    predicted_heldout = mean([task_prob(task, "p_heldout") for task in tasks])
    uniform_wrong = mean([step_sr ** task.k for task in tasks])
    mean_k = mean([task.k for task in tasks])
    return {
        "actual_tsr": actual,
        "predicted_tsr_binned_leave_one_out": predicted_binned,
        "predicted_tsr_heldout_cv": predicted_heldout,
        "uniform_mean_p_power_wrong_by_task_k": uniform_wrong,
        "uniform_mean_p_power_wrong_at_mean_k": step_sr ** mean_k if mean_k else 0.0,
        "mean_step_accuracy": step_sr,
        "mean_k": mean_k,
        "abs_error_heldout": abs(predicted_heldout - actual),
        "abs_error_binned": abs(predicted_binned - actual),
        "uniform_abs_error": abs(uniform_wrong - actual),
    }


def _pooled_k_groups(tasks: Sequence[TaskRecord], min_group_n: int) -> List[Tuple[str, List[TaskRecord]]]:
    by_k: Dict[int, List[TaskRecord]] = defaultdict(list)
    for task in tasks:
        by_k[task.k].append(task)
    groups: List[Tuple[str, List[TaskRecord]]] = []
    pending: List[TaskRecord] = []
    pending_ks: List[int] = []
    for k in sorted(by_k):
        pending.extend(by_k[k])
        pending_ks.append(k)
        if len(pending) >= min_group_n:
            label = str(pending_ks[0]) if len(pending_ks) == 1 else f"{pending_ks[0]}-{pending_ks[-1]}"
            groups.append((label, pending))
            pending = []
            pending_ks = []
    if pending:
        if groups:
            prev_label, prev_tasks = groups.pop()
            all_tasks = prev_tasks + pending
            first = min(task.k for task in all_tasks)
            last = max(task.k for task in all_tasks)
            label = str(first) if first == last else f"{first}+"
            groups.append((label, all_tasks))
        else:
            first = min(pending_ks)
            last = max(pending_ks)
            label = str(first) if first == last else f"{first}+"
            groups.append((label, pending))
    return groups


def _k_table(groups: Sequence[Tuple[str, List[TaskRecord]]], step_sr: float) -> List[Dict[str, Any]]:
    rows = []
    for label, values in groups:
        n = len(values)
        successes = sum(1 for task in values if task.actual_success)
        actual = successes / n if n else 0.0
        actual_smoothed = (successes + ALPHA) / (n + 2 * ALPHA) if n else 0.0
        pred_heldout = mean([task_prob(task, "p_heldout") for task in values])
        pred_binned = mean([task_prob(task, "p_binned") for task in values])
        uniform_wrong = mean([step_sr ** task.k for task in values])
        mean_k = mean([task.k for task in values])
        mean_log_p = mean([math.log(max(EPS, step.p_heldout)) for task in values for step in task.steps])
        se = math.sqrt(max(EPS, pred_heldout * (1.0 - pred_heldout) / max(1, n)))
        z_actual_minus_predicted = (actual - pred_heldout) / se if se > 0 else 0.0
        rows.append(
            {
                "k_group": label,
                "mean_k": mean_k,
                "n": n,
                "successes": successes,
                "actual_tsr": actual,
                "actual_tsr_smoothed": actual_smoothed,
                "predicted_tsr_heldout_cv": pred_heldout,
                "predicted_tsr_binned": pred_binned,
                "uniform_wrong": uniform_wrong,
                "log_actual_tsr_smoothed": math.log(max(EPS, actual_smoothed)),
                "log_predicted_tsr_heldout_cv": math.log(max(EPS, pred_heldout)),
                "mean_log_p_i": mean_log_p,
                "actual_minus_predicted": actual - pred_heldout,
                "binomial_z_actual_minus_predicted": z_actual_minus_predicted,
            }
        )
    return rows


def layer2(tasks: Sequence[TaskRecord], step_sr: float, min_group_n: int) -> Dict[str, Any]:
    raw_labels: Dict[str, List[TaskRecord]] = defaultdict(list)
    for task in tasks:
        raw_labels[_k_bin(task.k)].append(task)
    raw_order = sorted(raw_labels, key=lambda label: int(label.rstrip("+")) if label.rstrip("+").isdigit() else 999)
    raw_groups = [(label, raw_labels[label]) for label in raw_order]
    pooled_groups = _pooled_k_groups(tasks, min_group_n)
    raw_rows = _k_table(raw_groups, step_sr)
    pooled_rows = _k_table(pooled_groups, step_sr)
    actual_fit = ols_fit([(row["mean_k"], row["log_actual_tsr_smoothed"]) for row in pooled_rows])
    predicted_fit = ols_fit([(row["mean_k"], row["log_predicted_tsr_heldout_cv"]) for row in pooled_rows])
    max_negative_gap = min((row["actual_minus_predicted"] for row in pooled_rows), default=0.0)
    min_negative_z = min((row["binomial_z_actual_minus_predicted"] for row in pooled_rows), default=0.0)
    return {
        "raw_by_k": raw_rows,
        "pooled_by_k": pooled_rows,
        "actual_log_tsr_slope": actual_fit,
        "predicted_log_tsr_slope": predicted_fit,
        "mean_log_p_i": mean([math.log(max(EPS, step.p_heldout)) for task in tasks for step in task.steps]),
        "slope_gap_actual_minus_predicted": actual_fit["slope"] - predicted_fit["slope"],
        "max_negative_actual_minus_predicted_by_group": max_negative_gap,
        "min_binomial_z_actual_minus_predicted_by_group": min_negative_z,
    }


def calibration(tasks: Sequence[TaskRecord], bins: int) -> Dict[str, Any]:
    rows = []
    pairs = sorted(
        [(task_prob(task, "p_heldout"), 1.0 if task.actual_success else 0.0, task.episode_id) for task in tasks],
        key=lambda item: item[0],
    )
    n = len(pairs)
    ece = 0.0
    for bin_index in range(bins):
        start = round(bin_index * n / bins)
        end = round((bin_index + 1) * n / bins)
        chunk = pairs[start:end]
        if not chunk:
            continue
        pred = mean([item[0] for item in chunk])
        actual = mean([item[1] for item in chunk])
        gap = actual - pred
        ece += (len(chunk) / n) * abs(gap)
        rows.append(
            {
                "bin": bin_index,
                "n": len(chunk),
                "predicted_mean": pred,
                "actual_success_rate": actual,
                "gap_actual_minus_predicted": gap,
                "predicted_min": chunk[0][0],
                "predicted_max": chunk[-1][0],
            }
        )
    return {"bins": rows, "ece": ece}


def _candidate_indexes(records: Sequence[StepRecord], levels: Sequence[str]) -> Dict[str, Dict[Tuple[Any, ...], List[int]]]:
    out: Dict[str, Dict[Tuple[Any, ...], List[int]]] = {level: defaultdict(list) for level in levels}
    for index, record in enumerate(records):
        for level in levels:
            out[level][record.key(level)].append(index)
    return out


def _choose_candidate(
    target: StepRecord,
    records: Sequence[StepRecord],
    indexes: Dict[str, Dict[Tuple[Any, ...], List[int]]],
    used_episode_ids: set[str],
    rng: random.Random,
    top_n: int,
) -> Tuple[StepRecord, str, bool]:
    levels = ("fine", "mid", "coarse", "action_bbox", "action", "global")
    relaxed = False
    for level in levels:
        pool = [
            records[idx]
            for idx in indexes[level].get(target.key(level), [])
            if records[idx].episode_id != target.episode_id and records[idx].episode_id not in used_episode_ids
        ]
        if not pool and level == "global":
            pool = [records[idx] for idx in indexes[level].get(target.key(level), []) if records[idx].episode_id != target.episode_id]
            relaxed = True
        if not pool:
            continue
        pool.sort(key=lambda candidate: abs(candidate.p_heldout - target.p_heldout))
        shortlist = pool[: max(1, min(top_n, len(pool)))]
        return rng.choice(shortlist), level, relaxed
    raise RuntimeError(f"no pseudo-task candidate for episode={target.episode_id} step={target.step_idx}")


def pseudo_tasks(
    tasks: Sequence[TaskRecord],
    records: Sequence[StepRecord],
    *,
    repeats: int,
    seed: int,
    top_n: int,
    output_path: Path,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    levels = ("fine", "mid", "coarse", "action_bbox", "action", "global")
    indexes = _candidate_indexes(records, levels)
    rows_written = 0
    successes = 0
    total_abs_mean_p_diff = 0.0
    total_abs_logp_diff = 0.0
    relaxed_count = 0
    matched_level_counts: Counter = Counter()
    by_k: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            target_p = [step.p_heldout for step in task.steps]
            target_logp = sum(math.log(max(EPS, value)) for value in target_p)
            target_mean_p = mean(target_p)
            for repeat in range(repeats):
                used_episode_ids: set[str] = set()
                sampled_steps: List[StepRecord] = []
                match_levels: List[str] = []
                relaxed = False
                for step in task.steps:
                    candidate, level, was_relaxed = _choose_candidate(step, records, indexes, used_episode_ids, rng, top_n)
                    used_episode_ids.add(candidate.episode_id)
                    sampled_steps.append(candidate)
                    match_levels.append(level)
                    matched_level_counts[level] += 1
                    relaxed = relaxed or was_relaxed
                sampled_p = [step.p_heldout for step in sampled_steps]
                sampled_logp = sum(math.log(max(EPS, value)) for value in sampled_p)
                sampled_mean_p = mean(sampled_p)
                success = all(step.success for step in sampled_steps)
                successes += 1 if success else 0
                rows_written += 1
                total_abs_mean_p_diff += abs(sampled_mean_p - target_mean_p)
                total_abs_logp_diff += abs(sampled_logp - target_logp)
                relaxed_count += 1 if relaxed else 0
                payload = {
                    "pseudo_id": f"{task.episode_id}:{repeat}",
                    "matched_real_episode_id": task.episode_id,
                    "k": task.k,
                    "success": success,
                    "target_pred_prob": product(target_p),
                    "sampled_pred_prob": product(sampled_p),
                    "target_p_i": target_p,
                    "sampled_p_i": sampled_p,
                    "sampled_step_successes": [step.success for step in sampled_steps],
                    "source_episode_ids": [step.episode_id for step in sampled_steps],
                    "match_levels": match_levels,
                    "relaxed_distinct_source_constraint": relaxed,
                }
                by_k[_k_bin(task.k)].append(payload)
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    by_k_summary = []
    for label in sorted(by_k, key=lambda item: int(item.rstrip("+")) if item.rstrip("+").isdigit() else 999):
        values = by_k[label]
        by_k_summary.append(
            {
                "k_group": label,
                "n_pseudo": len(values),
                "pseudo_tsr": mean([1.0 if item["success"] else 0.0 for item in values]),
                "target_pred_mean": mean([item["target_pred_prob"] for item in values]),
                "sampled_pred_mean": mean([item["sampled_pred_prob"] for item in values]),
            }
        )
    return {
        "pseudo_tasks": rows_written,
        "pseudo_tsr": successes / rows_written if rows_written else 0.0,
        "mean_abs_mean_p_diff": total_abs_mean_p_diff / rows_written if rows_written else 0.0,
        "mean_abs_logp_sum_diff": total_abs_logp_diff / rows_written if rows_written else 0.0,
        "relaxed_distinct_source_rate": relaxed_count / rows_written if rows_written else 0.0,
        "match_level_counts": dict(matched_level_counts),
        "by_k": by_k_summary,
        "output_path": str(output_path),
    }


def critical_step_summary(tasks: Sequence[TaskRecord]) -> Dict[str, Any]:
    bottom1 = []
    bottom2 = []
    min_ps = []
    second_min_ps = []
    for task in tasks:
        values = [step.p_heldout for step in task.steps]
        share1, share2 = neg_log_shares(values)
        bottom1.append(share1)
        bottom2.append(share2)
        ordered = sorted(values)
        if ordered:
            min_ps.append(ordered[0])
            second_min_ps.append(ordered[1] if len(ordered) > 1 else ordered[0])
    return {
        "bottom1_log_failure_share_mean": mean(bottom1),
        "bottom1_log_failure_share_median": median(bottom1),
        "bottom1_log_failure_share_p25": quantile(bottom1, 0.25),
        "bottom1_log_failure_share_p75": quantile(bottom1, 0.75),
        "bottom2_log_failure_share_mean": mean(bottom2),
        "bottom2_log_failure_share_median": median(bottom2),
        "bottom2_log_failure_share_p25": quantile(bottom2, 0.25),
        "bottom2_log_failure_share_p75": quantile(bottom2, 0.75),
        "min_p_mean": mean(min_ps),
        "second_min_p_mean": mean(second_min_ps),
    }


def gate(layer1_data: Dict[str, Any], layer2_data: Dict[str, Any], layer3_data: Dict[str, Any]) -> Dict[str, Any]:
    actual = layer1_data["actual_tsr"]
    pred = layer1_data["predicted_tsr_heldout_cv"]
    pseudo = layer3_data["pseudo"]["pseudo_tsr"]
    ece = layer3_data["calibration"]["ece"]
    product_gap = actual - pred
    pseudo_gap = actual - pseudo
    slope_gap = layer2_data["slope_gap_actual_minus_predicted"]
    high_k_gap = layer2_data["max_negative_actual_minus_predicted_by_group"]
    min_binomial_z = layer2_data["min_binomial_z_actual_minus_predicted_by_group"]

    tolerances = {
        "layer1_abs_tsr_gap": 0.03,
        "layer3_abs_real_pseudo_gap": 0.03,
        "calibration_ece": 0.05,
        "diagnostic_slope_abs_gap": 0.20,
        "high_k_negative_gap": -0.05,
        "layer2_negative_binomial_z": -2.0,
    }
    layer1_pass = abs(product_gap) <= tolerances["layer1_abs_tsr_gap"]
    pseudo_pass = abs(pseudo_gap) <= tolerances["layer3_abs_real_pseudo_gap"]
    calibration_pass = ece <= tolerances["calibration_ece"]
    slope_warning = abs(slope_gap) > tolerances["diagnostic_slope_abs_gap"]
    layer2_pass = high_k_gap >= tolerances["high_k_negative_gap"] and min_binomial_z >= tolerances["layer2_negative_binomial_z"]

    if layer1_pass and pseudo_pass and calibration_pass and layer2_pass:
        verdict = "COMPOUND_CONFIRMED"
        explanation = "Held-out product, per-k no-negative-bend check, calibration, and matched pseudo-task shuffle control are all within tolerance."
    elif pseudo_gap < -tolerances["layer3_abs_real_pseudo_gap"] or high_k_gap < tolerances["high_k_negative_gap"]:
        verdict = "RESIDUAL_COUPLING_DETECTED"
        explanation = "Real tasks underperform matched independent pseudo-tasks or the product line in at least one powered k group."
    else:
        verdict = "INCONCLUSIVE"
        explanation = "Some evidence is close to compound, but at least one required layer misses the tolerance."

    return {
        "verdict": verdict,
        "explanation": explanation,
        "tolerances": tolerances,
        "checks": {
            "layer1_product_gap_actual_minus_predicted": product_gap,
            "layer1_pass": layer1_pass,
            "layer2_slope_gap_actual_minus_predicted_diagnostic": slope_gap,
            "layer2_slope_warning": slope_warning,
            "layer2_high_k_min_gap": high_k_gap,
            "layer2_min_binomial_z_actual_minus_predicted": min_binomial_z,
            "layer2_pass": layer2_pass,
            "layer3_real_minus_pseudo_tsr": pseudo_gap,
            "layer3_pseudo_pass": pseudo_pass,
            "layer3_calibration_ece": ece,
            "layer3_calibration_pass": calibration_pass,
        },
    }


def write_per_task(tasks: Sequence[TaskRecord], output_path: Path, step_sr: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            p_heldout = [step.p_heldout for step in task.steps]
            p_binned = [step.p_binned for step in task.steps]
            share1, share2 = neg_log_shares(p_heldout)
            payload = {
                "episode_id": task.episode_id,
                "k": task.k,
                "actual_success": task.actual_success,
                "actual_progress": task.actual_progress,
                "predicted_prob_heldout_cv": product(p_heldout),
                "predicted_prob_binned_leave_one_out": product(p_binned),
                "uniform_wrong_prob": step_sr ** task.k,
                "per_step_p_heldout_cv": p_heldout,
                "per_step_p_binned_leave_one_out": p_binned,
                "per_step_success": [step.success for step in task.steps],
                "per_step_sources_heldout_cv": [step.p_heldout_source for step in task.steps],
                "per_step_sources_binned": [step.p_binned_source for step in task.steps],
                "bottom1_log_failure_share": share1,
                "bottom2_log_failure_share": share2,
                "min_p_i": min(p_heldout) if p_heldout else None,
                "step_features": [step.features for step in task.steps],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def fmt(value: float) -> str:
    return f"{value:.6f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---" for _ in headers]) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def write_report(
    output_path: Path,
    *,
    args: argparse.Namespace,
    n_tasks: int,
    n_steps: int,
    step_sr: float,
    source_counts_binned: Counter,
    source_counts_heldout: Counter,
    layer1_data: Dict[str, Any],
    layer2_data: Dict[str, Any],
    calibration_data: Dict[str, Any],
    pseudo_data: Dict[str, Any],
    critical_data: Dict[str, Any],
    gate_data: Dict[str, Any],
) -> None:
    lines: List[str] = []
    lines.append("# GUI-360 Compound Structure Proof")
    lines.append("")
    lines.append("Date: 2026-06-30")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This is a zero-training diagnostic over frozen per-step eval outputs.")
    lines.append("")
    lines.append(f"History/eval condition: {args.condition_label}. The same condition supplies both per-step p_i estimates and TSR.")
    lines.append("")
    lines.append(f"Tasks: {n_tasks}; steps: {n_steps}; observed StepSR: {pct(step_sr)}.")
    lines.append("")
    lines.append("Observable difficulty features used: action type, target bbox area bin, bbox aspect bin, 3x3 screen position bin, step phase, task-length bin, and label completeness. Similar-element density was not present in the existing eval artifacts, so it is marked unavailable rather than inferred.")
    lines.append("")
    lines.append("p_i estimates:")
    lines.append("")
    lines.append(f"- Difficulty-binned leave-one-out source counts: `{dict(source_counts_binned)}`")
    lines.append(f"- Held-out 5-fold source counts: `{dict(source_counts_heldout)}`")
    lines.append("")
    lines.append("## Layer 1 - Overall Compound Fit")
    lines.append("")
    rows = [
        ["Actual TSR", pct(layer1_data["actual_tsr"])],
        ["Predicted TSR, held-out product(p_i)", pct(layer1_data["predicted_tsr_heldout_cv"])],
        ["Predicted TSR, difficulty-binned LOO product(p_i)", pct(layer1_data["predicted_tsr_binned_leave_one_out"])],
        ["Uniform mean-p^k baseline (wrong estimator)", pct(layer1_data["uniform_mean_p_power_wrong_by_task_k"])],
        ["Uniform mean-p^mean(k) baseline (wrong estimator)", pct(layer1_data["uniform_mean_p_power_wrong_at_mean_k"])],
    ]
    lines.append(markdown_table(["quantity", "value"], rows))
    lines.append("")
    lines.append("The uniform baseline is intentionally reported as the labeled-wrong Trap-1 estimator. It collapses non-uniform step difficulty into one mean p and is not used for the verdict.")
    lines.append("")
    lines.append("## Layer 2 - Scaling With Step Count")
    lines.append("")
    pooled_rows = []
    for row in layer2_data["pooled_by_k"]:
        pooled_rows.append(
            [
                row["k_group"],
                row["n"],
                f"{row['mean_k']:.2f}",
                pct(row["actual_tsr"]),
                pct(row["predicted_tsr_heldout_cv"]),
                f"{row['log_actual_tsr_smoothed']:.3f}",
                f"{row['log_predicted_tsr_heldout_cv']:.3f}",
                f"{row['actual_minus_predicted']:+.4f}",
                f"{row['binomial_z_actual_minus_predicted']:+.2f}",
            ]
        )
    lines.append(markdown_table(["k group", "n", "mean k", "actual TSR", "pred product", "log actual", "log pred", "actual-pred", "z"], pooled_rows))
    lines.append("")
    lines.append(f"Actual log-TSR slope: {layer2_data['actual_log_tsr_slope']['slope']:.4f} (R2={layer2_data['actual_log_tsr_slope']['r2']:.3f}).")
    lines.append(f"Predicted product log-TSR slope: {layer2_data['predicted_log_tsr_slope']['slope']:.4f} (R2={layer2_data['predicted_log_tsr_slope']['r2']:.3f}).")
    lines.append(f"Mean log p_i: {layer2_data['mean_log_p_i']:.4f}.")
    lines.append(f"Slope gap actual-predicted: {layer2_data['slope_gap_actual_minus_predicted']:+.4f}.")
    lines.append(f"Most negative per-k binomial residual z(actual-predicted): {layer2_data['min_binomial_z_actual_minus_predicted_by_group']:+.2f}.")
    lines.append("")
    lines.append("Raw k bins are included in `compound_fit.json`; pooled bins are used above so each reported scaling bin has enough support when possible. The log-slope fit is retained as a diagnostic, but bins with near-zero expected TSR make smoothed log(actual TSR) noisy. The coupling gate therefore uses the direct residual question from the spec: whether actual TSR bends significantly below the product line as k grows.")
    lines.append("")
    lines.append("## Layer 3 - Calibration And Shuffle Control")
    lines.append("")
    cal_rows = []
    for row in calibration_data["bins"]:
        cal_rows.append(
            [
                row["bin"],
                row["n"],
                pct(row["predicted_mean"]),
                pct(row["actual_success_rate"]),
                f"{row['gap_actual_minus_predicted']:+.4f}",
                f"{row['predicted_min']:.5f}-{row['predicted_max']:.5f}",
            ]
        )
    lines.append(markdown_table(["bin", "n", "predicted", "actual", "actual-pred", "pred range"], cal_rows))
    lines.append("")
    lines.append(f"Calibration ECE: {calibration_data['ece']:.4f}.")
    lines.append("")
    lines.append("Shuffle control pseudo-tasks recombine steps from different real tasks, matched to the real task's k and per-step held-out p_i distribution.")
    lines.append("")
    pseudo_rows = [
        ["Real-task TSR", pct(layer1_data["actual_tsr"])],
        ["Pseudo-task TSR", pct(pseudo_data["pseudo_tsr"])],
        ["Real minus pseudo", f"{layer1_data['actual_tsr'] - pseudo_data['pseudo_tsr']:+.4f}"],
        ["Pseudo tasks", pseudo_data["pseudo_tasks"]],
        ["Mean abs mean-p mismatch", f"{pseudo_data['mean_abs_mean_p_diff']:.6f}"],
        ["Mean abs sum-log-p mismatch", f"{pseudo_data['mean_abs_logp_sum_diff']:.6f}"],
        ["Relaxed source-distinct rate", pct(pseudo_data["relaxed_distinct_source_rate"])],
    ]
    lines.append(markdown_table(["quantity", "value"], pseudo_rows))
    lines.append("")
    pseudo_k_rows = []
    for row in pseudo_data["by_k"]:
        pseudo_k_rows.append([row["k_group"], row["n_pseudo"], pct(row["pseudo_tsr"]), pct(row["target_pred_mean"]), pct(row["sampled_pred_mean"])])
    lines.append(markdown_table(["k group", "n pseudo", "pseudo TSR", "target pred", "sampled pred"], pseudo_k_rows))
    lines.append("")
    lines.append(f"Pseudo match levels: `{pseudo_data['match_level_counts']}`")
    lines.append("")
    lines.append("## Critical-Step Corroboration")
    lines.append("")
    crit_rows = [
        ["Bottom-1 share mean", pct(critical_data["bottom1_log_failure_share_mean"])],
        ["Bottom-1 share median", pct(critical_data["bottom1_log_failure_share_median"])],
        ["Bottom-1 share IQR", f"{pct(critical_data['bottom1_log_failure_share_p25'])} - {pct(critical_data['bottom1_log_failure_share_p75'])}"],
        ["Bottom-2 share mean", pct(critical_data["bottom2_log_failure_share_mean"])],
        ["Bottom-2 share median", pct(critical_data["bottom2_log_failure_share_median"])],
        ["Bottom-2 share IQR", f"{pct(critical_data['bottom2_log_failure_share_p25'])} - {pct(critical_data['bottom2_log_failure_share_p75'])}"],
        ["Mean min p_i", pct(critical_data["min_p_mean"])],
        ["Mean second-min p_i", pct(critical_data["second_min_p_mean"])],
    ]
    lines.append(markdown_table(["quantity", "value"], crit_rows))
    lines.append("")
    lines.append("These shares measure how much of each task's negative log product is contributed by the lowest-p steps. A high bottom-1/bottom-2 share means the task product is dominated by a few critical grounding steps.")
    lines.append("")
    lines.append("## Gate Verdict")
    lines.append("")
    lines.append(f"Verdict: **{gate_data['verdict']}**")
    lines.append("")
    lines.append(gate_data["explanation"])
    lines.append("")
    gate_rows = [[key, value] for key, value in gate_data["checks"].items()]
    lines.append(markdown_table(["check", "value"], gate_rows))
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `outputs/compound_proof/compound_fit.md`")
    lines.append("- `outputs/compound_proof/compound_fit.json`")
    lines.append("- `outputs/compound_proof/per_task.jsonl`")
    lines.append("- `outputs/compound_proof/pseudo_tasks.jsonl`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="GUI-360 compound independent-step proof diagnostic")
    parser.add_argument("--test-data", default=DEFAULT_TEST_DATA)
    parser.add_argument("--eval-results", nargs="+", default=DEFAULT_RESULTS)
    parser.add_argument("--step-rows", default=DEFAULT_STEP_ROWS, help="Optional compact step rows JSONL; if set, overrides --eval-results")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--condition-label", default=None)
    parser.add_argument("--min-bucket", type=int, default=25)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-k-group", type=int, default=30)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--pseudo-repeats", type=int, default=50)
    parser.add_argument("--pseudo-seed", type=int, default=20260630)
    parser.add_argument("--pseudo-top-n", type=int, default=32)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_episodes = read_test_episodes(Path(args.test_data))
    if args.step_rows:
        eval_results = read_step_rows(Path(args.step_rows))
        if args.condition_label is None:
            args.condition_label = "compact matched ShareGPT, GT-history teacher-forced, frozen action_match matcher"
    else:
        eval_results = read_eval_results([Path(path) for path in args.eval_results])
        if args.condition_label is None:
            args.condition_label = "original GUI-360 template, GT-history teacher-forced, full-history mode, frozen compute_step_reward matcher"
    tasks = build_records(test_episodes, eval_results)
    records = flatten_steps(tasks)
    if not tasks or not records:
        raise SystemExit("no task/step records loaded")

    source_counts_binned = estimate_leave_one_out(records, min_bucket=args.min_bucket, alpha=ALPHA)
    source_counts_heldout = estimate_heldout_cv(records, folds=args.folds, min_bucket=args.min_bucket, alpha=ALPHA)

    n_tasks = len(tasks)
    n_steps = len(records)
    step_sr = mean([1.0 if step.success else 0.0 for step in records])

    l1 = layer1(tasks, step_sr)
    l2 = layer2(tasks, step_sr, args.min_k_group)
    cal = calibration(tasks, args.calibration_bins)
    pseudo = pseudo_tasks(
        tasks,
        records,
        repeats=args.pseudo_repeats,
        seed=args.pseudo_seed,
        top_n=args.pseudo_top_n,
        output_path=output_dir / "pseudo_tasks.jsonl",
    )
    critical = critical_step_summary(tasks)
    l3 = {"calibration": cal, "pseudo": pseudo}
    gate_data = gate(l1, l2, l3)

    write_per_task(tasks, output_dir / "per_task.jsonl", step_sr)

    fit_json = {
        "inputs": {
            "test_data": args.test_data,
            "eval_results": args.eval_results,
            "step_rows": args.step_rows,
            "history_condition": args.condition_label,
            "min_bucket": args.min_bucket,
            "folds": args.folds,
            "pseudo_repeats": args.pseudo_repeats,
            "pseudo_seed": args.pseudo_seed,
        },
        "n_tasks": n_tasks,
        "n_steps": n_steps,
        "step_sr": step_sr,
        "source_counts_binned_leave_one_out": dict(source_counts_binned),
        "source_counts_heldout_cv": dict(source_counts_heldout),
        "layer1": l1,
        "layer2": l2,
        "layer3": l3,
        "critical_steps": critical,
        "gate": gate_data,
    }
    (output_dir / "compound_fit.json").write_text(json.dumps(fit_json, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(
        output_dir / "compound_fit.md",
        args=args,
        n_tasks=n_tasks,
        n_steps=n_steps,
        step_sr=step_sr,
        source_counts_binned=source_counts_binned,
        source_counts_heldout=source_counts_heldout,
        layer1_data=l1,
        layer2_data=l2,
        calibration_data=cal,
        pseudo_data=pseudo,
        critical_data=critical,
        gate_data=gate_data,
    )
    print(json.dumps({"output_dir": str(output_dir), "verdict": gate_data["verdict"], "layer1": l1, "layer3_real_minus_pseudo": gate_data["checks"]["layer3_real_minus_pseudo_tsr"]}, indent=2))


if __name__ == "__main__":
    main()