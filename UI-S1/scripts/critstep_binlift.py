#!/usr/bin/env python3
"""High-lift observable-bin sweep for operational critical-step definitions.

Bins are defined only from inference-observable features. Measured lift uses
actual verifier selections when all-step verifier outcomes are available. If the
required full-step verifier outcomes are missing, the script writes a pending
audit/report instead of substituting oracle or critical-only verifier results.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.critstep_reward_structure_uia import controls_for_step  # noqa: E402
from scripts.score_critstep_verifier_v2_cot_voting import candidate_distinct_key  # noqa: E402
from scripts.verifier_e2e_eval import load_verifiers, strict_weight  # noqa: E402


DEFAULT_TEST_CANDIDATES = "outputs/verifier_e2e/slice200/candidates/per_step.jsonl"
DEFAULT_TEST_DATA = "outputs/gui360_history_ab/original_eval/gui360_test_1000_balanced_uia.jsonl"
DEFAULT_TRAIN_DATA = "outputs/gui360_history_ab/original_eval/gui360_train_balanced_uia.jsonl"
DEFAULT_TEST_TASKS = "outputs/critstep_eval/per_task.jsonl"
DEFAULT_STRICT_SUMMARY = "outputs/critstep_verifier_v2/strict/combine/strict_summary.json"
DEFAULT_OUTPUT_DIR = "outputs/critstep_binlift"
BUDGETS = (0.10, 0.20, 0.30, 0.50, 1.00)
QUANTILE_GRIDS = (3, 4, 10)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def entropy_from_values(values: Sequence[str]) -> float:
    vals = [str(value) for value in values if value not in (None, "")]
    if not vals:
        return 0.0
    counts = Counter(vals)
    total = len(vals)
    return -sum((count / total) * math.log(count / total, 2) for count in counts.values())


def norm_entropy(values: Sequence[str]) -> float:
    vals = [str(value) for value in values if value not in (None, "")]
    n_unique = len(set(vals))
    if n_unique <= 1:
        return 0.0
    return entropy_from_values(vals) / math.log(n_unique, 2)


def modal_fraction(values: Sequence[str]) -> float:
    vals = [str(value) for value in values if value not in (None, "")]
    if not vals:
        return 0.0
    return Counter(vals).most_common(1)[0][1] / len(vals)


def decode_key(candidate: Mapping[str, Any]) -> str:
    try:
        return candidate_distinct_key(candidate)
    except Exception:
        control = candidate.get("control") if isinstance(candidate.get("control"), dict) else {}
        payload = {
            "action_signature": candidate.get("action_signature"),
            "pred_type": candidate.get("pred_type"),
            "control_key": control.get("key"),
            "control_rect": control.get("rect"),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def read_episode_index(path: str) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    return {str(row.get("episode_id")): row for row in read_jsonl(Path(path))}


def read_task_index(path: str) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    return {str(row.get("episode_id")): row for row in read_jsonl(Path(path))}


def task_length_from(row: Mapping[str, Any], task: Optional[Mapping[str, Any]], episode: Optional[Mapping[str, Any]]) -> int:
    if task and task.get("k") is not None:
        return int(task.get("k"))
    if episode and isinstance(episode.get("steps"), list):
        return len(episode["steps"])
    return int(row.get("task_k") or row.get("n_steps") or 0)


def task_length_bin(k: int) -> str:
    if k <= 1:
        return "len_1"
    if k <= 3:
        return "len_2_3"
    if k <= 5:
        return "len_4_5"
    if k <= 10:
        return "len_6_10"
    return "len_11_plus"


def position_bin(step_idx: int, task_k: int) -> str:
    if task_k <= 1:
        return "pos_only"
    rel = step_idx / max(1, task_k - 1)
    if rel < 1 / 3:
        return "pos_early"
    if rel < 2 / 3:
        return "pos_mid"
    return "pos_late"


def candidate_score_stats(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    values = []
    for candidate in candidates:
        for key in ("stage1_score_k8", "verifier_score", "verifier_margin"):
            if candidate.get(key) is not None:
                values.append(safe_float(candidate.get(key)))
                break
    vals = sorted(values, reverse=True)
    return {
        "verifier_score_available_frac": len(vals) / len(candidates) if candidates else 0.0,
        "verifier_score_max": vals[0] if vals else 0.0,
        "verifier_score_top2_gap": vals[0] - vals[1] if len(vals) >= 2 else 0.0,
        "verifier_score_mean": float(np.mean(vals)) if vals else 0.0,
    }


def observable_features(row: Mapping[str, Any], task: Optional[Mapping[str, Any]], episode: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    candidates = row.get("candidates") if isinstance(row.get("candidates"), list) else []
    greedy = candidates[0] if candidates else {}
    step_idx = int(row.get("step_idx") or 0)
    task_k = task_length_from(row, task, episode)
    rel_pos = step_idx / max(1, task_k - 1) if task_k else 0.0
    pred_types = [str(candidate.get("pred_type") or "unknown") for candidate in candidates]
    decode_keys = [decode_key(candidate) for candidate in candidates]
    control_keys = []
    for candidate in candidates:
        control = candidate.get("control") if isinstance(candidate.get("control"), dict) else {}
        control_keys.append(str(control.get("key") or "NO_CONTROL"))
    screen_n_controls = 0
    if episode and isinstance(episode.get("steps"), list) and step_idx < len(episode["steps"]):
        screen_n_controls = len(controls_for_step(episode["steps"][step_idx]))
    features: Dict[str, Any] = {
        "greedy_pred_type": str(greedy.get("pred_type") or "unknown"),
        "greedy_pred_category": str(greedy.get("pred_category") or "unknown"),
        "position_bin": position_bin(step_idx, task_k),
        "task_length_bin": task_length_bin(task_k),
        "step_idx": float(step_idx),
        "task_k": float(task_k),
        "relative_position": float(rel_pos),
        "distinct_decode_count": float(len(set(decode_keys))),
        "distinct_action_type_count": float(len(set(pred_types))),
        "distinct_control_count": float(len(set(key for key in control_keys if key != "NO_CONTROL"))),
        "decode_entropy_norm": norm_entropy(decode_keys),
        "control_entropy_norm": norm_entropy(control_keys),
        "one_minus_modal_decode_frac": 1.0 - modal_fraction(decode_keys),
        "one_minus_modal_control_frac": 1.0 - modal_fraction(control_keys),
        "screen_n_controls": float(screen_n_controls),
    }
    features.update(candidate_score_stats(candidates))
    return features


def load_verifier_map(verifier_root: str, n: int, strict_summary: str) -> Dict[str, Dict[str, Any]]:
    if not verifier_root:
        return {}
    weight = strict_weight(Path(strict_summary)) if strict_summary else 0.0
    return load_verifiers(verifier_root, [n], weight).get(n, {})


def build_split_rows(
    *,
    split: str,
    candidates_path: str,
    data_path: str,
    tasks_path: str,
    verifier_root: str,
    n_candidates: int,
    strict_summary: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidate_rows = read_jsonl(Path(candidates_path)) if candidates_path else []
    episodes = read_episode_index(data_path)
    tasks = read_task_index(tasks_path)
    verifier_by_target = load_verifier_map(verifier_root, n_candidates, strict_summary)
    rows: List[Dict[str, Any]] = []
    missing = Counter()
    for raw in candidate_rows:
        episode_id = str(raw.get("episode_id"))
        step_idx = int(raw.get("step_idx") or 0)
        task = tasks.get(episode_id)
        episode = episodes.get(episode_id)
        task_k = task_length_from(raw, task, episode)
        candidates = raw.get("candidates") if isinstance(raw.get("candidates"), list) else []
        if not candidates:
            missing["no_candidates"] += 1
            continue
        greedy = candidates[0]
        verifier_item = verifier_by_target.get(str(raw.get("target_id")))
        verifier_candidate_id = verifier_item.get("candidate_id") if verifier_item else None
        verifier_candidate = None
        if verifier_candidate_id is not None:
            for candidate in candidates:
                if str(candidate.get("candidate_id")) == str(verifier_candidate_id):
                    verifier_candidate = candidate
                    break
        if verifier_item and verifier_candidate is None:
            missing["verifier_candidate_missing_in_pool"] += 1
        p_i = None
        bottom2 = False
        if task and isinstance(task.get("per_step_p_heldout_cv"), list) and step_idx < len(task["per_step_p_heldout_cv"]):
            p_i = safe_float(task["per_step_p_heldout_cv"][step_idx], default=float("nan"))
            bottom2 = step_idx in {int(index) for index in task.get("bottom2_critical_indices", [])}
        row = {
            "row_id": f"{split}:{episode_id}:{step_idx}",
            "target_id": raw.get("target_id"),
            "split": split,
            "episode_id": episode_id,
            "episode_key": f"{split}:{episode_id}",
            "step_idx": step_idx,
            "task_k": task_k,
            "complete_episode_expected_steps": task_k,
            "greedy_success": bool(greedy.get("is_correct")),
            "oracle_success": any(bool(candidate.get("is_correct")) for candidate in candidates[:n_candidates]),
            "verifier_available": verifier_candidate is not None,
            "verifier_success": bool(verifier_candidate.get("is_correct")) if verifier_candidate is not None else None,
            "verifier_helped": bool((not greedy.get("is_correct")) and verifier_candidate is not None and verifier_candidate.get("is_correct")),
            "verifier_hurt": bool(greedy.get("is_correct") and verifier_candidate is not None and not verifier_candidate.get("is_correct")),
            "p_i_heldout_compare_only": p_i,
            "bottom_p_i_compare_only": bool(bottom2),
            "score_set_requires_verifier": bool((not greedy.get("is_correct")) and any(bool(candidate.get("is_correct")) for candidate in candidates[:n_candidates])),
            "deterministic_skip_reason": "greedy_correct" if bool(greedy.get("is_correct")) else ("missing_no_correct_candidate" if not any(bool(candidate.get("is_correct")) for candidate in candidates[:n_candidates]) else None),
            "features": observable_features(raw, task, episode),
            "bin_memberships": [],
        }
        rows.append(row)
    manifest = {
        "split": split,
        "candidates_path": candidates_path,
        "rows_in": len(candidate_rows),
        "rows_out": len(rows),
        "episodes": len({row["episode_key"] for row in rows}),
        "verifier_root": verifier_root,
        "verifier_predictions": len(verifier_by_target),
        "verifier_available_rows": sum(1 for row in rows if row["verifier_available"]),
        "missing": dict(missing),
    }
    return rows, manifest


def complete_episode_keys(rows: Sequence[Mapping[str, Any]]) -> Set[str]:
    by_episode: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[str(row["episode_key"])].append(row)
    complete = set()
    for episode_key, episode_rows in by_episode.items():
        expected = int(episode_rows[0].get("complete_episode_expected_steps") or 0)
        if expected > 0 and len({int(row["step_idx"]) for row in episode_rows}) == expected:
            complete.add(episode_key)
    return complete


def tsr_for_rows(rows: Sequence[Mapping[str, Any]], verify_ids: Set[str], mode: str = "verifier") -> Optional[float]:
    complete = complete_episode_keys(rows)
    if not complete:
        return None
    by_episode: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["episode_key"] in complete:
            by_episode[str(row["episode_key"])].append(row)
    success_count = 0
    for episode_rows in by_episode.values():
        ok = True
        for row in episode_rows:
            if row["row_id"] in verify_ids:
                if mode == "oracle":
                    step_success = bool(row.get("oracle_success"))
                else:
                    if row.get("greedy_success"):
                        step_success = True
                    elif not row.get("oracle_success"):
                        step_success = False
                    elif not row.get("verifier_available"):
                        return None
                    else:
                        step_success = bool(row.get("verifier_success"))
            else:
                step_success = bool(row.get("greedy_success"))
            ok = ok and step_success
        success_count += int(ok)
    return success_count / len(by_episode) if by_episode else None


def projected_tsr(rows: Sequence[Mapping[str, Any]], verify_ids: Set[str], mode: str = "real") -> Optional[float]:
    by_episode: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("p_i_heldout_compare_only") is None or not math.isfinite(float(row.get("p_i_heldout_compare_only"))):
            return None
        by_episode[str(row["episode_key"])].append(row)
    values = []
    for episode_rows in by_episode.values():
        prob = 1.0
        for row in episode_rows:
            p_i = max(0.0, min(1.0, float(row["p_i_heldout_compare_only"])))
            if row["row_id"] in verify_ids:
                if mode == "oracle":
                    p_i = 1.0
                else:
                    # Analysis-time bin gating uses verifier only on greedy-wrong recoverable rows.
                    if row.get("greedy_success"):
                        p_i = 1.0
                    elif not row.get("oracle_success"):
                        p_i = 0.0
                    elif not row.get("verifier_available"):
                        return None
                    else:
                        p_i = 1.0 if row.get("verifier_success") else 0.0
            prob *= p_i
        values.append(prob)
    return float(np.mean(values)) if values else None


def quantile_edges(values: Sequence[float], q: int) -> List[float]:
    vals = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not vals:
        return []
    return sorted(set(float(np.quantile(vals, frac)) for frac in np.linspace(0, 1, q + 1)[1:-1]))


def quantile_label(value: float, edges: Sequence[float]) -> int:
    label = 0
    for edge in edges:
        if value > edge:
            label += 1
    return label


def build_bins(train_rows: Sequence[Mapping[str, Any]], all_rows: Sequence[Dict[str, Any]], min_train_steps: int, selection_split: str = "train") -> Dict[str, Dict[str, Any]]:
    categorical = ["greedy_pred_type", "greedy_pred_category", "position_bin", "task_length_bin"]
    continuous = [
        "relative_position", "distinct_decode_count", "distinct_action_type_count", "distinct_control_count", "decode_entropy_norm",
        "control_entropy_norm", "one_minus_modal_decode_frac", "one_minus_modal_control_frac", "screen_n_controls",
        "verifier_score_available_frac", "verifier_score_max", "verifier_score_top2_gap", "verifier_score_mean",
    ]
    quantile_specs: Dict[str, List[float]] = {}
    for name in continuous:
        values = [safe_float(row["features"].get(name), default=float("nan")) for row in train_rows]
        for q in QUANTILE_GRIDS:
            quantile_specs[f"{name}_q{q}"] = quantile_edges(values, q)
    bins: Dict[str, Dict[str, Any]] = {}

    def add(row: Dict[str, Any], bin_id: str, definition: Dict[str, Any]) -> None:
        row["bin_memberships"].append(bin_id)
        item = bins.setdefault(bin_id, {"bin_id": bin_id, "definition": definition, "train_ids": set(), "test_ids": set()})
        item[f"{row['split']}_ids"].add(row["row_id"])

    for row in all_rows:
        features = row["features"]
        atomic: List[Tuple[str, Dict[str, Any]]] = []
        for name in categorical:
            value = str(features.get(name) or "unknown")
            bin_id = f"{name}={value}"
            definition = {"type": "categorical", "feature": name, "value": value, "observable": True}
            add(row, bin_id, definition)
            atomic.append((bin_id, definition))
        for spec, edges in quantile_specs.items():
            feature = spec.rsplit("_q", 1)[0]
            q = int(spec.rsplit("_q", 1)[1])
            value = safe_float(features.get(feature), default=0.0)
            label = quantile_label(value, edges)
            bin_id = f"{spec}=bin{label}"
            definition = {"type": "quantile", "feature": feature, "q": q, "bin": label, "edges": edges, "observable": True}
            add(row, bin_id, definition)
            if q in (3, 4):
                atomic.append((bin_id, definition))
        for left_idx, (left, left_def) in enumerate(atomic[:12]):
            for right, right_def in atomic[left_idx + 1 : 12]:
                bin_id = f"{left} && {right}"
                definition = {"type": "cross2", "parts": [left_def, right_def], "observable": True}
                add(row, bin_id, definition)
        # A restrained 3-way set: action type x position x one continuous uncertainty quantile.
        for uncertainty in [item for item in atomic if item[0].startswith("decode_entropy_norm_q4=") or item[0].startswith("verifier_score_top2_gap_q4=")]:
            type_id = next((item for item in atomic if item[0].startswith("greedy_pred_type=")), None)
            pos_id = next((item for item in atomic if item[0].startswith("position_bin=")), None)
            if type_id and pos_id:
                bin_id = f"{type_id[0]} && {pos_id[0]} && {uncertainty[0]}"
                definition = {"type": "cross3", "parts": [type_id[1], pos_id[1], uncertainty[1]], "observable": True}
                add(row, bin_id, definition)
    count_field = "train_ids" if selection_split == "train" else "test_ids"
    return {bin_id: item for bin_id, item in bins.items() if len(item.get(count_field, set())) >= min_train_steps}


def score_bin(bin_item: Mapping[str, Any], train_rows: Sequence[Mapping[str, Any]], test_rows: Sequence[Mapping[str, Any]], baseline_cache: Mapping[str, Optional[float]]) -> Dict[str, Any]:
    train_ids = set(bin_item.get("train_ids", set()))
    test_ids = set(bin_item.get("test_ids", set()))
    train_measured = tsr_for_rows(train_rows, train_ids, mode="verifier")
    test_measured = tsr_for_rows(test_rows, test_ids, mode="verifier")
    train_oracle = tsr_for_rows(train_rows, train_ids, mode="oracle")
    test_oracle = tsr_for_rows(test_rows, test_ids, mode="oracle")
    train_projected = projected_tsr(train_rows, train_ids, mode="real")
    test_projected = projected_tsr(test_rows, test_ids, mode="real")
    test_projected_oracle = projected_tsr(test_rows, test_ids, mode="oracle")
    test_cost = len(test_ids) / len(test_rows) if test_rows else 0.0
    train_cost = len(train_ids) / len(train_rows) if train_rows else 0.0
    def delta(value: Optional[float], baseline: Optional[float]) -> Optional[float]:
        return None if value is None or baseline is None else value - baseline
    test_bin_rows = [row for row in test_rows if row["row_id"] in test_ids]
    greedy_wrong = sum(1 for row in test_bin_rows if not row.get("greedy_success"))
    helped = sum(1 for row in test_bin_rows if row.get("verifier_helped"))
    score_set_rows = [row for row in test_bin_rows if row.get("score_set_requires_verifier")]
    verifier_available = sum(1 for row in score_set_rows if row.get("verifier_available"))
    bottom_ids = {row["row_id"] for row in test_rows if row.get("bottom_p_i_compare_only")}
    jaccard = len(test_ids & bottom_ids) / len(test_ids | bottom_ids) if (test_ids or bottom_ids) else 0.0
    measured_lift = delta(test_measured, baseline_cache.get("test_greedy_tsr"))
    lift_per_cost = None if measured_lift is None or test_cost <= 0 else measured_lift / test_cost
    return {
        "bin_id": bin_item["bin_id"],
        "definition": bin_item["definition"],
        "train_steps": len(train_ids),
        "test_steps": len(test_ids),
        "train_cost": train_cost,
        "test_cost": test_cost,
        "test_score_set_steps": len(score_set_rows),
        "test_verifier_available_fraction": verifier_available / len(score_set_rows) if score_set_rows else 1.0,
        "test_verifier_fix_rate_on_greedy_wrong": helped / greedy_wrong if greedy_wrong else None,
        "train_measured_tsr": train_measured,
        "test_measured_tsr": test_measured,
        "test_measured_lift": measured_lift,
        "test_lift_per_cost": lift_per_cost,
        "train_projected_tsr": train_projected,
        "test_projected_tsr": test_projected,
        "test_projected_lift": delta(test_projected, baseline_cache.get("test_projected_baseline")),
        "test_oracle_measured_tsr": test_oracle,
        "test_oracle_measured_lift": delta(test_oracle, baseline_cache.get("test_greedy_tsr")),
        "test_oracle_projected_tsr": test_projected_oracle,
        "test_oracle_projected_lift": delta(test_projected_oracle, baseline_cache.get("test_projected_baseline")),
        "jaccard_vs_bottom_p_i": jaccard,
    }


def greedy_bin_set(train_rows: Sequence[Mapping[str, Any]], test_rows: Sequence[Mapping[str, Any]], scored_bins: Sequence[Mapping[str, Any]], budgets: Sequence[float], baseline: Mapping[str, Optional[float]]) -> List[Dict[str, Any]]:
    eligible = [row for row in scored_bins if row.get("train_measured_tsr") is not None]
    eligible.sort(key=lambda row: ((row.get("train_measured_tsr") or 0.0) - (baseline.get("train_greedy_tsr") or 0.0)) / max(1e-9, row.get("train_cost") or 0.0), reverse=True)
    out = []
    for budget in budgets:
        selected_bins = []
        selected_train_ids: Set[str] = set()
        selected_test_ids: Set[str] = set()
        best_train_tsr = baseline.get("train_greedy_tsr")
        for bin_row in eligible:
            candidate_test_ids = selected_test_ids | set(bin_row.get("_test_ids", set()))
            # Cost budget is enforced on TEST rows for reporting consistency.
            if len(candidate_test_ids) / len(test_rows) > budget:
                continue
            candidate_train_ids = selected_train_ids | set(bin_row.get("_train_ids", set()))
            train_tsr = tsr_for_rows(train_rows, candidate_train_ids, mode="verifier")
            if train_tsr is not None and (best_train_tsr is None or train_tsr > best_train_tsr + 1e-12):
                selected_bins.append(bin_row["bin_id"])
                selected_train_ids = candidate_train_ids
                selected_test_ids = candidate_test_ids
                best_train_tsr = train_tsr
        test_tsr = tsr_for_rows(test_rows, selected_test_ids, mode="verifier")
        out.append({
            "budget": budget,
            "selected_bins": selected_bins,
            "test_verified_steps": len(selected_test_ids),
            "test_cost": len(selected_test_ids) / len(test_rows) if test_rows else 0.0,
            "test_measured_tsr": test_tsr,
            "test_measured_lift": None if test_tsr is None or baseline.get("test_greedy_tsr") is None else test_tsr - baseline["test_greedy_tsr"],
        })
    return out


def episode_folds(rows: Sequence[Mapping[str, Any]], n_folds: int, seed: int) -> List[Set[str]]:
    episode_keys = sorted({str(row["episode_key"]) for row in rows})
    rng = np.random.default_rng(seed)
    shuffled = list(episode_keys)
    rng.shuffle(shuffled)
    folds = [set() for _ in range(max(2, n_folds))]
    for index, episode_key in enumerate(shuffled):
        folds[index % len(folds)].add(episode_key)
    return folds


def cv_budget_curve(rows: Sequence[Mapping[str, Any]], scored_bins: Sequence[Mapping[str, Any]], budgets: Sequence[float], n_folds: int, seed: int) -> List[Dict[str, Any]]:
    if not rows:
        return []
    folds = episode_folds(rows, n_folds=n_folds, seed=seed)
    by_budget: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
    for fold_index, eval_episodes in enumerate(folds):
        train_rows = [row for row in rows if str(row["episode_key"]) not in eval_episodes]
        eval_rows = [row for row in rows if str(row["episode_key"]) in eval_episodes]
        if not train_rows or not eval_rows:
            continue
        train_ids = {row["row_id"] for row in train_rows}
        eval_ids = {row["row_id"] for row in eval_rows}
        fold_bins = []
        for bin_row in scored_bins:
            item = dict(bin_row)
            item["_train_ids"] = list(set(bin_row.get("_test_ids", [])) & train_ids)
            item["_test_ids"] = list(set(bin_row.get("_test_ids", [])) & eval_ids)
            item["train_cost"] = len(item["_train_ids"]) / len(train_rows) if train_rows else 0.0
            item["test_cost"] = len(item["_test_ids"]) / len(eval_rows) if eval_rows else 0.0
            fold_bins.append(item)
        fold_baseline = {
            "train_greedy_tsr": tsr_for_rows(train_rows, set(), mode="verifier"),
            "test_greedy_tsr": tsr_for_rows(eval_rows, set(), mode="verifier"),
        }
        fold_curve = greedy_bin_set(train_rows, eval_rows, fold_bins, budgets, fold_baseline)
        for row in fold_curve:
            row = dict(row)
            row["fold"] = fold_index
            row["eval_episodes"] = len({r["episode_key"] for r in eval_rows})
            by_budget[float(row["budget"])].append(row)
    out = []
    for budget in budgets:
        fold_rows = by_budget.get(float(budget), [])
        if not fold_rows:
            out.append({"budget": budget, "mode": "cv_on_test", "folds": 0, "test_cost": 0.0, "test_measured_tsr": None, "test_measured_lift": None, "selected_bins": []})
            continue
        total_weight = sum(max(1, int(row.get("eval_episodes") or 1)) for row in fold_rows)
        def weighted(field: str) -> Optional[float]:
            vals = [(row.get(field), max(1, int(row.get("eval_episodes") or 1))) for row in fold_rows if row.get(field) is not None]
            if not vals:
                return None
            denom = sum(weight for _, weight in vals)
            return sum(float(value) * weight for value, weight in vals) / denom
        selected_counter: Counter[str] = Counter()
        for row in fold_rows:
            selected_counter.update(row.get("selected_bins") or [])
        out.append({
            "budget": budget,
            "mode": "cv_on_test",
            "folds": len(fold_rows),
            "test_cost": weighted("test_cost"),
            "test_measured_tsr": weighted("test_measured_tsr"),
            "test_measured_lift": weighted("test_measured_lift"),
            "selected_bins": [item for item, _ in selected_counter.most_common(10)],
            "fold_details": fold_rows,
            "eval_episode_weight": total_weight,
        })
    return out


def fmt_pct(value: Optional[float]) -> str:
    return "NA" if value is None else f"{100.0 * value:.2f}%"


def fmt_num(value: Optional[float]) -> str:
    return "NA" if value is None else f"{value:.4f}"


def render_report(summary: Mapping[str, Any], output_dir: Path) -> str:
    lines = ["# High-Lift Observable Bin Identification", ""]
    lines.append("Bins are defined only from inference-observable features. Measured lift requires full-step verifier outcomes; missing coverage is reported as pending rather than replaced by oracle labels.")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- train candidates: `{summary['inputs']['train_candidates']}`")
    lines.append(f"- test candidates: `{summary['inputs']['test_candidates']}`")
    lines.append(f"- train rows: `{summary['dataset']['train_rows']}`; test rows: `{summary['dataset']['test_rows']}`")
    lines.append(f"- test verifier availability: `{summary['dataset']['test_verifier_available_fraction']*100:.2f}%`")
    lines.append(f"- test score-set verifier availability: `{summary['dataset'].get('test_score_set_verifier_available_fraction', 0.0)*100:.2f}%` over `{summary['dataset'].get('test_score_set_steps', 0)}` greedy-wrong recoverable steps")
    lines.append(f"- strict split: `{summary['dataset']['strict_train_test']}`")
    lines.append(f"- bin-set selection mode: `{summary['dataset'].get('selection_mode')}`")
    lines.append("- analysis-time semantics: verifier is applied only to greedy-wrong bin steps; greedy-correct and MISSING steps are filled deterministically. Deployment damage on greedy-correct bin steps remains a flagged follow-up.")
    lines.append("")
    lines.append("## Baselines")
    lines.append("")
    base = summary["baselines"]
    lines.append("| split | greedy measured TSR | projected baseline | complete episodes |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| TRAIN | {fmt_pct(base.get('train_greedy_tsr'))} | {fmt_pct(base.get('train_projected_baseline'))} | {base.get('train_complete_episodes', 0)} |")
    lines.append(f"| TEST | {fmt_pct(base.get('test_greedy_tsr'))} | {fmt_pct(base.get('test_projected_baseline'))} | {base.get('test_complete_episodes', 0)} |")
    lines.append("")
    lines.append("## Ranked Single Bins")
    lines.append("")
    lines.append("| rank | bin | test cost | verifier avail | measured lift | lift/cost | projected lift | oracle lift | Jaccard vs bottom-p_i |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for idx, row in enumerate(summary["top_bins"][:30], 1):
        lines.append(
            f"| {idx} | `{row['bin_id']}` | {fmt_pct(row.get('test_cost'))} | {fmt_pct(row.get('test_verifier_available_fraction'))} | "
            f"{fmt_pct(row.get('test_measured_lift'))} | {fmt_num(row.get('test_lift_per_cost'))} | {fmt_pct(row.get('test_projected_lift'))} | "
            f"{fmt_pct(row.get('test_oracle_measured_lift'))} | {fmt_pct(row.get('jaccard_vs_bottom_p_i'))} |"
        )
    lines.append("")
    lines.append("## Best Bin Set By Budget")
    lines.append("")
    lines.append("| budget | actual cost | measured TSR | measured lift | bins |")
    lines.append("|---:|---:|---:|---:|---|")
    for row in summary["budget_curve"]:
        lines.append(f"| {row['budget']*100:.0f}% | {fmt_pct(row.get('test_cost'))} | {fmt_pct(row.get('test_measured_tsr'))} | {fmt_pct(row.get('test_measured_lift'))} | `{'; '.join(row.get('selected_bins') or [])}` |")
    lines.append("")
    lines.append("## Leakage Audit")
    lines.append("")
    lines.append("- bin definitions use observable features only: greedy predicted type/category, step position, task length, sampling disagreement, verifier score stats when present, and screen complexity.")
    lines.append("- p_i / GT / matcher correctness are excluded from bin definitions; p_i appears only in projection and bottom-p_i comparison.")
    lines.append("- real operational lift uses actual verifier outcomes only when full-step verifier coverage exists; oracle lift is reported separately as ceiling.")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(f"**{summary['gate']['verdict']}**")
    lines.append("")
    lines.append(summary["gate"]["reason"])
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{output_dir / 'binlift.md'}`")
    lines.append(f"- `{output_dir / 'summary.json'}`")
    lines.append(f"- `{output_dir / 'per_bin.jsonl'}`")
    lines.append(f"- `{output_dir / 'per_step.jsonl'}`")
    lines.append("")
    lines.append("STOP for review.")
    return "\n".join(lines) + "\n"


def decide_gate(summary: Mapping[str, Any]) -> Dict[str, str]:
    train_required = summary["dataset"].get("train_score_set_steps", 0) > 0
    test_required = summary["dataset"].get("test_score_set_steps", 0) > 0
    train_ok = (not train_required) or summary["dataset"].get("train_score_set_verifier_available_fraction", 0.0) >= 0.99
    test_ok = (not test_required) or summary["dataset"].get("test_score_set_verifier_available_fraction", 0.0) >= 0.99
    if not (train_ok and test_ok):
        return {
            "verdict": "PENDING_MISSING_FULL_VERIFIER_OUTCOMES",
            "reason": "Observable bins were built, but real measured lift cannot be computed because score-set verifier outcomes are missing. Greedy-correct and MISSING rows are deterministic under the pinned analysis-time semantics.",
        }
    curve20 = next((row for row in summary["budget_curve"] if abs(row["budget"] - 0.20) < 1e-6), None)
    full = next((row for row in summary["budget_curve"] if abs(row["budget"] - 1.00) < 1e-6), None)
    if curve20 and full and curve20.get("test_measured_lift") is not None and full.get("test_measured_lift") is not None and full["test_measured_lift"] > 0:
        captured = curve20["test_measured_lift"] / full["test_measured_lift"]
        if captured >= 0.6 and curve20.get("test_cost", 1.0) <= 0.30:
            return {"verdict": "OPERATIONAL CRITICAL-STEP BIN FOUND", "reason": "A train-picked observable bin set captures most full-verification lift at a fraction of the cost."}
    return {"verdict": "NO USEFUL BIN", "reason": "Measured lift is not concentrated in an observable bin set enough to justify targeted verification."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-candidates", default="")
    parser.add_argument("--test-candidates", default=DEFAULT_TEST_CANDIDATES)
    parser.add_argument("--train-data", default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--test-data", default=DEFAULT_TEST_DATA)
    parser.add_argument("--train-tasks", default="")
    parser.add_argument("--test-tasks", default=DEFAULT_TEST_TASKS)
    parser.add_argument("--train-verifier-root", default="")
    parser.add_argument("--test-verifier-root", default="")
    parser.add_argument("--strict-summary", default=DEFAULT_STRICT_SUMMARY)
    parser.add_argument("--n-candidates", type=int, default=50)
    parser.add_argument("--min-train-steps", type=int, default=10)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows, train_manifest = build_split_rows(
        split="train",
        candidates_path=args.train_candidates,
        data_path=args.train_data,
        tasks_path=args.train_tasks,
        verifier_root=args.train_verifier_root,
        n_candidates=args.n_candidates,
        strict_summary=args.strict_summary,
    )
    test_rows, test_manifest = build_split_rows(
        split="test",
        candidates_path=args.test_candidates,
        data_path=args.test_data,
        tasks_path=args.test_tasks,
        verifier_root=args.test_verifier_root,
        n_candidates=args.n_candidates,
        strict_summary=args.strict_summary,
    )
    strict_split = bool(train_rows and test_rows and not ({row["episode_key"] for row in train_rows} & {row["episode_key"] for row in test_rows}))
    selection_rows = train_rows if train_rows else test_rows
    all_rows = [dict(row) for row in train_rows + test_rows]
    bins = build_bins(selection_rows, all_rows, args.min_train_steps, selection_split="train" if train_rows else "test")
    # Push bin membership changes back into split row objects.
    membership_by_id = {row["row_id"]: row.get("bin_memberships", []) for row in all_rows}
    for row in train_rows + test_rows:
        row["bin_memberships"] = membership_by_id.get(row["row_id"], [])
    baselines = {
        "train_greedy_tsr": tsr_for_rows(train_rows, set(), mode="verifier") if train_rows else None,
        "test_greedy_tsr": tsr_for_rows(test_rows, set(), mode="verifier") if test_rows else None,
        "train_projected_baseline": projected_tsr(train_rows, set(), mode="real") if train_rows else None,
        "test_projected_baseline": projected_tsr(test_rows, set(), mode="real") if test_rows else None,
        "train_complete_episodes": len(complete_episode_keys(train_rows)),
        "test_complete_episodes": len(complete_episode_keys(test_rows)),
    }
    scored_bins = []
    bin_items = list(bins.values())
    for item in bin_items:
        scored = score_bin(item, train_rows, test_rows, baselines)
        scored["_train_ids"] = list(item.get("train_ids", set()))
        scored["_test_ids"] = list(item.get("test_ids", set()))
        scored_bins.append(scored)
    ranked = sorted(scored_bins, key=lambda row: (row.get("test_lift_per_cost") if row.get("test_lift_per_cost") is not None else -1e9, row.get("test_projected_lift") if row.get("test_projected_lift") is not None else -1e9), reverse=True)
    if train_rows and test_rows:
        budget_curve = greedy_bin_set(train_rows, test_rows, ranked, BUDGETS, baselines)
        selection_mode = "strict_train_test"
    elif test_rows:
        budget_curve = cv_budget_curve(test_rows, ranked, BUDGETS, n_folds=args.cv_folds, seed=args.seed)
        selection_mode = "cv_on_test"
    else:
        budget_curve = []
        selection_mode = "unavailable"
    summary = {
        "inputs": {
            "train_candidates": args.train_candidates,
            "test_candidates": args.test_candidates,
            "train_verifier_root": args.train_verifier_root,
            "test_verifier_root": args.test_verifier_root,
        },
        "dataset": {
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "train_verifier_available_fraction": sum(1 for row in train_rows if row.get("verifier_available")) / len(train_rows) if train_rows else 0.0,
            "test_verifier_available_fraction": sum(1 for row in test_rows if row.get("verifier_available")) / len(test_rows) if test_rows else 0.0,
            "train_score_set_steps": sum(1 for row in train_rows if row.get("score_set_requires_verifier")),
            "test_score_set_steps": sum(1 for row in test_rows if row.get("score_set_requires_verifier")),
            "train_score_set_verifier_available_fraction": (sum(1 for row in train_rows if row.get("score_set_requires_verifier") and row.get("verifier_available")) / sum(1 for row in train_rows if row.get("score_set_requires_verifier"))) if any(row.get("score_set_requires_verifier") for row in train_rows) else 0.0,
            "test_score_set_verifier_available_fraction": (sum(1 for row in test_rows if row.get("score_set_requires_verifier") and row.get("verifier_available")) / sum(1 for row in test_rows if row.get("score_set_requires_verifier"))) if any(row.get("score_set_requires_verifier") for row in test_rows) else 0.0,
            "strict_train_test": strict_split,
            "train_manifest": train_manifest,
            "test_manifest": test_manifest,
            "n_bins": len(scored_bins),
            "selection_mode": selection_mode,
        },
        "baselines": baselines,
        "top_bins": [{k: v for k, v in row.items() if not k.startswith("_")} for row in ranked[:100]],
        "budget_curve": budget_curve,
    }
    summary["gate"] = decide_gate(summary)
    write_jsonl(output_dir / "per_step.jsonl", [
        {
            "row_id": row["row_id"],
            "split": row["split"],
            "episode_id": row["episode_id"],
            "step_idx": row["step_idx"],
            "greedy_success": row["greedy_success"],
            "verifier_available": row["verifier_available"],
            "verifier_success": row["verifier_success"],
            "verifier_helped": row["verifier_helped"],
            "score_set_requires_verifier": row["score_set_requires_verifier"],
            "deterministic_skip_reason": row["deterministic_skip_reason"],
            "bottom_p_i_compare_only": row["bottom_p_i_compare_only"],
            "p_i_heldout_compare_only": row["p_i_heldout_compare_only"],
            "features": row["features"],
            "bin_memberships": row.get("bin_memberships", []),
        }
        for row in train_rows + test_rows
    ])
    write_jsonl(output_dir / "per_bin.jsonl", [{k: v for k, v in row.items() if not k.startswith("_")} for row in ranked])
    write_json(output_dir / "summary.json", summary)
    (output_dir / "binlift.md").write_text(render_report(summary, output_dir), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "bins": len(scored_bins),
        "test_verifier_available_fraction": summary["dataset"]["test_verifier_available_fraction"],
        "gate": summary["gate"]["verdict"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()