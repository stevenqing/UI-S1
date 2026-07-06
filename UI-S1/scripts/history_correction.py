#!/usr/bin/env python3
"""Evaluate text-level history correction as an offline carry-blocking intervention.

H0/H3 reuse existing pred-history and GT-history evaluation outputs. H1/H2/BOTH
re-run the base model on GT screens with constructed history prefixes:
- H1 replaces wrong prior pred-history entries with GT actions.
- H2 replaces prior entries with the verifier-selected candidate action.
- BOTH uses H2 history and verifier selection for the current step.

The operational H2 policy is GT-free if the verifier map covers every prior step.
Coverage is reported explicitly because filtered verifier maps cannot measure
correct-to-wrong injection.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.carry_test_pred_vs_gt import gt_result_paths, load_eval_results  # noqa: E402
from scripts.verifier_e2e_eval import load_verifiers, strict_weight  # noqa: E402
from v13_gui_360.eval_gui360_template import _format_action_for_history, build_step_prompt, parse_tool_call  # noqa: E402
from v13_gui_360.reward import compute_step_reward  # noqa: E402

DEFAULT_TEST_DATA = "outputs/gui360_history_ab/original_eval/gui360_test_1000_balanced_uia.jsonl"
DEFAULT_GT_SUMMARY = "outputs/gui360_history_ab/original_sft_template_gt_history_merged_20260630/summary.json"
DEFAULT_PRED_RESULTS = "outputs/gui360_history_ab/original_sft_template_pred_history_full_20260701/eval_results_20260701_085620.json"
DEFAULT_CANDIDATES = "outputs/history_correction/n5_candidates/per_step.jsonl"
DEFAULT_VERIFIER_ROOT = "outputs/history_correction/verifier"
DEFAULT_POINTWISE_VERIFIER = "outputs/history_correction/verifier_pointwise_n5/per_step.jsonl"
DEFAULT_STRICT_SUMMARY = "outputs/critstep_verifier_v2/strict/combine/strict_summary.json"
DEFAULT_OUTPUT_DIR = "outputs/history_correction"
DEFAULT_MODEL = "gui360-fullparam-sft-step250"

write_lock = Lock()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with write_lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: Optional[float]) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{100.0 * float(value):.2f}%"


def pp(value: Optional[float]) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{100.0 * float(value):+.2f}pp"


def step_success(step: Mapping[str, Any]) -> bool:
    if "success" in step:
        return bool(step.get("success"))
    reward = step.get("reward")
    try:
        return float(reward) >= 0.5
    except (TypeError, ValueError):
        return False


def pred_action(step: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    action = step.get("pred_action")
    return dict(action) if isinstance(action, dict) else None


def action_signature(action: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(action, Mapping):
        return "null"
    payload = {}
    for key in sorted(action):
        value = action[key]
        if value is None:
            continue
        payload[key] = value
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def score_prediction(pred_text: str, gt_action: Mapping[str, Any], image_w: int, image_h: int) -> Dict[str, Any]:
    parsed = parse_tool_call(pred_text)
    fake_text = f"<action>{json.dumps(parsed, ensure_ascii=False)}</action>" if parsed else pred_text
    reward, info = compute_step_reward(fake_text, gt_action, image_w, image_h)
    return {
        "success": bool(reward >= 0.5),
        "reward": float(reward),
        "pred_action": info.get("pred_action") or parsed,
        "pred_type": info.get("pred_type"),
        "gt_type": info.get("gt_type"),
        "format_reward": info.get("format_reward", 0.0),
        "type_reward": info.get("type_reward", 0.0),
        "content_reward": info.get("content_reward", 0.0),
        "pred_text": pred_text[:700],
    }


def api_url_for(args: argparse.Namespace, key: str) -> str:
    urls = [item.strip().rstrip("/") for item in str(args.api_url).split(",") if item.strip()]
    if not urls:
        return str(args.api_url).rstrip("/")
    return urls[hash(key) % len(urls)]


def candidate_lookup(candidates_path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in read_jsonl(candidates_path):
        out[(str(row.get("episode_id")), int(row.get("step_idx") or 0))] = row
    return out


def load_aggregate_verifier_map(verifier_root: Path, n_candidates: int, strict_summary: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    weight = strict_weight(strict_summary) if strict_summary.exists() else 0.0
    by_target = load_verifiers(str(verifier_root), [n_candidates], weight).get(n_candidates, {})
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for target_id, item in by_target.items():
        parts = str(target_id).split(":")
        if len(parts) >= 3:
            episode_id = parts[-2]
            try:
                step_idx = int(parts[-1])
            except ValueError:
                continue
            out[(str(episode_id), step_idx)] = dict(item)
    return out


def load_pointwise_verifier_map(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in read_jsonl(path):
        episode_id = str(row.get("episode_id"))
        try:
            step_idx = int(row.get("step_idx") or 0)
        except (TypeError, ValueError):
            continue
        out[(episode_id, step_idx)] = {
            "candidate_id": row.get("verifier_candidate_id"),
            "verifier_score": row.get("verifier_score"),
            "verifier_correct": row.get("verifier_correct"),
            "verifier_kind": "pointwise_logits",
        }
    return out


def load_verifier_map(args: argparse.Namespace) -> Dict[Tuple[str, int], Dict[str, Any]]:
    if args.verifier_kind == "pointwise":
        return load_pointwise_verifier_map(Path(args.verifier_per_step))
    return load_aggregate_verifier_map(Path(args.verifier_root), args.n_candidates, Path(args.strict_summary))


def verifier_selected_candidate(candidate_row: Optional[Mapping[str, Any]], verifier_item: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidate_row or not verifier_item:
        return None
    candidate_id = str(verifier_item.get("candidate_id"))
    for candidate in candidate_row.get("candidates", []) or []:
        if str(candidate.get("candidate_id")) == candidate_id:
            return dict(candidate)
    return None


def load_baseline_results(gt_summary: Path, pred_results_path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    gt_results = load_eval_results(gt_result_paths(gt_summary))
    pred_results = load_eval_results([str(pred_results_path)])
    return gt_results, pred_results


def build_policy_actions(
    *,
    episodes: Mapping[str, Mapping[str, Any]],
    pred_results: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[Tuple[str, int], Mapping[str, Any]],
    verifier: Mapping[Tuple[str, int], Mapping[str, Any]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    by_episode: Dict[str, List[Dict[str, Any]]] = {}
    counters: Counter[str] = Counter()
    for episode_id, episode in episodes.items():
        steps = episode.get("steps") if isinstance(episode.get("steps"), list) else []
        pred_episode = pred_results.get(episode_id, {})
        pred_steps = pred_episode.get("steps") if isinstance(pred_episode.get("steps"), list) else []
        rows: List[Dict[str, Any]] = []
        for idx, step in enumerate(steps):
            gt_action = step.get("action") if isinstance(step.get("action"), dict) else {}
            pred_step = pred_steps[idx] if idx < len(pred_steps) and isinstance(pred_steps[idx], Mapping) else {}
            recorded = pred_action(pred_step)
            recorded_correct = step_success(pred_step)
            candidate_row = candidates.get((episode_id, idx))
            verifier_item = verifier.get((episode_id, idx))
            selected = verifier_selected_candidate(candidate_row, verifier_item)
            verifier_action = selected.get("action") if selected and isinstance(selected.get("action"), dict) else recorded
            verifier_correct = bool(selected.get("is_correct")) if selected else None
            verifier_available = selected is not None
            changed = action_signature(verifier_action) != action_signature(recorded)
            fix = bool(verifier_available and (not recorded_correct) and verifier_correct)
            inject = bool(verifier_available and recorded_correct and (not verifier_correct) and changed)
            wrong_to_wrong_change = bool(verifier_available and (not recorded_correct) and (not verifier_correct) and changed)
            correct_noop_or_correct_change = bool(verifier_available and recorded_correct and verifier_correct)
            counters["steps"] += 1
            counters["pred_recorded_correct"] += int(recorded_correct)
            counters["pred_recorded_wrong"] += int(not recorded_correct)
            counters["verifier_available"] += int(verifier_available)
            counters["verifier_changed"] += int(changed and verifier_available)
            counters["verifier_fix_wrong_to_correct"] += int(fix)
            counters["verifier_inject_correct_to_wrong"] += int(inject)
            counters["verifier_wrong_to_wrong_change"] += int(wrong_to_wrong_change)
            counters["verifier_keeps_or_changes_correct_to_correct"] += int(correct_noop_or_correct_change)
            rows.append({
                "episode_id": episode_id,
                "step_idx": idx,
                "gt_action": gt_action,
                "pred_action": recorded,
                "pred_recorded_correct": bool(recorded_correct),
                "verifier_available": bool(verifier_available),
                "verifier_candidate_id": selected.get("candidate_id") if selected else None,
                "verifier_action": verifier_action,
                "verifier_correct": verifier_correct,
                "verifier_changed_record": bool(changed and verifier_available),
                "verifier_fix_wrong_to_correct": fix,
                "verifier_inject_correct_to_wrong": inject,
                "verifier_wrong_to_wrong_change": wrong_to_wrong_change,
            })
        by_episode[episode_id] = rows
    summary = dict(counters)
    total = summary.get("steps", 0)
    summary["verifier_coverage"] = summary.get("verifier_available", 0) / total if total else 0.0
    summary["net_fixed_minus_injected"] = summary.get("verifier_fix_wrong_to_correct", 0) - summary.get("verifier_inject_correct_to_wrong", 0)
    summary["net_fixed_minus_injected_rate"] = summary["net_fixed_minus_injected"] / total if total else 0.0
    return by_episode, summary


def history_for(policy: str, policy_rows: Sequence[Mapping[str, Any]], target_idx: int) -> List[str]:
    history = []
    for idx in range(target_idx):
        row = policy_rows[idx]
        if policy == "H1_oracle_corrected":
            action = row["gt_action"] if not row.get("pred_recorded_correct") else row.get("pred_action")
        elif policy in {"H2_verifier_corrected", "BOTH_verifier_history_and_current"}:
            action = row.get("verifier_action") if row.get("verifier_available") else row.get("pred_action")
        else:
            action = row.get("pred_action")
        history.append(_format_action_for_history(action, idx + 1))
    return history


def build_targets(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    episodes = {str(row.get("episode_id")): row for row in read_jsonl(Path(args.test_data))}
    gt_results, pred_results = load_baseline_results(Path(args.gt_summary), Path(args.pred_results))
    candidates = candidate_lookup(Path(args.candidates))
    verifier = load_verifier_map(args)
    policy_rows, correction_summary = build_policy_actions(episodes=episodes, pred_results=pred_results, candidates=candidates, verifier=verifier)
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    targets: List[Dict[str, Any]] = []
    for episode_id in sorted(episodes, key=lambda value: int(value) if value.isdigit() else value):
        episode = episodes[episode_id]
        steps = episode.get("steps") if isinstance(episode.get("steps"), list) else []
        rows = policy_rows.get(episode_id, [])
        if not rows:
            continue
        for idx, step in enumerate(steps):
            for policy in policies:
                if policy in {"H0_pred_history", "H3_gt_history"}:
                    continue
                targets.append({
                    "condition_id": f"{episode_id}:{idx}:{policy}",
                    "episode_id": episode_id,
                    "step_idx": idx,
                    "policy": policy,
                    "instruction": episode.get("goal"),
                    "screenshot": step.get("screenshot"),
                    "gt_action": step.get("action"),
                    "image_w": int(step.get("image_w") or 1040),
                    "image_h": int(step.get("image_h") or 736),
                    "history": history_for(policy, rows, idx),
                    "prefix_len": idx,
                    "prefix_pred_errors": sum(1 for prior in rows[:idx] if not prior.get("pred_recorded_correct")),
                    "prefix_verifier_available": sum(1 for prior in rows[:idx] if prior.get("verifier_available")),
                    "prefix_verifier_fixes": sum(1 for prior in rows[:idx] if prior.get("verifier_fix_wrong_to_correct")),
                    "prefix_verifier_injections": sum(1 for prior in rows[:idx] if prior.get("verifier_inject_correct_to_wrong")),
                    "current_verifier_candidate_id": rows[idx].get("verifier_candidate_id"),
                    "current_verifier_action": rows[idx].get("verifier_action"),
                    "current_verifier_available": rows[idx].get("verifier_available"),
                    "current_verifier_correct": rows[idx].get("verifier_correct"),
                })
    manifest = {
        "test_data": args.test_data,
        "gt_summary": args.gt_summary,
        "pred_results": args.pred_results,
        "candidates": args.candidates,
        "verifier_root": args.verifier_root,
        "verifier_kind": args.verifier_kind,
        "verifier_per_step": args.verifier_per_step,
        "n_candidates": args.n_candidates,
        "strict_summary": args.strict_summary,
        "episodes": len(episodes),
        "baseline_gt_episodes": len(gt_results),
        "baseline_pred_episodes": len(pred_results),
        "target_rows": len(targets),
        "policies_to_infer": [policy for policy in policies if policy not in {"H0_pred_history", "H3_gt_history"}],
        "correction_summary": correction_summary,
        "operational_h2_is_full_coverage": correction_summary.get("verifier_available", 0) == correction_summary.get("steps", -1),
    }
    return targets, manifest


def done_ids(path: Path) -> set[str]:
    return {str(row.get("condition_id")) for row in read_jsonl(path) if row.get("condition_id")}


def eval_target(target: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if target["policy"] == "BOTH_verifier_history_and_current" and target.get("current_verifier_available"):
        selected_action = target.get("current_verifier_action") if isinstance(target.get("current_verifier_action"), dict) else None
        fake_text = f"<action>{json.dumps(selected_action, ensure_ascii=False)}</action>" if selected_action else ""
        scored = score_prediction(fake_text, target["gt_action"], int(target["image_w"]), int(target["image_h"]))
        return {k: v for k, v in target.items() if k != "history"} | scored | {"used_current_verifier_selection": True, "api_error_count": 0, "api_errors": []}
    client = OpenAI(base_url=api_url_for(args, str(target["condition_id"])), api_key="dummy", timeout=args.request_timeout)
    messages = build_step_prompt(
        str(target["instruction"]),
        str(target["screenshot"]),
        int(target["step_idx"]),
        list(target["history"]),
        image_max_pixels=args.image_max_pixels,
    )
    errors: List[str] = []
    pred_text = ""
    for attempt in range(args.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.model_name,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=0.0,
            )
            pred_text = response.choices[0].message.content or ""
            break
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc)[:300])
            if attempt < args.max_retries:
                time.sleep(min(10.0, 1.5 ** attempt))
    scored = score_prediction(pred_text, target["gt_action"], int(target["image_w"]), int(target["image_h"]))
    return {k: v for k, v in target.items() if k != "history"} | scored | {"used_current_verifier_selection": False, "api_error_count": len(errors), "api_errors": errors}


def run_eval(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets, manifest = build_targets(args)
    write_json(output_dir / "manifest.json", manifest)
    target_path = output_dir / "targets.jsonl"
    if args.force_targets or not target_path.exists():
        write_jsonl(target_path, targets)
    result_path = output_dir / "per_step.jsonl"
    completed = done_ids(result_path) if args.resume else set()
    work = [target for target in targets if str(target["condition_id"]) not in completed]
    print(json.dumps({"targets": len(targets), "completed": len(completed), "remaining": len(work), "manifest": manifest}, indent=2), flush=True)
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(eval_target, target, args): target for target in work}
        for index, future in enumerate(as_completed(futures), 1):
            target = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001
                row = {k: v for k, v in target.items() if k != "history"} | {"success": False, "reward": 0.0, "eval_error": str(exc)[:500]}
            append_jsonl(result_path, row)
            if index % 100 == 0 or index == len(work):
                print(f"completed {index}/{len(work)}", flush=True)
    summarize(args)


def baseline_rows_from_results(policy: str, results: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for episode_id, episode in results.items():
        for step in episode.get("steps", []) or []:
            out.append({
                "condition_id": f"{episode_id}:{step.get('step_idx')}:{policy}",
                "episode_id": str(episode_id),
                "step_idx": int(step.get("step_idx") or 0),
                "policy": policy,
                "success": step_success(step),
                "reward": float(step.get("reward") or 0.0),
                "pred_action": pred_action(step),
                "pred_type": step.get("pred_type"),
                "api_error_count": 0,
                "api_errors": [],
            })
    return out


def metric_for(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_episode: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[str(row.get("episode_id"))].append(row)
    task_success = 0
    progress_sum = 0.0
    total_steps = 0
    correct_steps = 0
    reward_sum = 0.0
    for steps in by_episode.values():
        ordered = sorted(steps, key=lambda item: int(item.get("step_idx") or 0))
        if not ordered:
            continue
        total_steps += len(ordered)
        correct_steps += sum(1 for step in ordered if step.get("success"))
        reward_sum += sum(float(step.get("reward") or 0.0) for step in ordered)
        first_error_pos = next((pos for pos, step in enumerate(ordered, 1) if not step.get("success")), None)
        if first_error_pos is None:
            task_success += 1
            progress_sum += 1.0
        else:
            progress_sum += (first_error_pos - 1) / len(ordered)
    n_ep = len(by_episode)
    return {
        "episodes": n_ep,
        "total_steps": total_steps,
        "correct_steps": correct_steps,
        "task_success": task_success,
        "tsr": task_success / n_ep if n_ep else 0.0,
        "step_sr": correct_steps / total_steps if total_steps else 0.0,
        "mean_reward": reward_sum / total_steps if total_steps else 0.0,
        "avg_progress": progress_sum / n_ep if n_ep else 0.0,
    }


def load_all_policy_rows(args: argparse.Namespace) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    gt_results, pred_results = load_baseline_results(Path(args.gt_summary), Path(args.pred_results))
    policies: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    policies.update({
        "H0_pred_history": baseline_rows_from_results("H0_pred_history", pred_results),
        "H3_gt_history": baseline_rows_from_results("H3_gt_history", gt_results),
    })
    inferred = read_jsonl(Path(args.output_dir) / "per_step.jsonl")
    for row in inferred:
        policies[str(row.get("policy"))].append(row)
    manifest_path = Path(args.output_dir) / "manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    return policies, manifest


def load_current_step_selection_metrics(args: argparse.Namespace) -> Dict[str, Any]:
    candidates = candidate_lookup(Path(args.candidates))
    verifier = load_verifier_map(args)
    rows_by_policy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for (episode_id, step_idx), cand_row in candidates.items():
        cands = cand_row.get("candidates", []) or []
        greedy = cands[0] if cands else {}
        selected = verifier_selected_candidate(cand_row, verifier.get((episode_id, step_idx)))
        rows_by_policy["current_greedy"].append({
            "episode_id": episode_id,
            "step_idx": step_idx,
            "success": bool(greedy.get("is_correct")),
            "reward": float(greedy.get("reward") or 0.0),
        })
        rows_by_policy["current_step_selection"].append({
            "episode_id": episode_id,
            "step_idx": step_idx,
            "success": bool(selected.get("is_correct")) if selected else bool(greedy.get("is_correct")),
            "reward": float((selected or greedy).get("reward") or 0.0),
            "selector_available": bool(selected),
        })
    return {name: metric_for(rows) for name, rows in rows_by_policy.items()} | {
        "verifier_available_steps": len(verifier),
        "candidate_rows": len(candidates),
        "selection_coverage": len(verifier) / len(candidates) if candidates else 0.0,
    }


def pair_delta(metrics: Mapping[str, Mapping[str, Any]], a: str, b: str, field: str) -> Optional[float]:
    if a not in metrics or b not in metrics:
        return None
    return float(metrics[a].get(field) or 0.0) - float(metrics[b].get(field) or 0.0)


def decide_gate(summary: Mapping[str, Any]) -> Dict[str, str]:
    oracle = summary.get("reclaim", {}).get("oracle_tsr")
    operational = summary.get("reclaim", {}).get("operational_tsr")
    net = summary.get("h2_net_effect", {}).get("net_fixed_minus_injected", 0)
    h2_full = bool((summary.get("manifest") or {}).get("operational_h2_is_full_coverage"))
    if oracle is not None and oracle >= 0.03:
        if operational is not None and operational >= 0.01 and net > 0 and h2_full:
            return {"verdict": "HISTORY-CORRECTION EFFECTIVE", "reason": "Oracle correction reclaims substantial TSR and full-coverage verifier correction is positive with more fixes than injections."}
        if operational is not None and operational >= 0.01 and net > 0:
            return {"verdict": "HISTORY-CORRECTION PARTIAL / COVERAGE-LIMITED", "reason": "Operational correction is positive, but verifier coverage is not full enough to satisfy the strict Trap-2 deployment test."}
        return {"verdict": "HISTORY-CORRECTION ORACLE-ONLY", "reason": "Oracle correction reclaims carry, but operational verifier correction is negligible, negative, or coverage-limited."}
    return {"verdict": "HISTORY-CORRECTION WEAK", "reason": "Even oracle text correction does not materially improve TSR over pred-history."}


def render_report(summary: Mapping[str, Any], output_dir: Path) -> str:
    lines: List[str] = ["# History-Correction to Block Error-Carry", ""]
    lines.append("Teacher-forced GT screens; only history text is edited. Frozen matcher, base model unchanged. H1 uses GT and is an oracle mechanism ceiling; H2 uses verifier-selected candidate actions and is operational only when verifier coverage is full.")
    lines.append("")
    lines.append("## TSR / StepSR")
    lines.append("")
    lines.append("| policy | TSR | StepSR | Avg progress | mean reward | task success | correct steps | total steps |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for policy in ["H0_pred_history", "H1_oracle_corrected", "H2_verifier_corrected", "BOTH_verifier_history_and_current", "H3_gt_history"]:
        item = (summary.get("metrics") or {}).get(policy)
        if not item:
            continue
        lines.append(f"| {policy} | {pct(item.get('tsr'))} | {pct(item.get('step_sr'))} | {pct(item.get('avg_progress'))} | {item.get('mean_reward', 0.0):.4f} | {item.get('task_success', 0)} | {item.get('correct_steps', 0)} | {item.get('total_steps', 0)} |")
    lines.append("")
    rec = summary.get("reclaim") or {}
    lines.append("## Reclaim")
    lines.append("")
    lines.append("| metric | TSR | StepSR |")
    lines.append("|---|---:|---:|")
    lines.append(f"| oracle reclaim H1-H0 | {pp(rec.get('oracle_tsr'))} | {pp(rec.get('oracle_step_sr'))} |")
    lines.append(f"| operational reclaim H2-H0 | {pp(rec.get('operational_tsr'))} | {pp(rec.get('operational_step_sr'))} |")
    lines.append(f"| fraction of oracle captured | {pct(rec.get('fraction_oracle_tsr'))} | {pct(rec.get('fraction_oracle_step_sr'))} |")
    lines.append("")
    net = summary.get("h2_net_effect") or {}
    lines.append("## H2 Net Effect")
    lines.append("")
    lines.append("| item | count | rate |")
    lines.append("|---|---:|---:|")
    total = max(1, int(net.get("steps", 0) or 0))
    for key in ["verifier_available", "verifier_changed", "verifier_fix_wrong_to_correct", "verifier_inject_correct_to_wrong", "verifier_wrong_to_wrong_change", "net_fixed_minus_injected"]:
        value = int(net.get(key, 0) or 0)
        lines.append(f"| {key} | {value} | {pct(value / total)} |")
    lines.append("")
    lines.append(f"Operational H2 full coverage: `{bool((summary.get('manifest') or {}).get('operational_h2_is_full_coverage'))}`; verifier coverage `{pct(net.get('verifier_coverage'))}`.")
    lines.append("")
    comp = summary.get("selection_comparison") or {}
    lines.append("## Same Verifier: Selection vs History Correction")
    lines.append("")
    lines.append("| use | TSR | StepSR | ΔTSR vs H0/greedy | coverage |")
    lines.append("|---|---:|---:|---:|---:|")
    h0 = (summary.get("metrics") or {}).get("H0_pred_history", {})
    h2 = (summary.get("metrics") or {}).get("H2_verifier_corrected", {})
    both = (summary.get("metrics") or {}).get("BOTH_verifier_history_and_current", {})
    current = comp.get("current_step_selection", {})
    greedy = comp.get("current_greedy", {})
    lines.append(f"| current-step selection | {pct(current.get('tsr'))} | {pct(current.get('step_sr'))} | {pp(float(current.get('tsr') or 0.0) - float(greedy.get('tsr') or 0.0))} | {pct(comp.get('selection_coverage'))} |")
    if h2:
        lines.append(f"| history correction | {pct(h2.get('tsr'))} | {pct(h2.get('step_sr'))} | {pp(float(h2.get('tsr') or 0.0) - float(h0.get('tsr') or 0.0))} | {pct(net.get('verifier_coverage'))} |")
    if both:
        lines.append(f"| both | {pct(both.get('tsr'))} | {pct(both.get('step_sr'))} | {pp(float(both.get('tsr') or 0.0) - float(h0.get('tsr') or 0.0))} | {pct(net.get('verifier_coverage'))} |")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    gate = summary.get("gate") or {}
    lines.append(f"**{gate.get('verdict', 'UNKNOWN')}**")
    lines.append("")
    lines.append(str(gate.get("reason", "")))
    lines.append("")
    lines.append("## Guardrails")
    lines.append("")
    lines.append("- H1 is oracle GT correction and is not deployable.")
    verifier_kind = str((summary.get("manifest") or {}).get("verifier_kind") or "aggregate")
    lines.append(f"- H2 verifier kind: `{verifier_kind}`. `aggregate` means Stage1/Stage2 weighted aggregation; `pointwise` means the earlier logits verifier used as a full-coverage operational proxy.")
    lines.append("- H2 is GT-free, but only counts as the strict operational Trap-2 test when verifier coverage is full; otherwise injection is under-measured.")
    lines.append("- Current-step selection and BOTH use the static GT-history candidate pool; they are compute-matched offline comparisons, not online resampling under corrected history.")
    lines.append("- Correction is a text edit on teacher-forced GT screens, so this is offline and does not require a world model or online environment.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{output_dir / 'correction.md'}`")
    lines.append(f"- `{output_dir / 'summary.json'}`")
    lines.append(f"- `{output_dir / 'per_step.jsonl'}`")
    lines.append(f"- `{output_dir / 'targets.jsonl'}`")
    lines.append(f"- `{output_dir / 'manifest.json'}`")
    lines.append("")
    lines.append("STOP for review.")
    return "\n".join(lines) + "\n"


def summarize(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    policies, manifest = load_all_policy_rows(args)
    metrics = {policy: metric_for(rows) for policy, rows in policies.items() if rows}
    h2_net = (manifest.get("correction_summary") or {}) if isinstance(manifest, Mapping) else {}
    reclaim = {
        "oracle_tsr": pair_delta(metrics, "H1_oracle_corrected", "H0_pred_history", "tsr"),
        "oracle_step_sr": pair_delta(metrics, "H1_oracle_corrected", "H0_pred_history", "step_sr"),
        "operational_tsr": pair_delta(metrics, "H2_verifier_corrected", "H0_pred_history", "tsr"),
        "operational_step_sr": pair_delta(metrics, "H2_verifier_corrected", "H0_pred_history", "step_sr"),
    }
    if reclaim["oracle_tsr"] and abs(reclaim["oracle_tsr"]) > 1e-12 and reclaim["operational_tsr"] is not None:
        reclaim["fraction_oracle_tsr"] = reclaim["operational_tsr"] / reclaim["oracle_tsr"]
    else:
        reclaim["fraction_oracle_tsr"] = None
    if reclaim["oracle_step_sr"] and abs(reclaim["oracle_step_sr"]) > 1e-12 and reclaim["operational_step_sr"] is not None:
        reclaim["fraction_oracle_step_sr"] = reclaim["operational_step_sr"] / reclaim["oracle_step_sr"]
    else:
        reclaim["fraction_oracle_step_sr"] = None
    selection = load_current_step_selection_metrics(args) if Path(args.candidates).exists() else {}
    summary = {
        "manifest": manifest,
        "metrics": metrics,
        "reclaim": reclaim,
        "h2_net_effect": h2_net,
        "selection_comparison": selection,
        "teacher_forced_bound": "GT screens at every target step; history is text only.",
        "greedy_baseline_reference_tsr": 0.222,
    }
    summary["gate"] = decide_gate(summary)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "correction.md").write_text(render_report(summary, output_dir), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics, "gate": summary["gate"]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-data", default=DEFAULT_TEST_DATA)
    parser.add_argument("--gt-summary", default=DEFAULT_GT_SUMMARY)
    parser.add_argument("--pred-results", default=DEFAULT_PRED_RESULTS)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--verifier-root", default=DEFAULT_VERIFIER_ROOT)
    parser.add_argument("--verifier-kind", default="aggregate", choices=["aggregate", "pointwise"])
    parser.add_argument("--verifier-per-step", default=DEFAULT_POINTWISE_VERIFIER)
    parser.add_argument("--strict-summary", default=DEFAULT_STRICT_SUMMARY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-candidates", type=int, default=5)
    parser.add_argument("--policies", default="H1_oracle_corrected,H2_verifier_corrected,BOTH_verifier_history_and_current")
    parser.add_argument("--api-url", default="http://127.0.0.1:8142/v1")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--image-max-pixels", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-targets", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.summarize_only:
        summarize(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
