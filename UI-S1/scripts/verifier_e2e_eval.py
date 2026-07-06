#!/usr/bin/env python3
"""Compute teacher-forced full-trajectory TSR from E2E candidate pools."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.combine_critstep_verifier_v2 import aggregate_pick, attach_scores  # noqa: E402


DEFAULT_CANDIDATES = "outputs/verifier_e2e/slice200/candidates/per_step.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/verifier_e2e"
DEFAULT_CRITICAL_POOL = "outputs/critstep_elicit_uia/per_step.jsonl"
DEFAULT_STRICT_SUMMARY = "outputs/critstep_verifier_v2/strict/combine/strict_summary.json"
N_SWEEP = (5, 10, 20, 50)


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


def expand_paths(values: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for value in values:
        matches = sorted(glob.glob(value))
        paths.extend(Path(path) for path in (matches or [value]))
    seen = set()
    unique = []
    for path in paths:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def candidate_index(row: Mapping[str, Any], candidate_id: str) -> Optional[Dict[str, Any]]:
    for candidate in row.get("candidates", []):
        if str(candidate.get("candidate_id")) == str(candidate_id):
            return dict(candidate)
    return None


def prefix_candidates(row: Mapping[str, Any], n: int) -> List[Dict[str, Any]]:
    candidates = row.get("candidates") if isinstance(row.get("candidates"), list) else []
    return [dict(candidate) for candidate in candidates[:n]]


def action_without_coords(action: Mapping[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in action.items():
        if key in {"coordinate", "endCoordinate", "start_coordinate", "end_coordinate"}:
            continue
        out[key] = value
    return out


def majority_key(candidate: Mapping[str, Any]) -> str:
    action = candidate.get("action") if isinstance(candidate.get("action"), dict) else {}
    control = candidate.get("control") if isinstance(candidate.get("control"), dict) else {}
    pred_type = str(candidate.get("pred_type") or action.get("action") or "")
    control_key = control.get("key")
    if control_key:
        payload = {"type": pred_type, "control_key": control_key, "action": action_without_coords(action)}
    else:
        coord = action.get("coordinate")
        rounded_coord = None
        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
            try:
                rounded_coord = [round(float(coord[0]) / 25.0) * 25, round(float(coord[1]) / 25.0) * 25]
            except (TypeError, ValueError):
                rounded_coord = None
        payload = {"type": pred_type, "coord25": rounded_coord, "action": action_without_coords(action)}
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def select_greedy(row: Mapping[str, Any], n: int, ctx: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    candidates = prefix_candidates(row, n)
    return (candidates[0] if candidates else None), {"selector_available": bool(candidates)}


def select_oracle(row: Mapping[str, Any], n: int, ctx: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    candidates = prefix_candidates(row, n)
    correct = [candidate for candidate in candidates if candidate.get("is_correct")]
    if correct:
        return correct[0], {"selector_available": True, "oracle_used": True}
    return (candidates[0] if candidates else None), {"selector_available": bool(candidates), "oracle_used": False}


def select_majority(row: Mapping[str, Any], n: int, ctx: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    candidates = prefix_candidates(row, n)
    if not candidates:
        return None, {"selector_available": False}
    first_by_key: Dict[str, Dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for candidate in candidates:
        key = majority_key(candidate)
        counts[key] += 1
        first_by_key.setdefault(key, candidate)
    best_key, best_count = max(counts.items(), key=lambda item: (item[1], -candidates.index(first_by_key[item[0]])))
    return first_by_key[best_key], {"selector_available": True, "modal_count": best_count, "modal_key": best_key}


def select_logprob(row: Mapping[str, Any], n: int, ctx: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    candidates = prefix_candidates(row, n)
    scored = [candidate for candidate in candidates if candidate.get("model_logprob_avg") is not None]
    if not candidates:
        return None, {"selector_available": False, "logprob_available": False}
    if not scored:
        return candidates[0], {"selector_available": False, "logprob_available": False, "fallback": "greedy"}
    best = max(scored, key=lambda candidate: (float(candidate.get("model_logprob_avg")), -int(candidate.get("sample_rank") or 0)))
    return best, {"selector_available": True, "logprob_available": True}


def select_verifier(row: Mapping[str, Any], n: int, ctx: Mapping[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    verifier_by_n = ctx.get("verifier_by_n") if isinstance(ctx.get("verifier_by_n"), dict) else {}
    verifier = verifier_by_n.get(n)
    if verifier is None:
        candidates = prefix_candidates(row, n)
        return (candidates[0] if candidates else None), {"selector_available": False, "fallback": "greedy", "reason": "missing verifier stage1/stage2 files"}
    target_id = str(row.get("target_id"))
    item = verifier.get(target_id)
    if item is None:
        candidates = prefix_candidates(row, n)
        return (candidates[0] if candidates else None), {"selector_available": False, "fallback": "greedy", "reason": "target missing verifier score"}
    selected_id = item.get("candidate_id")
    selected = candidate_index(row, str(selected_id))
    if selected is None:
        candidates = prefix_candidates(row, n)
        selected = candidates[0] if candidates else None
        return selected, {"selector_available": False, "fallback": "greedy", "reason": "verifier candidate id not in candidate pool"}
    meta = dict(item)
    meta["selector_available"] = True
    return selected, meta


SELECTORS = {
    "greedy": select_greedy,
    "bon_majority": select_majority,
    "bon_logprob": select_logprob,
    "bon_verifier": select_verifier,
    "oracle": select_oracle,
}


def load_critical_tags(path: Path, temperature: float) -> Dict[Tuple[str, int], Dict[str, Any]]:
    if not path.exists():
        return {}
    tags: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in read_jsonl(path):
        if str(row.get("population")) != "critical":
            continue
        try:
            row_temperature = float(row.get("temperature"))
        except (TypeError, ValueError):
            continue
        if row_temperature != temperature:
            continue
        key = (str(row.get("episode_id")), int(row.get("step_idx")))
        tags[key] = {"critical": True, "critical_recoverable_at50": bool(row.get("recoverable")), "critical_missing_at50": not bool(row.get("recoverable"))}
    return tags


def strict_weight(path: Path) -> float:
    if not path.exists():
        return 0.0
    data = json.loads(path.read_text(encoding="utf-8"))
    value = data.get("models", {}).get("scalar_weight", {}).get("weight_stage1")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def load_verifier_for_n(verifier_root: Path, n: int, weight_stage1: float) -> Optional[Dict[str, Dict[str, Any]]]:
    stage1_path = verifier_root / f"n{n}" / "stage1" / "stage1_per_step.jsonl"
    stage2_path = verifier_root / f"n{n}" / "stage2" / "stage2_per_step.jsonl"
    if not stage1_path.exists() or not stage2_path.exists():
        return None
    stage1_rows = {str(row["target_id"]): row for row in read_jsonl(stage1_path)}
    stage2_rows = {str(row["target_id"]): row for row in read_jsonl(stage2_path)}
    out: Dict[str, Dict[str, Any]] = {}
    for target_id, stage1_row in stage1_rows.items():
        stage2_row = stage2_rows.get(target_id)
        if stage2_row is None:
            continue
        candidates = attach_scores(stage1_row, stage2_row)
        pick = aggregate_pick(candidates, weight_stage1)
        out[target_id] = {
            "candidate_id": pick.get("representative_candidate_id"),
            "distinct_key": pick.get("distinct_key"),
            "aggregate_score": pick.get("aggregate_score"),
            "weight_stage1": weight_stage1,
            "stage2_net_wins": pick.get("stage2_net_wins"),
            "stage1_score": pick.get("stage1_score"),
        }
    return out


def load_verifiers(verifier_root: str, n_sweep: Sequence[int], weight_stage1: float) -> Dict[int, Dict[str, Dict[str, Any]]]:
    if not verifier_root:
        return {}
    root = Path(verifier_root)
    out = {}
    for n in n_sweep:
        loaded = load_verifier_for_n(root, n, weight_stage1)
        if loaded is not None:
            out[n] = loaded
    return out


def evaluate_method(rows: Sequence[Mapping[str, Any]], method: str, n: int, ctx: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    selector = SELECTORS[method]
    by_episode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_steps = 0
    correct_steps = 0
    total_reward = 0.0
    available_steps = 0
    logprob_available_steps = 0
    recoverable_steps = 0
    recoverable_selected_correct = 0
    missing_steps = 0
    missing_selected_reward = 0.0
    missing_greedy_reward = 0.0
    missing_hurt = 0
    missing_helped = 0
    critical_steps = 0
    critical_recoverable_steps = 0
    critical_missing_steps = 0
    for row in rows:
        selected, meta = selector(row, n, ctx)
        candidates = prefix_candidates(row, n)
        greedy = candidates[0] if candidates else {}
        success = bool(selected and selected.get("is_correct"))
        reward = float(selected.get("reward") if selected else 0.0)
        greedy_reward = float(greedy.get("reward") or 0.0)
        pool_recoverable = any(bool(candidate.get("is_correct")) for candidate in candidates)
        pool_missing = not pool_recoverable
        tag = ctx.get("critical_tags", {}).get((str(row.get("episode_id")), int(row.get("step_idx") or 0)), {})
        critical = bool(tag.get("critical"))
        total_steps += 1
        correct_steps += int(success)
        total_reward += reward
        available_steps += int(bool(meta.get("selector_available")))
        logprob_available_steps += int(bool(meta.get("logprob_available")))
        recoverable_steps += int(pool_recoverable)
        recoverable_selected_correct += int(pool_recoverable and success)
        missing_steps += int(pool_missing)
        if pool_missing:
            missing_selected_reward += reward
            missing_greedy_reward += greedy_reward
            missing_hurt += int(reward < greedy_reward)
            missing_helped += int(reward > greedy_reward)
        critical_steps += int(critical)
        critical_recoverable_steps += int(critical and bool(tag.get("critical_recoverable_at50")))
        critical_missing_steps += int(critical and bool(tag.get("critical_missing_at50")))
        by_episode[str(row.get("episode_id"))].append({
            "step_idx": int(row.get("step_idx") or 0),
            "selected_candidate_id": selected.get("candidate_id") if selected else None,
            "selected_source": selected.get("source") if selected else None,
            "selected_success": success,
            "selected_reward": reward,
            "greedy_reward": greedy_reward,
            "pool_recoverable": pool_recoverable,
            "pool_missing": pool_missing,
            "critical": critical,
            "critical_recoverable_at50": bool(tag.get("critical_recoverable_at50")),
            "critical_missing_at50": bool(tag.get("critical_missing_at50")),
            "selector_available": bool(meta.get("selector_available")),
            "selector_meta": meta,
        })
    task_success = 0
    progress_sum = 0.0
    for episode_steps in by_episode.values():
        episode_steps.sort(key=lambda item: item["step_idx"])
        first_error = next((idx for idx, step in enumerate(episode_steps, 1) if not step["selected_success"]), None)
        if first_error is None:
            task_success += 1
            progress_sum += 1.0
        else:
            progress_sum += (first_error - 1) / len(episode_steps) if episode_steps else 0.0
    num_episodes = len(by_episode)
    metrics = {
        "method": method,
        "n": n,
        "num_episodes": num_episodes,
        "total_steps": total_steps,
        "task_success": task_success,
        "tsr": task_success / num_episodes if num_episodes else 0.0,
        "step_sr": correct_steps / total_steps if total_steps else 0.0,
        "mean_reward": total_reward / total_steps if total_steps else 0.0,
        "avg_progress": progress_sum / num_episodes if num_episodes else 0.0,
        "selector_available_steps": available_steps,
        "selector_available_fraction": available_steps / total_steps if total_steps else 0.0,
        "logprob_available_steps": logprob_available_steps,
        "recoverable_steps": recoverable_steps,
        "recoverable_selection_accuracy": recoverable_selected_correct / recoverable_steps if recoverable_steps else None,
        "missing_steps": missing_steps,
        "missing_selected_mean_reward": missing_selected_reward / missing_steps if missing_steps else None,
        "missing_greedy_mean_reward": missing_greedy_reward / missing_steps if missing_steps else None,
        "missing_reward_delta_vs_greedy": (missing_selected_reward - missing_greedy_reward) / missing_steps if missing_steps else None,
        "missing_hurt_steps": missing_hurt,
        "missing_helped_steps": missing_helped,
        "critical_steps_tagged": critical_steps,
        "critical_recoverable_at50_steps_tagged": critical_recoverable_steps,
        "critical_missing_at50_steps_tagged": critical_missing_steps,
    }
    return metrics, by_episode


def merge_per_episode(method_steps: Mapping[Tuple[str, int], Dict[str, List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    episode_ids = sorted({episode_id for by_episode in method_steps.values() for episode_id in by_episode})
    rows = []
    for episode_id in episode_ids:
        item: Dict[str, Any] = {"episode_id": episode_id, "methods": {}}
        for (method, n), by_episode in sorted(method_steps.items(), key=lambda pair: (pair[0][1], pair[0][0])):
            steps = by_episode.get(episode_id, [])
            first_error = next((step["step_idx"] for step in steps if not step["selected_success"]), None)
            item["methods"][f"{method}@N{n}"] = {
                "task_success": first_error is None if steps else False,
                "first_error_step_idx": first_error,
                "steps": steps,
            }
        rows.append(item)
    return rows


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.2f}%"


def fmt_pp(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.2f}pp"


def metric_available(item: Mapping[str, Any]) -> bool:
    method = str(item.get("method") or "")
    if method in {"bon_logprob", "bon_verifier"}:
        return float(item.get("selector_available_fraction") or 0.0) > 0.0
    return True


def render_report(metrics: Sequence[Mapping[str, Any]], output_dir: Path, args: argparse.Namespace, verifier_by_n: Mapping[int, Any], weight_stage1: float) -> str:
    by_key = {(str(item["method"]), int(item["n"])): item for item in metrics}
    lines = ["# End-to-End Verifier TSR", ""]
    lines.append("Teacher-forced static-data evaluation: each step uses the GT screen/history, the selector chooses from sampled candidates, the frozen matcher scores the selected action, and stop-on-error defines TSR. This is not autonomous rollout; online/KVM is required for that.")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append("| N | method | TSR | StepSR | Avg progress | ΔTSR vs greedy | ΔTSR vs best no-verifier | selector available |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for n in args.n_sweep:
        greedy = by_key.get(("greedy", n), {})
        no_verifier = [by_key.get(("bon_majority", n)), by_key.get(("bon_logprob", n))]
        no_verifier = [item for item in no_verifier if item and item.get("selector_available_fraction", 0.0) > 0.0]
        best_no_verifier = max((float(item.get("tsr") or 0.0) for item in no_verifier), default=None)
        for method in ("greedy", "bon_majority", "bon_logprob", "bon_verifier", "oracle"):
            item = by_key.get((method, n))
            if not item:
                continue
            if metric_available(item):
                delta_greedy = float(item.get("tsr") or 0.0) - float(greedy.get("tsr") or 0.0)
                delta_fair = None if best_no_verifier is None or method != "bon_verifier" else float(item.get("tsr") or 0.0) - best_no_verifier
                tsr = fmt_pct(item.get("tsr"))
                step_sr = fmt_pct(item.get("step_sr"))
                progress = fmt_pct(item.get("avg_progress"))
            else:
                delta_greedy = None
                delta_fair = None
                tsr = step_sr = progress = "NA"
            lines.append(
                f"| {n} | {method} | {tsr} | {step_sr} | {progress} | "
                f"{fmt_pp(delta_greedy)} | {fmt_pp(delta_fair)} | {fmt_pct(item.get('selector_available_fraction'))} |"
            )
    lines.append("")
    lines.append("## Missing-Step Drag")
    lines.append("")
    lines.append("Pool MISSING means no correct candidate exists in the same prefix-N pool, so the selector must choose a wrong action. The key question is whether it scores worse than greedy on those steps.")
    lines.append("")
    lines.append("| N | method | missing steps | selected mean reward | greedy mean reward | reward delta | hurt | helped |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for n in args.n_sweep:
        for method in ("bon_majority", "bon_logprob", "bon_verifier"):
            item = by_key.get((method, n))
            if not item:
                continue
            if not metric_available(item):
                lines.append(f"| {n} | {method} | {item.get('missing_steps', 0)} | NA | NA | NA | NA | NA |")
                continue
            lines.append(
                f"| {n} | {method} | {item.get('missing_steps', 0)} | "
                f"{(item.get('missing_selected_mean_reward') if item.get('missing_selected_mean_reward') is not None else 0.0):.4f} | "
                f"{(item.get('missing_greedy_mean_reward') if item.get('missing_greedy_mean_reward') is not None else 0.0):.4f} | "
                f"{(item.get('missing_reward_delta_vs_greedy') if item.get('missing_reward_delta_vs_greedy') is not None else 0.0):+.4f} | "
                f"{item.get('missing_hurt_steps', 0)} | {item.get('missing_helped_steps', 0)} |"
            )
    lines.append("")
    lines.append("## Recoverable-Step Selection")
    lines.append("")
    lines.append("| N | method | recoverable steps | selection accuracy on recoverable |")
    lines.append("|---:|---|---:|---:|")
    for n in args.n_sweep:
        for method in ("bon_majority", "bon_logprob", "bon_verifier", "oracle"):
            item = by_key.get((method, n))
            if item:
                accuracy = fmt_pct(item.get('recoverable_selection_accuracy')) if metric_available(item) else "NA"
                lines.append(f"| {n} | {method} | {item.get('recoverable_steps', 0)} | {accuracy} |")
    lines.append("")
    lines.append("## Compute Cost")
    lines.append("")
    lines.append("| N | base candidates per step | extra base samples vs greedy | verifier status | strict aggregation weight_stage1 |")
    lines.append("|---:|---:|---:|---|---:|")
    for n in args.n_sweep:
        status = "available" if n in verifier_by_n else "pending"
        lines.append(f"| {n} | {n} | {max(0, n - 1)} | {status} | {weight_stage1:.2f} |")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    primary_n = max(args.n_sweep)
    verifier = by_key.get(("bon_verifier", primary_n))
    majority = by_key.get(("bon_majority", primary_n))
    logprob = by_key.get(("bon_logprob", primary_n))
    no_verifier_values = [float(item.get("tsr") or 0.0) for item in (majority, logprob) if item and item.get("selector_available_fraction", 0.0) > 0.0]
    if not verifier or verifier.get("selector_available_fraction", 0.0) <= 0.0:
        gate = "VERIFIER E2E PENDING"
        reason = "Candidate/no-verifier E2E metrics are ready, but Stage1/Stage2 verifier scores are not available for the primary N yet."
    else:
        best_no = max(no_verifier_values) if no_verifier_values else float(by_key.get(("greedy", primary_n), {}).get("tsr") or 0.0)
        fair_delta = float(verifier.get("tsr") or 0.0) - best_no
        if fair_delta >= 0.01:
            gate = "VERIFIER REAL E2E GAIN"
            reason = f"At N={primary_n}, verifier TSR exceeds the best compute-matched no-verifier selector by {fair_delta*100:.2f}pp."
        elif fair_delta >= -0.005:
            gate = "VERIFIER = MORE SAMPLING"
            reason = f"At N={primary_n}, verifier TSR is within {fair_delta*100:.2f}pp of the best compute-matched no-verifier selector."
        else:
            gate = "PROJECTION DID NOT REALIZE"
            reason = f"At N={primary_n}, verifier TSR is {fair_delta*100:.2f}pp below the best compute-matched no-verifier selector."
    lines.append(f"**{gate}**")
    lines.append("")
    lines.append(reason)
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{output_dir / 'e2e_tsr.md'}`")
    lines.append(f"- `{output_dir / 'e2e_summary.json'}`")
    lines.append(f"- `{output_dir / 'per_episode.jsonl'}`")
    lines.append("")
    lines.append("STOP for review.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-files", nargs="+", default=[DEFAULT_CANDIDATES])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-sweep", type=lambda value: [int(item.strip()) for item in value.split(",") if item.strip()], default=list(N_SWEEP))
    parser.add_argument("--critical-pool", default=DEFAULT_CRITICAL_POOL)
    parser.add_argument("--critical-temperature", type=float, default=0.7)
    parser.add_argument("--verifier-root", default="")
    parser.add_argument("--strict-summary", default=DEFAULT_STRICT_SUMMARY)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for path in expand_paths(args.candidate_files):
        rows.extend(read_jsonl(path))
    rows.sort(key=lambda row: (int(row.get("episode_order") or 0), int(row.get("step_idx") or 0), str(row.get("target_id"))))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    critical_tags = load_critical_tags(Path(args.critical_pool), args.critical_temperature)
    weight_stage1 = strict_weight(Path(args.strict_summary))
    verifier_by_n = load_verifiers(args.verifier_root, args.n_sweep, weight_stage1)
    ctx = {"critical_tags": critical_tags, "verifier_by_n": verifier_by_n}

    metrics: List[Dict[str, Any]] = []
    method_steps: Dict[Tuple[str, int], Dict[str, List[Dict[str, Any]]]] = {}
    for n in args.n_sweep:
        for method in ("greedy", "bon_majority", "bon_logprob", "bon_verifier", "oracle"):
            item, by_episode = evaluate_method(rows, method, n, ctx)
            metrics.append(item)
            method_steps[(method, n)] = by_episode
    per_episode = merge_per_episode(method_steps)
    write_jsonl(output_dir / "per_episode.jsonl", per_episode)
    summary = {
        "candidate_files": [str(path) for path in expand_paths(args.candidate_files)],
        "rows": len(rows),
        "episodes": len({str(row.get("episode_id")) for row in rows}),
        "n_sweep": args.n_sweep,
        "strict_weight_stage1": weight_stage1,
        "verifier_available_n": sorted(verifier_by_n.keys()),
        "metrics": metrics,
        "teacher_forced_bound": "Static-data E2E uses GT screens/history at every step; it is not autonomous rollout.",
    }
    write_json(output_dir / "e2e_summary.json", summary)
    report = render_report(metrics, output_dir, args, verifier_by_n, weight_stage1)
    (output_dir / "e2e_tsr.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "rows": len(rows), "episodes": summary["episodes"], "verifier_available_n": summary["verifier_available_n"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()