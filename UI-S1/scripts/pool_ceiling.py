#!/usr/bin/env python3
"""Measure pool-limited oracle ceiling as candidate pool size grows."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_N50_POOL = "outputs/critstep_binlift_lean/test_candidates/per_step.jsonl"
DEFAULT_N5_POOL = "outputs/history_correction/n5_candidates/per_step.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/pool_ceiling"
DEFAULT_POOL_SIZES = (1, 5, 10, 20, 50)
DEFAULT_SELECTION_ACCURACIES = (0.35, 0.50, 0.73)
HISTORICAL_N5_ORACLE_GATE_TSR = 0.244


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def pct(value: Optional[float]) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{100.0 * float(value):.2f}%"


def pp(value: Optional[float]) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{100.0 * float(value):+.2f}pp"


def fmt(value: Optional[float]) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.3f}"


def row_key(row: Mapping[str, Any]) -> Tuple[str, int]:
    episode_key = str(row.get("episode_key") or row.get("episode_id"))
    return episode_key, int(row.get("step_idx") or 0)


def sort_key(row: Mapping[str, Any]) -> Tuple[int, str, int]:
    episode_order = row.get("episode_order")
    if episode_order is None:
        episode_id = str(row.get("episode_id"))
        episode_order = int(episode_id) if episode_id.isdigit() else 10**9
    return int(episode_order), str(row.get("episode_id")), int(row.get("step_idx") or 0)


def task_metrics(rows: Sequence[Mapping[str, Any]], success_values: Sequence[bool]) -> Dict[str, Any]:
    by_episode: Dict[str, List[Tuple[int, bool]]] = defaultdict(list)
    for row, success in zip(rows, success_values, strict=True):
        by_episode[str(row.get("episode_key") or row.get("episode_id"))].append((int(row.get("step_idx") or 0), bool(success)))
    task_success = 0
    progress_sum = 0.0
    for steps in by_episode.values():
        ordered = [success for _, success in sorted(steps)]
        first_error = next((idx for idx, ok in enumerate(ordered, 1) if not ok), None)
        if first_error is None:
            task_success += 1
            progress_sum += 1.0
        else:
            progress_sum += (first_error - 1) / len(ordered) if ordered else 0.0
    total = len(success_values)
    correct = sum(1 for ok in success_values if ok)
    episodes = len(by_episode)
    return {
        "episodes": episodes,
        "total_steps": total,
        "correct_steps": correct,
        "task_success": task_success,
        "tsr": task_success / episodes if episodes else 0.0,
        "step_sr": correct / total if total else 0.0,
        "avg_progress": progress_sum / episodes if episodes else 0.0,
    }


def candidate_signature(candidate: Mapping[str, Any]) -> str:
    signature = candidate.get("action_signature")
    if signature is not None:
        return str(signature)
    action = candidate.get("action") or {}
    return json.dumps(action, ensure_ascii=False, sort_keys=True)


def candidate_is_correct(candidate: Mapping[str, Any]) -> bool:
    return bool(candidate.get("is_correct"))


def first_correct_rank(candidates: Sequence[Mapping[str, Any]]) -> Tuple[Optional[int], Optional[str]]:
    for index, candidate in enumerate(candidates, 1):
        if candidate_is_correct(candidate):
            return index, str(candidate.get("candidate_id"))
    return None, None


def flags_for_row(row: Mapping[str, Any], pool_sizes: Sequence[int]) -> Dict[str, Any]:
    candidates = list(row.get("candidates") or [])
    greedy_correct = bool(row.get("greedy_correct"))
    recoverable_at: Dict[str, bool] = {}
    missing_at: Dict[str, bool] = {}
    oracle_correct_at: Dict[str, bool] = {}
    n_correct_at: Dict[str, int] = {}
    first_rank_by_n: Dict[str, Optional[int]] = {}
    for pool_size in pool_sizes:
        prefix = candidates[:pool_size]
        n_correct = sum(1 for candidate in prefix if candidate_is_correct(candidate))
        recoverable = n_correct > 0
        recoverable_at[str(pool_size)] = recoverable
        missing_at[str(pool_size)] = bool((not greedy_correct) and (not recoverable))
        oracle_correct_at[str(pool_size)] = bool(greedy_correct or ((not greedy_correct) and recoverable))
        n_correct_at[str(pool_size)] = n_correct
        rank, _ = first_correct_rank(prefix)
        first_rank_by_n[str(pool_size)] = rank
    full_rank, full_candidate_id = first_correct_rank(candidates)
    return {
        "recoverable_at": recoverable_at,
        "missing_at": missing_at,
        "oracle_correct_at": oracle_correct_at,
        "n_correct_at": n_correct_at,
        "first_correct_rank_at": first_rank_by_n,
        "first_correct_rank_full_pool": full_rank,
        "first_correct_candidate_id_full_pool": full_candidate_id,
    }


def compare_n5_reference(rows: Sequence[Mapping[str, Any]], n5_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n50_by_key = {row_key(row): row for row in rows}
    total = len(n5_rows)
    covered = 0
    exact_first5_signature_match = 0
    recoverability_match = 0
    greedy_correct_match = 0
    for n5_row in n5_rows:
        n50_row = n50_by_key.get(row_key(n5_row))
        if n50_row is None:
            continue
        covered += 1
        n5_candidates = list(n5_row.get("candidates") or [])[:5]
        n50_candidates = list(n50_row.get("candidates") or [])[:5]
        if [candidate_signature(candidate) for candidate in n5_candidates] == [candidate_signature(candidate) for candidate in n50_candidates]:
            exact_first5_signature_match += 1
        n5_recoverable = any(candidate_is_correct(candidate) for candidate in n5_candidates)
        n50_recoverable = any(candidate_is_correct(candidate) for candidate in n50_candidates)
        recoverability_match += int(n5_recoverable == n50_recoverable)
        greedy_correct_match += int(bool(n5_row.get("greedy_correct")) == bool(n50_row.get("greedy_correct")))
    return {
        "n5_rows": total,
        "covered_by_n50": covered,
        "coverage_fraction": covered / total if total else 0.0,
        "exact_first5_signature_match": exact_first5_signature_match,
        "exact_first5_signature_match_fraction": exact_first5_signature_match / total if total else 0.0,
        "recoverability_match": recoverability_match,
        "recoverability_match_fraction": recoverability_match / total if total else 0.0,
        "greedy_correct_match": greedy_correct_match,
        "greedy_correct_match_fraction": greedy_correct_match / total if total else 0.0,
    }


def build_per_step(rows: Sequence[Mapping[str, Any]], pool_sizes: Sequence[int]) -> List[Dict[str, Any]]:
    per_step: List[Dict[str, Any]] = []
    for row in rows:
        flags = flags_for_row(row, pool_sizes)
        per_step.append({
            "target_id": row.get("target_id"),
            "split": row.get("split"),
            "episode_id": str(row.get("episode_id")),
            "episode_key": str(row.get("episode_key") or row.get("episode_id")),
            "episode_order": row.get("episode_order"),
            "step_idx": int(row.get("step_idx") or 0),
            "instruction": row.get("instruction"),
            "screenshot": row.get("screenshot"),
            "gt_action_type": row.get("gt_action_type"),
            "gt_action_category": row.get("gt_action_category"),
            "greedy_correct": bool(row.get("greedy_correct")),
            "greedy_reward": row.get("greedy_reward"),
            "n_candidates_available": len(row.get("candidates") or []),
            "n_correct_candidates_full_pool": sum(1 for candidate in row.get("candidates") or [] if candidate_is_correct(candidate)),
            **flags,
        })
    return per_step


def summarize_pool_sizes(rows: Sequence[Mapping[str, Any]], per_step: Sequence[Mapping[str, Any]], pool_sizes: Sequence[int], selection_accuracies: Sequence[float]) -> Dict[str, Any]:
    greedy_success = [bool(row.get("greedy_correct")) for row in rows]
    greedy_metrics = task_metrics(rows, greedy_success)
    total_steps = len(rows)
    greedy_wrong = sum(1 for row in rows if not row.get("greedy_correct"))
    curve = []
    realistic_projection = []
    for pool_size in pool_sizes:
        pool_key = str(pool_size)
        recoverable_flags = [bool(row["recoverable_at"][pool_key]) for row in per_step]
        missing_wrong_flags = [bool(row["missing_at"][pool_key]) for row in per_step]
        oracle_success = [bool(row["oracle_correct_at"][pool_key]) for row in per_step]
        wrong_recoverable = sum(1 for original, recoverable in zip(rows, recoverable_flags, strict=True) if (not original.get("greedy_correct")) and recoverable)
        missing_wrong = sum(1 for flag in missing_wrong_flags if flag)
        metrics = task_metrics(rows, oracle_success)
        item = {
            "N": pool_size,
            "steps": total_steps,
            "greedy_wrong_steps": greedy_wrong,
            "recoverable_steps": sum(1 for flag in recoverable_flags if flag),
            "recoverable_rate_overall": sum(1 for flag in recoverable_flags if flag) / total_steps if total_steps else 0.0,
            "recoverable_greedy_wrong_steps": wrong_recoverable,
            "recoverable_rate_greedy_wrong": wrong_recoverable / greedy_wrong if greedy_wrong else 0.0,
            "missing_greedy_wrong_steps": missing_wrong,
            "missing_rate_greedy_wrong": missing_wrong / greedy_wrong if greedy_wrong else 0.0,
            "oracle_tsr_ceiling": metrics["tsr"],
            "oracle_step_sr_ceiling": metrics["step_sr"],
            "oracle_task_success": metrics["task_success"],
            "delta_tsr_vs_greedy": metrics["tsr"] - greedy_metrics["tsr"],
            "delta_step_sr_vs_greedy": metrics["step_sr"] - greedy_metrics["step_sr"],
            "avg_progress_ceiling": metrics["avg_progress"],
        }
        curve.append(item)
        for selection_accuracy in selection_accuracies:
            projected_tsr = greedy_metrics["tsr"] + (metrics["tsr"] - greedy_metrics["tsr"]) * selection_accuracy
            realistic_projection.append({
                "N": pool_size,
                "selection_accuracy": selection_accuracy,
                "projected_tsr": projected_tsr,
                "delta_tsr_vs_greedy": projected_tsr - greedy_metrics["tsr"],
            })
    return {
        "greedy_metrics": greedy_metrics,
        "curve": curve,
        "realistic_projection": realistic_projection,
    }


def saturation_summary(curve: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    increments = []
    previous = None
    for item in curve:
        if previous is not None:
            delta = float(item["recoverable_rate_greedy_wrong"]) - float(previous["recoverable_rate_greedy_wrong"])
            span = int(item["N"]) - int(previous["N"])
            increments.append({
                "from_N": previous["N"],
                "to_N": item["N"],
                "gain_recoverable_greedy_wrong": delta,
                "gain_pp": 100.0 * delta,
                "gain_per_extra_candidate": delta / span if span else None,
            })
        previous = item
    last_gain = float(increments[-1]["gain_recoverable_greedy_wrong"]) if increments else 0.0
    if last_gain >= 0.05:
        verdict = "not_saturated_at_50"
        reason = "Recoverable greedy-wrong rate still rises materially from N=20 to N=50. More samples may add headroom, although marginal gain per sample is falling."
    elif last_gain >= 0.02:
        verdict = "weakly_saturated_at_50"
        reason = "Recoverable rate still rises from N=20 to N=50, but with smaller marginal returns."
    else:
        verdict = "saturated_before_50"
        reason = "Recoverable rate changes little after N=20; extra sampling is unlikely to help much."
    return {"increments": increments, "verdict": verdict, "reason": reason}


def decide_gate(curve: Sequence[Mapping[str, Any]], projection: Sequence[Mapping[str, Any]], greedy_tsr: float) -> Dict[str, str]:
    by_n = {int(row["N"]): row for row in curve}
    n5 = by_n.get(5)
    n50 = by_n.get(50) or curve[-1]
    n50_delta = float(n50["delta_tsr_vs_greedy"])
    n5_to_n50 = float(n50["oracle_tsr_ceiling"] - n5["oracle_tsr_ceiling"]) if n5 else n50_delta
    n50_low_projection = next((row for row in projection if int(row["N"]) == int(n50["N"]) and abs(float(row["selection_accuracy"]) - 0.35) < 1e-9), None)
    low_projection_delta = float(n50_low_projection["delta_tsr_vs_greedy"]) if n50_low_projection else 0.0
    if n50_delta >= 0.08 and n5_to_n50 >= 0.05 and low_projection_delta > 0.01:
        return {
            "verdict": "POOL HAS HEADROOM",
            "reason": "The N=50 pool-limited oracle ceiling rises substantially over N=5, and even the conservative selection projection beats greedy. Retraining an N=50 verifier has real headroom; the remaining limit is selection quality, not just pool availability.",
        }
    if n50_delta <= 0.03 or float(n50["recoverable_rate_greedy_wrong"]) <= 0.25:
        return {
            "verdict": "POOL IS THE CAP",
            "reason": "Even N=50 leaves little pool-limited TSR lift or low greedy-wrong recoverability, so candidate generation/base sampling caps verifier retraining.",
        }
    return {
        "verdict": "MIXED",
        "reason": "N=50 has bounded but nontrivial pool headroom. Verifier retraining may help, but gains depend heavily on selection accuracy and candidate generation still matters.",
    }


def render_report(summary: Mapping[str, Any], output_dir: Path) -> str:
    lines = ["# Pool-Limited Ceiling at N=50", ""]
    lines.append("Ceiling analysis for candidate-pool headroom. Recoverable uses frozen-matcher correctness and is an ORACLE pool-contains-correct metric, not an operational verifier result.")
    lines.append("")
    ds = summary["dataset"]
    reuse = summary["reuse_and_cost"]
    lines.append("## Scope And Reuse")
    lines.append("")
    lines.append(f"- rows: `{ds['rows']}` across `{ds['episodes']}` episodes")
    lines.append(f"- greedy-correct: `{ds['greedy_correct_steps']}`; greedy-wrong: `{ds['greedy_wrong_steps']}`")
    lines.append(f"- N=50 pool coverage: `{reuse['n50_rows']}/{ds['rows']}` steps (`{pct(reuse['n50_coverage_fraction'])}`); greedy-wrong coverage `{reuse['n50_greedy_wrong_covered']}/{ds['greedy_wrong_steps']}` (`{pct(reuse['n50_greedy_wrong_coverage_fraction'])}`)")
    lines.append(f"- N=50 candidate count min/max: `{reuse['n50_candidate_count_min']}` / `{reuse['n50_candidate_count_max']}`")
    lines.append(f"- New sampling cost: `{reuse['new_sampling_steps']}` steps; full TEST N=50 already exists, so no new decoding was needed.")
    n5_ref = summary["n5_reference"]
    lines.append(f"- Existing N=5 file is covered by N=50 at `{pct(n5_ref['coverage_fraction'])}`; first-5 exact signature match `{pct(n5_ref['exact_first5_signature_match_fraction'])}`; recoverability match `{pct(n5_ref['recoverability_match_fraction'])}`.")
    lines.append("")
    lines.append("## Metric 1: Recoverable And Missing")
    lines.append("")
    lines.append("| N | recoverable overall | recoverable greedy-wrong | MISSING greedy-wrong | recoverable wrong steps | missing wrong steps |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in summary["pool_curve"]:
        lines.append(f"| {row['N']} | {pct(row['recoverable_rate_overall'])} | {pct(row['recoverable_rate_greedy_wrong'])} | {pct(row['missing_rate_greedy_wrong'])} | {row['recoverable_greedy_wrong_steps']} | {row['missing_greedy_wrong_steps']} |")
    lines.append("")
    lines.append("## Metric 2: Pool-Limited Oracle TSR Ceiling")
    lines.append("")
    lines.append("This is perfect selection whenever the top-N pool contains a matcher-correct action. It is a ceiling, not achieved verifier performance.")
    lines.append("")
    lines.append("| N | oracle TSR ceiling | oracle StepSR ceiling | ΔTSR vs greedy | successful tasks |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in summary["pool_curve"]:
        lines.append(f"| {row['N']} | {pct(row['oracle_tsr_ceiling'])} | {pct(row['oracle_step_sr_ceiling'])} | {pp(row['delta_tsr_vs_greedy'])} | {row['oracle_task_success']} |")
    lines.append("")
    ref = summary["reference_note"]
    lines.append(f"N=5 -> N=50 pool-limited ceiling rise: `{pp(ref['n5_to_n50_ceiling_rise'])}`. Historical N=5 oracle-gating-with-existing-verifier was `{pct(ref['historical_n5_oracle_gate_tsr'])}` (`{pp(ref['historical_n5_oracle_gate_delta_vs_greedy'])}`), which is lower because it used the existing verifier's selection rather than perfect pool selection.")
    lines.append("")
    lines.append("## Metric 3: Realistic Projection")
    lines.append("")
    lines.append("Projected TSR = greedy TSR + (pool ceiling - greedy TSR) * selection accuracy. This keeps pool-contains-correct distinct from verifier-selects-correct.")
    lines.append("")
    lines.append("| N | selection acc 0.35 | selection acc 0.50 | selection acc 0.73 |")
    lines.append("|---:|---:|---:|---:|")
    by_n: Dict[int, Dict[float, Mapping[str, Any]]] = defaultdict(dict)
    for row in summary["realistic_projection"]:
        by_n[int(row["N"])][round(float(row["selection_accuracy"]), 2)] = row
    for row in summary["pool_curve"]:
        n = int(row["N"])
        cells = []
        for acc in (0.35, 0.50, 0.73):
            projection = by_n[n][round(acc, 2)]
            cells.append(f"{pct(projection['projected_tsr'])} ({pp(projection['delta_tsr_vs_greedy'])})")
        lines.append(f"| {n} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")
    lines.append("## Metric 4: Saturation")
    lines.append("")
    lines.append("| interval | greedy-wrong recoverable gain | gain per extra candidate |")
    lines.append("|---:|---:|---:|")
    for row in summary["saturation"]["increments"]:
        lines.append(f"| {row['from_N']} -> {row['to_N']} | {pp(row['gain_recoverable_greedy_wrong'])} | {fmt(row['gain_per_extra_candidate'])} |")
    lines.append("")
    lines.append(f"**{summary['saturation']['verdict']}**: {summary['saturation']['reason']}")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(f"**{summary['gate']['verdict']}**")
    lines.append("")
    lines.append(summary["gate"]["reason"])
    lines.append("")
    lines.append("## Guardrails")
    lines.append("")
    lines.append("- CEILING IS ORACLE: recoverable uses GT/frozen matcher labels only to measure the upper bound.")
    lines.append("- RECOVERABLE != FINDABLE: a correct action in the pool still needs a verifier to select it.")
    lines.append("- REUSE FIRST: this run reused full TEST N=50 and full TEST N=5; no new samples were decoded.")
    lines.append("- Frozen matcher and base model unchanged.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{output_dir / 'pool_ceiling.md'}`")
    lines.append(f"- `{output_dir / 'summary.json'}`")
    lines.append(f"- `{output_dir / 'per_step.jsonl'}`")
    lines.append("")
    lines.append("STOP for review.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n50-pool", default=DEFAULT_N50_POOL)
    parser.add_argument("--n5-pool", default=DEFAULT_N5_POOL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pool-sizes", default=",".join(str(value) for value in DEFAULT_POOL_SIZES))
    parser.add_argument("--selection-accuracies", default=",".join(str(value) for value in DEFAULT_SELECTION_ACCURACIES))
    parser.add_argument("--historical-n5-oracle-gate-tsr", type=float, default=HISTORICAL_N5_ORACLE_GATE_TSR)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    pool_sizes = [int(item) for item in args.pool_sizes.split(",") if item.strip()]
    selection_accuracies = [float(item) for item in args.selection_accuracies.split(",") if item.strip()]
    rows = sorted(read_jsonl(Path(args.n50_pool)), key=sort_key)
    n5_rows = read_jsonl(Path(args.n5_pool)) if Path(args.n5_pool).exists() else []
    if not rows:
        raise RuntimeError(f"no rows loaded from {args.n50_pool}")
    max_pool_size = max(pool_sizes)
    short_rows = [row_key(row) for row in rows if len(row.get("candidates") or []) < max_pool_size]
    if short_rows:
        raise RuntimeError(f"{len(short_rows)} rows have fewer than {max_pool_size} candidates; first missing key={short_rows[0]}")

    per_step = build_per_step(rows, pool_sizes)
    pool_summary = summarize_pool_sizes(rows, per_step, pool_sizes, selection_accuracies)
    greedy = pool_summary["greedy_metrics"]
    by_n = {int(row["N"]): row for row in pool_summary["curve"]}
    n5 = by_n.get(5)
    n50 = by_n.get(50) or pool_summary["curve"][-1]
    candidate_counts = [len(row.get("candidates") or []) for row in rows]
    greedy_wrong_rows = [row for row in rows if not row.get("greedy_correct")]
    n50_covered_keys = {row_key(row) for row in rows if len(row.get("candidates") or []) >= 50}
    n50_greedy_wrong_covered = sum(1 for row in greedy_wrong_rows if row_key(row) in n50_covered_keys)
    summary = {
        "inputs": {"n50_pool": args.n50_pool, "n5_pool": args.n5_pool, "pool_sizes": pool_sizes, "selection_accuracies": selection_accuracies},
        "dataset": {
            "rows": len(rows),
            "episodes": len({str(row.get("episode_key") or row.get("episode_id")) for row in rows}),
            "greedy_correct_steps": sum(1 for row in rows if row.get("greedy_correct")),
            "greedy_wrong_steps": sum(1 for row in rows if not row.get("greedy_correct")),
            "greedy_tsr": greedy["tsr"],
            "greedy_step_sr": greedy["step_sr"],
            "greedy_task_success": greedy["task_success"],
        },
        "reuse_and_cost": {
            "n50_rows": len(rows),
            "n50_coverage_fraction": 1.0,
            "n50_greedy_wrong_covered": n50_greedy_wrong_covered,
            "n50_greedy_wrong_coverage_fraction": n50_greedy_wrong_covered / len(greedy_wrong_rows) if greedy_wrong_rows else 0.0,
            "n50_candidate_count_min": min(candidate_counts),
            "n50_candidate_count_max": max(candidate_counts),
            "n50_candidate_count_unique": sorted(set(candidate_counts)),
            "new_sampling_steps": 0,
            "new_sampling_reason": "full TEST N=50 pool already exists; no greedy-wrong steps lack N=50 coverage",
        },
        "n5_reference": compare_n5_reference(rows, n5_rows) if n5_rows else {},
        "pool_curve": pool_summary["curve"],
        "realistic_projection": pool_summary["realistic_projection"],
        "saturation": saturation_summary(pool_summary["curve"]),
        "reference_note": {
            "historical_n5_oracle_gate_tsr": args.historical_n5_oracle_gate_tsr,
            "historical_n5_oracle_gate_delta_vs_greedy": args.historical_n5_oracle_gate_tsr - greedy["tsr"],
            "pool_limited_n5_tsr": n5["oracle_tsr_ceiling"] if n5 else None,
            "pool_limited_n5_delta_vs_greedy": n5["delta_tsr_vs_greedy"] if n5 else None,
            "pool_limited_n50_tsr": n50["oracle_tsr_ceiling"],
            "pool_limited_n50_delta_vs_greedy": n50["delta_tsr_vs_greedy"],
            "n5_to_n50_ceiling_rise": n50["oracle_tsr_ceiling"] - n5["oracle_tsr_ceiling"] if n5 else None,
        },
    }
    summary["gate"] = decide_gate(summary["pool_curve"], summary["realistic_projection"], greedy["tsr"])

    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "per_step.jsonl", per_step)
    (output_dir / "pool_ceiling.md").write_text(render_report(summary, output_dir), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "greedy_tsr": greedy["tsr"],
        "n5_pool_limited_tsr": n5["oracle_tsr_ceiling"] if n5 else None,
        "n50_pool_limited_tsr": n50["oracle_tsr_ceiling"],
        "n50_recoverable_greedy_wrong": n50["recoverable_rate_greedy_wrong"],
        "gate": summary["gate"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()