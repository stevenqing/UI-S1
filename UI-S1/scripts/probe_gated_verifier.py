#!/usr/bin/env python3
"""Recompose teacher-forced TSR for probe-gated verifier selection."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BUDGETS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0)
DEFAULT_PROBE = "outputs/critstep_probe_leakage/per_step.jsonl"
DEFAULT_VERIFIER = "outputs/history_correction/verifier_pointwise_n5/per_step.jsonl"
DEFAULT_CANDIDATES = "outputs/history_correction/n5_candidates/per_step.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/probe_gated_verifier"
GREEDY_TSR_REF = 0.222
EVERY_STEP_VERIFIER_TSR_REF = 0.19


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
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
    if value is None:
        return "NA"
    return f"{100.0 * float(value):.2f}%"


def pp(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):+.2f}pp"


def key(row: Mapping[str, Any]) -> Tuple[str, int]:
    return str(row.get("episode_id")), int(row.get("step_idx") or 0)


def task_metrics(rows: Sequence[Mapping[str, Any]], success_field: str) -> Dict[str, Any]:
    by_episode: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    total = 0
    correct = 0
    for row in rows:
        by_episode[str(row["episode_id"])].append(row)
        total += 1
        correct += int(bool(row[success_field]))
    task_success = 0
    progress_sum = 0.0
    for steps in by_episode.values():
        ordered = sorted(steps, key=lambda item: int(item["step_idx"]))
        first_error = next((idx for idx, step in enumerate(ordered, 1) if not step[success_field]), None)
        if first_error is None:
            task_success += 1
            progress_sum += 1.0
        else:
            progress_sum += (first_error - 1) / len(ordered) if ordered else 0.0
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


def selected_indices_by_budget(scores: Sequence[float], budget: float) -> set[int]:
    n = len(scores)
    if budget <= 0:
        return set()
    if budget >= 1:
        return set(range(n))
    k = max(1, int(round(n * budget)))
    order = sorted(range(n), key=lambda idx: (-float(scores[idx]), idx))
    return set(order[:k])


def compose(rows: Sequence[Mapping[str, Any]], selected: set[int], label: str, cost_override: Optional[float] = None) -> Dict[str, Any]:
    out_rows: List[Dict[str, Any]] = []
    fixed = injected = missing = recoverable_not_fixed = gated_greedy_wrong = gated_greedy_correct = 0
    retained_correct = 0
    for idx, row in enumerate(rows):
        gated = idx in selected
        greedy_correct = bool(row["greedy_correct"])
        verifier_correct = bool(row["verifier_correct"])
        recoverable = bool(row["recoverable"])
        selected_correct = verifier_correct if gated else greedy_correct
        if gated:
            gated_greedy_correct += int(greedy_correct)
            gated_greedy_wrong += int(not greedy_correct)
            fixed += int((not greedy_correct) and verifier_correct)
            injected += int(greedy_correct and (not verifier_correct))
            missing += int((not greedy_correct) and (not recoverable))
            recoverable_not_fixed += int((not greedy_correct) and recoverable and (not verifier_correct))
            retained_correct += int(greedy_correct and verifier_correct)
        out = dict(row)
        out[f"{label}_gated"] = gated
        out[f"{label}_selected_correct"] = selected_correct
        out_rows.append(out)
    metrics = task_metrics(out_rows, f"{label}_selected_correct")
    gated_count = len(selected)
    metrics.update({
        "label": label,
        "gated_steps": gated_count,
        "verify_fraction": cost_override if cost_override is not None else (gated_count / len(rows) if rows else 0.0),
        "fixed_wrong_to_correct": fixed,
        "injected_correct_to_wrong": injected,
        "missing_gated_wrong_no_correct_candidate": missing,
        "recoverable_wrong_not_fixed": recoverable_not_fixed,
        "gated_greedy_wrong": gated_greedy_wrong,
        "gated_greedy_correct": gated_greedy_correct,
        "retained_correct_under_verifier": retained_correct,
        "fix_per_injection": fixed / injected if injected else None,
        "net_step_effect_fixed_minus_injected": fixed - injected,
        "tsr_per_verified_fraction": (metrics["tsr"] - GREEDY_TSR_REF) / (gated_count / len(rows)) if gated_count else None,
    })
    return metrics


def build_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    probe = {key(row): row for row in read_jsonl(Path(args.probe_per_step))}
    verifier = {key(row): row for row in read_jsonl(Path(args.verifier_per_step))}
    candidates = {key(row): row for row in read_jsonl(Path(args.candidates))}
    rows: List[Dict[str, Any]] = []
    missing = []
    for key_item, cand in sorted(candidates.items(), key=lambda item: (int(item[0][0]) if item[0][0].isdigit() else item[0][0], item[0][1])):
        p = probe.get(key_item)
        v = verifier.get(key_item)
        if p is None or v is None:
            missing.append(key_item)
            continue
        greedy_correct = bool(cand.get("greedy_correct"))
        verifier_correct = bool(v.get("verifier_correct"))
        recoverable = bool(cand.get("n_correct_candidates", 0) > 0)
        rows.append({
            "episode_id": key_item[0],
            "step_idx": key_item[1],
            "target_id": cand.get("target_id"),
            "probe_score": float(p.get(args.probe_score_field)),
            "probe_percentile": p.get("best_probe_percentile"),
            "bottom2_critical": bool(p.get("bottom2_critical")),
            "bottom1_critical": bool(p.get("bottom1_critical")),
            "p_i_heldout_label_only": p.get("p_i_heldout_label_only"),
            "greedy_correct": greedy_correct,
            "verifier_correct": verifier_correct,
            "recoverable": recoverable,
            "n_correct_candidates": cand.get("n_correct_candidates"),
            "verifier_candidate_id": v.get("verifier_candidate_id"),
            "verifier_score": v.get("verifier_score"),
            "fixed_if_verified": bool((not greedy_correct) and verifier_correct),
            "injected_if_verified": bool(greedy_correct and (not verifier_correct)),
            "missing_if_verified": bool((not greedy_correct) and (not recoverable)),
            "recoverable_not_fixed_if_verified": bool((not greedy_correct) and recoverable and (not verifier_correct)),
        })
    if missing:
        raise RuntimeError(f"missing probe/verifier rows for {len(missing)} candidate rows")
    return rows


def oracle_rank_indices(rows: Sequence[Mapping[str, Any]], budget: float) -> set[int]:
    if budget <= 0:
        return set()
    wrong = [idx for idx, row in enumerate(rows) if not row["greedy_correct"]]
    if budget >= 1:
        return set(wrong)
    k = max(1, int(round(len(rows) * budget)))
    return set(wrong[: min(k, len(wrong))])


def decide_gate(curve: Sequence[Mapping[str, Any]], greedy_tsr: float, every_tsr: float) -> Dict[str, str]:
    probe_rows = [row for row in curve if row.get("kind") == "probe"]
    best = max(probe_rows, key=lambda row: float(row.get("tsr") or 0.0))
    if float(best["tsr"]) > greedy_tsr + 0.001:
        return {"verdict": "PROBE-GATED VERIFIER POSITIVE", "reason": f"Best probe-gated K={best['budget_percent']} beats greedy by {(best['tsr'] - greedy_tsr) * 100:.2f}pp."}
    if any(float(row["tsr"]) > every_tsr + 0.001 for row in probe_rows):
        return {"verdict": "PROBE-GATED BETTER BUT STILL NOT POSITIVE", "reason": "Probe gating reduces every-step verifier damage but no threshold beats greedy TSR."}
    return {"verdict": "PROBE-GATING DOESN'T HELP", "reason": "No probe-gated threshold improves materially over every-step verifier or greedy."}


def render_report(summary: Mapping[str, Any], output_dir: Path) -> str:
    lines = ["# Probe-Gated Verifier", ""]
    lines.append("Teacher-forced static recomposition. Gated steps use pointwise verifier-selected correctness; ungated steps keep greedy correctness. Frozen matcher; no model inference.")
    lines.append("")
    lines.append("## Sanity Checks")
    lines.append("")
    sanity = summary["sanity"]
    lines.append(f"- K=0 greedy TSR: `{pct(sanity['k0_tsr'])}`; reference greedy TSR `{pct(summary['references']['greedy_tsr'])}`")
    lines.append(f"- K=100 every-step verifier TSR: `{pct(sanity['k100_tsr'])}`; reference every-step verifier TSR `{pct(summary['references']['every_step_verifier_tsr'])}`")
    lines.append(f"- sanity passed: `{sanity['passed']}`")
    lines.append("")
    lines.append("## TSR Curve")
    lines.append("")
    lines.append("| K | verified | TSR | StepSR | ΔTSR vs greedy | ΔTSR vs every-step | fixed | injected | missing | fix/inject | net fix-inject | TSR per compute |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary["probe_curve"]:
        lines.append(
            f"| {row['budget_percent']} | {pct(row['verify_fraction'])} | {pct(row['tsr'])} | {pct(row['step_sr'])} | "
            f"{pp(row['delta_tsr_vs_greedy'])} | {pp(row['delta_tsr_vs_every_step'])} | {row['fixed_wrong_to_correct']} | "
            f"{row['injected_correct_to_wrong']} | {row['missing_gated_wrong_no_correct_candidate']} | {fmt(row.get('fix_per_injection'))} | "
            f"{row['net_step_effect_fixed_minus_injected']} | {pp(row.get('tsr_per_verified_fraction'))} |"
        )
    lines.append("")
    lines.append("## Best Probe Gate")
    lines.append("")
    best = summary["best_probe_gate"]
    lines.append(f"Best K: `{best['budget_percent']}` with TSR `{pct(best['tsr'])}` and ΔTSR vs greedy `{pp(best['delta_tsr_vs_greedy'])}`.")
    lines.append("")
    lines.append("## Probe vs Oracle Gating")
    lines.append("")
    lines.append("| policy | verified | TSR | StepSR | ΔTSR vs greedy | fixed | injected | missing | fix/inject |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary["oracle_comparison"]:
        lines.append(
            f"| {row['label']} | {pct(row['verify_fraction'])} | {pct(row['tsr'])} | {pct(row['step_sr'])} | {pp(row['delta_tsr_vs_greedy'])} | "
            f"{row['fixed_wrong_to_correct']} | {row['injected_correct_to_wrong']} | {row['missing_gated_wrong_no_correct_candidate']} | {fmt(row.get('fix_per_injection'))} |"
        )
    lines.append("")
    lines.append("## Mechanism")
    lines.append("")
    mech = summary["mechanism_summary"]
    lines.append(f"Every-step verifier: fixed `{mech['every_step']['fixed']}`, injected `{mech['every_step']['injected']}`, fix/inject `{fmt(mech['every_step']['fix_per_injection'])}`.")
    lines.append(f"Best probe gate: fixed `{mech['best_probe']['fixed']}`, injected `{mech['best_probe']['injected']}`, fix/inject `{fmt(mech['best_probe']['fix_per_injection'])}`.")
    lines.append(f"Oracle greedy-wrong gate: fixed `{mech['oracle_greedy_wrong']['fixed']}`, injected `{mech['oracle_greedy_wrong']['injected']}`, fix/inject `{fmt(mech['oracle_greedy_wrong']['fix_per_injection'])}`.")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(f"**{summary['gate']['verdict']}**")
    lines.append("")
    lines.append(summary["gate"]["reason"])
    lines.append("")
    lines.append("## Guardrails")
    lines.append("")
    lines.append("- K=0 and K=100 sanity checks are included; K=100 matches the every-step pointwise verifier recomposition.")
    lines.append("- Missing means no correct action exists in the static N=5 candidate pool; verifier cannot fix those steps.")
    lines.append("- This is teacher-forced static-data recomposition, not autonomous rollout.")
    lines.append("- Oracle gating uses true greedy-wrong and is reported only as a ceiling, not deployable.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{output_dir / 'gated.md'}`")
    lines.append(f"- `{output_dir / 'summary.json'}`")
    lines.append(f"- `{output_dir / 'per_step.jsonl'}`")
    lines.append("")
    lines.append("STOP for review.")
    return "\n".join(lines) + "\n"


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-per-step", default="outputs/critstep_probe_leakage/per_step.jsonl")
    parser.add_argument("--verifier-per-step", default="outputs/history_correction/verifier_pointwise_n5/per_step.jsonl")
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--probe-score-field", default="best_probe_score")
    parser.add_argument("--budgets", default=",".join(str(v) for v in BUDGETS))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    budgets = [float(item) for item in args.budgets.split(",") if item.strip()]
    rows = build_rows(args)
    scores = [float(row["probe_score"]) for row in rows]
    probe_curve = []
    for budget in budgets:
        selected = selected_indices_by_budget(scores, budget)
        metrics = compose(rows, selected, f"probe_k{int(round(budget * 100))}")
        metrics.update({
            "kind": "probe",
            "budget": budget,
            "budget_percent": f"{int(round(budget * 100))}%",
            "delta_tsr_vs_greedy": metrics["tsr"] - GREEDY_TSR_REF,
            "delta_tsr_vs_every_step": metrics["tsr"] - EVERY_STEP_VERIFIER_TSR_REF,
        })
        probe_curve.append(metrics)
    greedy = probe_curve[0]
    every = probe_curve[-1]
    oracle_all_wrong = compose(rows, {idx for idx, row in enumerate(rows) if not row["greedy_correct"]}, "oracle_greedy_wrong_all")
    oracle_all_wrong.update({
        "kind": "oracle",
        "budget": oracle_all_wrong["verify_fraction"],
        "budget_percent": f"{oracle_all_wrong['verify_fraction']*100:.2f}%",
        "delta_tsr_vs_greedy": oracle_all_wrong["tsr"] - GREEDY_TSR_REF,
        "delta_tsr_vs_every_step": oracle_all_wrong["tsr"] - EVERY_STEP_VERIFIER_TSR_REF,
    })
    oracle_same_cost = []
    for budget in budgets:
        if budget <= 0:
            continue
        selected = oracle_rank_indices(rows, budget)
        metrics = compose(rows, selected, f"oracle_k{int(round(budget * 100))}")
        metrics.update({
            "kind": "oracle_budget",
            "budget": budget,
            "budget_percent": f"{int(round(budget * 100))}%",
            "delta_tsr_vs_greedy": metrics["tsr"] - GREEDY_TSR_REF,
            "delta_tsr_vs_every_step": metrics["tsr"] - EVERY_STEP_VERIFIER_TSR_REF,
        })
        oracle_same_cost.append(metrics)
    all_curve = probe_curve + [oracle_all_wrong] + oracle_same_cost
    best_probe = max(probe_curve, key=lambda row: row["tsr"])
    sanity = {
        "k0_tsr": greedy["tsr"],
        "k0_step_sr": greedy["step_sr"],
        "k100_tsr": every["tsr"],
        "k100_step_sr": every["step_sr"],
        "passed": abs(greedy["tsr"] - GREEDY_TSR_REF) < 1e-9 and abs(every["tsr"] - EVERY_STEP_VERIFIER_TSR_REF) < 1e-9,
    }
    summary = {
        "inputs": {"probe_per_step": args.probe_per_step, "verifier_per_step": args.verifier_per_step, "candidates": args.candidates, "probe_score_field": args.probe_score_field},
        "references": {"greedy_tsr": GREEDY_TSR_REF, "every_step_verifier_tsr": EVERY_STEP_VERIFIER_TSR_REF, "every_step_delta_vs_greedy": EVERY_STEP_VERIFIER_TSR_REF - GREEDY_TSR_REF},
        "dataset": {"steps": len(rows), "episodes": len({row["episode_id"] for row in rows}), "greedy_correct_steps": sum(1 for row in rows if row["greedy_correct"]), "greedy_wrong_steps": sum(1 for row in rows if not row["greedy_correct"]), "recoverable_greedy_wrong_steps": sum(1 for row in rows if (not row["greedy_correct"] and row["recoverable"]))},
        "sanity": sanity,
        "probe_curve": probe_curve,
        "best_probe_gate": best_probe,
        "oracle_comparison": [every, best_probe, oracle_all_wrong],
        "oracle_budget_curve": oracle_same_cost,
        "mechanism_summary": {
            "every_step": {"fixed": every["fixed_wrong_to_correct"], "injected": every["injected_correct_to_wrong"], "fix_per_injection": every.get("fix_per_injection")},
            "best_probe": {"fixed": best_probe["fixed_wrong_to_correct"], "injected": best_probe["injected_correct_to_wrong"], "fix_per_injection": best_probe.get("fix_per_injection")},
            "oracle_greedy_wrong": {"fixed": oracle_all_wrong["fixed_wrong_to_correct"], "injected": oracle_all_wrong["injected_correct_to_wrong"], "fix_per_injection": oracle_all_wrong.get("fix_per_injection")},
        },
    }
    summary["gate"] = decide_gate(probe_curve, GREEDY_TSR_REF, EVERY_STEP_VERIFIER_TSR_REF)
    # Keep per-step compact but include flags for each K.
    per_step_rows = []
    selected_by_budget = {budget: selected_indices_by_budget(scores, budget) for budget in budgets}
    for idx, row in enumerate(rows):
        item = dict(row)
        item["gated_flags"] = {f"K{int(round(budget * 100))}": idx in selected for budget, selected in selected_by_budget.items()}
        per_step_rows.append(item)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "per_step.jsonl", per_step_rows)
    (output_dir / "gated.md").write_text(render_report(summary, output_dir), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "best_probe_gate": {"K": best_probe["budget_percent"], "tsr": best_probe["tsr"], "delta_vs_greedy": best_probe["delta_tsr_vs_greedy"]}, "gate": summary["gate"], "sanity": sanity}, indent=2), flush=True)


if __name__ == "__main__":
    main()
