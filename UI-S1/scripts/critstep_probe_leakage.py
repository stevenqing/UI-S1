#!/usr/bin/env python3
"""Disentangle whether critical-step representation probes detect difficulty or imminent error."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.critstep_identifiability import auc_score, write_json, write_jsonl  # noqa: E402
from scripts.critstep_representation_probe import (  # noqa: E402
    BASELINE_SURFACE_AUC,
    BASELINE_SURFACE_TOP20_RECALL,
    BUDGETS,
    load_activation_shards,
    logistic_cv_array,
    orient_score_for_auc,
    parse_csv_strs,
    random_project,
    read_jsonl,
    topk_table_from_scores,
)

DEFAULT_REP_DIR = "outputs/critstep_representation"
DEFAULT_CANDIDATES = "outputs/critstep_binlift_lean/test_candidates/per_step.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/critstep_probe_leakage"


def mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def std(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if len(vals) <= 1:
        return 0.0 if vals else None
    return float(np.std(np.asarray(vals, dtype=float)))


def median(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(np.asarray(vals, dtype=float))) if vals else None


def pct(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):.2f}%"


def fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def read_candidate_map(path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out = {}
    for row in read_jsonl(path):
        key = (str(row.get("episode_id")), int(row.get("step_idx") or 0))
        out[key] = row
    return out


def group_stats(rows: Sequence[Mapping[str, Any]], score_field: str) -> Dict[str, Any]:
    scores = [float(row[score_field]) for row in rows]
    p_vals = [float(row["p_i_heldout_label_only"]) for row in rows]
    return {
        "n": len(rows),
        "mean_score": mean(scores),
        "median_score": median(scores),
        "std_score": std(scores),
        "mean_p_i": mean(p_vals),
        "correct_rate": mean([1.0 if row.get("greedy_correct") else 0.0 for row in rows]),
    }


def ols_r2(y: np.ndarray, x: np.ndarray) -> float:
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    design = np.column_stack([np.ones(len(y)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    if controls.ndim == 1:
        controls = controls.reshape(-1, 1)
    design = np.column_stack([np.ones(len(values)), controls])
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coef


def corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) <= 1 or float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def score_representations(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray], Dict[str, Any]]:
    rep_root = Path(args.representation_dir)
    meta_rows, arrays = load_activation_shards(rep_root, args.num_shards)
    allowed = set(parse_csv_strs(args.representations)) if args.representations else set(arrays)
    y = np.asarray([int(row["bottom2"]) for row in meta_rows], dtype=int)
    results = []
    scores: Dict[str, np.ndarray] = {}
    for rep_name in sorted(arrays):
        if rep_name not in allowed:
            continue
        seed = args.seed + (abs(hash(rep_name)) % 100000)
        x = random_project(arrays[rep_name], args.probe_dim, seed)
        item = logistic_cv_array(x, y, args.folds, args.seed, l2=args.l2)
        raw_score = np.asarray(item["scores"], dtype=float)
        oriented = orient_score_for_auc(y, raw_score).astype(float)
        scores[rep_name] = oriented
        results.append({
            "representation": rep_name,
            "auc_bottom2": auc_score(y.tolist(), oriented.tolist()),
            "balanced_accuracy": item["balanced_accuracy"],
            "n": int(len(y)),
            "positives": int(np.sum(y == 1)),
        })
    results.sort(key=lambda row: row["auc_bottom2"] if row["auc_bottom2"] is not None else -1.0, reverse=True)
    manifest = {
        "representation_dir": args.representation_dir,
        "num_rows": len(meta_rows),
        "scored_representations": [row["representation"] for row in results],
        "probe_dim": args.probe_dim,
        "folds": args.folds,
        "pre_decision_note": "Uses the same prompt-forward pre-decision activations extracted by critstep_representation_probe.py.",
    }
    return meta_rows, scores, {"results": results, "manifest": manifest}


def position_family(rep_name: str) -> str:
    if rep_name.endswith("prompt_last"):
        return "prompt_last_action_generation_position"
    if rep_name.endswith("text_mean"):
        return "text_mean_pre_decision"
    if rep_name.endswith("vision_mean"):
        return "vision_mean_pre_decision"
    if rep_name.endswith("vision_max"):
        return "vision_max_pre_decision"
    return "unknown"


def build_rows(meta_rows: Sequence[Mapping[str, Any]], candidates: Mapping[Tuple[str, int], Mapping[str, Any]], scores: Mapping[str, np.ndarray], best_rep: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, meta in enumerate(meta_rows):
        key = (str(meta["episode_id"]), int(meta["step_idx"]))
        cand = candidates.get(key, {})
        greedy_correct = bool(cand.get("greedy_correct"))
        row = {
            "row_index": int(meta["row_index"]),
            "target_id": meta.get("target_id"),
            "episode_id": str(meta["episode_id"]),
            "step_idx": int(meta["step_idx"]),
            "bottom2_critical": bool(meta["bottom2"]),
            "bottom1_critical": bool(meta["bottom1"]),
            "p_i_heldout_label_only": float(meta["p_i_heldout_label_only"]),
            "difficulty_score": 1.0 - float(meta["p_i_heldout_label_only"]),
            "greedy_correct": greedy_correct,
            "this_sample_wrong": not greedy_correct,
            "greedy_reward": cand.get("greedy_reward"),
            "best_representation": best_rep,
            "best_probe_score": float(scores[best_rep][idx]),
            "probe_scores": {name: float(value[idx]) for name, value in scores.items()},
        }
        rows.append(row)
    return rows


def test_a(rows: Sequence[Mapping[str, Any]], score_field: str = "best_probe_score") -> Dict[str, Any]:
    groups = {
        "hard_correct": [row for row in rows if row["bottom2_critical"] and row["greedy_correct"]],
        "hard_wrong": [row for row in rows if row["bottom2_critical"] and not row["greedy_correct"]],
        "easy_correct": [row for row in rows if not row["bottom2_critical"] and row["greedy_correct"]],
        "easy_wrong": [row for row in rows if not row["bottom2_critical"] and not row["greedy_correct"]],
    }
    out = {name: group_stats(group, score_field) for name, group in groups.items()}
    hc = out["hard_correct"]["mean_score"]
    hw = out["hard_wrong"]["mean_score"]
    ec = out["easy_correct"]["mean_score"]
    ew = out["easy_wrong"]["mean_score"]
    out["contrasts"] = {
        "hard_wrong_minus_hard_correct": (hw - hc) if hw is not None and hc is not None else None,
        "easy_wrong_minus_easy_correct": (ew - ec) if ew is not None and ec is not None else None,
        "hard_correct_minus_easy_correct": (hc - ec) if hc is not None and ec is not None else None,
        "hard_wrong_minus_easy_wrong": (hw - ew) if hw is not None and ew is not None else None,
    }
    return out


def test_b(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    score = np.asarray([float(row["best_probe_score"]) for row in rows], dtype=float)
    difficulty = np.asarray([float(row["difficulty_score"]) for row in rows], dtype=float)
    error = np.asarray([1.0 if row["this_sample_wrong"] else 0.0 for row in rows], dtype=float)
    hard = np.asarray([1 if row["bottom2_critical"] else 0 for row in rows], dtype=int)
    r_score_difficulty = corr(score, difficulty)
    r_score_error = corr(score, error)
    partial_score_difficulty_given_error = corr(residualize(score, error), residualize(difficulty, error))
    partial_score_error_given_difficulty = corr(residualize(score, difficulty), residualize(error, difficulty))
    r2_diff_error = ols_r2(difficulty, error)
    r2_diff_error_score = ols_r2(difficulty, np.column_stack([error, score]))
    r2_error_diff = ols_r2(error, difficulty)
    r2_error_diff_score = ols_r2(error, np.column_stack([difficulty, score]))
    return {
        "score_vs_difficulty_corr": r_score_difficulty,
        "score_vs_error_corr": r_score_error,
        "partial_score_difficulty_given_error": partial_score_difficulty_given_error,
        "partial_score_error_given_difficulty": partial_score_error_given_difficulty,
        "difficulty_r2_error_only": r2_diff_error,
        "difficulty_r2_error_plus_probe": r2_diff_error_score,
        "difficulty_delta_r2_probe_beyond_error": r2_diff_error_score - r2_diff_error,
        "error_r2_difficulty_only": r2_error_diff,
        "error_r2_difficulty_plus_probe": r2_error_diff_score,
        "error_delta_r2_probe_beyond_difficulty": r2_error_diff_score - r2_error_diff,
        "probe_auc_for_bottom2": auc_score(hard.tolist(), score.tolist()),
        "probe_auc_for_this_sample_error": auc_score(error.astype(int).tolist(), score.tolist()),
        "difficulty_auc_for_this_sample_error": auc_score(error.astype(int).tolist(), difficulty.tolist()),
    }


def test_c(rep_results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_family: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rep_results:
        by_family[position_family(str(row["representation"]))].append(row)
    out = {}
    for family, items in sorted(by_family.items()):
        best = max(items, key=lambda item: item.get("auc_bottom2") or -1.0)
        out[family] = {
            "best_representation": best["representation"],
            "best_auc": best.get("auc_bottom2"),
            "representations": [{"representation": item["representation"], "auc": item.get("auc_bottom2")} for item in sorted(items, key=lambda item: item.get("auc_bottom2") or -1.0, reverse=True)],
        }
    return out


def triage(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    meta_like = [{"bottom2": row["bottom2_critical"]} for row in rows]
    scores = [float(row["best_probe_score"]) for row in rows]
    error_meta = [{"error": row["this_sample_wrong"]} for row in rows]
    return {
        "bottom2_by_probe": topk_table_from_scores(meta_like, scores, BUDGETS, "bottom2"),
        "error_by_probe": topk_table_from_scores(error_meta, scores, BUDGETS, "error"),
    }


def decide_gate(test_a_out: Mapping[str, Any], test_b_out: Mapping[str, Any], test_c_out: Mapping[str, Any]) -> Dict[str, str]:
    hard_delta = abs(float((test_a_out.get("contrasts") or {}).get("hard_wrong_minus_hard_correct") or 0.0))
    hard_easy = float((test_a_out.get("contrasts") or {}).get("hard_correct_minus_easy_correct") or 0.0)
    diff_delta = float(test_b_out.get("difficulty_delta_r2_probe_beyond_error") or 0.0)
    err_delta = float(test_b_out.get("error_delta_r2_probe_beyond_difficulty") or 0.0)
    prompt_auc = float((test_c_out.get("prompt_last_action_generation_position") or {}).get("best_auc") or 0.0)
    text_auc = float((test_c_out.get("text_mean_pre_decision") or {}).get("best_auc") or 0.0)
    vision_auc = max(
        float((test_c_out.get("vision_mean_pre_decision") or {}).get("best_auc") or 0.0),
        float((test_c_out.get("vision_max_pre_decision") or {}).get("best_auc") or 0.0),
    )
    if hard_delta <= 0.05 and hard_easy > 0.10 and diff_delta > err_delta + 0.01 and prompt_auc > BASELINE_SURFACE_AUC and text_auc > BASELINE_SURFACE_AUC and vision_auc > BASELINE_SURFACE_AUC:
        return {"verdict": "DIFFICULTY SIGNAL", "reason": "Hard-correct and hard-wrong scores remain close, probe adds more difficulty information than outcome information, and signal survives at prompt/text/vision pre-decision positions."}
    if hard_delta > 0.10 and err_delta > diff_delta and vision_auc < BASELINE_SURFACE_AUC:
        return {"verdict": "IMMINENT-ERROR LEAKAGE", "reason": "Probe score tracks hard-wrong over hard-correct, adds more outcome information than difficulty information, and vision/pre-decision signal is weak."}
    return {"verdict": "MIXED", "reason": "The probe contains a real difficulty component beyond outcome, but part of the strongest late-token signal also tracks this-sample correctness and vision-token survival is only marginal."}


def render_report(summary: Mapping[str, Any], output_dir: Path) -> str:
    lines: List[str] = ["# Critical-Step Probe Leakage Check", ""]
    lines.append("Disentangles intrinsic difficulty from this-sample imminent error using pre-decision probe scores, held-out p_i labels, and greedy correctness.")
    lines.append("")
    lines.append("## Scope")
    ds = summary["dataset"]
    lines.append(f"- rows: `{ds['rows']}`")
    lines.append(f"- bottom-2 hard steps: `{ds['bottom2_critical']}`")
    lines.append(f"- greedy wrong steps: `{ds['greedy_wrong']}`")
    lines.append(f"- best representation tested: `{summary['best_representation']}`")
    lines.append("")
    lines.append("## Test A: Hard-Correct vs Hard-Wrong")
    lines.append("")
    lines.append("| group | n | mean probe score | median | mean p_i | correct rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for group in ["hard_correct", "hard_wrong", "easy_correct", "easy_wrong"]:
        item = summary["test_a"][group]
        lines.append(f"| {group} | {item['n']} | {fmt(item.get('mean_score'))} | {fmt(item.get('median_score'))} | {fmt(item.get('mean_p_i'))} | {pct(item.get('correct_rate'))} |")
    c = summary["test_a"]["contrasts"]
    lines.append("")
    lines.append(f"Hard wrong - hard correct probe-score gap: `{fmt(c.get('hard_wrong_minus_hard_correct'))}`.")
    lines.append(f"Hard correct - easy correct difficulty gap: `{fmt(c.get('hard_correct_minus_easy_correct'))}`.")
    lines.append(f"Easy wrong - easy correct outcome-control gap: `{fmt(c.get('easy_wrong_minus_easy_correct'))}`.")
    lines.append("")
    lines.append("## Test B: p_i Difficulty vs This-Sample Outcome")
    lines.append("")
    tb = summary["test_b"]
    lines.append("| quantity | value |")
    lines.append("|---|---:|")
    for key in [
        "score_vs_difficulty_corr",
        "score_vs_error_corr",
        "partial_score_difficulty_given_error",
        "partial_score_error_given_difficulty",
        "difficulty_delta_r2_probe_beyond_error",
        "error_delta_r2_probe_beyond_difficulty",
        "probe_auc_for_bottom2",
        "probe_auc_for_this_sample_error",
        "difficulty_auc_for_this_sample_error",
    ]:
        lines.append(f"| `{key}` | {fmt(tb.get(key))} |")
    lines.append("")
    lines.append("## Test C: Position Survival")
    lines.append("")
    lines.append("| position family | best representation | bottom-2 AUC |")
    lines.append("|---|---|---:|")
    for family, item in summary["test_c"].items():
        lines.append(f"| {family} | `{item['best_representation']}` | {pct(item.get('best_auc'))} |")
    lines.append("")
    lines.append("## Operational Triage")
    lines.append("")
    lines.append("Bottom-2 by probe score:")
    lines.append("")
    lines.append("| budget | selected | recall | precision | random recall |")
    lines.append("|---:|---:|---:|---:|---:|")
    for item in summary["triage"]["bottom2_by_probe"]:
        lines.append(f"| {pct(item['budget_fraction'])} | {item['selected_steps']} | {pct(item['recall'])} | {pct(item['precision'])} | {pct(item['random_recall'])} |")
    lines.append("")
    lines.append("This-sample error by same probe score:")
    lines.append("")
    lines.append("| budget | selected | recall | precision | random recall |")
    lines.append("|---:|---:|---:|---:|---:|")
    for item in summary["triage"]["error_by_probe"]:
        lines.append(f"| {pct(item['budget_fraction'])} | {item['selected_steps']} | {pct(item['recall'])} | {pct(item['precision'])} | {pct(item['random_recall'])} |")
    lines.append("")
    lines.append("## Test D: Sample Stability")
    lines.append("")
    lines.append("Skipped: extracted activations are one pre-decision prompt state per step, not per stochastic sampled action. Tests A-C are the leakage decision tests here.")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    gate = summary["gate"]
    lines.append(f"**{gate['verdict']}**")
    lines.append("")
    lines.append(gate["reason"])
    lines.append("")
    lines.append("## Guardrails")
    lines.append("")
    lines.append("- Probe inputs are pre-decision prompt-forward activations from the prior representation run.")
    lines.append("- p_i/bottom-k are labels only; greedy correctness is used only for leakage/disentangling evaluation.")
    lines.append("- No base model training or reward/matcher signal enters probe inputs.")
    lines.append("- Current available extracted positions are prompt_last/action-generation-position proxy, text_mean, vision_mean, and vision_max; no post-action generated token state is used.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{output_dir / 'leakage.md'}`")
    lines.append(f"- `{output_dir / 'summary.json'}`")
    lines.append(f"- `{output_dir / 'per_step.jsonl'}`")
    lines.append("")
    lines.append("STOP for review.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation-dir", default=DEFAULT_REP_DIR)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--probe-dim", type=int, default=256)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--representations", default="", help="Optional comma-separated subset")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_rows, scores, scored = score_representations(args)
    if not scored["results"]:
        raise RuntimeError("no representation scores produced")
    best_rep = scored["results"][0]["representation"]
    candidates = read_candidate_map(Path(args.candidates))
    rows = build_rows(meta_rows, candidates, scores, best_rep)
    hard_count = sum(1 for row in rows if row["bottom2_critical"])
    wrong_count = sum(1 for row in rows if row["this_sample_wrong"])
    summary = {
        "inputs": {"representation_dir": args.representation_dir, "candidates": args.candidates},
        "dataset": {"rows": len(rows), "bottom2_critical": hard_count, "greedy_wrong": wrong_count},
        "best_representation": best_rep,
        "representation_results": scored["results"],
        "scoring_manifest": scored["manifest"],
        "test_a": test_a(rows),
        "test_b": test_b(rows),
        "test_c": test_c(scored["results"]),
        "triage": triage(rows),
        "test_d": {"status": "skipped", "reason": "No per-sample activations; one pre-decision prompt state per step."},
    }
    summary["gate"] = decide_gate(summary["test_a"], summary["test_b"], summary["test_c"])
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "per_step.jsonl", rows)
    (output_dir / "leakage.md").write_text(render_report(summary, output_dir), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "best_representation": best_rep, "gate": summary["gate"], "test_b": summary["test_b"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
