#!/usr/bin/env python3
"""Fast targeted N=50 discriminator-gated verifier recomposition."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.critstep_identifiability import auc_score, balanced_accuracy, fit_logistic  # noqa: E402
from scripts.critstep_representation_probe import load_activation_shards, random_project, read_jsonl, standardize_train_test  # noqa: E402


DEFAULT_CANDIDATES = "outputs/critstep_binlift_lean/test_candidates/per_step.jsonl"
DEFAULT_REPRESENTATION_DIR = "outputs/critstep_representation"
DEFAULT_VERIFIER_SCORED = "outputs/n50_gated_e2e_fast/verifier_pointwise_top10/per_step.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/n50_gated_e2e_fast"
DEFAULT_REPRESENTATION = "L24_prompt_last"
BUDGETS = (0.0, 0.01, 0.02, 0.05, 0.10)
PROJECTION_N50_035 = 0.30985
POOL_CEILING_N50 = 0.473
N5_OLD_VERIFIER_TSR = 0.190


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


def key(row: Mapping[str, Any]) -> Tuple[str, int]:
    return str(row.get("episode_id")), int(row.get("step_idx") or 0)


def target_key(row: Mapping[str, Any]) -> str:
    return str(row.get("target_id") or f"{row.get('episode_id')}:{row.get('step_idx')}")


def episode_split(rows: Sequence[Mapping[str, Any]], train_fraction: float, seed: int) -> Tuple[set[str], set[str]]:
    by_episode: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_episode[str(row.get("episode_id"))].append(idx)
    episodes = list(by_episode)
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    episodes.sort(key=lambda ep: sum(1 for idx in by_episode[ep] if not rows[idx].get("greedy_correct")), reverse=True)
    n_train = int(round(len(episodes) * train_fraction))
    train_eps: set[str] = set()
    test_eps: set[str] = set()
    train_wrong = test_wrong = 0
    for ep in episodes:
        wrong = sum(1 for idx in by_episode[ep] if not rows[idx].get("greedy_correct"))
        if len(train_eps) < n_train and (train_wrong <= test_wrong or len(test_eps) >= len(episodes) - n_train):
            train_eps.add(ep)
            train_wrong += wrong
        else:
            test_eps.add(ep)
            test_wrong += wrong
    return train_eps, test_eps


def train_linear(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, l2: float) -> np.ndarray:
    xtr, xte = standardize_train_test(x_train, x_test)
    params = fit_logistic(xtr, y_train, l2=l2)
    logits = np.clip(params[0] + xte @ params[1:], -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def train_mlp(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int, epochs: int, hidden_dim: int) -> np.ndarray:
    torch.manual_seed(seed)
    xtr, xte = standardize_train_test(x_train, x_test)
    y_float = y_train.astype(np.float32)
    model = torch.nn.Sequential(
        torch.nn.Linear(xtr.shape[1], hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.10),
        torch.nn.Linear(hidden_dim, 1),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    tx = torch.tensor(xtr, dtype=torch.float32)
    ty = torch.tensor(y_float.reshape(-1, 1), dtype=torch.float32)
    pos = max(1.0, float(np.sum(y_float == 1)))
    neg = max(1.0, float(np.sum(y_float == 0)))
    weights = torch.tensor(np.where(y_float == 1, 0.5 / pos, 0.5 / neg).reshape(-1, 1), dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        logits = model(tx)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, ty, weight=weights, reduction="sum")
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = model(torch.tensor(xte, dtype=torch.float32)).squeeze(1).numpy()
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def task_metrics(rows: Sequence[Mapping[str, Any]], success: Sequence[bool]) -> Dict[str, Any]:
    by_episode: Dict[str, List[Tuple[int, bool]]] = defaultdict(list)
    for row, ok in zip(rows, success, strict=True):
        by_episode[str(row.get("episode_id"))].append((int(row.get("step_idx") or 0), bool(ok)))
    task_success = 0
    progress_sum = 0.0
    for steps in by_episode.values():
        ordered = [ok for _, ok in sorted(steps)]
        first_error = next((idx for idx, ok in enumerate(ordered, 1) if not ok), None)
        if first_error is None:
            task_success += 1
            progress_sum += 1.0
        else:
            progress_sum += (first_error - 1) / len(ordered) if ordered else 0.0
    return {
        "episodes": len(by_episode),
        "total_steps": len(success),
        "correct_steps": sum(1 for ok in success if ok),
        "task_success": task_success,
        "tsr": task_success / len(by_episode) if by_episode else 0.0,
        "step_sr": sum(1 for ok in success if ok) / len(success) if success else 0.0,
        "avg_progress": progress_sum / len(by_episode) if by_episode else 0.0,
    }


def recoverable(row: Mapping[str, Any]) -> bool:
    return any(bool(candidate.get("is_correct")) for candidate in (row.get("candidates") or [])[:50])


def prepare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(Path(args.candidates))
    meta_rows, arrays = load_activation_shards(Path(args.representation_dir), args.num_shards)
    if len(rows) != len(meta_rows):
        raise RuntimeError(f"row mismatch: candidates={len(rows)} activations={len(meta_rows)}")
    if args.representation not in arrays:
        raise KeyError(f"missing representation {args.representation}; available={sorted(arrays)[:20]}")
    train_eps, test_eps = episode_split(rows, args.train_fraction, args.seed)
    train_idx = np.asarray([idx for idx, row in enumerate(rows) if str(row.get("episode_id")) in train_eps], dtype=int)
    test_idx = np.asarray([idx for idx, row in enumerate(rows) if str(row.get("episode_id")) in test_eps], dtype=int)
    y = np.asarray([0 if row.get("greedy_correct") else 1 for row in rows], dtype=int)
    x = random_project(arrays[args.representation], args.probe_dim, args.seed + 1009)
    linear_scores = train_linear(x[train_idx], y[train_idx], x[test_idx], args.l2)
    mlp_scores = train_mlp(x[train_idx], y[train_idx], x[test_idx], args.seed, args.mlp_epochs, args.mlp_hidden_dim)
    test_rows = [rows[int(idx)] for idx in test_idx]
    y_test = y[test_idx]
    method_scores = {"linear": linear_scores, "mlp": mlp_scores}
    metrics = {}
    for name, scores in method_scores.items():
        metrics[name] = {
            "auc_wrong": auc_score(y_test.tolist(), scores.tolist()),
            "balanced_accuracy_at_0_5": balanced_accuracy(y_test, (scores >= 0.5).astype(int)),
        }
    chosen_scores = method_scores[args.discriminator]
    order = np.argsort(-chosen_scores)
    max_k = max(1, int(round(len(test_rows) * args.max_verify_fraction)))
    selected_indices = set(int(idx) for idx in order[:max_k])
    score_rows: List[Dict[str, Any]] = []
    missing_selected = 0
    for local_idx, row in enumerate(test_rows):
        if local_idx not in selected_indices:
            continue
        if (not row.get("greedy_correct")) and (not recoverable(row)):
            missing_selected += 1
            continue
        item = dict(row)
        item["candidates"] = (row.get("candidates") or [])[:50]
        item["n_candidates"] = len(item["candidates"])
        item["score_set_requires_verifier"] = True
        item["filter_reason"] = "targeted_gate_selected_for_fast_e2e"
        score_rows.append(item)
    score_set_path = output_dir / "selected_for_verifier.jsonl"
    write_jsonl(score_set_path, score_rows)
    score_records = []
    for local_idx, row in enumerate(test_rows):
        score_records.append({
            "target_id": row.get("target_id"),
            "episode_id": str(row.get("episode_id")),
            "step_idx": int(row.get("step_idx") or 0),
            "greedy_correct": bool(row.get("greedy_correct")),
            "recoverable_n50": recoverable(row),
            "wrong_score": float(chosen_scores[local_idx]),
            "correct_score": float(1.0 - chosen_scores[local_idx]),
            "rank_by_wrong_score": int(np.where(order == local_idx)[0][0]) + 1,
            "selected_for_scoring": local_idx in selected_indices,
            "needs_verifier_score": local_idx in selected_indices and not ((not row.get("greedy_correct")) and (not recoverable(row))),
        })
    write_jsonl(output_dir / "discriminator_scores.jsonl", score_records)
    summary = {
        "phase": "prepare",
        "inputs": {"candidates": args.candidates, "representation_dir": args.representation_dir, "representation": args.representation},
        "split": {
            "train_episodes": len(train_eps),
            "test_episodes": len(test_eps),
            "train_steps": int(len(train_idx)),
            "test_steps": int(len(test_idx)),
            "episode_intersection": len(train_eps & test_eps),
            "train_fraction": args.train_fraction,
        },
        "discriminator": {"chosen": args.discriminator, "metrics": metrics, "probe_dim": args.probe_dim, "mlp_epochs": args.mlp_epochs},
        "targeted_scoring": {
            "max_verify_fraction": args.max_verify_fraction,
            "selected_test_steps": max_k,
            "rows_written_for_verifier": len(score_rows),
            "selected_missing_wrong_no_score_needed": missing_selected,
            "candidate_jobs": sum(len(row.get("candidates") or []) for row in score_rows),
            "score_set_path": str(score_set_path),
        },
        "budgets": list(BUDGETS),
    }
    write_json(output_dir / "prepare_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    rows = read_jsonl(Path(args.candidates))
    score_records = read_jsonl(output_dir / "discriminator_scores.jsonl")
    by_score_key = {target_key(row): row for row in score_records}
    test_rows = [row for row in rows if target_key(row) in by_score_key]
    test_rows.sort(key=lambda row: (int(row.get("episode_id") or 0), int(row.get("step_idx") or 0)))
    scored_rows = {target_key(row): row for row in read_jsonl(Path(args.verifier_scored))}
    ordered = sorted(test_rows, key=lambda row: int(by_score_key[target_key(row)]["rank_by_wrong_score"]))
    base_metrics = task_metrics(test_rows, [bool(row.get("greedy_correct")) for row in test_rows])
    curve = []
    per_step_by_budget: Dict[str, List[Dict[str, Any]]] = {}
    for budget in BUDGETS:
        k = int(round(len(test_rows) * budget))
        selected_targets = {target_key(row) for row in ordered[:k]}
        success = []
        fixed = injected = missing = recoverable_not_fixed = selected_gw = selected_gc = verifier_calls = 0
        rows_out = []
        for row in test_rows:
            tkey = target_key(row)
            selected = tkey in selected_targets
            greedy_correct = bool(row.get("greedy_correct"))
            row_recoverable = recoverable(row)
            scored = scored_rows.get(tkey)
            verifier_correct = bool(scored.get("verifier_correct")) if scored is not None else False
            if selected:
                selected_gc += int(greedy_correct)
                selected_gw += int(not greedy_correct)
                if (not greedy_correct) and (not row_recoverable):
                    ok = False
                    missing += 1
                elif scored is None:
                    raise RuntimeError(f"selected row lacks verifier score and is not missing: {tkey}")
                else:
                    verifier_calls += 1
                    ok = verifier_correct
                    fixed += int((not greedy_correct) and verifier_correct)
                    injected += int(greedy_correct and not verifier_correct)
                    recoverable_not_fixed += int((not greedy_correct) and row_recoverable and not verifier_correct)
            else:
                ok = greedy_correct
            success.append(ok)
            rows_out.append({
                "target_id": row.get("target_id"),
                "episode_id": str(row.get("episode_id")),
                "step_idx": int(row.get("step_idx") or 0),
                "wrong_score": by_score_key[tkey]["wrong_score"],
                "rank_by_wrong_score": by_score_key[tkey]["rank_by_wrong_score"],
                "selected_for_verifier": selected,
                "greedy_correct": greedy_correct,
                "recoverable_n50": row_recoverable,
                "verifier_scored": scored is not None,
                "verifier_candidate_id": scored.get("verifier_candidate_id") if scored else None,
                "verifier_correct": verifier_correct if scored is not None else None,
                "policy_correct": ok,
                "fixed": bool(selected and (not greedy_correct) and verifier_correct),
                "injected": bool(selected and greedy_correct and scored is not None and not verifier_correct),
                "missing": bool(selected and (not greedy_correct) and (not row_recoverable)),
            })
        metrics = task_metrics(test_rows, success)
        curve.append({
            "budget_fraction": budget,
            "selected_steps": k,
            "verifier_calls": verifier_calls,
            "verify_fraction": verifier_calls / len(test_rows) if test_rows else 0.0,
            **metrics,
            "delta_tsr_vs_greedy_test": metrics["tsr"] - base_metrics["tsr"],
            "delta_tsr_vs_n5_old_verifier_ref": metrics["tsr"] - N5_OLD_VERIFIER_TSR,
            "delta_tsr_vs_projection_n50_035_ref": metrics["tsr"] - PROJECTION_N50_035,
            "ceiling_fraction_of_n50_pool": (metrics["tsr"] - base_metrics["tsr"]) / max(1e-9, POOL_CEILING_N50 - base_metrics["tsr"]),
            "fixed_wrong_to_correct": fixed,
            "injected_correct_to_wrong": injected,
            "missing_selected_wrong": missing,
            "recoverable_wrong_not_fixed": recoverable_not_fixed,
            "selected_greedy_wrong": selected_gw,
            "selected_greedy_correct": selected_gc,
            "fix_per_injection": fixed / injected if injected else None,
            "tsr_gain_per_verify_fraction": (metrics["tsr"] - base_metrics["tsr"]) / (verifier_calls / len(test_rows)) if verifier_calls else None,
        })
        per_step_by_budget[f"K{int(round(budget * 100))}"] = rows_out
    best = max(curve, key=lambda item: item["tsr"])
    selected_recoverable_wrong = [row for row in test_rows if target_key(row) in {target_key(item) for item in ordered[: int(round(len(test_rows) * max(BUDGETS)))]} and (not row.get("greedy_correct")) and recoverable(row)]
    sel_acc_den = len(selected_recoverable_wrong)
    sel_acc_num = sum(1 for row in selected_recoverable_wrong if scored_rows.get(target_key(row), {}).get("verifier_correct"))
    gate = decide_gate(best, base_metrics)
    summary = {
        "phase": "evaluate",
        "inputs": {"candidates": args.candidates, "verifier_scored": args.verifier_scored},
        "fast_mode_note": "Exact only for thresholds up to the prepared max budget; missing-wrong selected rows need no verifier score and are counted wrong.",
        "references": {"n5_old_verifier_tsr": N5_OLD_VERIFIER_TSR, "n50_pool_ceiling_tsr": POOL_CEILING_N50, "n50_projection_selection_035_tsr": PROJECTION_N50_035},
        "test_greedy": base_metrics,
        "actual_verifier_selection_accuracy_on_scored_selected_recoverable_wrong": sel_acc_num / sel_acc_den if sel_acc_den else None,
        "selected_recoverable_wrong_scored": sel_acc_den,
        "curve": curve,
        "best": best,
        "gate": gate,
    }
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "per_step.jsonl", per_step_by_budget[f"K{int(round(best['budget_fraction'] * 100))}"])
    (output_dir / "gated_e2e.md").write_text(render_report(summary, output_dir), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "best_tsr": best["tsr"], "best_delta": best["delta_tsr_vs_greedy_test"], "gate": gate}, indent=2), flush=True)


def decide_gate(best: Mapping[str, Any], greedy: Mapping[str, Any]) -> Dict[str, str]:
    delta = float(best["tsr"] - greedy["tsr"])
    projection_gap = float(PROJECTION_N50_035 - best["tsr"])
    if delta > 0.02 and projection_gap <= 0.03:
        return {"verdict": "METHOD POSITIVE", "reason": "Best measured fast-sweep threshold clearly beats greedy and is near the +8.79pp projection."}
    if delta > 0.0:
        return {"verdict": "METHOD PARTIAL", "reason": "Best measured fast-sweep threshold beats greedy, but remains well below the +8.79pp projection."}
    return {"verdict": "METHOD STILL NEGATIVE", "reason": "No measured threshold in the fast targeted sweep beats greedy; current verifier selection/injection remains the bottleneck."}


def render_report(summary: Mapping[str, Any], output_dir: Path) -> str:
    lines = ["# N=50 Gated E2E Fast Sweep", ""]
    lines.append("Teacher-forced GT-screen recomposition. Discriminator is trained on TRAIN episodes only and applied to disjoint TEST episodes. This fast run measures thresholds up to the prepared verifier budget instead of every-step verifier.")
    lines.append("")
    lines.append("## References")
    lines.append("")
    greedy = summary["test_greedy"]
    lines.append(f"- TEST greedy TSR: `{pct(greedy['tsr'])}`; StepSR `{pct(greedy['step_sr'])}`")
    lines.append(f"- N=5 old verifier reference: `{pct(summary['references']['n5_old_verifier_tsr'])}`")
    lines.append(f"- N=50 pool ceiling reference: `{pct(summary['references']['n50_pool_ceiling_tsr'])}`")
    lines.append(f"- N=50 + selection 0.35 projection: `{pct(summary['references']['n50_projection_selection_035_tsr'])}`")
    lines.append("")
    lines.append("## Measured TSR Curve")
    lines.append("")
    lines.append("| budget | selected | verifier calls | TSR | StepSR | ΔTSR vs greedy | fixed | injected | missing | fix/inject | TSR per compute |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary["curve"]:
        fix_inj = "NA" if row.get("fix_per_injection") is None else f"{row['fix_per_injection']:.3f}"
        compute = "NA" if row.get("tsr_gain_per_verify_fraction") is None else pp(row["tsr_gain_per_verify_fraction"])
        lines.append(f"| {pct(row['budget_fraction'])} | {row['selected_steps']} | {row['verifier_calls']} | {pct(row['tsr'])} | {pct(row['step_sr'])} | {pp(row['delta_tsr_vs_greedy_test'])} | {row['fixed_wrong_to_correct']} | {row['injected_correct_to_wrong']} | {row['missing_selected_wrong']} | {fix_inj} | {compute} |")
    lines.append("")
    best = summary["best"]
    lines.append("## Best Threshold")
    lines.append("")
    lines.append(f"Best measured budget: `{pct(best['budget_fraction'])}` with TSR `{pct(best['tsr'])}` and ΔTSR vs TEST greedy `{pp(best['delta_tsr_vs_greedy_test'])}`.")
    lines.append(f"Gap vs +8.79pp projection reference: `{pp(best['delta_tsr_vs_projection_n50_035_ref'])}`; fraction of N=50 ceiling captured: `{pct(best['ceiling_fraction_of_n50_pool'])}`.")
    lines.append("")
    lines.append("## Selection And Injection")
    lines.append("")
    lines.append(f"Actual verifier selection accuracy on scored selected recoverable-greedy-wrong steps: `{pct(summary['actual_verifier_selection_accuracy_on_scored_selected_recoverable_wrong'])}` over `{summary['selected_recoverable_wrong_scored']}` steps.")
    lines.append(f"At best budget: fixed `{best['fixed_wrong_to_correct']}`, injected `{best['injected_correct_to_wrong']}`, missing `{best['missing_selected_wrong']}`, verifier calls `{best['verifier_calls']}`.")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(f"**{summary['gate']['verdict']}**")
    lines.append("")
    lines.append(summary["gate"]["reason"])
    lines.append("")
    lines.append("## Guardrails")
    lines.append("")
    lines.append("- Clean split: discriminator training rows are disjoint from TEST recomposition rows.")
    lines.append("- Frozen matcher; base model unchanged; teacher-forced GT screens.")
    lines.append("- Fast-mode limitation: every-step verifier sanity is not measured here; this run is exact only for thresholds up to the prepared targeted verifier budget.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{output_dir / 'gated_e2e.md'}`")
    lines.append(f"- `{output_dir / 'summary.json'}`")
    lines.append(f"- `{output_dir / 'per_step.jsonl'}`")
    lines.append(f"- `{output_dir / 'selected_for_verifier.jsonl'}`")
    lines.append("")
    lines.append("STOP for review.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["prepare", "evaluate"], required=True)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--representation-dir", default=DEFAULT_REPRESENTATION_DIR)
    parser.add_argument("--representation", default=DEFAULT_REPRESENTATION)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--verifier-scored", default=DEFAULT_VERIFIER_SCORED)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--probe-dim", type=int, default=256)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--mlp-epochs", type=int, default=24)
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--discriminator", choices=["linear", "mlp"], default="mlp")
    parser.add_argument("--max-verify-fraction", type=float, default=0.10)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()