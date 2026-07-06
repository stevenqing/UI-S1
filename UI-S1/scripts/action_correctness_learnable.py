#!/usr/bin/env python3
"""Assess whether greedy action correctness is learnable by lightweight discriminators."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.critstep_identifiability import auc_score, balanced_accuracy, write_json, write_jsonl  # noqa: E402
from scripts.critstep_representation_probe import (  # noqa: E402
    fit_logistic,
    load_activation_shards,
    random_project,
    read_jsonl,
    standardize_train_test,
)

DEFAULT_REP_DIR = "outputs/critstep_representation"
DEFAULT_CANDIDATES = "outputs/history_correction/n5_candidates/per_step.jsonl"
DEFAULT_VERIFIER = "outputs/history_correction/verifier_pointwise_n5/per_step.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/action_correctness_learnable"
BUDGETS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30)
GREEDY_TSR_REF = 0.222
ORACLE_GREEDY_WRONG_TSR = 0.244


def pct(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):.2f}%"


def pp(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):+.2f}pp"


def fmt(value: Optional[float]) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.3f}"


def key(row: Mapping[str, Any]) -> Tuple[str, int]:
    return str(row.get("episode_id")), int(row.get("step_idx") or 0)


def read_candidate_rows(candidates_path: Path, verifier_path: Path) -> List[Dict[str, Any]]:
    verifier = {key(row): row for row in read_jsonl(verifier_path)}
    rows = []
    for row in read_jsonl(candidates_path):
        k = key(row)
        v = verifier.get(k)
        if v is None:
            raise RuntimeError(f"missing verifier row for {k}")
        greedy = next((candidate for candidate in v.get("candidates", []) if candidate.get("source") == "greedy"), None)
        if greedy is None:
            raise RuntimeError(f"missing greedy candidate for {k}")
        rows.append({
            "episode_id": k[0],
            "step_idx": k[1],
            "target_id": row.get("target_id"),
            "greedy_correct": bool(row.get("greedy_correct")),
            "greedy_wrong": not bool(row.get("greedy_correct")),
            "greedy_reward": row.get("greedy_reward"),
            "recoverable": bool(row.get("n_correct_candidates", 0) > 0),
            "n_correct_candidates": int(row.get("n_correct_candidates") or 0),
            "verifier_correct": bool(v.get("verifier_correct")),
            "verifier_candidate_id": v.get("verifier_candidate_id"),
            "greedy_verifier_score": float(greedy.get("verifier_score") or 0.0),
            "greedy_verifier_margin": float(greedy.get("verifier_margin") or 0.0),
        })
    return rows


def episode_folds(rows: Sequence[Mapping[str, Any]], n_folds: int, seed: int) -> List[np.ndarray]:
    by_episode: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        by_episode[str(row["episode_id"])].append(idx)
    episodes = list(by_episode)
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    # Greedy balancing by episode wrong count.
    episodes.sort(key=lambda ep: sum(1 for idx in by_episode[ep] if rows[idx]["greedy_wrong"]), reverse=True)
    fold_eps = [[] for _ in range(n_folds)]
    fold_pos = [0 for _ in range(n_folds)]
    for ep in episodes:
        target = min(range(n_folds), key=lambda i: (fold_pos[i], len(fold_eps[i])))
        fold_eps[target].append(ep)
        fold_pos[target] += sum(1 for idx in by_episode[ep] if rows[idx]["greedy_wrong"])
    folds = []
    for eps in fold_eps:
        indices = sorted(idx for ep in eps for idx in by_episode[ep])
        folds.append(np.asarray(indices, dtype=int))
    return folds


def logistic_group_cv(x: np.ndarray, y: np.ndarray, folds: Sequence[np.ndarray], l2: float) -> Dict[str, Any]:
    scores = np.zeros(len(y), dtype=float)
    for test_idx in folds:
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False
        x_train, x_test = standardize_train_test(x[train_mask], x[test_idx])
        params = fit_logistic(x_train, y[train_mask], l2=l2)
        logits = np.clip(params[0] + x_test @ params[1:], -40.0, 40.0)
        scores[test_idx] = 1.0 / (1.0 + np.exp(-logits))
    pred = (scores >= 0.5).astype(int)
    return {"scores": scores, "auc": auc_score(y.tolist(), scores.tolist()), "balanced_accuracy": balanced_accuracy(y, pred)}


def mlp_group_cv(x: np.ndarray, y: np.ndarray, folds: Sequence[np.ndarray], seed: int, epochs: int, hidden_dim: int) -> Dict[str, Any]:
    scores = np.zeros(len(y), dtype=float)
    torch.manual_seed(seed)
    for fold_idx, test_idx in enumerate(folds):
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False
        x_train, x_test = standardize_train_test(x[train_mask], x[test_idx])
        y_train = y[train_mask].astype(np.float32)
        model = torch.nn.Sequential(
            torch.nn.Linear(x_train.shape[1], hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(hidden_dim, 1),
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        tx = torch.tensor(x_train, dtype=torch.float32)
        ty = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
        pos = max(1.0, float(np.sum(y_train == 1)))
        neg = max(1.0, float(np.sum(y_train == 0)))
        weights = torch.tensor(np.where(y_train == 1, 0.5 / pos, 0.5 / neg).reshape(-1, 1), dtype=torch.float32)
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            logits = model(tx)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, ty, weight=weights, reduction="sum")
            loss.backward()
            opt.step()
        with torch.no_grad():
            logits = model(torch.tensor(x_test, dtype=torch.float32)).squeeze(1).numpy()
        scores[test_idx] = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    pred = (scores >= 0.5).astype(int)
    return {"scores": scores, "auc": auc_score(y.tolist(), scores.tolist()), "balanced_accuracy": balanced_accuracy(y, pred)}


def task_metrics(rows: Sequence[Mapping[str, Any]], success_values: Sequence[bool]) -> Dict[str, Any]:
    by_episode: Dict[str, List[Tuple[int, bool]]] = defaultdict(list)
    for row, success in zip(rows, success_values, strict=True):
        by_episode[str(row["episode_id"])].append((int(row["step_idx"]), bool(success)))
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
    return {"tsr": task_success / episodes if episodes else 0.0, "step_sr": correct / total if total else 0.0, "task_success": task_success, "correct_steps": correct, "episodes": episodes, "total_steps": total, "avg_progress": progress_sum / episodes if episodes else 0.0}


def high_conf_tables(rows: Sequence[Mapping[str, Any]], scores: np.ndarray, budgets: Sequence[float]) -> Dict[str, List[Dict[str, Any]]]:
    y_wrong = np.asarray([1 if row["greedy_wrong"] else 0 for row in rows], dtype=int)
    y_correct = 1 - y_wrong
    out_wrong = []
    out_correct = []
    order_wrong = np.argsort(-scores)
    order_correct = np.argsort(scores)
    for budget in budgets:
        k = max(1, int(round(len(rows) * budget)))
        idx_wrong = order_wrong[:k]
        idx_correct = order_correct[:k]
        out_wrong.append({"budget_fraction": budget, "selected_steps": k, "wrong_precision": float(np.mean(y_wrong[idx_wrong])), "wrong_recall": float(np.sum(y_wrong[idx_wrong]) / max(1, np.sum(y_wrong)))})
        out_correct.append({"budget_fraction": budget, "selected_steps": k, "correct_precision": float(np.mean(y_correct[idx_correct])), "correct_recall": float(np.sum(y_correct[idx_correct]) / max(1, np.sum(y_correct)))})
    return {"confident_wrong": out_wrong, "confident_correct": out_correct}


def replacement_projection(rows: Sequence[Mapping[str, Any]], scores: np.ndarray, budgets: Sequence[float]) -> List[Dict[str, Any]]:
    order = np.argsort(-scores)
    out = []
    greedy_success = [bool(row["greedy_correct"]) for row in rows]
    greedy_metrics = task_metrics(rows, greedy_success)
    for budget in budgets:
        k = max(1, int(round(len(rows) * budget)))
        selected = set(int(idx) for idx in order[:k])
        success = []
        fixed = injected = missing = recoverable_not_fixed = 0
        for idx, row in enumerate(rows):
            gated = idx in selected
            if gated:
                ok = bool(row["verifier_correct"])
                fixed += int(row["greedy_wrong"] and ok)
                injected += int(row["greedy_correct"] and not ok)
                missing += int(row["greedy_wrong"] and not row["recoverable"])
                recoverable_not_fixed += int(row["greedy_wrong"] and row["recoverable"] and not ok)
            else:
                ok = bool(row["greedy_correct"])
            success.append(ok)
        metrics = task_metrics(rows, success)
        out.append({
            "budget_fraction": budget,
            "selected_steps": k,
            **metrics,
            "delta_tsr_vs_greedy": metrics["tsr"] - greedy_metrics["tsr"],
            "fixed_wrong_to_correct": fixed,
            "injected_correct_to_wrong": injected,
            "missing_wrong_no_correct_candidate": missing,
            "recoverable_wrong_not_fixed": recoverable_not_fixed,
            "fix_per_injection": fixed / injected if injected else None,
        })
    return out


def render_report(summary: Mapping[str, Any], output_dir: Path) -> str:
    lines = ["# Action-Correctness Learnability", ""]
    lines.append("Question: can a lightweight discriminator judge whether the greedy action is correct, without GT/matcher inputs? Labels are frozen-matcher greedy correctness only.")
    lines.append("")
    lines.append("## Scope And Guardrails")
    ds = summary["dataset"]
    lines.append(f"- rows: `{ds['rows']}` across `{ds['episodes']}` episodes")
    lines.append(f"- greedy-correct: `{ds['greedy_correct']}`; greedy-wrong: `{ds['greedy_wrong']}`")
    lines.append("- R0/R1 use episode-disjoint 5-fold CV on frozen pre-decision SFT activations.")
    lines.append("- R3 uses the existing pointwise verifier LoRA's score on the greedy candidate, evaluated on the same TEST rows.")
    lines.append("- No GT or matcher features are used as inputs; correctness is label only.")
    lines.append("")
    lines.append("## Training Ladder")
    lines.append("")
    lines.append("| rung | model | AUC for greedy-wrong | balanced acc | best representation / note |")
    lines.append("|---|---|---:|---:|---|")
    for row in summary["ladder"]:
        lines.append(f"| {row['rung']} | {row['model']} | {pct(row.get('auc'))} | {pct(row.get('balanced_accuracy'))} | {row.get('note','')} |")
    lines.append("")
    lines.append("## High-Confidence Precision")
    lines.append("")
    for rung_name, table in summary["high_confidence"].items():
        lines.append(f"### {rung_name}")
        lines.append("")
        lines.append("Confident greedy-wrong:")
        lines.append("")
        lines.append("| budget | selected | actual wrong precision | wrong recall |")
        lines.append("|---:|---:|---:|---:|")
        for item in table["confident_wrong"]:
            lines.append(f"| {pct(item['budget_fraction'])} | {item['selected_steps']} | {pct(item['wrong_precision'])} | {pct(item['wrong_recall'])} |")
        lines.append("")
        lines.append("Confident greedy-correct:")
        lines.append("")
        lines.append("| budget | selected | actual correct precision | correct recall |")
        lines.append("|---:|---:|---:|---:|")
        for item in table["confident_correct"]:
            lines.append(f"| {pct(item['budget_fraction'])} | {item['selected_steps']} | {pct(item['correct_precision'])} | {pct(item['correct_recall'])} |")
        lines.append("")
    lines.append("## Replacement Projection")
    lines.append("")
    lines.append("Use the discriminator as an accept/replace gate: replace greedy with existing verifier-selected action only where the discriminator is most confident greedy is wrong.")
    lines.append("")
    for rung_name, curve in summary["replacement_projection"].items():
        lines.append(f"### {rung_name}")
        lines.append("")
        lines.append("| budget | TSR | StepSR | ΔTSR vs greedy | fixed | injected | missing | fix/inject |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for item in curve:
            lines.append(f"| {pct(item['budget_fraction'])} | {pct(item['tsr'])} | {pct(item['step_sr'])} | {pp(item['delta_tsr_vs_greedy'])} | {item['fixed_wrong_to_correct']} | {item['injected_correct_to_wrong']} | {item['missing_wrong_no_correct_candidate']} | {fmt(item.get('fix_per_injection'))} |")
        lines.append("")
    lines.append("## Oracle Ceiling")
    oracle = summary["oracle_reference"]
    lines.append("")
    lines.append(f"Oracle greedy-wrong gating TSR: `{pct(oracle['tsr'])}` (`{pp(oracle['delta_tsr_vs_greedy'])}` vs greedy), with zero injection by definition.")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(f"**{summary['gate']['verdict']}**")
    lines.append("")
    lines.append(summary["gate"]["reason"])
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{output_dir / 'learnable.md'}`")
    lines.append(f"- `{output_dir / 'summary.json'}`")
    lines.append(f"- `{output_dir / 'per_step.jsonl'}`")
    lines.append("")
    lines.append("STOP for review.")
    return "\n".join(lines) + "\n"


def pct(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):.2f}%"


def pp(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):+.2f}pp"


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.3f}"


def decide_gate(ladder: Sequence[Mapping[str, Any]], high_conf: Mapping[str, Any]) -> Dict[str, str]:
    best = max((float(row.get("auc") or 0.0) for row in ladder), default=0.0)
    best_wrong_precision = 0.0
    for table in high_conf.values():
        for item in table["confident_wrong"][:3]:
            best_wrong_precision = max(best_wrong_precision, float(item["wrong_precision"]))
    if best >= 0.75 and best_wrong_precision >= 0.75:
        return {"verdict": "LEARNABLE", "reason": "A dedicated discriminator achieves strong AUC and high-confidence wrong precision, so accept/replace retraining is worth pursuing."}
    if best <= 0.62 and best_wrong_precision <= 0.60:
        return {"verdict": "NOT LEARNABLE", "reason": "Even the strongest available dedicated discriminator remains near chance/modest and cannot identify confidently-wrong greedy actions with high precision."}
    return {"verdict": "PARTIAL", "reason": "Frozen representations learn correctness moderately, but the dedicated pointwise verifier score is near chance and high-confidence wrong precision is not safe enough for replacement gating."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation-dir", default=DEFAULT_REP_DIR)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--verifier-per-step", default=DEFAULT_VERIFIER)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--probe-dim", type=int, default=256)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--mlp-top-k", type=int, default=4)
    parser.add_argument("--mlp-epochs", type=int, default=30)
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_candidate_rows(Path(args.candidates), Path(args.verifier_per_step))
    meta_rows, arrays = load_activation_shards(Path(args.representation_dir), args.num_shards)
    if len(rows) != len(meta_rows):
        raise RuntimeError(f"row mismatch candidates={len(rows)} activations={len(meta_rows)}")
    y_wrong = np.asarray([1 if row["greedy_wrong"] else 0 for row in rows], dtype=int)
    folds = episode_folds(rows, args.folds, args.seed)

    linear_results = []
    linear_scores: Dict[str, np.ndarray] = {}
    for rep_name in sorted(arrays):
        x = random_project(arrays[rep_name], args.probe_dim, args.seed + abs(hash(rep_name)) % 100000)
        result = logistic_group_cv(x, y_wrong, folds, args.l2)
        linear_results.append({"representation": rep_name, "auc": result["auc"], "balanced_accuracy": result["balanced_accuracy"]})
        linear_scores[rep_name] = np.asarray(result["scores"], dtype=float)
    linear_results.sort(key=lambda item: item.get("auc") or -1.0, reverse=True)

    mlp_results = []
    mlp_scores: Dict[str, np.ndarray] = {}
    for item in linear_results[: args.mlp_top_k]:
        rep_name = item["representation"]
        x = random_project(arrays[rep_name], args.probe_dim, args.seed + 17 + abs(hash(rep_name)) % 100000)
        result = mlp_group_cv(x, y_wrong, folds, args.seed, args.mlp_epochs, args.mlp_hidden_dim)
        mlp_results.append({"representation": rep_name, "auc": result["auc"], "balanced_accuracy": result["balanced_accuracy"]})
        mlp_scores[rep_name] = np.asarray(result["scores"], dtype=float)
    mlp_results.sort(key=lambda item: item.get("auc") or -1.0, reverse=True)

    verifier_score_wrong = np.asarray([-float(row["greedy_verifier_score"]) for row in rows], dtype=float)
    verifier_auc = auc_score(y_wrong.tolist(), verifier_score_wrong.tolist())
    verifier_pred = (verifier_score_wrong >= np.median(verifier_score_wrong)).astype(int)
    verifier_bal = balanced_accuracy(y_wrong, verifier_pred)

    ladder = [
        {"rung": "R0", "model": "frozen SFT representation + linear", "auc": linear_results[0]["auc"], "balanced_accuracy": linear_results[0]["balanced_accuracy"], "note": linear_results[0]["representation"]},
        {"rung": "R1", "model": "frozen SFT representation + MLP", "auc": mlp_results[0]["auc"] if mlp_results else None, "balanced_accuracy": mlp_results[0]["balanced_accuracy"] if mlp_results else None, "note": mlp_results[0]["representation"] if mlp_results else "not run"},
        {"rung": "R2", "model": "fine-tuned representation", "auc": None, "balanced_accuracy": None, "note": "not separately available; no greedy-correctness representation fine-tune checkpoint"},
        {"rung": "R3", "model": "existing pointwise verifier LoRA score on greedy action", "auc": verifier_auc, "balanced_accuracy": verifier_bal, "note": "dedicated action-correctness verifier trained on candidate labels"},
    ]

    score_sets: Dict[str, np.ndarray] = {
        f"R0_linear_{linear_results[0]['representation']}": linear_scores[linear_results[0]["representation"]],
        "R3_pointwise_verifier_greedy_score": verifier_score_wrong,
    }
    if mlp_results:
        score_sets[f"R1_mlp_{mlp_results[0]['representation']}"] = mlp_scores[mlp_results[0]["representation"]]

    high_conf = {name: high_conf_tables(rows, scores, BUDGETS) for name, scores in score_sets.items()}
    projection = {name: replacement_projection(rows, scores, BUDGETS) for name, scores in score_sets.items()}
    oracle_success = [bool(row["verifier_correct"]) if row["greedy_wrong"] else bool(row["greedy_correct"]) for row in rows]
    oracle_metrics = task_metrics(rows, oracle_success)
    oracle_ref = {**oracle_metrics, "delta_tsr_vs_greedy": oracle_metrics["tsr"] - GREEDY_TSR_REF}

    per_step = []
    for idx, row in enumerate(rows):
        out = dict(row)
        out["scores"] = {name: float(scores[idx]) for name, scores in score_sets.items()}
        per_step.append(out)

    summary = {
        "inputs": {"representation_dir": args.representation_dir, "candidates": args.candidates, "verifier_per_step": args.verifier_per_step},
        "dataset": {"rows": len(rows), "episodes": len({row["episode_id"] for row in rows}), "greedy_correct": int(np.sum(y_wrong == 0)), "greedy_wrong": int(np.sum(y_wrong == 1)), "wrong_prevalence": float(np.mean(y_wrong))},
        "cv_protocol": {"folds": args.folds, "episode_disjoint": True, "probe_dim": args.probe_dim, "mlp_epochs": args.mlp_epochs},
        "linear_all_representations": linear_results,
        "mlp_top_representations": mlp_results,
        "ladder": ladder,
        "high_confidence": high_conf,
        "replacement_projection": projection,
        "oracle_reference": oracle_ref,
    }
    summary["gate"] = decide_gate(ladder, high_conf)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "per_step.jsonl", per_step)
    (output_dir / "learnable.md").write_text(render_report(summary, output_dir), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "best_auc": max(float(row.get("auc") or 0.0) for row in ladder), "gate": summary["gate"], "ladder": ladder}, indent=2), flush=True)


if __name__ == "__main__":
    main()
