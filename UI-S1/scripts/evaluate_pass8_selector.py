#!/usr/bin/env python3
"""Evaluate frozen Pass@8 selectors after paired outputs are complete."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v13_gui_360.reward import _normalize_action_type, compute_step_reward


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rounded_coord(coord: Any, bucket: int) -> list[int] | None:
    if not isinstance(coord, (list, tuple)) or len(coord) < 2:
        return None
    try:
        values = [float(coord[0]), float(coord[1])]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    return [int(round(value / bucket) * bucket) for value in values]


def action_key(action: Mapping[str, Any] | None, coord_bucket: int) -> str:
    if not isinstance(action, Mapping):
        return "__unparsed__"
    action_type = _normalize_action_type(str(action.get("action", "")))
    payload: dict[str, Any] = {"type": action_type}
    if action_type in {"click", "long_press"}:
        payload["coord"] = rounded_coord(action.get("coordinate"), coord_bucket)
    elif action_type == "swipe":
        payload["start"] = rounded_coord(action.get("startCoordinate") or action.get("coordinate"), coord_bucket)
        payload["end"] = rounded_coord(action.get("endCoordinate"), coord_bucket)
    elif action_type in {"type", "open", "answer", "key"}:
        payload["text"] = str(action.get("text", "")).strip().lower()[:160]
        coordinate = rounded_coord(action.get("coordinate"), coord_bucket)
        if coordinate:
            payload["coord"] = coordinate
    elif action_type == "system_button":
        payload["button"] = str(action.get("button", "")).strip().lower()
    else:
        payload["raw"] = json.dumps(dict(action), ensure_ascii=False, sort_keys=True)[:240]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def score_action(
    action: Mapping[str, Any] | None,
    gt_action: Mapping[str, Any],
    image_w: int,
    image_h: int,
    threshold: float,
    coord_bucket: int,
) -> dict[str, Any]:
    text = f"<action>{json.dumps(dict(action), ensure_ascii=False)}</action>" if isinstance(action, Mapping) else ""
    try:
        reward, info = compute_step_reward(text, dict(gt_action), image_w=image_w, image_h=image_h)
    except Exception as exc:  # diagnostic evaluation should fail closed per action, not abort the study
        return {
            "reward": 0.0,
            "correct": False,
            "pred_action": dict(action) if isinstance(action, Mapping) else None,
            "action_key": "__diagnostic_score_error__",
            "diagnostic_error": f"{type(exc).__name__}: {exc}"[:500],
        }
    predicted = info.get("pred_action")
    return {
        "reward": float(reward),
        "correct": bool(reward >= threshold),
        "pred_action": predicted,
        "action_key": action_key(predicted, coord_bucket),
    }


def index_unique(rows: Sequence[Mapping[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        tid = str(row["target_id"])
        if tid in indexed:
            raise ValueError(f"duplicate target_id in {source}: {tid}")
        indexed[tid] = row
    return indexed


def registered_hash(manifest: Mapping[str, Any], manifest_dir: Path, path: Path) -> str:
    resolved = path.resolve()
    for relative, metadata in manifest.get("artifacts", {}).items():
        if (manifest_dir / relative).resolve() == resolved:
            return str(metadata["sha256"])
    raise ValueError(f"artifact is not registered in frozen manifest: {path}")


def classify(baseline_correct: bool, selected_correct: bool) -> str:
    if baseline_correct and selected_correct:
        return "preserve_correct"
    if baseline_correct and not selected_correct:
        return "regress"
    if not baseline_correct and selected_correct:
        return "rescue"
    return "unresolved"


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot take percentile of an empty sequence")
    return float(sorted(values)[round((len(values) - 1) * q)])


def cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    value: Callable[[Mapping[str, Any]], float],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["episode_id"]), []).append(row)
    units = list(grouped.values())
    if not units:
        raise ValueError("cannot bootstrap an empty evaluation")
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        total = count = 0.0
        for _index in range(len(units)):
            unit = units[rng.randrange(len(units))]
            total += sum(value(row) for row in unit)
            count += len(unit)
        estimates.append(total / count)
    return {
        "mean": sum(estimates) / len(estimates),
        "lo": percentile(estimates, 0.025),
        "hi": percentile(estimates, 0.975),
        "draws": draws,
        "sampling_unit": "episode_id",
        "units": len(units),
    }


def parse_selector(spec: str) -> tuple[str, Path]:
    name, path = spec.split(":", 1)
    return name, Path(path)


def materialize_rows(
    name: str,
    blind: Mapping[str, Mapping[str, Any]],
    sealed: Mapping[str, Mapping[str, Any]],
    selected: Mapping[str, Mapping[str, Any]],
    threshold: float,
    coord_bucket: int,
) -> list[dict[str, Any]]:
    rows = []
    for tid in sorted(blind):
        packet = blind[tid]
        label = sealed[tid]
        output = selected[tid]
        if output.get("packet_sha256") != packet.get("packet_sha256") or label.get("packet_sha256") != packet.get("packet_sha256"):
            raise ValueError(f"packet hash mismatch after unsealing: {tid}")
        candidates = {str(candidate["candidate_id"]): candidate for candidate in packet.get("candidates") or []}
        selected_id = str(output.get("selected_candidate_id") or "BASELINE")
        if selected_id not in candidates:
            selected_id = "BASELINE"
        gt_action = label.get("gt_action") or {}
        image_w = int(packet.get("image_w") or 1040)
        image_h = int(packet.get("image_h") or 736)
        baseline_score = score_action(packet.get("baseline_action"), gt_action, image_w, image_h, threshold, coord_bucket)
        selected_score = score_action(candidates[selected_id].get("action"), gt_action, image_w, image_h, threshold, coord_bucket)
        candidate_scores = {
            candidate_id: score_action(candidate.get("action"), gt_action, image_w, image_h, threshold, coord_bucket)
            for candidate_id, candidate in candidates.items()
        }
        baseline_correct = bool(baseline_score["correct"])
        selected_correct = bool(selected_score["correct"])
        rows.append({
            "selector": name,
            "target_id": tid,
            "episode_id": packet["episode_id"],
            "step_idx": packet["step_idx"],
            "packet_sha256": packet["packet_sha256"],
            "selected_candidate_id": selected_id,
            "changed_from_baseline": selected_id != "BASELINE" and selected_score.get("action_key") != baseline_score.get("action_key"),
            "parse_ok": bool(output.get("parse_ok")),
            "fallback_reason": output.get("fallback_reason"),
            "baseline_correct": baseline_correct,
            "selected_correct": selected_correct,
            "utility_outcome": classify(baseline_correct, selected_correct),
            "utility": int(selected_correct) - int(baseline_correct),
            "oracle_correct": any(bool(score["correct"]) for score in candidate_scores.values()),
            "candidate_count": len(candidates),
            "baseline_action_key": baseline_score.get("action_key"),
            "selected_action_key": selected_score.get("action_key"),
        })
    return rows


def summarize(rows: Sequence[Mapping[str, Any]], draws: int, seed: int) -> dict[str, Any]:
    counts = Counter(str(row["utility_outcome"]) for row in rows)
    n = len(rows)
    baseline_correct = sum(bool(row["baseline_correct"]) for row in rows)
    selected_correct = sum(bool(row["selected_correct"]) for row in rows)
    oracle_correct = sum(bool(row["oracle_correct"]) for row in rows)
    oracle_headroom = oracle_correct - baseline_correct
    selected_gain = selected_correct - baseline_correct
    bootstrap = cluster_bootstrap(rows, lambda row: float(row["utility"]), draws, seed)
    gate = counts["rescue"] > counts["regress"] and (selected_correct - baseline_correct) / n > 0 and bootstrap["lo"] > 0
    return {
        "steps": n,
        "episodes": len({str(row["episode_id"]) for row in rows}),
        "baseline_accuracy": baseline_correct / n,
        "selector_accuracy": selected_correct / n,
        "oracle_packet_accuracy": oracle_correct / n,
        "oracle_headroom_over_baseline": oracle_headroom / n,
        "oracle_headroom_capture": selected_gain / oracle_headroom if oracle_headroom else None,
        "selection_regret_to_oracle": (oracle_correct - selected_correct) / n,
        "net_student_relative_utility": (counts["rescue"] - counts["regress"]) / n,
        "outcomes": {name: counts[name] for name in ("rescue", "regress", "preserve_correct", "unresolved")},
        "changed_coverage": sum(bool(row["changed_from_baseline"]) for row in rows) / n,
        "parse_rate": sum(bool(row["parse_ok"]) for row in rows) / n,
        "fallback_rate": sum(row.get("fallback_reason") is not None for row in rows) / n,
        "episode_cluster_bootstrap": bootstrap,
        "predeclared_gate_pass": gate,
    }


def paired_delta(
    left_name: str,
    left: Sequence[Mapping[str, Any]],
    right_name: str,
    right: Sequence[Mapping[str, Any]],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    left_by_id = {str(row["target_id"]): row for row in left}
    right_by_id = {str(row["target_id"]): row for row in right}
    if set(left_by_id) != set(right_by_id):
        raise ValueError(f"paired selector target mismatch: {left_name} vs {right_name}")
    paired = [{
        "target_id": tid,
        "episode_id": left_by_id[tid]["episode_id"],
        "delta": int(right_by_id[tid]["selected_correct"]) - int(left_by_id[tid]["selected_correct"]),
        "same_selected_action": right_by_id[tid]["selected_action_key"] == left_by_id[tid]["selected_action_key"],
    } for tid in sorted(left_by_id)]
    bootstrap = cluster_bootstrap(paired, lambda row: float(row["delta"]), draws, seed)
    point = sum(row["delta"] for row in paired) / len(paired)
    return {
        "reference": left_name,
        "challenger": right_name,
        "challenger_minus_reference_accuracy": point,
        "challenger_only_correct": sum(row["delta"] == 1 for row in paired),
        "reference_only_correct": sum(row["delta"] == -1 for row in paired),
        "selected_action_agreement": sum(bool(row["same_selected_action"]) for row in paired) / len(paired),
        "episode_cluster_bootstrap": bootstrap,
        "predeclared_stronger_model_win": point > 0 and bootstrap["lo"] > 0,
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    return f"{100.0 * value:+.2f}pp"


def render_report(summary: Mapping[str, Any]) -> str:
    passing_selectors = [name for name, item in summary["selectors"].items() if item["predeclared_gate_pass"]]
    lines = [
        "# Frozen Pass@8 Selector Evaluation",
        "",
        f"Split: **{summary['split']}**. Labels were unsealed only after all paired outputs passed completeness and packet-hash checks.",
        "",
        "Primary utility is student-relative:",
        "",
        "$$u = \\frac{N_{rescue}-N_{regress}}{N_{steps}}.$$",
        "",
        "| selector | baseline acc | selected acc | oracle ceiling | oracle captured | net utility | rescue | regress | changed | parse | 95% cluster CI | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, item in summary["selectors"].items():
        ci = item["episode_cluster_bootstrap"]
        lines.append(
            f"| {name} | {pct(item['baseline_accuracy'])} | {pct(item['selector_accuracy'])} | "
            f"{pct(item['oracle_packet_accuracy'])} | "
            f"{pct(item['oracle_headroom_capture']) if item['oracle_headroom_capture'] is not None else 'NA'} | "
            f"{pp(item['net_student_relative_utility'])} | "
            f"{item['outcomes']['rescue']} | {item['outcomes']['regress']} | {pct(item['changed_coverage'])} | "
            f"{pct(item['parse_rate'])} | [{pp(ci['lo'])}, {pp(ci['hi'])}] | "
            f"{'PASS' if item['predeclared_gate_pass'] else 'FAIL-CLOSED'} |"
        )
    lines.extend(["", "## Paired Corrector Deltas", ""])
    for item in summary.get("paired_deltas", []):
        ci = item["episode_cluster_bootstrap"]
        lines.append(
            f"- **{item['challenger']} − {item['reference']}**: {pp(item['challenger_minus_reference_accuracy'])}, "
            f"95% CI [{pp(ci['lo'])}, {pp(ci['hi'])}] — "
            f"{'stronger-model win' if item['predeclared_stronger_model_win'] else 'no locked win'}; "
            f"discordant correct {item['challenger_only_correct']} vs {item['reference_only_correct']}, "
            f"action agreement {pct(item['selected_action_agreement'])}."
        )
    lines.extend([
        "",
        "## Decision",
        "",
        (
            "The predeclared selector gate **passes** for " + ", ".join(passing_selectors) + ". "
            "This authorizes preparation of a new train-split 25% selected-revision + 75% clean-replay arm; "
            "the dev/locked rows in this study must never be used for training."
            if passing_selectors
            else "The predeclared selector gate fails. No policy training is authorized from this selector study."
        ),
        "The oracle ceiling is diagnostic only and is not a deployable selector.",
        "",
        "## Scope Warning",
        "",
        "This is selector-fresh, not benchmark-fresh: the underlying benchmark episodes and the GT-conditioned critical target set predate this frozen comparison.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--blind", required=True)
    parser.add_argument("--sealed", required=True)
    parser.add_argument("--selector", action="append", required=True, help="name:path; repeatable, reference first")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--match-threshold", type=float, default=0.85)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260213)
    parser.add_argument("--allow-incomplete", action="store_true", help="debug only; forbidden for locked_test")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    blind_path = Path(args.blind)
    expected_blind_hash = registered_hash(manifest, manifest_path.parent, blind_path)
    if sha256_file(blind_path) != expected_blind_hash:
        raise ValueError("blind artifact hash mismatch")
    blind_rows = read_jsonl(blind_path)
    blind = index_unique(blind_rows, str(blind_path))
    split_names = {str(row.get("split")) for row in blind_rows}
    if len(split_names) != 1:
        raise ValueError(f"blind file contains mixed splits: {split_names}")
    split = next(iter(split_names))
    if split == "locked_test" and args.allow_incomplete:
        raise ValueError("--allow-incomplete is forbidden for locked_test")

    # Verify paired output completeness before opening the sealed-label artifact.
    selector_outputs: dict[str, dict[str, dict[str, Any]]] = {}
    selector_models: dict[str, str] = {}
    for spec in args.selector:
        name, path = parse_selector(spec)
        if name in selector_outputs:
            raise ValueError(f"duplicate selector name: {name}")
        rows = read_jsonl(path)
        models = {str(row.get("model") or "") for row in rows}
        if len(models) != 1 or not next(iter(models), ""):
            raise ValueError(f"selector {name} does not have one consistent model identity: {sorted(models)}")
        selector_models[name] = next(iter(models))
        indexed = index_unique(rows, str(path))
        missing = set(blind) - set(indexed)
        extra = set(indexed) - set(blind)
        if extra or (missing and not args.allow_incomplete):
            raise ValueError(f"selector {name} is not paired-complete: missing={len(missing)}, extra={len(extra)}")
        if missing:
            blind = {tid: row for tid, row in blind.items() if tid in indexed}
            for previous_name in selector_outputs:
                selector_outputs[previous_name] = {tid: row for tid, row in selector_outputs[previous_name].items() if tid in blind}
        for tid, row in indexed.items():
            if tid in blind and row.get("packet_sha256") != blind[tid].get("packet_sha256"):
                raise ValueError(f"selector {name} has stale packet hash: {tid}")
            if tid in blind and row.get("selector_name") != name:
                raise ValueError(f"selector identity mismatch for {name}: {tid}")
            if tid in blind and row.get("prompt_version") != manifest.get("prompt_version"):
                raise ValueError(f"selector prompt version mismatch for {name}: {tid}")
        selector_outputs[name] = {tid: row for tid, row in indexed.items() if tid in blind}
    if len(selector_outputs) < 2:
        raise ValueError("paired evaluation requires at least two selector outputs")

    sealed_path = Path(args.sealed)
    expected_sealed_hash = registered_hash(manifest, manifest_path.parent, sealed_path)
    if sha256_file(sealed_path) != expected_sealed_hash:
        raise ValueError("sealed artifact hash mismatch")
    sealed_all = index_unique(read_jsonl(sealed_path), str(sealed_path))
    if not set(blind).issubset(sealed_all):
        raise ValueError("sealed labels do not cover the paired blind targets")
    sealed = {tid: sealed_all[tid] for tid in blind}
    coord_bucket = int(manifest.get("scope", {}).get("coord_bucket") or 25)

    materialized = {
        name: materialize_rows(name, blind, sealed, rows, args.match_threshold, coord_bucket)
        for name, rows in selector_outputs.items()
    }
    summaries = {
        name: summarize(rows, args.bootstrap_draws, args.seed + index)
        for index, (name, rows) in enumerate(materialized.items())
    }
    names = list(materialized)
    paired = [
        paired_delta(names[0], materialized[names[0]], name, materialized[name], args.bootstrap_draws, args.seed + 100 + index)
        for index, name in enumerate(names[1:])
    ]
    summary = {
        "protocol_version": manifest.get("protocol_version"),
        "prompt_version": manifest.get("prompt_version"),
        "split": split,
        "match_threshold": args.match_threshold,
        "paired_complete": not args.allow_incomplete,
        "selector_models": selector_models,
        "selectors": summaries,
        "paired_deltas": paired,
        "predeclared_gate": manifest.get("predeclared_gate"),
        "scope_limitations": manifest.get("leakage_contract", {}).get("known_limitations"),
    }
    out_dir = Path(args.output_dir)
    for name, rows in materialized.items():
        write_jsonl(out_dir / f"{name}_per_step.jsonl", rows)
    write_json(out_dir / "summary.json", summary)
    (out_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(out_dir),
        "split": split,
        "selectors": {name: {"net_utility": item["net_student_relative_utility"], "gate": item["predeclared_gate_pass"]} for name, item in summaries.items()},
        "paired_deltas": paired,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()