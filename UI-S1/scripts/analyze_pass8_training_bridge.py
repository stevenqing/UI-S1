#!/usr/bin/env python3
"""Diagnose whether a positive Pass@8 selector is pure enough for SFT."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def index_unique(rows: Iterable[Mapping[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        target_id = str(row["target_id"])
        if target_id in indexed:
            raise ValueError(f"duplicate target_id in {source}: {target_id}")
        indexed[target_id] = row
    return indexed


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(probability * (1.0 - probability) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def purity_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(bool(row["selected_correct"]) for row in rows)
    rescue = sum(str(row["utility_outcome"]) == "rescue" for row in rows)
    regress = sum(str(row["utility_outcome"]) == "regress" for row in rows)
    return {
        "rows": total,
        "selected_correct": correct,
        "selected_wrong": total - correct,
        "label_purity": correct / total if total else None,
        "label_purity_wilson95": wilson_interval(correct, total),
        "rescue": rescue,
        "regress": regress,
        "student_relative_net_utility": (rescue - regress) / total if total else None,
    }


def source_set(label: Mapping[str, Any], candidate_id: str) -> frozenset[str]:
    provenance = (label.get("candidate_provenance") or {}).get(candidate_id) or {}
    return frozenset(str(item.get("source")) for item in provenance.get("occurrences") or [] if item.get("source"))


def candidate_lookup(packet: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(candidate["candidate_id"]): dict(candidate) for candidate in packet.get("candidates") or []}


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.2f}%"


def pp(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:+.2f}pp"


def ci_text(value: Sequence[float] | None) -> str:
    return "NA" if value is None else f"[{pct(value[0])}, {pct(value[1])}]"


def render_report(summary: Mapping[str, Any]) -> str:
    variants = summary["training_variants"]
    source_rows = summary["qwen35_self_selection"]["strata"]
    lines = [
        "# Pass@8 Selector-to-Training Bridge Diagnostic",
        "",
        "Positive student-relative utility is not equivalent to clean SFT labels: on student-wrong rows, a wrong replacement has utility zero but remains an actively wrong training target.",
        "",
        "## Candidate training variants",
        "",
        "| variant | changed rows | correct labels | purity | Wilson 95% | rescue / regress | utility |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("all_9b_changes", "all_consensus_changes", "9b_consensus_same_action"):
        item = variants[name]
        lines.append(
            f"| {name} | {item['rows']} | {item['selected_correct']} | {pct(item['label_purity'])} | "
            f"{ci_text(item['label_purity_wilson95'])} | {item['rescue']} / {item['regress']} | "
            f"{pp(item['student_relative_net_utility'])} |"
        )
    lines.extend([
        "",
        "## Qwen3.5-9B self-selection",
        "",
        f"Among all changed 9B selections, {summary['qwen35_self_selection']['selected_with_qwen35_source']} / "
        f"{summary['qwen35_self_selection']['changed_rows']} exact selected actions contain a Qwen3.5 source occurrence "
        f"({pct(summary['qwen35_self_selection']['selected_with_qwen35_source_rate'])}).",
        "",
        "| selected exact-source stratum | rows | correct | purity |",
        "|---|---:|---:|---:|",
    ])
    for name in ("qwen35_only", "qwen35_mixed", "no_qwen35"):
        item = source_rows[name]
        lines.append(f"| {name} | {item['rows']} | {item['selected_correct']} | {pct(item['label_purity'])} |")
    lines.extend([
        "",
        "## Safety boundary",
        "",
        f"The locked population contains only {summary['scope']['student_correct_rows']} student-correct rows out of "
        f"{summary['scope']['rows']}. Its rescue/regress ratio therefore cannot establish arbitrary-state regression safety.",
        "",
        "## Decision",
        "",
        "Do not train directly from these selected changes. First measure a controlled 100/80/60/40% purity-response curve at fixed 25/75 revision-to-clean replay, then measure train-split aggregate purity for frozen GT-free construction variants. A training variant is eligible only if its conservative purity bound clears the empirically tolerated purity threshold.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind", required=True)
    parser.add_argument("--sealed", required=True)
    parser.add_argument("--current-eval", required=True)
    parser.add_argument("--consensus-eval", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--self-source", default="qwen35")
    args = parser.parse_args()

    blind = index_unique(read_jsonl(Path(args.blind)), args.blind)
    sealed = index_unique(read_jsonl(Path(args.sealed)), args.sealed)
    current = index_unique(read_jsonl(Path(args.current_eval)), args.current_eval)
    consensus = index_unique(read_jsonl(Path(args.consensus_eval)), args.consensus_eval)
    target_ids = set(blind)
    for name, rows in (("sealed", sealed), ("current", current), ("consensus", consensus)):
        if set(rows) != target_ids:
            raise ValueError(f"{name} target mismatch: expected={len(target_ids)} actual={len(rows)}")
    for target_id in target_ids:
        packet_hash = blind[target_id]["packet_sha256"]
        if sealed[target_id].get("packet_sha256") != packet_hash:
            raise ValueError(f"sealed packet hash mismatch: {target_id}")
        if current[target_id].get("packet_sha256") != packet_hash or consensus[target_id].get("packet_sha256") != packet_hash:
            raise ValueError(f"selector packet hash mismatch: {target_id}")

    current_changed = [row for row in current.values() if bool(row["changed_from_baseline"])]
    consensus_changed = [row for row in consensus.values() if bool(row["changed_from_baseline"])]
    intersection = [
        row for target_id, row in current.items()
        if bool(row["changed_from_baseline"])
        and bool(consensus[target_id]["changed_from_baseline"])
        and row["selected_action_key"] == consensus[target_id]["selected_action_key"]
    ]

    strata: dict[str, list[Mapping[str, Any]]] = {"qwen35_only": [], "qwen35_mixed": [], "no_qwen35": []}
    patterns: Counter[str] = Counter()
    candidate_total = candidate_with_self = 0
    selected_with_self = 0
    neighborhood_cross_source = 0
    for target_id, packet in blind.items():
        candidates = candidate_lookup(packet)
        label = sealed[target_id]
        for candidate_id in candidates:
            if candidate_id == "BASELINE":
                continue
            candidate_total += 1
            if args.self_source in source_set(label, candidate_id):
                candidate_with_self += 1
        row = current[target_id]
        if not bool(row["changed_from_baseline"]):
            continue
        candidate_id = str(row["selected_candidate_id"])
        sources = source_set(label, candidate_id)
        patterns["+".join(sorted(sources)) or "unknown"] += 1
        has_self = args.self_source in sources
        selected_with_self += int(has_self)
        if has_self and len(sources) == 1:
            strata["qwen35_only"].append(row)
        elif has_self:
            strata["qwen35_mixed"].append(row)
        else:
            strata["no_qwen35"].append(row)
        selected_candidate = candidates[candidate_id]
        neighborhood_cross_source += int(int(selected_candidate.get("source_count") or 0) >= 2)

    summary = {
        "scope": {
            "rows": len(target_ids),
            "episodes": len({str(row["episode_id"]) for row in current.values()}),
            "student_correct_rows": sum(bool(row["baseline_correct"]) for row in current.values()),
            "student_wrong_rows": sum(not bool(row["baseline_correct"]) for row in current.values()),
        },
        "training_variants": {
            "all_9b_changes": purity_summary(current_changed),
            "all_consensus_changes": purity_summary(consensus_changed),
            "9b_consensus_same_action": purity_summary(intersection),
        },
        "qwen35_self_selection": {
            "self_source": args.self_source,
            "changed_rows": len(current_changed),
            "selected_with_qwen35_source": selected_with_self,
            "selected_with_qwen35_source_rate": selected_with_self / max(1, len(current_changed)),
            "selected_cross_source_neighborhood": neighborhood_cross_source,
            "selected_cross_source_neighborhood_rate": neighborhood_cross_source / max(1, len(current_changed)),
            "candidate_actions": candidate_total,
            "candidate_actions_with_qwen35_source": candidate_with_self,
            "candidate_actions_with_qwen35_source_rate": candidate_with_self / max(1, candidate_total),
            "selection_enrichment_over_candidate_rate": (
                (selected_with_self / max(1, len(current_changed))) /
                (candidate_with_self / max(1, candidate_total))
                if candidate_with_self else None
            ),
            "strata": {name: purity_summary(rows) for name, rows in strata.items()},
            "exact_source_set_counts": dict(sorted(patterns.items(), key=lambda item: (-item[1], item[0]))),
        },
        "interpretation": {
            "utility_is_not_sft_purity": True,
            "reason": "Wrong selected actions on student-wrong states have zero utility but are actively incorrect SFT labels.",
            "oracle_revision_purity": 1.0,
            "unfiltered_revision_accuracy_reference": 0.2604,
        },
    }
    out_dir = Path(args.output_dir)
    write_json(out_dir / "summary.json", summary)
    (out_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(out_dir),
        "training_variants": summary["training_variants"],
        "qwen35_self_selection": summary["qwen35_self_selection"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()