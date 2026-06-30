#!/usr/bin/env python3
"""Phase 1 coordinate-vs-symbol mechanism check for GUI-360.

Compares unconditional full-a11y V+A (with control_rect) against V+A_symbolic,
type-only, and label-only on an expanded test slice. Measurement only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


BUCKETS = ("far_miss", "type_mismatch")
SOURCES = ("VA", "symbolic", "type_only", "label_only")
SOURCE_LABELS = {
    "VA": "unconditional_full_a11y",
    "symbolic": "symbolic_type+label_no_coords",
    "type_only": "type_only_no_coords",
    "label_only": "label_only_no_coords",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def mean_ci(values: Sequence[float], seed: int, samples: int) -> Tuple[float, float, float]:
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    mean = float(arr.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(samples, len(arr)))
    boot = arr[idx].mean(axis=1)
    return mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def source_success(row: Dict[str, Any], source: str) -> bool:
    return bool(row[source]["success"])


def build_joined(modality_rows: Sequence[Dict[str, Any]], symbolic_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    symbolic_by_id = {row["state_id"]: row for row in symbolic_rows}
    rows = []
    for mod in modality_rows:
        sym = symbolic_by_id.get(mod["state_id"])
        if sym is None:
            continue
        row = {
            "state_id": mod["state_id"],
            "episode_id": mod.get("episode_id"),
            "step_idx": mod.get("step_idx"),
            "goal": mod.get("goal"),
            "gt_action": mod.get("gt_action"),
            "a11y_present": mod.get("a11y_present"),
            "num_controls": mod.get("num_controls"),
            "V_bucket": mod["V"].get("bucket"),
            "V": mod["V"],
            "VA": mod["VA"],
            "symbolic": sym["symbolic"],
            "type_only": sym["type_only"],
            "label_only": sym["label_only"],
            "symbolic_serialization": sym.get("symbolic_serialization"),
            "type_only_serialization": sym.get("type_only_serialization"),
            "label_only_serialization": sym.get("label_only_serialization"),
        }
        rows.append(row)
    return rows


def bucket_table(rows: Sequence[Dict[str, Any]], seed: int, samples: int) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for bucket in BUCKETS:
        sub = [row for row in rows if row["V_bucket"] == bucket]
        out[bucket] = {"n": len(sub), "support_ok": len(sub) >= 30, "sources": {}}
        for source in SOURCES:
            values = [float(source_success(row, source)) - float(source_success(row, "V")) for row in sub]
            out[bucket]["sources"][source] = {
                "correct_rate": sum(float(source_success(row, source)) for row in sub) / max(len(sub), 1),
                "gain_vs_V": mean_ci(values, seed, samples),
            }
    return out


def paired_diff(rows: Sequence[Dict[str, Any]], bucket: str, source_a: str, source_b: str, seed: int, samples: int) -> Dict[str, Any]:
    sub = [row for row in rows if row["V_bucket"] == bucket]
    values = [float(source_success(row, source_a)) - float(source_success(row, source_b)) for row in sub]
    return {"n": len(sub), "diff": mean_ci(values, seed, samples)}


def summarize(rows: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    buckets = bucket_table(rows, args.seed, args.bootstrap_samples)
    c1 = paired_diff(rows, "far_miss", "VA", "symbolic", args.seed, args.bootstrap_samples)
    c2 = paired_diff(rows, "far_miss", "symbolic", "type_only", args.seed, args.bootstrap_samples)
    support_ok = all(buckets[bucket]["support_ok"] for bucket in BUCKETS)
    c1_pass = support_ok and c1["diff"][1] > 0.0
    if not support_ok:
        c2_verdict = "PENDING_SUPPORT"
    elif c2["diff"][1] > 0.0:
        c2_verdict = "FAIL_LABEL_HELPS"
    elif abs(c2["diff"][0]) <= args.c2_abs_tolerance and c2["diff"][1] <= 0.0 <= c2["diff"][2]:
        c2_verdict = "CONFIRMED_NO_LABEL_EFFECT"
    else:
        c2_verdict = "INCONCLUSIVE"
    if not support_ok:
        verdict = "PENDING_SUPPORT"
        consequent = "expanded slice still lacks n>=30 in one or both grounding buckets; do not pivot"
    elif c1_pass and c2_verdict == "CONFIRMED_NO_LABEL_EFFECT":
        verdict = "COORD_MECHANISM_CONFIRMED"
        consequent = "C1+C2 hold; stop for review before Phase 2 conditional full-a11y"
    elif not c1_pass:
        verdict = "C1_FAIL"
        consequent = "coordinates did not significantly outperform symbolic on far-miss; do not assume coordinate mechanism"
    else:
        verdict = "C2_NOT_CONFIRMED"
        consequent = "label/no-label far-miss dissociation is not confirmed; revisit symbolic mechanism before Phase 2"
    return {
        "n": len(rows),
        "bucket_table": buckets,
        "c1_full_minus_symbolic_far_miss": c1,
        "c2_symbolic_minus_type_only_far_miss": c2,
        "support_ok": support_ok,
        "c1_pass": c1_pass,
        "c2_verdict": c2_verdict,
        "verdict": verdict,
        "consequent": consequent,
    }


def fmt_ci(values: Sequence[float]) -> str:
    return f"{values[0]:+.4f} [{values[1]:+.4f}, {values[2]:+.4f}]"


def render(summary: Dict[str, Any], args: argparse.Namespace) -> str:
    lines = [
        "# Phase 1: Coordinate vs Symbol Mechanism",
        "",
        "## Gate Verdict",
        "",
        f"**{summary['verdict']}**",
        "",
        summary["consequent"],
        "",
        "## Inputs",
        "",
        f"- modality rows: `{args.modality_rows}`",
        f"- symbolic rows: `{args.symbolic_rows}`",
        f"- joined states: `{summary['n']}`",
        "- split: `GUI-360 balanced test`",
        "- matcher: frozen GUI-360 step matcher / trichotomy buckets from V",
        "- real UIA: `uia_controls_info`; full-a11y includes `control_rect`; symbolic variants exclude coordinates",
        "",
        "## Expanded-Support Repair Table",
        "",
        "| bucket | n | support | source | source correct | repair vs V |",
        "|---|---:|---|---|---:|---:|",
    ]
    for bucket in BUCKETS:
        info = summary["bucket_table"][bucket]
        for source in SOURCES:
            row = info["sources"][source]
            lines.append(
                f"| {bucket} | {info['n']} | {info['support_ok']} | {SOURCE_LABELS[source]} | "
                f"{row['correct_rate']:.4f} | {fmt_ci(row['gain_vs_V'])} |"
            )
    c1 = summary["c1_full_minus_symbolic_far_miss"]
    c2 = summary["c2_symbolic_minus_type_only_far_miss"]
    lines += [
        "",
        "## Confirmations",
        "",
        "| check | comparison | n | paired difference | verdict |",
        "|---|---|---:|---:|---|",
        f"| C1 coords drive far-miss | unconditional full-a11y - symbolic on V far_miss | {c1['n']} | {fmt_ci(c1['diff'])} | {'PASS' if summary['c1_pass'] else 'FAIL/PENDING'} |",
        f"| C2 label does not drive far-miss | symbolic - type_only on V far_miss | {c2['n']} | {fmt_ci(c2['diff'])} | {summary['c2_verdict']} |",
        "",
        "## Phase Decision",
        "",
        f"- adequate support n>=30: `{summary['support_ok']}`",
        f"- C1 pass: `{summary['c1_pass']}`",
        f"- C2 verdict: `{summary['c2_verdict']}`",
        f"- final: `{summary['verdict']}`",
        "",
        "No Phase 2 conditional full-a11y and no Phase 3 verifier are run by this script. Stop for review at Phase 1.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality_rows", required=True)
    parser.add_argument("--symbolic_rows", required=True)
    parser.add_argument("--output_dir", default="outputs/candidate_orthogonality/cond_full_a11y")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--c2_abs_tolerance", type=float, default=0.03)
    args = parser.parse_args()

    modality = read_jsonl(Path(args.modality_rows))
    symbolic = read_jsonl(Path(args.symbolic_rows))
    rows = build_joined(modality, symbolic)
    summary = summarize(rows, args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase1_mechanism.json").write_text(json.dumps({"summary": summary, "args": vars(args)}, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "phase1_mechanism.md").write_text(render(summary, args))
    write_jsonl(out_dir / "phase1_per_state.jsonl", rows)
    print(f"Wrote {out_dir / 'phase1_mechanism.md'}")
    print(f"Wrote {out_dir / 'phase1_per_state.jsonl'}")
    print(f"PHASE1: {summary['verdict']} - {summary['consequent']}")


if __name__ == "__main__":
    main()
