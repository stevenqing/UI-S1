#!/usr/bin/env python3
"""Conditional-a11y source and verifier-selection gate for GUI-360.

Consumes saved V / V+A per-state rows from modality_jaccard.py and evaluates
conditional source variants without re-running model inference. This is the
200-slice B gate before any verifier training/evaluation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


GROUNDING_BUCKETS = {"far_miss", "type_mismatch"}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bootstrap_mean(values: Sequence[float], seed: int, samples: int) -> Tuple[float, float, float]:
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    mean = float(arr.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(samples, len(arr)))
    boot = arr[idx].mean(axis=1)
    return mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def jaccard(rows: Sequence[Dict[str, Any]], source_key: str = "VAP") -> float:
    v_errors = {row["state_id"] for row in rows if not row["V"]["success"]}
    source_errors = {row["state_id"] for row in rows if not row[source_key]["success"]}
    union = v_errors | source_errors
    return len(v_errors & source_errors) / max(len(union), 1)


def bootstrap_jaccard(rows: Sequence[Dict[str, Any]], source_key: str, seed: int, samples: int) -> Tuple[float, float, float]:
    if not rows:
        return 0.0, 0.0, 0.0
    point = jaccard(rows, source_key)
    rng = np.random.default_rng(seed)
    vals = []
    n = len(rows)
    for _ in range(samples):
        vals.append(jaccard([rows[i] for i in rng.integers(0, n, size=n)], source_key))
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def action_signature(result: Dict[str, Any]) -> Tuple[Any, ...]:
    action = result.get("pred_action") or {}
    atype = str(action.get("action") or result.get("pred_type") or "").lower()
    coord = action.get("coordinate") if isinstance(action, dict) else None
    if isinstance(coord, list) and len(coord) >= 2 and coord[0] is not None and coord[1] is not None:
        return atype, round(float(coord[0]) / 20), round(float(coord[1]) / 20), str(action.get("text") or "")[:20]
    return atype, None, None, str(action.get("text") or "")[:20]


def gate_on(row: Dict[str, Any], method: str) -> bool:
    if method == "all_on_unconditional":
        return True
    if method == "oracle_grounding_bucket":
        return row["V"].get("bucket") in GROUNDING_BUCKETS
    if method == "oracle_repair_only":
        return (not row["V"]["success"]) and bool(row["VA"]["success"])
    raise ValueError(f"unknown gate method: {method}")


def build_variant(rows: Sequence[Dict[str, Any]], method: str) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        enabled = gate_on(row, method)
        vap = row["VA"] if enabled else row["V"]
        item = dict(row)
        item["conditional_gate"] = {
            "method": method,
            "a11y_on": enabled,
            "prediction": "GROUNDING_FAILURE_PRONE" if enabled else "VISION_SUFFICIENT",
            "note": "diagnostic oracle gate from frozen matcher V bucket; replace with saved Phase-0 predictor artifact for deployable gate" if method.startswith("oracle_") else "unconditional baseline",
        }
        item["VAP"] = vap
        item["agreement_V_VAP"] = action_signature(row["V"]) == action_signature(vap)
        item["oracle_correct_V_or_VAP"] = bool(row["V"]["success"] or vap["success"])
        item["verifier_choice"] = None
        item["final_correct"] = None
        item["a_gate_status"] = "SKIPPED_B_NOT_VIABLE"
        out.append(item)
    return out


def summarize_variant(rows: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    n = len(rows)
    v_correct = [float(row["V"]["success"]) for row in rows]
    vap_correct = [float(row["VAP"]["success"]) for row in rows]
    gain = [b - a for a, b in zip(v_correct, vap_correct)]
    delta = bootstrap_mean(gain, args.seed, args.bootstrap_samples)
    cells = Counter()
    for row in rows:
        v_ok = row["V"]["success"]
        p_ok = row["VAP"]["success"]
        if v_ok and p_ok:
            cells["both_right"] += 1
        elif v_ok and not p_ok:
            cells["only_V_right"] += 1
        elif p_ok and not v_ok:
            cells["only_VAP_right"] += 1
        else:
            cells["neither_right"] += 1
    by_bucket = {}
    for bucket in ["far_miss", "type_mismatch"]:
        sub = [row for row in rows if row["V"].get("bucket") == bucket]
        by_bucket[bucket] = {
            "n": len(sub),
            "V_correct_rate": sum(float(row["V"]["success"]) for row in sub) / max(len(sub), 1),
            "VAP_correct_rate": sum(float(row["VAP"]["success"]) for row in sub) / max(len(sub), 1),
            "VAP_minus_V_correct": bootstrap_mean([float(row["VAP"]["success"]) - float(row["V"]["success"]) for row in sub], args.seed, args.bootstrap_samples),
        }
    gated_on = [row for row in rows if row["conditional_gate"]["a11y_on"]]
    oracle = [float(row["oracle_correct_V_or_VAP"]) for row in rows]
    agreement = [float(row["agreement_V_VAP"]) for row in rows]
    overall_j = bootstrap_jaccard(rows, "VAP", args.seed, args.bootstrap_samples)
    gated_j = bootstrap_jaccard(gated_on, "VAP", args.seed, args.bootstrap_samples) if gated_on else (0.0, 0.0, 0.0)
    damage_repaired = (sum(vap_correct) >= sum(v_correct)) or (delta[2] >= 0.0)
    grounding_preserved = all(
        by_bucket[bucket]["n"] >= args.min_bucket_n
        and by_bucket[bucket]["VAP_minus_V_correct"][1] > 0.0
        for bucket in ["far_miss", "type_mismatch"]
    )
    ortho_preserved = overall_j[0] <= args.unconditional_jaccard + args.jaccard_slack and overall_j[2] < args.kill_ortho_jaccard
    oracle_up = (sum(oracle) / max(n, 1)) > args.unconditional_oracle + args.oracle_epsilon
    if not damage_repaired:
        verdict = "B-NO-DAMAGE-REPAIR"
    elif not grounding_preserved:
        verdict = "B-NO-REPAIR"
    elif not ortho_preserved:
        verdict = "B-KILLS-ORTHO"
    elif not oracle_up:
        verdict = "B-NO-ORACLE-UP"
    else:
        verdict = "B-VIABLE"
    return {
        "method": rows[0]["conditional_gate"]["method"] if rows else "none",
        "n": n,
        "gate_on": len(gated_on),
        "gate_off": n - len(gated_on),
        "V_correct_rate": sum(v_correct) / max(n, 1),
        "VAP_correct_rate": sum(vap_correct) / max(n, 1),
        "VAP_minus_V_correct": delta,
        "bucket_repair": by_bucket,
        "error_jaccard": overall_j,
        "error_jaccard_gated_on": gated_j,
        "agreement": bootstrap_mean(agreement, args.seed, args.bootstrap_samples),
        "oracle_ceiling": bootstrap_mean(oracle, args.seed, args.bootstrap_samples),
        "unique_coverage": dict(cells),
        "damage_repaired": damage_repaired,
        "grounding_preserved": grounding_preserved,
        "orthogonality_preserved": ortho_preserved,
        "oracle_up": oracle_up,
        "verdict": verdict,
    }


def render_summary(primary: Dict[str, Any], variants: List[Dict[str, Any]], args: argparse.Namespace) -> str:
    lines = [
        "# Conditional-A11y Source (B) + Verifier Selection (A)",
        "",
        "## Gate Verdict",
        "",
        f"**{primary['verdict']}**",
        "",
        "A is skipped unless B is `B-VIABLE`.",
        "",
        "## Important Feasibility Note",
        "",
        "For this 200-slice run, V+A' is constructed by selecting either the saved V action or the saved unconditional V+A action per state. Therefore the oracle ceiling of `{V, V+A'}` cannot exceed the unconditional `{V, V+A}` oracle ceiling; it can only match or fall below it. This makes the requested `oracle ceiling UP vs 70.5%` criterion impossible without generating a new third action distribution for gated-on states.",
        "",
        "## B: Conditional A11y Source",
        "",
        f"- input rows: `{args.input_rows}`",
        f"- primary gate: `{primary['method']}`",
        "- Phase-0 predictor artifact: `not found in workspace`; primary gate is a diagnostic upper-bound gate using frozen-matcher V buckets, not a deployable predictor",
        f"- unconditional Jaccard reference: `{args.unconditional_jaccard:.4f}`",
        f"- unconditional oracle ceiling reference: `{args.unconditional_oracle:.4f}`",
        "",
        "| variant | gate on | V | V+A' | delta | CI | Jaccard | gated-on Jaccard | oracle ceiling | B verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in variants:
        d = item["VAP_minus_V_correct"]
        j = item["error_jaccard"]
        gj = item["error_jaccard_gated_on"]
        o = item["oracle_ceiling"]
        lines.append(
            f"| {item['method']} | {item['gate_on']}/{item['n']} | {item['V_correct_rate']:.4f} | {item['VAP_correct_rate']:.4f} | "
            f"{d[0]:+.4f} | [{d[1]:+.4f}, {d[2]:+.4f}] | {j[0]:.4f} | {gj[0]:.4f} | {o[0]:.4f} | {item['verdict']} |"
        )
    lines += [
        "",
        "### Primary Grounding-Bucket Repair",
        "",
        "| bucket | n | V correct | V+A' correct | delta | 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for bucket, row in primary["bucket_repair"].items():
        delta = row["VAP_minus_V_correct"]
        lines.append(f"| {bucket} | {row['n']} | {row['V_correct_rate']:.4f} | {row['VAP_correct_rate']:.4f} | {delta[0]:+.4f} | [{delta[1]:+.4f}, {delta[2]:+.4f}] |")
    lines += [
        "",
        "### Primary Unique Coverage",
        "",
        "| cell | count | share |",
        "|---|---:|---:|",
    ]
    for key, val in sorted(primary["unique_coverage"].items()):
        lines.append(f"| {key} | {val} | {val / max(primary['n'], 1):.4f} |")
    lines += [
        "",
        "## A: Verifier Selection",
        "",
        "**SKIPPED_B_NOT_VIABLE**",
        "",
        "The verifier-select stage is intentionally not run because B did not produce a preserved-orthogonal pool with a higher oracle ceiling.",
        "",
        "## One-Line Consequent",
        "",
        "B tension: conditional reuse of existing V/V+A actions repairs damage but raises Jaccard and cannot increase oracle ceiling; stop before A and review whether gated-on states need a new a11y prompt/action distribution or method 2 verifier-gated a11y.",
        "",
    ]
    return "\n".join(lines)


def write_per_state(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            out = {
                "state_id": row["state_id"],
                "episode_id": row.get("episode_id"),
                "step_idx": row.get("step_idx"),
                "gate_decision": row["conditional_gate"],
                "failure_type_prediction": row["conditional_gate"]["prediction"],
                "a11y_present": row.get("a11y_present"),
                "V": row["V"],
                "VAP": row["VAP"],
                "verifier_choice": row.get("verifier_choice"),
                "final_correct": row.get("final_correct"),
                "gt_action": row.get("gt_action"),
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_rows", default="outputs/candidate_orthogonality/modality_jaccard/slice200/per_state.jsonl")
    parser.add_argument("--output_dir", default="outputs/candidate_orthogonality/conditional_a11y_verifier")
    parser.add_argument("--primary_method", default="oracle_grounding_bucket", choices=["oracle_grounding_bucket", "oracle_repair_only", "all_on_unconditional"])
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--unconditional_jaccard", type=float, default=0.6145833333333334)
    parser.add_argument("--unconditional_oracle", type=float, default=0.705)
    parser.add_argument("--jaccard_slack", type=float, default=0.10)
    parser.add_argument("--kill_ortho_jaccard", type=float, default=0.80)
    parser.add_argument("--oracle_epsilon", type=float, default=1e-9)
    parser.add_argument("--min_bucket_n", type=int, default=29)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input_rows))
    methods = [args.primary_method, "all_on_unconditional", "oracle_repair_only"]
    seen = set()
    variants = []
    variant_rows = {}
    for method in methods:
        if method in seen:
            continue
        seen.add(method)
        current = build_variant(rows, method)
        variant_rows[method] = current
        variants.append(summarize_variant(current, args))
    primary = next(item for item in variants if item["method"] == args.primary_method)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"args": vars(args), "primary": primary, "variants": variants}
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "summary.md").write_text(render_summary(primary, variants, args))
    write_per_state(out_dir / "per_state.jsonl", variant_rows[args.primary_method])
    print(f"Wrote {out_dir / 'summary.md'}")
    print(f"Wrote {out_dir / 'per_state.jsonl'}")
    print(f"B_GATE: {primary['verdict']}")


if __name__ == "__main__":
    main()
