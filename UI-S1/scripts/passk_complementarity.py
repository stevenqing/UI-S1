#!/usr/bin/env python3
"""pass@k complementarity vs matched-budget single-SFT on critical steps."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def pct(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.2f}%"


def pp(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:+.2f}pp"


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def load_targets(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        values = data.get("target_ids") or []
    else:
        values = data
    return [str(value) for value in values]


def target_id(row: Mapping[str, Any]) -> str:
    return str(row.get("target_id") or f"{row['episode_id']}:{row['step_idx']}")


def sample_key(sample: Mapping[str, Any]) -> str:
    key = sample.get("action_key")
    if key is not None:
        return str(key)
    action = sample.get("pred_action")
    if isinstance(action, Mapping):
        return json.dumps(action, sort_keys=True, ensure_ascii=False)
    return "__unparsed__"


def normalize_rows(path: Path, model: str, tier: str, targets: set[str]) -> dict[str, dict[str, Any]]:
    by_target: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        tid = target_id(row)
        if tid not in targets or tid in by_target:
            continue
        samples = list(row.get("samples") or [])
        # Some cached rows store greedy separately; include it only if no sample list exists.
        if not samples and row.get("greedy_pred_action") is not None:
            samples = [{
                "correct": bool(row.get("greedy_correct")),
                "parse_ok": row.get("greedy_pred_action") is not None,
                "pred_action": row.get("greedy_pred_action"),
                "action_key": row.get("greedy_key"),
                "reward": row.get("greedy_reward"),
            }]
        by_target[tid] = {
            "target_id": tid,
            "episode_id": str(row.get("episode_id")),
            "step_idx": int(row.get("step_idx") or 0),
            "model": model,
            "tier": tier,
            "samples": samples,
            "sample_count": len(samples),
        }
    return by_target


def support_metrics(samples: list[Mapping[str, Any]], k: int, min_share: float, min_count: int) -> dict[str, Any]:
    group = list(samples[:k])
    parse_count = sum(1 for sample in group if sample.get("parse_ok", sample.get("pred_action") is not None))
    correct_samples = [sample for sample in group if bool(sample.get("correct"))]
    correct_key_counts = Counter(sample_key(sample) for sample in correct_samples)
    best_count = max(correct_key_counts.values(), default=0)
    quality_threshold = max(min_count, math.ceil(min_share * max(1, len(group))))
    pass_any = bool(correct_samples)
    high_quality = best_count >= quality_threshold
    return {
        "available_samples": len(group),
        "parse_count": parse_count,
        "parse_rate": parse_count / max(1, len(group)),
        "correct_count": len(correct_samples),
        "best_correct_action_count": best_count,
        "quality_threshold": quality_threshold,
        "pass_any": pass_any,
        "high_quality": high_quality,
        "noise_correct": pass_any and not high_quality,
        "best_correct_action_key": max(correct_key_counts, key=correct_key_counts.get, default=None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default="outputs/multiagent_complementarity/target_ids.json")
    parser.add_argument("--output-dir", default="outputs/passk_complementarity")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--quality-share", type=float, default=0.25)
    parser.add_argument("--quality-min-count", type=int, default=2)
    parser.add_argument("--sft-matched", default="outputs/temp_restores_signal/sft_T1p5_critical.jsonl")
    parser.add_argument("--models", nargs="+", required=True, help="name:tier:path")
    parser.add_argument("--lift-threshold-low", type=float, default=0.05)
    parser.add_argument("--lift-threshold-high", type=float, default=0.10)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_list = load_targets(Path(args.targets))
    targets = set(target_list)

    model_rows: dict[str, dict[str, dict[str, Any]]] = {}
    model_tiers: dict[str, str] = {}
    for spec in args.models:
        name, tier, path_text = spec.split(":", 2)
        model_tiers[name] = tier
        model_rows[name] = normalize_rows(Path(path_text), name, tier, targets)

    complete_models = [name for name in model_rows if len(model_rows[name]) == len(target_list)]
    incomplete_models = {name: {"rows": len(rows), "expected": len(target_list)} for name, rows in model_rows.items() if len(rows) != len(target_list)}
    active_models = complete_models
    matched_budget = len(active_models) * args.k
    sft_rows = normalize_rows(Path(args.sft_matched), "single_sft_matched", "single_policy", targets)
    sft_budget_available = min((len(row["samples"]) for row in sft_rows.values()), default=0)
    if sft_budget_available < matched_budget:
        raise SystemExit(f"single-SFT matched budget unavailable: need {matched_budget}, have {sft_budget_available}")

    per_step: list[dict[str, Any]] = []
    per_model_counts = {name: Counter() for name in active_models}
    per_model_parse_sum = {name: 0.0 for name in active_models}
    contrib_beyond_sft = {name: [] for name in active_models if not name.startswith("sft")}
    union_any = union_hq = sft_hq_count = sft_any_count = 0

    for tid in target_list:
        sft_payload = support_metrics(sft_rows[tid]["samples"], matched_budget, args.quality_share, args.quality_min_count)
        sft_hq = bool(sft_payload["high_quality"])
        sft_any = bool(sft_payload["pass_any"])
        sft_hq_count += int(sft_hq)
        sft_any_count += int(sft_any)
        model_payloads = {}
        step_union_any = False
        step_union_hq = False
        for name in active_models:
            payload = support_metrics(model_rows[name][tid]["samples"], args.k, args.quality_share, args.quality_min_count)
            model_payloads[name] = payload
            per_model_counts[name]["pass_any"] += int(payload["pass_any"])
            per_model_counts[name]["high_quality"] += int(payload["high_quality"])
            per_model_counts[name]["noise_correct"] += int(payload["noise_correct"])
            per_model_counts[name]["correct_count"] += int(payload["correct_count"])
            per_model_counts[name]["steps"] += 1
            per_model_parse_sum[name] += float(payload["parse_rate"])
            step_union_any = step_union_any or payload["pass_any"]
            step_union_hq = step_union_hq or payload["high_quality"]
            if name in contrib_beyond_sft and payload["high_quality"] and not sft_hq:
                contrib_beyond_sft[name].append(tid)
        union_any += int(step_union_any)
        union_hq += int(step_union_hq)
        per_step.append({
            "target_id": tid,
            "matched_budget": matched_budget,
            "sft_matched": sft_payload,
            "models": model_payloads,
            "union_any_correct": step_union_any,
            "union_high_quality": step_union_hq,
            "adds_high_quality_beyond_sft": [name for name, payload in model_payloads.items() if payload["high_quality"] and not sft_hq],
        })

    n = len(target_list)
    model_summary = {}
    for name in active_models:
        counts = per_model_counts[name]
        model_summary[name] = {
            "tier": model_tiers[name],
            "steps": counts["steps"],
            "parse_rate": per_model_parse_sum[name] / max(1, counts["steps"]),
            "pass_any_coverage": counts["pass_any"] / max(1, n),
            "high_quality_coverage": counts["high_quality"] / max(1, n),
            "noise_correct_coverage": counts["noise_correct"] / max(1, n),
            "mean_correct_count": counts["correct_count"] / max(1, n),
            "unique_hq_beyond_sft_count": len(contrib_beyond_sft.get(name, [])),
            "unique_hq_beyond_sft_coverage": len(contrib_beyond_sft.get(name, [])) / max(1, n),
            "noise_only_flag": counts["high_quality"] == 0 and counts["pass_any"] > 0,
        }

    union_hq_cov = union_hq / max(1, n)
    sft_hq_cov = sft_hq_count / max(1, n)
    margin = union_hq_cov - sft_hq_cov
    if margin >= args.lift_threshold_high:
        gate = "MULTI-AGENT HAS SPACE (pursue it)"
        reason = "High-quality union beats matched-budget single-SFT by at least the high threshold."
    elif margin >= args.lift_threshold_low and sum(len(v) for v in contrib_beyond_sft.values()) > 0:
        gate = "MIXED"
        reason = "High-quality union beats matched-budget single-SFT by a modest margin, with some model-specific additions."
    else:
        gate = "NO SPACE (abandon multi-agent on GUI-360)"
        reason = "High-quality union does not beat matched-budget single-SFT by the pre-agreed 5-10pp margin, or added coverage is negligible."

    summary = {
        "gate": gate,
        "reason": reason,
        "targets": n,
        "k": args.k,
        "models": active_models,
        "incomplete_models": incomplete_models,
        "matched_budget": matched_budget,
        "quality_share": args.quality_share,
        "quality_min_count": args.quality_min_count,
        "single_sft_matched": {
            "source": args.sft_matched,
            "budget": matched_budget,
            "pass_any_coverage": sft_any_count / max(1, n),
            "high_quality_coverage": sft_hq_cov,
        },
        "union": {
            "pass_any_coverage": union_any / max(1, n),
            "high_quality_coverage": union_hq_cov,
            "margin_high_quality_vs_sft": margin,
        },
        "per_model": model_summary,
        "contrib_beyond_sft": {name: ids for name, ids in contrib_beyond_sft.items()},
    }
    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "per_step.jsonl", per_step)

    lines = [
        "# pass@k Long-Tail Complementarity",
        "",
        "Frozen matcher; sampling only; no training. High-quality means the same correct action key reaches the configured probability-mass threshold, not merely an isolated lucky hit.",
        "",
        "## Setup",
        "",
        table(["field", "value"], [
            ["targets", n],
            ["k per model", args.k],
            ["active models", ", ".join(active_models)],
            ["matched single-SFT budget", matched_budget],
            ["high-quality threshold", f"max({args.quality_min_count}, ceil({args.quality_share} * samples))"],
            ["incomplete models", json.dumps(incomplete_models, ensure_ascii=False)],
        ]),
        "",
        "## Metric 1 - Per-Model pass@k",
        "",
        table(["model", "tier", "parse", "pass@k any", "high-quality", "noise", "mean correct", "HQ beyond SFT", "noise-only"], [
            [name, item["tier"], pct(item["parse_rate"]), pct(item["pass_any_coverage"]), pct(item["high_quality_coverage"]), pct(item["noise_correct_coverage"]), f"{item['mean_correct_count']:.2f}", f"{item['unique_hq_beyond_sft_count']} ({pct(item['unique_hq_beyond_sft_coverage'])})", item["noise_only_flag"]]
            for name, item in model_summary.items()
        ]),
        "",
        "## Metric 2 - Union pass@k",
        "",
        table(["metric", "coverage"], [
            ["union any-correct", pct(union_any / max(1, n))],
            ["union high-quality", pct(union_hq_cov)],
            ["single-SFT matched any-correct", pct(sft_any_count / max(1, n))],
            ["single-SFT matched high-quality", pct(sft_hq_cov)],
        ]),
        "",
        "## Metric 3 - Decision Metric",
        "",
        f"High-quality union minus matched-budget single-SFT: `{pp(margin)}`.",
        "",
        "## Metric 4 - Per-Model Complementary HQ Steps Beyond SFT",
        "",
        table(["model", "unique HQ beyond SFT"], [[name, len(ids)] for name, ids in contrib_beyond_sft.items()]),
        "",
        "## Gate",
        "",
        gate,
        "",
        reason,
        "",
        "STOP for review.",
        "",
    ]
    (out_dir / "passk.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "gate": gate, "margin": margin}, indent=2), flush=True)


if __name__ == "__main__":
    main()