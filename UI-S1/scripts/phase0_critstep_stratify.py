#!/usr/bin/env python3
"""Critical-step stratification for re-examination Phase 0 outputs.

Uses a held-out p_i estimator derived from the baseline compound proof artifacts
and applies it to TRAIN episodes. Then tags existing Phase 0 per-state rows with
bottom-1/bottom-2 critical labels and counts real-a11y teacher positives.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prove_gui360_compound_structure import _bbox_features, _k_bin, _label_detail, _step_phase


LEVELS = ("fine", "mid", "coarse", "action_bbox", "action", "global")
PRIMARY_BUCKETS = ("far_miss", "type_mismatch")


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def feature_key(feature: Dict[str, Any], level: str) -> Tuple[Any, ...]:
    if level == "fine":
        return (
            feature["action_type"],
            feature["bbox_area_bin"],
            feature["bbox_aspect_bin"],
            feature["position_bin"],
            feature["step_phase"],
            feature["k_bin"],
            feature["label_detail"],
        )
    if level == "mid":
        return (
            feature["action_type"],
            feature["bbox_area_bin"],
            feature["position_bin"],
            feature["step_phase"],
            feature["label_detail"],
        )
    if level == "coarse":
        return (feature["action_type"], feature["bbox_area_bin"], feature["step_phase"], feature["label_detail"])
    if level == "action_bbox":
        return (feature["action_type"], feature["bbox_area_bin"], feature["label_detail"])
    if level == "action":
        return (feature["action_type"], feature["label_detail"])
    return ("global",)


def step_feature(step: Dict[str, Any], step_idx: int, k: int) -> Dict[str, Any]:
    action = step.get("action") if isinstance(step.get("action"), dict) else {}
    action_type = str(action.get("action") or "unknown").strip().lower() or "unknown"
    area_frac, area_bin, aspect_bin, position_bin, has_bbox = _bbox_features(step)
    return {
        "action_type": action_type,
        "bbox_area_frac": area_frac,
        "bbox_area_bin": area_bin,
        "bbox_aspect_bin": aspect_bin,
        "position_bin": position_bin,
        "step_phase": _step_phase(step_idx, k),
        "k_bin": _k_bin(k),
        "label_detail": _label_detail(action_type, action),
        "has_bbox": has_bbox,
    }


def build_heldout_p_estimator(compound_per_task: Path, min_bucket: int) -> Dict[str, Any]:
    stats: Dict[str, Dict[Tuple[Any, ...], List[float]]] = {level: defaultdict(list) for level in LEVELS}
    for task in load_jsonl(compound_per_task):
        ps = [float(value) for value in task["per_step_p_heldout_cv"]]
        features = task.get("step_features") or []
        for idx, p in enumerate(ps):
            feature = features[idx]
            for level in LEVELS:
                stats[level][feature_key(feature, level)].append(p)
    means = {level: {key: mean(values) for key, values in table.items()} for level, table in stats.items()}
    counts = {level: {key: len(values) for key, values in table.items()} for level, table in stats.items()}
    return {"means": means, "counts": counts, "min_bucket": min_bucket}


def estimate_p(feature: Dict[str, Any], estimator: Dict[str, Any]) -> Tuple[float, str, int]:
    for level in LEVELS:
        key = feature_key(feature, level)
        count = estimator["counts"].get(level, {}).get(key, 0)
        if count >= estimator["min_bucket"] or level == "global":
            return float(estimator["means"][level][key]), level, count
    raise RuntimeError("unreachable estimator fallback")


def read_train_episodes(data_dir: Path) -> Dict[str, Dict[str, Any]]:
    episodes: Dict[str, Dict[str, Any]] = {}
    for parquet_path in sorted(data_dir.glob("train-*.parquet")):
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(batch_size=64, columns=["episode_id", "goal", "steps"]):
            for row in batch.to_pylist():
                episode_id = str(row["episode_id"])
                steps = json.loads(row["steps"])
                episodes[episode_id] = {"episode_id": episode_id, "goal": row.get("goal", ""), "steps": steps}
    return episodes


def bottom_indices(values: Sequence[float], n: int) -> List[int]:
    keep = min(n, len(values))
    return [idx for idx, _ in sorted(enumerate(values), key=lambda item: (item[1], item[0]))[:keep]]


def build_train_critical_map(episodes: Dict[str, Dict[str, Any]], estimator: Dict[str, Any]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for episode_id, episode in episodes.items():
        steps = episode["steps"]
        k = len(steps)
        step_infos = []
        for step_pos, step in enumerate(steps):
            step_idx = int(step.get("step_idx", step_pos))
            feature = step_feature(step, step_pos, k)
            p_hat, source, source_count = estimate_p(feature, estimator)
            step_infos.append({"step_idx": step_idx, "step_pos": step_pos, "p_hat": p_hat, "p_source": source, "p_source_count": source_count, "feature": feature})
        ps = [item["p_hat"] for item in step_infos]
        bottom1 = set(bottom_indices(ps, 1))
        bottom2 = set(bottom_indices(ps, 2))
        for pos, item in enumerate(step_infos):
            payload = dict(item)
            payload["episode_id"] = episode_id
            payload["k"] = k
            payload["bottom1"] = pos in bottom1
            payload["bottom2"] = pos in bottom2
            payload["critical_rank"] = sorted(range(len(ps)), key=lambda i: (ps[i], i)).index(pos) + 1
            out[(episode_id, item["step_idx"])] = payload
    return out


def action_type_from_gt(row: Dict[str, Any]) -> str:
    action = row.get("gt_action") if isinstance(row.get("gt_action"), dict) else {}
    return str(action.get("action") or row.get("V", {}).get("gt_type") or "unknown").lower()


def critical_status(row: Dict[str, Any]) -> str:
    if row.get("bottom1"):
        return "bottom1"
    if row.get("bottom2"):
        return "bottom2_only"
    return "noncritical"


def count_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    denom = Counter()
    positives = [row for row in rows if row.get("distill_positive_real_a11y")]
    v_correct = [row for row in rows if row.get("V", {}).get("success")]
    teacher_broke = [row for row in rows if row.get("V", {}).get("success") and not row.get("real_a11y", {}).get("success")]
    for row in rows:
        status = critical_status(row)
        denom[status] += 1
        if row.get("bottom1"):
            denom["bottom1_any"] += 1
        if row.get("bottom2"):
            denom["bottom2_any"] += 1
        denom["total"] += 1
    def status_counts(selected: Sequence[Dict[str, Any]]) -> Dict[str, int]:
        counts = Counter()
        for row in selected:
            counts[critical_status(row)] += 1
            if row.get("bottom1"):
                counts["bottom1_any"] += 1
            if row.get("bottom2"):
                counts["bottom2_any"] += 1
            counts["total"] += 1
        return dict(counts)
    pos_by_bucket = Counter(row.get("V_bucket") for row in positives)
    pos_by_action = Counter(action_type_from_gt(row) for row in positives)
    pos_by_status_bucket = Counter((critical_status(row), row.get("V_bucket")) for row in positives)
    pos_by_status_action = Counter((critical_status(row), action_type_from_gt(row)) for row in positives)
    bottom1_pos = [row for row in positives if row.get("bottom1")]
    bottom2_pos = [row for row in positives if row.get("bottom2")]
    bottom2_click_far = [row for row in bottom2_pos if action_type_from_gt(row) == "click" and row.get("V_bucket") == "far_miss"]
    bottom1_click_far = [row for row in bottom1_pos if action_type_from_gt(row) == "click" and row.get("V_bucket") == "far_miss"]
    return {
        "denominators": dict(denom),
        "positives_total": len(positives),
        "positives_primary_total": sum(pos_by_bucket[bucket] for bucket in PRIMARY_BUCKETS),
        "positives_by_v_bucket": dict(pos_by_bucket),
        "positives_by_action_type": dict(pos_by_action),
        "positives_by_status": status_counts(positives),
        "positives_by_status_bucket": {f"{status}|{bucket}": count for (status, bucket), count in pos_by_status_bucket.items()},
        "positives_by_status_action": {f"{status}|{action}": count for (status, action), count in pos_by_status_action.items()},
        "bottom1_positive_count": len(bottom1_pos),
        "bottom2_positive_count": len(bottom2_pos),
        "bottom1_click_far_miss_count": len(bottom1_click_far),
        "bottom2_click_far_miss_count": len(bottom2_click_far),
        "bottom1_click_far_miss_share": len(bottom1_click_far) / len(bottom1_pos) if bottom1_pos else 0.0,
        "bottom2_click_far_miss_share": len(bottom2_click_far) / len(bottom2_pos) if bottom2_pos else 0.0,
        "v_correct_total": len(v_correct),
        "v_correct_by_status": status_counts(v_correct),
        "teacher_broke_total": len(teacher_broke),
        "teacher_broke_by_status": status_counts(teacher_broke),
        "teacher_correct_total": sum(1 for row in rows if row.get("real_a11y", {}).get("success")),
    }


def verdict(counts: Dict[str, Any]) -> Tuple[str, str]:
    total = counts["positives_total"]
    bottom2 = counts["bottom2_positive_count"]
    if total < 150:
        return "DATA-STARVED", "total real-a11y positives are below 150 in the evaluated train slice"
    if bottom2 < 50:
        return "CRITICAL-STARVED", "overall positives may exist, but bottom-2 critical positives are below 50"
    if total < 250 or bottom2 < 80:
        return "MARGINAL", "positive pool or critical-positive pool is underpowered for P4"
    return "VIABLE", "overall and critical-step positive pools meet the pre-registered thresholds"


def render_report(counts: Dict[str, Any], phase0_summary: Dict[str, Any], args: argparse.Namespace, full_train_denoms: Dict[str, int], verdict_name: str, verdict_reason: str) -> str:
    lines = []
    lines.append("# Re-examination Phase 0 Critical-Step Stratification")
    lines.append("")
    lines.append("Date: 2026-06-30")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("Regime: TRAIN split, original SFT baseline `checkpoints/gui360-fullparam-sft-step250`, V no-a11y prompt vs real full-a11y teacher prompt, frozen matcher. No training performed.")
    lines.append("")
    lines.append("Critical-step definition: bottom-1 / bottom-2 by held-out p_i within each TRAIN task. p_i is estimated from the baseline compound proof feature buckets, not from the state outcome being counted.")
    lines.append("")
    lines.append("Important coverage note: this report stratifies the existing Phase 0 run, which evaluated a seeded 500-state TRAIN slice (`limit=500`), not the full TRAIN split. Full TRAIN denominators are shown for context; the gate below is for the evaluated Phase 0 slice unless a full run is executed.")
    lines.append("")
    lines.append("## Denominators")
    lines.append("")
    lines.append("| denominator | count |")
    lines.append("|---|---:|")
    lines.append(f"| full train episodes | {full_train_denoms['episodes']} |")
    lines.append(f"| full train steps | {full_train_denoms['steps']} |")
    lines.append(f"| evaluated Phase 0 states | {counts['denominators'].get('total', 0)} |")
    lines.append(f"| evaluated bottom-1 states | {counts['denominators'].get('bottom1_any', 0)} |")
    lines.append(f"| evaluated bottom-2 states | {counts['denominators'].get('bottom2_any', 0)} |")
    lines.append(f"| evaluated non-bottom2 states | {counts['denominators'].get('noncritical', 0)} |")
    lines.append("")
    lines.append("## Distillation Positives: V Wrong, Real-A11y Teacher Right")
    lines.append("")
    lines.append("| metric | count |")
    lines.append("|---|---:|")
    lines.append(f"| positives total | {counts['positives_total']} |")
    lines.append(f"| positives primary far/type | {counts['positives_primary_total']} |")
    lines.append(f"| positives bottom-1 | {counts['bottom1_positive_count']} |")
    lines.append(f"| positives bottom-2 | {counts['bottom2_positive_count']} |")
    lines.append(f"| positives non-critical | {counts['positives_by_status'].get('noncritical', 0)} |")
    lines.append(f"| bottom-1 click far_miss positives | {counts['bottom1_click_far_miss_count']} |")
    lines.append(f"| bottom-2 click far_miss positives | {counts['bottom2_click_far_miss_count']} |")
    lines.append(f"| bottom-1 click far_miss share | {counts['bottom1_click_far_miss_share']:.4f} |")
    lines.append(f"| bottom-2 click far_miss share | {counts['bottom2_click_far_miss_share']:.4f} |")
    lines.append("")
    lines.append("### Positives by V Bucket")
    lines.append("")
    lines.append("| V bucket | count |")
    lines.append("|---|---:|")
    for bucket, count in sorted(counts["positives_by_v_bucket"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {bucket} | {count} |")
    lines.append("")
    lines.append("### Positives by Critical Status and Bucket")
    lines.append("")
    lines.append("| status | bucket | count |")
    lines.append("|---|---|---:|")
    for key, count in sorted(counts["positives_by_status_bucket"].items()):
        status, bucket = key.split("|", 1)
        lines.append(f"| {status} | {bucket} | {count} |")
    lines.append("")
    lines.append("### Positives by Action Type")
    lines.append("")
    lines.append("| action type | count |")
    lines.append("|---|---:|")
    for action, count in sorted(counts["positives_by_action_type"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {action} | {count} |")
    lines.append("")
    lines.append("## Preservation And Teacher-Broke Health Check")
    lines.append("")
    lines.append("| metric | total | bottom-1 | bottom-2 | non-critical |")
    lines.append("|---|---:|---:|---:|---:|")
    vc = counts["v_correct_by_status"]
    tb = counts["teacher_broke_by_status"]
    lines.append(f"| V-correct preservation pool | {counts['v_correct_total']} | {vc.get('bottom1_any', 0)} | {vc.get('bottom2_any', 0)} | {vc.get('noncritical', 0)} |")
    lines.append(f"| teacher-broke V-correct | {counts['teacher_broke_total']} | {tb.get('bottom1_any', 0)} | {tb.get('bottom2_any', 0)} | {tb.get('noncritical', 0)} |")
    lines.append("")
    lines.append("## Phase 0 Original Summary Context")
    lines.append("")
    lines.append(f"- V-correct rate on evaluated slice: `{phase0_summary['v_correct_rate']:.4f}`")
    lines.append(f"- real-a11y teacher correct total: `{counts['teacher_correct_total']}`")
    lines.append(f"- original Phase 0 verdict: `{phase0_summary['verdict']}`")
    lines.append("")
    lines.append("## Data Viability Verdict")
    lines.append("")
    lines.append(f"**{verdict_name}**")
    lines.append("")
    lines.append(verdict_reason)
    lines.append("")
    if args.phase0_per_state.endswith("phase0_limit500_retry/phase0_per_state.jsonl"):
        lines.append("Because this is the existing 500-state Phase 0 slice, this verdict should be read as a slice-level data-viability result. A full TRAIN run would be needed to make the absolute 150/80 thresholds definitive for the entire train split.")
        lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `outputs/reexam_distill/phase0_critstep.md`")
    lines.append("- `outputs/reexam_distill/phase0_critstep.json`")
    lines.append("- `outputs/reexam_distill/phase0_critstep_per_state.jsonl`")
    return "\n".join(lines) + "\n"


def full_train_denominators(episodes: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    return {"episodes": len(episodes), "steps": sum(len(ep["steps"]) for ep in episodes.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Critical-step stratification for Phase 0 re-examination outputs")
    parser.add_argument("--phase0-per-state", default="outputs/reexam_distill/phase0_limit500_retry/phase0_per_state.jsonl")
    parser.add_argument("--phase0-summary", default="outputs/reexam_distill/phase0_limit500_retry/phase0_data.json")
    parser.add_argument("--compound-per-task", default="outputs/compound_proof/per_task.jsonl")
    parser.add_argument("--balanced-data-dir", default="datasets/gui360-balanced/data")
    parser.add_argument("--output-dir", default="outputs/reexam_distill")
    parser.add_argument("--min-bucket", type=int, default=25)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.phase0_per_state))
    phase0 = json.loads(Path(args.phase0_summary).read_text(encoding="utf-8"))["summary"]
    estimator = build_heldout_p_estimator(Path(args.compound_per_task), args.min_bucket)
    episodes = read_train_episodes(Path(args.balanced_data_dir))
    critical_map = build_train_critical_map(episodes, estimator)

    enriched = []
    missing = []
    for row in rows:
        key = (str(row["episode_id"]), int(row["step_idx"]))
        info = critical_map.get(key)
        if info is None:
            missing.append(row["state_id"])
            continue
        new_row = dict(row)
        new_row.update({
            "p_hat_heldout_bucket": info["p_hat"],
            "p_hat_source": info["p_source"],
            "p_hat_source_count": info["p_source_count"],
            "critical_rank": info["critical_rank"],
            "bottom1": info["bottom1"],
            "bottom2": info["bottom2"],
            "critical_feature": info["feature"],
            "task_k": info["k"],
        })
        enriched.append(new_row)
    counts = count_rows(enriched)
    counts["missing_state_ids"] = missing[:20]
    counts["missing_count"] = len(missing)
    verdict_name, verdict_reason = verdict(counts)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl = out_dir / "phase0_critstep_per_state.jsonl"
    with write_jsonl.open("w", encoding="utf-8") as handle:
        for row in enriched:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {
        "phase0_source": args.phase0_per_state,
        "critical_source": args.compound_per_task,
        "coverage": {"enriched": len(enriched), "missing": len(missing)},
        "full_train_denominators": full_train_denominators(episodes),
        "counts": counts,
        "verdict": verdict_name,
        "verdict_reason": verdict_reason,
    }
    (out_dir / "phase0_critstep.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "phase0_critstep.md").write_text(render_report(counts, phase0, args, payload["full_train_denominators"], verdict_name, verdict_reason), encoding="utf-8")
    print(json.dumps({"output": str(out_dir / "phase0_critstep.md"), "verdict": verdict_name, "positives_total": counts["positives_total"], "bottom2_positive_count": counts["bottom2_positive_count"], "bottom2_click_far_miss_count": counts["bottom2_click_far_miss_count"], "missing": len(missing)}, indent=2))


if __name__ == "__main__":
    main()