#!/usr/bin/env python3
"""Report paired held-out results for equal-budget revision LoRA screens."""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def flatten(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    output = {}
    for episode in rows:
        episode_id = str(episode["episode_id"])
        for fallback_idx, step in enumerate(episode.get("steps", [])):
            step_idx = int(step.get("step_idx", fallback_idx))
            target_id = str(step.get("target_id") or f"{episode_id}:{step_idx}")
            if target_id in output:
                raise ValueError(f"duplicate target_id: {target_id}")
            output[target_id] = step
    return output


def bootstrap_delta(base: list[int], post: list[int], draws: int, seed: int) -> dict[str, float]:
    if len(base) != len(post) or not base:
        raise ValueError("paired bootstrap requires equal vectors")
    rng = random.Random(seed)
    n = len(base)
    values = []
    for _ in range(draws):
        values.append(sum(post[idx := rng.randrange(n)] - base[idx] for _ in range(n)) / n)
    values.sort()
    return {
        "mean": sum(p - b for p, b in zip(post, base)) / n,
        "lo": values[int(0.025 * draws)],
        "hi": values[min(draws - 1, int(0.975 * draws))],
        "draws": draws,
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    return f"{100.0 * value:+.2f}pp"


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--test-episodes", required=True)
    parser.add_argument("--post-glob", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--training-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    episodes = read_jsonl(Path(args.test_episodes))
    selected_ids = [
        str(row["episode_id"]) for idx, row in enumerate(episodes)
        if idx % args.shard_count == args.shard_index
    ]
    baseline_all = {str(row["episode_id"]): row for row in read_jsonl(Path(args.baseline))}
    if any(episode_id not in baseline_all for episode_id in selected_ids):
        raise ValueError("baseline missing selected episode")
    baseline = [baseline_all[episode_id] for episode_id in selected_ids]
    base_by_ep = {str(row["episode_id"]): row for row in baseline}
    base_steps = flatten(baseline)
    base_task = [int(bool(base_by_ep[episode_id]["task_success"])) for episode_id in selected_ids]
    base_tsr = sum(base_task) / len(base_task)
    base_step_accuracy = sum(bool(step["success"]) for step in base_steps.values()) / len(base_steps)

    data_manifest = read_json(Path(args.data_manifest))
    arms = []
    for post_name in sorted(glob.glob(args.post_glob)):
        post_path = Path(post_name)
        post_rows = read_jsonl(post_path)
        post_all = {str(row["episode_id"]): row for row in post_rows}
        missing_post = set(selected_ids) - set(post_all)
        if missing_post:
            raise ValueError(f"post episode grid missing {list(missing_post)[:10]}: {post_path}")
        post_by_ep = {episode_id: post_all[episode_id] for episode_id in selected_ids}
        post_rows = [post_by_ep[episode_id] for episode_id in selected_ids]
        post_steps = flatten(post_rows)
        if set(post_steps) != set(base_steps):
            raise ValueError(f"post step grid mismatch: {post_path}")
        arm = str(post_rows[0].get("arm") or post_path.stem)
        post_task = [int(bool(post_by_ep[episode_id]["task_success"])) for episode_id in selected_ids]
        post_tsr = sum(post_task) / len(post_task)
        post_step_accuracy = sum(bool(step["success"]) for step in post_steps.values()) / len(post_steps)
        tsr_delta = post_tsr - base_tsr
        step_delta = post_step_accuracy - base_step_accuracy
        if tsr_delta > 0 and step_delta >= -0.01:
            gate = "HELPS"
        elif tsr_delta < -0.01 or step_delta < -0.02:
            gate = "HARMS"
        else:
            gate = "NO_CLEAR_SIGNAL"
        training = None
        if args.training_root:
            metrics_path = Path(args.training_root) / arm / "metrics.jsonl"
            if metrics_path.exists():
                metrics = read_jsonl(metrics_path)
                training = metrics[-1] if metrics else None
        data_info = data_manifest["arms"].get(arm)
        oracle_control = bool(
            data_info and (data_info.get("selection_uses_matcher") or data_info.get("oracle_target_used"))
        )
        research_role = str((data_info or {}).get("research_role") or "unclassified")
        candidate_arm = research_role.startswith("candidate_")
        arms.append({
            "arm": arm,
            "post": str(post_path),
            "episodes": len(selected_ids),
            "steps": len(base_steps),
            "baseline_tsr": base_tsr,
            "post_tsr": post_tsr,
            "tsr_delta": tsr_delta,
            "baseline_step_accuracy": base_step_accuracy,
            "post_step_accuracy": post_step_accuracy,
            "step_accuracy_delta": step_delta,
            "paired_tsr_bootstrap": bootstrap_delta(base_task, post_task, args.bootstrap_draws, args.seed),
            "task_wrong_to_right": sum(not before and after for before, after in zip(base_task, post_task)),
            "task_right_to_wrong": sum(before and not after for before, after in zip(base_task, post_task)),
            "step_wrong_to_right": sum(not bool(base_steps[key]["success"]) and bool(post_steps[key]["success"]) for key in base_steps),
            "step_right_to_wrong": sum(bool(base_steps[key]["success"]) and not bool(post_steps[key]["success"]) for key in base_steps),
            "gate": gate,
            "oracle_control": oracle_control,
            "research_role": research_role,
            "candidate_arm": candidate_arm,
            "deployable_selector": candidate_arm and not oracle_control,
            "data": data_info,
            "last_training_metrics": training,
        })
    if not arms:
        raise ValueError("no post files matched")
    arms.sort(key=lambda row: (row["tsr_delta"], row["step_accuracy_delta"]), reverse=True)

    summary = {
        "protocol": {
            "selected_episodes": len(selected_ids),
            "selected_steps": len(base_steps),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "baseline_tsr": base_tsr,
            "baseline_step_accuracy": base_step_accuracy,
            "same_held_out_grid": True,
            "same_training_update_budget": data_manifest["training_policy"]["same_update_budget"],
            "screening_only": args.shard_count > 1,
            "held_out_fraction": 1.0 / args.shard_count,
        },
        "arms": arms,
        "full_eval_candidates": [
            row["arm"] for row in arms if row["gate"] == "HELPS" and row["deployable_selector"]
        ],
    }
    out_dir = Path(args.output_dir)
    write_json(out_dir / "summary.json", summary)
    report_rows = [
        [row["arm"], row["research_role"], pct(row["post_tsr"]), pp(row["tsr_delta"]), pct(row["post_step_accuracy"]), pp(row["step_accuracy_delta"]), row["gate"]]
        for row in arms
    ]
    lines = [
        "# Revision LoRA Equal-Budget Screen",
        "",
        f"Screening grid: held-out shard {args.shard_index}/{args.shard_count}, {len(selected_ids)} episodes / {len(base_steps)} steps ({pct(1.0 / args.shard_count)} of the full test set). Baseline TSR {pct(base_tsr)}, step accuracy {pct(base_step_accuracy)}.",
        "",
        table(["arm", "status", "TSR", "ΔTSR", "step acc", "Δstep", "gate"], report_rows),
        "",
        "All arms use the same optimizer-step budget. Matcher-selected arms are oracle diagnostic controls and cannot be promoted as deployable selectors. Any subset gate is screening evidence only; full 1,000-episode confirmation is required before a method claim.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"arms": len(arms), "full_eval_candidates": summary["full_eval_candidates"], "report": str(out_dir / "report.md")}, indent=2))


if __name__ == "__main__":
    main()
