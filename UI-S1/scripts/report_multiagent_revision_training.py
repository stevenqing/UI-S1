#!/usr/bin/env python3
"""Report held-out causal deltas after training on noisy global revisions."""

from __future__ import annotations

import argparse
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


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def pp(value: float) -> str:
    return f"{100.0 * value:+.2f}pp"


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def bootstrap_delta(base: list[int], post: list[int], draws: int = 10000, seed: int = 42) -> dict[str, float]:
    if len(base) != len(post) or not base:
        raise ValueError("paired bootstrap requires equal non-empty vectors")
    rng = random.Random(seed)
    n = len(base)
    values = []
    for _ in range(draws):
        indices = [rng.randrange(n) for _ in range(n)]
        values.append(sum(post[idx] - base[idx] for idx in indices) / n)
    values.sort()
    return {
        "mean": sum(p - b for p, b in zip(post, base)) / n,
        "lo": values[int(0.025 * draws)],
        "hi": values[min(draws - 1, int(0.975 * draws))],
    }


def flatten(rows: list[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    out = {}
    for episode in rows:
        episode_id = str(episode["episode_id"])
        for fallback_idx, step in enumerate(episode.get("steps", [])):
            step_idx = int(step.get("step_idx", fallback_idx))
            out[str(step.get("target_id") or f"{episode_id}:{step_idx}")] = step
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--post", required=True)
    parser.add_argument("--correction-summary", required=True)
    parser.add_argument("--sft-summary", required=True)
    parser.add_argument("--training-metrics", required=True)
    parser.add_argument("--training-state", required=True)
    parser.add_argument("--output-dir", default="outputs/multiagent_trajectory_revision/pilot_v1/training_eval")
    parser.add_argument("--expected-episodes", type=int, default=0)
    parser.add_argument("--expected-steps", type=int, default=0)
    args = parser.parse_args()

    baseline_rows = read_jsonl(Path(args.baseline))
    post_rows = read_jsonl(Path(args.post))
    base_by_ep = {str(row["episode_id"]): row for row in baseline_rows}
    post_by_ep = {str(row["episode_id"]): row for row in post_rows}
    if set(base_by_ep) != set(post_by_ep) or (args.expected_episodes > 0 and len(base_by_ep) != args.expected_episodes):
        raise ValueError("held-out episode grid mismatch")
    base_steps = flatten(baseline_rows)
    post_steps = flatten(post_rows)
    if set(base_steps) != set(post_steps) or (args.expected_steps > 0 and len(base_steps) != args.expected_steps):
        raise ValueError("held-out step grid mismatch")

    episode_ids = [str(row["episode_id"]) for row in baseline_rows]
    base_task = [int(bool(base_by_ep[eid]["task_success"])) for eid in episode_ids]
    post_task = [int(bool(post_by_ep[eid]["task_success"])) for eid in episode_ids]
    paired = bootstrap_delta(base_task, post_task)
    base_step_acc = sum(bool(step["success"]) for step in base_steps.values()) / len(base_steps)
    post_step_acc = sum(bool(step["success"]) for step in post_steps.values()) / len(post_steps)
    base_tsr = sum(base_task) / len(base_task)
    post_tsr = sum(post_task) / len(post_task)
    task_wrong_to_right = sum(b == 0 and p == 1 for b, p in zip(base_task, post_task))
    task_right_to_wrong = sum(b == 1 and p == 0 for b, p in zip(base_task, post_task))
    step_wrong_to_right = sum(not bool(base_steps[tid]["success"]) and bool(post_steps[tid]["success"]) for tid in base_steps)
    step_right_to_wrong = sum(bool(base_steps[tid]["success"]) and not bool(post_steps[tid]["success"]) for tid in base_steps)
    tsr_delta = post_tsr - base_tsr
    step_delta = post_step_acc - base_step_acc

    if tsr_delta > 0 and step_delta >= -0.01:
        gate = "NOISY GLOBAL REVISION HELPS ON HELD-OUT GRID"
        reason = "Held-out TSR rises and held-out step accuracy stays within the -1pp guardrail."
    elif tsr_delta < -0.01 or step_delta < -0.02:
        gate = "NOISY GLOBAL REVISION HARMS ON HELD-OUT GRID"
        reason = "Held-out TSR or step accuracy crosses the predeclared harm threshold."
    else:
        gate = "NO CLEAR HELD-OUT SIGNAL"
        reason = f"The {len(base_task)}-episode evaluation does not show a directional effect beyond the guardrails."

    correction = read_json(Path(args.correction_summary))
    sft = read_json(Path(args.sft_summary))
    metrics = read_jsonl(Path(args.training_metrics))
    try:
        import torch
        state = torch.load(args.training_state, map_location="cpu", weights_only=True)
    except Exception:
        state = {}
    finetuning_type = str(state.get("finetuning_type") or "unknown")
    summary = {
        "gate": gate,
        "reason": reason,
        "teacher_forced": True,
        "held_out_episodes": len(episode_ids),
        "held_out_steps": len(base_steps),
        "baseline_tsr": base_tsr,
        "post_tsr": post_tsr,
        "tsr_delta": tsr_delta,
        "baseline_step_accuracy": base_step_acc,
        "post_step_accuracy": post_step_acc,
        "step_accuracy_delta": step_delta,
        "paired_tsr_delta_bootstrap": paired,
        "task_wrong_to_right": task_wrong_to_right,
        "task_right_to_wrong": task_right_to_wrong,
        "step_wrong_to_right": step_wrong_to_right,
        "step_right_to_wrong": step_right_to_wrong,
        "training_global_step": state.get("global_step"),
        "finetuning_type": finetuning_type,
        "last_training_metrics": metrics[-1] if metrics else None,
        "correction_diagnostics": correction,
        "sft_data": sft,
        "semantic_quality_filter_used": False,
        "phase2_started": False,
    }
    out_dir = Path(args.output_dir)
    write_json(out_dir / "summary.json", summary)
    rows = [
        ["baseline", len(episode_ids), pct(base_tsr), pct(base_step_acc)],
        [f"noisy-global-revision {finetuning_type}", len(episode_ids), pct(post_tsr), pct(post_step_acc)],
        ["delta", "", pp(tsr_delta), pp(step_delta)],
    ]
    lines = [
        "# Noisy Global-Trajectory Revision Training — Held-Out Evaluation",
        "",
        f"Train split only: global revisions from heterogeneous actor/corrector pairs. Every fully parseable revision was retained without matcher-based semantic filtering. Evaluation uses {len(episode_ids)} frozen GUI-360 test episodes / {len(base_steps)} steps and teacher-forced screenshots/history.",
        "",
        "## Held-Out Metrics",
        "",
        table(["arm", "episodes", "TSR", "step accuracy"], rows),
        "",
        f"Paired TSR delta bootstrap (fixed {len(episode_ids)} episodes, 10k draws): [{pp(paired['lo'])}, {pp(paired['hi'])}].",
        "",
        table(["flip", "count"], [
            ["task wrong->right", task_wrong_to_right],
            ["task right->wrong", task_right_to_wrong],
            ["step wrong->right", step_wrong_to_right],
            ["step right->wrong", step_right_to_wrong],
        ]),
        "",
        "## Data Diagnostics (not used for selection)",
        "",
        table(["metric", "value"], [
            ["revision parse", pct(correction["overall"]["parse_rate"])],
            ["actor step accuracy", pct(correction["overall"]["actor_step_accuracy"])],
            ["revised step accuracy", pct(correction["overall"]["revised_step_accuracy_diagnostic_only"])],
            ["noisy SFT rows", sft["sft_rows"]],
            ["diagnostic label accuracy", pct(sft["diagnostic_matcher_accuracy_not_used_for_selection"])],
            ["semantic filter", False],
        ]),
        "",
        "## Gate",
        "",
        gate,
        "",
        reason,
        "",
        "STOP for review. No additional training phase was started.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"gate": gate, "report": str(out_dir / "report.md")}, indent=2))


if __name__ == "__main__":
    main()
