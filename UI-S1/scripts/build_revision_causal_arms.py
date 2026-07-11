#!/usr/bin/env python3
"""Build paired causal training arms for heterogeneous trajectory revisions.

The builder changes target source, history source, or selection policy while
preserving the same frozen train screenshots and row schema. It does not train a
model. Arms that use matcher/ground-truth information are explicitly marked as
oracle controls and must not be described as deployable selectors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.multiagent_trajectory_revision import normalize_action, score_action  # noqa: E402
from scripts.rl_feasibility_sampling import action_key  # noqa: E402
from v13_gui_360.eval_gui360_template import _format_action_for_history  # noqa: E402
from v23_visual_transition.prepare_offline_data import tool_call_text  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def history_entry(action: Mapping[str, Any] | None, step_number: int, source: str) -> str:
    if action is None:
        return f"Step {step_number}: <unparseable_{source}_action>"
    return _format_action_for_history(dict(action), step_number)


def canonical_action(action: Any) -> dict[str, Any] | None:
    if not isinstance(action, Mapping) or not str(action.get("action") or "").strip():
        return None
    normalized = normalize_action(action)
    return normalized if normalized is not None else dict(action)


def build_base_steps(
    actor_rows: Sequence[Mapping[str, Any]], correction_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    actors = {str(row["trajectory_id"]): row for row in actor_rows}
    if len(actors) != len(actor_rows):
        raise ValueError("duplicate actor trajectory_id")

    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for correction in correction_rows:
        if not bool(correction.get("parse_ok")):
            continue
        trajectory_id = str(correction["trajectory_id"])
        if trajectory_id not in actors:
            raise ValueError(f"missing actor trajectory: {trajectory_id}")
        actor = actors[trajectory_id]
        if str(actor.get("split")) != "train" or str(correction.get("split")) != "train":
            raise ValueError("non-train row detected")
        source_steps = sorted(actor["steps"], key=lambda row: int(row["step_idx"]))
        revised_steps = sorted(correction["revised_steps"], key=lambda row: int(row["step_idx"]))
        if len(source_steps) != len(revised_steps):
            raise ValueError(f"actor/revision length mismatch: {trajectory_id}")

        gt_history: list[str] = []
        actor_history: list[str] = []
        revision_history: list[str] = []
        prior_revision_wrong = 0
        for source_step, revised_step in zip(source_steps, revised_steps):
            step_idx = int(source_step["step_idx"])
            if step_idx != int(revised_step["step_idx"]):
                raise ValueError(f"actor/revision index mismatch: {trajectory_id}:{step_idx}")
            key = (str(correction["correction_id"]), step_idx)
            if key in seen:
                raise ValueError(f"duplicate correction step: {key}")
            seen.add(key)

            gt_action = canonical_action(source_step.get("gt_action"))
            actor_action = normalize_action(source_step.get("actor_action"))
            revision_action = normalize_action(revised_step.get("action"))
            if gt_action is None or revision_action is None:
                raise ValueError(f"invalid GT/revision action: {trajectory_id}:{step_idx}")

            output.append(
                {
                    "correction_id": str(correction["correction_id"]),
                    "trajectory_id": trajectory_id,
                    "episode_id": str(actor["episode_id"]),
                    "step_idx": step_idx,
                    "num_steps": int(actor["num_steps"]),
                    "goal": str(actor["goal"]),
                    "target_id": str(source_step["target_id"]),
                    "image": str(source_step["screenshot"]),
                    "screenshot": str(source_step["screenshot"]),
                    "image_w": int(source_step.get("image_w") or 1040),
                    "image_h": int(source_step.get("image_h") or 736),
                    "gt_action": gt_action,
                    "actor_action": actor_action,
                    "revision_action": revision_action,
                    "gt_history": list(gt_history),
                    "actor_history": list(actor_history),
                    "revision_history": list(revision_history),
                    "actor": str(correction["actor"]),
                    "corrector": str(correction["corrector"]),
                    "correction_confidence": correction.get("confidence"),
                    "actor_correct": bool(source_step["actor_correct"]),
                    "revision_correct": bool(revised_step["diagnostic_correct"]),
                    "revision_changed_from_actor": bool(revised_step["changed_from_actor"]),
                    "prefix_wrong_count": prior_revision_wrong,
                    "prefix_clean": prior_revision_wrong == 0,
                    "is_last_step": step_idx == int(actor["num_steps"]) - 1,
                }
            )

            gt_history.append(history_entry(gt_action, step_idx + 1, "gt"))
            actor_history.append(history_entry(actor_action, step_idx + 1, "actor"))
            revision_history.append(history_entry(revision_action, step_idx + 1, "revision"))
            prior_revision_wrong += int(not bool(revised_step["diagnostic_correct"]))
    if not output:
        raise ValueError("no usable corrected steps")
    return output


def shuffled_marginal_actions(base_steps: Sequence[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Return an exact-marginal permutation minimizing same-key assignments."""
    rng = random.Random(seed)
    actions = [dict(row["gt_action"]) for row in base_steps]
    rng.shuffle(actions)
    gt_keys = [action_key(row["gt_action"], 25) for row in base_steps]
    assigned_keys = [action_key(action, 25) for action in actions]
    for _round in range(4):
        bad = [idx for idx, (assigned, target) in enumerate(zip(assigned_keys, gt_keys)) if assigned == target]
        if not bad:
            break
        rng.shuffle(bad)
        for idx in bad:
            if assigned_keys[idx] != gt_keys[idx]:
                continue
            for _attempt in range(256):
                other = rng.randrange(len(actions))
                if other == idx:
                    continue
                if assigned_keys[other] == gt_keys[idx] or assigned_keys[idx] == gt_keys[other]:
                    continue
                actions[idx], actions[other] = actions[other], actions[idx]
                assigned_keys[idx], assigned_keys[other] = assigned_keys[other], assigned_keys[idx]
                break
    return actions


def make_row(
    base: Mapping[str, Any],
    *,
    arm: str,
    target: Mapping[str, Any],
    target_source: str,
    history: Sequence[str],
    history_source: str,
    selection_policy: str,
    selection_uses_matcher: bool,
    oracle_target_used: bool,
    match_threshold: float,
) -> dict[str, Any]:
    score = score_action(
        target,
        base["gt_action"],
        int(base["image_w"]),
        int(base["image_h"]),
        match_threshold,
    )
    normalized_target = canonical_action(score.get("pred_action") or target)
    if normalized_target is None:
        raise ValueError(f"unparseable target for {arm}:{base['correction_id']}:{base['step_idx']}")
    teacher = {
        "gt": "ground_truth",
        "random_marginal": "random_marginal",
        "actor": str(base["actor"]),
        "revision": str(base["corrector"]),
    }[target_source]
    return {
        "sample_id": f"{arm}:{base['correction_id']}:{base['step_idx']}",
        "target_id": base["target_id"],
        "episode_id": base["episode_id"],
        "step_idx": base["step_idx"],
        "num_steps": base["num_steps"],
        "goal": base["goal"],
        "history": list(history),
        "image": base["image"],
        "screenshot": base["screenshot"],
        "gt_action": base["gt_action"],
        "actor_action": base["actor_action"],
        "revision_action": base["revision_action"],
        "chosen_action": normalized_target,
        "chosen_action_key": action_key(normalized_target, 25),
        "chosen_teacher": teacher,
        "target_text": tool_call_text(normalized_target, bool(base["is_last_step"])),
        "is_last_step": bool(base["is_last_step"]),
        "weight": 1.0,
        "source": "multiagent_revision_causal_arm",
        "treatment_arm": arm,
        "target_source": target_source,
        "history_source": history_source,
        "selection_policy": selection_policy,
        "actor": base["actor"],
        "corrector": base["corrector"],
        "correction_id": base["correction_id"],
        "trajectory_id": base["trajectory_id"],
        "correction_confidence": base["correction_confidence"],
        "actor_correct": bool(base["actor_correct"]),
        "revision_correct": bool(base["revision_correct"]),
        "diagnostic_matcher_correct": bool(score["correct"]),
        "diagnostic_reward": float(score["reward"]),
        "revision_changed_from_actor": bool(base["revision_changed_from_actor"]),
        "prefix_wrong_count": int(base["prefix_wrong_count"]),
        "prefix_clean": bool(base["prefix_clean"]),
        "selection_uses_matcher": selection_uses_matcher,
        "semantic_quality_filter_used": selection_uses_matcher,
        "oracle_target_used": oracle_target_used,
        "teacher_forced_screens": True,
        "global_future_screen_access": True,
    }


def summarize_arm(arm: str, rows: Sequence[Mapping[str, Any]], path: Path) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"empty arm: {arm}")
    keys = [(str(row["correction_id"]), int(row["step_idx"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate correction step in arm: {arm}")
    return {
        "arm": arm,
        "rows": len(rows),
        "episodes": len({str(row["episode_id"]) for row in rows}),
        "trajectories": len({str(row["correction_id"]) for row in rows}),
        "target_source": sorted({str(row["target_source"]) for row in rows}),
        "history_source": sorted({str(row["history_source"]) for row in rows}),
        "selection_policy": sorted({str(row["selection_policy"]) for row in rows}),
        "selection_uses_matcher": any(bool(row["selection_uses_matcher"]) for row in rows),
        "oracle_target_used": any(bool(row["oracle_target_used"]) for row in rows),
        "diagnostic_label_accuracy": sum(bool(row["diagnostic_matcher_correct"]) for row in rows) / len(rows),
        "prefix_clean_fraction": sum(bool(row["prefix_clean"]) for row in rows) / len(rows),
        "actor_counts": dict(Counter(str(row["actor"]) for row in rows)),
        "action_type_counts": dict(Counter(str(row["chosen_action"].get("action")) for row in rows)),
        "output": str(path),
        "output_sha256": sha256(path),
    }


def validate_reference(rows: Sequence[Mapping[str, Any]], reference_path: Path) -> dict[str, Any]:
    reference_rows = read_jsonl(reference_path)
    reference = {
        (str(row["correction_id"]), int(row["step_idx"])): row for row in reference_rows
    }
    candidate = {
        (str(row["correction_id"]), int(row["step_idx"])): row for row in rows
    }
    if set(reference) != set(candidate):
        raise ValueError("A4/reference correction-step grid mismatch")
    fields = ("history", "chosen_action_key", "target_text", "image", "target_id")
    mismatches = Counter()
    for key in reference:
        for field in fields:
            mismatches[field] += int(reference[key].get(field) != candidate[key].get(field))
    if any(mismatches.values()):
        raise ValueError(f"A4/reference core mismatch: {dict(mismatches)}")
    return {
        "reference": str(reference_path),
        "reference_sha256": sha256(reference_path),
        "rows": len(reference_rows),
        "core_fields": list(fields),
        "mismatches": dict(mismatches),
        "core_match": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor-trajectories", required=True)
    parser.add_argument("--corrections", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-revision-sft")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--match-threshold", type=float, default=0.5)
    args = parser.parse_args()

    actor_path = Path(args.actor_trajectories)
    correction_path = Path(args.corrections)
    output_dir = Path(args.output_dir)
    base_steps = build_base_steps(read_jsonl(actor_path), read_jsonl(correction_path))
    random_actions = shuffled_marginal_actions(base_steps, args.seed)

    arm_rows: dict[str, list[dict[str, Any]]] = {
        "a1_gt_target_gt_history": [],
        "a2_random_target_gt_history": [],
        "a3_actor_target_actor_history": [],
        "a4_revision_target_revision_history": [],
        "a5_revision_target_gt_history": [],
        "a6_gt_target_revision_history": [],
        "a7_revision_clean_prefix": [],
        "a8_revision_dirty_prefix": [],
        "a9_revision_internvl3_only": [],
        "a10_revision_qwen3_vl_only": [],
        "a11_oracle_revision_correct": [],
        "a12_oracle_revision_correct_clean_prefix": [],
    }

    for base, random_action in zip(base_steps, random_actions):
        common = {"match_threshold": args.match_threshold}
        arm_rows["a1_gt_target_gt_history"].append(make_row(
            base, arm="a1_gt_target_gt_history", target=base["gt_action"], target_source="gt",
            history=base["gt_history"], history_source="gt", selection_policy="all_usable",
            selection_uses_matcher=False, oracle_target_used=True, **common,
        ))
        arm_rows["a2_random_target_gt_history"].append(make_row(
            base, arm="a2_random_target_gt_history", target=random_action, target_source="random_marginal",
            history=base["gt_history"], history_source="gt", selection_policy="all_usable",
            selection_uses_matcher=False, oracle_target_used=False, **common,
        ))
        if base["actor_action"] is not None:
            arm_rows["a3_actor_target_actor_history"].append(make_row(
                base, arm="a3_actor_target_actor_history", target=base["actor_action"], target_source="actor",
                history=base["actor_history"], history_source="actor", selection_policy="actor_target_parseable",
                selection_uses_matcher=False, oracle_target_used=False, **common,
            ))
        a4 = make_row(
            base, arm="a4_revision_target_revision_history", target=base["revision_action"], target_source="revision",
            history=base["revision_history"], history_source="revision", selection_policy="all_usable",
            selection_uses_matcher=False, oracle_target_used=False, **common,
        )
        arm_rows["a4_revision_target_revision_history"].append(a4)
        arm_rows["a5_revision_target_gt_history"].append(make_row(
            base, arm="a5_revision_target_gt_history", target=base["revision_action"], target_source="revision",
            history=base["gt_history"], history_source="gt", selection_policy="all_usable",
            selection_uses_matcher=False, oracle_target_used=False, **common,
        ))
        arm_rows["a6_gt_target_revision_history"].append(make_row(
            base, arm="a6_gt_target_revision_history", target=base["gt_action"], target_source="gt",
            history=base["revision_history"], history_source="revision", selection_policy="all_usable",
            selection_uses_matcher=False, oracle_target_used=True, **common,
        ))
        if bool(base["prefix_clean"]):
            arm_rows["a7_revision_clean_prefix"].append(make_row(
                base, arm="a7_revision_clean_prefix", target=base["revision_action"], target_source="revision",
                history=base["revision_history"], history_source="revision", selection_policy="matcher_clean_prefix",
                selection_uses_matcher=True, oracle_target_used=False, **common,
            ))
        else:
            arm_rows["a8_revision_dirty_prefix"].append(make_row(
                base, arm="a8_revision_dirty_prefix", target=base["revision_action"], target_source="revision",
                history=base["revision_history"], history_source="revision", selection_policy="matcher_dirty_prefix",
                selection_uses_matcher=True, oracle_target_used=False, **common,
            ))
        if str(base["actor"]) == "internvl3":
            arm_rows["a9_revision_internvl3_only"].append(make_row(
                base, arm="a9_revision_internvl3_only", target=base["revision_action"], target_source="revision",
                history=base["revision_history"], history_source="revision", selection_policy="source_internvl3",
                selection_uses_matcher=False, oracle_target_used=False, **common,
            ))
        if str(base["actor"]) == "qwen3_vl":
            arm_rows["a10_revision_qwen3_vl_only"].append(make_row(
                base, arm="a10_revision_qwen3_vl_only", target=base["revision_action"], target_source="revision",
                history=base["revision_history"], history_source="revision", selection_policy="source_qwen3_vl",
                selection_uses_matcher=False, oracle_target_used=False, **common,
            ))
        if bool(base["revision_correct"]):
            arm_rows["a11_oracle_revision_correct"].append(make_row(
                base, arm="a11_oracle_revision_correct", target=base["revision_action"], target_source="revision",
                history=base["revision_history"], history_source="revision", selection_policy="matcher_revision_correct",
                selection_uses_matcher=True, oracle_target_used=False, **common,
            ))
            if bool(base["prefix_clean"]):
                arm_rows["a12_oracle_revision_correct_clean_prefix"].append(make_row(
                    base, arm="a12_oracle_revision_correct_clean_prefix", target=base["revision_action"], target_source="revision",
                    history=base["revision_history"], history_source="revision", selection_policy="matcher_correct_and_clean_prefix",
                    selection_uses_matcher=True, oracle_target_used=False, **common,
                ))

    summaries: dict[str, Any] = {}
    for arm, rows in arm_rows.items():
        path = output_dir / f"{arm}.jsonl"
        write_jsonl(path, rows)
        summaries[arm] = summarize_arm(arm, rows, path)
        write_json(path.with_suffix(".summary.json"), summaries[arm])

    manifest: dict[str, Any] = {
        "version": "multiagent-revision-causal-arms-v1",
        "seed": args.seed,
        "match_threshold": args.match_threshold,
        "actor_trajectories": str(actor_path),
        "actor_trajectories_sha256": sha256(actor_path),
        "corrections": str(correction_path),
        "corrections_sha256": sha256(correction_path),
        "base_usable_steps": len(base_steps),
        "arms": summaries,
        "notes": {
            "a1_a6": "target/history factorial and controls; GT target arms are oracle controls",
            "a7_a8": "matcher-defined prefix selection; diagnostic, not deployable",
            "a9_a10": "source-conditioned non-oracle selection",
            "a11_a12": "matcher-defined oracle ceilings",
            "random_arm": "exact empirical GT-action multiset permutation minimizing same-key assignments",
        },
    }
    if args.reference_revision_sft:
        manifest["a4_reference_validation"] = validate_reference(
            arm_rows["a4_revision_target_revision_history"], Path(args.reference_revision_sft)
        )
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({
        "base_usable_steps": len(base_steps),
        "arms": {arm: {"rows": item["rows"], "accuracy": item["diagnostic_label_accuracy"]} for arm, item in summaries.items()},
        "a4_reference_core_match": (manifest.get("a4_reference_validation") or {}).get("core_match"),
        "output_dir": str(output_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
