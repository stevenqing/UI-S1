#!/usr/bin/env python3
"""Freeze label-blind Pass@8 packets and separately sealed evaluation labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from multiagent_trajectory_revision import normalize_action
from rl_feasibility_sampling import action_key


PROTOCOL_VERSION = "pass8-selector-v1"
PROMPT_VERSION = "pass8-fixed-choice-v1"
DEFAULT_SOURCES = (
    "qwen3_vl:outputs/multiagent_complementarity/qwen3_vl_candidates.jsonl",
    "qwen35:outputs/multiagent_complementarity/qwen35_candidates.jsonl",
    "llava15:outputs/multiagent_complementarity/extra_tiers/llava15_candidates.jsonl",
)


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


def stable_digest(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def target_id(row: Mapping[str, Any]) -> str:
    return str(row.get("target_id") or f"{row['episode_id']}:{row['step_idx']}")


def parse_source(spec: str) -> tuple[str, Path]:
    name, path = spec.split(":", 1)
    return name, Path(path)


def index_unique(rows: Iterable[Mapping[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        key = target_id(row)
        if key in indexed:
            raise ValueError(f"duplicate target_id in {source}: {key}")
        indexed[key] = row
    return indexed


def episode_step(episodes: Mapping[str, Mapping[str, Any]], tid: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    episode_id, step_text = tid.rsplit(":", 1)
    episode = episodes.get(episode_id)
    if episode is None:
        raise KeyError(f"target episode absent from episode data: {tid}")
    step_idx = int(step_text)
    steps = list(episode.get("steps") or [])
    if step_idx < 0 or step_idx >= len(steps):
        raise IndexError(f"target step is out of range: {tid}")
    return episode, steps[step_idx]


def history_strings(episode: Mapping[str, Any], step_idx: int) -> list[str]:
    history = []
    for index, previous in enumerate(list(episode.get("steps") or [])[:step_idx], start=1):
        history.append(f"Step {index}: {json.dumps(previous.get('action'), ensure_ascii=False, sort_keys=True)}")
    return history


def split_episodes(episode_ids: set[str], seed: str, smoke_count: int, dev_fraction: float) -> dict[str, str]:
    ordered = sorted(
        episode_ids,
        key=lambda episode_id: hashlib.sha256(f"{seed}|{episode_id}".encode("utf-8")).hexdigest(),
    )
    if smoke_count < 0 or smoke_count >= len(ordered):
        raise ValueError("smoke episode count must leave at least one non-smoke episode")
    smoke = set(ordered[:smoke_count])
    remaining = ordered[smoke_count:]
    dev_count = round(len(remaining) * dev_fraction)
    if dev_count <= 0 or dev_count >= len(remaining):
        raise ValueError("dev fraction must produce non-empty dev and locked-test splits")
    dev = set(remaining[:dev_count])
    return {
        episode_id: "smoke" if episode_id in smoke else "dev" if episode_id in dev else "locked_test"
        for episode_id in ordered
    }


def representative_action(entries: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    actions: dict[str, dict[str, Any]] = {}
    for entry in entries:
        action = dict(entry["action"])
        key = json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        counts[key] = counts.get(key, 0) + 1
        actions[key] = action
    best = min(counts, key=lambda key: (-counts[key], key))
    return actions[best]


def build_packet(
    tid: str,
    split: str,
    episode: Mapping[str, Any],
    step: Mapping[str, Any],
    baseline_row: Mapping[str, Any],
    sampled_sources: Mapping[str, Mapping[str, Any]],
    k: int,
    coord_bucket: int,
    seed: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline_action = normalize_action(baseline_row.get("greedy_pred_action"))
    grouped: dict[str, list[dict[str, Any]]] = {}
    sources = {"sft_anchor": baseline_row, **sampled_sources}
    for source_name, source_row in sources.items():
        samples = list(source_row.get("samples") or [])[:k]
        if len(samples) != k:
            raise ValueError(f"{tid}: {source_name} has {len(samples)} candidates, expected {k}")
        for sample_idx, sample in enumerate(samples):
            action = normalize_action(sample.get("pred_action"))
            if action is None:
                continue
            exact_key = json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            grouped.setdefault(exact_key, []).append({
                "source": source_name,
                "sample_idx": sample_idx,
                "action": action,
                "action_key": action_key(action, coord_bucket),
            })

    candidate_groups = list(grouped.items())
    rng_seed = int(hashlib.sha256(f"{seed}|candidates|{tid}".encode("utf-8")).hexdigest(), 16)
    random.Random(rng_seed).shuffle(candidate_groups)
    candidates: list[dict[str, Any]] = [{
        "candidate_id": "BASELINE",
        "action": baseline_action,
        "support_count": 1,
        "source_count": 1,
        "is_baseline": True,
    }]
    sealed_provenance: dict[str, Any] = {}
    neighborhood_entries: dict[str, list[dict[str, Any]]] = {}
    for entries in grouped.values():
        for entry in entries:
            neighborhood_entries.setdefault(str(entry["action_key"]), []).append(entry)
    for index, (_exact_key, entries) in enumerate(candidate_groups, start=1):
        candidate_id = f"C{index:02d}"
        neighborhood = neighborhood_entries[str(entries[0]["action_key"])]
        candidates.append({
            "candidate_id": candidate_id,
            "action": representative_action(entries),
            "support_count": len(entries),
            "neighborhood_support_count": len(neighborhood),
            "source_count": len({entry["source"] for entry in neighborhood}),
            "is_baseline": False,
        })
        sealed_provenance[candidate_id] = {
            "action_key": entries[0]["action_key"],
            "occurrences": [{"source": entry["source"], "sample_idx": entry["sample_idx"]} for entry in entries],
        }

    step_idx = int(step.get("step_idx") if step.get("step_idx") is not None else tid.rsplit(":", 1)[1])
    blind = {
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "target_id": tid,
        "split": split,
        "episode_id": str(episode["episode_id"]),
        "step_idx": step_idx,
        "goal": str(episode.get("goal") or ""),
        "history": history_strings(episode, step_idx),
        "screenshot": str(step["screenshot"]),
        "image_w": int(step.get("image_w") or baseline_row.get("image_w") or 1040),
        "image_h": int(step.get("image_h") or baseline_row.get("image_h") or 736),
        "baseline_action": baseline_action,
        "candidates": candidates,
    }
    blind["packet_sha256"] = stable_digest(blind)
    sealed = {
        "protocol_version": PROTOCOL_VERSION,
        "target_id": tid,
        "split": split,
        "episode_id": str(episode["episode_id"]),
        "step_idx": step_idx,
        "gt_action": normalize_action(step.get("action")),
        "candidate_provenance": sealed_provenance,
        "packet_sha256": blind["packet_sha256"],
    }
    return blind, sealed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default="outputs/multiagent_complementarity/target_ids.json")
    parser.add_argument("--episodes", default="outputs/validation_2k/data/test_episodes.jsonl")
    parser.add_argument("--baseline", default="outputs/rl_feasibility/per_step.jsonl")
    parser.add_argument("--candidate-source", action="append", default=[], help="name:path; repeatable")
    parser.add_argument("--output-dir", default="outputs/pass8_selector_study/frozen_v1")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--coord-bucket", type=int, default=25)
    parser.add_argument("--seed", default="pass8-selector-frozen-v1-20260213")
    parser.add_argument("--smoke-episodes", type=int, default=12)
    parser.add_argument("--dev-fraction", type=float, default=0.25)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite frozen holdout: {out_dir}")

    targets_path = Path(args.targets)
    target_payload = json.loads(targets_path.read_text(encoding="utf-8"))
    target_ids = [str(value) for value in (target_payload.get("target_ids") if isinstance(target_payload, dict) else target_payload)]
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("target manifest contains duplicate target IDs")
    target_set = set(target_ids)

    episodes_path = Path(args.episodes)
    episode_rows = read_jsonl(episodes_path)
    episodes = {str(row["episode_id"]): row for row in episode_rows}
    if len(episodes) != len(episode_rows):
        raise ValueError("episode data contains duplicate episode IDs")
    baseline_path = Path(args.baseline)
    baseline = index_unique(read_jsonl(baseline_path), str(baseline_path))
    missing_baseline = sorted(target_set - set(baseline))
    if missing_baseline:
        raise ValueError(f"baseline is missing {len(missing_baseline)} targets; first={missing_baseline[:3]}")

    source_specs = tuple(args.candidate_source) if args.candidate_source else DEFAULT_SOURCES
    source_paths: dict[str, Path] = {}
    source_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in source_specs:
        name, path = parse_source(spec)
        if name in source_rows or name == "sft_anchor":
            raise ValueError(f"duplicate or reserved candidate source: {name}")
        rows = index_unique(read_jsonl(path), str(path))
        missing = sorted(target_set - set(rows))
        if missing:
            raise ValueError(f"{name} is missing {len(missing)} targets; first={missing[:3]}")
        source_paths[name] = path
        source_rows[name] = rows

    target_episode_ids = {tid.rsplit(":", 1)[0] for tid in target_ids}
    split_by_episode = split_episodes(target_episode_ids, args.seed, args.smoke_episodes, args.dev_fraction)
    blind_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in ("smoke", "dev", "locked_test")}
    sealed_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in blind_rows}
    screenshot_rows = []
    screenshot_cache: dict[str, str] = {}
    for tid in target_ids:
        episode, step = episode_step(episodes, tid)
        expected_gt = normalize_action(step.get("action"))
        for source_name, source_row in {"sft_anchor": baseline[tid], **{name: rows[tid] for name, rows in source_rows.items()}}.items():
            source_gt = normalize_action(source_row.get("gt_action"))
            if source_gt != expected_gt:
                raise ValueError(f"{tid}: GT mismatch between episode data and {source_name}")
        episode_id = str(episode["episode_id"])
        split = split_by_episode[episode_id]
        packet, sealed = build_packet(
            tid,
            split,
            episode,
            step,
            baseline[tid],
            {name: rows[tid] for name, rows in source_rows.items()},
            args.k,
            args.coord_bucket,
            args.seed,
        )
        blind_rows[split].append(packet)
        sealed_rows[split].append(sealed)
        screenshot = str(step["screenshot"])
        if screenshot not in screenshot_cache:
            screenshot_cache[screenshot] = sha256_file(Path(screenshot))
        screenshot_rows.append({"target_id": tid, "screenshot": screenshot, "sha256": screenshot_cache[screenshot]})

    out_dir.mkdir(parents=True, exist_ok=False)
    for split in blind_rows:
        blind_rows[split].sort(key=lambda row: row["target_id"])
        sealed_rows[split].sort(key=lambda row: row["target_id"])
        write_jsonl(out_dir / "blind" / f"{split}.jsonl", blind_rows[split])
        sealed_path = out_dir / "sealed_labels" / f"{split}.jsonl"
        write_jsonl(sealed_path, sealed_rows[split])
        os.chmod(sealed_path, 0o600)
    write_json(out_dir / "screenshot_namespace.json", {
        "unique_screenshots": len(screenshot_cache),
        "namespace_sha256": stable_digest(screenshot_rows),
        "rows": screenshot_rows,
    })

    artifacts = {}
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            artifacts[str(path.relative_to(out_dir))] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    source_files = {str(path): {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in [targets_path, episodes_path, baseline_path, *source_paths.values()]}
    split_summary = {}
    for split, rows in blind_rows.items():
        split_summary[split] = {
            "steps": len(rows),
            "episodes": len({row["episode_id"] for row in rows}),
            "target_ids_sha256": stable_digest([row["target_id"] for row in rows]),
            "mean_candidates": sum(len(row["candidates"]) for row in rows) / max(1, len(rows)),
        }
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "gpu_policy": {"allowed_physical_gpus": [4, 5, 6, 7], "forbidden_physical_gpus": [0, 1, 2, 3], "protected_pid": 1911},
        "scope": {
            "targets": len(target_ids),
            "episodes": len(target_episode_ids),
            "k_per_source": args.k,
            "candidate_sources": ["sft_anchor", *source_rows],
            "coord_bucket": args.coord_bucket,
            "split_unit": "episode_id",
            "splits": split_summary,
        },
        "leakage_contract": {
            "blind_packets_exclude": ["gt_action", "candidate correctness", "reward", "model/source identity", "GT-derived criticality scores"],
            "sealed_labels_must_not_be_read_by_selector": True,
            "locked_test_must_not_be_evaluated_until_all_paired_outputs_are_complete": True,
            "known_limitations": [
                "The underlying GUI-360 benchmark test episodes are not benchmark-fresh and were used by earlier studies.",
                "The 962-step target set was selected previously using GT-derived critical-step diagnostics.",
                "This split is fresh only for the fixed-choice selector/corrector comparison.",
            ],
        },
        "predeclared_gate": {
            "primary_metric": "(rescue - regress) / locked_test_steps relative to frozen student greedy action",
            "pass": "net utility > 0, rescue > regress, and episode-cluster bootstrap 95% lower bound > 0",
            "stronger_model_win": "paired selected-accuracy delta vs current corrector has episode-cluster bootstrap 95% lower bound > 0",
            "invalid_or_failed_selection": "fall back to BASELINE",
            "policy_training": "forbidden unless the locked gate passes",
        },
        "source_files": source_files,
        "artifacts": artifacts,
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(out_dir), "splits": split_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()