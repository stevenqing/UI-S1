#!/usr/bin/env python3
"""Discover latent segmentation/capability schema from raw GUI trajectories.

This script does not consume the weak segment labels produced by
analyze_trajectory_segments.py. It loads raw/canonicalized train trajectories,
extracts benchmark-agnostic transition and local-window feature sets, then uses a
small dependency-free greedy Jaccard clustering procedure to surface recurring
boundary and capability archetypes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_trajectory_segments import (  # noqa: E402
    iter_android_control,
    iter_gui_odyssey,
    resolve_path,
    text_blob,
    tokenize,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]
FeatureSet = frozenset[str]

EXTRA_STOPWORDS = {
    "action", "allows", "appropriate", "begin", "choosing", "current", "displayed", "helps",
    "interface", "necessary", "needed", "need", "screen", "shows", "task", "using", "view",
    "access", "accessed", "continue", "process", "proceed", "relevant", "specific",
}


def bucket(value: float, bins: Sequence[tuple[float, str]]) -> str:
    for threshold, name in bins:
        if value <= threshold:
            return name
    return bins[-1][1]


def action_type(step: JsonDict) -> str:
    return str(step.get("action", {}).get("type", "unknown"))


def action_args(step: JsonDict) -> JsonDict:
    args = step.get("action", {}).get("args", {})
    return args if isinstance(args, dict) else {}


def button_name(step: JsonDict) -> str:
    return str(action_args(step).get("button", "")).lower()


def coord_region(step: JsonDict) -> str | None:
    coord = step.get("grounding", {}).get("coordinate") or action_args(step).get("coordinate")
    if not isinstance(coord, list) or len(coord) < 2:
        return None
    try:
        x = float(coord[0])
        y = float(coord[1])
    except (TypeError, ValueError):
        return None
    x_bucket = "left" if x < 333 else "mid_x" if x < 667 else "right"
    y_bucket = "top" if y < 333 else "mid_y" if y < 667 else "bottom"
    return f"region:{x_bucket}:{y_bucket}"


def text_tokens(step: JsonDict) -> set[str]:
    tokens = set(tokenize(text_blob(step)))
    tokens = {token for token in tokens if token not in EXTRA_STOPWORDS}
    args = action_args(step)
    for key in ("text", "button", "status"):
        value = args.get(key)
        if isinstance(value, str):
            tokens.update(token for token in tokenize(value) if token not in EXTRA_STOPWORDS)
    return tokens


def jaccard(left: set[str] | FeatureSet, right: set[str] | FeatureSet) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def transition_features(prev_step: JsonDict, curr_step: JsonDict, goal_tokens: set[str]) -> FeatureSet:
    prev_tokens = text_tokens(prev_step)
    curr_tokens = text_tokens(curr_step)
    text_shift = 1.0 - jaccard(prev_tokens, curr_tokens)
    prev_goal = jaccard(prev_tokens, goal_tokens)
    curr_goal = jaccard(curr_tokens, goal_tokens)
    delta_goal = curr_goal - prev_goal

    features = {
        "kind:transition",
        f"prev_action:{action_type(prev_step)}",
        f"curr_action:{action_type(curr_step)}",
        f"action_bigram:{action_type(prev_step)}->{action_type(curr_step)}",
        f"text_shift:{bucket(text_shift, [(0.25, 'low'), (0.55, 'mid'), (0.80, 'high'), (1.01, 'very_high')])}",
        f"goal_delta:{bucket(delta_goal, [(-0.10, 'down'), (0.10, 'flat'), (1.01, 'up')])}",
    }
    features.add("same_action_type" if action_type(prev_step) == action_type(curr_step) else "changed_action_type")
    if button_name(prev_step) in {"home", "back", "menu", "appselect"}:
        features.add("prev_system_nav")
    if button_name(curr_step) in {"home", "back", "menu", "appselect"}:
        features.add("curr_system_nav")
    if action_type(curr_step) == "terminate":
        features.add("curr_terminal")
    if action_type(curr_step) == "open":
        features.add("curr_open_surface")
    if action_type(prev_step) == "type":
        features.add("prev_entered_value")
    if action_type(curr_step) == "type":
        features.add("curr_enters_value")
    if prev_step.get("grounding", {}).get("bbox"):
        features.add("prev_has_bbox")
    if curr_step.get("grounding", {}).get("bbox"):
        features.add("curr_has_bbox")
    for token, _ in Counter((prev_tokens | curr_tokens) & goal_tokens).most_common(6):
        features.add(f"goal_tok:{token}")
    return frozenset(features)


def window_features(steps: Sequence[JsonDict], start: int, width: int, goal_tokens: set[str]) -> FeatureSet:
    window = steps[start : min(start + width, len(steps))]
    actions = [action_type(step) for step in window]
    tokens: set[str] = set()
    features = {"kind:window", f"len:{len(window)}"}
    for action, count in Counter(actions).most_common():
        features.add(f"action_count:{action}:{bucket(float(count), [(1.0, 'one'), (2.0, 'two'), (4.0, 'few'), (999.0, 'many')])}")
    if actions:
        features.add("action_seq:" + ">".join(actions[:5]))
        features.add(f"first_action:{actions[0]}")
        features.add(f"last_action:{actions[-1]}")
    for step in window:
        tokens.update(text_tokens(step))
        if step.get("grounding", {}).get("bbox"):
            features.add("window_has_bbox")
        if action_args(step).get("text"):
            features.add("window_has_text_arg")
        region = coord_region(step)
        if region:
            features.add(region)
    goal_overlap = jaccard(tokens, goal_tokens)
    features.add(f"goal_overlap:{bucket(goal_overlap, [(0.02, 'none'), (0.10, 'low'), (0.25, 'mid'), (1.01, 'high')])}")
    for token, _ in Counter(tokens).most_common(10):
        features.add(f"tok:{token}")
    return frozenset(features)


def add_to_clusters(clusters: list[JsonDict], features: FeatureSet, example: JsonDict, threshold: float, max_active_clusters: int) -> None:
    best_index = None
    best_score = 0.0
    for index, cluster in enumerate(clusters):
        score = jaccard(features, cluster["prototype"])
        if score > best_score:
            best_index = index
            best_score = score
    if (best_index is None or best_score < threshold) and len(clusters) < max_active_clusters:
        clusters.append({"prototype": set(features), "count": 1, "feature_counts": Counter(features), "examples": [example]})
        return
    if best_index is None:
        best_index = 0

    cluster = clusters[best_index]
    cluster["count"] += 1
    cluster["feature_counts"].update(features)
    if len(cluster["examples"]) < 6:
        cluster["examples"].append(example)
    min_support = max(2, int(cluster["count"] * 0.35))
    cluster["prototype"] = {feature for feature, count in cluster["feature_counts"].items() if count >= min_support}


def cluster_records(records: Iterable[tuple[FeatureSet, JsonDict]], threshold: float, min_count: int, max_active_clusters: int) -> list[JsonDict]:
    clusters: list[JsonDict] = []
    for features, example in records:
        add_to_clusters(clusters, features, example, threshold, max_active_clusters)
    clusters = [cluster for cluster in clusters if cluster["count"] >= min_count]
    clusters.sort(key=lambda item: item["count"], reverse=True)
    return clusters


def summarize_cluster(cluster: JsonDict, cluster_id: int) -> JsonDict:
    top_features = [feature for feature, _ in cluster["feature_counts"].most_common(20)]
    top_actions = [feature for feature in top_features if "action" in feature][:8]
    top_tokens = [feature.replace("tok:", "") for feature in top_features if feature.startswith("tok:")][:8]
    suggested_name_parts = []
    for feature in top_actions[:2]:
        suggested_name_parts.append(feature.replace("action_bigram:", "").replace("action_count:", ""))
    suggested_name_parts.extend(top_tokens[:3])
    suggested_name = " | ".join(suggested_name_parts) if suggested_name_parts else f"cluster_{cluster_id:02d}"
    return {
        "cluster_id": cluster_id,
        "count": cluster["count"],
        "suggested_name": suggested_name,
        "top_features": top_features,
        "examples": cluster["examples"],
    }


def load_episodes(args: argparse.Namespace) -> list[JsonDict]:
    max_examples = None if args.max_examples == 0 else args.max_examples
    episodes = []
    if not args.skip_gui_odyssey:
        gui_dir = resolve_path(args.gui_odyssey_dir)
        if gui_dir is not None:
            episodes.extend(iter_gui_odyssey(gui_dir, None, args.gui_split, args.gui_subset, max_examples))
    if not args.skip_android_control:
        android_path = resolve_path(args.android_control_jsonl)
        if android_path is not None:
            episodes.extend(iter_android_control(android_path, max_examples))
    return episodes


def discover(episodes: Sequence[JsonDict], args: argparse.Namespace) -> JsonDict:
    transition_records: list[tuple[FeatureSet, JsonDict]] = []
    window_records: list[tuple[FeatureSet, JsonDict]] = []
    action_counts: Counter[str] = Counter()
    total_transitions = 0
    total_windows = 0

    for episode in episodes:
        goal_tokens = set(tokenize(episode.get("task_goal", "")))
        steps = episode.get("steps", [])
        for step in steps:
            action_counts[action_type(step)] += 1
        for index in range(1, len(steps)):
            total_transitions += 1
            if len(transition_records) >= args.max_transition_records:
                continue
            features = transition_features(steps[index - 1], steps[index], goal_tokens)
            transition_records.append(
                (
                    features,
                    {
                        "benchmark": episode.get("benchmark"),
                        "episode_id": episode.get("episode_id"),
                        "after_step": index - 1,
                        "prev_action": action_type(steps[index - 1]),
                        "curr_action": action_type(steps[index]),
                        "prev_instruction": steps[index - 1].get("text_fields", {}).get("instruction", ""),
                        "curr_instruction": steps[index].get("text_fields", {}).get("instruction", ""),
                    },
                )
            )
        for start in range(0, len(steps), args.window_stride):
            total_windows += 1
            if len(window_records) >= args.max_window_records:
                continue
            features = window_features(steps, start, args.window_size, goal_tokens)
            window_records.append(
                (
                    features,
                    {
                        "benchmark": episode.get("benchmark"),
                        "episode_id": episode.get("episode_id"),
                        "start_step": start,
                        "end_step": min(start + args.window_size, len(steps)) - 1,
                        "actions": [action_type(step) for step in steps[start : min(start + args.window_size, len(steps))]],
                        "first_instruction": steps[start].get("text_fields", {}).get("instruction", "") if start < len(steps) else "",
                    },
                )
            )

    boundary_clusters = cluster_records(transition_records, args.boundary_threshold, args.min_cluster_count, args.max_active_clusters)
    capability_clusters = cluster_records(window_records, args.window_threshold, args.min_cluster_count, args.max_active_clusters)
    return {
        "summary": {
            "episodes": len(episodes),
            "steps": sum(len(episode.get("steps", [])) for episode in episodes),
            "transitions_seen": total_transitions,
            "transitions_clustered": len(transition_records),
            "windows_seen": total_windows,
            "windows_clustered": len(window_records),
            "action_counts": dict(action_counts.most_common()),
            "boundary_clusters": len(boundary_clusters),
            "capability_clusters": len(capability_clusters),
        },
        "boundary_archetypes": [summarize_cluster(cluster, index) for index, cluster in enumerate(boundary_clusters[: args.top_clusters])],
        "capability_archetypes": [summarize_cluster(cluster, index) for index, cluster in enumerate(capability_clusters[: args.top_clusters])],
    }


def format_counter(counter: dict[str, int]) -> list[str]:
    total = sum(counter.values()) or 1
    return [f"- `{key}`: {value} ({value / total:.1%})" for key, value in counter.items()]


def write_report(path: Path, result: JsonDict) -> None:
    lines = ["# Raw-Data Segmentation Schema Discovery", ""]
    summary = result["summary"]
    lines.extend(
        [
            "## Summary",
            "",
            f"- Episodes: {summary['episodes']}",
            f"- Steps: {summary['steps']}",
            f"- Transitions seen: {summary['transitions_seen']}",
            f"- Transitions clustered: {summary['transitions_clustered']}",
            f"- Local windows seen: {summary['windows_seen']}",
            f"- Local windows clustered: {summary['windows_clustered']}",
            f"- Boundary archetypes found: {summary['boundary_clusters']}",
            f"- Capability/window archetypes found: {summary['capability_clusters']}",
            "",
            "## Action Distribution",
            "",
            *format_counter(summary["action_counts"]),
        ]
    )
    for title, key in [("Boundary Archetypes", "boundary_archetypes"), ("Capability / Window Archetypes", "capability_archetypes")]:
        lines.extend(["", f"## {title}", ""])
        for cluster in result[key]:
            lines.append(f"### Cluster {cluster['cluster_id']} - {cluster['suggested_name']}")
            lines.append(f"- Count: {cluster['count']}")
            lines.append("- Top features: " + ", ".join(f"`{feature}`" for feature in cluster["top_features"][:12]))
            lines.append("- Examples:")
            for example in cluster["examples"][:3]:
                compact = {key: value for key, value in example.items() if value not in ("", [], None)}
                lines.append("  - " + json.dumps(compact, ensure_ascii=False))
            lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover segmentation schema from raw GUI trajectory data")
    parser.add_argument("--gui-odyssey-dir", type=str, default=str(PROJECT_ROOT / "datasets" / "GUI-Odyssey"))
    parser.add_argument("--gui-split", type=str, default="random_split")
    parser.add_argument("--gui-subset", type=str, default="train")
    parser.add_argument("--android-control-jsonl", type=str, default=str(PROJECT_ROOT / "datasets" / "ui_s1_dataset" / "train_1000.jsonl"))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "datasets" / "schema_discovery_train"))
    parser.add_argument("--max-examples", type=int, default=0, help="Max episodes per benchmark; 0 means all")
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--window-stride", type=int, default=2)
    parser.add_argument("--boundary-threshold", type=float, default=0.46)
    parser.add_argument("--window-threshold", type=float, default=0.42)
    parser.add_argument("--min-cluster-count", type=int, default=25)
    parser.add_argument("--top-clusters", type=int, default=20)
    parser.add_argument("--max-transition-records", type=int, default=50000)
    parser.add_argument("--max-window-records", type=int, default=50000)
    parser.add_argument("--max-active-clusters", type=int, default=300)
    parser.add_argument("--skip-gui-odyssey", action="store_true")
    parser.add_argument("--skip-android-control", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_episodes(args)
    logger.info("Loaded %d episodes for raw schema discovery", len(episodes))
    result = discover(episodes, args)

    json_path = output_dir / "discovered_schema.json"
    report_path = output_dir / "discovered_schema_report.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(report_path, result)
    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
