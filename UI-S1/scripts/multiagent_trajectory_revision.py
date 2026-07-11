#!/usr/bin/env python3
"""Heterogeneous actor trajectories -> cross-agent global revisions -> noisy SFT data.

This pipeline is deliberately explicit about its offline bound:
- Screens are teacher-forced GUI-360 screenshots, not autonomous rollout states.
- Correctors see the complete actor trajectory and all screenshots.
- Corrected actions are retained without matcher-based quality filtering.
- The frozen matcher is used only for diagnostics after generation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.minimal_validation import encode_image  # noqa: E402
from scripts.rl_feasibility_sampling import action_key, sanitize_jsonable  # noqa: E402
from v13_gui_360.eval_gui360_template import _format_action_for_history  # noqa: E402
from v13_gui_360.reward import compute_step_reward  # noqa: E402
from v23_visual_transition.prepare_offline_data import tool_call_text  # noqa: E402

ALLOWED_ACTIONS = {"click", "type", "swipe", "drag", "wheel_mouse_input", "press", "key"}
SUPPORTED_MANIFEST_VERSIONS = {"multiagent-global-revision-pilot-v1", "multiagent-global-revision-full-v1"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(sanitize_jsonable(dict(row)), ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(sanitize_jsonable(dict(row)), ensure_ascii=False) + "\n")
        handle.flush()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(seed: int, split: str, episode_id: str) -> str:
    return hashlib.sha256(f"{seed}|{split}|{episode_id}".encode()).hexdigest()


def sort_episode_id(value: str) -> tuple[int, Any]:
    return (0, int(value)) if value.isdigit() else (1, value)


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(item) for item in row) + " |" for row in rows)
    return "\n".join(lines)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def score_action(action: Mapping[str, Any] | None, gt_action: Mapping[str, Any], image_w: int, image_h: int, threshold: float) -> dict[str, Any]:
    if isinstance(action, Mapping):
        text = f"<action>{json.dumps(dict(action), ensure_ascii=False)}</action>"
    else:
        text = ""
    try:
        reward, info = compute_step_reward(text, dict(gt_action), image_w=image_w, image_h=image_h)
    except Exception as exc:  # diagnostic scoring must never terminate data generation
        return {
            "reward": 0.0,
            "correct": False,
            "pred_action": dict(action) if isinstance(action, Mapping) else None,
            "action_key": "__diagnostic_score_error__",
            "pred_type": None,
            "gt_type": str(gt_action.get("action") or ""),
            "diagnostic_error": f"{type(exc).__name__}: {exc}"[:500],
        }
    pred_action = info.get("pred_action")
    return {
        "reward": float(reward),
        "correct": bool(reward >= threshold),
        "pred_action": pred_action,
        "action_key": action_key(pred_action, 25),
        "pred_type": info.get("pred_type"),
        "gt_type": info.get("gt_type"),
    }


def freeze_pilot(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    train_path = Path(args.train_episodes)
    test_path = Path(args.test_episodes)
    train_rows = read_jsonl(train_path)
    test_rows = read_jsonl(test_path)

    def choose(rows: list[dict[str, Any]], split: str, count: int) -> list[dict[str, Any]]:
        eligible = [
            row for row in rows
            if args.min_steps <= len(row.get("steps", [])) <= args.max_steps
        ]
        eligible.sort(key=lambda row: stable_rank(args.seed, split, str(row["episode_id"])))
        if len(eligible) < count:
            raise ValueError(f"{split}: need {count} eligible episodes, found {len(eligible)}")
        return eligible[:count]

    actor_rows = list(train_rows) if args.scope == "full" else choose(train_rows, "train_actor", args.actor_episodes)
    eval_rows = list(test_rows) if args.scope == "full" else choose(test_rows, "test_eval", args.eval_episodes)
    actor_ids = [str(row["episode_id"]) for row in actor_rows]
    eval_ids = [str(row["episode_id"]) for row in eval_rows]
    target_ids = [
        f"{row['episode_id']}:{step_idx}"
        for row in actor_rows
        for step_idx, _step in enumerate(row.get("steps", []))
    ]
    write_jsonl(out_dir / "train_actor_episodes.jsonl", actor_rows)
    write_jsonl(out_dir / "test_eval_episodes.jsonl", eval_rows)
    write_json(out_dir / "train_target_ids.json", {"target_ids": target_ids})
    manifest = {
        "version": f"multiagent-global-revision-{args.scope}-v1",
        "scope": args.scope,
        "seed": args.seed,
        "teacher_forced_screens": True,
        "global_corrector_sees_future_screens": True,
        "semantic_quality_filter_for_training": False,
        "train_source": str(train_path),
        "test_source": str(test_path),
        "train_source_sha256": sha256(train_path),
        "test_source_sha256": sha256(test_path),
        "train_actor_episode_ids": actor_ids,
        "test_eval_episode_ids": eval_ids,
        "train_actor_episodes": len(actor_rows),
        "train_actor_steps": len(target_ids),
        "test_eval_episodes": len(eval_rows),
        "test_eval_steps": sum(len(row.get("steps", [])) for row in eval_rows),
        "episode_ids_are_split_local": True,
        "numeric_episode_id_overlap_count": len(set(actor_ids) & set(eval_ids)),
        "train_screenshot_namespace": "outputs/validation_2k/data/images/train/",
        "test_screenshot_namespace": "outputs/validation_2k/data/images/test/",
        "min_steps": None if args.scope == "full" else args.min_steps,
        "max_steps": None if args.scope == "full" else args.max_steps,
        "split_isolation": "Distinct train/test source hashes and screenshot namespaces; numeric episode IDs are split-local and may repeat",
    }
    write_json(out_dir / "pilot_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def load_agent_rows(spec: str) -> tuple[str, Path]:
    name, path_text = spec.split(":", 1)
    return name, Path(path_text)


def first_sample(row: Mapping[str, Any]) -> Mapping[str, Any]:
    samples = list(row.get("samples") or [])
    return samples[0] if samples else {}


def build_actor_trajectories(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    manifest = read_json(Path(args.manifest))
    if manifest.get("version") not in SUPPORTED_MANIFEST_VERSIONS:
        raise ValueError("unexpected pilot manifest")
    if args.split != "train":
        raise ValueError("pilot actor trajectories must use the train split")
    episodes = read_jsonl(Path(args.episode_data))
    episode_ids = [str(row["episode_id"]) for row in episodes]
    if set(episode_ids) != set(manifest["train_actor_episode_ids"]):
        raise ValueError("actor episode IDs do not match the frozen train pilot")
    expected_targets = {
        f"{episode['episode_id']}:{idx}"
        for episode in episodes
        for idx, _step in enumerate(episode.get("steps", []))
    }

    agents: dict[str, dict[str, dict[str, Any]]] = {}
    sources = {}
    for spec in args.agent_rows:
        name, path = load_agent_rows(spec)
        rows = read_jsonl(path)
        by_target = {str(row.get("target_id")): row for row in rows if str(row.get("target_id")) in expected_targets}
        if set(by_target) != expected_targets:
            missing = sorted(expected_targets - set(by_target))[:10]
            raise ValueError(f"{name}: incomplete actor rows {len(by_target)}/{len(expected_targets)}, missing={missing}")
        agents[name] = by_target
        sources[name] = {"path": str(path), "sha256": sha256(path), "rows": len(rows)}

    trajectories = []
    error_trajectories = []
    for agent, by_target in agents.items():
        for episode in sorted(episodes, key=lambda row: sort_episode_id(str(row["episode_id"]))):
            episode_id = str(episode["episode_id"])
            steps = []
            for idx, source_step in enumerate(episode.get("steps", [])):
                actor_row = by_target[f"{episode_id}:{idx}"]
                sample = first_sample(actor_row)
                actor_action = sample.get("pred_action") if isinstance(sample.get("pred_action"), Mapping) else None
                score = score_action(
                    actor_action,
                    source_step.get("action") or {},
                    int(source_step.get("image_w") or 1040),
                    int(source_step.get("image_h") or 736),
                    args.match_threshold,
                )
                steps.append({
                    "step_idx": idx,
                    "target_id": f"{episode_id}:{idx}",
                    "screenshot": source_step.get("screenshot"),
                    "image_w": int(source_step.get("image_w") or 1040),
                    "image_h": int(source_step.get("image_h") or 736),
                    "gt_action": source_step.get("action") or {},
                    "actor_raw_output": sample.get("raw_output") or "",
                    "actor_action": score["pred_action"],
                    "actor_action_key": score["action_key"],
                    "actor_parse_ok": score["pred_action"] is not None,
                    "actor_reward": score["reward"],
                    "actor_correct": score["correct"],
                })
            first_error = next((step["step_idx"] + 1 for step in steps if not step["actor_correct"]), None)
            row = {
                "trajectory_id": f"train:{agent}:{episode_id}",
                "split": "train",
                "episode_id": episode_id,
                "goal": episode.get("goal", ""),
                "actor": agent,
                "num_steps": len(steps),
                "correct_steps": sum(step["actor_correct"] for step in steps),
                "step_accuracy": sum(step["actor_correct"] for step in steps) / max(1, len(steps)),
                "task_success": first_error is None,
                "first_error_step": first_error,
                "teacher_forced_screens": True,
                "steps": steps,
            }
            trajectories.append(row)
            if first_error is not None:
                error_trajectories.append(row)

    by_agent = defaultdict(list)
    for row in trajectories:
        by_agent[row["actor"]].append(row)
    diversity = []
    agent_names = sorted(by_agent)
    error_sets = {
        agent: {
            step["target_id"]
            for row in by_agent[agent]
            for step in row["steps"]
            if not step["actor_correct"]
        }
        for agent in agent_names
    }
    action_keys = {
        agent: {
            step["target_id"]: step["actor_action_key"]
            for row in by_agent[agent]
            for step in row["steps"]
        }
        for agent in agent_names
    }
    for left_idx, left in enumerate(agent_names):
        for right in agent_names[left_idx + 1:]:
            union = error_sets[left] | error_sets[right]
            common_targets = sorted(set(action_keys[left]) & set(action_keys[right]))
            diversity.append({
                "pair": [left, right],
                "error_jaccard": len(error_sets[left] & error_sets[right]) / max(1, len(union)),
                "action_disagreement": sum(action_keys[left][tid] != action_keys[right][tid] for tid in common_targets) / max(1, len(common_targets)),
            })
    agent_summary = {
        agent: {
            "episodes": len(rows),
            "error_episodes": sum(not row["task_success"] for row in rows),
            "steps": sum(row["num_steps"] for row in rows),
            "step_accuracy": sum(row["correct_steps"] for row in rows) / max(1, sum(row["num_steps"] for row in rows)),
            "parse_rate": sum(step["actor_parse_ok"] for row in rows for step in row["steps"]) / max(1, sum(row["num_steps"] for row in rows)),
        }
        for agent, rows in by_agent.items()
    }
    write_jsonl(out_dir / "actor_trajectories.jsonl", trajectories)
    write_jsonl(out_dir / "error_trajectories.jsonl", error_trajectories)
    summary = {
        "manifest_version": manifest.get("version"),
        "scope": manifest.get("scope", "pilot"),
        "split": "train",
        "episodes": len(episodes),
        "actors": agent_names,
        "trajectories": len(trajectories),
        "error_trajectories": len(error_trajectories),
        "agent_summary": agent_summary,
        "pairwise_diversity": diversity,
        "sources": sources,
        "teacher_forced_screens": True,
    }
    write_json(out_dir / "actor_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def normalize_action(action: Any) -> dict[str, Any] | None:
    if not isinstance(action, Mapping):
        return None
    payload = dict(action)
    action_type = str(payload.get("action") or payload.get("type") or "").strip().lower()
    if action_type not in ALLOWED_ACTIONS:
        return None
    if action_type == "drag":
        action_type = "swipe"
    payload["action"] = action_type
    payload.pop("type", None)
    if "keys" in payload and "text" not in payload:
        payload["text"] = payload.pop("keys")

    def coord(value: Any) -> list[float] | None:
        if isinstance(value, Mapping):
            x = value.get("x", value.get("X"))
            y = value.get("y", value.get("Y"))
            value = [x, y]
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
            return None
        try:
            values = [float(value[0]), float(value[1])]
        except (TypeError, ValueError):
            return None
        return values if all(math.isfinite(item) for item in values) else None

    if "coordinate" in payload:
        normalized = coord(payload.get("coordinate"))
        if normalized is None:
            if action_type in {"click", "swipe"}:
                return None
            payload.pop("coordinate", None)
        else:
            payload["coordinate"] = normalized
    if action_type == "click" and "coordinate" not in payload:
        return None
    if action_type == "swipe":
        normalized_end = coord(payload.get("endCoordinate"))
        if "coordinate" not in payload or normalized_end is None:
            return None
        payload["endCoordinate"] = normalized_end
    return payload


def action_from_revision_row(row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    nested = row.get("action")
    if isinstance(nested, Mapping):
        return normalize_action(nested), "nested"
    if isinstance(nested, str):
        if nested.strip().lower() in {"", "null", "none", "noop", "no-op"}:
            return None, "omitted_noop"
        payload = {"action": nested}
        for key in ("coordinate", "endCoordinate", "text", "keys", "button", "double", "pressed"):
            if key in row:
                payload[key] = row[key]
        if "keys" in payload and "text" not in payload:
            payload["text"] = payload.pop("keys")
        return normalize_action(payload), "flattened"
    return None, "omitted_noop" if nested is None else "missing"


def extract_json_object(text: str) -> dict[str, Any] | None:
    candidates = []
    match = re.search(r"<trajectory_revision>\s*(\{.*?\})\s*</trajectory_revision>", text, re.DOTALL)
    if match:
        candidates.append(match.group(1))
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE))
    first = text.find("{")
    if first >= 0:
        candidates.append(text[first:])
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value, _end = decoder.raw_decode(candidate.strip())
            if isinstance(value, dict):
                return value
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def parse_revision(text: str, num_steps: int) -> tuple[dict[str, Any] | None, str]:
    payload = extract_json_object(text)
    if payload is None:
        return None, "no_json_object"
    rows = payload.get("revised_actions")
    if not isinstance(rows, list):
        return None, "missing_revised_actions"
    by_step = {}
    flattened_count = 0
    ignored_noop_count = 0
    ignored_out_of_range_count = 0
    for row in rows:
        if not isinstance(row, Mapping):
            return None, "non_object_revision"
        try:
            step_idx = int(row.get("step_idx"))
        except (TypeError, ValueError):
            return None, "invalid_step_idx"
        action, action_schema = action_from_revision_row(row)
        if action is None and action_schema == "omitted_noop":
            ignored_noop_count += 1
            continue
        if action is None or step_idx in by_step:
            return None, "invalid_or_duplicate_action"
        if step_idx < 0 or step_idx >= num_steps:
            ignored_out_of_range_count += 1
            continue
        flattened_count += action_schema == "flattened"
        by_step[step_idx] = {"step_idx": step_idx, "action": action, "rationale": str(row.get("rationale") or "")[:500], "schema": action_schema}
    missing_steps = sorted(set(range(num_steps)) - set(by_step))
    confidence = payload.get("confidence")
    try:
        confidence_value = min(1.0, max(0.0, float(confidence)))
    except (TypeError, ValueError):
        confidence_value = None
    return {
        "confidence": confidence_value,
        "analysis": str(payload.get("analysis") or "")[:2000],
        "revised_actions": [by_step[idx] for idx in sorted(by_step)],
        "missing_steps": missing_steps,
        "flattened_action_count": flattened_count,
        "ignored_noop_count": ignored_noop_count,
        "ignored_out_of_range_count": ignored_out_of_range_count,
    }, "ok" if not missing_steps and flattened_count == 0 and ignored_out_of_range_count == 0 else "recoverable_schema"


def build_corrector_messages(trajectory: Mapping[str, Any], image_max_pixels: int) -> list[dict[str, Any]]:
    intro = (
        "You are a global GUI trajectory reviser. Review the complete teacher-forced screenshot sequence and the actor's actions, "
        "then rewrite EVERY action as one globally coherent trajectory for the stated goal. Later screenshots are available because "
        "this is an offline global-revision experiment; do not assume the actor caused those screenshots. Do not ask for ground truth. "
        "Return exactly one JSON object inside <trajectory_revision> tags with keys: analysis (string), confidence (0..1), and "
        "revised_actions (a list of exactly one object per step). Each list object must contain step_idx, action, and rationale. "
        "Allowed action payloads use action=click/type/swipe/press with coordinate/text/endCoordinate as needed.\n\n"
        f"GOAL: {trajectory.get('goal', '')}\nACTOR: {trajectory.get('actor')}\nSTEPS: {trajectory.get('num_steps')}"
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": intro}]
    for step in trajectory.get("steps", []):
        action_text = json.dumps(step.get("actor_action"), ensure_ascii=False)
        raw_excerpt = str(step.get("actor_raw_output") or "").replace("\n", " ")[:300]
        content.append({
            "type": "text",
            "text": f"STEP {step['step_idx']} actor_action={action_text}\nactor_output_excerpt={raw_excerpt}\nScreenshot for STEP {step['step_idx']}:",
        })
        b64, _w, _h = encode_image(str(step["screenshot"]), image_max_pixels)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": "Now return the complete revised trajectory only in the required tagged JSON format."})
    return [{"role": "user", "content": content}]


def call_corrector(api_url: str, model: str, messages: list[dict[str, Any]], args: argparse.Namespace) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": bool(args.enable_thinking)},
    }
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            response = requests.post(
                api_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer EMPTY"},
                json=payload,
                timeout=args.request_timeout,
            )
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
            message = response.json()["choices"][0]["message"]
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""
            return f"<think>{reasoning}</think>\n{content}" if reasoning else content
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < args.retries:
                time.sleep(min(3.0 * (attempt + 1), 15.0))
    return f"ERROR: {last_error}"


def parse_api_spec(spec: str) -> tuple[str, str, str]:
    name, model, url = spec.split(":", 2)
    return name, model, url


def correct_one(
    trajectory: Mapping[str, Any],
    corrector: str,
    model: str,
    url: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if str(trajectory.get("actor")) == corrector:
        raise ValueError("actor and corrector must differ")
    messages = build_corrector_messages(trajectory, args.image_max_pixels)
    raw_output = call_corrector(url, model, messages, args)
    return materialize_correction(trajectory, corrector, model, raw_output, args)


def materialize_correction(
    trajectory: Mapping[str, Any],
    corrector: str,
    model: str,
    raw_output: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    revision, parse_status = parse_revision(raw_output, int(trajectory["num_steps"]))
    revised_steps = []
    imputed_steps: list[int] = []
    flattened_action_count = 0
    ignored_noop_count = 0
    ignored_out_of_range_count = 0
    if revision is not None:
        revised_by_step = {int(row["step_idx"]): row for row in revision["revised_actions"]}
        flattened_action_count = int(revision.get("flattened_action_count") or 0)
        ignored_noop_count = int(revision.get("ignored_noop_count") or 0)
        ignored_out_of_range_count = int(revision.get("ignored_out_of_range_count") or 0)
        materialization_failed = False
        for source_step in trajectory.get("steps", []):
            step_idx = int(source_step["step_idx"])
            revised = revised_by_step.get(step_idx)
            if revised is None:
                actor_action = normalize_action(source_step.get("actor_action"))
                if actor_action is None:
                    materialization_failed = True
                    break
                revised = {
                    "step_idx": step_idx,
                    "action": actor_action,
                    "rationale": "Corrector omitted this step; retained the actor action for fixed-length alignment.",
                    "schema": "imputed_actor",
                }
                imputed_steps.append(step_idx)
            score = score_action(
                revised["action"],
                source_step.get("gt_action") or {},
                int(source_step.get("image_w") or 1040),
                int(source_step.get("image_h") or 736),
                args.match_threshold,
            )
            revised_steps.append({
                **revised,
                "action": score["pred_action"] or revised["action"],
                "action_key": score["action_key"],
                "diagnostic_reward": score["reward"],
                "diagnostic_correct": score["correct"],
                "changed_from_actor": score["action_key"] != source_step.get("actor_action_key"),
                "schema": revised.get("schema"),
            })
        if materialization_failed:
            revised_steps = []
            parse_status = "missing_step_with_unparsed_actor_fallback"
        elif imputed_steps or flattened_action_count or ignored_noop_count or ignored_out_of_range_count:
            parse_status = "recovered_format_only"
    parse_ok = len(revised_steps) == int(trajectory["num_steps"])
    return {
        "correction_id": f"{trajectory['trajectory_id']}->{corrector}",
        "trajectory_id": trajectory["trajectory_id"],
        "split": trajectory.get("split"),
        "episode_id": trajectory.get("episode_id"),
        "goal": trajectory.get("goal"),
        "actor": trajectory.get("actor"),
        "corrector": corrector,
        "corrector_model": model,
        "teacher_forced_screens": True,
        "global_future_screen_access": True,
        "selection_uses_matcher": False,
        "parse_ok": parse_ok,
        "parse_status": parse_status,
        "confidence": None if revision is None else revision["confidence"],
        "analysis": "" if revision is None else revision["analysis"],
        "raw_output": raw_output,
        "num_steps": trajectory.get("num_steps"),
        "revised_steps": revised_steps,
        "diagnostic_correct_steps": sum(step["diagnostic_correct"] for step in revised_steps),
        "diagnostic_task_success": bool(revised_steps) and all(step["diagnostic_correct"] for step in revised_steps),
        "changed_steps": sum(step["changed_from_actor"] for step in revised_steps),
        "imputed_actor_steps": imputed_steps,
        "flattened_action_count": flattened_action_count,
        "ignored_noop_count": ignored_noop_count,
        "ignored_out_of_range_count": ignored_out_of_range_count,
    }


def summarize_corrections(rows: list[dict[str, Any]], trajectories: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[f"{row.get('actor')}->{row.get('corrector')}"] .append(row)

    def metrics(items: list[dict[str, Any]]) -> dict[str, Any]:
        parsed = [row for row in items if row.get("parse_ok")]
        actor_steps = sum(int(trajectories.get(str(row.get("trajectory_id")), {}).get("num_steps") or 0) for row in items)
        actor_correct = sum(int(trajectories.get(str(row.get("trajectory_id")), {}).get("correct_steps") or 0) for row in items)
        revised_steps = sum(len(row.get("revised_steps") or []) for row in parsed)
        revised_correct = sum(int(row.get("diagnostic_correct_steps") or 0) for row in parsed)
        return {
            "trajectories": len(items),
            "parse_rate": len(parsed) / max(1, len(items)),
            "actor_step_accuracy": actor_correct / max(1, actor_steps),
            "revised_step_accuracy_diagnostic_only": revised_correct / max(1, revised_steps),
            "revised_task_success_diagnostic_only": sum(bool(row.get("diagnostic_task_success")) for row in parsed) / max(1, len(parsed)),
            "changed_step_rate": sum(int(row.get("changed_steps") or 0) for row in parsed) / max(1, revised_steps),
            "mean_confidence": sum(float(row.get("confidence") or 0.0) for row in parsed) / max(1, len(parsed)),
            "format_recovered_trajectories": sum(row.get("parse_status") == "recovered_format_only" for row in parsed),
            "imputed_actor_steps": sum(len(row.get("imputed_actor_steps") or []) for row in parsed),
            "flattened_actions_recovered": sum(int(row.get("flattened_action_count") or 0) for row in parsed),
            "ignored_noop_rows": sum(int(row.get("ignored_noop_count") or 0) for row in parsed),
            "ignored_out_of_range_rows": sum(int(row.get("ignored_out_of_range_count") or 0) for row in parsed),
        }

    overall = metrics(rows)
    return {
        "overall": overall,
        "by_pair": {pair: metrics(items) for pair, items in sorted(by_pair.items())},
        "semantic_filter_used": False,
        "training_policy": "retain every fully parseable global revision, even matcher-wrong revisions",
        "schema_gate_pass": overall["parse_rate"] >= 0.80 and overall["trajectories"] > 0,
    }


def run_correctors(args: argparse.Namespace) -> None:
    trajectory_rows = read_jsonl(Path(args.trajectories))
    trajectories = {str(row["trajectory_id"]): row for row in trajectory_rows}
    apis: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for name, model, url in (parse_api_spec(spec) for spec in args.corrector_apis):
        apis[name].append((model, url))
    pair_specs = [tuple(spec.split(":", 1)) for spec in args.pairs]
    jobs = []
    for actor, corrector in pair_specs:
        if actor == corrector:
            raise ValueError(f"self-correction forbidden: {actor}")
        if corrector not in apis:
            raise ValueError(f"missing API for corrector {corrector}")
        actor_rows = [row for row in trajectory_rows if row.get("actor") == actor]
        endpoints = apis[corrector]
        jobs.extend((row, corrector, *endpoints[index % len(endpoints)]) for index, row in enumerate(actor_rows))
    output = Path(args.output)
    existing = read_jsonl(output) if args.resume else []
    done = {str(row.get("correction_id")) for row in existing}
    pending = [job for job in jobs if f"{job[0]['trajectory_id']}->{job[1]}" not in done]
    print(json.dumps({"jobs": len(jobs), "done": len(done), "pending": len(pending)}, indent=2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = [pool.submit(correct_one, trajectory, corrector, model, url, args) for trajectory, corrector, model, url in pending]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            append_jsonl(output, future.result())
            if index % 5 == 0 or index == len(futures):
                print(f"corrected {index}/{len(futures)}", flush=True)
    rows = read_jsonl(output)
    summary = summarize_corrections(rows, trajectories)
    summary["pairs"] = [list(pair) for pair in pair_specs]
    summary["source"] = args.trajectories
    summary["source_sha256"] = sha256(Path(args.trajectories))
    write_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def rescore_corrections(args: argparse.Namespace) -> None:
    trajectories = {str(row["trajectory_id"]): row for row in read_jsonl(Path(args.trajectories))}
    source_rows = read_jsonl(Path(args.corrections))
    rescored = []
    for row in source_rows:
        trajectory = trajectories.get(str(row.get("trajectory_id")))
        if trajectory is None:
            raise ValueError(f"missing trajectory {row.get('trajectory_id')}")
        rescored.append(materialize_correction(
            trajectory,
            str(row.get("corrector")),
            str(row.get("corrector_model")),
            str(row.get("raw_output") or ""),
            args,
        ))
    write_jsonl(Path(args.output), rescored)
    summary = summarize_corrections(rescored, trajectories)
    summary.update({
        "source": args.corrections,
        "source_sha256": sha256(Path(args.corrections)),
        "schema_recovery_only": True,
        "matcher_used_for_selection": False,
    })
    write_json(Path(args.output).with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def prepare_sft(args: argparse.Namespace) -> None:
    manifest = read_json(Path(args.manifest))
    if manifest.get("version") not in SUPPORTED_MANIFEST_VERSIONS:
        raise ValueError("unexpected pilot manifest")
    trajectories = {str(row["trajectory_id"]): row for row in read_jsonl(Path(args.trajectories))}
    corrections = read_jsonl(Path(args.corrections))
    if any(row.get("split") != "train" for row in corrections):
        raise ValueError("non-train correction detected")
    if any(str(row.get("episode_id")) not in set(manifest["train_actor_episode_ids"]) for row in corrections):
        raise ValueError("correction episode is outside frozen train actor split")

    output_rows = []
    parsed_trajectories = 0
    for correction in corrections:
        if not correction.get("parse_ok"):
            continue
        trajectory = trajectories.get(str(correction.get("trajectory_id")))
        if trajectory is None:
            raise ValueError(f"missing trajectory {correction.get('trajectory_id')}")
        revised_steps = list(correction.get("revised_steps") or [])
        if len(revised_steps) != int(trajectory.get("num_steps") or -1):
            continue
        parsed_trajectories += 1
        revised_history: list[str] = []
        for source_step, revised in zip(trajectory.get("steps", []), revised_steps):
            action = normalize_action(revised.get("action"))
            if action is None:
                break
            step_idx = int(source_step["step_idx"])
            output_rows.append({
                "sample_id": f"{correction['correction_id']}:{step_idx}",
                "target_id": source_step["target_id"],
                "episode_id": trajectory["episode_id"],
                "step_idx": step_idx,
                "num_steps": trajectory["num_steps"],
                "goal": trajectory["goal"],
                "history": list(revised_history),
                "image": source_step["screenshot"],
                "screenshot": source_step["screenshot"],
                "gt_action": source_step["gt_action"],
                "chosen_action": action,
                "chosen_action_key": action_key(action, 25),
                "chosen_teacher": correction["corrector"],
                "target_text": tool_call_text(action, step_idx == int(trajectory["num_steps"]) - 1),
                "is_last_step": step_idx == int(trajectory["num_steps"]) - 1,
                "weight": 1.0,
                "source": "multiagent_global_revision_noisy",
                "actor": correction["actor"],
                "corrector": correction["corrector"],
                "correction_id": correction["correction_id"],
                "correction_confidence": correction.get("confidence"),
                "diagnostic_matcher_correct": revised.get("diagnostic_correct"),
                "selection_uses_matcher": False,
                "teacher_forced_screens": True,
                "global_future_screen_access": True,
            })
            revised_history.append(_format_action_for_history(action, step_idx + 1))
    if not output_rows:
        raise ValueError("no parseable correction rows")
    if any(row["selection_uses_matcher"] for row in output_rows):
        raise AssertionError("matcher-based selection is forbidden for noisy arm")
    unique_sft_rows = len(output_rows)
    padding_rows = 0
    if args.pad_to_multiple > 0:
        padding_rows = (-len(output_rows)) % args.pad_to_multiple
        for index in range(padding_rows):
            clone = dict(output_rows[index % unique_sft_rows])
            clone["sample_id"] = f"{clone['sample_id']}:padding-{index}"
            clone["padding_repeat"] = True
            output_rows.append(clone)
    write_jsonl(Path(args.output), output_rows)
    summary = {
        "correction_rows": len(corrections),
        "parsed_correction_trajectories": parsed_trajectories,
        "sft_rows": len(output_rows),
        "unique_sft_rows": unique_sft_rows,
        "padding_rows": padding_rows,
        "pad_to_multiple": args.pad_to_multiple,
        "episode_ids": sorted({str(row["episode_id"]) for row in output_rows}, key=sort_episode_id),
        "actor_counts": dict(Counter(str(row["actor"]) for row in output_rows)),
        "corrector_counts": dict(Counter(str(row["corrector"]) for row in output_rows)),
        "diagnostic_matcher_accuracy_not_used_for_selection": sum(bool(row["diagnostic_matcher_correct"]) for row in output_rows) / len(output_rows),
        "action_type_counts": dict(Counter(str((row.get("chosen_action") or {}).get("action") or "unknown") for row in output_rows)),
        "pair_counts": dict(Counter(f"{row['actor']}->{row['corrector']}" for row in output_rows)),
        "pair_diagnostic_accuracy_not_used_for_selection": {
            pair: sum(bool(row["diagnostic_matcher_correct"]) for row in rows) / len(rows)
            for pair, rows in {
                pair: [row for row in output_rows if f"{row['actor']}->{row['corrector']}" == pair]
                for pair in sorted({f"{row['actor']}->{row['corrector']}" for row in output_rows})
            }.items()
        },
        "selection_uses_matcher": False,
        "semantic_quality_filter_used": False,
        "source": args.corrections,
        "source_sha256": sha256(Path(args.corrections)),
        "output": args.output,
        "output_sha256": sha256(Path(args.output)),
        "split": "train",
    }
    write_json(Path(args.output).with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def report(args: argparse.Namespace) -> None:
    actor_summary = read_json(Path(args.actor_summary))
    correction_summary = read_json(Path(args.correction_summary))
    sft_summary = read_json(Path(args.sft_summary)) if args.sft_summary and Path(args.sft_summary).exists() else None
    out_dir = Path(args.output_dir)
    scope = str(actor_summary.get("scope") or "pilot")
    train_gate = "TRAIN NOISY FULL DATASET" if scope == "full" else "TRAIN NOISY PILOT"
    lines = [
        f"# Heterogeneous Multi-Agent Global Trajectory Revision — {scope.title()} Data Report",
        "",
        "Offline teacher-forced bound. Actors run on GT screenshots/history. Correctors see the complete screenshot sequence, including future screenshots, and globally rewrite every action. Matcher scores are diagnostic only; every fully parseable correction is retained for the noisy-training arm.",
        "",
        "## Actor Errors",
        "",
        table(["actor", "episodes", "error episodes", "parse", "step accuracy"], [
            [agent, item["episodes"], item["error_episodes"], pct(item["parse_rate"]), pct(item["step_accuracy"])]
            for agent, item in sorted(actor_summary["agent_summary"].items())
        ]),
        "",
        "## Error Diversity",
        "",
        table(["pair", "error Jaccard", "action disagreement"], [
            [" vs ".join(item["pair"]), pct(item["error_jaccard"]), pct(item["action_disagreement"])]
            for item in actor_summary["pairwise_diversity"]
        ]),
        "",
        "## Global Revision Diagnostics (not selection)",
        "",
        table(["pair", "trajectories", "parse", "actor step acc", "revised step acc", "revised TSR", "changed steps", "confidence"], [
            [pair, item["trajectories"], pct(item["parse_rate"]), pct(item["actor_step_accuracy"]), pct(item["revised_step_accuracy_diagnostic_only"]), pct(item["revised_task_success_diagnostic_only"]), pct(item["changed_step_rate"]), pct(item["mean_confidence"])]
            for pair, item in correction_summary["by_pair"].items()
        ]),
        "",
        f"Schema gate: **{'PASS' if correction_summary['schema_gate_pass'] else 'FAIL'}**. Semantic filtering used: **{correction_summary['semantic_filter_used']}**.",
        "",
    ]
    if sft_summary is not None:
        lines.extend([
            "## Noisy SFT Data",
            "",
            table(["field", "value"], [
                ["rows", sft_summary["sft_rows"]],
                ["episodes", len(sft_summary["episode_ids"])],
                ["diagnostic matcher accuracy", pct(sft_summary["diagnostic_matcher_accuracy_not_used_for_selection"])],
                ["selection uses matcher", sft_summary["selection_uses_matcher"]],
            ]),
            "",
        ])
    lines.extend([
        "## Gate",
        "",
        train_gate if correction_summary["schema_gate_pass"] else "STOP — CORRECTION SCHEMA UNHEALTHY",
        "",
        "This gate checks only parse/completeness. It does not require corrected labels to be matcher-correct.",
        "",
    ])
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / ("data_report.md" if scope == "full" else "pilot_report.md")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    write_json(out_dir / "pilot_summary.json", {
        "gate": train_gate if correction_summary["schema_gate_pass"] else "STOP — CORRECTION SCHEMA UNHEALTHY",
        "scope": scope,
        "actor": actor_summary,
        "correction": correction_summary,
        "sft": sft_summary,
    })
    print(json.dumps({"report": str(report_path), "schema_gate": correction_summary["schema_gate_pass"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("freeze-pilot")
    p.add_argument("--train-episodes", default="outputs/validation_2k/data/train_episodes.jsonl")
    p.add_argument("--test-episodes", default="outputs/validation_2k/data/test_episodes.jsonl")
    p.add_argument("--output-dir", default="outputs/multiagent_trajectory_revision/pilot_v1")
    p.add_argument("--actor-episodes", type=int, default=12)
    p.add_argument("--eval-episodes", type=int, default=24)
    p.add_argument("--min-steps", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    p.set_defaults(func=freeze_pilot)

    p = sub.add_parser("build-actor-trajectories")
    p.add_argument("--episode-data", required=True)
    p.add_argument("--manifest", default="outputs/multiagent_trajectory_revision/pilot_v1/pilot_manifest.json")
    p.add_argument("--agent-rows", nargs="+", required=True, help="name:path")
    p.add_argument("--split", default="train")
    p.add_argument("--output-dir", default="outputs/multiagent_trajectory_revision/pilot_v1")
    p.add_argument("--match-threshold", type=float, default=0.5)
    p.set_defaults(func=build_actor_trajectories)

    p = sub.add_parser("run-correctors")
    p.add_argument("--trajectories", default="outputs/multiagent_trajectory_revision/pilot_v1/error_trajectories.jsonl")
    p.add_argument("--pairs", nargs="+", required=True, help="actor:corrector")
    p.add_argument("--corrector-apis", nargs="+", required=True, help="name:model:http://host:port/v1")
    p.add_argument("--output", default="outputs/multiagent_trajectory_revision/pilot_v1/global_corrections.jsonl")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=3072)
    p.add_argument("--image-max-pixels", type=int, default=301056)
    p.add_argument("--match-threshold", type=float, default=0.5)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--request-timeout", type=int, default=900)
    p.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.set_defaults(func=run_correctors)

    p = sub.add_parser("prepare-sft")
    p.add_argument("--manifest", default="outputs/multiagent_trajectory_revision/pilot_v1/pilot_manifest.json")
    p.add_argument("--trajectories", default="outputs/multiagent_trajectory_revision/pilot_v1/error_trajectories.jsonl")
    p.add_argument("--corrections", default="outputs/multiagent_trajectory_revision/pilot_v1/global_corrections.jsonl")
    p.add_argument("--output", default="outputs/multiagent_trajectory_revision/pilot_v1/noisy_global_corrections_train.jsonl")
    p.add_argument("--pad-to-multiple", type=int, default=0)
    p.set_defaults(func=prepare_sft)

    p = sub.add_parser("rescore-corrections")
    p.add_argument("--trajectories", default="outputs/multiagent_trajectory_revision/pilot_v1/error_trajectories.jsonl")
    p.add_argument("--corrections", default="outputs/multiagent_trajectory_revision/pilot_v1/global_corrections.jsonl")
    p.add_argument("--output", default="outputs/multiagent_trajectory_revision/pilot_v1/global_corrections_recovered.jsonl")
    p.add_argument("--match-threshold", type=float, default=0.5)
    p.set_defaults(func=rescore_corrections)

    p = sub.add_parser("report")
    p.add_argument("--actor-summary", default="outputs/multiagent_trajectory_revision/pilot_v1/actor_summary.json")
    p.add_argument("--correction-summary", default="outputs/multiagent_trajectory_revision/pilot_v1/global_corrections.summary.json")
    p.add_argument("--sft-summary", default="outputs/multiagent_trajectory_revision/pilot_v1/noisy_global_corrections_train.summary.json")
    p.add_argument("--output-dir", default="outputs/multiagent_trajectory_revision/pilot_v1")
    p.set_defaults(func=report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
