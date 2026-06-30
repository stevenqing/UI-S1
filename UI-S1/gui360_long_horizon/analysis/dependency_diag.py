"""GUI-360 cross-step dependency diagnostic gate.

This module labels candidates only; it never trains a model. Accessibility and
OCR are offline referee inputs and are not passed into model prompts.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from gui360_long_horizon.analysis.controls import DEFAULT_DEPENDENCY_THRESHOLDS, DependencyGateThresholds, assert_thresholds_frozen, null_model_reuse_test, thresholds_to_dict
from gui360_long_horizon.data.availability import OcrReferee, classify_layer1_availability, load_ocr_cache, resurfaced_between
from gui360_long_horizon.data.carried_value import CandidateExtractionResult, DependencyCandidate, extract_candidates, get_attr, get_step_id
from gui360_long_horizon.data.defect_localize import load_prediction_records, summarize_defects
from gui360_long_horizon.data.distance import classify_distance
from gui360_long_horizon.data.pseudo_consumption import build_routine_profile, classify_pseudo_consumption


BUCKET_ORDER = [
    "given",
    "coincidence",
    "noise",
    "onscreen_a11y",
    "onscreen_ocr",
    "default",
    "clipboard",
    "resurfaced",
    "derivable",
    "forced",
    "routine",
    "adjacent",
    "nointerf",
    "persistent",
    "survivor",
]


def _as_namespace_steps(rows: Sequence[Mapping[str, Any]], episode_id: str) -> List[Any]:
    steps = []
    for index, row in enumerate(rows):
        payload = dict(row)
        payload.setdefault("exec_id", episode_id)
        payload.setdefault("step_id", index)
        steps.append(SimpleNamespace(**payload))
    return steps


def load_episode_json(path: str | Path) -> List[Sequence[Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    episodes: List[Sequence[Any]] = []
    if isinstance(payload, Mapping):
        if isinstance(payload.get("episodes"), list):
            payload = payload["episodes"]
        else:
            for episode_id, rows in payload.items():
                if isinstance(rows, list):
                    episodes.append(_as_namespace_steps(rows, str(episode_id)))
            return episodes
    if not isinstance(payload, list):
        raise ValueError(f"episode JSON must be a list, mapping, or {{episodes: [...]}}: {path}")
    for index, item in enumerate(payload):
        if isinstance(item, Mapping) and isinstance(item.get("steps"), list):
            episodes.append(_as_namespace_steps(item["steps"], str(item.get("exec_id") or item.get("episode_id") or index)))
        elif isinstance(item, list):
            episodes.append(_as_namespace_steps(item, str(index)))
    return episodes


def load_raw_gui360(repo: str, split: str, app: str, tag: str, *, limit_shards: Optional[int] = None) -> List[Sequence[Any]]:
    from gui360_long_horizon.data.loader import load_trajectories

    trajectories = load_trajectories(repo, split, app, tag, limit=limit_shards)
    return [sorted(steps, key=lambda step: get_step_id(step, 0)) for steps in trajectories.values()]


def _parse_tool_json(text: str) -> Dict[str, Any]:
    value = str(text or "")
    for pattern in (r"<tool_call>\s*(\{.*?\})\s*</tool_call>", r"```(?:json)?\s*(\{.*?\})\s*```"):
        match = re.search(pattern, value, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(1))
                return dict(parsed) if isinstance(parsed, Mapping) else {}
            except json.JSONDecodeError:
                pass
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", value):
        try:
            parsed, _ = decoder.raw_decode(value[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return {}


def _balanced_action(step: Mapping[str, Any]) -> Dict[str, Any]:
    raw_action = dict(step.get("action") or {})
    tool = _parse_tool_json(str(step.get("conversation_gpt") or ""))
    tool_args = tool.get("args") if isinstance(tool.get("args"), Mapping) else {}
    function = str(tool.get("function") or raw_action.get("function") or raw_action.get("action") or "").strip().lower()
    action: Dict[str, Any] = {"function": function, "action": function}
    coordinate = tool_args.get("coordinate") or raw_action.get("coordinate") or raw_action.get("xy")
    if coordinate is not None:
        action["coordinate"] = coordinate
        try:
            action["coordinate_x"] = float(coordinate[0])
            action["coordinate_y"] = float(coordinate[1])
        except (TypeError, ValueError, IndexError):
            pass
    text = raw_action.get("text") or tool_args.get("text") or tool_args.get("keys")
    if text is not None:
        action["text"] = str(text)
        action["args"] = {"text": str(text)}
    elif tool_args:
        action["args"] = dict(tool_args)
    return action


def _balanced_rect(step: Mapping[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    bbox = step.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        left, top, right, bottom = map(float, bbox[:4])
    except (TypeError, ValueError):
        return None
    return min(left, right), min(top, bottom), max(left, right), max(top, bottom)


def _balanced_xy(action: Mapping[str, Any]) -> Optional[Tuple[float, float]]:
    coord = action.get("coordinate") or action.get("xy")
    if coord is None:
        return None
    try:
        return float(coord[0]), float(coord[1])
    except (TypeError, ValueError, IndexError):
        return None


def _balanced_step(row: Mapping[str, Any], step: Mapping[str, Any], split: str) -> Any:
    episode_id = row.get("episode_id")
    step_idx = int(step.get("step_idx", step.get("step_id", 0)) or 0)
    action = _balanced_action(step)
    exec_id = f"gui360-balanced:{split}:{episode_id}"
    return SimpleNamespace(
        exec_id=exec_id,
        episode_id=exec_id,
        app="gui360-balanced",
        tag=str(row.get("goal") or "")[:80],
        split=split,
        step_id=step_idx,
        request=str(row.get("goal") or ""),
        template=str(row.get("goal") or ""),
        subtask="",
        observation="",
        thought="",
        status=str(step.get("status") or ""),
        screenshot_clean=str(step.get("screenshot") or ""),
        screenshot_desktop="",
        screenshot_annotated="",
        image_rel_path=str(step.get("screenshot") or ""),
        desktop_image_rel_path="",
        annotated_image_rel_path="",
        ui_tree=None,
        control_infos={},
        has_a11y=False,
        contiguous=True,
        gt_action=action,
        gt_function=str(action.get("function") or ""),
        gt_xy=_balanced_xy(action),
        gt_rect=_balanced_rect(step),
        raw={"balanced_row": {"episode_id": episode_id, "goal": row.get("goal"), "split": split}, "balanced_step": dict(step)},
    )


def load_balanced_parquet(data_dir: str | Path, split: str, *, max_episodes: int = 0) -> List[Sequence[Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required to read gui360-balanced parquet; run this through the uv diagnostic environment") from exc

    root = Path(data_dir)
    files = sorted(root.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split}-*.parquet files under {root}")
    episodes: List[Sequence[Any]] = []
    for path in files:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=128, columns=["episode_id", "goal", "steps"]):
            for row in batch.to_pylist():
                try:
                    raw_steps = json.loads(row.get("steps") or "[]")
                except json.JSONDecodeError:
                    continue
                if not isinstance(raw_steps, list):
                    continue
                steps = [_balanced_step(row, step, split) for step in raw_steps if isinstance(step, Mapping)]
                if steps:
                    episodes.append(steps)
                    if max_episodes > 0 and len(episodes) >= max_episodes:
                        return episodes
    return episodes


def _trajectory_map(episodes: Sequence[Sequence[Any]]) -> Dict[str, Sequence[Any]]:
    out: Dict[str, Sequence[Any]] = {}
    for index, raw_steps in enumerate(episodes):
        steps = sorted(list(raw_steps), key=lambda step: get_step_id(step, 0))
        if not steps:
            continue
        episode_id = str(get_attr(steps[0], "exec_id") or get_attr(steps[0], "episode_id") or f"episode-{index}")
        out[episode_id] = steps
    return out


def _classify_candidate(
    candidate: DependencyCandidate,
    trajectory: Sequence[Any],
    ocr: OcrReferee,
    routine_profile: Any,
    thresholds: DependencyGateThresholds,
) -> Tuple[str, Dict[str, Any]]:
    if candidate.layer0_bucket:
        return candidate.layer0_bucket, {}
    consume_step = trajectory[candidate.consume_index]
    availability = classify_layer1_availability(candidate, consume_step, ocr, thresholds=thresholds)
    if availability.bucket:
        return availability.bucket, availability.metadata
    if resurfaced_between(candidate, trajectory, ocr):
        return "resurfaced", {}
    pseudo_bucket, pseudo_meta = classify_pseudo_consumption(candidate, consume_step, routine_profile, thresholds=thresholds)
    if pseudo_bucket:
        return pseudo_bucket, pseudo_meta
    distance_bucket, distance_meta = classify_distance(candidate, trajectory, ocr, thresholds=thresholds)
    if distance_bucket:
        return distance_bucket, distance_meta
    return "survivor", {}


def _distance_histogram(candidates: Sequence[DependencyCandidate]) -> Dict[str, int]:
    counter = Counter(candidate.distance for candidate in candidates)
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _decide_verdict(q1: Mapping[str, Any], q2: Mapping[str, Any], q3: Mapping[str, Any], thresholds: DependencyGateThresholds) -> Dict[str, Any]:
    share = float(q1.get("battlefield_share", 0.0))
    if int(q1.get("candidate_total", 0)) == 0 or share < thresholds.q1_no_battlefield_share_max:
        return {"label": "NO_BATTLEFIELD", "reason": "Q1 battlefield share below pre-registered no-battlefield threshold"}
    if q3.get("available") and not q3.get("more_fail_at_consumption"):
        return {"label": "NO_BATTLEFIELD", "reason": "Q3 failures are not more likely at consumption steps"}
    battlefield = bool(
        share >= thresholds.q1_battlefield_share_min
        and int(q2.get("distance_ge3_n", 0)) >= thresholds.q2_min_distance_ge3_n
        and q3.get("available")
        and q3.get("more_fail_at_consumption")
        and float(q3.get("memory_4_4_fraction", 0.0)) >= thresholds.q3_min_memory_44_fraction
    )
    if battlefield:
        return {"label": "BATTLEFIELD", "reason": "Q1/Q2/Q3 all pass the pre-registered gate; proceed only to consumption-step-focused training"}
    return {"label": "MARGINAL", "reason": "Gate evidence is incomplete, small, or weak; do not train multi-turn on this basis"}


def run_dependency_diagnostic(
    episodes: Sequence[Sequence[Any]],
    *,
    ocr: Optional[OcrReferee] = None,
    prediction_rows: Optional[Sequence[Any]] = None,
    thresholds: DependencyGateThresholds = DEFAULT_DEPENDENCY_THRESHOLDS,
    max_candidates: int = -1,
) -> Dict[str, Any]:
    ocr_referee = ocr or OcrReferee(cache={}, missing_is_available=thresholds.missing_ocr_is_available)
    preliminary = extract_candidates(episodes, thresholds=thresholds, null_model_passed=True, max_candidates=max_candidates)
    null_model = null_model_reuse_test(
        preliminary.produced_by_episode,
        preliminary.consumed_by_episode,
        shuffles=thresholds.null_model_shuffles,
        seed=thresholds.null_model_seed,
        margin=thresholds.null_model_margin,
    )
    extraction: CandidateExtractionResult
    if preliminary.candidates and not null_model.passed:
        extraction = extract_candidates(episodes, thresholds=thresholds, null_model_passed=False, max_candidates=max_candidates)
    else:
        extraction = preliminary
    candidates = extraction.candidates
    routine_profile = build_routine_profile(candidates, thresholds=thresholds)
    trajectories = _trajectory_map(episodes)
    bucket_counts = Counter({bucket: 0 for bucket in BUCKET_ORDER})
    survivors: List[DependencyCandidate] = []
    candidate_rows = []
    for candidate in candidates:
        trajectory = trajectories.get(candidate.episode_id)
        if trajectory is None:
            bucket, metadata = "noise", {"reason": "missing_trajectory"}
        else:
            bucket, metadata = _classify_candidate(candidate, trajectory, ocr_referee, routine_profile, thresholds)
        bucket_counts[bucket] += 1
        if bucket == "survivor":
            survivors.append(candidate)
        candidate_rows.append({"candidate": asdict(candidate), "bucket": bucket, "metadata": metadata})
    total_accounted = sum(bucket_counts[bucket] for bucket in BUCKET_ORDER)
    if total_accounted != len(candidates):
        raise AssertionError(f"bucket accounting mismatch: accounted={total_accounted}, total={len(candidates)}")

    q1_share = float(len(survivors) / len(candidates)) if candidates else 0.0
    q1 = {"candidate_total": len(candidates), "survivor_n": len(survivors), "battlefield_share": q1_share}
    ge3 = [candidate for candidate in survivors if candidate.distance >= thresholds.long_horizon_min_distance]
    q2 = {
        "distance_histogram": _distance_histogram(survivors),
        "distance_ge3_n": len(ge3),
        "distance_ge3_share": float(len(ge3) / len(survivors)) if survivors else 0.0,
    }
    q3 = summarize_defects(survivors, list(prediction_rows or [])) if prediction_rows is not None else {"available": False, "reason": "prediction rows not supplied"}
    verdict = _decide_verdict(q1, q2, q3, thresholds)
    return {
        "schema_version": 1,
        "thresholds": thresholds_to_dict(thresholds),
        "bucket_counts": {bucket: int(bucket_counts[bucket]) for bucket in BUCKET_ORDER},
        "bucket_total": total_accounted,
        "null_model": asdict(null_model),
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "verdict": verdict,
        "candidate_rows_preview": candidate_rows[:50],
    }


def write_dependency_verdict(payload: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the GUI-360 cross-step dependency diagnostic gate")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episode-json", default="", help="Local JSON episodes for offline diagnostic tests")
    source.add_argument("--raw-gui360", action="store_true", help="Load raw GUI-360 trajectories via the existing loader")
    source.add_argument("--balanced-data-dir", default="", help="Local datasets/gui360-balanced/data parquet directory")
    parser.add_argument("--repo", default="vyokky/GUI-360")
    parser.add_argument("--split", default="test")
    parser.add_argument("--app", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--limit-shards", type=int, default=0)
    parser.add_argument("--ocr-cache", default="", help="JSON/JSONL cache from the offline OCR referee")
    parser.add_argument("--pred-rows", default="", help="Teacher-forced prediction rows for Layer 4/Q3")
    parser.add_argument("--thresholds", default="", help="Optional threshold JSON; must exactly match the frozen defaults")
    parser.add_argument("--max-candidates", type=int, default=-1)
    parser.add_argument("--out", default="reports/dependency_verdict.json")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    thresholds = DEFAULT_DEPENDENCY_THRESHOLDS
    if args.thresholds:
        thresholds = assert_thresholds_frozen(json.loads(Path(args.thresholds).read_text(encoding="utf-8")))
    if args.episode_json:
        episodes = load_episode_json(args.episode_json)
    elif args.balanced_data_dir:
        episodes = load_balanced_parquet(args.balanced_data_dir, args.split, max_episodes=args.max_episodes)
    else:
        if not args.app or not args.tag:
            raise ValueError("--raw-gui360 requires --app and --tag")
        episodes = load_raw_gui360(args.repo, args.split, args.app, args.tag, limit_shards=args.limit_shards or None)
    ocr = load_ocr_cache(args.ocr_cache)
    pred_rows = load_prediction_records(args.pred_rows) if args.pred_rows else None
    payload = run_dependency_diagnostic(episodes, ocr=ocr, prediction_rows=pred_rows, thresholds=thresholds, max_candidates=args.max_candidates)
    output = write_dependency_verdict(payload, args.out)
    print(json.dumps({"out": str(output), "verdict": payload["verdict"], "q1": payload["q1"], "q2": payload["q2"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())