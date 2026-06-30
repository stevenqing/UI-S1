"""Recoverable overnight runner for GUI-360 long-horizon gates.

This module does not start or stop model servers. The shell launcher owns process
management via PID files; this Python runner only performs data/model API work
and writes resumable JSONL/JSON artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tarfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
import threading
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:  # pragma: no cover - exercised in lightweight/offline envs
    class HfApi:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("huggingface_hub is required for remote GUI-360 dataset access")

    def hf_hub_download(*args: Any, **kwargs: Any) -> str:  # type: ignore[no-redef]
        raise ImportError("huggingface_hub is required for remote GUI-360 dataset access")

from gui360_long_horizon.analysis.stats import DecisionAborted, decision
from gui360_long_horizon.data.difficulty import DifficultyProxyInvalid, fit_buckets, validity_gate
from gui360_long_horizon.data.divergence import ScreenIndex, delta, detect_t_star, make_screen_point
from gui360_long_horizon.data.loader import Step, load_image, load_trajectories
from gui360_long_horizon.harness.correctness import step_correct
from gui360_long_horizon.harness.model import VLLMClient
from gui360_long_horizon.harness.prompt import build_messages
from gui360_long_horizon.recovery_oracle import recovery_oracle
from gui360_long_horizon.run_all import load_config
from gui360_long_horizon.stages import loader_smoke
from gui360_long_horizon.types import task_key_for_step


DEFAULT_CONFIG = "gui360_long_horizon/configs/default.yaml"
DEFAULT_OUT_DIR = "outputs/gui360_long_horizon/overnight/manual"
FAIL_TAR_MEMBER_SMOKE = "image/excel/in_app/fail/excel_1_1/action_step1.png"


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def latest_by_step_uid(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for row in rows:
        uid = str(row.get("step_uid") or "")
        if not uid:
            continue
        if uid not in latest:
            order.append(uid)
        latest[uid] = row
    return [latest[uid] for uid in order if uid in latest]


def latest_by_textmem_key(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for row in rows:
        uid = str(row.get("step_uid") or "")
        mode = str((row.get("cond") or {}).get("history_mode") or "")
        key = f"{uid}:{mode}"
        if not uid or not mode:
            continue
        if key not in latest:
            order.append(key)
        latest[key] = row
    return [latest[key] for key in order if key in latest]


def latest_by_textdrift_key(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for row in rows:
        uid = str(row.get("step_uid") or "")
        injected = str((row.get("cond") or {}).get("injected_error") or "")
        key = f"{uid}:{injected}"
        if not uid or not injected:
            continue
        if key not in latest:
            order.append(key)
        latest[key] = row
    return [latest[key] for key in order if key in latest]


def latest_by_plan_key(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for row in rows:
        uid = str(row.get("step_uid") or "")
        plan = str((row.get("cond") or {}).get("plan") or "")
        key = f"{uid}:{plan}"
        if not uid or not plan:
            continue
        if key not in latest:
            order.append(key)
        latest[key] = row
    return [latest[key] for key in order if key in latest]


def rewrite_jsonl(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")
    tmp.replace(target)


def _step_uid(step: Step) -> str:
    return f"{step.split}:{step.app}:{step.tag}:{step.exec_id}:{step.step_id}"


def _iter_shards(config: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    for shard in config.get("shards") or [{"app": "excel", "tag": "in_app"}]:
        yield str(shard.get("app", "excel")), str(shard.get("tag", "in_app"))


def collect_success_steps(config: Dict[str, Any], limit: int) -> List[Step]:
    repo = str(config.get("repo") or "vyokky/GUI-360")
    steps: List[Step] = []
    for app, tag in _iter_shards(config):
        trajectories = load_trajectories(repo, "test", app, tag)
        for traj in trajectories.values():
            for step in traj:
                if step.gt_rect is not None and step.gt_function is not None:
                    steps.append(step)
                if limit and len(steps) >= limit:
                    return steps
    return steps


def collect_success_steps_with_history(config: Dict[str, Any], limit: int) -> List[Step]:
    repo = str(config.get("repo") or "vyokky/GUI-360")
    out: List[Step] = []
    for app, tag in _iter_shards(config):
        trajectories = load_trajectories(repo, "test", app, tag)
        for traj in trajectories.values():
            history_lines: List[str] = []
            summary_lines: List[str] = []
            native_lines: List[str] = []
            action_only_lines: List[str] = []
            for step in traj:
                if step.gt_rect is not None and step.gt_function is not None:
                    raw = dict(step.raw or {})
                    raw["history_text"] = "\n".join(history_lines) if history_lines else "None"
                    raw["history_summary"] = "; ".join(summary_lines[-6:]) if summary_lines else "No previous completed steps."
                    raw["history_native"] = "\n".join(native_lines) if native_lines else "None"
                    raw["history_native_last3"] = "\n".join(native_lines[-3:]) if native_lines else "None"
                    raw["history_action_only"] = "\n".join(action_only_lines) if action_only_lines else "None"
                    raw["history_action_last3"] = "\n".join(action_only_lines[-3:]) if action_only_lines else "None"
                    raw["corrupt_history"] = _corrupt_history(raw["history_text"], step)
                    out.append(replace(step, raw=raw))
                    if limit and len(out) >= limit:
                        return out
                history_lines.append(_history_line(step))
                native_lines.append(_native_action_history_line(step))
                action_only_lines.append(_action_only_history_line(step))
                if step.subtask:
                    summary_lines.append(f"Step {step.step_id}: {step.subtask}")
    return out


def collect_success_plan_items(config: Dict[str, Any], limit: int) -> List[Tuple[Step, str]]:
    repo = str(config.get("repo") or "vyokky/GUI-360")
    out: List[Tuple[Step, str]] = []
    for app, tag in _iter_shards(config):
        trajectories = load_trajectories(repo, "test", app, tag)
        for traj in trajectories.values():
            plan_text = _oracle_plan_for_traj(traj)
            for step in traj:
                if step.gt_rect is not None and step.gt_function is not None:
                    out.append((step, plan_text))
                    if limit and len(out) >= limit:
                        return out
    return out


def _oracle_plan_for_traj(traj: Sequence[Step]) -> str:
    lines = []
    for step in sorted(traj, key=lambda item: int(item.step_id)):
        if step.subtask:
            lines.append(f"Step {step.step_id}: {step.subtask}")
    return "\n".join(lines) if lines else "No explicit subtask sequence available."


def _history_line(step: Step) -> str:
    action = step.gt_action or {}
    pieces = [f"Step {step.step_id}"]
    if step.subtask:
        pieces.append(f"subtask={step.subtask}")
    if step.observation:
        pieces.append(f"observation={step.observation[:160]}")
    if action:
        action_bits = [str(action.get("function") or "")]
        if step.gt_xy is not None:
            action_bits.append(f"coordinate=[{int(step.gt_xy[0])}, {int(step.gt_xy[1])}]")
        pieces.append("action=" + " ".join(bit for bit in action_bits if bit))
    return " | ".join(pieces)


def _action_dict_for_history(step: Step) -> Dict[str, Any]:
    action = {"action": str(step.gt_function or "")}
    if step.gt_xy is not None:
        action["coordinate"] = [float(step.gt_xy[0]), float(step.gt_xy[1])]
    raw_action = step.gt_action or {}
    args = raw_action.get("args") if isinstance(raw_action.get("args"), dict) else {}
    text = args.get("keys") or args.get("text") or raw_action.get("text") or raw_action.get("control_text")
    if text:
        action["text"] = str(text)
    if raw_action.get("end_coordinate"):
        action["endCoordinate"] = raw_action.get("end_coordinate")
    return action


def _native_action_history_line(step: Step) -> str:
    action = _action_dict_for_history(step)
    atype = action.get("action", "")
    coord = action.get("coordinate")

    def valid_coord(value: Any) -> bool:
        return isinstance(value, (list, tuple)) and len(value) >= 2 and value[0] is not None and value[1] is not None

    if atype == "click":
        if valid_coord(coord):
            return f"Step {step.step_id}: click(coordinate=[{int(float(coord[0]))}, {int(float(coord[1]))}])"
        return f"Step {step.step_id}: click()"
    if atype == "type":
        text = str(action.get("text") or "")
        text = text[:30] + "..." if len(text) > 30 else text
        if valid_coord(coord) and text:
            return f"Step {step.step_id}: type(coordinate=[{int(float(coord[0]))}, {int(float(coord[1]))}], keys='{text}')"
        if valid_coord(coord):
            return f"Step {step.step_id}: type(coordinate=[{int(float(coord[0]))}, {int(float(coord[1]))}])"
        if text:
            return f"Step {step.step_id}: type(keys='{text}')"
        return f"Step {step.step_id}: type()"
    if atype in {"swipe", "drag"}:
        end = action.get("endCoordinate")
        if valid_coord(coord) and valid_coord(end):
            return f"Step {step.step_id}: drag(start_coordinate=[{int(float(coord[0]))}, {int(float(coord[1]))}], end_coordinate=[{int(float(end[0]))}, {int(float(end[1]))}])"
        return f"Step {step.step_id}: drag()"
    return f"Step {step.step_id}: {atype}()"


def _action_only_history_line(step: Step) -> str:
    action = _action_dict_for_history(step)
    atype = str(action.get("action") or "")
    coord = action.get("coordinate")
    if isinstance(coord, (list, tuple)) and len(coord) >= 2:
        return f"{step.step_id}: {atype} [{int(float(coord[0]))}, {int(float(coord[1]))}]"
    return f"{step.step_id}: {atype}"


def _corrupt_history(history_text: str, step: Step) -> str:
    corrupt_line = f"Step {max(1, int(step.step_id) - 1)} | action=click coordinate=[0, 0] | injected_error=true"
    if not history_text or history_text == "None":
        return corrupt_line
    return history_text + "\n" + corrupt_line


def collect_fail_trajectories(config: Dict[str, Any], limit: int) -> Dict[str, List[Step]]:
    repo = str(config.get("repo") or "vyokky/GUI-360")
    out: Dict[str, List[Step]] = {}
    for app, tag in _iter_shards(config):
        trajectories = load_trajectories(repo, "fail", app, tag)
        for exec_id, traj in trajectories.items():
            out[exec_id] = traj
            if limit and len(out) >= limit:
                return out
    return out


def fail_text_snippets(config: Dict[str, Any], limit_fail_traj: int, max_snippets: int = 512) -> List[str]:
    trajectories = collect_fail_trajectories(config, limit_fail_traj)
    snippets: List[str] = []
    for traj in trajectories.values():
        for step in traj:
            parts = []
            if step.subtask:
                parts.append(f"subtask={step.subtask}")
            if step.thought:
                parts.append(f"thought={step.thought[:180]}")
            if step.observation:
                parts.append(f"observation={step.observation[:180]}")
            if step.status:
                parts.append(f"status={step.status}")
            text = " | ".join(parts).strip()
            if text:
                snippets.append(text)
            if len(snippets) >= max_snippets:
                return snippets
    return snippets or ["Step 1: wrong prior action"]


def _remote_jsonl_paths(repo: str, split: str, app: str, tag: str) -> List[str]:
    prefix = f"{split}/data/{app}/{tag}"
    api = HfApi()
    try:
        entries = api.list_repo_tree(repo_id=repo, repo_type="dataset", path_in_repo=prefix, recursive=True)
        paths = [getattr(entry, "path", str(entry)) for entry in entries]
    except Exception:
        paths = api.list_repo_files(repo_id=repo, repo_type="dataset")
    return sorted(path for path in paths if path.startswith(prefix + "/") and path.endswith(".jsonl") and "processed_data/" not in path)


def phase_prefetch_jsonl(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    repo = str(config.get("repo") or "vyokky/GUI-360")
    cache_dir = Path(args.raw_local_dir)
    requested_splits = [item.strip() for item in args.prefetch_splits.split(",") if item.strip()]
    rows: List[Dict[str, Any]] = []
    for app, tag in _iter_shards(config):
        for split in requested_splits:
            paths = _remote_jsonl_paths(repo, split, app, tag)
            if args.prefetch_limit_per_split:
                paths = paths[: args.prefetch_limit_per_split]
            downloaded = 0
            skipped = 0
            for path in paths:
                local_path = cache_dir / path
                if local_path.exists():
                    skipped += 1
                    continue
                hf_hub_download(repo_id=repo, repo_type="dataset", filename=path, local_dir=str(cache_dir))
                downloaded += 1
            rows.append({"repo": repo, "app": app, "tag": tag, "split": split, "n_paths": len(paths), "downloaded": downloaded, "already_local": skipped})
    write_json(Path(args.out_dir) / "data" / "prefetch_jsonl.json", {"rows": rows, "prefetch_splits": requested_splits})


def _assembled_size_matches(path: Path, part_paths: Sequence[Path]) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    return path.stat().st_size == sum(part.stat().st_size for part in part_paths)


def phase_ensure_fail_images(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    repo = str(config.get("repo") or "vyokky/GUI-360")
    cache_dir = Path(args.raw_local_dir)
    fail_dir = cache_dir / "fail"
    fail_dir.mkdir(parents=True, exist_ok=True)
    assembled = fail_dir / "image.tar.gz"
    local_parts = sorted(fail_dir.glob("image.tar.gz[0-9][0-9][0-9]"))
    if local_parts:
        part_paths = local_parts
        parts = ["fail/" + path.name for path in part_paths]
    else:
        api = HfApi()
        files = api.list_repo_files(repo_id=repo, repo_type="dataset")
        parts = sorted(path for path in files if re.fullmatch(r"fail/image\.tar\.gz\d{3}", path))
        if not parts:
            raise RuntimeError("no fail/image.tar.gzNNN split parts found in HF repo")
        part_paths = []
        for part in parts:
            local = Path(hf_hub_download(repo_id=repo, repo_type="dataset", filename=part, local_dir=str(cache_dir)))
            part_paths.append(local)
    valid_before = _assembled_size_matches(assembled, part_paths)
    if not valid_before:
        tmp = assembled.with_suffix(".tar.gz.tmp")
        with tmp.open("wb") as out_handle:
            for part_path in part_paths:
                with part_path.open("rb") as in_handle:
                    while True:
                        chunk = in_handle.read(1024 * 1024 * 16)
                        if not chunk:
                            break
                        out_handle.write(chunk)
        tmp.replace(assembled)
    valid_after = _assembled_size_matches(assembled, part_paths)
    write_json(
        Path(args.out_dir) / "data" / "fail_images.json",
        {"repo": repo, "n_parts": len(parts), "assembled": str(assembled), "size_bytes": assembled.stat().st_size if assembled.exists() else 0, "valid": valid_after},
    )
    if not valid_after:
        raise RuntimeError(f"assembled fail image tar is not valid: {assembled}")


def _tar_member_for_fail_image(step: Step) -> str:
    rel = step.image_rel_path.replace("\\", "/").lstrip("/")
    prefix = "fail/image/"
    if not rel.startswith(prefix):
        raise ValueError(f"expected fail image path, got {rel!r}")
    return "image/" + rel[len(prefix):]


def _tar_member_for_split_image(step: Step, split: str) -> str:
    rel = step.image_rel_path.replace("\\", "/").lstrip("/")
    prefix = f"{split}/image/"
    if not rel.startswith(prefix):
        raise ValueError(f"expected {split} image path, got {rel!r}")
    return "image/" + rel[len(prefix):]


def _extract_members_from_tar(tar_path: Path, output_root: Path, members: Sequence[str]) -> Dict[str, Any]:
    members = sorted(set(members))
    existing = [member for member in members if (output_root / member).exists()]
    missing = [member for member in members if not (output_root / member).exists()]
    report: Dict[str, Any] = {"tar_path": str(tar_path), "n_members": len(members), "n_existing_before": len(existing), "n_missing_before": len(missing)}
    if missing:
        needed = set(missing)
        extracted = 0
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar:
                if member.name not in needed:
                    continue
                tar.extract(member, path=output_root)
                needed.remove(member.name)
                extracted += 1
                if not needed:
                    break
        if needed:
            report["missing_after_scan"] = sorted(needed)
        report["n_extracted_this_run"] = extracted
    existing_after = [member for member in members if (output_root / member).exists()]
    report.update({"n_existing_after": len(existing_after), "passed": len(existing_after) == len(members)})
    return report


def phase_extract_fail_images(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    cache_dir = Path(args.raw_local_dir)
    tar_path = cache_dir / "fail" / "image.tar.gz"
    if not tar_path.exists():
        raise RuntimeError(f"fail image tar is missing: {tar_path}")
    trajectories = collect_fail_trajectories(config, args.limit_fail_traj)
    members: List[str] = []
    for traj in trajectories.values():
        for step in traj:
            members.append(_tar_member_for_fail_image(step))
    report = _extract_members_from_tar(tar_path, cache_dir / "fail", members)
    report.update({
        "tar_path": str(tar_path),
        "limit_fail_traj": args.limit_fail_traj,
        "n_fail_traj": len(trajectories),
    })
    write_json(Path(args.out_dir) / "data" / "fail_image_extract.json", report)
    if not report["passed"]:
        raise RuntimeError("not all requested fail images were extracted")


def phase_extract_success_images(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    cache_dir = Path(args.raw_local_dir)
    tar_path = cache_dir / "test" / "image.tar.gz"
    if not tar_path.exists():
        raise RuntimeError(f"test image tar is missing: {tar_path}")
    steps = collect_success_steps(config, args.limit_success_images)
    members = [_tar_member_for_split_image(step, "test") for step in steps]
    report = _extract_members_from_tar(tar_path, cache_dir / "test", members)
    report.update({"limit_success_images": args.limit_success_images, "n_success_steps": len(steps)})
    write_json(Path(args.out_dir) / "data" / "success_image_extract.json", report)
    if not report["passed"]:
        raise RuntimeError("not all requested success images were extracted")


def phase_loader_smoke(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    result = loader_smoke(config)
    write_json(Path(args.out_dir) / "data" / "loader_smoke.json", {"passed": result.passed, "details": result.details})
    if not result.passed:
        raise RuntimeError("loader smoke gate failed")


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    patterns = [
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        r"```(?:json)?\s*(\{.*?\})\s*```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            continue
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _pred_dict_from_text(text: str) -> Dict[str, Any]:
    obj = _extract_json_object(text) or {}
    function = str(obj.get("function") or obj.get("action") or "").strip().lower()
    args = obj.get("args") if isinstance(obj.get("args"), dict) else obj
    if not function:
        func_match = re.search(r"<tool_call>\s*([A-Za-z_][A-Za-z0-9_]*)", text)
        if func_match:
            candidate = func_match.group(1).strip().lower()
            if candidate not in {"tool_call", "json"}:
                function = candidate
    xy = args.get("coordinate") or args.get("xy") or args.get("start_coordinate")
    if xy is None and "x" in args and "y" in args:
        xy = [args.get("x"), args.get("y")]
    pred = {"function": function, "raw_json": obj}
    if xy is not None:
        pred["coordinate"] = xy
    if function == "type" and "text" in args:
        pred["text"] = args.get("text")
    return pred


def _query_action(client: VLLMClient, step: Step, history_mode: str, input_mode: str, max_tokens: int, plan: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    messages = build_messages(step, history_mode, input_mode, plan=plan)
    decode = client.generate(messages, n=1, logprobs=False, max_tokens=max_tokens, temperature=0.0)[0]
    pred = _pred_dict_from_text(decode.text)
    return decode.text, pred


def phase_score_success(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    out_path = Path(args.out_dir) / "scores" / f"{args.score_label}.jsonl"
    summary_path = Path(args.out_dir) / "scores" / f"{args.score_label}_summary.json"
    initial_rows = latest_by_step_uid(read_jsonl(out_path))
    existing = {str(row.get("step_uid")) for row in initial_rows if row.get("ok") and row.get("step_uid")}
    steps = collect_success_steps(config, args.limit_steps)
    started = time.time()
    n_new = 0
    parse_fail = 0
    errors = 0
    steps_to_score = [step for step in steps if _step_uid(step) not in existing]
    local_state = threading.local()

    def get_client() -> VLLMClient:
        client = getattr(local_state, "client", None)
        if client is None:
            client = VLLMClient(args.api_url, args.model_name, timeout=args.timeout)
            local_state.client = client
        return client

    def score_one(step: Step) -> Dict[str, Any]:
        uid = _step_uid(step)
        row: Dict[str, Any] = {
            "step_uid": uid,
            "exec_id": step.exec_id,
            "app": step.app,
            "tag": step.tag,
            "split": step.split,
            "step_id": step.step_id,
            "total_steps": step.total_steps,
            "task_key": task_key_for_step(step),
            "model": args.model_name,
            "score_label": args.score_label,
            "history_mode": args.history_mode,
            "input_mode": args.input_mode,
        }
        try:
            text, pred = _query_action(get_client(), step, args.history_mode, args.input_mode, args.max_tokens)
            correct = step_correct(pred, step)
            row.update({"ok": True, "correct": bool(correct), "difficulty": float(1.0 - int(bool(correct))), "prediction": pred, "text": text})
        except Exception as exc:
            row.update({"ok": False, "correct": None, "difficulty": None, "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
        return row

    if args.threads <= 1:
        iterator = (score_one(step) for step in steps_to_score)
    else:
        executor = ThreadPoolExecutor(max_workers=args.threads)
        futures = [executor.submit(score_one, step) for step in steps_to_score]
        iterator = (future.result() for future in as_completed(futures))

    try:
        for row in iterator:
            if row.get("ok"):
                if not (row.get("prediction") or {}).get("function"):
                    parse_fail += 1
            else:
                errors += 1
            uid = str(row.get("step_uid"))
            append_jsonl(out_path, row)
            if row.get("ok"):
                existing.add(uid)
            n_new += 1
            if args.flush_every and n_new % args.flush_every == 0:
                rows = latest_by_step_uid(read_jsonl(out_path))
                write_json(summary_path, summarize_score_rows(rows, extra={"elapsed_sec": time.time() - started, "in_progress": True, "threads": args.threads}))
    finally:
        if args.threads > 1:
            executor.shutdown(wait=False, cancel_futures=True)
    rows = latest_by_step_uid(read_jsonl(out_path))
    rewrite_jsonl(out_path, rows)
    write_json(summary_path, summarize_score_rows(rows, extra={"elapsed_sec": time.time() - started, "in_progress": False, "n_new": n_new, "errors_this_run": errors, "parse_fail_this_run": parse_fail, "threads": args.threads}))


def phase_textmem_gate(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    out_path = Path(args.out_dir) / "scores" / "textmem_gate.jsonl"
    summary_path = Path(args.out_dir) / "reports" / "textmem_gate.json"
    modes = [mode.strip() for mode in args.history_modes.split(",") if mode.strip()]
    if not modes:
        raise ValueError("history_modes must not be empty")
    initial_rows = latest_by_textmem_key(read_jsonl(out_path))
    existing = {f"{row.get('step_uid')}:{(row.get('cond') or {}).get('history_mode')}" for row in initial_rows if row.get("ok") and row.get("step_uid")}
    steps = collect_success_steps_with_history(config, args.limit_steps)
    local_state = threading.local()
    started = time.time()

    def get_client() -> VLLMClient:
        client = getattr(local_state, "client", None)
        if client is None:
            client = VLLMClient(args.api_url, args.model_name, timeout=args.timeout)
            local_state.client = client
        return client

    def score_one(item: Tuple[Step, str]) -> Dict[str, Any]:
        step, mode = item
        uid = _step_uid(step)
        row: Dict[str, Any] = {
            "step_uid": uid,
            "exec_id": step.exec_id,
            "app": step.app,
            "tag": step.tag,
            "split": step.split,
            "step_id": step.step_id,
            "total_steps": step.total_steps,
            "task_key": task_key_for_step(step),
            "model": args.model_name,
            "score_label": "textmem_gate",
            "cond": {"history_mode": mode, "input_mode": args.input_mode, "plan": None, "injected_error": 1 if mode == "corrupt" else 0},
        }
        try:
            text, pred = _query_action(get_client(), step, mode, args.input_mode, args.max_tokens)
            correct = step_correct(pred, step)
            row.update({"ok": True, "correct": bool(correct), "prediction": pred, "text": text})
        except Exception as exc:
            row.update({"ok": False, "correct": None, "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
        return row

    items = [(step, mode) for step in steps for mode in modes if f"{_step_uid(step)}:{mode}" not in existing]
    n_new = 0
    errors = 0
    parse_fail = 0
    if args.threads <= 1:
        iterator = (score_one(item) for item in items)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=args.threads)
        futures = [executor.submit(score_one, item) for item in items]
        iterator = (future.result() for future in as_completed(futures))
    try:
        for row in iterator:
            if row.get("ok"):
                if not (row.get("prediction") or {}).get("function"):
                    parse_fail += 1
            else:
                errors += 1
            append_jsonl(out_path, row)
            n_new += 1
            if args.flush_every and n_new % args.flush_every == 0:
                write_json(summary_path, summarize_textmem_rows(latest_by_textmem_key(read_jsonl(out_path)), args.gate_eps, extra={"in_progress": True, "elapsed_sec": time.time() - started, "threads": args.threads}))
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
    rows = latest_by_textmem_key(read_jsonl(out_path))
    rewrite_jsonl(out_path, rows)
    write_json(summary_path, summarize_textmem_rows(rows, args.gate_eps, extra={"in_progress": False, "elapsed_sec": time.time() - started, "threads": args.threads, "n_new": n_new, "errors_this_run": errors, "parse_fail_this_run": parse_fail}))


def phase_textdrift_gate(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    out_path = Path(args.out_dir) / "scores" / "textdrift_gate.jsonl"
    summary_path = Path(args.out_dir) / "reports" / "textdrift_gate.json"
    initial_rows = latest_by_textdrift_key(read_jsonl(out_path))
    existing = {f"{row.get('step_uid')}:{(row.get('cond') or {}).get('injected_error')}" for row in initial_rows if row.get("ok") and row.get("step_uid")}
    baseline_correct: Optional[set[str]] = None
    if args.baseline_rows:
        baseline_correct = set()
        for row in latest_by_textmem_key(read_jsonl(args.baseline_rows)):
            if (row.get("cond") or {}).get("history_mode") == "none" and row.get("ok") and row.get("correct"):
                baseline_correct.add(str(row.get("step_uid")))
    steps = collect_success_steps_with_history(config, args.limit_steps)
    if baseline_correct is not None:
        steps = [step for step in steps if _step_uid(step) in baseline_correct]
    if args.limit_textdrift_base and len(steps) > args.limit_textdrift_base:
        steps = steps[: args.limit_textdrift_base]
    snippets = fail_text_snippets(config, args.limit_fail_traj, args.max_drift_snippets)
    local_state = threading.local()
    started = time.time()

    def get_client() -> VLLMClient:
        client = getattr(local_state, "client", None)
        if client is None:
            client = VLLMClient(args.api_url, args.model_name, timeout=args.timeout)
            local_state.client = client
        return client

    def corrupt_step(step: Step, injected_count: int) -> Step:
        selected = [snippets[(hash(_step_uid(step)) + idx) % len(snippets)] for idx in range(injected_count)]
        raw = dict(step.raw or {})
        raw["corrupt_history"] = "\n".join(f"Injected wrong history {idx + 1}: {text}" for idx, text in enumerate(selected))
        return replace(step, raw=raw)

    def score_one(item: Tuple[Step, int]) -> Dict[str, Any]:
        step, injected_count = item
        uid = _step_uid(step)
        row: Dict[str, Any] = {
            "step_uid": uid,
            "exec_id": step.exec_id,
            "app": step.app,
            "tag": step.tag,
            "split": step.split,
            "step_id": step.step_id,
            "total_steps": step.total_steps,
            "task_key": task_key_for_step(step),
            "model": args.model_name,
            "score_label": "textdrift_gate",
            "cond": {"history_mode": "corrupt", "input_mode": args.input_mode, "plan": None, "injected_error": injected_count},
        }
        try:
            query_step_obj = corrupt_step(step, injected_count)
            text, pred = _query_action(get_client(), query_step_obj, "corrupt", args.input_mode, args.max_tokens)
            correct = step_correct(pred, step)
            row.update({"ok": True, "correct": bool(correct), "prediction": pred, "text": text})
        except Exception as exc:
            row.update({"ok": False, "correct": None, "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
        return row

    injected_counts = list(range(1, args.max_injected + 1))
    items = [(step, injected_count) for step in steps for injected_count in injected_counts if f"{_step_uid(step)}:{injected_count}" not in existing]
    n_new = 0
    errors = 0
    parse_fail = 0
    if args.threads <= 1:
        iterator = (score_one(item) for item in items)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=args.threads)
        futures = [executor.submit(score_one, item) for item in items]
        iterator = (future.result() for future in as_completed(futures))
    try:
        for row in iterator:
            if row.get("ok"):
                if not (row.get("prediction") or {}).get("function"):
                    parse_fail += 1
            else:
                errors += 1
            append_jsonl(out_path, row)
            n_new += 1
            if args.flush_every and n_new % args.flush_every == 0:
                rows = latest_by_textdrift_key(read_jsonl(out_path))
                write_json(summary_path, summarize_textdrift_rows(rows, len(steps), extra={"in_progress": True, "elapsed_sec": time.time() - started, "threads": args.threads}))
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
    rows = latest_by_textdrift_key(read_jsonl(out_path))
    rewrite_jsonl(out_path, rows)
    write_json(summary_path, summarize_textdrift_rows(rows, len(steps), extra={"in_progress": False, "elapsed_sec": time.time() - started, "threads": args.threads, "n_new": n_new, "errors_this_run": errors, "parse_fail_this_run": parse_fail, "n_snippets": len(snippets)}))


def phase_plan_gate(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    out_path = Path(args.out_dir) / "scores" / "plan_gate.jsonl"
    summary_path = Path(args.out_dir) / "reports" / "plan_gate.json"
    initial_rows = latest_by_plan_key(read_jsonl(out_path))
    existing = {f"{row.get('step_uid')}:{(row.get('cond') or {}).get('plan')}" for row in initial_rows if row.get("ok") and row.get("step_uid")}
    plan_items = collect_success_plan_items(config, args.limit_steps)
    local_state = threading.local()
    started = time.time()

    def get_client() -> VLLMClient:
        client = getattr(local_state, "client", None)
        if client is None:
            client = VLLMClient(args.api_url, args.model_name, timeout=args.timeout)
            local_state.client = client
        return client

    def score_one(item: Tuple[Step, str, str]) -> Dict[str, Any]:
        step, plan_name, plan_text = item
        uid = _step_uid(step)
        row: Dict[str, Any] = {
            "step_uid": uid,
            "exec_id": step.exec_id,
            "app": step.app,
            "tag": step.tag,
            "split": step.split,
            "step_id": step.step_id,
            "total_steps": step.total_steps,
            "task_key": task_key_for_step(step),
            "model": args.model_name,
            "score_label": "plan_gate",
            "cond": {"history_mode": args.history_mode, "input_mode": args.input_mode, "plan": plan_name, "injected_error": 0},
        }
        try:
            plan = plan_text if plan_name == "oracle" else None
            text, pred = _query_action(get_client(), step, args.history_mode, args.input_mode, args.max_tokens, plan=plan)
            correct = step_correct(pred, step)
            row.update({"ok": True, "correct": bool(correct), "prediction": pred, "text": text})
        except Exception as exc:
            row.update({"ok": False, "correct": None, "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
        return row

    items: List[Tuple[Step, str, str]] = []
    for step, plan_text in plan_items:
        for plan_name in ("none", "oracle"):
            key = f"{_step_uid(step)}:{plan_name}"
            if key not in existing:
                items.append((step, plan_name, plan_text))
    n_new = 0
    errors = 0
    parse_fail = 0
    if args.threads <= 1:
        iterator = (score_one(item) for item in items)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=args.threads)
        futures = [executor.submit(score_one, item) for item in items]
        iterator = (future.result() for future in as_completed(futures))
    try:
        for row in iterator:
            if row.get("ok"):
                if not (row.get("prediction") or {}).get("function"):
                    parse_fail += 1
            else:
                errors += 1
            append_jsonl(out_path, row)
            n_new += 1
            if args.flush_every and n_new % args.flush_every == 0:
                write_json(summary_path, summarize_plan_rows(latest_by_plan_key(read_jsonl(out_path)), extra={"in_progress": True, "elapsed_sec": time.time() - started, "threads": args.threads}))
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
    rows = latest_by_plan_key(read_jsonl(out_path))
    rewrite_jsonl(out_path, rows)
    write_json(summary_path, summarize_plan_rows(rows, extra={"in_progress": False, "elapsed_sec": time.time() - started, "threads": args.threads, "n_new": n_new, "errors_this_run": errors, "parse_fail_this_run": parse_fail, "caveat": "value of correct global decomposition (horizon construct), not single-step"}))


def summarize_textmem_rows(rows: Sequence[Dict[str, Any]], gate_eps: float, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    by_mode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        mode = str((row.get("cond") or {}).get("history_mode") or "")
        if mode:
            by_mode[mode].append(row)
    acc_by_mode: Dict[str, Optional[float]] = {}
    n_by_mode: Dict[str, int] = {}
    errors_by_mode: Dict[str, int] = {}
    for mode, mode_rows in sorted(by_mode.items()):
        ok_rows = [row for row in mode_rows if row.get("ok") and row.get("correct") is not None]
        acc_by_mode[mode] = (sum(bool(row.get("correct")) for row in ok_rows) / len(ok_rows)) if ok_rows else None
        n_by_mode[mode] = len(ok_rows)
        errors_by_mode[mode] = len(mode_rows) - len(ok_rows)
    full = acc_by_mode.get("full")
    none = acc_by_mode.get("none")
    delta = None if full is None or none is None else full - none
    gate_passed = None if delta is None else abs(delta) <= gate_eps
    payload: Dict[str, Any] = {
        "acc_by_mode": acc_by_mode,
        "n_by_mode": n_by_mode,
        "errors_by_mode": errors_by_mode,
        "gate_eps": gate_eps,
        "full_minus_none": delta,
        "gate_passed": gate_passed,
        "n_rows": len(rows),
    }
    if extra:
        payload.update(extra)
    return payload


def summarize_textdrift_rows(rows: Sequence[Dict[str, Any]], n_base_steps: int, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    by_count: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        count = str((row.get("cond") or {}).get("injected_error") or "")
        if count:
            by_count[count].append(row)
    acc_by_injected: Dict[str, Optional[float]] = {}
    n_by_injected: Dict[str, int] = {}
    errors_by_injected: Dict[str, int] = {}
    for count, count_rows in sorted(by_count.items(), key=lambda item: int(item[0])):
        ok_rows = [row for row in count_rows if row.get("ok") and row.get("correct") is not None]
        acc_by_injected[count] = (sum(bool(row.get("correct")) for row in ok_rows) / len(ok_rows)) if ok_rows else None
        n_by_injected[count] = len(ok_rows)
        errors_by_injected[count] = len(count_rows) - len(ok_rows)
    payload: Dict[str, Any] = {
        "n_base_steps": n_base_steps,
        "acc_by_injected": acc_by_injected,
        "n_by_injected": n_by_injected,
        "errors_by_injected": errors_by_injected,
        "baseline_acc": 1.0,
        "drop_from_baseline": {key: None if value is None else value - 1.0 for key, value in acc_by_injected.items()},
        "n_rows": len(rows),
    }
    if extra:
        payload.update(extra)
    return payload


def summarize_plan_rows(rows: Sequence[Dict[str, Any]], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    by_plan: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        plan = str((row.get("cond") or {}).get("plan") or "")
        if plan:
            by_plan[plan].append(row)
    acc_by_plan: Dict[str, Optional[float]] = {}
    n_by_plan: Dict[str, int] = {}
    errors_by_plan: Dict[str, int] = {}
    for plan, plan_rows in sorted(by_plan.items()):
        ok_rows = [row for row in plan_rows if row.get("ok") and row.get("correct") is not None]
        acc_by_plan[plan] = (sum(bool(row.get("correct")) for row in ok_rows) / len(ok_rows)) if ok_rows else None
        n_by_plan[plan] = len(ok_rows)
        errors_by_plan[plan] = len(plan_rows) - len(ok_rows)
    none = acc_by_plan.get("none")
    oracle = acc_by_plan.get("oracle")
    delta = None if none is None or oracle is None else oracle - none
    payload: Dict[str, Any] = {
        "acc_by_plan": acc_by_plan,
        "n_by_plan": n_by_plan,
        "errors_by_plan": errors_by_plan,
        "oracle_minus_none": delta,
        "n_rows": len(rows),
    }
    if extra:
        payload.update(extra)
    return payload


def summarize_score_rows(rows: Sequence[Dict[str, Any]], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ok_rows = [row for row in rows if row.get("ok")]
    correct_rows = [row for row in ok_rows if row.get("correct") is not None]
    payload = {
        "n_rows": len(rows),
        "n_ok": len(ok_rows),
        "n_errors": len(rows) - len(ok_rows),
        "n_correct_eval": len(correct_rows),
        "correct": sum(bool(row.get("correct")) for row in correct_rows),
        "accuracy": (sum(bool(row.get("correct")) for row in correct_rows) / len(correct_rows)) if correct_rows else None,
    }
    if extra:
        payload.update(extra)
    return payload


def phase_validity_report(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    strong_rows = {row["step_uid"]: row for row in read_jsonl(Path(args.out_dir) / "scores" / f"{args.strong_label}.jsonl") if row.get("ok") and row.get("difficulty") is not None}
    test_rows = {row["step_uid"]: row for row in read_jsonl(Path(args.out_dir) / "scores" / f"{args.test_label}.jsonl") if row.get("ok") and row.get("correct") is not None}
    keys = sorted(set(strong_rows) & set(test_rows))
    d_scores = [float(strong_rows[key]["difficulty"]) for key in keys]
    test_correct = [bool(test_rows[key]["correct"]) for key in keys]
    report: Dict[str, Any] = {"n_joined": len(keys), "strong_label": args.strong_label, "test_label": args.test_label}
    if len(keys) >= 10:
        try:
            gate = validity_gate(d_scores, test_correct)
            report["gate"] = asdict(gate)
            report["passed"] = bool(gate.passed)
        except (DifficultyProxyInvalid, ValueError) as exc:
            report["passed"] = False
            report["error"] = str(exc)
    else:
        report["passed"] = False
        report["error"] = "not enough joined rows for validity_gate (need n>=10)"
    if keys:
        bucketizer = fit_buckets(np.asarray(d_scores, dtype=np.float64), k=min(10, max(2, len(set(d_scores)))))
        buckets = bucketizer.transform(d_scores)
        bucket_rows = []
        for bucket in sorted(set(int(x) for x in buckets)):
            idx = [pos for pos, value in enumerate(buckets) if int(value) == bucket]
            bucket_rows.append({"bucket": bucket, "n": len(idx), "mean_d": float(np.mean([d_scores[pos] for pos in idx])), "test_acc": float(np.mean([test_correct[pos] for pos in idx]))})
        report["buckets"] = bucket_rows
    write_json(Path(args.out_dir) / "reports" / "difficulty_validity.json", report)


def phase_recovery_scan(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    trajectories = collect_fail_trajectories(config, args.limit_fail_traj)
    out_path = Path(args.out_dir) / "reports" / "recovery_oracle_candidates.jsonl"
    if out_path.exists() and not args.overwrite:
        rows = read_jsonl(out_path)
    else:
        if out_path.exists():
            out_path.unlink()
        rows = []
        for traj in trajectories.values():
            for step in traj:
                target = recovery_oracle(step)
                if target is None:
                    continue
                row = {
                    "exec_id": step.exec_id,
                    "app": step.app,
                    "tag": step.tag,
                    "step_id": step.step_id,
                    "task_key": task_key_for_step(step),
                    "kind": target.kind,
                    "reason": target.reason,
                    "action": asdict(target.correct_action),
                    "image_rel_path": step.image_rel_path,
                }
                append_jsonl(out_path, row)
                rows.append(row)
    write_json(Path(args.out_dir) / "reports" / "recovery_oracle_summary.json", {"n_fail_traj": len(trajectories), "n_candidates": len(rows)})


def phase_divergence_scan(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    repo = str(config.get("repo") or "vyokky/GUI-360")
    success_steps = collect_success_steps(config, args.limit_success_steps)
    success_by_task: Dict[str, List[Step]] = defaultdict(list)
    for step in success_steps:
        success_by_task[task_key_for_step(step)].append(step)
    fail_trajectories = collect_fail_trajectories(config, args.limit_fail_traj)
    out_path = Path(args.out_dir) / "reports" / "divergence_candidates.jsonl"
    if out_path.exists() and not args.overwrite:
        rows = read_jsonl(out_path)
    else:
        if out_path.exists():
            out_path.unlink()
        rows = []
        for exec_id, traj in fail_trajectories.items():
            if not traj:
                continue
            key = task_key_for_step(traj[0])
            candidates = success_by_task.get(key) or []
            if not candidates:
                continue
            try:
                index = ScreenIndex(task_key=key, points=[make_screen_point(step) for step in candidates], alpha=args.divergence_alpha, band_width=args.divergence_band_width)
                t_star = detect_t_star(traj, index, args.divergence_tau)
                for step in traj:
                    row = {
                        "exec_id": exec_id,
                        "app": step.app,
                        "tag": step.tag,
                        "step_id": step.step_id,
                        "task_key": key,
                        "t_star": t_star,
                        "delta": delta(step, t_star),
                        "nearest_similarity": index.nearest_similarity(step),
                        "image_rel_path": step.image_rel_path,
                    }
                    append_jsonl(out_path, row)
                    rows.append(row)
            except Exception as exc:
                append_jsonl(out_path, {"exec_id": exec_id, "task_key": key, "error": f"{type(exc).__name__}: {str(exc)[:500]}"})
    n_with_t = sum(1 for row in rows if row.get("t_star") is not None)
    write_json(Path(args.out_dir) / "reports" / "divergence_summary.json", {"n_fail_traj": len(fail_trajectories), "n_rows": len(rows), "n_rows_with_t_star": n_with_t})


def phase_summary(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    out_dir = Path(args.out_dir)
    payload: Dict[str, Any] = {"out_dir": str(out_dir), "files": {}}
    for path in sorted(out_dir.rglob("*.json")):
        try:
            payload["files"][str(path.relative_to(out_dir))] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            payload["files"][str(path.relative_to(out_dir))] = {"error": str(exc)}
    try:
        verdict = decision({})
        payload["placeholder_decision"] = asdict(verdict)
    except DecisionAborted as exc:
        payload["placeholder_decision"] = asdict(exc.verdict)
    write_json(out_dir / "summary.json", payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GUI-360 long-horizon overnight phase runner")
    parser.add_argument("phase", choices=["ensure-fail-images", "prefetch-jsonl", "extract-success-images", "extract-fail-images", "loader-smoke", "score-success", "textmem-gate", "textdrift-gate", "plan-gate", "validity-report", "recovery-scan", "divergence-scan", "summary"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--raw_local_dir", default="datasets/GUI-360-raw-jsonl")
    parser.add_argument("--api_url", default="http://localhost:8001/v1")
    parser.add_argument("--model_name", default="checkpoints/Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--score_label", default="strong_72b")
    parser.add_argument("--strong_label", default="strong_72b")
    parser.add_argument("--test_label", default="test_sft")
    parser.add_argument("--limit_steps", type=int, default=1000)
    parser.add_argument("--limit_success_steps", type=int, default=1000)
    parser.add_argument("--limit_success_images", type=int, default=1000)
    parser.add_argument("--limit_fail_traj", type=int, default=300)
    parser.add_argument("--prefetch_splits", default="test,fail")
    parser.add_argument("--prefetch_limit_per_split", type=int, default=0)
    parser.add_argument("--history_mode", default="none", choices=["none", "full", "summary", "corrupt"])
    parser.add_argument("--history_modes", default="full,summary,corrupt,none")
    parser.add_argument("--input_mode", default="visual", choices=["visual"])
    parser.add_argument("--gate_eps", type=float, default=0.01)
    parser.add_argument("--baseline_rows", default="")
    parser.add_argument("--max_injected", type=int, default=3)
    parser.add_argument("--limit_textdrift_base", type=int, default=0)
    parser.add_argument("--max_drift_snippets", type=int, default=512)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--flush_every", type=int, default=25)
    parser.add_argument("--divergence_alpha", type=float, default=0.65)
    parser.add_argument("--divergence_tau", type=float, default=0.50)
    parser.add_argument("--divergence_band_width", type=float, default=0.20)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.phase == "ensure-fail-images":
        phase_ensure_fail_images(args, config)
    elif args.phase == "prefetch-jsonl":
        phase_prefetch_jsonl(args, config)
    elif args.phase == "extract-success-images":
        phase_extract_success_images(args, config)
    elif args.phase == "extract-fail-images":
        phase_extract_fail_images(args, config)
    elif args.phase == "loader-smoke":
        phase_loader_smoke(args, config)
    elif args.phase == "score-success":
        phase_score_success(args, config)
    elif args.phase == "textmem-gate":
        phase_textmem_gate(args, config)
    elif args.phase == "textdrift-gate":
        phase_textdrift_gate(args, config)
    elif args.phase == "plan-gate":
        phase_plan_gate(args, config)
    elif args.phase == "validity-report":
        phase_validity_report(args, config)
    elif args.phase == "recovery-scan":
        phase_recovery_scan(args, config)
    elif args.phase == "divergence-scan":
        phase_divergence_scan(args, config)
    elif args.phase == "summary":
        phase_summary(args, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
