#!/usr/bin/env python3
"""Run a multimodal fixed-choice selector over frozen label-blind Pass@8 packets."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import re
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests
from PIL import Image


PROMPT_VERSION = "pass8-fixed-choice-v1"
REQUIRED_BLIND_FIELDS = {
    "target_id",
    "episode_id",
    "step_idx",
    "goal",
    "screenshot",
    "baseline_action",
    "candidates",
    "packet_sha256",
}
FORBIDDEN_BLIND_KEYS = {
    "gt_action",
    "correct",
    "is_correct",
    "reward",
    "diagnostic_correct",
    "diagnostic_reward",
    "oracle",
    "candidate_provenance",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def packet_digest(row: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in row.items() if key != "packet_sha256"}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def encode_image(path: str, image_max_pixels: int | None) -> tuple[str, int, int]:
    image = Image.open(path).convert("RGB")
    image_w, image_h = image.size
    if image_max_pixels and image_w * image_h > image_max_pixels:
        scale = (image_max_pixels / (image_w * image_h)) ** 0.5
        image = image.resize((max(1, int(image_w * scale)), max(1, int(image_h * scale))), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8"), image_w, image_h


def append_jsonl(path: Path, row: Mapping[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
    with lock, path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()


def walk_forbidden(value: Any, location: str = "row") -> list[str]:
    found = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in FORBIDDEN_BLIND_KEYS:
                found.append(f"{location}.{key}")
            found.extend(walk_forbidden(nested, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            found.extend(walk_forbidden(nested, f"{location}[{index}]"))
    return found


def verify_frozen_blind(manifest_path: Path, blind_path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("prompt_version") != PROMPT_VERSION:
        raise ValueError(f"unsupported prompt version: {manifest.get('prompt_version')}")
    artifact = None
    for relative, metadata in manifest.get("artifacts", {}).items():
        candidate = (manifest_path.parent / relative).resolve()
        if candidate == blind_path.resolve():
            artifact = metadata
            break
    if artifact is None or not str(blind_path.resolve()).startswith(str((manifest_path.parent / "blind").resolve())):
        raise ValueError("input is not a registered frozen blind artifact")
    actual_hash = sha256_file(blind_path)
    if actual_hash != artifact.get("sha256"):
        raise ValueError(f"frozen blind hash mismatch: expected {artifact.get('sha256')}, got {actual_hash}")
    namespace_path = manifest_path.parent / "screenshot_namespace.json"
    namespace_artifact = manifest.get("artifacts", {}).get("screenshot_namespace.json")
    if namespace_artifact is None or sha256_file(namespace_path) != namespace_artifact.get("sha256"):
        raise ValueError("frozen screenshot namespace hash mismatch")
    namespace = json.loads(namespace_path.read_text(encoding="utf-8"))
    screenshot_hashes = {str(row["screenshot"]): str(row["sha256"]) for row in namespace.get("rows") or []}
    verified_screenshots: set[str] = set()
    for row in rows:
        missing = REQUIRED_BLIND_FIELDS - set(row)
        if missing:
            raise ValueError(f"blind packet {row.get('target_id')} is missing required fields: {sorted(missing)}")
        forbidden = walk_forbidden(row)
        if forbidden:
            raise ValueError(f"label leakage fields found in blind packet {row.get('target_id')}: {forbidden[:5]}")
        if row.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(f"packet prompt version mismatch: {row.get('target_id')}")
        if packet_digest(row) != row.get("packet_sha256"):
            raise ValueError(f"packet content hash mismatch: {row.get('target_id')}")
        screenshot = str(row["screenshot"])
        if screenshot not in screenshot_hashes:
            raise ValueError(f"screenshot is absent from frozen namespace: {screenshot}")
        if screenshot not in verified_screenshots:
            actual_screenshot_hash = sha256_file(Path(screenshot))
            if actual_screenshot_hash != screenshot_hashes[screenshot]:
                raise ValueError(f"frozen screenshot content hash mismatch: {screenshot}")
            verified_screenshots.add(screenshot)
        candidate_ids = [str(candidate.get("candidate_id")) for candidate in row.get("candidates") or []]
        if not candidate_ids or candidate_ids.count("BASELINE") != 1 or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"invalid candidate IDs in blind packet: {row.get('target_id')}")
    return manifest


def build_messages(row: Mapping[str, Any], image_max_pixels: int | None) -> list[dict[str, Any]]:
    candidates = [{
        "candidate_id": candidate["candidate_id"],
        "action": candidate.get("action"),
        "support_count": candidate.get("support_count"),
        "neighborhood_support_count": candidate.get("neighborhood_support_count", candidate.get("support_count")),
        "independent_support_count": candidate.get("source_count"),
        "is_student_baseline": bool(candidate.get("is_baseline")),
    } for candidate in row.get("candidates") or []]
    history = "\n".join(str(item) for item in row.get("history") or []) or "None"
    prompt = (
        "You are a conservative GUI action selector. Inspect the current screenshot, goal, and prior action history. "
        "Choose exactly one candidate action for the NEXT step. The candidates are fixed: do not invent, rewrite, "
        "merge, or adjust coordinates. BASELINE is the frozen student's action and is the safe choice whenever the "
        "evidence for changing it is weak. support_count reports exact repeats, neighborhood_support_count reports nearby "
        "coordinate/action agreement, and independent_support_count reports agreement across anonymized generators; none "
        "guarantees correctness.\n\n"
        "Return exactly one object inside <selection> tags with keys candidate_id, confidence (0..1), and reason.\n"
        "Example: <selection>{\"candidate_id\":\"BASELINE\",\"confidence\":0.72,\"reason\":\"brief reason\"}</selection>\n\n"
        f"GOAL:\n{row.get('goal', '')}\n\n"
        f"PRIOR ACTION HISTORY:\n{history}\n\n"
        f"SCREEN SIZE: {row.get('image_w')} x {row.get('image_h')}\n\n"
        f"FIXED CANDIDATES:\n{json.dumps(candidates, ensure_ascii=False, indent=2, sort_keys=True)}"
    )
    b64, _image_w, _image_h = encode_image(str(row["screenshot"]), image_max_pixels)
    return [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ],
    }]


def extract_selection(text: str) -> dict[str, Any] | None:
    candidates = []
    tagged = re.search(r"<selection>\s*(\{.*?\})\s*</selection>", text, re.DOTALL | re.IGNORECASE)
    if tagged:
        candidates.append(tagged.group(1))
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    first = text.find("{")
    if first >= 0:
        candidates.append(text[first:])
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value, _end = decoder.raw_decode(candidate.strip())
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def call_chat(api_url: str, model: str, messages: list[dict[str, Any]], args: argparse.Namespace) -> tuple[str, str | None]:
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
            return (f"<think>{reasoning}</think>\n{content}" if reasoning else content), None
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < args.retries:
                time.sleep(min(2.0 * (attempt + 1), 10.0))
    error = f"{type(last_error).__name__}: {last_error}" if last_error is not None else "unknown request failure"
    return f"ERROR: {error}", error


def select_one(row: Mapping[str, Any], api_url: str, args: argparse.Namespace) -> dict[str, Any]:
    raw_output, request_error = call_chat(api_url, args.model, build_messages(row, args.image_max_pixels), args)
    parsed = extract_selection(raw_output)
    valid_ids = {str(candidate["candidate_id"]) for candidate in row.get("candidates") or []}
    attempted = str(parsed.get("candidate_id")) if parsed and parsed.get("candidate_id") is not None else None
    if attempted in valid_ids:
        selected = attempted
        fallback_reason = None
    else:
        selected = "BASELINE"
        fallback_reason = "api_error" if request_error else "invalid_or_missing_candidate_id"
    confidence = parsed.get("confidence") if parsed else None
    try:
        confidence = min(1.0, max(0.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = None
    candidate_by_id = {str(candidate["candidate_id"]): candidate for candidate in row.get("candidates") or []}
    return {
        "protocol_version": row.get("protocol_version"),
        "prompt_version": PROMPT_VERSION,
        "selector_name": args.selector_name,
        "model": args.model,
        "target_id": row["target_id"],
        "episode_id": row["episode_id"],
        "step_idx": row["step_idx"],
        "packet_sha256": row["packet_sha256"],
        "attempted_candidate_id": attempted,
        "selected_candidate_id": selected,
        "selected_action": candidate_by_id.get(selected, {}).get("action"),
        "confidence": confidence,
        "reason": str(parsed.get("reason") or "")[:1000] if parsed else "",
        "parse_ok": parsed is not None and attempted in valid_ids,
        "fallback_reason": fallback_reason,
        "request_error": request_error,
        "raw_output": raw_output[:8000],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--blind", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--selector-name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-urls", nargs="+", required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--image-max-pixels", type=int, default=1040 * 736)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    blind_path = Path(args.blind)
    rows = read_jsonl(blind_path)
    verify_frozen_blind(Path(args.manifest), blind_path, rows)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    output_path = Path(args.output)
    existing_rows = read_jsonl(output_path) if output_path.exists() else []
    existing = {str(row["target_id"]): row for row in existing_rows}
    if len(existing) != len(existing_rows):
        raise ValueError("selector output contains duplicate target IDs")
    expected_packets = {str(row["target_id"]): str(row["packet_sha256"]) for row in rows}
    for tid, row in existing.items():
        if tid not in expected_packets or row.get("packet_sha256") != expected_packets[tid]:
            raise ValueError(f"stale or foreign existing output row: {tid}")
        if row.get("selector_name") != args.selector_name or row.get("model") != args.model:
            raise ValueError(f"existing output selector/model mismatch: {tid}")
    pending = [row for row in rows if str(row["target_id"]) not in existing]
    print(json.dumps({
        "selector": args.selector_name,
        "model": args.model,
        "blind_rows": len(rows),
        "already_done": len(existing),
        "pending": len(pending),
        "api_urls": args.api_urls,
    }, indent=2), flush=True)
    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {
            pool.submit(select_one, row, args.api_urls[index % len(args.api_urls)], args): row["target_id"]
            for index, row in enumerate(pending)
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            append_jsonl(output_path, result, lock)
            if completed % 25 == 0 or completed == len(pending):
                print(f"{args.selector_name}: completed {completed}/{len(pending)}", flush=True)


if __name__ == "__main__":
    main()