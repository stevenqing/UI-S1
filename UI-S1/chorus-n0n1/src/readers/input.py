from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping


FORBIDDEN_READER_FIELD_RE = re.compile(r"(gt|ground.?truth|label|target_action)", re.IGNORECASE)


@dataclass(frozen=True, init=False)
class ReaderInput:
    goal: str
    current_screenshot: str
    schema_payload: Mapping[str, Any]
    episode_id: str
    step_idx: int
    episode_len: int

    def __init__(
        self,
        source_record: Mapping[str, Any] | None = None,
        *,
        goal: str | None = None,
        current_screenshot: str | None = None,
        schema_payload: Mapping[str, Any] | None = None,
        episode_id: str | None = None,
        step_idx: int | None = None,
        episode_len: int | None = None,
    ) -> None:
        if source_record is not None:
            assert_reader_safe_record(source_record)
            goal = str(source_record.get("goal", "") if goal is None else goal)
            current_screenshot = str(
                source_record.get("current_screenshot", source_record.get("screenshot", ""))
                if current_screenshot is None
                else current_screenshot
            )
            schema_payload = source_record.get("schema_payload", {}) if schema_payload is None else schema_payload
            episode_id = str(source_record.get("episode_id", "") if episode_id is None else episode_id)
            step_idx = int(source_record.get("step_idx", 0) if step_idx is None else step_idx)
            episode_len = int(source_record.get("episode_len", source_record.get("num_steps", 0)) if episode_len is None else episode_len)

        object.__setattr__(self, "goal", goal or "")
        object.__setattr__(self, "current_screenshot", current_screenshot or "")
        object.__setattr__(self, "schema_payload", dict(schema_payload or {}))
        object.__setattr__(self, "episode_id", episode_id or "")
        object.__setattr__(self, "step_idx", int(step_idx or 0))
        object.__setattr__(self, "episode_len", int(episode_len or 0))

    def to_json(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "current_screenshot": self.current_screenshot,
            "schema_payload": dict(self.schema_payload),
            "episode_id": self.episode_id,
            "step_idx": self.step_idx,
            "episode_len": self.episode_len,
        }


def assert_reader_safe_record(record: Mapping[str, Any]) -> None:
    forbidden = sorted(_forbidden_key_paths(record))
    if forbidden:
        joined = ", ".join(forbidden)
        raise ValueError(f"ReaderInput source record contains GT/label fields: {joined}")


def build_reader_prompt(reader_input: ReaderInput) -> str:
    if not isinstance(reader_input, ReaderInput):
        raise TypeError("reader prompt builders accept ReaderInput exclusively")
    payload = reader_input.to_json()
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _forbidden_key_paths(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, Mapping):
        paths: list[str] = []
        for key, nested in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if FORBIDDEN_READER_FIELD_RE.search(key_text):
                paths.append(path)
            paths.extend(_forbidden_key_paths(nested, path))
        return paths
    if isinstance(value, list):
        paths = []
        for index, nested in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            paths.extend(_forbidden_key_paths(nested, path))
        return paths
    return []