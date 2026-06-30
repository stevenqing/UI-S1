"""Harness protocol for own-history arm construction.

The own-history arm is semi-online: it asks a policy for the model's own thought
and action, but it always patches the recorded action back to the expert action
and always advances on expert screenshots. The default `_UnwiredHarness` raises
so O-arm data cannot be silently fabricated without a real model endpoint.
"""

from __future__ import annotations

import json
import base64
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol

from .model import VLLMClient


@dataclass(frozen=True)
class HarnessPrediction:
    thought: str
    action: Dict[str, Any]
    text: str


class Harness(Protocol):
    def predict(self, conversations: List[Dict[str, str]], images: List[str], step: Dict[str, Any]) -> HarnessPrediction:
        """Return the model's own thought/action for the next expert screen."""


class UnwiredHarness:
    """Placeholder that prevents own-history rollout from running accidentally."""

    def predict(self, conversations: List[Dict[str, str]], images: List[str], step: Dict[str, Any]) -> HarnessPrediction:
        raise RuntimeError("own_history arm requires a wired Harness; refusing to run against _Unwired")


def image_path_to_data_url(path: str) -> str:
    image_path = Path(path)
    mime = mimetypes.guess_type(image_path.name)[0] or "image/png"
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def sharegpt_to_openai_messages(conversations: List[Dict[str, str]], images: List[str]) -> List[Dict[str, Any]]:
    marker_count = sum(str(turn.get("value", "")).count("<image>") for turn in conversations if turn.get("from") == "human")
    if marker_count != len(images):
        raise ValueError(f"OpenAI message conversion image mismatch: markers={marker_count}, images={len(images)}")

    image_iter = iter(images)
    messages: List[Dict[str, Any]] = []
    for turn in conversations:
        role = "assistant" if turn.get("from") == "gpt" else "user"
        value = str(turn.get("value") or "")
        if role == "assistant":
            messages.append({"role": role, "content": value})
            continue

        content: List[Dict[str, Any]] = []
        chunks = value.split("<image>")
        for chunk_index, chunk in enumerate(chunks):
            if chunk_index > 0:
                content.append({"type": "image_url", "image_url": {"url": image_path_to_data_url(next(image_iter))}})
            text = chunk.strip()
            if text:
                content.append({"type": "text", "text": text})
        if not content:
            content.append({"type": "text", "text": ""})
        messages.append({"role": role, "content": content})
    return messages


def parse_tool_action(text: str) -> Dict[str, Any]:
    """Parse a loose GUI-360 model output into an action dict."""

    obj: Dict[str, Any] = {}
    for pattern in (r"<tool_call>\s*(\{.*?\})\s*</tool_call>", r"```(?:json)?\s*(\{.*?\})\s*```"):
        match = re.search(pattern, text, flags=re.DOTALL)
        if not match:
            continue
        try:
            obj = json.loads(match.group(1))
            break
        except json.JSONDecodeError:
            pass
    if not obj:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                candidate, _ = decoder.raw_decode(text[match.start():])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                obj = candidate
                break
    if not obj:
        func_match = re.search(r"<tool_call>\s*([A-Za-z_][A-Za-z0-9_]*)", text)
        return {"function": func_match.group(1).lower()} if func_match else {"function": ""}

    function = str(obj.get("function") or obj.get("action") or "").strip().lower()
    args = obj.get("args") if isinstance(obj.get("args"), dict) else obj
    action: Dict[str, Any] = {"function": function, "raw_json": obj}
    coordinate = args.get("coordinate") or args.get("xy") or args.get("start_coordinate")
    if coordinate is not None:
        action["coordinate"] = coordinate
    text_value = args.get("keys") or args.get("text")
    if text_value is not None:
        action["text"] = text_value
    return action


class VLLMHarness:
    """Minimal OpenAI/vLLM-backed harness for own-history rollout."""

    def __init__(self, client: VLLMClient, max_tokens: int = 256, temperature: float = 0.0):
        self.client = client
        self.max_tokens = max_tokens
        self.temperature = temperature

    def predict(self, conversations: List[Dict[str, str]], images: List[str], step: Dict[str, Any]) -> HarnessPrediction:
        human = str(step.get("conversation_human") or "")
        messages = sharegpt_to_openai_messages(conversations + [{"from": "human", "value": human}], images)
        decode = self.client.generate(messages, n=1, max_tokens=self.max_tokens, temperature=self.temperature)[0]
        action = parse_tool_action(decode.text)
        thought = decode.text.split("<tool_call>", 1)[0].strip()
        return HarnessPrediction(thought=thought, action=action, text=decode.text)
