from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import resolve_path


@dataclass(frozen=True)
class InferenceRequest:
    model: str
    messages: List[Dict[str, Any]]
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    logprobs: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def cache_key(self) -> str:
        payload = {
            "model": self.model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "logprobs": self.logprobs,
            "extra": self.extra,
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CachedOpenAIClient:
    def __init__(
        self,
        api_url: str,
        cache_dir: str | Path,
        cost_log: str | Path,
        api_key: str = "dummy",
        retries: int = 2,
        retry_sleep: float = 2.0,
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=api_url, api_key=api_key)
        self.cache_dir = resolve_path(cache_dir) or Path(cache_dir)
        self.cost_log = resolve_path(cost_log) or Path(cost_log)
        self.retries = retries
        self.retry_sleep = retry_sleep
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cost_log.parent.mkdir(parents=True, exist_ok=True)

    def chat(self, request: InferenceRequest) -> Dict[str, Any]:
        key = request.cache_key()
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["cache_hit"] = True
            return payload

        start = time.time()
        last_error: Optional[str] = None
        for attempt in range(self.retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=request.model,
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    logprobs=request.logprobs,
                    **request.extra,
                )
                payload = self._normalize_response(response, key, time.time() - start)
                with cache_path.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                self._append_cost_log(payload)
                payload["cache_hit"] = False
                return payload
            except Exception as exc:
                last_error = repr(exc)
                if attempt < self.retries:
                    time.sleep(self.retry_sleep)

        payload = {
            "cache_key": key,
            "text": "",
            "finish_reason": "error",
            "truncated": False,
            "error": last_error,
            "usage": {},
            "mean_logprob": None,
            "min_logprob": None,
            "latency_s": time.time() - start,
            "cache_hit": False,
        }
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        self._append_cost_log(payload)
        return payload

    def _normalize_response(self, response: Any, key: str, latency_s: float) -> Dict[str, Any]:
        choice = response.choices[0]
        text = choice.message.content or ""
        finish_reason = choice.finish_reason
        token_logprobs = _extract_token_logprobs(getattr(choice, "logprobs", None))
        return {
            "cache_key": key,
            "text": text,
            "finish_reason": finish_reason,
            "truncated": finish_reason == "length",
            "error": None,
            "usage": _model_dump(getattr(response, "usage", None)),
            "mean_logprob": _mean(token_logprobs),
            "min_logprob": min(token_logprobs) if token_logprobs else None,
            "latency_s": latency_s,
        }

    def _append_cost_log(self, payload: Dict[str, Any]) -> None:
        row = {
            "cache_key": payload.get("cache_key"),
            "finish_reason": payload.get("finish_reason"),
            "truncated": payload.get("truncated", False),
            "usage": payload.get("usage", {}),
            "latency_s": payload.get("latency_s"),
            "error": payload.get("error"),
        }
        with self.cost_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_token_logprobs(logprobs: Any) -> List[float]:
    if not logprobs:
        return []
    content = getattr(logprobs, "content", None) or []
    values = []
    for item in content:
        value = getattr(item, "logprob", None)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _model_dump(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    return dict(getattr(obj, "__dict__", {}))
