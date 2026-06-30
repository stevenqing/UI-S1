"""OpenAI-compatible vLLM client wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised in offline/unit-test envs
    OpenAI = None


@dataclass(frozen=True)
class Decode:
    text: str
    logprobs: Any = None
    raw: Any = None


class VLLMClient:
    """Small OpenAI-compatible client used by experiment runners."""

    def __init__(self, base_url: str, model_name: str, api_key: str = "dummy", timeout: float = 600.0):
        if OpenAI is None:
            raise ImportError("openai is required to instantiate VLLMClient")
        self.base_url = base_url
        self.model_name = model_name
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    @property
    def cache_id(self) -> str:
        return f"{self.base_url}|{self.model_name}"

    def generate(self, messages: List[Dict[str, Any]], n: int = 1, logprobs: bool = False, max_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0) -> List[Decode]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            n=n,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        return [Decode(text=choice.message.content or "", logprobs=getattr(choice, "logprobs", None), raw=choice) for choice in response.choices]
