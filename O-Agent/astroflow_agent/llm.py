"""Minimal OpenAI-compatible chat client. No LangChain chains."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class StaticLLM:
    """Deterministic stand-in for tests and first planning-quality cycles."""

    def __init__(self, response: str):
        self.response = response

    def complete(self, system: str, user: str) -> str:
        return self.response


class SequenceLLM:
    """Returns canned replies in order, then repeats the last one."""

    def __init__(self, responses: list[str]):
        if not responses:
            raise ValueError("SequenceLLM needs at least one response")
        self.responses = responses
        self.index = 0

    def complete(self, system: str, user: str) -> str:
        reply = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return reply


class OpenAICompatibleClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.environ.get("ASTROFLOW_AGENT_MODEL", "gpt-4o-mini")

    def complete(self, system: str, user: str) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        payload = json.dumps(
            {
                "model": self.model,
                "temperature": 0.1,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        return body["choices"][0]["message"]["content"]


def extract_json(text: str) -> dict:
    """Parse a JSON object from a model reply, ignoring markdown fences."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.endswith("```"):
            stripped = stripped[: -3]
        stripped = stripped.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"No JSON object in LLM output: {text[:400]}")
    return json.loads(stripped[start : end + 1])
