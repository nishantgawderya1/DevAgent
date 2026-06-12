"""Shared LLM client for the agent nodes.

DevAgent routes all chat-completion calls through **OpenRouter** (OpenAI-compatible
API). The client is lazily constructed so importing this module never requires a
key, and ``complete`` accepts an injected ``client`` so nodes are testable offline.
"""

from __future__ import annotations

import os
from typing import Any, Sequence


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"

_client: Any | None = None


def get_client() -> Any:
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(
            base_url=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    return _client


def get_model() -> str:
    return os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)


def complete(
    messages: Sequence[dict[str, str]],
    *,
    temperature: float = 0.0,
    response_format: dict[str, Any] | None = None,
    client: Any | None = None,
    model: str | None = None,
) -> str:
    """Run one chat completion and return the assistant message text."""
    client = client or get_client()
    model = model or get_model()

    kwargs: dict[str, Any] = {"model": model, "messages": list(messages), "temperature": temperature}
    if response_format is not None:
        kwargs["response_format"] = response_format

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""
