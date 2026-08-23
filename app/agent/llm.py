"""Shared LLM client for the agent nodes.

DevAgent talks to **NVIDIA NIM** (Nemotron) by default. Nothing here is
NVIDIA-specific though: the endpoint is OpenAI-compatible, so this is the OpenAI
SDK pointed at a configurable ``base_url``. Any OpenAI-compatible provider —
OpenRouter, vLLM, Ollama, an internal gateway — works by changing three env vars.

Configuration is read from provider-neutral names, falling back to the older
``OPENROUTER_*`` ones so existing setups keep working:

===================  ==========================  ==========================
Setting              Preferred                   Fallback
===================  ==========================  ==========================
Endpoint             ``LLM_BASE_URL``            ``OPENROUTER_BASE_URL``
Credential           ``LLM_API_KEY``             ``OPENROUTER_API_KEY``, ``NVIDIA_API_KEY``
Model                ``LLM_MODEL``               ``OPENROUTER_MODEL``
===================  ==========================  ==========================

The client is built lazily so importing this module never requires a key, and
``complete`` accepts an injected ``client`` so nodes stay testable offline.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Sequence


logger = logging.getLogger(__name__)

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Verify against build.nvidia.com before relying on this — NVIDIA's catalog moves
# and model ids get retired. It is only a default; LLM_MODEL overrides it.
DEFAULT_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"

DEFAULT_TIMEOUT_SECONDS = 120.0

_client: Any | None = None


def _first_env(*names: str) -> str | None:
    """First non-empty value among ``names``, preferring the earlier ones."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def get_base_url() -> str:
    return _first_env("LLM_BASE_URL", "OPENROUTER_BASE_URL") or NVIDIA_BASE_URL


def get_api_key() -> str | None:
    return _first_env("LLM_API_KEY", "OPENROUTER_API_KEY", "NVIDIA_API_KEY")


def get_model() -> str:
    return _first_env("LLM_MODEL", "OPENROUTER_MODEL") or DEFAULT_MODEL


def get_timeout() -> float:
    """Per-request timeout.

    Without this a stalled provider hangs a background run indefinitely, and the
    run is invisible rather than failed.
    """
    raw = os.getenv("LLM_TIMEOUT_SECONDS")
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        return float(raw)
    except ValueError:
        logger.warning("LLM_TIMEOUT_SECONDS=%r is not a number; using default.", raw)
        return DEFAULT_TIMEOUT_SECONDS


def get_client() -> Any:
    global _client
    if _client is None:
        from openai import OpenAI

        base_url = get_base_url()
        logger.info("LLM client: %s (model %s)", base_url, get_model())
        _client = OpenAI(base_url=base_url, api_key=get_api_key(), timeout=get_timeout())
    return _client


def reset_client() -> None:
    """Drop the cached client so a config change takes effect."""
    global _client
    _client = None


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
