from __future__ import annotations

import pytest

from app.agent import llm


_ALL_VARS = (
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_TIMEOUT_SECONDS",
    "OPENROUTER_BASE_URL",
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
    "NVIDIA_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in _ALL_VARS:
        monkeypatch.delenv(name, raising=False)
    llm.reset_client()
    yield
    llm.reset_client()


def test_defaults_to_nvidia_nim() -> None:
    assert llm.get_base_url() == llm.NVIDIA_BASE_URL
    assert llm.get_model() == llm.DEFAULT_MODEL
    assert llm.get_api_key() is None


def test_llm_vars_are_preferred(monkeypatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "primary")
    monkeypatch.setenv("LLM_MODEL", "vendor/model-x")

    assert llm.get_base_url() == "https://example.test/v1"
    assert llm.get_api_key() == "primary"
    assert llm.get_model() == "vendor/model-x"


def test_openrouter_vars_still_work(monkeypatch) -> None:
    """Existing setups must not break on the rename."""
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "legacy")
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")

    assert llm.get_base_url() == "https://openrouter.ai/api/v1"
    assert llm.get_api_key() == "legacy"
    assert llm.get_model() == "anthropic/claude-3.5-sonnet"


def test_llm_vars_win_over_openrouter(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "legacy")
    monkeypatch.setenv("LLM_API_KEY", "primary")

    assert llm.get_api_key() == "primary"


def test_nvidia_api_key_is_accepted(monkeypatch) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-xxx")

    assert llm.get_api_key() == "nvapi-xxx"


def test_empty_env_var_falls_through(monkeypatch) -> None:
    """An exported-but-blank var is a common .env slip; treat it as unset."""
    monkeypatch.setenv("LLM_MODEL", "")
    monkeypatch.setenv("OPENROUTER_MODEL", "fallback/model")

    assert llm.get_model() == "fallback/model"


def test_timeout_defaults_and_overrides(monkeypatch) -> None:
    assert llm.get_timeout() == llm.DEFAULT_TIMEOUT_SECONDS

    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")
    assert llm.get_timeout() == 30.0


def test_unparseable_timeout_falls_back(monkeypatch) -> None:
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "soon")

    # A typo must not make every request hang forever.
    assert llm.get_timeout() == llm.DEFAULT_TIMEOUT_SECONDS


def test_complete_passes_model_and_temperature() -> None:
    from types import SimpleNamespace

    calls: list[dict] = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    result = llm.complete([{"role": "user", "content": "q"}], client=client, model="m")

    assert result == "hi"
    assert calls[0]["model"] == "m"
    assert calls[0]["temperature"] == 0.0
    assert "response_format" not in calls[0]


def test_reset_client_drops_the_cache(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(llm, "_client", sentinel)

    llm.reset_client()

    assert llm._client is None
