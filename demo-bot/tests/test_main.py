from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI

from demo_bot import __main__ as entry_point


def test_starts_uvicorn_with_the_configured_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_BOT_LLM_MODEL", "test-model")
    monkeypatch.setenv("DEMO_BOT_LLM_API_KEY", "test-key")
    monkeypatch.setenv("DEMO_BOT_PORT", "8123")
    calls: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append((app, kw)))

    entry_point.main()

    [(app, options)] = calls
    assert isinstance(app, FastAPI)
    assert options == {"host": "127.0.0.1", "port": 8123, "server_header": False}


def test_missing_configuration_names_the_variables_to_set() -> None:
    with pytest.raises(SystemExit) as caught:
        entry_point.main()

    message = str(caught.value)
    assert "DEMO_BOT_LLM_MODEL: Field required" in message
    assert "DEMO_BOT_LLM_API_KEY: Field required" in message


def test_configuration_errors_never_echo_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_BOT_LLM_MODEL", "test-model")
    monkeypatch.setenv("DEMO_BOT_LLM_API_KEY", "test-key")
    monkeypatch.setenv("DEMO_BOT_PORT", "sk-pasted-into-the-wrong-variable")

    with pytest.raises(SystemExit) as caught:
        entry_point.main()

    assert "DEMO_BOT_PORT" in str(caught.value)
    assert "sk-pasted" not in str(caught.value)


def test_rule_violations_read_as_plain_sentences(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_BOT_LLM_MODEL", "test-model")
    monkeypatch.setenv("DEMO_BOT_LLM_API_KEY", "test-key")
    monkeypatch.setenv("DEMO_BOT_HOST", "0.0.0.0")

    with pytest.raises(SystemExit) as caught:
        entry_point.main()

    assert "settings: DEMO_BOT_AUTH_TOKEN must be set" in str(caught.value)
