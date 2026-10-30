from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from demo_bot.config import Settings


def test_defaults_are_local_and_safe(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings()

    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.auth_token is None
    assert settings.prompt_version == "v1"
    assert str(settings.llm_base_url) == "https://api.openai.com/v1"


def test_reads_prefixed_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_BOT_LLM_MODEL", "env-model")
    monkeypatch.setenv("DEMO_BOT_LLM_API_KEY", "env-key")
    monkeypatch.setenv("DEMO_BOT_PROMPT_VERSION", "v2")

    settings = Settings()

    assert settings.llm_model == "env-model"
    assert settings.llm_api_key.get_secret_value() == "env-key"
    assert settings.prompt_version == "v2"


def test_reads_dotenv_file_in_working_directory(tmp_path: Path) -> None:
    # The isolated_settings fixture has already moved into tmp_path.
    (tmp_path / ".env").write_text(
        "DEMO_BOT_LLM_MODEL=file-model\nDEMO_BOT_LLM_API_KEY=file-key\nDEMO_BOT_AUTH_TOKEN=\n",
        encoding="utf-8",
    )

    settings = Settings()

    assert settings.llm_model == "file-model"
    assert settings.auth_token is None  # an empty value counts as not set


def test_secrets_are_hidden_when_printed(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(auth_token="a-very-secret-token-value")

    assert "a-very-secret-token-value" not in repr(settings)
    assert "test-key" not in repr(settings)


def test_exposed_host_requires_a_token(make_settings: Callable[..., Settings]) -> None:
    # 0.0.0.0 means "every network interface", so other machines could reach the bot.
    with pytest.raises(ValidationError, match="DEMO_BOT_AUTH_TOKEN must be set"):
        make_settings(host="0.0.0.0")

    assert make_settings(host="0.0.0.0", auth_token="x" * 16).host == "0.0.0.0"


def test_short_token_is_rejected(make_settings: Callable[..., Settings]) -> None:
    with pytest.raises(ValidationError, match="at least 16 characters"):
        make_settings(auth_token="x" * 15)


@pytest.mark.parametrize(
    "overrides",
    [
        {"prompt_version": "v3"},
        {"llm_base_url": "file:///etc/passwd"},
        {"llm_model": ""},
        {"llm_api_key": ""},
        {"retrieval_top_k": 0},
        {"port": 70_000},
    ],
)
def test_invalid_values_are_rejected(
    make_settings: Callable[..., Settings], overrides: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        make_settings(**overrides)
