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
    "url",
    ["http://llm.example/v1", "http://192.168.1.20:11434/v1", "http://127.0.0.1.evil.example/v1"],
)
def test_the_key_is_never_sent_unencrypted_to_another_machine(
    make_settings: Callable[..., Settings], url: str
) -> None:
    with pytest.raises(ValidationError, match="must use https"):
        make_settings(llm_base_url=url)


@pytest.mark.parametrize(
    "url",
    [
        "https://api.anthropic.com/v1/",
        "http://localhost:11434/v1",
        "http://127.0.0.1:11434/v1",
        "http://[::1]:11434/v1",
    ],
)
def test_encrypted_and_local_model_addresses_are_accepted(
    make_settings: Callable[..., Settings], url: str
) -> None:
    assert make_settings(llm_base_url=url).llm_base_url.host


@pytest.mark.parametrize(
    "secret",
    [
        pytest.param("   ", id="blank"),
        pytest.param("sk ant key", id="inner-space"),
        pytest.param("sk-key\nsecond-line", id="line-break"),
        pytest.param("sk-“quoted”-key", id="smart-quotes"),
        pytest.param("sk-kéy", id="accented"),
    ],
)
@pytest.mark.parametrize("field", ["llm_api_key", "auth_token"])
def test_secrets_that_cannot_travel_in_a_header_stop_startup(
    make_settings: Callable[..., Settings], field: str, secret: str
) -> None:
    padded = secret.ljust(16, "x") if field == "auth_token" and secret != "   " else secret

    with pytest.raises(ValidationError) as caught:
        make_settings(**{field: padded})

    assert padded not in caught.value.json(include_input=False)


def test_text_settings_are_trimmed(make_settings: Callable[..., Settings]) -> None:
    settings = make_settings(llm_model="  claude-haiku-5-5 \n", host=" localhost ")

    assert settings.llm_model == "claude-haiku-5-5"
    assert settings.host == "localhost"


def test_environment_overrides_the_dotenv_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live checks switch prompt version this way, without editing any file."""
    (tmp_path / ".env").write_text(
        "DEMO_BOT_LLM_MODEL=m\nDEMO_BOT_LLM_API_KEY=k\nDEMO_BOT_PROMPT_VERSION=v1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEMO_BOT_PROMPT_VERSION", "v2")

    assert Settings().prompt_version == "v2"


@pytest.mark.parametrize(
    "overrides",
    [
        {"prompt_version": "v3"},
        {"prompt_version": "../../secrets"},
        {"llm_base_url": "file:///etc/passwd"},
        {"llm_base_url": "ftp://llm.example/"},
        {"llm_base_url": "not a url"},
        {"llm_model": ""},
        {"llm_model": "   "},
        {"llm_api_key": ""},
        {"llm_timeout_seconds": 0},
        {"llm_timeout_seconds": 301},
        {"llm_timeout_seconds": float("nan")},
        {"retrieval_top_k": 0},
        {"chat_timeout_seconds": 0},
        {"chat_timeout_seconds": 601},
        {"max_concurrent_chats": 0},
        {"chats_per_minute": 0},
        {"retrieval_top_k": 11},
        {"port": 0},
        {"port": 70_000},
        {"host": "   "},
    ],
)
def test_invalid_values_are_rejected(
    make_settings: Callable[..., Settings], overrides: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        make_settings(**overrides)
