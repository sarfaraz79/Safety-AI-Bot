from pathlib import Path

import pytest
from pydantic import ValidationError

from audit_engine.config import Settings


def settings(**overrides: object) -> Settings:
    return Settings.model_validate({"llm_model": "m", "llm_api_key": "k", **overrides})


def test_defaults() -> None:
    loaded = settings()

    assert str(loaded.llm_base_url) == "https://api.openai.com/v1"
    assert loaded.llm_timeout_seconds == 120


def test_reads_prefixed_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUDIT_ENGINE_LLM_MODEL", "env-model")
    monkeypatch.setenv("AUDIT_ENGINE_LLM_API_KEY", "env-key")

    loaded = Settings()

    assert loaded.llm_model == "env-model"
    assert loaded.llm_api_key.get_secret_value() == "env-key"


def test_reads_dotenv_alongside_the_demo_bots_settings(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "DEMO_BOT_LLM_MODEL=other\nAUDIT_ENGINE_LLM_MODEL=file-model\nAUDIT_ENGINE_LLM_API_KEY=k\n",
        encoding="utf-8",
    )

    assert Settings().llm_model == "file-model"


def test_the_key_is_hidden_when_printed() -> None:
    assert "very-secret" not in repr(settings(llm_api_key="very-secret"))


@pytest.mark.parametrize(
    "url", ["http://llm.example/v1", "http://192.168.1.20/v1", "http://127.0.0.1.evil.example/v1"]
)
def test_the_key_is_never_sent_unencrypted_to_another_machine(url: str) -> None:
    with pytest.raises(ValidationError, match="must use https"):
        settings(llm_base_url=url)


@pytest.mark.parametrize(
    "url",
    [
        "https://api.anthropic.com/v1/",
        "http://localhost:11434/v1",
        "http://LOCALHOST:11434/v1",
        "http://127.0.0.1:11434/v1",
        "http://[::1]:11434/v1",
    ],
)
def test_encrypted_and_local_model_addresses_are_accepted(url: str) -> None:
    assert settings(llm_base_url=url).llm_base_url.host


@pytest.mark.parametrize(
    "key",
    [
        pytest.param("", id="empty"),
        pytest.param("   ", id="blank"),
        pytest.param("sk ant key", id="inner-space"),
        pytest.param("sk-key\nsecond-line", id="line-break"),
        pytest.param("sk-“quoted”-key", id="smart-quotes"),
    ],
)
def test_keys_that_cannot_travel_in_a_header_are_refused(key: str) -> None:
    with pytest.raises(ValidationError) as caught:
        settings(llm_api_key=key)

    assert key.strip() == "" or key not in caught.value.json(include_input=False)


@pytest.mark.parametrize(
    "overrides",
    [
        {"llm_model": ""},
        {"llm_model": "   "},
        {"llm_base_url": "file:///etc/passwd"},
        {"llm_base_url": "not a url"},
        {"llm_timeout_seconds": 0},
        {"llm_timeout_seconds": 601},
        {"llm_timeout_seconds": float("nan")},
    ],
)
def test_invalid_values_are_refused(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        settings(**overrides)
