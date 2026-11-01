"""Settings, read from environment variables prefixed AUDIT_ENGINE_ or from a .env file."""

from pydantic import Field, HttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from audit_engine.fields import HEADER_SAFE, EncryptedUrl


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AUDIT_ENGINE_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        str_strip_whitespace=True,
    )

    # Any provider with an OpenAI-compatible Chat Completions API.
    llm_model: str = Field(min_length=1)
    llm_api_key: SecretStr = Field(min_length=1)
    llm_base_url: EncryptedUrl = HttpUrl("https://api.openai.com/v1")
    llm_timeout_seconds: float = Field(default=120.0, gt=0, le=600)

    @field_validator("llm_api_key")
    @classmethod
    def _check_key_characters(cls, value: SecretStr) -> SecretStr:
        if not HEADER_SAFE.fullmatch(value.get_secret_value()):
            raise ValueError("must contain only printable ASCII characters, with no spaces.")
        return value
