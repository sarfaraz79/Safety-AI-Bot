"""Settings, read from environment variables prefixed DEMO_BOT_ or from a .env file."""

from typing import Literal, Self

from pydantic import Field, HttpUrl, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from demo_bot.security import is_loopback

PromptVersion = Literal["v1", "v2"]

MIN_AUTH_TOKEN_LENGTH = 16


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DEMO_BOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # Any provider with an OpenAI-compatible Chat Completions API.
    llm_model: str = Field(min_length=1)
    llm_api_key: SecretStr = Field(min_length=1)
    llm_base_url: HttpUrl = HttpUrl("https://api.openai.com/v1")
    llm_timeout_seconds: float = Field(default=30.0, gt=0, le=300)

    prompt_version: PromptVersion = "v1"
    retrieval_top_k: int = Field(default=2, ge=1, le=10)

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65_535)
    auth_token: SecretStr | None = None

    @model_validator(mode="after")
    def _check_auth_token(self) -> Self:
        if self.auth_token is None:
            if not is_loopback(self.host):
                raise ValueError(
                    "DEMO_BOT_AUTH_TOKEN must be set when DEMO_BOT_HOST is not loopback."
                )
        elif len(self.auth_token.get_secret_value()) < MIN_AUTH_TOKEN_LENGTH:
            raise ValueError(
                f"DEMO_BOT_AUTH_TOKEN must be at least {MIN_AUTH_TOKEN_LENGTH} characters long."
            )
        return self
