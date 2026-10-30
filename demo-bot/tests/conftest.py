"""Shared fixtures. Tests never call a real model and never read the developer's settings."""

import os
from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from demo_bot.api import create_app
from demo_bot.config import Settings
from demo_bot.llm import LLMClient

LOCAL_CLIENT = ("127.0.0.1", 50000)

SettingsFactory = Callable[..., Settings]
ClientFactory = Callable[..., TestClient]


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Hide DEMO_BOT_* environment variables and any .env file from every test."""
    for name in list(os.environ):
        if name.startswith("DEMO_BOT_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def make_settings() -> SettingsFactory:
    def factory(**overrides: object) -> Settings:
        return Settings.model_validate(
            {"llm_model": "test-model", "llm_api_key": "test-key", **overrides}
        )

    return factory


@pytest.fixture
def make_client(make_settings: SettingsFactory) -> ClientFactory:
    """Build a test client for the app, calling from 127.0.0.1 unless `client` says otherwise."""

    def factory(
        llm: LLMClient, *, client: tuple[str, int] = LOCAL_CLIENT, **settings: object
    ) -> TestClient:
        return TestClient(
            create_app(make_settings(**settings), llm=llm),
            client=client,
            base_url="http://127.0.0.1:8000",
        )

    return factory
