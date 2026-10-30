"""Fixtures shared by every package's tests."""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_engine_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Hide AUDIT_ENGINE_* environment variables and any .env file from every test."""
    for name in list(os.environ):
        if name.startswith("AUDIT_ENGINE_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
