"""What the commands share: failing cleanly, settings, and safe file handling."""

import contextlib
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import NoReturn

import typer
from pydantic import ValidationError

from audit_engine.config import Settings
from audit_engine.llm import LLMClient, OpenAICompatibleClient

EXIT_FAILED = 1
EXIT_INCOMPLETE = 3


def fail(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(EXIT_FAILED)


def load_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        fail(f"Invalid configuration:\n{_describe_settings_errors(exc)}")


def _describe_settings_errors(exc: ValidationError) -> str:
    """List each problem by environment variable name, never echoing values: they may be secret."""
    lines = []
    for error in exc.errors(include_url=False, include_input=False):
        name = str(error["loc"][0]).upper()
        lines.append(f"  AUDIT_ENGINE_{name}: {error['msg'].removeprefix('Value error, ')}")
    return "\n".join(lines)


def make_client(settings: Settings, model: str | None = None) -> LLMClient:
    return OpenAICompatibleClient(
        model=model or settings.llm_model,
        api_key=settings.llm_api_key.get_secret_value(),
        base_url=str(settings.llm_base_url),
        timeout_seconds=settings.llm_timeout_seconds,
    )


def check_outputs(outputs: Iterable[Path], inputs: Iterable[Path], *, force: bool) -> None:
    """Runs first, so a mistake here costs nothing. Results are evidence, so they're never
    replaced by accident, and an input can never be replaced at all.
    """
    protected = {path.resolve() for path in inputs}
    for output in outputs:
        if output.resolve() in protected:
            fail(f"{output} would replace one of the input files.")
        if not output.parent.is_dir():
            fail(f"The folder {output.parent} does not exist.")
        if not force and output.exists():
            fail(f"{output} already exists. Use --force to replace it.")


def read_bytes(path: Path, *, limit: int) -> bytes:
    """Read at most `limit` + 1 bytes, which is enough for the caller to see it's too large."""
    try:
        with path.open("rb") as file:
            return file.read(limit + 1)
    except OSError as exc:
        fail(f"Could not read {path}: {exc.strerror}.")


def read_text(path: Path, *, limit: int) -> str:
    try:
        return read_bytes(path, limit=limit * 4).decode()
    except UnicodeDecodeError:
        fail(f"{path} is not valid UTF-8 text.")


def write_atomically(path: Path, text: str) -> None:
    """Write everything or nothing, so a crash can't leave half a file behind."""
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(text)
        Path(temporary).replace(path)
    except OSError:
        with contextlib.suppress(OSError):
            Path(temporary).unlink()
        raise


def write_results(path: Path, text: str) -> None:
    try:
        write_atomically(path, text)
    except OSError as exc:
        fail(f"Could not write {path}: {exc.strerror}.")
