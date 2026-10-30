"""The command line: `audit-engine extract policy.md -o contract.yaml`."""

import asyncio
import contextlib
import os
import tempfile
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import ValidationError

from audit_engine.config import Settings
from audit_engine.contract import render_contract
from audit_engine.documents import MAX_DOCUMENT_BYTES, Document, DocumentError, parse_document
from audit_engine.extract import ExtractionResult, render_rejections
from audit_engine.extract import extract as run_extraction
from audit_engine.llm import LLMClient, LLMError, OpenAICompatibleClient

EXIT_FAILED = 1
EXIT_INCOMPLETE = 3

# Pretty exceptions print local variables, which would include the API key.
app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)


@app.callback()
def main() -> None:
    """Test a customer-facing AI agent against a company's own policies."""


@app.command("extract")
def extract_command(
    policy: Annotated[Path, typer.Argument(help="The Markdown policy document to read.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to write the contract.")],
    force: Annotated[
        bool, typer.Option("--force", help="Replace an existing contract, losing its approvals.")
    ] = False,
) -> None:
    """List a policy's facts and rules in a contract, each with its source quote."""
    rejections = output.with_suffix(".rejected.yaml")
    _check_outputs(policy, output, rejections, force=force)
    settings = _load_settings()
    document = _read_policy(policy)

    typer.echo(f"Sections to read: {len(document.sections)}. Model: {settings.llm_model}.")
    try:
        result = asyncio.run(_extract(document, _make_client(settings)))
    except LLMError as exc:
        _fail(f"The model request failed ({exc}). Nothing was written.")

    try:
        _write_atomically(output, render_contract(result.contract))
        _write_atomically(rejections, render_rejections(result))
    except OSError as exc:
        _fail(f"Could not write the results: {exc.strerror}.")
    _report(result, output, rejections)


def _check_outputs(policy: Path, output: Path, rejections: Path, *, force: bool) -> None:
    """Runs first, so a mistake here costs nothing."""
    if output.resolve() == policy.resolve() or rejections.resolve() == policy.resolve():
        _fail("The output would replace the policy document.")
    if not output.parent.is_dir():
        _fail(f"The folder {output.parent} does not exist.")
    if not force:
        for path in (output, rejections):
            if path.exists():
                _fail(f"{path} already exists, and may hold approvals. Use --force to replace it.")


def _load_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        _fail(f"Invalid configuration:\n{_describe_settings_errors(exc)}")


def _describe_settings_errors(exc: ValidationError) -> str:
    """List each problem by environment variable name, never echoing values: they may be secret."""
    lines = []
    for error in exc.errors(include_url=False, include_input=False):
        name = str(error["loc"][0]).upper()
        lines.append(f"  AUDIT_ENGINE_{name}: {error['msg'].removeprefix('Value error, ')}")
    return "\n".join(lines)


def _read_policy(path: Path) -> Document:
    try:
        with path.open("rb") as file:
            raw = file.read(MAX_DOCUMENT_BYTES + 1)
    except OSError as exc:
        _fail(f"Could not read {path}: {exc.strerror}.")
    try:
        return parse_document(path.name, raw)
    except DocumentError as exc:
        _fail(f"Could not use {path}: {exc}")


def _make_client(settings: Settings) -> LLMClient:
    return OpenAICompatibleClient(
        model=settings.llm_model,
        api_key=settings.llm_api_key.get_secret_value(),
        base_url=str(settings.llm_base_url),
        timeout_seconds=settings.llm_timeout_seconds,
    )


async def _extract(document: Document, llm: LLMClient) -> ExtractionResult:
    try:
        return await run_extraction(document, llm)
    finally:
        await llm.aclose()


def _write_atomically(path: Path, text: str) -> None:
    """Write everything or nothing, so a crash can't leave half a contract behind."""
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(text)
        Path(temporary).replace(path)
    except OSError:
        with contextlib.suppress(OSError):
            Path(temporary).unlink()
        raise


def _report(result: ExtractionResult, output: Path, rejections: Path) -> None:
    items = len(result.contract.items)
    typer.echo(
        f"Items written to {output}: {items}. Nothing is approved until a person reviews them."
    )
    if result.rejected:
        typer.echo(f"Suggestions rejected: {len(result.rejected)}. See {rejections}.")
    if result.failed_sections:
        typer.echo(
            f"The contract is incomplete. Sections with no usable reply: "
            f"{len(result.failed_sections)}. See {rejections}.",
            err=True,
        )
        raise typer.Exit(EXIT_INCOMPLETE)


def _fail(message: str) -> NoReturn:
    typer.echo(message, err=True)
    raise typer.Exit(EXIT_FAILED)
