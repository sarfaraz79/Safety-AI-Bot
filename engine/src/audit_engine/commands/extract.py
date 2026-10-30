"""`audit-engine extract`: list a policy's facts and rules, for a person to review."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from audit_engine.commands import common
from audit_engine.contract import render_contract
from audit_engine.documents import MAX_DOCUMENT_BYTES, Document, DocumentError, parse_document
from audit_engine.extract import ExtractionResult, extract, render_rejections
from audit_engine.llm import LLMClient, LLMError


def extract_command(
    policy: Annotated[Path, typer.Argument(help="The Markdown policy document to read.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to write the contract.")],
    force: Annotated[
        bool, typer.Option("--force", help="Replace an existing contract, losing its approvals.")
    ] = False,
) -> None:
    """List a policy's facts and rules in a contract, each with its source quote."""
    rejections = output.with_suffix(".rejected.yaml")
    common.check_outputs([output, rejections], [policy], force=force)
    settings = common.load_settings()
    document = read_policy(policy)

    typer.echo(f"Sections to read: {len(document.sections)}. Model: {settings.llm_model}.")
    try:
        result = asyncio.run(_extract(document, common.make_client(settings)))
    except LLMError as exc:
        common.fail(f"The model request failed ({exc}). Nothing was written.")

    common.write_results(output, render_contract(result.contract))
    common.write_results(rejections, render_rejections(result))
    _report(result, output, rejections)


def read_policy(path: Path) -> Document:
    raw = common.read_bytes(path, limit=MAX_DOCUMENT_BYTES)
    try:
        return parse_document(path.name, raw)
    except DocumentError as exc:
        common.fail(f"Could not use {path}: {exc}")


async def _extract(document: Document, llm: LLMClient) -> ExtractionResult:
    try:
        return await extract(document, llm)
    finally:
        await llm.aclose()


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
        raise typer.Exit(common.EXIT_INCOMPLETE)
