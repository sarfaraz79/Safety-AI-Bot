"""`audit-engine run`: put a test file's questions to a bot and save its answers."""

import asyncio
import os
from pathlib import Path
from typing import Annotated

import typer

from audit_engine.commands import common
from audit_engine.run import (
    MAX_CONCURRENCY,
    MAX_REPEATS,
    MAX_REQUESTS_PER_SECOND,
    RunLimits,
    RunSummary,
    run_cases,
)
from audit_engine.target import (
    MAX_TARGET_CHARS,
    Target,
    TargetClient,
    TargetError,
    parse_target,
    read_token,
)
from audit_engine.testcases import (
    MAX_TESTS_FILE_CHARS,
    Case,
    CaseFile,
    CaseFileError,
    parse_cases,
)
from audit_engine.transcripts import TranscriptEntry, render_entry


def run_command(
    tests_path: Annotated[Path, typer.Argument(help="The test file.")],
    target_path: Annotated[Path, typer.Option("--target", help="How to reach the bot.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to save the answers.")],
    repeats: Annotated[
        int,
        typer.Option(min=1, max=MAX_REPEATS, help="Times to ask each question. Multiplies cost."),
    ] = 1,
    concurrency: Annotated[
        int, typer.Option(min=1, max=MAX_CONCURRENCY, help="Most requests at once.")
    ] = 4,
    rate: Annotated[
        float,
        typer.Option(min=0.1, max=MAX_REQUESTS_PER_SECOND, help="Most new requests per second."),
    ] = 5.0,
    force: Annotated[bool, typer.Option("--force", help="Replace an existing transcript.")] = False,
) -> None:
    """Put a test file's questions to a bot, and save every answer as it arrives."""
    common.check_outputs([output], [tests_path, target_path], force=force)
    cases = read_tests(tests_path)
    target = read_target(target_path)
    try:
        token = read_token(target, os.environ)
    except TargetError as exc:
        common.fail(str(exc))
    if not cases.tests:
        common.fail(f"{tests_path} has no tests.")

    limits = RunLimits(repeats=repeats, concurrency=concurrency, requests_per_second=rate)
    typer.echo(
        f"Requests to send: {len(cases.tests) * repeats} "
        f"({len(cases.tests)} tests, {repeats} each). Each may cost the bot's owner money."
    )
    client = make_target_client(target, token)
    try:
        summary = asyncio.run(_run(cases.tests, client, output, limits))
    except OSError as exc:
        common.fail(f"Could not write {output}: {exc.strerror}. Answers so far were kept.")
    _report(summary, output)


def make_target_client(target: Target, token: str | None) -> TargetClient:
    return TargetClient(target, token)


def read_tests(path: Path) -> CaseFile:
    text = common.read_text(path, limit=MAX_TESTS_FILE_CHARS)
    try:
        return parse_cases(text)
    except CaseFileError as exc:
        common.fail(f"Could not use {path}: {exc}")


def read_target(path: Path) -> Target:
    text = common.read_text(path, limit=MAX_TARGET_CHARS)
    try:
        return parse_target(text)
    except TargetError as exc:
        common.fail(f"Could not use {path}: {exc}")


async def _run(
    cases: list[Case], client: TargetClient, output: Path, limits: RunLimits
) -> RunSummary:
    try:
        with common.open_for_this_user(output) as file:

            def record(entry: TranscriptEntry) -> None:
                file.write(render_entry(entry))
                file.flush()  # saved as it arrives, so a crash keeps every answer so far

            return await run_cases(cases, client.ask, record, limits)
    finally:
        await client.aclose()


def _report(summary: RunSummary, output: Path) -> None:
    typer.echo(f"Answers saved to {output}: {summary.answered}.")
    if summary.failed:
        typer.echo(f"Requests that got no usable answer: {summary.failed}. See {output}.", err=True)
        raise typer.Exit(common.EXIT_INCOMPLETE)
