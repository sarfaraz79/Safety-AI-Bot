"""`audit-engine judge`: decide whether each answer in a transcript was right."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from audit_engine.commands import common
from audit_engine.commands.extract import read_policy
from audit_engine.commands.run import read_tests
from audit_engine.judge import (
    JudgeContext,
    JudgeError,
    JudgeSummary,
    check_policies,
    check_transcript,
    judge_transcript,
)
from audit_engine.llm import LLMClient, LLMError
from audit_engine.results import JudgedAnswer, render_result
from audit_engine.testcases import CaseFile
from audit_engine.transcripts import (
    MAX_TRANSCRIPT_CHARS,
    TranscriptEntry,
    TranscriptError,
    parse_transcript,
)


def judge_command(
    transcript: Annotated[Path, typer.Argument(help="The transcript from `audit-engine run`.")],
    tests_path: Annotated[Path, typer.Option("--tests", help="The tests that were run.")],
    policies: Annotated[
        list[Path],
        typer.Option("--policy", help="A current policy to judge against. Repeatable."),
    ],
    output: Annotated[Path, typer.Option("--output", "-o", help="Where to save the verdicts.")],
    model: Annotated[
        str | None, typer.Option("--model", help="Judge with this model instead of the default.")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Replace existing results.")] = False,
) -> None:
    """Decide whether each answer was right, using automatic checks and then a model."""
    common.check_outputs([output], [transcript, tests_path, *policies], force=force)
    cases = read_tests(tests_path)
    entries, cut_short = _read_transcript(transcript)
    context = JudgeContext(policies=tuple(read_policy(path) for path in policies))
    try:
        check_policies(cases, context)
        check_transcript(cases, entries)
    except JudgeError as exc:
        common.fail(str(exc))
    settings = common.load_settings()

    typer.echo(f"Answers to judge: {len(entries)}. Model: {model or settings.llm_model}.")
    try:
        summary = asyncio.run(
            _judge(cases, entries, common.make_client(settings, model), output, context)
        )
    except LLMError as exc:
        common.fail(f"The model request failed ({exc}). Verdicts so far are in {output}.")
    except OSError as exc:
        common.fail(f"Could not write {output}: {exc.strerror}.")
    _report(summary, output, cut_short=cut_short)


def _read_transcript(path: Path) -> tuple[list[TranscriptEntry], bool]:
    text = common.read_text(path, limit=MAX_TRANSCRIPT_CHARS)
    try:
        return parse_transcript(text)
    except TranscriptError as exc:
        common.fail(f"Could not use {path}: {exc}")


async def _judge(
    cases: CaseFile,
    entries: list[TranscriptEntry],
    llm: LLMClient,
    output: Path,
    context: JudgeContext,
) -> JudgeSummary:
    try:
        with common.open_for_this_user(output) as file:

            def record(result: JudgedAnswer) -> None:
                file.write(render_result(result))
                file.flush()  # saved as it's decided, as each verdict can cost a model call

            return await judge_transcript(cases, entries, llm, record, context)
    finally:
        await llm.aclose()


def _report(summary: JudgeSummary, output: Path, *, cut_short: bool) -> None:
    counts = summary.counts
    typer.echo(
        f"Verdicts saved to {output}. Passed: {counts['pass']}. Failed: {counts['fail']}. "
        f"Unclear, for a person to check: {counts['unclear']}."
    )
    typer.echo("The judge's accuracy hasn't been measured, so treat these verdicts with care.")
    if cut_short:
        typer.echo("The transcript was cut short, so its last answer is missing.", err=True)
    if counts["error"] or counts["not_run"]:
        typer.echo(
            f"The bot gave no usable answer: {counts['error']}. "
            f"Tests with no answer at all: {counts['not_run']}.",
            err=True,
        )
    if cut_short or counts["error"] or counts["not_run"]:
        raise typer.Exit(common.EXIT_INCOMPLETE)
