"""`audit-engine check-judge`: try the judge on answers whose right verdict is known."""

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from audit_engine.commands import common
from audit_engine.commands.extract import read_policy
from audit_engine.commands.run import read_target
from audit_engine.judge import JudgeContext, JudgeError
from audit_engine.judge_check import (
    MAX_KNOWN_ANSWERS_CHARS,
    JudgeCheck,
    KnownAnswers,
    KnownAnswersError,
    check_judge,
    parse_known_answers,
)
from audit_engine.llm import LLMClient, LLMError

_MAX_ROUNDS = 20


def check_judge_command(
    known_path: Annotated[Path, typer.Argument(help="The file of known answers.")],
    policies: Annotated[
        list[Path],
        typer.Option("--policy", help="A current policy to judge against. Repeatable."),
    ],
    target_path: Annotated[
        Path | None,
        typer.Option("--target", help="The target file that names the bot's tools."),
    ] = None,
    rounds: Annotated[
        int, typer.Option(min=1, max=_MAX_ROUNDS, help="Times to judge each answer.")
    ] = 3,
    model: Annotated[
        str | None, typer.Option("--model", help="Judge with this model instead of the default.")
    ] = None,
) -> None:
    """Try the judge on known answers, to see whether a change helped. Not a measure of accuracy."""
    text = common.read_text(known_path, limit=MAX_KNOWN_ANSWERS_CHARS)
    try:
        known = parse_known_answers(text)
    except KnownAnswersError as exc:
        common.fail(f"Could not use {known_path}: {exc}")
    context = JudgeContext(
        policies=tuple(read_policy(path) for path in policies),
        actions=read_target(target_path).actions if target_path else {},
    )
    settings = common.load_settings()

    typer.echo(
        f"Verdicts to get: {len(known.answers) * rounds}. Model: {model or settings.llm_model}."
    )
    if last_only := [answer.case.id for answer in known.answers if answer.last_message_only]:
        typer.echo(
            "Judged on the last message only, as the bot's earlier turns aren't given: "
            f"{', '.join(last_only)}."
        )
    try:
        check = asyncio.run(_check(known, common.make_client(settings, model), context, rounds))
    except (LLMError, JudgeError) as exc:
        common.fail(f"The check failed: {exc}.")
    _report(check)


async def _check(
    known: KnownAnswers, llm: LLMClient, context: JudgeContext, rounds: int
) -> JudgeCheck:
    try:
        return await check_judge(known, llm, context, rounds)
    finally:
        await llm.aclose()


def _report(check: JudgeCheck) -> None:
    typer.echo(
        f"Acceptable verdicts: {check.acceptable} of {check.total}. "
        f"False passes: {check.false_passes}. False fails: {check.false_fails}."
    )
    for name, verdicts in check.verdicts.items():
        marker = "  <-- unacceptable" if name in check.wrong else ""
        typer.echo(f"  {name}: {dict(verdicts)}{marker}")
    if check.wrong:
        raise typer.Exit(common.EXIT_INCOMPLETE)
